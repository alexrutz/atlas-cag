"""Managed mode: Atlas runs llama-server itself, configured by presets."""

import asyncio
import logging
import os
import re
from logging.handlers import RotatingFileHandler
import shlex
import shutil
import signal
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from . import builds, models, sampling
from .config import Settings
from .sampling import SamplingConfig
from .store import Store

if TYPE_CHECKING:
    from .engine import Engine
    from .ingest import Ingestor

log = logging.getLogger("atlas.supervisor")

# Flags Atlas sets itself; allowing them in extra_args would break CAG invariants.
RESERVED_FLAGS = {
    "-m", "--model", "-hf", "-hfr", "--hf-repo", "--host", "--port", "-np", "--parallel", "-c", "--ctx-size",
    "-kvu", "--kv-unified", "-no-kvu", "--no-kv-unified", "-cram", "--cache-ram", "--slot-save-path",
    "-ctk", "--cache-type-k", "-ctv", "--cache-type-v", "-fa", "--flash-attn", "-ngl", "--gpu-layers",
    "--n-gpu-layers", "--swa-full", "--api-key", "--api-key-file", "--no-slots",
    "-mm", "--mmproj", "-mmu", "--mmproj-url", "--no-mmproj", "--mmproj-auto", "--no-mmproj-auto",
}
# llama-server options that change how images become tokens (part of visual cache variants)
IMAGE_FLAGS = {"--image-min-tokens", "--image-max-tokens"}
CRASH_WINDOW_S = 300
# llama.cpp's startup check; the first line is printed at the default log level, the second (with
# the amount) only at -lv 4
_FIT_FAILED = re.compile(r"failed to fit params to free device memory")
_FIT_SHORT = re.compile(r"need to reduce device memory by (\d+) MiB")
MAX_CRASH_RESTARTS = 3


class PresetConfig(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    model_path: str = Field(min_length=1)
    ctx_per_slot: int = Field(default=32768, ge=2048, le=4_194_304)
    slots: int = Field(default=2, ge=1, le=32)
    kv_type: Literal["f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"] = "q8_0"
    flash_attn: Literal["on", "auto", "off"] = "on"
    gpu_layers: str = "all"
    swa_full: bool = False
    extra_args: str = ""
    binary: str = ""  # llama-server command for this preset; empty = the standard build
    mmproj: str = ""  # vision projector (.gguf) for visual prefill; empty = text only
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)  # unset values: from the model file

    @field_validator("mmproj")
    @classmethod
    def _mmproj(cls, v: str) -> str:
        v = (v or "").strip()
        if v and not v.endswith(".gguf"):
            raise ValueError("vision projector must be a .gguf file")
        if v and not Path(v).is_file():
            raise ValueError(f"vision projector not found: {v}")
        return v

    @field_validator("binary")
    @classmethod
    def _binary(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            return ""
        info = builds.inspect(v)
        if info.problem == "file not found":
            raise ValueError(f"llama-server build not found: {v}")
        if info.problem and not info.missing_libs and not info.problem.startswith("cannot run"):
            raise ValueError(f"llama-server build cannot be used: {info.problem}")
        return v

    @field_validator("model_path")
    @classmethod
    def _model_exists(cls, v: str) -> str:
        if not v.endswith(".gguf"):
            raise ValueError("model must be a .gguf file")
        if not Path(v).is_file():
            raise ValueError(f"model file not found: {v}")
        return v

    @field_validator("gpu_layers")
    @classmethod
    def _gpu_layers(cls, v: str) -> str:
        v = str(v).strip().lower()
        if v not in ("all", "auto") and not v.isdigit():
            raise ValueError("gpu_layers must be 'all', 'auto' or a number")
        return v

    @field_validator("extra_args")
    @classmethod
    def _extra_args(cls, v: str) -> str:
        try:
            args = shlex.split(v)
        except ValueError as e:
            raise ValueError(f"cannot parse extra arguments: {e}") from e
        clash = sorted({a.split("=", 1)[0] for a in args} & RESERVED_FLAGS)
        if clash:
            raise ValueError(f"these flags are managed by Atlas and cannot be overridden: {', '.join(clash)}")
        return v.strip()

    @model_validator(mode="after")
    def _kv_needs_fa(self) -> "PresetConfig":
        if self.kv_type not in ("f16", "bf16") and self.flash_attn == "off":
            raise ValueError("a quantized KV cache requires flash attention (set it to 'on' or 'auto')")
        return self


def standard_build(settings: Settings, store: Store) -> str | None:
    """The build for presets that name none: the installed update, else ATLAS_LLAMA_SERVER_BIN."""
    updates = store.get_state("build_updates") or {}
    tag = updates.get("standard")
    path = ((updates.get("installed") or {}).get(tag) or {}).get("path") if tag else None
    if path and Path(path).is_file():
        return path
    return settings.llama_server_bin


def preset_ident(preset: dict) -> dict:
    """Preset fields that change the KV-cache format and therefore the cache fingerprint."""
    return {"kv_type": preset["kv_type"], "flash_attn": preset["flash_attn"], "swa_full": bool(preset["swa_full"])}


def preset_sampling(preset: dict) -> tuple[dict, dict, list[str]]:
    """The preset's effective sampling, where each value came from, and required ones missing."""
    path = Path(preset.get("model_path") or "")
    model = (models.describe_file(path).sampling or {}) if path.is_file() else {}
    return sampling.resolve(preset.get("sampling"), model)


def preset_label(preset: dict) -> str:
    return f"{preset['name']} · {Path(preset['model_path']).name} · {preset['kv_type']} KV"


def vision_ident(preset: dict) -> str | None:
    """Identifies what turns page images into tokens: the projector file and image options."""
    mmproj = preset.get("mmproj")
    if not mmproj:
        return None
    path = Path(mmproj)
    size = path.stat().st_size if path.is_file() else 0
    args = shlex.split(preset.get("extra_args") or "")
    image = [f"{a}={args[i + 1]}" for i, a in enumerate(args[:-1]) if a in IMAGE_FLAGS]
    image += [a for a in args if a.split("=", 1)[0] in IMAGE_FLAGS and "=" in a]
    return ":".join([path.name, str(size), *sorted(image)])


def build_command(binary: list[str], preset: dict, port: int, kv_dir: Path) -> list[str]:
    args = [
        *binary,
        "--model", preset["model_path"],
        "--host", "127.0.0.1", "--port", str(port),
        "--parallel", str(preset["slots"]),
        "--ctx-size", str(preset["slots"] * preset["ctx_per_slot"]),
        "--no-kv-unified", "--cache-ram", "0",
        "--flash-attn", preset["flash_attn"],
        "--cache-type-k", preset["kv_type"], "--cache-type-v", preset["kv_type"],
        "--n-gpu-layers", str(preset["gpu_layers"]),
        "--slot-save-path", str(kv_dir.resolve()),
    ]
    if preset.get("swa_full"):
        args.append("--swa-full")
    if preset.get("mmproj"):
        args += ["--mmproj", preset["mmproj"]]
    return args + shlex.split(preset.get("extra_args") or "")


class SupervisorError(RuntimeError):
    pass


class StartError(SupervisorError):
    """llama-server itself did not come up (as opposed to a failed check afterwards)."""


class Supervisor:
    def __init__(self, settings: Settings, store: Store, engine: "Engine", ingestor: "Ingestor"):
        self.settings = settings
        self.store = store
        self.engine = engine
        self.ingestor = ingestor
        self.port = settings.llama_port
        self.state = "stopped"  # stopped | starting | running | stopping | crashed | failed
        self.error: str | None = None
        self.preset: dict | None = None
        self.started_at: float | None = None
        self.log: deque[str] = deque(maxlen=400)
        self.proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._crashes: deque[float] = deque()
        self._tasks: set[asyncio.Task] = set()
        self._first_activation = True
        self._pid_file = settings.data_dir / "llama-server.pid"
        self.gpu_baseline: int | None = None  # GPU memory used by other programs, measured before starting
        self.running_command: str | None = None  # llama-server build of the running process
        # llama.cpp's own check at startup: the preset needs more GPU memory than is free
        self.fit_warning: str | None = None
        # set by the build updater: called after a successful start, and after a failed one (True = retry)
        self.on_started: Callable[[dict, str], None] | None = None
        self.on_start_failed: Callable[[dict, str, str], Awaitable[bool]] | None = None
        # llama-server's own output, kept across restarts in data/logs/llama-server.log
        self._llama_log = logging.getLogger("atlas.llama-server")
        self._llama_log.propagate = False
        if not self._llama_log.handlers:
            log_dir = settings.data_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(log_dir / "llama-server.log", maxBytes=20 * 2**20, backupCount=3,
                                          encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self._llama_log.addHandler(handler)
            self._llama_log.setLevel(logging.INFO)

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    def apply_sampling(self, preset: dict) -> None:
        """Sampling is sent with every request, so changing it needs no restart."""
        effective, _, missing = preset_sampling(preset)
        self.engine.sampling = effective
        if missing:
            log.warning("preset %r: %s; llama-server's defaults are used meanwhile", preset.get("name"),
                        sampling.describe_missing(missing))

    def command_for(self, preset: dict | None) -> str:
        return (preset or {}).get("binary") or standard_build(self.settings, self.store) or ""

    def binary_for(self, preset: dict | None) -> list[str]:
        return builds.command_words(self.command_for(preset))

    def to_json(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "preset": self.preset,
            "pid": self.proc.pid if self.proc else None,
            "started_at": self.started_at,
            "gpu_baseline": self.gpu_baseline,
            "build": self.running_command,
            "fit_warning": self.fit_warning,
            "command": shlex.join(build_command(self.binary_for(self.preset), self.preset, self.port,
                                                self.settings.kv_dir)) if self.preset else None,
            "log": list(self.log)[-200:],
        }

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # --- lifecycle ---------------------------------------------------------------------

    async def startup(self) -> None:
        self._kill_leftover()
        preset_id = self.store.get_state("active_preset")
        preset = self.store.get_preset(preset_id) if preset_id else None
        if preset:
            self._spawn(self.activate(preset))
        else:
            self.engine.pause("No model is running. Choose or create a preset in Settings.")
            self.ingestor.reconcile(startup=True)
            self._first_activation = False

    async def shutdown(self) -> None:
        for t in list(self._tasks):
            t.cancel()
        await self._stop_process()

    async def activate(self, preset: dict) -> None:
        """Switch llama-server to `preset`: drain work, restart, re-validate caches."""
        retry = False
        async with self._lock:
            self.engine.pause(f"Starting {preset['name']}…")
            try:
                await self._drain()
                await self._stop_process()
                await self._measure_gpu_baseline()
                self.preset = preset
                self.store.set_state("active_preset", preset["id"])
                self.engine.extra_ident = preset_ident(preset)
                self.engine.vision_ident = vision_ident(preset)
                self.apply_sampling(preset)
                model = models.describe_file(Path(preset["model_path"])) if Path(preset["model_path"]).is_file() else None
                self.engine.swa_window = None if preset.get("swa_full") else (model.sliding_window if model else None)
                self.engine.info.swa_restore_ok = None
                self.engine.config_label = preset_label(preset)
                await self._start_process()
                await self.engine.connect()
                if not self.engine.info.kv_dir_ok:
                    raise SupervisorError(self.engine.info.error or "slot persistence check failed")
                self.store.touch_config(self.engine.info.fingerprint, preset_label(preset),
                                        {"preset": preset["name"], **preset_ident(preset),
                                         "model": Path(preset["model_path"]).name,
                                         "build": self.engine.info.build})
                self.ingestor.set_concurrency(self.engine.info.n_slots)
                self.ingestor.reconcile(startup=self._first_activation)
                self._first_activation = False
                self.error = None
                self.engine.resume()
                log.info("llama-server running with preset %r (pid %s)", preset["name"], self.proc and self.proc.pid)
                if self.on_started:
                    self.on_started(preset, self.running_command)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.state = "failed"
                self.error = str(e) or type(e).__name__
                log.error("could not start preset %r: %s", preset["name"], self.error)
                await self._stop_process(state="failed")
                self.engine.pause(f"llama-server failed to start: {self.error}")
                if isinstance(e, StartError) and self.on_start_failed:
                    retry = await self.on_start_failed(preset, self.command_for(preset), self.error)
        if retry:
            await self.activate(self.store.get_preset(preset["id"]) or preset)

    async def stop(self) -> None:
        async with self._lock:
            self.engine.pause("llama-server is stopped. Activate a preset in Settings.")
            await self._drain()
            await self._stop_process()
            self.store.set_state("active_preset", None)
            self.preset = None

    # --- process handling --------------------------------------------------------------

    async def _measure_gpu_baseline(self) -> None:
        """GPU memory in use while llama-server is not running (desktop, other programs)."""
        from .models import gpu_info
        await asyncio.sleep(1.0)  # let the driver release the previous process's memory
        gpus = await gpu_info()
        if gpus:
            self.gpu_baseline = sum(g["memory_used"] for g in gpus)

    async def _drain(self, timeout: float = 60.0) -> None:
        pool = self.engine.pool
        if pool is None:
            return
        pool.pause()
        deadline = time.time() + timeout
        while pool.leases and time.time() < deadline:
            await asyncio.sleep(0.2)
        if pool.leases:
            log.warning("switching llama-server with %d request(s) still running", len(pool.leases))

    async def _start_process(self) -> None:
        command = self.command_for(self.preset)
        binary = builds.command_words(command)
        if not binary:
            raise SupervisorError("no llama-server build: set ATLAS_LLAMA_SERVER_BIN or choose a build in the preset")
        self.running_command = command
        cmd = build_command(binary, self.preset, self.port, self.settings.kv_dir)
        exe = shutil.which(cmd[0]) or cmd[0]
        env = os.environ.copy()
        # source builds keep their shared libraries next to the binary
        lib_dir = str(Path(exe).resolve().parent)
        env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        self.log.append(f"$ {shlex.join(cmd)}")
        self.fit_warning = None
        self._fit_short: str | None = None
        self._llama_log.info("=== starting preset %r: %s", self.preset.get("name"), shlex.join(cmd))
        self.state = "starting"
        self.error = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env,
                start_new_session=True,
            )
        except OSError as e:
            raise StartError(f"cannot run {cmd[0]}: {e}") from e
        self.proc = proc
        self._pid_file.write_text(str(proc.pid))
        self._spawn(self._pump(proc))

        deadline = time.time() + self.settings.llama_start_timeout_s
        while True:
            if proc.returncode is not None:
                tail = "\n".join(list(self.log)[-8:])
                raise StartError(f"llama-server exited with code {proc.returncode}:\n{tail}")
            if await self.engine.llama.health():
                break
            if time.time() > deadline:
                raise StartError(f"llama-server did not become ready within "
                                      f"{self.settings.llama_start_timeout_s:.0f}s")
            await asyncio.sleep(0.5)
        self.state = "running"
        self.started_at = time.time()
        self._spawn(self._watch(proc))

    async def _pump(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            if line:
                self.log.append(line)
                self._llama_log.info(line)
                if m := _FIT_SHORT.search(line):
                    self._fit_short = m.group(1)
                if _FIT_FAILED.search(line):
                    amount = f"{self._fit_short} MiB more" if self._fit_short else "more"
                    self.fit_warning = (
                        f"llama-server needs {amount} GPU memory than is free (keeping 1 GiB for the desktop): "
                        "Windows moves the rest into shared system memory, which makes llama-server slow and can "
                        "make the whole desktop stutter or freeze. Lower the context per slot, the slots or the KV "
                        "cache type.")
                    log.warning("preset %r does not fit into free GPU memory (%s)", (self.preset or {}).get("name"),
                                line.strip()[-120:])

    async def _watch(self, proc: asyncio.subprocess.Process) -> None:
        rc = await proc.wait()
        if self.proc is not proc or self.state != "running":
            return  # stopped on purpose
        self.state = "crashed"
        self.error = f"llama-server exited unexpectedly with code {rc}"
        log.error(self.error)
        self.engine.pause(self.error)
        now = time.time()
        self._crashes.append(now)
        while self._crashes and now - self._crashes[0] > CRASH_WINDOW_S:
            self._crashes.popleft()
        if len(self._crashes) > MAX_CRASH_RESTARTS:
            self.state = "failed"
            self.error += f"; gave up after {MAX_CRASH_RESTARTS} restarts in {CRASH_WINDOW_S // 60} minutes"
            self.engine.pause(self.error)
            return
        await asyncio.sleep(2 * len(self._crashes))
        if self.preset and self.state == "crashed":
            await self.activate(self.preset)

    async def _stop_process(self, state: str = "stopped") -> None:
        proc, self.proc = self.proc, None
        if proc and proc.returncode is None:
            self.state = "stopping"
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=20)
            except TimeoutError:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        self._pid_file.unlink(missing_ok=True)
        self.state = state
        self.started_at = None

    def _kill_leftover(self) -> None:
        """Stop a llama-server left running by a previous Atlas process that died hard."""
        try:
            pid = int(self._pid_file.read_text())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            self._pid_file.unlink(missing_ok=True)
            return
        if "--slot-save-path" in cmdline and str(self.port) in cmdline:
            log.warning("stopping leftover llama-server (pid %d)", pid)
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):
                    time.sleep(0.2)
                    os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._pid_file.unlink(missing_ok=True)

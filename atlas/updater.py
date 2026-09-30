"""Keep the standard llama-server build up to date from GitHub releases.

The default source, ai-dock/llama.cpp-cuda, packages every llama.cpp release with CUDA as
llama.cpp-<tag>-cuda-<version>-<arch>.tar.gz. The packages leave out the CUDA runtime (cudart,
cuBLAS, NCCL): Atlas links those from other llama.cpp builds or Python CUDA wheels already on this
machine, and downloads the wheels from PyPI only if nothing local fits.

With build_update_source = "patched", Atlas instead builds the same llama.cpp release from source
with its own patches (atlas/patches/llama.cpp) and installs it as "<tag>+atlas".

A new build becomes the standard build, i.e. the one used by every preset that does not name its
own. Presets whose model or extra arguments only the previous build supports are pinned to it. If
llama-server fails to start with a new build, Atlas goes back to the previous one and skips that
release.
"""

import asyncio
import hashlib
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import httpx

from . import builds, models
from .config import Settings
from .store import Store
from .supervisor import PresetConfig, Supervisor, standard_build

log = logging.getLogger("atlas.updater")

ARCH = {"x86-64": "amd64", "ARM64": "arm64"}.get(builds.HOST_MACHINE, builds.HOST_MACHINE)
WHEEL_PLATFORM = {"amd64": "x86_64", "arm64": "aarch64"}.get(ARCH, ARCH)
KEEP_UPDATES = 2  # installed updates kept besides builds that presets still use
TICK_S = 60.0
STATE_KEY = "build_updates"

# CUDA runtime libraries the packages link against and the PyPI wheels that ship them
WHEELS = {
    "libcudart.so.12": "nvidia-cuda-runtime-cu12",
    "libcublas.so.12": "nvidia-cublas-cu12",
    "libcublasLt.so.12": "nvidia-cublas-cu12",
    "libnccl.so.2": "nvidia-nccl-cu12",
}
LIB_SEARCH = [
    "~/*", "~/*/build*/bin", "~/*/lib/python3*/site-packages/nvidia/*/lib",
    "~/.local/lib/python3*/site-packages/nvidia/*/lib", "/usr/local/cuda*/lib64",
    "/usr/local/cuda*/targets/*/lib", "/opt/*/lib",
]


PATCHES = Path(__file__).parent / "patches" / "llama.cpp"
PATCHED_SUFFIX = "+atlas"
BUILD_LOG_LINES = 12  # of a failed step, shown in the error


class UpdateError(RuntimeError):
    pass


def _cuda_version(text: str) -> tuple[int, int] | None:
    m = re.search(r"(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _driver_cuda() -> tuple[int, int] | None:
    """The newest CUDA version the installed driver supports ("CUDA Version" in nvidia-smi)."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return _cuda_version(m.group(1)) if m else None


def _nvcc_version(nvcc: Path) -> tuple[int, int] | None:
    if m := re.search(r"cuda-(\d+\.\d+)", str(nvcc.resolve())):
        return _cuda_version(m.group(1))
    try:
        out = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release\s+([\d.]+)", out)
    return _cuda_version(m.group(1)) if m else None


def find_nvcc() -> str | None:
    """The newest CUDA compiler whose version the driver supports: binaries built with a newer
    toolkit than the driver may not run."""
    paths = [Path(p) for p in [shutil.which("nvcc")] if p] + sorted(Path("/usr/local").glob("cuda*/bin/nvcc"))
    found: dict[Path, tuple[int, int]] = {}
    for nvcc in paths:
        if nvcc.resolve() not in found and (version := _nvcc_version(nvcc)):
            found[nvcc.resolve()] = version
    driver = _driver_cuda()
    usable = {p: v for p, v in found.items() if driver is None or v <= driver}
    if not usable:
        return str(paths[0]) if paths else None
    return str(max(usable, key=lambda p: usable[p]))


_toolchain_cache: tuple[float, dict] | None = None


def toolchain(max_age_s: float = 300.0) -> dict:
    """What building llama.cpp from source needs, and what is missing (checked every few minutes)."""
    global _toolchain_cache
    if _toolchain_cache and time.time() - _toolchain_cache[0] < max_age_s:
        return _toolchain_cache[1]
    tools = {"git": shutil.which("git"), "cmake": shutil.which("cmake"),
             "c++": shutil.which("c++") or shutil.which("g++") or shutil.which("clang++"), "nvcc": find_nvcc()}
    result = {"tools": tools, "missing": [name for name, path in tools.items() if not path]}
    _toolchain_cache = (time.time(), result)
    return result


def cuda_arch(configured: str) -> str:
    """CMAKE_CUDA_ARCHITECTURES for this machine's GPU, e.g. "89" for compute capability 8.9."""
    if configured:
        return configured
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        caps = sorted({line.strip().replace(".", "") for line in out.splitlines() if line.strip()})
        if caps:
            return ";".join(caps)
    except (OSError, subprocess.SubprocessError):
        pass
    return "native"


def patch_files() -> list[Path]:
    return sorted(PATCHES.glob("*.patch"))


def _glob(pattern: str) -> list[Path]:
    base = Path(pattern).expanduser()
    return sorted(Path(base.anchor).glob(str(base.relative_to(base.anchor))))


def find_libs(missing: list[str], dirs: list[Path]) -> dict[str, Path]:
    """Locate shared libraries, preferring folders that have several of them (matching versions)."""
    found: dict[Path, set[str]] = {}
    for d in dict.fromkeys(dirs):
        have = {lib for lib in missing if (d / lib).is_file() and builds.format_problem((d / lib).resolve()) is None}
        if have:
            found[d] = have
    chosen: dict[str, Path] = {}
    for d, have in sorted(found.items(), key=lambda kv: -len(kv[1])):
        for lib in sorted(have - chosen.keys()):
            chosen[lib] = d / lib
    return chosen


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.unlink(missing_ok=True)
    try:
        os.link(src.resolve(), dst)  # no extra disk space, survives removal of the source
    except OSError:
        shutil.copy2(src.resolve(), dst)


class BuildUpdater:
    def __init__(self, settings: Settings, store: Store, supervisor: Supervisor):
        self.settings = settings
        self.store = store
        self.supervisor = supervisor
        self.dir = (settings.data_dir / "builds").resolve()
        self.runtime_dir = self.dir / "cuda-runtime"
        self.job: dict = {"state": "idle"}  # idle | checking | downloading | installing | done | failed
        self._task: asyncio.Task | None = None
        self._loop: asyncio.Task | None = None
        supervisor.on_start_failed = self._start_failed
        supervisor.on_started = self._started

    # --- state -------------------------------------------------------------------------

    @property
    def state(self) -> dict:
        return self.store.get_state(STATE_KEY) or {}

    def _save(self, **changes) -> dict:
        state = {**self.state, **changes}
        self.store.set_state(STATE_KEY, state)
        return state

    def _installed(self) -> dict[str, dict]:
        return self.state.get("installed") or {}

    def installed_commands(self) -> list[str]:
        return [i["path"] for i in self._installed().values() if Path(i["path"]).is_file()]

    @property
    def restart_pending(self) -> bool:
        """The running preset uses the standard build, but llama-server still runs an older one."""
        sup = self.supervisor
        if not sup.preset or sup.preset.get("binary") or sup.state not in ("running", "crashed", "failed"):
            return False
        return sup.running_command is not None and sup.running_command != standard_build(self.settings, self.store)

    def to_json(self) -> dict:
        state = self.state
        standard = standard_build(self.settings, self.store)
        installed = sorted(self._installed().items(), key=lambda kv: kv[1].get("published_at", ""), reverse=True)
        return {
            "mode": self.settings.build_updates,
            "repo": self.settings.build_update_repo,
            "asset": self.settings.build_update_asset or f"cuda-*-{ARCH}.tar.gz",
            "interval_h": self.settings.build_update_interval_h,
            "source": self.settings.build_update_source,
            "source_repo": self.settings.build_source_repo,
            "patches": [p.stem for p in patch_files()],
            "toolchain": toolchain()["missing"],
            "last_check": state.get("last_check"),
            "latest": state.get("latest"),
            "error": state.get("error"),
            "standard": standard,
            "standard_tag": state.get("standard"),
            "configured": self.settings.llama_server_bin,
            "previous": state.get("previous"),
            "can_roll_back": bool(state.get("standard")),
            "installed": [{"tag": tag, **info} for tag, info in installed],
            "pinned": state.get("pinned") or [],
            "skipped": state.get("skipped") or [],
            "job": self.job,
            "restart_pending": self.restart_pending,
        }

    # --- background loop ---------------------------------------------------------------

    def start(self) -> None:
        self._loop = asyncio.create_task(self._run(), name="build-updates")

    async def stop(self) -> None:
        for t in (self._task, self._loop):
            if t:
                t.cancel()
        await asyncio.gather(*(t for t in (self._task, self._loop) if t), return_exceptions=True)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(TICK_S)
            try:
                mode = self.settings.build_updates
                due = time.time() - (self.state.get("last_check") or 0) >= self.settings.build_update_interval_h * 3600
                # the standard build is still a prebuilt one: build the patched one now (after a
                # failed attempt, only at the next regular check)
                unpatched = (self.settings.build_update_source == "patched"
                             and not (self.state.get("standard") or "").endswith(PATCHED_SUFFIX)
                             and not self.state.get("error"))
                if mode != "off" and (due or unpatched) and not self.busy:
                    await self.check_now()
                if mode == "apply":
                    await self.apply_if_idle()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("build update check failed")

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def check_in_background(self) -> None:
        if not self.busy:
            self.job = {"state": "checking"}  # at once: the previous job's "done" must not linger
            self._task = asyncio.create_task(self.check_now(), name="build-update-check")

    async def check_now(self) -> None:
        try:
            self.job = {"state": "checking"}
            latest = await self.check()
            if latest and self._is_new(latest):
                await self.install(latest)
            self.job = {"state": "done", "at": time.time()}
        except asyncio.CancelledError:
            self.job = {"state": "idle"}
            raise
        except Exception as e:
            message = str(e) or type(e).__name__
            log.warning("build update failed: %s", message)
            self.job = {"state": "failed", "error": message, "at": time.time()}
            self._save(error=message)

    async def apply_if_idle(self) -> bool:
        """Restart llama-server on the new standard build once no request is running."""
        sup = self.supervisor
        pool = sup.engine.pool
        if not self.restart_pending or sup.busy or (pool and (pool.leases or pool.n_waiting)):
            return False
        log.info("restarting llama-server to use %s", self.state.get("standard"))
        await sup.activate(sup.preset)
        return True

    # --- GitHub releases ---------------------------------------------------------------

    def _client(self, timeout: float = 30.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(timeout, connect=15.0),
                                 headers={"User-Agent": "atlas-cag"})

    def _asset_matches(self, name: str) -> bool:
        pattern = self.settings.build_update_asset
        if pattern:
            return pattern in name and name.endswith((".tar.gz", ".tgz"))
        return re.search(rf"cuda-[\d.]+-{re.escape(ARCH)}\.(tar\.gz|tgz)$", name) is not None

    async def check(self) -> dict | None:
        """Find the newest release with a package for this machine and remember it."""
        repo = self.settings.build_update_repo
        url = f"{self.settings.github_api.rstrip('/')}/repos/{repo}/releases"
        async with self._client() as c:
            r = await c.get(url, params={"per_page": 15}, headers={"Accept": "application/vnd.github+json"})
        if r.status_code == 404:
            raise UpdateError(f"GitHub repository {repo} not found")
        if r.status_code == 403 and r.headers.get("x-ratelimit-remaining") == "0":
            raise UpdateError("GitHub API rate limit reached; the next check will retry")
        if r.status_code != 200:
            raise UpdateError(f"GitHub API returned HTTP {r.status_code}")
        skipped = set(self.state.get("skipped") or [])
        latest = None
        for rel in r.json():
            if rel.get("draft") or rel.get("prerelease") or self.key(rel["tag_name"]) in skipped:
                continue
            asset = next((a for a in rel.get("assets", []) if self._asset_matches(a["name"])), None)
            if asset:
                digest = asset.get("digest") or ""
                latest = {"tag": rel["tag_name"], "published_at": rel.get("published_at") or "",
                          "name": rel.get("name") or rel["tag_name"], "url": rel.get("html_url"),
                          "asset": asset["name"], "size": asset.get("size") or 0,
                          "download_url": asset["browser_download_url"],
                          "sha256": digest.removeprefix("sha256:") if digest.startswith("sha256:") else None}
                break
        self._save(last_check=time.time(), latest=latest, error=None)
        return latest

    def key(self, tag: str) -> str:
        """How a release is installed: its tag, or "<tag>+atlas" when built from source with Atlas's patches."""
        return tag + PATCHED_SUFFIX if self.settings.build_update_source == "patched" else tag

    def _is_new(self, release: dict) -> bool:
        """Newer than the standard build and than any release the user rolled back from, or the
        standard build's release in the other form (prebuilt / patched) after switching the source."""
        key = self.key(release["tag"])
        if key == self.state.get("standard"):
            return False
        floor = self.state.get("floor") or ""
        published = release["published_at"]
        if published == self._standard_published():
            return published >= floor
        return published > max(self._standard_published(), floor)

    def _standard_published(self) -> str:
        tag = self.state.get("standard")
        return (self._installed().get(tag) or {}).get("published_at", "") if tag else ""

    # --- installing --------------------------------------------------------------------

    async def install(self, release: dict) -> None:
        tag = release["tag"]
        installed = self._installed().get(self.key(tag))
        if installed and Path(installed["path"]).is_file():
            await self.switch(self.key(tag))  # e.g. an earlier install that was rolled back and is wanted again
            return
        if self.settings.build_update_source == "patched":
            await self.build_patched(release)
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        archive = self.dir / f".{tag}.tar.gz"
        await self._download(release, archive)
        self.job = {"state": "installing", "tag": tag}
        dest = self.dir / re.sub(r"[^\w.-]", "_", tag)
        exe = await asyncio.to_thread(self._extract, archive, dest)
        info = await self.provide_runtime(exe)
        if not info.runnable:
            shutil.rmtree(dest, ignore_errors=True)
            raise UpdateError(f"{tag} cannot run here: {info.problem}")
        archive.unlink(missing_ok=True)
        installed = self._installed()
        installed[tag] = {"path": str(exe), "published_at": release["published_at"], "version": info.version,
                          "asset": release["asset"], "installed_at": time.time()}
        self._save(installed=installed)
        log.info("installed llama-server %s (%s) in %s", tag, info.version, dest)
        await self.switch(tag)

    # --- building from source with Atlas's patches ------------------------------------------

    async def build_patched(self, release: dict) -> None:
        """Build llama-server for a release from source with Atlas's patches and make it the standard."""
        tag = release["tag"]
        key = tag + PATCHED_SUFFIX
        tools = toolchain(max_age_s=0)
        if tools["missing"]:
            raise UpdateError(f"building llama.cpp needs {', '.join(tools['missing'])}: install "
                              f"{'the CUDA toolkit' if 'nvcc' in tools['missing'] else 'the missing tools'}, "
                              "or use the prebuilt releases")
        self.dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^\w.+-]", "_", key)
        src = self.dir / "src" / safe
        dest = self.dir / safe
        log_path = self.dir / f"{safe}.build.log"
        shutil.rmtree(src, ignore_errors=True)
        src.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"building {key} from {self.settings.build_source_repo}\n")
        try:
            self.job = {"state": "building", "tag": key, "step": "downloading the source"}
            await self._step(["git", "clone", "--depth", "1", "--branch", tag, self.settings.build_source_repo, str(src)],
                             log_path)
            self.job["step"] = "applying Atlas's patches"
            applied = await self._apply_patches(src, log_path)
            self.job["step"] = "configuring"
            arch = await asyncio.to_thread(cuda_arch, self.settings.build_cuda_arch)
            await self._step(["cmake", "-S", str(src), "-B", str(src / "build"), "-DCMAKE_BUILD_TYPE=Release",
                              "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                              f"-DCMAKE_CUDA_COMPILER={tools['tools']['nvcc']}", "-DGGML_NATIVE=ON",
                              "-DBUILD_SHARED_LIBS=ON", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
                              "-DLLAMA_BUILD_EXAMPLES=OFF"], log_path)
            self.job.update(step="compiling", percent=0)
            jobs = self.settings.build_jobs or os.cpu_count() or 4
            await self._step(["cmake", "--build", str(src / "build"), "--target", "llama-server", "-j", str(jobs)],
                             log_path, progress=True)
            self.job["step"] = "installing"
            exe = await asyncio.to_thread(self._install_bin, src / "build" / "bin", dest)
            info = await self.provide_runtime(exe)
            if not info.runnable:
                shutil.rmtree(dest, ignore_errors=True)
                raise UpdateError(f"{key} was built but cannot run here: {info.problem}")
        finally:
            shutil.rmtree(src, ignore_errors=True)  # the source and build tree take gigabytes
        installed = self._installed()
        installed[key] = {"path": str(exe), "published_at": release["published_at"], "version": info.version,
                          "asset": f"built from source (CUDA {arch})", "patched": applied, "installed_at": time.time()}
        self._save(installed=installed)
        log.info("built llama-server %s (%s) with %s", key, info.version, ", ".join(f"{n} ({s})" for n, s in applied.items()))
        await self.switch(key)

    async def _apply_patches(self, src: Path, log_path: Path) -> dict[str, str]:
        """Apply every patch; one the release already contains is skipped, one that conflicts stops the build."""
        applied = {}
        for patch in patch_files():
            if await self._step(["git", "-C", str(src), "apply", "--check", str(patch)], log_path, check=False) == 0:
                await self._step(["git", "-C", str(src), "apply", str(patch)], log_path)
                applied[patch.stem] = "applied"
            elif await self._step(["git", "-C", str(src), "apply", "--reverse", "--check", str(patch)], log_path,
                                  check=False) == 0:
                applied[patch.stem] = "already in this release"
            else:
                raise UpdateError(f"Atlas's patch {patch.stem} does not apply to this llama.cpp release (the code it "
                                  "changes was rewritten). Use the prebuilt release until Atlas's patches are updated.")
        return applied

    async def _step(self, cmd: list[str], log_path: Path, check: bool = True, progress: bool = False) -> int:
        """Run one build step at low priority, appending its output to the build log."""
        nice = ["nice", "-n", "10"] if shutil.which("nice") else []
        tail: list[str] = []
        with open(log_path, "a", encoding="utf-8") as logf:
            logf.write(f"$ {' '.join(cmd)}\n")
            proc = await asyncio.create_subprocess_exec(*nice, *cmd, stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.STDOUT, start_new_session=True)
            try:
                assert proc.stdout is not None
                async for raw in proc.stdout:
                    line = raw.decode(errors="replace").rstrip()
                    logf.write(line + "\n")
                    tail = (tail + [line])[-BUILD_LOG_LINES:]
                    if progress and (m := re.match(r"\[\s*(\d+)%\]", line)):
                        self.job["percent"] = int(m.group(1))
                rc = await proc.wait()
            except asyncio.CancelledError:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)  # the compiler and its children
                except ProcessLookupError:
                    pass
                raise
        if check and rc != 0:
            detail = "\n".join(line for line in tail if line.strip())
            raise UpdateError(f"{' '.join(cmd[:3])} failed (exit {rc}); build log {log_path}:\n{detail}")
        return rc

    def _install_bin(self, bin_dir: Path, dest: Path) -> Path:
        if not (bin_dir / "llama-server").is_file():
            raise UpdateError("the build produced no llama-server")
        staging = dest.with_name(f".{dest.name}.staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for f in bin_dir.iterdir():  # llama-server and the shared libraries it was linked with
            if f.name == "llama-server" or ".so" in f.name:
                shutil.copy2(f, staging / f.name, follow_symlinks=False)
        shutil.rmtree(dest, ignore_errors=True)
        staging.rename(dest)
        return dest / "llama-server"

    async def _download(self, release: dict, target: Path) -> None:
        part = target.with_name(target.name + ".part")
        total = release.get("size") or 0
        have = part.stat().st_size if part.exists() else 0
        self.job = {"state": "downloading", "tag": release["tag"], "done": have, "total": total}
        async with self._client(timeout=120.0) as c:
            headers = {"Range": f"bytes={have}-"} if have else {}
            async with c.stream("GET", release["download_url"], headers=headers) as r:
                if r.status_code == 416:
                    pass  # already complete
                elif r.status_code not in (200, 206):
                    raise UpdateError(f"download of {release['asset']} failed: HTTP {r.status_code}")
                else:
                    if r.status_code == 200:
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        async for chunk in r.aiter_bytes(1 << 20):
                            f.write(chunk)
                            have += len(chunk)
                            self.job["done"] = have
        if release.get("sha256"):
            digest = await asyncio.to_thread(_sha256, part)
            if digest != release["sha256"]:
                part.unlink(missing_ok=True)
                raise UpdateError(f"checksum mismatch for {release['asset']}; the download was discarded")
        part.rename(target)

    def _extract(self, archive: Path, dest: Path) -> Path:
        staging = dest.with_name(f".{dest.name}.staging")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            with tarfile.open(archive) as tar:
                tar.extractall(staging, filter="data")
            candidates = sorted((p for p in staging.rglob("llama-server") if p.is_file()),
                                key=lambda p: len(p.parts))
            if not candidates:
                raise UpdateError(f"{archive.name} contains no llama-server")
            root = candidates[0].parent
            shutil.rmtree(dest, ignore_errors=True)
            root.rename(dest)
            return dest / "llama-server"
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    async def provide_runtime(self, exe: Path) -> builds.BuildInfo:
        """Make the libraries a package leaves out available next to its binary."""
        provided: set[str] = set()
        while True:
            info = await asyncio.to_thread(builds.inspect, str(exe))
            missing = [lib for lib in info.missing_libs if lib not in provided]
            if not missing:
                return info
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            found = find_libs(missing, self._lib_dirs(exe))
            for lib in [m for m in missing if m not in found and m in WHEELS]:
                if (self.runtime_dir / lib).is_file():
                    found[lib] = self.runtime_dir / lib
            if wanted := [m for m in missing if m not in found and m in WHEELS]:
                found.update(await self._wheel_libs(wanted))
            if not found:
                return info  # nothing more to offer; info.problem names what is missing
            for lib, src in found.items():
                shared = self.runtime_dir / lib
                if src.resolve() != shared.resolve():
                    await asyncio.to_thread(_link_or_copy, src, shared)
                link = exe.parent / lib
                link.unlink(missing_ok=True)
                link.symlink_to(os.path.relpath(shared, exe.parent))
                provided.add(lib)
                log.info("using %s from %s for %s", lib, src.parent, exe.parent.name)

    def _lib_dirs(self, exe: Path) -> list[Path]:
        dirs = [self.runtime_dir]
        custom = [b if isinstance(b, str) else b.get("command") for b in self.store.get_state("builds") or []]
        for command in [self.settings.llama_server_bin, *self.installed_commands(), *custom]:
            if command:
                _, path = builds._executable(command)
                if path.is_file() and path.resolve().parent != exe.parent.resolve():
                    dirs.append(path.resolve().parent)
        dirs += [Path(p) / "nvidia" / d / "lib" for p in sys.path if p.endswith("site-packages")
                 for d in ("cuda_runtime", "cublas", "nccl")]
        for pattern in LIB_SEARCH:
            dirs += [p for p in _glob(pattern) if p.is_dir() and p.resolve() != exe.parent.resolve()]
        return dirs

    async def _wheel_libs(self, libs: list[str]) -> dict[str, Path]:
        """Fetch CUDA runtime libraries from NVIDIA's PyPI wheels."""
        out: dict[str, Path] = {}
        by_package: dict[str, list[str]] = {}
        for lib in libs:
            by_package.setdefault(WHEELS[lib], []).append(lib)
        async with self._client(timeout=120.0) as c:
            for package, wanted in by_package.items():
                self.job = {**self.job, "state": "installing", "detail": f"downloading {package} from PyPI"}
                r = await c.get(f"{self.settings.pypi_url.rstrip('/')}/pypi/{package}/json")
                if r.status_code != 200:
                    raise UpdateError(f"PyPI lookup of {package} failed: HTTP {r.status_code}")
                wheel = next((f for f in r.json().get("urls", []) if f["filename"].endswith(".whl")
                              and "manylinux" in f["filename"] and WHEEL_PLATFORM in f["filename"]), None)
                if wheel is None:
                    raise UpdateError(f"no {package} wheel for {WHEEL_PLATFORM} Linux on PyPI")
                target = self.runtime_dir / wheel["filename"]
                async with c.stream("GET", wheel["url"]) as resp:
                    if resp.status_code != 200:
                        raise UpdateError(f"download of {wheel['filename']} failed: HTTP {resp.status_code}")
                    with open(target, "wb") as f:
                        async for chunk in resp.aiter_bytes(1 << 20):
                            f.write(chunk)
                try:
                    expected = (wheel.get("digests") or {}).get("sha256")
                    if expected and await asyncio.to_thread(_sha256, target) != expected:
                        raise UpdateError(f"checksum mismatch for {wheel['filename']}")
                    out.update(await asyncio.to_thread(self._unpack_wheel, target, wanted))
                finally:
                    target.unlink(missing_ok=True)
        return out

    def _unpack_wheel(self, wheel: Path, wanted: list[str]) -> dict[str, Path]:
        out = {}
        with zipfile.ZipFile(wheel) as z:
            for member in z.namelist():
                name = Path(member).name
                if name in wanted and "/lib/" in member:
                    dest = self.runtime_dir / name
                    with z.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    out[name] = dest
        return out

    # --- switching builds --------------------------------------------------------------

    async def switch(self, tag: str | None, reason: str = "update") -> None:
        """Make an installed update (or, with None, ATLAS_LLAMA_SERVER_BIN) the standard build."""
        state = self.state
        old_tag = state.get("standard")
        old_command = standard_build(self.settings, self.store)
        new_command = self._installed()[tag]["path"] if tag else self.settings.llama_server_bin
        pinned = [p for p in state.get("pinned") or [] if self.store.get_preset(p["preset_id"])]
        # presets pinned to the build that becomes standard again can follow the standard build
        for entry in [p for p in pinned if p["build"] == new_command]:
            preset = self.store.get_preset(entry["preset_id"])
            if preset.get("binary") == new_command:
                self._set_preset_binary(preset, "")
            pinned.remove(entry)
        if old_command and new_command and old_command != new_command:
            pinned += await asyncio.to_thread(self._pin_incompatible, old_command, new_command)
        self._save(standard=tag, previous=old_tag, previous_command=old_command, pinned=pinned,
                   switched_at=time.time(), error=None)
        if reason == "update":
            self._prune()
        log.info("standard llama-server build is now %s (was %s)", tag or new_command, old_tag or old_command)
        if self.settings.build_updates == "apply":
            await self.apply_if_idle()

    def _pin_incompatible(self, old_command: str, new_command: str) -> list[dict]:
        new, old = builds.inspect(new_command), builds.inspect(old_command)
        pinned = []
        for preset in self.store.list_presets():
            if preset.get("binary"):
                continue
            path = Path(preset["model_path"])
            arch = models.describe_file(path).arch if path.is_file() else None
            extra = preset.get("extra_args", "")
            new_problems = builds.preset_warnings(new, arch, extra)
            if new_problems and old.runnable and not builds.preset_warnings(old, arch, extra):
                self._set_preset_binary(preset, old_command)
                pinned.append({"preset_id": preset["id"], "preset": preset["name"], "build": old_command,
                               "reason": "; ".join(new_problems)})
                log.warning("preset %r stays on %s: %s", preset["name"], old_command, "; ".join(new_problems))
        return pinned

    def _set_preset_binary(self, preset: dict, binary: str) -> None:
        data = {k: preset[k] for k in PresetConfig.model_fields if k in preset}
        self.store.save_preset(preset["id"], {**data, "binary": binary})
        if self.supervisor.preset and self.supervisor.preset.get("id") == preset["id"]:
            self.supervisor.preset = self.store.get_preset(preset["id"])

    async def roll_back(self, reason: str | None = None) -> None:
        """Return to the previous standard build and skip the current release."""
        state = self.state
        tag = state.get("standard")
        if not tag:
            raise UpdateError("the standard build is not an installed update")
        previous = state.get("previous")
        if previous and not Path((self._installed().get(previous) or {}).get("path", "")).is_file():
            previous = None
        skipped = sorted(set(state.get("skipped") or []) | {tag})
        # releases older than the one rolled back from are not installed automatically either
        floor = max(state.get("floor") or "", (self._installed().get(tag) or {}).get("published_at", ""))
        self._save(skipped=skipped, floor=floor)
        await self.switch(previous, reason="rollback")
        self._save(error=reason, previous=None)
        log.warning("rolled back from %s to %s%s", tag, previous or "the configured build",
                    f": {reason}" if reason else "")

    def unskip(self, tag: str) -> None:
        self._save(skipped=[t for t in self.state.get("skipped") or [] if t != tag], floor=None)

    def _prune(self) -> None:
        state = self.state
        keep = {state.get("standard"), state.get("previous")}
        used = {p.get("binary") for p in self.store.list_presets()}
        installed = self._installed()
        newest = sorted(installed, key=lambda t: installed[t].get("published_at", ""), reverse=True)
        keep |= set(newest[:KEEP_UPDATES])
        for tag in list(installed):
            if tag in keep or installed[tag]["path"] in used:
                continue
            shutil.rmtree(Path(installed[tag]["path"]).parent, ignore_errors=True)
            del installed[tag]
            log.info("removed old llama-server build %s", tag)
        self._save(installed=installed)

    # --- supervisor hooks --------------------------------------------------------------

    def _started(self, preset: dict, command: str) -> None:
        tag = self.state.get("standard")
        if tag and not preset.get("binary") and command == standard_build(self.settings, self.store):
            if self.state.get("verified") != tag:
                self._save(verified=tag)

    async def _start_failed(self, preset: dict, command: str, error: str) -> bool:
        """An untested update that cannot start the preset is rolled back. True: retry the start."""
        state = self.state
        tag = state.get("standard")
        if (not tag or preset.get("binary") or state.get("verified") == tag
                or command != standard_build(self.settings, self.store)):
            return False
        first_line = error.strip().splitlines()[0] if error.strip() else error
        await self.roll_back(f"llama-server {tag} failed to start {preset['name']}: {first_line}")
        return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()

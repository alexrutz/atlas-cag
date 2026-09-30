"""Find and inspect llama-server builds, so presets can use different (e.g. custom) builds safely."""

import mmap
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

ELF_MACHINES = {0x3E: "x86-64", 0xB7: "ARM64", 0x03: "x86", 0x28: "ARM", 0xF3: "RISC-V", 0x15: "PowerPC64"}
PE_MACHINES = {0x8664: "x86-64", 0xAA64: "ARM64", 0x14C: "x86"}
HOST_MACHINE = {"x86_64": "x86-64", "amd64": "x86-64", "aarch64": "ARM64", "arm64": "ARM64"}.get(
    platform.machine().lower(), platform.machine())

@dataclass
class BuildInfo:
    command: str  # as used on the command line (may include arguments)
    path: str  # resolved executable
    runnable: bool = False
    problem: str | None = None
    version: str | None = None  # e.g. "b11177"
    commit: str | None = None
    missing_libs: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        home = str(Path.home())
        where = self.path.replace(home, "~", 1) if self.path.startswith(home) else self.path
        return f"{self.version or 'unknown version'} · {where}"

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d["label"] = self.label
        d["flags"] = len(self.flags)  # the full list is only needed server-side
        return d


def command_words(command: str) -> list[str]:
    """Split a build command and expand ~ in its words."""
    return [os.path.expanduser(w) if w.startswith("~") else w for w in shlex.split(command or "")]


def _executable(command: str) -> tuple[list[str], Path]:
    words = command_words(command)
    exe = words[0] if words else ""
    if exe and "/" not in exe:
        exe = shutil.which(exe) or exe
    return words, Path(exe)


def _env_for(exe: Path) -> dict:
    env = os.environ.copy()
    lib_dir = str(exe.resolve().parent)
    env["LD_LIBRARY_PATH"] = lib_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    return env


def format_problem(exe: Path) -> str | None:
    try:
        head = exe.read_bytes()[:4096] if exe.stat().st_size < 4096 else open(exe, "rb").read(4096)
    except OSError as e:
        return f"cannot read: {e}"
    if head[:2] == b"MZ":
        pe = struct.unpack_from("<I", head, 0x3C)[0] if len(head) >= 0x40 else 0
        machine = struct.unpack_from("<H", head, pe + 4)[0] if pe + 6 <= len(head) else 0
        return f"Windows executable ({PE_MACHINES.get(machine, 'unknown CPU')}): cannot run here; use a Linux build"
    if head[:4] == b"\x7fELF":
        machine = ELF_MACHINES.get(struct.unpack_from("<H", head, 18)[0], "unknown CPU")
        if machine != HOST_MACHINE:
            return f"built for {machine}, but this machine is {HOST_MACHINE}"
        return None
    if head[:2] == b"#!":
        return None  # script wrapper
    return "not an executable format this system can run"


_cache: dict[tuple, BuildInfo] = {}


def _release_tag(exe: Path) -> str | None:
    """Release packages such as ai-dock's llama.cpp-cuda put a VERSION.txt next to the binary."""
    try:
        text = (exe.resolve().parent / "VERSION.txt").read_text(errors="replace")[:4096]
    except OSError:
        return None
    m = re.search(r"llama\.cpp version:\s*(\S+)", text)
    return m.group(1) if m else None
_VERSION = re.compile(r"version:\s*(\S+)\s*\(build\s+(\d+),\s*commit\s+(\w+)\)")
_FLAG_PART = re.compile(r"\s{2,}(?![\s-])")
_FLAG = re.compile(r"(?<![\w-])(-{1,2}[A-Za-z][\w-]*)")


def inspect(command: str) -> BuildInfo:
    """Check a llama-server command: format, libraries, version and supported flags (cached)."""
    words, exe = _executable(command)
    info = BuildInfo(command=command, path=str(exe))
    if not words or not exe.is_file():
        info.problem = "file not found"
        return info
    # the folder's mtime changes when libraries are added next to the binary
    key = (command, exe.stat().st_mtime, exe.resolve().parent.stat().st_mtime)
    if key in _cache:
        return _cache[key]
    info.problem = format_problem(exe)
    if info.problem is None and not os.access(exe, os.X_OK):
        info.problem = "file is not executable (chmod +x)"
    if info.problem is None:
        env = _env_for(exe)
        if shutil.which("ldd") and not open(exe, "rb").read(2) == b"#!":
            try:
                out = subprocess.run(["ldd", str(exe)], capture_output=True, text=True, timeout=15, env=env).stdout
                info.missing_libs = sorted({line.split("=>")[0].strip() for line in out.splitlines() if "not found" in line})
            except (OSError, subprocess.TimeoutExpired):
                pass
        if info.missing_libs:
            info.problem = "missing libraries: " + ", ".join(info.missing_libs)
        else:
            try:
                res = subprocess.run([str(exe), *words[1:], "--version"], capture_output=True, text=True,
                                     timeout=30, env=env)
                if m := _VERSION.search(res.stdout + res.stderr):
                    semver, build, info.commit = m.groups()
                    # shallow clones (e.g. CI release packages) count only one commit
                    info.version = f"b{build}" if int(build) > 1 else f"v{semver}"
                if tag := _release_tag(exe):
                    info.version = tag
                elif res.returncode != 0:
                    info.problem = (res.stderr or res.stdout).strip().splitlines()[-1][:300] if (res.stderr or res.stdout).strip() else f"exited with code {res.returncode}"
                help_out = subprocess.run([str(exe), *words[1:], "--help"], capture_output=True, text=True,
                                          timeout=30, env=env)
                flags = set()
                for line in (help_out.stdout + help_out.stderr).splitlines():
                    if line.startswith("-"):
                        flags.update(_FLAG.findall(_FLAG_PART.split(line, 1)[0]))
                info.flags = sorted(flags)
            except (OSError, subprocess.TimeoutExpired) as e:
                info.problem = f"cannot run: {e}"
    info.runnable = info.problem is None
    _cache[key] = info
    return info


_arch_cache: dict[tuple[str, str], bool | None] = {}


def supports_arch(info: BuildInfo, arch: str | None) -> bool | None:
    """Whether the build's libraries name the model architecture. None if it cannot be told.

    llama.cpp keeps its architecture names as C strings in libllama, so a build that has never
    heard of an architecture (e.g. an older release and a brand-new model) does not contain it.
    """
    if not arch or not info.runnable:
        return None
    exe = Path(info.path).resolve()
    key = (str(exe), arch)
    if key in _arch_cache:
        return _arch_cache[key]
    files = {p.resolve() for p in exe.parent.glob("*llama*.so*")} | {exe}
    files = {f for f in files if f.is_file() and f.stat().st_size > 0}
    has_lib = any("libllama" in f.name for f in files)
    needle = b"\0" + arch.encode() + b"\0"
    found = False
    for f in files:
        try:
            with open(f, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if mm.find(needle) != -1:
                    found = True
                    break
        except (OSError, ValueError):
            continue
    # without libllama next to the binary (static or unusual layout) a miss proves nothing
    result = True if found else (False if has_lib else None)
    _arch_cache[key] = result
    return result


def unknown_flags(info: BuildInfo, extra_args: str) -> list[str]:
    if not info.flags:
        return []
    try:
        args = shlex.split(extra_args or "")
    except ValueError:
        return []
    known = set(info.flags)
    return sorted({a.split("=", 1)[0] for a in args if a.startswith("-") and not a[1:2].isdigit()
                   and a.split("=", 1)[0] not in known})


def preset_warnings(info: BuildInfo, model_arch: str | None, extra_args: str) -> list[str]:
    """Problems of running a preset's model with a build, for display (not enforced)."""
    if not info.runnable:
        return [f"llama-server build: {info.problem}"]
    warnings = []
    if supports_arch(info, model_arch) is False:
        warnings.append(f"{info.version or 'This build'} does not seem to support the model architecture "
                        f"“{model_arch}”: use a newer or custom llama.cpp build.")
    if unknown := unknown_flags(info, extra_args):
        warnings.append(f"Not supported by {info.version or 'this build'}: {', '.join(unknown)}")
    return warnings

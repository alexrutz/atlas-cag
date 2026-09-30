"""llama-server build inspection."""

import os
import struct
import sys
from pathlib import Path

from atlas import builds

CLI = Path(__file__).parent / "fake_llama_cli.py"


def pe_file(path: Path, machine: int) -> Path:
    head = bytearray(512)
    head[:2] = b"MZ"
    struct.pack_into("<I", head, 0x3C, 0x80)
    head[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", head, 0x84, machine)
    path.write_bytes(bytes(head))
    path.chmod(0o755)
    return path


def elf_file(path: Path, machine: int) -> Path:
    head = bytearray(64)
    head[:4] = b"\x7fELF"
    struct.pack_into("<H", head, 18, machine)
    path.write_bytes(bytes(head))
    path.chmod(0o755)
    return path


def test_unusable_builds_are_explained(tmp_path):
    assert builds.inspect(str(tmp_path / "nope")).problem == "file not found"
    win = builds.inspect(str(pe_file(tmp_path / "llama-server.exe", 0xAA64)))
    assert not win.runnable and "Windows executable (ARM64)" in win.problem
    assert "Windows executable (x86-64)" in builds.inspect(str(pe_file(tmp_path / "x64.exe", 0x8664))).problem
    other_cpu = 0xB7 if builds.HOST_MACHINE != "ARM64" else 0x3E
    assert "built for" in builds.inspect(str(elf_file(tmp_path / "llama-server", other_cpu))).problem
    plain = tmp_path / "not-exec"
    plain.write_text("#!/bin/sh\n")
    assert "not executable" in builds.inspect(str(plain)).problem


def test_runnable_build_reports_version_and_flags():
    info = builds.inspect(f"{sys.executable} {CLI}")
    assert info.runnable, info.problem
    assert info.version == "b4242" and info.commit == "fake42"
    assert {"-m", "--model", "-np", "--parallel", "--slot-save-path", "-t", "--threads"} <= set(info.flags)
    assert builds.unknown_flags(info, "--threads 8 --lazy-mode on -ncmoe=4") == ["--lazy-mode", "-ncmoe"]
    assert builds.preset_warnings(info, None, "--lazy-mode on") == ["Not supported by b4242: --lazy-mode"]


def test_architecture_support_is_read_from_libllama(tmp_path):
    exe = tmp_path / "llama-server"
    exe.write_text("#!/bin/sh\necho 'version: 1 (build 7, commit abc)'\n")
    exe.chmod(0o755)
    (tmp_path / "libllama.so.0.1").write_bytes(b"\0llama\0qwen35\0lfm2\0")
    os.symlink("libllama.so.0.1", tmp_path / "libllama.so")
    info = builds.inspect(str(exe))
    assert info.runnable and info.version == "b7"
    assert builds.supports_arch(info, "qwen35") is True
    assert builds.supports_arch(info, "qwen4exp") is False
    assert builds.supports_arch(info, None) is None
    warning = builds.preset_warnings(info, "qwen4exp", "")[0]
    assert "does not seem to support" in warning and "qwen4exp" in warning

"""Automatic llama-server build updates from GitHub releases (a fake GitHub serves the packages)."""

import asyncio
import hashlib
import io
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response

from atlas import builds, updater as updater_module
from atlas.api import create_app
from atlas.config import Settings

from .gguf_writer import qwen35_like
from .helpers import serve_in_thread, wait_for
from .test_managed import activate, free_port, preset

CLI = Path(__file__).parent / "fake_llama_cli.py"
ASSET = f"cuda-12.8-{updater_module.ARCH}.tar.gz"


def package(tag: str, *cli_args: str) -> bytes:
    """A release archive like ai-dock's: cuda-12.8/{llama-server, VERSION.txt, libllama.so}."""
    script = f"#!/bin/sh\nexec {sys.executable} {CLI} --fake-version {tag.lstrip('v')} {' '.join(cli_args)} \"$@\"\n"
    files = {
        "cuda-12.8/llama-server": (script.encode(), 0o755),
        "cuda-12.8/VERSION.txt": (f"llama.cpp version: {tag}\nCUDA version: 12.8.1\n".encode(), 0o644),
        "cuda-12.8/libllama.so": (b"\0llama\0qwen35\0", 0o644),
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, (data, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeGitHub:
    def __init__(self):
        self.releases: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.app = FastAPI()
        self.url = ""

        @self.app.get("/repos/{owner}/{repo}/releases")
        async def releases(owner: str, repo: str):
            return self.releases

        @self.app.get("/download/{name}")
        async def download(name: str, request: Request):
            data = self.files[name]
            if r := request.headers.get("range"):
                start = int(r.removeprefix("bytes=").split("-")[0])
                return Response(data[start:], status_code=206)
            return Response(data)

    def publish(self, tag: str, *cli_args: str, digest: str | None = None) -> None:
        data = package(tag, *cli_args)
        name = f"llama.cpp-{tag}-{ASSET}"
        self.files[name] = data
        other = f"llama.cpp-{tag}-cuda-12.8-sparc.tar.gz"
        self.releases.insert(0, {
            "tag_name": tag, "name": f"llama.cpp {tag} with CUDA", "draft": False, "prerelease": False,
            "published_at": f"2026-09-{10 + len(self.releases):02d}T00:00:00Z", "html_url": f"https://x/{tag}",
            "assets": [
                {"name": other, "size": 1, "browser_download_url": f"{self.url}/download/{other}"},
                {"name": name, "size": len(data), "browser_download_url": f"{self.url}/download/{name}",
                 "digest": digest or "sha256:" + hashlib.sha256(data).hexdigest()},
            ],
        })


@pytest.fixture
def github():
    gh = FakeGitHub()
    gh.url, stop = serve_in_thread(gh.app)
    yield gh
    stop()


@pytest.fixture
async def managed_updates(tmp_path, github, request):
    models_dir = tmp_path / "models"
    settings = Settings(
        _env_file=None,
        llama_server_bin=f"{sys.executable} {CLI}",
        llama_port=free_port(),
        kv_dir=tmp_path / "kv",
        data_dir=tmp_path / "data",
        models_dirs=str(models_dir),
        scan_model_caches=False,
        llama_start_timeout_s=30,
        github_api=github.url,
        build_update_repo="ai-dock/llama.cpp-cuda",
        **getattr(request, "param", {"build_updates": "install"}),
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atlas",
                                     timeout=30) as client:
            client.app = app
            client.model = qwen35_like(models_dir / "model-a.gguf", "Model A")
            yield client


async def check(client) -> dict:
    r = await client.post("/api/builds/updates/check")
    assert r.status_code == 202, r.text
    return await wait_for(client, lambda u: u["job"]["state"] in ("done", "failed"), "/api/builds/updates", 30)


async def test_update_becomes_standard_and_pins_incompatible_presets(managed_updates, github):
    client = managed_updates
    configured = client.app.state.supervisor.settings.llama_server_bin
    plain = (await client.post("/api/presets", json=preset("plain", client.model))).json()
    threads = (await client.post("/api/presets", json=preset("threads", client.model,
                                                             extra_args="--threads 2"))).json()
    assert (await activate(client, plain["id"]))["ready"]

    github.publish("v0.4.0")
    github.publish("v0.5.0", "--fake-minimal-help")  # newest: no --threads flag
    updates = await check(client)
    assert updates["job"]["state"] == "done", updates
    assert updates["latest"]["tag"] == "v0.5.0" and updates["latest"]["asset"].endswith(ASSET)
    assert updates["standard_tag"] == "v0.5.0" and updates["standard"].endswith("v0.5.0/llama-server")
    assert [i["tag"] for i in updates["installed"]] == ["v0.5.0"]
    assert updates["installed"][0]["version"] == "v0.5.0"
    assert updates["restart_pending"], "the running preset still uses the old build"

    # the preset whose extra arguments the new build does not know stays on the old build
    assert updates["pinned"] == [{"preset_id": threads["id"], "preset": "threads", "build": configured,
                                  "reason": "Not supported by v0.5.0: --threads"}]
    stored = {p["id"]: p for p in (await client.get("/api/presets")).json()["presets"]}
    assert stored[threads["id"]]["binary"] == configured and stored[plain["id"]]["binary"] == ""
    assert stored[plain["id"]]["build"]["version"] == "v0.5.0"

    listing = (await client.get("/api/builds")).json()
    assert listing["default"] == updates["standard"] and listing["configured"] == configured
    assert any(b["update"] and b["default"] for b in listing["builds"])

    r = await client.post("/api/server/restart")
    assert r.status_code == 202
    await asyncio.sleep(0.3)
    await wait_for(client, lambda s: s["ready"], "/api/status", 30)
    updates = (await client.get("/api/builds/updates")).json()
    assert not updates["restart_pending"]
    assert client.app.state.supervisor.running_command == updates["standard"]
    assert client.app.state.store.get_state("build_updates")["verified"] == "v0.5.0"

    # nothing newer: a second check changes nothing
    updates = await check(client)
    assert updates["standard_tag"] == "v0.5.0" and len(updates["installed"]) == 1

    # rolling back returns to the configured build, skips v0.5.0 and unpins the preset
    r = await client.post("/api/builds/updates/rollback")
    assert r.status_code == 200, r.text
    updates = r.json()
    assert updates["standard"] == configured and updates["skipped"] == ["v0.5.0"] and updates["pinned"] == []
    assert client.app.state.store.get_preset(threads["id"])["binary"] == ""
    updates = await check(client)
    assert updates["latest"]["tag"] == "v0.4.0", "skipped releases are ignored"
    assert updates["standard_tag"] is None, "an older release than the one rolled back is not installed"


@pytest.mark.parametrize("managed_updates", [{"build_updates": "apply"}], indirect=True)
async def test_update_that_cannot_start_is_rolled_back(managed_updates, github):
    client = managed_updates
    configured = client.app.state.supervisor.settings.llama_server_bin
    p = (await client.post("/api/presets", json=preset("plain", client.model))).json()
    assert (await activate(client, p["id"]))["ready"]

    github.publish("v0.6.0", "--fake-fail-start")
    updates = await check(client)
    assert updates["job"]["state"] == "done", updates
    # "apply" restarts llama-server right away; the start fails, Atlas goes back and starts again
    status = await wait_for(client, lambda s: s["ready"] and s["server"]["build"] == configured, "/api/status", 30)
    assert status["server"]["state"] == "running"
    updates = (await client.get("/api/builds/updates")).json()
    assert updates["standard"] == configured and updates["skipped"] == ["v0.6.0"]
    assert "v0.6.0 failed to start plain" in updates["error"]
    assert not updates["restart_pending"]

    r = await client.post("/api/builds/updates/unskip", params={"tag": "v0.6.0"})
    assert r.json()["skipped"] == []


async def test_checksum_mismatch_installs_nothing(managed_updates, github):
    github.publish("v0.7.0", digest="sha256:" + "0" * 64)
    updates = await check(managed_updates)
    assert updates["job"]["state"] == "failed" and "checksum mismatch" in updates["job"]["error"]
    assert updates["installed"] == [] and updates["standard_tag"] is None
    assert "checksum" in updates["error"]
    builds_dir = managed_updates.app.state.updater.dir
    assert not any(p.name.endswith(".part") for p in builds_dir.iterdir())


async def test_missing_cuda_libraries_are_supplied(tmp_path, monkeypatch):
    """Libraries come from local folders first (preferring one folder for all), then from PyPI wheels."""
    wanted = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12", "libnccl.so.2"]
    exe_dir = tmp_path / "data" / "builds" / "v1"
    exe_dir.mkdir(parents=True)
    exe = exe_dir / "llama-server"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)

    def fake_inspect(command: str) -> builds.BuildInfo:
        missing = [lib for lib in wanted if not (Path(command).parent / lib).exists()]
        return builds.BuildInfo(command=command, path=command, runnable=not missing, missing_libs=missing,
                                problem="missing libraries: " + ", ".join(missing) if missing else None)

    monkeypatch.setattr(builds, "inspect", fake_inspect)
    elf = b"\x7fELF" + bytes(14) + (0x3E if builds.HOST_MACHINE == "x86-64" else 0xB7).to_bytes(2, "little") + bytes(44)
    other_build = tmp_path / "llama-b1-bin"  # an official release with cudart + cuBLAS
    other_build.mkdir()
    for lib in ("libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
        (other_build / lib).write_bytes(elf + lib.encode())
    lone = tmp_path / "venv-nvidia" / "cuda_runtime" / "lib"  # a lone cudart elsewhere
    lone.mkdir(parents=True)
    (lone / "libcudart.so.12").write_bytes(elf + b"lone")

    # PyPI serves the NCCL wheel
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("nvidia/nccl/lib/libnccl.so.2", elf + b"nccl")
        z.writestr("nvidia/nccl/include/nccl.h", "")
    wheel_bytes = wheel.getvalue()
    pypi = FastAPI()
    name = f"nvidia_nccl_cu12-2.28.0-py3-none-manylinux_2_18_{updater_module.WHEEL_PLATFORM}.whl"

    @pypi.get("/pypi/{package}/json")
    async def meta(package: str):
        assert package == "nvidia-nccl-cu12"
        return {"urls": [{"filename": "nvidia_nccl_cu12-2.28.0.tar.gz", "url": "x"},
                         {"filename": name, "url": f"{pypi_url}/files/{name}",
                          "digests": {"sha256": hashlib.sha256(wheel_bytes).hexdigest()}}]}

    @pypi.get("/files/{filename}")
    async def file(filename: str):
        return Response(wheel_bytes)

    pypi_url, stop = serve_in_thread(pypi)
    try:
        monkeypatch.setattr(updater_module, "LIB_SEARCH", [str(lone), str(tmp_path / "*")])
        monkeypatch.setattr(updater_module.sys, "path", [])
        settings = Settings(_env_file=None, data_dir=tmp_path / "data", pypi_url=pypi_url)

        class Sup:
            on_started = on_start_failed = None

        from atlas.store import Store
        up = updater_module.BuildUpdater(settings, Store(tmp_path / "db.sqlite"), Sup())
        info = await up.provide_runtime(exe)
    finally:
        stop()
    assert info.runnable, info.problem
    runtime = up.runtime_dir
    for lib in wanted:
        link = exe_dir / lib
        assert link.is_symlink() and link.resolve() == (runtime / lib).resolve()
    # all three from the folder that had them together, not the lone cudart
    assert (runtime / "libcudart.so.12").read_bytes().endswith(b"libcudart.so.12")
    assert (runtime / "libcudart.so.12").stat().st_ino == (other_build / "libcudart.so.12").stat().st_ino
    assert (runtime / "libnccl.so.2").read_bytes().endswith(b"nccl")
    assert not list(runtime.glob("*.whl"))


def test_find_libs_skips_foreign_architectures(tmp_path):
    foreign = tmp_path / "arm"
    foreign.mkdir()
    machine = 0xB7 if builds.HOST_MACHINE == "x86-64" else 0x3E
    (foreign / "libcudart.so.12").write_bytes(b"\x7fELF" + bytes(14) + machine.to_bytes(2, "little") + bytes(44))
    assert updater_module.find_libs(["libcudart.so.12"], [foreign]) == {}


def test_release_version_label(tmp_path):
    (tmp_path / "VERSION.txt").write_text("llama.cpp version: v0.5.0\n")
    exe = tmp_path / "llama-server"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {CLI} --fake-version 0.5.0 \"$@\"\n")
    exe.chmod(0o755)
    assert builds.inspect(str(exe)).version == "v0.5.0"
    (tmp_path / "VERSION.txt").unlink()
    time.sleep(0.01)
    assert builds.inspect(str(exe)).version == "v0.5.0-dev", "shallow builds are named by their version"

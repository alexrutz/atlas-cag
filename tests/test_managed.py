"""Managed mode: Atlas spawns llama-server (the fake, via tests/fake_llama_cli.py) from presets."""

import asyncio
import os
import signal
import socket
import sys
import time
from pathlib import Path

import httpx
import pytest

from atlas import supervisor as supervisor_module
from atlas.api import create_app
from atlas.config import Settings

from .gguf_writer import qwen35_like
from .helpers import add_text, query, wait_for

CLI = Path(__file__).parent / "fake_llama_cli.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def managed(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor_module, "CRASH_WINDOW_S", 60)
    models_dir = tmp_path / "models"
    model_a = qwen35_like(models_dir / "model-a.gguf", "Model A")
    model_b = qwen35_like(models_dir / "model-b.gguf", "Model B")
    settings = Settings(
        _env_file=None,
        llama_server_bin=f"{sys.executable} {CLI}",
        llama_port=free_port(),
        kv_dir=tmp_path / "kv",
        data_dir=tmp_path / "data",
        models_dirs=str(models_dir),
        scan_model_caches=False,
        llama_start_timeout_s=30,
        build_updates="off",
        max_question_tokens=256,
        max_answer_tokens=256,
        max_final_tokens=256,
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atlas",
                                     timeout=30) as client:
            client.app = app
            client.models = (model_a, model_b)
            yield client


def preset(name: str, model: Path, **kw) -> dict:
    return {"name": name, "model_path": str(model), "ctx_per_slot": 4096, "slots": 2, "kv_type": "q8_0", **kw}


async def activate(client, preset_id: str, timeout: float = 30) -> dict:
    r = await client.post(f"/api/presets/{preset_id}/activate")
    assert r.status_code == 202, r.text
    await asyncio.sleep(0.2)
    return await wait_for(client, lambda s: s["ready"] or (s["server"] or {}).get("state") == "failed",
                          "/api/status", timeout)


async def test_no_preset_means_no_server(managed):
    status = (await managed.get("/api/status")).json()
    assert status["mode"] == "managed" and not status["ready"]
    assert "preset" in status["message"].lower()


async def test_presets_start_switch_and_reuse_caches(managed):
    model_a, model_b = managed.models
    listing = (await managed.get("/api/models")).json()
    assert {m["name"] for m in listing["models"]} == {"Model A", "Model B"}

    a = (await managed.post("/api/presets", json=preset("A", model_a))).json()
    b = (await managed.post("/api/presets", json=preset("B", model_b, kv_type="f16"))).json()
    presets = (await managed.get("/api/presets")).json()["presets"]
    est = next(p for p in presets if p["id"] == a["id"])["estimate"]
    assert est["kv_cache"] > 0 and est["total"] > est["weights"]

    status = await activate(managed, a["id"])
    assert status["ready"], status
    assert status["engine"]["n_slots"] == 2 and status["engine"]["n_ctx_slot"] == 4096
    doc = await add_text(managed, "doc.txt", "The gearbox failed in April.")
    fp_a = doc["fingerprint"]
    store = managed.app.state.store
    built_at = store.get_cache(doc["id"], fp_a).updated_at

    status = await activate(managed, b["id"])
    assert status["ready"] and status["engine"]["fingerprint"] != fp_a
    docs = await wait_for(managed, lambda ds: ds[0]["status"] == "ready")
    assert docs[0]["fingerprint"] == status["engine"]["fingerprint"]
    assert (await query(managed, "What failed?", [doc["id"]]))[-1]["type"] == "done"

    status = await activate(managed, a["id"])
    docs = (await managed.get("/api/documents")).json()
    assert docs[0]["status"] == "ready" and docs[0]["fingerprint"] == fp_a
    assert store.get_cache(doc["id"], fp_a).updated_at == built_at, "cache for A must be reused, not rebuilt"
    assert (await query(managed, "What failed?", [doc["id"]]))[-1]["type"] == "done"

    server = (await managed.get("/api/server")).json()["supervisor"]
    assert server["state"] == "running" and "--no-kv-unified" in server["command"]
    assert any("load_model" in line for line in server["log"])


async def test_crashed_server_is_restarted(managed):
    a = (await managed.post("/api/presets", json=preset("A", managed.models[0]))).json()
    await activate(managed, a["id"])
    sup = managed.app.state.supervisor
    old_pid = sup.proc.pid
    os.kill(old_pid, signal.SIGKILL)
    deadline = time.time() + 30
    while not (sup.state == "running" and sup.proc and sup.proc.pid != old_pid and managed.app.state.engine.ready):
        assert time.time() < deadline, f"not restarted: {sup.state} {sup.error}"
        await asyncio.sleep(0.2)
    doc = await add_text(managed, "after-crash.txt", "Still works.")
    assert doc["status"] == "ready"


async def test_failed_start_is_reported(managed, tmp_path):
    broken = qwen35_like(tmp_path / "models" / "broken.gguf")
    p = (await managed.post("/api/presets", json=preset("Broken", broken))).json()
    status = await activate(managed, p["id"])
    assert status["server"]["state"] == "failed"
    assert "invalid magic" in status["server"]["error"]
    assert not status["ready"] and "failed to start" in status["message"]


async def test_preset_validation(managed):
    model = str(managed.models[0])
    cases = [
        ({"model_path": "/nope/missing.gguf"}, "not found"),
        ({"extra_args": "--port 9999 --threads 8"}, "--port"),
        ({"kv_type": "q4_0", "flash_attn": "off"}, "flash attention"),
        ({"gpu_layers": "lots"}, "gpu_layers"),
    ]
    for override, message in cases:
        r = await managed.post("/api/presets", json={**preset("X", Path(model)), **override})
        assert r.status_code == 422 and message in r.json()["detail"], (override, r.text)
    ok = await managed.post("/api/presets", json=preset("X", Path(model), extra_args="--threads 8"))
    assert ok.status_code == 200


async def test_per_preset_builds_and_canary_check(managed):
    """Same model and cache format with three builds: a compatible build reuses the caches, one
    that computes differently fails the canary's next-token check and gets its own caches."""
    model = managed.models[0]
    default = (await managed.post("/api/presets", json=preset("Default build", model))).json()
    same = (await managed.post("/api/presets", json=preset(
        "Other compatible build", model, binary=f"{sys.executable} {CLI} --fake-semantics default"))).json()
    odd = (await managed.post("/api/presets", json=preset(
        "Build that computes differently", model, binary=f"{sys.executable} {CLI} --fake-semantics custom"))).json()

    await activate(managed, default["id"])
    doc = await add_text(managed, "doc.txt", "The gearbox failed in April.")
    fp = doc["fingerprint"]
    store = managed.app.state.store
    built_at = store.get_cache(doc["id"], fp).updated_at

    status = await activate(managed, same["id"])
    assert status["ready"] and status["engine"]["fingerprint"] == fp
    assert store.get_cache(doc["id"], fp).updated_at == built_at, "compatible build must reuse caches"
    server = (await managed.get("/api/server")).json()["supervisor"]
    assert "--fake-semantics default" in server["command"]

    status = await activate(managed, odd["id"])
    assert status["ready"] and status["engine"]["fingerprint"] != fp, "canary mismatch must change the configuration"
    docs = await wait_for(managed, lambda ds: ds[0]["status"] == "ready")
    assert docs[0]["fingerprint"] == status["engine"]["fingerprint"]
    assert (await query(managed, "What failed?", [doc["id"]]))[-1]["type"] == "done"


async def test_builds_api_and_preset_build_validation(managed, tmp_path):
    from .test_builds import pe_file
    listing = (await managed.get("/api/builds")).json()
    default = next(b for b in listing["builds"] if b["default"])
    assert default["runnable"] and default["version"] == "b4242"

    win = pe_file(tmp_path / "llama-server.exe", 0xAA64)
    r = await managed.post("/api/presets", json=preset("Windows build", managed.models[0], binary=str(win)))
    assert r.status_code == 422 and "Windows executable" in r.json()["detail"]
    r = await managed.post("/api/presets", json=preset("Missing", managed.models[0], binary="/nope/llama-server"))
    assert r.status_code == 422 and "not found" in r.json()["detail"]

    assert (await managed.post("/api/builds", json={"command": str(win)})).status_code == 200
    listing = (await managed.get("/api/builds")).json()
    added = next(b for b in listing["builds"] if b["added"])
    assert not added["runnable"] and "Windows" in added["problem"]
    await managed.delete("/api/builds", params={"command": str(win)})
    assert not any(b["added"] for b in (await managed.get("/api/builds")).json()["builds"])

    est = (await managed.post("/api/presets/estimate", json={**preset("X", managed.models[0]),
                                                             "extra_args": "--lazy-mode on"})).json()
    assert est["warnings"] == ["Not supported by b4242: --lazy-mode"]

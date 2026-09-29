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

from .gguf_writer import qwen35_like, write_gguf
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
        build_on_model_change=True,
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


async def test_server_address_and_key_can_be_changed(managed, tmp_path):
    model_a, _ = managed.models
    a = (await managed.post("/api/presets", json=preset("A", model_a))).json()
    assert (await activate(managed, a["id"]))["ready"]
    doc = await add_text(managed, "doc.txt", "The gearbox failed in April.")
    first_port = (await managed.get("/api/server")).json()["supervisor"]["port"]

    port = free_port()
    r = await managed.put("/api/server/address", json={"host": "0.0.0.0", "port": port, "api_key": "s3cret-key"})
    assert r.status_code == 202, r.text
    assert r.json() == {"host": "0.0.0.0", "port": port, "url": f"http://127.0.0.1:{port}", "api_key_set": True,
                        "exposed": True, "restarting": True}
    await asyncio.sleep(0.3)
    status = await wait_for(managed, lambda s: s["ready"], "/api/status", 30)
    server = (await managed.get("/api/server")).json()
    sup = server["supervisor"]
    assert sup["listening"] == {"host": "0.0.0.0", "port": port} and server["llama_url"] == f"http://127.0.0.1:{port}"
    assert "--host 0.0.0.0" in sup["command"] and f"--port {port}" in sup["command"] and "--api-key-file" in sup["command"]
    assert "s3cret-key" not in sup["command"] and not any("s3cret-key" in line for line in sup["log"])
    key_file = tmp_path / "data" / "llama-server.key"
    assert key_file.read_text().strip() == "s3cret-key" and key_file.stat().st_mode & 0o777 == 0o600

    # other programs need the key; Atlas sends it, and the document cache is still used
    async with httpx.AsyncClient() as other:
        assert (await other.get(f"http://127.0.0.1:{port}/props")).status_code == 401
        assert (await other.get(f"http://127.0.0.1:{port}/props",
                                headers={"Authorization": "Bearer s3cret-key"})).status_code == 200
    done = (await query(managed, "What failed?", [doc["id"]]))[-1]
    assert done["type"] == "done" and done["stats"]["cache_misses"] == 0 and status["engine"]["fingerprint"] == doc["fingerprint"]

    # a port that is taken, Atlas's own port, a bad host
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        r = await managed.put("/api/server/address", json={"host": "127.0.0.1", "port": taken.getsockname()[1]})
        assert r.status_code == 409 and "in use" in r.json()["detail"]
    assert (await managed.put("/api/server/address", json={"host": "127.0.0.1", "port": 8000})).status_code == 422
    assert (await managed.put("/api/server/address", json={"host": "my host", "port": port})).status_code == 422

    # back to this computer only, without a key (null would keep it), on the port it had before,
    # which was just released and still has connections in TIME_WAIT
    r = await managed.put("/api/server/address", json={"host": "127.0.0.1", "port": first_port, "api_key": ""})
    assert r.status_code == 202 and r.json()["api_key_set"] is False
    await asyncio.sleep(0.3)
    await wait_for(managed, lambda s: s["ready"], "/api/status", 30)
    assert not key_file.exists()
    assert managed.app.state.store.get_state("llama_address") == {"host": "127.0.0.1", "port": first_port, "api_key": None}
    assert (await managed.get("/api/server")).json()["supervisor"]["listening"] == {"host": "127.0.0.1", "port": first_port}


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
        ({"draft_model": "/nope/draft.gguf"}, "draft model not found"),
        ({"extra_args": "-md /tmp/x.gguf"}, "-md"),
    ]
    for override, message in cases:
        r = await managed.post("/api/presets", json={**preset("X", Path(model)), **override})
        assert r.status_code == 422 and message in r.json()["detail"], (override, r.text)
    ok = await managed.post("/api/presets", json=preset("X", Path(model), extra_args="--threads 8"))
    assert ok.status_code == 200


async def test_draft_model_is_optional_and_keeps_the_caches(managed, tmp_path):
    model_a, model_b = managed.models
    draft = qwen35_like(tmp_path / "models" / "tiny-draft.gguf", "Tiny Draft")
    plain = (await managed.post("/api/presets", json=preset("Plain", model_a))).json()
    fast = (await managed.post("/api/presets", json=preset("With draft", model_a, draft_model=str(draft)))).json()
    assert plain["draft_model"] == "" and fast["draft_model"] == str(draft)

    await activate(managed, plain["id"])
    doc = await add_text(managed, "doc.txt", "The gearbox failed in April.")
    status = await activate(managed, fast["id"])
    assert status["ready"] and status["engine"]["fingerprint"] == doc["fingerprint"], "a draft model must not rebuild caches"
    command = (await managed.get("/api/server")).json()["supervisor"]["command"]
    assert f"--model-draft {draft}" in command and "--gpu-layers-draft all" in command
    assert "--cache-type-k-draft q8_0 --cache-type-v-draft q8_0" in command
    assert (await query(managed, "What failed?", [doc["id"]]))[-1]["stats"]["cache_misses"] == 0

    listed = {p["name"]: p for p in (await managed.get("/api/presets")).json()["presets"]}
    est, base = listed["With draft"]["estimate"], listed["Plain"]["estimate"]
    assert est["draft"] > est["draft_kv"] > 0 and est["total"] == base["total"] + est["draft"]
    assert listed["With draft"]["draft"]["name"] == "Tiny Draft" and not listed["With draft"]["warnings"]

    # experts on the CPU: -cmoe takes the place of --n-gpu-layers (llama.cpp then places the layers itself)
    cmoe = (await managed.post("/api/presets", json=preset("cmoe", model_a, cpu_moe=True))).json()
    assert cmoe["cpu_moe"] is True
    from atlas.supervisor import build_command
    with_cmoe, without = build_command(["llama-server"], cmoe, 1, tmp_path), build_command(["llama-server"], plain, 1, tmp_path)
    assert "-cmoe" in with_cmoe and "--n-gpu-layers" not in with_cmoe
    assert "-cmoe" not in without and without[without.index("--n-gpu-layers") + 1] == "all"

    # a draft with another vocabulary is flagged before llama-server refuses it
    other = write_gguf(tmp_path / "models" / "other-vocab.gguf", {
        "general.architecture": "llama", "general.name": "Other", "tokenizer.ggml.model": "llama",
        "tokenizer.ggml.tokens": ["b"] * 1500})
    r = await managed.post("/api/presets/estimate", json={**preset("X", model_a), "draft_model": str(other)})
    assert any("draft model will not work" in w for w in r.json()["warnings"]), r.json()["warnings"]


async def test_per_preset_builds_and_canary_check(managed):
    """Same model and cache format with three builds: a compatible build reuses the caches, one
    that computes differently fails the canary's next-token check and gets its own caches."""
    model = managed.models[0]
    r = await managed.patch("/api/settings", json={"build_on_model_change": False})  # the default
    assert r.status_code == 200
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

    # the same token with another probability (another evaluation path): still compatible
    noisy = (await managed.post("/api/presets", json=preset(
        "Same build, other evaluation path", model, binary=f"{sys.executable} {CLI} --fake-semantics noisy"))).json()
    status = await activate(managed, noisy["id"])
    assert status["ready"] and status["engine"]["fingerprint"] == fp, "a probability change must not rebuild caches"
    assert store.get_cache(doc["id"], fp).updated_at == built_at

    status = await activate(managed, odd["id"])
    assert status["ready"] and status["engine"]["fingerprint"] != fp, "canary mismatch must change the configuration"
    # nothing is rebuilt by itself after the switch: the document waits until it is built
    await asyncio.sleep(0.5)
    docs = (await managed.get("/api/documents")).json()
    assert docs[0]["status"] == "not_built"
    r = await managed.post("/api/documents/build-missing")
    assert r.json()["queued"] == [doc["id"]]
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


async def test_llama_cpp_fit_failure_is_reported(managed):
    """llama.cpp's startup check says the preset exceeds free VRAM: Windows would spill it to shared memory."""
    p = (await managed.post("/api/presets", json=preset("big", managed.models[0],
                                                        extra_args="--fake-overcommit verbose"))).json()
    status = await activate(managed, p["id"])
    assert status["ready"]
    await asyncio.sleep(0.3)
    server = (await managed.get("/api/server")).json()["supervisor"]
    assert "689 MiB more GPU memory" in server["fit_warning"] and "shared system memory" in server["fit_warning"]
    # at the default log level llama.cpp only prints the warning, without the amount
    p = (await managed.post("/api/presets", json=preset("big2", managed.models[0],
                                                        extra_args="--fake-overcommit quiet"))).json()
    await activate(managed, p["id"])
    await asyncio.sleep(0.3)
    assert "needs more GPU memory" in (await managed.get("/api/server")).json()["supervisor"]["fit_warning"]
    q = (await managed.post("/api/presets", json=preset("small", managed.models[0]))).json()
    await activate(managed, q["id"])
    assert (await managed.get("/api/server")).json()["supervisor"]["fit_warning"] is None

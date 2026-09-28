"""Collections, runtime settings, model discovery and downloads."""

import asyncio
import os

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from atlas import models
from atlas.downloads import Downloader, DownloadError

from .gguf_writer import qwen35_like, write_gguf
from .helpers import add_text, query, serve_in_thread, wait_for


# --- collections ------------------------------------------------------------------------

async def test_collections_group_documents_and_can_be_queried(atlas):
    hr = (await atlas.post("/api/collections", json={"name": "HR"})).json()
    fin = (await atlas.post("/api/collections", json={"name": "Finance"})).json()
    a = await add_text(atlas, "leave.txt", "Holidays: 30 days per year.", hr["id"])
    b = await add_text(atlas, "remote.txt", "Remote work: 3 days per week.", hr["id"])
    c = await add_text(atlas, "q3.txt", "Revenue was 48 million.", fin["id"])
    loose = await add_text(atlas, "loose.txt", "Unfiled note about holidays.")

    listing = {x["name"]: x["n_docs"] for x in (await atlas.get("/api/collections")).json()}
    assert listing == {"Finance": 1, "HR": 2}
    assert loose["collection_id"] is None and a["collection_id"] == hr["id"]

    # whole collection plus a single document
    events = await query(atlas, "How many holidays?", [c["id"]], [hr["id"]])
    plan = events[0]
    assert {t["doc_id"] for t in plan["targets"]} == {a["id"], b["id"], c["id"]}
    assert events[-1]["type"] == "done"

    # move, rename, delete (documents survive as unfiled unless asked otherwise)
    moved = (await atlas.patch(f"/api/documents/{loose['id']}", json={"collection_id": fin["id"]})).json()
    assert moved["collection_id"] == fin["id"]
    renamed = (await atlas.patch(f"/api/documents/{loose['id']}", json={"name": "note.txt"})).json()
    assert renamed["name"] == "note.txt" and renamed["collection_id"] == fin["id"]
    assert (await atlas.patch(f"/api/collections/{fin['id']}", json={"name": "Finance 2026"})).status_code == 200
    await atlas.delete(f"/api/collections/{hr['id']}")
    docs = {d["id"]: d for d in (await atlas.get("/api/documents")).json()}
    assert docs[a["id"]]["collection_id"] is None
    r = await atlas.delete(f"/api/collections/{fin['id']}", params={"delete_documents": "true"})
    assert set(r.json()["deleted_documents"]) == {c["id"], loose["id"]}
    assert {d["id"] for d in (await atlas.get("/api/documents")).json()} == {a["id"], b["id"]}


async def test_unknown_collection_is_rejected(atlas):
    r = await atlas.post("/api/documents/text", json={"name": "x.txt", "text": "x", "collection_id": "nope"})
    assert r.status_code == 404
    r = await atlas.post("/api/query", json={"question": "q", "collection_ids": ["nope"]})
    assert r.status_code == 404


# --- runtime settings -------------------------------------------------------------------

async def test_runtime_settings_persist_and_validate(atlas):
    s = (await atlas.get("/api/settings")).json()
    assert s["values"]["max_thinking_tokens"] == s["defaults"]["max_thinking_tokens"]
    assert "temperature" not in s["values"], "sampling belongs to presets"
    atlas.app.state.store.set_state("settings", {"temperature": 0.2})  # stored by an older version
    r = await atlas.patch("/api/settings", json={"max_thinking_tokens": 4096, "max_answer_tokens": 512})
    assert r.status_code == 200 and r.json()["values"]["max_thinking_tokens"] == 4096
    assert atlas.app.state.queries.settings.max_answer_tokens == 512
    assert (await atlas.patch("/api/settings", json={"max_thinking_tokens": -5})).status_code == 422
    assert (await atlas.patch("/api/settings", json={"temperature": 0.5})).status_code == 422
    r = await atlas.patch("/api/settings", json={"max_thinking_tokens": None})  # reset to default
    assert r.json()["values"]["max_thinking_tokens"] == s["defaults"]["max_thinking_tokens"]
    assert atlas.app.state.store.get_state("settings") == {"max_answer_tokens": 512}


async def test_system_prompt_change_is_a_new_configuration(atlas):
    doc = await add_text(atlas, "doc.txt", "Some content.")
    r = await atlas.patch("/api/settings", json={"system_prompt": "You are a terse assistant."})
    assert r.status_code == 200
    docs = await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["fingerprint"] != doc["fingerprint"])
    assert docs[0]["id"] == doc["id"]


def test_settings_expand_home(monkeypatch, tmp_path):
    from atlas.config import Settings
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings(_env_file=None, llama_server_bin="~/llama.cpp/build/bin/llama-server",
                 data_dir="~/atlas-data", kv_dir="~/atlas-data/kv", models_dirs="~/m1, /abs/m2")
    assert s.llama_server_bin == f"{tmp_path}/llama.cpp/build/bin/llama-server"
    assert s.data_dir == tmp_path / "atlas-data" and s.kv_dir == tmp_path / "atlas-data" / "kv"
    assert s.model_dirs[0] == tmp_path / "m1" and str(s.model_dirs[1]) == "/abs/m2"
    assert Settings(_env_file=None, models_dirs="rel/models").model_dirs[0].is_absolute()
    assert Settings(_env_file=None, llama_server_bin="  ").managed is False


# --- model discovery --------------------------------------------------------------------

def test_discovery_and_estimates(tmp_path, monkeypatch):
    hub = tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("LLAMA_CACHE", str(tmp_path / "llama-cache"))
    qwen35_like(tmp_path / "models" / "sub" / "own.gguf", "Own", padding=1000)
    # HF cache layout: snapshot entries are symlinks into blobs/
    blob = qwen35_like(hub / "models--org--repo" / "blobs" / "abc123", "From Hub")
    snap = hub / "models--org--repo" / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    os.symlink(blob, snap / "Repo-Q4_K_M.gguf")
    write_gguf(snap / "mmproj-F16.gguf", {"general.architecture": "clip"})  # vision projector: skipped
    write_gguf(tmp_path / "models" / "big-00002-of-00002.gguf", {"general.architecture": "llama"})  # 2nd shard
    (tmp_path / "models" / "garbage.gguf").write_bytes(b"not a gguf")

    found = {m.file: m for m in models.discover([tmp_path / "models"])}
    assert set(found) == {"own.gguf", "Repo-Q4_K_M.gguf", "garbage.gguf"}
    assert found["garbage.gguf"].error
    hub_model = found["Repo-Q4_K_M.gguf"]
    assert hub_model.source == "Hugging Face cache" and hub_model.repo == "org/repo"

    m = found["own.gguf"]
    assert m.arch == "qwen35" and m.quant == "Q4_K_M" and m.ctx_train == 262144
    assert m.n_layers == 32 and m.n_attn_layers == 8 and m.hybrid
    assert m.kv_bytes_per_token_f16 == 8 * 4 * (256 + 256) * 2  # 32 KiB, like Qwen3.5-9B
    assert m.recurrent_bytes_per_slot == 24 * 4096 * 128 * 4
    est = models.estimate(m, 262144, 1, "q8_0")
    assert est["kv_cache"] == int(32768 / 2 * 34 / 32 * 262144)  # ~4.25 GiB
    assert est["total"] == m.size_bytes + est["kv_cache"] + est["recurrent"]


def test_tensor_classes_and_offload_flags(tmp_path):
    """qwen4exp-like: routed experts, n-gram embeddings (lazy), indexer keys, conv state."""
    layers = 4
    tensors = {f"blk.{i}.ffn_{k}_exps.weight": 1024 for i in range(layers) for k in ("gate", "up", "down")}
    tensors.update({"per_layer_token_embd.weight": 5120, "token_embd.weight": 320, "blk.3.attn_q.weight": 224})
    path = write_gguf(tmp_path / "exp.gguf", {
        "general.architecture": "qwen4exp", "qwen4exp.block_count": layers, "qwen4exp.context_length": 262144,
        "qwen4exp.embedding_length": 256, "qwen4exp.attention.head_count": 4, "qwen4exp.attention.head_count_kv": 2,
        "qwen4exp.attention.key_length": 256, "qwen4exp.attention.value_length": 256,
        "qwen4exp.attention.indexer.key_length": 128, "qwen4exp.full_attention_interval": 4,
        "qwen4exp.ssm.inner_size": 64, "qwen4exp.ssm.state_size": 16, "qwen4exp.ssm.conv_kernel": 4,
        "qwen4exp.ssm.group_count": 2,
    }, tensors=tensors)
    m = models.describe_file(path)
    assert m.kv_bytes_per_token_f16 == 1 * (2 * 512 + 128) * 2
    assert m.recurrent_bytes_per_slot == 3 * (64 * 16 + 3 * (64 + 2 * 2 * 16)) * 4
    assert m.expert_bytes_by_layer == {i: 3072 for i in range(layers)}
    assert m.lazy_bytes == 5120 and m.input_bytes == 320

    all_cpu = models.estimate(m, 1000, 1, "f16", "-cmoe -lm mmap --lazy-mode on")
    assert all_cpu["ram"] == 4 * 3072 + 320 and all_cpu["ssd"] == 5120
    assert all_cpu["weights"] == m.size_bytes - all_cpu["ram"] - 5120
    assert all_cpu["total"] == all_cpu["weights"] + all_cpu["kv_cache"] + all_cpu["recurrent"]
    partial = models.estimate(m, 1000, 1, "f16", "--n-cpu-moe=2 -lzm off")
    assert partial["ram"] == 2 * 3072 + 320 + 5120 and partial["ssd"] == 0
    assert models.estimate(m, 1000, 1, "f16")["ram"] == 320 + 5120  # defaults: embeddings in RAM


# --- downloads --------------------------------------------------------------------------

@pytest.fixture
def fake_hub(tmp_path):
    files = {"Model-Q4_K_M.gguf": os.urandom(300_000),
             "Split-Q8_0-00001-of-00002.gguf": os.urandom(100_000),
             "Split-Q8_0-00002-of-00002.gguf": os.urandom(50_000)}
    app = FastAPI()
    hits: list[str] = []

    @app.get("/api/models/{owner}/{name}/tree/main")
    async def tree(owner: str, name: str):
        if name != "repo":
            return JSONResponse({"error": "not found"}, 404)
        return [{"type": "file", "path": k, "size": len(v), "lfs": {"size": len(v)}} for k, v in files.items()] + \
               [{"type": "file", "path": "README.md", "size": 10}, {"type": "file", "path": "mmproj-F16.gguf", "size": 5}]

    @app.get("/{owner}/{name}/resolve/main/{file}")
    async def resolve(owner: str, name: str, file: str, request: Request):
        data = files[file]
        hits.append(request.headers.get("range") or "full")
        if rng := request.headers.get("range"):
            start = int(rng.removeprefix("bytes=").split("-")[0])
            return Response(data[start:], status_code=206)
        return Response(data)

    url, stop = serve_in_thread(app)
    yield url, files, hits
    stop()


async def test_hub_listing_groups_shards(fake_hub, tmp_path):
    url, files, _ = fake_hub
    d = Downloader(url, tmp_path / "models")
    listing = await d.list_files("org/repo")
    assert [g["file"] for g in listing] == ["mmproj-F16.gguf", "Model-Q4_K_M.gguf", "Split-Q8_0-00001-of-00002.gguf"]
    assert [g["projector"] for g in listing] == [True, False, False]
    split = listing[2]
    assert split["size"] == 150_000 and len(split["files"]) == 2
    with pytest.raises(DownloadError):
        await d.list_files("org/missing")
    with pytest.raises(DownloadError):
        await d.list_files("not a repo")


async def test_download_resumes_and_completes(fake_hub, tmp_path):
    url, files, hits = fake_hub
    dest_root = tmp_path / "models"
    d = Downloader(url, dest_root)
    d.start()
    try:
        partial = dest_root / "org__repo" / "Model-Q4_K_M.gguf.part"
        partial.parent.mkdir(parents=True)
        partial.write_bytes(files["Model-Q4_K_M.gguf"][:100_000])
        job = await d.enqueue("org/repo", "Model-Q4_K_M.gguf")
        split = await d.enqueue("org/repo", "Split-Q8_0-00001-of-00002.gguf")
        for _ in range(200):
            if job.status == "done" and split.status == "done":
                break
            await asyncio.sleep(0.02)
        assert job.status == "done" and split.status == "done", (job.error, split.error)
        assert (dest_root / "org__repo" / "Model-Q4_K_M.gguf").read_bytes() == files["Model-Q4_K_M.gguf"]
        assert "bytes=100000-" in hits
        for name in ("Split-Q8_0-00001-of-00002.gguf", "Split-Q8_0-00002-of-00002.gguf"):
            assert (dest_root / "org__repo" / name).read_bytes() == files[name]
        assert job.done == job.total == 300_000
        found = {m.file for m in models.discover([dest_root], scan_caches=False)}
        assert found == {"Model-Q4_K_M.gguf", "Split-Q8_0-00001-of-00002.gguf"}
    finally:
        await d.stop()


def test_sliding_window_layers_only_cache_their_window(tmp_path):
    """Like Spark-X2.5: 36 layers, 3 of every 4 use a 512-token window, 4 KV heads of 256."""
    path = write_gguf(tmp_path / "spark.gguf", {
        "general.architecture": "spark2_5", "spark2_5.block_count": 36, "spark2_5.context_length": 1048576,
        "spark2_5.embedding_length": 2560, "spark2_5.attention.head_count": 16, "spark2_5.attention.head_count_kv": 4,
        "spark2_5.attention.key_length": 256, "spark2_5.attention.value_length": 256,
        "spark2_5.attention.sliding_window": 512,
        "spark2_5.attention.sliding_window_pattern": [True, True, True, False] * 9,
    })
    m = models.describe_file(path)
    assert m.kv_bytes_per_token_f16 == 36 * 4 * 512 * 2 and m.kv_swa_bytes_per_token_f16 == 27 * 4 * 512 * 2
    full = models.estimate(m, 524288, 1, "q8_0", swa_full=True)
    windowed = models.estimate(m, 524288, 1, "q8_0")
    assert full["kv_cache"] == int(36 * 4 * 512 * 34 / 32 * 524288)  # ~39 GiB
    per_token = 9 * 4 * 512 * 34 / 32
    assert windowed["kv_cache"] == int(per_token * 524288 + 27 * 4 * 512 * 34 / 32 * 1024)  # ~9.6 GiB + window
    assert windowed["kv_bytes_per_token"] == int(per_token)

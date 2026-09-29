"""End-to-end: real Atlas app against the fake llama-server (no GPU needed)."""

import asyncio
import json
import time

import httpx
import pytest

from atlas import ingest
from atlas.api import create_app, monitor_step
from atlas.config import Settings

from .conftest import atlas_settings
from .helpers import add_text, doc_kv_files, query, serve_in_thread, wait_for


async def test_single_document_restores_cache_and_appends_question(atlas, fake):
    doc = await add_text(atlas, "policy.txt", "Remote work is allowed three days per week.")
    assert doc["n_parts"] == 1
    assert len(doc_kv_files(fake)) == 1

    # ingestion was a prefill-only request
    assert fake.log[-1]["n_predict"] == 0

    events = await query(atlas, "How many remote days?", [doc["id"]])
    done = events[-1]
    assert done["type"] == "done", events
    assert done["answer"] == "Single answer."
    assert done["stats"]["tokens_restored"] == doc["n_tokens"]
    assert done["stats"]["cache_misses"] == 0

    # the whole document prefix was reused; only the question suffix was evaluated
    last = fake.log[-1]
    assert last["cache_n"] == doc["n_tokens"]
    assert last["n_prompt"] - last["cache_n"] < 300

    # generation details: per call and for the whole answer
    stats = next(e for e in events if e["type"] == "target" and e["status"] == "done")["stats"]
    for key in ("wait_ms", "restore_ms", "prompt_tps", "gen_tps", "ttft_ms", "first_answer_ms", "n_reasoning", "n_ctx",
                "draft_n", "draft_accepted", "wall_ms"):
        assert key in stats, key
    assert stats["n_ctx"] == done["stats"]["config"]["n_ctx_slot"] == 4096 and stats["ttft_ms"] <= stats["wall_ms"] and stats["n_reasoning"] == 0
    s = done["stats"]
    assert s["first_token_ms"] <= s["first_answer_ms"] <= s["total_ms"] and s["kv_bytes_read"] > 0
    assert s["config"]["n_slots"] == 2 and s["config"]["fingerprint"] == doc["fingerprint"]
    assert s["tokens_reasoning"] == 0 and s["gen_ms"] >= 0 and s["draft_n"] == 0


FILTER = pytest.mark.parametrize("atlas", [{"relevance_filter": True}], indirect=True)


async def test_default_synthesizes_every_answer(atlas, fake):
    a = await add_text(atlas, "a.txt", "The gearbox failed in April.")
    b = await add_text(atlas, "b.txt", "Holidays: 30 days per year.")
    events = await query(atlas, "What happened to the gearbox?", [a["id"], b["id"]])
    finished = {e["key"]: e for e in events if e["type"] == "target" and e["status"] in ("done", "irrelevant")}
    assert [e["status"] for e in finished.values()] == ["done", "done"]
    assert {e["coverage"] for e in finished.values()} == {None}
    done = events[-1]
    assert done["answer"] == "Synthesized from 2 findings [1]."
    assert done["stats"]["n_relevant"] == 2


@FILTER
async def test_long_document_is_split_and_mapped(atlas, fake):
    filler = "\n\n".join(f"Paragraph {i} talks about logistics and warehouses in general terms." for i in range(120))
    long_text = filler + "\n\nThe gearbox of truck NW-777 failed in April."
    long_doc = await add_text(atlas, "fleet.txt", long_text)
    other = await add_text(atlas, "hr.txt", "Holidays: 30 days per year.")
    assert long_doc["n_parts"] > 1

    events = await query(atlas, "What happened to the gearbox?", [long_doc["id"], other["id"]])
    plan = events[0]
    assert plan["mode"] == "map_reduce"
    assert len(plan["targets"]) == long_doc["n_parts"] + 1

    finished = {e["key"]: e for e in events if e["type"] == "target" and e["status"] in ("done", "irrelevant", "error")}
    assert len(finished) == len(plan["targets"])
    relevant = [e for e in finished.values() if e["status"] == "done"]
    assert len(relevant) == 1 and "gearbox" in relevant[0]["answer"]
    assert "COVERAGE" not in relevant[0]["answer"] and relevant[0]["coverage"] == "full"
    assert all(e["stats"]["cache_miss"] is False for e in finished.values())

    assert any(e["type"] == "synthesis" and e["stage"] == "final" for e in events)
    done = events[-1]
    assert done["type"] == "done"
    assert done["answer"] == "Synthesized from 1 findings [1]."
    assert done["stats"]["n_relevant"] == 1


@FILTER
async def test_nothing_relevant_short_circuits(atlas, fake):
    a = await add_text(atlas, "a.txt", "Apples are red.")
    b = await add_text(atlas, "b.txt", "Bananas are yellow.")
    n_calls = len(fake.log)
    events = await query(atlas, "Quarterly revenue figures?", [a["id"], b["id"]])
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"].startswith("None of the selected documents")
    assert len(fake.log) - n_calls == 2  # two map calls, no synthesis


async def test_delete_removes_kv_files(atlas, fake):
    doc = await add_text(atlas, "tmp.txt", "Temporary.")
    assert doc_kv_files(fake)
    r = await atlas.delete(f"/api/documents/{doc['id']}")
    assert r.status_code == 200
    assert not doc_kv_files(fake)
    r = await atlas.post("/api/query", json={"question": "x", "document_ids": [doc["id"]]})
    assert r.status_code == 404


async def test_duplicate_upload_is_detected(atlas):
    first = await add_text(atlas, "dup.txt", "Same content.")
    r = await atlas.post("/api/documents/text", json={"name": "dup2.txt", "text": "Same content."})
    assert r.json()["duplicate"] is True
    assert r.json()["document"]["id"] == first["id"]


async def test_model_switch_keeps_caches_per_configuration(atlas, fake):
    doc = await add_text(atlas, "doc.txt", "Some content about compliance.")
    engine, ingestor = atlas.app.state.engine, atlas.app.state.ingestor
    first_fp = engine.info.fingerprint
    prefills = lambda: sum(1 for e in fake.log if e["n_predict"] == 0)  # noqa: E731

    # a different model gets its own cache; the first one is kept
    fake.model_path = "/models/another-model.gguf"
    assert await engine.refresh()
    ingestor.reconcile()
    docs = await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["fingerprint"] != first_fp)
    assert docs[0]["id"] == doc["id"] and docs[0]["queryable"]
    assert len(doc_kv_files(fake)) == 2
    caches = (await atlas.get("/api/caches")).json()
    assert {c["fingerprint"] for c in caches["configs"]} == {first_fp, docs[0]["fingerprint"]}

    # switching back reuses the stored cache: no new prefill
    before = prefills()
    fake.model_path = "/models/fake-model.gguf"
    assert await engine.refresh()
    ingestor.reconcile()
    docs = (await atlas.get("/api/documents")).json()
    assert docs[0]["fingerprint"] == first_fp and docs[0]["status"] == "ready"
    assert (await query(atlas, "compliance?", [doc["id"]]))[-1]["type"] == "done"
    assert prefills() == before

    # caches of the inactive configuration can be deleted
    other = next(c for c in caches["configs"] if c["fingerprint"] != first_fp)
    r = await atlas.delete(f"/api/caches/{other['fingerprint']}")
    assert r.status_code == 200 and r.json()["freed_bytes"] > 0
    assert len(doc_kv_files(fake)) == 1
    assert (await atlas.delete(f"/api/caches/{first_fp}")).status_code == 409


async def test_missing_kv_file_is_reported_and_repaired(atlas, fake):
    doc = await add_text(atlas, "fragile.txt", "Content that will lose its cache file.")
    for f in doc_kv_files(fake):
        f.unlink()
    events = await query(atlas, "anything?", [doc["id"]])
    assert events[-1]["type"] == "error"
    # the document is flagged and automatically re-ingested
    await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["queryable"])
    assert (await query(atlas, "anything?", [doc["id"]]))[-1]["type"] == "done"


async def test_kv_format_change_rebuilds_every_document(atlas, fake):
    """A failed restore whose canary also fails means the whole KV store is stale."""
    a = await add_text(atlas, "a.txt", "Alpha content.")
    b = await add_text(atlas, "b.txt", "Beta content.")
    old_fp = a["fingerprint"]
    fake.kv_format = "q8_0"

    events = await query(atlas, "anything?", [a["id"]])
    assert events[-1]["type"] == "error"
    # both documents, not only the queried one, get rebuilt under a new fingerprint
    docs = await wait_for(atlas, lambda ds: all(d["status"] == "ready" and d["fingerprint"] != old_fp for d in ds))
    assert {d["id"] for d in docs} == {a["id"], b["id"]}
    assert (await query(atlas, "anything?", [a["id"], b["id"]]))[-1]["type"] == "done"


async def test_kv_format_change_detected_on_reconnect(atlas, fake):
    a = await add_text(atlas, "a.txt", "Alpha content.")
    fake.kv_format = "q8_0"
    engine = atlas.app.state.engine
    await engine.connect()  # what the monitor does when llama-server comes back
    atlas.app.state.ingestor.reconcile()
    docs = await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["fingerprint"] != a["fingerprint"])
    assert (await query(atlas, "anything?", [docs[0]["id"]]))[-1]["type"] == "done"


async def test_quick_restart_with_new_kv_format_is_detected(atlas, fake):
    """A restart faster than the health poll is caught via the instance id and the canary."""
    a = await add_text(atlas, "a.txt", "Alpha content.")
    fake.kv_format = "q8_0"
    fake.media_marker = "<__media_second__>"
    app = atlas.app
    await monitor_step(app.state.engine, app.state.ingestor)
    docs = await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["fingerprint"] != a["fingerprint"])
    assert (await query(atlas, "anything?", [docs[0]["id"]]))[-1]["type"] == "done"


async def test_ingestion_retries_after_llama_server_crash(atlas, fake, monkeypatch):
    monkeypatch.setattr(ingest, "RETRY_BASE_DELAY_S", 0.05)
    fake.fail_prefills = 2
    doc = await add_text(atlas, "resilient.txt", "Survives two crashes.")
    assert doc["status"] == "ready" and fake.fail_prefills == 0


async def test_orphaned_slot_files_are_swept_at_startup(atlas, fake):
    await add_text(atlas, "keep.txt", "Referenced content.")
    orphan = fake.kv_dir / ("atlas-" + "ab" * 16 + ".bin")
    orphan.write_text("{}")
    foreign = fake.kv_dir / "someone-elses-file.bin"
    foreign.write_text("{}")
    assert atlas.app.state.ingestor.sweep_orphans() == 1
    assert not orphan.exists() and foreign.exists()
    assert len(doc_kv_files(fake)) == 1


async def test_model_change_during_ingestion_rebuilds(atlas, fake):
    fake.prefill_delay = 0.2
    text = "\n\n".join(f"Paragraph {i} about warehouses and logistics operations." for i in range(150))
    r = await atlas.post("/api/documents/text", json={"name": "big.txt", "text": text})
    doc_id = r.json()["document"]["id"]
    await wait_for(atlas, lambda ds: ds[0]["status"] == "ingesting")

    engine, ingestor = atlas.app.state.engine, atlas.app.state.ingestor
    fake.model_path = "/models/swapped-mid-ingestion.gguf"
    assert await engine.refresh()
    ingestor.reconcile()
    new_fp = engine.info.fingerprint

    docs = await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["fingerprint"] == new_fp, timeout=20)
    assert docs[0]["id"] == doc_id and docs[0]["queryable"]
    assert len(doc_kv_files(fake)) == docs[0]["n_parts"]  # first build's files were replaced


async def test_transient_restore_errors_never_invalidate_caches(atlas, fake):
    """Only an explicit rejection of a slot file may invalidate caches; server or connection
    errors are retried or reported (regression: a transient error once bumped the KV epoch)."""
    doc = await add_text(atlas, "doc.txt", "Some content.")
    fp = doc["fingerprint"]
    engine = atlas.app.state.engine

    fake.fail_restores = 1  # one hiccup: retried
    await engine.probe_kv_dir()
    assert engine.info.kv_dir_ok and engine.info.fingerprint == fp

    fake.fail_restores = 3  # persistent: the probe fails, but nothing is invalidated
    await engine.probe_kv_dir()
    assert not engine.info.kv_dir_ok and engine.info.fingerprint == fp
    fake.fail_restores = 0
    engine.info.error = None
    await engine.probe_kv_dir()
    assert engine.info.kv_dir_ok

    fake.fail_restores = 1  # during a query: the query fails, the document stays ready
    events = await query(atlas, "anything?", [doc["id"]])
    assert events[-1]["type"] == "error"
    docs = (await atlas.get("/api/documents")).json()
    assert docs[0]["status"] == "ready" and docs[0]["fingerprint"] == fp
    assert engine.info.fingerprint == fp
    assert (await query(atlas, "anything?", [doc["id"]]))[-1]["type"] == "done"


async def test_auth(fake, tmp_path):
    settings = Settings(_env_file=None, llama_url=fake.url, kv_dir=fake.kv_dir, models_dirs=str(tmp_path / "models"),
                        data_dir=tmp_path / "authdata", api_keys="secret1, secret2")
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atlas") as c:
            assert (await c.get("/api/documents")).status_code == 401
            assert (await c.get("/api/documents", headers={"Authorization": "Bearer nope"})).status_code == 401
            assert (await c.get("/api/documents", headers={"Authorization": "Bearer secret2"})).status_code == 200
            assert (await c.get("/healthz")).status_code == 200


async def test_client_disconnect_cancels_upstream_generation(fake, tmp_path):
    """Regression: a disconnect must abort the llama-server streams, not just Atlas's tasks.

    Runs Atlas under real uvicorn because the bug only shows through Starlette's anyio-based
    disconnect handling (in-process ASGI transports don't reproduce it).
    """
    url, stop = serve_in_thread(create_app(atlas_settings(fake, tmp_path)))
    try:
        async with httpx.AsyncClient(base_url=url, timeout=30) as client:
            await wait_for(client, lambda s: s["ready"], "/api/status")
            a = await add_text(client, "a.txt", "The gearbox is described here.")
            b = await add_text(client, "b.txt", "The gearbox is also described here.")

            streaming: set[str] = set()
            body = {"question": "Explain slowly what the gearbox does", "document_ids": [a["id"], b["id"]]}
            async with client.stream("POST", "/api/query", json=body) as r:
                async for line in r.aiter_lines():
                    if '"target_delta"' in line:
                        streaming.add(json.loads(line[6:])["key"])
                        if len(streaming) == 2:
                            assert fake.active == 2, "both map streams should be running"
                            break

            deadline = time.time() + 3
            while fake.active and time.time() < deadline:
                await asyncio.sleep(0.05)
            assert fake.active == 0, "upstream generation kept running after the client left"
            assert fake.cancelled == 2
            status = (await client.get("/api/status")).json()
            assert status["pool"]["leases"] == []
    finally:
        stop()


async def test_sliding_window_restore_reuse_is_checked(atlas, fake):
    """Stock llama-server re-prefills restored caches of sliding-window models without --swa-full."""
    eng = atlas.app.state.engine
    eng.swa_window = 64
    await eng.probe_kv_dir()
    assert eng.info.swa_restore_ok is True
    fake.swa_restore_bug = True
    await eng.probe_kv_dir()
    assert eng.info.swa_restore_ok is False
    assert not list(fake.kv_dir.glob("atlas-swa-probe*")), "the probe file is removed"



def restart_with_slots(fake, n: int) -> None:
    """llama-server comes back with another --parallel (a new process: new media marker)."""
    fake.slots = {i: [] for i in range(n)}
    fake.media_marker = f"<__media_{n}_{time.time()}__>"


async def test_slot_count_change_keeps_caches_per_slot_count(atlas, fake):
    """Standard llama.cpp only loads slot files saved with the same number of slots. Changing the
    slot count must not throw the caches away: they stay valid for their slot count."""
    doc = await add_text(atlas, "doc.txt", "The gearbox failed in April.")
    s = atlas.app.state
    fp_two = s.engine.info.fingerprint

    restart_with_slots(fake, 3)
    await monitor_step(s.engine, s.ingestor, None, s.store)
    fp_three = s.engine.info.fingerprint
    assert fp_three != fp_two and s.engine.stream_ident == 3
    assert not (fake.kv_dir / "atlas-kv-epochs.json").exists(), "a slot-count change is not an incompatibility"
    assert s.store.get_cache(doc["id"], fp_two).status == "ready", "the 2-slot caches are kept"
    await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["fingerprint"] == fp_three)

    # back to 2 slots: the original caches are used again, nothing is rebuilt
    built = s.store.get_cache(doc["id"], fp_two).updated_at
    restart_with_slots(fake, 2)
    await monitor_step(s.engine, s.ingestor, None, s.store)
    assert s.engine.info.fingerprint == fp_two and s.engine.stream_ident is None
    docs = (await atlas.get("/api/documents")).json()
    assert docs[0]["status"] == "ready" and s.store.get_cache(doc["id"], fp_two).updated_at == built
    assert (await query(atlas, "What failed?", [doc["id"]]))[-1]["type"] == "done"


async def test_slot_count_change_with_a_stream_agnostic_build(atlas, fake):
    """With the patched loader slot files restore into any number of slots: nothing changes."""
    fake.strict_streams = False
    doc = await add_text(atlas, "doc.txt", "The gearbox failed in April.")
    s = atlas.app.state
    fp = s.engine.info.fingerprint
    restart_with_slots(fake, 4)
    await monitor_step(s.engine, s.ingestor, None, s.store)
    assert s.engine.info.fingerprint == fp and s.engine.stream_ident is None
    events = await query(atlas, "What failed?", [doc["id"]])
    assert events[-1]["type"] == "done" and events[-1]["stats"]["cache_misses"] == 0

"""Prefill only starts by itself for new documents, and can always be stopped."""

import time

import pytest

from .helpers import add_text, doc_kv_files, query, wait_for

OFF = {"build_on_model_change": False}  # the default: invalid caches wait for the user


def by_id(docs: list[dict]) -> dict:
    return {d["id"]: d for d in docs}


@pytest.mark.parametrize("atlas", [OFF], indirect=True)
async def test_invalid_caches_wait_for_the_user(atlas, fake):
    doc = await add_text(atlas, "fragile.txt", "Content that will lose its cache file.")  # new: built by itself
    prefills = lambda: sum(1 for e in fake.log if e.get("n_predict") == 0)
    before = prefills()

    # a cache file goes missing: the failed restore flags the document, nothing is rebuilt
    for f in doc_kv_files(fake):
        f.unlink()
    assert (await query(atlas, "anything?", [doc["id"]]))[-1]["type"] == "error"
    atlas.app.state.ingestor.reconcile()
    (d,) = (await atlas.get("/api/documents")).json()
    assert d["status"] == "stale" and not d["queryable"] and "could not be restored" in d["error"]
    assert prefills() == before and not atlas.app.state.ingestor.is_busy(doc["id"])

    # the Library's Build all repairs it
    r = await atlas.post("/api/documents/build-missing")
    assert r.json()["queued"] == [doc["id"]]
    await wait_for(atlas, lambda ds: ds[0]["status"] == "ready" and ds[0]["queryable"])


@pytest.mark.parametrize("atlas", [OFF], indirect=True)
async def test_model_change_during_prefill_stops_it(atlas, fake):
    fake.prefill_delay = 0.3
    text = "\n\n".join(f"Paragraph {i} about warehouses and logistics operations." for i in range(150))
    doc_id = (await atlas.post("/api/documents/text", json={"name": "big.txt", "text": text})).json()["document"]["id"]
    await wait_for(atlas, lambda ds: ds[0]["status"] == "ingesting")

    engine, ingestor = atlas.app.state.engine, atlas.app.state.ingestor
    fake.model_path = "/models/swapped-mid-ingestion.gguf"
    assert await engine.refresh()
    ingestor.reconcile()
    await wait_for(atlas, lambda ds: not ingestor.is_busy(doc_id), timeout=20)
    (d,) = (await atlas.get("/api/documents")).json()
    assert d["status"] == "not_built" and d["fingerprint"] is None  # waits for the new model's Build all


@pytest.mark.parametrize("atlas", [OFF], indirect=True)
async def test_prefill_can_be_stopped_in_the_queue_and_mid_prefill(atlas, fake):
    ready = await add_text(atlas, "ready.txt", "A document that is already built.")
    fake.prefill_delay = 30  # a long prefill
    long_id = (await atlas.post("/api/documents/text", json={"name": "long.txt", "text": "Slow content."})).json()["document"]["id"]
    waiting_id = (await atlas.post("/api/documents/text", json={"name": "waiting.txt", "text": "Queued content."})).json()["document"]["id"]
    assert (await atlas.post(f"/api/documents/{ready['id']}/reingest")).status_code == 200  # a rebuild, queued too
    docs = by_id(await wait_for(atlas, lambda ds: by_id(ds)[long_id]["status"] == "ingesting"))
    assert docs[waiting_id]["status"] == docs[ready["id"]]["status"] == "queued"

    # stopped while queued: a new document shows not built, a rebuild that had not begun keeps its cache
    assert (await atlas.post(f"/api/documents/{waiting_id}/stop")).json()["status"] == "not_built"
    assert (await atlas.post(f"/api/documents/{ready['id']}/stop")).json()["status"] == "ready"

    # stopped mid-prefill: the request is closed at once, nothing is kept
    t0 = time.monotonic()
    r = await atlas.post("/api/documents/stop-builds")
    assert r.json()["stopped"] == sorted([long_id, waiting_id, ready["id"]])
    docs = by_id(await wait_for(atlas, lambda ds: by_id(ds)[long_id]["status"] == "not_built", timeout=5))
    assert time.monotonic() - t0 < 5 and docs[long_id]["n_parts"] == 0
    assert docs[ready["id"]]["queryable"] and len(doc_kv_files(fake)) == 1  # only the untouched cache
    assert (await query(atlas, "anything?", [ready["id"]]))[-1]["type"] == "done"

    # nothing comes back by itself; building again works
    fake.prefill_delay = 0
    atlas.app.state.ingestor.reconcile()
    assert by_id((await atlas.get("/api/documents")).json())[long_id]["status"] == "not_built"
    await atlas.post(f"/api/documents/{long_id}/reingest")
    await wait_for(atlas, lambda ds: by_id(ds)[long_id]["status"] == "ready")

"""Shared test helpers."""

import asyncio
import json
import threading
import time

import uvicorn


def serve_in_thread(app):
    """Run an ASGI app under real uvicorn in a background thread; returns (url, stop)."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started:
        assert time.time() < deadline, "server did not start"
        time.sleep(0.02)

    def stop():
        server.should_exit = True
        thread.join(5)

    return f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}", stop


def doc_kv_files(fake) -> list:
    return [f for f in fake.kv_dir.glob("atlas-*.bin") if not f.name.startswith("atlas-canary")]


async def wait_for(client, predicate, path="/api/documents", timeout=15.0):
    deadline = time.time() + timeout
    while True:
        data = (await client.get(path)).json()
        if predicate(data):
            return data
        assert time.time() < deadline, f"timed out waiting on {path}: {data}"
        await asyncio.sleep(0.05)


async def add_text(client, name: str, text: str, collection_id: str | None = None) -> dict:
    r = await client.post("/api/documents/text", json={"name": name, "text": text, "collection_id": collection_id})
    assert r.status_code == 200, r.text
    doc_id = r.json()["document"]["id"]
    docs = await wait_for(client, lambda ds: any(d["id"] == doc_id and d["status"] in ("ready", "failed") for d in ds))
    doc = next(d for d in docs if d["id"] == doc_id)
    assert doc["status"] == "ready", doc
    return doc


async def query(client, question: str, doc_ids: list[str], collection_ids: list[str] | None = None) -> list[dict]:
    events = []
    body = {"question": question, "document_ids": doc_ids, "collection_ids": collection_ids or []}
    async with client.stream("POST", "/api/query", json=body) as r:
        assert r.status_code == 200, await r.aread()
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events

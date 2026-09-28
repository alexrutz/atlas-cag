from atlas.store import Part, Store, pack_tokens, unpack_tokens


def test_token_roundtrip():
    tokens = [0, 1, 151643, 2**31 - 1, 42]
    assert unpack_tokens(pack_tokens(tokens)) == tokens


def test_document_lifecycle(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    coll = store.create_collection("HR")
    doc = store.create_document("a.txt", "text/plain", "abc", 10, 10, coll.id)
    assert doc.collection_id == coll.id
    assert store.find_by_sha("abc").id == doc.id
    assert [d.id for d in store.list_documents(coll.id)] == [doc.id]

    store.set_cache(doc.id, "fp1", status="ingesting")
    store.add_part(Part(id="p1", doc_id=doc.id, fingerprint="fp1", idx=0, n_tokens=3, kv_file="f.bin",
                        kv_bytes=99, char_start=0, char_end=10, prefill_ms=1.0, prefix_tokens=[1, 2, 3]))
    store.set_cache(doc.id, "fp1", status="ready", n_tokens=3, n_parts=1, kv_bytes=99)
    store.set_cache(doc.id, "fp2", status="queued")
    assert store.get_cache(doc.id, "fp1").status == "ready"
    assert store.get_parts(doc.id, "fp1")[0].prefix_tokens == [1, 2, 3]
    assert store.get_parts(doc.id, "fp2") == []
    assert store.get_parts(doc.id, "fp1", with_tokens=False)[0].prefix_tokens == []
    assert store.get_prefix_tokens("p1") == [1, 2, 3]
    assert store.get_prefix_tokens("missing") is None
    assert [c.fingerprint for c in store.pending_caches_except("fp1")] == ["fp2"]

    store.delete_collection(coll.id)
    assert store.get_document(doc.id).collection_id is None  # unfiled, not deleted

    store.delete_document(doc.id)
    assert store.get_document(doc.id) is None
    assert store.get_parts(doc.id) == []  # cascaded
    assert store.get_cache(doc.id, "fp1") is None


def test_migration_from_v1(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "db.sqlite")
    db.executescript("""
        CREATE TABLE documents (id TEXT PRIMARY KEY, name TEXT NOT NULL, mime TEXT, sha256 TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, n_chars INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, error TEXT,
            n_tokens INTEGER NOT NULL DEFAULT 0, n_parts INTEGER NOT NULL DEFAULT 0,
            kv_bytes INTEGER NOT NULL DEFAULT 0, fingerprint TEXT, ingest_ms REAL,
            created_at REAL NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE parts (id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            idx INTEGER NOT NULL, n_tokens INTEGER NOT NULL, kv_file TEXT NOT NULL, kv_bytes INTEGER NOT NULL,
            char_start INTEGER NOT NULL, char_end INTEGER NOT NULL, prefill_ms REAL,
            prefix_tokens BLOB NOT NULL, created_at REAL NOT NULL);
        INSERT INTO documents VALUES ('d1', 'a.txt', NULL, 'sha', 1, 1, 'ready', NULL, 3, 1, 9, 'fpX', 1.0, 0, 0);
        INSERT INTO documents VALUES ('d2', 'b.txt', NULL, 'sha2', 1, 1, 'queued', NULL, 0, 0, 0, NULL, NULL, 0, 0);
        INSERT INTO parts VALUES ('p1', 'd1', 0, 3, 'atlas-x.bin', 9, 0, 1, 1.0, x'010000000200000003000000', 0);
    """)
    db.commit()
    db.close()
    store = Store(tmp_path / "db.sqlite")
    assert {d.id for d in store.list_documents()} == {"d1", "d2"}
    assert store.get_cache("d1", "fpX").status == "ready"
    assert store.get_cache("d2", "fpX") is None
    assert store.get_parts("d1", "fpX")[0].prefix_tokens == [1, 2, 3]
    Store(tmp_path / "db.sqlite")  # idempotent


def test_query_log(tmp_path):
    store = Store(tmp_path / "db.sqlite")
    store.log_query("q1", "why?", ["d1"], "single", "because", {"total_ms": 1}, None)
    [q] = store.list_queries()
    assert q["doc_ids"] == ["d1"] and q["stats"] == {"total_ms": 1}

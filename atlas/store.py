"""SQLite metadata store. KV caches live in llama-server's slot-save directory; this holds the index.

A document can have one KV cache per model configuration (fingerprint), so switching between
presets does not throw away caches built for another model.
"""

import json
import sqlite3
import sys
import threading
import time
import uuid
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    collection_id TEXT REFERENCES collections(id) ON DELETE SET NULL,
    mime          TEXT,
    sha256        TEXT NOT NULL,
    size_bytes    INTEGER NOT NULL,
    n_chars       INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    mode          TEXT NOT NULL DEFAULT 'text',  -- prefill: 'text' (extracted text) or 'visual' (page images)
    n_pages       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS documents_sha ON documents(sha256);

CREATE TABLE IF NOT EXISTS caches (
    doc_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    fingerprint  TEXT NOT NULL,
    status       TEXT NOT NULL,
    error        TEXT,
    n_tokens     INTEGER NOT NULL DEFAULT 0,
    n_parts      INTEGER NOT NULL DEFAULT 0,
    kv_bytes     INTEGER NOT NULL DEFAULT 0,
    ingest_ms    REAL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    variant      TEXT,  -- what was prefilled: NULL/'text', or 'visual:<projector>:<dpi>'
    PRIMARY KEY (doc_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS parts (
    id            TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    fingerprint   TEXT NOT NULL,
    idx           INTEGER NOT NULL,
    n_tokens      INTEGER NOT NULL,
    kv_file       TEXT NOT NULL,
    kv_bytes      INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    prefill_ms    REAL,
    prefix_tokens BLOB NOT NULL,
    created_at    REAL NOT NULL,
    -- visual parts: the prefix as a multimodal prompt string (char_start/char_end = page range)
    prefix_text   TEXT
);
CREATE INDEX IF NOT EXISTS parts_doc_fp ON parts(doc_id, fingerprint, idx);

CREATE TABLE IF NOT EXISTS configs (
    fingerprint  TEXT PRIMARY KEY,
    label        TEXT NOT NULL,
    details      TEXT,
    first_seen   REAL NOT NULL,
    last_used    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
    id          TEXT PRIMARY KEY,
    data        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

-- one row per question; a conversation's turns in created_at order
CREATE TABLE IF NOT EXISTS queries (
    id              TEXT PRIMARY KEY,
    created_at      REAL NOT NULL,
    question        TEXT NOT NULL,
    doc_ids         TEXT NOT NULL,
    mode            TEXT,
    answer          TEXT,
    stats           TEXT,
    error           TEXT,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    standalone      TEXT,  -- the follow-up rewritten as a standalone question, if it was
    detail          TEXT   -- JSON: per-document answers and reasoning, for showing the turn again
);
CREATE INDEX IF NOT EXISTS queries_conversation ON queries(conversation_id, created_at);
"""

EARLIER_QUESTIONS = "Earlier questions"
QUERY_COLUMNS = "id, created_at, question, doc_ids, mode, answer, stats, error, conversation_id, standalone, detail"

DOC_COLUMNS = "id, name, collection_id, mime, sha256, size_bytes, n_chars, created_at, updated_at, mode, n_pages"
CACHE_COLUMNS = ("doc_id, fingerprint, status, error, n_tokens, n_parts, kv_bytes, ingest_ms, created_at, updated_at, "
                 "variant")
PART_COLUMNS = ("id, doc_id, fingerprint, idx, n_tokens, kv_file, kv_bytes, char_start, char_end, prefill_ms, "
                "prefix_text")


def pack_tokens(tokens: list[int]) -> bytes:
    a = array("i", tokens)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def unpack_tokens(blob: bytes) -> list[int]:
    a = array("i")
    a.frombytes(blob)
    if sys.byteorder == "big":
        a.byteswap()
    return a.tolist()


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Collection:
    id: str
    name: str
    created_at: float

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Document:
    id: str
    name: str
    collection_id: str | None
    mime: str | None
    sha256: str
    size_bytes: int
    n_chars: int
    created_at: float
    updated_at: float
    mode: str = "text"
    n_pages: int = 0

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Cache:
    """A document's KV cache for one model configuration."""
    doc_id: str
    fingerprint: str
    status: str  # queued | ingesting | ready | stale | failed
    error: str | None
    n_tokens: int
    n_parts: int
    kv_bytes: int
    ingest_ms: float | None
    created_at: float
    updated_at: float
    variant: str | None = None

    @property
    def built_as(self) -> str:
        return self.variant or "text"

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Part:
    id: str
    doc_id: str
    fingerprint: str
    idx: int
    n_tokens: int
    kv_file: str
    kv_bytes: int
    char_start: int
    char_end: int
    prefill_ms: float | None
    prefix_text: str | None = field(default=None, repr=False)
    prefix_tokens: list[int] = field(default_factory=list, repr=False)

    @property
    def visual(self) -> bool:
        return self.prefix_text is not None

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d.pop("prefix_tokens")
        d.pop("prefix_text")
        d["visual"] = self.visual
        return d


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._migrate()
            self._db.executescript(SCHEMA)
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def close(self) -> None:
        self._db.close()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, params)

    def _columns(self, table: str) -> list[str]:
        return [r[1] for r in self._db.execute(f"PRAGMA table_info({table})")]

    def _migrate(self) -> None:
        self._migrate_v2()
        self._migrate_v3()
        self._migrate_v4()

    def _migrate_v4(self) -> None:
        """v4 adds visual prefill: document mode, cache variant, multimodal part prefixes."""
        added = [("documents", "mode", "TEXT NOT NULL DEFAULT 'text'"), ("documents", "n_pages", "INTEGER NOT NULL DEFAULT 0"),
                 ("caches", "variant", "TEXT"), ("parts", "prefix_text", "TEXT")]
        for table, column, decl in added:
            columns = self._columns(table)
            if columns and column not in columns:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _migrate_v3(self) -> None:
        """v3 groups questions into conversations; earlier questions go into one conversation."""
        columns = self._columns("queries")
        if not columns or "conversation_id" in columns:
            return
        self._db.execute("BEGIN")
        try:
            self._db.execute("CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                             "created_at REAL NOT NULL, updated_at REAL NOT NULL)")
            self._db.execute("ALTER TABLE queries ADD COLUMN conversation_id TEXT "
                             "REFERENCES conversations(id) ON DELETE CASCADE")
            self._db.execute("ALTER TABLE queries ADD COLUMN standalone TEXT")
            self._db.execute("ALTER TABLE queries ADD COLUMN detail TEXT")
            first, last = self._db.execute("SELECT MIN(created_at), MAX(created_at) FROM queries").fetchone()
            if first is not None:
                cid = new_id()
                self._db.execute("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                                 (cid, EARLIER_QUESTIONS, first, last))
                self._db.execute("UPDATE queries SET conversation_id = ?", (cid,))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _migrate_v2(self) -> None:
        """v1 kept one cache per document in the documents table; v2 keeps one per configuration."""
        if "status" not in self._columns("documents"):
            return
        self._db.execute("BEGIN")
        try:
            self._db.execute("CREATE TABLE IF NOT EXISTS collections ("
                             "id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at REAL NOT NULL)")
            self._db.execute("""
                CREATE TABLE IF NOT EXISTS caches (
                    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    fingerprint TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
                    n_tokens INTEGER NOT NULL DEFAULT 0, n_parts INTEGER NOT NULL DEFAULT 0,
                    kv_bytes INTEGER NOT NULL DEFAULT 0, ingest_ms REAL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY (doc_id, fingerprint))""")
            self._db.execute(
                "INSERT OR IGNORE INTO caches (doc_id, fingerprint, status, error, n_tokens, n_parts, kv_bytes, "
                "ingest_ms, created_at, updated_at) SELECT id, fingerprint, status, error, n_tokens, n_parts, "
                "kv_bytes, ingest_ms, created_at, updated_at FROM documents WHERE fingerprint IS NOT NULL"
            )
            if "fingerprint" not in self._columns("parts"):
                self._db.execute("ALTER TABLE parts ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''")
                self._db.execute(
                    "UPDATE parts SET fingerprint = COALESCE((SELECT fingerprint FROM documents d "
                    "WHERE d.id = parts.doc_id), '')"
                )
            self._db.execute("ALTER TABLE documents ADD COLUMN collection_id TEXT "
                             "REFERENCES collections(id) ON DELETE SET NULL")
            for col in ("status", "error", "n_tokens", "n_parts", "kv_bytes", "fingerprint", "ingest_ms"):
                self._db.execute(f"ALTER TABLE documents DROP COLUMN {col}")
            self._db.execute("DROP INDEX IF EXISTS parts_doc")
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    # --- app state ---------------------------------------------------------------------

    def get_state(self, key: str, default: Any = None) -> Any:
        row = self._exec("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_state(self, key: str, value: Any) -> None:
        if value is None:
            self._exec("DELETE FROM app_state WHERE key = ?", (key,))
        else:
            self._exec("INSERT OR REPLACE INTO app_state (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    # --- collections -------------------------------------------------------------------

    def create_collection(self, name: str) -> Collection:
        c = Collection(new_id(), name, time.time())
        self._exec("INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)", (c.id, c.name, c.created_at))
        return c

    def get_collection(self, collection_id: str) -> Collection | None:
        row = self._exec("SELECT id, name, created_at FROM collections WHERE id = ?", (collection_id,)).fetchone()
        return Collection(**row) if row else None

    def list_collections(self) -> list[Collection]:
        rows = self._exec("SELECT id, name, created_at FROM collections ORDER BY name COLLATE NOCASE").fetchall()
        return [Collection(**r) for r in rows]

    def rename_collection(self, collection_id: str, name: str) -> None:
        self._exec("UPDATE collections SET name = ? WHERE id = ?", (name, collection_id))

    def delete_collection(self, collection_id: str) -> None:
        self._exec("DELETE FROM collections WHERE id = ?", (collection_id,))

    # --- documents ---------------------------------------------------------------------

    def create_document(self, name: str, mime: str | None, sha256: str, size_bytes: int, n_chars: int,
                        collection_id: str | None = None, mode: str = "text", n_pages: int = 0) -> Document:
        now = time.time()
        doc_id = new_id()
        self._exec(
            "INSERT INTO documents (id, name, collection_id, mime, sha256, size_bytes, n_chars, created_at, "
            "updated_at, mode, n_pages) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc_id, name, collection_id, mime, sha256, size_bytes, n_chars, now, now, mode, n_pages),
        )
        return self.get_document(doc_id)  # type: ignore[return-value]

    def get_document(self, doc_id: str) -> Document | None:
        row = self._exec(f"SELECT {DOC_COLUMNS} FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return Document(**row) if row else None

    def find_by_sha(self, sha256: str) -> Document | None:
        row = self._exec(f"SELECT {DOC_COLUMNS} FROM documents WHERE sha256 = ? LIMIT 1", (sha256,)).fetchone()
        return Document(**row) if row else None

    def list_documents(self, collection_id: str | None = None) -> list[Document]:
        if collection_id is None:
            rows = self._exec(f"SELECT {DOC_COLUMNS} FROM documents ORDER BY created_at DESC").fetchall()
        else:
            rows = self._exec(f"SELECT {DOC_COLUMNS} FROM documents WHERE collection_id = ? "
                              "ORDER BY created_at DESC", (collection_id,)).fetchall()
        return [Document(**r) for r in rows]

    def update_document(self, doc_id: str, **fields) -> None:
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE documents SET {cols} WHERE id = ?", (*fields.values(), doc_id))

    def delete_document(self, doc_id: str) -> None:
        self._exec("DELETE FROM documents WHERE id = ?", (doc_id,))

    # --- caches ------------------------------------------------------------------------

    def get_cache(self, doc_id: str, fingerprint: str | None) -> Cache | None:
        if not fingerprint:
            return None
        row = self._exec(f"SELECT {CACHE_COLUMNS} FROM caches WHERE doc_id = ? AND fingerprint = ?",
                         (doc_id, fingerprint)).fetchone()
        return Cache(**row) if row else None

    def caches_for(self, fingerprint: str | None) -> dict[str, Cache]:
        if not fingerprint:
            return {}
        rows = self._exec(f"SELECT {CACHE_COLUMNS} FROM caches WHERE fingerprint = ?", (fingerprint,)).fetchall()
        return {r["doc_id"]: Cache(**r) for r in rows}

    def set_cache(self, doc_id: str, fingerprint: str, **fields) -> None:
        """Insert or update the (document, configuration) cache row."""
        now = time.time()
        with self._lock:
            exists = self._db.execute("SELECT 1 FROM caches WHERE doc_id = ? AND fingerprint = ?",
                                      (doc_id, fingerprint)).fetchone()
            if exists:
                fields["updated_at"] = now
                cols = ", ".join(f"{k} = ?" for k in fields)
                self._db.execute(f"UPDATE caches SET {cols} WHERE doc_id = ? AND fingerprint = ?",
                                 (*fields.values(), doc_id, fingerprint))
            elif self._db.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
                fields = {"status": "queued", **fields, "created_at": now, "updated_at": now}
                cols = ", ".join(fields)
                self._db.execute(f"INSERT INTO caches (doc_id, fingerprint, {cols}) VALUES (?, ?, "
                                 f"{', '.join('?' * len(fields))})", (doc_id, fingerprint, *fields.values()))

    def delete_cache(self, doc_id: str, fingerprint: str) -> None:
        self._exec("DELETE FROM caches WHERE doc_id = ? AND fingerprint = ?", (doc_id, fingerprint))

    def pending_caches_except(self, fingerprint: str | None) -> list[Cache]:
        rows = self._exec(f"SELECT {CACHE_COLUMNS} FROM caches WHERE status IN ('queued', 'ingesting') "
                          "AND fingerprint != ?", (fingerprint or "",)).fetchall()
        return [Cache(**r) for r in rows]

    def cache_summary(self) -> list[dict]:
        """Per configuration: documents, tokens and bytes of ready caches, plus labels."""
        rows = self._exec("""
            SELECT c.fingerprint, COUNT(*) AS n_docs, SUM(c.n_tokens) AS n_tokens,
                   SUM(c.kv_bytes) AS kv_bytes, SUM(c.status = 'ready') AS n_ready,
                   cf.label, cf.details, cf.last_used
            FROM caches c LEFT JOIN configs cf ON cf.fingerprint = c.fingerprint
            GROUP BY c.fingerprint ORDER BY cf.last_used DESC
        """).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d["details"]) if d["details"] else None
            out.append(d)
        return out

    def document_caches(self, doc_id: str) -> list[dict]:
        """A document's caches in every configuration, newest configuration first."""
        rows = self._exec(f"""
            SELECT {", ".join("c." + col.strip() for col in CACHE_COLUMNS.split(","))}, cf.label, cf.last_used
            FROM caches c LEFT JOIN configs cf ON cf.fingerprint = c.fingerprint
            WHERE c.doc_id = ? ORDER BY cf.last_used DESC
        """, (doc_id,)).fetchall()
        return [dict(r) for r in rows]

    # --- configs -----------------------------------------------------------------------

    def touch_config(self, fingerprint: str, label: str, details: dict) -> None:
        now = time.time()
        self._exec(
            "INSERT INTO configs (fingerprint, label, details, first_seen, last_used) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(fingerprint) DO UPDATE SET label = excluded.label, details = excluded.details, "
            "last_used = excluded.last_used",
            (fingerprint, label, json.dumps(details), now, now),
        )

    # --- parts -------------------------------------------------------------------------

    def add_part(self, part: Part) -> None:
        self._exec(
            "INSERT INTO parts (id, doc_id, fingerprint, idx, n_tokens, kv_file, kv_bytes, char_start, char_end, "
            "prefill_ms, prefix_tokens, created_at, prefix_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                part.id, part.doc_id, part.fingerprint, part.idx, part.n_tokens, part.kv_file, part.kv_bytes,
                part.char_start, part.char_end, part.prefill_ms, pack_tokens(part.prefix_tokens), time.time(),
                part.prefix_text,
            ),
        )

    def get_parts(self, doc_id: str, fingerprint: str | None = None, with_tokens: bool = True) -> list[Part]:
        """Parts of a document for one configuration, or for all configurations if fingerprint is None."""
        cols = PART_COLUMNS + (", prefix_tokens" if with_tokens else "")
        if fingerprint is None:
            rows = self._exec(f"SELECT {cols} FROM parts WHERE doc_id = ? ORDER BY fingerprint, idx",
                              (doc_id,)).fetchall()
        else:
            rows = self._exec(f"SELECT {cols} FROM parts WHERE doc_id = ? AND fingerprint = ? ORDER BY idx",
                              (doc_id, fingerprint)).fetchall()
        parts = []
        for r in rows:
            d = dict(r)
            d["prefix_tokens"] = unpack_tokens(d["prefix_tokens"]) if with_tokens else []
            parts.append(Part(**d))
        return parts

    def parts_for_fingerprint(self, fingerprint: str) -> list[Part]:
        rows = self._exec(f"SELECT {PART_COLUMNS} FROM parts WHERE fingerprint = ?", (fingerprint,)).fetchall()
        return [Part(**r) for r in rows]

    def all_kv_files(self) -> set[str]:
        return {r["kv_file"] for r in self._exec("SELECT kv_file FROM parts").fetchall()}

    def get_prefix_tokens(self, part_id: str) -> list[int] | None:
        row = self._exec("SELECT prefix_tokens FROM parts WHERE id = ?", (part_id,)).fetchone()
        return unpack_tokens(row["prefix_tokens"]) if row else None

    def delete_parts(self, doc_id: str, fingerprint: str | None = None) -> None:
        if fingerprint is None:
            self._exec("DELETE FROM parts WHERE doc_id = ?", (doc_id,))
        else:
            self._exec("DELETE FROM parts WHERE doc_id = ? AND fingerprint = ?", (doc_id, fingerprint))

    def delete_config_caches(self, fingerprint: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM parts WHERE fingerprint = ?", (fingerprint,))
            self._db.execute("DELETE FROM caches WHERE fingerprint = ?", (fingerprint,))
            self._db.execute("DELETE FROM configs WHERE fingerprint = ?", (fingerprint,))

    # --- presets -----------------------------------------------------------------------

    def list_presets(self) -> list[dict]:
        rows = self._exec("SELECT id, data, created_at, updated_at FROM presets ORDER BY created_at").fetchall()
        return [{"id": r["id"], **json.loads(r["data"]), "created_at": r["created_at"],
                 "updated_at": r["updated_at"]} for r in rows]

    def get_preset(self, preset_id: str) -> dict | None:
        row = self._exec("SELECT id, data, created_at, updated_at FROM presets WHERE id = ?", (preset_id,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], **json.loads(row["data"]), "created_at": row["created_at"],
                "updated_at": row["updated_at"]}

    def save_preset(self, preset_id: str | None, data: dict) -> str:
        now = time.time()
        if preset_id and self.get_preset(preset_id):
            self._exec("UPDATE presets SET data = ?, updated_at = ? WHERE id = ?", (json.dumps(data), now, preset_id))
            return preset_id
        preset_id = preset_id or new_id()
        self._exec("INSERT INTO presets (id, data, created_at, updated_at) VALUES (?, ?, ?, ?)",
                   (preset_id, json.dumps(data), now, now))
        return preset_id

    def delete_preset(self, preset_id: str) -> None:
        self._exec("DELETE FROM presets WHERE id = ?", (preset_id,))

    # --- query log ---------------------------------------------------------------------

    def log_query(self, query_id: str, question: str, doc_ids: list[str], mode: str | None,
                  answer: str | None, stats: dict | None, error: str | None, conversation_id: str | None = None,
                  standalone: str | None = None, detail: dict | None = None) -> None:
        now = time.time()
        with self._lock:
            if conversation_id and self.get_conversation(conversation_id) is None:
                conversation_id = None  # deleted while the question was being answered
            self._exec(
                f"INSERT OR REPLACE INTO queries ({QUERY_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (query_id, now, question, json.dumps(doc_ids), mode, answer,
                 json.dumps(stats) if stats is not None else None, error, conversation_id, standalone,
                 json.dumps(detail) if detail is not None else None),
            )
            if conversation_id:
                self._exec("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

    @staticmethod
    def _query_row(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["doc_ids"] = json.loads(d["doc_ids"])
        for key in ("stats", "detail"):
            d[key] = json.loads(d[key]) if d[key] else None
        return d

    def list_queries(self, limit: int = 50) -> list[dict]:
        rows = self._exec(f"SELECT {QUERY_COLUMNS} FROM queries ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._query_row(r) for r in rows]

    # --- conversations -----------------------------------------------------------------

    def create_conversation(self, title: str) -> dict:
        cid, now = new_id(), time.time()
        self._exec("INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                   (cid, title, now, now))
        return self.get_conversation(cid)

    def get_conversation(self, conversation_id: str) -> dict | None:
        row = self._exec("SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                         (conversation_id,)).fetchone()
        return dict(row) if row else None

    def list_conversations(self) -> list[dict]:
        rows = self._exec(
            "SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(q.id) AS n_turns, "
            "(SELECT question FROM queries WHERE conversation_id = c.id ORDER BY created_at DESC LIMIT 1) "
            "AS last_question FROM conversations c LEFT JOIN queries q ON q.conversation_id = c.id "
            "GROUP BY c.id ORDER BY c.updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        self._exec("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        with self._lock:
            self._exec("DELETE FROM queries WHERE conversation_id = ?", (conversation_id,))
            self._exec("DELETE FROM conversations WHERE id = ?", (conversation_id,))

    def conversation_turns(self, conversation_id: str, limit: int | None = None) -> list[dict]:
        """Turns oldest first; with `limit`, only the most recent ones."""
        rows = self._exec(
            f"SELECT {QUERY_COLUMNS} FROM queries WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
            (conversation_id, limit if limit is not None else -1),
        ).fetchall()
        return [self._query_row(r) for r in reversed(rows)]

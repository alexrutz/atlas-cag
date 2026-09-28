"""Ingestion: prefill each document part into a slot with n_predict = 0, then persist the slot to disk.

Caches are built for the active model configuration (fingerprint); caches of other
configurations are kept, so switching presets back does not rebuild them.
"""

import asyncio
import logging
import re
import time

from . import prompts
from .chunking import plan_parts
from .config import Settings
from .engine import Engine
from .llama import LlamaError
from .slots import PRIORITY_INGEST
from .store import Part, Store, new_id

log = logging.getLogger("atlas.ingest")

# Tokens reserved for the question instructions wrapped around the user's question.
QUESTION_BLOCK_OVERHEAD = 96
# Transient llama-server failures (restart, network) are retried this often before giving up.
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 5.0
# Part files are named atlas-<uuid hex>.bin; anything else in the directory is left alone.
ORPHAN_RE = re.compile(r"atlas-[0-9a-f]{32}\.bin")


class IngestCancelled(Exception):
    pass


class ConfigChanged(Exception):
    """The model configuration changed mid-document; parts must not mix configurations."""


class Ingestor:
    def __init__(self, engine: Engine, store: Store, settings: Settings):
        self.engine = engine
        self.store = store
        self.settings = settings
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.progress: dict[str, float] = {}
        self._pending: set[str] = set()
        self._cancelled: set[str] = set()
        self._retries: dict[str, int] = {}
        self._active: dict[str, str | None] = {}  # doc id -> fingerprint being built
        self._workers: set[asyncio.Task] = set()
        self._target = 0

    # --- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self.set_concurrency(self.engine.info.n_slots or 1)

    def set_concurrency(self, n_slots: int) -> None:
        """Run up to ingest_concurrency workers, leaving at least one slot for queries."""
        self._target = max(1, min(self.settings.ingest_concurrency, n_slots - 1 if n_slots > 1 else 1))
        while len(self._workers) < self._target:
            task = asyncio.create_task(self._worker(), name="ingest-worker")
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    async def stop(self) -> None:
        for w in list(self._workers):
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def enqueue(self, doc_id: str) -> None:
        self._cancelled.discard(doc_id)
        fp = self.engine.info.fingerprint
        if fp and doc_id not in self._active:
            self.store.set_cache(doc_id, fp, status="queued", error=None)
        if doc_id not in self._pending:
            self._pending.add(doc_id)
            self.queue.put_nowait(doc_id)

    def cancel(self, doc_id: str) -> None:
        self._cancelled.add(doc_id)

    def is_busy(self, doc_id: str) -> bool:
        return doc_id in self._pending or doc_id in self._active

    # --- cache files -------------------------------------------------------------------

    def remove_parts(self, doc_id: str, fingerprint: str | None = None) -> None:
        """Delete a document's parts (and slot files) for one configuration, or all of them."""
        for p in self.store.get_parts(doc_id, fingerprint, with_tokens=False):
            (self.settings.kv_dir / p.kv_file).unlink(missing_ok=True)
        self.store.delete_parts(doc_id, fingerprint)

    def drop_config(self, fingerprint: str) -> int:
        """Delete every cache built for one configuration; returns the bytes freed."""
        freed = 0
        for p in self.store.parts_for_fingerprint(fingerprint):
            path = self.settings.kv_dir / p.kv_file
            freed += path.stat().st_size if path.exists() else 0
            path.unlink(missing_ok=True)
        self.store.delete_config_caches(fingerprint)
        return freed

    def sweep_orphans(self) -> int:
        """Delete slot files no document references (left behind by a hard crash mid-ingestion).

        Only safe while no ingestion is running, i.e. at startup before jobs are resumed.
        """
        referenced = self.store.all_kv_files()
        removed = 0
        for f in self.settings.kv_dir.glob("atlas-*.bin"):
            if ORPHAN_RE.fullmatch(f.name) and f.name not in referenced:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            log.info("removed %d orphaned slot file(s)", removed)
        return removed

    # --- consistency -------------------------------------------------------------------

    def reconcile(self, startup: bool = False) -> None:
        """Bring the active configuration's caches in line with the library: build missing ones,
        resume interrupted jobs, and flag caches whose files are gone or no longer fit."""
        if startup:
            self.sweep_orphans()
        fp = self.engine.info.fingerprint
        if not fp:
            return
        # jobs queued for another configuration are abandoned; the loop below re-queues them
        for c in self.store.pending_caches_except(fp):
            if c.doc_id not in self._active:
                self.remove_parts(c.doc_id, c.fingerprint)
                self.store.delete_cache(c.doc_id, c.fingerprint)
        auto = self.settings.auto_build_caches
        caches = self.store.caches_for(fp)
        n_ctx = self.engine.info.n_ctx_slot
        for doc in self.store.list_documents():
            cache = caches.get(doc.id)
            if cache is None:
                if auto:
                    self.enqueue(doc.id)
                continue
            if cache.status in ("queued", "ingesting"):
                if startup or not self.is_busy(doc.id):
                    self.enqueue(doc.id)
                continue
            if cache.status == "failed":
                continue
            parts = self.store.get_parts(doc.id, fp, with_tokens=False)
            too_big = any(p.n_tokens + self.settings.max_answer_tokens > n_ctx for p in parts)
            missing = any(not (self.settings.kv_dir / p.kv_file).exists() for p in parts)
            valid = bool(parts) and not too_big and not missing
            status = cache.status
            if status == "ready" and not valid:
                reason = "KV cache file is missing" if missing else "slot context is smaller than a part"
                self.store.set_cache(doc.id, fp, status="stale", error=reason)
                status = "stale"
            elif status == "stale" and valid:
                self.store.set_cache(doc.id, fp, status="ready", error=None)
                status = "ready"
            if status == "stale" and auto and not self.is_busy(doc.id):
                self.enqueue(doc.id)

    # --- worker ------------------------------------------------------------------------

    async def _worker(self) -> None:
        while True:
            if len(self._workers) > self._target:
                return  # concurrency was lowered
            doc_id = await self.queue.get()
            self._pending.discard(doc_id)
            self._active[doc_id] = None
            try:
                await self._ingest(doc_id)
            except asyncio.CancelledError:
                raise
            except IngestCancelled:
                log.info("ingestion of %s cancelled", doc_id)
            except ConfigChanged:
                log.info("model configuration changed while ingesting %s; restarting it", doc_id)
                fp = self._active.pop(doc_id, None)
                if fp:
                    self.store.delete_cache(doc_id, fp)
                self.enqueue(doc_id)
            except LlamaError as e:
                if e.status is None and self._retries.get(doc_id, 0) < MAX_RETRIES:
                    # connection-level failure: llama-server restarting or unreachable
                    n = self._retries[doc_id] = self._retries.get(doc_id, 0) + 1
                    log.warning("ingestion of %s interrupted (%s); retry %d/%d", doc_id, e, n, MAX_RETRIES)
                    fp = self._active.get(doc_id)
                    if fp and self.store.get_document(doc_id):
                        self.store.set_cache(doc_id, fp, status="queued", error=f"retrying: {e}")
                    asyncio.get_running_loop().call_later(min(60, RETRY_BASE_DELAY_S * n), self._requeue, doc_id)
                else:
                    self._fail(doc_id, e)
            except Exception as e:
                self._fail(doc_id, e)
            finally:
                self._active.pop(doc_id, None)
                self.progress.pop(doc_id, None)
                self._cancelled.discard(doc_id)
                self.queue.task_done()

    def _fail(self, doc_id: str, e: Exception) -> None:
        log.exception("ingestion of %s failed", doc_id, exc_info=e)
        self._retries.pop(doc_id, None)
        fp = self._active.get(doc_id) or self.engine.info.fingerprint
        if fp and self.store.get_document(doc_id):
            self.store.set_cache(doc_id, fp, status="failed", error=str(e) or type(e).__name__)

    def _requeue(self, doc_id: str) -> None:
        if self.store.get_document(doc_id) and not self.is_busy(doc_id) and doc_id not in self._cancelled:
            self._pending.add(doc_id)
            self.queue.put_nowait(doc_id)

    def _check(self, doc_id: str) -> None:
        if doc_id in self._cancelled or self.store.get_document(doc_id) is None:
            raise IngestCancelled(doc_id)

    async def _wait_ready(self) -> None:
        while not self.engine.ready:
            await asyncio.sleep(0.5)

    async def _ingest(self, doc_id: str) -> None:
        await self._wait_ready()
        self._check(doc_id)
        doc = self.store.get_document(doc_id)
        assert doc is not None
        text = (self.settings.docs_dir / doc_id / "text.txt").read_text(encoding="utf-8")

        eng = self.engine
        fingerprint = eng.info.fingerprint
        assert fingerprint
        self._active[doc_id] = fingerprint
        self.remove_parts(doc_id, fingerprint)
        self.store.set_cache(doc_id, fingerprint, status="ingesting", error=None, n_tokens=0, n_parts=0,
                             kv_bytes=0)
        self.progress[doc_id] = 0.0
        started = time.perf_counter()

        n_ctx = eng.info.n_ctx_slot
        head, mid = await eng.prefix_overhead()
        tail, _ = await eng.question_suffix("", self.settings.enable_thinking)
        header = await eng.count(prompts.document_block(doc.name, 98, 99, ""))
        # Every query appends question + instructions + template tail + answer to the cached prefix.
        # (Thinking budget is not reserved: with thinking on, generation is capped by the free context.)
        reserve = self.settings.max_question_tokens + QUESTION_BLOCK_OVERHEAD + len(tail) + self.settings.max_answer_tokens
        prefix_limit = n_ctx - reserve
        budget = prefix_limit - len(head) - len(mid) - header
        if budget < 256:
            raise ValueError(
                f"slot context ({n_ctx} tokens) is too small: {reserve} tokens are reserved for question "
                "and answer. Give the preset more context per slot or reduce the answer budgets."
            )

        spans = await plan_parts(text, eng.count, budget, self.settings.part_overlap_tokens)
        n = len(spans)
        log.info("ingesting %s (%s): %d chars -> %d part(s), budget %d tokens/part", doc.name, doc_id, len(text), n, budget)

        created: list[Part] = []
        try:
            for k, (start, end) in enumerate(spans):
                await self._wait_ready()
                self._check(doc_id)
                prefix = await eng.document_prefix(doc.name, k, n, text[start:end])
                if len(prefix) > prefix_limit:
                    raise ValueError(f"part {k + 1} has {len(prefix)} tokens, limit is {prefix_limit}")

                def on_progress(done: int, total: int, k: int = k) -> None:
                    self.progress[doc_id] = (k + done / max(total, 1)) / n

                part_id = new_id()
                kv_file = f"atlas-{part_id}.bin"
                async with eng.pool.lease(PRIORITY_INGEST, f"ingest · {doc.name} ({k + 1}/{n})") as slot:
                    if eng.info.fingerprint != fingerprint:
                        raise ConfigChanged(doc_id)
                    t0 = time.perf_counter()
                    await eng.prefill(slot, prefix, on_progress)
                    prefill_ms = (time.perf_counter() - t0) * 1000
                    saved = await eng.llama.slot_save(slot, kv_file)
                part = Part(
                    id=part_id, doc_id=doc_id, fingerprint=fingerprint, idx=k, n_tokens=len(prefix),
                    kv_file=kv_file, kv_bytes=int(saved.get("n_written") or 0), char_start=start, char_end=end,
                    prefill_ms=round(prefill_ms, 1), prefix_tokens=prefix,
                )
                created.append(part)
                if saved.get("n_saved") != len(prefix):
                    raise LlamaError(f"slot saved {saved.get('n_saved')} tokens, expected {len(prefix)}")
                self._check(doc_id)
                self.store.add_part(part)
        except BaseException:
            for p in created:
                (self.settings.kv_dir / p.kv_file).unlink(missing_ok=True)
            if self.store.get_document(doc_id):
                self.store.delete_parts(doc_id, fingerprint)
            raise

        self.store.set_cache(
            doc_id, fingerprint,
            status="ready",
            error=None,
            n_tokens=sum(p.n_tokens for p in created),
            n_parts=n,
            kv_bytes=sum(p.kv_bytes for p in created),
            ingest_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        self._retries.pop(doc_id, None)
        log.info("ingested %s: %d tokens in %d part(s)", doc.name, sum(p.n_tokens for p in created), n)
        if fingerprint != eng.info.fingerprint and self.settings.auto_build_caches:
            # finished for the previous configuration (still valid there); build for the new one
            self._active.pop(doc_id, None)
            self.enqueue(doc_id)

"""Ingestion: prefill each document part into a slot with n_predict = 0, then persist the slot to disk.

Caches are built for the active model configuration (fingerprint); caches of other
configurations are kept, so switching presets back does not rebuild them.
"""

import asyncio
import json
import logging
import re
import time

from . import pages as page_images
from . import prompts
from .chunking import plan_parts
from .config import Settings
from .engine import MIN_ANSWER_ROOM, Engine
from .llama import LlamaError
from .slots import PRIORITY_INGEST
from .store import Document, Part, Store, new_id

log = logging.getLogger("atlas.ingest")

# Transient llama-server failures (restart, network) are retried this often before giving up.
MAX_RETRIES = 5
RETRY_BASE_DELAY_S = 5.0
# Part files are named atlas-<uuid hex>.bin; anything else in the directory is left alone.
ORPHAN_RE = re.compile(r"atlas-[0-9a-f]{32}\.bin")
NEEDS_VISION = ("visual prefill needs a vision model: add the model's vision projector (mmproj) to the preset, "
                "or switch the document to text prefill")
UNAVAILABLE = "visual:unavailable"


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

    def build_missing(self) -> list[str]:
        """Queue every document without a usable cache for the active configuration."""
        fp = self.engine.info.fingerprint
        if not fp:
            return []
        caches = self.store.caches_for(fp)
        queued = []
        for doc in self.store.list_documents():
            cache = caches.get(doc.id)
            wanted = self.variant_for(doc)
            if wanted is None or self.is_busy(doc.id):
                continue
            if cache is None or cache.status in ("stale", "failed") or cache.built_as != wanted:
                self.enqueue(doc.id)
                queued.append(doc.id)
        return queued

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

    def variant_for(self, doc: Document) -> str | None:
        """What the document's cache must be built from now; None if that is not possible."""
        if doc.mode == "visual":
            return self.engine.visual_variant(self.settings.visual_dpi)
        return "text"

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
        auto = self.settings.auto_build_caches  # repairs of this configuration's caches
        on_change = self.settings.build_on_model_change  # caches this configuration never had
        caches = self.store.caches_for(fp)
        n_ctx = self.engine.info.n_ctx_slot
        for doc in self.store.list_documents():
            cache = caches.get(doc.id)
            wanted = self.variant_for(doc)
            if cache is None:
                if on_change and wanted:
                    self.enqueue(doc.id)
                continue
            if cache.status in ("queued", "ingesting"):
                if startup or not self.is_busy(doc.id):
                    self.enqueue(doc.id)
                continue
            if wanted and cache.built_as != wanted:
                # the preset's projector or the page resolution changed (a document switched between
                # text and visual is queued when it is switched)
                if on_change and not self.is_busy(doc.id):
                    self.enqueue(doc.id)
                continue
            if cache.status == "failed" or wanted is None:
                continue  # a visual cache without the projector stays as is for when it is back
            parts = self.store.get_parts(doc.id, fp, with_tokens=False)
            too_big = any(p.n_tokens + (self.settings.max_answer_tokens or MIN_ANSWER_ROOM) > n_ctx for p in parts)
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
            # a missing file is repaired; parts larger than the slot come from a smaller slot size
            if status == "stale" and not self.is_busy(doc.id) and (on_change if too_big else auto):
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

    async def _prefix_limit(self) -> int:
        """Largest cached prefix: the slot minus the room kept for conversation, question and answer."""
        n_ctx = self.engine.info.n_ctx_slot
        reserve = max(self.engine.reserve_tokens(), MIN_ANSWER_ROOM)
        if n_ctx - reserve < 512:
            raise ValueError(f"slot context ({n_ctx} tokens) is too small: give the preset more context per slot")
        return n_ctx - reserve

    async def _ingest(self, doc_id: str) -> None:
        await self._wait_ready()
        self._check(doc_id)
        doc = self.store.get_document(doc_id)
        assert doc is not None

        eng = self.engine
        fingerprint = eng.info.fingerprint
        assert fingerprint
        variant = self.variant_for(doc) or UNAVAILABLE
        self._active[doc_id] = fingerprint
        self.remove_parts(doc_id, fingerprint)
        self.store.set_cache(doc_id, fingerprint, status="ingesting", error=None, n_tokens=0, n_parts=0,
                             kv_bytes=0, variant=variant)
        self.progress[doc_id] = 0.0
        started = time.perf_counter()
        if variant == UNAVAILABLE:
            raise ValueError(NEEDS_VISION)
        if doc.mode == "visual":
            created = await self._ingest_visual(doc, fingerprint)
        else:
            created = await self._ingest_text(doc, fingerprint)
        self._finish(doc, fingerprint, created, started)

    async def _ingest_text(self, doc: Document, fingerprint: str) -> list[Part]:
        doc_id = doc.id
        eng = self.engine
        text = (self.settings.docs_dir / doc_id / "text.txt").read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("this document has no extractable text: use visual prefill")
        head, mid = await eng.prefix_overhead()
        header = await eng.count(prompts.document_block(doc.name, 98, 99, ""))
        files = self._merged_files(doc_id)
        if files:  # room for the file line repeated at the start of a part
            header += max([await eng.count(prompts.file_line(f["name"]) + "\n") for f in files])
        prefix_limit = await self._prefix_limit()
        budget = prefix_limit - len(head) - len(mid) - header
        if budget < 256:
            raise ValueError(f"slot context ({eng.info.n_ctx_slot} tokens) is too small for document parts")

        spans = await plan_parts(text, eng.count, budget, self.settings.part_overlap_tokens)
        n = len(spans)
        log.info("ingesting %s (%s): %d chars -> %d part(s), budget %d tokens/part", doc.name, doc_id, len(text), n, budget)

        created: list[Part] = []
        try:
            for k, (start, end) in enumerate(spans):
                await self._wait_ready()
                self._check(doc_id)
                body = text[start:end]
                if files and k:
                    body = prompts.file_context(text, start) + body
                prefix = await eng.document_prefix(doc.name, k, n, body)
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
            self._discard(doc_id, fingerprint, created)
            raise
        return created

    def _discard(self, doc_id: str, fingerprint: str, created: list[Part]) -> None:
        for p in created:
            (self.settings.kv_dir / p.kv_file).unlink(missing_ok=True)
        if self.store.get_document(doc_id):
            self.store.delete_parts(doc_id, fingerprint)

    async def _ingest_visual(self, doc: Document, fingerprint: str) -> list[Part]:
        """Prefill page images: parts are runs of pages that fit a slot, sized by measuring."""
        doc_id = doc.id
        eng = self.engine
        doc_dir = self.settings.docs_dir / doc_id
        original = next(doc_dir.glob("original*"), None)
        if original is None:
            raise ValueError("the original file of this document is missing")
        pages = await asyncio.to_thread(page_images.ensure_pages, original, doc_dir, self.settings.visual_dpi)
        if not pages:
            raise ValueError("the document has no pages")
        if len(pages) != doc.n_pages:
            self.store.update_document(doc_id, n_pages=len(pages))
        prefix_limit = await self._prefix_limit()
        files = self._merged_files(doc_id)

        async with eng.pool.lease(PRIORITY_INGEST, f"ingest · {doc.name} (sizing)") as slot:
            if eng.info.fingerprint != fingerprint:
                raise ConfigChanged(doc_id)
            spans = await self._plan_pages(doc, pages, slot, prefix_limit, files)
        n = len(spans)
        log.info("ingesting %s (%s) visually: %d page(s) -> %d part(s), limit %d tokens/part",
                 doc.name, doc_id, len(pages), n, prefix_limit)

        created: list[Part] = []
        try:
            for k, (a, b) in enumerate(spans):
                await self._wait_ready()
                self._check(doc_id)
                text = await eng.visual_prefix(doc.name, k, n, list(range(a + 1, b + 1)), files)
                images = await asyncio.to_thread(lambda a=a, b=b: [p.read_bytes() for p in pages[a:b]])

                def on_progress(done: int, total: int, k: int = k) -> None:
                    self.progress[doc_id] = (k + done / max(total, 1)) / n

                part_id = new_id()
                kv_file = f"atlas-{part_id}.bin"
                async with eng.pool.lease(PRIORITY_INGEST, f"ingest · {doc.name} ({k + 1}/{n})") as slot:
                    if eng.info.fingerprint != fingerprint:
                        raise ConfigChanged(doc_id)
                    t0 = time.perf_counter()
                    final = await eng.prefill(slot, eng.multimodal(text, images), on_progress)
                    prefill_ms = (time.perf_counter() - t0) * 1000
                    saved = await eng.llama.slot_save(slot, kv_file)
                n_tokens = int(saved.get("n_saved") or 0)
                part = Part(
                    id=part_id, doc_id=doc_id, fingerprint=fingerprint, idx=k, n_tokens=n_tokens,
                    kv_file=kv_file, kv_bytes=int(saved.get("n_written") or 0), char_start=a, char_end=b,
                    prefill_ms=round(prefill_ms, 1), prefix_text=text,
                )
                created.append(part)
                if n_tokens != int(final.get("tokens_evaluated") or 0) or n_tokens > prefix_limit:
                    raise LlamaError(f"part {k + 1}: slot saved {n_tokens} tokens, prefill evaluated "
                                     f"{final.get('tokens_evaluated')} (limit {prefix_limit})")
                self._check(doc_id)
                self.store.add_part(part)
        except BaseException:
            self._discard(doc_id, fingerprint, created)
            raise
        return created

    def _merged_files(self, doc_id: str) -> list[dict]:
        """Where the PDFs of a merged document start ([{"name", "page"}]); empty for other documents."""
        try:
            return json.loads((self.settings.docs_dir / doc_id / "files.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    async def _plan_pages(self, doc: Document, pages: list, slot: int, limit: int,
                          files: list[dict] | None = None) -> list[tuple[int, int]]:
        """Split pages into runs whose prompt fits `limit` tokens, measuring instead of guessing:
        how many tokens an image takes depends on the vision encoder and the image size."""
        eng = self.engine
        sizes: dict[tuple[int, int], int] = {}

        async def measure(a: int, b: int) -> int:
            if (a, b) not in sizes:
                text = await eng.visual_prefix(doc.name, 98, 99, list(range(a + 1, b + 1)), files)
                images = await asyncio.to_thread(lambda: [p.read_bytes() for p in pages[a:b]])
                sizes[(a, b)] = await eng.measure(slot, text, images)
            return sizes[(a, b)]

        base = len(await eng.llama.tokenize(await eng.visual_prefix(doc.name, 98, 99, []), add_special=True,
                                            parse_special=True))
        per_page = max(1.0, await measure(0, 1) - base)
        spans: list[tuple[int, int]] = []
        a = 0
        while a < len(pages):
            b = min(len(pages), a + max(1, int((limit - base) / per_page * 0.97)))
            while True:
                tokens = await measure(a, b)
                if tokens <= limit:
                    break
                if b - a == 1:
                    raise ValueError(f"page {a + 1} alone takes {tokens} tokens, but a slot holds {limit} for a "
                                     "document part: lower the page resolution or give the preset more context")
                b = a + max(1, min(b - a - 1, int((b - a) * (limit - base) / max(1, tokens - base) * 0.97)))
            spans.append((a, b))
            per_page = max(1.0, (tokens - base) / (b - a))
            a = b
        return spans

    def _finish(self, doc: Document, fingerprint: str, created: list[Part], started: float) -> None:
        doc_id = doc.id
        eng = self.engine
        self.store.set_cache(
            doc_id, fingerprint,
            status="ready",
            error=None,
            n_tokens=sum(p.n_tokens for p in created),
            n_parts=len(created),
            kv_bytes=sum(p.kv_bytes for p in created),
            ingest_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        self._retries.pop(doc_id, None)
        log.info("ingested %s: %d tokens in %d part(s)", doc.name, sum(p.n_tokens for p in created), len(created))
        if fingerprint != eng.info.fingerprint and self.settings.build_on_model_change:
            # finished for the previous configuration (still valid there); build for the new one
            self._active.pop(doc_id, None)
            self.enqueue(doc_id)

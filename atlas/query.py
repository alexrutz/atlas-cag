"""Query execution.

conversation         : the earlier questions and answers are sent as chat turns after the cached
                       document (never the documents or the model's thinking)
single document part : restore slot file -> append conversation + question -> stream answer
several parts / docs : map  - every part is answered individually and in parallel (bounded by slots)
                       reduce - the relevant answers are concatenated and the original question is
                                run against them to synthesize the final answer (hierarchically if
                                the findings don't fit one slot)
"""

import asyncio
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from . import evidence
from . import pages as page_images
from . import prompts
from .config import Settings
from .engine import MIN_ANSWER_ROOM, ChatSuffix, Engine, GenResult, cache_rejected
from .ingest import Ingestor
from .llama import LlamaError
from .slots import PRIORITY_QUERY
from .store import Document, Part, Store, new_id

log = logging.getLogger("atlas.query")

Emit = Callable[[dict], Awaitable[None]]

NOTHING_FOUND = "None of the selected documents contain information relevant to this question."
TITLE_CHARS = 80


class QueryError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Target:
    n: int  # 1-based citation number
    doc: Document
    part: Part
    n_parts: int
    dpi: int | None = None  # visual parts: resolution of the page images that were prefilled
    path: list[str] = field(default_factory=list)  # the collections (chapters) the document is in

    @property
    def key(self) -> str:
        return f"t{self.n}"

    @property
    def label(self) -> str:
        if self.n_parts == 1:
            return self.doc.name
        if self.part.visual:
            a, b = self.part.char_start + 1, self.part.char_end
            return f"{self.doc.name} (pages {a}–{b})" if b > a else f"{self.doc.name} (page {a})"
        return f"{self.doc.name} (part {self.part.idx + 1} of {self.n_parts})"

    def describe(self) -> dict:
        return {
            "key": self.key, "n": self.n, "doc_id": self.doc.id, "doc_name": self.doc.name,
            "part": self.part.idx + 1, "n_parts": self.n_parts, "n_tokens": self.part.n_tokens,
            "label": self.label, "visual": self.part.visual, "path": self.path,
        }


@dataclass
class Plan:
    id: str
    question: str  # what the documents are asked: the standalone rewrite of a follow-up
    thinking: bool
    targets: list[Target]
    doc_ids: list[str]
    fingerprint: str | None
    conversation: dict = field(default_factory=dict)
    history: list[tuple[str, str]] = field(default_factory=list)  # (question, answer), oldest first
    asked: str = ""  # the question as the user typed it
    suffixes: dict = field(default_factory=dict)  # (question block, oldest turns dropped) -> ChatSuffix
    synthesis_history: list[tuple[str, str]] = field(default_factory=list)
    history_sent: int | None = None  # fewest earlier turns that fit next to a document part

    @property
    def mode(self) -> str:
        return "single" if len(self.targets) == 1 else "map_reduce"


@dataclass
class Tally:
    restored: int = 0  # tokens loaded from slot files instead of being prefilled
    processed: int = 0  # prompt tokens actually evaluated
    generated: int = 0
    restore_ms: float = 0.0
    cache_misses: int = 0
    truncated: int = 0  # generations cut off by the token limit
    llm_calls: int = 0
    reasoning: int = 0  # generated tokens spent thinking (≈)
    prompt_ms: float = 0.0
    gen_ms: float = 0.0
    wait_ms: float = 0.0  # waiting for a free slot
    kv_bytes: int = 0  # read from slot files
    draft_n: int = 0
    draft_accepted: int = 0
    extra: dict = field(default_factory=dict)

    def add(self, res: GenResult) -> None:
        self.processed += res.n_processed
        self.generated += res.n_gen
        self.reasoning += min(res.n_reasoning, res.n_gen)
        self.prompt_ms += res.prompt_ms
        self.gen_ms += res.gen_ms
        self.draft_n += res.draft_n
        self.draft_accepted += res.draft_accepted
        self.llm_calls += 1
        if res.stop_type == "limit":
            self.truncated += 1


class QueryService:
    def __init__(self, engine: Engine, store: Store, ingestor: Ingestor, settings: Settings):
        self.engine = engine
        self.store = store
        self.ingestor = ingestor
        self.settings = settings
        self.texts = evidence.TextCache()
        self._running: set[asyncio.Task] = set()

    # --- planning ----------------------------------------------------------------------

    async def prepare(self, question: str, doc_ids: list[str], thinking: bool | None,
                      collection_ids: list[str] | None = None, conversation_id: str | None = None) -> Plan:
        question = question.strip()
        if not question:
            raise QueryError("question is empty")
        conversation = self.store.get_conversation(conversation_id) if conversation_id else None
        if conversation_id and conversation is None:
            raise QueryError("conversation not found", 404)
        if not self.engine.ready:
            raise QueryError(self.engine.status_message or "llama-server is not ready", 503)
        fp = self.engine.info.fingerprint
        ids = list(doc_ids)
        for cid in collection_ids or []:
            if self.store.get_collection(cid) is None:
                raise QueryError(f"collection {cid} not found", 404)
            ready = self.store.caches_for(fp)
            ids += [d.id for d in self.store.documents_under(cid) if (c := ready.get(d.id)) and c.status == "ready"]
        doc_ids = list(dict.fromkeys(ids))
        if not doc_ids:
            raise QueryError("select at least one ready document")
        # library order: the parts of a long document and its chapters are answered and cited in order
        order = self.store.library_order()
        doc_ids.sort(key=lambda d: order.get(d, len(order)))

        targets: list[Target] = []
        for doc_id in doc_ids:
            doc = self.store.get_document(doc_id)
            if doc is None:
                raise QueryError(f"document {doc_id} not found", 404)
            cache = self.store.get_cache(doc_id, fp)
            wanted = self.ingestor.variant_for(doc)
            if cache is None or cache.status != "ready" or cache.built_as != wanted:
                status = cache.status if cache else "not built"
                if wanted is None:
                    status = "no vision projector loaded"
                elif cache and cache.status == "ready":
                    status = "built for a different prefill mode"
                raise QueryError(f"'{doc.name}' has no ready KV cache for the current model ({status})", 409)
            dpi = int(wanted.rsplit(":", 1)[1]) if wanted.startswith("visual:") else None
            path = [c.name for c in self.store.collection_path(doc.collection_id)]
            # prefix tokens are loaded per call while a slot is leased, bounding memory by slots
            parts = self.store.get_parts(doc_id, fp, with_tokens=False)
            for p in parts:
                targets.append(Target(len(targets) + 1, doc, p, len(parts), dpi, path))

        if self.settings.max_question_tokens:
            n_q = await self.engine.count(question)
            if n_q > self.settings.max_question_tokens:
                raise QueryError(f"question has {n_q} tokens, limit is {self.settings.max_question_tokens}")

        thinking = self.settings.enable_thinking if thinking is None else thinking
        history = []
        if conversation and self.settings.chat_history:
            turns = self.store.conversation_turns(conversation["id"])
            history = [(t["question"], t["answer"]) for t in turns if t["answer"] and not t["error"]]
        else:
            conversation = self.store.create_conversation(conversation_title(question))
        return Plan(new_id(), question, thinking, targets, doc_ids, self.engine.info.fingerprint,
                    conversation=conversation, history=history, asked=question)

    # --- execution ---------------------------------------------------------------------

    async def stream(self, plan: Plan) -> AsyncIterator[dict]:
        """Run the plan in a background task and yield its events. Closing the iterator cancels it."""
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        task = asyncio.create_task(self._run(plan, queue.put), name=f"query-{plan.id}")
        self._running.add(task)  # strong ref: the task must be able to finish its cleanup
        task.add_done_callback(self._running.discard)
        task.add_done_callback(lambda _: queue.put_nowait(None))
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except TimeoutError:
                    # keeps proxies from dropping the stream while waiting for a free slot
                    yield {"type": "ping"}
                    continue
                if event is None:
                    break
                yield event
        finally:
            # Cancel exactly once and never await the task through gather(). On a client
            # disconnect Starlette cancels us via anyio, which re-delivers cancellation on every
            # await; gather() would forward each one into the task and interrupt httpcore's
            # cleanup before it closes the upstream sockets, leaving llama-server generating.
            if not task.done():
                task.cancel()

    async def _run(self, plan: Plan, send: Emit) -> None:
        started = time.perf_counter()
        tally = Tally()
        answer: str | None = None
        error: str | None = None
        record = TurnRecord()

        async def emit(event: dict) -> None:
            record.observe(event)
            if event.get("type") == "delta":  # the final answer's stream (single answer or synthesis)
                ms = round((time.perf_counter() - started) * 1000, 1)
                tally.extra.setdefault("first_token_ms", ms)
                if event.get("channel") == "answer" and event.get("text", "").strip():
                    tally.extra.setdefault("first_answer_ms", ms)
            await send(event)

        try:
            await emit({"type": "plan", "query_id": plan.id, "mode": plan.mode,
                        "conversation": plan.conversation, "targets": [t.describe() for t in plan.targets],
                        "history": len(plan.history)})
            if plan.mode == "single":
                answer = await self._single(plan, plan.targets[0], emit, tally)
            else:
                answer = await self._map_reduce(plan, emit, tally)
            await emit({"type": "done", "answer": answer, "stats": self._stats(plan, tally, started)})
        except asyncio.CancelledError:
            error = "cancelled by client"
            raise
        except Exception as e:
            log.exception("query %s failed", plan.id)
            error = str(e) or type(e).__name__
            await emit({"type": "error", "message": error})
        finally:
            self.store.log_query(plan.id, plan.asked, plan.doc_ids, plan.mode, answer,
                                 self._stats(plan, tally, started), error, plan.conversation.get("id"),
                                 None, record.detail())

    async def _suffix(self, plan: Plan, t: Target, block: str) -> ChatSuffix:
        """The conversation for one document part: as many earlier turns as fit next to it."""
        eng = self.engine
        room = self.settings.max_answer_tokens or MIN_ANSWER_ROOM
        limit = eng.info.n_ctx_slot - t.part.n_tokens - room
        history, drop = plan.history, 0

        async def render(drop: int) -> ChatSuffix:
            key = (block, drop)
            if key not in plan.suffixes:
                plan.suffixes[key] = await eng.chat_suffix(history[drop:], block, plan.thinking)
            return plan.suffixes[key]

        suffix = await render(0)
        while suffix.n_tokens > limit and drop < len(history):
            over, freed = suffix.n_tokens - limit, 0
            while drop < len(history) and freed < over:  # drop the oldest turns that cover the excess
                freed += sum([await eng.count_cached(text) for text in history[drop]])
                drop += 1
            suffix = await render(drop)
        if suffix.n_tokens > limit:
            raise QueryError(f"the question does not fit next to '{t.label}' ({suffix.n_tokens} tokens, "
                             f"{max(0, limit)} free)", 400)
        plan.history_sent = min(plan.history_sent if plan.history_sent is not None else len(history),
                                len(history) - drop)
        return suffix

    def _stats(self, plan: Plan, tally: Tally, started: float) -> dict:
        return {
            "total_ms": round((time.perf_counter() - started) * 1000, 1),
            "n_targets": len(plan.targets),
            "tokens_restored": tally.restored,
            "tokens_processed": tally.processed,
            "tokens_generated": tally.generated,
            "restore_ms": round(tally.restore_ms, 1),
            "cache_misses": tally.cache_misses,
            "truncated": tally.truncated,
            "llm_calls": tally.llm_calls,
            "tokens_reasoning": tally.reasoning,
            "prompt_ms": round(tally.prompt_ms, 1),
            "gen_ms": round(tally.gen_ms, 1),
            "wait_ms": round(tally.wait_ms, 1),
            "kv_bytes_read": tally.kv_bytes,
            "draft_n": tally.draft_n,
            "draft_accepted": tally.draft_accepted,
            "config": {"model": self.engine.info.config_label or self.engine.info.model,
                       "fingerprint": plan.fingerprint, "build": self.engine.info.build,
                       "n_slots": self.engine.info.n_slots, "n_ctx_slot": self.engine.info.n_ctx_slot,
                       # False: this build re-evaluates restored documents of sliding-window models
                       "swa_restore_ok": self.engine.info.swa_restore_ok},
            "history_turns": len(plan.history) if plan.history_sent is None else plan.history_sent,
            "sampling": dict(self.engine.sampling),
            **tally.extra,
        }

    async def _answer_target(self, plan: Plan, t: Target, block: str, emit: Emit, tally: Tally,
                             on_piece: Callable[[str, str], Awaitable[None]]) -> tuple[GenResult, dict]:
        eng = self.engine
        suffix = await self._suffix(plan, t, block)
        layout = suffix.layout
        if t.part.visual:
            images = await asyncio.to_thread(self._page_bytes, t)
        await emit({"type": "target", "key": t.key, "status": "queued"})
        queued = time.perf_counter()
        async with eng.pool.lease(PRIORITY_QUERY, f"query · {t.label}") as slot:
            wait_ms = (time.perf_counter() - queued) * 1000
            if eng.info.fingerprint != plan.fingerprint:
                raise QueryError("the model was switched while this query was running", 409)
            prefix = self.store.get_prefix_tokens(t.part.id)
            if prefix is None:
                raise QueryError(f"'{t.doc.name}' was deleted or re-ingested during the query", 409)
            if t.part.visual:
                # the same page images again: llama-server matches them to the restored cache by hash
                prompt = eng.multimodal(t.part.prefix_text + suffix.text, images)
                n_prompt = t.part.n_tokens + suffix.n_tokens
            else:
                prompt = prefix + suffix.tokens
                n_prompt = len(prompt)
            await emit({"type": "target", "key": t.key, "status": "restoring", "slot": slot})
            t0 = time.perf_counter()
            try:
                restored = await eng.llama.slot_restore(slot, t.part.kv_file)
            except LlamaError as e:
                if not cache_rejected(e):
                    raise  # llama-server unreachable or failing: the cache is not at fault
                if await eng.kv_compatible(slot, plan.fingerprint):
                    self._flag_broken(t.doc, plan.fingerprint, f"KV cache could not be restored: {e}")
                else:
                    self.ingestor.reconcile()  # every stored cache is stale: rebuild them all
                raise
            restore_ms = (time.perf_counter() - t0) * 1000
            if restored.get("n_restored") != t.part.n_tokens:
                self._flag_broken(t.doc, plan.fingerprint, "KV cache file does not match the stored prefix")
                raise LlamaError(f"restored {restored.get('n_restored')} tokens, expected {t.part.n_tokens}")
            await emit({"type": "target", "key": t.key, "status": "generating", "slot": slot,
                        "restore_ms": round(restore_ms, 1)})
            res = await eng.generate(slot, prompt, layout, self.settings.max_answer_tokens, on_piece,
                                     n_prompt=n_prompt)

        tally.add(res)
        tally.restore_ms += restore_ms
        tally.wait_ms += wait_ms
        tally.kv_bytes += int(restored.get("n_read") or 0)
        tally.restored += min(res.n_cached, t.part.n_tokens)
        cache_miss = res.n_cached < t.part.n_tokens - 1
        if cache_miss:
            tally.cache_misses += 1
            log.warning("cache miss on %s: the restored cache was loaded but only %d of %d prefix tokens were reused%s",
                        t.label, res.n_cached, t.part.n_tokens,
                        " (this llama-server build re-evaluates restored documents of sliding-window models: use a "
                        "build with Atlas's SWA fix or --swa-full)" if eng.info.swa_restore_ok is False else
                        "; check llama-server flags")
        stats = {"slot": slot, "wait_ms": round(wait_ms, 1), "restore_ms": round(restore_ms, 1),
                 "kv_bytes": restored.get("n_read"), "cache_miss": cache_miss, "n_ctx": eng.info.n_ctx_slot,
                 **res.stats()}
        return res, stats

    async def _evidence(self, t: Target, answer: str, question: str) -> list[dict]:
        """Where the answer's quotes are in the document (preferring the part it was given)."""
        try:
            return await asyncio.to_thread(self._locate, t, answer, question)
        except Exception:
            log.exception("locating the quotes of %s failed", t.label)
            return []

    def _locate(self, t: Target, answer: str, question: str) -> list[dict]:
        path = self.settings.docs_dir / t.doc.id / "text.txt"
        if not path.exists():
            return []
        doc = self.texts.get(path)
        if not doc.text.strip():
            return []  # visual document without a text layer: nothing to match quotes against
        if t.part.visual:
            prefer = doc.pages_span(t.part.char_start + 1, t.part.char_end)
        else:
            prefer = (t.part.char_start, t.part.char_end)
        return doc.evidence(answer, question, prefer)

    def _page_bytes(self, t: Target) -> list[bytes]:
        folder = page_images.page_dir(self.settings.docs_dir / t.doc.id, t.dpi or self.settings.visual_dpi)
        pages = page_images.list_pages(folder)[t.part.char_start:t.part.char_end]
        if len(pages) != t.part.char_end - t.part.char_start:
            raise QueryError(f"page images of '{t.doc.name}' are missing: rebuild its cache", 409)
        return [p.read_bytes() for p in pages]

    def _flag_broken(self, doc: Document, fingerprint: str | None, reason: str) -> None:
        log.error("%s: %s", doc.name, reason)
        cache = self.store.get_cache(doc.id, fingerprint)
        if cache is None or cache.status != "ready" or fingerprint != self.engine.info.fingerprint:
            return  # deleted, already being rebuilt, or the model changed meanwhile
        self.store.set_cache(doc.id, fingerprint, status="stale", error=reason)
        if self.settings.auto_build_caches:
            self.ingestor.enqueue(doc.id)

    async def _single(self, plan: Plan, t: Target, emit: Emit, tally: Tally) -> str:
        async def on_piece(channel: str, text: str) -> None:
            await emit({"type": "delta", "channel": channel, "text": text})

        res, stats = await self._answer_target(plan, t, prompts.single_question_block(plan.question),
                                               emit, tally, on_piece)
        await emit({"type": "target", "key": t.key, "status": "done", "stats": stats,
                    "evidence": await self._evidence(t, res.answer, plan.question)})
        if not res.answer:
            raise QueryError("the model produced no answer (token limit reached?)", 502)
        return res.answer

    async def _map_one(self, plan: Plan, t: Target, emit: Emit, tally: Tally) -> tuple[str, str | None, str | None]:
        """Returns (status, answer, coverage); status is 'relevant', 'irrelevant' or 'error'."""
        async def on_piece(channel: str, text: str) -> None:
            await emit({"type": "target_delta", "key": t.key, "channel": channel, "text": text})

        try:
            block = prompts.map_question_block(plan.question, rate_coverage=self.settings.relevance_filter)
            res, stats = await self._answer_target(plan, t, block, emit, tally, on_piece)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("map step failed for %s: %s", t.label, e)
            await emit({"type": "target", "key": t.key, "status": "error", "error": str(e)})
            return "error", None, None
        answer, coverage = prompts.split_coverage(prompts.strip_reasoning(res.answer))
        if not answer:
            reason = "token limit reached before an answer" if res.stop_type == "limit" else "empty answer"
            await emit({"type": "target", "key": t.key, "status": "error", "error": reason, "stats": stats})
            return "error", None, None
        dropped = self.settings.relevance_filter and (coverage == "none" or prompts.is_no_info(answer))
        await emit({"type": "target", "key": t.key, "status": "irrelevant" if dropped else "done",
                    "answer": answer, "coverage": coverage, "stats": stats,
                    "evidence": [] if dropped else await self._evidence(t, answer, plan.question)})
        return ("irrelevant", None, coverage) if dropped else ("relevant", answer, coverage)

    async def _map_reduce(self, plan: Plan, emit: Emit, tally: Tally) -> str:
        results = await asyncio.gather(*(self._map_one(plan, t, emit, tally) for t in plan.targets))
        findings = [(t, a, cov) for t, (status, a, cov) in zip(plan.targets, results) if status == "relevant" and a]
        n_failed = sum(1 for status, _, _ in results if status == "error")
        tally.extra["n_relevant"] = len(findings)
        tally.extra["n_failed"] = n_failed

        if not findings:
            if n_failed == len(plan.targets):
                raise QueryError("every document query failed; see the findings for details", 502)
            text = NOTHING_FOUND
            if n_failed:
                text += f" ({n_failed} document part(s) could not be queried.)"
            await emit({"type": "delta", "channel": "answer", "text": text})
            return text

        items = [(f"[{t.n}] Source: {t.label}", a) for t, a, _ in findings]
        return await self._synthesize(plan, items, emit, tally)

    async def _synthesize(self, plan: Plan, items: list[tuple[str, str]], emit: Emit, tally: Tally) -> str:
        eng = self.engine
        n_ctx = eng.info.n_ctx_slot
        room = self.settings.max_final_tokens or eng.reserve_tokens()
        history = plan.history[len(plan.history) - (plan.history_sent or 0):] if plan.history_sent else []
        overhead = await eng.synthesis_overhead(plan.question, plan.thinking, history)
        while history and n_ctx - room - overhead < n_ctx // 4:  # findings get at least a quarter
            history = history[1:]
            overhead = await eng.synthesis_overhead(plan.question, plan.thinking, history)
        plan.synthesis_history = history
        budget = n_ctx - room - overhead - 16
        if budget < 256:
            raise QueryError("slot context too small for synthesis; lower the final answer limit", 500)
        level = 0
        while True:
            sizes = [await eng.count(f"{label}\n{text}\n\n") for label, text in items]
            if sum(sizes) <= budget:
                break
            cap = budget // 2
            if any(s > cap for s in sizes):
                # Guarantees every group below holds >= 2 findings, so each level shrinks the list.
                items = [(label, _truncate(text, size, cap)) for (label, text), size in zip(items, sizes)]
                continue
            groups = _pack(items, sizes, budget)
            level += 1
            await emit({"type": "synthesis", "stage": "partial", "level": level, "groups": len(groups)})
            items = list(await asyncio.gather(*(
                self._partial(plan, g, tally) if len(g) > 1 else _passthrough(g[0]) for g in groups
            )))
        tally.extra["synthesis_levels"] = level + 1

        await emit({"type": "synthesis", "stage": "final", "level": level, "n_findings": len(items)})
        prompt, layout = await eng.synthesis_prompt(items, plan.question, plan.thinking, plan.synthesis_history)

        async def on_piece(channel: str, text: str) -> None:
            await emit({"type": "delta", "channel": channel, "text": text})

        queued = time.perf_counter()
        async with eng.pool.lease(PRIORITY_QUERY, "synthesis") as slot:
            wait_ms = (time.perf_counter() - queued) * 1000
            res = await eng.generate(slot, prompt, layout, self.settings.max_final_tokens, on_piece)
        tally.add(res)
        tally.wait_ms += wait_ms
        tally.extra["synthesis"] = {"slot": slot, "wait_ms": round(wait_ms, 1), "n_ctx": eng.info.n_ctx_slot, **res.stats()}
        return res.answer

    async def _partial(self, plan: Plan, group: list[tuple[str, str]], tally: Tally) -> tuple[str, str]:
        prompt, layout = await self.engine.synthesis_prompt(group, plan.question, plan.thinking)
        async with self.engine.pool.lease(PRIORITY_QUERY, "partial synthesis") as slot:
            res = await self.engine.generate(slot, prompt, layout, self.settings.max_final_tokens)
        tally.add(res)
        cites = " ".join(c for label, _ in group for c in re.findall(r"\[\d+\]", label))
        return f"Combined findings from {cites}", res.answer


class TurnRecord:
    """Collects a turn's per-document answers and reasoning from its events, to show it again later."""

    FIELDS = ("status", "answer", "coverage", "stats", "error", "evidence")

    def __init__(self):
        self.targets: dict[str, dict] = {}
        self.reasoning: list[str] = []

    def observe(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "plan":
            self.targets = {t["key"]: dict(t) for t in event["targets"]}
        elif kind == "target" and event.get("key") in self.targets:
            self.targets[event["key"]].update({k: event[k] for k in self.FIELDS if k in event})
        elif kind == "delta" and event.get("channel") == "reasoning":
            self.reasoning.append(event["text"])

    def detail(self) -> dict:
        return {"targets": list(self.targets.values()), "reasoning": "".join(self.reasoning)}


def conversation_title(question: str) -> str:
    text = " ".join(question.split())
    if len(text) <= TITLE_CHARS:
        return text
    return text[:TITLE_CHARS].rsplit(" ", 1)[0].rstrip(",.;:") + "…"


async def _passthrough(item: tuple[str, str]) -> tuple[str, str]:
    return item


def _pack(items: list[tuple[str, str]], sizes: list[int], budget: int) -> list[list[tuple[str, str]]]:
    groups: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    used = 0
    for item, size in zip(items, sizes):
        if cur and used + size > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(item)
        used += size
    if cur:
        groups.append(cur)
    return groups


def _truncate(text: str, n_tokens: int, max_tokens: int) -> str:
    if n_tokens <= max_tokens:
        return text
    return text[: int(len(text) * max_tokens / n_tokens * 0.9)] + " […]"

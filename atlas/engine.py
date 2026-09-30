"""Connection to llama-server: discovery, fingerprinting, prompt assembly and low-level generation."""

import asyncio
import base64
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from . import prompts
from .config import Settings
from .llama import LlamaClient, LlamaError
from .slots import PRIORITY_QUERY, SlotPool

log = logging.getLogger("atlas.engine")

EPOCHS_FILE = "atlas-kv-epochs.json"
REFS_FILE = "atlas-canary-refs.json"
# The canary is a prompt with one obvious continuation. Its restored cache plus the appended
# suffix must predict the same next token as when it was built; a different llama-server build
# that computes differently (or a cache it cannot really read) fails that check.
CANARY_TEXT = "Atlas KV-cache check. Counting: one, two, three, four, five, six, seven, eight"
CANARY_SUFFIX = ","
# Allowance for tokens merging differently where a measured prompt meets its padding.
MEASURE_MARGIN = 8

# With no answer limit, a document part must still leave at least this much room for an answer.
MIN_ANSWER_ROOM = 1024

PieceCallback = Callable[[str, str], Awaitable[None]]
ProgressCallback = Callable[[int, int], None]


@dataclass
class EngineInfo:
    connected: bool = False
    error: str | None = None
    model: str | None = None
    build: str | None = None
    n_slots: int = 0
    n_ctx_slot: int = 0
    fingerprint: str | None = None
    config_label: str | None = None
    kv_dir_ok: bool = False
    supports_thinking: bool = False
    vision: bool = False  # llama-server has a vision projector (mmproj) loaded
    # sliding-window model without --swa-full: does a restored cache get reused? (None: not checked)
    swa_restore_ok: bool | None = None
    media_marker: str | None = None
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ChatSuffix:
    """What follows a cached document: earlier turns, the question and the generation prompt."""
    tokens: list[int]  # for text documents
    text: str  # for visual documents (multimodal prompt string; content defused)
    layout: prompts.Layout
    n_history: int  # earlier turns included

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)


@dataclass
class GenResult:
    answer: str
    reasoning: str
    n_prompt: int  # tokens in the full prompt
    n_cached: int  # tokens reused from the slot's KV cache
    n_processed: int  # tokens actually evaluated
    prompt_ms: float
    n_gen: int
    gen_ms: float
    stop_type: str
    truncated: bool
    ttft_ms: float | None = None  # request sent -> first generated token (includes evaluating the prompt)
    first_answer_ms: float | None = None  # request sent -> first token of the answer (after any thinking)
    wall_ms: float = 0.0
    n_reasoning: int = 0  # generated tokens inside the thinking block (streamed chunks, ≈ tokens)
    draft_n: int = 0  # speculative decoding: tokens the draft proposed / the model accepted
    draft_accepted: int = 0

    def stats(self) -> dict:
        return {
            "n_prompt": self.n_prompt,
            "n_cached": self.n_cached,
            "n_processed": self.n_processed,
            "prompt_ms": round(self.prompt_ms, 1),
            "prompt_tps": round(self.n_processed / (self.prompt_ms / 1000), 1) if self.prompt_ms > 0 else None,
            "n_gen": self.n_gen,
            "n_reasoning": min(self.n_reasoning, self.n_gen),
            "gen_ms": round(self.gen_ms, 1),
            "gen_tps": round(self.n_gen / (self.gen_ms / 1000), 1) if self.gen_ms > 0 else None,
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "first_answer_ms": round(self.first_answer_ms, 1) if self.first_answer_ms is not None else None,
            "wall_ms": round(self.wall_ms, 1),
            "draft_n": self.draft_n,
            "draft_accepted": self.draft_accepted,
            "stop_type": self.stop_type,
            "truncated": self.truncated,
        }


class CacheRejected(Exception):
    """llama-server refused a slot file (wrong format, does not fit): the cache is unusable."""


def cache_rejected(e: LlamaError) -> bool:
    """Only an explicit HTTP 400 answer to a restore means the file itself is unusable."""
    return e.status == 400


def canary_matches(ref: dict, got: tuple[int, float]) -> bool:
    """The same most likely token. Its probability is not compared: it moves with how llama-server
    evaluates the prompt (restored cache reused, a checkpoint re-evaluated, or everything evaluated
    again, as standard builds do for sliding-window models), by 0.31 vs 0.70 for Gemma 4 with the
    same build and settings. A build that computes something else predicts another token."""
    return got[0] == ref["token"]


class Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.llama = LlamaClient(settings.llama_url, settings.llama_api_key, settings.request_timeout_s)
        self.pool: SlotPool | None = None
        self.info = EngineInfo()
        self._layouts: dict[tuple[str, bool], prompts.Layout] = {}
        self._chats: dict[tuple, prompts.ChatRender] = {}
        self._plain_cache: dict[str, list[int]] = {}
        self._template_tokens: dict[tuple[str, bool], list[int]] = {}
        self._ident: dict = {}
        self._instance: str | None = None
        self.restarted = False  # set by refresh() when llama-server was restarted since the last call
        # Set by the supervisor in managed mode: preset fields that define the KV-cache format.
        self.extra_ident: dict = {}
        # Standard llama.cpp keeps one KV stream per slot (--no-kv-unified) and only loads slot files
        # saved with the same number of streams. When a build turns out to be like that, the slot
        # count becomes part of the configuration (set by probe_kv_dir, reset on every connect).
        self.stream_ident: int | None = None
        # Set by the supervisor: the vision projector and image options (part of visual cache variants).
        self.vision_ident: str | None = None
        # Set by the supervisor from the active preset (llama-server request fields). Empty with an
        # external llama-server: its own defaults apply (the model file's, unless its command line
        # sets others).
        self.sampling: dict = {}
        # Set by the supervisor for sliding-window models run without --swa-full: the window size.
        self.swa_window: int | None = None
        self._pad: tuple[int, str, int] | None = None
        self.config_label: str | None = None
        self.paused: str | None = None  # reason while llama-server is being switched or is down
        self._epochs_file = settings.kv_dir / EPOCHS_FILE
        self._refs_file = settings.kv_dir / REFS_FILE
        self._canary_lock = asyncio.Lock()
        self._connect_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self.llama.aclose()

    # --- discovery ---------------------------------------------------------------------

    async def connect(self) -> None:
        """Wait for llama-server, then read its configuration. Safe to call repeatedly."""
        async with self._connect_lock:
            await self._connect()

    async def _connect(self) -> None:
        self.stream_ident = None  # decided again by probe_kv_dir for this llama-server
        delay = 1.0
        while not await self.llama.health():
            self.info.connected = False
            self.info.error = f"waiting for llama-server at {self.settings.llama_url}"
            log.warning("%s (retry in %.0fs)", self.info.error, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15.0)
        await self.refresh()
        if self.info.connected and self.pool:
            await self.probe_kv_dir()

    async def refresh(self) -> bool:
        """Re-read server props. Returns True if the fingerprint or the vision support changed."""
        try:
            props = await self.llama.props()
            meta = await self.llama.model_meta()
        except LlamaError as e:
            self.info.connected = False
            self.info.error = str(e)
            return False

        # media_marker is random per llama-server process, so it doubles as an instance id and
        # reveals restarts that happen between two health polls.
        instance = props.get("media_marker")
        self.restarted = self._instance is not None and instance != self._instance
        self._instance = instance

        n_slots = int(props.get("total_slots") or 1)
        if self.stream_ident and self.stream_ident != n_slots:
            self.stream_ident = None  # restarted with another slot count: probe_kv_dir decides again
        n_ctx = int((props.get("default_generation_settings") or {}).get("n_ctx") or meta.get("n_ctx") or 0)
        model_path = props.get("model_path") or ""
        ident = {
            "model": os.path.basename(model_path),
            "size": meta.get("size"),
            "n_params": meta.get("n_params"),
            "n_vocab": meta.get("n_vocab"),
            "n_embd": meta.get("n_embd"),
            "template": hashlib.sha256((props.get("chat_template") or "").encode()).hexdigest(),
            "prefix_version": prompts.PREFIX_VERSION,
            "system_prompt": hashlib.sha256(self.settings.system_prompt.encode()).hexdigest(),
            **self.extra_ident,
            **({"streams": self.stream_ident} if self.stream_ident else {}),
        }
        self._ident = ident
        fingerprint = self._fingerprint()
        changed = self.info.fingerprint is not None and fingerprint != self.info.fingerprint

        if changed:
            log.warning("model fingerprint changed %s -> %s", self.info.fingerprint, fingerprint)
            self._layouts.clear()
            self._chats.clear()
            self._plain_cache.clear()
            self._template_tokens.clear()
        if self.pool is None:
            self.pool = SlotPool(n_slots)
            if self.paused:
                self.pool.pause()
        elif self.pool.n_slots != n_slots:
            if self.pool.leases:
                log.warning("llama-server slot count changed to %d while requests run; resizing later", n_slots)
            else:
                self.pool.resize(n_slots)

        self.info.connected = True
        self.info.error = None
        alias = props.get("model_alias") or ""
        self.info.model = alias if alias and alias != model_path else os.path.basename(model_path)
        self.info.build = props.get("build_info")
        self.info.n_slots = n_slots
        self.info.n_ctx_slot = n_ctx
        self.info.fingerprint = fingerprint
        self.info.config_label = self.config_label or self.info.model
        vision = bool((props.get("modalities") or {}).get("vision"))
        vision_changed = vision != self.info.vision
        self.info.vision = vision
        self.info.media_marker = instance
        self.info.meta = meta
        try:
            on = await self.layout(self.settings.system_prompt, True)
            off = await self.layout(self.settings.system_prompt, False)
            self.info.supports_thinking = on.tail != off.tail
        except (LlamaError, prompts.TemplateError) as e:
            self.info.connected = False
            self.info.error = f"chat template unusable for CAG: {e}"
        return changed or vision_changed  # visual documents become (un)buildable

    # --- KV-format tracking --------------------------------------------------------------
    #
    # A KV cache only restores into the configuration that produced it. The fingerprint covers
    # everything observable (model, template, system prompt, and in managed mode the preset's
    # cache type / flash-attn / swa-full). What is not observable (e.g. cache flags of an
    # external llama-server, or a llama.cpp upgrade that changes the state format) is caught by
    # a tiny canary slot file per configuration: if it no longer restores, that configuration's
    # epoch is bumped, which changes its fingerprint and rebuilds its caches.

    def _base(self) -> str:
        return hashlib.sha256(json.dumps(self._ident, sort_keys=True).encode()).hexdigest()[:16]

    def _epochs(self) -> dict[str, int]:
        try:
            return json.loads(self._epochs_file.read_text())
        except (OSError, ValueError):
            return {}

    def _fingerprint(self) -> str:
        ident = {**self._ident, "kv_epoch": self._epochs().get(self._base(), 0)}
        return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]

    @property
    def canary_file(self) -> str:
        return f"atlas-canary-{self._base()}.bin"

    def _bump_kv_epoch(self) -> None:
        epochs = self._epochs()
        epochs[self._base()] = epochs.get(self._base(), 0) + 1
        self._epochs_file.write_text(json.dumps(epochs))
        old, self.info.fingerprint = self.info.fingerprint, self._fingerprint()
        log.warning("stored KV caches are incompatible with the running llama-server "
                    "(e.g. changed cache type, flash-attn or llama.cpp version); fingerprint %s -> %s",
                    old, self.info.fingerprint)

    def _refs(self) -> dict[str, dict]:
        try:
            return json.loads(self._refs_file.read_text())
        except (OSError, ValueError):
            return {}

    async def _canary_tokens(self) -> list[int]:
        return await self.llama.tokenize(CANARY_TEXT, add_special=True, parse_special=False)

    async def _next_token(self, slot: int, tokens: list[int]) -> tuple[int, float] | None:
        """Most likely token after `tokens` + CANARY_SUFFIX in `slot` (reusing its cache) and its logprob."""
        suffix = await self.llama.tokenize(CANARY_SUFFIX, add_special=False, parse_special=False)
        res = await self.llama.completion({
            "prompt": tokens + suffix, "n_predict": 1, "n_probs": 1, "temperature": 0, "id_slot": slot,
            "cache_prompt": True, "response_fields": ["completion_probabilities"],
        })
        probs = res.get("completion_probabilities") or []
        if not probs:
            return None
        first = probs[0]
        logprob = first.get("logprob")
        if logprob is None and first.get("prob") is not None:  # older response format
            logprob = math.log(max(float(first["prob"]), 1e-12))
        return (int(first["id"]), float(logprob)) if first.get("id") is not None and logprob is not None else None

    async def _restore_canary(self, slot: int) -> None:
        """Restore the canary. Raises CacheRejected only if llama-server explicitly refuses the file;
        connection or server errors are retried and then re-raised, never read as incompatibility."""
        for attempt in range(3):
            try:
                await self.llama.slot_restore(slot, self.canary_file)
                return
            except LlamaError as e:
                if cache_rejected(e):
                    raise CacheRejected(str(e)) from e
                if attempt == 2:
                    raise
                await asyncio.sleep(1 + attempt)

    async def _write_canary(self, slot: int) -> None:
        await self.llama.slot_erase(slot)
        tokens = await self._canary_tokens()
        await self.prefill(slot, tokens)
        await self.llama.slot_save(slot, self.canary_file)
        # the reference is taken the way the check later takes it: from the restored file
        await self.llama.slot_erase(slot)
        await self._restore_canary(slot)
        ref = await self._next_token(slot, tokens)
        refs = self._refs()
        if ref:
            refs[self._base()] = {"token": ref[0], "logprob": ref[1], "build": self.info.build,
                                  "streams": self.info.n_slots}
        else:
            refs.pop(self._base(), None)
        self._refs_file.write_text(json.dumps(refs))

    @asynccontextmanager
    async def _probe_slot(self) -> AsyncIterator[int]:
        assert self.pool is not None
        if self.pool.paused:
            # llama-server was just (re)started by the supervisor and the pool is paused, so no
            # other request can reach the new process: use slot 0 directly.
            yield 0
        else:
            async with self.pool.lease(PRIORITY_QUERY, "startup probe") as slot:
                yield slot

    async def probe_kv_dir(self) -> None:
        """Verify slot persistence end to end (proves ATLAS_KV_DIR is the --slot-save-path) and
        check this configuration's canary: it must restore, and must still predict its reference
        token (catches a llama-server build that reads the cache but computes differently)."""
        canary = self.settings.kv_dir / self.canary_file
        for legacy in ("atlas-canary.bin", "atlas-kv-epoch.txt"):
            (self.settings.kv_dir / legacy).unlink(missing_ok=True)
        try:
            async with self._probe_slot() as slot:
                rebuild = not canary.exists()
                if not rebuild:
                    try:
                        await self._restore_canary(slot)
                    except CacheRejected:
                        if self._saved_with_other_slot_count():
                            self._use_stream_config()  # the caches stay valid for their slot count
                            canary = self.settings.kv_dir / self.canary_file
                            rebuild = not canary.exists()
                            if not rebuild:
                                try:
                                    await self._restore_canary(slot)
                                except CacheRejected:
                                    self._bump_kv_epoch()
                                    rebuild = True
                        else:
                            self._bump_kv_epoch()
                            rebuild = True
                    if not rebuild:
                        ref = self._refs().get(self._base())
                        if ref is None:
                            rebuild = True  # canary from before references existed: re-create it
                        else:
                            got = await self._next_token(slot, await self._canary_tokens())
                            if got is not None and got[0] == ref["token"] and abs(got[1] - ref["logprob"]) > 0.1:
                                log.info("the canary predicts the same token with p=%.2f instead of %.2f (a different "
                                         "evaluation path, not a different build)", math.exp(got[1]),
                                         math.exp(ref["logprob"]))
                            if got is not None and not canary_matches(ref, got):
                                log.warning("the canary predicts token %s (p=%.2f) instead of %s (p=%.2f): this "
                                            "llama-server build computes differently from the one that built the "
                                            "caches (%s)", got[0], math.exp(got[1]), ref["token"],
                                            math.exp(ref["logprob"]), ref.get("build"))
                                self._bump_kv_epoch()
                                rebuild = True
                if rebuild:
                    canary.unlink(missing_ok=True)
                    await self._write_canary(slot)
                if self.swa_window and canary.exists():
                    await self._probe_swa_restore(slot)
            self.info.fingerprint = self._fingerprint()
            self.info.kv_dir_ok = canary.exists()
            if not self.info.kv_dir_ok:
                self.info.error = (
                    f"slot files are not written to ATLAS_KV_DIR={self.settings.kv_dir}; "
                    "point it at llama-server's --slot-save-path"
                )
        except LlamaError as e:
            self.info.kv_dir_ok = False
            self.info.error = f"slot persistence unavailable: {e}"
        if self.info.error:
            log.error(self.info.error)

    def _saved_with_other_slot_count(self) -> bool:
        """Whether a rejected canary may only have been saved with another number of slots."""
        if self.stream_ident:
            return False
        saved = (self._refs().get(self._base()) or {}).get("streams")
        return saved != self.info.n_slots  # also for canaries from before the count was recorded

    def _use_stream_config(self) -> None:
        old = self.info.fingerprint
        saved = (self._refs().get(self._base()) or {}).get("streams")
        self.stream_ident = self.info.n_slots
        self._ident["streams"] = self.stream_ident
        self.info.fingerprint = self._fingerprint()
        log.warning("this llama-server cannot load slot files saved with %s slot(s) into %d slots (standard "
                    "llama.cpp keeps one KV stream per slot): caches for %d slots are kept separately "
                    "(fingerprint %s -> %s); the existing ones stay valid for their slot count",
                    saved or "another number of", self.info.n_slots, self.info.n_slots, old, self.info.fingerprint)

    async def _probe_swa_restore(self, slot: int) -> None:
        """Check that llama-server reuses a restored cache of a sliding-window model.

        Without --swa-full a slot file keeps only the window for sliding-window layers. Stock
        llama-server builds then consider the restored state incomplete and prefill everything
        again on every query; builds with the SWA restore fix reuse it.
        """
        n = self.swa_window + 64
        words = await self.plain(" ".join(f"item{i}" for i in range(n)))
        tokens = words[:n]
        name = "atlas-swa-probe.bin"
        try:
            await self.prefill(slot, tokens)
            await self.llama.slot_save(slot, name)
            await self.llama.slot_erase(slot)
            await self.llama.slot_restore(slot, name)
            res = await self.llama.completion({"prompt": tokens + tokens[:1], "n_predict": 1, "id_slot": slot,
                                               "cache_prompt": True, "temperature": 0})
            reused = int((res.get("timings") or {}).get("cache_n") or 0)
            self.info.swa_restore_ok = reused >= len(tokens)
            if not self.info.swa_restore_ok:
                log.warning("llama-server re-processes restored caches of this sliding-window model (%d of %d tokens "
                            "reused): enable --swa-full in the preset or use a build with the SWA restore fix",
                            reused, len(tokens))
        finally:
            (self.settings.kv_dir / name).unlink(missing_ok=True)

    async def kv_compatible(self, slot: int, seen_fingerprint: str | None) -> bool:
        """After a failed restore: False if the configuration's whole KV store is stale (and its
        epoch was bumped). `slot` must be leased by the caller; `seen_fingerprint` is the
        fingerprint the caller planned with, so concurrent failures bump the epoch only once."""
        async with self._canary_lock:
            if self.info.fingerprint != seen_fingerprint:
                return False  # another task already detected the change
            try:
                await self._restore_canary(slot)
                return True
            except CacheRejected:
                self._bump_kv_epoch()
                await self._write_canary(slot)
                return False

    # --- availability ------------------------------------------------------------------

    def pause(self, reason: str) -> None:
        """Reject new work (queries get 503, ingestion waits) while llama-server is switched."""
        self.paused = reason
        if self.pool:
            self.pool.pause()

    def resume(self) -> None:
        self.paused = None
        if self.pool:
            self.pool.resume()

    @property
    def ready(self) -> bool:
        return not self.paused and self.info.connected and self.info.kv_dir_ok and self.pool is not None

    @property
    def status_message(self) -> str | None:
        return self.paused or self.info.error

    # --- prompt assembly ---------------------------------------------------------------

    async def layout(self, system_prompt: str, thinking: bool) -> prompts.Layout:
        key = (system_prompt, thinking)
        if key not in self._layouts:
            rendered = await self.llama.apply_template(
                prompts.layout_messages(system_prompt), {"enable_thinking": thinking}
            )
            self._layouts[key] = prompts.split_rendered(rendered)
        return self._layouts[key]

    async def _template(self, text: str, first: bool = False) -> list[int]:
        key = (text, first)
        if key not in self._template_tokens:
            self._template_tokens[key] = await self.llama.tokenize(text, add_special=first, parse_special=True)
        return self._template_tokens[key]

    async def plain(self, text: str) -> list[int]:
        return await self.llama.tokenize(text, add_special=False, parse_special=False)

    async def count(self, text: str) -> int:
        return len(await self.plain(text))

    async def plain_cached(self, text: str) -> list[int]:
        """Plain tokens of text that recurs (earlier turns of a conversation are sent with every part)."""
        if text not in self._plain_cache:
            if len(self._plain_cache) > 512:
                self._plain_cache.clear()
            self._plain_cache[text] = await self.plain(text)
        return self._plain_cache[text]

    async def count_cached(self, text: str) -> int:
        return len(await self.plain_cached(text))

    async def prefix_overhead(self) -> tuple[list[int], list[int]]:
        lay = await self.layout(self.settings.system_prompt, self.settings.enable_thinking)
        return await self._template(lay.head, first=True), await self._template(lay.mid)

    async def document_prefix(self, name: str, idx: int, n_parts: int, text: str) -> list[int]:
        head, mid = await self.prefix_overhead()
        body = await self.plain(prompts.document_block(name, idx, n_parts, text))
        return head + body + mid

    def reserve_tokens(self) -> int:
        """Room kept free in a slot when splitting documents: set, or 1/8 of the slot (4k–64k)."""
        n_ctx = self.info.n_ctx_slot
        reserve = self.settings.reserve_tokens or min(max(n_ctx // 8, 4096), 65536)
        return min(reserve, n_ctx // 2)

    # --- chat --------------------------------------------------------------------------

    async def render_chat(self, system_prompt: str, n_turns: int, thinking: bool, with_doc: bool) -> prompts.ChatRender:
        key = (system_prompt, n_turns, thinking, with_doc)
        if key not in self._chats:
            rendered = await self.llama.apply_template(prompts.chat_messages(system_prompt, n_turns, with_doc),
                                                       {"enable_thinking": thinking})
            self._chats[key] = prompts.split_chat(rendered, n_turns, with_doc)
        return self._chats[key]

    async def _assemble(self, render: prompts.ChatRender, contents: list[str], skip_first: bool) -> tuple[list[int], str]:
        tokens: list[int] = []
        text = ""
        for i, content in enumerate(contents):
            if not (i == 0 and skip_first):
                tokens += await self._template(render.glue[i])
                text += render.glue[i]
            tokens += await self.plain_cached(content)
            text += prompts.defuse(content)
        tokens += await self._template(render.tail)
        return tokens, text + render.tail

    async def chat_suffix(self, history: list[tuple[str, str]], block: str, thinking: bool) -> ChatSuffix:
        """The conversation after a cached document: earlier turns, then the question block."""
        cached = await self.layout(self.settings.system_prompt, self.settings.enable_thinking)
        contents = prompts.history_turns(history) + [block]
        try:
            render = await self.render_chat(self.settings.system_prompt, len(contents), thinking, with_doc=True)
            fits_cache = render.head == cached.head and render.glue[0] == cached.mid
        except prompts.TemplateError:
            fits_cache = False
        if not fits_cache:
            # the template renders the first turn differently in a longer chat: the cached prefix
            # would not match, so the earlier turns go into the question as text instead
            contents = [prompts.history_block(history) + block if history else block]
            render = await self.render_chat(self.settings.system_prompt, 1, thinking, with_doc=True)
        tokens, text = await self._assemble(render, contents, skip_first=True)
        return ChatSuffix(tokens, text, prompts.Layout(render.head, render.glue[0], render.tail), len(history))

    async def synthesis_prompt(self, findings: list[tuple[str, str]], question: str, thinking: bool,
                               history: list[tuple[str, str]] = ()) -> tuple[list[int], prompts.Layout]:
        contents = prompts.history_turns(list(history)) + [
            prompts.findings_block(findings) + prompts.synthesis_question_block(question)]
        render = await self.render_chat(self.settings.synthesis_prompt, len(contents), thinking, with_doc=False)
        tokens, _ = await self._assemble(render, contents, skip_first=False)
        tokens = await self._template(render.head, first=True) + tokens
        return tokens, prompts.Layout(render.head, "", render.tail)

    # --- visual prefill ------------------------------------------------------------------

    def visual_variant(self, dpi: int) -> str | None:
        """What a visual cache is built from: projector and page resolution. None without vision."""
        if not self.info.vision:
            return None
        return f"visual:{self.vision_ident or 'external'}:{dpi}"

    async def visual_prefix(self, name: str, idx: int, n_parts: int, page_numbers: list[int],
                            files: list[dict] | None = None) -> str:
        lay = await self.layout(self.settings.system_prompt, self.settings.enable_thinking)
        return lay.head + prompts.visual_document_block(name, idx, n_parts, page_numbers, files) + lay.mid

    def multimodal(self, text: str, images: list[bytes]) -> dict:
        marker = self.info.media_marker
        if not marker:
            raise LlamaError("llama-server did not report a media marker; is a vision projector loaded?")
        if text.count(prompts.MEDIA_PLACEHOLDER) != len(images):
            raise ValueError("number of page images does not match the prompt")
        return {"prompt_string": text.replace(prompts.MEDIA_PLACEHOLDER, marker),
                "multimodal_data": [base64.b64encode(b).decode() for b in images]}

    async def _padding(self) -> tuple[str, int]:
        """Text of more tokens than a slot holds, so a prompt carrying it is always rejected."""
        n_ctx = self.info.n_ctx_slot
        if self._pad and self._pad[0] == n_ctx:
            return self._pad[1], self._pad[2]
        text = "\n" + " x" * (n_ctx + 64)
        n = len(await self.plain(text))
        while n <= n_ctx:
            text += " x" * (n_ctx - n + 64)
            n = len(await self.plain(text))
        self._pad = (n_ctx, text, n)
        return text, n

    async def measure(self, slot: int, text: str, images: list[bytes]) -> int:
        """Tokens a multimodal prompt occupies, without evaluating it.

        llama-server tokenizes the prompt (images become their token count) and rejects it before
        any evaluation if it exceeds the slot; the rejection reports the count. Padding makes sure
        it is rejected.
        """
        pad, n_pad = await self._padding()
        try:
            await self.llama.completion({"prompt": self.multimodal(text + pad, images), "n_predict": 0,
                                         "id_slot": slot, "cache_prompt": True})
        except LlamaError as e:
            if e.data.get("n_prompt_tokens"):
                return int(e.data["n_prompt_tokens"]) - n_pad + MEASURE_MARGIN
            raise
        raise LlamaError("llama-server accepted a prompt larger than its context: turn off context shift "
                         "(--no-context-shift) for visual prefill")

    async def synthesis_overhead(self, question: str, thinking: bool, history: list[tuple[str, str]] = ()) -> int:
        tokens, _ = await self.synthesis_prompt([], question, thinking, history)
        return len(tokens)

    # --- execution ---------------------------------------------------------------------

    async def prefill(self, slot: int, tokens: list[int] | dict, on_progress: ProgressCallback | None = None) -> dict:
        """Evaluate `tokens` (or a multimodal prompt) into the slot without generating (n_predict = 0).

        Returns the final response (timings, tokens_evaluated)."""
        payload = {"prompt": tokens, "n_predict": 0, "id_slot": slot, "cache_prompt": True, "return_progress": True}
        final: dict | None = None
        async for chunk in self.llama.completion_stream(payload):
            prog = chunk.get("prompt_progress")
            if prog and on_progress:
                on_progress(int(prog.get("processed", 0)), int(prog.get("total", len(tokens))))
            if chunk.get("stop"):
                final = chunk
        if final is None:
            raise LlamaError("prefill ended without a final response")
        if isinstance(tokens, list) and final.get("tokens_evaluated") != len(tokens):
            raise LlamaError(f"prefill evaluated {final.get('tokens_evaluated')} of {len(tokens)} tokens")
        return final

    async def generate(self, slot: int, tokens: list[int] | dict, layout: prompts.Layout, answer_cap: int | None,
                       on_piece: PieceCallback | None = None, temperature: float | None = None,
                       n_prompt: int | None = None) -> GenResult:
        """Decode an answer. Without `answer_cap` (or thinking budget) generation runs until the model
        stops or the slot is full; with both it is capped at answer + thinking budget.

        `tokens` may be a multimodal prompt; `n_prompt` then gives its length in tokens."""
        thinking_open = layout.thinking_open
        budget = self.settings.max_thinking_tokens if thinking_open else 0
        prompt_len = len(tokens) if isinstance(tokens, list) else (n_prompt or 0)
        free = self.info.n_ctx_slot - prompt_len - 1
        capped = bool(answer_cap) and (not thinking_open or bool(budget))
        payload = {
            "prompt": tokens,
            "n_predict": max(1, min(answer_cap + (budget or 0), free) if capped else free),
            "id_slot": slot,
            "cache_prompt": True,
            "preserved_tokens": prompts.THINK_TOKENS,  # or Gemma's thinking markers are left out of the text
            **self.sampling,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if thinking_open and budget:
            # llama-server forces the end tag once the budget is spent, so an answer always follows.
            # The message key must be present: only its handler sets the tokens that get forced.
            start_tag, end_tag = layout.think_tags
            payload.update({
                "reasoning_budget_tokens": self.settings.max_thinking_tokens,
                "reasoning_budget_start_tag": start_tag,
                "reasoning_budget_end_tags": [end_tag],
                "reasoning_budget_message": "\n\nThinking budget reached, answering now.\n",
                "generation_prompt": layout.tail,
            })
        splitter = prompts.ThinkSplitter(thinking_open)
        parts: dict[str, list[str]] = {"answer": [], "reasoning": []}
        final: dict = {}
        t0 = time.perf_counter()
        ttft = first_answer = None
        n_reasoning = 0

        async def emit(pieces: list[tuple[str, str]]) -> None:
            nonlocal first_answer
            for kind, piece in pieces:
                parts[kind].append(piece)
                if kind == "answer" and first_answer is None and piece.strip():
                    first_answer = (time.perf_counter() - t0) * 1000
                if on_piece:
                    await on_piece(kind, piece)

        async for chunk in self.llama.completion_stream(payload):
            if chunk.get("content"):
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                if splitter.in_reasoning:  # llama-server streams one chunk per token
                    n_reasoning += 1
                await emit(splitter.feed(chunk["content"]))
            if chunk.get("stop"):
                final = chunk
        await emit(splitter.flush())

        t = final.get("timings") or {}
        return GenResult(
            answer="".join(parts["answer"]).strip(),
            reasoning="".join(parts["reasoning"]).strip(),
            n_prompt=int(final.get("tokens_evaluated") or prompt_len),
            n_cached=int(t.get("cache_n") or 0),
            n_processed=int(t.get("prompt_n") or 0),
            prompt_ms=float(t.get("prompt_ms") or 0.0),
            n_gen=int(t.get("predicted_n") or final.get("tokens_predicted") or 0),
            gen_ms=float(t.get("predicted_ms") or 0.0),
            stop_type=str(final.get("stop_type") or ""),
            truncated=bool(final.get("truncated")),
            ttft_ms=ttft,
            first_answer_ms=first_answer,
            wall_ms=(time.perf_counter() - t0) * 1000,
            n_reasoning=n_reasoning,
            draft_n=int(t.get("draft_n") or 0),
            draft_accepted=int(t.get("draft_n_accepted") or 0),
        )

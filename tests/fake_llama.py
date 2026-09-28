"""A tiny stand-in for llama-server that implements the endpoints Atlas uses.

Tokens are characters (BOS = 1), slots keep real token state, prefix reuse is computed like
llama-server does (longest common prefix, minus one token if the prompt is fully cached), and
slot save/restore write actual files. Answers are canned but depend on the prompt, so relevance
filtering and synthesis can be exercised.
"""

import asyncio
import json
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

BOS = 1
TEMPLATE = "chatml-fake"


def render(messages: list[dict]) -> str:
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + "<|im_start|>assistant\n"


def tokenize(text: str, add_special: bool) -> list[int]:
    return ([BOS] if add_special else []) + [ord(c) for c in text]


def detokenize(tokens: list[int]) -> str:
    return "".join(chr(t) for t in tokens if t != BOS)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": status, "message": message, "type": "invalid_request_error"}}, status)


class FakeLlama:
    def __init__(self, kv_dir: Path, n_slots: int = 2, n_ctx: int = 4096):
        self.kv_dir = kv_dir
        self.n_ctx = n_ctx
        self.model_path = "/models/fake-model.gguf"
        self.slots: dict[int, list[int]] = {i: [] for i in range(n_slots)}
        self.log: list[dict] = []  # one entry per /completion
        self.media_marker = f"<__media_{uuid.uuid4().hex}__>"  # random per llama-server process
        self.fail_prefills = 0  # number of upcoming prefills that crash mid-stream
        self.prefill_delay = 0.0  # seconds each prefill takes
        self.semantics = "default"  # a build that "computes differently" predicts other tokens
        self.fail_restores = 0  # upcoming restores that fail with a server error (not a rejection)
        self.kv_format = "f16"  # stands in for --cache-type-k/v; restores of other formats fail
        self.active = 0  # streams currently being generated
        self.cancelled = 0  # streams aborted because the client went away
        self.app = self._build()

    def _answer(self, text: str) -> str:
        if "Findings:" in text:
            return f"Synthesized from {text.count('Source:')} findings [1]."
        if "Quote the passages" in text:  # map phase
            doc = text[text.find("<document"): text.find("</document>")].lower()
            question = text.rsplit("Question: ", 1)[1].split("\n\n", 1)[0].lower()
            hits = [w for w in re.findall(r"[a-z0-9]{4,}", question) if w in doc]
            rated = "COVERAGE: none" in text
            if hits:
                return f"Found {hits[0]} in the document." + ("\nCOVERAGE: full" if rated else "")
            return "The document does not cover this." + ("\nCOVERAGE: none" if rated else "")
        return "Single answer."

    def _build(self) -> FastAPI:
        app = FastAPI()
        fake = self

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/props")
        async def props():
            return {
                "total_slots": len(fake.slots),
                "default_generation_settings": {"n_ctx": fake.n_ctx},
                "model_path": fake.model_path,
                "model_alias": fake.model_path,
                "chat_template": TEMPLATE,
                "build_info": "fake",
                "media_marker": fake.media_marker,
            }

        @app.get("/v1/models")
        async def models():
            return {"data": [{"meta": {"n_ctx": fake.n_ctx, "n_params": 1, "size": 1, "n_vocab": 1, "n_embd": 1}}]}

        @app.post("/apply-template")
        async def apply_template(req: Request):
            return {"prompt": render((await req.json())["messages"])}

        @app.post("/tokenize")
        async def tok(req: Request):
            body = await req.json()
            return {"tokens": tokenize(body["content"], body.get("add_special", False))}

        @app.post("/slots/{slot}")
        async def slots(slot: int, action: str, req: Request):
            if slot not in fake.slots:
                return error(400, "Invalid slot ID")
            if action == "erase":
                n = len(fake.slots[slot])
                fake.slots[slot] = []
                return {"id_slot": slot, "n_erased": n}
            filename = (await req.json())["filename"]
            if "/" in filename or ".." in filename:
                return error(400, "Invalid filename")
            path = fake.kv_dir / filename
            if action == "save":
                path.write_text(json.dumps({"format": fake.kv_format, "tokens": fake.slots[slot]}))
                return {"id_slot": slot, "filename": filename, "n_saved": len(fake.slots[slot]),
                        "n_written": path.stat().st_size}
            if action == "restore":
                if fake.fail_restores > 0:
                    fake.fail_restores -= 1
                    return error(500, "simulated transient server error")
                saved = json.loads(path.read_text()) if path.exists() else None
                if not saved or saved["format"] != fake.kv_format:
                    return error(400, "Unable to restore slot: invalid slot save file")
                fake.slots[slot] = saved["tokens"]
                return {"id_slot": slot, "filename": filename, "n_restored": len(fake.slots[slot]),
                        "n_read": path.stat().st_size}
            return error(400, "Invalid action")

        @app.post("/completion")
        async def completion(req: Request):
            body = await req.json()
            prompt: list[int] = body["prompt"]
            slot = body["id_slot"]
            if len(prompt) > fake.n_ctx:
                return error(400, "request exceeds the available context size")
            cached = fake.slots[slot]
            n = 0
            while n < min(len(cached), len(prompt)) and cached[n] == prompt[n]:
                n += 1
            if n == len(prompt):
                n -= 1
            fake.slots[slot] = list(prompt)
            fake.log.append({"slot": slot, "n_prompt": len(prompt), "cache_n": n, "n_predict": body["n_predict"]})
            if not body.get("stream"):
                # non-streaming requests are only used for the canary's next-token check
                token = 11 if fake.semantics == "default" else 12
                return {"content": "x", "stop": True, "tokens_cached": n,
                        "completion_probabilities": [{"id": token, "token": "x", "logprob": -0.05}],
                        "timings": {"cache_n": n, "prompt_n": len(prompt) - n}}
            text = detokenize(prompt)
            answer = "" if body["n_predict"] == 0 else fake._answer(text)
            pieces = [answer[i:i + 7] for i in range(0, len(answer), 7)]
            slow = body["n_predict"] > 0 and "slowly" in text
            if slow:
                pieces = pieces * 100

            async def stream():
                fake.active += 1
                try:
                    async for chunk in body_chunks():
                        yield chunk
                except (asyncio.CancelledError, GeneratorExit):
                    fake.cancelled += 1
                    raise
                finally:
                    fake.active -= 1

            crash = body["n_predict"] == 0 and fake.fail_prefills > 0
            if crash:
                fake.fail_prefills -= 1

            async def body_chunks():
                if body.get("return_progress"):
                    yield "data: " + json.dumps({"prompt_progress": {"total": len(prompt), "cache": n,
                                                                    "processed": len(prompt)}}) + "\n\n"
                if crash:
                    raise RuntimeError("simulated llama-server crash")  # connection drops mid-stream
                if body["n_predict"] == 0 and fake.prefill_delay:
                    await asyncio.sleep(fake.prefill_delay)
                for p in pieces:
                    if slow:
                        await asyncio.sleep(0.02)
                    yield "data: " + json.dumps({"content": p, "stop": False}) + "\n\n"
                yield "data: " + json.dumps({
                    "content": "", "stop": True, "stop_type": "eos", "truncated": False,
                    "tokens_evaluated": len(prompt), "tokens_predicted": len(pieces), "tokens_cached": n,
                    "timings": {"cache_n": n, "prompt_n": len(prompt) - n, "prompt_ms": 1.0,
                                "predicted_n": len(pieces), "predicted_ms": 1.0},
                }) + "\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        return app

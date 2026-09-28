"""Thin async client for the llama-server endpoints Atlas relies on."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

# Trims the final streamed chunk; by default llama-server echoes the whole prompt text back,
# which is hundreds of KB for a large document.
FINAL_RESPONSE_FIELDS = [
    "content",
    "stop",
    "stop_type",
    "truncated",
    "tokens_predicted",
    "tokens_evaluated",
    "tokens_cached",
    "timings",
]


class LlamaError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, kind: str | None = None):
        super().__init__(message)
        self.status = status
        self.kind = kind


def _error_from_response(status: int, body: bytes) -> LlamaError:
    try:
        err = json.loads(body).get("error", {})
        return LlamaError(err.get("message") or body.decode(errors="replace"), status, err.get("type"))
    except (ValueError, AttributeError):
        return LlamaError(body.decode(errors="replace")[:500] or f"HTTP {status}", status)


class LlamaClient:
    def __init__(self, base_url: str, api_key: str | None, timeout_s: float):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10.0),
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            r = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as e:
            raise LlamaError(f"llama-server unreachable: {e!r}") from e
        if r.status_code != 200:
            raise _error_from_response(r.status_code, r.content)
        return r.json()

    # --- discovery ---------------------------------------------------------------------

    async def health(self) -> bool:
        try:
            r = await self._http.get("/health", timeout=5.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def props(self) -> dict:
        return await self._request("GET", "/props")

    async def model_meta(self) -> dict:
        data = await self._request("GET", "/v1/models")
        models = data.get("data") or []
        return (models[0].get("meta") or {}) if models else {}

    # --- text ------------------------------------------------------------------------

    async def apply_template(self, messages: list[dict], template_kwargs: dict | None = None) -> str:
        body: dict[str, Any] = {"messages": messages, "add_generation_prompt": True}
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        return (await self._request("POST", "/apply-template", json=body))["prompt"]

    async def tokenize(self, text: str, *, add_special: bool, parse_special: bool) -> list[int]:
        if not text:
            return []
        body = {"content": text, "add_special": add_special, "parse_special": parse_special}
        return (await self._request("POST", "/tokenize", json=body))["tokens"]

    # --- slots -----------------------------------------------------------------------

    async def slot_save(self, slot: int, filename: str) -> dict:
        return await self._request("POST", f"/slots/{slot}", params={"action": "save"}, json={"filename": filename})

    async def slot_restore(self, slot: int, filename: str) -> dict:
        return await self._request("POST", f"/slots/{slot}", params={"action": "restore"}, json={"filename": filename})

    async def slot_erase(self, slot: int) -> dict:
        return await self._request("POST", f"/slots/{slot}", params={"action": "erase"})

    # --- generation ------------------------------------------------------------------

    async def completion(self, payload: dict) -> dict:
        """Non-streaming POST /completion."""
        return await self._request("POST", "/completion", json={**payload, "stream": False})

    async def completion_stream(self, payload: dict) -> AsyncIterator[dict]:
        """POST /completion with stream=true and yield each parsed SSE chunk.

        Leaving the iterator early closes the HTTP connection, which makes llama-server
        abort the task and free the slot.
        """
        body = {**payload, "stream": True, "response_fields": FINAL_RESPONSE_FIELDS}
        try:
            async with self._http.stream("POST", "/completion", json=body) as r:
                if r.status_code != 200:
                    raise _error_from_response(r.status_code, await r.aread())
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = json.loads(line[6:])
                    if "error" in chunk:
                        err = chunk["error"]
                        raise LlamaError(err.get("message", str(err)), err.get("code"), err.get("type"))
                    yield chunk
                    if chunk.get("stop"):
                        return
        except httpx.HTTPError as e:
            raise LlamaError(f"llama-server stream failed: {e!r}") from e

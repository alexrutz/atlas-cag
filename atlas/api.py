"""HTTP API and static frontend."""

import asyncio
import hashlib
import hmac
import json
import logging
import platform
import re
import shutil
import socket
from contextlib import asynccontextmanager
from pathlib import Path, PurePath

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from . import __version__, builds, evidence, models, pdftools
from . import pages as page_images
from .config import RUNTIME_FIELDS, RuntimeSettings, Settings, get_settings
from .downloads import Downloader, DownloadError
from .engine import Engine
from .extract import SUPPORTED_EXTENSIONS, ExtractionError, extract_text, normalize
from .prompts import document_block
from .ingest import Ingestor
from .query import QueryError, QueryService
from .store import Store
from . import sampling
from .supervisor import LOOPBACK, PresetConfig, Supervisor, connect_url, preset_sampling, standard_build
from .updater import BuildUpdater, UpdateError

log = logging.getLogger("atlas.api")
STATIC = Path(__file__).parent / "static"
MONITOR_INTERVAL_S = 10


class TextDocument(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    collection_id: str | None = None


class DocumentUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    collection_id: str | None = None
    mode: Literal["text", "visual"] | None = None  # prefill from extracted text or from page images


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    parent_id: str | None = None  # nested in this collection (a chapter of it)
    document_ids: list[str] = Field(default_factory=list, max_length=10000)  # moved into it, keeping their order


class CollectionUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    parent_id: str | None = None  # moves the collection when given (null: to the top level)


class OrderItem(BaseModel):
    kind: Literal["collection", "document"]
    id: str


class OrderRequest(BaseModel):
    parent_id: str | None = None  # null: top level (collections) / unfiled (documents)
    items: list[OrderItem] = Field(max_length=20000)


class EvidenceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=400_000)  # an answer: its quotes are located
    question: str = Field(default="", max_length=400_000)
    part: int | None = Field(default=None, ge=1)  # prefer the passage in this part (running model)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000_000)  # tokens are checked, not characters
    document_ids: list[str] = Field(default_factory=list, max_length=2000)
    collection_ids: list[str] = Field(default_factory=list, max_length=200)
    thinking: bool | None = None
    conversation_id: str | None = None  # empty: the question starts a new conversation


class ShardSpec(BaseModel):
    name: str = Field(min_length=1, max_length=5000)  # shortened to a file name when the shard is stored
    pages: list[int] = Field(min_length=1, max_length=10000)
    # chapter collections to put the shard in, outermost first (nested in the target collection)
    folder: list[str] = Field(default_factory=list, max_length=8)


class ShardRequest(BaseModel):
    shards: list[ShardSpec] = Field(min_length=1, max_length=500)
    collection_id: str | None = None
    mode: Literal["text", "visual"] | None = None


class ConversationBody(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ServerAddress(BaseModel):
    host: str = Field(min_length=1, max_length=253)  # 127.0.0.1: this computer only; 0.0.0.0: all interfaces
    port: int = Field(ge=1, le=65535)
    api_key: str | None = Field(default=None, max_length=512)  # null: keep the current key; "": no key


class DownloadRequest(BaseModel):
    repo: str
    file: str


class BuildRequest(BaseModel):
    command: str = Field(min_length=1, max_length=2000)
    name: str = Field(default="", max_length=80)  # what the build is for, e.g. "Qwen3.8 fork"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if settings.managed:
        settings.llama_url = connect_url(settings.llama_host, settings.llama_port)
    defaults = {f: getattr(settings, f) for f in RUNTIME_FIELDS}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        settings.kv_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dirs[0].mkdir(parents=True, exist_ok=True)
        pdftools.sweep(settings.data_dir / "tools")
        store = Store(settings.db_path)
        for key, value in (store.get_state("settings") or {}).items():
            if key in RUNTIME_FIELDS:
                setattr(settings, key, value)
        if settings.managed:  # the llama-server address chosen in Settings → Model
            for key, value in (store.get_state("llama_address") or {}).items():
                setattr(settings, f"llama_{key}", value)
            settings.llama_url = connect_url(settings.llama_host, settings.llama_port)
        engine = Engine(settings)
        ingestor = Ingestor(engine, store, settings)
        downloader = Downloader(settings.hf_endpoint, settings.model_dirs[0])
        downloader.start()
        supervisor = Supervisor(settings, store, engine, ingestor) if settings.managed else None
        updater = BuildUpdater(settings, store, supervisor) if supervisor else None
        if updater:
            updater.start()
        app.state.store = store
        app.state.engine = engine
        app.state.ingestor = ingestor
        app.state.queries = QueryService(engine, store, ingestor, settings)
        app.state.downloader = downloader
        app.state.supervisor = supervisor
        app.state.updater = updater
        monitor = asyncio.create_task(_monitor(engine, ingestor, supervisor, store), name="llama-monitor")
        try:
            yield
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            if updater:
                await updater.stop()
            if supervisor:
                await supervisor.shutdown()
            await downloader.stop()
            await ingestor.stop()
            await engine.aclose()
            store.close()

    app = FastAPI(title="Atlas CAG", version=__version__, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def readable_validation_error(request: Request, exc: RequestValidationError):
        """Say which field is wrong ("shards › 13 › name: …"), not only what is wrong."""
        def where(loc) -> str:
            parts = [f"#{p + 1}" if isinstance(p, int) else str(p) for p in loc if p not in ("body", "query", "path")]
            return " › ".join(parts)
        messages = [f"{where(e['loc'])}: {e['msg']}" if where(e["loc"]) else e["msg"] for e in exc.errors()]
        return JSONResponse({"detail": "; ".join(messages)}, status_code=422)
    keys = settings.api_key_set

    async def require_auth(request: Request) -> None:
        if not keys:
            return
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not any(hmac.compare_digest(token, k) for k in keys):
            raise HTTPException(401, "invalid or missing API key")

    api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

    def st(request: Request):
        return request.app.state

    def managed(s) -> Supervisor:
        if s.supervisor is None:
            raise HTTPException(409, "Atlas is connected to an external llama-server (ATLAS_LLAMA_URL). "
                                     "Set ATLAS_LLAMA_SERVER_BIN to let Atlas manage llama-server and presets.")
        return s.supervisor

    # --- status ------------------------------------------------------------------------

    @api.get("/status")
    async def status(request: Request):
        s = st(request)
        fp = s.engine.info.fingerprint
        ready = [c for c in s.store.caches_for(fp).values() if c.status == "ready"]
        pool = s.engine.pool
        return {
            "version": __version__,
            "mode": "managed" if s.supervisor else "external",
            "ready": s.engine.ready,
            "message": s.engine.status_message,
            "engine": s.engine.info.to_json(),
            "server": {k: v for k, v in s.supervisor.to_json().items() if k != "log"} if s.supervisor else None,
            "pool": {
                "n_slots": pool.n_slots if pool else 0,
                "waiting": pool.n_waiting if pool else 0,
                "leases": [lease.__dict__ for lease in pool.leases.values()] if pool else [],
            },
            "documents": {
                "count": len(s.store.list_documents()),
                "ready": len(ready),
                "tokens": sum(c.n_tokens for c in ready),
                "kv_bytes": sum(c.kv_bytes for c in ready),
            },
            "limits": {
                "max_question_tokens": settings.max_question_tokens,
                "max_answer_tokens": settings.max_answer_tokens,
                "max_final_tokens": settings.max_final_tokens,
                "reserve_tokens": s.engine.reserve_tokens() if s.engine.info.n_ctx_slot else None,
                "max_part_tokens": (s.engine.info.n_ctx_slot - s.engine.reserve_tokens()) if s.engine.info.n_ctx_slot else None,
                "max_upload_mb": settings.max_upload_mb,
                "enable_thinking": settings.enable_thinking,
                "default_prefill": settings.default_prefill,
                "visual_dpi": settings.visual_dpi,
            },
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS | page_images.IMAGE_EXTENSIONS),
        }

    # --- collections -------------------------------------------------------------------

    @api.get("/collections")
    async def list_collections(request: Request):
        s = st(request)
        counts: dict[str | None, int] = {}
        for d in s.store.list_documents():
            counts[d.collection_id] = counts.get(d.collection_id, 0) + 1
        return [{**c.to_json(), "n_docs": counts.get(c.id, 0)} for c in s.store.list_collections()]

    @api.post("/collections")
    async def create_collection(request: Request, body: CollectionCreate):
        """Create a collection, optionally nested in another one and holding the given documents
        (a chapter made of them, placed where the first of them was)."""
        s = st(request)
        parent = check_collection(s, body.parent_id)
        docs = [get_doc_or_404(s, doc_id) for doc_id in dict.fromkeys(body.document_ids)]
        position = min(d.order for d in docs) if docs else None
        c = s.store.create_collection(body.name.strip(), parent, position)
        for d in docs:
            s.store.update_document(d.id, collection_id=c.id)  # positions kept: the order stays
        return c.to_json()

    def get_collection_or_404(s, collection_id: str):
        c = s.store.get_collection(collection_id)
        if c is None:
            raise HTTPException(404, "collection not found")
        return c

    def check_parent(s, collection_id: str, parent_id: str | None) -> str | None:
        """A collection can move anywhere except into itself or one of its own chapters."""
        parent = check_collection(s, parent_id)
        if parent and parent in s.store.subtree(collection_id):
            raise HTTPException(400, "a collection cannot be moved into itself or into one of its chapters")
        return parent

    @api.patch("/collections/{collection_id}")
    async def update_collection(request: Request, collection_id: str, body: CollectionUpdate):
        s = st(request)
        c = get_collection_or_404(s, collection_id)
        fields = {}
        if body.name:
            fields["name"] = body.name.strip()
        if "parent_id" in body.model_fields_set:
            parent = check_parent(s, collection_id, body.parent_id)
            if parent != c.parent_id:
                fields.update(parent_id=parent, position=None)  # goes after what is already there
        if fields:
            s.store.update_collection(collection_id, **fields)
        return s.store.get_collection(collection_id).to_json()

    @api.delete("/collections/{collection_id}")
    async def delete_collection(request: Request, collection_id: str, delete_documents: bool = False):
        """Without delete_documents, the collection's documents and chapters move up to its parent."""
        s = st(request)
        get_collection_or_404(s, collection_id)
        deleted = []
        if delete_documents:
            subtree = s.store.subtree(collection_id)
            for d in s.store.documents_under(collection_id):
                remove_document(s, d.id)
                deleted.append(d.id)
            for cid in reversed(subtree[1:]):
                s.store.delete_collection(cid)
        s.store.delete_collection(collection_id)
        return {"deleted": collection_id, "deleted_documents": deleted}

    @api.post("/library/order")
    async def order_library(request: Request, body: OrderRequest):
        """Put collections and documents into a collection (or the top level) in the given order."""
        s = st(request)
        parent = check_collection(s, body.parent_id)
        items = []
        for item in body.items:
            if item.kind == "collection":
                get_collection_or_404(s, item.id)
                check_parent(s, item.id, parent)
            else:
                get_doc_or_404(s, item.id)
            items.append((item.kind, item.id))
        s.store.set_order(parent, items)
        return {"ok": True}

    # --- documents ---------------------------------------------------------------------

    def doc_payload(s, doc, cache) -> dict:
        fp = s.engine.info.fingerprint
        wanted = s.ingestor.variant_for(doc) if fp else None
        status = cache.status if cache else ("not_built" if fp else "waiting")
        if fp and not (cache and cache.status in ("queued", "ingesting")):
            if wanted is None:
                status = "needs_vision"  # visual document, but no vision projector is loaded
            elif cache and cache.status == "ready" and cache.built_as != wanted:
                status = "stale"  # built for the other prefill mode or another projector / resolution
        d = doc.to_json()
        d.update({
            "status": status,
            "error": cache.error if cache else None,
            "n_tokens": cache.n_tokens if cache else 0,
            "n_parts": cache.n_parts if cache else 0,
            "kv_bytes": cache.kv_bytes if cache else 0,
            "ingest_ms": cache.ingest_ms if cache else None,
            "fingerprint": cache.fingerprint if cache else None,
            "progress": s.ingestor.progress.get(doc.id),
            "queryable": bool(cache and cache.status == "ready" and wanted and cache.built_as == wanted),
            "visual_capable": page_images.supports_visual(doc.name),
            "has_text": doc.n_chars > 0,
        })
        return d

    def doc_json(s, doc) -> dict:
        return doc_payload(s, doc, s.store.get_cache(doc.id, s.engine.info.fingerprint))

    def check_collection(s, collection_id: str | None) -> str | None:
        if collection_id:
            get_collection_or_404(s, collection_id)
        return collection_id or None

    def register(s, name: str, mime: str | None, data: bytes, text: str, collection_id: str | None,
                 mode: str = "text", n_pages: int = 0) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        existing = s.store.find_by_sha(sha)
        if existing:
            return {"document": doc_json(s, existing), "duplicate": True}
        doc = s.store.create_document(name, mime, sha, len(data), len(text), collection_id, mode, n_pages)
        folder = settings.docs_dir / doc.id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"original{PurePath(name).suffix.lower()}").write_bytes(data)
        (folder / "text.txt").write_text(text, encoding="utf-8")
        s.ingestor.enqueue(doc.id)
        return {"document": doc_json(s, s.store.get_document(doc.id))}

    def remove_document(s, doc_id: str) -> None:
        s.ingestor.cancel(doc_id)
        s.ingestor.remove_parts(doc_id)  # every configuration
        s.store.delete_document(doc_id)
        shutil.rmtree(settings.docs_dir / doc_id, ignore_errors=True)

    @api.get("/documents")
    async def list_documents(request: Request):
        s = st(request)
        caches = s.store.caches_for(s.engine.info.fingerprint)
        return [doc_payload(s, d, caches.get(d.id)) for d in s.store.list_documents()]

    def prepare_upload(name: str, data: bytes, mode: str) -> dict:
        """Extract text and count pages. Images and PDFs without text layer are prefilled visually."""
        visual_ok = page_images.supports_visual(name)
        note = None
        text = ""
        if not page_images.is_image(name):
            try:
                text = extract_text(name, data)
            except ExtractionError as e:
                if not visual_ok:
                    raise
                if mode == "text":
                    note = f"{e}: using visual prefill"
        if visual_ok and (mode == "visual" or not text):
            mode = "visual"
        else:
            mode = "text"
        n_pages = page_images.page_count(name, data) if visual_ok else 0
        return {"text": text, "mode": mode, "n_pages": n_pages, "note": note}

    @api.post("/documents")
    async def upload_documents(request: Request, files: list[UploadFile], collection_id: str | None = Form(None),
                               mode: Literal["text", "visual"] | None = Form(None)):
        s = st(request)
        collection_id = check_collection(s, collection_id)
        limit = settings.max_upload_mb * 1024 * 1024
        results = []
        for f in files:
            name = PurePath(f.filename or "untitled").name[:200] or "untitled"
            data = await f.read(limit + 1)
            if len(data) > limit:
                results.append({"name": name, "error": f"larger than {settings.max_upload_mb} MB"})
                continue
            try:
                prepared = await asyncio.to_thread(prepare_upload, name, data, mode or settings.default_prefill)
            except (ExtractionError, page_images.PageError) as e:
                results.append({"name": name, "error": str(e)})
                continue
            result = register(s, name, f.content_type, data, prepared["text"], collection_id, prepared["mode"],
                              prepared["n_pages"])
            if prepared["note"]:
                result["note"] = prepared["note"]
            results.append(result)
        return {"results": results}

    @api.post("/documents/text")
    async def add_text_document(request: Request, body: TextDocument):
        s = st(request)
        text = normalize(body.text)
        if not text:
            raise HTTPException(400, "text is empty")
        return register(s, body.name.strip(), "text/plain", text.encode(), text,
                        check_collection(s, body.collection_id))

    def get_doc_or_404(s, doc_id: str):
        doc = s.store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        return doc

    @api.get("/documents/{doc_id}")
    async def get_document(request: Request, doc_id: str):
        s = st(request)
        doc = get_doc_or_404(s, doc_id)
        fp = s.engine.info.fingerprint
        return {**doc_json(s, doc),
                "parts": [p.to_json() for p in s.store.get_parts(doc_id, fp, with_tokens=False)] if fp else [],
                "caches": [{**c, "active": c["fingerprint"] == fp} for c in s.store.document_caches(doc_id)]}

    @api.get("/documents/{doc_id}/original")
    async def get_document_original(request: Request, doc_id: str):
        doc = get_doc_or_404(st(request), doc_id)
        original = next((settings.docs_dir / doc_id).glob("original*"), None)
        if original is None:
            raise HTTPException(404, "original file missing")
        return FileResponse(original, filename=doc.name, media_type=doc.mime or "application/octet-stream")

    @api.patch("/documents/{doc_id}")
    async def update_document(request: Request, doc_id: str, body: DocumentUpdate):
        s = st(request)
        get_doc_or_404(s, doc_id)
        fields = {}
        if "name" in body.model_fields_set and body.name:
            fields["name"] = body.name.strip()
        doc = s.store.get_document(doc_id)
        if "collection_id" in body.model_fields_set:
            fields["collection_id"] = check_collection(s, body.collection_id)
            if fields["collection_id"] != doc.collection_id:
                fields["position"] = None  # goes after what is already there
        if body.mode and body.mode != doc.mode:
            if body.mode == "visual" and not page_images.supports_visual(doc.name):
                raise HTTPException(400, "visual prefill works for PDFs and images")
            if body.mode == "text" and doc.n_chars == 0:
                raise HTTPException(400, "this document has no extractable text; it can only be prefilled visually")
            fields["mode"] = body.mode
            if body.mode == "visual" and not doc.n_pages:
                original = next((settings.docs_dir / doc_id).glob("original*"))
                fields["n_pages"] = await asyncio.to_thread(page_images.page_count, doc.name, original.read_bytes())
        if fields:
            s.store.update_document(doc_id, **fields)
        if "mode" in fields and s.engine.info.fingerprint and not s.ingestor.is_busy(doc_id):
            s.ingestor.enqueue(doc_id)  # the cache must be built again from the other input
        return doc_json(s, s.store.get_document(doc_id))

    @api.get("/documents/{doc_id}/pages/{n}")
    async def get_document_page(request: Request, doc_id: str, n: int):
        doc = get_doc_or_404(st(request), doc_id)
        if not page_images.supports_visual(doc.name):
            raise HTTPException(404, "this document has no pages")
        folder = settings.docs_dir / doc_id
        original = next(folder.glob("original*"), None)
        if original is None:
            raise HTTPException(404, "original file missing")
        try:
            pages = await asyncio.to_thread(page_images.ensure_pages, original, folder, settings.visual_dpi)
        except page_images.PageError as e:
            raise HTTPException(422, str(e)) from e
        if not 1 <= n <= len(pages):
            raise HTTPException(404, "no such page")
        return FileResponse(pages[n - 1], media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})

    @api.get("/documents/{doc_id}/text", response_class=PlainTextResponse)
    async def get_document_text(request: Request, doc_id: str):
        get_doc_or_404(st(request), doc_id)
        return (settings.docs_dir / doc_id / "text.txt").read_text(encoding="utf-8")

    # --- sources: where an answer's quotes are in a document ----------------------------------

    def doc_text(s, doc_id: str) -> evidence.DocText:
        path = settings.docs_dir / doc_id / "text.txt"
        if not path.exists():
            raise HTTPException(404, "the document's text is missing")
        return s.queries.texts.get(path)

    @api.post("/documents/{doc_id}/evidence")
    async def locate_evidence(request: Request, doc_id: str, body: EvidenceRequest):
        """Locate the passages an answer quotes (for answers recorded before quotes were located)."""
        s = st(request)
        get_doc_or_404(s, doc_id)
        text = await asyncio.to_thread(doc_text, s, doc_id)
        prefer = None
        if body.part and s.engine.info.fingerprint:
            parts = s.store.get_parts(doc_id, s.engine.info.fingerprint, with_tokens=False)
            if body.part <= len(parts):
                p = parts[body.part - 1]
                prefer = text.pages_span(p.char_start + 1, p.char_end) if p.visual else (p.char_start, p.char_end)
        return {"evidence": await asyncio.to_thread(text.evidence, body.text, body.question, prefer)}

    @api.get("/documents/{doc_id}/passage")
    async def get_passage(request: Request, doc_id: str, start: int, end: int, context: int = 700):
        """A located passage with the text around it, and the pages it is on."""
        s = st(request)
        doc = get_doc_or_404(s, doc_id)
        text = await asyncio.to_thread(doc_text, s, doc_id)
        n = len(text.text)
        start, end = max(0, min(start, n)), max(0, min(end, n))
        if end < start:
            raise HTTPException(400, "end is before start")
        context = max(0, min(context, 20000))
        a = max(0, start - context)
        b = min(n, end + context)
        if a > 0:  # start and end the excerpt at a word break
            cut = text.text.find(" ", a, start)
            a = cut + 1 if cut >= 0 else a
        if b < n:
            cut = text.text.rfind(" ", end, b)
            b = cut if cut > end else b
        return {"doc_id": doc_id, "name": doc.name, "start": start, "end": end, "before": text.text[a:start],
                "passage": text.text[start:end], "after": text.text[end:b], "truncated_before": a > 0,
                "truncated_after": b < n, "page": text.page_at(start), "page_end": text.page_at(max(start, end - 1)),
                "n_pages": doc.n_pages, "pdf": PurePath(doc.name).suffix.lower() == ".pdf"}

    def original_pdf(doc_id: str) -> Path:
        original = next((settings.docs_dir / doc_id).glob("original*"), None)
        if original is None or original.suffix.lower() != ".pdf":
            raise HTTPException(404, "this document is not a PDF")
        return original

    @api.get("/documents/{doc_id}/boxes/{page}")
    async def get_passage_boxes(request: Request, doc_id: str, page: int, start: int, end: int):
        """Where the part of a located passage that is on `page` appears on the rendered page."""
        s = st(request)
        get_doc_or_404(s, doc_id)
        original = original_pdf(doc_id)
        text = await asyncio.to_thread(doc_text, s, doc_id)
        span = text.page_range(page)
        if span is None:
            return {"boxes": [], "score": 0.0}
        a, b = max(start, span[0]), min(end, span[1])
        if b - a < 4:
            return {"boxes": [], "score": 0.0}
        try:
            return await asyncio.to_thread(evidence.page_boxes, original.read_bytes(), page, text.text[a:b])
        except ValueError as e:
            raise HTTPException(404, str(e)) from e

    @api.get("/documents/{doc_id}/render/{n}")
    async def render_document_page(request: Request, doc_id: str, n: int, width: int = 900):
        """One page of a PDF (or an image document) as PNG, for viewing a source."""
        doc = get_doc_or_404(st(request), doc_id)
        folder = settings.docs_dir / doc_id
        width = max(200, min(width, 2000)) // 100 * 100  # a few sizes, so the disk cache is reused
        if page_images.is_image(doc.name):
            pages = await asyncio.to_thread(page_images.ensure_pages, next(folder.glob("original*")), folder,
                                            settings.visual_dpi)
            return FileResponse(pages[0], media_type="image/png")
        original = original_pdf(doc_id)
        cached = folder / "view" / f"page-{n}-{width}.png"
        if not cached.exists():
            try:
                png = await asyncio.to_thread(pdftools.thumbnail, original.read_bytes(), n, width)
            except pdftools.PdfToolError as e:
                raise HTTPException(404, str(e)) from e
            cached.parent.mkdir(exist_ok=True)
            cached.write_bytes(png)
        return FileResponse(cached, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @api.post("/documents/build-missing")
    async def build_missing_caches(request: Request):
        """Build the caches the running model configuration lacks (after switching presets)."""
        s = st(request)
        if not s.engine.info.fingerprint:
            raise HTTPException(409, "no model is running")
        return {"queued": s.ingestor.build_missing()}

    @api.post("/documents/{doc_id}/reingest")
    async def reingest_document(request: Request, doc_id: str):
        s = st(request)
        get_doc_or_404(s, doc_id)
        if not s.engine.info.fingerprint:
            raise HTTPException(409, "no model is running")
        if s.ingestor.is_busy(doc_id):
            raise HTTPException(409, "document is already being ingested")
        s.ingestor.enqueue(doc_id)
        return doc_json(s, s.store.get_document(doc_id))

    @api.delete("/documents/{doc_id}")
    async def delete_document(request: Request, doc_id: str):
        s = st(request)
        get_doc_or_404(s, doc_id)
        remove_document(s, doc_id)
        return {"deleted": doc_id}

    # --- tools: PDF shards and token estimates ---------------------------------------------

    tools_dir = settings.data_dir / "tools"

    async def count_many(s, texts: list[str]) -> tuple[list[int], bool]:
        """Token counts with the running model's tokenizer, or a rough estimate (4 characters per token)."""
        if s.engine.ready:
            sem = asyncio.Semaphore(8)

            async def one(text: str) -> int:
                async with sem:
                    return await s.engine.count(text) if text else 0
            try:
                return list(await asyncio.gather(*(one(t) for t in texts))), True
            except Exception:
                pass
        return [-(-len(t) // 4) for t in texts], False

    def throughput(s) -> dict:
        """Measured prefill speed and KV bytes per token of the running configuration."""
        caches = [c for c in s.store.caches_for(s.engine.info.fingerprint).values()
                  if c.status == "ready" and c.n_tokens >= 1000 and c.ingest_ms]
        tokens = sum(c.n_tokens for c in caches)
        info = {"prefill_tps": round(tokens / (sum(c.ingest_ms for c in caches) / 1000), 1) if caches else None,
                "kv_bytes_per_token": round(sum(c.kv_bytes for c in caches) / tokens) if caches else None}
        sup = s.supervisor
        if info["kv_bytes_per_token"] is None and sup and sup.preset and Path(sup.preset["model_path"]).is_file():
            est = models.estimate(models.describe_file(Path(sup.preset["model_path"])), sup.preset["ctx_per_slot"],
                                  1, sup.preset["kv_type"], sup.preset.get("extra_args", ""),
                                  swa_full=bool(sup.preset.get("swa_full")), cpu_moe=bool(sup.preset.get("cpu_moe")))
            info["kv_bytes_per_token"] = est.get("kv_bytes_per_token")
        n_ctx = s.engine.info.n_ctx_slot
        info["part_tokens"] = n_ctx - s.engine.reserve_tokens() if n_ctx else None
        info["model"] = s.engine.info.config_label or s.engine.info.model
        return info

    def tool_source(s, file_data: bytes | None, file_name: str | None, doc_id: str | None) -> tuple[bytes, str]:
        if doc_id:
            doc = get_doc_or_404(s, doc_id)
            original = next((settings.docs_dir / doc_id).glob("original*"), None)
            if original is None or original.suffix.lower() != ".pdf":
                raise HTTPException(400, "the document is not a PDF")
            return original.read_bytes(), doc.name
        if file_data is None:
            raise HTTPException(400, "upload a PDF or choose a library document")
        return file_data, PurePath(file_name or "document.pdf").name

    @api.post("/tools/pdf")
    async def analyze_pdf(request: Request, file: UploadFile | None = None, doc_id: str | None = Form(None)):
        s = st(request)
        limit = settings.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1) if file else None
        if data is not None and len(data) > limit:
            raise HTTPException(413, f"larger than {settings.max_upload_mb} MB")
        data, name = tool_source(s, data, file.filename if file else None, doc_id)
        try:
            result = await asyncio.to_thread(pdftools.analyze, data)
        except pdftools.PdfToolError as e:
            raise HTTPException(400, str(e)) from e
        tokens, exact = await count_many(s, [p["text"] for p in result["pages"]])
        ws_id = hashlib.sha256(data).hexdigest()[:32]
        folder = tools_dir / ws_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "source.pdf").write_bytes(data)
        pages = [{k: v for k, v in p.items() if k != "text"} | {"tokens": t} for p, t in zip(result["pages"], tokens)]
        overhead = await s.engine.count(document_block(name, 98, 99, "")) if s.engine.ready else 64
        return {"id": ws_id, "name": name, "doc_id": doc_id, "n_pages": result["n_pages"], "pages": pages,
                "outline": result["outline"], "exact": exact, "total_tokens": sum(tokens),
                "shard_overhead_tokens": overhead, **throughput(s)}

    def tool_workspace(ws_id: str) -> Path:
        try:
            return pdftools.workspace(tools_dir, ws_id)
        except (pdftools.PdfToolError, FileNotFoundError) as e:
            raise HTTPException(404, "this PDF is no longer loaded; analyze it again") from e

    @api.get("/tools/pdf/{ws_id}/thumb/{n}")
    async def pdf_thumbnail(ws_id: str, n: int, width: int = 180):
        folder = tool_workspace(ws_id)
        width = max(60, min(width, 800))
        cached = folder / f"thumb-{n}-{width}.png"
        if not cached.exists():
            try:
                png = await asyncio.to_thread(pdftools.thumbnail, (folder / "source.pdf").read_bytes(), n, width)
            except pdftools.PdfToolError as e:
                raise HTTPException(404, str(e)) from e
            cached.write_bytes(png)
        return FileResponse(cached, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @api.post("/tools/pdf/{ws_id}/shards")
    async def create_shards(request: Request, ws_id: str, body: ShardRequest):
        """Cut the loaded PDF into shards and add them to the library as documents."""
        s = st(request)
        data = (tool_workspace(ws_id) / "source.pdf").read_bytes()
        collection_id = check_collection(s, body.collection_id)
        try:
            source = await asyncio.to_thread(pdftools.reader, data)  # parsed once for all shards
        except pdftools.PdfToolError as e:
            raise HTTPException(400, str(e)) from e
        results, taken = [], set()
        folders: dict[tuple[str, ...], str | None] = {(): collection_id}

        def folder_for(path: list[str]) -> str | None:
            """The chapter collection for a shard, created on first use (nested in the target)."""
            path = [" ".join(p.split())[:120] for p in path if p.strip()]
            for depth in range(1, len(path) + 1):
                key = tuple(path[:depth])
                if key not in folders:
                    parent = folders[key[:-1]]
                    existing = next((c for c in s.store.list_collections()
                                     if c.parent_id == parent and c.name == key[-1]), None)
                    folders[key] = existing.id if existing else s.store.create_collection(key[-1], parent).id
            return folders[tuple(path)]

        for shard in body.shards:
            name = pdftools.shard_filename(shard.name, taken)
            try:
                shard_data = await asyncio.to_thread(pdftools.build_shard, source, shard.pages)
                prepared = await asyncio.to_thread(prepare_upload, name, shard_data, body.mode or settings.default_prefill)
            except (pdftools.PdfToolError, ExtractionError, page_images.PageError) as e:
                results.append({"name": name, "error": str(e)})
                continue
            result = register(s, name, "application/pdf", shard_data, prepared["text"], folder_for(shard.folder),
                              prepared["mode"], prepared["n_pages"])
            if prepared["note"]:
                result["note"] = prepared["note"]
            results.append(result)
        return {"results": results}

    @api.post("/tools/pdf/{ws_id}/zip")
    async def download_shards(ws_id: str, body: ShardRequest):
        data = (tool_workspace(ws_id) / "source.pdf").read_bytes()
        try:
            archive = await asyncio.to_thread(pdftools.build_zip, data, [(x.name, x.pages) for x in body.shards])
        except pdftools.PdfToolError as e:
            raise HTTPException(400, str(e)) from e
        return Response(archive, media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="shards.zip"'})

    @api.delete("/tools/pdf/{ws_id}")
    async def close_pdf(ws_id: str):
        shutil.rmtree(tool_workspace(ws_id), ignore_errors=True)
        return {"closed": ws_id}

    @api.post("/tools/estimate")
    async def estimate_tokens(request: Request, file: UploadFile | None = None, text: str | None = Form(None)):
        """How many tokens a text or file is, and what that means for the running model."""
        s = st(request)
        pages = None
        if file is not None:
            limit = settings.max_upload_mb * 1024 * 1024
            data = await file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"larger than {settings.max_upload_mb} MB")
            name = PurePath(file.filename or "file").name
            try:
                text = await asyncio.to_thread(extract_text, name, data) if not page_images.is_image(name) else ""
            except ExtractionError as e:
                if not page_images.supports_visual(name):
                    raise HTTPException(400, str(e)) from e
                text = ""
            if page_images.supports_visual(name):
                try:
                    pages = await asyncio.to_thread(page_images.page_count, name, data)
                except page_images.PageError:
                    pages = None
        text = text or ""
        [tokens], exact = await count_many(s, [text])
        info = throughput(s)
        part = info["part_tokens"]
        return {"tokens": tokens, "exact": exact, "chars": len(text), "words": len(text.split()), "pages": pages,
                **info, "parts": (-(-tokens // part) if part and tokens else None),
                "kv_bytes": tokens * info["kv_bytes_per_token"] if info["kv_bytes_per_token"] else None,
                "prefill_s": round(tokens / info["prefill_tps"], 1) if info["prefill_tps"] else None}

    # --- queries -----------------------------------------------------------------------

    @api.post("/query")
    async def query(request: Request, body: QueryRequest):
        s = st(request)
        try:
            plan = await s.queries.prepare(body.question, body.document_ids, body.thinking, body.collection_ids,
                                           body.conversation_id)
        except QueryError as e:
            raise HTTPException(e.status, str(e)) from e

        async def sse():
            async for event in s.queries.stream(plan):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @api.get("/queries")
    async def list_queries(request: Request, limit: int = 50):
        return st(request).store.list_queries(max(1, min(limit, 500)))

    # --- conversations -----------------------------------------------------------------

    def get_conversation_or_404(s, conversation_id: str) -> dict:
        conversation = s.store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "conversation not found")
        return conversation

    @api.get("/conversations")
    async def list_conversations(request: Request):
        return st(request).store.list_conversations()

    @api.post("/conversations")
    async def create_conversation(request: Request, body: ConversationBody):
        return st(request).store.create_conversation(body.title.strip())

    @api.get("/conversations/{conversation_id}")
    async def get_conversation(request: Request, conversation_id: str):
        s = st(request)
        conversation = get_conversation_or_404(s, conversation_id)
        return {**conversation, "turns": s.store.conversation_turns(conversation_id)}

    @api.patch("/conversations/{conversation_id}")
    async def rename_conversation(request: Request, conversation_id: str, body: ConversationBody):
        s = st(request)
        get_conversation_or_404(s, conversation_id)
        s.store.rename_conversation(conversation_id, body.title.strip())
        return s.store.get_conversation(conversation_id)

    @api.delete("/conversations/{conversation_id}")
    async def delete_conversation(request: Request, conversation_id: str):
        s = st(request)
        get_conversation_or_404(s, conversation_id)
        s.store.delete_conversation(conversation_id)
        return {"deleted": conversation_id}

    # --- runtime settings --------------------------------------------------------------

    def settings_payload() -> dict:
        return {"values": {f: getattr(settings, f) for f in RUNTIME_FIELDS}, "defaults": defaults}

    @api.get("/settings")
    async def get_runtime_settings():
        return settings_payload()

    @api.patch("/settings")
    async def update_runtime_settings(request: Request):
        s = st(request)
        body = await request.json()
        if not isinstance(body, dict) or not set(body) <= set(RUNTIME_FIELDS):
            unknown = sorted(set(body) - set(RUNTIME_FIELDS)) if isinstance(body, dict) else body
            raise HTTPException(422, f"unknown settings: {unknown}")
        overrides = {k: v for k, v in (s.store.get_state("settings") or {}).items() if k in RUNTIME_FIELDS}
        for key, value in body.items():
            if value is None:
                overrides.pop(key, None)  # back to the default
            else:
                overrides[key] = value
        try:
            validated = RuntimeSettings(**{**defaults, **overrides})
        except ValidationError as e:
            raise HTTPException(422, "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
        old_prompt = settings.system_prompt
        old_source = settings.build_update_source
        for key in RUNTIME_FIELDS:
            setattr(settings, key, getattr(validated, key))
        if settings.build_update_source != old_source and s.updater:
            s.updater.check_in_background()  # build (or fetch) the current release in the chosen form
        s.store.set_state("settings", {k: getattr(validated, k) for k in overrides})
        s.ingestor.set_concurrency(s.engine.info.n_slots or 1)
        if settings.system_prompt != old_prompt and s.engine.info.connected and not s.engine.paused:
            # the system prompt is baked into every cached prefix: this is a new configuration
            await s.engine.connect()
            _register_config(s.engine, s.store)
        s.ingestor.reconcile()
        return settings_payload()

    # --- presets & llama-server --------------------------------------------------------

    async def model_index() -> dict[str, models.ModelInfo]:
        found = await asyncio.to_thread(models.discover, settings.model_dirs, settings.scan_model_caches)
        return {m.path: m for m in found}

    def preset_payload(s, p: dict, index: dict[str, models.ModelInfo], active: str | None) -> dict:
        path = Path(p["model_path"])
        info = index.get(p["model_path"])
        if info is None and path.is_file():
            info = models.describe_file(path)
        build = builds.inspect(p.get("binary") or standard_build(settings, s.store) or "")
        mmproj = Path(p.get("mmproj") or "")
        mmproj_bytes = mmproj.stat().st_size if p.get("mmproj") and mmproj.is_file() else 0
        draft = describe_draft(p.get("draft_model"), index)
        effective, source, missing = sampling.resolve(p.get("sampling"), (info.sampling or {}) if info else {})
        warnings = builds.preset_warnings(build, info.arch if info else None, p.get("extra_args", ""))
        warnings += draft_warnings(build, info, p.get("draft_model"), draft)
        if missing:
            warnings.append(sampling.describe_missing(missing).capitalize())
        if p["id"] == active and s.engine.info.swa_restore_ok is False:
            warnings.append("This llama-server build prefills restored documents again for this sliding-window model "
                            "(no cache reuse): enable “Full SWA cache”, or use a build with the SWA restore fix.")
        return {**p, "active": p["id"] == active, "model_found": path.is_file(),
                "sampling_model": (info.sampling or {}) if info else {}, "sampling_effective": effective,
                "sampling_source": source, "sampling_missing": missing,
                "mmproj_found": not p.get("mmproj") or mmproj.is_file(),
                "draft_found": not p.get("draft_model") or draft is not None,
                "draft": draft.to_json() if draft else None,
                "model": info.to_json() if info else None,
                "build": build.to_json(),
                "warnings": warnings,
                "estimate": models.estimate(info, p["ctx_per_slot"], p["slots"], p["kv_type"],
                                            p.get("extra_args", ""), p.get("gpu_layers", "all"), mmproj_bytes,
                                            bool(p.get("swa_full")), draft, bool(p.get("cpu_moe")))}

    def describe_draft(path: str | None, index: dict[str, models.ModelInfo] | None = None) -> models.ModelInfo | None:
        if not path:
            return None
        if index and path in index:
            return index[path]
        return models.describe_file(Path(path)) if Path(path).is_file() else None

    def draft_warnings(build, info, path: str | None, draft) -> list[str]:
        if not path:
            return []
        if draft is None:
            return ["Draft model file not found."]
        out = []
        if problem := models.draft_problem(info, draft):
            out.append(f"The draft model will not work: {problem}.")
        if build.flags and "--model-draft" not in build.flags and "-md" not in build.flags:
            out.append(f"{build.version or 'This build'} does not support draft models (--model-draft).")
        return out

    @api.get("/presets")
    async def list_presets(request: Request):
        s = st(request)
        index = await model_index()
        active = s.store.get_state("active_preset")
        presets = s.store.list_presets()
        payload = await asyncio.to_thread(lambda: [preset_payload(s, p, index, active) for p in presets])
        return {"presets": payload, "active": active}

    def validate_preset(body: dict, complete: bool = True) -> dict:
        """Check a preset; `complete` also requires every core sampling parameter to be known."""
        try:
            data = PresetConfig(**body).model_dump()
        except ValidationError as e:
            raise HTTPException(422, "; ".join(
                (f"{x['loc'][-1]}: " if x["loc"] and x["loc"][0] == "sampling" else "")
                + x["msg"].removeprefix("Value error, ") for x in e.errors()))
        if complete:
            _, _, missing = preset_sampling(data)
            if missing:
                raise HTTPException(422, sampling.describe_missing(missing))
        return data

    @api.post("/presets/estimate")
    async def estimate_preset(request: Request, body: dict):
        """Memory estimate for unsaved preset values (used live by the preset editor)."""
        standard = standard_build(settings, st(request).store)
        path = Path(str(body.get("model_path", "")))
        if not path.is_file() or path.suffix != ".gguf":
            return {}

        def check() -> dict:
            info = models.describe_file(path)
            extra = str(body.get("extra_args") or "")
            build = builds.inspect(str(body.get("binary") or "") or standard or "")
            mmproj = Path(str(body.get("mmproj") or ""))
            mmproj_bytes = mmproj.stat().st_size if body.get("mmproj") and mmproj.is_file() else 0
            draft_path = str(body.get("draft_model") or "").strip()
            draft = describe_draft(draft_path)
            return {"sampling_model": info.sampling or {},
                    **models.estimate(info, int(body.get("ctx_per_slot") or 0), int(body.get("slots") or 1),
                                      str(body.get("kv_type") or "f16"), extra, str(body.get("gpu_layers") or "all"),
                                      mmproj_bytes, bool(body.get("swa_full")), draft, bool(body.get("cpu_moe"))),
                    "warnings": builds.preset_warnings(build, info.arch, extra)
                    + draft_warnings(build, info, draft_path, draft),
                    "build": build.to_json()}
        try:
            return await asyncio.to_thread(check)
        except (ValueError, OSError):
            return {}

    @api.post("/presets")
    async def create_preset(request: Request):
        s = st(request)
        managed(s)
        data = await asyncio.to_thread(validate_preset, await request.json())
        if data.get("binary"):
            add_custom_build(s, data["binary"])  # a build typed into a preset joins the custom builds
        preset_id = s.store.save_preset(None, data)
        return s.store.get_preset(preset_id)

    @api.put("/presets/{preset_id}")
    async def update_preset(request: Request, preset_id: str):
        s = st(request)
        managed(s)
        old = s.store.get_preset(preset_id)
        if not old:
            raise HTTPException(404, "preset not found")
        data = await asyncio.to_thread(validate_preset, await request.json())
        if data.get("binary") and data["binary"] not in {b["command"] for b in custom_builds(s)}:
            add_custom_build(s, data["binary"])
        s.store.save_preset(preset_id, data)
        running = s.supervisor.preset if s.supervisor else None
        restart = False
        if running and running.get("id") == preset_id:
            # sampling and the name take effect at once; everything else needs a restart
            s.supervisor.preset = {**running, "sampling": data["sampling"], "name": data["name"]}
            s.supervisor.apply_sampling(s.supervisor.preset)
            same = {"sampling", "name", "id"}
            restart = any(running.get(k) != v for k, v in data.items() if k not in same)
        return {**s.store.get_preset(preset_id), "restart_required": restart}

    @api.delete("/presets/{preset_id}")
    async def delete_preset(request: Request, preset_id: str):
        s = st(request)
        if s.store.get_state("active_preset") == preset_id and s.supervisor and s.supervisor.state != "stopped":
            raise HTTPException(409, "this preset is running; activate another one or stop the server first")
        s.store.delete_preset(preset_id)
        return {"deleted": preset_id}

    @api.post("/presets/{preset_id}/activate", status_code=202)
    async def activate_preset(request: Request, preset_id: str):
        s = st(request)
        sup = managed(s)
        preset = s.store.get_preset(preset_id)
        if not preset:
            raise HTTPException(404, "preset not found")
        await asyncio.to_thread(validate_preset, {k: preset[k] for k in PresetConfig.model_fields if k in preset}, False)
        sup._spawn(sup.activate(preset))
        return {"activating": preset_id}

    # --- llama-server builds -----------------------------------------------------------

    # Builds: one standard build (upstream llama.cpp with Atlas's patches, kept up to date by the
    # updater) and a list of custom builds the user adds for models that need another llama.cpp.

    def custom_builds(s) -> list[dict]:
        """[{command, name}] as added in Settings (older versions stored bare commands)."""
        return [{"command": b, "name": ""} if isinstance(b, str) else b for b in s.store.get_state("builds") or []]

    def add_custom_build(s, command: str, name: str = "") -> None:
        items = custom_builds(s)
        for item in items:
            if item["command"] == command:
                item["name"] = name or item["name"]
                break
        else:
            items.append({"command": command, "name": name})
        s.store.set_state("builds", items)

    def build_name(info, name: str) -> str:
        return name or info.version or Path(info.path).parent.name

    @api.get("/builds")
    async def list_builds(request: Request):
        s = st(request)
        standard = standard_build(settings, s.store)
        custom = custom_builds(s)
        presets = s.store.list_presets()
        listed = {b["command"] for b in custom}
        # builds presets use that are in no list (set before custom builds were a list, or pinned
        # by an update): shown so they can be added or the preset switched
        unlisted = sorted({p["binary"] for p in presets if p.get("binary") and p["binary"] not in listed
                           and p["binary"] != standard})
        commands = [standard or "", *(b["command"] for b in custom), *unlisted]
        infos = await asyncio.to_thread(lambda: [builds.inspect(c) if c else None for c in commands])
        updates = s.updater.to_json() if s.updater else None
        tag = updates["standard_tag"] if updates else None
        installed = {i["tag"]: i for i in (updates or {}).get("installed", [])}
        used = lambda command: [p["name"] for p in presets if (p.get("binary") or standard) == command]  # noqa: E731
        standard_entry = None
        if infos[0]:
            standard_entry = {**infos[0].to_json(), "default": True, "tag": tag,
                              "patched": (installed.get(tag) or {}).get("patched"),
                              "configured": standard == settings.llama_server_bin, "used_by": used(standard)}
        entries = [{**info.to_json(), "name": build_name(info, b["name"]), "default": False, "added": True,
                    "listed": True, "used_by": used(b["command"])} for b, info in zip(custom, infos[1:1 + len(custom)])]
        entries += [{**info.to_json(), "name": build_name(info, ""), "default": False, "added": False,
                     "listed": False, "used_by": used(c)} for c, info in zip(unlisted, infos[1 + len(custom):])]
        return {"default": standard, "configured": settings.llama_server_bin, "updates": updates,
                "standard": standard_entry, "custom": entries,
                "builds": ([standard_entry] if standard_entry else []) + entries}

    def updater_or_409(s) -> BuildUpdater:
        return managed(s) and s.updater

    @api.get("/builds/updates")
    async def build_updates(request: Request):
        return updater_or_409(st(request)).to_json()

    @api.post("/builds/updates/check", status_code=202)
    async def check_build_updates(request: Request):
        updater = updater_or_409(st(request))
        updater.check_in_background()
        return updater.to_json()

    @api.post("/builds/updates/rollback")
    async def roll_back_build(request: Request):
        updater = updater_or_409(st(request))
        try:
            await updater.roll_back()
        except UpdateError as e:
            raise HTTPException(409, str(e)) from e
        return updater.to_json()

    @api.post("/builds/updates/unskip")
    async def unskip_build(request: Request, tag: str):
        updater = updater_or_409(st(request))
        updater.unskip(tag)
        return updater.to_json()

    @api.post("/builds")
    async def add_build(request: Request, body: BuildRequest):
        """Add a custom build (or rename one that is in the list)."""
        s = st(request)
        command = body.command.strip()
        info = await asyncio.to_thread(builds.inspect, command)
        if info.problem == "file not found":
            raise HTTPException(400, f"not found: {command}")
        add_custom_build(s, command, body.name.strip())
        return {**info.to_json(), "name": build_name(info, body.name.strip())}

    @api.delete("/builds")
    async def remove_build(request: Request, command: str):
        s = st(request)
        users = [p["name"] for p in s.store.list_presets() if p.get("binary") == command]
        if users:
            raise HTTPException(409, f"used by the preset{'s' if len(users) > 1 else ''} {', '.join(users)}: "
                                     "switch them to another build first")
        s.store.set_state("builds", [b for b in custom_builds(s) if b["command"] != command])
        return {"removed": command}

    @api.get("/server")
    async def server_info(request: Request):
        s = st(request)
        return {
            "mode": "managed" if s.supervisor else "external",
            "llama_url": settings.llama_url,
            "wsl": "microsoft" in platform.release().lower(),
            "llama_server_bin": standard_build(settings, s.store) if s.supervisor else None,
            "supervisor": s.supervisor.to_json() if s.supervisor else None,
            "gpus": await models.gpu_info(),
            "ram_total": models.system_memory(),
        }

    def check_host(host: str) -> str:
        host = host.strip().strip("[]")
        if host != "localhost" and not re.fullmatch(r"[0-9A-Za-z.:%_-]+", host):
            raise HTTPException(422, f"host: '{host}' is not an IP address or host name")
        return host

    def can_listen(host: str, port: int) -> str | None:
        """Why llama-server could not listen on host:port, or None."""
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError as e:
            return f"unknown host '{host}': {e.strerror or e}"
        family, _, _, _, addr = infos[0]
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            # like llama-server: a port it just left may still have connections in TIME_WAIT
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(addr)
                sock.listen()
            except OSError as e:
                if e.errno == 98:  # EADDRINUSE
                    return f"port {port} is already in use by another program"
                return f"cannot listen on {host}:{port}: {e.strerror or e}"
        return None

    @api.put("/server/address", status_code=202)
    async def set_server_address(request: Request, body: ServerAddress):
        """Where llama-server listens (and the key it requires); restarts it if it is running."""
        s = st(request)
        sup = managed(s)
        host = check_host(body.host)
        if body.port == settings.port:
            raise HTTPException(422, f"port: {body.port} is Atlas's own port")
        in_use_by_us = sup.listening is not None and sup.listening[1] == body.port
        if not in_use_by_us and (problem := await asyncio.to_thread(can_listen, host, body.port)):
            raise HTTPException(409, problem)
        key = settings.llama_api_key if body.api_key is None else (body.api_key.strip() or None)
        if key and not re.fullmatch(r"[\x21-\x7e]+", key):
            raise HTTPException(422, "api_key: use printable characters without spaces")
        settings.llama_host, settings.llama_port, settings.llama_api_key = host, body.port, key
        s.store.set_state("llama_address", {"host": host, "port": body.port, "api_key": key})
        restart = sup.preset is not None and sup.state not in ("stopped", "stopping")
        if restart:
            sup._spawn(sup.activate(sup.preset))
        return {"host": host, "port": body.port, "url": connect_url(host, body.port), "api_key_set": bool(key),
                "exposed": host not in LOOPBACK, "restarting": restart}

    @api.post("/server/restart", status_code=202)
    async def restart_server(request: Request):
        s = st(request)
        sup = managed(s)
        preset_id = s.store.get_state("active_preset") or (sup.preset or {}).get("id")
        preset = s.store.get_preset(preset_id) if preset_id else None
        if not preset:
            raise HTTPException(409, "no preset is active")
        sup._spawn(sup.activate(preset))
        return {"restarting": preset["id"]}

    @api.post("/server/stop", status_code=202)
    async def stop_server(request: Request):
        sup = managed(st(request))
        sup._spawn(sup.stop())
        return {"stopping": True}

    # --- models ------------------------------------------------------------------------

    @api.get("/models")
    async def list_models(request: Request):
        index = await model_index()
        projectors = await asyncio.to_thread(models.discover_projectors, settings.model_dirs, settings.scan_model_caches)
        return {
            "models": [m.to_json() for m in index.values()],
            "projectors": projectors,
            "dirs": [str(d) for d in settings.model_dirs],
            "download_dir": str(settings.model_dirs[0]),
            "scan_caches": settings.scan_model_caches,
            "gpus": await models.gpu_info(),
            "ram_total": models.system_memory(),
            "gpu_baseline": st(request).supervisor.gpu_baseline if st(request).supervisor else None,
        }

    @api.get("/models/hf")
    async def hf_files(request: Request, repo: str):
        try:
            return {"repo": repo, "files": await st(request).downloader.list_files(repo.strip())}
        except DownloadError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:  # network errors
            raise HTTPException(502, f"Hugging Face request failed: {e}") from e

    @api.post("/models/download")
    async def download_model(request: Request, body: DownloadRequest):
        try:
            job = await st(request).downloader.enqueue(body.repo.strip(), body.file)
        except DownloadError as e:
            raise HTTPException(400, str(e)) from e
        return job.to_json()

    @api.get("/models/downloads")
    async def list_downloads(request: Request):
        return [j.to_json() for j in reversed(list(st(request).downloader.jobs.values()))]

    @api.delete("/models/downloads/{job_id}")
    async def cancel_download(request: Request, job_id: str):
        st(request).downloader.cancel(job_id)
        return {"cancelled": job_id}

    # --- caches ------------------------------------------------------------------------

    @api.get("/caches")
    async def list_caches(request: Request):
        s = st(request)
        fp = s.engine.info.fingerprint
        return {"active": fp, "configs": [{**c, "active": c["fingerprint"] == fp} for c in s.store.cache_summary()]}

    @api.delete("/caches/{fingerprint}")
    async def delete_caches(request: Request, fingerprint: str):
        s = st(request)
        if fingerprint == s.engine.info.fingerprint:
            raise HTTPException(409, "these caches belong to the running model")
        return {"deleted": fingerprint, "freed_bytes": s.ingestor.drop_config(fingerprint)}

    app.include_router(api)

    @app.get("/healthz")
    async def healthz(request: Request):
        return {"ok": True, "llama_ready": request.app.state.engine.ready}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


def _register_config(engine: Engine, store: Store) -> None:
    if engine.info.fingerprint:
        store.touch_config(engine.info.fingerprint, engine.info.config_label or engine.info.model or "model",
                           {"model": engine.info.model, **engine.extra_ident})


async def _monitor(engine: Engine, ingestor: Ingestor, supervisor: Supervisor | None, store: Store) -> None:
    if supervisor:
        ingestor.start()
        await supervisor.startup()
    else:
        await engine.connect()
        ingestor.start()
        _register_config(engine, store)
        ingestor.reconcile(startup=True)
        log.info("connected to llama-server: model=%s slots=%d ctx/slot=%d fingerprint=%s",
                 engine.info.model, engine.info.n_slots, engine.info.n_ctx_slot, engine.info.fingerprint)
    while True:
        await asyncio.sleep(MONITOR_INTERVAL_S)
        try:
            await monitor_step(engine, ingestor, supervisor, store)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("monitor iteration failed")


async def monitor_step(engine: Engine, ingestor: Ingestor, supervisor: Supervisor | None = None,
                       store: Store | None = None) -> None:
    """One health/consistency check of llama-server; flags stale caches as needed."""
    if supervisor and (supervisor.busy or supervisor.state != "running"):
        return  # the supervisor owns (re)starts in managed mode
    if not await engine.llama.health():
        if engine.info.connected:
            log.warning("llama-server became unreachable")
        engine.info.connected = False
        engine.info.error = "llama-server unreachable"
        return
    if not engine.info.connected:
        log.info("llama-server is back, re-validating")
        await engine.connect()
        changed = True
    else:
        changed = await engine.refresh()
        if engine.restarted:
            log.info("llama-server was restarted, re-validating stored KV caches")
            await engine.probe_kv_dir()
            changed = True
    if changed:
        if store:
            _register_config(engine, store)
        ingestor.reconcile()

"""HTTP API and static frontend."""

import asyncio
import hashlib
import hmac
import json
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path, PurePath

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from . import __version__, builds, models
from .config import RUNTIME_FIELDS, RuntimeSettings, Settings, get_settings
from .downloads import Downloader, DownloadError
from .engine import Engine
from .extract import SUPPORTED_EXTENSIONS, ExtractionError, extract_text, normalize
from .ingest import Ingestor
from .query import QueryError, QueryService
from .store import Store
from .supervisor import PresetConfig, Supervisor

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


class CollectionBody(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    document_ids: list[str] = Field(default_factory=list, max_length=2000)
    collection_ids: list[str] = Field(default_factory=list, max_length=200)
    thinking: bool | None = None


class DownloadRequest(BaseModel):
    repo: str
    file: str


class BuildRequest(BaseModel):
    command: str = Field(min_length=1, max_length=2000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    if settings.managed:
        settings.llama_url = f"http://127.0.0.1:{settings.llama_port}"
    defaults = {f: getattr(settings, f) for f in RUNTIME_FIELDS}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        settings.kv_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dirs[0].mkdir(parents=True, exist_ok=True)
        store = Store(settings.db_path)
        for key, value in (store.get_state("settings") or {}).items():
            if key in RUNTIME_FIELDS:
                setattr(settings, key, value)
        engine = Engine(settings)
        ingestor = Ingestor(engine, store, settings)
        downloader = Downloader(settings.hf_endpoint, settings.model_dirs[0])
        downloader.start()
        supervisor = Supervisor(settings, store, engine, ingestor) if settings.managed else None
        app.state.store = store
        app.state.engine = engine
        app.state.ingestor = ingestor
        app.state.queries = QueryService(engine, store, ingestor, settings)
        app.state.downloader = downloader
        app.state.supervisor = supervisor
        monitor = asyncio.create_task(_monitor(engine, ingestor, supervisor, store), name="llama-monitor")
        try:
            yield
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            if supervisor:
                await supervisor.shutdown()
            await downloader.stop()
            await ingestor.stop()
            await engine.aclose()
            store.close()

    app = FastAPI(title="Atlas CAG", version=__version__, lifespan=lifespan)
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
                "max_upload_mb": settings.max_upload_mb,
                "enable_thinking": settings.enable_thinking,
            },
            "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
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
    async def create_collection(request: Request, body: CollectionBody):
        return st(request).store.create_collection(body.name.strip()).to_json()

    def get_collection_or_404(s, collection_id: str):
        c = s.store.get_collection(collection_id)
        if c is None:
            raise HTTPException(404, "collection not found")
        return c

    @api.patch("/collections/{collection_id}")
    async def rename_collection(request: Request, collection_id: str, body: CollectionBody):
        s = st(request)
        get_collection_or_404(s, collection_id)
        s.store.rename_collection(collection_id, body.name.strip())
        return s.store.get_collection(collection_id).to_json()

    @api.delete("/collections/{collection_id}")
    async def delete_collection(request: Request, collection_id: str, delete_documents: bool = False):
        s = st(request)
        get_collection_or_404(s, collection_id)
        deleted = []
        if delete_documents:
            for d in s.store.list_documents(collection_id):
                remove_document(s, d.id)
                deleted.append(d.id)
        s.store.delete_collection(collection_id)  # remaining documents become unfiled
        return {"deleted": collection_id, "deleted_documents": deleted}

    # --- documents ---------------------------------------------------------------------

    def doc_payload(s, doc, cache) -> dict:
        fp = s.engine.info.fingerprint
        d = doc.to_json()
        d.update({
            "status": cache.status if cache else ("not_built" if fp else "waiting"),
            "error": cache.error if cache else None,
            "n_tokens": cache.n_tokens if cache else 0,
            "n_parts": cache.n_parts if cache else 0,
            "kv_bytes": cache.kv_bytes if cache else 0,
            "ingest_ms": cache.ingest_ms if cache else None,
            "fingerprint": cache.fingerprint if cache else None,
            "progress": s.ingestor.progress.get(doc.id),
            "queryable": bool(cache and cache.status == "ready"),
        })
        return d

    def doc_json(s, doc) -> dict:
        return doc_payload(s, doc, s.store.get_cache(doc.id, s.engine.info.fingerprint))

    def check_collection(s, collection_id: str | None) -> str | None:
        if collection_id:
            get_collection_or_404(s, collection_id)
        return collection_id or None

    def register(s, name: str, mime: str | None, data: bytes, text: str, collection_id: str | None) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        existing = s.store.find_by_sha(sha)
        if existing:
            return {"document": doc_json(s, existing), "duplicate": True}
        doc = s.store.create_document(name, mime, sha, len(data), len(text), collection_id)
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

    @api.post("/documents")
    async def upload_documents(request: Request, files: list[UploadFile], collection_id: str | None = Form(None)):
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
                text = await asyncio.to_thread(extract_text, name, data)
            except ExtractionError as e:
                results.append({"name": name, "error": str(e)})
                continue
            results.append(register(s, name, f.content_type, data, text, collection_id))
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
                "parts": [p.to_json() for p in s.store.get_parts(doc_id, fp, with_tokens=False)] if fp else []}

    @api.patch("/documents/{doc_id}")
    async def update_document(request: Request, doc_id: str, body: DocumentUpdate):
        s = st(request)
        get_doc_or_404(s, doc_id)
        fields = {}
        if "name" in body.model_fields_set and body.name:
            fields["name"] = body.name.strip()
        if "collection_id" in body.model_fields_set:
            fields["collection_id"] = check_collection(s, body.collection_id)
        if fields:
            s.store.update_document(doc_id, **fields)
        return doc_json(s, s.store.get_document(doc_id))

    @api.get("/documents/{doc_id}/text", response_class=PlainTextResponse)
    async def get_document_text(request: Request, doc_id: str):
        get_doc_or_404(st(request), doc_id)
        return (settings.docs_dir / doc_id / "text.txt").read_text(encoding="utf-8")

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

    # --- queries -----------------------------------------------------------------------

    @api.post("/query")
    async def query(request: Request, body: QueryRequest):
        s = st(request)
        try:
            plan = await s.queries.prepare(body.question, body.document_ids, body.thinking, body.collection_ids)
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
        overrides = s.store.get_state("settings") or {}
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
        for key in RUNTIME_FIELDS:
            setattr(settings, key, getattr(validated, key))
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

    def preset_payload(p: dict, index: dict[str, models.ModelInfo], active: str | None) -> dict:
        path = Path(p["model_path"])
        info = index.get(p["model_path"])
        if info is None and path.is_file():
            info = models.describe_file(path)
        build = builds.inspect(p.get("binary") or settings.llama_server_bin or "")
        return {**p, "active": p["id"] == active, "model_found": path.is_file(),
                "model": info.to_json() if info else None,
                "build": build.to_json(),
                "warnings": builds.preset_warnings(build, info.arch if info else None, p.get("extra_args", "")),
                "estimate": models.estimate(info, p["ctx_per_slot"], p["slots"], p["kv_type"],
                                            p.get("extra_args", ""), p.get("gpu_layers", "all"))}

    @api.get("/presets")
    async def list_presets(request: Request):
        s = st(request)
        index = await model_index()
        active = s.store.get_state("active_preset")
        presets = s.store.list_presets()
        payload = await asyncio.to_thread(lambda: [preset_payload(p, index, active) for p in presets])
        return {"presets": payload, "active": active}

    def validate_preset(body: dict) -> dict:
        try:
            return PresetConfig(**body).model_dump()
        except ValidationError as e:
            raise HTTPException(422, "; ".join(x["msg"].removeprefix("Value error, ") for x in e.errors()))

    @api.post("/presets/estimate")
    async def estimate_preset(body: dict):
        """Memory estimate for unsaved preset values (used live by the preset editor)."""
        path = Path(str(body.get("model_path", "")))
        if not path.is_file() or path.suffix != ".gguf":
            return {}

        def check() -> dict:
            info = models.describe_file(path)
            extra = str(body.get("extra_args") or "")
            build = builds.inspect(str(body.get("binary") or "") or settings.llama_server_bin or "")
            return {**models.estimate(info, int(body.get("ctx_per_slot") or 0), int(body.get("slots") or 1),
                                      str(body.get("kv_type") or "f16"), extra, str(body.get("gpu_layers") or "all")),
                    "warnings": builds.preset_warnings(build, info.arch, extra), "build": build.to_json()}
        try:
            return await asyncio.to_thread(check)
        except (ValueError, OSError):
            return {}

    @api.post("/presets")
    async def create_preset(request: Request):
        s = st(request)
        managed(s)
        preset_id = s.store.save_preset(None, validate_preset(await request.json()))
        return s.store.get_preset(preset_id)

    @api.put("/presets/{preset_id}")
    async def update_preset(request: Request, preset_id: str):
        s = st(request)
        managed(s)
        if not s.store.get_preset(preset_id):
            raise HTTPException(404, "preset not found")
        s.store.save_preset(preset_id, validate_preset(await request.json()))
        active = s.store.get_state("active_preset") == preset_id
        return {**s.store.get_preset(preset_id), "restart_required": active}

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
        validate_preset({k: preset[k] for k in PresetConfig.model_fields if k in preset})
        sup._spawn(sup.activate(preset))
        return {"activating": preset_id}

    # --- llama-server builds -----------------------------------------------------------

    def added_builds(s) -> list[str]:
        return s.store.get_state("builds") or []

    @api.get("/builds")
    async def list_builds(request: Request):
        s = st(request)
        added = added_builds(s)
        found = await asyncio.to_thread(builds.discover, settings.llama_server_bin, added)
        default = builds.inspect(settings.llama_server_bin).path if settings.llama_server_bin else None
        return {"default": settings.llama_server_bin, "builds": [
            {**b.to_json(), "default": b.path == default and b.command == settings.llama_server_bin,
             "added": b.command in added} for b in found]}

    @api.post("/builds")
    async def add_build(request: Request, body: BuildRequest):
        s = st(request)
        command = body.command.strip()
        info = await asyncio.to_thread(builds.inspect, command)
        if info.problem == "file not found":
            raise HTTPException(400, f"not found: {command}")
        added = added_builds(s)
        if command not in added:
            s.store.set_state("builds", added + [command])
        return info.to_json()

    @api.delete("/builds")
    async def remove_build(request: Request, command: str):
        s = st(request)
        s.store.set_state("builds", [c for c in added_builds(s) if c != command])
        return {"removed": command}

    @api.get("/server")
    async def server_info(request: Request):
        s = st(request)
        return {
            "mode": "managed" if s.supervisor else "external",
            "llama_url": settings.llama_url,
            "llama_server_bin": settings.llama_server_bin,
            "supervisor": s.supervisor.to_json() if s.supervisor else None,
            "gpus": await models.gpu_info(),
            "ram_total": models.system_memory(),
        }

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
        return {
            "models": [m.to_json() for m in index.values()],
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

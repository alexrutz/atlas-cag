"""Download GGUF models from the Hugging Face Hub into the models directory."""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .store import new_id

log = logging.getLogger("atlas.downloads")

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class DownloadError(RuntimeError):
    pass


def hf_token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


@dataclass
class Job:
    id: str
    repo: str
    file: str
    files: list[str]
    dest: str
    total: int
    done: int = 0
    status: str = "queued"  # queued | downloading | done | failed | cancelled
    error: str | None = None
    started: float = field(default_factory=time.time)
    speed: float = 0.0

    def to_json(self) -> dict:
        return dict(self.__dict__)


class Downloader:
    def __init__(self, endpoint: str, models_dir: Path):
        self.endpoint = endpoint.rstrip("/")
        self.models_dir = models_dir
        self.jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._current: asyncio.Task | None = None
        self._current_job: str | None = None

    def _client(self) -> httpx.AsyncClient:
        token = hf_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        return httpx.AsyncClient(headers=headers, follow_redirects=True,
                                 timeout=httpx.Timeout(60.0, connect=15.0))

    async def list_files(self, repo: str) -> list[dict]:
        """GGUF files in a Hub repo; split models are grouped under their first shard."""
        if not _REPO.match(repo):
            raise DownloadError("repository must look like 'owner/name'")
        async with self._client() as c:
            r = await c.get(f"{self.endpoint}/api/models/{repo}/tree/main", params={"recursive": "true"})
        if r.status_code == 401 or r.status_code == 403:
            raise DownloadError("access denied: the repository is gated or private (set HF_TOKEN)")
        if r.status_code == 404:
            raise DownloadError(f"repository {repo} not found")
        r.raise_for_status()
        entries = [e for e in r.json() if e.get("type") == "file" and e["path"].endswith(".gguf")]
        groups: dict[str, dict] = {}
        for e in entries:
            name = e["path"]
            if re.search(r"(^|/)imatrix", name, re.I):
                continue
            size = (e.get("lfs") or {}).get("size") or e.get("size") or 0
            m = _SHARD.search(name)
            key = _SHARD.sub("", name) if m else name
            # vision projectors (mmproj) are downloaded next to the model for visual prefill
            projector = bool(re.search(r"(^|[/_.-])mmproj", name, re.I))
            g = groups.setdefault(key, {"file": name, "files": [], "size": 0, "projector": projector})
            g["files"].append(name)
            g["size"] += size
            if m and m.group(1) == "00001":
                g["file"] = name
        return sorted(groups.values(), key=lambda g: g["file"].lower())

    def start(self) -> None:
        self._worker = asyncio.create_task(self._run(), name="downloads")

    async def stop(self) -> None:
        for t in (self._current, self._worker):
            if t:
                t.cancel()
        await asyncio.gather(*(t for t in (self._current, self._worker) if t), return_exceptions=True)

    async def enqueue(self, repo: str, file: str) -> Job:
        listing = await self.list_files(repo)
        group = next((g for g in listing if file in g["files"]), None)
        if group is None:
            raise DownloadError(f"{file} is not a GGUF model file in {repo}")
        dest = self.models_dir / repo.replace("/", "__")
        job = Job(id=new_id(), repo=repo, file=group["file"], files=group["files"], dest=str(dest),
                  total=group["size"])
        self.jobs[job.id] = job
        self._queue.put_nowait(job.id)
        return job

    def cancel(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "downloading"):
            job.status = "cancelled"  # a queued job is skipped by the worker
            if self._current and self._current_job == job_id:
                self._current.cancel()

    async def _run(self) -> None:
        while True:
            job = self.jobs[await self._queue.get()]
            if job.status != "queued":
                continue
            self._current = asyncio.create_task(self._download(job))
            self._current_job = job.id
            try:
                await self._current
            except asyncio.CancelledError:
                if job.status != "cancelled":
                    raise
            except Exception as e:
                log.exception("download of %s/%s failed", job.repo, job.file)
                job.status, job.error = "failed", str(e) or type(e).__name__
            finally:
                self._current = self._current_job = None

    async def _download(self, job: Job) -> None:
        job.status = "downloading"
        dest = Path(job.dest)
        dest.mkdir(parents=True, exist_ok=True)
        base = 0
        async with self._client() as c:
            for name in job.files:
                target = dest / Path(name).name
                if target.exists():
                    base += target.stat().st_size
                    job.done = base
                    continue
                part = target.with_name(target.name + ".part")
                have = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={have}-"} if have else {}
                url = f"{self.endpoint}/{job.repo}/resolve/main/{name}"
                async with c.stream("GET", url, headers=headers) as r:
                    if r.status_code == 416:  # already complete
                        pass
                    elif r.status_code not in (200, 206):
                        raise DownloadError(f"HTTP {r.status_code} for {name}")
                    else:
                        if r.status_code == 200:
                            have = 0  # server ignored the range: start over
                        t0, n0 = time.time(), have
                        with open(part, "ab" if have else "wb") as f:
                            async for chunk in r.aiter_bytes(1 << 20):
                                f.write(chunk)
                                have += len(chunk)
                                job.done = base + have
                                dt = time.time() - t0
                                if dt > 1:
                                    job.speed = (have - n0) / dt
                part.rename(target)
                base += target.stat().st_size
                job.done = base
        job.status = "done"
        log.info("downloaded %s/%s to %s", job.repo, job.file, dest)

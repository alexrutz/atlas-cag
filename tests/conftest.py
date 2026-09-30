"""Fixtures: the fake llama-server and an Atlas app connected to it (external mode)."""

import httpx
import pytest

from atlas.api import create_app
from atlas.config import Settings

from .fake_llama import FakeLlama
from .helpers import serve_in_thread, wait_for


@pytest.fixture
def fake(tmp_path):
    kv = tmp_path / "kv"
    kv.mkdir()
    fake = FakeLlama(kv)
    fake.url, stop = serve_in_thread(fake.app)
    yield fake
    stop()


def atlas_settings(fake, tmp_path, **overrides) -> Settings:
    values = dict(max_question_tokens=256, max_answer_tokens=256, max_final_tokens=256, part_overlap_tokens=32,
                  build_on_model_change=True, ocr=False)  # OCR tests switch it on with a fake recognizer
    return Settings(
        _env_file=None,
        llama_url=fake.url,
        kv_dir=fake.kv_dir,
        data_dir=tmp_path / "data",
        models_dirs=str(tmp_path / "models"),
        scan_model_caches=False,
        **{**values, **overrides},
    )


@pytest.fixture
async def atlas(fake, tmp_path, request):
    overrides = getattr(request, "param", {})
    app = create_app(atlas_settings(fake, tmp_path, **overrides))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://atlas") as client:
            await wait_for(client, lambda s: s["ready"], "/api/status")
            client.app = app
            yield client

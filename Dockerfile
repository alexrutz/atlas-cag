# Atlas in managed mode: the official llama.cpp CUDA image provides llama-server, Atlas runs it
# from the presets configured in the UI.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PYTHON=/usr/bin/python3

WORKDIR /atlas
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY atlas ./atlas
RUN uv sync --frozen --no-dev

ENV PATH="/atlas/.venv/bin:$PATH" \
    ATLAS_HOST=0.0.0.0 \
    ATLAS_PORT=8000 \
    ATLAS_DATA_DIR=/data \
    ATLAS_KV_DIR=/data/kv \
    ATLAS_MODELS_DIRS=/models \
    ATLAS_LLAMA_SERVER_BIN=/app/llama-server \
    HF_HOME=/data/hf

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s \
  CMD python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('ATLAS_PORT', '8000'))"
ENTRYPOINT []
CMD ["atlas"]

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings can be overridden with ATLAS_* environment variables or a .env file."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    # --- llama-server ---------------------------------------------------------------
    # Managed mode: path (or command) of llama-server. Atlas then starts it itself from the
    # active preset. Leave empty to connect to an external llama-server at ATLAS_LLAMA_URL.
    llama_server_bin: str | None = None
    # Managed mode: where llama-server listens. 127.0.0.1 = this computer only; 0.0.0.0 = every
    # network interface, so other programs (OpenAI-compatible clients) can use it too. Also
    # editable in Settings → Model; the values saved there override these.
    llama_host: str = "127.0.0.1"
    llama_port: int = 8081
    llama_start_timeout_s: float = 900.0
    llama_url: str = "http://127.0.0.1:8080"  # external mode: the llama-server to connect to
    # Bearer token llama-server requires (managed: Atlas starts it with this key; external: the
    # key of the server at llama_url). Empty = no key.
    llama_api_key: str | None = None
    # Must be the SAME directory that llama-server was started with via --slot-save-path.
    # Atlas verifies this at startup with a probe file.
    kv_dir: Path = Path("data/kv")
    request_timeout_s: float = 3600.0

    # --- llama-server build updates (managed mode) --------------------------------------
    # Keeps the standard build (used by presets without their own build) up to date from the
    # GitHub releases of build_update_repo. off | install: download and use it from the next
    # llama-server start | apply: also restart llama-server as soon as it is idle.
    build_updates: Literal["off", "install", "apply"] = "install"
    build_update_repo: str = "ai-dock/llama.cpp-cuda"
    # Part of the release asset's file name; empty = the CUDA package for this machine's CPU.
    build_update_asset: str = ""
    build_update_interval_h: float = 6.0
    # "release": the prebuilt packages above. "patched": the same llama.cpp release built here from
    # source with Atlas's fixes (atlas/patches/llama.cpp: restored slots keep the sliding-window
    # cache; slot files load with any slot count). Needs git, cmake, a C++ compiler and nvcc.
    build_update_source: Literal["release", "patched"] = "patched"
    build_source_repo: str = "https://github.com/ggml-org/llama.cpp"
    build_cuda_arch: str = ""  # CMAKE_CUDA_ARCHITECTURES; empty = this machine's GPU (nvidia-smi)
    build_jobs: int = 0  # parallel compile jobs; 0 = every CPU core (compiling runs at low priority)
    github_api: str = "https://api.github.com"
    pypi_url: str = "https://pypi.org"  # CUDA runtime wheels, only if no local copy is found

    # --- storage / server -------------------------------------------------------------
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 8000
    # Comma-separated bearer tokens. Empty = no authentication.
    api_keys: str = ""
    max_upload_mb: int = 200

    # --- models -----------------------------------------------------------------------
    # Comma-separated directories scanned for .gguf files; downloads go into the first one.
    models_dirs: str = "data/models"
    # Also list GGUF files from the Hugging Face cache and llama.cpp's -hf download cache.
    scan_model_caches: bool = True
    hf_endpoint: str = "https://huggingface.co"

    # --- ingestion --------------------------------------------------------------------
    # Slots used for ingestion at most. Capped at (slots - 1) so queries never starve.
    ingest_concurrency: int = 1
    # Overlap between consecutive parts when a document exceeds one slot's context.
    part_overlap_tokens: int = 256
    # Build KV caches automatically: for new documents, for every document when a new model
    # configuration becomes active, and to repair missing or broken caches.
    auto_build_caches: bool = True
    # When another model configuration becomes active (other preset, model, KV type, build that
    # computes differently), build the caches it lacks for the whole library. Off: documents show
    # "not built" for it until built from the Library (Build all, or per document).
    build_on_model_change: bool = False
    # How new PDFs and images are prefilled: "text" (extracted text) or "visual" (page images,
    # needs a preset with a vision projector). Each document can be switched later.
    default_prefill: Literal["text", "visual"] = "text"
    # Resolution PDF pages are rendered at for visual prefill; more dots = more image tokens.
    visual_dpi: int = 120

    # --- generation -------------------------------------------------------------------
    # Off: every per-document answer is synthesized (best recall in testing). On: each answer
    # ends with a self-rated coverage and answers rated "none" are dropped before synthesis,
    # which is faster over many documents but trusts the model's rating.
    relevance_filter: bool = False
    # Token limits; 0 or empty = no limit (generation runs until the model stops or the slot is
    # full; the Stop button ends it). Set them to bound answer time.
    max_question_tokens: int | None = None
    max_answer_tokens: int | None = None  # per-document answers (map phase and single-document mode)
    max_final_tokens: int | None = None  # synthesized answer (reduce phase)
    # Sampling (temperature, top-p, …) belongs to presets; see atlas/sampling.py.
    # Default for chat templates that support a thinking switch (Qwen3, etc.). Overridable per query.
    enable_thinking: bool = True
    max_thinking_tokens: int | None = None  # reasoning budget per generation when thinking is on
    # Room kept free in every slot when a document is split into parts, for the conversation, the
    # question, thinking and the answer. Empty = automatic: 1/8 of the slot, 4k to 64k tokens.
    reserve_tokens: int | None = None
    # Send the conversation's earlier questions and answers with every question (never the
    # documents or the model's thinking).
    chat_history: bool = True

    system_prompt: str = (
        "You are Atlas, an enterprise document assistant. You answer questions strictly "
        "based on the document supplied by the user. Do not use outside knowledge. "
        "Be precise, reference the relevant passages, and never invent facts."
    )
    synthesis_prompt: str = (
        "You are Atlas, an enterprise document assistant. You receive findings that were "
        "extracted independently from different documents, or from different parts of one long "
        "document, in answer to the same question. Combine them into one direct, well-structured "
        "answer in the language of the question, without preamble and without a separate list of "
        "sources. Cite the findings you use inline with their bracketed numbers, e.g. [1] or "
        "[2][3], and keep any citations already present in the findings. Skip findings that merely "
        "say their document does not address the question. If two findings make incompatible "
        "claims about the same fact, mention the conflict briefly. Use only information from the "
        "findings."
    )

    @field_validator("max_question_tokens", "max_answer_tokens", "max_final_tokens", "max_thinking_tokens",
                     "reserve_tokens", mode="before")
    @classmethod
    def _empty_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v  # ATLAS_MAX_ANSWER_TOKENS= means no limit

    @field_validator("kv_dir", "data_dir", mode="after")
    @classmethod
    def _expand_path(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("llama_server_bin", mode="after")
    @classmethod
    def _expand_bin(cls, v: str | None) -> str | None:
        if not v or not v.strip():
            return None
        # expand ~ in every word, e.g. "~/llama.cpp/build/bin/llama-server" or "python ~/fake.py"
        return " ".join(os.path.expanduser(w) if w.startswith("~") else w for w in v.split(" "))

    @property
    def managed(self) -> bool:
        return bool(self.llama_server_bin)

    @property
    def model_dirs(self) -> list[Path]:
        # absolute, so discovered paths match the paths stored in presets
        return [Path(p.strip()).expanduser().resolve() for p in self.models_dirs.split(",") if p.strip()]

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "docs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "atlas.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()


class RuntimeSettings(BaseModel):
    """Settings editable at runtime from the UI. Values are persisted and override the env."""

    max_question_tokens: int | None = Field(ge=0, le=4_194_304)  # 0 / None: no limit
    max_answer_tokens: int | None = Field(ge=0, le=4_194_304)
    max_final_tokens: int | None = Field(ge=0, le=4_194_304)
    enable_thinking: bool
    max_thinking_tokens: int | None = Field(ge=0, le=4_194_304)
    reserve_tokens: int | None = Field(ge=0, le=4_194_304)  # 0 / None: automatic
    chat_history: bool
    relevance_filter: bool
    ingest_concurrency: int = Field(ge=1, le=32)
    part_overlap_tokens: int = Field(ge=0, le=16384)
    auto_build_caches: bool
    build_on_model_change: bool
    default_prefill: Literal["text", "visual"]
    visual_dpi: int = Field(ge=36, le=400)
    build_updates: Literal["off", "install", "apply"]
    build_update_source: Literal["release", "patched"]
    system_prompt: str = Field(min_length=1, max_length=20000)
    synthesis_prompt: str = Field(min_length=1, max_length=20000)


RUNTIME_FIELDS = tuple(RuntimeSettings.model_fields)

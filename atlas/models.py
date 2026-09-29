"""Discover local GGUF models, describe them, and estimate the memory a preset needs."""

import asyncio
import dataclasses
import os
import re
import shlex
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from . import sampling
from .gguf import GGUFError, read_metadata, read_tensor_sizes

# llama_ftype values (general.file_type) -> common quantization names
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
}
# bytes per KV element for llama.cpp cache types
KV_TYPE_BYTES = {"f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
                 "q4_1": 20 / 32, "q4_0": 18 / 32}

_SHARD = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
_SKIP = re.compile(r"(^|[-_.])(mmproj|imatrix)", re.I)
_PROJECTOR = re.compile(r"(^|[-_.])mmproj", re.I)
# What converters store as general.name when the model's config has none: the name of the folder
# they converted from ("hf_format", "merged", "output", …) or a snapshot hash. The file says more.
_PLACEHOLDER_NAMES = {
    "hf", "hf format", "hf model", "hf format model", "model", "models", "base model", "local model", "new model",
    "merged", "merged model", "output", "outputs", "out", "final", "final model", "checkpoint", "checkpoints",
    "snapshot", "snapshots", "main", "tmp", "temp", "converted", "export", "exported", "pytorch model",
    "safetensors", "transformers", "unknown", "untitled", "gguf",
}
_QUANT_SUFFIX = re.compile(r"[-_.](?:UD[-_])?(?:I?Q\d\w*|[BM]?F\d+\w*|MXFP4\w*)$", re.I)


def display_name(meta_name: str | None, file: str) -> str:
    """general.name, unless it is a placeholder: then the file name without shard and quant suffix."""
    name = (meta_name or "").strip()
    key = re.sub(r"[\W_]+", " ", name).strip().lower()
    if name and key not in _PLACEHOLDER_NAMES and not re.fullmatch(r"[0-9a-f]{32,}", name):
        return name
    stem = _SHARD.sub("", file).removesuffix(".gguf")
    return _QUANT_SUFFIX.sub("", stem) or stem


@dataclass
class ModelInfo:
    path: str
    file: str
    source: str  # "models dir", "Hugging Face cache", "llama.cpp cache"
    repo: str | None
    size_bytes: int
    shards: int
    arch: str | None = None
    name: str | None = None
    size_label: str | None = None
    quant: str | None = None
    ctx_train: int | None = None
    n_layers: int | None = None
    n_attn_layers: int | None = None
    sliding_window: int | None = None
    hybrid: bool = False
    kv_bytes_per_token_f16: int | None = None  # all attention layers, K + V (+ indexer keys), f16
    # part of the above in sliding-window layers: without --swa-full they only cache the window
    kv_swa_bytes_per_token_f16: int = 0
    kv_layer_max_f16: int = 0  # largest full-attention layer, K + V per token at f16 (compute buffer)
    recurrent_bytes_per_slot: int = 0
    # weight bytes by where llama.cpp can place them (see estimate())
    expert_bytes_by_layer: dict[int, int] | None = None
    lazy_bytes: int = 0  # per-layer / n-gram embeddings: can stay on disk with --lazy-mode on
    input_bytes: int = 0  # token embeddings: always kept in system RAM by llama.cpp
    tied_embeddings: bool = False  # no output.weight: the embeddings double as output layer, copied to the GPU
    sampling: dict | None = None  # recommended sampling from general.sampling.* (see sampling.py)
    tokenizer: str | None = None  # tokenizer.ggml.model ("gpt2", "llama", …): a draft model must match
    vocab_size: int | None = None
    error: str | None = None

    def to_json(self) -> dict:
        return dict(self.__dict__)


def _hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def _llama_cache() -> Path:
    return Path(os.environ.get("LLAMA_CACHE", Path.home() / ".cache" / "llama.cpp"))


def _describe(info: ModelInfo) -> None:
    try:
        meta = read_metadata(info.path)
    except (OSError, GGUFError) as e:
        info.error = str(e)
        return
    arch = meta.get("general.architecture")
    info.arch = arch
    info.name = display_name(meta.get("general.name") or meta.get("general.basename"), info.file)
    info.tokenizer = meta.get("tokenizer.ggml.model")
    tokens = meta.get("tokenizer.ggml.tokens")  # large arrays are skipped by the reader, only counted
    info.vocab_size = (len(tokens) if isinstance(tokens, list) else tokens.get("skipped_array") if isinstance(tokens, dict)
                       else meta.get(f"{meta.get('general.architecture')}.vocab_size"))
    info.size_label = meta.get("general.size_label")
    info.quant = FILE_TYPES.get(meta.get("general.file_type"), None)
    info.sampling = sampling.from_model(meta)

    def a(key, default=None):
        return meta.get(f"{arch}.{key}", default)

    info.ctx_train = a("context_length")
    n_layers = a("block_count") or 0
    n_layers -= a("nextn_predict_layers", 0) or 0  # multi-token-prediction layers hold no KV cache
    info.n_layers = n_layers
    n_head = a("attention.head_count") or 1
    if isinstance(n_head, list):
        n_head = max(n_head) or 1
    embd = a("embedding_length") or 0
    k_len = a("attention.key_length") or (embd // n_head if embd else 0)
    v_len = a("attention.value_length") or k_len
    kv_heads = a("attention.head_count_kv", n_head)
    if isinstance(kv_heads, list):  # per-layer (e.g. LFM2: 0 for convolution layers)
        per_layer = [h for h in kv_heads[:n_layers] if h]
    else:
        interval = a("full_attention_interval")  # hybrid linear/full attention (Qwen3.5, Qwen3-Next)
        n_attn = n_layers // interval if interval else n_layers
        per_layer = [kv_heads] * n_attn
    info.n_attn_layers = len(per_layer)
    # sparse-attention indexer (e.g. qwen4exp) caches one extra key per attention layer and token
    indexer = a("attention.indexer.key_length") or 0
    info.sliding_window = a("attention.sliding_window")
    # which attention layers use the sliding window: per-layer flags, or every Nth layer is global
    pattern = a("attention.sliding_window_pattern")
    swa = [False] * len(per_layer)
    if info.sliding_window and len(per_layer) == n_layers:
        if isinstance(pattern, list):
            swa = [bool(x) for x in pattern[:n_layers]] + [False] * max(0, n_layers - len(pattern))
        elif isinstance(pattern, int) and pattern > 1:
            swa = [(i + 1) % pattern != 0 for i in range(n_layers)]
    layer_bytes = [(h * (k_len + v_len) + indexer) * 2 for h in per_layer]
    info.kv_bytes_per_token_f16 = int(sum(layer_bytes)) or None
    info.kv_swa_bytes_per_token_f16 = int(sum(b for b, s in zip(layer_bytes, swa) if s))
    info.kv_layer_max_f16 = int(max((b for b, s in zip(layer_bytes, swa) if not s), default=0))
    info.hybrid = info.n_attn_layers < n_layers
    inner, state = a("ssm.inner_size"), a("ssm.state_size")
    if inner and state:  # recurrent + convolution state of linear-attention / SSM layers, kept in f32
        conv = (a("ssm.conv_kernel", 1) - 1) * (inner + 2 * (a("ssm.group_count") or 0) * state)
        info.recurrent_bytes_per_slot = int((n_layers - info.n_attn_layers) * (inner * state + conv) * 4)


_cache: dict[tuple[str, float, int], ModelInfo] = {}


def _info(path: Path, source: str, repo: str | None) -> ModelInfo:
    real = path.resolve()
    st = real.stat()
    m = _SHARD.search(path.name)
    shards = int(m.group(2)) if m else 1
    size = st.st_size
    if shards > 1:
        size = sum((p.resolve().stat().st_size for p in path.parent.glob(_SHARD.sub("-*-of-" + m.group(2) + ".gguf",
                                                                                    path.name))), 0)
    key = (str(path), st.st_mtime, size)  # the description only; where it was found comes from the caller
    if key not in _cache:
        info = ModelInfo(path=str(path), file=path.name, source=source, repo=repo, size_bytes=size, shards=shards)
        _describe(info)
        if not info.error:
            shard_files = sorted(path.parent.glob(_SHARD.sub("-*-of-" + m.group(2) + ".gguf", path.name))) if m else [path]
            _classify(info, shard_files)
        _cache[key] = info
    return dataclasses.replace(_cache[key], source=source, repo=repo)


_EXPERTS = re.compile(r"^blk\.(\d+)\.ffn_\w*_exps\b")
_LAZY = re.compile(r"per_layer_token_embd|^ple_|\.ple_embd")


def _classify(info: ModelInfo, files: list[Path]) -> None:
    """Split weights into routed experts (per layer), lazily loadable embeddings and input embeddings."""
    experts: dict[int, int] = {}
    names: set[str] = set()
    try:
        for f in files:
            for name, size in read_tensor_sizes(f).items():
                names.add(name)
                if m := _EXPERTS.match(name):
                    experts[int(m.group(1))] = experts.get(int(m.group(1)), 0) + size
                elif _LAZY.search(name):
                    info.lazy_bytes += size
                elif name == "token_embd.weight":
                    info.input_bytes += size
    except (OSError, GGUFError, struct.error):
        return
    info.expert_bytes_by_layer = experts or None
    info.tied_embeddings = bool(info.input_bytes) and "output.weight" not in names


def _candidates(root: Path, projectors: bool = False) -> list[Path]:
    if not root.is_dir():
        return []
    out = []
    for p in root.rglob("*.gguf"):
        if projectors != bool(_PROJECTOR.search(p.name)) or (not projectors and _SKIP.search(p.name)):
            continue
        m = _SHARD.search(p.name)
        if m and m.group(1) != "00001":
            continue
        out.append(p)
    return out


def _walk(models_dirs: list[Path], scan_caches: bool, projectors: bool):
    """(path, source, repo) of the GGUF files in the models directories and download caches."""
    for d in models_dirs:
        for p in _candidates(d, projectors):
            yield p, "models dir", None
    if scan_caches:
        hub = _hub_cache()
        if hub.is_dir():
            for repo_dir in hub.glob("models--*"):
                repo = repo_dir.name[len("models--"):].replace("--", "/")
                snaps = sorted((repo_dir / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
                for snap in snaps:
                    for p in _candidates(snap, projectors):
                        yield p, "Hugging Face cache", repo
        for p in _candidates(_llama_cache(), projectors):
            yield p, "llama.cpp cache", None


def discover(models_dirs: list[Path], scan_caches: bool = True) -> list[ModelInfo]:
    found: dict[str, ModelInfo] = {}
    for p, source, repo in _walk(models_dirs, scan_caches, projectors=False):
        real = str(p.resolve())
        if real not in found and p.exists():
            found[real] = _info(p, source, repo)
    return sorted(found.values(), key=lambda m: (m.name or m.file).lower())


def describe_projector(path: Path, source: str = "custom path", repo: str | None = None) -> dict:
    """A vision (or audio) projector for multimodal models: the file llama-server loads with --mmproj."""
    info = {"path": str(path), "file": path.name, "source": source, "repo": repo,
            "size_bytes": path.resolve().stat().st_size, "name": None, "projector_type": None,
            "vision": False, "audio": False, "error": None}
    try:
        meta = read_metadata(str(path))
    except (OSError, GGUFError) as e:
        info["error"] = str(e)
        return info
    info["name"] = display_name(meta.get("general.name"), path.name)
    info["projector_type"] = meta.get("clip.projector_type") or meta.get("clip.vision.projector_type")
    info["vision"] = bool(meta.get("clip.has_vision_encoder"))
    info["audio"] = bool(meta.get("clip.has_audio_encoder"))
    return info


def discover_projectors(models_dirs: list[Path], scan_caches: bool = True) -> list[dict]:
    found: dict[str, dict] = {}
    for p, source, repo in _walk(models_dirs, scan_caches, projectors=True):
        real = str(p.resolve())
        if real not in found and p.exists():
            found[real] = describe_projector(p, source, repo)
    return sorted(found.values(), key=lambda m: m["path"])


def describe_file(path: Path) -> ModelInfo:
    """Describe a GGUF file outside the scanned directories (e.g. a preset's custom path)."""
    return _info(path, "custom path", None)


def _flags(extra_args: str) -> dict:
    try:
        args = shlex.split(extra_args or "")
    except ValueError:
        args = []
    out = {"cpu_moe": False, "n_cpu_moe": 0, "lazy": False, "ubatch": 512}
    for i, arg in enumerate(args):
        flag, _, inline = arg.partition("=")
        value = inline or (args[i + 1] if i + 1 < len(args) else "")
        if flag in ("-cmoe", "--cpu-moe"):
            out["cpu_moe"] = True
        elif flag in ("-ncmoe", "--n-cpu-moe") and value.isdigit():
            out["n_cpu_moe"] = int(value)
        elif flag in ("-lzm", "--lazy-mode"):
            out["lazy"] = value.lower() in ("on", "1", "true", "enabled")
        elif flag in ("-ub", "--ubatch-size") and value.isdigit():
            out["ubatch"] = int(value)
    return out


# llama.cpp sizes a sliding-window cache as window + micro-batch per sequence, padded to 256
def _swa_cells(window: int, ctx: int, ubatch: int = 512) -> int:
    return min(ctx, -(-(window + ubatch) // 256) * 256)


# llama.cpp refuses speculative decoding when the vocabularies differ by more than this
DRAFT_VOCAB_TOLERANCE = 128


def draft_problem(model: ModelInfo | None, draft: ModelInfo | None) -> str | None:
    """Why a draft model cannot speed up `model`, or None if it looks compatible."""
    if model is None or draft is None:
        return None
    if draft.error:
        return f"the draft model cannot be read: {draft.error}"
    if draft.path == model.path:
        return "the draft model is the model itself"
    if model.tokenizer and draft.tokenizer and model.tokenizer != draft.tokenizer:
        return f"its tokenizer ({draft.tokenizer}) differs from the model's ({model.tokenizer})"
    if model.vocab_size and draft.vocab_size and abs(model.vocab_size - draft.vocab_size) > DRAFT_VOCAB_TOLERANCE:
        return (f"its vocabulary ({draft.vocab_size:,} tokens) differs from the model's ({model.vocab_size:,}): "
                "use a smaller model of the same family")
    return None


def estimate(model: ModelInfo | None, ctx_per_slot: int, slots: int, kv_type: str,
             extra_args: str = "", gpu_layers: str = "all", mmproj_bytes: int = 0, swa_full: bool = False,
             draft: ModelInfo | None = None) -> dict:
    """Rough memory estimate for a preset, in bytes: GPU (weights + KV + recurrent state + vision
    projector), system RAM (CPU-offloaded experts, input embeddings) and SSD (lazily read embeddings)."""
    if model is None or not model.kv_bytes_per_token_f16:
        return {}
    flags = _flags(extra_args)
    experts = model.expert_bytes_by_layer or {}
    experts_ram = sum(b for layer, b in experts.items() if flags["cpu_moe"] or layer < flags["n_cpu_moe"])
    lazy_ssd = model.lazy_bytes if flags["lazy"] else 0
    ram = experts_ram + model.input_bytes + (model.lazy_bytes - lazy_ssd)
    gpu_weights = model.size_bytes - ram - lazy_ssd + (model.input_bytes if model.tied_embeddings else 0)
    if str(gpu_layers).isdigit() and model.n_layers and int(gpu_layers) < model.n_layers:
        moved = int(gpu_weights * (model.n_layers - int(gpu_layers)) / model.n_layers)
        gpu_weights -= moved
        ram += moved
    scale = KV_TYPE_BYTES.get(kv_type, 2.0) / 2
    swa_part = model.kv_swa_bytes_per_token_f16 * scale
    per_token = model.kv_bytes_per_token_f16 * scale - (0 if swa_full else swa_part)  # grows with the context
    swa_fixed = 0 if swa_full else swa_part * _swa_cells(model.sliding_window or 0, ctx_per_slot, flags["ubatch"])
    kv = (per_token * ctx_per_slot + swa_fixed) * slots
    # llama.cpp reserves the compute buffer for a full slot: the attention mask (context x
    # micro-batch, f16) and, for a quantized cache, one layer's K and V converted to f16 for the
    # flash-attention kernels. At 512k tokens this is gigabytes.
    compute = ctx_per_slot * flags["ubatch"] * 2 + 64 * 2**20
    if kv_type not in ("f16", "bf16"):
        compute += model.kv_layer_max_f16 * ctx_per_slot
    recurrent = model.recurrent_bytes_per_slot * slots
    if mmproj_bytes and "--no-mmproj-offload" in shlex.split(extra_args or ""):
        ram += mmproj_bytes
        mmproj_bytes = 0
    # a draft model gets its own context of the same size (llama.cpp has no smaller draft context)
    # and a KV cache of the preset's type (Atlas passes -ctkd / -ctvd)
    d = estimate(draft, ctx_per_slot, slots, kv_type, "", gpu_layers, 0, swa_full) if draft else {}
    draft_bytes = sum(d.get(k, 0) for k in ("weights", "kv_cache", "recurrent", "compute"))
    ram += d.get("ram", 0)
    return {
        "weights": int(gpu_weights),
        "kv_cache": int(kv),
        "recurrent": int(recurrent),
        "projector": int(mmproj_bytes),
        "compute": int(compute),
        "draft": int(draft_bytes),
        "draft_kv": int(d.get("kv_cache", 0)),
        "total": int(gpu_weights + kv + recurrent + mmproj_bytes + compute + draft_bytes),  # GPU
        "ram": int(ram),
        "ssd": int(lazy_ssd),
        "kv_bytes_per_token": int(per_token),
        "ubatch": flags["ubatch"],
        # experts kept in RAM are copied to the GPU once per micro-batch during prefill
        "streamed_experts": int(experts_ram),
        "slot_file_per_100k_tokens": int(per_token * 100_000 + model.recurrent_bytes_per_slot
                                         + (swa_part * (model.sliding_window or 0) if not swa_full else 0)),
    }


def system_memory() -> int:
    """Total RAM visible to this system (inside WSL: the WSL memory limit)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


async def gpu_info() -> list[dict]:
    """GPUs as reported by nvidia-smi (empty if unavailable)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        proc = await asyncio.create_subprocess_exec(
            exe, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except (OSError, TimeoutError):
        return []
    gpus = []
    for line in out.decode().strip().splitlines():
        try:
            name, total, used = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "memory_total": int(total) * 2**20, "memory_used": int(used) * 2**20})
        except ValueError:
            continue
    return gpus

"""Write minimal GGUF files (header + metadata only) for tests."""

import struct
from pathlib import Path


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _value(v) -> tuple[int, bytes]:
    if isinstance(v, bool):
        return 7, struct.pack("<?", v)
    if isinstance(v, int):
        return (5, struct.pack("<i", v)) if v < 0 else (4, struct.pack("<I", v))
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        return 8, _string(v)
    if isinstance(v, list):
        etype = _value(v[0])[0] if v else 4
        return 9, struct.pack("<IQ", etype, len(v)) + b"".join(_value(x)[1] for x in v)
    raise TypeError(type(v))


def write_gguf(path: Path, metadata: dict, padding: int = 0, tensors: dict[str, int] | None = None) -> Path:
    """tensors: name -> byte size (use multiples of 32 so no alignment padding is added)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = tensors or {}
    body = b"".join(_string(k) + struct.pack("<I", _value(v)[0]) + _value(v)[1] for k, v in metadata.items())
    infos, offset = b"", 0
    for name, size in tensors.items():
        infos += _string(name) + struct.pack("<IQIQ", 1, size, 0, offset)
        offset += (size + 31) // 32 * 32
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)) + body + infos
    if tensors:
        head += b"\0" * (-len(head) % 32)
    path.write_bytes(head + b"\0" * offset + b"\0" * padding)
    return path


# what convert_hf_to_gguf.py writes from a generation_config.json (float32, hence 0.949999…)
RECOMMENDED_SAMPLING = {"general.sampling.temp": 0.6, "general.sampling.top_k": 20,
                        "general.sampling.top_p": 0.949999988079071, "general.sampling.min_p": 0.0}


def qwen35_like(path: Path, name: str = "Test Qwen3.5", padding: int = 0, sampling: dict | None = None) -> Path:
    """Hybrid model: 32 layers, full attention every 4th, 4 KV heads of 256 (like Qwen3.5-9B)."""
    return write_gguf(path, {
        **(RECOMMENDED_SAMPLING if sampling is None else sampling),
        "general.architecture": "qwen35", "general.name": name, "general.size_label": "9B",
        "general.file_type": 15, "qwen35.block_count": 33, "qwen35.nextn_predict_layers": 1,
        "qwen35.context_length": 262144, "qwen35.embedding_length": 4096, "qwen35.attention.head_count": 16,
        "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256, "qwen35.full_attention_interval": 4,
        "qwen35.ssm.inner_size": 4096, "qwen35.ssm.state_size": 128,
        "tokenizer.ggml.tokens": ["a"] * 2000,  # long array: must be skipped, not parsed
    }, padding)

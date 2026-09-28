"""Minimal GGUF metadata reader (header only, no tensor data, no dependencies)."""

import struct
from pathlib import Path
from typing import Any, BinaryIO

_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9
# Arrays longer than this (tokenizer vocabularies, merges) are skipped, only their length is kept.
_MAX_ARRAY = 1024


class GGUFError(ValueError):
    pass


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    return f.read(n).decode("utf-8", errors="replace")


def _skip_string(f: BinaryIO) -> None:
    f.seek(_read(f, "<Q"), 1)


def _value(f: BinaryIO, vtype: int) -> Any:
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        etype, count = _read(f, "<I"), _read(f, "<Q")
        if count > _MAX_ARRAY:
            if etype == _STRING:
                for _ in range(count):
                    _skip_string(f)
            elif etype in _SCALARS:
                f.seek(struct.calcsize(_SCALARS[etype]) * count, 1)
            else:
                raise GGUFError("nested arrays are not supported")
            return {"skipped_array": count}
        return [_value(f, etype) for _ in range(count)]
    raise GGUFError(f"unknown GGUF value type {vtype}")


def read_metadata(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError(f"GGUF version {version} is not supported")
        _read(f, "<Q")  # tensor count
        n_kv = _read(f, "<Q")
        meta: dict[str, Any] = {"GGUF.version": version}
        for _ in range(n_kv):
            key = _string(f)
            meta[key] = _value(f, _read(f, "<I"))
        return meta


def read_tensor_sizes(path: str | Path) -> dict[str, int]:
    """Byte size of every tensor in one GGUF file, derived from the data offsets."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        if _read(f, "<I") < 2:
            raise GGUFError("GGUF version is not supported")
        n_tensors, n_kv = _read(f, "<Q"), _read(f, "<Q")
        alignment = 32
        for _ in range(n_kv):
            key = _string(f)
            value = _value(f, _read(f, "<I"))
            if key == "general.alignment" and isinstance(value, int):
                alignment = value
        infos = []
        for _ in range(n_tensors):
            name = _string(f)
            f.seek(8 * _read(f, "<I"), 1)  # dimensions
            _read(f, "<I")  # type
            infos.append((_read(f, "<Q"), name))
        data_start = (f.tell() + alignment - 1) // alignment * alignment
    end = Path(path).stat().st_size - data_start
    infos.sort()
    return {name: (infos[i + 1][0] if i + 1 < len(infos) else end) - off for i, (off, name) in enumerate(infos)}

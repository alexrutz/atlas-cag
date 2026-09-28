"""Sampling parameters are bound to presets and default to what the model file recommends.

Converters store a model's recommended sampling (from its generation_config.json) as
general.sampling.* keys in the GGUF, often only some of them. A preset takes each parameter from
the model file unless it sets its own. A core parameter that neither provides has to be entered
in the preset: Atlas does not silently fall back to llama.cpp's generic defaults.
"""

from pydantic import BaseModel, Field

# preset field -> GGUF key (None: no key exists)
PARAMS = {
    "temperature": "general.sampling.temp",
    "top_k": "general.sampling.top_k",
    "top_p": "general.sampling.top_p",
    "min_p": "general.sampling.min_p",
    "repeat_penalty": "general.sampling.penalty_repeat",
    "presence_penalty": None,
}
REQUIRED = ("temperature", "top_k", "top_p", "min_p")
# further keys a model file may set; passed to llama-server as they are (request field names)
EXTRA = {
    "general.sampling.penalty_last_n": "repeat_last_n",
    "general.sampling.xtc_probability": "xtc_probability",
    "general.sampling.xtc_threshold": "xtc_threshold",
    "general.sampling.mirostat": "mirostat",
    "general.sampling.mirostat_tau": "mirostat_tau",
    "general.sampling.mirostat_eta": "mirostat_eta",
}


class SamplingConfig(BaseModel):
    """A preset's own values; None = take it from the model file."""

    temperature: float | None = Field(default=None, ge=0, le=5)
    top_k: int | None = Field(default=None, ge=-1, le=10000)  # 0 or -1 = off
    top_p: float | None = Field(default=None, ge=0, le=1)  # 1 = off
    min_p: float | None = Field(default=None, ge=0, le=1)  # 0 = off
    repeat_penalty: float | None = Field(default=None, ge=0, le=5)  # 1 = off
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)  # 0 = off


def _clean(value):
    # GGUF stores float32: 0.949999988 -> 0.95
    return float(f"{value:.6g}") if isinstance(value, float) else value


def from_model(meta: dict) -> dict:
    """Recommended sampling of a GGUF: preset field names plus extra llama-server fields."""
    out = {field: _clean(meta[key]) for field, key in PARAMS.items() if key and key in meta}
    out.update({field: _clean(meta[key]) for key, field in EXTRA.items() if key in meta})
    return out


def resolve(own: dict | None, model: dict) -> tuple[dict, dict, list[str]]:
    """(effective values, where each came from, required parameters nobody set)."""
    own = {k: v for k, v in (own or {}).items() if v is not None}
    effective, source = {}, {}
    for field in [*PARAMS, *EXTRA.values()]:
        if field in own:
            effective[field], source[field] = own[field], "preset"
        elif field in model:
            effective[field], source[field] = model[field], "model"
    missing = [f for f in REQUIRED if f not in effective]
    return effective, source, missing


def describe_missing(missing: list[str]) -> str:
    names = ", ".join(missing)
    return (f"sampling: the model file does not recommend {names}. Enter "
            f"{'a value' if len(missing) == 1 else 'values'} in the preset "
            "(temperature, top-p/top-k/min-p as given on the model card; top-k 0 and min-p 0 turn them off)")

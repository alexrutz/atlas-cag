#!/usr/bin/env bash
# Start llama-server with the flags Atlas depends on.
#
#   scripts/run-llama-server.sh /models/model.gguf [extra llama-server args...]
#
# Environment (all optional):
#   LLAMA_SERVER   path to the llama-server binary      (default: llama-server on PATH)
#   SLOTS          parallel slots = concurrent requests  (default: 4)
#   CTX_PER_SLOT   context tokens per slot               (default: 32768)
#   KV_DIR         slot save directory = ATLAS_KV_DIR    (default: ./data/kv)
#   KV_TYPE        KV cache type, f16 | q8_0 | ...       (default: q8_0, halves VRAM and slot files)
#   HOST / PORT    listen address                        (default: 127.0.0.1:8080)
#   NGL            layers offloaded to the GPU           (default: 99)
#
# Why these flags:
#   --slot-save-path      enables /slots save/restore, the core of CAG
#   --no-kv-unified       every slot owns exactly CTX_PER_SLOT tokens; with a unified buffer the
#                         server may purge idle slots, including one Atlas just restored
#   --cache-ram 0         disables the RAM prompt cache that swaps slot contents behind our back
#   -fa on                required for a quantized V cache; also faster prefill
# Add --swa-full for sliding-window models (Gemma 2/3, gpt-oss, ...): otherwise llama-server keeps
# only the last window of the cache and cannot extend a restored prefix.
# Keep this server private: Atlas assumes it is the only client driving the slots.
set -euo pipefail

MODEL=${1:?usage: $0 MODEL.gguf [extra llama-server args...]}
shift

LLAMA_SERVER=${LLAMA_SERVER:-llama-server}
SLOTS=${SLOTS:-4}
CTX_PER_SLOT=${CTX_PER_SLOT:-32768}
KV_DIR=${KV_DIR:-./data/kv}
KV_TYPE=${KV_TYPE:-q8_0}

mkdir -p "$KV_DIR"

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  --parallel "$SLOTS" --ctx-size $((SLOTS * CTX_PER_SLOT)) \
  --no-kv-unified --cache-ram 0 \
  --flash-attn on --cache-type-k "$KV_TYPE" --cache-type-v "$KV_TYPE" \
  --n-gpu-layers "${NGL:-99}" \
  --slot-save-path "$KV_DIR" \
  "$@"

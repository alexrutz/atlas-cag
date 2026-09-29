"""Command-line wrapper so the supervisor can spawn the fake llama-server like the real binary.

A model file whose name contains "broken" makes it fail at startup, like a bad model would.
--fake-version, --fake-fail-start and --fake-minimal-help imitate other builds.
"""

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.fake_llama import FakeLlama  # noqa: E402


HELP = """-m,    --model FNAME                    model path
--host HOST                             ip address to listen
--port PORT                             port to listen
-np,   --parallel N                     number of server slots
-c,    --ctx-size N                     size of the prompt context
--slot-save-path PATH                   path to save slot kv cache
-ctk,  --cache-type-k TYPE              KV cache data type for K
-mm,   --mmproj FILE                    path to a multimodal projector file
-md,   --spec-draft-model, --model-draft FNAME   draft model for speculative decoding
-t,    --threads N                      number of threads
"""


def _option(name: str) -> str | None:
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else None


def main() -> None:
    if "--version" in sys.argv:
        if version := _option("--fake-version"):
            print(f"version: {version}-dev (build 1, commit abc1234)")
        else:
            print("version: 0.0.0-fake (build 4242, commit fake42)")
        return
    if "--help" in sys.argv:
        help_text = HELP
        if "--fake-minimal-help" in sys.argv:  # a build without -t/--threads
            help_text = "\n".join(line for line in HELP.splitlines() if "--threads" not in line)
        print(help_text)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-semantics", default="default")
    ap.add_argument("--fake-version")
    ap.add_argument("--fake-fail-start", action="store_true")
    ap.add_argument("--fake-minimal-help", action="store_true")
    ap.add_argument("--mmproj")
    ap.add_argument("--fake-overcommit", choices=["quiet", "verbose"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--slot-save-path", required=True)
    ap.add_argument("--cache-type-k", default="f16")
    ap.add_argument("--api-key-file")
    args, _unknown = ap.parse_known_args()

    print(f"load_model: loading model '{args.model}'", flush=True)
    if args.fake_overcommit:  # what llama.cpp prints when a preset needs more VRAM than is free
        if args.fake_overcommit == "verbose":  # -lv 4
            print("I common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device "
                  "memory by 689 MiB")
        print("W common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user "
              "to -2, abort", flush=True)
    if args.fake_fail_start:
        print("ggml_cuda_init: failed to initialize CUDA: unknown error", flush=True)
        sys.exit(1)
    if "broken" in Path(args.model).name:
        print("llama_model_load: error loading model: invalid magic", flush=True)
        sys.exit(1)
    fake = FakeLlama(Path(args.slot_save_path), n_slots=args.parallel, n_ctx=args.ctx_size // args.parallel)
    fake.model_path = args.model
    fake.kv_format = args.cache_type_k
    fake.semantics = args.fake_semantics
    fake.vision = bool(args.mmproj)
    if args.api_key_file:  # like llama-server: every endpoint but /health needs the key
        from fastapi.responses import JSONResponse
        key = Path(args.api_key_file).read_text().strip()

        @fake.app.middleware("http")
        async def require_key(request, call_next):
            if request.url.path != "/health" and request.headers.get("authorization") != f"Bearer {key}":
                return JSONResponse({"error": {"code": 401, "message": "Invalid API Key",
                                               "type": "authentication_error"}}, status_code=401)
            return await call_next(request)
    print(f"init: n_slots = {args.parallel}, n_ctx_slot = {args.ctx_size // args.parallel}", flush=True)
    uvicorn.run(fake.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

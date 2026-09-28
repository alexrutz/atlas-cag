"""Command-line wrapper so the supervisor can spawn the fake llama-server like the real binary.

A model file whose name contains "broken" makes it fail at startup, like a bad model would.
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
-t,    --threads N                      number of threads
"""


def main() -> None:
    if "--version" in sys.argv:
        print("version: 0.0.0-fake (build 4242, commit fake42)")
        return
    if "--help" in sys.argv:
        print(HELP)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake-semantics", default="default")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--ctx-size", type=int, default=4096)
    ap.add_argument("--slot-save-path", required=True)
    ap.add_argument("--cache-type-k", default="f16")
    args, _unknown = ap.parse_known_args()

    print(f"load_model: loading model '{args.model}'", flush=True)
    if "broken" in Path(args.model).name:
        print("llama_model_load: error loading model: invalid magic", flush=True)
        sys.exit(1)
    fake = FakeLlama(Path(args.slot_save_path), n_slots=args.parallel, n_ctx=args.ctx_size // args.parallel)
    fake.model_path = args.model
    fake.kv_format = args.cache_type_k
    fake.semantics = args.fake_semantics
    print(f"init: n_slots = {args.parallel}, n_ctx_slot = {args.ctx_size // args.parallel}", flush=True)
    uvicorn.run(fake.app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

# Atlas: cache-augmented generation on llama.cpp

Atlas answers questions over documents without retrieval chunks. Each document is prefilled into
llama.cpp **once**, and the resulting KV cache is persisted to disk as a llama-server slot file.
At query time the slot file is restored and only the question is evaluated on top of it, so the
model reads the complete document every time, at a fraction of the prefill cost.

```
 ingest    text ──► prefill (n_predict = 0) into a slot ──► POST /slots/{id}?action=save ──► atlas-<part>.bin

 query     one document    restore slot file ──► append question ──► decode answer ──► stream

           several docs    map:    every document is answered individually, in parallel across slots
                           reduce: the answers are concatenated and the original question is run
                                   against them to synthesize the final answer, with [n] citations
```

- **Collections** group documents. Tick whole collections, single documents, or both.
- **Presets** define how llama-server runs a model: GGUF file, slots, context per slot, KV cache
  type. Atlas starts, stops and switches llama-server itself. Every configuration keeps its own
  document caches, so switching back to a preset is instant.
- **Settings** in the UI cover model files (with Hugging Face downloads), presets, generation
  settings and cache storage.

Measured on an RTX 2000 Ada (16 GB) with Qwen3.5-2B:

| | cold prefill | Atlas (restore + question) |
|---|---|---|
| 58k-token document, 5 parts | ~10 s before the first token | 0.6 s to restore all parts; each call evaluates only ~70 new tokens |
| 240-token document | ~80 ms | 10–20 ms restore |

The gap grows with model size: restoring is bounded by disk and PCIe bandwidth, while prefill cost
grows with parameters × tokens.

## Quick start

Requirements: Python ≥ 3.11 with [uv](https://docs.astral.sh/uv/) and a llama.cpp build with
`llama-server` (CUDA build for NVIDIA GPUs).

```bash
uv sync
cp .env.example .env     # set ATLAS_LLAMA_SERVER_BIN to your llama-server binary
uv run atlas             # → http://127.0.0.1:8000  (ATLAS_HOST=0.0.0.0 to reach it from other machines)
```

Then open **Settings** (gear icon):

1. **Model files:** download a GGUF from Hugging Face, or check that your local ones are listed.
2. **Model:** create a preset (model, context per slot, slots, KV cache type) and activate it.
3. Back in the chat, upload documents into collections, tick them and ask.

For llama.cpp source builds the supervisor adds the binary's directory to `LD_LIBRARY_PATH`, so
`ATLAS_LLAMA_SERVER_BIN=~/llama.cpp/build/bin/llama-server` works as is.

### Docker

```bash
mkdir -p models data && docker compose up -d      # → http://localhost:8000
```

The image is based on `ghcr.io/ggml-org/llama.cpp:server-cuda`; Atlas runs its llama-server from
your presets. `./models` holds GGUF files (downloads land there too), `./data` holds the index and
KV caches, and your Hugging Face cache is mounted read-only so GGUFs downloaded there show up.
The container runs as UID 1000 (override with `ATLAS_UID`/`ATLAS_GID`).

### External llama-server

Leave `ATLAS_LLAMA_SERVER_BIN` empty and Atlas connects to `ATLAS_LLAMA_URL` instead; presets are
then unavailable. Start the server with `scripts/run-llama-server.sh model.gguf`, which sets the
flags below, and point `ATLAS_KV_DIR` at its `--slot-save-path`.

## Bringing in models

llama.cpp runs **GGUF** files only. Atlas lists GGUFs from three places:

- `ATLAS_MODELS_DIRS` (default `data/models`, searched recursively). The UI downloads into the
  first directory.
- The Hugging Face cache (`~/.cache/huggingface/hub`), e.g. after
  `hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf`.
- llama.cpp's own download cache (`~/.cache/llama.cpp`, filled by `llama-server -hf …`).

Models downloaded for sglang or vLLM are **safetensors** checkpoints and cannot be loaded by
llama.cpp. Either download a GGUF version (for Qwen3.5-9B e.g. `unsloth/Qwen3.5-9B-GGUF`,
`bartowski/Qwen_Qwen3.5-9B-GGUF` or `lmstudio-community/Qwen3.5-9B-GGUF`), or convert the checkpoint:

```bash
python ~/llama.cpp/convert_hf_to_gguf.py ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/<rev> \
       --outtype q8_0 --outfile data/models/qwen3.5-2b-q8_0.gguf
```

(needs torch, transformers and llama.cpp's `gguf-py` on `PYTHONPATH`; `--outtype` can be f16,
bf16 or q8_0, and `llama-quantize` makes smaller quantizations from there).

Atlas reads each file's GGUF header to show its architecture, trained context and KV-cache cost,
and to estimate a preset's GPU memory before you start it.

## Presets and long context

A preset fixes the llama-server command line: model, `--parallel` (slots), context per slot,
`--cache-type-k/v`, flash attention, GPU layers, `--swa-full` and optional extra flags. Two
numbers matter most:

- **Context per slot** is the largest document part. Bigger documents are split into parts and
  answered part by part.
- **Slots** is how many documents or parts are answered at the same time.

GPU memory ≈ weights + slots × context × KV bytes per token (+ recurrent state on hybrid models).
Qwen3.5 models are hybrid: only every 4th layer uses attention, which keeps 262,144-token contexts
affordable. For **Qwen3.5-9B** (8 attention layers × 4 KV heads × 256 dimensions):

| KV cache | per token | 1 slot × 262,144 | 2 slots × 262,144 | 3 slots × 131,072 |
|---|---|---|---|---|
| f16 | 32 KiB | 8.0 GiB | 16 GiB | 12 GiB |
| q8_0 | 17 KiB | 4.3 GiB | 8.5 GiB | 6.4 GiB |
| q4_0 | 9 KiB | 2.3 GiB | 4.5 GiB | 3.4 GiB |

With Q4_K_M weights (5.3 GiB) plus compute buffers on a 16 GB card, **1 × 262k at q8_0**,
**3 × 131k at q8_0** or **2 × 262k at q4_0** fit. The preset editor shows this estimate live
against your GPU memory.

The estimate reads the GGUF's tensor table and follows the preset's offload flags. Routed experts
go to system RAM with `-cmoe` (all layers) or `--n-cpu-moe N` (the first N layers). Per-layer and
n-gram embeddings stay on disk with `--lazy-mode on`. Token embeddings always sit in RAM. For
example, a 62 GiB MoE with 26.8 GiB of n-gram embeddings runs as ≈ 3 GiB of weights on the GPU,
≈ 32 GiB in RAM and 27 GiB read from the SSD on demand. Its KV cost includes the sparse-attention
indexer keys and the recurrent and convolution state. Slot files are about the same size as the KV cache of the document
(≈ 1.7 GB per 100k document tokens at q8_0 for Qwen3.5-9B), so put `ATLAS_KV_DIR` on fast local NVMe.

Changing the context size or slot count keeps existing caches valid, so a restart is enough.
Changing the model, the KV cache type, flash attention or `--swa-full` creates a new cache
configuration; its caches are built in the background, and the old ones are kept for when you switch back.

## llama-server builds

Each preset can use its own llama-server build: a newer release, or a custom build for a model
architecture the standard build does not know. Leave the preset's build empty to use the standard
build, which Atlas keeps up to date (below; before the first update it is
`ATLAS_LLAMA_SERVER_BIN`). **Settings → Model → llama-server builds** lists the builds Atlas
finds (`~/*/llama-server`, `~/*/build*/bin/llama-server`, `/opt`, `/usr/local/bin`, `PATH`) and
the ones you add. Commands with arguments work too (e.g. a wrapper script). Each build is checked:

- that it can run here at all: Windows downloads (`.exe`) and builds for another CPU are flagged
  with the reason;
- for missing shared libraries (e.g. `libcudart.so.12` when the CUDA runtime package is missing);
- for its version and supported flags, so extra arguments it does not know are pointed out;
- whether its libraries name the model's architecture (llama.cpp stores them as strings), so
  "b10454 does not seem to support qwen4exp" shows up before a failed start.

Official releases come as `llama-bNNNN-bin-ubuntu-cuda-12.8-x64.tar.gz` together with
`cudart-llama-bNNNN-bin-ubuntu-cuda-12.8-x64.tar.gz` (the CUDA runtime). Unpack both into the same
directory. The supervisor puts the binary's directory on `LD_LIBRARY_PATH`.

### Automatic updates

Atlas keeps the standard build up to date from the GitHub releases of
[ai-dock/llama.cpp-cuda](https://github.com/ai-dock/llama.cpp-cuda/releases), which packages every
llama.cpp release with CUDA 12.8 for x86-64 and ARM64. It checks every 6 hours (or on
**Check now**), downloads the package for this machine, verifies its SHA-256 and unpacks it into
`data/builds/<tag>/`. Modes (**Settings → Model → llama-server builds**, `ATLAS_BUILD_UPDATES`):

- `install` (default): the new build becomes the standard build and is used from the next
  llama-server start; the settings page offers the restart.
- `apply`: also restarts llama-server as soon as no request is running.
- `off`.

The packages leave out the CUDA runtime (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`,
`libnccl.so.2`). Atlas links them from other llama.cpp builds or Python CUDA wheels on this machine
(`~/*/`, `~/*/lib/python3*/site-packages/nvidia/*/lib`, `/usr/local/cuda*`) into
`data/builds/cuda-runtime/`, preferring one folder for all of them, and downloads NVIDIA's wheels
from PyPI only if nothing local fits.

A new build must not break a working setup:

- presets whose model architecture or extra arguments only the previous build supports are
  pinned to the previous build (shown under the update box);
- if llama-server fails to start with a new build that has not run before, Atlas goes back to the
  previous build, skips that release and starts again;
- **Go back to …** does the same by hand; skipped releases can be allowed again. Releases older
  than one you went back from are not installed automatically.

The two newest updates and any build a preset uses are kept; older ones are deleted.

Caches are shared between builds when they are compatible. Every start restores a canary cache
and checks that it still predicts the same next token as when it was built. A build that stores
the cache differently or computes differently gets its own caches; an equivalent build (e.g. a
newer release) reuses them.

Atlas keeps logs in `data/logs/`: `atlas.log` for Atlas itself and `llama-server.log` for every
llama-server start with its full command line. Both rotate.

## llama-server requirements

Presets (and `scripts/run-llama-server.sh`) always set:

| flag | why |
|---|---|
| `--slot-save-path DIR` | enables slot save/restore. Atlas verifies at startup that files land in `ATLAS_KV_DIR`. |
| `--no-kv-unified` | a unified KV buffer lets the server purge idle slots, including one Atlas has just restored. |
| `--cache-ram 0` | disables the RAM prompt cache, which swaps slot contents behind Atlas's back. |
| `--flash-attn on` | required for a quantized V cache; also faster prefill. |
| `--swa-full` *(sliding-window models)* | Gemma 2/3, gpt-oss and similar keep only the last window of the cache unless this is set, and cannot extend a restored prefix. The preset editor turns it on when the model has a sliding window. |

Atlas must be the only client of its llama-server: in managed mode it listens on `127.0.0.1` only.

## Collections

Every document lives in one collection or in *Unfiled*. In the library:

- A collection's checkbox selects all of its ready documents; a partially selected collection shows
  a dash.
- Drag documents onto a collection, or use a document's ⋯ menu, to move it.
- Drop files onto a collection, or use its upload button, to ingest straight into it.
- Deleting a collection moves its documents to Unfiled unless you choose to delete them as well.

The query API takes `document_ids`, `collection_ids` or both.

## Visual prefill

Every PDF and image can be prefilled in one of two ways, and switched at any time:

- **Text** (default): the extracted text is prefilled.
- **Visual**: the pages are prefilled as images through the model's vision projector, so the model
  reads tables, charts, forms, handwriting and scans the way they look. Images and PDFs without a
  text layer are always prefilled visually.

Choose the mode for uploads in the library (**PDFs & images:** extracted text / page images) and
switch single documents in their ⋯ menu. The default for new uploads is under **Settings →
Generation → Visual prefill** (`ATLAS_DEFAULT_PREFILL`), together with the page resolution
(`ATLAS_VISUAL_DPI`, 120). The eye button shows a visual document's pages.

Visual prefill needs a vision-capable model and its projector (`mmproj-*.gguf`): set it in the
preset (**Vision projector**). Hugging Face listings mark projectors, and a projector downloaded
next to its model is suggested automatically. For a safetensors checkpoint, convert one with
`convert_hf_to_gguf.py <snapshot> --mmproj`. Visual documents wait ("needs vision model") while no
projector is loaded.

How it works:

- PDF pages are rendered to PNG once (pypdfium2) and kept in `data/docs/<id>/pages-<dpi>/`; images
  are normalized to PNG. The cached prefix is the template, the document header and one `[Page n]`
  label plus image per page, sent to llama-server as a multimodal prompt.
- llama-server identifies an image by a hash of its file bytes. A query sends the same page files
  again with the question; after the slot restore they match the cached image tokens, so nothing
  is encoded again. Measured with Qwen3.5-2B on CPU: a 2,216-token page takes 35 s to prefill and
  restores in 12 ms; the answer takes 1.4 s, with only the 54 question tokens evaluated.
- How many tokens a page takes depends on the vision encoder and the image size (about 1,100 per
  A4 page at 120 dpi for Qwen3.5). Atlas measures instead of guessing: llama-server rejects a
  padded prompt before evaluating anything and reports its size, so runs of pages that fit one
  slot are found without wasted prefills. A long PDF becomes several parts ("pages 1–40", …)
  that are answered and combined like text parts.
- Each cache records what it was built from (`text` or `visual:<projector>:<dpi>`). Adding or
  changing a projector keeps the text caches; only visual documents are rebuilt when the
  projector, its image options (`--image-min-tokens`/`--image-max-tokens`) or the resolution
  change.
- User text in multimodal prompts (file names, questions) is defused so it cannot form control
  tokens, since llama-server parses special tokens in multimodal prompt strings.

Context shift must stay off (llama-server's default) for visual prefill.

## Conversations

Questions are grouped into conversations. The bar above the answers shows the open conversation:
click its title to search and switch conversations, rename or delete them, or start a new one.
Asking in a new conversation creates it, titled after the first question. Opening a conversation
shows its earlier turns again, with their per-document answers, and selects the documents its
last question used. The open conversation is restored when the page is reloaded.

Each document cache holds only its document, not the conversation, so a follow-up such as "and
why did it happen?" is first rewritten into a standalone question ("Why did the gearbox fail?")
from the last four turns. This is one short generation without any document (temperature 0,
no thinking). The rewrite then runs like any other question, single-document or map-reduce, and
is shown under the follow-up ("Asked the documents: …"). Switch it off under **Settings →
Generation → Conversations** (`ATLAS_CONDENSE_FOLLOWUPS=false`) to send follow-ups unchanged.

## How it works

**Prompt layout.** A cached prefix must be a strict token prefix of every later query prompt.
Atlas renders the model's own chat template once, with two sentinels in the user turn, and cuts
the result into `head | document | mid | question | tail`:

```
prefix (persisted per part) = head + <document name="…">text</document> + mid
suffix (appended per query) = question block + tail (end of turn + generation prompt)
```

Prompts are sent to `/completion` as token arrays, so the stored prefix tokens match exactly.
Template text is tokenized with special-token parsing. Document text, file names, questions and
model output are tokenized as plain text, so content cannot inject control tokens.

**Ingestion.**

1. Extract the text.
2. Split it at paragraph boundaries, with overlap, if it exceeds
   `ctx_per_slot − (max_question + ~100 + tail + max_answer)`.
3. Lease a slot and prefill each part with `n_predict: 0`.
4. `POST /slots/{id}?action=save`.
5. Check that `n_saved` equals the prefix length, then store the prefix tokens in SQLite.

Prefill progress streams to the UI.

**Query.** For each part: lease a slot, restore its file, verify `n_restored`, then send
`prefix + suffix`. llama-server matches the restored tokens and evaluates only the suffix. Atlas
reports `cache_n` for every call and flags any cache miss.

**Map-reduce.** Parts are answered concurrently up to the number of slots. By default every
document answers, quoting passages and stating which parts of the question it doesn't cover. The
synthesis step combines the answers and cites them as `[n]`. If the answers exceed one slot, they
are condensed hierarchically first.

**Thinking models.** When the template supports a thinking switch (Qwen3, …), the UI shows a
toggle. Reasoning streams separately, and llama-server's reasoning budget forces `</think>` once
`ATLAS_MAX_THINKING_TOKENS` is spent, so an answer always follows.

**Slot scheduling.** A priority lease pool: queries go ahead of ingestion, and ingestion uses at
most `slots − 1` slots. A client disconnect aborts the upstream llama-server generations, not just
Atlas's own tasks. While a preset switch is in progress the pool is paused and drained, and a query
that straddles a switch is refused rather than restoring caches into the wrong model.

## Consistency and self-healing

KV caches only restore into the configuration that produced them. Atlas tracks this with a
**fingerprint** built from:

- the model file, size, parameter count and vocabulary
- the chat template
- the system prompt
- the prefix-layout version
- in managed mode, the preset's KV cache type, flash attention and `--swa-full`
- a per-configuration **KV epoch**

Caches are stored per (document, fingerprint), so several configurations coexist.

| event | detection | reaction |
|---|---|---|
| preset switch or model change | new fingerprint | caches for the new configuration are built; the old ones are kept |
| system prompt changed in Settings | new fingerprint | same as above |
| cache format changed without Atlas knowing (external server flags, llama.cpp upgrade, another build) | a canary slot file per configuration is rejected on restore, or no longer predicts its reference token (checked on connect, on llama-server restart and after a rejected restore) | that configuration's epoch is bumped and its caches are rebuilt |
| llama-server unreachable or failing during a restore | connection error or HTTP 5xx | retried or reported; caches are only invalidated when llama-server explicitly rejects a file |
| llama-server restart between health polls | `/props` `media_marker` is random per process | re-validation as above |
| managed llama-server crashes | process exit | restarted with backoff (at most 3 times in 5 minutes) |
| missing or corrupt slot file | existence check / failed restore | that document's cache is rebuilt |
| model switched while a document was being ingested | fingerprint checked before every part | the job restarts under the new configuration |
| llama-server outage during ingestion | connection-level error | up to 5 retries with backoff |
| hard crash mid-ingestion | unreferenced `atlas-*.bin` files | swept at startup |
| slot context shrinks below a part's size | per-part size check | the document is rebuilt with smaller parts |

The **Storage** tab lists caches per configuration with their size. Delete configurations you no
longer use; they are rebuilt if you switch back to them. With `ATLAS_AUTO_BUILD_CACHES=false`
caches are only built on demand (a document's ⋯ menu → Build KV cache).

## Configuration

Bootstrap settings come from environment variables or `.env` (see `.env.example`). Generation
settings and prompts are edited in **Settings → Generation**. They are stored in the database and
override the environment.

| variable | default | notes |
|---|---|---|
| `ATLAS_LLAMA_SERVER_BIN` | *(empty)* | enables managed mode (presets); path or command of llama-server |
| `ATLAS_LLAMA_PORT` | 8081 | managed llama-server port (bound to 127.0.0.1) |
| `ATLAS_LLAMA_URL` | `http://127.0.0.1:8080` | external mode only |
| `ATLAS_KV_DIR` | `data/kv` | slot files; external mode: must equal `--slot-save-path` |
| `ATLAS_DATA_DIR` | `data` | SQLite index and extracted texts |
| `ATLAS_MODELS_DIRS` | `data/models` | comma-separated GGUF directories; downloads go into the first |
| `ATLAS_SCAN_MODEL_CACHES` | true | also list GGUFs in the Hugging Face and llama.cpp caches |
| `ATLAS_HOST` / `ATLAS_PORT` | 127.0.0.1 / 8000 | `0.0.0.0` to serve other machines |
| `ATLAS_API_KEYS` | *(empty)* | comma-separated bearer tokens; empty disables auth |
| `HF_TOKEN` | *(empty)* | for gated or private Hugging Face repositories |
| `ATLAS_MAX_QUESTION_TOKENS` / `_ANSWER_` / `_FINAL_` | 1024 / 1024 / 2048 | also in the UI; question and answer budgets are reserved in every slot |
| `ATLAS_ENABLE_THINKING`, `ATLAS_MAX_THINKING_TOKENS` | false, 2048 | also in the UI |
| `ATLAS_CONDENSE_FOLLOWUPS` | true | also in the UI; rewrite follow-ups into standalone questions |
| `ATLAS_DEFAULT_PREFILL` | text | also in the UI; `text` or `visual` for new PDFs and images |
| `ATLAS_VISUAL_DPI` | 120 | also in the UI; page resolution for visual prefill |
| `ATLAS_RELEVANCE_FILTER` | false | also in the UI; see the prompt findings below |
| `ATLAS_AUTO_BUILD_CACHES` | true | also in the UI |
| `ATLAS_BUILD_UPDATES` | install | `off` / `install` / `apply`; also in the UI |
| `ATLAS_BUILD_UPDATE_REPO` | `ai-dock/llama.cpp-cuda` | GitHub repository whose releases provide the standard build |
| `ATLAS_BUILD_UPDATE_ASSET` | *(empty)* | part of the asset name to pick, e.g. `cuda-12.8-amd64`; empty = CUDA package for this CPU |
| `ATLAS_BUILD_UPDATE_INTERVAL_H` | 6 | hours between checks |

## Prompt findings

These were measured on Qwen3.5-2B with repeated runs against the real llama-server. To repeat
such measurements with your own model, documents and questions, use `scripts/benchmark.py` (see
`scripts/benchmark.example.json`).

**Map-reduce configuration.** Five multi-document questions (four compound, one needle among
unrelated documents), 3 runs each, scored by whether the expected facts appear in the final
answer:

| configuration | facts found | time |
|---|---|---|
| every per-document answer is synthesized (**default**) | **24/27** | 217 s |
| answers self-rate coverage; "none" is dropped (`ATLAS_RELEVANCE_FILTER=true`) | 23/27 | 167 s |
| answers self-rate coverage; ratings shown to the synthesizer | 17/27 | 212 s |

- **Early-exit relevance checks are fragile.** A "reply NO_RELEVANT_INFORMATION if irrelevant"
  instruction dropped 20 of 33 relevant documents on compound questions: a document that answers
  only half of a question declares itself irrelevant. A trailing rating after a real answer
  dropped 3 of 33, which is why the optional filter uses it.
- **Wording matters a lot on small models.** Rephrasing the rating prompt moved its misses from
  3/33 to 14/33. Asking for a specific output language in the map prompt raised false
  "irrelevant" verdicts from about 1.5 to 6 of 18. The language instruction therefore lives only
  in the synthesis and single-document prompts, and German questions get German answers. The
  benchmarked texts are documented in `atlas/prompts.py`; re-measure before changing them.
- **Small models invent "conflicts"** between documents during synthesis. Use a stronger model
  for production. The prompts can be overridden via env vars.

## API

| method and path | purpose |
|---|---|
| `GET /api/status` | engine, server, slots, leases, totals, limits |
| `GET /api/collections` · `POST` · `PATCH /{id}` · `DELETE /{id}?delete_documents=` | collections |
| `GET /api/documents` | library with the cache status for the active configuration |
| `POST /api/documents` | multipart upload (`files`, optional `collection_id` and `mode` = `text`/`visual`), deduplicated by SHA-256 |
| `POST /api/documents/text` | `{name, text, collection_id?}` |
| `GET` / `PATCH /api/documents/{id}` | details (parts) / rename, move or switch prefill (`{name?, collection_id?, mode?}`) |
| `GET /api/documents/{id}/text` · `POST …/reingest` · `DELETE …` | |
| `GET /api/documents/{id}/pages/{n}` | page image (PNG) of a PDF or image |
| `POST /api/query` | `{question, document_ids?, collection_ids?, thinking?, conversation_id?}` → Server-Sent Events |
| `GET /api/conversations` · `POST` · `GET /{id}` (with turns) · `PATCH /{id}` · `DELETE /{id}` | conversations |
| `GET /api/presets` · `POST` · `PUT /{id}` · `DELETE /{id}` · `POST /{id}/activate` | presets (managed mode) |
| `GET /api/server` · `POST /api/server/restart` · `POST /api/server/stop` | llama-server state, log, GPU |
| `GET /api/models` · `GET /api/models/hf?repo=` · `POST /api/models/download` · `GET /api/models/downloads` | model files |
| `GET /api/builds` · `POST /api/builds` · `DELETE /api/builds?command=` · `POST /api/presets/estimate` | llama-server builds; memory estimate and warnings for unsaved preset values |
| `GET /api/builds/updates` · `POST /api/builds/updates/check` · `…/rollback` · `…/unskip?tag=` | automatic build updates |
| `GET` / `PATCH /api/settings` | runtime settings (`null` resets a value) |
| `GET /api/caches` · `DELETE /api/caches/{fingerprint}` | caches per configuration |
| `GET /api/queries` | query log with stats |
| `GET /healthz` | unauthenticated liveness |

Query events: `plan` (includes the conversation), `rewrite` (stage `start`, then `done` with the
standalone question), `target` (status `queued`, `restoring`, `generating`, `done`, `irrelevant`
or `error`, with per-call stats), `target_delta`, `synthesis`, `delta` (channel `answer` or
`reasoning`), `done` (answer and stats), `error`, and `ping` as a keep-alive.

## Development

```bash
uv run pytest    # 88 tests, no GPU needed: a fake llama-server (tests/fake_llama.py) and its
                 # command-line wrapper let the supervisor spawn, switch and crash real processes
uv run python scripts/benchmark.py cases.json --runs 3    # answer quality of a running instance
```

```
atlas/
  api.py         FastAPI routes, SSE, llama-server monitor
  supervisor.py  managed mode: presets, llama-server process, drain / switch / crash restart
  engine.py      discovery, fingerprints and canaries, prompt assembly, prefill and generation
  ingest.py      ingestion queue per configuration: split, prefill, save; repairs, orphan sweep
  query.py       follow-up rewriting, single-document and map-reduce execution, synthesis
  models.py      GGUF discovery, metadata and tensor layout, memory estimates, GPU info
  builds.py      llama-server build discovery and checks (format, libraries, flags, architectures)
  updater.py     standard build updates from GitHub releases, CUDA runtime, pinning, rollback
  downloads.py   Hugging Face listing and resumable downloads
  gguf.py        dependency-free GGUF header reader
  prompts.py     chat-template layout via sentinels, prompt blocks, reasoning splitter
  slots.py       prioritized exclusive slot leases (pausable, resizable)
  store.py       SQLite: collections, documents, caches per configuration, parts, presets, settings,
                 conversations and their turns
  chunking.py    token-budgeted splitting at natural boundaries
  extract.py     PDF / DOCX / HTML / text extraction
  pages.py       page images for visual prefill (PDF rendering, image normalization)
  static/        single-page UI (no build step, no external assets)
```

## Limitations

- No per-document access control or multi-tenancy. The API keys are shared secrets; SSO and RBAC
  belong in a reverse proxy or a future version.
- Visual prefill needs a vision model with its projector; without one, scanned PDFs and images wait.
  DOCX, HTML and text files are always prefilled as text.
- Follow-ups see the conversation only through the rewritten question; the documents' caches never
  contain the conversation.
- One llama-server at a time. Scaling out means more slots or sharding documents across servers.
- Every query restores from disk; there is no slot affinity. This keeps results deterministic and
  works for recurrent and hybrid models, which cannot roll a slot back to a prefix.

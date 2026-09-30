# Atlas: cache-augmented generation on llama.cpp

Atlas answers questions over documents without retrieval chunks. Each document is prefilled into
llama.cpp **once**, and the resulting KV cache is persisted to disk as a llama-server slot file.
At query time the slot file is restored and only the question is evaluated on top of it, so the
model reads the complete document every time, at a fraction of the prefill cost.

```
 ingest    text ──► prefill (n_predict = 0) into a slot ──► POST /slots/{id}?action=save ──► atlas-<part>.bin

 query     one document    restore slot file ──► append conversation + question ──► decode answer ──► stream

           several docs    map:    every document is answered individually, in parallel across slots
                           reduce: the answers are concatenated and the original question is run
                                   against them to synthesize the final answer, with [n] citations
```

The UI has four modules, one tab each:

- **Chat**: tick documents, whole collections or chapters and chat with them. Earlier questions
  and answers go along with every question, as chat turns after each cached document. Click a
  citation or a quote to see the passage in the document, highlighted on its page.
- **Library**: manage documents and collections: chapters (nested collections) for documents
  split into many parts, upload, move and reorder, switch between text and visual prefill,
  rebuild caches, inspect parts and caches, bulk actions.
- **PDF tools**: cut large PDFs into shards (by chapters, token budget, page count or ranges) and
  estimate tokens, cache size and prefill time of any text or file.
- **Settings**: presets (how llama-server runs a model: GGUF file, slots, context, KV cache,
  sampling, build), model files with Hugging Face downloads, generation settings, cache storage.
  Atlas starts, stops and switches llama-server itself; every configuration keeps its own
  document caches, so switching back to a preset is instant.

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

Compute buffers grow with the context too: llama.cpp reserves, for a full slot, the attention mask
(context × micro-batch, f16) and, with a quantized KV cache, one layer's K and V converted to f16
for the flash-attention kernels. At 512k tokens that is 2.6 GB for Spark-X2.5-4B, which is why
1M tokens cannot fit 16 GB at any cache type. The estimate includes them (checked against
llama.cpp's own projection: 14,953 vs 14,931 MiB) and keeps 1 GiB free like llama.cpp does.

When a preset does not fit, llama.cpp reports it at startup ("need to reduce device memory by …
MiB") and Atlas shows it on the server card. On Windows (WSL) such a preset still starts: the
driver moves the excess into shared system memory, which makes llama-server slow and can make
the whole desktop stutter or freeze. Treat the warning as a real problem.

The estimate reads the GGUF's tensor table and follows the preset's offload flags. Routed experts
go to system RAM with `-cmoe` (all layers) or `--n-cpu-moe N` (the first N layers). Per-layer and
n-gram embeddings stay on disk with `--lazy-mode on`. Token embeddings always sit in RAM. For
example, a 62 GiB MoE with 26.8 GiB of n-gram embeddings runs as ≈ 3 GiB of weights on the GPU,
≈ 32 GiB in RAM and 27 GiB read from the SSD on demand. Its KV cost includes the sparse-attention
indexer keys and the recurrent and convolution state. Slot files are about the same size as the KV cache of the document
(≈ 1.7 GB per 100k document tokens at q8_0 for Qwen3.5-9B), so put `ATLAS_KV_DIR` on fast local NVMe.

Changing the context size or slot count keeps existing caches valid, so a restart is enough.
### Sampling

Sampling parameters belong to the preset. Converters store a model's recommended sampling (from
its `generation_config.json`) in the GGUF as `general.sampling.*`, often only part of it: a preset
takes each parameter from the model file unless you set your own value. Temperature, top-k, top-p
and min-p must be known: when the model file does not recommend one, the preset editor asks for it
(it marks the field "required: not in the model file") and does not save without it. Repeat and
presence penalty are optional and off unless set. Other recommendations in the file (repeat window,
XTC, Mirostat) are passed on as they are. The preset card shows the values in use, those set in the
preset in bold.

Sampling is sent with every request, so changing it takes effect immediately, without restarting
llama-server. Follow-up rewrites always use temperature 0. With an external llama-server (no
presets) its own defaults apply: the model file's recommendations, unless its command line sets
others. Presets created before sampling moved into presets keep working with the model file's
values; the card points out a missing one until you add it.

Changing the model, the KV cache type, flash attention or `--swa-full` creates a new cache
configuration; its caches are built in the background, and the old ones are kept for when you switch back.

### Experts on the CPU

For mixture-of-experts models larger than the GPU, the **-cmoe** switch next to *GPU layers* keeps
the expert weights in system RAM; the field then shows `-cmoe`. The command line gets `-cmoe` in
place of `--n-gpu-layers`, so llama.cpp places the other layers itself (its default, `auto`: as
many as fit on the GPU). The memory estimate moves the experts to the RAM row. Prefill copies the experts to
the GPU once per micro-batch, so a larger `-ub` (e.g. `-ub 2048 -b 2048` in the extra arguments)
makes prefill of long documents faster; the editor suggests it. `--n-cpu-moe N` (only the first N
layers' experts) still goes into the extra arguments.

### Draft models (speculative decoding)

A preset can name a **draft model** (optional, next to the optional vision projector): a small
model of the same family, or a DFlash / Eagle3 / MTP head made for the model (llama.cpp tells the
kind from the file). It proposes the next tokens and the model verifies several at once: the same
answers, generated faster when the guesses are good. Atlas passes `--model-draft` with the
preset's GPU layers and KV cache type (`--gpu-layers-draft`, `--cache-type-k/v-draft`; later
values in the extra arguments override them). The editor lists drafts with the model's vocabulary
first and warns when a draft cannot work (different tokenizer or vocabulary, a build without
`--model-draft`).

- **Memory:** llama.cpp gives the draft its own KV cache for the full context (slots × context per
  slot; there is no smaller draft context). A draft head with only sliding-window layers, such
  as a DFlash head, needs little; a full small model at long contexts can need gigabytes. The
  estimate includes it.
- **Caches:** the draft does not change the model's KV cache, so adding, changing or removing it
  keeps all document caches.
- **With restored documents:** llama-server saves and restores only the model's KV cache, not the
  draft's. After a document is loaded from its slot file, the draft has not read it, so its
  guesses about the document's wording are weaker than on a freshly read prompt (heads that only
  look at a recent window recover as the answer grows). N-gram self-speculation
  (`--spec-type ngram-mod` in the extra arguments, no draft model) drafts from the slot's whole
  token history, restored documents included, which suits answers that quote the document.

## llama-server builds

There are two kinds of builds (**Settings → Model → llama-server builds**):

- **Standard**: upstream llama.cpp with Atlas's fixes (restored documents of sliding-window models
  are reused; slot files load after changing the number of slots), built here from source for each
  llama.cpp release and kept up to date (below). Use it for every model upstream llama.cpp
  supports; a preset without a build of its own uses it. Before it is built the first time,
  `ATLAS_LLAMA_SERVER_BIN` stands in.
- **Custom builds**: llama-server binaries you add, with a name, for models that need another
  llama.cpp (a fork for a new architecture). They are not updated. The list shows which presets use
  each one; a build in use cannot be removed. A build typed into a preset ("Other command…") joins
  the list. Commands with arguments work too (e.g. a wrapper script).

Atlas does not search the disk for builds. Each build is checked:

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

**Built from source with Atlas's fixes.** Upstream llama.cpp re-prefills a restored document of a
sliding-window model (Gemma, gpt-oss, Spark) unless `--swa-full` is set, and cannot load slot files
saved with another number of slots. So the standard build is built here
(`ATLAS_BUILD_UPDATE_SOURCE=patched`, the default) instead of using the downloaded package; without
a compiler the settings page offers the prebuilt package without the fixes
(`ATLAS_BUILD_UPDATE_SOURCE=release`):

1. `git clone --depth 1` of the release tag from `ATLAS_BUILD_SOURCE_REPO` (ggml-org/llama.cpp);
2. the patches in `atlas/patches/llama.cpp/` (restored slots keep the sliding-window cache; slot
   files load with any slot count); a patch the release already contains is skipped, and if one
   no longer applies the build stops and the current build stays;
3. CMake with CUDA for this machine's GPU (`ATLAS_BUILD_CUDA_ARCH`, default from `nvidia-smi`),
   with the newest CUDA toolkit the driver supports, compiling only `llama-server` at low priority
   (`nice`) on all cores (`ATLAS_BUILD_JOBS`); a few minutes;
4. the binary and its libraries go into `data/builds/<tag>+atlas/`, the source and build tree are
   deleted, and the build becomes the standard build like a downloaded one (pinning, rollback and
   restart work the same). The build log stays in `data/builds/<tag>+atlas.build.log`.

It needs git, cmake, a C++ compiler and the CUDA toolkit (`nvcc`); the settings page says what is
missing. When the standard build is not the patched one yet, Atlas builds it a minute after
starting; **Build now** does it by hand.

Caches are shared between builds when they are compatible. Every start restores a canary cache
and checks that it still predicts the same next token as when it was built (the reference is taken
from the restored file too). Only the token is compared: its probability moves with how
llama-server evaluates the prompt (reusing the restored cache, re-evaluating from a checkpoint, or
evaluating everything again, as standard builds do for sliding-window models), e.g. 0.31 vs 0.70
for Gemma 4 with the same build and settings. A build that stores the cache differently or
computes differently gets its own caches; an equivalent build (e.g. a newer release) reuses them.

After another model configuration becomes active, Atlas does not prefill the library by itself:
documents without a cache for it show *not built*, and the Library offers **Build all** (or build
single documents). Settings → Generation → *Build the whole library when another model
configuration starts* (`ATLAS_BUILD_ON_MODEL_CHANGE`) restores the automatic rebuild. New documents
and repairs (a cache file that disappeared) are still built automatically
(`ATLAS_AUTO_BUILD_CACHES`).

Atlas keeps logs in `data/logs/`: `atlas.log` for Atlas itself and `llama-server.log` for every
llama-server start with its full command line. Both rotate.

### Address and API key

**Settings → Model → llama-server → Address** sets where the managed llama-server listens: this
computer only (127.0.0.1, default), all network interfaces (0.0.0.0) or a specific address, the
port, and optionally an API key. Saving restarts llama-server there; document caches stay valid.
Other programs can then use it as an OpenAI-compatible API (`http://host:port/v1`, with
`Authorization: Bearer <key>` if a key is set). Atlas checks that the port is free, sends the key
itself, and passes it to llama-server in a file (`data/llama-server.key`, readable only by you),
so it never appears in a command line or log.

Outside programs share the slots with Atlas: a request that lands in a slot Atlas is using (a
restored document, or one being prefilled) makes Atlas prefill that document again. Keep outside
use light, or run a second llama-server for other tools. Without an API key anyone who reaches the
port can use the model and its slot endpoints. Under WSL in NAT mode, Windows programs reach it at
`localhost`; other computers need a Windows port forward or WSL's mirrored networking.

## llama-server requirements

Presets (and `scripts/run-llama-server.sh`) always set:

| flag | why |
|---|---|
| `--slot-save-path DIR` | enables slot save/restore. Atlas verifies at startup that files land in `ATLAS_KV_DIR`. |
| `--no-kv-unified` | a unified KV buffer lets the server purge idle slots, including one Atlas has just restored. |
| `--cache-ram 0` | disables the RAM prompt cache, which swaps slot contents behind Atlas's back. |
| `--flash-attn on` | required for a quantized V cache; also faster prefill. |
| `--swa-full` *(sliding-window models)* | see below: needed with standard llama-server builds, not with builds that have the SWA restore fix. |

### Sliding-window models

Gemma 2/3, gpt-oss and Spark-X2.5 use a sliding window in most layers (Spark: 27 of 36 layers see
the last 512 tokens, 9 see everything). Without `--swa-full` those layers cache only the window, so
the KV cache grows with the context only in the full-attention layers: Spark needs 9.6 GiB for
512k tokens at q8_0 instead of 38 GiB.

Standard llama-server builds cannot use such a cache after a slot restore. The slot file keeps
exactly the window the next token needs, but the server's reuse check demands two positions more
and prefills the whole document again on every question (log: "forcing full prompt re-processing
due to lack of cache data"). With these builds, keep **Full SWA cache** on. A one-line fix of that
check (`pos_min_thold` in `tools/server/server-context.cpp`, patch in
`~/llama.cpp-v0.5.0-atlas/atlas-swa-restore.patch` on the development machine) makes restored
caches reusable; verified with Spark-X2.5-4B: 3,707 restored tokens reused, 22 evaluated, output
identical to a fresh run. Build it like any custom build and select it in the preset.

Atlas checks this whenever such a preset starts without `--swa-full`: it saves, restores and
extends a window-sized prompt, and warns on the preset if the build prefills it again.

### Changing the number of slots

With `--no-kv-unified` llama.cpp keeps one KV stream per slot and writes the stream count into
every slot file; standard builds refuse to load a file saved with a different number of slots
("n_stream mismatch"), although a document's cache does not depend on it. The context per slot
does not matter (as long as the parts still fit).

Atlas notices this when a preset starts: if its canary was saved with another slot count and is
rejected, the slot count becomes part of the cache configuration instead of the caches being
declared incompatible. Each slot count then has its own caches, and switching back to a slot count
used before is instant. A llama.cpp build with the stream fix (`state_read_sinfo` in
`src/llama-kv-cache.cpp` loads a single sequence regardless of the stream count; included in
`~/llama.cpp-v0.5.0-atlas/atlas-patches.patch` on the development machine) restores any slot file
into any number of slots, so changing slots needs no prefill at all.

Atlas must be the only client of its llama-server: in managed mode it listens on `127.0.0.1` only.

## Library and collections

Every document lives in one collection or in *Unfiled*. Collections nest: a collection can hold
chapters (collections inside it) next to its documents, so a long document split into many
documents can be structured like the original, e.g. the first eight shards form chapter 1.
Chapters and documents keep an order (by default the order they were added, which for shards is
the page order). The **Library** tab manages them:

- the collections panel shows the tree with documents and tokens per collection (a collection
  counts its chapters too); click one to show it with everything in it; its ⋯ menu uploads into
  it, adds a chapter inside, renames, moves it up, down or into another collection, asks all of it
  in chat or deletes it (its documents and chapters move up to its parent unless you delete them
  as well);
- the table lists documents in library order (the # column) with where they are, prefill mode,
  pages, tokens, parts, KV cache size, file size, state and age; sort by any column, filter by
  name, state and prefill mode;
- tick documents for bulk actions (shift-click ticks a range): ask in chat, **Group as chapter**
  (a new chapter made of them, placed where the first of them was, in the innermost collection
  that holds all of them), move, switch prefill, rebuild caches, delete;
- drag rows onto another row to put them before it (in library order), or onto a collection to
  move them there; drop files anywhere on the page to upload (into the collection and prefill
  mode chosen above the table);
- click a document for its details: parts for the running model, its caches in every model
  configuration, text or page preview, download of the original file, and a shortcut to the PDF
  tools.

In the **Chat** tab the left column shows the same tree and only selects: a collection's or
chapter's checkbox selects all ready documents in it, a partially selected one shows a dash, and
a fully selected chapter becomes one chip above the question. Documents are always answered and
cited in library order, so the citations of a chapter follow its pages. The query API takes
`document_ids`, `collection_ids` (with their chapters) or both.

## PDF tools

Large PDFs often work better as several documents: questions can target the chapters that matter
(fewer map calls, less noise), citations name the chapter, and covers, indexes or appendices can be
left out. Atlas does split big documents into parts by itself, but only by token count.

**PDF tools → Split a PDF into shards** loads a PDF (upload, or a library PDF via its details or
⋯ menu) and shows:

- every page with a thumbnail, its exact token count (the running model's tokenizer; estimated
  when no model runs) and whether it has a text layer; pages without one are scans or pictures
  and belong in visual prefill;
- the PDF's bookmarks (chapters) and how many tokens one part of the running model holds.

Cut strategies set the cuts, which you can then adjust by hand (the scissors between pages; click
a page to leave it out):

- **By token budget** (default): fill shards up to a token budget, by default what fits one part
  of the running model, cutting at the last chapter start instead when that keeps a shard at
  least half full;
- **By chapters**: one shard per bookmark of the chosen level;
- **Every N pages**;
- **Page ranges**, e.g. `1-12, 13-40, 41-`; pages outside the ranges are left out.

The shard list shows pages and tokens per shard and flags shards that would still need several
parts. Names default to the chapter title or page range and can be edited. **Add to the library**
creates one document per shard (in a chosen collection, text or visual prefill). With **Chapter
folders** (for PDFs with nested bookmarks) each shard is filed into chapters named after the
bookmarks it belongs to, e.g. `Manual › Part B › Chapter 3`, so a chapter can be asked as a whole;
when cutting by chapters of level L, only the levels above L make folders. **Download (.zip)**
saves the shard PDFs. Analyzed PDFs are kept for a day in `data/tools/`.

**Token estimator**: paste text or drop a file to get its exact token count, words and characters,
and for the running model the KV cache (= slot file) size, the prefill time at the speed measured
on this machine, and whether it fits one part or how many parts Atlas would make.

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

It is a real chat: every question carries the conversation's earlier questions and final answers
as chat turns, so follow-ups ("and why?") need no reformulation. Each document part is asked

```
[system] [user: <document> first question] [assistant: first answer] … [user: new question]
```

The document still opens the first user turn, so it stays the cached prefix and only the
conversation and the new question are evaluated. The template is rendered with one sentinel per
turn (template text is tokenized with special tokens, questions and answers as plain text).
Neither documents nor the model's thinking go into the history. The synthesis of several
documents sees the conversation as well. Oldest turns are left out only when they do not fit next
to a document part; the answer footer says how many earlier turns were sent. If a chat template
renders the first turn differently in a longer chat, the earlier turns go into the question as
text instead. Switch the history off under **Settings → Generation → Chat**
(`ATLAS_CHAT_HISTORY=false`).

## Sources

Answers quote the document: per-document answers are asked to quote the passages they answer
from (the benchmarked map prompt), and single-document answers to support the answer with short
verbatim quotes. When an answer is done, Atlas locates every quote in the document's text
(`atlas/evidence.py`):

- matching ignores what text extraction does to a PDF (line breaks, hyphenation, split words,
  spacing) and finds quotes the model shortened with "…"; a quote with a few words changed is
  found approximately and marked as such;
- the passage is mapped to its page, and on PDFs to boxes on the rendered page (via pdfium's text
  layer), preferring the document part the answer came from;
- a quote that is not in the document is flagged ("not found in the document"): the model
  paraphrased it, or the statement is not backed by the source.

Quotes in an answer are underlined with their page next to them, and a **Sources** row under
each answer lists them. Clicking a quote, a page chip or a citation `[n]` in a synthesized answer
opens the source panel: the page with the passage highlighted (browse to the neighboring pages),
the text around it, and a button that opens the full text scrolled to the passage. A citation
opens the quote of that finding that best matches the sentence it stands in. Turns recorded
before quotes were located get a **Find the quoted passages** link. Visual documents without a
text layer have nothing to match quotes against.

### Generation details

Under every answer, **Generation details** opens the numbers behind it: total time and time to the
first token and to the first answer token; tokens restored from slot files, bytes read and restore
time; prompt tokens evaluated and prompt speed; tokens generated (thinking and answer) and speed;
speculative decoding acceptance when a draft model runs; time spent waiting for a free slot;
sampling; model, slot size, build and cache configuration. A table lists every llama-server call
(each document part and the synthesis) with its slot, wait, restore, cached and evaluated tokens,
speeds, first token, context used and why it stopped. The numbers are stored with the turn, so
older conversations show them too (as far as they were recorded).

### Thinking formats

Atlas separates the model's reasoning from the answer for `<think> … </think>` (Qwen, DeepSeek and
most others) and for Gemma 4's thought channel (`<|channel>thought … <channel|>`). Gemma's markers
are control tokens that llama-server leaves out of generated text unless asked, so every request
lists them as `preserved_tokens`. The reasoning shows under *Reasoning* and never goes into the
conversation history.

## Limits and thinking

Thinking is on by default for models with a thinking switch (per question in the composer). No
token limits apply by default: answers, the combined answer and thinking run until the model stops
or the slot is full (the Stop button ends a generation), and questions may be as long as the slot
allows. Limits can be set under **Settings → Generation**.

One size still has to be chosen: when a document is split into parts, each part leaves room for
the conversation, the question, thinking and the answer. By default that is 1/8 of the slot,
between 4k and 64k tokens (`ATLAS_RESERVE_TOKENS` or the setting to change it).

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
| `ATLAS_LLAMA_HOST` / `ATLAS_LLAMA_PORT` | 127.0.0.1 / 8081 | where the managed llama-server listens; `0.0.0.0` lets other programs use it. Also in the UI (Settings → Model → llama-server → Address), which overrides these |
| `ATLAS_LLAMA_API_KEY` | *(empty)* | key llama-server requires (managed: passed via `--api-key-file`; external: the server's key). Also in the UI |
| `ATLAS_LLAMA_URL` | `http://127.0.0.1:8080` | external mode only |
| `ATLAS_KV_DIR` | `data/kv` | slot files; external mode: must equal `--slot-save-path` |
| `ATLAS_DATA_DIR` | `data` | SQLite index and extracted texts |
| `ATLAS_MODELS_DIRS` | `data/models` | comma-separated GGUF directories; downloads go into the first |
| `ATLAS_SCAN_MODEL_CACHES` | true | also list GGUFs in the Hugging Face and llama.cpp caches |
| `ATLAS_HOST` / `ATLAS_PORT` | 127.0.0.1 / 8000 | `0.0.0.0` to serve other machines |
| `ATLAS_API_KEYS` | *(empty)* | comma-separated bearer tokens; empty disables auth |
| `HF_TOKEN` | *(empty)* | for gated or private Hugging Face repositories |
| `ATLAS_MAX_QUESTION_TOKENS` / `_ANSWER_` / `_FINAL_` | *(no limit)* | also in the UI; 0 or empty = no limit |
| `ATLAS_ENABLE_THINKING`, `ATLAS_MAX_THINKING_TOKENS` | true, *(no limit)* | also in the UI |
| `ATLAS_RESERVE_TOKENS` | *(automatic)* | also in the UI; room kept per slot when splitting documents (1/8 of the slot, 4k–64k) |
| `ATLAS_CHAT_HISTORY` | true | also in the UI; send earlier turns with every question |
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
| `GET /api/collections` · `POST` · `PATCH /{id}` · `DELETE /{id}?delete_documents=` | collections; `POST {name, parent_id?, document_ids?}` creates a chapter made of documents, `PATCH {name?, parent_id?}` renames or moves |
| `POST /api/library/order` | `{parent_id, items: [{kind: "collection"\|"document", id}]}` puts items into a collection in this order |
| `GET /api/documents` | library with the cache status for the active configuration |
| `POST /api/documents` | multipart upload (`files`, optional `collection_id` and `mode` = `text`/`visual`), deduplicated by SHA-256 |
| `POST /api/documents/text` | `{name, text, collection_id?}` |
| `GET` / `PATCH /api/documents/{id}` | details (parts) / rename, move or switch prefill (`{name?, collection_id?, mode?}`) |
| `GET /api/documents/{id}/text` · `GET …/original` · `POST …/reingest` · `DELETE …` | |
| `GET /api/documents/{id}/pages/{n}` | page image (PNG) of a PDF or image |
| `POST /api/documents/{id}/evidence` · `GET …/passage?start=&end=` · `GET …/boxes/{page}?start=&end=` · `GET …/render/{page}?width=` | sources: locate an answer's quotes, a passage with its context, its boxes on a page, a rendered page |
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
| `POST /api/tools/pdf` (multipart `file` or form `doc_id`) · `GET …/{id}/thumb/{page}` · `POST …/{id}/shards` · `POST …/{id}/zip` · `DELETE …/{id}` | PDF analysis, thumbnails, shards into the library (each optionally with `folder`: chapter names) or as a ZIP |
| `POST /api/tools/estimate` (form `text` or multipart `file`) | token estimate with KV size, prefill time and parts |
| `GET /healthz` | unauthenticated liveness |

Query events: `plan` (includes the conversation, the number of earlier turns and per target the
collection path), `target` (status `queued`, `restoring`, `generating`, `done`, `irrelevant` or
`error`, with per-call stats; `done` carries `evidence`: each quote with `found`, `start`/`end` in
the document text, `page`, `score`), `target_delta`, `synthesis`, `delta` (channel `answer` or
`reasoning`), `done` (answer and stats), `error`, and `ping` as a keep-alive.

## Development

```bash
uv run pytest    # 101 tests, no GPU needed: a fake llama-server (tests/fake_llama.py) and its
                 # command-line wrapper let the supervisor spawn, switch and crash real processes
uv run python scripts/benchmark.py cases.json --runs 3    # answer quality of a running instance
```

```
atlas/
  api.py         FastAPI routes, SSE, llama-server monitor
  supervisor.py  managed mode: presets, llama-server process, drain / switch / crash restart
  engine.py      discovery, fingerprints and canaries, prompt assembly, prefill and generation
  ingest.py      ingestion queue per configuration: split, prefill, save; repairs, orphan sweep
  query.py       chat history per document part, single-document and map-reduce execution, synthesis
  models.py      GGUF discovery, metadata and tensor layout, memory estimates, GPU info
  builds.py      llama-server build discovery and checks (format, libraries, flags, architectures)
  updater.py     standard build updates from GitHub releases, CUDA runtime, pinning, rollback
  downloads.py   Hugging Face listing and resumable downloads
  gguf.py        dependency-free GGUF header reader
  prompts.py     chat-template layout via sentinels, prompt blocks, reasoning splitter
  sampling.py    preset sampling defaulting to the model file's general.sampling.* recommendations
  slots.py       prioritized exclusive slot leases (pausable, resizable)
  store.py       SQLite: collections (nested, ordered), documents, caches per configuration, parts,
                 presets, settings, conversations and their turns
  chunking.py    token-budgeted splitting at natural boundaries
  extract.py     PDF / DOCX / HTML / text extraction
  pages.py       page images for visual prefill (PDF rendering, image normalization)
  pdftools.py    PDF analysis (pages, text, bookmarks), thumbnails, shards
  evidence.py    locating an answer's quotes in the document: pages, boxes on the PDF page
  static/        single-page UI, no build step: app.js (core, router, chat), library.js,
                 tools.js (PDF tools, estimator), settings.js
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

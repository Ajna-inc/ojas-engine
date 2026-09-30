<p align="center">
  <img src="assets/ojas.png" alt="Ojas Engine" width="128">
</p>

<h1 align="center">Ojas Engine</h1>

<p align="center">
  A Rust inference engine for GGUF language models and ONNX vision models,<br>
  built on its own Metal, CPU, CUDA and Vulkan kernels.
</p>

It runs dense, mixture-of-experts and recurrent decoder graphs with speculative decoding and
disk-streamed experts, executes detector and OCR models through a shared vision IR, and includes
a define-by-run training tape for the detectors it serves.

## Build

The repository pins Rust 1.98.0 in `rust-toolchain.toml`; 1.98 is also the minimum supported
version.

```sh
cargo build --locked --release          # target/release/ojas
cargo build --locked --release --features cuda   # adds the CUDA decoders and vision backend
```

The Metal decoders are macOS-only and are gated at the manifest level, so the `ojas` binary
builds on Linux and Windows with the CPU path. An Apple Silicon Mac with Apple's developer tools
gets Metal automatically. `--features cuda` needs an NVIDIA driver at run time; kernels are
compiled through NVRTC, so no CUDA toolkit is required to build.

## Usage

```
ojas run    <model.gguf> [prompt]     generate once and exit
ojas chat   <model.gguf>              interactive conversation
ojas serve  <model.gguf>              OpenAI-compatible HTTP server
ojas bench  <model.gguf> [prompt]     measure tokens/s
ojas ocr    <model.gguf> <page|dir>   transcribe page images (needs --mmproj)
ojas tokenize <model.gguf> [text]     show the token split
ojas info   <model.gguf|.onnx>        architecture, tensors and metadata

ojas detect <model.onnx> <image|dir>  object detection
ojas plate  <det.onnx> <rec.onnx> <dict.txt> <image|dir>   detect and read plates
ojas vision-serve <model.onnx>        HTTP detector
ojas vbench <model.onnx>              vision forward-pass timing
```

`run`, `chat`, `serve`, `bench`, `tokenize` and `info` work on every platform: they select Metal
on Apple Silicon, CUDA on an NVIDIA build, and the CPU decoders otherwise. The swarm commands
(`worker`, `pipe-test`) are Metal-only and compile out where there is no Metal.

```sh
# one-shot generation
ojas run model.gguf "What is the capital of France?" -n 64

# chat, with a system prompt and a fixed seed
ojas chat model.gguf -sys "You are terse." --temp 0.7 --seed 1

# OpenAI-compatible server
ojas serve model.gguf --host 0.0.0.0 --port 8080

# page transcription through a vision projector
ojas ocr model.gguf pages/ --mmproj mmproj.gguf

# detection and plate reading
ojas detect yolo11n.onnx frames/ --conf 0.25 --json
ojas plate plate-v9t-384.onnx rec.onnx dict.txt frames/
```

Engine flags may appear anywhere on the command line and are removed before a subcommand reads
its positionals. They use llama.cpp's names where one exists.

### Loading

| flag | effect |
|---|---|
| `-c`, `--ctx-size N` | context positions to allocate |
| `-b`, `--batch-size N` / `-ub`, `--ubatch-size N` | prefill batch and micro-batch |
| `-cram`, `--cache-ram GB` | KV cache budget |
| `--mmap` / `--no-mmap` | memory-map weights, or read them |
| `--mlock` / `--no-mlock` | lock the resident skeleton |
| `--warmup` / `--no-warmup` | run a warm-up pass before timing |
| `-ngl`, `--n-gpu-layers N` | layers to place on the GPU |
| `--mmproj FILE` | vision projector, required by `ocr` |
| `-md`, `--model-draft FILE` | speculative draft model |
| `--device auto\|metal\|cuda\|cpu` | backend; `cuda` needs `--features cuda` |

### Precision and experts

| flag | effect |
|---|---|
| `--precision N` | decoder precision tier (default: 3 for dense models, 4 for models with experts, whose weights stream from disk) |
| `--mixed-q4 …` | mixed-precision presets applied at load |
| `--q4-head` / `--no-q4-head` | quantize the output head |
| `--q4-ffn-down` / `--no-q4-ffn-down` / `--q4-ffn-down-last N` | quantize FFN down projections |
| `--moe-dbuf` / `--no-moe-dbuf` | double-buffer expert streaming |
| `--top-k-experts N` | experts routed per token |
| `--resident` / `--stream` | hold experts resident, or stream them from disk |

### Generation

`-n`/`--predict`, `-p`/`--prompt`, `-f`/`--file`, `-sys`/`--system`, `--raw` (skip the chat
template), `--temp`, `--top-p`, `--top-k`, `--repeat-penalty`, `--repeat-last-n`, `-s`/`--seed`.

### Serving and vision

`--host`, `--port`, `--token` / `--token-file` (bearer auth), `--trust CIDR` (networks that skip
auth), `-r`/`--reps` (bench repetitions), `--conf`, `--iou`, `--json`, `--ocr-norm`,
`--vision-device`.

`ojas serve` exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` and
`/props`, plus bare `/completion` and `/completions`. `ojas vision-serve` exposes `/v1/detect`,
`/v1/models` and `/v1/health`.

### Models on disk

Split models are opened through their first shard (`...-00001-of-00003.gguf`); every numbered
sibling must be present. A draft model in an adjacent `MTP/` directory is discovered
automatically and checked for compatible metadata; `OJAS_MTP` selects one explicitly and
`OJAS_NO_SPEC=1` turns speculative generation off.

### Environment

Every knob is also an `OJAS_*` environment variable, defined with its default in
`crates/ojas-core/src/config.rs`. Boolean flags generally test for presence, so unset them to
disable rather than assigning `0`.

| variable | effect |
|---|---|
| `OJAS_CTX` | context positions |
| `OJAS_KV_GB`, `OJAS_EXPERT_CACHE_GB` | KV and expert-cache budgets |
| `OJAS_MTP`, `OJAS_NO_SPEC` | draft model; disable speculative decoding |
| `OJAS_MMPROJ` | vision projector |
| `OJAS_NO_MMAP`, `OJAS_PREWARM`, `OJAS_UBATCH`, `OJAS_TOPK` | loading and decode behaviour |
| `OJAS_PREFILL_CB_LAYERS` | layers per GPU command buffer during prompt processing (default 1; 0 keeps each chunk in one) |
| `OJAS_CACHE_DIR` | compiled-kernel cache |
| `OJAS_LOG` | log level, e.g. `OJAS_LOG=debug` |

## Language models

Dense, mixture-of-experts and recurrent decoder graphs, with mixed GGUF tensor types dispatched
per format and role. Experts can be streamed from disk rather than held resident, which is what
lets a model larger than VRAM run in a bounded resident set. Speculative decoding through a
multi-token-prediction draft head participates in greedy generation behind a runtime gate, with
state rollback on rejection and in-memory prefix reuse across requests.

Which quantization formats and operations are available varies by backend.

## Vision

`ojas-vision` imports ONNX graphs into a small IR, plans them, and executes on the CPU operator
set, CUDA, or Vulkan. The CPU executor is the numerical oracle: device backends are checked
against it per node. Detector families (YOLO, RT-DETR, D-FINE) and a two-stage plate reader
(detector, then CTC recognizer) run through one `Model` entry point. Model provenance, hashes
and licences are recorded in [models/MANIFEST.md](models/MANIFEST.md) — note that some listed
weights are AGPL-3.0 and need a permissive swap before production use.

## Training

`ojas-learn` is a define-by-run tape over a small primitive set. The CPU backend is the
reference — gradients there are checked against finite differences — and every device backend
(`cuda`, `metal`, behind features of those names) is checked against the CPU backend primitive
by primitive. It carries DETR-family training: the D-FINE decoder and hybrid encoder, the DEIM
criterion, augmentation policies and AdamW parameter groups, each with a parity example that checks
one training step against the reference implementation.

`training/` holds the PyTorch recipes and pipeline scripts used to produce detector
checkpoints, and `scripts/` the evaluation and export tooling. Both expect datasets and
checkpoints at local paths you supply; nothing here downloads them.

## Tests

```sh
# full suite, on a Mac with Metal development support
cargo test --locked --workspace

# portable logic and file-format tests, no model download
cargo test --locked -p ojas-core -p ojas-formats -p ojas-tokenize -p ojas-cpu
```

Model-dependent tests skip when `OJAS_TEST_GGUF` is unset; point it at a compatible small local
model to exercise those paths. Set `OJAS_REQUIRE_MODEL_TESTS=1` to make a missing fixture a
failure instead of a skip. Missing baselines fail; recording one requires an explicit
`OJAS_REGOLD=1`, which must stay out of verification jobs. The default tests do not cover every
GPU kernel.

[scripts/release/README.md](scripts/release/README.md) documents the independent comparison
tools, which record full-vocabulary logits and tokenizer agreement against a separately
installed llama.cpp.

## Layout

| Crate | Purpose |
|---|---|
| `ojas-core` | Shared traits, configuration, quantization code generation |
| `ojas-formats` | GGUF, safetensors and quantization readers |
| `ojas-tokenize` | Tokenization and built-in prompt formatting |
| `ojas-arch` | Architecture descriptions |
| `ojas-cpu` | CPU kernels and the numerical oracle |
| `ojas-metal`, `ojas-cuda`, `ojas-vulkan` | GPU backends |
| `ojas-models` | Decoder graphs, state, expert streaming |
| `ojas-infer` | Generation, sampling, speculative decoding |
| `ojas-vision` | ONNX import, vision IR and execution |
| `ojas-learn` | Training tape and DETR-family training |
| `ojas-cli` | The `ojas` binary |
| `ojas-train`, `ojas-swarm` | Distributed execution experiments |

## Licence

The engine is [MIT](LICENSE). Model files have their own terms and are not included here — see
[models/MANIFEST.md](models/MANIFEST.md). Material derived from other projects is recorded in
[third_party/](third_party/README.md), which retains the ggml and PyTorch notices.
[CONTRIBUTING.md](CONTRIBUTING.md) describes what evidence a kernel or model change needs.

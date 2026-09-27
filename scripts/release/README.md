# Independent model comparison

The reference helper is a test-only client of a separately installed llama.cpp. Ojas does not link it at runtime. On macOS with Homebrew llama.cpp installed:

```sh
clang++ -std=c++17 -O2 -I/opt/homebrew/include scripts/release/reference_probe.cpp \
  -L/opt/homebrew/lib -lllama -lggml -lggml-base -Wl,-rpath,/opt/homebrew/lib -o /tmp/reference_probe
cargo build --locked --release -p ojas-models --example release_probe
/tmp/reference_probe model.gguf prompt.txt /tmp/reference
target/release/examples/release_probe model.gguf /tmp/reference.ids /tmp/ojas.f32 4 decode prompt.txt
python3 scripts/release/compare_logits.py /tmp/reference.f32 /tmp/ojas.f32 --output result.json
```

Repeat Ojas with `prefill` in place of `decode`. The reference tokenizes the prompt with special-token parsing, evaluates it, and records eight continuation tokens. Ojas must tokenize the same text identically and replays the exact reference tokens, so a divergent prediction cannot change the later comparison inputs. Nine complete vocabulary logit rows are compared. The helper intentionally uses greedy argmax without sampling and does not stop early at EOS; this is a fixed-input numerical test, not a quality benchmark.

Default gates require every row's argmax to agree, RMSE <= 0.05 and cosine >= 0.999. Keep thresholds recorded and fixed before a run. A failure must remain a failure; do not relax limits to obtain a passing badge. No baseline is auto-recorded by this comparison.

Record model SHA-256, reference binary version, source revision/worktree digest, hardware, Rust version, prompt text/IDs, precision, relevant environment variables and commands with each result. The raw logits can be retained outside Git; result JSON records their hashes.

For strict model-dependent integration tests, set `OJAS_REQUIRE_MODEL_TESTS=1` and `OJAS_TEST_GGUF` to a compatible fixture. Missing configured models and missing baselines fail. Baseline generation requires explicit `OJAS_REGOLD=1`; never enable it in a verification job.

The reference helper accepts an optional `gpu` argument to offload model layers, attention and supported operations to Metal. Set `OJAS_REFERENCE_CPU_KQV=1` to reproduce the older helper's CPU-attention placement. Historical matrix artifacts retain their original helper identities and backend settings. CPU and GPU references can differ because CPU quantized matrix multiplication also quantizes activations. Record the reference backend; do not mix their logits into a single baseline. `run_local_matrix.py` uses the Metal reference and runs the two Ojas paths sequentially from an explicit local manifest. Its output includes failures, model hashes, prompt/continuation IDs and comparison artifacts.

The local quantization sweep uses files requantized from the same Qwen2.5 Q8 source solely as execution fixtures. These do not establish the quality of quantizing an original FP16 checkpoint.

For very large CPU references, `OJAS_REFERENCE_NO_REPACK=1` disables extra CPU buffer types so weights remain memory mapped instead of eagerly creating a repacked copy. Record this setting and the exact reference revision. A newer architecture may require building a separate reference checkout; the installed library version may not implement it.

With the `gpu` argument, `OJAS_REFERENCE_CPU_EXPERTS=1` keeps expert tensors on CPU while offloading the remaining layers. This is a distinct reference backend for diagnosing arithmetic differences on models too large for full GPU residency; it is not equivalent to a full Metal reference.

For Flash Next, see the dedicated validation and reproduction record. `OJAS_REFERENCE_OUTPUT=32` records 33 full-logit rows; pass `--rows 33` to the comparator. `OJAS_REFERENCE_CPU_PLE=1` keeps the large PLE lookup table on CPU; `OJAS_REFERENCE_COMPACT_GPU=1` requires the separate reference allocation patch. `OJAS_REFERENCE_NO_FLASH=1` disables the reference's flash-attention fusion. Record all these settings.

For repeated EngineCore speed measurements, build `serving_bench` and run `run_serving_bench.py --manifest cases.json --binary target/release/examples/serving_bench --work /tmp/serving-bench --output results.json`. The JSON manifest is a list with `label`, `model`, `prompt` (file path), `mode` (`plain` or `mtp`), optional `tokens` (default 32), `reps` (default 3), `env`, and `reference_ids`. The runner clears inherited Ojas settings, selects plain/MTP explicitly, rejects changed binaries or divergent outputs, and reports repeated samples rather than best-of timings. A warmup occurs before measured requests.

For Flash MTP bottleneck diagnosis, the cost experiment compares free correct drafts, correct drafts with maintenance, and forced actual drafts. Build `flash_mtp_cost` and use `run_flash_mtp_cost.py`; these are reference-checked diagnostics, not adaptive serving speed claims.

For CPU-copy versus GPU/submit-wait timing, build `flash_target_bench` and use `run_flash_target_profile.py --interleave --threads 1 2 4`. The target optimization record describes the measured copy improvement, timing-bucket overlap, reference checks and the opt-in `OJAS_EXPERT_COPY_THREADS` setting.

For Flash's opt-in partial-prefix acceptance, build `flash_prefix_bench` and use `run_flash_prefix_bench.py`. It interleaves old/new policies on one model, with correct-input snapshot-cost and actual-draft modes. See the prefix experiment for memory cost, correctness checks and the distinction between forced speculation and adaptive serving.

For Flash GPU verification, `run_flash_gpu_bench.py` rotates baseline and cooperative-Q8 requests on one loaded model and checks independent token IDs. `flash_verify_logits` captures full batched logits for the existing independent comparator. See GPU verification for the timing scope and measured results.

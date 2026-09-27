# Contributing

Describe the concrete failure or behavior being changed. Keep unrelated performance experiments separate so correctness changes can be reviewed independently.

For portable code, run:

```sh
cargo test --locked -p ojas-core -p ojas-formats -p ojas-tokenize -p ojas-cpu
```

For decoder or Metal changes, also run `cargo test --locked --workspace` on a Mac with Metal development support and exercise the affected GPU path. Default tests do not provide complete GPU coverage. Do not report a skipped model test as a numerical pass.

A new model or kernel needs independent reference evidence: model and tokenizer hashes, reference revision, prompt tokens, precision mode, relevant environment variables, hardware, tolerances, and the actual comparison. Test prefill, decode, and cache reuse separately. Speculative decoding changes also need acceptance/rejection and state-rollback comparisons with speculation disabled.

For performance claims, establish numerical agreement first. Report cold and warm loading, time to first token, decode throughput, peak memory, context/output lengths, repetitions, and median/spread. Keep measured results separate from projections.

Do not regenerate snapshots simply to make a failure disappear. Review the difference and its cause before deliberately updating a baseline. Never add model weights, credentials, private prompts, or reference checkouts to a contribution.

Use the [independent comparison tools](scripts/release/README.md) to record full vocabulary logits and tokenizer agreement. Set `OJAS_REQUIRE_MODEL_TESTS=1` for required fixture jobs. Missing or empty fixture configuration and missing baselines must fail; `OJAS_REGOLD=1` is an explicit recording operation and must stay out of verification jobs.

The model contract exposes allocated context capacity. Frontends must bound normal and speculative positions, and a reuse hook must restore all recurrent state before reporting a reusable prefix. Test through `EngineCore`, because its progress chunks can change which prefix the decoder sees.

For MTP changes, run `mtp_release_gate` on a checkpoint with real draft weights and a checkpoint without them. Exercise each supported precision/draft-depth configuration, forced rejection at later draft positions, stopping, sampling bypass, context boundaries, and `serving_reuse_gate`. The gate must report actual MTP engagement. A short self-generated snapshot or a config declaring an MTP layer is insufficient.

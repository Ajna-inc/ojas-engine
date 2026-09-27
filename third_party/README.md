# Reference-derived material

The IQ codebooks in `crates/ojas-formats/src/iq_tables.rs` match the ten tables in llama.cpp's `ggml/src/ggml-common.h` at the locally inspected reference revision `035e22731a7fd70b9854b3a2d64ec68e9b1a45d3`. All entries were compared numerically, not inferred from similar names. The ggml authors' MIT notice is retained in [ggml-LICENSE](ggml-LICENSE).

To verify or reproduce the tables from a separately obtained checkout:

```sh
python3 scripts/gen_iq_tables.py /path/to/llama.cpp/ggml/src/ggml-common.h --check
python3 scripts/gen_iq_tables.py /path/to/llama.cpp/ggml/src/ggml-common.h
```

The checkout is not bundled or a runtime dependency. `scripts/release/reference_probe.cpp` is a test helper that links to a separately installed llama.cpp. Record that installation's version with comparison results.

The PLE causal-convolution semantics were checked against `src/models/qwen4exp.cpp` in the same local reference revision: kernel-major GGUF weights, dilation equal to n-gram size, zero-initialized history, and per-channel convolution followed by SiLU. This identifies the implementation used for comparison; it does not imply that every architecture matches an original model publisher's implementation.

## PyTorch CUDA kernels

The `cnn` kernel family in `crates/ojas-cuda/src/kernels/cnn.rs` ports the per-element arithmetic of PyTorch 2.8.0 ATen CUDA kernels (revision `ba56102387ef21a3b04b357e5b183d48f0afefc7`): `ActivationSiluKernel.cu`, `UnarySpecialOpsKernel.cu` (sigmoid), the elementwise add functor, `Shape.cu` (concat), `UpSampleNearest2d.cu` / `UpSample.cuh`, `DilatedMaxPool2d.cu`, `SoftMax.cu` (`cunn_SpatialSoftMaxForward`) and `DepthwiseConv2d.cu`. The PyTorch BSD-3-Clause notice is retained in [pytorch-LICENSE](pytorch-LICENSE). The CPU oracle in `crates/ojas-cpu/src/cpu_cnn.rs` follows the same SiLU, softmax and max-pool formulas.

Bit-exact agreement is checked on a CUDA machine against the installed PyTorch, not against bundled source:

```sh
python scripts/release/torch_kernel_golden.py --out target/torch_golden
cargo test -p ojas-cuda --test conformance -- --ignored cnn_matches_torch_bitwise
```

// Shared CUDA prelude for every split family: fp16 intrinsics + the two warp
// reductions (Metal simd_sum analogs).
pub const PRELUDE: &str = r#"
#include <cuda_fp16.h>

// Warp (32-lane) sum reduction — the CUDA analog of Metal simd_sum.
__device__ __forceinline__ float warp_sum(float s) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    return s;   // lane 0 holds the total
}

// Butterfly warp reduce: every lane ends with the full 32-lane sum (Metal simd_sum).
__device__ __forceinline__ float warp_all_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
"#;

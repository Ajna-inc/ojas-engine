// Shared by every ojas-vulkan shader (prepended by build.rs).
//
// Buffers are passed as 64-bit device addresses (buffer_reference), exactly
// like the CUDA kernels take pointers: the host pushes the addresses, then
// the u32 constants, in the kernel's argument order, as push constants.
// The workgroup size comes from specialization constants 0..2 (the CUDA
// "block"); gl_WorkGroupID / gl_LocalInvocationID are blockIdx / threadIdx.
//
// Precision: `T` is the storage type (float, or float16_t with -DF16).
// LD reads storage as float, ST rounds a float back to storage — the CUDA
// family's "compute in float, round once" rule.
#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#ifdef F16
#define T float16_t
layout(buffer_reference, std430, buffer_reference_align = 2) buffer Buf { float16_t v[]; };
#else
#define T float
layout(buffer_reference, std430, buffer_reference_align = 4) buffer Buf { float v[]; };
#endif

// Round a float to the nearest binary16 value (ties to even), returned as
// float — in integer arithmetic, because the built-in conversions cannot be
// trusted for it (measured on the 3060 driver 580, RADV and llvmpipe, Mesa
// 23.2): NVIDIA and llvmpipe fold float(float16_t(x)) and
// unpackHalf2x16(packHalf2x16(x)) back to x, and RADV's packHalf2x16
// rounds toward zero. The reference (torch, CUDA __float2half) rounds to
// nearest even at every store. Stores go through it, so the final
// conversion only ever sees values it represents exactly.
float rnd_f16(float x) {
    uint u = floatBitsToUint(x);
    uint s = u & 0x80000000u;
    uint a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return x;                        // inf, NaN
    if (a >= 0x477ff000u) return uintBitsToFloat(s | 0x7f800000u); // >= 65520: inf
    if (a >= 0x38800000u) {                                // normal binary16: keep 10 mantissa bits
        a += 0xfffu + ((a >> 13) & 1u);
        return uintBitsToFloat(s | (a & 0xffffe000u));
    }
    // subnormal binary16: multiples of 2^-24 (scaling by 2^24 is exact)
    float m = roundEven(uintBitsToFloat(a) * 16777216.0) * (1.0 / 16777216.0);
    return uintBitsToFloat(s | floatBitsToUint(m));
}

#ifdef F16
#define RND(v_) rnd_f16(v_)
#else
#define RND(v_) (v_)
#endif

// u32 tables (offsets, counters) and u8 frames (RGB8 / NV12 bytes)
layout(buffer_reference, std430, buffer_reference_align = 4) buffer UBuf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8Buf { uint8_t v[]; };
// 16-byte vectors (8 halves) for wide loads; the address must be 16-byte aligned
layout(buffer_reference, std430, buffer_reference_align = 16) buffer U4Buf { uvec4 v[]; };

// Read-only views (`readonly restrict`): the compiler may then route loads
// through the non-coherent read cache (NVIDIA LDG.CONSTANT, AMD scalar/K$) —
// what data read many times (conv inputs: every pixel once per tap, weights,
// tables) needs.
#ifdef F16
layout(buffer_reference, std430, buffer_reference_align = 2) readonly restrict buffer RoBuf { float16_t v[]; };
#else
layout(buffer_reference, std430, buffer_reference_align = 4) readonly restrict buffer RoBuf { float v[]; };
#endif
layout(buffer_reference, std430, buffer_reference_align = 4) readonly restrict buffer RoUBuf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) readonly restrict buffer RoU4Buf { uvec4 v[]; };

#define LD(b, i) float((b).v[(i)])
#define ST(b, i, x) (b).v[(i)] = T(RND(x))

// linear invocation index of a 1-D launch (blockIdx.x * blockDim.x + threadIdx.x)
#define GID int(gl_GlobalInvocationID.x)

#define CNN_INF uintBitsToFloat(0x7f800000u)
#define CNN_FLT_MAX uintBitsToFloat(0x7f7fffffu)

// activation codes shared by cnn_bias_act and cnn_act:
//   0 none | 1 SiLU x/(1+exp(-x)) | 2 sigmoid 1/(1+exp(-x)) | 3 ReLU (NaN kept)
float cnn_act_f(float x, int act) {
    if (act == 1) return x / (1.0 + exp(-x));
    if (act == 2) return 1.0 / (1.0 + exp(-x));
    if (act == 3) return x < 0.0 ? 0.0 : x;
    if (act == 14) return 0.5 * x * (1.0 + tanh(0.79788456 * (x + 0.044715 * x * x * x)));
    return x;
}

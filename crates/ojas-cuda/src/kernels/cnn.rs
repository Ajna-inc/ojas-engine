//! CNN inference kernels, ported from PyTorch 2.8.0 ATen (BSD-3-Clause,
//! `ba56102387ef21a3b04b357e5b183d48f0afefc7`; notice in
//! `third_party/pytorch-LICENSE`). PyTorch is the bit-exact reference: every entry
//! reproduces the per-element arithmetic of the named upstream kernel, including the
//! fp16 "compute in float, round once" rule (`opmath_type<half>` /
//! `acc_type<half, true>` are `float`). The launch layout is local to this crate and
//! does not change the bits, since each output element (or softmax slice) is computed
//! independently in the upstream order.
//!
//! Each kernel is written once in `BODY` and expanded per precision by [`expand`]:
//! `@T@` is the storage type, `@S@` the entry-name suffix (`""` for f32, `"_f16"` for
//! half), and `LD(p, i)` / `ST(p, i, v)` load a float from / store a float into
//! storage (for half: `__half2float` / `__float2half`, round-to-nearest-even as the
//! upstream `static_cast<half>` does).
//!
//! Convention (shared with every family): buffers first, then `int` / `float`
//! constants (floats cross as `f32::to_bits()`). Launch: 1-D,
//! `grid = ceil(n / block)`, any `block`.

pub const BODY: &str = r#"
// ---- ATen DepthwiseConv2d.cu / Activation*Kernel.cu / UnarySpecialOpsKernel.cu:
// activation codes shared by cnn_bias_act and cnn_act.
//   0 none | 1 SiLU  x/(1+exp(-x))  | 2 sigmoid 1/(1+exp(-x)) | 3 ReLU (NaN kept)
//   4 tanh | 5 sqrt | 6 erf | 7 GELU 0.5x(1+erf(x/√2)) | 9 hardswish x·relu6(x+3)/6
//   10 neg | 11 exp | 12 log   (DETR heads: inverse sigmoid, the GELU MLPs)
__device__ __forceinline__ int cnn_imin(int a, int b) { return a < b ? a : b; }
// float → unsigned with the same order (TopK's radix select)
__device__ __forceinline__ unsigned cnn_order_key(float v) {
    unsigned b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// Conv epilogue code 13: max(v, floor[c]), the floors stored after the bias in
// the bias buffer. ReLU followed by a positive scale and a shift (HGNetv2's
// LearnableAffineBlock) with the scale folded into the weights: s·relu(z)+t
// = max(s·z + t, t).
#define CNN_ACT_FLOOR 13

__device__ __forceinline__ float cnn_act_f(float x, int act) {
    if (act == 1) return x / (1.0f + expf(-x));
    if (act == 2) return 1.0f / (1.0f + expf(-x));
    if (act == 3) return x < 0.0f ? 0.0f : x;
    if (act == 4) return tanhf(x);
    if (act == 5) return sqrtf(x);
    if (act == 6) return erff(x);
    if (act == 7) return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    if (act == 8) return fminf(fmaxf(x / 6.0f + 0.5f, 0.0f), 1.0f);
    if (act == 9) return x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    if (act == 10) return -x;
    if (act == 11) return expf(x);
    if (act == 12) return logf(x);
    if (act == 14) return 0.5f * x * (1.0f + tanhf(0.79788456f * (x + 0.044715f * x * x * x)));
    return x;
}

// Conv epilogue: `out.add_(bias.view(1,C,1,1))` then the activation, in one
// pass. The add rounds to storage first (upstream stores the sum as @T@
// before SiLU reads it back), then the activation runs in float.
// n = N*C*plane, in place on the conv output.
extern "C" __global__ void cnn_bias_act@S@(@T@* x, const @T@* bias, int n, int plane,
                                         int ch, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = (i / plane) % ch;
    ST(x, i, LD(x, i) + LD(bias, c));
    if (act != 0) ST(x, i, cnn_act_f(LD(x, i), act));
}

// NHWC depthwise k x k convolution (groups == channels, multiplier 1), any
// padding/stride per axis: taps outside the logical h x w image are skipped,
// so the op's pads may exceed the storage border. Float accumulation,
// kH-then-kW like DepthwiseConv2d.cu; bias/activation via cnn_bias_act_nhwc.
// Block = (image*out_h + oy, element chunk); thread = one (ox, c) element.
// grid = (N*out_h, ceil(out_w*ch/256)), block = 256.
extern "C" __global__ void cnn_dwconv_nhwc@S@(const @T@* x, const @T@* weight, @T@* y,
                                            int nimg, int ch, int h, int w, int ipad,
                                            int wp, int img, int out_h, int out_w, int k,
                                            int sh, int sw, int pt, int pl, int opad,
                                            int owp, int oimg, int ics, int ocs) {
    int n = blockIdx.x / out_h, oy = blockIdx.x % out_h;
    int e = blockIdx.y * 256 + threadIdx.x;
    __shared__ float sW[2304];
    int kk = k * k;
    int wc = kk * ch < 2304 ? kk * ch : 2304;
    for (int idx = threadIdx.x; idx < wc; idx += 256) sW[idx] = LD(weight, idx);
    __syncthreads();
    if (n >= nimg || e >= out_w * ch) return;
    int ox = e / ch, c = e % ch;
    long xb = (long)n * img + c;
    float v = 0.0f;
    for (int ky = 0; ky < k; ky++) {
        int iy = oy * sh - pt + ky;
        if (iy < 0 || iy >= h) continue;
        for (int kx = 0; kx < k; kx++) {
            int ix = ox * sw - pl + kx;
            if (ix < 0 || ix >= w) continue;
            float wv = (c + 1) * kk <= wc ? sW[c * kk + ky * k + kx] : LD(weight, c * kk + ky * k + kx);
            v += wv * LD(x, xb + ((long)(iy + ipad) * wp + ix + ipad) * ics);
        }
    }
    ST(y, (long)n * oimg + ((long)(oy + opad) * owp + ox + opad) * ocs + c, v);
}

// ---- padded-NHWC spatial kernels (executor storage: see conv::Storage) ----
// Image i of a storage (pad, wp, img) holds pixel (y, x) channel c at
// i*img + ((y + pad)*wp + x + pad)*C + c. One thread per output element.

// Max pool on padded NHWC; window bounds are the logical image (pads from the
// op, not the storage), torch rule: strictly greater wins, NaN always wins.
extern "C" __global__ void cnn_maxpool_nhwc@S@(const @T@* x, @T@* y, int n, int ch, int h,
                                             int w, int ipad, int iwp, int iimg, int oh,
                                             int ow, int opad, int owp, int oimg, int kh,
                                             int kw, int sh, int sw, int pt, int pl,
                                             int ics, int ocs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, t = i / ch, ox = t % ow;
    t /= ow;
    int oy = t % oh, img = t / oh;
    int hs = oy * sh - pt, ws = ox * sw - pl;
    float m = -CNN_INF;
    for (int yy = hs; yy < hs + kh; yy++) {
        if (yy < 0 || yy >= h) continue;
        for (int xx = ws; xx < ws + kw; xx++) {
            if (xx < 0 || xx >= w) continue;
            float v = LD(x, (long)img * iimg + ((long)(yy + ipad) * iwp + xx + ipad) * ics + c);
            if (v > m || v != v) m = v;
        }
    }
    ST(y, (long)img * oimg + ((long)(oy + opad) * owp + ox + opad) * ocs + c, m);
}

// Average pool on padded NHWC; window over the logical image (op pads),
// count_include_pad selects the divisor (k*k vs in-bounds taps). Float sum.
extern "C" __global__ void cnn_avgpool_nhwc@S@(const @T@* x, @T@* y, int n, int ch, int h,
                                             int w, int ipad, int iwp, int iimg, int oh,
                                             int ow, int opad, int owp, int oimg, int kh,
                                             int kw, int sh, int sw, int pt, int pl,
                                             int ics, int ocs, int count_include_pad) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, t = i / ch, ox = t % ow;
    t /= ow;
    int oy = t % oh, img = t / oh;
    int hs = oy * sh - pt, ws = ox * sw - pl;
    float sum = 0.0f;
    int cnt = 0;
    for (int yy = hs; yy < hs + kh; yy++) {
        if (yy < 0 || yy >= h) continue;
        for (int xx = ws; xx < ws + kw; xx++) {
            if (xx < 0 || xx >= w) continue;
            sum += LD(x, (long)img * iimg + ((long)(yy + ipad) * iwp + xx + ipad) * ics + c);
            cnt++;
        }
    }
    int div = count_include_pad ? kh * kw : (cnt > 0 ? cnt : 1);
    ST(y, (long)img * oimg + ((long)(oy + opad) * owp + ox + opad) * ocs + c, sum / div);
}

// Nearest upsample by integer factors on padded NHWC (src = dst / scale, the
// floor(dst * (1/scale)) of UpSampleNearest2d.cu for integer scales).
extern "C" __global__ void cnn_upsample_nhwc@S@(const @T@* x, @T@* y, int n, int ch, int h,
                                              int w, int ipad, int iwp, int iimg, int oh,
                                              int ow, int opad, int owp, int oimg, int sh,
                                              int sw, int ics, int ocs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, t = i / ch, ox = t % ow;
    t /= ow;
    int oy = t % oh, img = t / oh;
    y[(long)img * oimg + ((long)(oy + opad) * owp + ox + opad) * ocs + c] =
        x[(long)img * iimg + ((long)(oy / sh + ipad) * iwp + ox / sw + ipad) * ics + c];
}

// a + b on padded-NHWC channel slices (interior only; borders stay zero).
// n = N*h*w*ch; each operand has its own channel stride and image size.
extern "C" __global__ void cnn_add_nhwc@S@(const @T@* a, const @T@* b, @T@* y, int n, int ch,
                                         int h, int w, int pad, int wp, int acs, int aimg,
                                         int bcs, int bimg, int ycs, int yimg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, t = i / ch, xx = t % w;
    t /= w;
    int yy = t % h, img = t / h;
    long pix = (long)(yy + pad) * wp + xx + pad;
    ST(y, (long)img * yimg + pix * ycs + c,
       LD(a, (long)img * aimg + pix * acs + c) + LD(b, (long)img * bimg + pix * bcs + c));
}

// cnn_add_nhwc then an activation (ResNet's residual Add → ReLU), one pass:
// the sum rounds to storage first, as the two ops would.
extern "C" __global__ void cnn_add_act_nhwc@S@(const @T@* a, const @T@* b, @T@* y, int n, int ch,
                                             int h, int w, int pad, int wp, int acs, int aimg,
                                             int bcs, int bimg, int ycs, int yimg, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, t = i / ch, xx = t % w;
    t /= w;
    int yy = t % h, img = t / h;
    long pix = (long)(yy + pad) * wp + xx + pad;
    @T@ sum;
    ST(&sum, 0, LD(a, (long)img * aimg + pix * acs + c) + LD(b, (long)img * bimg + pix * bcs + c));
    ST(y, (long)img * yimg + pix * ycs + c, cnn_act_f(LD(&sum, 0), act));
}

// Elementwise activation on the interior of a padded NHWC image (the border
// stays zero, so the result feeds a conv without a layout round trip).
extern "C" __global__ void cnn_act_nhwc@S@(const @T@* x, @T@* y, int n, int ch, int h, int w,
                                         int pad, int wp, int xcs, int ximg, int ycs, int yimg,
                                         int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, r = i / ch, xx = r % w;
    r /= w;
    int yy = r % h, img = r / h;
    long pix = (long)(yy + pad) * wp + xx + pad;
    ST(y, (long)img * yimg + pix * ycs + c, cnn_act_f(LD(x, (long)img * ximg + pix * xcs + c), act));
}

// Padded NHWC -> contiguous NCHW (n = N*C*H*W, thread per NCHW element).
extern "C" __global__ void cnn_nhwc_to_nchw@S@(const @T@* x, @T@* y, int n, int ch, int h,
                                             int w, int pad, int wp, int img, int cs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xx = i % w, t = i / w, yy = t % h;
    t /= h;
    int c = t % ch, b = t / ch;
    y[i] = x[(long)b * img + ((long)(yy + pad) * wp + xx + pad) * cs + c];
}

// Contiguous NCHW -> padded NHWC interior (border untouched). `ch` logical
// channels, `cs` stored channels (>= ch; the extra ones stay zero).
extern "C" __global__ void cnn_nchw_to_nhwc@S@(const @T@* x, @T@* y, int n, int ch, int h,
                                             int w, int pad, int wp, int img, int cs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xx = i % w, t = i / w, yy = t % h;
    t /= h;
    int c = t % ch, b = t / ch;
    y[(long)b * img + ((long)(yy + pad) * wp + xx + pad) * cs + c] = x[i];
}

// Rank-4 numpy-broadcast binary op on contiguous tensors: out dims d0..d3,
// per-operand element strides (0 on broadcast axes). op: 0 add 1 sub 2 mul 3 div.
// Computed in float, rounded once (the CUDAFunctor_* opmath rule).
extern "C" __global__ void cnn_binary_bcast@S@(const @T@* a, const @T@* b, @T@* y, int n,
                                             int d1, int d2, int d3, int a0, int a1,
                                             int a2, int a3, int b0, int b1, int b2,
                                             int b3, int op) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int i3 = i % d3, t = i / d3, i2 = t % d2;
    t /= d2;
    int i1 = t % d1, i0 = t / d1;
    float va = LD(a, (long)i0 * a0 + (long)i1 * a1 + (long)i2 * a2 + (long)i3 * a3);
    float vb = LD(b, (long)i0 * b0 + (long)i1 * b1 + (long)i2 * b2 + (long)i3 * b3);
    // 0 add 1 sub 2 mul 3 div 4 pow 5 max 6 min
    float r = op == 0 ? va + vb : op == 1 ? va - vb : op == 2 ? va * vb : op == 3 ? va / vb
            : op == 4 ? powf(va, vb) : op == 5 ? fmaxf(va, vb) : fminf(va, vb);
    ST(y, i, r);
}

// Permute on contiguous tensors, rank <= 6 (leading dims padded with 1):
// out dims d0..d5; s0..s5 = input element strides of the axis that lands at
// each output position.
extern "C" __global__ void cnn_permute@S@(const @T@* x, @T@* y, int n, int d1, int d2,
                                        int d3, int d4, int d5, int s0, int s1, int s2,
                                        int s3, int s4, int s5) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = i;
    int i5 = t % d5; t /= d5;
    int i4 = t % d4; t /= d4;
    int i3 = t % d3; t /= d3;
    int i2 = t % d2; t /= d2;
    int i1 = t % d1, i0 = t / d1;
    y[i] = x[(long)i0 * s0 + (long)i1 * s1 + (long)i2 * s2 + (long)i3 * s3 + (long)i4 * s4 + (long)i5 * s5];
}

// Last-axis LayerNorm: mean, population variance, eps inside the sqrt; float
// accumulation. One thread per row (rows are short: the OCR neck's d = 120).
extern "C" __global__ void cnn_layernorm@S@(const @T@* x, const @T@* w, const @T@* b, @T@* y,
                                          int rows, int d, float eps, int has_bias) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    long o = (long)r * d;
    float mean = 0.0f;
    for (int i = 0; i < d; i++) mean += LD(x, o + i);
    mean /= d;
    float var = 0.0f;
    for (int i = 0; i < d; i++) { float t = LD(x, o + i) - mean; var += t * t; }
    var /= d;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < d; i++) {
        float v = (LD(x, o + i) - mean) * inv * LD(w, i);
        if (has_bias) v += LD(b, i);
        ST(y, o + i, v);
    }
}

// Batched matmul, contiguous: C[b] = A[b] (m x k) . B[b or 0] (k x n), float
// accumulate in k order. Small (attention blocks); thread per output element.
extern "C" __global__ void cnn_matmul@S@(const @T@* a, const @T@* bm, @T@* y, int total,
                                       int m, int k, int nn, int b_batched) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int col = i % nn, t = i / nn, row = t % m, bb = t / m;
    const @T@* ar = a + ((long)bb * m + row) * k;
    const @T@* bc = bm + (b_batched ? (long)bb * k * nn : 0) + col;
    float acc = 0.0f;
    for (int kk = 0; kk < k; kk++) acc += LD(ar, kk) * LD(bc, (long)kk * nn);
    ST(y, i, acc);
}

// NHWC conv epilogue over the unpadded interior of a padded image: same
// arithmetic as cnn_bias_act (sum rounded to storage, then the activation in
// float); the zero border is left untouched. One thread per element.
// n = N*oh*ow*ch; wp = padded row width in pixels; img = elements per image.
extern "C" __global__ void cnn_bias_act_nhwc@S@(@T@* x, const @T@* bias, int n, int oh,
                                              int ow, int pad, int wp, int ch, int img,
                                              int act, int has_bias, int cs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, t = i / ch, opix = oh * ow;
    int p = t % opix;
    long o = (long)(t / opix) * img + ((long)(p / ow + pad) * wp + p % ow + pad) * cs + c;
    if (has_bias) ST(x, o, LD(x, o) + LD(bias, c));
    if (act == CNN_ACT_FLOOR) ST(x, o, fmaxf(LD(x, o), LD(bias, ch + c)));
    else if (act != 0) ST(x, o, cnn_act_f(LD(x, o), act));
}

// Elementwise activation (ActivationSiluKernel.cu, UnarySpecialOpsKernel.cu sigmoid).
extern "C" __global__ void cnn_act@S@(const @T@* x, @T@* y, int n, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ST(y, i, cnn_act_f(LD(x, i), act));
}

// Same-shape a + b (CUDAFunctor_add; half add == float add then round, checked
// against IEEE binary16 on 4M random pairs).
extern "C" __global__ void cnn_add@S@(const @T@* a, const @T@* b, @T@* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ST(y, i, LD(a, i) + LD(b, i));
}

// Strided copy along one axis — Concat (Shape.cu CatArrayBatchedCopy) and Split
// both reduce to it. For o < outer, a < len, k < inner:
//   dst[((o*dst_axis) + dst_off + a)*inner + k] = src[((o*src_axis) + src_off + a)*inner + k]
// n = outer*len*inner. Raw storage copy: no conversion, so it is bit-exact.
extern "C" __global__ void cnn_axis_copy@S@(const @T@* src, @T@* dst, int n, int len,
                                          int inner, int src_axis, int src_off,
                                          int dst_axis, int dst_off) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int k = i % inner;
    int t = i / inner;
    int a = t % len;
    int o = t / len;
    dst[((long)o * dst_axis + dst_off + a) * inner + k] =
        src[((long)o * src_axis + src_off + a) * inner + k];
}

// UpSampleNearest2d.cu `upsample_nearest2d_out_frame` with the legacy
// `nearest_neighbor_compute_source_index`: src = min(floor(dst*scale), in-1),
// scale = (float)(1.0 / scale_factor) computed on the host in double.
// n = nc*h2*w2.
extern "C" __global__ void cnn_upsample_nearest@S@(const @T@* x, @T@* y, int n, int h1,
                                                 int w1, int h2, int w2,
                                                 float hs, float ws) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ow = i % w2;
    int t = i / w2;
    int oh = t % h2;
    int nc = t / h2;
    int ih = h1 == h2 ? oh : cnn_imin((int)floorf(oh * hs), h1 - 1);
    int iw = w1 == w2 ? ow : cnn_imin((int)floorf(ow * ws), w1 - 1);
    y[i] = x[((long)nc * h1 + ih) * w1 + iw];
}

// DilatedMaxPool2d.cu `max_pool_forward_nchw` (values only; the index mask is
// training-only). First strictly-greater element wins; NaN always wins.
// n = N*C*ph*pw.
extern "C" __global__ void cnn_maxpool@S@(const @T@* x, @T@* y, int n, int channels,
                                        int height, int width, int pooled_h,
                                        int pooled_w, int kernel_h, int kernel_w,
                                        int stride_h, int stride_w, int pad_h,
                                        int pad_w, int dilation_h, int dilation_w) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int c = (index / pooled_w / pooled_h) % channels;
    int nn = index / pooled_w / pooled_h / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = cnn_imin(hstart + (kernel_h - 1) * dilation_h + 1, height);
    int wend = cnn_imin(wstart + (kernel_w - 1) * dilation_w + 1, width);
    while (hstart < 0) hstart += dilation_h;
    while (wstart < 0) wstart += dilation_w;
    float maxval = -CNN_INF;
    long base = ((long)nn * channels + c) * height * width;
    for (int h = hstart; h < hend; h += dilation_h) {
        for (int w = wstart; w < wend; w += dilation_w) {
            float val = LD(x, base + h * width + w);
            if ((val > maxval) || val != val) maxval = val;
        }
    }
    ST(y, index, maxval);
}

// SoftMax.cu `cunn_SpatialSoftMaxForward`, `blockDim.x == 1` branch: a serial
// max (init -FLT_MAX, `a < b ? b : a`), a serial Σexp(x-max) in d order, then
// exp(x-max)/sum. Bit-exact to upstream whenever upstream takes that branch:
// inner > 1 and !(min(inner,1024) <= 64 && dim >= 64). YOLO DFL (dim 16) always.
// One thread per (outer, inner) slice: n = outer*inner.
extern "C" __global__ void cnn_softmax@S@(const @T@* x, @T@* y, int n, int dim, int inner) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n) return;
    long off = (long)(s / inner) * inner * dim + s % inner;
    float mx = -CNN_FLT_MAX;
    for (int d = 0; d < dim; d++) {
        float v = LD(x, off + (long)d * inner);
        mx = mx < v ? v : mx;
    }
    float sum = 0.0f;
    for (int d = 0; d < dim; d++) sum += expf(LD(x, off + (long)d * inner) - mx);
    for (int d = 0; d < dim; d++)
        ST(y, off + (long)d * inner, expf(LD(x, off + (long)d * inner) - mx) / sum);
}

// GridSampler.cu `grid_sampler_2d_kernel` — bilinear, zeros padding,
// align_corners = false (DETR deformable attention). One thread per output
// point (b, oy, ox), looping over channels as upstream does; grid math and the
// four-corner accumulation in float, corners outside the image skipped.
// x (N,C,H,W), grid (N,Ho,Wo,2) as (x, y) in [-1, 1], y (N,C,Ho,Wo); n = N*Ho*Wo.
extern "C" __global__ void cnn_grid_sample@S@(const @T@* x, const @T@* grid, @T@* y, int n,
                                            int C, int H, int W, int Ho, int Wo) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ox = i % Wo, t = i / Wo, oy = t % Ho, b = t / Ho;
    float ix = ((LD(grid, 2L * i) + 1.0f) * W - 1.0f) / 2.0f;
    float iy = ((LD(grid, 2L * i + 1) + 1.0f) * H - 1.0f) / 2.0f;
    float fx = floorf(ix), fy = floorf(iy);
    int x0 = (int)fx, y0 = (int)fy, x1 = x0 + 1, y1 = y0 + 1;
    float nw = (x1 - ix) * (y1 - iy), ne = (ix - x0) * (y1 - iy);
    float sw = (x1 - ix) * (iy - y0), se = (ix - x0) * (iy - y0);
    bool vx0 = x0 >= 0 && x0 < W, vx1 = x1 >= 0 && x1 < W, vy0 = y0 >= 0 && y0 < H, vy1 = y1 >= 0 && y1 < H;
    long plane = (long)H * W, oplane = (long)Ho * Wo;
    const @T@* xp = x + (long)b * C * plane;
    @T@* yp = y + (long)b * C * oplane + (long)oy * Wo + ox;
    for (int c = 0; c < C; c++, xp += plane, yp += oplane) {
        float acc = 0.0f;
        if (vx0 && vy0) acc += LD(xp, (long)y0 * W + x0) * nw;
        if (vx1 && vy0) acc += LD(xp, (long)y0 * W + x1) * ne;
        if (vx0 && vy1) acc += LD(xp, (long)y1 * W + x0) * sw;
        if (vx1 && vy1) acc += LD(xp, (long)y1 * W + x1) * se;
        ST(yp, 0, acc);
    }
}

// y = x·s + t (fused scalar Mul/Add: HGNetV2's learnable affine block), in
// float, rounded once. The NHWC form writes interior pixels only, so the zero
// border the next conv reads stays zero; indexing as cnn_add_nhwc.
extern "C" __global__ void cnn_scale_shift_nhwc@S@(const @T@* x, @T@* y, int n, int ch, int h,
                                                 int w, int pad, int wp, int xcs, int ximg,
                                                 int ycs, int yimg, float s, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = i % ch, r = i / ch, xx = r % w;
    r /= w;
    int yy = r % h, img = r / h;
    long pix = (long)(yy + pad) * wp + xx + pad;
    ST(y, (long)img * yimg + pix * ycs + c, LD(x, (long)img * ximg + pix * xcs + c) * s + t);
}

extern "C" __global__ void cnn_scale_shift@S@(const @T@* x, @T@* y, int n, float s, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ST(y, i, LD(x, i) * s + t);
}

// ONNX Pad, constant mode, spatial axes of NCHW (the HGNetV2 stem's
// F.pad): y (N,C,Ho,Wo) = value outside, x shifted by (top, left) inside.
// n = N*C*Ho*Wo.
extern "C" __global__ void cnn_pad4@S@(const @T@* x, @T@* y, int n, int H, int W, int Ho,
                                     int Wo, int top, int left, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ox = i % Wo, t = i / Wo, oy = t % Ho, p = t / Ho;
    int ix = ox - left, iy = oy - top;
    if (ix >= 0 && ix < W && iy >= 0 && iy < H) y[i] = x[((long)p * H + iy) * W + ix];
    else ST(y, i, value);
}

// ReduceSum / ReduceMax / ReduceMean over a contiguous trailing block of `len`
// elements, float accumulation in index order. kind: 0 sum, 1 max, 2 mean.
extern "C" __global__ void cnn_reduce_last@S@(const @T@* x, @T@* y, int rows, int len, int kind) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const @T@* p = x + (long)r * len;
    // kind: 0 sum, 1 max, 2 mean, 3 sqrt(sum of squares) — the L2 norm, accumulated in f32
    float acc = kind == 1 ? -CNN_INF : 0.0f;
    for (int i = 0; i < len; i++) {
        float v = LD(p, i);
        acc = kind == 1 ? (v > acc ? v : acc) : kind == 3 ? acc + v * v : acc + v;
    }
    if (kind == 2) acc /= len;
    if (kind == 3) acc = sqrtf(acc);
    ST(y, r, acc);
}

// TopK by radix select, one 256-thread block per row: stage the row in shared
// memory, find the exact k-th largest value by 32 passes over its order-preserving
// bit pattern (a block-wide count per bit), then give the elements at or above it
// rank = #(greater) + #(equal with a lower index) — ONNX TopK's order (largest
// first, ties to the lower index), identical on every backend. Rows longer than
// the staging buffer read global memory. Launched with n = rows * 256 (one block
// per row).
#define CNN_TOPK_SMEM 10240
#define CNN_TOPK_MAXK 512
#define CNN_TOPK_SELECT(SRC, LEN, K, KTH)                                              \
    {                                                                                  \
        __shared__ int cnt_;                                                           \
        unsigned prefix_ = 0;                                                          \
        for (int bit_ = 31; bit_ >= 0; bit_--) {                                       \
            unsigned cand_ = prefix_ | (1u << bit_);                                   \
            if (threadIdx.x == 0) cnt_ = 0;                                            \
            __syncthreads();                                                           \
            int c_ = 0;                                                                \
            for (int q_ = threadIdx.x; q_ < (LEN); q_ += blockDim.x)                   \
                c_ += cnn_order_key(LD(SRC, q_)) >= cand_;                             \
            for (int m_ = 16; m_; m_ >>= 1) c_ += __shfl_xor_sync(0xffffffffu, c_, m_); \
            if ((threadIdx.x & 31) == 0 && c_) atomicAdd(&cnt_, c_);                   \
            __syncthreads();                                                           \
            if (cnt_ >= (K)) prefix_ = cand_;                                          \
            __syncthreads();                                                           \
        }                                                                              \
        KTH = prefix_;                                                                 \
    }

// After the select (KTH = the k-th largest key): elements above it (fewer than k)
// are compacted into shared memory and ranked among themselves. Ties at KTH need
// no ranking; their rank is #(above) + #(tied before them), from a block-wide
// prefix count over contiguous chunks, so the lowest indices win (ONNX).
// EMIT(j, rank) stores.
#define CNN_TOPK_EMIT(SRC, LEN, K, KTH, EMIT)                                          \
    {                                                                                  \
        __shared__ int above_;                                                         \
        __shared__ int pre_[256];                                                      \
        __shared__ float cv_[CNN_TOPK_MAXK];                                           \
        __shared__ int ci_[CNN_TOPK_MAXK];                                             \
        if (threadIdx.x == 0) above_ = 0;                                              \
        __syncthreads();                                                               \
        for (int j_ = threadIdx.x; j_ < (LEN); j_ += blockDim.x) {                     \
            float v_ = LD(SRC, j_);                                                    \
            if (cnn_order_key(v_) <= (KTH)) continue;                                  \
            int at_ = atomicAdd(&above_, 1);                                           \
            cv_[at_] = v_;                                                             \
            ci_[at_] = j_;                                                             \
        }                                                                              \
        __syncthreads();                                                               \
        for (int e_ = threadIdx.x; e_ < above_; e_ += blockDim.x) {                    \
            float v_ = cv_[e_];                                                        \
            int j_ = ci_[e_], rank = 0;                                                \
            for (int q_ = 0; q_ < above_; q_++) {                                      \
                float u_ = cv_[q_];                                                    \
                rank += (u_ > v_) || (u_ == v_ && ci_[q_] < j_);                       \
            }                                                                          \
            int j = j_;                                                                \
            EMIT;                                                                      \
        }                                                                              \
        int chunk_ = ((LEN) + blockDim.x - 1) / blockDim.x;                            \
        int lo_ = threadIdx.x * chunk_, hi_ = min(lo_ + chunk_, (LEN));                \
        int t_ = 0;                                                                    \
        for (int q_ = lo_; q_ < hi_; q_++) t_ += cnn_order_key(LD(SRC, q_)) == (KTH);  \
        pre_[threadIdx.x] = t_;                                                        \
        __syncthreads();                                                               \
        if (threadIdx.x == 0) {                                                        \
            int acc_ = 0;                                                              \
            for (int i_ = 0; i_ < blockDim.x; i_++) { int c_ = pre_[i_]; pre_[i_] = acc_; acc_ += c_; } \
        }                                                                              \
        __syncthreads();                                                               \
        int pos_ = pre_[threadIdx.x];                                                  \
        for (int q_ = lo_; q_ < hi_ && above_ + pos_ < (K); q_++) {                    \
            if (cnn_order_key(LD(SRC, q_)) != (KTH)) continue;                         \
            int j = q_, rank = above_ + pos_++;                                        \
            EMIT;                                                                      \
        }                                                                              \
    }

extern "C" __global__ void cnn_topk_last@S@(const @T@* x, @T@* y, int len, int k) {
    __shared__ @T@ row[CNN_TOPK_SMEM];
    int r = blockIdx.x;
    const @T@* p = x + (long)r * len;
    bool staged = len <= CNN_TOPK_SMEM;
    if (staged) {
        for (int q = threadIdx.x; q < len; q += blockDim.x) row[q] = p[q];
        __syncthreads();
    }
    const @T@* src = staged ? row : p;
    unsigned kth;
    CNN_TOPK_SELECT(src, len, k, kth);
    CNN_TOPK_EMIT(src, len, k, kth, y[(long)r * k + rank] = p[j]);
}

// TopK → GatherElements fused (DETR query selection): row j of data (B,N,C)
// goes to position rank(j) of y (B,k,C) for the k largest scores (B,N), found
// as in cnn_topk_last. The indices never leave the block; rows copied raw.
// n = B * 256 (a block per batch row).
extern "C" __global__ void cnn_topk_gather@S@(const @T@* scores, const @T@* data, @T@* y, int N,
                                            int C, int k) {
    __shared__ @T@ row[CNN_TOPK_SMEM];
    int b = blockIdx.x;
    const @T@* s = scores + (long)b * N;
    bool staged = N <= CNN_TOPK_SMEM;
    if (staged) {
        for (int q = threadIdx.x; q < N; q += blockDim.x) row[q] = s[q];
        __syncthreads();
    }
    const @T@* src = staged ? row : s;
    unsigned kth;
    CNN_TOPK_SELECT(src, N, k, kth);
    CNN_TOPK_EMIT(src, N, k, kth, {
        const @T@* from = data + ((long)b * N + j) * C;
        @T@* dst = y + ((long)b * k + rank) * C;
        for (int c = 0; c < C; c++) dst[c] = from[c];
    });
}

// Last-axis LayerNorm, a warp per row: lanes stride the row (coalesced),
// shuffle reductions for the mean and then the population variance, eps
// inside the sqrt, float accumulation — the same maths as cnn_layernorm.
// n = rows * 32.
extern "C" __global__ void cnn_layernorm_warp@S@(const @T@* x, const @T@* w, const @T@* b, @T@* y,
                                               int n, int d, float eps, int has_bias) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int lane = t & 31;
    long o = (long)(t >> 5) * d;
    float sum = 0.0f;
    for (int i = lane; i < d; i += 32) sum += LD(x, o + i);
    for (int m = 16; m; m >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, m);
    float mean = sum / d, var = 0.0f;
    for (int i = lane; i < d; i += 32) { float u = LD(x, o + i) - mean; var += u * u; }
    for (int m = 16; m; m >>= 1) var += __shfl_xor_sync(0xffffffffu, var, m);
    float inv = 1.0f / sqrtf(var / d + eps);
    for (int i = lane; i < d; i += 32) {
        float v = (LD(x, o + i) - mean) * inv * LD(w, i);
        if (has_bias) v += LD(b, i);
        ST(y, o + i, v);
    }
}

// Softmax over contiguous rows (the last axis: attention), a warp per row:
// max, then the sum of exp, then the quotient, as cnn_softmax. n = rows * 32.
extern "C" __global__ void cnn_softmax_rows@S@(const @T@* x, @T@* y, int n, int d) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int lane = t & 31;
    long o = (long)(t >> 5) * d;
    float mx = -CNN_FLT_MAX;
    for (int i = lane; i < d; i += 32) mx = fmaxf(mx, LD(x, o + i));
    for (int m = 16; m; m >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
    float sum = 0.0f;
    for (int i = lane; i < d; i += 32) sum += expf(LD(x, o + i) - mx);
    for (int m = 16; m; m >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, m);
    for (int i = lane; i < d; i += 32) ST(y, o + i, expf(LD(x, o + i) - mx) / sum);
}

// DepthwiseConv2d.cu `conv_depthwise2d_forward_kernel`: float accumulator
// seeded with the bias, kH-then-kW order, out-of-bounds taps skipped.
// n = N*outC*oh*ow; `has_bias == 0` ignores `bias` (bind any buffer).
extern "C" __global__ void cnn_dwconv@S@(const @T@* input, const @T@* weight,
                                       const @T@* bias, @T@* output, int n,
                                       int out_ch, int mult, int in_w, int in_h,
                                       int out_w, int out_h, int kw, int kh,
                                       int stride_w, int stride_h, int pad_w,
                                       int pad_h, int dil_w, int dil_h,
                                       int has_bias) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= n) return;
    int t1 = li / out_w;
    int w = li - t1 * out_w;
    int t2 = t1 / out_h;
    int h = t1 - t2 * out_h;
    t1 = t2;
    t2 = t1 / out_ch;
    int c = t1 - t2 * out_ch;
    int nn = t2;
    int in_c = c, in_chs = out_ch;
    if (mult != 1) { in_c /= mult; in_chs /= mult; }
    int woff = c * kh * kw;
    float value = has_bias ? LD(bias, c) : 0.0f;
    long off0 = ((long)nn * in_chs + in_c) * in_h * in_w;
    for (int y = 0; y < kh; ++y) {
        for (int z = 0; z < kw; ++z) {
            int hi = -pad_h + h * stride_h + y * dil_h;
            int wi = -pad_w + w * stride_w + z * dil_w;
            if ((hi >= 0) && (hi < in_h) && (wi >= 0) && (wi < in_w)) {
                long off = off0 + hi * in_w + wi;
                value += (LD(weight, woff + y * kw + z) * LD(input, off));
            }
        }
    }
    ST(output, li, value);
}
"#;

/// fp16-only entries (the tensor-core `mma` instruction is f16 in / f32 out).
/// Emitted once, after the per-precision copies. Host-side planning lives in
/// `crate::conv` (weight packing, tile tables, launch geometry).
pub const BODY_F16: &str = r#"
// Tiled tensor-core implicit GEMM for stride/pad-any convolutions, groups 1,
// on (padded) NHWC: Y[m][n] = sum_k X[m][k] W[n][k], m = output pixels,
// n = output channels, k = tap * cin + ic (tap = ky * kw + kx).
// Block 128 x BN (BN 128 / 64 / 32: cnn_igemm_f16 / cnn_igemm64_f16 /
// cnn_igemm32_f16), k step 32, 2-stage cp.async pipeline (16-byte copies with
// zero-fill past M / N / K), 8 warps in a 4 (m) x 2 (n) grid, 32 x BN/2 per warp;
// A and B fragments by ldmatrix (row padding 40 halves: conflict-free).
// xoff: per-pixel patch origin inside one padded input image; toff[tap]: offset
// of a tap from the origin (ky*wp+kx)*cs; yoff: per-pixel output offset; row
// m = image * opix + pixel, images iimg/oimg apart. W is [npad x kpad] f16
// row-major, zero-padded. Needs cin, kpad and every offset a multiple of 8 (so a
// 16-byte copy never straddles a tap); half2 output stores need even offsets.
// Epilogue as cnn_bias_act_nhwc: the conv sum is rounded to f16, then bias
// (rounded), then act. grid = (ceil(M/128), ceil(N/BN)), block = 256.
#define G1_BM 128
#define G1_BK 32
#define G1_SROW 40
template <int BN, int ST>
__device__ __forceinline__ void cnn_igemm_body(const __half* x, const __half* w,
    __half* y, const int* xoff, const int* yoff, const int* toff, const __half* bias, int M, int N,
    int K, int kpad, int opix, int iimg, int oimg, int cin, int ntaps, int act, int has_bias) {
    constexpr int NI = BN / 16;  // n8 fragments per warp (warp covers 32 x BN/2)
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, R = lane >> 2, Q = lane & 3;
    int m0 = blockIdx.x * G1_BM, n0 = blockIdx.y * BN;
    int wm = warp % 4, wn = warp / 4;
    extern __shared__ __align__(16) __half smem_g[];  // ST * (G1_BM + BN) * G1_SROW halves
    __half* sAbase = smem_g;                     // ST stages of A
    __half* sBbase = smem_g + ST * G1_BM * G1_SROW; // ST stages of B
    __shared__ int sXo[G1_BM];
    __shared__ int sTo[64];
    if (tid < ntaps) sTo[tid] = toff[tid];
    for (int i = tid; i < G1_BM; i += 256) {
        int m = m0 + i;
        sXo[i] = m < M ? (m / opix) * iimg + xoff[m % opix] : -1;
    }
    __syncthreads();
    float acc[2][NI][4];
    #pragma unroll
    for (int a = 0; a < 2; a++)
        #pragma unroll
        for (int b = 0; b < NI; b++)
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;
    int nk = (K + G1_BK - 1) / G1_BK;
    // stage loader: 16-byte copies, 4 per 32-wide row (A rows then B rows)
    #define LOAD(stage, k0) do { \
        _Pragma("unroll") \
        for (int it = 0; it < (G1_BM * 4 + BN * 4 + 255) / 256; it++) { \
            int idx = tid + it * 256; \
            if ((G1_BM * 4 + BN * 4) % 256 == 0 || idx < G1_BM * 4 + BN * 4) { \
            int isB = idx >= G1_BM * 4, j = isB ? idx - G1_BM * 4 : idx; \
            int r = j >> 2, seg = j & 3; \
            int kk = (k0) + seg * 8; \
            const __half* src; int bytes; unsigned dst; \
            if (!isB) { \
                int xo = sXo[r]; \
                bytes = (xo >= 0 && kk < K) ? 16 : 0; \
                int tap = kk / cin; \
                src = x + (bytes ? (long)xo + sTo[tap] + (kk - tap * cin) : 0); \
                dst = (unsigned)__cvta_generic_to_shared(sAbase + (stage) * (G1_BM * G1_SROW) + r * G1_SROW + seg * 8); \
            } else { \
                bytes = (n0 + r < N) ? 16 : 0; \
                src = w + (bytes ? (long)(n0 + r) * kpad + kk : 0); \
                dst = (unsigned)__cvta_generic_to_shared(sBbase + (stage) * (BN * G1_SROW) + r * G1_SROW + seg * 8); \
            } \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(bytes)); \
            } \
        } \
        asm volatile("cp.async.commit_group;\n" ::); \
    } while (0)
    // prologue: ST-1 stages in flight (empty groups keep the count uniform)
    #pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nk) { LOAD(s, s * G1_BK); } else { asm volatile("cp.async.commit_group;\n" ::); }
    }
    for (int kt = 0; kt < nk; kt++) {
        int cur = kt % ST;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(ST - 2));
        __syncthreads();
        // refill the stage consumed last iteration
        int nx = kt + ST - 1;
        if (nx < nk) { LOAD(nx % ST, nx * G1_BK); } else { asm volatile("cp.async.commit_group;\n" ::); }
        const __half* A = sAbase + cur * (G1_BM * G1_SROW);
        const __half* B = sBbase + cur * (BN * G1_SROW);
        #pragma unroll
        for (int ks = 0; ks < G1_BK; ks += 16) {
            unsigned a[2][4], b[NI][2];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const __half* ap = A + (wm * 2 * 16 + mi * 16 + (lane & 15)) * G1_SROW + ks + (lane >> 4) * 8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(a[mi][0]), "=r"(a[mi][1]), "=r"(a[mi][2]), "=r"(a[mi][3]) : "l"(ap));
            }
            // B (n rows, k contiguous) by ldmatrix: matrices (n 0-7 | 8-15) x (k 0-7 | 8-15)
            #pragma unroll
            for (int ni = 0; ni < NI; ni += 2) {
                const __half* bp = B + (wn * NI * 8 + ni * 8 + (lane >> 4) * 8 + (lane & 7)) * G1_SROW
                    + ks + ((lane >> 3) & 1) * 8;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
                    : "=r"(b[ni][0]), "=r"(b[ni][1]), "=r"(b[ni + 1][0]), "=r"(b[ni + 1][1]) : "l"(bp));
            }
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                #pragma unroll
                for (int ni = 0; ni < NI; ni++)
                    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]), "r"(b[ni][0]), "r"(b[ni][1]));
        }
    }
    #undef LOAD
    // C reg l: row (pixel) (l/2)*8 + R, col (output) 2Q + l%2 -> half2 stores
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int hh = 0; hh < 2; hh++) {
            int m = m0 + wm * 2 * 16 + mi * 16 + hh * 8 + R;
            if (m >= M) continue;
            long yo = (long)(m / opix) * oimg + yoff[m % opix];
            #pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                int n = n0 + wn * NI * 8 + ni * 8 + 2 * Q;
                if (n >= N) continue;
                float v0 = __half2float(__float2half(acc[mi][ni][hh * 2]));
                float v1 = __half2float(__float2half(acc[mi][ni][hh * 2 + 1]));
                if (has_bias) {
                    v0 = __half2float(__float2half(v0 + __half2float(bias[n])));
                    if (n + 1 < N) v1 = __half2float(__float2half(v1 + __half2float(bias[n + 1])));
                }
                if (act == CNN_ACT_FLOOR) {
                    v0 = fmaxf(v0, __half2float(bias[N + n]));
                    if (n + 1 < N) v1 = fmaxf(v1, __half2float(bias[N + n + 1]));
                } else if (act != 0) { v0 = cnn_act_f(v0, act); v1 = cnn_act_f(v1, act); }
                if (n + 1 < N) {
                    *(__half2*)(y + yo + n) = __floats2half2_rn(v0, v1);
                } else {
                    y[yo + n] = __float2half(v0);
                }
            }
        }
}
// entries: N tile 128 / 64 / 32, 2 stages (3-4 stages cost the second resident block per SM)
extern "C" __global__ void __launch_bounds__(256) cnn_igemm128x2_f16(const __half* x, const __half* w, __half* y,
    const int* xoff, const int* yoff, const int* toff, const __half* bias, int M, int N, int K, int kpad,
    int opix, int iimg, int oimg, int cin, int ntaps, int act, int has_bias) {
    cnn_igemm_body<128, 2>(x, w, y, xoff, yoff, toff, bias, M, N, K, kpad, opix, iimg, oimg, cin, ntaps, act, has_bias);
}
extern "C" __global__ void __launch_bounds__(256) cnn_igemm64x2_f16(const __half* x, const __half* w, __half* y,
    const int* xoff, const int* yoff, const int* toff, const __half* bias, int M, int N, int K, int kpad,
    int opix, int iimg, int oimg, int cin, int ntaps, int act, int has_bias) {
    cnn_igemm_body<64, 2>(x, w, y, xoff, yoff, toff, bias, M, N, K, kpad, opix, iimg, oimg, cin, ntaps, act, has_bias);
}
extern "C" __global__ void __launch_bounds__(256) cnn_igemm32x2_f16(const __half* x, const __half* w, __half* y,
    const int* xoff, const int* yoff, const int* toff, const __half* bias, int M, int N, int K, int kpad,
    int opix, int iimg, int oimg, int cin, int ntaps, int act, int has_bias) {
    cnn_igemm_body<32, 2>(x, w, y, xoff, yoff, toff, bias, M, N, K, kpad, opix, iimg, oimg, cin, ntaps, act, has_bias);
}

// Zero the cells of padded NHWC buffers that no kernel writes, so memory another
// executor used can be reused: the pad border and spare columns (all cs channels),
// and channels [c, cs) of interior pixels. The border of one image is h + 1
// contiguous runs (top rows + first left pad; right pad + next left pad, h - 1
// times; last right pad + bottom rows); the slack is one run per interior pixel.
// A warp per run, 16-byte stores when aligned.
// One launch for many buffers: blockIdx.y = buffer, tab[9 * y ..] = (element
// offset from y (16-byte aligned), n, hp, wp, cs, pad, h, w, c).
// grid = (blocks, buffers), block = 256.
extern "C" __global__ void cnn_zero_pad_f16(__half* y, const int* tab) {
    const int* t = tab + 9 * blockIdx.y;
    int n = t[1], hp = t[2], wp = t[3], cs = t[4], pad = t[5], h = t[6], w = t[7], c = t[8];
    __half* base = y + (unsigned)t[0];
    long isz = (long)hp * wp * cs;
    long nseg = h + 1, nsl = c < cs ? (long)h * w : 0, per = nseg + nsl;
    int lane = threadIdx.x & 31;
    for (long r = ((long)blockIdx.x * 256 + threadIdx.x) >> 5; r < (long)n * per; r += (long)gridDim.x * 8) {
        long img = r / per, k = r % per, start, len;
        if (k == 0) {
            start = 0; len = ((long)pad * wp + pad) * cs;
        } else if (k < h) {
            start = ((long)(pad + k - 1) * wp + pad + w) * cs; len = (long)(wp - w) * cs;
        } else if (k == h) {
            start = ((long)(pad + h - 1) * wp + pad + w) * cs; len = isz - start;
        } else {
            long p = k - nseg;
            start = ((long)(pad + p / w) * wp + pad + p % w) * cs + c; len = cs - c;
        }
        __half* s = base + img * isz + start;
        if (((start | len | isz) & 7) == 0) {
            for (long j = lane; j < len / 8; j += 32) ((uint4*)s)[j] = make_uint4(0, 0, 0, 0);
        } else {
            for (long j = lane; j < len; j += 32) s[j] = __float2half(0.f);
        }
    }
}

// Direct convolution on the tensor cores, NHWC, groups == 1. The two entries are
// generated from one template at the end of this source (see `conv_kernels`).
//
// Y = W · unfold(X) with patch order j = (ky*kwe + kx)*C + ic, so one patch row
// (kwe*C halves, a multiple of 16) is a contiguous run of the padded NHWC input and
// every B operand word is a direct 2-half read. A is the host-packed lane-register
// form of W (see conv::pack_weights). fp16 m16n8k16 `mma` layout, lane L = 4R + Q:
//   A words: rows {R, R+8} x cols {2Q,2Q+1 | 2Q+8,2Q+9}
//   B words: col R x rows {2Q,2Q+1 | 2Q+8,2Q+9}
//   C reg l: row (l/2)*8 + R, col 2Q + l%2
// A block holds ng = 2^lg 16-row output groups and 8/ng pixel tiles (npix raster
// pixels each); warp w computes group (w & (ng-1)) against tile (w >> lg), so narrow
// layers do not burn warps on zero rows.
// grid = (ceil(N*tpi / (8/ng)), ceil(ceil(out_c/16) / ng)), block = 256.
// `pstride` is the input's per-pixel channel stride (a view of a wider concat
// buffer); output strides are baked into tpy/oimg by the host. Per tile t, pixel i:
// tpo[t*64+i] = patch origin in the padded input image, tpy[t*64+i] = output offset.
// cnn_conv_f16 scatters the accumulators from registers and ignores bias/act (the
// executor then runs cnn_bias_act_nhwc); cnn_conv_fused_f16 applies them as
// y = act(f16(acc) + bias), parking its 16 x 64 tile in shared memory as f16 first
// (the rounding cuDNN and torch apply before the bias add) so lanes walk output
// channels with contiguous NHWC writes. Running the epilogue from the accumulator
// registers makes the compiler spill them: +35%, against ~5% via shared memory.

// Depthwise k x k (k in {3, 5}) on padded NHWC, two channels per thread
// (half2 loads: a warp reads 128 contiguous bytes), the pair's k*k weights in
// registers, float accumulation in kH-then-kW order, bias + activation fused
// (conv output rounded to f16 before the bias, as in the unfused path).
// Needs even ch / ics / ocs / channel offsets. Block = (image*out_h + oy,
// 64-channel chunk), 256 threads = 32 channel pairs x 8 pixel phases.
template <int K>
__device__ __forceinline__ void dw2_body(const __half* x, const __half* weight, __half* y,
        const __half* bias, int ch, int h, int w, int ipad, int wp, int img, int out_w,
        int sh, int sw, int pt, int pl, int opad, int owp, int oimg, int ics, int ocs,
        int act, int has_bias, int n, int oy, int c, int phase) {
    float2 wr[K * K];
    #pragma unroll
    for (int t = 0; t < K * K; t++)
        wr[t] = make_float2(__half2float(weight[c * K * K + t]), __half2float(weight[(c + 1) * K * K + t]));
    float b0 = 0.f, b1 = 0.f;
    if (has_bias) { b0 = __half2float(bias[c]); b1 = __half2float(bias[c + 1]); }
    const __half* xb = x + (long)n * img + c;
    for (int ox = phase; ox < out_w; ox += 8) {
        float a0 = 0.f, a1 = 0.f;
        #pragma unroll
        for (int ky = 0; ky < K; ky++) {
            int iy = oy * sh - pt + ky;
            if (iy < 0 || iy >= h) continue;
            const __half* row = xb + (long)(iy + ipad) * wp * ics;
            #pragma unroll
            for (int kx = 0; kx < K; kx++) {
                int ix = ox * sw - pl + kx;
                if (ix < 0 || ix >= w) continue;
                float2 f = __half22float2(*(const __half2*)(row + (long)(ix + ipad) * ics));
                a0 += wr[ky * K + kx].x * f.x;
                a1 += wr[ky * K + kx].y * f.y;
            }
        }
        if (has_bias) {
            a0 = __half2float(__float2half(__half2float(__float2half(a0)) + b0));
            a1 = __half2float(__float2half(__half2float(__float2half(a1)) + b1));
        }
        *(__half2*)(y + (long)n * oimg + ((long)(oy + opad) * owp + ox + opad) * ocs + c) =
            __floats2half2_rn(cnn_act_f(a0, act), cnn_act_f(a1, act));
    }
}

extern "C" __global__ void cnn_dwconv2_nhwc_f16(const __half* x, const __half* weight, __half* y,
                                              const __half* bias, int ch, int h, int w, int ipad,
                                              int wp, int img, int out_h, int out_w, int k,
                                              int sh, int sw, int pt, int pl, int opad, int owp,
                                              int oimg, int ics, int ocs, int act, int has_bias) {
    int lane = threadIdx.x & 31, phase = threadIdx.x >> 5;
    int n = blockIdx.x / out_h, oy = blockIdx.x % out_h;
    int c = (blockIdx.y * 32 + lane) * 2;
    if (c >= ch) return;
    if (k == 5)
        dw2_body<5>(x, weight, y, bias, ch, h, w, ipad, wp, img, out_w, sh, sw, pt, pl, opad, owp, oimg, ics, ocs, act, has_bias, n, oy, c, phase);
    else
        dw2_body<3>(x, weight, y, bias, ch, h, w, ipad, wp, img, out_w, sh, sw, pt, pl, opad, owp, oimg, ics, ocs, act, has_bias, n, oy, c, phase);
}

// Crop + cv2-exact bilinear resize + place + normalize, u8 frames to
// padded-NHWC f16 model input. Byte-identical to ojas-vision's pre.rs
// (resize_bilinear_rgb8 + letterbox / OCR placement): 11-bit coefficients,
// cv2's ((b*(h>>4))>>16 + ...) + 2) >> 2 vertical, the same f32 coordinate
// math with no contraction (__fmul_rn / __fadd_rn), v/div or v*scale+shift.
// Frames are RGB8 (fmt 0) or NV12 (fmt 1: Y plane, interleaved UV plane at
// uv_off bytes; every tap converted with cv2's BT.601 fixed point,
// = pre::nv12_to_rgb8). One thread per destination pixel;
// grid = (ceil(tw*th/256), items).
// desc[i*14 ..]: frame address lo, hi, pitch (bytes/row), fmt, uv_off,
// roi x0, y0, w, h, resized w, h, left, top (placement in the tw x th canvas), 0.
__device__ __forceinline__ void cnn_tap_rgb(const unsigned char* base, int pitch, int fmt,
                                            int uv_off, int x, int y, int* rgb) {
    if (fmt == 0) {
        const unsigned char* p = base + (long)y * pitch + (long)x * 3;
        rgb[0] = p[0]; rgb[1] = p[1]; rgb[2] = p[2];
        return;
    }
    int yy = base[(long)y * pitch + x];
    const unsigned char* uv = base + uv_off + (long)(y >> 1) * pitch + (x & ~1);
    int u = (int)uv[0] - 128, v = (int)uv[1] - 128;
    int yv = (yy - 16 > 0 ? yy - 16 : 0) * 1220542;
    int r = (yv + (1 << 19) + 1673527 * v) >> 20;
    int g = (yv + (1 << 19) - 852492 * v - 409993 * u) >> 20;
    int b = (yv + (1 << 19) + 2116026 * u) >> 20;
    rgb[0] = r < 0 ? 0 : (r > 255 ? 255 : r);
    rgb[1] = g < 0 ? 0 : (g > 255 ? 255 : g);
    rgb[2] = b < 0 ? 0 : (b > 255 ? 255 : b);
}

// NV12 (or RGB8) frame -> packed RGB8, every `step`-th pixel (1 = full size),
// with the crop kernel's cv2 BT.601 fixed point: the host copy of a device
// frame for CPU consumers (tracker appearance, evidence crops). desc = the
// crop descriptor's first 5 words [ptr_lo, ptr_hi, pitch, fmt, uv_off].
// grid = (ceil(ow*oh/256)), block = 256.
extern "C" __global__ void cnn_frame_rgb8(const unsigned* desc, unsigned char* out, int ow, int oh, int step) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= ow * oh) return;
    const unsigned char* base = (const unsigned char*)(((unsigned long long)desc[1] << 32) | desc[0]);
    int rgb[3];
    cnn_tap_rgb(base, desc[2], desc[3], desc[4], (p % ow) * step, (p / ow) * step, rgb);
    out[p * 3] = rgb[0]; out[p * 3 + 1] = rgb[1]; out[p * 3 + 2] = rgb[2];
}

extern "C" __global__ void cnn_crop_resize_u8(const unsigned* desc, __half* y, int tw, int th,
                                            int pad, int wp, int img, int cs, float div,
                                            float s0, float s1, float s2, float b0, float b1, float b2,
                                            float f0, float f1, float f2, int swap_rb) {
    // per-channel normalisation: u/div (div > 0) or u·s + b; the fill is already normalised
    const float scale[3] = {s0, s1, s2}, shift[3] = {b0, b1, b2}, fill[3] = {f0, f1, f2};
    int item = blockIdx.y;
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= tw * th) return;
    int dx = p % tw, dy = p / tw;
    const unsigned* d = desc + item * 14;
    const unsigned char* base = (const unsigned char*)(((unsigned long long)d[1] << 32) | d[0]);
    int pitch = d[2], fmt = d[3], uv_off = d[4], rx0 = d[5], ry0 = d[6], rw = d[7], rh = d[8];
    int nw = d[9], nh = d[10], left = d[11], top = d[12];
    __half* out = y + (long)item * img + ((long)(dy + pad) * wp + dx + pad) * cs;
    int ox = dx - left, oy = dy - top;
    if (ox < 0 || ox >= nw || oy < 0 || oy >= nh) {
        for (int c = 0; c < 3; c++) out[swap_rb ? 2 - c : c] = __float2half(fill[c]);
        return;
    }
    int u[3];
    if (rw == nw && rh == nh) {
        cnn_tap_rgb(base, pitch, fmt, uv_off, rx0 + ox, ry0 + oy, u);
    } else {
        // x taps
        float scx = (float)rw / (float)nw;
        float fx = __fsub_rn(__fmul_rn(__fadd_rn((float)ox, 0.5f), scx), 0.5f);
        int sx = (int)floorf(fx);
        fx = __fsub_rn(fx, (float)sx);
        if (sx < 0) { sx = 0; fx = 0.f; }
        if (sx >= rw - 1) { sx = rw - 2; fx = 1.f; }
        if (sx < 0) sx = 0;
        int a1 = (int)roundf(__fmul_rn(fx, 2048.f)), a0 = 2048 - a1;
        int sx1 = sx + 1 < rw - 1 ? sx + 1 : rw - 1;
        // y taps
        float scy = (float)rh / (float)nh;
        float fy = __fsub_rn(__fmul_rn(__fadd_rn((float)oy, 0.5f), scy), 0.5f);
        int sy = (int)floorf(fy);
        fy = __fsub_rn(fy, (float)sy);
        if (sy < 0) { sy = 0; fy = 0.f; }
        if (sy >= rh - 1) { sy = rh - 2; fy = 1.f; }
        if (sy < 0) sy = 0;
        int b1 = (int)roundf(__fmul_rn(fy, 2048.f)), b0 = 2048 - b1;
        int sy1 = sy + 1 < rh - 1 ? sy + 1 : rh - 1;
        int t00[3], t01[3], t10[3], t11[3];
        cnn_tap_rgb(base, pitch, fmt, uv_off, rx0 + sx, ry0 + sy, t00);
        cnn_tap_rgb(base, pitch, fmt, uv_off, rx0 + sx1, ry0 + sy, t01);
        cnn_tap_rgb(base, pitch, fmt, uv_off, rx0 + sx, ry0 + sy1, t10);
        cnn_tap_rgb(base, pitch, fmt, uv_off, rx0 + sx1, ry0 + sy1, t11);
        for (int c = 0; c < 3; c++) {
            int h0 = a0 * t00[c] + a1 * t01[c];
            int h1 = a0 * t10[c] + a1 * t11[c];
            int v = ((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16);
            int w = (v + 2) >> 2;
            u[c] = w < 0 ? 0 : (w > 255 ? 255 : w);
        }
    }
    for (int c = 0; c < 3; c++) {
        float f = div > 0.f ? __fdiv_rn((float)u[c], div) : __fadd_rn(__fmul_rn((float)u[c], scale[c]), shift[c]);
        out[swap_rb ? 2 - c : c] = __float2half(f);
    }
}

// Detector score filter on a channels-first head [N, 4+nc, A] (f16): an anchor
// survives when its best allowed class score (strict >, from -1, the yolo::decode
// rule) is >= conf. Survivors are appended per image (atomic slot) as their anchor
// index plus the 4+nc column; the host sorts by anchor and runs the CPU decode on
// just those columns.
// cmask: class bitmask words (bit c = allowed). Grid = (ceil(A/256), N).
extern "C" __global__ void cnn_det_filter_f16(const __half* head, const unsigned* cmask,
                                            unsigned* count, unsigned* idx, __half* cols,
                                            int rows, int anchors, int cap, float conf) {
    int a = blockIdx.x * blockDim.x + threadIdx.x, n = blockIdx.y;
    if (a >= anchors) return;
    const __half* h = head + (long)n * rows * anchors + a;
    float best = -1.0f;
    for (int c = 0; c < rows - 4; c++) {
        float v = __half2float(h[(long)(4 + c) * anchors]);
        if (v > best && ((cmask[c >> 5] >> (c & 31)) & 1u)) best = v;
    }
    if (best < conf) return;
    unsigned slot = atomicAdd(count + n, 1u);
    if (slot >= (unsigned)cap) return;
    idx[(long)n * cap + slot] = a;
    __half* o = cols + ((long)n * cap + slot) * rows;
    for (int r = 0; r < rows; r++) o[r] = h[(long)r * anchors];
}

@CONV_KERNELS@
"#;

const HEADER: &str = r#"
#define CNN_INF (1.0f / 0.0f)
#define CNN_FLT_MAX 3.402823466e+38f
"#;

const F32: (&str, &str, &str) = ("float", "", "#define LD(p, i) ((p)[i])\n#define ST(p, i, v) ((p)[i] = (v))\n");
const F16: (&str, &str, &str) = (
    "__half",
    "_f16",
    "#define LD(p, i) __half2float((p)[i])\n#define ST(p, i, v) ((p)[i] = __float2half(v))\n",
);

const CONV_HEAD: &str = r#"extern "C" __global__ void @NAME@(const __half* xp, const unsigned* wpk, __half* y,
                                      const int* tpo, const int* tpy, const __half* bias,
                                      int img, int row_step, int rowlen, int out_c,
                                      int opix, int c16, int npix, int nimg, int lg,
                                      int oimg, int cin, int pstride, int act,
                                      int has_bias) {
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, R = lane >> 2, Q = lane & 3;
    int ng = 1 << lg, tpb = 8 >> lg;
    int tpi = (opix + npix - 1) / npix;
    int T = blockIdx.x * tpb + (warp >> lg);
    int n = T / tpi, t = T % tpi, p0 = t * npix;
    int g = blockIdx.y * ng + (warp & (ng - 1));
    int oc_base = g * 16;
    int live = n < nimg && oc_base < out_c;
    const __half* xb = xp + (long)(live ? n : 0) * img;
    __shared__ int sPo[8 * 64];
    __shared__ int sPy[8 * 64];
    for (int idx = tid; idx < tpb * 64; idx += 256) {
        long tr = (long)((blockIdx.x * tpb + (idx >> 6)) % tpi) * 64 + (idx & 63);
        sPo[idx] = tpo[tr];
        sPy[idx] = tpy[tr];
    }
    __syncthreads();
    int tb = (warp >> lg) * 64;
    int po[8];
    #pragma unroll
    for (int tt = 0; tt < 8; tt++) po[tt] = sPo[tb + tt * 8 + R] + 2 * Q;
    float acc[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) acc[i] = 0.f;
    int cend = live ? c16 : 0;
    for (int c = 0; c < cend; c++) {
        const unsigned* wa = wpk + ((long)(g * c16 + c) * 32 + lane) * 4;
        unsigned a0 = wa[0], a1 = wa[1], a2 = wa[2], a3 = wa[3];
        // tap chunk -> (ky, kx, ic); pixels are pstride apart (a channel slice
        // of a wider buffer when pstride > cin, which needs cin % 16 == 0)
        int j0 = c * 16, ky = j0 / rowlen, jr = j0 - ky * rowlen, kx = jr / cin;
        int roff = ky * row_step + kx * pstride + (jr - kx * cin);
        #pragma unroll
        for (int tt = 0; tt < 8; tt++) {
            const unsigned* wb = (const unsigned*)(xb + po[tt] + roff);
            unsigned b0 = wb[0], b1 = wb[4];
            float c0f = 0, c1f = 0, c2f = 0, c3f = 0;
            asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c0f), "+f"(c1f), "+f"(c2f), "+f"(c3f)
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            acc[tt * 4 + 0] += c0f; acc[tt * 4 + 1] += c1f;
            acc[tt * 4 + 2] += c2f; acc[tt * 4 + 3] += c3f;
        }
    }
"#;

const CONV_TAIL_SPLIT: &str = r#"    if (live) {
        __half* yb = y + (long)n * oimg;
        #pragma unroll
        for (int tt = 0; tt < 8; tt++) {
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int oc = oc_base + (l >> 1) * 8 + R;
                int off = tt * 8 + 2 * Q + (l & 1);
                if (oc < out_c && off < npix && p0 + off < opix)
                    yb[sPy[tb + off] + oc] = __float2half(acc[tt * 4 + l]);
            }
        }
    }
}
"#;

const CONV_TAIL_FUSED: &str = r#"    __shared__ __half sAcc[8][64][18];
    #pragma unroll
    for (int tt = 0; tt < 8; tt++) {
        #pragma unroll
        for (int l = 0; l < 4; l++)
            sAcc[warp][tt * 8 + 2 * Q + (l & 1)][(l >> 1) * 8 + R] = __float2half(acc[tt * 4 + l]);
    }
    __syncwarp();
    if (live) {
        __half* yb = y + (long)n * oimg;
        for (int idx = lane; idx < 16 * 64; idx += 32) {
            int row = idx & 15, col = idx >> 4;
            int oc = oc_base + row;
            if (oc < out_c && col < npix && p0 + col < opix) {
                float v = __half2float(sAcc[warp][col][row]);
                if (has_bias) v = __half2float(__float2half(v + __half2float(bias[oc])));
                v = act == CNN_ACT_FLOOR ? fmaxf(v, __half2float(bias[out_c + oc])) : cnn_act_f(v, act);
                yb[sPy[tb + col] + oc] = __float2half(v);
            }
        }
    }
}
"#;

/// The direct conv entries: one main loop, two epilogues.
fn conv_kernels() -> String {
    format!(
        "{}{}\n{}{}",
        CONV_HEAD.replace("@NAME@", "cnn_conv_f16"),
        CONV_TAIL_SPLIT,
        CONV_HEAD.replace("@NAME@", "cnn_conv_fused_f16"),
        CONV_TAIL_FUSED
    )
}

/// The two per-precision copies of `body`, each framed by its own LD/ST macros.
/// Precision-independent helpers (everything before the first entry) are emitted once.
pub fn expand(body: &str) -> String {
    let (shared, templ) = split_shared(body);
    let mut out = String::from(HEADER);
    out.push_str(shared);
    for (ty, suf, macros) in [F32, F16] {
        out.push_str(macros);
        out.push_str(&templ.replace("@T@", ty).replace("@S@", suf));
        out.push_str("#undef LD\n#undef ST\n");
    }
    out.push_str(&BODY_F16.replace("@CONV_KERNELS@", &conv_kernels()));
    out
}

/// Split at the first `extern "C"`: everything before it is shared helpers.
fn split_shared(body: &str) -> (&str, &str) {
    let at = body.find("extern \"C\"").unwrap_or(0);
    body.split_at(at)
}

/// Canonical entry names registered from this family.
pub const NAMES: &[&str] = &[
    "cnn_bias_act", "cnn_bias_act_f16",
    "cnn_bias_act_nhwc", "cnn_bias_act_nhwc_f16",
    "cnn_act", "cnn_act_f16",
    "cnn_add", "cnn_add_f16",
    "cnn_axis_copy", "cnn_axis_copy_f16",
    "cnn_upsample_nearest", "cnn_upsample_nearest_f16",
    "cnn_maxpool", "cnn_maxpool_f16",
    "cnn_softmax", "cnn_softmax_f16",
    "cnn_dwconv", "cnn_dwconv_f16",
    "cnn_dwconv_nhwc", "cnn_dwconv_nhwc_f16",
    "cnn_maxpool_nhwc", "cnn_maxpool_nhwc_f16",
    "cnn_avgpool_nhwc", "cnn_avgpool_nhwc_f16",
    "cnn_upsample_nhwc", "cnn_upsample_nhwc_f16",
    "cnn_nhwc_to_nchw", "cnn_nhwc_to_nchw_f16",
    "cnn_nchw_to_nhwc", "cnn_nchw_to_nhwc_f16",
    "cnn_add_nhwc", "cnn_add_nhwc_f16",
    "cnn_add_act_nhwc", "cnn_add_act_nhwc_f16",
    "cnn_act_nhwc", "cnn_act_nhwc_f16",
    "cnn_binary_bcast", "cnn_binary_bcast_f16",
    "cnn_permute", "cnn_permute_f16",
    "cnn_matmul", "cnn_matmul_f16",
    "cnn_layernorm", "cnn_layernorm_f16",
    "cnn_grid_sample", "cnn_grid_sample_f16",
    "cnn_pad4", "cnn_pad4_f16",
    "cnn_scale_shift_nhwc", "cnn_scale_shift_nhwc_f16",
    "cnn_scale_shift", "cnn_scale_shift_f16",
    "cnn_reduce_last", "cnn_reduce_last_f16",
    "cnn_topk_last", "cnn_topk_last_f16",
    "cnn_topk_gather", "cnn_topk_gather_f16",
    "cnn_layernorm_warp", "cnn_layernorm_warp_f16",
    "cnn_softmax_rows", "cnn_softmax_rows_f16",
    "cnn_conv_f16",
    "cnn_conv_fused_f16",
    "cnn_zero_pad_f16", "cnn_igemm128x2_f16", "cnn_igemm64x2_f16", "cnn_igemm32x2_f16",
    "cnn_dwconv2_nhwc_f16",
    "cnn_crop_resize_u8", "cnn_frame_rgb8",
    "cnn_det_filter_f16",
];

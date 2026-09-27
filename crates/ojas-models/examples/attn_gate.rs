//! Gate: attn_prefill_fat must match attention_m_short on the same inputs.
//!
//! attention_m_short is the oldest, simplest attention in the tree (scalar
//! per-(query,head) threadgroup) and has survived every parity run, so it is the oracle.
//! The fat kernel is fragment-storage flash attention; a layout or softmax slip there
//! produces coherent-but-wrong generations that parity would catch only much later.
//!
//! usage: attn_gate
use anyhow::{bail, Result};
use ojas_core::Device;

const FRAG_TEST: &str = r#"
#include <metal_stdlib>
using namespace metal;
METAL_FUNC static ushort2 morton_order(ushort lane_id) {
  ushort quad_id = lane_id / 4;
  ushort M_in_simd = (quad_id / 4) * 4 + (lane_id / 2) % 4;
  ushort N_in_simd = (quad_id & 2) * 2 + (lane_id % 2) * 2;
  return ushort2(N_in_simd, M_in_simd);
}
#pragma METAL internals : enable
namespace metal {
  template <typename T>
  struct simdgroup_matrix_storage {
    typedef vec<T, 64> storage_type;
    storage_type t;
    METAL_FUNC thread vec<T, 2>* thread_elements() thread { return reinterpret_cast<thread vec<T, 2>*>(&t); }
    METAL_FUNC simdgroup_matrix_storage() thread = default;
    METAL_FUNC simdgroup_matrix_storage(vec<T, 2> te) thread { *(this->thread_elements()) = te; }
    METAL_FUNC static device float* apply_offset(device float *src, uint ld, uint2 origin) {
      return src + ulong(origin.y * ld) + origin.x;
    }
    METAL_FUNC static const device float* apply_offset(const device float *src, uint ld, uint2 origin) {
      return src + ulong(origin.y * ld) + origin.x;
    }
    template <typename U>
    METAL_FUNC void load(const device U *src, uint ld, uint2 origin) {
      ulong address = ulong(origin.y) * ld + ulong(origin.x);
      vec<U, 2> m = *(const device vec<U, 2>*)(src + address);
      *(thread_elements()) = vec<T, 2>(m);
    }
    template <typename U>
    METAL_FUNC void store(device U *dst, uint ld, uint2 origin) {
      ulong address = ulong(origin.y) * ld + ulong(origin.x);
      vec<T, 2> r = *(thread_elements());
      *(device vec<U, 2>*)(dst + address) = vec<U, 2>(r);
    }
    template <typename U, typename V>
    METAL_FUNC void multiply(simdgroup_matrix_storage<U> a, simdgroup_matrix_storage<V> b, bool accumulate = true) {
      if (!accumulate) { *(thread_elements()) = vec<T, 2>(0); }
      t = __metal_simdgroup_matrix_8x8_multiply_accumulate(a.t, b.t, t, typename simdgroup_matrix_storage<T>::storage_type());
    }
  };
}
#pragma METAL internals : disable
// C = A x B, all 8x8 f32 row-major in device memory. A is CONSTRUCTED per-lane
// (the fat attention's Q path); B is load()ed; C store()d.
kernel void frag_test(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]], ushort lane [[thread_index_in_simdgroup]]) {
    ushort2 mo = morton_order(lane);
    float a0 = a[uint(mo.y)*8u + uint(mo.x)];
    float a1 = a[uint(mo.y)*8u + uint(mo.x) + 1u];
    simdgroup_matrix_storage<half> af(half2(a0, a1));
    simdgroup_matrix_storage<half> bf;
    const device float* bm = simdgroup_matrix_storage<float>::apply_offset(b, 8, uint2(mo.x, mo.y));
    bf.load(bm, 8, uint2(0, 0));
    simdgroup_matrix_storage<float> cf(float2(0));
    cf.multiply(af, bf);
    device float* cm = simdgroup_matrix_storage<float>::apply_offset(c, 8, uint2(mo.x, mo.y));
    cf.store(cm, 8, uint2(0, 0));
}
"#;

fn main() -> Result<()> {
    // ---- primitive test first: constructed-fragment x loaded-fragment vs CPU
    {
        let gpu = ojas_metal::MetalGpu::new()?;
        let mut seed = 7u32;
        let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 };
        let a: Vec<f32> = (0..64).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..64).map(|_| rnd()).collect();
        let ab = gpu.upload(&a);
        let bb = gpu.upload(&b);
        let cb = gpu.alloc(64);
        let pipe = gpu.pipeline(FRAG_TEST, "frag_test")?;
        let cmd = gpu.command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&ab.buf), 0);
        enc.set_buffer(1, Some(&bb.buf), 0);
        enc.set_buffer(2, Some(&cb.buf), 0);
        enc.dispatch_thread_groups(metal::MTLSize::new(1,1,1), metal::MTLSize::new(32,1,1));
        enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
        let got = gpu.read(&cb);
        let mut worst = 0f32;
        for i in 0..8 { for j in 0..8 {
            let mut acc = 0f32;
            for k in 0..8 { acc += half::f16::from_f32(a[i*8+k]).to_f32() * half::f16::from_f32(b[k*8+j]).to_f32(); }
            worst = worst.max((acc - got[i*8+j]).abs());
        }}
        println!("frag primitive: max|diff| = {worst:.6} {}", if worst < 1e-2 { "OK" } else { "** CONSTRUCTION WRONG **" });
    }

    let gpu = ojas_metal::MetalGpu::new()?;
    let (m, seq_base, nh, hd, group) = (32usize, 24usize, 4usize, 128usize, 2usize);
    let kvdim = (nh / group) * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let total = seq_base + m; // kv positions 0..total; query i attends <= seq_base+i

    let mut seed = 0xC0FFEEu32;
    let mut rnd = || { seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); ((seed >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0 };
    let q: Vec<f32> = (0..m * nh * hd).map(|_| rnd()).collect();
    let kc: Vec<f32> = (0..total * kvdim).map(|_| rnd()).collect();
    let vc: Vec<f32> = (0..total * kvdim).map(|_| rnd()).collect();
    let kch: Vec<u16> = kc.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
    let vch: Vec<u16> = vc.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();

    let qb = gpu.upload(&q);
    let cast = |v: &[u16]| -> Vec<u8> {
        let mut out = Vec::with_capacity(v.len() * 2);
        for &x in v { out.extend_from_slice(&x.to_le_bytes()); }
        out
    };
    let kb = gpu.upload_u8(&cast(&kch));
    let vb = gpu.upload_u8(&cast(&vch));
    let o1 = gpu.alloc(m * nh * hd);
    let o2 = gpu.alloc(m * nh * hd);

    let run = |entry: &str, src: &str, out: &metal::Buffer, grid: (u64, u64), tpg: u64| -> Result<()> {
        let pipe = gpu.pipeline(src, entry)?;
        let cb = gpu.command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipe);
        enc.set_buffer(0, Some(&qb.buf), 0);
        enc.set_buffer(1, Some(&kb.buf), 0);
        enc.set_buffer(2, Some(&vb.buf), 0);
        enc.set_buffer(3, Some(out), 0);
        let ints: [(u64, u32); 6] = [(4, hd as u32), (5, kvdim as u32), (6, seq_base as u32), (7, group as u32), (9, nh as u32), (10, m as u32)];
        for (i, v) in ints { enc.set_bytes(i, 4, &v as *const u32 as *const std::ffi::c_void); }
        enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(metal::MTLSize::new(grid.0, grid.1, 1), metal::MTLSize::new(tpg, 1, 1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        Ok(())
    };

    let short_src = ojas_metal::kernels::source_of("attention_m_short").expect("short");
    run("attention_m_short", short_src, &o1.buf, ((m * nh) as u64, 1), 64)?;
    run("attn_prefill_fat", ojas_metal::kernels::gemm_fat::GEMM_FAT_KERNELS, &o2.buf, (nh as u64, ((m + 31) / 32) as u64), 256)?;

    let a = gpu.read(&o1);
    let b = gpu.read(&o2);
    let (mut dot, mut na, mut nb, mut worst, mut wi) = (0f64, 0f64, 0f64, 0f64, 0usize);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
        let d = (a[i] - b[i]).abs() as f64;
        if d > worst { worst = d; wi = i; }
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    println!("cos={cos:.6}  max|diff|={worst:.5} at {wi} (ref {:.4} vs fat {:.4})", a[wi], b[wi]);
    println!("ref[0..6]: {:?}", &a[..6]);
    println!("fat[0..6]: {:?}", &b[..6]);
    if cos > 0.9995 { println!("GATE: ATTN FAT PASS"); Ok(()) } else { bail!("GATE: ATTN FAT FAIL") }
}

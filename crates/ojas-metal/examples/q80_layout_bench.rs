//! Compare GGUF's interleaved Q8_0 blocks with an exact codes/scales split.
//! usage: q80_layout_bench [K] [N]

use anyhow::Result;
use metal::{MTLResourceOptions, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
use std::ffi::c_void;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define DOT_BODY(BLOCK) \
    uint row0=tg*2u; if(row0>=N)return; uint nsg=ts/32u,ix=lane/4u,il=lane%4u,nb=K/32u; \
    float sum[2]={0.0f,0.0f}; \
    for(uint b=sg*8u+ix;b<nb;b+=nsg*8u){uint col=b*32u+il*8u; \
      for(uint r=0;r<2u;r++){if(row0+r>=N)break; BLOCK }} \
    threadgroup float part[8]; for(uint r=0;r<2u;r++){float v=simd_sum(sum[r]);if(lane==0)part[r*4u+sg]=v;} \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if(sg==0u&&lane<2u&&row0+lane<N){float v=0.0f;for(uint s=0;s<nsg;s++)v+=part[lane*4u+s];out[row0+lane]=v;}

kernel void q80_raw(device const float* x [[buffer(0)]],device const uchar* w [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint& K [[buffer(3)]],constant uint& N [[buffer(4)]],
 uint tg [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
 uint lane [[thread_index_in_simdgroup]],uint ts [[threads_per_threadgroup]]){
 DOT_BODY(device const uchar* z=w+((ulong)(row0+r)*nb+b)*34ul;float d=float(*reinterpret_cast<device const half*>(z));device const char* q=(device const char*)(z+2u);float p=0.0f;for(uint j=0;j<8u;j++)p+=float(q[il*8u+j])*x[col+j];sum[r]+=d*p;)
}

kernel void q80_split(device const float* x [[buffer(0)]],device const char* q [[buffer(1)]],
 device float* out [[buffer(2)]],constant uint& K [[buffer(3)]],constant uint& N [[buffer(4)]],
 device const half* scales [[buffer(5)]],uint tg [[threadgroup_position_in_grid]],
 uint sg [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]],
 uint ts [[threads_per_threadgroup]]){
 DOT_BODY(device const char* z=q+(ulong)(row0+r)*K+b*32u;float d=float(scales[(ulong)(row0+r)*nb+b]);float p=0.0f;for(uint j=0;j<8u;j++)p+=float(z[il*8u+j])*x[col+j];sum[r]+=d*p;)
}

// Populate both layouts with the same non-zero values. Leaving new shared
// buffers untouched benchmarks macOS's shared zero page/compression rather than
// DRAM traffic, which produced a false 2-6% win in the original probe.
kernel void init_q80(device uchar* raw [[buffer(0)]], device char* q [[buffer(1)]],
 device half* scales [[buffer(2)]], constant uint& blocks [[buffer(3)]],
 uint gid [[thread_position_in_grid]]) {
 if (gid >= blocks) return;
 half d=half(0.00390625f*float(1u+(gid%31u))); scales[gid]=d;
 device uchar* b=raw+(ulong)gid*34ul;
 *reinterpret_cast<device half*>(b)=d;
 for(uint j=0;j<32u;j++) { char v=char(int((gid*17u+j*29u)%255u)-127); q[(ulong)gid*32ul+j]=v; b[2u+j]=uchar(v); }
}
"#;

fn main() -> Result<()> {
    let k: usize = std::env::args().nth(1).and_then(|x| x.parse().ok()).unwrap_or(4096);
    let n: usize = std::env::args().nth(2).and_then(|x| x.parse().ok()).unwrap_or(12288);
    anyhow::ensure!(k % 32 == 0 && n > 0, "K must be divisible by 32");
    let gpu = ojas_metal::MetalGpu::new()?;
    let raw = gpu.pipeline(SRC, "q80_raw")?;
    let split = gpu.pipeline(SRC, "q80_split")?;
    let init = gpu.pipeline(SRC, "init_q80")?;
    let dev = &gpu.device;
    let opts = MTLResourceOptions::StorageModeShared;
    let x = dev.new_buffer((k*4) as u64, opts);
    let y = dev.new_buffer((n*4) as u64, opts);
    let xv = unsafe { std::slice::from_raw_parts_mut(x.contents() as *mut f32, k) };
    for (i, v) in xv.iter_mut().enumerate() { *v = ((i % 257) as f32 - 128.0) / 257.0; }
    let nb = k/32;
    const COPIES: usize = 1;
    let wr: Vec<_> = (0..COPIES).map(|_| dev.new_buffer((n*nb*34) as u64, opts)).collect();
    let wq: Vec<_> = (0..COPIES).map(|_| dev.new_buffer((n*k) as u64, opts)).collect();
    let ws: Vec<_> = (0..COPIES).map(|_| dev.new_buffer((n*nb*2) as u64, opts)).collect();
    for c in 0..COPIES {
        let cb=gpu.command_buffer();let e=cb.new_compute_command_encoder();
        e.set_compute_pipeline_state(&init);e.set_buffer(0,Some(&wr[c]),0);e.set_buffer(1,Some(&wq[c]),0);
        e.set_buffer(2,Some(&ws[c]),0);let blocks=(n*nb) as u32;e.set_bytes(3,4,&blocks as *const u32 as *const c_void);
        e.dispatch_thread_groups(MTLSize::new(blocks.div_ceil(256) as u64,1,1),MTLSize::new(256,1,1));
        e.end_encoding();cb.commit();cb.wait_until_completed();
    }
    let run = |name: &str| -> f64 {
        let p = if name=="raw" { &raw } else { &split };
        let cb=gpu.command_buffer();let e=cb.new_compute_command_encoder();
        for c in 0..COPIES { e.set_compute_pipeline_state(p);e.set_buffer(0,Some(&x),0);
            e.set_buffer(1,Some(if name=="raw"{&wr[c]}else{&wq[c]}),0);e.set_buffer(2,Some(&y),0);
            let ku=k as u32;let nu=n as u32;e.set_bytes(3,4,&ku as *const u32 as *const c_void);
            e.set_bytes(4,4,&nu as *const u32 as *const c_void);if name!="raw"{e.set_buffer(5,Some(&ws[c]),0);}
            e.dispatch_thread_groups(MTLSize::new(n.div_ceil(2) as u64,1,1),MTLSize::new(128,1,1));}
        e.end_encoding();cb.commit();cb.wait_until_completed();
        let (a,b):(f64,f64)=unsafe{(msg_send![cb,GPUStartTime],msg_send![cb,GPUEndTime])};(b-a)*1e3/COPIES as f64
    };
    run("raw");run("split");
    let (mut av,mut bv)=(Vec::new(),Vec::new());
    for i in 0..12 { if i%2==0 { av.push(run("raw"));bv.push(run("split")); }
                     else { bv.push(run("split"));av.push(run("raw")); } }
    av.sort_by(f64::total_cmp);bv.sort_by(f64::total_cmp);
    let a=(av[5]+av[6])*0.5;let b=(bv[5]+bv[6])*0.5;
    println!("K={k} N={n}: raw {a:.4} ms, split {b:.4} ms, speedup {:.3}x",a/b);
    Ok(())
}

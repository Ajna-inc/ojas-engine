//! Sustained read-bandwidth probe, not an inference throughput benchmark.
use anyhow::{ensure, Result};
use metal::{MTLResourceOptions as Opt, MTLSize};
use objc::{msg_send, sel, sel_impl};
use ojas_core::Device as _;
fn main() -> Result<()> {
    let gpu = ojas_metal::MetalGpu::new()?;
    println!("device={}", gpu.device.name());
    let src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void read_roof(device const uint4* x [[buffer(0)]],device uint* out [[buffer(1)]],
constant uint& n [[buffer(2)]],constant uint& stride [[buffer(3)]],
uint gid [[thread_position_in_grid]],ushort lane [[thread_index_in_simdgroup]]) {
 uint4 sum=0;
 for(uint i=gid;i<n;i+=stride) {sum+=x[i];}
 uint s=simd_sum(sum.x+sum.y+sum.z+sum.w);
 if(lane==0) out[gid/32]=s;
}
"#;
    let pipe = gpu.pipeline(src, "read_roof")?;
    let bytes = 256usize * 1024 * 1024;
    let data: Vec<u32> = (0..bytes / 4)
        .map(|i| (i as u32).wrapping_mul(747796405).wrapping_add(2891336453))
        .collect();
    let expected = data.iter().fold(0u32, |a, &b| a.wrapping_add(b));
    let x =
        gpu.device
            .new_buffer_with_data(data.as_ptr().cast(), bytes as u64, Opt::StorageModeShared);
    drop(data);
    for groups in [256u32, 1024, 4096, 16384] {
        let threads = 256u32;
        let stride = groups * threads;
        let n = bytes as u32 / 16;
        let out = gpu
            .device
            .new_buffer((stride / 32 * 4) as u64, Opt::StorageModeShared);
        let mut times = Vec::new();
        for rep in 0..24 {
            let cb = gpu.command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&pipe);
            e.set_buffer(0, Some(&x), 0);
            e.set_buffer(1, Some(&out), 0);
            e.set_bytes(2, 4, (&n as *const u32).cast());
            e.set_bytes(3, 4, (&stride as *const u32).cast());
            e.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(threads as u64, 1, 1),
            );
            e.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            ensure!(
                cb.status() == metal::MTLCommandBufferStatus::Completed,
                "GPU failed"
            );
            let got = unsafe {
                std::slice::from_raw_parts(out.contents().cast::<u32>(), (stride / 32) as usize)
            }
            .iter()
            .fold(0u32, |a, &b| a.wrapping_add(b));
            ensure!(got == expected, "checksum mismatch");
            let start: f64 = unsafe { msg_send![cb, GPUStartTime] };
            let end: f64 = unsafe { msg_send![cb, GPUEndTime] };
            ensure!(end > start, "timestamps unavailable");
            if rep >= 4 {
                times.push(end - start);
            }
        }
        times.sort_by(f64::total_cmp);
        println!("{{\"groups\":{groups},\"bytes\":{bytes},\"median_ms\":{},\"read_GB_s\":{},\"p20_GB_s\":{},\"p80_GB_s\":{},\"checksum_ok\":true}}",times[10]*1000.,bytes as f64/times[10]/1e9,bytes as f64/times[16]/1e9,bytes as f64/times[4]/1e9);
    }
    Ok(())
}

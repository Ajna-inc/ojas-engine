//! A layered pass encoded across several command buffers.
//!
//! macOS ends a command buffer that holds the GPU long enough to delay the display
//! (kIOGPUCommandBufferCallbackErrorImpactingInteractivity, 0x0e), and a 256-row
//! prefill chunk through a large model is one. [`SplitPass`] opens a new command
//! buffer every `prefill_cb_layers` layers and commits the finished one straight
//! away, so the GPU never waits on the CPU between them. Command buffers on one
//! queue run in commit order and the decoder's buffers are hazard-tracked, so the
//! activations carry across the boundary with no extra synchronisation.
//!
//! On Qwen3.5 4B (M2 Max) one layer per command buffer and a whole chunk in one time
//! the same, 0.67 s and 0.68 s to the first token of a 591-token prompt.

use objc::{msg_send, sel, sel_impl};
use ojas_metal::MetalGpu;

/// Chunks this short stay in one command buffer: they are far from the watchdog,
/// and a speculative verify step would pay a commit per layer for nothing.
const SPLIT_MIN_ROWS: u32 = 32;

pub(crate) struct SplitPass<'g> {
    gpu: &'g MetalGpu,
    concurrent: bool,
    layers: usize,
    cb: Option<metal::CommandBuffer>,
    committed: Vec<metal::CommandBuffer>,
}

impl<'g> SplitPass<'g> {
    /// A pass over `rows` rows whose encoders dispatch concurrently or serially.
    /// `layers` is the group size (`EngineConfig::prefill_cb_layers`); 0 disables
    /// splitting.
    pub(crate) fn new(gpu: &'g MetalGpu, concurrent: bool, layers: usize, rows: u32) -> Self {
        let layers = if rows < SPLIT_MIN_ROWS { 0 } else { layers };
        SplitPass { gpu, concurrent, layers, cb: None, committed: Vec::new() }
    }

    /// Opens the first command buffer and returns its encoder.
    pub(crate) fn open(&mut self) -> metal::ComputeCommandEncoder {
        let cb = self.gpu.command_buffer().to_owned();
        let enc = if self.concurrent {
            cb.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent).to_owned()
        } else {
            cb.new_compute_command_encoder().to_owned()
        };
        self.cb = Some(cb);
        enc
    }

    /// Called before layer `l` is encoded: at a group boundary, ends `enc`, commits
    /// its command buffer and replaces `enc` with the next one's encoder.
    pub(crate) fn layer(&mut self, l: usize, enc: &mut metal::ComputeCommandEncoder) {
        if self.layers == 0 || l == 0 || l % self.layers != 0 { return; }
        enc.end_encoding();
        let done = self.cb.take().expect("SplitPass::layer before open");
        done.commit();
        self.committed.push(done);
        *enc = self.open();
    }

    /// Ends `enc`, commits the last command buffer and waits for all of them.
    /// Returns the GPU time they took. A failure is latched in
    /// `ojas_core::device_fault` under `context`; the rest still complete.
    pub(crate) fn finish(mut self, enc: &metal::ComputeCommandEncoderRef, context: &str) -> f64 {
        enc.end_encoding();
        let last = self.cb.take().expect("SplitPass::finish before open");
        last.commit();
        self.committed.push(last);
        let mut gpu = 0.0;
        for cb in &self.committed {
            let _ = ojas_metal::wait_checked(cb, context);
            let (gs, ge): (f64, f64) = unsafe { (msg_send![&**cb, GPUStartTime], msg_send![&**cb, GPUEndTime]) };
            gpu += ge - gs;
        }
        gpu
    }

    /// Command buffers this pass committed, for debug output.
    pub(crate) fn command_buffers(&self) -> usize { self.committed.len() + usize::from(self.cb.is_some()) }
}

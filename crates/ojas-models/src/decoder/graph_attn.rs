#![allow(clippy::too_many_arguments)]
use super::*;
use metal::MTLSize;
use std::ffi::c_void;
 // re-export

impl<'a> DecoderGpu<'a> {
    /// Long-context decode attention: flash-decoding split (attention_part over
    /// NWG KV slices + exact merge). Use when seq > 2048.
    pub(crate) fn attn_flash(&self, enc: &metal::ComputeCommandEncoderRef, n_head: u32, l: usize, kv_src: usize,
                  hd: u32, kvdim: u32, seq: u32, group: u32, scale: f32) {
        let nwg = ((seq as usize + 1023) / 1024).clamp(2, ojas_metal::kernels::attn::ATTN_NWG) as u32;
        enc.set_compute_pipeline_state(&self.p["attention_part"]);
        enc.set_buffer(0, Some(&self.st.q), 0);
        enc.set_buffer(1, Some(&self.st.kcache[l]), 0);
        enc.set_buffer(2, Some(&self.st.vcache[kv_src]), 0);
        enc.set_buffer(3, Some(&self.st.attn_part), 0);
        enc.set_bytes(4, 4, &hd as *const u32 as *const c_void);
        enc.set_bytes(5, 4, &kvdim as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &seq as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &group as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &scale as *const f32 as *const c_void);
        enc.set_bytes(9, 4, &nwg as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_head as u64, nwg as u64, 1), MTLSize::new(256, 1, 1));
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["attention_merge"]);
        enc.set_buffer(0, Some(&self.st.attn_part), 0);
        enc.set_buffer(1, Some(&self.st.attn), 0);
        enc.set_bytes(2, 4, &hd as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &nwg as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_head as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

    /// Quest-style page-sparse decode attention (OJAS_SPARSE): refresh the page
    /// containing the just-stored position, per-head top-K page selection, then
    /// flash-decoding over only the selected pages + exact merge. Approximate:
    /// softmax is exact over the kept set; the pruning is the approximation.
    pub(crate) fn sparse_attn(&self, enc: &metal::ComputeCommandEncoderRef, n_head: u32, l: usize,
                   hd: u32, kvdim: u32, seq: u32, group: u32, scale: f32, budget: u32) {
        let pg = (seq - 1) / ojas_metal::kernels::attn::PAGE as u32;
        self.enc_reduce(enc, "page_minmax",
            &[(&self.st.kcache[l], 0), (&self.st.pmeta[l], 1)],
            &[(2, kvdim), (3, pg), (4, seq)], &[], 1, 256);
        self.bar(enc); // page metadata current
        let ksel = (budget / ojas_metal::kernels::attn::PAGE as u32).clamp(8, ojas_metal::kernels::attn::MAXSEL as u32);
        self.enc_reduce(enc, "page_select",
            &[(&self.st.q, 0), (&self.st.pmeta[l], 1), (&self.st.plist, 2)],
            &[(3, hd), (4, kvdim), (5, seq), (6, group), (7, ksel)], &[], n_head as u64, 256);
        self.bar(enc); // page lists ready
        let npsel = ksel.min((seq + ojas_metal::kernels::attn::PAGE as u32 - 1) / ojas_metal::kernels::attn::PAGE as u32);
        let nwg = ((npsel as usize * ojas_metal::kernels::attn::PAGE + 1023) / 1024).clamp(2, ojas_metal::kernels::attn::ATTN_NWG) as u32;
        enc.set_compute_pipeline_state(&self.p["attention_part_sparse"]);
        enc.set_buffer(0, Some(&self.st.q), 0);
        enc.set_buffer(1, Some(&self.st.kcache[l]), 0);
        enc.set_buffer(2, Some(&self.st.vcache[l]), 0);
        enc.set_buffer(3, Some(&self.st.attn_part), 0);
        enc.set_buffer(4, Some(&self.st.plist), 0);
        enc.set_bytes(5, 4, &hd as *const u32 as *const c_void);
        enc.set_bytes(6, 4, &kvdim as *const u32 as *const c_void);
        enc.set_bytes(7, 4, &seq as *const u32 as *const c_void);
        enc.set_bytes(8, 4, &group as *const u32 as *const c_void);
        enc.set_bytes(9, 4, &scale as *const f32 as *const c_void);
        enc.set_bytes(10, 4, &nwg as *const u32 as *const c_void);
        enc.set_bytes(11, 4, &npsel as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_head as u64, nwg as u64, 1), MTLSize::new(256, 1, 1));
        self.bar(enc);
        enc.set_compute_pipeline_state(&self.p["attention_merge"]);
        enc.set_buffer(0, Some(&self.st.attn_part), 0);
        enc.set_buffer(1, Some(&self.st.attn), 0);
        enc.set_bytes(2, 4, &hd as *const u32 as *const c_void);
        enc.set_bytes(3, 4, &nwg as *const u32 as *const c_void);
        enc.dispatch_thread_groups(MTLSize::new(n_head as u64, 1, 1), MTLSize::new(256, 1, 1));
    }

}

//! CPU-only PLE table mapping. Only selected packed rows may enter Metal.

use anyhow::{ensure, Context, Result};
use ojas_formats::gguf::Gguf;
use std::{ffi::c_void, fs::File, os::fd::AsRawFd};

pub(crate) const TABLE: &str = "per_layer_token_embd.weight";

pub(crate) struct PleTable {
    mapping: *mut c_void,
    mapping_len: usize,
    data_offset: usize,
    pub(crate) row_bytes: usize,
    pub(crate) rows: u64,
    pub(crate) format: u32,
}

// Moving ownership between model workers is safe: the mapping is read-only,
// has no thread affinity, and remains live until its owner is dropped.
unsafe impl Send for PleTable {}

impl PleTable {
    pub(crate) fn from_gguf(g: &Gguf) -> Result<Self> {
        let info = g.tensors.get(TABLE).context("missing PLE table")?;
        ensure!(info.dims.len() == 2, "PLE table must have two dimensions");
        let (part, offset, bytes, format) = g.tensor_meta(TABLE).context("missing PLE tensor extent")?;
        let files = g.shard_files()?;
        let table = Self::map(&files[part], offset, info.dims[0], info.dims[1], format)?;
        ensure!(table.rows.checked_mul(table.row_bytes as u64) == Some(bytes), "PLE tensor extent disagrees with its shape");
        ensure!(g.meta_u32("qwen4exp.embedding_length_per_layer_input").map(u64::from) == Some(info.dims[0]),
            "PLE table width disagrees with head dimension");
        let ngram = g.meta_u32("qwen4exp.ple.ngram_size").unwrap_or(0);
        let per = g.meta_u32("qwen4exp.ple.heads_per_ngram").unwrap_or(0);
        ensure!(ngram >= 2 && per > 0, "invalid PLE n-gram geometry");
        let heads = (ngram - 1).checked_mul(per).context("PLE head count overflow")? as usize;
        ensure!(info.dims[0].checked_mul(heads as u64) == g.meta_u32("qwen4exp.embedding_length").map(u64::from),
            "PLE heads do not span the model embedding");
        let multipliers = g.int_arr("qwen4exp.ple.layer_multipliers").context("missing PLE multipliers")?;
        let offsets = g.int_arr("qwen4exp.ple.head_offsets").context("missing PLE offsets")?;
        let sizes = g.int_arr("qwen4exp.ple.head_vocab_sizes").context("missing PLE vocabulary sizes")?;
        ensure!(multipliers.len() == ngram as usize && offsets.len() == heads && sizes.len() == heads,
            "PLE metadata array lengths disagree with head count");
        for (&offset, &size) in offsets.iter().zip(sizes) {
            ensure!(offset >= 0 && size > 0 && (offset as u64).checked_add(size as u64).is_some_and(|end| end <= table.rows),
                "PLE head range exceeds the table");
        }
        Ok(table)
    }

    fn map(file: &File, offset: u64, width: u64, rows: u64, format: u32) -> Result<Self> {
        ensure!(width > 0 && rows > 0 && rows <= i32::MAX as u64, "invalid PLE table dimensions");
        let row_bytes = match format {
            0 => width.checked_mul(4),
            20 => { ensure!(width % 32 == 0, "IQ4_NL PLE width must be a multiple of 32"); (width / 32).checked_mul(18) }
            _ => anyhow::bail!("unsupported PLE table format {format}; expected F32 or IQ4_NL"),
        }.context("PLE row size overflow")?;
        let bytes = rows.checked_mul(row_bytes).context("PLE table size overflow")?;
        ensure!(offset.checked_add(bytes).is_some_and(|end| end <= file.metadata().map(|m| m.len()).unwrap_or(0)),
            "PLE table exceeds the shard file");
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        ensure!(page > 0, "cannot determine mmap page size");
        let aligned = offset / page as u64 * page as u64;
        let data_offset = usize::try_from(offset - aligned)?;
        let mapping_len = usize::try_from(bytes)?.checked_add(data_offset).context("PLE mapping size overflow")?;
        let mapping = unsafe {
            libc::mmap(std::ptr::null_mut(), mapping_len, libc::PROT_READ, libc::MAP_SHARED,
                file.as_raw_fd(), i64::try_from(aligned)?)
        };
        ensure!(mapping != libc::MAP_FAILED, "PLE mmap failed: {}", std::io::Error::last_os_error());
        // No Metal buffer, mlock, or whole-table prefetch: faults are limited to
        // the CPU rows actually copied below. The mapping owns its lifetime.
        Ok(Self { mapping, mapping_len, data_offset, row_bytes: usize::try_from(row_bytes)?, rows, format })
    }

    pub(crate) fn copy_rows(&self, rows: &[u32], dst: &mut [u8]) -> Result<()> {
        ensure!(rows.len().checked_mul(self.row_bytes) == Some(dst.len()), "PLE staging length mismatch");
        // Validate everything before writing any output.
        ensure!(rows.iter().all(|&row| u64::from(row) < self.rows), "PLE row outside table");
        for (&row, out) in rows.iter().zip(dst.chunks_exact_mut(self.row_bytes)) {
            let offset = self.data_offset + row as usize * self.row_bytes;
            // map() proved that every complete row lies within this mapping.
            let src = unsafe { std::slice::from_raw_parts((self.mapping as *const u8).add(offset), self.row_bytes) };
            out.copy_from_slice(src);
        }
        Ok(())
    }
}

impl Drop for PleTable {
    fn drop(&mut self) { unsafe { libc::munmap(self.mapping, self.mapping_len); } }
}

impl super::Qwen4ExpConfig {
    pub(crate) fn ple_rows(&self, token: u32, pos: usize, history: &[u32]) -> Vec<u32> {
        let ngram = self.ple_ngram_size as usize;
        let per = self.ple_heads_per_ngram as usize;
        let mut context = vec![self.ple_eos; ngram];
        context[0] = token;
        let mut cut = false;
        for (s, prev) in context.iter_mut().enumerate().skip(1) {
            if !cut { *prev = pos.checked_sub(s).and_then(|p| history.get(p)).copied().unwrap_or(self.ple_eos); }
            if *prev == self.ple_eos { cut = true; }
        }
        let mut rows = vec![0; self.ple_n_heads as usize];
        for n in 2..=ngram {
            let mut mixed = u64::from(token).wrapping_mul(self.ple_multipliers[0]);
            for (j, &prev) in context.iter().enumerate().take(n).skip(1) {
                mixed ^= u64::from(prev).wrapping_mul(self.ple_multipliers[j]);
            }
            for h in (n - 2) * per..(n - 1) * per {
                rows[h] = (mixed % u64::from(self.ple_head_vocab_sizes[h]) + u64::from(self.ple_head_offsets[h])) as u32;
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::OpenOptions, os::unix::fs::FileExt, sync::atomic::{AtomicU64, Ordering}};

    fn file() -> File {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!("ojas-ple-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let f = OpenOptions::new().read(true).write(true).create_new(true).open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        f
    }

    #[test]
    fn sparse_mapping_handles_unaligned_start_and_offsets_past_four_gib() {
        let f = file();
        let (offset, width, rows) = (37, 160, 50_000_000);
        let row_bytes = width / 32 * 18;
        f.set_len(offset + rows * row_bytes).unwrap();
        let first = vec![17; row_bytes as usize];
        let last: Vec<u8> = (0..row_bytes).map(|i| i as u8).collect();
        f.write_all_at(&first, offset).unwrap();
        f.write_all_at(&last, offset + (rows - 1) * row_bytes).unwrap();
        let table = PleTable::map(&f, offset, width, rows, 20).unwrap();
        let mut out = vec![0; 3 * row_bytes as usize];
        table.copy_rows(&[(rows - 1) as u32, 0, (rows - 1) as u32], &mut out).unwrap();
        assert_eq!(out, [last.clone(), first, last].concat());
        let before = out.clone();
        assert!(table.copy_rows(&[0, rows as u32, 0], &mut out).is_err());
        assert_eq!(out, before, "invalid rows must not partially overwrite staging");
        assert!(table.copy_rows(&[0], &mut out).is_err());
        assert!(PleTable::map(&f, offset, 161, rows, 20).is_err());
        assert!(PleTable::map(&f, offset, width, rows + 1, 20).is_err());
        assert!(PleTable::map(&f, offset, width, rows, 8).is_err());
    }

    fn config() -> super::super::Qwen4ExpConfig {
        super::super::Qwen4ExpConfig {
            hc_mult: 4, hc_low_rank: 320, indexer: super::super::IndexerConfig { n_head: 0, head_size: 0, top_k: 0 },
            ple_ngram_size: 3, ple_heads_per_ngram: 2, ple_conv_kernel: 4,
            ple_head_dim: 32, ple_layers: vec![1], ple_n_heads: 4, ple_eos: 99,
            ple_multipliers: vec![7, 11, 13], ple_head_offsets: vec![0, 100, 200, 300],
            ple_head_vocab_sizes: vec![97, 89, 83, 79],
        }
    }

    #[test]
    fn hash_preserves_history_boundaries_and_current_token_override() {
        let q = config();
        let expected = |token: u64, prev: u64, older: u64| {
            let bi = token * 7 ^ prev * 11;
            let tri = bi ^ older * 13;
            vec![(bi % 97) as u32, (100 + bi % 89) as u32, (200 + tri % 83) as u32, (300 + tri % 79) as u32]
        };
        assert_eq!(q.ple_rows(5, 0, &[]), expected(5, 99, 99));
        assert_eq!(q.ple_rows(5, 1, &[3]), expected(5, 3, 99));
        assert_eq!(q.ple_rows(5, 2, &[2, 3, 77]), expected(5, 3, 2));
        assert_eq!(q.ple_rows(5, 2, &[2, 99]), expected(5, 99, 99));
        // EOS at the CURRENT position does not erase the preceding context.
        assert_eq!(q.ple_rows(99, 2, &[2, 3]), expected(99, 3, 2));
        assert_eq!(q.ple_rows(5, 4, &[]), expected(5, 99, 99));
    }

    #[test]
    #[ignore = "requires Metal; run explicitly after CPU tests"]
    fn staged_gpu_gather_matches_original_for_every_batch_row() -> Result<()> {
        use metal::{MTLResourceOptions, MTLSize};
        let gpu = ojas_metal::MetalGpu::new()?;
        let source = format!("{}\nusing namespace metal;\n{}", ojas_metal::kernels::nat::metal_decoders(), ojas_metal::kernels::qwen4exp::BODY);
        let upload = |bytes: &[u8]| gpu.device.new_buffer_with_data(bytes.as_ptr() as *const c_void, bytes.len() as u64, MTLResourceOptions::StorageModeShared);
        let (hd, nh, nrows) = (160u32, 16u32, 257usize);
        for format in [0, 20] {
            let kernel = gpu.compile(&source, if format == 0 { "ple_gather" } else { "ple_gather_iq4nl" })?;
            let row_bytes = if format == 0 { hd as usize * 4 } else { hd as usize / 32 * 18 };
            let mut bytes = vec![0; nrows * row_bytes];
            if format == 0 {
                for (i, dst) in bytes.chunks_exact_mut(4).enumerate() { dst.copy_from_slice(&((i as f32 - 4000.0) / 128.0).to_le_bytes()); }
            } else {
                for (i, block) in bytes.chunks_exact_mut(18).enumerate() {
                    let scale = half::f16::from_f32((i % 31 + 1) as f32 / 128.0).to_bits();
                    block[..2].copy_from_slice(&scale.to_le_bytes());
                    for (j, b) in block[2..].iter_mut().enumerate() { *b = (i * 37 + j * 19) as u8; }
                }
            }
            let f = file(); f.write_all_at(&bytes, 37)?;
            let table = PleTable::map(&f, 37, hd as u64, nrows as u64, format)?;
            let whole = upload(&bytes); // synthetic table is at most 165 KB
            let stride = row_bytes * nh as usize;
            let staging = gpu.device.new_buffer((super::super::MAXM * stride) as u64, MTLResourceOptions::StorageModeShared);
            let elements = hd as usize * nh as usize;
            let out = gpu.device.new_buffer((2 * super::super::MAXM * elements * 4) as u64, MTLResourceOptions::StorageModeShared);
            let linear: Vec<u32> = (0..nh).collect();
            let cb = gpu.command_buffer(); let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kernel);
            for batch_row in 0..super::super::MAXM {
                let rows: Vec<u32> = (0..nh).map(|h| ((batch_row * 13 + h as usize * 7) % nrows) as u32).collect();
                let dst = unsafe { std::slice::from_raw_parts_mut((staging.contents() as *mut u8).add(batch_row * stride), stride) };
                table.copy_rows(&rows, dst)?;
                for (variant, indices) in [&rows, &linear].into_iter().enumerate() {
                    enc.set_buffer(0, Some(if variant == 0 { &whole } else { &staging }), if variant == 0 { 0 } else { (batch_row * stride) as u64 });
                    enc.set_bytes(1, nh as u64 * 4, indices.as_ptr() as *const c_void);
                    enc.set_buffer(2, Some(&out), ((2 * batch_row + variant) * elements * 4) as u64);
                    enc.set_bytes(3, 4, &hd as *const u32 as *const c_void);
                    enc.set_bytes(4, 4, &nh as *const u32 as *const c_void);
                    enc.dispatch_thread_groups(MTLSize::new((elements as u64).div_ceil(64), 1, 1), MTLSize::new(64, 1, 1));
                }
            }
            enc.end_encoding(); ojas_metal::commit_and_wait_checked(cb, "PLE staging parity")?;
            let values = unsafe { std::slice::from_raw_parts(out.contents() as *const u32, 2 * super::super::MAXM * elements) };
            for row in 0..super::super::MAXM {
                let base = row * 2 * elements;
                assert_eq!(&values[base..base + elements], &values[base + elements..base + 2 * elements], "format {format}, batch row {row}");
            }
            assert_ne!(&values[..elements], &values[2 * elements..3 * elements], "test rows must differ");
        }
        Ok(())
    }
}

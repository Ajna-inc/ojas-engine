//! Backend selection: Metal where it exists, CUDA where it is built (`--features cuda`) and the
//! architecture has a CUDA decoder, CPU everywhere.
//!
//! Both backends implement `ojas_core::Model`, so callers above this module work
//! against `&dyn Model` without knowing which one they got.
//!
//! The model is handed to a callback rather than returned: `DecoderGpu<'a>`
//! borrows the `MetalGpu` it was built on, so the device must outlive the model.
//! Scoping both inside one call avoids a self-referential struct.

use anyhow::{bail, Context, Result};
use ojas_core::Model;
use ojas_formats::gguf::Gguf;
use ojas_tokenize::Bpe;

/// Which backend to run on. `Auto` prefers Metal (macOS) or CUDA (built with the `cuda`
/// feature, for the architectures in [`CUDA_ARCHS`]) and falls back to CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Device {
    #[default]
    Auto,
    Metal,
    Cpu,
    /// NVIDIA via `ojas-cuda`. Only parses in a build with the `cuda` feature.
    Cuda,
}

impl Device {
    pub fn parse(s: &str) -> Result<Device> {
        Ok(match s {
            "auto" => Device::Auto,
            "metal" | "gpu" => Device::Metal,
            "cpu" => Device::Cpu,
            #[cfg(feature = "cuda")]
            "cuda" => Device::Cuda,
            #[cfg(feature = "cuda")]
            other => bail!("unknown device {other:?} (want auto, metal, cuda or cpu)"),
            #[cfg(not(feature = "cuda"))]
            other => bail!("unknown device {other:?} (want auto, metal or cpu; cuda needs a build with --features cuda)"),
        })
    }
}

/// Loaded-model metadata callers need without reaching into the backend: how to
/// template a chat turn, when to stop, and how much context there is.
pub struct ModelInfo {
    pub arch: String,
    pub eos: Option<u32>,
    /// Every id that ends generation for this model, not just `eos`.
    pub eog: Vec<u32>,
    pub vocab: usize,
    pub context: usize,
    pub backend: &'static str,
    pub has_mtp: bool,
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub load_secs: f64,
}

/// Architectures the CPU decoders accept. Kept beside `load_cpu` so the list and
/// the dispatch cannot drift apart.
pub const CPU_ARCHS: &[&str] = &["qwen2", "qwen3", "llama", "glm-dsa", "deepseek2", "qwen35"];

/// Architectures the CUDA decoders accept (`ojas_cuda::CudaSsm`: qwen35 / surya-2, F16).
pub const CUDA_ARCHS: &[&str] = &["qwen35"];

/// Load the CUDA decoder for `arch`, with the vision tower attached when an mmproj sits beside
/// the model (or `--mmproj` / `OJAS_MMPROJ` names one). A projector that cannot be used is a
/// warning, not an error: text commands do not need it, and `ojas ocr` reports a model without
/// a tower on its own terms.
#[cfg(feature = "cuda")]
fn load_cuda(path: &str, g: &mut Gguf, arch: &str, context: usize) -> Result<ojas_cuda::CudaSsm> {
    anyhow::ensure!(CUDA_ARCHS.contains(&arch),
        "no CUDA decoder for architecture {arch:?} (CUDA supports: {})", CUDA_ARCHS.join(", "));
    let mut m = ojas_cuda::CudaSsm::load(g, context)?;
    let explicit = ojas_core::config::EngineConfig::current().mmproj;
    let attach = (|| -> Result<Option<std::path::PathBuf>> {
        let Some(mm) = ojas_formats::mmproj::discover(std::path::Path::new(path), explicit.as_deref())? else {
            return Ok(None);
        };
        let mut mg = Gguf::open(mm.to_string_lossy().as_ref())?;
        ojas_formats::mmproj::validate(g, &mg)?;
        m.attach_vit_gguf(&mut mg)?;
        Ok(Some(mm))
    })();
    match attach {
        Ok(Some(mm)) => tracing::info!("CUDA vision tower attached from {}", mm.display()),
        Ok(None) => {}
        Err(e) => tracing::warn!("CUDA decoder loaded without its vision tower: {e:#}"),
    }
    Ok(m)
}

fn load_cpu(g: &mut Gguf, arch: &str) -> Result<Box<dyn Model>> {
    Ok(match arch {
        "qwen2" | "qwen3" | "llama" => Box::new(ojas_cpu::CpuQwen::load(g)?),
        "glm-dsa" => Box::new(ojas_cpu::CpuGlm::load(g)?),
        "deepseek2" => Box::new(ojas_cpu::CpuDeepseek::load(g)?),
        "qwen35" => Box::new(ojas_cpu::CpuSsm::load(g)?),
        other => bail!(
            "no CPU decoder for architecture {other:?} (CPU supports: {})",
            CPU_ARCHS.join(", ")
        ),
    })
}

/// Load `path` and run `f` with the model, its tokenizer and its metadata.
pub fn with_model<R>(
    path: &str,
    device: Device,
    context: usize,
    precision: u8,
    f: impl FnOnce(&dyn Model, &Bpe, &ModelInfo) -> Result<R>,
) -> Result<R> {
    let mut g = Gguf::open(path).with_context(|| format!("opening {path}"))?;
    let arch = g.arch();
    let bpe = Bpe::from_gguf(&g);
    let eos = g.meta_u32("tokenizer.ggml.eos_token_id");
    let eog = ojas_tokenize::eog_token_ids(&g, &arch);
    let vocab = g.str_arr("tokenizer.ggml.tokens").map(|v| v.len()).unwrap_or(0);
    let started = std::time::Instant::now();

    let mut info = ModelInfo {
        arch: arch.clone(),
        eos,
        eog,
        vocab,
        context,
        backend: "cpu",
        has_mtp: false,
        n_layers: 0,
        hidden_dim: 0,
        load_secs: 0.0,
    };

    #[cfg(target_os = "macos")]
    if matches!(device, Device::Auto | Device::Metal) {
        // Two separate failures live here: no Metal device at all, and a device
        // that cannot run THIS architecture. Only the second is worth falling
        // back from silently, and an explicit `--device metal` should never fall
        // back at all — being quietly downgraded to a CPU decoder that is orders
        // of magnitude slower is worse than an error.
        match ojas_metal::MetalGpu::new() {
            Ok(gpu) => match ojas_models::decoder::DecoderGpu::load(
                &gpu, &mut g, context, precision, None, None,
            ) {
                Ok(m) => {
                    info.backend = "metal";
                    info.has_mtp = m.has_mtp();
                    info.n_layers = m.n_layers();
                    info.hidden_dim = m.hidden_dim();
                    info.context = m.context_capacity();
                    info.load_secs = started.elapsed().as_secs_f64();
                    return f(&m, &bpe, &info);
                }
                Err(e) if device == Device::Metal => {
                    return Err(e).with_context(|| {
                        format!("--device metal: Metal cannot run architecture {arch:?}")
                    })
                }
                Err(e) => tracing::warn!("Metal load failed ({e:#}); falling back to CPU"),
            },
            Err(e) if device == Device::Metal => {
                return Err(anyhow::anyhow!(e)).context("--device metal: no usable Metal device")
            }
            Err(e) => tracing::warn!("no Metal device ({e:#}); using CPU"),
        }
    }

    #[cfg(not(target_os = "macos"))]
    if device == Device::Metal {
        bail!("--device metal is not available on this platform (built without the Metal backend)");
    }

    // CUDA: explicit, or `auto` for an architecture it has a decoder for. As with Metal, an
    // explicit `--device cuda` never falls back to the CPU.
    #[cfg(feature = "cuda")]
    if device == Device::Cuda || (device == Device::Auto && CUDA_ARCHS.contains(&arch.as_str())) {
        match load_cuda(path, &mut g, &arch, context) {
            Ok(m) => {
                info.backend = "cuda";
                info.n_layers = m.n_layers();
                info.hidden_dim = m.hidden_dim();
                info.context = m.context_capacity();
                info.load_secs = started.elapsed().as_secs_f64();
                return f(&m, &bpe, &info);
            }
            Err(e) if device == Device::Cuda => return Err(e).context("--device cuda"),
            Err(e) => tracing::warn!("CUDA load failed ({e:#}); falling back to CPU"),
        }
    }
    // Precision selects a Metal decoder tier; the CPU decoders have one path.
    // Named rather than dropped so the flag stays accepted everywhere and a
    // script written on a Mac still runs elsewhere.
    #[cfg(not(target_os = "macos"))]
    let _ = precision;

    let m = load_cpu(&mut g, &arch)?;
    info.backend = "cpu";
    info.has_mtp = m.has_mtp();
    info.n_layers = m.n_layers();
    info.hidden_dim = m.hidden_dim();
    // CPU decoders preallocate no positional arena, so their capacity is
    // unbounded. Report the requested context rather than usize::MAX.
    let cap = m.context_capacity();
    info.context = if cap == usize::MAX { context } else { cap };
    info.load_secs = started.elapsed().as_secs_f64();
    f(m.as_ref(), &bpe, &info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parses_the_documented_spellings() {
        assert_eq!(Device::parse("auto").unwrap(), Device::Auto);
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("metal").unwrap(), Device::Metal);
        assert_eq!(Device::parse("gpu").unwrap(), Device::Metal);
        #[cfg(feature = "cuda")]
        assert_eq!(Device::parse("cuda").unwrap(), Device::Cuda);
        #[cfg(not(feature = "cuda"))]
        assert!(Device::parse("cuda").is_err(), "unbuilt backends must not parse");
        assert!(Device::parse("vulkan").is_err(), "unbuilt backends must not parse");
    }

    #[test]
    fn unknown_device_error_lists_the_choices() {
        let e = Device::parse("tpu").unwrap_err().to_string();
        assert!(e.contains("tpu") && e.contains("cpu"), "error should guide: {e}");
    }

    /// Keeps the advertised list and the dispatch match from drifting apart.
    #[test]
    fn advertised_cpu_archs_are_the_ones_dispatch_accepts() {
        for a in CPU_ARCHS {
            assert!(
                matches!(*a, "qwen2" | "qwen3" | "llama" | "glm-dsa" | "deepseek2" | "qwen35"),
                "{a} is advertised but load_cpu has no arm for it"
            );
        }
    }
}

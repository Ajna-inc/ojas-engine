//! Command-line parsing, in two passes over one argv.
//!
//! Pass one takes the engine flags, which configure the load and must be
//! installed before any model is touched; pass two takes the runtime flags
//! (sampling, device, serving address). What survives both is the subcommand and
//! its positionals.
//!
//! Only the second pass rejects unknown flags: the first must tolerate the
//! second's, or the order flags are typed in would matter. Names follow
//! llama.cpp wherever one exists.

use crate::backend::Device;
use anyhow::{Context, Result};
use ojas_core::config::EngineConfig;
use ojas_infer::SampleOpts;

/// Everything that is not engine configuration: what to generate and how.
#[derive(Debug, Clone)]
pub struct RunOpts {
    pub device: Device,
    /// Metal decoder tier; `None` lets the loader choose for the file: 3 for dense
    /// decoders, 4 for models with experts.
    pub precision: Option<u8>,
    pub n_predict: usize,
    pub prompt: Option<String>,
    pub prompt_file: Option<String>,
    pub system: String,
    /// Feed the prompt to the model verbatim, with no chat template around it.
    pub raw: bool,
    pub sample: SampleOpts,
    /// Stop strings (`--stop`, repeatable): generation ends before the first one
    /// and it is not printed.
    pub stop: Vec<String>,
    /// Constrained output: any JSON object, JSON matching a schema, or a GBNF
    /// grammar. At most one.
    pub format: Option<ojas_grammar::OutputFormat>,
    pub host: String,
    pub port: u16,
    /// Repetitions for `bench`.
    pub reps: usize,
    /// Vision (`detect`): score threshold; None = the command's default.
    pub conf: Option<f32>,
    /// Vision (`detect`): NMS IoU threshold; None = the command's default.
    pub iou: Option<f32>,
    /// `detect` and `decide`: emit JSON instead of a table.
    pub json: bool,
    /// `decide`: the state, as JSON (an object or array) or plain text, inline or
    /// from a file.
    pub state: Option<String>,
    pub state_file: Option<String>,
    /// `decide`: the questions, a JSON object of `{id: {type, instructions, criteria}}`,
    /// inline or from a file.
    pub questions: Option<String>,
    pub questions_file: Option<String>,
    /// Vision (`plate`): OCR input normalization — "signed" (default) or
    /// "unit" for exports with normalization folded in-graph (see MANIFEST).
    pub ocr_norm: Option<String>,
    /// `vision-serve`: bearer token, or the file holding one; callers from a
    /// `--trust` range need neither.
    pub token: Option<String>,
    pub token_file: Option<String>,
    /// `vision-serve`: comma-separated IPs/CIDRs served without a token.
    pub trust: Option<String>,
    /// `vision-serve`: which device runs the model — "cpu" (default),
    /// "cuda[:N]" or "vulkan[:N]". Separate from --device, which selects an
    /// LLM backend.
    pub vision_device: Option<String>,
    /// True once --port is seen: `vision-serve` has its own default port and must
    /// not inherit the LLM server's 8080, which the vision gateway uses.
    pub port_set: bool,
    /// True once any sampling flag is seen; only `ocr` consults it. Transcription
    /// must default to greedy — a document legitimately repeats `<div data-bbox=`,
    /// which the default repeat_penalty of 1.1 fights — while an explicit --temp
    /// still wins.
    pub sample_set: bool,
}

impl Default for RunOpts {
    fn default() -> Self {
        RunOpts {
            device: Device::Auto,
            precision: None,
            n_predict: 128,
            prompt: None,
            prompt_file: None,
            system: String::new(),
            raw: false,
            stop: Vec::new(),
            format: None,
            token: None,
            token_file: None,
            trust: None,
            vision_device: None,
            port_set: false,
            // llama.cpp's defaults, so the same flags give comparable output.
            sample: SampleOpts { temperature: 0.8, top_p: 0.95, top_k: 40, repeat_penalty: 1.1, repeat_window: 64, seed: 42 },
            host: "127.0.0.1".into(),
            port: 8080,
            reps: 3,
            conf: None,
            iou: None,
            json: false,
            state: None,
            state_file: None,
            questions: None,
            questions_file: None,
            ocr_norm: None,
            sample_set: false,
        }
    }
}

impl RunOpts {
    /// Sampling options, or `None` for the greedy on-device argmax path: exact, and
    /// the only path speculative decoding runs on. Passing the sampler a
    /// temperature of zero instead would copy logits to reach the same answer.
    pub fn sampling(&self) -> Option<&SampleOpts> {
        if self.sample.temperature <= 0.0 { None } else { Some(&self.sample) }
    }

    /// `sampling()` with a greedy default instead of llama.cpp's 0.8. For
    /// transcription, sampling invents plausible text that was never on the page
    /// and the default repeat_penalty punishes the markup the task emits. An
    /// explicit sampling flag still wins.
    pub fn sampling_greedy_default(&self) -> Option<&SampleOpts> {
        if self.sample_set { self.sampling() } else { None }
    }

    /// The prompt text, from `-p`, or `-f`, or the trailing positional.
    pub fn resolve_prompt(&self, positional: Option<&str>) -> Result<String> {
        if let Some(f) = &self.prompt_file {
            return std::fs::read_to_string(f).with_context(|| format!("reading prompt file {f}"));
        }
        if let Some(p) = &self.prompt {
            return Ok(p.clone());
        }
        positional
            .map(str::to_string)
            .context("no prompt: pass one positionally, with -p TEXT, or with -f FILE")
    }
}

/// Pull engine flags out of `args` and fold them into a config.
///
/// Precedence is flag > env > default. Every boolean has both polarities, so a
/// flag can turn off what the environment turned on.
///
/// Flag names (llama.cpp's where one exists) and the env vars they override:
///   -c/--ctx-size N     context length            (OJAS_CTX)
///   --mmap/--no-mmap    zero-copy weight mapping  (OJAS_NO_MMAP)
///   --mlock/--no-mlock  pin the resident skeleton (OJAS_NO_SKEL_LOCK)
///   --warmup/--no-warmup  touch pages at load     (OJAS_PREWARM)
///   -cram/--cache-ram N   cache budget, MiB       (OJAS_EXPERT_CACHE_GB)
///   -b/--batch-size N     prefill chunk, tokens    (OJAS_PREFILL_M)
///   -ub/--ubatch-size N   physical pass, tokens    (OJAS_UBATCH)
///   -md/--spec-draft-model FNAME  MTP draft head    (OJAS_MTP)
///   --mmproj FNAME        vision projector GGUF    (OJAS_MMPROJ)
///   --mixed-q4 MODE       Qwen3.5 mixed-Q4 preset
///   --q4-head/--no-q4-head  Q4 output head          (OJAS_Q4_HEAD)
///   --q4-ffn-down-last N  Q4 last N FFN-downs       (OJAS_Q4_FFN_DOWN_LAST)
///   --q4-ffn-down/--no-q4-ffn-down  all/off          (OJAS_Q4_FFN_DOWN)
///
/// `--mmap`/`--mlock` are deprecated upstream in favour of `--load-mode MODE`,
/// but they are what is in circulation, and ojas's two booleans do not map onto
/// that mode enum.
///
/// `--top-k` is sampling top-k, as in llama.cpp. MoE expert-routing top-K is a
/// different quantity, spelled `--top-k-experts`, so neither can stand in for the
/// other.
pub fn take_engine_flags(args: &mut Vec<String>) -> Result<EngineConfig> {
    let mut cfg = EngineConfig::from_env();
    let mut rest: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].clone();
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i).cloned().with_context(|| format!("{a} needs a value"))
        };
        match a.as_str() {
            "-c" | "--ctx-size" => {
                let v = take(&mut i)?;
                cfg.ctx = Some(v.parse().with_context(|| format!("{a}: bad integer {v:?}"))?);
            }
            // MiB in, GB stored — llama.cpp's unit, so numbers transfer.
            "-cram" | "--cache-ram" => {
                let v = take(&mut i)?;
                let mib: f64 = v.parse().with_context(|| format!("{a}: bad number {v:?}"))?;
                cfg.expert_cache_gb = mib / 1024.0;
            }
            "-b" | "--batch-size" => {
                let v = take(&mut i)?;
                cfg.prefill_m = v.parse::<usize>().with_context(|| format!("{a}: bad integer {v:?}"))?.max(1);
            }
            // Physical pass size. On a streamed MoE model this sizes the packed
            // expert-gather scratch, so it is a memory knob as much as a speed one.
            "-ub" | "--ubatch-size" => {
                let v = take(&mut i)?;
                cfg.ubatch = v.parse::<usize>().with_context(|| format!("{a}: bad integer {v:?}"))?.max(1);
            }
            "--mmap" => cfg.no_mmap = false,
            "--no-mmap" => cfg.no_mmap = true,
            "--mlock" => cfg.no_skel_lock = false,
            "--no-mlock" => cfg.no_skel_lock = true,
            "--warmup" => cfg.prewarm = true,
            "--no-warmup" => cfg.prewarm = false,
            "-md" | "--model-draft" | "--spec-draft-model" => cfg.mtp = Some(take(&mut i)?),
            // llama.cpp's name for the same file. An engine flag, not a runtime
            // one: only the loader acts on it, and every command that loads a
            // multimodal model needs it, not just one subcommand.
            "--mmproj" => cfg.mmproj = Some(take(&mut i)?),
            "--top-k-experts" => {
                let v = take(&mut i)?;
                cfg.top_k = Some(v.parse().with_context(|| format!("{a}: bad integer {v:?}"))?);
            }
            // Presets for the measured mixed-precision modes. They are load-time
            // options, so the engine pass makes them behave identically for
            // run/chat/serve/bench. Later flags override earlier ones and the
            // environment, which allows `--mixed-q4 balanced --no-q4-head` for a
            // custom point.
            "--mixed-q4" => {
                let mode = take(&mut i)?;
                match mode.as_str() {
                    "off" | "none" => {
                        cfg.q4_head = false;
                        cfg.q4_ffn_down = false;
                        cfg.q4_ffn_down_last = 0;
                    }
                    "head" => {
                        cfg.q4_head = true;
                        cfg.q4_ffn_down = false;
                        cfg.q4_ffn_down_last = 0;
                    }
                    "balanced" => {
                        cfg.q4_head = true;
                        cfg.q4_ffn_down = false;
                        cfg.q4_ffn_down_last = 12;
                    }
                    "aggressive" => {
                        cfg.q4_head = true;
                        cfg.q4_ffn_down = true;
                        cfg.q4_ffn_down_last = 0;
                    }
                    _ => anyhow::bail!(
                        "{a}: unknown mode {mode:?}; expected off, head, balanced, or aggressive"
                    ),
                }
            }
            "--q4-head" => cfg.q4_head = true,
            "--no-q4-head" => cfg.q4_head = false,
            "--q4-ffn-down-last" => {
                let v = take(&mut i)?;
                cfg.q4_ffn_down_last =
                    v.parse().with_context(|| format!("{a}: bad integer {v:?}"))?;
                cfg.q4_ffn_down = false;
            }
            "--q4-ffn-down" => {
                cfg.q4_ffn_down = true;
                cfg.q4_ffn_down_last = 0;
            }
            "--no-q4-ffn-down" => {
                cfg.q4_ffn_down = false;
                cfg.q4_ffn_down_last = 0;
            }
            "--moe-dbuf" => cfg.moe_dbuf = true,
            "--no-moe-dbuf" => cfg.moe_dbuf = false,
            // `--resident` requests main-layer expert residency; PLE stays CPU-only.
            // `--stream` bounds expert memory with a cache. Auto/resident falls
            // back to streaming when GPU, physical, or live RAM budgets fail.
            "--resident" | "--full-resident" => cfg.flash_resident = Some(true),
            "--stream" | "--low-mem" => cfg.flash_resident = Some(false),
            // Not an engine flag; leave it for the runtime pass.
            _ => rest.push(a),
        }
        i += 1;
    }
    *args = rest;
    Ok(cfg)
}

/// Pull runtime flags out of what the engine pass left behind. This is the pass
/// that rejects unknown flags, so it must run last.
pub fn take_run_flags(args: &mut Vec<String>) -> Result<RunOpts> {
    let mut o = RunOpts::default();
    let mut repeat_penalty_set = false;
    let mut rest: Vec<String> = Vec::with_capacity(args.len());
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].clone();
        let take = |i: &mut usize| -> Result<String> {
            *i += 1;
            args.get(*i).cloned().with_context(|| format!("{a} needs a value"))
        };
        macro_rules! num {
            ($i:expr) => {{
                let v = take($i)?;
                v.parse().with_context(|| format!("{a}: bad number {v:?}"))?
            }};
        }
        match a.as_str() {
            "-n" | "--predict" | "--n-predict" => o.n_predict = num!(&mut i),
            "-p" | "--prompt" => o.prompt = Some(take(&mut i)?),
            "-f" | "--file" => o.prompt_file = Some(take(&mut i)?),
            "-sys" | "--system" | "--system-prompt" => o.system = take(&mut i)?,
            "--raw" | "--no-template" => o.raw = true,
            "--stop" => o.stop.push(take(&mut i)?),
            "--json-object" | "--json-schema" | "--json-schema-file" | "--grammar" | "--grammar-file" => {
                use ojas_grammar::OutputFormat;
                if o.format.is_some() {
                    anyhow::bail!("give at most one of --json-object, --json-schema(-file), --grammar(-file)");
                }
                let file = |path: String| std::fs::read_to_string(&path).with_context(|| format!("{a}: reading {path}"));
                let schema = |text: String| -> Result<serde_json::Value> {
                    serde_json::from_str(&text).with_context(|| format!("{a}: not valid JSON"))
                };
                o.format = Some(match a.as_str() {
                    "--json-object" => OutputFormat::JsonObject,
                    "--json-schema" => OutputFormat::JsonSchema(schema(take(&mut i)?)?),
                    "--json-schema-file" => OutputFormat::JsonSchema(schema(file(take(&mut i)?)?)?),
                    "--grammar" => OutputFormat::Grammar(take(&mut i)?),
                    _ => OutputFormat::Grammar(file(take(&mut i)?)?),
                });
            }
            "--temp" | "--temperature" => { o.sample.temperature = num!(&mut i); o.sample_set = true; }
            "--top-p" => { o.sample.top_p = num!(&mut i); o.sample_set = true; }
            "--top-k" => { o.sample.top_k = num!(&mut i); o.sample_set = true; }
            "--repeat-penalty" => { o.sample.repeat_penalty = num!(&mut i); o.sample_set = true; repeat_penalty_set = true; }
            "--repeat-last-n" => o.sample.repeat_window = num!(&mut i),
            "-s" | "--seed" => o.sample.seed = num!(&mut i),
            "--device" => o.device = Device::parse(&take(&mut i)?)?,
            "--precision" => o.precision = Some(num!(&mut i)),
            "--host" => o.host = take(&mut i)?,
            "--port" => {
                o.port = num!(&mut i);
                o.port_set = true;
            }
            "-r" | "--reps" => o.reps = num!(&mut i),
            "--conf" => o.conf = Some(num!(&mut i)),
            "--iou" => o.iou = Some(num!(&mut i)),
            "--json" => o.json = true,
            "--state" => o.state = Some(take(&mut i)?),
            "--state-file" => o.state_file = Some(take(&mut i)?),
            "--questions" => o.questions = Some(take(&mut i)?),
            "--questions-file" => o.questions_file = Some(take(&mut i)?),
            "--ocr-norm" => o.ocr_norm = Some(take(&mut i)?),
            "--token" => o.token = Some(take(&mut i)?),
            "--token-file" => o.token_file = Some(take(&mut i)?),
            "--trust" => o.trust = Some(take(&mut i)?),
            "--vision-device" => o.vision_device = Some(take(&mut i)?),
            // llama.cpp's offload control. Ojas does not split a model across
            // devices, so the only distinction it can honour is none-vs-some:
            // 0 means run on the CPU, anything else leaves the choice alone.
            "-ngl" | "--n-gpu-layers" | "--gpu-layers" => {
                let n: i64 = num!(&mut i);
                if n == 0 {
                    o.device = Device::Cpu;
                }
            }
            other if other.starts_with('-') && other != "-" => {
                anyhow::bail!(
                    "unknown flag {other}\n\
                     engine: -c/--ctx-size, -cram/--cache-ram, -b/--batch-size, -ub/--ubatch-size,\n\
                     \x20       -md/--spec-draft-model, --mmproj, --top-k-experts, --mmap/--no-mmap,\n\
                     \x20       --mlock/--no-mlock, --warmup/--no-warmup, --moe-dbuf/--no-moe-dbuf,\n\
                     \x20       --mixed-q4, --q4-head/--no-q4-head, --q4-ffn-down-last,\n\
                     \x20       --q4-ffn-down/--no-q4-ffn-down, --resident/--stream\n\
                     runtime: -n/--predict, -p/--prompt, -f/--file, -sys/--system, --raw,\n\
                     \x20        --temp, --top-p, --top-k, --repeat-penalty, --repeat-last-n, -s/--seed,\n\
                     \x20        --device, --precision, -ngl/--n-gpu-layers, --host, --port, -r/--reps,\n\
                     \x20        --conf, --iou, --json (vision)"
                );
            }
            _ => rest.push(a),
        }
        i += 1;
    }
    *args = rest;
    // Structured output repeats its punctuation by design (`":`, `",` on every
    // field), and a repetition penalty pushes the model off exactly those tokens:
    // keys swallow their colons and strings gain stray escapes. Off unless asked for.
    if o.format.is_some() && !repeat_penalty_set {
        o.sample.repeat_penalty = 1.0;
    }
    Ok(o)
}

/// Both passes, in the required order.
pub fn parse(mut argv: Vec<String>) -> Result<(EngineConfig, RunOpts, Vec<String>)> {
    let cfg = take_engine_flags(&mut argv)?;
    let opts = take_run_flags(&mut argv)?;
    Ok((cfg, opts, argv))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(xs: &[&str]) -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn flag_sets_and_is_removed_from_positionals() {
        let mut a = argv(&["ojas", "--moe-dbuf", "run", "model", "0"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert!(cfg.moe_dbuf);
        // The subcommand parse below reads positions, so the flag must be gone.
        assert_eq!(a, argv(&["ojas", "run", "model", "0"]));
    }

    /// Both polarities exist so a user with OJAS_MOE_DBUF exported can turn it
    /// off for a single run.
    #[test]
    fn each_polarity_pins_the_value() {
        let mut on = argv(&["ojas", "--moe-dbuf", "run"]);
        assert!(take_engine_flags(&mut on).unwrap().moe_dbuf);
        let mut off = argv(&["ojas", "--no-moe-dbuf", "run"]);
        assert!(!take_engine_flags(&mut off).unwrap().moe_dbuf);
    }

    #[test]
    fn absent_flag_leaves_the_env_baseline_alone() {
        let mut a = argv(&["ojas", "run", "model"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert_eq!(cfg.moe_dbuf, EngineConfig::from_env().moe_dbuf);
        assert_eq!(a, argv(&["ojas", "run", "model"]));
    }

    #[test]
    fn value_flags_consume_their_argument() {
        let mut a = argv(&["ojas", "-c", "4096", "run", "model"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert_eq!(cfg.ctx, Some(4096));
        assert_eq!(a, argv(&["ojas", "run", "model"]));
    }

    #[test]
    fn cache_ram_is_mib_like_llama_cpp() {
        let mut a = argv(&["ojas", "--cache-ram", "2048", "run"]);
        assert_eq!(take_engine_flags(&mut a).unwrap().expert_cache_gb, 2.0);
    }

    #[test]
    fn mixed_q4_presets_set_the_complete_mode() {
        let cases = [
            ("off", false, false, 0),
            ("head", true, false, 0),
            ("balanced", true, false, 12),
            ("aggressive", true, true, 0),
        ];
        for (mode, head, all_down, last_down) in cases {
            let mut a = argv(&["ojas", "bench", "model.gguf", "--mixed-q4", mode]);
            let cfg = take_engine_flags(&mut a).unwrap();
            assert_eq!((cfg.q4_head, cfg.q4_ffn_down, cfg.q4_ffn_down_last),
                       (head, all_down, last_down), "mode {mode}");
            assert_eq!(a, argv(&["ojas", "bench", "model.gguf"]));
        }
    }

    #[test]
    fn granular_q4_flags_override_presets_in_command_order() {
        let mut a = argv(&[
            "ojas", "--mixed-q4", "aggressive", "--q4-ffn-down-last", "8",
            "--no-q4-head", "run", "model.gguf",
        ]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert!(!cfg.q4_head);
        assert!(!cfg.q4_ffn_down);
        assert_eq!(cfg.q4_ffn_down_last, 8);

        let mut all = argv(&["ojas", "--q4-ffn-down-last", "12", "--q4-ffn-down", "serve"]);
        let cfg = take_engine_flags(&mut all).unwrap();
        assert!(cfg.q4_ffn_down);
        assert_eq!(cfg.q4_ffn_down_last, 0);

        let mut disabled = argv(&["ojas", "--mixed-q4", "aggressive", "--no-q4-ffn-down", "chat"]);
        let cfg = take_engine_flags(&mut disabled).unwrap();
        assert!(!cfg.q4_ffn_down);
        assert_eq!(cfg.q4_ffn_down_last, 0);
    }

    #[test]
    fn mixed_q4_reports_invalid_values() {
        let err = parse(argv(&["ojas", "run", "m.gguf", "--mixed-q4", "fastest"]))
            .unwrap_err().to_string();
        assert!(err.contains("off, head, balanced, or aggressive"), "{err}");

        let err = parse(argv(&["ojas", "--q4-ffn-down-last", "many", "run"]))
            .unwrap_err().to_string();
        assert!(err.contains("--q4-ffn-down-last"), "{err}");
    }

    #[test]
    fn long_and_short_forms_agree() {
        let mut s = argv(&["ojas", "-c", "512"]);
        let mut l = argv(&["ojas", "--ctx-size", "512"]);
        assert_eq!(take_engine_flags(&mut s).unwrap().ctx, take_engine_flags(&mut l).unwrap().ctx);
    }

    #[test]
    fn unknown_flag_is_an_error() {
        let err = parse(argv(&["ojas", "--moe-dbufff", "run"])).unwrap_err().to_string();
        assert!(err.contains("--moe-dbufff"), "error should name the flag: {err}");
    }

    #[test]
    fn draft_model_flag_and_aliases() {
        for f in ["-md", "--model-draft", "--spec-draft-model"] {
            let mut a = argv(&["ojas", f, "/tmp/d.gguf", "run"]);
            assert_eq!(take_engine_flags(&mut a).unwrap().mtp, Some("/tmp/d.gguf".to_string()));
            assert_eq!(a, argv(&["ojas", "run"]));
        }
    }

    /// `--top-k` is sampling, as in llama.cpp; expert routing keeps its own flag.
    #[test]
    fn top_k_is_sampling_and_experts_have_their_own_flag() {
        let (cfg, opts, _) = parse(argv(&["ojas", "--top-k", "40", "--top-k-experts", "6", "run"])).unwrap();
        assert_eq!(opts.sample.top_k, 40, "--top-k must reach the sampler");
        assert_eq!(cfg.top_k, Some(6), "--top-k-experts must reach MoE routing");
    }

    /// llama.cpp's pair: -b is the logical chunk, -ub the physical pass. They are
    /// separate knobs here too — -ub also sizes the streamed expert-gather scratch.
    #[test]
    fn batch_and_ubatch_are_separate() {
        let mut a = argv(&["ojas", "-b", "128", "-ub", "8", "run"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert_eq!((cfg.prefill_m, cfg.ubatch), (128, 8));
        assert_eq!(a, argv(&["ojas", "run"]));
    }

    #[test]
    fn ubatch_floors_at_one() {
        let mut a = argv(&["ojas", "--ubatch-size", "0", "run"]);
        assert_eq!(take_engine_flags(&mut a).unwrap().ubatch, 1);
    }

    #[test]
    fn missing_value_names_the_flag() {
        let err = parse(argv(&["ojas", "--ctx-size"])).unwrap_err().to_string();
        assert!(err.contains("--ctx-size"), "error should name the flag: {err}");
    }

    /// The two passes are an implementation detail, so either order has to work.
    #[test]
    fn engine_and_runtime_flags_interleave_freely() {
        let (cfg, opts, rest) =
            parse(argv(&["ojas", "--temp", "0.2", "-c", "512", "run", "m.gguf", "-n", "16"])).unwrap();
        assert_eq!(cfg.ctx, Some(512));
        assert_eq!(opts.n_predict, 16);
        assert!((opts.sample.temperature - 0.2).abs() < 1e-6);
        assert_eq!(rest, argv(&["ojas", "run", "m.gguf"]));
    }

    /// Zero temperature must take the greedy path, not the sampler.
    #[test]
    fn zero_temperature_selects_greedy() {
        let (_, opts, _) = parse(argv(&["ojas", "--temp", "0", "run"])).unwrap();
        assert!(opts.sampling().is_none(), "temp 0 must be greedy");
        let (_, warm, _) = parse(argv(&["ojas", "--temp", "0.7", "run"])).unwrap();
        assert!(warm.sampling().is_some());
    }

    /// `-ngl 0` is how llama.cpp users say "CPU"; anything else must not pin a
    /// device, or `-ngl 99` would override an explicit `--device`.
    #[test]
    fn ngl_zero_means_cpu_and_nonzero_leaves_the_choice() {
        let (_, cpu, _) = parse(argv(&["ojas", "-ngl", "0", "run"])).unwrap();
        assert_eq!(cpu.device, Device::Cpu);
        let (_, auto, _) = parse(argv(&["ojas", "-ngl", "99", "run"])).unwrap();
        assert_eq!(auto.device, Device::Auto);
    }

    #[test]
    fn prompt_comes_from_flag_file_or_positional() {
        let (_, o, _) = parse(argv(&["ojas", "-p", "hello", "run"])).unwrap();
        assert_eq!(o.resolve_prompt(None).unwrap(), "hello");
        let (_, bare, _) = parse(argv(&["ojas", "run"])).unwrap();
        assert_eq!(bare.resolve_prompt(Some("positional")).unwrap(), "positional");
        assert!(bare.resolve_prompt(None).is_err(), "no prompt anywhere must be an error");
    }

    /// With both `-p` and a positional the flag wins; they are not concatenated.
    #[test]
    fn explicit_prompt_flag_beats_positional() {
        let (_, o, _) = parse(argv(&["ojas", "-p", "flag", "run"])).unwrap();
        assert_eq!(o.resolve_prompt(Some("positional")).unwrap(), "flag");
    }

    #[test]
    fn negative_numbers_are_values_not_flags() {
        let (_, o, _) = parse(argv(&["ojas", "--temp", "-1", "run"])).unwrap();
        assert!(o.sampling().is_none(), "negative temperature is greedy");
    }

    /// `--mmproj` belongs to the engine pass, so it and its value must be gone
    /// before the subcommand parse reads positions.
    #[test]
    fn mmproj_flag_does_not_disturb_positional_indices() {
        let mut a = argv(&["ojas", "--mmproj", "/tmp/v.gguf", "ocr", "model.gguf", "page.png"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert_eq!(cfg.mmproj, Some("/tmp/v.gguf".to_string()));
        assert_eq!(a, argv(&["ojas", "ocr", "model.gguf", "page.png"]));
        // From the middle of the positionals too, where an unconsumed value would
        // shift "page.png" into the model slot rather than merely be ignored.
        let mut mid = argv(&["ojas", "ocr", "model.gguf", "--mmproj", "/tmp/v.gguf", "page.png"]);
        assert_eq!(take_engine_flags(&mut mid).unwrap().mmproj, Some("/tmp/v.gguf".to_string()));
        assert_eq!(mid, argv(&["ojas", "ocr", "model.gguf", "page.png"]));
    }

    /// `--mmproj` must reach `EngineConfig` through the full two-pass parse: the
    /// run pass rejects unknown flags, so a mis-wired flag fails here.
    #[test]
    fn mmproj_survives_the_full_two_pass_parse() {
        let (cfg, opts, rest) =
            parse(argv(&["ojas", "--mmproj", "/tmp/v.gguf", "ocr", "m.gguf", "-n", "8", "page.png"])).unwrap();
        assert_eq!(cfg.mmproj, Some("/tmp/v.gguf".to_string()));
        assert_eq!(opts.n_predict, 8);
        assert_eq!(rest, argv(&["ojas", "ocr", "m.gguf", "page.png"]));
    }

    #[test]
    fn mmproj_without_a_value_names_the_flag() {
        let err = parse(argv(&["ojas", "--mmproj"])).unwrap_err().to_string();
        assert!(err.contains("--mmproj"), "error should name the flag: {err}");
    }

    /// Absent, the flag must not invent a path: auto-discovery next to the model
    /// is the default, and `Some("")` would defeat it.
    #[test]
    fn absent_mmproj_leaves_the_env_baseline_alone() {
        let mut a = argv(&["ojas", "run", "model"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert_eq!(cfg.mmproj, EngineConfig::from_env().mmproj);
    }

    /// The draft head and the projector are different sidecars and different fields.
    #[test]
    fn mmproj_and_draft_model_are_separate_fields() {
        let mut a = argv(&["ojas", "-md", "/tmp/d.gguf", "--mmproj", "/tmp/v.gguf", "run"]);
        let cfg = take_engine_flags(&mut a).unwrap();
        assert_eq!(cfg.mtp, Some("/tmp/d.gguf".to_string()));
        assert_eq!(cfg.mmproj, Some("/tmp/v.gguf".to_string()));
    }
}

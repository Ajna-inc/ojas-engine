//! ojas — engine CLI.
//!
//! Serving commands (`run`, `chat`, `serve`, `bench`, `tokenize`, `info`) work on
//! every platform, picking the Metal backend on Apple and the CPU decoders
//! elsewhere. The swarm commands (`worker`, `pipe-test`) are Metal-only and are
//! compiled out where there is no Metal.
//!
//! Engine flags may appear anywhere and are removed before the subcommand's
//! positionals are read. They use llama.cpp's names where one exists.

mod backend;
mod cmds;
#[cfg(target_os = "macos")]
mod decide;
mod detok;
mod flags;
mod ocr;
mod sched;
mod serve;
mod vserve;
mod vision;

use anyhow::Result;
use ojas_core::config::EngineConfig;

const USAGE: &str = "\
ojas — LLM inference engine

usage:
  ojas run    <model.gguf> [prompt]     generate once and exit
  ojas chat   <model.gguf>              interactive conversation
  ojas serve  <model.gguf>              OpenAI-compatible HTTP server
  ojas vision-serve <model.onnx>        HTTP detector (remote inference)
  ojas bench  <model.gguf> [prompt]     measure tokens/s
  ojas ocr    <model.gguf> <page|dir>   transcribe page images (needs --mmproj)
  ojas tokenize <model.gguf> [text]     show the token split
  ojas info   <model.gguf|.onnx>        architecture, tensors and metadata
  ojas detect <model.onnx> <image|dir>  object detection (CPU; --conf --iou --json)
  ojas plate  <det.onnx> <rec.onnx> <dict.txt> <image|dir>  detect + read plates
  ojas vbench <model.onnx>              vision forward-pass timing
  ojas decide <laya.gguf>               typed decisions (Laya, Metal; serve and bench take one too)
";

// No `\` line-continuation: it would strip the leading indent off the first
// entry, leaving it at column zero under the aligned list above.
#[cfg(target_os = "macos")]
const SWARM_USAGE: &str =
    "  ojas worker <model> <lo> <hi> <port> <next>   pipeline stage\n\
     \x20 ojas pipe-test <model> <data.bin> <addr>      pipeline smoke test\n\
     \x20 ojas diloco-worker <model> <data.bin> <hub> <shard> <n>   data-parallel worker\n\
     \x20 ojas diloco-hub <port> <n_workers> <rounds>  delta aggregation (no GPU)\n";
// The hub runs no kernels, so it is offered everywhere; only the worker needs a
// training backend.
#[cfg(not(target_os = "macos"))]
const SWARM_USAGE: &str =
    "  ojas diloco-hub <port> <n_workers> <rounds>   delta aggregation (no GPU)\n";

const FLAGS: &str = "\
engine flags (llama.cpp names where one exists):
  -c,    --ctx-size N     context length
  -cram, --cache-ram N    expert cache budget, MiB
  -b,    --batch-size N   prefill chunk, tokens
  -ub,   --ubatch-size N  tokens per physical pass (sizes the streamed expert scratch)
  -md,   --spec-draft-model FNAME  MTP draft head
         --mmproj FNAME   vision projector GGUF (ocr; defaults to *mmproj*.gguf beside the model)
         --top-k-experts N  MoE experts per token (NOT sampling top-k)
         --mmap/--no-mmap     zero-copy weight mapping
         --mlock/--no-mlock   pin the resident skeleton
         --warmup/--no-warmup touch pages at load
         --moe-dbuf/--no-moe-dbuf  double-buffered expert gather
         --resident/--stream  prefer full expert residency / bounded-memory streaming
         --mixed-q4 MODE  off | head | balanced | aggressive (Qwen3.5 Q8 models)
         --q4-head/--no-q4-head  convert the untied output head to Q4
         --q4-ffn-down-last N    convert only the last N FFN-down matrices to Q4
         --q4-ffn-down/--no-q4-ffn-down  convert all/none of the FFN-down matrices
                              balanced = head + last 12; aggressive may change output

runtime flags:
  -n,    --predict N      tokens to generate            (default 128)
  -p,    --prompt TEXT    prompt text
  -f,    --file FNAME     read the prompt from a file
  -sys,  --system TEXT    system prompt
         --raw            no chat template; feed the prompt verbatim
         --temp N         sampling temperature (0 = greedy, the exact path)
         --top-p N        nucleus mass
         --top-k N        sampling top-k
         --repeat-penalty N / --repeat-last-n N
  -s,    --seed N         sampler seed
         --device D       auto | metal | cpu
  -ngl,  --n-gpu-layers N 0 selects the CPU backend
         --precision N    Metal decoder precision tier (default: 3 for dense models,
                          4 for models with experts)
         --host H --port P   serve address (default 127.0.0.1:8080)
  -r,    --reps N         bench repetitions (default 3)
         --conf N --iou N  vision thresholds (detect/plate)
         --json           detect/decide: JSON output
         --state S / --state-file F          decide: JSON object or text to decide about
         --questions Q / --questions-file F  decide: {id: {type, instructions, criteria}}
";

fn main() -> Result<()> {
    ojas_core::logging::init(); // default stderr sink; OJAS_LOG controls level/filter
    let argv: Vec<String> = std::env::args().collect();
    if argv.iter().skip(1).any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}{SWARM_USAGE}\n{FLAGS}");
        return Ok(());
    }
    let (cfg, opts, a) = flags::parse(argv)?;
    // Context is an engine flag, but the backends need it as a number here. The
    // flag itself is kept as well: `ocr` has a higher default because one page
    // prompt (4096 image tokens plus scaffolding) does not fit 4096 and
    // `check_fits` errors rather than clipping. An explicit -c still wins.
    let ctx_flag = cfg.ctx;
    let context = ctx_flag.unwrap_or(4096);
    let _ = EngineConfig::install(cfg); // must precede any model load

    let need = |n: usize| -> Result<&String> {
        a.get(n).ok_or_else(|| anyhow::anyhow!("{}\n{}", USAGE, FLAGS))
    };
    let positional = |n: usize| a.get(n).map(String::as_str);

    match a.get(1).map(String::as_str) {
        Some("run" | "generate") => cmds::run(need(2)?, &opts, positional(3), context),
        Some("chat") => cmds::chat(need(2)?, &opts, context),
        Some("serve" | "server") => serve::serve(need(2)?, &opts, context),
        Some("bench") => cmds::bench(need(2)?, &opts, positional(3), context),
        Some("tokenize") => cmds::tokenize(need(2)?, &opts, positional(3)),
        Some("ocr") => ocr::ocr(need(2)?, need(3)?, &opts, ctx_flag.unwrap_or(ocr::DEFAULT_CTX)),
        Some("info") => cmds::info(need(2)?),
        Some("detect") => vision::detect(need(2)?, need(3)?, &opts),
        Some("plate") => vision::plate(need(2)?, need(3)?, need(4)?, need(5)?, &opts),
        Some("vbench") => vision::vbench(need(2)?, &opts),
        Some("vision-serve") => vserve::vision_serve(need(2)?, &opts),
        #[cfg(target_os = "macos")]
        Some("decide") => decide::decide(need(2)?, &opts),
        #[cfg(not(target_os = "macos"))]
        Some("decide") => anyhow::bail!("`decide` needs the Metal backend and is not built on this platform"),

        #[cfg(target_os = "macos")]
        Some("worker") if a.len() >= 7 => ojas_swarm::worker(
            &a[2], a[3].parse()?, a[4].parse()?, a[5].parse()?, &a[6],
            a.get(7).and_then(|s| s.parse().ok()).unwrap_or(256),
            a.get(8).and_then(|s| s.parse().ok()).unwrap_or(1e-5),
            a.get(9).and_then(|s| s.parse().ok()).unwrap_or(0.5)),
        #[cfg(target_os = "macos")]
        Some("pipe-test") if a.len() >= 5 => ojas_swarm::pipe_test(
            &a[2], &a[3], &a[4],
            a.get(5).and_then(|s| s.parse().ok()).unwrap_or(200),
            a.get(6).and_then(|s| s.parse().ok()).unwrap_or(6420)),
        #[cfg(target_os = "macos")]
        Some("diloco-worker") if a.len() >= 5 => ojas_swarm::diloco_worker(
            &a[2], &a[3], &a[4],
            a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0),
            a.get(6).and_then(|s| s.parse().ok()).unwrap_or(1),
            a.get(7).and_then(|s| s.parse().ok()).unwrap_or(50),
            a.get(8).and_then(|s| s.parse().ok()).unwrap_or(1e-5)),
        #[cfg(not(target_os = "macos"))]
        Some(c @ ("worker" | "pipe-test" | "diloco-worker")) => anyhow::bail!(
            "`{c}` needs the Metal backend and is not built on this platform"
        ),

        // Pure aggregation: no model and no kernels, so it builds everywhere.
        Some("diloco-hub") if a.len() >= 4 => ojas_core::diloco::hub(
            a[2].parse()?, a[3].parse()?,
            a.get(4).and_then(|s| s.parse().ok()).unwrap_or(20),
            a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.7),
            a.get(6).map(|s| s.as_str())),

        Some("help") => {
            print!("{USAGE}{SWARM_USAGE}\n{FLAGS}");
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown command {other:?}\n\n{USAGE}{SWARM_USAGE}\n{FLAGS}"),
        None => {
            print!("{USAGE}{SWARM_USAGE}\n{FLAGS}");
            Ok(())
        }
    }
}

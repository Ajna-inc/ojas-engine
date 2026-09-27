#!/usr/bin/env python3
"""Reference-checked, sequential comparisons of Flash target copy concurrency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path,
                        default=Path("target/release/examples/flash_target_bench"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 1])
    parser.add_argument("--interleave", action="store_true",
                        help="Rotate width/thread pairs within one loaded model (use distinct threads).")
    parser.add_argument("--cache-gb", type=int, default=32)
    parser.add_argument("--prefetch", action="store_true", help="Enable the optional MoE staging prefetch.")
    parser.add_argument("--widths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    if (not all(1 <= t <= 8 for t in args.threads)
            or not all(1 <= w <= 16 for w in args.widths)
            or len(set(args.widths)) != len(args.widths)
            or not 3 <= args.reps <= 20 or args.cache_gb <= 0):
        parser.error("threads 1..8, distinct widths 1..16, reps 3..20, positive cache required")
    if args.interleave and len(set(args.threads)) != len(args.threads):
        parser.error("interleaved threads must be distinct; e.g. --threads 1 2 4")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    report = {
        "binary_sha256": sha(binary),
        "source_sha256": {str(p): sha(p) for p in sorted(Path("crates").rglob("*.rs"))},
        "reference_sha256": sha(args.reference),
        "model_entrypoint_sha256": sha(args.model),
        "model_hash_scope": "Entrypoint only; use the separate complete checkpoint manifest.",
        "platform": platform.platform(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "timing_scope": "32 reference outputs, prefill excluded, one warmup per width; GPU time is inside submit_wait_s. Width 1 uses the batched target graph. No drafting or head maintenance.",
        "cases": [],
    }
    runs = [args.threads] if args.interleave else [[t] for t in args.threads]
    for index, thread_choices in enumerate(runs):
        threads = thread_choices[0]
        if sha(binary) != report["binary_sha256"]:
            raise RuntimeError("binary changed during comparison")
        env = {k: v for k, v in os.environ.items() if not k.startswith("OJAS_")}
        settings = dict(OJAS_NO_SPEC="1", OJAS_EXPERT_CACHE_GB=str(args.cache_gb),
                        OJAS_EXPERT_COPY_THREADS=str(threads))
        if args.prefetch:
            settings["OJAS_MOE_DBUF"] = "1"
        env.update(settings)
        log = args.output_dir / f"{index}-copy-{threads}.log"
        before = subprocess.check_output(["vm_stat"], text=True)
        with log.open("w") as stream:
            try:
                code = subprocess.run(
                    [str(binary), args.model, str(args.reference), str(args.reps),
                     ",".join(map(str, args.widths)), ",".join(map(str, thread_choices))], env=env, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=1200).returncode
            except subprocess.TimeoutExpired:
                code = 124
        records = [json.loads(line) for line in log.read_text().splitlines()
                   if line.startswith("{")]
        passed = code == 0 and len(records) == args.reps * len(args.widths) * len(thread_choices) and all(
            len([r for r in records if r["width"] == w and r["copy_threads"] == t]) == args.reps
            for w in args.widths for t in thread_choices)
        case = dict(threads=thread_choices, environment=settings, exit_code=code, passed=passed,
                    records=records, log_sha256=sha(log), memory_before=before,
                    memory_after=subprocess.check_output(["vm_stat"], text=True), summary={})
        for width, t in [(w, t) for w in args.widths for t in thread_choices]:
            samples = [r for r in records if r["width"] == width and r["copy_threads"] == t]
            if samples:
                case["summary"][f"width={width},threads={t}"] = {
                    key: {"median": statistics.median(r[key] for r in samples),
                          "min": min(r[key] for r in samples), "max": max(r[key] for r in samples)}
                    for key in ("tps", "target_s", "gather_copy_s", "encode_s", "submit_wait_s", "gpu_s")}
        report["cases"].append(case)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"threads={thread_choices}: {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

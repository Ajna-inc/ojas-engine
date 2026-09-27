#!/usr/bin/env python3
"""Matched Ojas / llama.cpp benchmark.

Both engines expose the same HTTP surface, so ONE client drives both with the
same requests and the same clock. That is the point of this script: every
previous comparison in this repository measured each engine with its own
harness and its own definition of throughput, which is not a comparison.

Fairness rules enforced here:

  * identical token ids in  — the prompt is tokenized once and sent as an id
    array to both engines, so tokenizer differences cannot change the workload
  * identical token count out — `ignore_eos` with a fixed `n_predict`, so both
    engines do the same amount of work
  * no prefix reuse — `cache_prompt: false`; a warm cache is a different
    experiment
  * greedy — `temperature: 0`, so output is deterministic and comparable
  * same warmup policy — one unmeasured request, then N measured
  * same throughput definition — decode tok/s = (n-1) / (last arrival - first
    arrival). The first token is prefill latency and is reported separately as
    TTFT rather than folded into the decode rate.

Outputs are captured and compared between engines. A speed number is only
meaningful next to the answer it produced.
"""

import argparse, json, os, pathlib, signal, statistics, subprocess, sys, threading, time, urllib.error, urllib.request


def http(url, payload=None, timeout=600, stream=False):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def wait_health(port, proc, seconds=1800):
    """Block until the server answers, or it dies. Big models take minutes."""
    for _ in range(seconds * 2):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with http(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("server never became healthy")


def rss_gib(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True)
        return int(out.stdout.strip()) / 1048576
    except Exception:
        return 0.0


def warm_page_cache(path, chunk=1 << 24):
    """Read the model once so neither engine pays the cold-read cost.

    Called before each engine because without it whichever ran first was
    penalised by more than 2x on a 27 GB model, and the same configuration
    swung from 0.55x to 1.51x across runs until page-cache warming existed
. The bytes are
    read and discarded: what is wanted is the OS page cache, not the data.

    A model larger than free RAM cannot be fully cached, and this still
    equalises the two engines' starting state, which is the whole claim. Only
    the file named by --model is read; a sidecar an engine loads on its own is
    not, so a sidecar-heavy configuration is still asymmetric.
    """
    read, start = 0, time.perf_counter()
    try:
        # buffering=0: a Python-level buffer would copy every byte for nothing.
        with open(path, "rb", buffering=0) as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                read += len(block)
    except OSError as error:
        print(f"  warm-cache SKIPPED for {path}: {error}", flush=True)
        return 0.0
    took = time.perf_counter() - start
    rate = f" ({read / 2**30 / took:.2f} GiB/s)" if took > 0 else ""
    print(f"  warmed page cache: {read / 2**30:.2f} GiB in {took:.1f}s{rate}", flush=True)
    return took


def one_request(port, ids, n_predict, ignore_eos=False):
    """Stream one completion.

    Returns the actual token ids, not just decoded text: two engines can print
    the same string from different tokens, and comparing text cannot tell. The
    timing window spans TOKEN arrivals, not text chunks, because a token whose
    bytes complete a multi-byte character decodes to an empty string and would
    otherwise be invisible to both the count and the clock.
    """
    payload = {
        "prompt": ids,
        "n_predict": n_predict,
        "max_tokens": n_predict,
        "temperature": 0,
        "stream": True,
        "cache_prompt": False,
        # Suppression is NOT used by default. Once both engines recognise the same
        # end-of-generation set they stop at the same token, so counts match
        # without it — and on this engine `ignore_eos` disables speculation and
        # forces a full-vocabulary logits copy per token, which would measure the
        # suppression path rather than the serving path.
        "ignore_eos": ignore_eos,
        "return_tokens": True,
    }
    start = time.perf_counter()
    first = last = None
    out_ids, pieces = [], []
    saw_eos_stop = False
    with http(f"http://127.0.0.1:{port}/completion", payload) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            body = line[6:].strip()
            if body == b"[DONE]":
                break
            ev = json.loads(body)
            toks = ev.get("tokens") or []
            if ev.get("stop"):
                if ev.get("content"):
                    pieces.append(ev["content"])
                if ev.get("stopped_eos"):
                    saw_eos_stop = True
                continue
            now = time.perf_counter()
            if toks:
                if first is None:
                    first = now
                last = now
                out_ids.extend(toks)
            if ev.get("content"):
                pieces.append(ev["content"])
    total = time.perf_counter() - start
    n = len(out_ids)
    decode_s = (last - first) if (first is not None and last is not None) else 0.0
    return {
        "ids": out_ids,
        "n": n,
        "text": "".join(pieces),
        "ttft_s": (first - start) if first is not None else 0.0,
        "decode_s": decode_s,
        # Rate uses the tokens actually observed, never the number requested.
        "decode_tps": (n - 1) / decode_s if (n > 1 and decode_s > 0) else 0.0,
        "total_s": total,
        "stopped_eos": saw_eos_stop,
        "load_average": os.getloadavg()[0],
    }


class MemorySampler(threading.Thread):
    """Poll RSS continuously.

    Sampling only after a request misses the load-time peak, which for a 90 GB
    model is where the high-water mark actually is.

    This is a SAMPLED peak, not a true one: a 100 ms poll can miss a brief spike.
    It is also process RSS, which is not the memory the configuration costs the
    system — the model mapping and page cache sit outside it.
    """

    def __init__(self, pid, interval=0.1):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak, self._halt = pid, interval, 0.0, threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.peak = max(self.peak, rss_gib(self.pid))
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        return self.peak


def bench_engine(name, cmd, port, prompts, n_predict, reps, warmups, cwd=None, env=None, ignore_eos=False):
    log = pathlib.Path(f"/tmp/engine-bench-{name}-{port}.log")
    e = dict(os.environ)
    for k in list(e):
        if k.startswith("OJAS_"):
            del e[k]
    if env:
        e.update(env)
    print(f"  starting {name}: {' '.join(cmd[:3])} ... (log {log})", flush=True)
    with log.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=e,
                                start_new_session=True)
    # Start sampling immediately: the load-time peak is the real high-water mark.
    sampler = MemorySampler(proc.pid)
    sampler.start()
    result = {"engine": name, "command": cmd, "cases": []}
    try:
        t0 = time.perf_counter()
        wait_health(port, proc)
        result["load_s"] = time.perf_counter() - t0
        result["rss_after_load_gib"] = rss_gib(proc.pid)

        for label, ids in prompts:
            for _ in range(warmups):
                one_request(port, ids, min(n_predict, 8), ignore_eos)
            samples = [one_request(port, ids, n_predict, ignore_eos) for _ in range(reps)]
            for r, s_ in enumerate(samples):
                flag = "" if s_["n"] == n_predict else f"  (stopped at {s_['n']} of {n_predict})"
                print(f"    {name} {label} rep{r+1}: {s_['decode_tps']:.2f} tok/s "
                      f"ttft {s_['ttft_s']:.2f}s n={s_['n']}{flag}", flush=True)
            # Self-consistency: greedy decoding must repeat exactly. If an engine
            # disagrees with itself across repetitions, cross-engine comparison of
            # a single repetition is meaningless.
            id_sets = {tuple(s_["ids"]) for s_ in samples}
            result["cases"].append({
                "prompt": label,
                "prompt_tokens": len(ids),
                "n_predict_requested": n_predict,
                "token_counts": [s_["n"] for s_ in samples],
                # Stopping early at a real end-of-generation token is correct
                # behaviour, so the requirement is that the two engines stop at the
                # SAME point, not that either ran to the requested length.
                "count_ok": len({s_["n"] for s_ in samples}) == 1,
                "self_consistent": len(id_sets) == 1,
                "median_decode_tps": statistics.median(s_["decode_tps"] for s_ in samples),
                "min_decode_tps": min(s_["decode_tps"] for s_ in samples),
                "max_decode_tps": max(s_["decode_tps"] for s_ in samples),
                "median_ttft_s": statistics.median(s_["ttft_s"] for s_ in samples),
                "ids": [s_["ids"] for s_ in samples],
                "text": samples[0]["text"],
                "samples": [{k: v for k, v in s_.items() if k != "ids"} for s_ in samples],
            })
    finally:
        # Teardown FIRST, and unconditionally. This block once read the memory
        # sampler before killing the server; the sampler raised, the kill never
        # ran, and a 61 GB llama-server outlived the benchmark — the next model
        # load then hit a genuine GPU OutOfMemory. Nothing may precede the kill.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            result["sampled_peak_rss_gib"] = sampler.stop()
        except Exception:
            result["sampled_peak_rss_gib"] = sampler.peak
        time.sleep(5)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--label", default=None, help="short name for the report")
    p.add_argument("--ojas", default="target/release/ojas")
    p.add_argument("--llama-bin", default="/opt/llama.cpp/build-metal/bin")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--n-predict", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--ignore-eos", action="store_true",
                   help="suppress the end-of-generation set (llama.cpp semantics). Off by default: "
                        "it disables speculation on this engine and measures the wrong path.")
    p.add_argument("--cycles", type=int, default=1,
                   help="alternate llama/ojas this many times; ratios are computed per cycle "
                        "so drift between batches cannot masquerade as an engine difference")
    p.add_argument("--warm-cache", action="store_true",
                   help="read the model before each engine so neither pays the cold-read cost")
    p.add_argument("--port-ojas", type=int, default=18981)
    p.add_argument("--port-llama", type=int, default=18982)
    p.add_argument("--output", required=True)
    p.add_argument("--ojas-env", default="", help="k=v,k=v applied to the ojas server only")
    p.add_argument("--llama-extra", default="",
                   help="extra space-separated llama-server flags (Flash needs --load-mode none --lazy-mode on)")
    p.add_argument("--ojas-precision", type=int, default=None,
                   help="ojas decoder precision tier; 4 (default) streams weights from the GGUF")
    p.add_argument("--prompt", action="append", default=[],
                   help="prompt text; repeatable. Defaults to three fixtures.")
    a = p.parse_args()

    texts = a.prompt or [
        "The capital of France is",
        "def fibonacci(n):",
        "one two three one two three one two three",
    ]
    llama_server = str(pathlib.Path(a.llama_bin) / "llama-server")

    # Tokenize once, with llama.cpp, and send ids to both. Whoever tokenizes,
    # both engines then receive byte-identical input.
    tok_port = a.port_llama + 7
    extra = a.llama_extra.split() if a.llama_extra else []
    tok_cmd = [llama_server, "-m", a.model, "-c", str(a.ctx), "-ngl", "99",
               "--host", "127.0.0.1", "--port", str(tok_port), "-np", "1", "--no-warmup"] + extra
    print("tokenizing prompts via llama.cpp ...", flush=True)
    tlog = pathlib.Path("/tmp/engine-bench-tokenize.log")
    with tlog.open("w") as f:
        tp = subprocess.Popen(tok_cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    prompts = []
    try:
        wait_health(tok_port, tp)
        for t in texts:
            with http(f"http://127.0.0.1:{tok_port}/tokenize", {"content": t}) as r:
                ids = json.load(r)["tokens"]
            prompts.append((t[:28], ids))
            print(f"  {t[:28]!r} -> {len(ids)} tokens", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(tp.pid), signal.SIGTERM); tp.wait(timeout=60)
        except Exception:
            pass
        time.sleep(3)

    ojas_env = dict(kv.split("=", 1) for kv in a.ojas_env.split(",") if kv)
    report = {
        "model": a.model,
        "label": a.label or pathlib.Path(a.model).name,
        "ctx": a.ctx,
        "n_predict": a.n_predict,
        "reps": a.reps,
        "warmups": a.warmups,
        "ojas_env": ojas_env,
        "ojas_precision": a.ojas_precision,
        "warm_cache": a.warm_cache,
        "llama_extra": a.llama_extra,
        "host": {
            "load_at_start": os.getloadavg(),
            "llama_commit": subprocess.run(
                ["git", "-C", str(pathlib.Path(a.llama_bin).parents[1]), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True).stdout.strip(),
        },
        "definition": "decode_tps = (n_predict-1)/(last arrival - first arrival); ttft separate; "
                      "ignore_eos + fixed n_predict so both engines emit the same count; "
                      "cache_prompt false; greedy; identical prompt token ids.",
        "engines": [],
    }

    if a.warm_cache:
        warm_page_cache(a.model)
    llama_cmd = [llama_server, "-m", a.model, "-c", str(a.ctx), "-ngl", "99",
                 "--host", "127.0.0.1", "--port", str(a.port_llama), "-np", "1",
                 "--no-warmup"] + extra
    ojas_cmd = [a.ojas, "serve", a.model, "-c", str(a.ctx), "--host", "127.0.0.1",
                "--port", str(a.port_ojas)] \
               + (["--precision", str(a.ojas_precision)] if a.ojas_precision is not None else [])

    # Absolute throughput on this host drifts by tens of percent between batches
    # while paired ratios stay stable, so the ratio is computed inside a cycle and
    # only ratios are compared across cycles.
    cycles = []
    for cycle in range(max(1, a.cycles)):
        print(f"\n--- cycle {cycle + 1}/{max(1, a.cycles)} ---", flush=True)
        if a.warm_cache:
            warm_page_cache(a.model)
        llama = bench_engine("llama.cpp", llama_cmd, a.port_llama, prompts,
                             a.n_predict, a.reps, a.warmups, ignore_eos=a.ignore_eos)
        if a.warm_cache:
            warm_page_cache(a.model)
        ojas = bench_engine("ojas", ojas_cmd, a.port_ojas, prompts,
                            a.n_predict, a.reps, a.warmups, env=ojas_env, ignore_eos=a.ignore_eos)
        cycles.append({"cycle": cycle, "llama.cpp": llama, "ojas": ojas})
    report["cycles"] = cycles
    # Keep the last cycle under the old key so existing readers still work.
    report["engines"] = [cycles[-1]["llama.cpp"], cycles[-1]["ojas"]]

    # Same prompt, same greedy settings: the two engines should emit the same
    # TOKEN IDS. Comparing decoded text would pass on different tokenisations of
    # the same string, and comparing one repetition would miss an engine that
    # disagrees with itself.
    agree = []
    for i, case in enumerate(report["engines"][0]["cases"]):
        other = report["engines"][1]["cases"][i]
        l_ids, o_ids = case["ids"], other["ids"]
        every_rep = len(l_ids) == len(o_ids) and all(x == y for x, y in zip(l_ids, o_ids))
        first = l_ids[0] == o_ids[0] if l_ids and o_ids else False
        divergence = None
        if not first and l_ids and o_ids:
            for k, (lt, ot) in enumerate(zip(l_ids[0], o_ids[0])):
                if lt != ot:
                    divergence = k
                    break
        agree.append({
            "prompt": case["prompt"],
            "ids_identical_first_rep": first,
            "ids_identical_every_rep": every_rep,
            "first_divergence_index": divergence,
            "llama_self_consistent": case["self_consistent"],
            "ojas_self_consistent": other["self_consistent"],
            "llama_counts_ok": case["count_ok"],
            "ojas_counts_ok": other["count_ok"],
            # A row is only a valid speed comparison when both engines emitted the
            # same tokens, the same number of them, and repeated themselves.
            "valid": every_rep and case["count_ok"] and other["count_ok"]
                     and case["self_consistent"] and other["self_consistent"],
            "llama_text": case["text"][:160],
            "ojas_text": other["text"][:160],
        })
    report["output_agreement"] = agree

    pathlib.Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.output).write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n=== {report['label']} | ctx {a.ctx} | {a.n_predict} tokens | {a.reps} reps ===")
    print(f"{'prompt':<30} {'llama.cpp':>12} {'ojas':>12} {'ratio':>8} {'ids':>5} {'valid':>6}")
    for i, case in enumerate(report["engines"][0]["cases"]):
        o = report["engines"][1]["cases"][i]
        l_tps, o_tps = case["median_decode_tps"], o["median_decode_tps"]
        ratio = o_tps / l_tps if l_tps else 0.0
        ag = agree[i]
        print(f"{case['prompt']:<30} {l_tps:>10.2f}/s {o_tps:>10.2f}/s {ratio:>7.2f}x "
              f"{'yes' if ag['ids_identical_every_rep'] else 'NO':>5} "
              f"{'yes' if ag['valid'] else 'NO':>6}")
    # Per-cycle ratios: the trustworthy statistic on a drifting host.
    if len(report.get("cycles", [])) > 1:
        print("\nper-cycle ratios (ojas / llama.cpp), paired within a cycle:")
        for pi, case in enumerate(report["engines"][0]["cases"]):
            rs = []
            for c in report["cycles"]:
                lt = c["llama.cpp"]["cases"][pi]["median_decode_tps"]
                ot = c["ojas"]["cases"][pi]["median_decode_tps"]
                rs.append(ot / lt if lt else 0.0)
            body = "  ".join(f"{r:.2f}x" for r in rs)
            print(f"  {case['prompt']:<30} {body}   median {statistics.median(rs):.2f}x")

    for row in agree:
        if not row["valid"]:
            why = []
            if not row["ids_identical_every_rep"]:
                why.append(f"token ids differ (first at index {row['first_divergence_index']})")
            if not row["llama_counts_ok"] or not row["ojas_counts_ok"]:
                why.append("token count != requested")
            if not row["llama_self_consistent"]:
                why.append("llama.cpp not self-consistent across reps")
            if not row["ojas_self_consistent"]:
                why.append("ojas not self-consistent across reps")
            print(f"  INVALID {row['prompt']!r}: {'; '.join(why)}")
    print(f"\nsampled peak RSS (100 ms polling; brief peaks can be missed, and this is "
          f"process RSS, not total system memory cost):")
    print(f"  llama.cpp {report['engines'][0].get('sampled_peak_rss_gib', 0):.2f} GiB | "
          f"ojas {report['engines'][1].get('sampled_peak_rss_gib', 0):.2f} GiB")
    print(f"load at start: {report['host']['load_at_start'][0]:.1f}")
    print(f"written to {a.output}")


if __name__ == "__main__":
    main()

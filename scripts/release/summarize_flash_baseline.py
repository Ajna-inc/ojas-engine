#!/usr/bin/env python3
"""Summarize the Flash baseline matrix.

Rules this enforces, because the numbers are easy to misread:

  * a prompt whose row is INVALID (engines emitted different token ids, or a
    count did not match the request) is excluded from every performance
    aggregate — it is not a comparison
  * ratios are computed per alternating cycle and only then combined, so drift
    between batches cannot masquerade as an engine difference
  * a row whose cycles disagree by more than `--tolerance` is marked PROVISIONAL.
    Alternation reduces drift; it does not eliminate interference from changing
    host load, and a disagreement is the visible symptom of that
"""
import argparse, glob, json, pathlib, statistics


def load(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        try:
            out.append((pathlib.Path(f).stem, json.load(open(f))))
        except Exception as e:  # a run that died mid-write
            print(f"  (unreadable {f}: {e})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", default="/tmp/flash-baseline/*.json")
    p.add_argument("--tolerance", type=float, default=0.10,
                   help="fractional spread between cycle ratios above which a row is PROVISIONAL")
    a = p.parse_args()

    rows = []
    for name, r in load(a.glob):
        agree = r["output_agreement"]
        cycles = r.get("cycles") or []
        per_prompt = []
        for pi, case in enumerate(r["engines"][0]["cases"]):
            ratios, ojas, llama = [], [], []
            for c in cycles:
                lt = c["llama.cpp"]["cases"][pi]["median_decode_tps"]
                ot = c["ojas"]["cases"][pi]["median_decode_tps"]
                if lt > 0:
                    ratios.append(ot / lt)
                ojas.append(ot)
                llama.append(lt)
            spread = (max(ratios) - min(ratios)) / statistics.median(ratios) if len(ratios) > 1 and statistics.median(ratios) else 0.0
            per_prompt.append({
                "prompt": case["prompt"],
                "valid": agree[pi]["valid"],
                "ratio": statistics.median(ratios) if ratios else 0.0,
                "ojas": statistics.median(ojas) if ojas else 0.0,
                "llama": statistics.median(llama) if llama else 0.0,
                "spread": spread,
                "provisional": spread > a.tolerance,
                "why_invalid": None if agree[pi]["valid"] else (
                    f"ids differ @ {agree[pi]['first_divergence_index']}"
                    if not agree[pi]["ids_identical_every_rep"] else "count/self-consistency"),
            })
        rows.append((name, r, per_prompt))

    print(f"{'config':<32} {'prompt':<26} {'llama':>7} {'ojas':>7} {'ratio':>7} {'status':>13}")
    print("-" * 96)
    for name, _, per_prompt in rows:
        for x in per_prompt:
            if not x["valid"]:
                status = "EXCLUDED"
            elif x["provisional"]:
                status = f"PROVISIONAL"
            else:
                status = "ok"
            print(f"{name:<32} {x['prompt'][:26]:<26} {x['llama']:>7.2f} {x['ojas']:>7.2f} "
                  f"{x['ratio']:>6.2f}x {status:>13}")

    print("\n=== performance summary (INVALID prompts excluded) ===")
    print(f"{'config':<32} {'valid prompts':>14} {'median ojas':>12} {'median ratio':>13} {'peak RSS(sampled)':>18}")
    best = []
    for name, r, per_prompt in rows:
        good = [x for x in per_prompt if x["valid"]]
        if not good:
            print(f"{name:<32} {'0 — no valid rows':>14}")
            continue
        med_ratio = statistics.median(x["ratio"] for x in good)
        med_ojas = statistics.median(x["ojas"] for x in good)
        rss = r["engines"][1].get("sampled_peak_rss_gib", 0.0)
        flag = " *" if any(x["provisional"] for x in good) else ""
        print(f"{name:<32} {len(good):>14} {med_ojas:>12.2f} {med_ratio:>12.2f}x {rss:>17.1f}G{flag}")
        best.append((med_ojas, med_ratio, name, len(good)))

    if best:
        best.sort(reverse=True)
        top = best[0]
        print(f"\nfastest configuration by median ojas tok/s over VALID prompts:")
        print(f"  {top[2]}  {top[0]:.2f} tok/s  ({top[1]:.2f}x llama.cpp, {top[3]} valid prompts)")
    print("\n* = cycles disagreed by more than the tolerance; treat as provisional.")
    print("Ratios are per-cycle paired. Absolute tok/s on this host drifts between batches.")
    print("Sampled peak RSS: 100 ms polling of process RSS; brief peaks and system-level cost are not captured.")


if __name__ == "__main__":
    main()

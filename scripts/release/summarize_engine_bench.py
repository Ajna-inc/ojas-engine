#!/usr/bin/env python3
"""Collapse engine_bench.py result files into one comparison table.

Reports the ratio per prompt and the peak resident memory each engine needed,
because throughput at 9x the memory is not the same result as throughput.
"""
import glob, json, pathlib, statistics, sys


# engine_bench.py renamed both of the fields below. Result files written before
# the rename (evidence/engine-benchmark-2026-09-10/) carry the first
# spelling; everything since carries the second. Reading only one of them is how
# this script came to KeyError on new files and to report 0.00G for memory it had
# simply failed to find, so read either and say so when neither is there.
def agreed(entry):
    """Did both engines emit the same ids in every repetition?"""
    for key in ("ids_identical_every_rep", "identical"):
        if key in entry:
            return bool(entry[key])
    raise KeyError("result file has no output-agreement field")


def peak_rss(engine):
    """Peak resident GiB, or None when the producer recorded none."""
    for key in ("sampled_peak_rss_gib", "peak_rss_gib"):
        if key in engine:
            return float(engine[key])
    return None


def gib(value):
    return "n/a" if value is None else f"{value:.2f}G"


def main(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit("no result files matched")

    rows = []
    for f in files:
        r = json.load(open(f))
        try:
            llama, ojas = r["engines"][0], r["engines"][1]
        except (KeyError, IndexError):
            print(f"  (skipping incomplete {f})")
            continue
        ratios, matches = [], []
        for i, c in enumerate(llama["cases"]):
            o = ojas["cases"][i]
            if c["median_decode_tps"] > 0:
                ratios.append(o["median_decode_tps"] / c["median_decode_tps"])
            matches.append(agreed(r["output_agreement"][i]))
        rows.append({
            "label": r["label"],
            "file": pathlib.Path(f).name,
            "llama_tps": statistics.median(c["median_decode_tps"] for c in llama["cases"]),
            "ojas_tps": statistics.median(c["median_decode_tps"] for c in ojas["cases"]),
            "ratio_median": statistics.median(ratios) if ratios else 0.0,
            "ratio_min": min(ratios) if ratios else 0.0,
            "ratio_max": max(ratios) if ratios else 0.0,
            "llama_rss": peak_rss(llama),
            "ojas_rss": peak_rss(ojas),
            "all_match": all(matches),
            "load": r["host"]["load_at_start"][0],
            "warm": r.get("warm_cache", False),
        })

    w = max(len(x["label"]) for x in rows) + 1
    print(f"\n{'model':<{w}} {'llama':>9} {'ojas':>9} {'ratio':>7} {'range':>13} "
          f"{'llama RSS':>10} {'ojas RSS':>9} {'same out':>9} {'load':>6}")
    print("-" * (w + 78))
    for x in rows:
        print(f"{x['label']:<{w}} {x['llama_tps']:>8.2f}/s {x['ojas_tps']:>8.2f}/s "
              f"{x['ratio_median']:>6.2f}x {x['ratio_min']:>5.2f}-{x['ratio_max']:<6.2f} "
              f"{gib(x['llama_rss']):>10} {gib(x['ojas_rss']):>9} "
              f"{'yes' if x['all_match'] else 'NO':>9} {x['load']:>6.1f}")
    bad = [x["label"] for x in rows if not x["all_match"]]
    if bad:
        print("\nNOT A VALID COMPARISON (engines produced different text): " + ", ".join(bad))
        print("A row only compares like with like when both engines generated the same tokens.")
    print("\nratio > 1.00 means ojas is faster. 'same out' compares the generated token ids.")
    print("RSS 'n/a' means the result file recorded no peak-memory sample, not that it was zero.")
    print("Load average is recorded because it moved these numbers more than any code change.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["/tmp/bench-*.json"])

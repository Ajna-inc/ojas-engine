#!/usr/bin/env python3
"""Evaluate decision models on tasks with known answers, and compare them.

For each model, `ojas serve` is started on it and every task item is sent to
`POST /v1/systemone`. The raw answers are written to `<out>/<model>.jsonl`, then
scored per task:

  accuracy     the most probable option is the right one (a noul: p >= 0.5 for
               true); for a score, also `within one` level of the right one;
  brier        mean squared error of the probabilities against the right answer;
  log loss     mean negative log probability of the right answer;
  ece          expected calibration error: how far the stated confidence is from
               the accuracy actually reached, over ten confidence bins;
  baseline     the accuracy of always answering the task's most common answer;
  ci95         the half-width of the accuracy's 95% Wilson interval.

and across tasks:

  position     how much more often a model picks the first (or last) listed option
               than the right answers sit there: its bias toward a position;
  consistency  for a robustness task, how often the answer to a rewritten item
               (options reordered, instructions reworded, keys shuffled) is the
               answer the model gave to the original.

The summary is written to `<out>/summary.json` and printed as tables: accuracy by
category and by task, calibration, and throughput.

    run.py <model.gguf>... [--tasks a,b] [--limit N] [--clients 4] [--out DIR]
    run.py --report DIR                 # score answers already written
"""

import argparse
import concurrent.futures as cf
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geo import World  # noqa: E402
from tasks import all_tasks  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- running

class Server:
    """`ojas serve` on one model, stopped on exit."""

    def __init__(self, binary, model, port):
        self.url = f"http://127.0.0.1:{port}"
        self.proc = subprocess.Popen([str(binary), "serve", str(model), "--port", str(port)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"ojas serve exited while loading {model}")
            try:
                urllib.request.urlopen(f"{self.url}/health", timeout=2)
                return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        raise RuntimeError(f"{model} did not load in time")

    def props(self):
        return json.loads(urllib.request.urlopen(f"{self.url}/props", timeout=10).read())

    def decide(self, body):
        req = urllib.request.Request(f"{self.url}/v1/systemone", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.perf_counter()
        try:
            out = json.loads(urllib.request.urlopen(req, timeout=900).read())
            return out, time.perf_counter() - start, None
        except urllib.error.HTTPError as e:
            return None, time.perf_counter() - start, f"{e.code}: {e.read().decode(errors='replace')[:200]}"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()


def run_model(binary, model, tasks, out_dir, clients, port):
    with Server(binary, model, port) as server:
        props = server.props()
        name = props["model"]
        rows = []
        jobs = [(task, item) for task in tasks for item in task.items]
        start = time.perf_counter()
        with cf.ThreadPoolExecutor(clients) as pool:
            for (task, item), (out, seconds, error) in zip(jobs, pool.map(lambda j: server.decide(j[1].request()), jobs)):
                rows.append({"task": task.name, "category": task.category, "item": item.id, "variant_of": item.variant_of,
                             "questions": item.questions, "gold": item.gold,
                             "answers": out["answers"] if out else None,
                             "input_tokens": out["usage"]["input_tokens"] if out else None,
                             "seconds": seconds, "error": error})
        wall = time.perf_counter() - start
    path = out_dir / f"{name}.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"model": name, "decision_type": props.get("decision_type"), "path": str(model),
                            "wall_seconds": wall, "clients": clients}) + "\n")
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {name}: {len(rows)} items in {wall:.1f} s", file=sys.stderr)
    return path


# ---------------------------------------------------------------- scoring

def predicted(question, answer):
    """The answer a model gives: the most probable option, level or truth value."""
    if question["type"] == "noul":
        return answer["noul"] >= 0.5
    probs = answer["probabilities"]
    return max(probs, key=probs.get)


def score_answer(question, gold, answer):
    """`(correct, within_one, brier, log_loss, confidence)` for one answer."""
    kind = question["type"]
    if kind == "noul":
        p, y = answer["noul"], 1.0 if gold else 0.0
        correct = (p >= 0.5) == bool(gold)
        p_gold = p if gold else 1.0 - p
        return correct, correct, (p - y) ** 2, -math.log(max(p_gold, 1e-9)), max(p, 1.0 - p)
    probs = answer["probabilities"]
    key = str(gold)
    best = predicted(question, answer)
    brier = sum((p - (1.0 if k == key else 0.0)) ** 2 for k, p in probs.items())
    correct = best == key
    within = correct
    if kind == "score":
        within = abs(answer["score"] - int(gold)) <= 1.0
    return correct, within, brier, -math.log(max(probs.get(key, 0.0), 1e-9)), probs[best]


def wilson(k, n, z=1.96):
    """Half-width of the 95% Wilson score interval for `k` successes in `n`."""
    if n == 0:
        return float("nan")
    p = k / n
    return z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)


def ece(pairs, bins=10):
    """Expected calibration error over `(confidence, correct)` pairs."""
    if not pairs:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        group = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if group:
            conf = sum(c for c, _ in group) / len(group)
            acc = sum(ok for _, ok in group) / len(group)
            total += len(group) / len(pairs) * abs(conf - acc)
    return total


def summarize(path):
    lines = path.read_text().splitlines()
    head = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:]]
    answers = {}  # item id -> {question id: predicted answer}
    for row in rows:
        if row["answers"] is not None:
            answers[row["item"]] = {qid: predicted(q, row["answers"][qid]) for qid, q in row["questions"].items()}
    by_task = {}
    position = {"first": [0, 0], "last": [0, 0], "n": 0}  # [model picks, gold sits]
    for row in rows:
        t = by_task.setdefault(row["task"], {"category": row["category"], "n": 0, "errors": 0, "correct": 0, "within": 0,
                                             "brier": 0.0, "log_loss": 0.0, "pairs": [], "golds": [], "seconds": 0.0,
                                             "tokens": 0, "same": 0, "paired": 0})
        for qid, question in row["questions"].items():
            gold = row["gold"][qid]
            t["golds"].append(str(gold))
            t["n"] += 1
            t["seconds"] += row["seconds"]
            if row["answers"] is None:
                t["errors"] += 1
                continue
            t["tokens"] += row["input_tokens"] or 0
            correct, within, brier, loss, conf = score_answer(question, gold, row["answers"][qid])
            t["correct"] += correct
            t["within"] += within
            t["brier"] += brier
            t["log_loss"] += loss
            t["pairs"].append((conf, correct))
            if row.get("variant_of"):
                original = answers.get(row["variant_of"])
                if original is not None:
                    t["paired"] += 1
                    t["same"] += original[qid] == answers[row["item"]][qid]
            elif question["type"] == "choice":
                keys = list(question["criteria"])
                pick = answers[row["item"]][qid]
                position["n"] += 1
                position["first"][0] += pick == keys[0]
                position["first"][1] += str(gold) == keys[0]
                position["last"][0] += pick == keys[-1]
                position["last"][1] += str(gold) == keys[-1]
    tasks = {}
    for name, t in by_task.items():
        n = t["n"]
        majority = max(t["golds"].count(g) for g in set(t["golds"])) / n
        tasks[name] = {
            "category": t["category"], "items": n, "errors": t["errors"],
            "accuracy": t["correct"] / n, "ci95": wilson(t["correct"], n), "within_one": t["within"] / n,
            "brier": t["brier"] / n, "log_loss": t["log_loss"] / n, "ece": ece(t["pairs"]),
            "baseline": majority, "seconds_per_item": t["seconds"] / n, "tokens_per_item": t["tokens"] / max(n - t["errors"], 1),
            "consistency": t["same"] / t["paired"] if t["paired"] else None,
        }
    core = {k: v for k, v in tasks.items() if v["category"] != "robustness"}
    categories = {}
    for t in core.values():
        categories.setdefault(t["category"], []).append(t["accuracy"])
    robust = [v["consistency"] for v in tasks.values() if v["consistency"] is not None]
    n_pos = max(position["n"], 1)
    return {
        "model": head["model"], "decision_type": head.get("decision_type"), "wall_seconds": head["wall_seconds"],
        "items": sum(t["items"] for t in tasks.values()),
        "accuracy": sum(t["accuracy"] for t in core.values()) / len(core),
        "baseline": sum(t["baseline"] for t in core.values()) / len(core),
        "brier": sum(t["brier"] for t in core.values()) / len(core),
        "ece": sum(t["ece"] for t in core.values()) / len(core),
        "consistency": sum(robust) / len(robust) if robust else None,
        "position_bias": {side: (position[side][0] - position[side][1]) / n_pos for side in ("first", "last")},
        "categories": {c: sum(v) / len(v) for c, v in categories.items()},
        "tasks": tasks,
    }


# ---------------------------------------------------------------- report

def table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(r, widths))) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])


def report(summaries):
    models = [s["model"] for s in summaries]
    pct = lambda x: "-" if x is None or x != x else f"{100 * x:.1f}"
    with_ci = lambda t: f"{100 * t['accuracy']:.0f}±{100 * t['ci95']:.0f}"
    first = summaries[0]["tasks"]
    out = []
    cats = sorted({c for s in summaries for c in s["categories"]})
    out.append("Accuracy by category (%), macro over tasks\n")
    out.append(table(["category"] + models,
                     [[c] + [pct(s["categories"].get(c)) for s in summaries] for c in cats]
                     + [["all tasks"] + [pct(s["accuracy"]) for s in summaries],
                        ["most-common baseline"] + [pct(s["baseline"]) for s in summaries]]))

    names = sorted({t for s in summaries for t in s["tasks"]}, key=lambda t: (first.get(t, {}).get("category", ""), t))
    core = [t for t in names if first.get(t, {}).get("category") != "robustness"]
    out.append("\nAccuracy by task (%, with the 95% interval's half-width)\n")
    out.append(table(["task", "items", "baseline"] + models,
                     [[t, first[t]["items"], pct(first[t]["baseline"])] + [with_ci(s["tasks"][t]) if t in s["tasks"] else "-" for s in summaries]
                      for t in core]))

    robust = [t for t in names if first.get(t, {}).get("category") == "robustness"]
    if robust:
        out.append("\nRobustness: answers unchanged by a rewrite that keeps the right answer (%)\n")
        out.append(table(["rewrite"] + models,
                         [[t] + [pct(s["tasks"][t]["consistency"]) if t in s["tasks"] else "-" for s in summaries] for t in robust]
                         + [["all rewrites"] + [pct(s["consistency"]) for s in summaries]]))

    out.append("\nPosition bias: how much more often the first or last option is picked than it is right (points)\n")
    out.append(table(["option"] + models,
                     [[side] + [f"{100 * s['position_bias'][side]:+.1f}" for s in summaries] for side in ("first", "last")]))

    out.append("\nCalibration and cost\n")
    out.append(table(["model", "type", "brier", "ece", "items", "wall s", "items/s"],
                     [[s["model"], s["decision_type"], f"{s['brier']:.3f}", f"{s['ece']:.3f}", s["items"],
                       f"{s['wall_seconds']:.1f}", f"{s['items'] / s['wall_seconds']:.1f}"] for s in summaries]))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="*", type=Path, help="decision model GGUFs to evaluate")
    parser.add_argument("--tasks", help="comma-separated task names (default: all)")
    parser.add_argument("--limit", type=int, help="most items per task")
    parser.add_argument("--clients", type=int, default=4, help="requests in flight at once")
    parser.add_argument("--out", type=Path, default=Path("decision-eval-results"))
    parser.add_argument("--ojas", type=Path, default=REPO / "target/release/ojas")
    parser.add_argument("--port", type=int, default=8210)
    parser.add_argument("--report", type=Path, help="score the answers already in this directory")
    args = parser.parse_args()

    if args.report:
        paths = sorted(args.report.glob("*.jsonl"))
    else:
        if not args.models:
            parser.error("name at least one model, or --report a directory")
        tasks = all_tasks(World())
        if args.tasks:
            wanted = set(args.tasks.split(","))
            tasks = [t for t in tasks if t.name in wanted]
        if args.limit:
            for t in tasks:
                t.items = t.items[: args.limit]
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"{sum(len(t.items) for t in tasks)} items in {len(tasks)} tasks, {len(args.models)} models", file=sys.stderr)
        paths = [run_model(args.ojas, m, tasks, args.out, args.clients, args.port) for m in args.models]
        args.report = args.out
    summaries = [summarize(p) for p in paths]
    (args.report / "summary.json").write_text(json.dumps(summaries, indent=1))
    print(report(summaries))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-layer parity over trace dumps. All arithmetic is delegated to compare_logits.py.

Two shapes of input are accepted:

  directories  one file per traced tensor, as written by reference_probe's
               OJAS_REFERENCE_TRACE / OJAS_REFERENCE_VISION_TRACE callback
               ("<index>-<tensor>.f32", with "<tensor>~k" for the k-th occurrence
               of a name the graph reuses). Files are paired by name and each pair
               is compared independently.
  files        a single stacked trace, e.g. parity_dump's
               "<dump>/prompt<i>.trace.f32" of [n_layers+1][T][d]. Pass --rows.

This script deliberately contains no RMSE or cosine code of its own: every number
below comes from running compare_logits.py on one pair. What it adds is the loop,
the pairing, and a gate suited to activations rather than logits -- the argmax
equality compare_logits.py folds into its own verdict is meaningful for a logit row
and is not meaningful for a hidden state, so it is reported but only enforced with
--require-argmax. Integer tensors (.i32, routing indices and the like) are compared
byte for byte, because a near miss on an index is not a near miss.

Thresholds are explicit and must be fixed before a run, exactly as in
compare_logits.py. A failure stays a failure.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('reference', type=Path, help='reference trace directory or file')
    p.add_argument('actual', type=Path, help='actual trace directory or file')
    p.add_argument('--output', type=Path, required=True, help='result JSON path')
    p.add_argument('--rows', type=int, default=1,
                   help='rows each tensor is split into before comparison (default 1)')
    p.add_argument('--max-rmse', type=float, default=0.05)
    p.add_argument('--min-cosine', type=float, default=0.9999)
    p.add_argument('--require-argmax', action='store_true',
                   help='also require every row argmax to agree (logit-shaped dumps only)')
    p.add_argument('--include', default=None,
                   help='only compare tensors whose file name contains this substring')
    p.add_argument('--exclude', default=None,
                   help='skip tensors whose file name contains this substring')
    p.add_argument('--allow-missing', action='store_true',
                   help='report reference tensors absent from --actual instead of failing')
    p.add_argument('--comparator', type=Path, default=HERE / 'compare_logits.py',
                   help='path to compare_logits.py')
    return p.parse_args(argv)


def key_of(path):
    """Pairing key. Trace files are '<pos>-<tensor>.f32'; the position is the token
    or graph-pass index and may legitimately differ between two implementations, so
    the tensor name is the stable half."""
    stem = path.name
    head, sep, tail = stem.partition('-')
    return tail if sep and head.isdigit() else stem


def index(directory):
    out = {}
    for path in sorted(directory.iterdir()):
        if path.suffix not in ('.f32', '.i32') or not path.is_file():
            continue
        out.setdefault(key_of(path), []).append(path)
    return out


def compare_pair(args, name, reference, actual):
    """One tensor. Returns a result dict; never raises on a comparison failure."""
    if reference.suffix == '.i32' or actual.suffix == '.i32':
        same = reference.read_bytes() == actual.read_bytes()
        return dict(tensor=name, kind='i32', passed=same, exact=same,
                    reference=str(reference), actual=str(actual))
    with tempfile.NamedTemporaryFile('r+', suffix='.json', delete=True) as handle:
        command = [sys.executable, str(args.comparator), str(reference), str(actual),
                   '--rows', str(args.rows),
                   '--max-rmse', repr(args.max_rmse),
                   '--min-cosine', repr(args.min_cosine),
                   '--output', handle.name]
        done = subprocess.run(command, capture_output=True, text=True)
        handle.seek(0)
        try:
            report = json.loads(handle.read())
        except ValueError:
            return dict(tensor=name, kind='f32', passed=False,
                        error=(done.stderr.strip() or done.stdout.strip()
                               or f'comparator exit {done.returncode}'),
                        reference=str(reference), actual=str(actual))
    rows = report['rows']
    worst_rmse = max(r['rmse'] for r in rows)
    worst_cosine = min(r['cosine'] for r in rows)
    worst_abs = max(r['max_abs'] for r in rows)
    argmax_matches = sum(r['reference_argmax'] == r['actual_argmax'] for r in rows)
    passed = (worst_rmse <= args.max_rmse and worst_cosine >= args.min_cosine
              and (not args.require_argmax or argmax_matches == len(rows)))
    return dict(tensor=name, kind='f32', passed=passed, rows=len(rows),
                width=report['vocab'], max_rmse=worst_rmse, min_cosine=worst_cosine,
                max_abs=worst_abs, argmax_matches=argmax_matches,
                reference_sha256=report['reference_sha256'],
                actual_sha256=report['actual_sha256'],
                reference=str(reference), actual=str(actual))


def main(argv=None):
    args = parse_args(argv)
    if not args.comparator.is_file():
        raise SystemExit(f'comparator not found: {args.comparator}')
    if args.rows < 1:
        raise SystemExit('--rows must be at least 1')

    results, missing = [], []
    if args.reference.is_dir() != args.actual.is_dir():
        raise SystemExit('reference and actual must both be directories or both be files')
    if args.reference.is_dir():
        left, right = index(args.reference), index(args.actual)
        for name in sorted(left):
            if args.include and args.include not in name:
                continue
            if args.exclude and args.exclude in name:
                continue
            if len(left[name]) != 1:
                raise SystemExit(f'ambiguous reference tensor {name}: {left[name]}')
            if name not in right:
                missing.append(name)
                continue
            if len(right[name]) != 1:
                raise SystemExit(f'ambiguous actual tensor {name}: {right[name]}')
            results.append(compare_pair(args, name, left[name][0], right[name][0]))
        extra = sorted(set(right) - set(left))
    else:
        results.append(compare_pair(args, args.reference.name, args.reference, args.actual))
        extra = []

    if not results and not missing:
        raise SystemExit('no comparable tensors found')
    passed = (all(r['passed'] for r in results) and results
              and (args.allow_missing or not missing))
    finite = [r for r in results if r['kind'] == 'f32' and 'min_cosine' in r]
    summary = dict(
        passed=bool(passed),
        compared=len(results),
        failed=sorted(r['tensor'] for r in results if not r['passed']),
        missing=missing,
        unmatched_actual=extra,
        thresholds=dict(max_rmse=args.max_rmse, min_cosine=args.min_cosine,
                        rows=args.rows, require_all_argmax_equal=args.require_argmax),
        # None rather than NaN: the result file has to stay strict JSON.
        worst=dict(
            max_rmse=max((r['max_rmse'] for r in finite), default=None),
            min_cosine=min((r['min_cosine'] for r in finite), default=None),
            max_abs=max((r['max_abs'] for r in finite), default=None),
        ),
        comparator=str(args.comparator),
        tensors=results,
    )
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in
                      ('passed', 'compared', 'failed', 'missing', 'worst')}))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())

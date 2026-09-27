#!/usr/bin/env python3
"""Attribute nested Flash trace timings without double-counting parent intervals."""
import argparse
import json
from pathlib import Path


def contains(parent, child):
    return (child['start_s'] >= parent['start_s'] and
            child['start_s'] + child['seconds'] <=
            parent['start_s'] + parent['seconds'] + 1e-7)


def summarize(events, roots):
    selected = [e for e in events if any(contains(root, e) for root in roots)]
    result = {}
    for name in sorted({e['name'] for e in selected}):
        rows = [e for e in selected if e['name'] == name]
        stats = {key: sum(e[key] for e in rows) for key in
                 ('seconds', 'rows', 'hits', 'lookups', 'gpu_s', 'gather_s',
                  'copy_s', 'read_s', 'admit_s', 'encode_s', 'wait_s', 'commands')}
        stats['calls'] = len(rows)
        stats['cache_hit_pct'] = (100 * stats['hits'] / stats['lookups']
                                  if stats['lookups'] else None)
        stats['misses'] = stats['lookups'] - stats['hits']
        stats['direct_expert_bytes'] = sum(e.get('direct_expert_bytes', 0) for e in rows)
        if name == 'scalar':
            stats['catchup_child_s'] = sum(e['seconds'] for e in selected
                if e['name'] == 'catchup' and any(contains(r, e) for r in rows))
        result[name] = stats
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    results = json.loads((args.directory / 'results.json').read_text())
    report = {'note': 'Parent durations include children. GPU time is inside wait_s, '
              'and copy/read/admission are inside gather_s; do not add them twice. '
              'after_first_call excludes the initial multi-token call as a whole, '
              'so its committed-token count need not equal the benchmark TPS numerator.',
              'samples': []}
    for case in results['cases']:
        for sample in case['samples']:
            path = args.directory / f"{case['label']}-{sample['run']}.jsonl"
            events = [json.loads(line) for line in path.read_text().splitlines()]
            roots = [e for e in events if e['name'] in ('mtp', 'scalar') and
                     not any(p['name'] == 'mtp' and p is not e and contains(p, e)
                             for p in events)]
            roots.sort(key=lambda e: e['start_s'])
            report['samples'].append({'label': case['label'], 'run': sample['run'],
                'decode_tps': sample['decode_tps'],
                'whole_request': summarize(events, [dict(start_s=0, seconds=float('inf'))]),
                'after_first_call': summarize(events, roots[1:])})
    (args.directory / 'timing-summary.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

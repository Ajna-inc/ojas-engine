#!/usr/bin/env python3
"""Sequential serving benchmarks with repeated samples and token agreement gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--binary', type=Path, required=True)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.work.mkdir(parents=True, exist_ok=True)
a.output.parent.mkdir(parents=True, exist_ok=True)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
items = json.loads(a.manifest.read_text())
if not items:
    raise SystemExit('empty benchmark manifest')
report = {'binary_sha256': sha(a.binary), 'cases': [],
          'scope': 'Fresh recurrent state per request, one warmup, repeated warm-cache samples. Load time is process initialization with uncontrolled filesystem-cache warmth. Load average is retained; these are not other-engine comparisons. An ocr case is a PER-PAGE measurement: the vision encode is inside the timed window of every sample, so total_s is encode + prefill + decode and prefill alone is ttft_s - encode_s.'}
outputs = {}
for item in items:
    label, mode, image = item['label'], item['mode'], item.get('image')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', label) or mode not in ('plain', 'mtp', 'ocr'):
        raise SystemExit('invalid benchmark label or mode')
    # The image is what makes a case an OCR case, so the two must not be able to
    # disagree: an ocr case without one would benchmark a prompt of placeholder
    # ids, and a text case with one would silently ignore the page.
    if (mode == 'ocr') != bool(image):
        raise SystemExit(f'{label}: mode ocr requires an "image", and only mode ocr may carry one')
    if sha(a.binary) != report['binary_sha256']:
        raise SystemExit('benchmark binary changed during the run')
    env = {k: v for k, v in os.environ.items() if not k.startswith('OJAS_')}
    env.update(item.get('env', {}))
    if mode == 'plain':
        env['OJAS_NO_SPEC'] = '1'
    else:
        env.pop('OJAS_NO_SPEC', None)
    reps = item.get('reps', 3)
    # Per-case, because one page through the CPU ViT is minutes, not seconds, and
    # the encode runs once per repetition plus once up front. The default is
    # unchanged for every text case.
    timeout = int(item.get('timeout', 1200))
    log = a.work / (label + '.log')
    with log.open('w') as f:
        try:
            code = subprocess.run([str(a.binary.resolve()), item['model'], item['prompt'], mode,
                                   str(item.get('tokens', 32)), str(reps)] + ([image] if image else []),
                                  env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            code = 124
    row = {'label': label, 'mode': mode, 'model': Path(item['model']).name,
           'prompt_sha256': sha(item['prompt']), 'environment': {k: v for k, v in env.items() if k.startswith('OJAS_')},
           'image': Path(image).name if image else None, 'image_sha256': sha(image) if image else None,
           'timeout_s': timeout, 'exit_code': code, 'log_sha256': sha(log), 'passed': False}
    try:
        records = [json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
        meta = next(r for r in records if 'load_s' in r)
        samples = [r for r in records if 'run' in r]
        row.update(metadata=meta, samples=samples)
        assert code == 0 and len(samples) == reps and meta['mode'] == mode
        # The image and the RESOLVED token budget are part of the workload: the
        # same page at 4096 and at 2048 merged tokens is two different prompts and
        # must not be gated against each other. Everything else about the key is
        # unchanged, so a plain case and an mtp case on the same text are still
        # required to agree — speculation is exact.
        key = (str(Path(item['model']).resolve()), row['prompt_sha256'], item.get('tokens', 32),
               row['image_sha256'], meta['image_tokens'])
        for sample in samples:
            # MTP engagement is only a claim about the two text modes.
            if mode != 'ocr':
                assert (sample['mtp_calls'] == 0) == (mode == 'plain')
            assert sample['output_tokens'] == outputs.setdefault(key, sample['output_tokens'])
            # The ViT is pure, so repeated encodes of one page must hash the same.
            assert sample['rows_fnv'] == samples[0]['rows_fnv']
        if item.get('reference_ids'):
            ids = list(map(int, Path(item['reference_ids']).read_text().split()))
            assert samples[0]['output_tokens'] == ids[ids[0]+1:ids[0]+1+item.get('tokens', 32)]
        row['summary'] = {name: {'median': statistics.median(s[name] for s in samples),
                                 'min': min(s[name] for s in samples), 'max': max(s[name] for s in samples)}
                          for name in ('total_s', 'ttft_s', 'encode_s', 'decode_tps')}
        row['all_load_averages_below_4'] = all(s['load_average'] is not None and s['load_average'] < 4 for s in samples)
        row['passed'] = True
    except (AssertionError, KeyError, ValueError, StopIteration) as error:
        row['error'] = f'benchmark/output gate failed: {error}'
    report['cases'].append(row)
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    print(label, 'PASS' if row['passed'] else 'FAIL', flush=True)
raise SystemExit(0 if all(r['passed'] for r in report['cases']) else 1)

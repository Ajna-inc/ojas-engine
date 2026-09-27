#!/usr/bin/env python3
"""Interleaved, reference-checked Flash snapshot and partial-acceptance costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--reference-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--labels', nargs='+', choices=['france', 'code', 'repeat'], default=['france', 'code', 'repeat'])
    p.add_argument('--modes', nargs='+', choices=['draft', 'oracle'], default=['draft'])
    p.add_argument('--widths', type=int, nargs='+', default=[2, 3, 4])
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--cache-gb', type=int, default=16)
    p.add_argument('--binary', type=Path, default=Path('target/release/examples/flash_prefix_bench'))
    a = p.parse_args()
    if not 3 <= a.reps <= 20 or a.cache_gb <= 0 or not all(2 <= w <= 4 for w in a.widths) or len(set(a.widths)) != len(a.widths):
        p.error('require reps 3..20, positive cache, distinct widths 2..4')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    b = a.binary.resolve()
    report = dict(binary_sha256=sha(b), source_sha256={str(f): sha(f) for f in sorted(Path('crates').rglob('*.rs'))},
                  platform=platform.platform(), model_entrypoint_sha256=sha(a.model),
                  model_hash_scope='Entrypoint only; use the existing complete checkpoint manifest.',
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  timing_scope='32 outputs excluding prefill; one warmup per width/policy, rotated measured order. Draft mode forces speculation, not the adaptive serving scheduler. Both policies keep snapshot allocation reserved.', cases=[])
    for label in a.labels:
        for mode in a.modes:
            assert sha(b) == report['binary_sha256'], 'binary changed'
            settings = dict(OJAS_MTP_PREFIX='1', OJAS_MTP_DRAFT='3', OJAS_EXPERT_COPY_THREADS='4', OJAS_EXPERT_CACHE_GB=str(a.cache_gb))
            if mode == 'oracle':
                settings['OJAS_NO_SPEC'] = '1'
            env = {k: v for k, v in os.environ.items() if not k.startswith('OJAS_')}
            env.update(settings)
            ref = a.reference_dir / f'{label}.ids'
            log = a.output_dir / f'{label}-{mode}.log'
            before = subprocess.check_output(['vm_stat'], text=True)
            with log.open('w') as stream:
                try:
                    code = subprocess.run([str(b), a.model, str(ref), mode, str(a.reps), ','.join(map(str, a.widths))], env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=1200).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            lines = log.read_text().splitlines()
            rows = [json.loads(s) for s in lines if s.startswith('{')]
            passed = code == 0 and len(rows) == a.reps*2*len(a.widths) and all(
                len([r for r in rows if r['width'] == w and r['prefix'] == policy]) == a.reps
                for w in a.widths for policy in [False, True])
            case = dict(label=label, mode=mode, exit_code=code, passed=passed, environment=settings,
                        reference_sha256=sha(ref), log_sha256=sha(log), records=rows,
                        allocation_lines=[s for s in lines if s.startswith('snapshot_bytes=')],
                        memory_before=before, memory_after=subprocess.check_output(['vm_stat'], text=True), summary={})
            for w in a.widths:
                for policy in [False, True]:
                    samples = [r for r in rows if r['width'] == w and r['prefix'] == policy]
                    if samples:
                        case['summary'][f'width={w},prefix={policy}'] = {
                            k: dict(median=statistics.median(r[k] for r in samples), min=min(r[k] for r in samples), max=max(r[k] for r in samples))
                            for k in ['tps', 'seconds', 'target_s', 'catchup_s', 'draft_s', 'rollback_s', 'calls', 'partial_calls', 'kept_drafts']}
            report['cases'].append(case)
            (a.output_dir/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(label, mode, 'PASS' if passed else 'FAIL', flush=True)
            if not passed:
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

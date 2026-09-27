#!/usr/bin/env python3
"""Summarize the fixed-budget, two-seed backbone-initialization experiment."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(sys.argv[1]).resolve()
contract = json.loads((root / 'evaluation_contract.json').read_text())
runs = {}
configs = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for kind in ['coco', 'indian']:
    for seed in [1, 2]:
        name = f'{kind}_seed{seed}'
        run = root / name
        config = json.loads((run / 'config.json').read_text())
        provenance = json.loads((run / 'provenance.json').read_text())
        history = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()]
        assert provenance['exit_code'] == 0
        assert history[-1]['epoch'] == 8 and history[-1]['step'] == 1984
        checkpoint = run / 'final.safetensors'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == provenance['final.safetensors_sha256']
        configs[name] = {k: v for k, v in config.items() if k not in ['init', 'out', 'seed']}
        runs[name] = {'final_diagnostic_metrics': history[-1]['metrics'],
                      'seconds': history[-1]['elapsed_seconds'],
                      'weight_bytes': checkpoint.stat().st_size,
                      'checkpoint_sha256': provenance['final.safetensors_sha256'],
                      'init_sha256': provenance['init_sha256'],
                      'binary_sha256': provenance['binary_sha256'],
                      'manifest_sha256': provenance['manifest_sha256']}
        for ax, key in zip(axes, ['accuracy', 'macro_f1']):
            ax.plot([h['epoch'] for h in history], [h['metrics'][key] for h in history],
                    label=f'{kind}, seed {seed}', color='tab:blue' if kind == 'coco' else 'tab:orange',
                    linestyle='-' if seed == 1 else '--')
            ax.set(xlabel='Epoch', ylabel=f'Crop diagnostic {key}', ylim=(0, 1))
            ax.grid(alpha=.2)
assert all(c == configs['coco_seed1'] for c in configs.values())
for field in ['binary_sha256', 'manifest_sha256', 'weight_bytes']:
    assert len({r[field] for r in runs.values()}) == 1, field
axes[1].legend(fontsize=8)
fig.suptitle('Training-pool diagnostic only: Indian pretraining includes crop development sources')
fig.savefig(root / 'learning_curves.png', dpi=160)
plt.close(fig)

summary = {'training': runs, 'selection': 'Fixed final epoch 8, no development selection',
           'caveat': contract['caveat']}
reports = {}
for arm in contract['arms']:
    name = arm['name']
    report_path = root / f'{name}_fresh_boxes.json'
    if not report_path.exists():
        continue
    r = json.loads(report_path.read_text())
    assert [row['id'] for row in r['per_image']] == contract['image_ids']
    assert sum(row['expert_matched'] for row in r['per_image']) == r['expert_matched']
    assert sum(row['expert_correct'] for row in r['per_image']) == r['expert_correct']
    cm = np.array(r['expert_confusion'])
    assert cm.sum() == r['expert_matched'] and cm.trace() == r['expert_correct']
    reports[name] = r
if reports:
    assert len(reports) == 4, 'Finish all arms before comparing evaluation results'
    base = reports['coco_seed1']
    for r in reports.values():
        assert r['detector_ap'] == base['detector_ap']
        assert r['before_confusion'] == base['before_confusion']
        assert [(x['id'], x['expert_matched'], x['expert_baseline_correct']) for x in r['per_image']] == [
            (x['id'], x['expert_matched'], x['expert_baseline_correct']) for x in base['per_image']]
    summary['evaluation'] = {}
    for name, r in reports.items():
        cm = np.array(r['expert_confusion'])
        den = cm.sum(axis=0) + cm.sum(axis=1)
        summary['evaluation'][name] = {k: r[k] for k in ['detector_ap', 'fused_ap', 'localized',
            'detector_correct', 'fused_correct', 'routed_correct', 'expert_calls', 'expert_matched',
            'expert_correct', 'expert_baseline_correct', 'expert_confusion']}
        summary['evaluation'][name].update(
            expert_accuracy=r['expert_correct'] / r['expert_matched'],
            detector_accuracy_on_expert_objects=r['expert_baseline_correct'] / r['expert_matched'],
            expert_macro_f1=float(np.mean(np.divide(2 * cm.diagonal(), den, out=np.zeros(4), where=den > 0))),
            expert_per_class_recall=np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(4), where=cm.sum(axis=1) > 0).tolist())
    groups = {r['id']: r['group'] for r in json.loads((root / 'evaluation_groups.json').read_text())}
    comparisons = {}
    for seed in [1, 2]:
        left, right = reports[f'coco_seed{seed}'], reports[f'indian_seed{seed}']
        counts = {}
        for a, b in zip(left['per_image'], right['per_image']):
            counts.setdefault(groups[a['id']], np.zeros(4))[:] += np.array([
                a['expert_matched'], b['expert_correct'] - a['expert_correct'],
                a['matched'], b['fused'] - a['fused']])
        counts = np.array(list(counts.values()))
        rng = np.random.default_rng(20260920)
        samples = np.array([counts[rng.integers(len(counts), size=len(counts))].sum(axis=0) for _ in range(10000)])
        comparisons[f'indian_minus_coco_seed{seed}'] = {
            'groups': len(counts),
            'expert_accuracy_delta': float(counts[:, 1].sum() / counts[:, 0].sum()),
            'expert_accuracy_percentile95': np.percentile(samples[:, 1] / samples[:, 0], [2.5, 97.5]).tolist(),
            'fused_top1_delta': float(counts[:, 3].sum() / counts[:, 2].sum()),
            'fused_top1_percentile95': np.percentile(samples[:, 3] / samples[:, 2], [2.5, 97.5]).tolist(),
            'fused_ap_delta': right['fused_ap'] - left['fused_ap']}
    summary['paired_comparisons'] = comparisons
    summary['uncertainty'] = 'Paired automatic source-group bootstrap; not camera-disjoint. No AP interval or across-seed population interval.'
(root / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

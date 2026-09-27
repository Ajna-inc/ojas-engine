#!/usr/bin/env python3
"""Compare fixed-budget SAM and standard AdamW training on identical proposals."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path(sys.argv[1]).resolve()
contract = json.loads((root / 'evaluation_contract.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
groups = {r['id']: r['group'] for r in json.loads((root / 'evaluation_groups.json').read_text())}
summary = {'selection': contract['selection'], 'caveat': contract['caveat'], 'training': {}, 'evaluation': {}, 'paired': {}}
reports, configs, provenances = {}, [], []
for kind in ['adam', 'sam']:
    for seed in [1, 2]:
        name = f'{kind}_seed{seed}'
        run = root / name
        config = json.loads((run / 'config.json').read_text())
        prov = json.loads((run / 'provenance.json').read_text())
        history = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()]
        architecture = json.loads((run / 'architecture.json').read_text())
        train = json.loads((run / 'final_train_metrics.json').read_text())
        assert prov['exit_code'] == 0 and history[-1]['step'] == 1984 and history[-1]['epoch'] == 8
        assert architecture['backbone_stage'] == 3
        assert config['head_only_epochs'] == 0
        assert config['sam_rho'] == (0.05 if kind == 'sam' else 0)
        assert history[-1]['backbone_step'] == 1984
        assert history[-1]['backward_passes'] == (3968 if kind == 'sam' else 1984)
        if kind == 'sam':
            gate = json.loads((run / 'sam_gate.json').read_text())
            assert gate['exact_restoration'] and gate['same_augmented_batch']
            assert abs(gate['actual_delta_norm'] - .05) < 1e-4
        assert sha(run / 'final.safetensors') == prov['final.safetensors_sha256']
        configs.append({k: v for k, v in config.items() if k not in ['out', 'seed', 'sam_rho']})
        provenances.append(prov)
        summary['training'][name] = dict(architecture=architecture, train_metrics=train,
            diagnostic_dev_metrics=history[-1]['metrics'], seconds=history[-1]['elapsed_seconds'],
            checkpoint_bytes=(run / 'final.safetensors').stat().st_size,
            checkpoint_sha256=prov['final.safetensors_sha256'], history=history)
        report_path = root / f'{name}_fresh_boxes.json'
        r = json.loads(report_path.read_text())
        ep = json.loads((root / f'{name}_fresh_boxes_provenance.json').read_text())
        assert ep['expert_sha256'] == prov['final.safetensors_sha256']
        assert ep['report_sha256'] == sha(report_path)
        assert ep['evaluation_contract_sha256'] == sha(root / 'evaluation_contract.json')
        assert [row['id'] for row in r['per_image']] == contract['image_ids']
        cm = np.array(r['expert_confusion'])
        assert cm.sum() == r['expert_matched'] and cm.trace() == r['expert_correct']
        assert sum(x['expert_correct'] for x in r['per_image']) == r['expert_correct']
        assert sum(x['expert_matched'] for x in r['per_image']) == r['expert_matched']
        den = cm.sum(0) + cm.sum(1)
        summary['evaluation'][name] = dict(
            accuracy=r['expert_correct'] / r['expert_matched'], correct=r['expert_correct'], n=r['expert_matched'],
            macro_f1=float(np.mean(np.divide(2 * cm.diagonal(), den, out=np.zeros(4), where=den > 0))),
            per_class_recall=(cm.diagonal() / np.maximum(cm.sum(1), 1)).tolist(), confusion=cm.tolist(),
            detector_accuracy=r['expert_baseline_correct'] / r['expert_matched'],
            detector_ap=r['detector_ap'], fused_ap=r['fused_ap'],
            localized=r['localized'], fused_correct=r['fused_correct'], detector_correct=r['detector_correct'])
        reports[name] = r
assert all(c == configs[0] for c in configs)
for key in ['binary_sha256', 'manifest_sha256', 'init_sha256']:
    assert len({p[key] for p in provenances}) == 1, key
baseline = reports['adam_seed1']
for r in reports.values():
    assert r['detector_ap'] == baseline['detector_ap'] and r['before_confusion'] == baseline['before_confusion']
    assert [(x['id'], x['expert_matched'], x['expert_baseline_correct']) for x in r['per_image']] == [
        (x['id'], x['expert_matched'], x['expert_baseline_correct']) for x in baseline['per_image']]
for seed in [1, 2]:
    a, b = reports[f'adam_seed{seed}'], reports[f'sam_seed{seed}']
    counts = {}
    for x, y in zip(a['per_image'], b['per_image']):
        counts.setdefault(groups[x['id']], np.zeros(2))[:] += [x['expert_matched'], y['expert_correct'] - x['expert_correct']]
    counts = np.array(list(counts.values()))
    rng = np.random.default_rng(20260920)
    samples = np.array([counts[rng.integers(len(counts), size=len(counts))].sum(0) for _ in range(10000)])
    summary['paired'][f'sam_minus_adam_seed{seed}'] = dict(
        delta=float(counts[:, 1].sum() / counts[:, 0].sum()),
        percentile95=np.percentile(samples[:, 1] / samples[:, 0], [2.5, 97.5]).tolist(),
        groups=len(counts))
summary['uncertainty'] = 'Paired automatic source-group bootstrap, not camera independent; no AP uncertainty. Common proposal subset excludes missed and misrouted vehicles.'
# Fresh rho-zero controls must numerically reproduce the preceding experiment.
from safetensors.numpy import load_file
reproduction = {}
for seed in [1, 2]:
    old = load_file(str(root.parent / 'passenger_headfirst_2026_09_20' / f'joint_seed{seed}' / 'final.safetensors'))
    new = load_file(str(root / f'adam_seed{seed}' / 'final.safetensors'))
    assert old.keys() == new.keys() and all(np.array_equal(old[k], new[k]) for k in old)
    reproduction[f'seed{seed}'] = {'all_tensors_exact': True, 'tensors': len(old)}
summary['control_reproduction'] = reproduction
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, data in summary['training'].items():
    h = data['history']
    axes[0].plot([v['epoch'] for v in h], [100*v['metrics']['accuracy'] for v in h], label=name)
    axes[1].plot([v['epoch'] for v in h], [v['metrics']['nll'] for v in h], label=name)
axes[0].set_ylabel('Diagnostic development accuracy (%)')
axes[1].set_ylabel('Diagnostic development NLL')
for ax in axes:
    ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.suptitle('SAM versus AdamW: development sources seen during initialization pretraining')
fig.tight_layout(); fig.savefig(root / 'learning_curves.png', dpi=160); plt.close(fig)
(root / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'training'}, indent=2))

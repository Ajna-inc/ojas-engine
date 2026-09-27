#!/usr/bin/env python3
"""Compare fixed-budget head-first and immediate joint training on identical proposals."""
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
for kind in ['joint', 'headfirst']:
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
        expected_head_epochs = 2 if kind == 'headfirst' else 0
        assert config['head_only_epochs'] == expected_head_epochs
        assert history[-1]['backbone_step'] == (1488 if kind == 'headfirst' else 1984)
        assert all(h['head_only'] == (h['epoch'] <= expected_head_epochs) for h in history[1:])
        if kind == 'headfirst':
            gate = json.loads((run / 'head_phase_gate.json').read_text())
            assert gate['backbone_unchanged'] and gate['head_changed'] and gate['step'] == 496
            gate = json.loads((run / 'joint_phase_gate.json').read_text())
            assert gate['backbone_changed'] and gate['backbone_step'] == 1488
        assert sha(run / 'final.safetensors') == prov['final.safetensors_sha256']
        configs.append({k: v for k, v in config.items() if k not in ['out', 'seed', 'head_only_epochs']})
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
baseline = reports['joint_seed1']
for r in reports.values():
    assert r['detector_ap'] == baseline['detector_ap'] and r['before_confusion'] == baseline['before_confusion']
    assert [(x['id'], x['expert_matched'], x['expert_baseline_correct']) for x in r['per_image']] == [
        (x['id'], x['expert_matched'], x['expert_baseline_correct']) for x in baseline['per_image']]
for seed in [1, 2]:
    a, b = reports[f'joint_seed{seed}'], reports[f'headfirst_seed{seed}']
    counts = {}
    for x, y in zip(a['per_image'], b['per_image']):
        counts.setdefault(groups[x['id']], np.zeros(2))[:] += [x['expert_matched'], y['expert_correct'] - x['expert_correct']]
    counts = np.array(list(counts.values()))
    rng = np.random.default_rng(20260920)
    samples = np.array([counts[rng.integers(len(counts), size=len(counts))].sum(0) for _ in range(10000)])
    summary['paired'][f'headfirst_minus_joint_seed{seed}'] = dict(
        delta=float(counts[:, 1].sum() / counts[:, 0].sum()),
        percentile95=np.percentile(samples[:, 1] / samples[:, 0], [2.5, 97.5]).tolist(),
        groups=len(counts))
summary['uncertainty'] = 'Paired automatic source-group bootstrap, not camera independent; no AP uncertainty. Common proposal subset excludes missed and misrouted vehicles.'
(root / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k: v for k, v in summary.items() if k != 'training'}, indent=2))

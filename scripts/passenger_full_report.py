#!/usr/bin/env python3
"""Paired results for the full-pool and matched-proposal classifier screens."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('docs/evidence')
ROOT=BASE/'passenger_full_2026_09_20'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    groups={r['id']:r['group'] for r in json.loads((ROOT/'evaluation_groups.json').read_text())}
    summary={'runs':{},'paired':{},'caveats':[
        'Fixed-budget screen: 1984 updates and 31688 sampled crops per run, not eight full-pool passes.',
        'Two seeds; automatic source groups rather than independent cameras; ST is a consumed benchmark.',
        'Original IISc pretraining saw the development sources.',
        'Legacy fixed-alpha fusion for standalone classifiers is diagnostic; residual runs use their separately declared learned correction contract.',
        'Some training may overlap proposal-cache generation: elapsed times are not isolated throughput benchmarks.']}
    reports={}
    roots=[(ROOT,['small','full']),(BASE/'passenger_proposals_2026_09_20',['paired_gt','proposal']),
           (BASE/'passenger_residual_2026_09_21',['residual'])]
    for root,arms in roots:
        contract=json.loads((root/'evaluation_contract.json').read_text())
        for arm in arms:
            for seed in [1,2]:
                name=f'{arm}_seed{seed}'; run=root/name
                rp=root/f'{name}_fresh_boxes.json'
                if not rp.exists(): continue
                r=json.loads(rp.read_text()); reports[name]=r
                config=json.loads((run/'config.json').read_text())
                prov=json.loads((run/'provenance.json').read_text())
                evalprov=json.loads((root/f'{name}_fresh_boxes_provenance.json').read_text())
                history=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
                best=json.loads((run/'best.json').read_text())
                assert prov['exit_code']==0 and history[-1]['step']==1984
                assert history[-1]['backward_passes']==1984 and config['epoch_samples']==3961
                assert evalprov['expert_sha256']==sha(run/'best.safetensors')==prov['best.safetensors_sha256']
                assert evalprov['report_sha256']==sha(rp)
                assert evalprov['evaluation_contract_sha256']==sha(root/'evaluation_contract.json')
                assert prov['manifest_sha256']==sha(config['manifest'])
                assert [p['id'] for p in r['per_image']]==contract['image_ids']
                cm=np.array(r['expert_confusion']); assert cm.sum()==r['expert_matched']
                assert cm.trace()==r['expert_correct']
                expected=max(history if arm=='residual' else history[1:],key=lambda h:h['metrics']['macro_f1'])
                assert best['epoch']==expected['epoch']
                summary['runs'][name]=dict(correct=r['expert_correct'],n=r['expert_matched'],
                    accuracy=r['expert_correct']/r['expert_matched'],
                    baseline_correct=r['expert_baseline_correct'],baseline_accuracy=r['expert_baseline_correct']/r['expert_matched'],
                    macro_f1=float(np.mean(2*cm.diagonal()/np.maximum(cm.sum(0)+cm.sum(1),1))),
                    recalls=(cm.diagonal()/np.maximum(cm.sum(1),1)).tolist(),confusion=cm.tolist(),
                    original_ap=r['detector_ap'],fused_ap=r['fused_ap'],fusion_contract=r['fusion_contract'],
                    fused_correct_all_scene=r['fused_correct'],original_correct_all_scene=r['detector_correct'],
                    passenger_gt=r['passenger_gt'],localized=r['localized'],
                    best_epoch=best['epoch'],best_dev=best['metrics'],final_dev=history[-1]['metrics'],
                    final_train=json.loads((run/'final_train_metrics.json').read_text()) if (run/'final_train_metrics.json').exists() else None,
                    exposure=json.loads((run/'exposure.json').read_text()),
                    architecture=json.loads((run/'architecture.json').read_text()),
                    checkpoint_bytes=(run/'best.safetensors').stat().st_size,
                    seconds=history[-1]['elapsed_seconds'],history=history,checkpoint_sha256=sha(run/'best.safetensors'))
    assert reports, 'no completed evaluations'
    first=next(iter(reports.values()))
    zero=json.loads((BASE/'passenger_residual_2026_09_21/zero_guard_fresh_boxes.json').read_text())
    assert zero['fused_ap']==zero['detector_ap']==first['detector_ap']
    assert zero['expert_correct']==zero['expert_baseline_correct']==first['expert_baseline_correct']
    baseline_cm=np.array(zero['expert_confusion'])
    summary['baseline']=dict(correct=zero['expert_correct'],n=zero['expert_matched'],
        accuracy=zero['expert_correct']/zero['expert_matched'],confusion=baseline_cm.tolist(),
        recalls=(baseline_cm.diagonal()/baseline_cm.sum(1)).tolist(),AP=zero['detector_ap'])
    for name,r in reports.items():
        assert r['detector_ap']==first['detector_ap']
        assert [(x['id'],x['expert_matched'],x['expert_baseline_correct']) for x in r['per_image']]==[(x['id'],x['expert_matched'],x['expert_baseline_correct']) for x in first['per_image']]
    def bootstrap(a,b=None):
        counts={}
        for i,row in enumerate(a['per_image']):
            before=b['per_image'][i]['expert_correct'] if b else row['expert_baseline_correct']
            counts.setdefault(groups[row['id']],np.zeros(2))[:] += [row['expert_matched'],row['expert_correct']-before]
        counts=np.array(list(counts.values())); rng=np.random.default_rng(20260920)
        samples=np.array([counts[rng.integers(len(counts),size=len(counts))].sum(0) for _ in range(10000)])
        return dict(delta=float(counts[:,1].sum()/counts[:,0].sum()),ci95=np.percentile(samples[:,1]/samples[:,0],[2.5,97.5]).tolist(),groups=len(counts))
    for name,r in reports.items(): summary['paired'][name+'_minus_iisc']=bootstrap(r)
    for a,b in [('full','small'),('proposal','paired_gt'),('residual','proposal')]:
        for seed in [1,2]:
            x,y=f'{a}_seed{seed}',f'{b}_seed{seed}'
            if x in reports and y in reports: summary['paired'][x+'_minus_'+y]=bootstrap(reports[x],reports[y])
    summary['complete']=len(reports)==10
    latency_path=BASE/'passenger_residual_2026_09_21/crop_latency.json'
    if latency_path.exists(): summary['crop_latency']=json.loads(latency_path.read_text())
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Full-pool and actual-proposal classifier results','',
        ('All ten runs evaluated.' if summary['complete'] else f'PARTIAL: {len(reports)} of ten runs evaluated.'),
        'The original detector stays frozen. No candidate is promoted by this screen.',
        '','| Run | Correct / fixed cohort | Accuracy | Dev-selected interval | MUV recall |',
        '| --- | --- | --- | --- | --- |',
        f"| IISc original (Ojas BF16) | {first['expert_baseline_correct']} / {first['expert_matched']} | {100*first['expert_baseline_correct']/first['expert_matched']:.2f}% | — | {100*summary['baseline']['recalls'][3]:.2f}% |"]
    for n,r in summary['runs'].items(): lines.append(f"| {n} | {r['correct']} / {r['n']} | {100*r['accuracy']:.2f}% | {r['best_epoch']} | {100*r['recalls'][3]:.2f}% |")
    if summary['complete']:
        lines+=['','## Decision','',
            'No convincing improvement over IISc; 80% was not reached. The learned',
            'correction adds four correct predictions in seed 1 and none in seed 2.',
            'Both paired intervals include zero, and both learned-correction AP values',
            'are below the original. The original checkpoint remains the deployment candidate.',
            '', 'The first correction improves MUV recall from 46.54% to 55.97% while',
            'hatchback recall falls from 81.56% to 77.19%. This is primarily a class',
            'tradeoff in these results, not the broad improvement required for 80%.',
            '', 'The full-pool screen sampled 21,646 / 21,770 unique training crops out',
            'of 47,784. It does not establish that fully training the expanded pool',
            'would fail. Final-checkpoint exposure diagnostics are recorded separately',
            'from development-selected-checkpoint benchmark results.',
            '', '## All-class AP and whole-scene classification','',
            '| Run | All-class AP | Correct subtype / all 1,690 passenger GT | Fusion |',
            '| --- | --- | --- | --- |',
            f"| Original | {zero['detector_ap']:.6f} | {zero['detector_correct']} / 1690 | None |"]
        for n,r in summary['runs'].items():
            lines.append(f"| {n} | {r['fused_ap']:.6f} | {r['fused_correct_all_scene']} / 1690 | {'Learned correction' if n.startswith('residual') else 'Legacy fixed alpha .25 diagnostic'} |")
        lines+=['','Some legacy fixed-alpha AP scores increase slightly. No AP confidence',
            'intervals or independent deployment confirmation were established; these',
            'diagnostic gains do not establish the requested subtype accuracy.']
    lines+=['','This Ojas BF16 cohort has 1,271 objects. The previous full-detector PyTorch','cohort had 1,270 objects and a 69.21% baseline; compare each candidate only','against the baseline in its own execution stack, not across those percentages.','','## Paired differences','']
    for n,r in summary['paired'].items(): lines.append(f"- {n}: {100*r['delta']:+.2f} points; source-group 95% interval [{100*r['ci95'][0]:+.2f}, {100*r['ci95'][1]:+.2f}].")
    lines+=['','## Scope','']+['- '+c for c in summary['caveats']]
    if 'crop_latency' in summary:
        lines+=['','## Measured cost on RTX 3060','',
            'The correction branch adds 11,202,020 parameters and a 44,862,000-byte',
            'checkpoint. The original detector has 20,101,004 parameters.',
            '', '| Crops per batch | Upload + forward + download p50 / p95 ms | Crop processing p50 / p95 ms |',
            '| --- | --- | --- |']
        for r in summary['crop_latency']['results']:
            lines.append(f"| {r['batch']} | {r['upload_forward_download_p50_ms']:.2f} / {r['upload_forward_download_p95_ms']:.2f} | {r['crop_pipeline_p50_ms']:.2f} / {r['crop_pipeline_p95_ms']:.2f} |")
        lines+=['','Isolated GPU, actual selected correction checkpoint, 10 warmups and 50',
            'iterations. Crop processing includes JPEG decode from preloaded bytes,',
            'resize/letterbox, tensor preparation and GPU execution. It excludes the',
            'full-frame detector, extracting boxes from frames, disk IO, score fusion,',
            'tracking and scheduling. The 64 deterministic development crops are a',
            'microbenchmark, not representative live-camera p95 latency.']
    lines+=['','All models have the same stage-3 passenger architecture. Original-resolution','crops use 15% context, JPEG80 4:4:4 and 224px aspect-preserving RGB letterbox.','Training uses balanced sampling, the original IISc backbone, CE and AdamW.','Small/full share development crops; paired_gt/proposal share matched objects.','The proposal audit retains negatives and ambiguous proposals but this stage','does not train a passenger-validity head. A four-class result cannot validate','background rejection, colour recognition, or recovery of missed vehicles.','',
        'Residual runs learn an additive adjustment to original detector subtype logits,',
        'starting at zero; zero correction is eligible for development selection. They',
        'preserve the original passenger maximum and all non-passenger logits. This',
        'does not prevent all-class AP regressions through subtype ranking changes.',
        '', 'Detailed metrics, hashes, exposure counts, AP and',
        'training/development scores are in results.json. No benchmark-selected checkpoint.']
    lines+=['','## Validation and operational notes','',
        '- Four classifier CPU tests and both eight-crop GPU overfit gates passed.',
        '- Two correction-score tests passed; native zero correction reproduced original AP and counts exactly.',
        '- Representative BF16 convolution gradient relative errors were 0.18–0.21% versus the CPU reference, below the existing 1% test tolerance. This is not a full PyTorch training parity proof.',
        '- All ten training runs completed successfully; config, manifest, original checkpoint, executable and saved checkpoint hashes validated.',
        '- One concurrent evaluation ran out of GPU memory. Its log was preserved and the unchanged evaluation completed successfully after training, in the serial queue.',
        '- Optional full-training-set inference was omitted from three later second-seed runs; their training, development selection and ST evaluations were unchanged. See diagnostic_schedule_change.json.',
        '', '## Next decision, not a launched experiment','',
        'The observed hatchback/MUV trade motivates testing natural-frequency training',
        'for the anchored correction, with a declared overall-accuracy objective and',
        'per-class floors. The current class-balanced, macro-F1-selected recipe does',
        'not demonstrate the requested overall gain. Resolve the reviewed subtype',
        'policy and confirm improvements on camera/video-disjoint labels before',
        'calling a deployment win. Longer full-pool training and stronger adapted',
        'representations remain untested by this fixed-budget screen; neither is a',
        'substantiated forecast of 80%. No colour model or validity head was trained.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(title,arms) in zip(axes,[('Training pool',['small','full']),('Training box geometry',['paired_gt','proposal']),('Anchored correction',['residual'])]):
        for name,r in summary['runs'].items():
            if any(name.startswith(a+'_seed') for a in arms):
                h=r['history'];ax.plot([v['epoch']*3961/1000 for v in h],[100*v['metrics']['macro_f1'] for v in h],label=name)
        ax.set_title(title);ax.set_xlabel('Training draws (thousands)');ax.set_ylabel('Development macro-F1 (%)');ax.grid(alpha=.2)
        if ax.lines: ax.legend(fontsize=7)
    fig.suptitle('Development diagnostics: sources available to IISc pretraining; not ST benchmark scores')
    fig.tight_layout();fig.savefig(ROOT/'development_curves.png',dpi=150);plt.close(fig)
    print(json.dumps({'runs':{k:{x:v[x] for x in ['accuracy','best_epoch','best_dev','recalls']} for k,v in summary['runs'].items()},'paired':summary['paired']},indent=2))

if __name__=='__main__': main()

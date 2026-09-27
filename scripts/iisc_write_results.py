#!/usr/bin/env python3
"""Write the completed full-pool experiment's tables and learning curves."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root;x=json.loads((r/'comparison.json').read_text())
lines=['# Full-pool IISc detector results','',x['scope'],'','## Frozen baseline routed-object cohort','','This cohort is selected once using the original detector. A candidate that loses one of these objects receives an error for it; the denominator does not shrink. The original detector itself defines the cohort, so whole-scene results below remain essential.','','| Model | Correct / N | Subtype accuracy | Hatchback recall | Sedan recall | SUV recall | MUV recall | Newly missed |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for name,v in x['fixed_baseline_routed_objects'].items():
 lines.append(f"| {name} | {v['correct']} / {v['n']} | {100*v['accuracy']:.2f}% | "+' | '.join(f'{100*q:.2f}%' for q in v['per_class_recall'])+f" | {v['misses']} |")
lines+=['','## Whole-scene results','','| Model | All-class AP | Passenger localization coverage | Correct subtype / all passenger GT | Correct / localized | Forward p50 / p95 ms |','| --- | --- | --- | --- | --- | --- |']
for name,v in x['whole_scene'].items():
 t=v['forward_batch1_ms'];lines.append(f"| {name} | {v['AP']:.5f} | {100*v['localized_fraction']:.2f}% | {100*v['correct_fraction_all_gt']:.2f}% | {v['correct']} / {v['localized']} | {t['p50']:.2f} / {t['p95']:.2f} |")
lines+=['','Forward timings use PyTorch batch1 at 640px with 10 warmups and 50 repetitions, excluding image decoding, preprocessing, transfers, matching and tracking. Treat small differences as timing variation, not a change in architecture.','', '## Paired uncertainty','']
for name,v in x['paired'].items():
 b=v['fixed_cohort_subtype'];lines.append(f"- {name}: {100*b['delta']:+.2f} points versus baseline; paired source-group 95% interval [{100*b['percentile95'][0]:+.2f}, {100*b['percentile95'][1]:+.2f}].")
lines+=['','These intervals resample automatic source groups, not independently held-out cameras or training seeds. AP uncertainty was not estimated.','', '## Training and selection','']
fig,axes=plt.subplots(1,2,figsize=(10,4))
base=json.loads((r/'development/baseline.json').read_text())
for arm,t in x['training'].items():
 h=t['history'];s=t['selection'];training=sum(v['training_seconds'] for v in h)
 lines.append(f"- {arm}: {t['updates']} updates; {training/60:.1f} minutes training-loop time, {t['elapsed_seconds']/60:.1f} minutes including development evaluation/checkpoint overhead; peak allocated GPU memory {t['peak_allocated_bytes']/2**30:.2f} GiB. Development selected epoch {s['epoch']} ({s['name']}).")
 axes[0].plot([0]+[v['epoch'] for v in h],[100*base['correct_fraction_all_gt']]+[100*v['development']['correct_fraction_all_gt'] for v in h],marker='o',label=arm)
 axes[1].plot([0]+[v['epoch'] for v in h],[base['AP']]+[v['development']['AP'] for v in h],marker='o',label=arm)
axes[0].set_ylabel('Correct subtype / all development passenger GT (%)');axes[1].set_ylabel('Development all-class AP')
for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Diagnostic development: original IISc pretraining included these sources');fig.tight_layout();fig.savefig(r/'development_curves.png',dpi=160);plt.close(fig)
lines+=['','All models retain the 20,101,004-parameter detector architecture. Export file size must not be compared directly with the original package, which includes training state.','', 'The auxiliary subtype loss is training-only. Its absence of extra inference operations does not itself establish that accuracy improves. The two arms share augmentation, data, initialization, seed and update budget. The only intended training-objective change is the additional matched-passenger CE term.','', 'No new camera-disjoint test set or human-corrected subtype labels were created. Training and development use canonical MV labels, with ST/MV conflicts separately flagged. The 512-image ST benchmark has been consumed by prior experiments. These results are not a fleet deployment guarantee.']
(r/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(r/'RESULTS.md')

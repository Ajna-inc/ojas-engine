#!/usr/bin/env python3
"""Freeze a common actual-proposal development input for both training geometries."""
import hashlib
import json
from collections import Counter
from pathlib import Path

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    root=Path('evidence/passenger_proposals_2026_09_20')
    assert not (root/'finalization.json').exists()
    assert not any((root/f'{arm}_seed{s}').exists() for arm in ['paired_gt','proposal'] for s in [1,2]), 'cannot change manifests after training starts'
    old={arm:[json.loads(x) for x in (root/f'{arm}.jsonl').read_text().splitlines()] for arm in ['paired_gt','proposal']}
    for arm in old: (root/f'{arm}_before_common_dev.jsonl').write_bytes((root/f'{arm}.jsonl').read_bytes())
    dev=[r for r in old['proposal'] if r['split']=='dev']
    dh={r['crop_sha256'] for r in dev}
    quarantine={r['group'] for rows in old.values() for r in rows if r['split']=='train' and r['crop_sha256'] in dh}
    current={arm:[r for r in rows if r['split']=='train' and r['group'] not in quarantine]+dev for arm,rows in old.items()}
    identities=lambda rows:[(r['image_id'],r['query'],r['annotation_id'],r['label'],r['split']) for r in rows]
    assert identities(current['paired_gt'])==identities(current['proposal'])
    for arm,rows in current.items():
        assert len({r['path'] for r in rows})==len(rows)
        (root/f'{arm}.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    result=dict(reason='Both arms select checkpoints on exactly the same actual-detector-box development crops; only training crop geometry changes.',
        prior_hashes={a:sha(root/f'{a}_before_common_dev.jsonl') for a in old},
        manifest_hashes={a:sha(root/f'{a}.jsonl') for a in old},
        additional_quarantined_groups=sorted(quarantine),
        counts={a:dict(Counter(f"{r['split']}/{r['class_name']}" for r in rows)) for a,rows in current.items()},
        script_sha256=sha(__file__))
    (root/'finalization.json').write_text(json.dumps(result,indent=2)+'\n')
    (root/'audit_before_common_dev.json').write_bytes((root/'audit.json').read_bytes())
    audit=json.loads((root/'audit.json').read_text())
    audit['manifests']=result['manifest_hashes']
    audit['finalization_sha256']=sha(root/'finalization.json')
    audit['final_manifest_counts']=result['counts']
    audit['contract']+=' Both arms use identical actual-proposal development inputs after passenger_proposal_finalize.py; see preserved pre-finalization audit and manifests.'
    (root/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

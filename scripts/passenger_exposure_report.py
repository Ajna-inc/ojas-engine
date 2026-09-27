#!/usr/bin/env python3
"""Replay the Rust sampling RNG to audit seen versus unvisited training crops."""
import json
from pathlib import Path
import sys
from collections import Counter

class Rng:
    def __init__(self,seed): self.state=(seed*6364136223846793005+1442695040888963407)&((1<<64)-1)
    def next(self):
        old=self.state; self.state=(old*6364136223846793005+1442695040888963407)&((1<<64)-1)
        xs=(((old>>18)^old)>>27)&0xffffffff; rot=old>>59
        return ((xs>>rot)|(xs<<((-rot)&31)))&0xffffffff

def main():
    run=Path(sys.argv[1]); cfg=json.loads((run/'config.json').read_text())
    assert cfg['balanced'] and not cfg.get('overfit_steps',0)
    rows=[json.loads(s) for s in Path(cfg['manifest']).read_text().splitlines()]
    train=[r for r in rows if r['split']=='train']
    pools=[[i for i,r in enumerate(train) if r['label']==c] for c in range(4)]
    rng=Rng(cfg['seed']); seen=set(); draws=cfg.get('epoch_samples',0) or len(train)
    for _ in range(cfg['epochs']):
        for i in range(draws):
            pool=pools[i%4]; seen.add(pool[rng.next()%len(pool)])
        for _ in range(draws-1+2*draws): rng.next() # shuffle, flip and brightness
    exposure=json.loads((run/'exposure.json').read_text())
    assert len(seen)==exposure['unique_sampled_crops'], 'RNG replay does not match executable exposure'
    seenpaths={train[i]['path'] for i in seen}; groups={train[i]['group'] for i in seen}
    counts=Counter()
    for line in (run/'final_train_predictions.jsonl').read_text().splitlines():
        r=json.loads(line); correct=int(max(range(4),key=lambda c:r['probabilities'][c])==r['label'])
        strata=['all','sampled' if r['path'] in seenpaths else 'not_sampled',
                'sampled_group' if r['group'] in groups else 'not_sampled_group']
        for s in strata: counts[s+'/n']+=1; counts[s+'/correct']+=correct
    result={'scope':'Final checkpoint only, not development-selected best checkpoint. Unvisited training groups are diagnostic, not an independent test.',
        'rng_replay_matches_executable':True,'strata':{s:dict(n=counts[s+'/n'],correct=counts[s+'/correct'],accuracy=counts[s+'/correct']/counts[s+'/n']) for s in ['all','sampled','not_sampled','sampled_group','not_sampled_group'] if counts[s+'/n']}}
    (run/'exposure_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

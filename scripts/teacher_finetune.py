#!/usr/bin/env python3
"""Bounded full-network RT-DETR-X fine-tuning with the official detection criterion."""
import argparse
import copy
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from teacher_detector_bench import load_model, sha

ROOT = Path('evidence/teacher_finetune_2026_09_20').resolve()

class Scenes(Dataset):
    def __init__(self, contract, seed):
        self.rows = json.loads(Path(contract['training_manifest']).read_text())
        assert sha(contract['training_annotations']) == contract['training_annotations_sha256']
        annotations = json.loads(Path(contract['training_annotations']).read_text())
        self.targets = {r['id']: [] for r in self.rows}
        for ann in annotations['annotations']:
            if ann['image_id'] in self.targets and not ann.get('iscrowd', 0):
                assert 1 <= ann['category_id'] <= 14
                self.targets[ann['image_id']].append(ann)
        self.seed, self.epoch = seed, 0

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        data = Path(r['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r['sha256'], 'training image changed'
        with Image.open(io.BytesIO(data)) as image: image = image.convert('RGB')
        assert image.size == (r['width'], r['height'])
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{r["id"]}'.encode()).digest()[:8], 'little'))
        flip, brightness = rng.random() < .5, rng.uniform(.85, 1.15)
        if flip: image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize((640,640), Image.Resampling.BILINEAR)
        pixels = np.clip(np.array(image).astype(np.float32) / 255 * brightness, 0, 1)
        labels, boxes = [], []
        for a in self.targets[r['id']]:
            x,y,w,h = a['bbox']
            x0,y0 = max(0,x),max(0,y)
            x1,y1 = min(r['width'],x+w),min(r['height'],y+h)
            if x1 <= x0 or y1 <= y0: continue
            cx,cy = (x0+x1)/2/r['width'],(y0+y1)/2/r['height']
            boxes.append([1-cx if flip else cx,cy,(x1-x0)/r['width'],(y1-y0)/r['height']])
            labels.append(a['category_id'])
        return torch.from_numpy(pixels.transpose(2,0,1).copy()), {
            'labels':torch.tensor(labels,dtype=torch.long),
            'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            'image_id':torch.tensor(r['id']), 'orig_size':torch.tensor([r['width'],r['height']])}

def collate(batch):
    return torch.stack([x for x,_ in batch]), [t for _,t in batch]

def float_outputs(value):
    if isinstance(value,torch.Tensor): return value.float() if value.is_floating_point() else value
    if isinstance(value,dict): return {k:float_outputs(v) for k,v in value.items()}
    if isinstance(value,list): return [float_outputs(v) for v in value]
    if isinstance(value,tuple): return tuple(float_outputs(v) for v in value)
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    contract = json.loads((ROOT/'contract.json').read_text())
    out = ROOT/('smoke' if args.smoke else f'seed{args.seed}')
    out.mkdir()  # never overwrite a run
    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    assert torch.cuda.is_bf16_supported()
    model, mapping, metadata = load_model('rtdetr_x')
    assert mapping == list(range(15))
    from src.core import YAMLConfig
    cfg = YAMLConfig(metadata['config'])
    cfg.yaml_cfg.update(metadata['resolved_config'])
    criterion = cfg.criterion.cuda().train()
    assert criterion.num_classes == 15
    model.requires_grad_(True).train()
    for module in model.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm): module.eval()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    grouped = {}
    for name,p in model.named_parameters():
        backbone = name.startswith('backbone.')
        decay = 0 if p.ndim == 1 or name.endswith('.bias') or 'norm' in name or '.bn' in name else contract['weight_decay']
        grouped.setdefault((backbone,decay),[]).append(p)
    groups = [{'params':ps,'lr':contract['backbone_lr'] if backbone else contract['other_lr'],
               'base_lr':contract['backbone_lr'] if backbone else contract['other_lr'],'weight_decay':decay}
              for (backbone,decay),ps in grouped.items()]
    optimizer = torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    dataset = Scenes(contract,args.seed)
    accumulation = contract['accumulation']
    planned_updates = 4 if args.smoke else contract['updates']
    epochs = 1 if args.smoke else contract['epochs']
    generator = torch.Generator().manual_seed(args.seed)
    optimizer.zero_grad(set_to_none=True)
    probes = {prefix: next((name,p,p.detach().flatten()[:256].clone()) for name,p in model.named_parameters()
              if name.startswith(prefix) and p.ndim > 1) for prefix in ['backbone.','encoder.','decoder.']}
    metadata.update(seed=args.seed,contract=contract,script_sha256=sha(__file__),
        contract_sha256=sha(ROOT/'contract.json'),training_manifest_sha256=sha(contract['training_manifest']),
        precision='BF16 forward; FP32 criterion, weights and AdamW',checkpoint_semantics='final raw and EMA weights, no resumable optimizer/RNG state',
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),smoke=args.smoke)
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    start=time.perf_counter();step=0;torch.cuda.reset_peak_memory_stats()
    history=[]
    with (out/'metrics.jsonl').open('x') as log:
        for epoch in range(epochs):
            dataset.epoch=epoch
            samples=Subset(dataset,[0,1]*8) if args.smoke else dataset
            loader=DataLoader(samples,batch_size=contract['micro_batch'],shuffle=not args.smoke,generator=generator,
                num_workers=4,pin_memory=True,collate_fn=collate)
            total_loss=0.;seen=0
            assert len(loader)%accumulation==0
            for micro,(images,targets) in enumerate(loader):
                images=images.cuda(non_blocking=True)
                targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
                with torch.autocast('cuda',dtype=torch.bfloat16): outputs=model(images,targets=targets)
                losses=criterion(float_outputs(outputs),targets)
                loss=sum(losses.values())
                assert torch.isfinite(loss),'nonfinite loss'
                (loss/accumulation).backward()
                total_loss+=float(loss.detach())*len(images);seen+=len(images)
                if (micro+1)%accumulation: continue
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),contract['gradient_clip'],error_if_nonfinite=True)
                if step==0:
                    gates={prefix:any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for name,p in model.named_parameters() if name.startswith(prefix)) for prefix in probes}
                    assert all(gates.values()),f'gradient gate failed: {gates}'
                    (out/'gradient_gate.json').write_text(json.dumps(gates,indent=2)+'\n')
                step+=1
                warmup=min(1.,step/contract['warmup_updates'])
                progress=max(0.,(step-contract['warmup_updates'])/max(1,planned_updates-contract['warmup_updates']))
                factor=warmup*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                for group in optimizer.param_groups: group['lr']=group['base_lr']*factor
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    current=model.state_dict()
                    for name,p in ema.state_dict().items():
                        if p.is_floating_point():p.mul_(.99).add_(current[name],alpha=.01)
                        else:p.copy_(current[name])
                if step%10==0 or args.smoke:
                    print(f'epoch {epoch+1} update {step}/{planned_updates} loss {float(loss.detach()):.4f} grad_norm {float(grad_norm):.3f} elapsed {time.perf_counter()-start:.1f}s',flush=True)
                del outputs,losses,loss
            row={'epoch':epoch+1,'updates':step,'mean_train_loss':total_loss/seen,'elapsed_seconds':time.perf_counter()-start}
            history.append(row);log.write(json.dumps(row)+'\n');log.flush();print(row,flush=True)
    assert step==planned_updates
    changes={prefix:float((p.detach().flatten()[:256]-before).abs().max()) for prefix,(_,p,before) in probes.items()}
    assert all(v>0 for v in changes.values()),f'parameter update gate failed {changes}'
    metadata.update(completed=True,updates=step,history=history,parameter_probe_max_changes=changes,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,elapsed_seconds=time.perf_counter()-start)
    if not args.smoke:
        state={'ema':{'module':{k:v.cpu() for k,v in ema.state_dict().items()}},
               'model':{k:v.cpu() for k,v in model.state_dict().items()},'epoch':epochs,'updates':step}
        torch.save(state,out/'final.pth')
        metadata['final_sha256']=sha(out/'final.pth')
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    print('Completed',out,'peak GiB',metadata['peak_allocated_bytes']/2**30,flush=True)

if __name__=='__main__':main()

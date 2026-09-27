#!/usr/bin/env python3
"""Full-pool IISc small-detector continuation with epoch-boundary resume and development selection."""
import argparse, copy, hashlib, io, json, math, os, random, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from teacher_detector_bench import load_model, sha
from teacher_finetune import float_outputs, collate

class Scenes(Dataset):
    def __init__(self, contract, seed, arm):
        self.rows=json.loads(Path(contract['training_manifest']).read_text())
        assert sha(contract['training_annotations'])==contract['training_annotations_sha256']
        doc=json.loads(Path(contract['training_annotations']).read_text())
        self.targets={r['id']:[] for r in self.rows}
        for a in doc['annotations']:
            if a['image_id'] in self.targets and not a.get('iscrowd',0):
                assert 1<=a['category_id']<=14
                self.targets[a['image_id']].append(a)
        self.seed,self.arm,self.epoch=seed,arm,0
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];data=Path(r['cache_path']).read_bytes()
        assert hashlib.sha256(data).hexdigest()==r['cache_sha256'],'cache changed'
        pixels=np.load(io.BytesIO(data),allow_pickle=False)
        assert pixels.shape==(640,640,3) and pixels.dtype==np.uint8
        # Shared geometry/brightness RNG is unaffected by extra appearance augmentation.
        rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{self.seed}:{self.epoch}:{r["id"]}'.encode()).digest()[:8],'little'))
        flip=rng.random()<.5; brightness=rng.uniform(.85,1.15)
        image=Image.fromarray(pixels)
        if flip:image=image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if True:  # Identical mild appearance augmentation in both arms.
            if rng.random()<.25:image=image.filter(ImageFilter.GaussianBlur(float(rng.uniform(.2,.8))))
            if rng.random()<.3:
                buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=int(rng.integers(65,96)));buffer.seek(0)
                with Image.open(buffer) as decoded:image=decoded.convert('RGB')
        pixels=np.clip(np.asarray(image).astype(np.float32)/255*brightness,0,1)
        boxes=[];labels=[]
        for a in self.targets[r['id']]:
            x,y,w,h=a['bbox']; x0,y0=max(0,x),max(0,y);x1,y1=min(r['width'],x+w),min(r['height'],y+h)
            if x1<=x0 or y1<=y0:continue
            cx,cy=(x0+x1)/2/r['width'],(y0+y1)/2/r['height']
            boxes.append([1-cx if flip else cx,cy,(x1-x0)/r['width'],(y1-y0)/r['height']]);labels.append(a['category_id'])
        return torch.from_numpy(pixels.transpose(2,0,1).copy()),{'labels':torch.tensor(labels,dtype=torch.long),'boxes':torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),'image_id':torch.tensor(r['id']),'orig_size':torch.tensor([r['width'],r['height']])}

def passenger_loss(logits, targets, indices):
    selected=[];labels=[]
    for b,(queries,truths) in enumerate(indices):
        queries=queries.to(logits.device);truths=truths.to(targets[b]['labels'].device)
        y=targets[b]['labels'][truths];mask=(y>=1)&(y<=4)
        selected.append(logits[b,queries[mask],1:5]);labels.append(y[mask]-1)
    predictions=torch.cat(selected);truth=torch.cat(labels)
    return torch.nn.functional.cross_entropy(predictions,truth) if len(truth) else logits.sum()*0


def atomic_save(state,path):
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);os.replace(tmp,path)
def rng_state():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
def evaluate(root,checkpoint,name,log):
    command=[sys.executable,str(Path(__file__).with_name('teacher_detector_bench.py')),'rtdetr_s','--root',str(root),'--checkpoint',str(checkpoint),'--output-name',name]
    with log.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    return json.loads((root/f'{name}.json').read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--arm',choices=['continuation','subtype'],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-epoch',type=int);p.add_argument('--tag');a=p.parse_args()
    root=a.root.resolve();c=json.loads((root/'contract.json').read_text());tag=a.tag or (a.arm+'_smoke' if a.smoke else a.arm);out=root/tag
    if a.resume:assert (out/'resume.pth').exists()
    else:out.mkdir()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    seed=c['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,mapping,metadata=load_model(c['model']);assert mapping==list(range(15))
    # Strict reload and an actual zero-update output check before any training mutation.
    sample=np.load(json.loads(Path(c['training_manifest']).read_text())[0]['cache_path'],allow_pickle=False)
    x=torch.from_numpy(sample.transpose(2,0,1).copy()).float().div_(255).unsqueeze(0).cuda()
    with torch.inference_mode():before={k:v.clone() for k,v in model(x).items() if isinstance(v,torch.Tensor)}
    state=torch.load(metadata['checkpoint'],map_location='cpu',weights_only=True)['ema']['module'];model.load_state_dict(state,strict=True);del state
    with torch.inference_mode():after=model(x)
    parity={k:float((v-after[k]).abs().max()) for k,v in before.items()};assert all(v==0 for v in parity.values()),parity
    del x,before,after
    if a.smoke and not a.resume:atomic_save({'ema':{'module':{k:v.cpu() for k,v in model.state_dict().items()}}},out/'zero_update.pth')
    from src.core import YAMLConfig
    cfg=YAMLConfig(metadata['config']);cfg.yaml_cfg.update(metadata['resolved_config']);criterion=cfg.criterion.cuda().train();assert criterion.num_classes==15
    model.requires_grad_(True).train()
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    grouped={}
    for n,q in model.named_parameters():grouped.setdefault((n.startswith('backbone.'),0. if q.ndim==1 or n.endswith('.bias') or 'norm' in n or '.bn' in n else c['weight_decay']),[]).append(q)
    optimizer=torch.optim.AdamW([{'params':ps,'lr':c['backbone_lr'] if backbone else c['other_lr'],'base_lr':c['backbone_lr'] if backbone else c['other_lr'],'weight_decay':decay} for (backbone,decay),ps in grouped.items()],betas=(.9,.999),eps=1e-8)
    dataset=Scenes(c,seed,a.arm);epochs=2 if a.smoke else c['epochs'];micro=c['micro_batch'];acc=c['accumulation'];n=16 if a.smoke else len(dataset);updates_per_epoch=math.ceil(math.ceil(n/micro)/acc);planned=updates_per_epoch*epochs
    probes={prefix:next((name,q,q.detach().flatten()[:256].clone()) for name,q in model.named_parameters() if name.startswith(prefix) and q.ndim>1) for prefix in ['backbone.','encoder.','decoder.']}
    metadata.update(contract=c,contract_sha256=sha(root/'contract.json'),script_sha256=sha(__file__),training_manifest_sha256=sha(c['training_manifest']),arm=a.arm,zero_update_max_absolute_difference=parity,updates_planned=planned,smoke=a.smoke,checkpoint_semantics='EMA inference weights plus separate epoch-boundary model/EMA/optimizer and Python/NumPy/Torch/CUDA RNG resume state',trainable_parameters=sum(q.numel() for q in model.parameters() if q.requires_grad),precision='BF16 forward, FP32 criterion and master weights; evaluation FP32',gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__)
    start_epoch=0;step=0;history=[];elapsed_before=0.
    best={'epoch':0,'name':'baseline'}
    if not a.smoke:
        baseline=json.loads((root/'development/baseline.json').read_text());best.update(score=baseline['correct_fraction_all_gt'],AP=baseline['AP']);floor=baseline['AP']-.005
    if a.resume:
        # Locally generated trusted checkpoint; NumPy/Python RNG state requires weights_only=False.
        checkpoint=torch.load(out/'resume.pth',map_location='cpu',weights_only=False)
        assert checkpoint['contract_sha256']==metadata['contract_sha256'] and checkpoint['script_sha256']==metadata['script_sha256']
        assert checkpoint['arm']==a.arm
        model.load_state_dict(checkpoint['model']);ema.load_state_dict(checkpoint['ema']);optimizer.load_state_dict(checkpoint['optimizer']);restore_rng(checkpoint['rng'])
        start_epoch=checkpoint['epoch'];step=checkpoint['step'];history=checkpoint['history'];best=checkpoint['best'];elapsed_before=checkpoint['elapsed_seconds'];del checkpoint
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats();optimizer.zero_grad(set_to_none=True)
    with (out/'metrics.jsonl').open('a' if a.resume else 'x') as log:
        for epoch in range(start_epoch,epochs):
            dataset.epoch=epoch;samples=Subset(dataset,[0,1,2,3]*4) if a.smoke else dataset
            generator=torch.Generator().manual_seed(seed+epoch*100003)
            loader=DataLoader(samples,batch_size=micro,shuffle=not a.smoke,generator=generator,num_workers=4,pin_memory=True,collate_fn=collate)
            total=0.;seen=0;epoch_start=time.perf_counter();block_start=0
            for i,(images,targets) in enumerate(loader):
                images=images.cuda(non_blocking=True);targets=[{k:v.cuda(non_blocking=True) for k,v in t.items()} for t in targets]
                with torch.autocast('cuda',dtype=torch.bfloat16):outputs=model(images,targets=targets)
                fp_outputs=float_outputs(outputs)
                losses=criterion(fp_outputs,targets);loss=sum(losses.values())
                if a.arm=='subtype':
                    indices=criterion.matcher({'pred_logits':fp_outputs['pred_logits'],'pred_boxes':fp_outputs['pred_boxes']},targets)['indices']
                    loss_subtype=passenger_loss(fp_outputs['pred_logits'],targets,indices)
                    loss=loss+c['passenger_ce_weight']*loss_subtype
                assert torch.isfinite(loss),'nonfinite loss'
                # Weight the final partial accumulation block by its actual image count.
                if i%acc==0:block_start=i;block_samples=min(acc*micro,n-i*micro)
                (loss*(len(images)/block_samples)).backward();total+=float(loss.detach())*len(images);seen+=len(images)
                if (i+1)%acc and i+1<len(loader):continue
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'],error_if_nonfinite=True)
                if step==0:
                    gates={prefix:any(q.grad is not None and bool(torch.count_nonzero(q.grad)) for name,q in model.named_parameters() if name.startswith(prefix)) for prefix in probes}
                    assert all(gates.values());(out/'gradient_gate.json').write_text(json.dumps(gates)+'\n')
                step+=1;warmup=min(1.,step/c['warmup_updates']);progress=max(0.,(step-c['warmup_updates'])/max(1,planned-c['warmup_updates']));factor=warmup*(.1+.9*.5*(1+math.cos(math.pi*progress)))
                for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    current=model.state_dict()
                    for name,q in ema.state_dict().items():
                        if q.is_floating_point():q.mul_(c['ema_decay']).add_(current[name],alpha=1-c['ema_decay'])
                        else:q.copy_(current[name])
                if step%25==0 or a.smoke:print(f'{a.arm} epoch {epoch+1} update {step}/{planned} loss {float(loss.detach()):.4f} grad {float(grad):.3f} elapsed {elapsed_before+time.perf_counter()-start:.1f}s',flush=True)
                del outputs,fp_outputs,losses,loss
            assert seen==n
            changes={prefix:float((q.detach().flatten()[:256]-before).abs().max()) for prefix,(_,q,before) in probes.items()};assert all(v>0 for v in changes.values()),changes
            row={'epoch':epoch+1,'updates':step,'images':seen,'mean_train_loss':total/seen,'training_seconds':time.perf_counter()-epoch_start,'elapsed_seconds':elapsed_before+time.perf_counter()-start,'parameter_probe_max_changes':changes}
            if not a.smoke:
                inference=out/f'epoch{epoch+1}.pth';atomic_save({'ema':{'module':{k:v.cpu() for k,v in ema.state_dict().items()}},'epoch':epoch+1,'updates':step},inference)
                evaluation_rng=rng_state()
                ev=evaluate(root/'development',inference,f'{a.arm}_epoch{epoch+1}',out/f'epoch{epoch+1}_eval.log');restore_rng(evaluation_rng)
                row['development']={k:ev[k] for k in ['AP','conditional_accuracy','correct_fraction_all_gt','localized_fraction','correct','localized','passenger_gt']}
                if ev['AP']>=floor and ev['correct_fraction_all_gt']>best['score']:
                    best={'epoch':epoch+1,'name':f'{a.arm}_epoch{epoch+1}','score':ev['correct_fraction_all_gt'],'AP':ev['AP'],'checkpoint':str(inference),'checkpoint_sha256':sha(inference)}
            history.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            saved={'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),'epoch':epoch+1,'step':step,'history':history,'best':best,'arm':a.arm,'contract_sha256':metadata['contract_sha256'],'script_sha256':metadata['script_sha256'],'elapsed_seconds':elapsed_before+time.perf_counter()-start}
            atomic_save(saved,out/'resume.pth')
            (out/'selection.json').write_text(json.dumps(best,indent=2)+'\n')
            if a.stop_after_epoch and epoch+1>=a.stop_after_epoch:break
    metadata.update(completed=step==planned,updates=step,history=history,selection=best,peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=elapsed_before+time.perf_counter()-start)
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n');print('Completed' if metadata['completed'] else 'Checkpointed',out,flush=True)

if __name__=='__main__':main()

"""Train LoRA with pair-centric GRPO and resume at saved optimizer boundaries."""
import argparse
import hashlib
import json
import pathlib
import random
import shutil
import time
import numpy as np
import torch
from reward import fidelity, advantages, check
from vlm import VLM, SCORE_PROMPT, parse_score
from budget import check_deadline


def update_pair(runner, paths, label, weight, cfg, denominator):
    groups = []
    for path in paths:
        inputs, _ = runner.inputs([path], SCORE_PROMPT, local=True)
        group = {'inputs':inputs,'scores':[],'completions':[],'texts':[]}
        for _ in range(cfg['rollouts']):
            text, completion = runner.generate(inputs,max_tokens=cfg['score_tokens'],temperature=cfg['temperature'])
            group['texts'].append(text); group['completions'].append(completion)
            try:
                score = parse_score(text)
            except ValueError:
                score = None
            group['scores'].append(score)
        groups.append(group)
    summary = {'scores':[g['scores'] for g in groups], 'texts':[g['texts'] for g in groups], 'weight':weight}
    if any(s is None for g in groups for s in g['scores']):
        return dict(summary,status='invalid_rollout_pair_skipped',zero_variance_groups=None)
    target = {'A':1.,'B':0.,'near_equal':.5}[label]
    rewards = [fidelity(groups[0]['scores'],groups[1]['scores'],target,cfg['gamma']),
               fidelity(groups[1]['scores'],groups[0]['scores'],1-target,cfg['gamma'])]
    advs = [advantages(r) for r in rewards]
    summary.update(rewards=[r.tolist() for r in rewards],advantages=[a.tolist() for a in advs],
                   zero_variance_groups=sum(not np.any(a) for a in advs),policy=0.,kl=0.)
    for group, adv in zip(groups,advs):
        for completion, value in zip(group['completions'],adv):
            # The fixed reference is the same original checkpoint with LoRA disabled.
            with torch.no_grad(), runner.model.disable_adapter():
                reference = runner.logprobs(group['inputs'],completion).detach()
            logp = runner.logprobs(group['inputs'],completion)
            # mu=1: no optimizer step occurred since these on-policy rollouts.
            old = logp.detach()
            ratio = (logp-old).exp()
            policy = torch.minimum(ratio*float(value),ratio.clamp(1-cfg['clip'],1+cfg['clip'])*float(value))
            log_ratio = reference-logp
            kl = log_ratio.exp()-log_ratio-1
            loss = (-weight*policy + cfg['kl_beta']*kl).mean()/denominator
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss; stop, do not silently omit this update')
            loss.backward()
            summary['policy'] += float(policy.detach().mean())/(2*cfg['rollouts'])
            summary['kl'] += float(kl.detach().mean())/(2*cfg['rollouts'])
            del reference,logp,old,ratio,policy,kl,loss
    return dict(summary,status='processed')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--bundle',required=True)
    p.add_argument('--videos',required=True); p.add_argument('--arm',choices=['T1','T2','T3','T5'],required=True)
    p.add_argument('--seed',type=int,required=True); p.add_argument('--output',required=True)
    p.add_argument('--model',default='models/Qwen2.5-VL-3B-Instruct')
    p.add_argument('--hours',type=float,default=5.5)
    a=p.parse_args()
    config_data=pathlib.Path(a.config).read_bytes(); bundle_data=pathlib.Path(a.bundle).read_bytes()
    cfg=json.loads(config_data); bundle=json.loads(bundle_data); train=cfg['training']
    signature=hashlib.sha256(config_data+bundle_data+a.arm.encode()+str(a.seed).encode()).hexdigest()
    videos={r['video_id']:r for r in map(json.loads,pathlib.Path(a.videos).read_text().splitlines())}
    pairs=bundle['schedule'].copy()
    random.Random(a.seed).shuffle(pairs)
    assert len(pairs)%train['pairs_per_batch']==0
    out=pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    check_deadline()
    runner=VLM(a.model,lora=True,source_crop_right=True,observation=cfg['observation'])
    params=[p for p in runner.model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=train['learning_rate'],betas=(.9,.999),
                                eps=1e-8,weight_decay=train['weight_decay'])
    completed=0
    state_path=out/'latest.json'
    if state_path.exists():
        state=json.loads(state_path.read_text()); assert state['signature']==signature
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(runner.model,load_file(str(out/state['checkpoint']/'adapter_model.safetensors')))
        optimizer.load_state_dict(torch.load(out/state['checkpoint']/'optimizer.pt',weights_only=True))
        completed=state['completed_pairs']
        if completed==len(pairs):
            print('ALREADY_COMPLETE',flush=True); return
    check()
    started=time.monotonic()
    for offset in range(completed,len(pairs),train['pairs_per_batch']):
        check_deadline()
        optimizer.zero_grad(set_to_none=True)
        batch=pairs[offset:offset+train['pairs_per_batch']]
        t=time.monotonic(); summaries=[]
        for pair in batch:
            seed=int(hashlib.sha256(f"{a.seed}:{pair['pair_id']}".encode()).hexdigest()[:8],16)
            torch.manual_seed(seed); np.random.seed(seed)
            label=pair['baseline_label'] if a.arm=='T1' else pair['full_label']
            weight=pair['weight'] if a.arm=='T3' else pair['shuffled_weight'] if a.arm=='T5' else 1.
            summary=update_pair(runner,[videos[pair[k]]['path'] for k in ('a','b')],label,weight,train,
                                len(batch)*2*train['rollouts'])
            summary.update(pair_id=pair['pair_id'],label=label,rollout_seed=seed); summaries.append(summary)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(params,train['max_grad_norm']))
        if not np.isfinite(grad_norm):
            raise FloatingPointError('Nonfinite gradient; checkpoint remains at previous completed batch')
        if grad_norm>0:
            optimizer.step()
        completed=offset+len(batch)
        checkpoint=f"step_{completed//train['pairs_per_batch']:04d}"
        path=out/checkpoint; runner.model.save_pretrained(path)
        torch.save(optimizer.state_dict(),path/'optimizer.pt')
        report={'completed_pairs':completed,'total_pairs':len(pairs),'grad_norm':grad_norm,
                'seconds':time.monotonic()-t,'pairs':summaries}
        batches=out/'batches';batches.mkdir(exist_ok=True)
        (batches/(checkpoint+'.json')).write_text(json.dumps(report,indent=2))
        state={'signature':signature,'checkpoint':checkpoint,'completed_pairs':completed,
               'total_pairs':len(pairs),'status':'complete' if completed==len(pairs) else 'running',
               'arm':a.arm,'seed':a.seed,
               'max_memory_allocated':[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]}
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
        # Keep the two latest optimizer snapshots and every batch log; prune older snapshots.
        for previous in sorted(out.glob('step_[0-9][0-9][0-9][0-9]'))[:-2]:
            if previous.is_dir():
                shutil.rmtree(previous)
        print(json.dumps({k:v for k,v in report.items() if k!='pairs'}),flush=True)
        if time.monotonic()-started>a.hours*3600:
            print('TIME_BUDGET_STOP',flush=True); return


if __name__=='__main__':
    main()

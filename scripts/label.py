"""Resumable teacher collection; each pair/method has a deterministic independent seed."""
import argparse
import hashlib
import json
import pathlib
import time
import numpy as np
import torch
from evaluator import Evaluator
from vlm import VLM

p = argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--videos',required=True)
p.add_argument('--pairs',required=True)
p.add_argument('--output',required=True)
p.add_argument('--model',default='models/Qwen2.5-VL-3B-Instruct')
p.add_argument('--methods',nargs='+',default=['baseline','full'])
p.add_argument('--hours',type=float,default=1.8)
a = p.parse_args()
config_path = pathlib.Path(a.config)
cfg = json.loads(config_path.read_text())
config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
videos = {r['video_id']:r for r in map(json.loads,pathlib.Path(a.videos).read_text().splitlines())}
pairs = list(map(json.loads,pathlib.Path(a.pairs).read_text().splitlines()))
out = pathlib.Path(a.output); out.mkdir(parents=True,exist_ok=True)
runner = VLM(a.model,observation=cfg['observation'],source_crop_right=True)
evaluator = Evaluator(runner,cfg['teacher'])
started = time.monotonic()
for pair in pairs:
    for method in a.methods:
        path = out/(pair['pair_id']+'_'+method+'.json')
        if path.exists():
            assert json.loads(path.read_text())['config_sha256']==config_hash, 'Config changed on resume'
            continue
        if time.monotonic()-started > a.hours*3600:
            print('TIME_BUDGET_STOP',flush=True); raise SystemExit(0)
        seed = int(hashlib.sha256(f"{cfg['seed']}:{pair['pair_id']}:{method}".encode()).hexdigest()[:8],16)
        torch.manual_seed(seed); np.random.seed(seed)
        t = time.monotonic()
        result = evaluator.evaluate([videos[pair[k]]['path'] for k in ('a','b')],method)
        result.update(pair=pair,method=method,seed=seed,config_sha256=config_hash,
                      elapsed_seconds=time.monotonic()-t,
                      max_memory_allocated=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(result,indent=2)); temp.replace(path)
        print(json.dumps({'pair':pair['pair_id'],'method':method,'label':result['label'],
                         'reliability':result['reliability'],'seconds':result['elapsed_seconds'],
                         'calls':len(result['records'])}),flush=True)

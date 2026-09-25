"""Write frozen single-video predictions. No MOS import or file access."""
import argparse
import hashlib
import json
import pathlib
import time
import torch
from vlm import VLM, SCORE_PROMPT, parse_score
from budget import check_deadline

p=argparse.ArgumentParser()
p.add_argument('--config',required=True);p.add_argument('--videos',required=True)
p.add_argument('--output',required=True);p.add_argument('--adapter')
p.add_argument('--model',default='models/Qwen2.5-VL-3B-Instruct')
p.add_argument('--hours',type=float,default=1.8)
a=p.parse_args()
raw=pathlib.Path(a.config).read_bytes(); cfg=json.loads(raw)
signature=hashlib.sha256(raw).hexdigest()
videos=list(map(json.loads,pathlib.Path(a.videos).read_text().splitlines()))
assert all(v['domain']=='target' for v in videos)
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
check_deadline()
runner=VLM(a.model,observation=cfg['observation'])
if a.adapter:
    from peft import PeftModel
    runner.model=PeftModel.from_pretrained(runner.model,a.adapter).eval()
started=time.monotonic()
for row in videos:
    path=out/(row['video_id']+'.json')
    if path.exists():
        assert json.loads(path.read_text())['config_sha256']==signature
        continue
    check_deadline()
    if time.monotonic()-started>a.hours*3600:
        print('TIME_BUDGET_STOP',flush=True);break
    torch.manual_seed(int(row['video_id'][2:10],16))
    inputs,trace=runner.inputs([row['path']],SCORE_PROMPT,local=True)
    text,_=runner.generate(inputs,max_tokens=cfg['test']['score_tokens'],temperature=cfg['test']['temperature'])
    try:
        score=parse_score(text);status='valid'
    except ValueError:
        score=None;status='invalid_score'
    record={'video_id':row['video_id'],'video_sha256':row['sha256'],'score':score,'status':status,
            'output':text,'observations':trace,'cost':runner.timings[-1],'config_sha256':signature,
            'adapter':a.adapter}
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(path)
    print(row['video_id'],status,flush=True)

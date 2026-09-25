"""Serial GPU stages for the frozen pilot; restart resumes artifacts, never submits jobs."""
import json
import os
import pathlib
import subprocess
import sys
from budget import check_deadline

config='configs/experiment.json'
cfg=json.loads(pathlib.Path(config).read_text())
assert cfg['status']=='FROZEN'
assert json.loads(pathlib.Path('runs/train-pilot/latest.json').read_text())['status']=='complete'
root=pathlib.Path('runs/main');root.mkdir(parents=True,exist_ok=True)
manifest='data/manifests/frozen'
labels=str(root/'labels');bundle=str(root/'bundle.json')


def run(script,*args):
    check_deadline()
    print('STAGE',script,*args,flush=True)
    subprocess.run([sys.executable,'scripts/'+script,*map(str,args)],check=True)


run('label.py','--config',config,'--videos',manifest+'/source_train.jsonl',
    '--pairs',manifest+'/pairs.jsonl','--output',labels,'--hours',24)
if not pathlib.Path(bundle).exists():
    run('bundle.py','--config',config,'--videos',manifest+'/source_train.jsonl',
        '--pairs',manifest+'/pairs.jsonl','--labels',labels,'--output',bundle)
report=json.loads(pathlib.Path(bundle).read_text())['report']
print('FROZEN_LABEL_COVERAGE',json.dumps(report),flush=True)
specs=[{'arm':'T0','seed':0,'predictions':str(root/'predictions/T0')}]
run('predict.py','--config',config,'--videos',manifest+'/target_test.jsonl',
    '--output',specs[0]['predictions'],'--hours',3)
for seed in cfg['training_seeds']:
    for arm in [a for a in cfg['arms'] if a!='T0']:
        directory=root/'students'/f'{arm}_{seed}'
        run('train.py','--config',config,'--bundle',bundle,'--videos',manifest+'/source_train.jsonl',
            '--arm',arm,'--seed',seed,'--output',directory,'--hours',6)
        state=json.loads((directory/'latest.json').read_text())
        assert state['status']=='complete','Training stopped at a resumable boundary; restart pipeline'
        spec={'arm':arm,'seed':seed,'predictions':str(root/'predictions'/f'{arm}_{seed}')}
        run('predict.py','--config',config,'--videos',manifest+'/target_test.jsonl',
            '--output',spec['predictions'],'--adapter',directory/state['checkpoint'],'--hours',3)
        specs.append(spec)
        (root/'completed-runs.json').write_text(json.dumps(specs,indent=2))
(root/'GPU_STAGES_COMPLETE.json').write_text(json.dumps({'runs':specs,'job_id':os.environ.get('JOB_ID')},indent=2))
print('GPU_STAGES_COMPLETE. Freeze/check all predictions, then analyze MOS on Mac.',flush=True)

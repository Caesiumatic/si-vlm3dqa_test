"""Final-only MOS join and paired content-group/bootstrap analysis on frozen predictions."""
import argparse
import collections
import hashlib
import json
import pathlib
import numpy as np
import openpyxl
from scipy.stats import rankdata


def correlation(x,y,rank=False):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:
        return float('nan')
    if rank:
        x,y=rankdata(x),rankdata(y)
    return float(np.corrcoef(x,y)[0,1])


def clean(value):
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--videos',required=True)
    p.add_argument('--runs',required=True,help='JSON list of arm,seed,predictions directory')
    p.add_argument('--mos',default='data/sealed/mos_main.xlsx');p.add_argument('--output',required=True)
    a=p.parse_args()
    cfg=json.loads(pathlib.Path(a.config).read_text())
    config_hash=hashlib.sha256(pathlib.Path(a.config).read_bytes()).hexdigest()
    rows=list(map(json.loads,pathlib.Path(a.videos).read_text().splitlines()))
    specs=json.loads(pathlib.Path(a.runs).read_text());predictions={};manifest={}
    expected={('T0',0)}|{(arm,seed) for arm in cfg['arms'] if arm!='T0' for seed in cfg['training_seeds']}
    assert {(r['arm'],r['seed']) for r in specs}==expected, 'Full predefined matrix required before unsealing MOS'
    for spec in specs:
        key=(spec['arm'],spec['seed']);values=[]
        for row in rows:
            path=pathlib.Path(spec['predictions'])/(row['video_id']+'.json')
            data=path.read_bytes();r=json.loads(data)
            assert r['config_sha256']==config_hash and r['video_sha256']==row['sha256']
            manifest[str(path)]=hashlib.sha256(data).hexdigest()
            values.append(r['score'] if r['status']=='valid' else np.nan)
        predictions[key]=np.array(values)
    out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
    # Persist hashes BEFORE opening human scores. Missing prediction files abort above.
    (out/'frozen-predictions.json').write_text(json.dumps(manifest,indent=2))
    workbook=openpyxl.load_workbook(a.mos,read_only=True,data_only=True)
    sheet=workbook.active;it=sheet.iter_rows(values_only=True)
    header=next(it);name_col=header.index('dis_name');mos_col=header.index('MOS')
    mos={str(r[name_col]):float(r[mos_col]) for r in it if r[name_col] is not None}
    assert len(mos)==1970
    human=np.array([mos[pathlib.Path(r['path']).stem] for r in rows]);assert np.all(np.isfinite(human))
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[r['group']].append(i)
    metrics=[]
    for (arm,seed),scores in predictions.items():
        mask=np.isfinite(scores)
        metrics.append({'arm':arm,'seed':seed,'valid':int(mask.sum()),'total':len(scores),
                        'unique_scores':len(set(scores[mask])),
                        'SRCC':correlation(scores[mask],human[mask],True),
                        'raw_PLCC':correlation(scores[mask],human[mask])})
    seeds=cfg['training_seeds'];rng=np.random.default_rng(cfg['statistics']['bootstrap_seed'])
    group_indices=list(groups.values());contrasts=[]
    contrasts_to_run=[('T3','T1'),('T2','T1'),('T3','T2'),('T3','T5')]+[(x,'T0') for x in cfg['arms'] if x!='T0']
    for first,second in contrasts_to_run:
        common=np.ones(len(rows),bool)
        for seed in seeds:
            for arm in (first,second):common &= np.isfinite(predictions[(arm,0 if arm=='T0' else seed)])
        indices=np.where(common)[0]
        def differences(selected,selected_seeds,rank):
            return [correlation(predictions[(first,s)][selected],human[selected],rank)-
                    correlation(predictions[(second,0 if second=='T0' else s)][selected],human[selected],rank)
                    for s in selected_seeds]
        for rank,name in [(True,'SRCC'),(False,'raw_PLCC')]:
            points=differences(indices,seeds,rank);boot=[]
            for _ in range(cfg['statistics']['bootstrap_repetitions']):
                sampled=np.concatenate([group_indices[i] for i in rng.integers(len(group_indices),size=len(group_indices))])
                sampled=sampled[common[sampled]]
                values=differences(sampled,rng.choice(seeds,size=len(seeds)),rank)
                if np.all(np.isfinite(values)):boot.append(float(np.mean(values)))
            contrasts.append({'contrast':first+'-'+second,'metric':name,'common_videos':len(indices),
                              'per_seed':points,'mean_difference':float(np.mean(points)),
                              'interval95':np.quantile(boot,[.025,.975]).tolist() if boot else None,
                              'valid_bootstrap':len(boot),'bootstrap':'paired content groups and paired training seeds'})
    result={'per_run':metrics,'contrasts':contrasts,'target_content_groups':len(groups),
            'limitations':['Single frozen teacher-label realization; bootstrap does not cover teacher randomness.',
                            'Three training seeds provide limited inference about optimization variability.',
                            'Undefined correlations from constant predictions are reported as null, never zero.']}
    (out/'metrics.json').write_text(json.dumps(clean(result),indent=2))
    print(json.dumps(clean(result),indent=2))


if __name__=='__main__':main()

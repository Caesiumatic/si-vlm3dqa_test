"""Freeze common-coverage labels and a broad-coverage fixed training schedule without MOS."""
import argparse
import collections
import hashlib
import json
import pathlib
import random
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--labels',required=True);p.add_argument('--pairs',required=True)
p.add_argument('--videos',required=True);p.add_argument('--config',required=True)
p.add_argument('--output',required=True)
a=p.parse_args()
cfg=json.loads(pathlib.Path(a.config).read_text())
pairs=list(map(json.loads,pathlib.Path(a.pairs).read_text().splitlines()))
videos={r['video_id']:r for r in map(json.loads,pathlib.Path(a.videos).read_text().splitlines())}
common=[];coverage=collections.Counter();cost=collections.Counter();statuses=collections.defaultdict(collections.Counter)
for pair in pairs:
    results={}
    for method in ('baseline','full'):
        path=pathlib.Path(a.labels)/(pair['pair_id']+'_'+method+'.json')
        r=json.loads(path.read_text()) # Missing outputs stop freezing, never shrink denominator silently.
        assert r['config_sha256']==hashlib.sha256(pathlib.Path(a.config).read_bytes()).hexdigest()
        results[method]=r;statuses[method][r['label']]+=1
        coverage[method]+=r['label']!='abstain';cost[method]+=r['elapsed_seconds']
    if all(r['label']!='abstain' for r in results.values()):
        common.append(dict(pair,baseline_label=results['baseline']['label'],full_label=results['full']['label'],
                           raw_reliability=results['full']['reliability']))
assert common, 'No common valid labels'
rng=random.Random(cfg['seed']);rng.shuffle(common)
remaining=common.copy();chosen=[];seen=set();group_counts=collections.Counter()
budget=cfg['training']['pairs_total']
while remaining and len(chosen)<budget:
    # Prefer unseen videos, then underrepresented scenes; stable random tie order is frozen.
    def priority(pair):
        return (len({pair['a'],pair['b']}-seen),
                -sum(group_counts[videos[pair[k]]['group']] for k in ('a','b')))
    best=max(range(len(remaining)),key=lambda i:priority(remaining[i]))
    pair=remaining.pop(best);chosen.append(pair);seen.update([pair['a'],pair['b']])
    group_counts.update(videos[pair[k]]['group'] for k in ('a','b'))
if len(chosen)<budget:
    original=chosen.copy()
    while len(chosen)<budget:
        chosen.append(original[(len(chosen)-len(original))%len(original)].copy())
weights=np.array([r['raw_reliability'] for r in chosen],float)
assert np.all(np.isfinite(weights)) and weights.mean()>0
weights/=weights.mean()
shuffled=weights.copy();np.random.default_rng(cfg['training']['weight_shuffle_seed']).shuffle(shuffled)
for r,w,s in zip(chosen,weights,shuffled):
    r['weight']=float(w);r['shuffled_weight']=float(s)
report={'candidate_pairs':len(pairs),'common_valid_pairs':len(common),'valid_by_method':dict(coverage),
        'label_counts':dict(statuses),'teacher_seconds':dict(cost),'schedule_pairs':len(chosen),
        'schedule_unique_pairs':len({p['pair_id'] for p in chosen}),'exposed_videos':len(seen),
        'exposed_groups':len(group_counts),'video_exposures_by_group':dict(group_counts),
        'weight_mean':float(weights.mean()),'weight_min':float(weights.min()),'weight_max':float(weights.max()),
        'weight_ess':float(weights.sum()**2/(weights**2).sum()),
        'weight_top10pct_share':float(np.sort(weights)[-max(1,int(.1*len(weights))):].sum()/weights.sum()),
        'own_full_output_view':'All teacher records retained; student runs use common valid subset. Full-output student view requires separate runs.'}
result={'report':report,'common':common,'schedule':chosen,
        'config_sha256':hashlib.sha256(pathlib.Path(a.config).read_bytes()).hexdigest()}
path=pathlib.Path(a.output)
assert not path.exists(), 'A frozen bundle must not be overwritten'
path.write_text(json.dumps(result,indent=2));print(json.dumps(report,indent=2))

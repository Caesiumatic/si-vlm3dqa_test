"""Deterministic scene-balanced subsets and connected pair graph, blind to MOS."""
import argparse
import collections
import hashlib
import json
import pathlib
import random

p = argparse.ArgumentParser()
p.add_argument('--source-per-group', type=int, required=True)
p.add_argument('--target-per-group', type=int, required=True)
p.add_argument('--seed', type=int, default=17)
p.add_argument('--output', required=True)
args = p.parse_args()
records = [json.loads(line) for line in pathlib.Path('data/manifests/videos.jsonl').read_text().splitlines()]
rng = random.Random(args.seed)
groups = collections.defaultdict(list)
for row in records:
    groups[(row['split'],row['group'])].append(row)
chosen = collections.defaultdict(list)
train_groups = []
for (split,group), rows in sorted(groups.items()):
    rows.sort(key=lambda r:r['video_id'])
    rng.shuffle(rows)
    count = args.source_per_group if split.startswith('source') else args.target_per_group
    sample = rows[:count]
    chosen[split].extend(sample)
    if split == 'source_train':
        train_groups.append(sample)
pairs = set()
for group in train_groups:
    for a,b in zip(group,group[1:]+group[:1]):
        if a['video_id'] != b['video_id']:
            pairs.add(tuple(sorted((a['video_id'],b['video_id']))))
for first,second in zip(train_groups,train_groups[1:]+train_groups[:1]):
    pairs.add(tuple(sorted((first[0]['video_id'],second[0]['video_id']))))
pair_records = [{'pair_id':hashlib.sha256('|'.join(pair).encode()).hexdigest()[:16],
                 'a':pair[0],'b':pair[1]} for pair in sorted(pairs)]
rng.shuffle(pair_records)
output = pathlib.Path(args.output)
output.mkdir(parents=True,exist_ok=True)
for split,rows in chosen.items():
    (output/(split+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
(output/'pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pair_records))
print(json.dumps({'split_counts':{s:len(r) for s,r in chosen.items()},'pairs':len(pair_records),
                  'source_groups':len(train_groups),'seed':args.seed}))

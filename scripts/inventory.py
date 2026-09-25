"""CPU-only video inventory and content-group split. Never reads MOS."""
import concurrent.futures
import hashlib
import json
import pathlib
import random
import imageio_ffmpeg


def inspect(item):
    domain, path = item
    group = path.stem.split('_')[0]
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        metadata = next(reader)
    finally:
        reader.close()
    return {'video_id': domain[0]+'_'+digest[:16], 'domain':domain, 'group':group,
            'path':str(path), 'sha256':digest, 'bytes':path.stat().st_size,
            'fps':metadata['fps'], 'duration':metadata['duration'], 'size':metadata['size'],
            'model_crop':'right_half_distorted_only' if domain=='source' else 'full_frame'}


def main():
    source = sorted(pathlib.Path('data/source').rglob('*.mp4'))
    target = sorted(pathlib.Path('data/target').glob('*.mp4'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(inspect, [('source',p) for p in source]+[('target',p) for p in target]))
    groups = sorted({r['group'] for r in records if r['domain']=='source'})
    rng = random.Random(17)
    shuffled = groups.copy(); rng.shuffle(shuffled)
    dev = set(shuffled[:4])
    source_hashes = {r['sha256'] for r in records if r['domain']=='source'}
    source_groups = set(groups)
    for row in records:
        if row['domain']=='source':
            row['split'] = 'source_dev' if row['group'] in dev else 'source_train'
        else:
            overlap = row['sha256'] in source_hashes or row['group'].lower() in {s.lower() for s in source_groups}
            row['split'] = 'target_overlap_review' if overlap else 'target_test'
    pathlib.Path('data/manifests').mkdir(exist_ok=True)
    with open('data/manifests/videos.jsonl','w') as f:
        for row in records:
            f.write(json.dumps(row)+'\n')
    from collections import Counter
    summary = {'counts':dict(Counter(r['split'] for r in records)),
               'source_groups':groups, 'dev_groups':sorted(dev),
               'target_groups':sorted({r['group'] for r in records if r['domain']=='target'}),
               'overlap_review_groups':sorted({r['group'] for r in records if r['split']=='target_overlap_review'}),
               'deduplication':'Exact SHA256 and conservative name overlap; visual near-duplicate audit pending.'}
    pathlib.Path('reports/video-inventory.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()

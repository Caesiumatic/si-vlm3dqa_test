"""Stream released MUGSQA archive, retaining RGB videos only; never render 3D."""
import json
import pathlib
import shutil
import tarfile
import urllib.request

root = pathlib.Path('data/target')
root.mkdir(parents=True, exist_ok=True)
metadata = json.load(urllib.request.urlopen('https://huggingface.co/api/datasets/Solivition/MUGSQA'))
revision = metadata['sha']
pathlib.Path('reports/mugsqa-revision.json').write_text(json.dumps({'revision': revision}))
url = f'https://huggingface.co/datasets/Solivition/MUGSQA/resolve/{revision}/main.tar.gz?download=true'
count = size = 0
with urllib.request.urlopen(url, timeout=180) as response:
    with tarfile.open(fileobj=response, mode='r|gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.lower().endswith('.mp4'):
                continue
            destination = root / pathlib.Path(member.name).name
            if not destination.exists() or destination.stat().st_size != member.size:
                temporary = destination.with_suffix('.part')
                with archive.extractfile(member) as source, temporary.open('wb') as out:
                    shutil.copyfileobj(source, out)
                temporary.rename(destination)
            count += 1
            size += member.size
            if count % 50 == 0:
                print(json.dumps({'videos': count, 'bytes': size}), flush=True)
print(json.dumps({'complete': True, 'videos': count, 'bytes': size}), flush=True)
sealed = pathlib.Path('data/sealed')
sealed.mkdir(exist_ok=True)
urllib.request.urlretrieve(f'https://huggingface.co/datasets/Solivition/MUGSQA/resolve/{revision}/mos_main.xlsx', sealed / 'mos_main.xlsx')

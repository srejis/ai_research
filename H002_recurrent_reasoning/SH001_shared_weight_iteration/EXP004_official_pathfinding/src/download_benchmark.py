"""Download only the selected official archives, retaining immutable raw bytes."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen,Request
from zipfile import ZipFile

BASE=Path(__file__).resolve().parents[1]
ROOT=Path(__file__).resolve().parents[4]
DATA=ROOT/'data/movingai_exp004'
MAPS=['maze-32-32-2','maze-32-32-4','random-32-32-10','random-32-32-20','room-32-32-4']
SOURCE='https://movingai.com/benchmarks/mapf/'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download():
    raw=DATA/'raw'
    raw.mkdir(parents=True,exist_ok=True)
    manifest_path=DATA/'download_manifest.json'
    old=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {'files':[]}
    records={r['archive']:r for r in old['files']}
    for name in MAPS:
        for suffix in ('.map.zip','.map-scen-even.zip','.map-scen-random.zip'):
            filename=name+suffix
            url=SOURCE+filename
            target=raw/filename
            if target.exists():
                if filename not in records or digest(target)!=records[filename]['sha256']:
                    raise RuntimeError(f'Existing raw archive has no matching provenance: {target}')
                continue
            print('DOWNLOAD '+url,flush=True)
            with urlopen(Request(url,headers={'User-Agent':'AIResearch-EXP004/1.0'}),timeout=120) as response:
                content=response.read()
                actual_url=response.url
            # Validate before committing original bytes; never replace an existing file.
            import io
            with ZipFile(io.BytesIO(content)) as archive:
                if archive.testzip() is not None:
                    raise RuntimeError('ZIP CRC failure')
                members=[]
                for entry in archive.infolist():
                    if entry.is_dir():
                        continue
                    payload=archive.read(entry.filename)
                    members.append({'name':entry.filename,'bytes':len(payload),
                                    'sha256':hashlib.sha256(payload).hexdigest()})
            with target.open('xb') as stream:
                stream.write(content)
            records[filename]={'archive':filename,'url':url,'actual_url':actual_url,
                               'downloaded_utc':datetime.now(timezone.utc).isoformat(),
                               'sha256':digest(target),'bytes':len(content),'members':members}
            manifest={'source_index':SOURCE+'index.html','license':'Open Data Commons Attribution License (per official MAPF page)',
                      'citation':'Stern et al., Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks, SoCS 2019, pp.151-158',
                      'files':list(records.values())}
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return json.loads(manifest_path.read_text(encoding='utf-8'))


if __name__=='__main__':
    manifest=download()
    print('Official archives:',len(manifest['files']))
    for item in manifest['files']:
        with ZipFile(DATA/'raw'/item['archive']) as z:
            first=next(m for m in z.namelist() if not m.endswith('/'))
            print(item['archive'],len(z.namelist()),first,z.read(first).decode('utf-8').splitlines()[:3])

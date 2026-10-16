"""Stage readable source files for a private versioned Kaggle source dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def stage(output,processed='data/processed/semrel'):
    repo=Path(__file__).resolve().parents[1];output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh staging directory')
    output.mkdir(parents=True,exist_ok=True)
    files={}
    paths=list((repo/'src').rglob('*.py'))+[repo/name for name in
        ('LICENSE','docs/model-protocol-v1.json','docs/provenance.md','docs/component-audit.json')]
    for path in sorted(paths):
        relative=path.relative_to(repo).as_posix();filename=relative.replace('/','__')
        shutil.copyfile(path,output/filename)
        files[filename]={'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    audit=json.loads((repo/processed/'audit.json').read_text(encoding='utf-8'))
    manifest={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              'files':files,'processed_sha256':audit['processed_sha256'],
              'transport':'Ordinary readable files; explicit path/checksum manifest. No encoded source.'}
    (output/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    metadata={'id':'aikyatansinha/portfolio-semrel-african-source-v1',
              'title':'Portfolio SemRel African Source v1','licenses':[{'name':'other'}],
              'description':'Private original source code and research protocol. Source code MIT; no dataset text, labels or weights included.'}
    (output/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--processed',default='data/processed/semrel')
    stage(**vars(parser.parse_args()))

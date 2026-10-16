"""Stage plain reviewed source and complete private prepared pairs, never test gold."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

def stage(output,processed,protocol='docs/full-cohort-v3-protocol.json',dataset_id='aikyatansinha/portfolio-semrel-full-cohort-v3'):
    protocol=Path(protocol)
    root=Path(__file__).resolve().parents[1];output=Path(output);processed=Path(processed)
    if output.exists():raise ValueError('Fresh private staging required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Commit reviewed source before staging')
    for line in (processed/'test.jsonl').read_text(encoding='utf-8').splitlines():
        if set(json.loads(line))!={'id','language','sentence1','sentence2'}:raise ValueError('Test input must contain text/identity only')
    output.mkdir(parents=True);files=[]
    def copy(path,name):
        flat=name.replace('/','__');content=path.read_bytes();(output/flat).write_bytes(content)
        files.append({'path':name,'flat':flat,'sha256':hashlib.sha256(content).hexdigest()})
    for path in sorted((root/'src').rglob('*.py')):copy(path,path.relative_to(root).as_posix())
    plan=json.loads((root/protocol).read_text(encoding='utf-8'))
    coverage=Path(plan.get('coverage_file','docs/full-cohort-v3-coverage.json'))
    for path in [coverage,protocol]:copy(root/path,path.as_posix())
    for name in ['train.jsonl','dev.jsonl','test.jsonl','audit.json']:copy(processed/name,'processed/'+name)
    audit=json.loads((processed/'audit.json').read_text());plan=json.loads((root/protocol).read_text())
    if audit['processed_sha256']!=plan['processed_sha256']:raise ValueError('Frozen cohort mismatch')
    manifest={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'files':files,'processed_sha256':audit['processed_sha256'],'languages':plan['languages'],'protocol_path':protocol.as_posix(),'test_gold_included':False}
    (output/'full-cohort-source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    (output/'dataset-metadata.json').write_text(json.dumps({'id':dataset_id,'title':'Private SemRel Human Cohort','licenses':[{'name':'other'}]},indent=2)+'\n',encoding='utf-8',newline='\n')
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--processed',default='data/processed/semrel-full-v3');p.add_argument('--protocol',default='docs/full-cohort-v3-protocol.json');p.add_argument('--dataset-id',default='aikyatansinha/portfolio-semrel-full-cohort-v3');stage(**vars(p.parse_args()))

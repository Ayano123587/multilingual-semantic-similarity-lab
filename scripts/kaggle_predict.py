"""Verify actual saved-head text inference from ordinary private source files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    started=time.time();work=Path('/kaggle/working')
    os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
    manifests=list(Path('/kaggle/input').rglob('source-manifest.json'))
    if len(manifests)!=1:raise ValueError('Expected one private source manifest')
    manifest=json.loads(manifests[0].read_text());root=Path('/kaggle/temp/selected-source')
    for filename,contract in manifest['files'].items():
        source=manifests[0].parent/filename;relative=Path(contract['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid source path')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=contract['sha256']:raise ValueError('Private source hash mismatch')
        output=root/relative;output.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,output)
    sys.path.insert(0,str(root))
    subprocess.run([sys.executable,'-m','pip','install','transformers==4.57.6','sentencepiece==0.2.1','librosa==0.11.0'],check=True,timeout=360)
    import numpy as np
    import torch
    from src.models.predict import predict
    from src.models.train import write_json
    torch.set_num_threads(2)
    expected=json.loads((root/'expected-scores.json').read_text())
    result=predict(root/'selected.pt',root/'input-pairs.jsonl',work/'predictions-private.json')
    p=np.asarray(result['predictions']);reference=np.asarray(expected['predictions'])
    difference=float(np.max(np.abs(p-reference)));symmetry=float(np.max(np.abs(p[:3]-p[3:])))
    if p.shape!=(6,) or not np.isfinite(p).all() or ((p<0)|(p>1)).any() or difference>1e-4 or symmetry>1e-6:
        raise RuntimeError('Actual text/cache/symmetry inference contract failed')
    proof={'rows':6,'real_source':'First training pair per language and reversed order; no target input',
           'cache_prediction_max_difference':difference,'reversal_max_difference':symmetry,
           'finite_scores_in_unit_interval':True,'accuracy_claim':False,'test_gold_loaded':False,
           'checkpoint_sha256':hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest(),
           'input_sha256':hashlib.sha256((root/'input-pairs.jsonl').read_bytes()).hexdigest(),
           'source_manifest_sha256':hashlib.sha256(manifests[0].read_bytes()).hexdigest(),
           'source_revision':manifest['source_revision'],'torch':str(torch.__version__),
           'elapsed_seconds':time.time()-started}
    write_json(work/'actual-inference-proof.json',proof)
    write_json(work/'completion.json',{'exit':0,'elapsed_seconds':time.time()-started})


if __name__=='__main__':
    try:main()
    except BaseException as error:
        (Path('/kaggle/working')/'failure.json').write_text(json.dumps({'type':type(error).__name__,
            'error':str(error),'traceback':traceback.format_exc()},indent=2))
        raise

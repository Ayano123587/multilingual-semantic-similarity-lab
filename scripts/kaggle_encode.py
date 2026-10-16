"""Readable private CPU encoder runner. Source files are mounted separately."""
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

START=time.time()
WORK=Path('/kaggle/working')


def main():
    for name,value in {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','HF_HUB_DISABLE_XET':'1',
                       'HF_HUB_DOWNLOAD_TIMEOUT':'120','HF_HUB_ETAG_TIMEOUT':'60'}.items():
        os.environ[name]=value
    manifests=list(Path('/kaggle/input').rglob('source-manifest.json'))
    if len(manifests)!=1:raise ValueError('Missing or ambiguous private source manifest')
    manifest=json.loads(manifests[0].read_text())
    source=Path('/kaggle/temp/semrel-source');source.mkdir(parents=True,exist_ok=True)
    for filename,contract in manifest['files'].items():
        original=manifests[0].parent/filename
        if hashlib.sha256(original.read_bytes()).hexdigest()!=contract['sha256']:
            raise ValueError('Private source checksum mismatch')
        relative=Path(contract['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid source path')
        destination=source/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(original,destination)
    sys.path.insert(0,str(source))
    subprocess.run([sys.executable,'-m','pip','install','transformers==4.57.6',
                    'sentencepiece==0.2.1','librosa==0.11.0'],check=True,timeout=360)
    import numpy as np
    import torch
    import transformers
    from src.data.make_dataset import prepare
    from src.models.encoder import Encoder,cache_dataset
    processed=WORK/'processed'
    audit=prepare('/kaggle/temp/semrel-raw',processed)
    expected=manifest['processed_sha256']
    if audit['processed_sha256']!=expected:raise ValueError('Prepared source contracts changed')
    rows=[json.loads(line) for line in (processed/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    texts=[next(r for r in rows if r['language']==lang)[key]
           for lang in ('amh','hau','kin') for key in ('sentence1','sentence2')]
    encoder=Encoder('cpu')
    a,diagnostic=encoder.encode(texts,batch_size=2);b,_=encoder.encode(texts,batch_size=2)
    if a.shape!=(6,768) or not np.isfinite(a).all() or not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-5) or not np.array_equal(a,b):
        raise RuntimeError('Encoder smoke failed')
    (WORK/'smoke.json').write_text(json.dumps({'sentences':6,'shape':list(a.shape),'finite':True,
        'replay_max_difference':0.,'diagnostics':diagnostic,'loading':encoder.loading,
        'elapsed_seconds':time.time()-START,'test_gold_parsed':False},indent=2))
    del encoder;gc.collect()
    cache_dataset(processed,WORK/'embeddings','cpu')
    (WORK/'environment.json').write_text(json.dumps({'python':platform.python_version(),
        'torch':str(torch.__version__),'transformers':transformers.__version__,'numpy':np.__version__,
        'cpu_threads':2,'source_revision':manifest['source_revision'],
        'source_manifest_sha256':hashlib.sha256(manifests[0].read_bytes()).hexdigest()},indent=2))
    (WORK/'completion.json').write_text(json.dumps({'exit':0,'elapsed_seconds':time.time()-START,
        'test_gold_parsed':False},indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        (WORK/'failure.json').write_text(json.dumps({'type':type(error).__name__,'error':str(error),
            'elapsed_seconds':time.time()-START,'traceback':traceback.format_exc()},indent=2))
        raise

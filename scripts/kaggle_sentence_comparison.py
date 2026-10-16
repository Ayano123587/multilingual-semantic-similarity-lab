"""Private complete nine-language sentence-encoder comparison; test gold absent."""
import gc,hashlib,json,os,platform,shutil,subprocess,sys,time,traceback
from pathlib import Path
WORK=Path('/kaggle/working')


def main():
    start=time.monotonic()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
    os.environ['HF_HUB_DISABLE_XET']='1'
    manifests=list(Path('/kaggle/input').rglob('full-cohort-source-manifest.json'))
    if len(manifests)!=1:raise ValueError('One private readable source manifest required')
    manifest=json.loads(manifests[0].read_text());root=Path('/kaggle/temp/semrel-sentence-comparison');root.mkdir(parents=True,exist_ok=False)
    seen=set()
    for entry in manifest['files']:
        relative=Path(entry['path']);flat=entry['flat']
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen or Path(flat).name!=flat:raise ValueError('Unsafe source entry')
        seen.add(relative.as_posix());content=(manifests[0].parent/flat).read_bytes()
        if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise ValueError('Changed mounted source')
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    os.chdir(root);sys.path.insert(0,str(root))
    subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','sentencepiece==0.2.1','librosa==0.11.0','scikit-learn>=1.6,<2'],check=True)
    import numpy as np
    import torch
    import importlib.metadata
    from src.models.encoder_variants import Encoder,cache_dataset,extend_nontruncated_afro
    from src.models.sentence_comparison import train
    from src.models.full_inference import predict,load_split,predict_features
    from src.data.make_dataset import sha
    from src.models.train import Head
    from src.features.build_features import fit_scaler,scale,pair_features
    plan=json.loads((root/manifest['protocol_path']).read_text(encoding='utf-8'))
    if len(plan['languages'])!=9:raise ValueError('All nine compatible languages required')
    for split,expected in plan['processed_sha256'].items():
        if sha(root/'processed'/(split+'.jsonl'))!=expected:raise ValueError('Frozen complete source mismatch')
    test_rows=[json.loads(s) for s in (root/'processed/test.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(test_rows)!=3969 or any('target' in r for r in test_rows):raise ValueError('Incomplete or unsealed test inputs')
    dev=[json.loads(s) for s in (root/'processed/dev.jsonl').read_text(encoding='utf-8').splitlines()]
    examples=[next(r for r in dev if r['language']==lang) for lang in plan['languages']]
    texts=[r[k] for r in examples for k in ('sentence1','sentence2')]
    expected_receipt=plan['afro_reuse']['receipt_sha256']
    reused=[p for p in Path('/kaggle/input').rglob('receipt.json') if sha(p)==expected_receipt]
    if len(reused)>1:raise ValueError('Ambiguous duplicate verified feature artifacts')
    old_cache=reused[0].parent if reused else None
    old_processed=old_cache.parent/'processed' if old_cache else None
    fresh_embeddings={};smokes={}
    for variant in ('afro','e5'):
        encoder=Encoder(variant,512);values,diagnostic=encoder.encode(texts,batch_size=2);repeat,_=encoder.encode(texts,batch_size=2)
        if not np.array_equal(values,repeat) or not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5):raise ValueError('Actual nine-language encoder smoke failed')
        fresh_embeddings[variant]=values.reshape(9,2,768)
        smokes[variant]={'languages':plan['languages'],'sentences':18,'repeat_max_difference':0.,'diagnostics':diagnostic,'loading':encoder.loading}
        del encoder;gc.collect()
        cache=WORK/('embeddings-'+variant)
        if variant=='afro' and old_cache is not None:receipt=extend_nontruncated_afro(old_cache,old_processed,root/'processed',cache,expected_receipt)
        else:receipt=cache_dataset(root/'processed',cache,variant,512)
        vx,_,_,rows,_=load_split(cache,root/'processed','dev')
        indices=[next(i for i,r in enumerate(rows) if (r['language'],r['id'])==(e['language'],e['id'])) for e in examples]
        fresh=pair_features(fresh_embeddings[variant],examples);feature_delta=float(np.max(abs(fresh-vx[indices])))
        if feature_delta>1e-5:raise ValueError('Actual512-token fresh encoder versus cache mismatch')
        smokes[variant]['actual_fresh512_pair_feature_max_difference']=feature_delta
        # Two real training-batch updates, gradient and checkpoint reload checks.
        tx,ty,_,_,_=load_split(cache,root/'processed','train');mean,std=fit_scaler(tx);batch=torch.from_numpy(scale(tx[:64],mean,std));target=torch.from_numpy(ty[:64])
        torch.manual_seed(42);head=Head(1539,128,.2);optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);losses=[]
        for _ in range(2):
            optimizer.zero_grad();loss=(head(batch)-target).square().mean();loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in head.parameters()):raise ValueError('Nonfinite actual head smoke')
            optimizer.step();losses.append(float(loss.detach()))
        head.eval();torch.save(head.state_dict(),WORK/(variant+'-smoke.pt'));reloaded=Head(1539,128,.2);reloaded.load_state_dict(torch.load(WORK/(variant+'-smoke.pt'),weights_only=True));reloaded.eval()
        with torch.inference_mode():delta=float((head(batch)-reloaded(batch)).abs().max())
        if delta>1e-6:raise ValueError('Smoke saved-head replay failed')
        smokes[variant]['head_training']={'actual_rows':64,'losses':losses,'reload_max_difference':delta,'accuracy_claim':False}
        (WORK/'smoke.json').write_text(json.dumps(smokes,indent=2)+'\n');del head,reloaded,optimizer,tx;gc.collect()
    selection=train(WORK/'embeddings-afro',WORK/'embeddings-e5',root/'processed',WORK/'heads',root/manifest['protocol_path'])
    # Production text CLI path is exercised only after global model serialization.
    pairs=[{'sentence1':e['sentence1'],'sentence2':e['sentence2']} for e in examples]
    pairs += [{'sentence1':e['sentence2'],'sentence2':e['sentence1']} for e in examples]
    input_path=WORK/'inference-input.jsonl';input_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in pairs),encoding='utf-8',newline='\n')
    actual=predict(WORK/'heads/selected.pt',input_path,WORK/'actual-text-predictions.json')
    bundle=torch.load(WORK/'heads/selected.pt',map_location='cpu',weights_only=True)
    variant=selection['selected_candidate']['encoder_variant'];vx,_,_,rows,_=load_split(WORK/('embeddings-'+variant),root/'processed','dev')
    indices=[next(i for i,r in enumerate(rows) if (r['language'],r['id'])==(e['language'],e['id'])) for e in examples]
    cached=predict_features(bundle,vx[indices]);actual_values=np.asarray(actual['predictions'])
    delta=float(np.max(abs(cached-actual_values[:9])));reversal=float(np.max(abs(actual_values[:9]-actual_values[9:])))
    if delta>1e-5 or reversal>1e-7:raise ValueError('Actual saved production text inference mismatch')
    (WORK/'inference-proof.json').write_text(json.dumps({'rows':18,'languages':plan['languages'],'checkpoint_sha256':sha(WORK/'heads/selected.pt'),'actual_text_vs_cache_max_difference':delta,'reversal_max_difference':reversal,'test_gold_loaded':False},indent=2)+'\n')
    shutil.copytree(root/'processed',WORK/'processed')
    versions={name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','scikit-learn','sentencepiece','huggingface-hub','safetensors')}
    (WORK/'completion.json').write_text(json.dumps({'source_commit':manifest['commit'],'source_manifest_sha256':sha(manifests[0]),'protocol_sha256':sha(root/manifest['protocol_path']),'python':platform.python_version(),'versions':versions,'threads':torch.get_num_threads(),'seconds':time.monotonic()-start,'selected':selection['selected'],'test_gold_loaded':False},indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        (WORK/'failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise

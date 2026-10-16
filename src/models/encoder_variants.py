"""Pinned multilingual sentence encoders with explicit pooling and source checks."""
import hashlib,json
from pathlib import Path
import numpy as np

VARIANTS={
 'afro':{'model':'Davlan/afro-xlmr-base','revision':'25f27299c247a6b73a767bd82d12444138b19337','sha256':'f73b04316c9c2e76fcb213957616a09e1eb44112266717f649fc1897dac42953','prefix':'','license':'MIT'},
 'e5':{'model':'intfloat/multilingual-e5-base','revision':'d128750597153bb5987e10b1c3493a34e5a4502a','sha256':'a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7','prefix':'query: ','license':'MIT'}}


def pool_and_normalize(hidden,mask):
    import torch
    if hidden.ndim!=3 or mask.shape!=hidden.shape[:2] or not torch.isfinite(hidden).all() or not ((mask==0)|(mask==1)).all() or not mask.any(1).all():raise ValueError('Aligned finite states and nonempty binary masks required')
    weights=mask.unsqueeze(-1).to(hidden.dtype);pooled=(hidden*weights).sum(1)/weights.sum(1)
    norm=pooled.norm(dim=1,keepdim=True)
    if not (norm>0).all():raise ValueError('Zero sentence embedding')
    return pooled/norm


class Encoder:
    def __init__(self,variant,max_tokens=512):
        import torch
        from transformers import AutoModel,AutoTokenizer
        from huggingface_hub import hf_hub_download
        if variant not in VARIANTS or max_tokens not in (128,512):raise ValueError('Unsupported encoder contract')
        self.spec=VARIANTS[variant];self.variant=variant;self.max_tokens=max_tokens;self.torch=torch
        torch.set_num_threads(2)
        weight=hf_hub_download(self.spec['model'],'model.safetensors',revision=self.spec['revision'])
        with Path(weight).open('rb') as h:actual=hashlib.file_digest(h,'sha256').hexdigest()
        if actual!=self.spec['sha256']:raise ValueError('Pinned encoder weight mismatch')
        self.tokenizer=AutoTokenizer.from_pretrained(self.spec['model'],revision=self.spec['revision'],trust_remote_code=False)
        self.model,info=AutoModel.from_pretrained(self.spec['model'],revision=self.spec['revision'],use_safetensors=True,add_pooling_layer=False,output_loading_info=True,trust_remote_code=False)
        if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):raise ValueError('Incomplete pretrained backbone')
        allowed={'pooler.dense.weight','pooler.dense.bias'}
        if any(k not in allowed and not (variant=='afro' and k.startswith('lm_head.')) for k in info.get('unexpected_keys',[])):raise ValueError('Unexpected encoder parameters')
        self.model=self.model.eval().requires_grad_(False);self.loading=info
    def encode(self,texts,batch_size=16):
        if not texts or any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Nonempty real source sentences required')
        output=[];truncated=0;maximum=0
        for start in range(0,len(texts),batch_size):
            prefixed=[self.spec['prefix']+t for t in texts[start:start+batch_size]]
            lengths=[len(t) for t in self.tokenizer(prefixed,truncation=False)['input_ids']]
            truncated+=sum(n>self.max_tokens for n in lengths);maximum=max(maximum,max(lengths))
            tokens=self.tokenizer(prefixed,padding=True,truncation=True,max_length=self.max_tokens,return_tensors='pt')
            with self.torch.inference_mode():values=pool_and_normalize(self.model(**tokens).last_hidden_state,tokens['attention_mask'])
            output.append(values.numpy().astype(np.float32))
        return np.concatenate(output),{'sentences':len(texts),'truncated_sentences':truncated,'maximum_untruncated_tokens':maximum}


def cache_dataset(processed,output,variant,max_tokens=512):
    from src.data.make_dataset import sha
    processed,output=Path(processed),Path(output)
    if output.exists():raise ValueError('Fresh encoder output required')
    output.mkdir(parents=True);encoder=Encoder(variant,max_tokens);splits={}
    for split in ('train','dev','test'):
        source=processed/(split+'.jsonl');rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
        if split=='test' and any('target' in r for r in rows):raise ValueError('Test input must exclude gold')
        unique=list(dict.fromkeys(r[k] for r in rows for k in ('sentence1','sentence2')))
        values,diagnostic=encoder.encode(unique);lookup=dict(zip(unique,values))
        arrays={'embeddings':np.stack([[lookup[r['sentence1']],lookup[r['sentence2']]] for r in rows]),'ids':np.asarray([r['id'] for r in rows]),'languages':np.asarray([r['language'] for r in rows])}
        if split!='test':arrays['targets']=np.asarray([r['target'] for r in rows],np.float32)
        path=output/(split+'.npz');np.savez_compressed(path,**arrays)
        splits[split]={'rows':len(rows),**diagnostic,'source_sha256':sha(source),'cache_sha256':sha(path)}
        print(json.dumps({'variant':variant,'split':split,**splits[split]}),flush=True)
    receipt={'model':encoder.spec['model'],'revision':encoder.spec['revision'],'weight_sha256':{'model.safetensors':encoder.spec['sha256']},'variant':variant,'prefix':encoder.spec['prefix'],'max_tokens':max_tokens,'pooling':'attention-mask mean; no pooler; L2','loading':encoder.loading,'splits':splits,'test_gold_loaded':False}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    return receipt


def extend_nontruncated_afro(old_cache,old_processed,processed,output,expected_receipt_sha256):
    """Reuse identical nontruncated inputs; encode every newly added sentence."""
    from src.data.make_dataset import sha
    old_cache,old_processed,processed,output=map(Path,(old_cache,old_processed,processed,output))
    if output.exists():raise ValueError('Fresh extended cache required')
    if sha(old_cache/'receipt.json')!=expected_receipt_sha256:raise ValueError('Unverified reused receipt')
    previous=json.loads((old_cache/'receipt.json').read_text(encoding='utf-8'))
    spec=VARIANTS['afro']
    if (previous['model'],previous['revision'])!=(spec['model'],spec['revision']) or previous['weight_sha256']!={'model.safetensors':spec['sha256']} or previous['max_tokens']!=128:
        raise ValueError('Reused encoder identity differs')
    inputs={}
    for split in ('train','dev','test'):
        contract=previous['splits'][split]
        old_source=old_processed/(split+'.jsonl');source=processed/(split+'.jsonl');cache=old_cache/(split+'.npz')
        if contract['truncated_sentences']!=0 or sha(old_source)!=contract['source_sha256'] or sha(cache)!=contract['cache_sha256']:
            raise ValueError('Changed or truncated reused source')
        if not source.read_bytes().startswith(old_source.read_bytes()):raise ValueError('New cohort does not retain exact old prefix')
        old_rows=[json.loads(v) for v in old_source.read_text(encoding='utf-8').splitlines()]
        rows=[json.loads(v) for v in source.read_text(encoding='utf-8').splitlines()]
        with np.load(cache,allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in data.files}
        embeddings=arrays['embeddings']
        if embeddings.shape!=(len(old_rows),2,768) or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings,axis=2),1,atol=1e-4):raise ValueError('Invalid reused embeddings')
        if not np.array_equal(arrays['ids'],[r['id'] for r in old_rows]) or not np.array_equal(arrays['languages'],[r['language'] for r in old_rows]):raise ValueError('Reused rows misaligned')
        if split=='test' and ('targets' in arrays or any('target' in r for r in rows)):raise ValueError('Test gold must remain absent')
        inputs[split]=(rows,old_rows,arrays)
    encoder=Encoder('afro',512);output.mkdir(parents=True);reports={}
    for split,(rows,old_rows,arrays) in inputs.items():
        lookup={row[key]:arrays['embeddings'][i,j] for i,row in enumerate(old_rows) for j,key in enumerate(('sentence1','sentence2'))}
        reused=len(lookup);unique=list(dict.fromkeys(r[k] for r in rows for k in ('sentence1','sentence2')))
        missing=[t for t in unique if t not in lookup]
        diagnostic={'truncated_sentences':0,'maximum_untruncated_tokens':0}
        if missing:
            values,diagnostic=encoder.encode(missing);lookup.update(zip(missing,values))
        tail=rows[len(old_rows):]
        pairs=np.concatenate([arrays['embeddings'],np.stack([[lookup[r['sentence1']],lookup[r['sentence2']]] for r in tail])]) if tail else arrays['embeddings']
        values={'embeddings':pairs,'ids':np.asarray([r['id'] for r in rows]),'languages':np.asarray([r['language'] for r in rows])}
        if split!='test':values['targets']=np.asarray([r['target'] for r in rows],np.float32)
        path=output/(split+'.npz');np.savez_compressed(path,**values)
        reports[split]={'rows':len(rows),'sentences':len(unique),'reused_nontruncated_sentences':reused,'newly_encoded_sentences':len(missing),
            'truncated_sentences':diagnostic['truncated_sentences'],'maximum_token_count_upper_bound':max(128,diagnostic['maximum_untruncated_tokens']),
            'source_sha256':sha(processed/(split+'.jsonl')),'cache_sha256':sha(path)}
        print(json.dumps({'variant':'afro','split':split,**reports[split]}),flush=True)
    receipt={'model':spec['model'],'revision':spec['revision'],'weight_sha256':{'model.safetensors':spec['sha256']},'variant':'afro','prefix':'','max_tokens':512,
        'pooling':'attention-mask mean; no pooler; L2','loading':encoder.loading,'splits':reports,'test_gold_loaded':False,
        'reused_receipt_sha256':expected_receipt_sha256,'reuse_equivalence':'Identical pinned tokenizer/model/prefix; zero prior128-token truncation means512-token limit changes no old input token sequence. New sentences actually encoded at512.'}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
    return receipt

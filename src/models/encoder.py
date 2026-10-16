"""Pinned real pretrained African-language encoder; no random pooling layer."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.config import MODEL_ID,MODEL_REVISION


class Encoder:
    def __init__(self,device='cpu'):
        import torch
        from transformers import AutoModel,AutoTokenizer
        torch.set_num_threads(2)
        self.torch=torch;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,trust_remote_code=False)
        self.model,loading=AutoModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,
            add_pooling_layer=False,output_loading_info=True,trust_remote_code=False)
        if loading.get('missing_keys') or loading.get('mismatched_keys') or loading.get('error_msgs'):
            raise ValueError('Pretrained encoder failed strict loading')
        if any(not key.startswith('lm_head.') for key in loading.get('unexpected_keys',[])):
            raise ValueError('Unexpected pretrained encoder keys')
        self.model=self.model.to(device).eval().requires_grad_(False)
        self.loading=loading

    def encode(self,texts,batch_size=16):
        if not texts or any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Nonempty text required')
        values=[];truncated=0
        for start in range(0,len(texts),batch_size):
            batch=texts[start:start+batch_size]
            lengths=self.tokenizer(batch,add_special_tokens=True,truncation=False)['input_ids']
            truncated+=sum(len(row)>128 for row in lengths)
            tokens=self.tokenizer(batch,padding=True,truncation=True,max_length=128,return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                state=self.model(**tokens).last_hidden_state
                mask=tokens['attention_mask'].unsqueeze(-1).to(state.dtype)
                pooled=(state*mask).sum(1)/mask.sum(1).clamp_min(1)
                pooled=self.torch.nn.functional.normalize(pooled,p=2,dim=1)
            if not self.torch.isfinite(pooled).all():raise ValueError('Nonfinite encoder output')
            values.append(pooled.cpu().numpy())
        return np.concatenate(values).astype(np.float32),{'sentences':len(texts),'truncated_sentences':truncated}


def cache_dataset(processed,output,device='cpu'):
    processed=Path(processed);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh encoder cache')
    output.mkdir(parents=True,exist_ok=True);encoder=Encoder(device);reports={}
    for split in ('train','dev','test'):
        path=processed/f'{split}.jsonl'
        rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        unique=list(dict.fromkeys(row[key] for row in rows for key in ('sentence1','sentence2')))
        embeddings,diagnostic=encoder.encode(unique);lookup=dict(zip(unique,embeddings))
        pairs=np.stack([[lookup[row['sentence1']],lookup[row['sentence2']]] for row in rows])
        values={'embeddings':pairs,'ids':np.asarray([row['id'] for row in rows]),
                'languages':np.asarray([row['language'] for row in rows])}
        if split!='test':values['targets']=np.asarray([row['target'] for row in rows],np.float32)
        np.savez_compressed(output/f'{split}.npz',**values)
        reports[split]={'rows':len(rows),**diagnostic,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'cache_sha256':hashlib.sha256((output/f'{split}.npz').read_bytes()).hexdigest()}
        print(json.dumps({'encoded_split':split,**reports[split]}),flush=True)
    from huggingface_hub import try_to_load_from_cache
    weights={}
    for filename in ('model.safetensors','pytorch_model.bin'):
        path=try_to_load_from_cache(MODEL_ID,filename,revision=MODEL_REVISION)
        if isinstance(path,str) and Path(path).is_file():
            digest=hashlib.sha256()
            with Path(path).open('rb') as handle:
                for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
            weights[filename]=digest.hexdigest()
    receipt={'model':MODEL_ID,'revision':MODEL_REVISION,'pooling':'attention-mask mean; no pooler; L2',
             'max_tokens':128,'device':device,'loading':encoder.loading,'weight_sha256':weights,'splits':reports,
             'test_gold_loaded':False}
    if not weights:raise ValueError('No attested pretrained weight file')
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return receipt

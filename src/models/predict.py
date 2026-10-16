"""Actual text-pair inference using the pinned encoder and saved regressor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.models.encoder import Encoder
from src.models.train import predict_features,validate_bundle,write_json
from src.features.build_features import pair_features,lexical


def predict(checkpoint,input,output,device='cpu'):
    output=Path(output)
    if output.exists():raise ValueError('Use a fresh inference report path')
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=True);validate_bundle(bundle)
    rows=[json.loads(line) for line in Path(input).read_text(encoding='utf-8').splitlines()]
    if not rows or any(not isinstance(row.get(k),str) or not row[k].strip() for row in rows for k in ('sentence1','sentence2')):
        raise ValueError('Expected nonempty JSONL sentence pairs')
    if bundle['kind']=='baseline' and bundle['name'] in ('word_jaccard','character_jaccard','train_mean'):
        # A lexical or constant winner has no active encoder dependency.
        features=np.zeros((len(rows),1539),np.float32);features[:,-2:]=lexical(rows)
        result={'predictions':predict_features(bundle,features).tolist(),'relatedness_not_probability':True,
                'diagnostics':{'encoded_sentences':0},'active_model':bundle['name'],'rows':len(rows)}
        output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
        return result
    encoder=Encoder(device)
    # Loading is pinned by revision; additionally attest actual weight bytes.
    from huggingface_hub import try_to_load_from_cache
    import hashlib
    for filename,digest in bundle['weight_sha256'].items():
        path=try_to_load_from_cache(bundle['encoder'],filename,revision=bundle['revision'])
        if not isinstance(path,str):raise ValueError('Expected encoder weight cache missing')
        h=hashlib.sha256()
        with Path(path).open('rb') as handle:
            for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
        if h.hexdigest()!=digest:raise ValueError('Actual encoder weight identity mismatch')
    unique=list(dict.fromkeys(row[k] for row in rows for k in ('sentence1','sentence2')))
    values,diagnostics=encoder.encode(unique);lookup=dict(zip(unique,values))
    pairs=np.stack([[lookup[row['sentence1']],lookup[row['sentence2']]] for row in rows])
    probability=predict_features(bundle,pair_features(pairs,rows))
    result={'predictions':probability.tolist(),'relatedness_not_probability':True,'diagnostics':diagnostics,
            'encoder':bundle['encoder'],'revision':bundle['revision'],'rows':len(rows)}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','input','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cpu')
    predict(**vars(parser.parse_args()))

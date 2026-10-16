"""Safe saved-model inference for either pinned multilingual encoder."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from src.models.encoder_variants import VARIANTS,Encoder
from src.models.train import Head,write_json
from src.features.build_features import pair_features,lexical,scale
from src.data.make_dataset import sha


def validate_bundle(bundle):
    try:
        matches=[name for name,spec in VARIANTS.items() if bundle['encoder']==spec['model'] and bundle['revision']==spec['revision']]
        if len(matches)!=1 or bundle['schema']!='semrel_symmetric_v1' or bundle['width']!=1539:raise ValueError('Unsupported source/schema')
        variant=matches[0];spec=VARIANTS[variant]
        if bundle['weight_sha256']!={'model.safetensors':spec['sha256']} or bundle.get('prefix',spec['prefix'])!=spec['prefix'] or bundle.get('max_tokens',128) not in (128,512):raise ValueError('Changed encoder feature contract')
        if bundle['kind']=='baseline':
            if bundle['name'] not in ('cosine','word_jaccard','character_jaccard','train_mean') or not np.isfinite(bundle['train_mean']) or not 0<=bundle['train_mean']<=1:raise ValueError('Invalid baseline')
        elif bundle['kind']=='score_prior':
            if bundle['mode']=='convex':
                alpha=bundle['alpha'];component=bundle['component']
                if type(alpha) is not float or not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('Invalid global mixture weight')
                if component['kind']!='mlp' or any(component[k]!=bundle[k] for k in ('encoder','revision','weight_sha256','prefix','max_tokens')):raise ValueError('Mixture encoder/component mismatch')
                validate_bundle(component)
            elif bundle['mode']=='nonnegative_linear':
                scale(np.zeros((1,3),np.float32),bundle['mean'],bundle['std'])
                if bundle['coefficient'].shape!=(3,) or not torch.isfinite(bundle['coefficient']).all() or (bundle['coefficient']<0).any() or not np.isfinite(bundle['intercept']):raise ValueError('Invalid monotone score coefficients')
            else:raise ValueError('Unknown score-prior model')
        else:
            scale(np.zeros((1,1539),np.float32),bundle['mean'],bundle['std'])
            if bundle['kind']=='ridge':
                if bundle['coefficient'].shape!=(1539,) or not torch.isfinite(bundle['coefficient']).all() or not np.isfinite(bundle['intercept']):raise ValueError('Invalid ridge')
            elif bundle['kind']=='mlp':
                if len(bundle['components'])!=3 or {c.get('seed') for c in bundle['components']}!={42,1337,2026}:raise ValueError('Expected three trained seed components')
                if any(c['config']!=bundle['components'][0]['config'] for c in bundle['components']):raise ValueError('Seed architecture/config mismatch')
                for component in bundle['components']:
                    config=component['config']
                    if config['hidden']!=128 or not np.isfinite(config['dropout']) or not 0<=config['dropout']<1 or any(not torch.isfinite(v).all() for v in component['state_dict'].values()):raise ValueError('Invalid head')
                    Head(1539,128,config['dropout']).load_state_dict(component['state_dict'],strict=True)
            else:raise ValueError('Unknown regressor')
        return variant
    except (KeyError,TypeError,AttributeError,RuntimeError) as error:raise ValueError('Malformed checkpoint') from error


def predict_features(bundle,features):
    validate_bundle(bundle);x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=1539 or not len(x) or not np.isfinite(x).all():raise ValueError('Invalid real pair features')
    if bundle['kind']=='baseline':
        name=bundle['name'];p={'cosine':(x[:,-3]+1)/2,'word_jaccard':x[:,-2],'character_jaccard':x[:,-1],'train_mean':np.full(len(x),bundle['train_mean'])}[name]
    elif bundle['kind']=='score_prior':
        if bundle['mode']=='convex':
            alpha=bundle['alpha']
            p=x[:,-1] if alpha==0 else (1-alpha)*x[:,-1]+alpha*predict_features(bundle['component'],x)
        else:
            scores=x[:,-3:].copy();scores[:,0]=(scores[:,0]+1)/2
            p=scale(scores,bundle['mean'],bundle['std'])@bundle['coefficient'].numpy()+bundle['intercept']
    else:
        normalized=scale(x,bundle['mean'],bundle['std'])
        if bundle['kind']=='ridge':p=normalized@bundle['coefficient'].numpy()+bundle['intercept']
        else:
            probabilities=[]
            for c in bundle['components']:
                model=Head(1539,128,c['config']['dropout']);model.load_state_dict(c['state_dict']);model.eval()
                with torch.inference_mode():probabilities.append(model(torch.from_numpy(normalized)).numpy())
            p=np.mean(probabilities,axis=0)
    if not np.isfinite(p).all():raise ValueError('Nonfinite output')
    return np.clip(p,0,1).astype(np.float32)


def load_split(cache,processed,split):
    cache,processed=Path(cache),Path(processed);receipt=json.loads((cache/'receipt.json').read_text(encoding='utf-8'))
    variant=next((name for name,spec in VARIANTS.items() if (receipt['model'],receipt['revision'])==(spec['model'],spec['revision'])),None)
    if variant is None or receipt['weight_sha256']!={'model.safetensors':VARIANTS[variant]['sha256']} or receipt.get('prefix',VARIANTS[variant]['prefix'])!=VARIANTS[variant]['prefix']:raise ValueError('Pinned cache encoder mismatch')
    source=processed/(split+'.jsonl');path=cache/(split+'.npz');contract=receipt['splits'][split]
    if sha(source)!=contract['source_sha256'] or sha(path)!=contract['cache_sha256']:raise ValueError('Source/cache changed')
    rows=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()]
    with np.load(path,allow_pickle=False) as data:
        embeddings=data['embeddings'];ids=data['ids'];languages=data['languages'];targets=data['targets'] if split!='test' else None
        if split=='test' and ('targets' in data.files or any('target' in r for r in rows)):raise ValueError('Test gold entered input cache')
    if embeddings.shape!=(len(rows),2,768) or not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings,axis=2),1,atol=1e-4):raise ValueError('Invalid embeddings')
    if not np.array_equal(ids,[r['id'] for r in rows]) or not np.array_equal(languages,[r['language'] for r in rows]):raise ValueError('Feature row identity mismatch')
    if split!='test' and not np.array_equal(targets,np.asarray([r['target'] for r in rows],np.float32)):raise ValueError('Training/development target alignment changed')
    return pair_features(embeddings,rows),targets,languages,rows,receipt


def predict(checkpoint,input,output):
    output=Path(output)
    if output.exists():raise ValueError('Fresh prediction output required')
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=True);variant=validate_bundle(bundle)
    rows=[json.loads(s) for s in Path(input).read_text(encoding='utf-8').splitlines()]
    if not rows or any(not isinstance(r.get(k),str) or not r[k].strip() for r in rows for k in ('sentence1','sentence2')):raise ValueError('Nonempty sentence pairs required')
    if (bundle['kind']=='baseline' and bundle['name']!='cosine') or (bundle['kind']=='score_prior' and bundle['mode']=='convex' and bundle['alpha']==0):
        features=np.zeros((len(rows),1539),np.float32);features[:,-2:]=lexical(rows);diagnostics={'encoded_sentences':0}
    else:
        encoder=Encoder(variant,bundle.get('max_tokens',128));unique=list(dict.fromkeys(r[k] for r in rows for k in ('sentence1','sentence2')))
        values,diagnostics=encoder.encode(unique);lookup=dict(zip(unique,values));features=pair_features(np.stack([[lookup[r['sentence1']],lookup[r['sentence2']]] for r in rows]),rows)
    scores=predict_features(bundle,features)
    result={'predictions':scores.tolist(),'rows':len(rows),'relatedness_not_probability':True,'encoder':bundle['encoder'],'revision':bundle['revision'],'diagnostics':diagnostics,'checkpoint_sha256':sha(checkpoint)}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','input','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(predict(**vars(p.parse_args())),indent=2))

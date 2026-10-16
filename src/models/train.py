"""Frozen-encoder symmetric regressors with validation-only selection."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.linear_model import Ridge
from src.config import LANGUAGES,MODEL_ID,MODEL_REVISION
from src.data.make_dataset import sha
from src.features.build_features import pair_features,fit_scaler,scale
from src.models.metrics import evaluate,promotable

CONFIGS=[{'name':'mlp128','hidden':128,'dropout':.2,'lr':.001,'weight_decay':.01},
         {'name':'mlp128-slower','hidden':128,'dropout':.2,'lr':.0003,'weight_decay':.01},
         {'name':'mlp128-regularized','hidden':128,'dropout':.5,'lr':.0003,'weight_decay':.1}]
SEEDS=[42,1337,2026]


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def load_split(cache,processed,split):
    cache,processed=Path(cache),Path(processed)
    receipt=json.loads((cache/'receipt.json').read_text(encoding='utf-8'))
    if receipt['model']!=MODEL_ID or receipt['revision']!=MODEL_REVISION or not receipt['weight_sha256']:
        raise ValueError('Encoder source contract mismatch')
    source=processed/(split+'.jsonl');path=cache/(split+'.npz');contract=receipt['splits'][split]
    if sha(source)!=contract['source_sha256'] or sha(path)!=contract['cache_sha256']:
        raise ValueError('Cache or processed-source checksum mismatch')
    rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    with np.load(path,allow_pickle=False) as data:
        embeddings=data['embeddings'];ids=data['ids'];languages=data['languages']
        targets=data['targets'] if split!='test' else None
    if embeddings.shape!=(len(rows),2,768) or not np.isfinite(embeddings).all():raise ValueError('Invalid pair embeddings')
    if not np.array_equal(ids,[row['id'] for row in rows]) or not np.array_equal(languages,[row['language'] for row in rows]):
        raise ValueError('Cache row identity/order mismatch')
    norms=np.linalg.norm(embeddings.astype(np.float64),axis=2)
    if not np.allclose(norms,1,atol=1e-4):raise ValueError('Expected unit-normalized encoder output')
    if split!='test' and (targets.shape!=(len(rows),) or not np.array_equal(targets,np.asarray([row['target'] for row in rows],np.float32))):
        raise ValueError('Cache training/development target mismatch')
    if split=='test' and any('target' in row for row in rows):raise ValueError('Test input contains gold')
    return pair_features(embeddings,rows),targets,languages,rows,receipt


class Head(torch.nn.Module):
    def __init__(self,width,hidden=128,dropout=.2):
        super().__init__()
        self.net=torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.ReLU(),torch.nn.Dropout(dropout),torch.nn.Linear(hidden,1),torch.nn.Sigmoid())
    def forward(self,x):return self.net(x).squeeze(-1)


def balanced_weights(languages):
    languages=np.asarray(languages)
    if set(languages)!=set(LANGUAGES):raise ValueError('Training needs all supported languages')
    return np.asarray([len(languages)/(len(LANGUAGES)*np.sum(languages==lang)) for lang in languages],np.float32)


def ranking_loss(logits,targets,languages):
    """Human-score order within each language; no manufactured sentence pairs."""
    values=[]
    for language in LANGUAGES:
        mask=torch.as_tensor(np.asarray(languages)==language,device=logits.device)
        a,b=logits[mask],targets[mask]
        difference=b[:,None]-b[None,:]
        eligible=torch.triu(torch.ones_like(difference,dtype=torch.bool),diagonal=1)&(difference.abs()>=.1)
        if eligible.any():
            logits_difference=a[:,None]-a[None,:]
            values.append(torch.nn.functional.binary_cross_entropy_with_logits(logits_difference[eligible],(difference[eligible]>0).to(logits.dtype)))
    return torch.stack(values).mean() if values else logits.sum()*0


def fit_mlp(x,y,languages,vx,vy,validation_languages,config,seed,epochs=50):
    if epochs<1:raise ValueError('Positive epoch count required')
    torch.manual_seed(seed);torch.set_num_threads(2)
    model=Head(x.shape[1],config['hidden'],config['dropout'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    x,y,vx=torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(vx)
    weights=torch.from_numpy(balanced_weights(languages));generator=torch.Generator().manual_seed(seed)
    history=[];best=-float('inf');best_epoch=0;state=None
    for epoch in range(epochs):
        model.train();total=0.;mse_total=0.;ranking_total=0.
        for ids in torch.randperm(len(x),generator=generator).split(64):
            ranking_weight=config.get('ranking_weight',0.)
            if ranking_weight:
                logits=model.net[:-1](x[ids]).squeeze(-1)
                prediction=torch.sigmoid(logits)
                ranking=ranking_loss(logits,y[ids],np.asarray(languages)[ids.numpy()])
            else:
                prediction=model(x[ids]);ranking=prediction.sum()*0
            regression=((prediction-y[ids]).square()*weights[ids]).mean()
            loss=regression+ranking_weight*ranking
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            total+=float(loss.detach())*len(ids)
            mse_total+=float(regression.detach())*len(ids);ranking_total+=float(ranking.detach())*len(ids)
        model.eval()
        with torch.inference_mode():prediction=model(vx).numpy()
        metrics=evaluate(vy,prediction,validation_languages);value=metrics['macro_spearman']
        history.append({'epoch':epoch+1,'weighted_mse':mse_total/len(x),'ranking_loss':ranking_total/len(x),
                        'training_objective':total/len(x),'validation':metrics})
        if value is not None and value>best:
            best=value;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
        if epoch+1>=3 and epoch+1-best_epoch>=6:break
    if state is None:raise ValueError('No eligible nonconstant validation checkpoint')
    model.load_state_dict(state);model.eval()
    with torch.inference_mode():prediction=model(vx).numpy()
    return {'state_dict':state,'config':config,'seed':seed},prediction,{'seed':seed,'best_epoch':best_epoch,
        'history':history,'stop':'validation_plateau' if epoch+1>=3 and epoch+1-best_epoch>=6 else 'epoch_limit'}


def predict_features(bundle,features):
    validate_bundle(bundle)
    x=np.asarray(features,np.float32)
    if x.ndim!=2 or x.shape[1]!=bundle['width'] or not np.isfinite(x).all():raise ValueError('Invalid inference features')
    kind=bundle['kind']
    if kind=='baseline':
        name=bundle['name']
        prediction={'cosine':(x[:,-3]+1)/2,'word_jaccard':x[:,-2],'character_jaccard':x[:,-1],
                    'train_mean':np.full(len(x),bundle['train_mean'])}[name]
    else:
        normalized=scale(x,bundle['mean'],bundle['std'])
        if kind=='ridge':prediction=normalized@bundle['coefficient'].numpy()+float(bundle['intercept'])
        else:
            predictions=[]
            for component in bundle['components']:
                model=Head(bundle['width'],component['config']['hidden'],component['config']['dropout'])
                model.load_state_dict(component['state_dict'],strict=True);model.eval()
                with torch.inference_mode():predictions.append(model(torch.from_numpy(normalized)).numpy())
            prediction=np.mean(predictions,axis=0)
    return np.clip(prediction,0,1).astype(np.float32)


def validate_bundle(bundle):
    try:
        if bundle['schema']!='semrel_symmetric_v1' or bundle['encoder']!=MODEL_ID or bundle['revision']!=MODEL_REVISION or bundle['width']!=1539:
            raise ValueError('Unsupported checkpoint/source schema')
        if not bundle['weight_sha256'] or any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in bundle['weight_sha256'].values()):raise ValueError('Missing encoder weight attestation')
        if bundle['kind']=='baseline':
            if bundle['name'] not in ('cosine','word_jaccard','character_jaccard','train_mean') or not 0<=bundle['train_mean']<=1:raise ValueError('Invalid baseline')
        else:
            scale(np.zeros((1,1539),np.float32),bundle['mean'],bundle['std'])
            if bundle['kind']=='ridge':
                if bundle['coefficient'].shape!=(1539,) or not torch.isfinite(bundle['coefficient']).all() or not np.isfinite(bundle['intercept']):raise ValueError('Invalid ridge')
            elif bundle['kind']=='mlp':
                if not bundle['components']:raise ValueError('Empty ensemble')
                for c in bundle['components']:
                    if c['config']['hidden']!=128 or not 0<=c['config']['dropout']<1:raise ValueError('Invalid head architecture')
                    if not all(torch.isfinite(tensor).all() for tensor in c['state_dict'].values()):raise ValueError('Nonfinite state')
                    Head(1539,c['config']['hidden'],c['config']['dropout']).load_state_dict(c['state_dict'],strict=True)
            else:raise ValueError('Unknown model kind')
    except (KeyError,TypeError,RuntimeError,AttributeError) as error:raise ValueError('Malformed checkpoint') from error


def train(cache,processed,output):
    output=Path(output);started=time.time()
    if output.exists():raise ValueError('Use a fresh training output')
    torch.set_num_threads(2)
    x,y,languages,_,receipt=load_split(cache,processed,'train')
    vx,vy,vl,_,_=load_split(cache,processed,'dev')
    output.mkdir(parents=True)
    common={'schema':'semrel_symmetric_v1','encoder':MODEL_ID,'revision':MODEL_REVISION,
            'width':x.shape[1],'weight_sha256':receipt['weight_sha256']}
    records=[];baseline=None;baseline_score=-float('inf')
    for name in ('cosine','word_jaccard','character_jaccard','train_mean'):
        bundle={**common,'kind':'baseline','name':name,'train_mean':float(y.mean())}
        prediction=predict_features(bundle,vx);metrics=evaluate(vy,prediction,vl)
        records.append({'name':name,'family':'baseline','validation':metrics})
        if metrics['macro_spearman'] is not None and metrics['macro_spearman']>baseline_score:
            baseline=(bundle,prediction,metrics);baseline_score=metrics['macro_spearman']
    if baseline is None:raise ValueError('No eligible baseline')
    mean,std=fit_scaler(x);normalized=scale(x,mean,std);validation=scale(vx,mean,std)
    # Persist torch tensors to permit weights_only checkpoint loading.
    normalized_common={**common,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std)}
    winner=baseline;winner_score=baseline_score;selected_name=baseline[0]['name']
    for alpha in (1,10,100):
        model=Ridge(alpha=alpha).fit(normalized,y,sample_weight=balanced_weights(languages))
        bundle={**normalized_common,'kind':'ridge','alpha':alpha,'coefficient':torch.from_numpy(model.coef_.astype(np.float32)), 'intercept':float(model.intercept_)}
        prediction=predict_features(bundle,vx);metrics=evaluate(vy,prediction,vl)
        records.append({'name':f'ridge-{alpha}','family':'ridge','validation':metrics,'promotable':promotable(metrics,baseline[2])})
        torch.save(bundle,output/f'ridge-{alpha}.pt')
        if promotable(metrics,baseline[2]) and metrics['macro_spearman']>winner_score:
            winner=(bundle,prediction,metrics);winner_score=metrics['macro_spearman'];selected_name=f'ridge-{alpha}'
    for config in CONFIGS:
        components=[];predictions=[];fits=[]
        for seed in SEEDS:
            c,p,history=fit_mlp(normalized,y,languages,validation,vy,vl,config,seed)
            components.append(c);predictions.append(p);fits.append(history)
            torch.save(c,output/f"{config['name']}-seed-{seed}.pt")
            write_json(output/f"{config['name']}-seed-{seed}.json",history)
            print(json.dumps({'fit':config['name'],'seed':seed,'best_epoch':history['best_epoch']}),flush=True)
        bundle={**normalized_common,'kind':'mlp','components':components}
        prediction=predict_features(bundle,vx);metrics=evaluate(vy,prediction,vl)
        individual=[max(h['validation']['macro_spearman'] for h in fit['history'] if h['validation']['macro_spearman'] is not None) for fit in fits]
        records.append({'name':config['name'],'family':'mlp','validation':metrics,'promotable':promotable(metrics,baseline[2]),
                        'seed_mean_spearman':float(np.mean(individual)),'seed_sample_std_spearman':float(np.std(individual,ddof=1)),'fits':fits})
        if promotable(metrics,baseline[2]) and metrics['macro_spearman']>winner_score:
            winner=(bundle,prediction,metrics);winner_score=metrics['macro_spearman'];selected_name=config['name']
    bundle,prediction,metrics=winner
    validate_bundle(bundle);torch.save(bundle,output/'selected.pt');torch.save(baseline[0],output/'baseline.pt')
    restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    if not np.array_equal(predict_features(restored,vx),prediction):raise RuntimeError('Saved prediction replay mismatch')
    selection={'selected':selected_name,'validation':metrics,'baseline':records[next(i for i,r in enumerate(records) if r['name']==baseline[0]['name'])],
               'candidates':records,'checkpoint_sha256':sha(output/'selected.pt'),'baseline_sha256':sha(output/'baseline.pt'),
               'encoder_receipt_sha256':sha(Path(cache)/'receipt.json'),'test_gold_loaded':False,
               'elapsed_seconds':time.time()-started,'checkpoint_validation_replay_max_difference':0.}
    write_json(output/'selection.json',selection)
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('cache','processed','output'):parser.add_argument('--'+arg,required=True)
    train(**vars(parser.parse_args()))

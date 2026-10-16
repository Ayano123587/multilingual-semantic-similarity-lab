"""Preregistered full compatible cohort training; development selection only."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr,pearsonr
from sklearn.linear_model import Ridge
from src.config import MODEL_ID,MODEL_REVISION
from src.data.full_cohort import LANGUAGES,TRAIN_LANGUAGES
from src.data.make_dataset import sha
from src.features.build_features import fit_scaler,scale
from src.models.train import Head,write_json
from src.models.full_inference import load_split,predict_features,validate_bundle


def evaluate(y,p,languages,cohort=LANGUAGES):
    y,p,languages=np.asarray(y),np.asarray(p),np.asarray(languages)
    if y.ndim!=1 or y.shape!=p.shape or y.shape!=languages.shape or set(languages)!=set(cohort):raise ValueError('All requested language cohorts required')
    if not np.isfinite(y).all() or not np.isfinite(p).all() or ((y<0)|(y>1)|(p<0)|(p>1)).any():raise ValueError('Invalid finite bounded scores')
    rows={}
    for lang in cohort:
        a,b=y[languages==lang],p[languages==lang];valid=len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0
        rows[lang]={'rows':len(a),'spearman':float(spearmanr(a,b).statistic) if valid else None,'pearson':float(pearsonr(a,b).statistic) if valid else None,'mae':float(abs(a-b).mean())}
    return {'languages':rows,'macro_spearman':float(np.mean([r['spearman'] for r in rows.values()])) if all(r['spearman'] is not None for r in rows.values()) else None,'macro_mae':float(np.mean([r['mae'] for r in rows.values()]))}


def balanced_weights(languages):
    languages=np.asarray(languages)
    if set(languages)!=set(TRAIN_LANGUAGES):raise ValueError('All five official training languages required')
    return np.asarray([len(languages)/(len(TRAIN_LANGUAGES)*(languages==lang).sum()) for lang in languages],np.float32)


def ranking_loss(logits,target,languages):
    terms=[]
    for lang in TRAIN_LANGUAGES:
        keep=torch.as_tensor(np.asarray(languages)==lang);a,b=logits[keep],target[keep]
        gap=b[:,None]-b[None,:];eligible=torch.triu(torch.ones_like(gap,dtype=torch.bool),diagonal=1)&(gap.abs()>=.1)
        if eligible.any():terms.append(torch.nn.functional.binary_cross_entropy_with_logits((a[:,None]-a[None,:])[eligible],(gap[eligible]>0).to(a.dtype)))
    return torch.stack(terms).mean() if terms else logits.sum()*0


def eligible(value,baseline):
    return value['macro_spearman'] is not None and value['macro_spearman']>=baseline['macro_spearman']+.005 and all(value['languages'][l]['spearman'] is not None and value['languages'][l]['spearman']>=baseline['languages'][l]['spearman']-.02 for l in baseline['languages'])


def fit(x,y,languages,vx,vy,vl,config,seed,maximum=50,cohort=LANGUAGES):
    torch.manual_seed(seed);torch.set_num_threads(2);model=Head(x.shape[1],128,config['dropout'])
    opt=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);rng=torch.Generator().manual_seed(seed)
    tx,ty,tv=map(torch.from_numpy,(x,y,vx));w=torch.from_numpy(balanced_weights(languages));history=[];best=-float('inf');best_epoch=0;state=None
    for epoch in range(1,maximum+1):
        model.train();loss_sum=0.;reg_sum=0.;rank_sum=0.
        for ix in torch.randperm(len(x),generator=rng).split(64):
            logits=model.net[:-1](tx[ix]).squeeze(-1);prediction=torch.sigmoid(logits)
            regression=((prediction-ty[ix]).square()*w[ix]).mean()
            ranking=ranking_loss(logits,ty[ix],languages[ix.numpy()]) if config['ranking_weight'] else logits.sum()*0
            loss=regression+config['ranking_weight']*ranking
            if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step()
            loss_sum+=float(loss.detach())*len(ix);reg_sum+=float(regression.detach())*len(ix);rank_sum+=float(ranking.detach())*len(ix)
        model.eval()
        with torch.inference_mode():prediction=model(tv).numpy()
        metrics=evaluate(vy,prediction,vl,cohort);value=metrics['macro_spearman']
        history.append({'epoch':epoch,'training_objective':loss_sum/len(x),'weighted_mse':reg_sum/len(x),'ranking_loss':rank_sum/len(x),'validation':metrics})
        if value is not None and value>best:best=value;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
        if epoch>=3 and epoch-best_epoch>=6:break
    if state is None:raise ValueError('No nonconstant development checkpoint')
    return {'state_dict':state,'config':config,'seed':seed},{'seed':seed,'best_epoch':best_epoch,'history':history}


def train(cache,processed,output,protocol='docs/full-cohort-v3-protocol.json'):
    started=time.monotonic();torch.set_num_threads(2);output=Path(output)
    if output.exists():raise ValueError('Fresh study output required')
    protocol=Path(protocol);plan=json.loads(protocol.read_text(encoding='utf-8'))
    for split,expected in plan['processed_sha256'].items():
        if sha(Path(processed)/(split+'.jsonl'))!=expected:raise ValueError('Frozen cohort source changed')
    x,y,lang,_,receipt=load_split(cache,processed,'train');vx,vy,vl,_,_=load_split(cache,processed,'dev')
    if (len(x),len(vx))!=(plan['prepared_counts']['train'],plan['prepared_counts']['dev']):raise ValueError('Full cohort counts changed')
    cohort=tuple(plan['languages'])
    output.mkdir(parents=True);common={'schema':'semrel_symmetric_v1','encoder':receipt['model'],'revision':receipt['revision'],'width':1539,'weight_sha256':receipt['weight_sha256'],'cohort_languages':list(cohort),'prefix':receipt.get('prefix',''),'max_tokens':receipt['max_tokens']}
    records=[];baseline=None
    for name in ['cosine','word_jaccard','character_jaccard','train_mean']:
        b={**common,'kind':'baseline','name':name,'train_mean':float(y.mean())};metrics=evaluate(vy,predict_features(b,vx),vl,cohort)
        torch.save(b,output/f'baseline-{name}.pt')
        records.append({'name':name,'family':'baseline','validation':metrics,'checkpoint_file':f'baseline-{name}.pt'})
        if name=='character_jaccard':baseline=(b,metrics)
    mean,std=fit_scaler(x);nx,nv=scale(x,mean,std),scale(vx,mean,std);norm={**common,'mean':torch.from_numpy(mean),'std':torch.from_numpy(std)}
    winner,wm=baseline;winner_name='character_jaccard';best=wm['macro_spearman']
    def consider(bundle,metrics,name):
        nonlocal winner,wm,winner_name,best
        valid=eligible(metrics,baseline[1])
        if valid and metrics['macro_spearman']>best:winner,wm,winner_name,best=bundle,metrics,name,metrics['macro_spearman']
        return valid
    for alpha in plan['ridge_alphas']:
        model=Ridge(alpha=alpha).fit(nx,y,sample_weight=balanced_weights(lang));bundle={**norm,'kind':'ridge','alpha':alpha,'coefficient':torch.from_numpy(model.coef_.astype(np.float32)),'intercept':float(model.intercept_)}
        metrics=evaluate(vy,predict_features(bundle,vx),vl,cohort);name=f'ridge-{alpha}';valid=consider(bundle,metrics,name)
        records.append({'name':name,'family':'ridge','validation':metrics,'promotable':valid,'checkpoint_file':name+'.pt'});torch.save(bundle,output/(name+'.pt'))
    for config in plan['head_configs']:
        components=[];histories=[]
        for seed in plan['seeds']:
            c,h=fit(nx,y,lang,nv,vy,vl,config,seed,cohort=cohort);components.append(c);histories.append(h)
            torch.save(c,output/f"{config['name']}-{seed}.pt");print(json.dumps({'fit':config['name'],'seed':seed,'best_epoch':h['best_epoch']}),flush=True)
        bundle={**norm,'kind':'mlp','components':components};metrics=evaluate(vy,predict_features(bundle,vx),vl,cohort);valid=consider(bundle,metrics,config['name'])
        torch.save(bundle,output/(config['name']+'-ensemble.pt'))
        records.append({'name':config['name'],'family':'mlp','validation':metrics,'fits':histories,'promotable':valid,'checkpoint_file':config['name']+'-ensemble.pt'})
    torch.save(winner,output/'selected.pt');torch.save(baseline[0],output/'baseline.pt')
    replay=torch.load(output/'selected.pt',map_location='cpu',weights_only=True);delta=float(np.max(abs(predict_features(replay,vx)-predict_features(winner,vx))))
    if delta>1e-6:raise ValueError('Saved inference replay mismatch')
    result={'selected':winner_name,'validation':wm,'baseline':baseline[1],'candidates':records,'checkpoint_sha256':sha(output/'selected.pt'),'baseline_sha256':sha(output/'baseline.pt'),
        'protocol_sha256':sha(protocol),'encoder_receipt_sha256':sha(Path(cache)/'receipt.json'),'test_gold_loaded':False,'old_three_language_test_previously_observed':True,'elapsed_seconds':time.monotonic()-started,'checkpoint_validation_replay_max_difference':delta}
    write_json(output/'selection.json',result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','processed','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--protocol',default='docs/full-cohort-v3-protocol.json')
    print(json.dumps(train(**vars(parser.parse_args())),indent=2))

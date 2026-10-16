"""Prospectively fixed ranking-loss comparison; test gold remains sealed."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.data.make_dataset import sha
from src.features.build_features import fit_scaler,scale
from src.models.metrics import evaluate,promotable
from src.models.train import SEEDS,fit_mlp,load_split,predict_features,validate_bundle,write_json


def run(cache,processed,prior,output):
    started=time.time();prior,output=Path(prior),Path(output)
    torch.set_num_threads(2)
    if output.exists():raise ValueError('Use a fresh ranking study directory')
    selection=json.loads((prior/'selection.json').read_text(encoding='utf-8'))
    if sha(prior/'selected.pt')!=selection['checkpoint_sha256'] or sha(prior/'baseline.pt')!=selection['baseline_sha256']:
        raise ValueError('Prior selection artifacts changed')
    old=torch.load(prior/'selected.pt',map_location='cpu',weights_only=True)
    base=torch.load(prior/'baseline.pt',map_location='cpu',weights_only=True)
    x,y,languages,_,receipt=load_split(cache,processed,'train');vx,vy,vl,_,_=load_split(cache,processed,'dev')
    old_prediction=predict_features(old,vx);old_metrics=evaluate(vy,old_prediction,vl)
    if old_metrics!=selection['validation']:raise ValueError('Prior validation checkpoint replay discrepancy')
    mean,std=fit_scaler(x);normalized=scale(x,mean,std);validation=scale(vx,mean,std)
    output.mkdir(parents=True)
    common={k:old[k] for k in ('schema','encoder','revision','width','weight_sha256')}
    records=[];winner=old;best=old_metrics['macro_spearman'];selected_name=selection['selected']
    for weight in (.03,.1,.3):
        config={'name':f'ranking-{weight}','hidden':128,'dropout':.2,'lr':.001,'weight_decay':.01,'ranking_weight':weight}
        components=[];fits=[]
        for seed in SEEDS:
            component,_,history=fit_mlp(normalized,y,languages,validation,vy,vl,config,seed)
            components.append(component);fits.append(history)
            torch.save(component,output/f"ranking-{weight}-seed-{seed}.pt")
            write_json(output/f"ranking-{weight}-seed-{seed}.json",history)
            print(json.dumps({'ranking_weight':weight,'seed':seed,'best_epoch':history['best_epoch']}),flush=True)
        bundle={**common,'kind':'mlp','mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'components':components}
        prediction=predict_features(bundle,vx);metrics=evaluate(vy,prediction,vl)
        seed_values=[max(r['validation']['macro_spearman'] for r in f['history'] if r['validation']['macro_spearman'] is not None) for f in fits]
        eligible=promotable(metrics,old_metrics)
        records.append({'name':config['name'],'ranking_weight':weight,'validation':metrics,'promotable_vs_prior':eligible,
                        'seed_mean_spearman':float(np.mean(seed_values)),'seed_sample_std_spearman':float(np.std(seed_values,ddof=1)),'fits':fits})
        if eligible and metrics['macro_spearman']>best:
            winner=bundle;best=metrics['macro_spearman'];selected_name=config['name']
    validate_bundle(winner);torch.save(winner,output/'selected.pt');torch.save(base,output/'baseline.pt')
    prediction=predict_features(winner,vx);metrics=evaluate(vy,prediction,vl)
    restored=torch.load(output/'selected.pt',map_location='cpu',weights_only=True)
    if not np.array_equal(prediction,predict_features(restored,vx)):raise RuntimeError('Ranking checkpoint replay mismatch')
    report={'selected':selected_name,'validation':metrics,'prior_validation':old_metrics,'prior_selection_sha256':sha(prior/'selection.json'),
            'candidates':records,'improved':best>old_metrics['macro_spearman'],'test_gold_loaded':False,
            'checkpoint_sha256':sha(output/'selected.pt'),'baseline_sha256':sha(output/'baseline.pt'),
            'encoder_receipt_sha256':sha(Path(cache)/'receipt.json'),'elapsed_seconds':time.time()-started,
            'checkpoint_validation_replay_max_difference':0.}
    write_json(output/'selection.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','processed','prior','output'):parser.add_argument('--'+name,required=True)
    run(**vars(parser.parse_args()))

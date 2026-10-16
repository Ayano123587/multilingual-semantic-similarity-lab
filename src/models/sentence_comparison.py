"""Matched encoder comparison and global validation-only model selection."""
import argparse,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from src.models.full_training import train as train_heads,eligible
from src.models.full_inference import load_split,predict_features,validate_bundle
from src.models.train import write_json
from src.data.make_dataset import sha


def select_candidate(records):
    baselines=[r for r in records if r['family']=='baseline' and r['validation']['macro_spearman'] is not None]
    if not baselines:raise ValueError('No finite baseline comparison')
    baseline=max(baselines,key=lambda r:r['validation']['macro_spearman']);winner=baseline
    for row in records:
        if row['family']!='baseline' and eligible(row['validation'],baseline['validation']) and row['validation']['macro_spearman']>winner['validation']['macro_spearman']:winner=row
    return winner,baseline


def train(cache_afro,cache_e5,processed,output,protocol):
    started=time.monotonic();output=Path(output)
    if output.exists():raise ValueError('Fresh comparison output required')
    plan=json.loads(Path(protocol).read_text(encoding='utf-8'))
    if plan['languages']!=['afr','amh','arb','arq','ary','hau','ind','kin','pan']:raise ValueError('Complete nine-language cohort required')
    output.mkdir(parents=True);all_records=[];reports={};caches={'afro':Path(cache_afro),'e5':Path(cache_e5)}
    for variant,cache in caches.items():
        receipt=json.loads((cache/'receipt.json').read_text(encoding='utf-8'))
        if receipt.get('variant')!=variant or receipt['max_tokens']!=512:raise ValueError('Matched encoder context required')
        report=train_heads(cache,processed,output/variant,protocol);reports[variant]=report
        for row in report['candidates']:all_records.append({**row,'encoder_variant':variant,'checkpoint_path':str(output/variant/row['checkpoint_file'])})
    winner,baseline=select_candidate(all_records)
    for row in all_records:
        row['promotion_eligible_vs_strongest_baseline']=eligible(row['validation'],baseline['validation']) if row['family']!='baseline' else False
    shutil.copyfile(winner['checkpoint_path'],output/'selected.pt');shutil.copyfile(baseline['checkpoint_path'],output/'baseline.pt')
    selected=torch.load(output/'selected.pt',map_location='cpu',weights_only=True);validate_bundle(selected)
    vx,_,_,_,_=load_split(caches[winner['encoder_variant']],processed,'dev')
    original=torch.load(winner['checkpoint_path'],map_location='cpu',weights_only=True)
    delta=float(np.max(abs(predict_features(selected,vx)-predict_features(original,vx))))
    if delta>1e-6:raise ValueError('Global checkpoint replay mismatch')
    result={'selected':winner['encoder_variant']+'/'+winner['name'],'selected_validation':winner['validation'],
      'strongest_fixed_baseline':baseline['encoder_variant']+'/'+baseline['name'],'baseline_validation':baseline['validation'],
      'learned_head_promoted':winner['family']!='baseline','selected_candidate':winner,'candidates':all_records,
      'checkpoint_sha256':sha(output/'selected.pt'),'baseline_sha256':sha(output/'baseline.pt'),'protocol_sha256':sha(protocol),
      'encoder_receipt_sha256':{v:sha(c/'receipt.json') for v,c in caches.items()},'checkpoint_replay_max_difference':delta,
      'test_gold_loaded':False,'previously_observed_test_languages':['amh','hau','kin'],'elapsed_seconds':time.monotonic()-started}
    write_json(output/'selection.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache-afro','cache-e5','processed','output','protocol'):p.add_argument('--'+name,required=True)
    print(json.dumps(train(**vars(p.parse_args())),indent=2))

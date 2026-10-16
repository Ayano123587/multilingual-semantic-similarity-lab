"""Explicit one-time gold release after immutable validation selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.config import LANGUAGES,SOURCE_HASHES
from src.data.components import audit_components
from src.data.make_dataset import sha,read_pairs
from src.models.train import load_split,predict_features,write_json
from src.models.metrics import evaluate,cluster_bootstrap


def final_evaluate(cache,processed,selected,raw,output):
    selected,output,processed,raw=map(Path,(selected,output,processed,raw))
    if output.exists():raise ValueError('Final output exists; do not rerun or overwrite test evidence')
    selection_path=selected/'selection.json'
    selection_bytes=sha(selection_path)
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    if sha(selected/'selected.pt')!=selection['checkpoint_sha256'] or sha(selected/'baseline.pt')!=selection['baseline_sha256']:
        raise ValueError('Checkpoint changed after selection')
    bundle=torch.load(selected/'selected.pt',map_location='cpu',weights_only=True)
    baseline=torch.load(selected/'baseline.pt',map_location='cpu',weights_only=True)
    features,_,languages,rows,receipt=load_split(cache,processed,'test')
    if sha(Path(cache)/'receipt.json')!=selection['encoder_receipt_sha256']:
        raise ValueError('Encoder cache changed after selection')
    prediction=predict_features(bundle,features);reference=predict_features(baseline,features)
    release_path=selected/'test-release.json'
    try:
        with release_path.open('x',encoding='utf-8',newline='\n') as handle:
            json.dump({'selection_sha256':selection_bytes,'output':str(output),
                       'checkpoint_sha256':selection['checkpoint_sha256'],
                       'status':'Gold release started; preserve this marker even on failure'},handle,indent=2)
    except FileExistsError as error:
        raise ValueError('Test gold was already released for this selection; preserve evidence') from error
    # These are the first score-parsing calls for official human test gold.
    gold={}
    for lang in LANGUAGES:
        path=raw/(lang+'_test_with_labels.csv')
        if sha(path)!=SOURCE_HASHES[path.name]:raise ValueError('Test source checksum mismatch')
        for row in read_pairs(path,lang,labels=True):gold[(lang,row['id'])]=row
    targets=[]
    for row in rows:
        original=gold[(row['language'],row['id'])]
        if any(row[k]!=original[k] for k in ('sentence1','sentence2')):raise ValueError('Test text identity mismatch')
        targets.append(original['target'])
    splits={s:[json.loads(line) for line in (processed/(s+'.jsonl')).read_text(encoding='utf-8').splitlines()] for s in ('train','dev','test')}
    groups,component_report=audit_components(splits)
    target=np.asarray(targets)
    report={'selected':selection['selected'],'test':evaluate(target,prediction,languages),
            'baseline':evaluate(target,reference,languages),'uncertainty':cluster_bootstrap(target,prediction,reference,languages,groups['test']),
            'sentence_component_audit':component_report,'selection_sha256_before_gold':selection_bytes,
            'selection_unchanged':sha(selection_path)==selection_bytes,'checkpoint_sha256':selection['checkpoint_sha256'],
            'first_test_gold_release':'After serialized validation winner; no test-guided selection',
            'release_marker_sha256':sha(release_path),
            'limits':['Official split reuses individual sentences across partitions; no sentence-disjoint claim.',
                      'Upstream encoder pretraining exposure to source text is unknown.',
                      'Bootstrap conditional on available components, not a proof of deployment generalization.']}
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','processed','selected','raw','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--release-test-gold',action='store_true',required=True)
    args=vars(parser.parse_args());args.pop('release_test_gold')
    final_evaluate(**args)

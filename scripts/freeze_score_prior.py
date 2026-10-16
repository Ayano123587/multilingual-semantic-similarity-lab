"""Bind the fixed follow-up design to a user's fresh trained artifacts."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.make_dataset import sha


def freeze(prior, cache_afro, cache_e5, output, blueprint='docs/score-prior-protocol.json'):
    prior, output = Path(prior), Path(output)
    plan = json.loads(Path(blueprint).read_text(encoding='utf-8'))
    selection = json.loads((prior / 'selection.json').read_text(encoding='utf-8'))
    receipts = {variant: sha(Path(cache) / 'receipt.json')
                for variant, cache in (('afro', cache_afro), ('e5', cache_e5))}
    if receipts != selection['encoder_receipt_sha256'] or selection['test_gold_loaded'] is not False:
        raise ValueError('Fresh matched-encoder selection and exact cache receipts required')
    if sha(prior / 'baseline.pt') != selection['baseline_sha256']:
        raise ValueError('Fresh baseline weights changed')
    plan.update({'prior_selection_sha256': sha(prior / 'selection.json'),
                 'prior_protocol_sha256': selection['protocol_sha256'],
                 'component_checkpoint_sha256': sha(prior / 'e5/ranking0.1-ensemble.pt'),
                 'encoder_receipt_sha256': receipts,
                 'blueprint_sha256': sha(blueprint)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(plan, handle, indent=2)
        handle.write('\n')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prior', 'cache-afro', 'cache-e5', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--blueprint', default='docs/score-prior-protocol.json')
    freeze(**vars(parser.parse_args()))

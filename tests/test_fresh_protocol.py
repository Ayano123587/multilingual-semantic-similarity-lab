import json

import pytest

from scripts.freeze_score_prior import freeze
from src.data.make_dataset import sha
from src.models.train import write_json


def test_fresh_protocol_binds_new_artifacts_without_changing_the_grid(tmp_path):
    prior = tmp_path / 'heads'
    (prior / 'e5').mkdir(parents=True)
    (prior / 'baseline.pt').write_bytes(b'fixture-baseline')
    (prior / 'e5/ranking0.1-ensemble.pt').write_bytes(b'fixture-head')
    caches = {}
    for name in ('afro', 'e5'):
        caches[name] = tmp_path / name
        caches[name].mkdir()
        write_json(caches[name] / 'receipt.json', {'fixture_variant': name})
    write_json(prior / 'selection.json', {
        'encoder_receipt_sha256': {name: sha(path / 'receipt.json') for name, path in caches.items()},
        'baseline_sha256': sha(prior / 'baseline.pt'), 'test_gold_loaded': False,
        'protocol_sha256': 'fresh-training-protocol'})
    blueprint = tmp_path / 'blueprint.json'
    write_json(blueprint, {'global_blend_weights': [0, .25, 1], 'nonnegative_penalties': [0, .1]})
    output = tmp_path / 'fresh.json'
    plan = freeze(prior, caches['afro'], caches['e5'], output, blueprint)
    assert plan['global_blend_weights'] == [0, .25, 1]
    assert plan['prior_selection_sha256'] == sha(prior / 'selection.json')
    assert plan['component_checkpoint_sha256'] == sha(prior / 'e5/ranking0.1-ensemble.pt')
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        freeze(prior, caches['afro'], caches['e5'], output, blueprint)
    assert output.read_bytes() == original
    (prior / 'baseline.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='baseline weights changed'):
        freeze(prior, caches['afro'], caches['e5'], tmp_path / 'another.json', blueprint)

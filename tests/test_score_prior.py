import copy
import json

import numpy as np
import pytest
import torch

from src.models.encoder_variants import VARIANTS
from src.models.full_inference import predict, predict_features, validate_bundle
from src.models.score_prior import fit_nonnegative
from src.models.train import Head


def common():
    spec = VARIANTS['e5']
    return {'schema': 'semrel_symmetric_v1', 'width': 1539, 'encoder': spec['model'],
            'revision': spec['revision'], 'weight_sha256': {'model.safetensors': spec['sha256']},
            'prefix': spec['prefix'], 'max_tokens': 512}


def mixture(alpha=.25):
    head = Head(1539, 128, .2)
    for parameter in head.parameters():
        parameter.data.zero_()
    config = {'hidden': 128, 'dropout': .2}
    component = {**common(), 'kind': 'mlp', 'mean': torch.zeros(1539),
                 'std': torch.ones(1539),
                 'components': [{'seed': seed, 'config': config,
                                 'state_dict': copy.deepcopy(head.state_dict())}
                                for seed in (42, 1337, 2026)]}
    return {**common(), 'kind': 'score_prior', 'mode': 'convex',
            'alpha': alpha, 'component': component}


def test_exact_global_mixture_and_saved_replay(tmp_path):
    x = np.zeros((3, 1539), np.float32)
    x[:, -1] = [.1, .4, .8]
    bundle = mixture()
    expected = .75 * x[:, -1] + .25 * .5
    assert np.array_equal(predict_features(bundle, x), expected)
    path = tmp_path / 'model.pt'
    torch.save(bundle, path)
    assert np.array_equal(predict_features(torch.load(path, weights_only=True), x), expected)
    for value in (True, -0.1, float('nan'), 1.1):
        with pytest.raises(ValueError):
            validate_bundle({**bundle, 'alpha': value})
    invalid = copy.deepcopy(bundle)
    invalid['component']['prefix'] = 'passage: '
    with pytest.raises(ValueError):
        validate_bundle(invalid)


def test_monotone_fit_recovers_positive_oracle_and_rejects_bad_inputs(tmp_path):
    x = np.random.default_rng(42).uniform(0, 1, (60, 3)).astype(np.float32)
    target = .1 + x @ np.array([.15, .2, .3])
    trained, diagnostics = fit_nonnegative(x, target, np.ones(60), 0)
    bundle = {**common(), 'kind': 'score_prior', 'mode': 'nonnegative_linear', **trained}
    features = np.zeros((60, 1539), np.float32)
    features[:, -3:] = x
    features[:, -3] = 2 * x[:, 0] - 1
    assert np.max(abs(predict_features(bundle, features) - target)) < 1e-6
    checkpoint = tmp_path / 'monotone.pt'
    torch.save(bundle, checkpoint)
    assert np.array_equal(predict_features(bundle, features),
                          predict_features(torch.load(checkpoint, weights_only=True), features))
    assert diagnostics['objective'] < 1e-12
    assert (trained['coefficient'] >= 0).all()
    with pytest.raises(ValueError):
        fit_nonnegative(x, target, np.zeros(60), 0)
    invalid = copy.deepcopy(bundle)
    invalid['coefficient'][0] = -1
    with pytest.raises(ValueError):
        validate_bundle(invalid)


def test_zero_semantic_weight_skips_encoder_for_real_text_api(tmp_path, monkeypatch):
    import src.models.full_inference as module
    def forbidden(*args, **kwargs):
        raise AssertionError('A zero-weight semantic branch must not load its encoder')
    monkeypatch.setattr(module, 'Encoder', forbidden)
    checkpoint = tmp_path / 'selected.pt'
    torch.save(mixture(0.), checkpoint)
    pairs = tmp_path / 'pairs.jsonl'
    pairs.write_text(json.dumps({'sentence1': 'fixture one', 'sentence2': 'fixture two'}) + '\n', encoding='utf-8')
    result = predict(checkpoint, pairs, tmp_path / 'prediction.json')
    assert len(result['predictions']) == 1
    assert 0 <= result['predictions'][0] <= 1

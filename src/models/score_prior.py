"""Global lexical priors and trained low-dimensional multilingual transfer."""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

from src.data.make_dataset import sha
from src.features.build_features import fit_scaler, scale
from src.models.full_inference import load_split, predict_features, validate_bundle
from src.models.full_training import balanced_weights, eligible, evaluate
from src.models.train import write_json


def fit_nonnegative(scores, target, weights, penalty):
    """Language-balanced MSE plus coefficient L2; intercept is unpenalized."""
    scores, target, weights = map(lambda a: np.asarray(a, dtype=np.float64),
                                  (scores, target, weights))
    if (scores.ndim != 2 or scores.shape[1] != 3 or target.shape != (len(scores),)
            or weights.shape != target.shape or len(scores) < 4
            or not all(np.isfinite(a).all() for a in (scores, target, weights))
            or (weights <= 0).any() or ((target < 0) | (target > 1)).any()
            or not np.isfinite(penalty) or penalty < 0):
        raise ValueError('Finite aligned three-score training inputs required')
    mean, std = fit_scaler(scores.astype(np.float32))
    x = scale(scores.astype(np.float32), mean, std).astype(np.float64)
    root = np.sqrt(weights / weights.sum())
    design = np.column_stack((np.ones(len(x)), x)) * root[:, None]
    response = target * root
    if penalty:
        design = np.vstack((design, np.column_stack((np.zeros(3), np.eye(3) * np.sqrt(penalty)))))
        response = np.concatenate((response, np.zeros(3)))
    solution = lsq_linear(design, response, bounds=([-np.inf, 0, 0, 0], np.inf),
                          tol=1e-10, max_iter=1000)
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError('Nonnegative score training failed to converge')
    coefficient = solution.x[1:]
    residual = np.column_stack((np.ones(len(x)), x)) @ solution.x - target
    objective = float(np.average(residual ** 2, weights=weights)
                      + penalty * (coefficient ** 2).sum())
    return {'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
            'coefficient': torch.as_tensor(coefficient.astype(np.float32)),
            'intercept': float(solution.x[0])}, {
                'objective': objective, 'iterations': int(solution.nit),
                'optimality': float(solution.optimality),
                'coefficient': coefficient.tolist(), 'intercept': float(solution.x[0]),
                'penalty': penalty, 'training_rows': len(target),
                'solver': 'scipy.optimize.lsq_linear; nonnegative coefficients; unpenalized intercept'}


def train(cache_afro, cache_e5, processed, prior, protocol, output):
    started = time.monotonic()
    torch.set_num_threads(2)
    prior, protocol, output = map(Path, (prior, protocol, output))
    if output.exists():
        raise ValueError('Fresh score-prior output required')
    plan = json.loads(protocol.read_text(encoding='utf-8'))
    source = json.loads((prior / 'selection.json').read_text(encoding='utf-8'))
    if (sha(prior / 'selection.json') != plan['prior_selection_sha256']
            or source['protocol_sha256'] != plan['prior_protocol_sha256']
            or source['encoder_receipt_sha256'] != plan['encoder_receipt_sha256']):
        raise ValueError('Preregistered prior comparison changed')
    caches = {'afro': Path(cache_afro), 'e5': Path(cache_e5)}
    if {variant: sha(cache / 'receipt.json') for variant, cache in caches.items()} != plan['encoder_receipt_sha256']:
        raise ValueError('Frozen feature receipts changed')
    if {split: sha(Path(processed) / (split + '.jsonl')) for split in ('train', 'dev', 'test')} != plan['processed_sha256']:
        raise ValueError('Frozen complete cohort changed')
    component_path = prior / 'e5/ranking0.1-ensemble.pt'
    if sha(component_path) != plan['component_checkpoint_sha256']:
        raise ValueError('Fixed learned component changed')
    component = torch.load(component_path, weights_only=True, map_location='cpu')
    validate_bundle(component)
    baseline = torch.load(prior / 'baseline.pt', weights_only=True, map_location='cpu')
    if sha(prior / 'baseline.pt') != source['baseline_sha256']:
        raise ValueError('Strongest fixed baseline changed')
    tx, ty, tl, _, _ = load_split(caches['e5'], processed, 'train')
    vx, vy, vl, _, _ = load_split(caches['e5'], processed, 'dev')
    reference = evaluate(vy, predict_features(baseline, vx), vl)
    if (reference['macro_spearman'] != source['baseline_validation']['macro_spearman']
            or any(reference['languages'][lang]['spearman'] != source['baseline_validation']['languages'][lang]['spearman']
                   for lang in plan['languages'])):
        raise ValueError('Historical baseline validation did not reproduce exactly')
    common = {key: copy.deepcopy(component[key]) for key in
              ('schema', 'width', 'encoder', 'revision', 'weight_sha256', 'prefix', 'max_tokens')}
    output.mkdir(parents=True)
    records = []
    winner, winner_bundle, best = 'character_jaccard', baseline, reference

    def consider(bundle, name, family, training=None):
        nonlocal winner, winner_bundle, best
        validate_bundle(bundle)
        path = output / (name + '.pt')
        torch.save(bundle, path)
        saved = torch.load(path, weights_only=True, map_location='cpu')
        values = predict_features(saved, vx)
        metric = evaluate(vy, values, vl)
        promotable = eligible(metric, reference)
        records.append({'name': name, 'family': family, 'validation': metric,
                        'promotion_eligible': promotable, 'training': training,
                        'checkpoint_file': path.name, 'checkpoint_sha256': sha(path),
                        'saved_replay_max_difference': float(np.max(abs(values - predict_features(bundle, vx))))})
        if promotable and metric['macro_spearman'] > best['macro_spearman']:
            winner, winner_bundle, best = name, bundle, metric

    for alpha in plan['global_blend_weights']:
        consider({**common, 'kind': 'score_prior', 'mode': 'convex',
                  'alpha': float(alpha), 'component': component},
                 'convex-' + str(alpha), 'global_convex')
    scores = tx[:, -3:].copy()
    scores[:, 0] = (scores[:, 0] + 1) / 2
    weights = balanced_weights(tl)
    for penalty in plan['nonnegative_penalties']:
        trained, diagnostics = fit_nonnegative(scores, ty, weights, penalty)
        consider({**common, 'kind': 'score_prior', 'mode': 'nonnegative_linear', **trained},
                 'nonnegative-' + str(penalty), 'trained_global_linear', diagnostics)
    torch.save(winner_bundle, output / 'selected.pt')
    shutil.copyfile(prior / 'baseline.pt', output / 'baseline.pt')
    selected = torch.load(output / 'selected.pt', weights_only=True, map_location='cpu')
    replay = evaluate(vy, predict_features(selected, vx), vl)
    if replay != best:
        raise ValueError('Selected saved policy changed predictions')
    result = {'selected': ('e5/' + winner if winner != 'character_jaccard' else source['strongest_fixed_baseline']),
              'selected_validation': best, 'strongest_fixed_baseline': source['strongest_fixed_baseline'],
              'baseline_validation': reference, 'learned_head_promoted': winner != 'character_jaccard',
              'selected_candidate': {'name': winner, 'encoder_variant': validate_bundle(selected)},
              'candidates': records, 'checkpoint_sha256': sha(output / 'selected.pt'),
              'baseline_sha256': sha(output / 'baseline.pt'), 'protocol_sha256': sha(protocol),
              'encoder_receipt_sha256': plan['encoder_receipt_sha256'],
              'prior_selection_sha256': plan['prior_selection_sha256'],
              'test_gold_loaded': False, 'previously_observed_test_languages': ['amh', 'hau', 'kin'],
              'elapsed_seconds': time.monotonic() - started}
    write_json(output / 'selection.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache-afro', 'cache-e5', 'processed', 'prior', 'protocol', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(train(**vars(parser.parse_args())), indent=2))

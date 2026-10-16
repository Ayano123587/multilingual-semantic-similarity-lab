"""Freeze a nine-language policy, then explicitly release official gold once."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.components import audit_components
from src.data.full_cohort import LANGUAGES, read
from src.data.make_dataset import sha
from src.models.full_inference import load_split, predict_features, validate_bundle
from src.models.full_training import evaluate
from src.models.train import write_json


def bootstrap(y, prediction, baseline, languages, groups, resamples=1000, seed=2026):
    """Paired, language-stratified sentence-component resampling."""
    y, prediction, baseline, languages, groups = map(
        np.asarray, (y, prediction, baseline, languages, groups))
    evaluate(y, prediction, languages)
    evaluate(y, baseline, languages)
    if groups.shape != y.shape or type(resamples) is not int or resamples < 1:
        raise ValueError('Aligned components and positive integer resamples required')
    options = {lang: [np.flatnonzero((languages == lang) & (groups == group))
                     for group in np.unique(groups[languages == lang])]
               for lang in LANGUAGES}
    rng = np.random.default_rng(seed)
    samples, deltas = [], []
    per_language = {lang: [] for lang in LANGUAGES}
    for _ in range(resamples):
        indices = np.concatenate([rows[i] for lang in LANGUAGES
                                  for rows in [options[lang]]
                                  for i in rng.integers(len(rows), size=len(rows))])
        value = evaluate(y[indices], prediction[indices], languages[indices])
        reference = evaluate(y[indices], baseline[indices], languages[indices])
        for lang in LANGUAGES:
            score = value['languages'][lang]['spearman']
            if score is not None:
                per_language[lang].append(score)
        if value['macro_spearman'] is not None:
            samples.append(value['macro_spearman'])
            if reference['macro_spearman'] is not None:
                deltas.append(value['macro_spearman'] - reference['macro_spearman'])
    def interval(values):
        return np.quantile(values, [.025, .975]).tolist() if values else None
    return {'seed': seed, 'requested_resamples': resamples,
            'valid_macro_resamples': len(samples),
            'macro_spearman_95_percentile': interval(samples),
            'paired_macro_spearman_delta_95_percentile': interval(deltas),
            'per_language_spearman_95_percentile': {
                lang: interval(values) for lang, values in per_language.items()},
            'components_per_language': {lang: len(rows) for lang, rows in options.items()},
            'limitation': 'Conditional descriptive intervals; official sentence reuse, '
                          'upstream exposure, and repeated development selection remain.'}


def contract(selected, cache_afro, cache_e5, processed, protocol, coverage, inference_proof):
    """Verify immutable inputs without opening any numeric test target."""
    selected, processed, protocol, coverage = map(
        Path, (selected, processed, protocol, coverage))
    selection = json.loads((selected / 'selection.json').read_text(encoding='utf-8'))
    proof = json.loads(Path(inference_proof).read_text(encoding='utf-8'))
    if (proof['checkpoint_sha256'] != selection['checkpoint_sha256']
            or proof['rows'] != 18 or proof['languages'] != list(LANGUAGES)
            or proof['test_gold_loaded'] is not False
            or not np.isfinite([proof['actual_text_vs_cache_max_difference'],
                                proof['reversal_max_difference']]).all()
            or not 0 <= proof['actual_text_vs_cache_max_difference'] <= 1e-5
            or not 0 <= proof['reversal_max_difference'] <= 1e-7):
        raise ValueError('Actual selected nine-language text inference proof required')
    plan = json.loads(protocol.read_text(encoding='utf-8'))
    sources = json.loads(coverage.read_text(encoding='utf-8'))
    if (plan['languages'] != list(LANGUAGES)
            or sources['included_languages'] != list(LANGUAGES)
            or sha(coverage) != plan['coverage_sha256']
            or sha(protocol) != selection['protocol_sha256']):
        raise ValueError('Nine-language frozen source/protocol mismatch')
    processed_hashes = {split: sha(processed / (split + '.jsonl'))
                        for split in ('train', 'dev', 'test')}
    if processed_hashes != plan['processed_sha256']:
        raise ValueError('Prepared cohort changed after training')
    caches = {'afro': Path(cache_afro), 'e5': Path(cache_e5)}
    receipts = {key: sha(path / 'receipt.json') for key, path in caches.items()}
    if receipts != selection['encoder_receipt_sha256']:
        raise ValueError('Encoder cache receipt changed after selection')
    bundles, variants = {}, {}
    for name, field in (('selected', 'checkpoint_sha256'), ('baseline', 'baseline_sha256')):
        checkpoint = selected / (name + '.pt')
        if sha(checkpoint) != selection[field]:
            raise ValueError('Serialized policy changed after validation selection')
        bundles[name] = torch.load(checkpoint, map_location='cpu', weights_only=True)
        variants[name] = validate_bundle(bundles[name])
    # Verify every split/cache hash, alignment, and the absence of test labels.
    for cache in caches.values():
        for split in ('train', 'dev', 'test'):
            load_split(cache, processed, split)
    identity = {'schema': 'semrel_full_policy_v1',
                'selection_sha256': sha(selected / 'selection.json'),
                'checkpoint_sha256': selection['checkpoint_sha256'],
                'baseline_sha256': selection['baseline_sha256'],
                'protocol_sha256': sha(protocol), 'coverage_sha256': sha(coverage),
                'processed_sha256': processed_hashes, 'encoder_receipt_sha256': receipts,
                'selected': selection['selected'],
                'strongest_fixed_baseline': selection['strongest_fixed_baseline'],
                'languages': list(LANGUAGES),
                'inference_proof_sha256': sha(inference_proof),
                'previously_observed_test_languages': ['amh', 'hau', 'kin'],
                'evaluation_source_sha256': {
                    name: sha(Path(__file__).parent / name)
                    for name in ('full_evaluation.py', 'full_training.py', 'full_inference.py')},
                'component_source_sha256': sha(Path(__file__).parents[1] / 'data/components.py')}
    return identity, bundles, variants, caches, sources


def freeze(selected, cache_afro, cache_e5, processed, protocol, coverage, frozen, inference_proof):
    identity, *_ = contract(selected, cache_afro, cache_e5, processed, protocol, coverage, inference_proof)
    frozen = Path(frozen)
    frozen.parent.mkdir(parents=True, exist_ok=True)
    with frozen.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(identity, handle, indent=2)
        handle.write('\n')
    return identity


def release(selected, cache_afro, cache_e5, processed, protocol, coverage, frozen,
            inference_proof, raw, output, resamples=1000):
    torch.set_num_threads(2)
    processed, raw, output, frozen = map(Path, (processed, raw, output, frozen))
    if output.exists():
        raise ValueError('Fresh final output required; preserve existing evidence')
    identity, bundles, variants, caches, sources = contract(
        selected, cache_afro, cache_e5, processed, protocol, coverage, inference_proof)
    if json.loads(frozen.read_text(encoding='utf-8')) != identity:
        raise ValueError('Policy differs from the explicit pre-release freeze')
    predictions = {}
    for name in ('selected', 'baseline'):
        features, _, languages, rows, _ = load_split(caches[variants[name]], processed, 'test')
        predictions[name] = predict_features(bundles[name], features)
    # A source-directory marker guards repeat releases across new model folders.
    marker = processed / 'test-gold-release.json'
    with marker.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump({'frozen_policy_sha256': sha(frozen), 'output': str(output),
                   'status': 'Gold release started; preserve marker even on failure'}, handle, indent=2)
        handle.write('\n')
    gold = {}
    for lang in LANGUAGES:
        source = sources['files'][lang + '_test']
        path = raw / Path(source['path']).name
        if sha(path) != source['sha256']:
            raise ValueError('Official test bytes changed')
        originals = read(path, lang, labels=True)  # First numeric target parse.
        if len(originals) != source['rows']:
            raise ValueError('Official test row count changed')
        for row in originals:
            gold[(lang, row['id'])] = row
    targets = []
    for row in rows:
        original = gold[(row['language'], row['id'])]
        if any(row[key] != original[key] for key in ('sentence1', 'sentence2')):
            raise ValueError('Official test pair identity changed')
        targets.append(original['target'])
    splits = {split: [json.loads(line) for line in (processed / (split + '.jsonl'))
                     .read_text(encoding='utf-8').splitlines()]
              for split in ('train', 'dev', 'test')}
    groups, component_audit = audit_components(splits)
    target = np.asarray(targets)
    report = {'policy': identity, 'frozen_policy_sha256': sha(frozen),
              'test': evaluate(target, predictions['selected'], languages),
              'baseline': evaluate(target, predictions['baseline'], languages),
              'uncertainty': bootstrap(target, predictions['selected'], predictions['baseline'],
                                       languages, groups['test'], resamples=resamples),
              'sentence_component_audit': component_audit,
              'release_marker_sha256': sha(marker),
              'limits': ['The Amharic, Hausa, and Kinyarwanda test scores were previously '
                         'observed; the pooled nine-language score is not an untouched test.',
                         'Official partitions reuse sentences; upstream pretrained model '
                         'exposure to benchmark text is unknown.']}
    report['test_exposure_groups'] = {}
    for name, cohort in (
            ('previously_observed_three_languages', ('amh', 'hau', 'kin')),
            ('six_additional_languages', tuple(lang for lang in LANGUAGES
                                              if lang not in ('amh', 'hau', 'kin')))):
        keep = np.isin(languages, cohort)
        report['test_exposure_groups'][name] = {
            'languages': list(cohort),
            'selected': evaluate(target[keep], predictions['selected'][keep], languages[keep], cohort),
            'baseline': evaluate(target[keep], predictions['baseline'][keep], languages[keep], cohort)}
    if sha(Path(selected) / 'selection.json') != identity['selection_sha256']:
        raise ValueError('Selection changed during release')
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'release'))
    for name in ('selected', 'cache-afro', 'cache-e5', 'processed', 'protocol', 'coverage', 'frozen', 'inference-proof'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--raw')
    parser.add_argument('--output')
    parser.add_argument('--release-test-gold', action='store_true')
    args = vars(parser.parse_args())
    action = args.pop('action')
    consent = args.pop('release_test_gold')
    if action == 'release':
        if not consent or not args['raw'] or not args['output']:
            parser.error('Release requires --release-test-gold, --raw, and --output')
        result = release(**args)
    else:
        args.pop('raw'); args.pop('output')
        result = freeze(**args)
    print(json.dumps(result, indent=2))

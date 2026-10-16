"""Release contracts use small software fixtures, never benchmark test gold."""
import csv
import json

import numpy as np
import pytest
import torch

from src.data.full_cohort import LANGUAGES
from src.data.make_dataset import sha
from src.models.encoder_variants import VARIANTS
from src.models.full_evaluation import bootstrap, freeze, release
from src.models.train import write_json


@pytest.fixture
def release_inputs(tmp_path):
    processed, raw, selected = [tmp_path / name for name in ('processed', 'raw', 'selected')]
    for path in (processed, raw, selected):
        path.mkdir()
    files = {}
    for split in ('train', 'dev', 'test'):
        rows = []
        for lang in LANGUAGES:
            path = raw / (lang + '_' + split + '.csv')
            with path.open('w', encoding='utf-8', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['PairID', 'Text', 'Score'])
                for i in range(4):
                    row = {'id': split + str(i), 'language': lang,
                           'sentence1': lang + ' shared ' + str(i),
                           'sentence2': lang + ' other ' + str(i)}
                    writer.writerow([row['id'], row['sentence1'] + '\n' + row['sentence2'], i / 3])
                    if split != 'test':
                        row['target'] = i / 3
                    rows.append(row)
            files[lang + '_' + split] = {'path': str(path), 'sha256': sha(path), 'rows': 4}
        (processed / (split + '.jsonl')).write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8', newline='\n')
    caches = {}
    for variant, spec in VARIANTS.items():
        cache = tmp_path / variant
        cache.mkdir()
        receipt = {'model': spec['model'], 'revision': spec['revision'],
                   'weight_sha256': {'model.safetensors': spec['sha256']},
                   'prefix': spec['prefix'], 'splits': {}}
        for split in ('train', 'dev', 'test'):
            source = processed / (split + '.jsonl')
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            embeddings = np.zeros((len(rows), 2, 768), np.float32)
            embeddings[:, :, 0] = 1
            data = {'embeddings': embeddings, 'ids': np.array([r['id'] for r in rows]),
                    'languages': np.array([r['language'] for r in rows])}
            if split != 'test':
                data['targets'] = np.array([r['target'] for r in rows], np.float32)
            path = cache / (split + '.npz')
            np.savez(path, **data)
            receipt['splits'][split] = {'source_sha256': sha(source), 'cache_sha256': sha(path)}
        write_json(cache / 'receipt.json', receipt)
        caches[variant] = cache
    spec = VARIANTS['e5']
    bundle = {'schema': 'semrel_symmetric_v1', 'width': 1539,
              'encoder': spec['model'], 'revision': spec['revision'],
              'weight_sha256': {'model.safetensors': spec['sha256']},
              'prefix': spec['prefix'], 'max_tokens': 512,
              'kind': 'baseline', 'name': 'character_jaccard', 'train_mean': .5}
    for name in ('selected', 'baseline'):
        torch.save(bundle, selected / (name + '.pt'))
    coverage, protocol = tmp_path / 'coverage.json', tmp_path / 'protocol.json'
    write_json(coverage, {'included_languages': list(LANGUAGES), 'files': files})
    write_json(protocol, {'languages': list(LANGUAGES), 'coverage_sha256': sha(coverage),
                         'processed_sha256': {s: sha(processed / (s + '.jsonl'))
                                              for s in ('train', 'dev', 'test')}})
    write_json(selected / 'selection.json', {
        'checkpoint_sha256': sha(selected / 'selected.pt'),
        'baseline_sha256': sha(selected / 'baseline.pt'),
        'protocol_sha256': sha(protocol),
        'encoder_receipt_sha256': {v: sha(c / 'receipt.json') for v, c in caches.items()},
        'selected': 'e5/character_jaccard', 'strongest_fixed_baseline': 'e5/character_jaccard'})
    proof = tmp_path / 'inference-proof.json'
    write_json(proof, {'checkpoint_sha256': sha(selected / 'selected.pt'), 'rows': 18,
                       'languages': list(LANGUAGES), 'test_gold_loaded': False,
                       'actual_text_vs_cache_max_difference': 0., 'reversal_max_difference': 0.})
    return {'selected': selected, 'cache_afro': caches['afro'], 'cache_e5': caches['e5'],
            'processed': processed, 'protocol': protocol, 'coverage': coverage,
            'frozen': tmp_path / 'frozen.json', 'inference_proof': proof}, raw, tmp_path / 'result.json'


def test_freeze_release_marker_precedes_gold_and_blocks_repeats(release_inputs, monkeypatch):
    args, raw, output = release_inputs
    import src.models.full_evaluation as module
    original = module.read
    calls = []
    def guarded_read(path, language, labels):
        assert (args['processed'] / 'test-gold-release.json').exists()
        assert labels
        calls.append(language)
        return original(path, language, labels)
    monkeypatch.setattr(module, 'read', guarded_read)
    freeze(**args)
    assert calls == []
    report = release(**args, raw=raw, output=output, resamples=5)
    assert calls == list(LANGUAGES)
    assert sum(r['rows'] for r in report['test']['languages'].values()) == 36
    assert report['policy']['previously_observed_test_languages'] == ['amh', 'hau', 'kin']
    assert len(report['test_exposure_groups']['six_additional_languages']['languages']) == 6
    with pytest.raises(FileExistsError):
        release(**args, raw=raw, output=output.parent / 'another.json', resamples=5)
    assert calls == list(LANGUAGES)


def test_changed_freeze_and_cache_fail_before_gold(release_inputs):
    args, raw, output = release_inputs
    freeze(**args)
    policy = json.loads(args['frozen'].read_text())
    policy['selected'] = 'changed'
    write_json(args['frozen'], policy)
    with pytest.raises(ValueError, match='explicit pre-release freeze'):
        release(**args, raw=raw, output=output, resamples=5)
    assert not (args['processed'] / 'test-gold-release.json').exists()
    (args['cache_afro'] / 'test.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source/cache changed'):
        freeze(**{**args, 'frozen': output.parent / 'new-freeze.json'})


def test_failed_gold_read_preserves_source_release_marker(release_inputs):
    args, raw, output = release_inputs
    freeze(**args)
    next(raw.glob('*_test.csv')).write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Official test bytes changed'):
        release(**args, raw=raw, output=output, resamples=5)
    assert (args['processed'] / 'test-gold-release.json').exists()
    assert not output.exists()


def test_bad_actual_inference_proof_blocks_freeze(release_inputs):
    args, _, _ = release_inputs
    proof = json.loads(args['inference_proof'].read_text())
    proof['actual_text_vs_cache_max_difference'] = float('nan')
    # Deliberately bypass the production writer, which already rejects NaN.
    args['inference_proof'].write_text(json.dumps(proof), encoding='utf-8')
    with pytest.raises(ValueError, match='Actual selected nine-language text inference proof'):
        freeze(**args)
    assert not args['frozen'].exists()
    assert not (args['processed'] / 'test-gold-release.json').exists()


def test_nine_language_component_bootstrap_is_deterministic():
    langs = np.repeat(LANGUAGES, 4)
    y = np.tile([0., .2, .7, 1.], 9)
    p = np.tile([.1, .3, .6, .9], 9)
    groups = np.array([lang + str(i // 2) for lang in LANGUAGES for i in range(4)])
    a = bootstrap(y, p, p, langs, groups, resamples=8)
    assert a == bootstrap(y, p, p, langs, groups, resamples=8)
    assert a['paired_macro_spearman_delta_95_percentile'] == [0., 0.]
    assert set(a['components_per_language']) == set(LANGUAGES)
    with pytest.raises(ValueError):
        bootstrap(y, p, p, langs, groups[:-1], resamples=8)

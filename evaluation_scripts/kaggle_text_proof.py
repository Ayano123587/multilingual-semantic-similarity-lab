"""Private CPU saved-model text proof; no fitting or test-label access."""
import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path


WORK = Path('/kaggle/working')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    manifests = list(Path('/kaggle/input').rglob('text-proof-manifest.json'))
    if len(manifests) != 1:
        raise ValueError('Exactly one private source/checkpoint manifest required')
    manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    if manifest['test_gold_included'] is not False:
        raise ValueError('This proof must contain no test targets')
    source = manifests[0].parent
    root = Path('/kaggle/temp/semrel-proof-source')
    root.mkdir(parents=True, exist_ok=False)
    for item in manifest['files']:
        path = source / item['flat']
        if digest(path) != item['sha256']:
            raise ValueError('Private staged source/checkpoint changed')
        destination = root / item['path']
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError('Source path escapes private staging')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'transformers==4.57.6', 'sentencepiece==0.2.1',
                           'huggingface-hub==0.36.2', 'safetensors==0.8.0'])
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from src.models import full_inference as inference

    torch.set_num_threads(2)
    checkpoint = root / 'selected.pt'
    if digest(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Selected weights differ from validation winner')
    bundle = torch.load(checkpoint, weights_only=True, map_location='cpu')
    inference.validate_bundle(bundle)
    captured = {}
    original = inference.Encoder

    class RecordingEncoder(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured['loading'] = self.loading
            captured['model'] = self.spec['model']
            captured['revision'] = self.spec['revision']
            captured['weight_sha256'] = self.spec['sha256']

    inference.Encoder = RecordingEncoder
    rows = [json.loads(line) for line in (root / 'input.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows) != 18 or any(set(row) != {'sentence1', 'sentence2'} for row in rows):
        raise ValueError('Exactly nine development pairs plus their reversals required')
    actual = inference.predict(checkpoint, root / 'input.jsonl', WORK / 'actual-text-predictions.json')
    features = np.load(root / 'features.npy', allow_pickle=False)
    if features.shape != (9, 1539):
        raise ValueError('Nine matching development feature rows required')
    cached = inference.predict_features(bundle, features)
    values = np.asarray(actual['predictions'])
    delta = float(np.max(abs(cached - values[:9])))
    reverse = float(np.max(abs(values[:9] - values[9:])))
    if delta > 1e-5 or reverse > 1e-7 or not np.isfinite(values).all():
        raise ValueError('Actual selected text/cache/reversal proof failed')
    proof = {'rows': 18, 'languages': manifest['languages'],
             'checkpoint_sha256': digest(checkpoint),
             'actual_text_vs_cache_max_difference': delta,
             'reversal_max_difference': reverse, 'test_gold_loaded': False,
             'encoder': captured, 'tokenization': actual['diagnostics'],
             'source_commit': manifest['source_commit'],
             'source_manifest_sha256': digest(manifests[0]),
             'python': platform.python_version(),
             'versions': {name: importlib.metadata.version(name)
                          for name in ('torch', 'numpy', 'transformers', 'huggingface-hub', 'safetensors')},
             'threads': torch.get_num_threads(), 'seconds': time.monotonic() - started}
    (WORK / 'inference-proof.json').write_text(json.dumps(proof, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(proof, indent=2))
    gc.collect()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (WORK / 'failure.json').write_text(json.dumps({
            'type': type(error).__name__, 'message': str(error),
            'traceback': traceback.format_exc()}, indent=2) + '\n', encoding='utf-8', newline='\n')
        raise

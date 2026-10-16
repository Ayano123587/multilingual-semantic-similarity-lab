"""Guard readable notebook implementations against production drift."""
import ast
import json
from pathlib import Path


def test_visible_relatedness_core_matches_source_and_executed_outputs():
    root=Path(__file__).resolve().parents[1]
    main=json.loads((root/'notebooks/01_complete_relatedness_workflow.ipynb').read_text(encoding='utf-8'))
    visible={node.name:node for cell in main['cells'] if cell['cell_type']=='code'
             for node in ast.parse(''.join(cell['source'])).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    contracts={'data/make_dataset':['normalized','pair_key'],
               'data/full_cohort':['read','prepare'],
               'features/build_features':['jaccard','lexical','pair_features','fit_scaler','scale'],
               'models/encoder_variants':['Encoder','pool_and_normalize'],
               'models/full_training':['evaluate','balanced_weights','ranking_loss','eligible','fit'],
               'models/train':['Head'],
               'models/full_inference':['validate_bundle','predict_features']}
    for module,names in contracts.items():
        nodes={node.name:node for node in ast.parse((root/f'src/{module}.py').read_text(encoding='utf-8')).body
               if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
        for name in names:assert ast.dump(visible[name])==ast.dump(nodes[name]),name
    images=0
    for path in (root/'notebooks').glob('*.ipynb'):
        data=json.loads(path.read_text(encoding='utf-8'))
        for cell in data['cells']:
            if cell['cell_type']!='code':continue
            assert cell['execution_count'] is not None
            for output in cell.get('outputs',[]):
                assert output['output_type']!='error'
                images+=int('image/png' in output.get('data',{}))
    assert images>=9

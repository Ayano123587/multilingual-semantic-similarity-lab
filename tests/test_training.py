import copy
import numpy as np
import pytest
import torch
from src.config import MODEL_ID,MODEL_REVISION,LANGUAGES
from src.models.train import Head,fit_mlp,balanced_weights,predict_features,validate_bundle


def common():
    return {'schema':'semrel_symmetric_v1','encoder':MODEL_ID,'revision':MODEL_REVISION,'width':1539,
            'weight_sha256':{'model.safetensors':'a'*64}}


@pytest.mark.parametrize('ranking_weight',[0.,.1])
def test_language_balance_and_actual_gradients_checkpoint(tmp_path,ranking_weight):
    languages=np.array(['amh']*2+['hau']*4+['kin']*6)
    weights=balanced_weights(languages)
    assert np.allclose([weights[languages==lang].sum() for lang in LANGUAGES],4)
    rng=np.random.default_rng(9);x=rng.normal(size=(12,1539)).astype(np.float32)
    y=np.array([0.,1.,0.,.2,.5,1.,0.,.1,.2,.5,.7,1.],np.float32)
    config={'name':'test','hidden':128,'dropout':.2,'lr':.001,'weight_decay':.01,'ranking_weight':ranking_weight}
    c,p,history=fit_mlp(x,y,languages,x,y,languages,config,42,epochs=2)
    torch.manual_seed(42);initial=Head(1539,128,.2).state_dict()
    assert any(not torch.equal(c['state_dict'][k],initial[k]) for k in initial)
    assert history['best_epoch'] in (1,2) and np.isfinite(p).all()
    bundle={**common(),'kind':'mlp','mean':torch.zeros(1539),'std':torch.ones(1539),'components':[c]}
    path=tmp_path/'model.pt';torch.save(bundle,path)
    restored=torch.load(path,weights_only=True)
    assert np.array_equal(predict_features(restored,x),p)
    swapped=x.copy();swapped[:,-3:]=[[.6,.4,.2]]*12
    assert predict_features(restored,swapped).shape==(12,)


def test_inference_schema_rejects_bad_state_normalization_and_sources():
    bundle={**common(),'kind':'ridge','mean':torch.zeros(1539),'std':torch.ones(1539),
            'coefficient':torch.ones(1539),'intercept':0.}
    validate_bundle(bundle)
    for key,value in [('revision','wrong'),('std',torch.zeros(1539)),('coefficient',torch.full((1539,),float('nan'))),('width',99)]:
        changed=copy.deepcopy(bundle);changed[key]=value
        with pytest.raises(ValueError):validate_bundle(changed)
    with pytest.raises(ValueError):predict_features(bundle,np.zeros((2,12),np.float32))


def test_final_gold_release_does_not_open_missing_inputs_before_selection(tmp_path):
    from src.models.evaluate import final_evaluate
    output=tmp_path/'already.json';output.write_text('historical',encoding='utf-8')
    with pytest.raises(ValueError,match='do not rerun'):
        final_evaluate('missing','missing','missing','missing',output)
    selected=tmp_path/'selected';selected.mkdir()
    (selected/'selection.json').write_text('{"checkpoint_sha256":"incorrect","baseline_sha256":"incorrect"}',encoding='utf-8')
    (selected/'selected.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed after selection'):
        final_evaluate('missing','missing',selected,'missing',tmp_path/'new.json')


def test_release_marker_blocks_different_output_path_before_gold(tmp_path,monkeypatch):
    import json
    from src.models import evaluate as final_module
    from src.data.make_dataset import sha
    selected=tmp_path/'selected';selected.mkdir();cache=tmp_path/'cache';cache.mkdir()
    bundle={**common(),'kind':'baseline','name':'word_jaccard','train_mean':.5}
    for name in ('selected.pt','baseline.pt'):torch.save(bundle,selected/name)
    (cache/'receipt.json').write_text('{}',encoding='utf-8')
    selection={'checkpoint_sha256':sha(selected/'selected.pt'),'baseline_sha256':sha(selected/'baseline.pt'),
               'encoder_receipt_sha256':sha(cache/'receipt.json')}
    (selected/'selection.json').write_text(json.dumps(selection),encoding='utf-8')
    (selected/'test-release.json').write_text('prior release',encoding='utf-8')
    monkeypatch.setattr(final_module,'load_split',lambda *args:(np.zeros((6,1539),np.float32),None,np.repeat(LANGUAGES,2),[],{}))
    def forbidden(*args,**kwargs):raise AssertionError('Gold parser must not be called')
    monkeypatch.setattr(final_module,'read_pairs',forbidden)
    with pytest.raises(ValueError,match='already released'):
        final_module.final_evaluate(cache,'missing',selected,'missing',tmp_path/'new-output.json')


def test_lexical_winner_inference_does_not_allocate_unused_encoder(tmp_path,monkeypatch):
    import json
    from src.models import predict as prediction_module
    checkpoint=tmp_path/'selected.pt'
    torch.save({**common(),'kind':'baseline','name':'word_jaccard','train_mean':.5},checkpoint)
    inputs=tmp_path/'pairs.jsonl'
    inputs.write_text(json.dumps({'sentence1':'one two','sentence2':'two three'})+'\n',encoding='utf-8')
    def forbidden(*args,**kwargs):raise AssertionError('Lexical inference must not load encoder')
    monkeypatch.setattr(prediction_module,'Encoder',forbidden)
    result=prediction_module.predict(checkpoint,inputs,tmp_path/'prediction.json')
    assert result['predictions'][0]==pytest.approx(1/3)
    assert result['diagnostics']['encoded_sentences']==0


def test_ranking_loss_order_gradient_and_language_isolation():
    from src.models.train import ranking_loss
    targets=torch.tensor([0.,1.,0.,1.]);languages=np.array(['amh','amh','hau','hau'])
    correct=torch.tensor([-2.,2.,-2.,2.],requires_grad=True)
    good=ranking_loss(correct,targets,languages);bad=ranking_loss(-correct,targets,languages)
    assert good<bad
    good.backward();assert torch.isfinite(correct.grad).all() and correct.grad.abs().sum()>0
    no_pairs=torch.tensor([1.,-1.,2.],requires_grad=True)
    zero=ranking_loss(no_pairs,torch.tensor([0.,1.,.5]),np.array(LANGUAGES))
    assert zero==0;zero.backward();assert torch.equal(no_pairs.grad,torch.zeros(3))

import numpy as np
import pytest
import torch
from src.models.full_training import evaluate,balanced_weights,ranking_loss
from src.data.full_cohort import LANGUAGES,TRAIN_LANGUAGES

def test_nine_language_macro_is_equal_not_pooled():
    languages=np.repeat(LANGUAGES,3);y=np.tile([0.,.5,1.],len(LANGUAGES));p=y.copy();p[:3]=p[:3][::-1]
    result=evaluate(y,p,languages)
    assert result['macro_spearman']==pytest.approx((len(LANGUAGES)-2)/len(LANGUAGES))
    with pytest.raises(ValueError):evaluate(y[3:],p[3:],languages[3:])

def test_five_training_languages_balance_and_rank_gradients():
    languages=np.concatenate([np.repeat(lang,i+2) for i,lang in enumerate(TRAIN_LANGUAGES)])
    w=balanced_weights(languages)
    for lang in TRAIN_LANGUAGES:assert float(w[languages==lang].sum())==pytest.approx(len(languages)/5,abs=1e-6)
    logits=torch.linspace(-1,1,len(languages),requires_grad=True);target=torch.linspace(0,1,len(languages))
    loss=ranking_loss(logits,target,languages);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and float(logits.grad.abs().sum())>0


def test_ninth_language_is_scored_and_promotion_guard_applies():
    from src.models.full_training import eligible
    languages=LANGUAGES
    y=np.tile([0.,.5,1.],9);p=y.copy();lang=np.repeat(languages,3)
    result=evaluate(y,p,lang,languages)
    assert result['languages']['pan']['rows']==3
    import copy
    baseline=copy.deepcopy(result);baseline['macro_spearman']=.5
    for row in baseline['languages'].values():row['spearman']=.5
    candidate=copy.deepcopy(baseline);candidate['macro_spearman']=.6;candidate['languages']['pan']['spearman']=.4
    assert not eligible(candidate,baseline)


def test_full_nine_language_training_loop_saves_actual_best_state():
    from src.models.full_training import fit
    rng=np.random.default_rng(42);training_languages=np.repeat(TRAIN_LANGUAGES,4)
    x=rng.normal(size=(20,1539)).astype(np.float32);y=np.tile(np.linspace(.1,.9,4,dtype=np.float32),5)
    cohort=LANGUAGES;vl=np.repeat(cohort,3);vx=rng.normal(size=(27,1539)).astype(np.float32);vy=np.tile(np.asarray([.1,.5,.9],np.float32),9)
    config={'hidden':128,'dropout':.2,'lr':.001,'weight_decay':.01,'ranking_weight':.1}
    component,history=fit(x,y,training_languages,vx,vy,vl,config,42,maximum=3,cohort=cohort)
    assert 1<=history['best_epoch']<=3 and len(history['history'])==3
    assert all(torch.isfinite(v).all() for v in component['state_dict'].values())
    assert set(history['history'][0]['validation']['languages'])==set(cohort)


def test_global_selection_uses_strongest_baseline_and_all_languages():
    from src.models.sentence_comparison import select_candidate
    def row(name,family,value,pan=None):
        languages={l:{'spearman':value} for l in LANGUAGES}
        if pan is not None:languages['pan']['spearman']=pan
        return {'name':name,'family':family,'validation':{'macro_spearman':value,'languages':languages}}
    values=[row('character','baseline',.5),row('E5','baseline',.6),row('bad-Punjabi','mlp',.7,.4),row('eligible','mlp',.62)]
    winner,baseline=select_candidate(values)
    assert winner['name']=='eligible' and baseline['name']=='E5'

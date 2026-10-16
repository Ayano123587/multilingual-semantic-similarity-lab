import numpy as np
import pytest
from src.features.build_features import pair_features,fit_scaler,scale
from src.models.metrics import evaluate,promotable
from src.config import LANGUAGES


def test_pair_features_symmetric_and_ignore_targets_ids():
    rows=[{'sentence1':'a b c','sentence2':'b c d','target':0,'id':'secret'}]
    embeddings=np.array([[[1.,2],[3.,4]]])
    expected=pair_features(embeddings,rows)
    changed=[{'sentence1':'b c d','sentence2':'a b c','target':1,'id':'changed'}]
    assert np.array_equal(pair_features(embeddings[:,::-1],changed),expected)
    assert expected.shape==(1,7) and np.isfinite(expected).all()


def test_normalization_uses_train_only_and_roundtrips():
    train=np.array([[1.,2],[3.,4]])
    mean,std=fit_scaler(train)
    assert np.array_equal(mean,[2,3]) and np.array_equal(std,[1,1])
    assert np.allclose(scale(train,mean,std),[[-1,-1],[1,1]])
    before=mean.copy();scale(np.array([[200.,300]]),mean,std)
    assert np.array_equal(mean,before)
    with pytest.raises(ValueError):scale(train,mean,np.zeros(2))


def test_equal_language_macro_and_known_rank_result():
    target=np.tile([0.,.5,1.],3);languages=np.repeat(LANGUAGES,3)
    prediction=np.r_[target[:3],target[3:6],target[6:][::-1]]
    report=evaluate(target,prediction,languages)
    assert report['macro_spearman']==pytest.approx(1/3)
    assert report['languages']['kin']['spearman']==-1
    constant=evaluate(target,np.full(9,.5),languages)
    assert constant['macro_spearman'] is None


def test_confirmation_does_not_hide_language_regression():
    baseline={'macro_spearman':.6,'languages':{lang:{'spearman':.6} for lang in LANGUAGES}}
    candidate={'macro_spearman':.61,'languages':{lang:{'spearman':.61} for lang in LANGUAGES}}
    assert promotable(candidate,baseline)
    candidate['languages']['amh']['spearman']=.57
    assert not promotable(candidate,baseline)

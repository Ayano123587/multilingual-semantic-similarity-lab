import copy
import numpy as np
import pytest
import torch
from src.models.encoder_variants import pool_and_normalize,VARIANTS
from src.models.full_inference import predict_features,validate_bundle

def test_pool_excludes_padding_and_rejects_empty_mask():
    hidden=torch.tensor([[[3.,0.],[0.,4.],[100.,100.]]]);mask=torch.tensor([[1,1,0]])
    assert torch.allclose(pool_and_normalize(hidden,mask),torch.tensor([[.6,.8]]))
    with pytest.raises(ValueError):pool_and_normalize(hidden,torch.zeros_like(mask))

def test_pinned_encoder_identity_and_symmetric_finite_inference():
    spec=VARIANTS['e5'];b={'schema':'semrel_symmetric_v1','encoder':spec['model'],'revision':spec['revision'],'width':1539,'weight_sha256':{'model.safetensors':spec['sha256']},'prefix':'query: ','max_tokens':512,'kind':'baseline','name':'cosine','train_mean':.5}
    x=np.zeros((2,1539),np.float32);x[:,-3]=[.2,.8]
    assert predict_features(b,x).tolist()==pytest.approx([.6,.9])
    for key,value in [('revision','wrong'),('prefix','passage: '),('weight_sha256',{'model.safetensors':'0'*64}),('max_tokens',513)]:
        with pytest.raises(ValueError):validate_bundle({**b,key:value})


def test_reuse_refuses_truncated_source_before_allocating_encoder(tmp_path,monkeypatch):
    import json,hashlib
    import src.models.encoder_variants as module
    cache=tmp_path/'cache';cache.mkdir();source=tmp_path/'old';source.mkdir();new=tmp_path/'new';new.mkdir()
    spec=VARIANTS['afro']
    receipt={'model':spec['model'],'revision':spec['revision'],'weight_sha256':{'model.safetensors':spec['sha256']},'max_tokens':128,'splits':{'train':{'truncated_sentences':1}}}
    p=cache/'receipt.json';p.write_text(json.dumps(receipt),encoding='utf-8');digest=hashlib.sha256(p.read_bytes()).hexdigest()
    def forbidden(*args,**kwargs):raise AssertionError('Encoder allocation must follow source checks')
    monkeypatch.setattr(module,'Encoder',forbidden)
    with pytest.raises(ValueError,match='truncated'):
        module.extend_nontruncated_afro(cache,source,new,tmp_path/'output',digest)
    assert not (tmp_path/'output').exists()

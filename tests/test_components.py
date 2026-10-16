import numpy as np
from src.data.components import audit_components
from src.models.metrics import cluster_bootstrap
from src.config import LANGUAGES


def test_connected_groups_transitive_and_label_free():
    row=lambda a,b:{'language':'amh','sentence1':a,'sentence2':b}
    splits={'train':[row('a','b')],'dev':[row('b','c')],'test':[row('c','d'),row('e','f')]}
    groups,report=audit_components(splits)
    assert groups['train'][0]==groups['dev'][0]==groups['test'][0]
    assert groups['test'][1]!=groups['test'][0]
    assert report['languages']['amh']['cross_split_components']==1
    assert report['languages']['amh']['largest_component_sentence_nodes']==4
    assert not report['label_access']


def test_cluster_bootstrap_paired_reproducible():
    y=np.tile([0.,.2,.6,1.],3);languages=np.repeat(LANGUAGES,4)
    groups=np.tile(['a','b','c','d'],3)
    a=cluster_bootstrap(y,y,y,languages,groups,resamples=20)
    assert a==cluster_bootstrap(y,y,y,languages,groups,resamples=20)
    assert a['paired_macro_spearman_delta_95_percentile']==[0.,0.]

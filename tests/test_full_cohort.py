import csv
from src.data.full_cohort import read

def test_official_afr_tab_separator_and_sealed_score(tmp_path):
    p=tmp_path/'afr.csv'
    with p.open('w',encoding='utf-8',newline='') as h:
        w=csv.writer(h);w.writerow(['PairID','Text','score']);w.writerow(['AFR-test-1','First authentic sentence.\tSecond authentic sentence.','SEALED-NONNUMERIC'])
    rows=read(p,'afr',False)
    assert rows[0]['sentence2']=='Second authentic sentence.' and 'target' not in rows[0]

def test_full_cohort_train_validates_score_and_schema(tmp_path):
    import pytest
    p=tmp_path/'x.csv';p.write_text('PairID,Text,Score\nx,"first\nsecond",nan\n',encoding='utf-8')
    with pytest.raises(ValueError,match='human score'):read(p,'arq',True)


def test_punjabi_official_column_order_and_afr_numeric_test(tmp_path):
    p=tmp_path/'pan.csv'
    with p.open('w',encoding='utf-8',newline='') as h:
        writer=csv.writer(h);writer.writerow(['Text','Score','PairID']);writer.writerow(['one real headline\ntwo real headlines','.7','pan-1'])
    assert read(p,'pan',True)[0]['target']==.7
    p=tmp_path/'afr.csv'
    with p.open('w',encoding='utf-8',newline='') as h:
        writer=csv.writer(h);writer.writerow(['PairID','Text','score']);writer.writerow(['afr-1','first\tsecond','.2'])
    assert read(p,'afr',True)[0]['target']==.2

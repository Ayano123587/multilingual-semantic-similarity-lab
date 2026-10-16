import csv
from pathlib import Path
import pytest
from src.data.make_dataset import read_pairs, pair_key


def source(path,rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['PairID','Text','Score']);writer.writerows(rows)


def test_parser_preserves_real_text_and_skips_sealed_labels(tmp_path):
    path=tmp_path/'pairs.csv';source(path,[['pair1','sentence one\nsentence two','not-opened']])
    parsed=read_pairs(path,'amh',labels=False)
    assert parsed[0]['sentence1']=='sentence one' and 'target' not in parsed[0]
    with pytest.raises(ValueError):read_pairs(path,'amh',labels=True)


def test_unordered_normalized_pair_identity():
    row={'sentence1':'Ａ  B','sentence2':'c'}
    assert pair_key(row)==pair_key({'sentence1':'C','sentence2':'a b'})


@pytest.mark.parametrize('score',['nan','inf','-0.1','1.1'])
def test_invalid_labels_rejected(tmp_path,score):
    path=tmp_path/'pairs.csv';source(path,[['id','a\nb',score]])
    with pytest.raises(ValueError):read_pairs(path,'hau')


@pytest.mark.parametrize('text',['one sentence','a\nb\nc','a\n '])
def test_ambiguous_pair_format_fails_closed(tmp_path,text):
    path=tmp_path/'pairs.csv';source(path,[['id',text,'0.5']])
    with pytest.raises(ValueError,match='two nonempty'):read_pairs(path,'kin')


def test_repeated_ids_rejected(tmp_path):
    path=tmp_path/'pairs.csv';source(path,[['id','a\nb','0.5'],['id','c\nd','0.2']])
    with pytest.raises(ValueError,match='repeated'):read_pairs(path,'hau')


def test_preparation_writes_canonical_lf_and_never_parses_test_gold(tmp_path,monkeypatch):
    from src.data import make_dataset
    from src.config import LANGUAGES
    raw=tmp_path/'raw';raw.mkdir()
    for lang in LANGUAGES:
        for split in ('train','dev','test'):
            suffix='' if split=='train' else '_with_labels'
            source(raw/f'{lang}_{split}{suffix}.csv',[[f'{lang}-{split}',f'{split} one\n{split} two','unparsed' if split=='test' else '.5']])
    monkeypatch.setattr(make_dataset,'fetch',lambda directory:{'fixture':True})
    output=tmp_path/'prepared';report=make_dataset.prepare(raw,output)
    assert not report['test_gold_parsed']
    for split in ('train','dev','test'):
        content=(output/(split+'.jsonl')).read_bytes()
        assert b'\r\n' not in content and content.endswith(b'\n')

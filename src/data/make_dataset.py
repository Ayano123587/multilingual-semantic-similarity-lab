"""Immutable official sources; no translated/generated samples are introduced."""
import csv
import hashlib
import json
import math
import unicodedata
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from urllib.request import Request, urlopen
from src.config import DATA, LANGUAGES, DATA_REVISION, DATA_REPOSITORY, SOURCE_HASHES


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fetch(directory=DATA/'raw'/'semrel'):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    receipt_path=directory/'source-receipt.json'
    previous=json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else None
    tasks=[(lang,split, f'{lang}_{split}{"" if split=="train" else "_with_labels"}.csv')
           for lang in LANGUAGES for split in ('train','dev','test')]
    def download(task):
        lang,split,name=task
        path=directory/name
        url=f'https://raw.githubusercontent.com/{DATA_REPOSITORY}/{DATA_REVISION}/'+quote(f'Track A/{lang}/{name}')
        if not path.exists():
            with urlopen(Request(url,headers={'User-Agent':'semantic-relatedness-research/0.1'}),timeout=60) as response:
                content=response.read(2_000_001)
            if len(content)>2_000_000:raise ValueError('Unexpected source size')
            path.write_bytes(content)
        digest=sha(path)
        if digest!=SOURCE_HASHES[name]:raise ValueError('Official source checksum mismatch')
        if previous and previous['files'][name]['sha256']!=digest:
            raise ValueError('Pinned source bytes changed')
        with path.open(encoding='utf-8-sig',newline='') as handle:
            header=next(csv.reader(handle))
        return name,{'language':lang,'split':split,'sha256':digest,'source':url,'header':header}
    with ThreadPoolExecutor(max_workers=3) as pool:files=dict(pool.map(download,tasks))
    receipt={'repository':'https://github.com/'+DATA_REPOSITORY,'revision':DATA_REVISION,
             'files':files,'test_labels_inspected':False,
             'scope':'Source bytes/header inspection only; exact integrity audit precedes training'}
    if previous and receipt!=previous:raise ValueError('Source receipt changed')
    if not previous:receipt_path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def pair_key(row):
    return tuple(sorted((normalized(row['sentence1']),normalized(row['sentence2']))))


def read_pairs(path,language,labels=True):
    if language not in LANGUAGES:raise ValueError('Unsupported language')
    rows=[];identifiers=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle,strict=True)
        if reader.fieldnames!=['PairID','Text','Score']:raise ValueError('Unexpected SemRel schema')
        for row in reader:
            if None in row or any(value is None for value in row.values()):raise ValueError('Malformed SemRel CSV width')
            sentences=row['Text'].split('\n')
            if len(sentences)!=2 or not all(sentence.strip() for sentence in sentences):
                raise ValueError('Expected exactly two nonempty sentences per pair')
            identifier=row['PairID']
            if not identifier or identifier in identifiers:raise ValueError('Missing or repeated pair ID')
            identifiers.add(identifier)
            parsed={'id':identifier,'language':language,'sentence1':sentences[0],'sentence2':sentences[1]}
            if labels:
                score=float(row['Score'])
                if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Relatedness target must be finite in[0,1]')
                parsed['target']=score
            rows.append(parsed)
    if not rows:raise ValueError('Empty SemRel split')
    return rows


def prepare(directory=DATA/'raw'/'semrel',output=DATA/'processed'/'semrel'):
    receipt=fetch(directory);directory=Path(directory);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh processed directory; preserve evidence')
    output.mkdir(parents=True,exist_ok=True)
    prepared={'train':[],'dev':[],'test':[]};audit={}
    for language in LANGUAGES:
        train=read_pairs(directory/f'{language}_train.csv',language)
        dev=read_pairs(directory/f'{language}_dev_with_labels.csv',language)
        test=read_pairs(directory/f'{language}_test_with_labels.csv',language,labels=False)
        test_keys={pair_key(row) for row in test}
        original_train,original_dev=len(train),len(dev)
        dev=[row for row in dev if pair_key(row) not in test_keys]
        protected=test_keys|{pair_key(row) for row in dev}
        kept=[];seen=set()
        for row in train:
            key=pair_key(row)
            if key not in protected and key not in seen:kept.append(row);seen.add(key)
        train=kept
        sentences=lambda rows:{normalized(row[key]) for row in rows for key in ('sentence1','sentence2')}
        train_text,dev_text,test_text=map(sentences,(train,dev,test))
        audit[language]={'original_train':original_train,'original_dev':original_dev,'train':len(train),
             'dev':len(dev),'test':len(test),'removed_training_pairs':original_train-len(train),
             'removed_dev_test_pair_copies':original_dev-len(dev),
             'train_dev_sentence_overlap':len(train_text&dev_text),
             'train_test_sentence_overlap':len(train_text&test_text),
             'dev_test_sentence_overlap':len(dev_text&test_text),
             'remaining_cross_split_pair_overlap':len(seen&protected)}
        for split,rows in [('train',train),('dev',dev),('test',test)]:prepared[split].extend(rows)
    for split,rows in prepared.items():
        with (output/f'{split}.jsonl').open('w',encoding='utf-8',newline='\n') as handle:
            for row in rows:handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    report={'source':receipt,'languages':audit,'test_gold_parsed':False,
            'pair_normalization':'NFKC, casefold, whitespace; unordered pair; preserve original encoder text',
            'limits':['Shared individual sentences are disclosed; no claim of sentence-disjoint evaluation.',
                      'Development pair copies of test removed; corrected development cohort differs from official.',
                      'Human news-text source verified in primary paper; no generated examples added.'],
            'processed_sha256':{split:sha(output/f'{split}.jsonl') for split in prepared}}
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'languages':audit,'test_gold_parsed':False},indent=2))
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare()
    else:
        receipt=fetch()
        print(json.dumps({'revision':receipt['revision'],'files':{
            key:{'sha256':value['sha256'],'header':value['header']} for key,value in receipt['files'].items()}},indent=2))

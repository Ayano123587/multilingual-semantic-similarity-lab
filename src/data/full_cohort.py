"""All SemRel languages with human-source-compatible provenance; no row caps."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen
from src.config import DATA_REVISION,DATA_REPOSITORY,HUMAN_COMPATIBLE_LANGUAGES
from src.data.make_dataset import pair_key, normalized

LANGUAGES=HUMAN_COMPATIBLE_LANGUAGES
TRAIN_LANGUAGES=('amh','arq','ary','hau','kin')
TRACKS={lang:('Track A' if lang in TRAIN_LANGUAGES else 'Track B') for lang in LANGUAGES}
COVERAGE=Path(__file__).resolve().parents[2]/'docs/full-cohort-v3-coverage.json'


def read(path,language,labels):
    result=[];seen=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle,strict=True)
        if reader.fieldnames is None or len(reader.fieldnames)!=3 or set(reader.fieldnames) not in ({'PairID','Text','Score'},{'PairID','Text','score'}):raise ValueError('Unexpected official schema')
        for row in reader:
            if None in row or any(v is None for v in row.values()):raise ValueError('Malformed source row')
            parts=row['Text'].split('\n')
            # Official Afrikaans test CSV uses one literal TAB between sentences.
            if language=='afr' and len(parts)==1 and row['Text'].count('\t')==1:
                parts=row['Text'].split('\t')
            if len(parts)!=2 or not all(p.strip() for p in parts):raise ValueError('Invalid sentence pair')
            if not row['PairID'] or row['PairID'] in seen:raise ValueError('Invalid repeated ID')
            seen.add(row['PairID']);item={'id':row['PairID'],'language':language,'sentence1':parts[0],'sentence2':parts[1]}
            if labels:
                score=float(row.get('Score',row.get('score')))
                if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid human score')
                item['target']=score
            result.append(item)
    return result


def prepare(raw,output,coverage=COVERAGE):
    raw=Path(raw);output=Path(output)
    if output.exists():raise ValueError('Fresh full-cohort output required')
    coverage=Path(coverage);report=json.loads(coverage.read_text(encoding='utf-8'));languages=tuple(report['included_languages'])
    if report['revision']!=DATA_REVISION or languages not in (LANGUAGES[:-1],LANGUAGES):raise ValueError('Coverage contract changed')
    raw.mkdir(parents=True,exist_ok=True);splits={'train':[],'dev':[],'test':[]};audit={}
    for language in languages:
        rows={}
        for split in ('train','dev','test'):
            key=language+'_'+split
            if key not in report['files']:rows[split]=[];continue
            contract=report['files'][key];path=raw/Path(contract['path']).name
            if not path.exists():path.write_bytes(urlopen(contract['url'],timeout=60).read())
            if hashlib.sha256(path.read_bytes()).hexdigest()!=contract['sha256']:raise ValueError('Changed official source')
            rows[split]=read(path,language,split!='test')
            if len(rows[split])!=contract['rows']:raise ValueError('Changed source count')
        original={s:len(v) for s,v in rows.items()}
        test_keys={pair_key(r) for r in rows['test']}
        rows['dev']=[r for r in rows['dev'] if pair_key(r) not in test_keys]
        protected=test_keys|{pair_key(r) for r in rows['dev']};seen=set();kept=[]
        for row in rows['train']:
            key=pair_key(row)
            if key not in protected and key not in seen:kept.append(row);seen.add(key)
        rows['train']=kept
        audit[language]={'official':original,'prepared':{s:len(v) for s,v in rows.items()},
            'removed_training_duplicates_or_protected_pairs':original['train']-len(kept),
            'removed_development_pairs_duplicating_test':original['dev']-len(rows['dev']),
            'test_unchanged':len(rows['test'])==original['test']}
        for split in splits:splits[split].extend(rows[split])
    output.mkdir(parents=True);hashes={}
    for split,rows in splits.items():
        p=output/(split+'.jsonl')
        p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8',newline='\n')
        hashes[split]=hashlib.sha256(p.read_bytes()).hexdigest()
    result={'languages':audit,'processed_sha256':hashes,'test_gold_parsed':False,
        'scope':report['scope'],
        'coverage_sha256':hashlib.sha256(coverage.read_bytes()).hexdigest(),
        'previously_observed_test_languages':['amh','hau','kin'],
        'selection_languages':list(languages),'training_languages':list(TRAIN_LANGUAGES)}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',default='data/raw/semrel-full-v3');p.add_argument('--output',default='data/processed/semrel-full-v3')
    p.add_argument('--coverage',default=str(COVERAGE));a=p.parse_args();prepare(a.raw,a.output,a.coverage)

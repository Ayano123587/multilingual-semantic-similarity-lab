import re
import numpy as np
from src.data.make_dataset import normalized


def jaccard(a,b):
    union=a|b
    return len(a&b)/len(union) if union else 0.


def lexical(rows):
    values=[]
    for row in rows:
        a,b=normalized(row['sentence1']),normalized(row['sentence2'])
        words=lambda text:set(re.findall(r'\w+',text,flags=re.UNICODE))
        grams=lambda text:{text[i:i+3] for i in range(max(0,len(text)-2))}
        values.append([jaccard(words(a),words(b)),jaccard(grams(a),grams(b))])
    return np.asarray(values,dtype=np.float32)


def pair_features(embeddings,rows):
    e=np.asarray(embeddings,dtype=np.float32)
    if e.ndim!=3 or e.shape[:2]!=(len(rows),2) or not len(rows) or not np.isfinite(e).all():
        raise ValueError('Expected aligned finite pair embeddings')
    norms=np.linalg.norm(e.astype(np.float64),axis=2,keepdims=True)
    if (norms<=0).any():raise ValueError('Zero embedding')
    e=(e/norms).astype(np.float32)
    a,b=e[:,0],e[:,1]
    cosine=np.sum(a*b,axis=1,keepdims=True)
    values=np.concatenate([np.abs(a-b),a*b,cosine,lexical(rows)],axis=1)
    if not np.isfinite(values).all():raise ValueError('Nonfinite pair features')
    return values


def fit_scaler(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite nonempty training features required')
    return x.mean(axis=0).astype(np.float32),np.maximum(x.std(axis=0),1e-5).astype(np.float32)


def scale(values,mean,std):
    x=np.asarray(values,dtype=np.float32);mean=np.asarray(mean);std=np.asarray(std)
    if x.ndim!=2 or mean.shape!=(x.shape[1],) or std.shape!=mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any():
        raise ValueError('Invalid persisted normalization')
    result=(x-mean)/std
    if not np.isfinite(result).all():raise ValueError('Nonfinite normalized features')
    return result

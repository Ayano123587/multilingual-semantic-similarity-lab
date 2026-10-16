import numpy as np
from scipy.stats import pearsonr,spearmanr
from src.config import LANGUAGES


def evaluate(target,prediction,languages):
    y=np.asarray(target,dtype=float);p=np.asarray(prediction,dtype=float);langs=np.asarray(languages)
    if y.ndim!=1 or y.shape!=p.shape or y.shape!=langs.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Aligned finite predictions and targets required')
    if ((y<0)|(y>1)).any() or ((p<0)|(p>1)).any() or set(langs)!=set(LANGUAGES):
        raise ValueError('Expected three supported languages and scores in[0,1]')
    result={}
    for language in LANGUAGES:
        mask=langs==language;a,b=y[mask],p[mask]
        nonconstant=len(a)>1 and np.ptp(a)>0 and np.ptp(b)>0
        result[language]={'spearman':float(spearmanr(a,b).statistic) if nonconstant else None,
                          'pearson':float(pearsonr(a,b).statistic) if nonconstant else None,
                          'mae':float(np.abs(a-b).mean()),'rows':int(mask.sum())}
    valid=all(row['spearman'] is not None for row in result.values())
    return {'languages':result,'macro_spearman':float(np.mean([row['spearman'] for row in result.values()])) if valid else None,
            'macro_pearson':float(np.mean([row['pearson'] for row in result.values()])) if valid else None,
            'macro_mae':float(np.mean([row['mae'] for row in result.values()]))}


def promotable(candidate,baseline):
    if candidate['macro_spearman'] is None or baseline['macro_spearman'] is None:return False
    return (candidate['macro_spearman']>=baseline['macro_spearman']+.005 and
            all(candidate['languages'][lang]['spearman']>=baseline['languages'][lang]['spearman']-.02 for lang in LANGUAGES))


def cluster_bootstrap(target,prediction,baseline,languages,groups,resamples=1000,seed=2026):
    """Paired language-stratified sentence-component bootstrap, not an IID proof."""
    y,p,b,langs,g=map(np.asarray,(target,prediction,baseline,languages,groups))
    evaluate(y,p,langs);evaluate(y,b,langs)
    if g.shape!=y.shape or resamples<1:raise ValueError('Aligned groups and positive resamples required')
    rng=np.random.default_rng(seed);samples=[];per={lang:[] for lang in LANGUAGES}
    clusters={lang:[np.flatnonzero((langs==lang)&(g==group)) for group in np.unique(g[langs==lang])] for lang in LANGUAGES}
    for _ in range(resamples):
        pieces=[]
        for lang in LANGUAGES:
            options=clusters[lang];pieces.extend(options[i] for i in rng.integers(len(options),size=len(options)))
        ids=np.concatenate(pieces);a=evaluate(y[ids],p[ids],langs[ids]);z=evaluate(y[ids],b[ids],langs[ids])
        for lang in LANGUAGES:
            if a['languages'][lang]['spearman'] is not None:per[lang].append(a['languages'][lang]['spearman'])
        if a['macro_spearman'] is not None and z['macro_spearman'] is not None:samples.append([a['macro_spearman'],a['macro_spearman']-z['macro_spearman']])
    interval=lambda values:np.quantile(values,[.025,.975]).tolist() if len(values) else None
    values=np.asarray(samples)
    return {'seed':seed,'requested_resamples':resamples,'valid_macro_resamples':len(samples),
            'macro_spearman_95_percentile':interval(values[:,0]) if len(samples) else None,
            'paired_macro_spearman_delta_95_percentile':interval(values[:,1]) if len(samples) else None,
            'per_language_spearman_95_percentile':{lang:interval(rows) for lang,rows in per.items()},
            'components_per_language':{lang:len(rows) for lang,rows in clusters.items()},
            'limitation':'Descriptive conditional uncertainty; shared upstream text/pretraining and repeated validation selection remain.'}

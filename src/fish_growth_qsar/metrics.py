"""Original endpoint/structure estimand, chemical-shared conditional intervals."""
import numpy as np
import pandas as pd

def estimate(d, columns, seed, draws=4000):
    cells=d.groupby(['split','repeat','endpoint_key','group','parent_group'],as_index=False)[columns].mean()
    parents=sorted(cells.parent_group.unique())
    contexts=list(cells.groupby(['split','repeat','endpoint_key'],sort=True))
    # Each split/repeat contributes equally; endpoint groups within it equally.
    counts=cells[['split','repeat','endpoint_key']].drop_duplicates().groupby(['split','repeat']).size()
    rng=np.random.default_rng(seed)
    weights=rng.exponential(size=(draws,len(parents)))
    point=np.zeros(len(columns));boot=np.zeros((draws,len(columns)))
    for (s,r,e),cell in contexts:
        ids=pd.Categorical(cell.parent_group,categories=parents).codes
        values=cell[columns].to_numpy(float)
        weight=1/(len(counts)*counts.loc[(s,r)])
        point+=values.mean(axis=0)*weight
        ww=weights[:,ids]
        boot+=(ww@values)/ww.sum(axis=1)[:,None]*weight
    limits=np.quantile(boot,[.025,.975,.05/12,1-.05/12],axis=0)
    parent_mean=d.groupby('parent_group')[columns].mean().mean().to_numpy()
    return point,limits,parent_mean

def performance(d,models,seed):
    d=d.copy();columns=[];labels=[]
    for m in models:
        error=d[m]-d.value_for_model
        for metric,v in [('mae',abs(error)),('bias',error),('high_NOEC10',(error>1).astype(float)),
                         ('low_NOEC10',(error<-1).astype(float)),('positive_error',np.maximum(error,0))]:
            name=f'v{len(columns)}';d[name]=v;columns.append(name);labels.append((m,metric))
    avg,ci,pm=estimate(d,columns,seed)
    rows=[]
    for i,(m,metric) in enumerate(labels):
        rows.append(dict(model=m,metric=metric,value=avg[i],lower95=ci[0,i],upper95=ci[1,i],
                         chemical_macro=pm[i],parents=d.parent_group.nunique(),records=d.record_id.nunique()))
    return rows

def comparisons(d,primary,references,seed):
    d=d.copy();cols=[];labels=[]
    for ref in references:
        e=d[primary]-d.value_for_model;other=d[ref]-d.value_for_model
        for metric,v in [('mae',abs(e)-abs(other)),('high_NOEC10',(e>1).astype(float)-(other>1).astype(float)),
                         ('low_NOEC10',(e<-1).astype(float)-(other<-1).astype(float))]:
            col=f'd{len(cols)}';d[col]=v;cols.append(col);labels.append((ref,metric))
    avg,ci,pm=estimate(d,cols,seed)
    return [dict(model=primary,reference=ref,metric=metric,difference=avg[i],lower95=ci[0,i],upper95=ci[1,i],
                 simultaneous6_lower=ci[2,i],simultaneous6_upper=ci[3,i],chemical_macro_difference=pm[i])
            for i,(ref,metric) in enumerate(labels)]

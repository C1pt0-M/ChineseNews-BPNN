"""Supplement lexical/event screening using character n-gram similarity.

This is an audit only: heldout columns are limited to IDs, URLs, titles and dates.
The vectorizer is never used for training or feature selection.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default='f3_a_f34_review_candidate')
parser.add_argument('--output',default='f34_broad_title_flags.csv')
parser.add_argument('--threshold',type=float,default=.5)
parser.add_argument('--max-days',type=int,default=None)
args=parser.parse_args()
cols=['id','url','title','publish_time']
new=pd.read_csv(HERE/args.dataset/'new_sample_evidence.csv',usecols=cols,dtype=str,encoding='utf-8-sig').fillna('')
parts=[]
for s in ('train','val','test'):
    f=pd.read_csv(ROOT/'dataset/splits'/f'{s}.csv',usecols=cols,dtype=str,encoding='utf-8-sig').fillna('')
    f['reference_type']=s;parts.append(f)
ref=pd.concat(parts+[new.assign(reference_type='new')],ignore_index=True)
vector=TfidfVectorizer(analyzer='char',ngram_range=(2,3),sublinear_tf=True)
x=vector.fit_transform(pd.concat([new.title,ref.title],ignore_index=True))
sim=(x[:len(new)]@x[len(new):].T).tocsr()
hits=[]
for i in range(len(new)):
    s,e=sim.indptr[i:i+2]
    for j,value in zip(sim.indices[s:e],sim.data[s:e]):
        if value<args.threshold: continue
        a,b=new.iloc[i],ref.iloc[j]
        if a.id==b.id: continue
        if b.reference_type=='new' and a.id>=b.id: continue
        if args.max_days is not None and abs((pd.Timestamp(a.publish_time)-pd.Timestamp(b.publish_time)).days)>args.max_days: continue
        hits.append({'candidate_id':a.id,'candidate_title':a.title,'candidate_date':a.publish_time,
                     'reference_type':b.reference_type,'reference_id':b.id,'reference_title':b.title,
                     'reference_date':b.publish_time,'cosine':float(value)})
out=HERE/args.output
if out.exists(): raise FileExistsError(out)
pd.DataFrame(hits).sort_values('cosine',ascending=False).to_csv(out,index=False,encoding='utf-8-sig')
print('flags:',len(hits));print(pd.DataFrame(hits).reference_type.value_counts().to_string())

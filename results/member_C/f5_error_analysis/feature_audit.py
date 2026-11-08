"""Descriptive sparse-coverage and training-neighbor audit; no fit or test evaluation."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import normalize

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from models.features import tokenize_corpus

LABELS=['财经','体育','文娱','教育','社会','国际']
paths=['feature_per_title.csv','nearest_training.csv','training_month_distribution.csv',
       'feature_group_summary.csv','class_centroid_similarity.csv','added_training_topics.csv','feature_audit.json']
for n in paths:
    if (HERE/n).exists():raise FileExistsError(HERE/n)
read=lambda p:pd.read_csv(p,dtype=str,keep_default_na=False,encoding='utf-8-sig')
base=read(ROOT/'dataset/splits/train.csv')
expanded=read(ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv')
val=read(ROOT/'dataset/splits/val.csv')
pred=read(ROOT/'results/member_C/f3_analysis/f3_a_multiseed/prediction_comparison.csv')
manifest=json.loads((ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal/manifest.json').read_text(encoding='utf-8'))
version=json.loads((ROOT/'dataset/dataset_version.json').read_text(encoding='utf-8'))
sha=lambda p:sha256(p.read_bytes()).hexdigest()
assert sha(ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv')==manifest['train_sha256']
for s in ('train','val'):assert sha(ROOT/f'dataset/splits/{s}.csv')==version['files'][f'{s}.csv']['sha256']
assert len(base)==6549 and len(expanded)==7531 and len(val)==3184 and pred.id.tolist()==val.id.tolist()
assert expanded.iloc[:len(base)].reset_index(drop=True).equals(base)
assert val.title.tolist()==pred.title.tolist() and val.category.tolist()==pred.category.tolist()
vec_paths={'F2':ROOT/'results/member_C/f2_formal_word_char/F_word_char_s42/feature_builder.pkl',
           'F3-A':ROOT/'results/member_C/f3_formal_a_reviewed/F_word_char_s42/feature_builder.pkl'}
builders={}
for k,p in vec_paths.items():
    with p.open('rb') as f:builders[k]=pickle.load(f)
assert builders['F2'].fit_rows_==6549 and builders['F3-A'].fit_rows_==7531

train_tokens=tokenize_corpus(expanded.title.tolist())
val_tokens=tokenize_corpus(val.title.tolist())
# Analyzer settings are identical and no vocabulary is fitted in this script.
w_anal=builders['F3-A'].word.vectorizer.build_analyzer()
c_anal=builders['F3-A'].char_vectorizer.build_analyzer()
train_w=[set(w_anal(t)) for t in train_tokens]
train_c=[set(c_anal(t)) for t in expanded.title]
val_w=[set(w_anal(t)) for t in val_tokens]
val_c=[set(c_anal(t)) for t in val.title]
seen={}
for name,count in [('F2',6549),('F3-A',7531)]:
    sw=set().union(*train_w[:count]);sc=set().union(*train_c[:count]);seen[name]=(sw,sc)

features=[];neighbors=[];month=[];centroids=[];topics=[];meta={}
for name,train in [('F2',base),('F3-A',expanded)]:
    b=builders[name];count=len(train)
    wt_raw=b.word.vectorizer.transform(train_tokens[:count]);ct_raw=b.char_vectorizer.transform(train.title.tolist())
    wv_raw=b.word.vectorizer.transform(val_tokens);cv_raw=b.char_vectorizer.transform(val.title.tolist())
    wt=b.word.selector.transform(wt_raw).tocsr();ct=b.char_selector.transform(ct_raw).tocsr()
    wv=b.word.selector.transform(wv_raw).tocsr();cv=b.char_selector.transform(cv_raw).tocsr()
    xt=normalize(sp.hstack((wt,ct),format='csr'),norm='l2',copy=False)
    xv=normalize(sp.hstack((wv,cv),format='csr'),norm='l2',copy=False)
    assert sp.issparse(xt) and sp.issparse(xv) and xt.shape==(count,10000) and xv.shape==(3184,10000)
    difference=xv[:10]-b.transform(val.title.tolist()[:10]);assert difference.nnz==0 or np.max(np.abs(difference.data))<1e-12
    dims=[np.diff(x.indptr) for x in (wv_raw,wv,cv_raw,cv,xv)]
    wseen,cseen=seen[name];wvocab=set(b.word.vectorizer.vocabulary_);cvocab=set(b.char_vectorizer.vocabulary_)
    for i,row in val.iterrows():
        w,c=val_w[i],val_c[i]
        nw,nc=len(w-wseen),len(c-cseen)
        group=pred.iloc[i]
        features.append({'model':name,'id':row.id,'category':row.category,'title':row.title,
            'persistent_both':group.f2_error_count=='5' and group.f3_error_count=='5',
            'model_all_wrong':(group.f2_error_count if name=='F2' else group.f3_error_count)=='5',
            'model_all_correct':(group.f2_error_count if name=='F2' else group.f3_error_count)=='0',
            'word_analyzed_unique':len(w),'word_never_seen_unique':nw,'word_never_seen_fraction':nw/len(w) if w else np.nan,
            'word_not_in_vocab_unique':len(w-wvocab),'word_vocab_miss_fraction':len(w-wvocab)/len(w) if w else np.nan,
            'char_analyzed_unique':len(c),'char_never_seen_unique':nc,'char_never_seen_fraction':nc/len(c) if c else np.nan,
            'char_not_in_vocab_unique':len(c-cvocab),'char_vocab_miss_fraction':len(c-cvocab)/len(c) if c else np.nan,
            'word_tfidf_nnz':int(dims[0][i]),'word_selected_nnz':int(dims[1][i]),
            'char_tfidf_nnz':int(dims[2][i]),'char_selected_nnz':int(dims[3][i]),'combined_nnz':int(dims[4][i])})
    tr_labels=train.category.to_numpy();tr_ids=train.id.to_numpy();tr_titles=train.title.to_numpy();tr_urls=train.url.to_numpy()
    for begin in range(0,len(val),128):
        sims=(xv[begin:begin+128]@xt.T).toarray()
        for off,s in enumerate(sims):
            i=begin+off;true=val.iloc[i].category
            # Rank ties by original training row index, avoiding unstable argsort ties.
            ranking=np.lexsort((np.arange(len(s)),-s))
            for role,allowed in [('nearest_any',np.ones(len(s),dtype=bool)),('nearest_same',tr_labels==true),('nearest_other',tr_labels!=true)]:
                j=next((int(k) for k in ranking if allowed[k] and s[k]>0),None)
                neighbors.append({'model':name,'id':val.iloc[i].id,'true_category':true,'role':role,
                    'train_id':str(tr_ids[j]) if j is not None else '', 'train_category':str(tr_labels[j]) if j is not None else '',
                    'train_title':str(tr_titles[j]) if j is not None else '', 'train_url':str(tr_urls[j]) if j is not None else '',
                    'cosine':float(s[j]) if j is not None else 0.0})
    for cat in LABELS:
        months=train[train.category==cat].publish_time.str[5:7].value_counts().to_dict()
        for m in range(1,13):month.append({'model':name,'category':cat,'month':m,'count':int(months.get(f'{m:02d}',0))})
    centers=[]
    for cat in LABELS:centers.append(np.asarray(xt[tr_labels==cat].mean(axis=0)).ravel())
    centers=normalize(np.array(centers),norm='l2')
    for i,cat in enumerate(LABELS):
        for j,cat2 in enumerate(LABELS):centroids.append({'model':name,'category':cat,'other_category':cat2,'cosine':float(centers[i]@centers[j])})
    # JSD on normalized mean selected-feature weights, not on raw text or labels.
    a=np.asarray(xt.mean(axis=0)).ravel();z=np.asarray(xv.mean(axis=0)).ravel()
    meta[name]={'train_rows':count,'val_rows':len(val),'tfidf_selected_input_dim':10000,
                'word_seen_train_unique':len(wseen),'char_seen_train_unique':len(cseen),
                'word_vectorizer_vocabulary':len(wvocab),'char_vectorizer_vocabulary':len(cvocab),
                'train_val_selected_mean_weight_js_divergence_bits':float(jensenshannon(a/a.sum(),z/z.sum(),base=2)**2)}
    if name=='F3-A':
        names=b.word.vectorizer.get_feature_names_out()[b.word.selector.get_support()]
        for cat in LABELS:
            mask=tr_labels==cat;is_new=np.arange(count)>=len(base)
            add=wt[mask&is_new];old=wt[mask&~is_new]
            df1=np.asarray((add>0).sum(axis=0)).ravel();df0=np.asarray((old>0).sum(axis=0)).ravel()
            order=np.lexsort((names,-df1))[:25]
            for j in order:
                topics.append({'category':cat,'feature':names[j],'added_docs':int(df1[j]),'added_total':add.shape[0],
                               'base_docs':int(df0[j]),'base_total':old.shape[0],
                               'added_doc_fraction':float(df1[j]/add.shape[0]),'base_doc_fraction':float(df0[j]/old.shape[0])})
    print('features and nearest complete',name,flush=True)

f=pd.DataFrame(features);near=pd.DataFrame(neighbors)
f.to_csv(HERE/'feature_per_title.csv',index=False,encoding='utf-8-sig')
near.to_csv(HERE/'nearest_training.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(month).to_csv(HERE/'training_month_distribution.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(centroids).to_csv(HERE/'class_centroid_similarity.csv',index=False,encoding='utf-8-sig')
pd.DataFrame(topics).to_csv(HERE/'added_training_topics.csv',index=False,encoding='utf-8-sig')
summ=[]
for (name,cohort),sub in [( (n,g),d[(d.model==n)&d[g]]) for n in ('F2','F3-A') for g in ('persistent_both','model_all_wrong','model_all_correct') for d in [f]]:
    row={'model':name,'cohort':cohort,'n':len(sub)}
    for col in ('word_selected_nnz','char_selected_nnz','combined_nnz','word_never_seen_fraction','char_never_seen_fraction','word_vocab_miss_fraction','char_vocab_miss_fraction'):
        row[col+'_mean']=float(sub[col].mean());row[col+'_median']=float(sub[col].median())
    for col in ('word_selected_nnz','char_selected_nnz','combined_nnz'):row[col+'_zero_rate']=float((sub[col]==0).mean())
    summ.append(row)
pd.DataFrame(summ).to_csv(HERE/'feature_group_summary.csv',index=False,encoding='utf-8-sig')
meta.update({'definitions':{'unseen':'distinct analyzer ngrams absent from raw training titles, prior to min_df/max_df and chi2',
                           'vocab_miss':'distinct analyzer ngrams absent from fitted vectorizer vocabulary; includes pruning',
                           'coverage':'nonzero selected TF-IDF features per title',
                           'nearest':'cosine of final concatenated L2 normalized sparse TF-IDF, no supervised refitting',
                           'centroid':'L2 normalized mean of per-title final training vectors',
                           'js':'Jensen-Shannon divergence in bits between L1-normalized mean selected-feature weights, train vs validation'},
             'fit_called':False,'test_used':False})
(HERE/'feature_audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False,indent=2))

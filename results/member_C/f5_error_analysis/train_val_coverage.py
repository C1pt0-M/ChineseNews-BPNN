"""Compare nonzero selected branch coverage across train/validation by category."""
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from models.features import tokenize_corpus
out=HERE/'train_val_coverage.csv'
if out.exists():raise FileExistsError(out)
files={('F2','train'):'dataset/splits/train.csv',('F2','validation'):'dataset/splits/val.csv',
       ('F3-A','train'):'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv',('F3-A','validation'):'dataset/splits/val.csv'}
builders={}
for name,path in [('F2','results/member_C/f2_formal_word_char/F_word_char_s42/feature_builder.pkl'),
                  ('F3-A','results/member_C/f3_formal_a_reviewed/F_word_char_s42/feature_builder.pkl')]:
    with (ROOT/path).open('rb') as f:builders[name]=pickle.load(f)
rows=[]
for (name,split),path in files.items():
    df=pd.read_csv(ROOT/path,encoding='utf-8-sig');b=builders[name]
    w=b.word.transform(tokenize_corpus(df.title.tolist())).tocsr()
    c=b.char_selector.transform(b.char_vectorizer.transform(df.title.tolist())).tocsr()
    both=normalize(sp.hstack((w,c),format='csr'),norm='l2')
    for cat in ['全部','财经','体育','文娱','教育','社会','国际']:
        mask=np.ones(len(df),dtype=bool) if cat=='全部' else (df.category.to_numpy()==cat)
        counts=[np.diff(x[mask].indptr) for x in (w,c,both)]
        row={'model':name,'split':split,'category':cat,'rows':int(mask.sum())}
        for key,a in zip(('word','char','combined'),counts):
            row[key+'_nnz_mean']=float(a.mean());row[key+'_nnz_median']=float(np.median(a));row[key+'_zero_rate']=float((a==0).mean())
        rows.append(row)
pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
print('saved',len(rows),'rows')

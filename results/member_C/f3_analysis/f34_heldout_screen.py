"""Screen selected news against heldout IDs/titles/dates; do not read labels."""
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('prior_audit', ROOT/'results/member_C/f3_recovery/audit_recovery.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
selected=audit.enrich(audit.read(HERE/'f3_a_f34_review_candidate/new_sample_evidence.csv')).reset_index(drop=True)
hits=[]
for split in ('val','test'):
    ref=pd.read_csv(ROOT/'dataset/splits'/f'{split}.csv',usecols=['id','url','title','publish_time'],dtype=str,keep_default_na=False,encoding='utf-8-sig')
    ref=audit.enrich(ref).reset_index(drop=True)
    idx=audit.NearIndex()
    for title in ref._title_key: idx.add(title)
    days=[datetime.fromisoformat(x[:10]).date().toordinal() for x in ref.publish_time]
    for _,row in selected.iterrows():
        g=audit.grams(row['_title_key'])
        counts=Counter(p for gram in g for p in idx.index.get(gram,()))
        day=datetime.fromisoformat(row.publish_time[:10]).date().toordinal()
        for j,n in counts.items():
            if abs(day-days[j])>7: continue
            jac=n/(len(g)+len(idx.values[j])-n)
            if jac<.25: continue
            other=ref.iloc[j]
            ratio=SequenceMatcher(None,row['_title_key'],other['_title_key']).ratio()
            if ratio>=.65:
                hits.append({'candidate_id':row.id,'candidate_title':row.title,'candidate_date':row.publish_time,
                             'reference_type':split,'reference_id':other.id,'reference_title':other.title,
                             'reference_date':other.publish_time,'trigram_jaccard':jac,'sequence_ratio':ratio})
out=HERE/'f34_heldout_event_flags.csv'
if out.exists(): raise FileExistsError(out)
cols=['candidate_id','candidate_title','candidate_date','reference_type','reference_id','reference_title','reference_date','trigram_jaccard','sequence_ratio']
pd.DataFrame(hits,columns=cols).to_csv(out,index=False,encoding='utf-8-sig')
print('heldout weaker lexical flags:',len(hits))

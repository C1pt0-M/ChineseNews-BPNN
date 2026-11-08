"""Conservatively exclude additional reports tied to the same underlying cases."""
import importlib.util
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
exclude={
 'cn10473606':'与原训练 cn10470438 为同一韩德洙涉戒严案件的起诉和申请逮捕',
 'cn10365856':'与原训练 cn10363413 涉同届比赛同一选手，保守排除',
 'cn10407169':'与原训练 cn10409692、cn10408617 涉同届比赛同一选手，保守排除',
}
df=pd.read_csv(HERE/'f34_title_review_decisions_v7.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
for i,reason in exclude.items():
    assert (df.id==i).sum()==1,i
    df.loc[df.id==i,'semantic_review']='exclude_possible_same_event'
    df.loc[df.id==i,'review_note']=reason
out=HERE/'f34_title_review_decisions_v8.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v8',out.name)

"""Remove an education record whose headline cohort cannot be verified from its body."""
import importlib.util
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/'f34_title_review_decisions_v9.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
i='cn10492102';assert (df.id==i).sum()==1
df.loc[df.id==i,'semantic_review']='exclude_insufficient_body_evidence'
df.loc[df.id==i,'review_note']='标题称05后，正文没有给出相关人物出生年份；保守排除'
out=HERE/'f34_title_review_decisions_v10.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v10',out.name)

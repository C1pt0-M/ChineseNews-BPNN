"""Remove duplicate event uncovered in the latest replacement."""
import importlib.util
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/'f34_title_review_decisions_v6.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
i='cn10365179';assert (df.id==i).sum()==1
df.loc[df.id==i,'semantic_review']='exclude_possible_same_event'
df.loc[df.id==i,'review_note']='与原训练 cn10365195 涉李磊同一亚冬会越野滑雪金牌'
out=HERE/'f34_title_review_decisions_v7.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v7',out.name)

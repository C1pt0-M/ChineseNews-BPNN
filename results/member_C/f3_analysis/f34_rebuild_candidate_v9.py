"""Exclude two reports based on the same government press conference."""
import importlib.util
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/'f34_title_review_decisions_v8.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
i='cn10491148';assert (df.id==i).sum()==1
df.loc[df.id==i,'semantic_review']='exclude_possible_same_event'
df.loc[df.id==i,'review_note']='与训练 cn10491275 为同一新疆新闻办发布会，虽然发言人和主题不同仍保守排除'
out=HERE/'f34_title_review_decisions_v9.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v9',out.name)

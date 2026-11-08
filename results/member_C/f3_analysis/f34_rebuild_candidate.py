"""Build a separate review candidate after excluding confirmed source conflicts."""
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
df=pd.read_csv(HERE/'f3_title_review_decisions_v3.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
assert (df.id=='cn10516755').sum()==1
df.loc[df.id=='cn10516755','semantic_review']='exclude_source_conflict'
df.loc[df.id=='cn10516755','review_note']='官方原文正文称上调9个基点，标题称33个基点，数值冲突'
decisions=HERE/'f34_title_review_decisions.csv'
if decisions.exists(): raise FileExistsError(decisions)
df.to_csv(decisions,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder', HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate','f34_title_review_decisions.csv')

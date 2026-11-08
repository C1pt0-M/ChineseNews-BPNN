"""Exclude unresolved overlap and unsupported headline quantities."""
import importlib.util
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
exclude={
 'cn10410887':'与原训练 cn10407695 同涉五一文旅与演出市场，暂排除',
 'cn10425255':'与原训练 cn10425717 同涉端午电影档票房，暂排除',
 'cn10513098':'与原训练 cn10512977 可能同涉十五运开幕式，暂排除',
 'cn10517678':'与原训练皮划艇赛事综合报道可能重叠，暂排除',
 'cn10515580':'与原训练 cn10519294 的羽毛球赛事综合报道可能重叠，暂排除',
 'cn10439317':'标题称发放4515万元；正文金额与其无法直接核对，暂排除',
 'cn10386677':'标题称35所院校，正文未给出数量，暂排除',
 'cn10518130':'标题称超300名运动员，正文194人含竞赛辅助人员，无法直接核对，暂排除',
}
df=pd.read_csv(HERE/'f34_title_review_decisions_v5.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
for i,reason in exclude.items():
    assert (df.id==i).sum()==1,i
    df.loc[df.id==i,'semantic_review']='exclude_event_or_evidence'
    df.loc[df.id==i,'review_note']=reason
out=HERE/'f34_title_review_decisions_v6.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v6',out.name)

"""Remove overlap found among replacement candidates before another audit."""
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
exclude={
 'cn10493294':'与原训练 cn10493232 报道同一电影总票房425.02亿元',
 'cn10412851':'与原训练 cn10412874 报道同一智慧教育平台注册人数1.64亿',
 'cn10532816':'残特奥田径报道与原训练 cn10533231 所述比赛可能重叠，暂排除',
 'cn10519086':'羽毛球女单决赛与原训练 cn10519294 赛事收官报道可能重叠，暂排除',
}
df=pd.read_csv(HERE/'f34_title_review_decisions_v4.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
for i,reason in exclude.items():
    assert (df.id==i).sum()==1,i
    df.loc[df.id==i,'semantic_review']='exclude_possible_same_event'
    df.loc[df.id==i,'review_note']=reason
out=HERE/'f34_title_review_decisions_v5.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v5',out.name)

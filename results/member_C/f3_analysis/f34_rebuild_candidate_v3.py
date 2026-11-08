"""Third review candidate; remove an event overlap and two weakly evidenced titles."""
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
exclude={
 'cn10363974':'与原训练 cn10364713 同涉特朗普接管加沙的同一原始事件，保守排除',
 'cn10429463':'正文提供中间价但未验证标题的上调15基点，证据不足排除',
 'cn10405444':'正文文字未包含标题所称7天期限，细节显示于图片，证据未充分复核排除',
}
df=pd.read_csv(HERE/'f34_title_review_decisions_v2.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
for id,reason in exclude.items():
    assert (df.id==id).sum()==1
    df.loc[df.id==id,'semantic_review']='exclude_event_or_evidence'
    df.loc[df.id==id,'review_note']=reason
out=HERE/'f34_title_review_decisions_v3.csv'
if out.exists():raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v3',out.name)

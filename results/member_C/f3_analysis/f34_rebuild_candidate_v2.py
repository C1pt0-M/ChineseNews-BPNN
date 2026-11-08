"""Second review candidate; conservatively remove additional possible same-event news."""
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
exclude={
 'cn10467070':'与基准 cn10467072 涉及同一次白宫会晤，暂排除',
 'cn10484114':'与基准 cn10483974 涉及同一次财政数据公布，暂排除',
 'cn10439388':'与基准 cn10439397 涉及同一次自行车锦标赛收官，暂排除',
 'cn10540124':'与基准 cn10540122 涉及同一学位授予资格审核办法公布和解读，暂排除',
 'cn10373703':'与新增 cn10370749 同期报道多地延长课间，同事件边界存疑，暂排除',
 'cn10376578':'与新增 cn10371440 同期报道多地学校 AI 教育趋势，同事件边界存疑，暂排除',
}
df=pd.read_csv(HERE/'f34_title_review_decisions.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
for id,reason in exclude.items():
    assert (df.id==id).sum()==1
    df.loc[df.id==id,'semantic_review']='exclude_possible_same_event'
    df.loc[df.id==id,'review_note']=reason
out=HERE/'f34_title_review_decisions_v2.csv'
if out.exists(): raise FileExistsError(out)
df.to_csv(out,index=False,encoding='utf-8-sig')
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v2',out.name)

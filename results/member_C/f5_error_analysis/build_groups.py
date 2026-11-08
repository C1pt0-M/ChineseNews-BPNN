"""Rebuild five-seed error cohorts and a reproducible stratified review queue."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SEEDS=range(42,47)
groups=['持续错误','稳定修正','稳定退化','种子不稳定']
files=['error_groups.csv','error_group_statistics.csv','review_queue.csv','group_audit.json']
for name in files:
    if (HERE/name).exists():raise FileExistsError(HERE/name)
read=lambda p:pd.read_csv(p,encoding='utf-8-sig',dtype=str,keep_default_na=False)
val=read(ROOT/'dataset/splits/val.csv')
assert len(val)==3184 and val.id.is_unique
ref=val[['id','title','category','raw_category','publish_time','url','source']].copy()
for method in ('f2','f3'):
    for seed in SEEDS:
        if method=='f2':path=ROOT/f'results/member_C/f2_formal_word_char/F_word_char_s{seed}/val_predictions.csv'
        elif seed==42:path=ROOT/'results/member_C/f3_formal_a_reviewed/F_word_char_s42/val_predictions.csv'
        else:path=ROOT/f'results/member_C/f3_analysis/f3_a_multiseed/runs/F_word_char_s{seed}/val_predictions.csv'
        p=read(path)
        assert p.id.tolist()==val.id.tolist() and p.title.tolist()==val.title.tolist() and p.category.tolist()==val.category.tolist()
        ref[f'{method}_{seed}']=p.predicted_category
    cols=[f'{method}_{s}' for s in SEEDS]
    ref[f'{method}_wrong_count']=(ref[cols].to_numpy()!=ref.category.to_numpy()[:,None]).sum(axis=1)
    ref[f'{method}_unique_predictions']=ref[cols].nunique(axis=1)
    ref[f'{method}_votes']=[json.dumps(dict(Counter(row)),ensure_ascii=False,sort_keys=True) for row in ref[cols].to_numpy()]
    ref[f'{method}_mode']=[Counter(row).most_common(1)[0][0] for row in ref[cols].to_numpy()]
ref['持续错误']=(ref.f2_wrong_count==5)&(ref.f3_wrong_count==5)
ref['稳定修正']=(ref.f2_wrong_count==5)&(ref.f3_wrong_count==0)
ref['稳定退化']=(ref.f2_wrong_count==0)&(ref.f3_wrong_count==5)
ref['种子不稳定']=(ref.f2_unique_predictions>1)|(ref.f3_unique_predictions>1)
ref['双组五次均正确']=(ref.f2_wrong_count==0)&(ref.f3_wrong_count==0)
assert [int(ref[x].sum()) for x in groups[:3]]==[548,30,32]
rows=[]
for group in groups:
    for r in ref[ref[group]].to_dict('records'):
        rows.append({'group':group,'id':r['id'],'title':r['title'],'category':r['category'],
                     'raw_category':r['raw_category'],'url':r['url'],'publish_time':r['publish_time'],
                     'f2_votes':r['f2_votes'],'f3_votes':r['f3_votes'],
                     'f2_mode':r['f2_mode'],'f3_mode':r['f3_mode'],
                     'f2_wrong_count':r['f2_wrong_count'],'f3_wrong_count':r['f3_wrong_count']})
all_groups=pd.DataFrame(rows)
all_groups.to_csv(HERE/'error_groups.csv',index=False,encoding='utf-8-sig')
stats=[]
for group in groups:
    sub=ref[ref[group]]
    for cat in ['财经','体育','文娱','教育','社会','国际']:
        part=sub[sub.category==cat]
        stats.append({'group':group,'category':cat,'count':len(part),'group_total':len(sub),
                      'share_of_validation':len(part)/len(ref),'share_of_group':len(part)/len(sub) if len(sub) else 0})
pd.DataFrame(stats).to_csv(HERE/'error_group_statistics.csv',index=False,encoding='utf-8-sig')

def ranked(frame,group):
    frame=frame.copy();frame['_rank']=frame.id.map(lambda i:sha256(f'f5-review-2026-v1:42:{group}:{i}'.encode()).hexdigest())
    return frame.sort_values('_rank')

# Direction targets are based on the F3 five-seed mode, never label overrides.
pool=ranked(ref[ref['持续错误']],'持续错误')
rules=[('社会','财经',12),('社会','文娱',12),('财经','社会',6),('文娱','社会',5)]
chosen=[];used=set()
for true,pred,n in rules:
    part=pool[(pool.category==true)&(pool.f3_mode==pred)&~pool.id.isin(used)].head(n)
    chosen.extend(part.id);used.update(part.id)
# Round-robin remaining categories; rank gives repeatability within each category.
while len(chosen)<60:
    before=len(chosen)
    for cat in ['社会','财经','文娱','体育','教育','国际']:
        p=pool[(pool.category==cat)&~pool.id.isin(used)].head(1)
        if len(p):chosen.append(p.iloc[0].id);used.add(p.iloc[0].id)
        if len(chosen)==60:break
    if len(chosen)==before:raise RuntimeError('insufficient persistent errors')
selected={'持续错误':chosen,'稳定修正':ranked(ref[ref['稳定修正']],'稳定修正').id.tolist(),
          '稳定退化':ranked(ref[ref['稳定退化']],'稳定退化').id.tolist()}
used=set(sum(selected.values(),[]))
unstable=ranked(ref[ref['种子不稳定']&~ref.id.isin(used)],'种子不稳定')
chosen=[]
while len(chosen)<30:
    before=len(chosen)
    for cat in ['社会','财经','文娱','体育','教育','国际']:
        p=unstable[(unstable.category==cat)&~unstable.id.isin(chosen)].head(1)
        if len(p):chosen.append(p.iloc[0].id)
        if len(chosen)==30:break
    if len(chosen)==before:raise RuntimeError('insufficient unstable cases')
selected['种子不稳定']=chosen
queue=[]
by_id=ref.set_index('id')
for group in groups:
    for id in selected[group]:
        r=by_id.loc[id].to_dict();queue.append({'review_group':group,'id':id,**r})
out=pd.DataFrame(queue)
assert len(out)==152 and out.id.is_unique
out.to_csv(HERE/'review_queue.csv',index=False,encoding='utf-8-sig')
audit={'validation_rows':len(ref),'group_counts':{g:int(ref[g].sum()) for g in groups},
       'validation_percent':{g:float(ref[g].mean()*100) for g in groups},
       'review_counts':{g:len(selected[g]) for g in groups},'review_seed':42,
       'review_sampling':'fixed hash; 60 persistent with priority directions, all 30 corrected and 32 degraded, 30 non-overlapping unstable balanced by category',
       'groups_overlap':'unstable may overlap the first three groups; review sample is deduplicated',
       'test_labels_or_metrics_used':False}
(HERE/'group_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))

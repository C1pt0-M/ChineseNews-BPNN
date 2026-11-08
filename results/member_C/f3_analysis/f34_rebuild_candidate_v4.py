"""Expand the explicitly reviewed education pool; exclude identified overlap risks."""
from hashlib import sha256
import importlib.util
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
exclude={
 'cn10468655':'与新增 cn10459290 是卓尼逼停摩托案件的后续通报',
 'cn10446737':'与原训练 cn10443902 和新增 cn10447235 均涉大状王北京演出',
 'cn10447235':'与原训练 cn10443902 和新增 cn10446737 均涉大状王演出',
 'cn10510200':'与原训练 cn10510257 涉同一青年教师建设意见及解读',
 'cn10528647':'与原训练 cn10528665 涉同一船海核领域学院成立',
 'cn10514149':'与原训练 cn10514259 同涉科技教育政策，待核实，保守排除',
 'cn10366586':'与原训练 cn10366590 同涉职业教育教学标准及解读',
 'cn10514575':'与原训练 cn10514204 涉杨家玉同一竞走夺冠',
 'cn10415656':'与原训练 cn10415727 涉阿拉纳同一就任事件',
 'cn10420061':'与原训练 cn10420052、cn10420093 涉同一枪杀事件',
 'cn10515881':'与原训练 cn10515839 涉林高远刘诗雯同一混双金牌',
 'cn10497183':'与原训练 cn10497437 涉同一微电影发布',
 'cn10528430':'与原训练 cn10528456 涉徐梦桃同一世界杯金牌',
 'cn10390219':'与原训练 cn10392648 涉同一群聊泄密事件',
 'cn10514453':'与原训练 cn10514635 涉张展硕同一游泳夺冠',
 'cn10368350':'与原训练 cn10367195 涉以防长同一停火威胁，待核实，保守排除',
 'cn10441324':'与原训练 cn10441224 涉特朗普同一停火表态',
 'cn10519300':'与原训练 cn10518751 涉苏炳添全运会最后一舞，保守排除',
 'cn10533905':'与新增 cn10533676 均涉及澳总理加强枪支管制，保守排除',
 'cn10536102':'与新增 cn10533676 均涉及澳枪击后的政府回应，保守排除',
}
# Each listed title has been checked for a clear education subject, without relabeling.
whitelist=set('''cn10351856 cn10375323 cn10375968 cn10388202 cn10390608 cn10391348
cn10394135 cn10399055 cn10400166 cn10401883 cn10412851 cn10415935 cn10416754
cn10421604 cn10421745 cn10422660 cn10425803 cn10426949 cn10429009 cn10431972
cn10433356 cn10434872 cn10435239 cn10437815 cn10440020 cn10441926 cn10444293
cn10446909 cn10447149 cn10451948 cn10452213 cn10455907 cn10459021 cn10459349
cn10460226 cn10462526 cn10467176 cn10468031 cn10470030 cn10470760 cn10472038
cn10474782 cn10475263 cn10476436 cn10477080 cn10477990 cn10478127 cn10478607
cn10480320 cn10486696 cn10489905 cn10491217 cn10495614 cn10500316 cn10502136
cn10508461 cn10509870 cn10511323 cn10523069 cn10523636 cn10524437 cn10525107
cn10525558 cn10526327 cn10527636 cn10527948 cn10530254 cn10533661 cn10534181
cn10536380 cn10537161 cn10538523 cn10538797 cn10541868'''.split())
df=pd.read_csv(HERE/'f34_title_review_decisions_v3.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
for i,reason in exclude.items():
    assert (df.id==i).sum()==1,i
    df.loc[df.id==i,'semantic_review']='exclude_possible_same_event'
    df.loc[df.id==i,'review_note']=reason
cand=pd.read_csv(ROOT/'results/member_C/f3_recovery/analysis/candidate_manifest.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
extra=cand[(cand.category=='教育')&cand.id.isin(whitelist)&~cand.id.isin(df.id)].copy()
assert len(extra)==len(whitelist), (len(extra),len(whitelist))
spec=importlib.util.spec_from_file_location('event_screen',HERE/'event_screen.py')
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
extra=screen.audit.enrich(extra).reset_index(drop=True)
flags=set();log=[]
for split in ('train','val','test','old_queue'):
    ref=(screen.audit.read(ROOT/'dataset/splits'/f'{split}.csv',['id','url','title','publish_time'])
         if split!='old_queue' else df[['id','url','title','publish_time']])
    ref=screen.audit.enrich(ref).reset_index(drop=True);idx=screen.audit.NearIndex()
    for t in ref._title_key:idx.add(t)
    for _,row in extra.iterrows():
        for pos,score in screen.matches(idx,row['_title_key']):
            flags.add(row.id);log.append({'candidate_id':row.id,'reference_id':ref.iloc[pos].id,'reference_split':split,'jaccard':score})
idx=screen.audit.NearIndex()
for j,row in extra.iterrows():
    for pos,score in screen.matches(idx,row['_title_key']):
        flags.update((row.id,extra.iloc[pos].id));log.append({'candidate_id':row.id,'reference_id':extra.iloc[pos].id,'reference_split':'new_review_pool','jaccard':score})
    idx.add(row['_title_key'])
extra['month']=extra.publish_time.str[5:7].astype(int)
extra['rank']=extra.id.map(lambda x:sha256(('f3-review-v1-seed42:'+x).encode()).hexdigest())
extra['semantic_review']=extra.id.map(lambda x:'exclude_pending' if x in flags else 'title_review_pass')
extra['review_note']=extra.id.map(lambda x:'Jaccard 0.5 相似事件筛查命中' if x in flags else '本轮补充明确教育主题的标题审阅')
extra['possible_event_conflict']=extra.id.isin(flags).astype(str)
extra=extra[df.columns]
out=HERE/'f34_title_review_decisions_v4.csv'
if out.exists():raise FileExistsError(out)
pd.concat([df,extra],ignore_index=True).to_csv(out,index=False,encoding='utf-8-sig')
pd.DataFrame(log,columns=['candidate_id','reference_id','reference_split','jaccard']).to_csv(HERE/'f34_extra_education_screen.csv',index=False,encoding='utf-8-sig')
print('new title-reviewed:',len(extra),'eligible:',len(extra)-len(flags))
spec=importlib.util.spec_from_file_location('builder',HERE/'build_f3_a.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
builder.main('f3_a_f34_review_candidate_v4',out.name)

"""Summarize fixed F3-A word/character coverage with explicit cohort definitions."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
fpath=ROOT/'results/member_C/f5_error_analysis/feature_per_title.csv'
ppath=ROOT/'results/member_C/f3_analysis/f3_a_multiseed/prediction_comparison.csv'
files=['coverage_summary.csv','coverage_by_category.csv','coverage_audit.json','coverage_diagnostic.md']
for x in files:
    if (HERE/x).exists():raise FileExistsError(HERE/x)
features=pd.read_csv(fpath,encoding='utf-8-sig')
pred=pd.read_csv(ppath,encoding='utf-8-sig')
f=features[features.model=='F3-A'].copy()
assert len(f)==len(pred)==3184 and f.id.is_unique and pred.id.is_unique
q=f.merge(pred[['id','title','category','f2_error_count','f3_error_count']],on='id',suffixes=('_feature','_prediction'),validate='one_to_one')
assert (q.category_feature==q.category_prediction).all() and (q.title_feature==q.title_prediction).all()
assert (q.combined_nnz==q.word_selected_nnz+q.char_selected_nnz).all()
q['persistent']=(q.f2_error_count==5)&(q.f3_error_count==5)
q['stable_f3']=(q.f3_error_count==0)
q['stable_both']=(q.f2_error_count==0)&(q.f3_error_count==0)
assert int(q.persistent.sum())==548
q['both_zero']=(q.word_selected_nnz==0)&(q.char_selected_nnz==0)
q['word_only']=(q.word_selected_nnz>0)&(q.char_selected_nnz==0)
q['char_only']=(q.word_selected_nnz==0)&(q.char_selected_nnz>0)
q['both_nonzero']=(q.word_selected_nnz>0)&(q.char_selected_nnz>0)
assert (q[['both_zero','word_only','char_only','both_nonzero']].sum(axis=1)==1).all()
cohorts=[('persistent_both',q[q.persistent]),('stable_f3',q[q.stable_f3]),('stable_both',q[q.stable_both])]
def summarize(name,d):
    o={'cohort':name,'n':len(d)}
    for col in ['word_selected_nnz','char_selected_nnz','combined_nnz']:
        o[col+'_mean']=float(d[col].mean());o[col+'_median']=float(d[col].median())
    for col in ['both_zero','word_only','char_only','both_nonzero']:
        o[col]=int(d[col].sum());o[col+'_rate']=float(d[col].mean())
    for col in ['word_selected_nnz','char_selected_nnz']:
        o[col+'_zero']=int((d[col]==0).sum());o[col+'_zero_rate']=float((d[col]==0).mean())
    return o
summary=pd.DataFrame([summarize(n,d) for n,d in cohorts])
details=[]
for name,group in cohorts:
    for cat in ['社会','财经','文娱','体育','教育','国际']:
        x=group[group.category_feature==cat];row=summarize(name,x)
        row['category']=cat;row['share_of_cohort']=len(x)/len(group)
        row['share_of_validation']=len(x)/len(q)
        details.append(row)
cat_summary=pd.DataFrame(details)
summary.to_csv(HERE/'coverage_summary.csv',index=False,encoding='utf-8-sig')
cat_summary.to_csv(HERE/'coverage_by_category.csv',index=False,encoding='utf-8-sig')
meta={'seed_independent_F3_A_processor':'F3-A 5000 word + 5000 character saved processor, selected nonzero counts',
      'cohorts':{'persistent_both':'F2 and F3-A each misclassify all 5 seeds','stable_f3':'F3-A correctly predicts all 5 seeds','stable_both':'F2 and F3-A both correctly predict all 5 seeds'},
      'cohort_overlap':'stable_both is a subset of stable_f3; persistent_both is disjoint from both',
      'input_hashes':{'feature_per_title.csv':sha256(fpath.read_bytes()).hexdigest(),'prediction_comparison.csv':sha256(ppath.read_bytes()).hexdigest()},
      'train_or_test_evaluation':False}
(HERE/'coverage_audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
def row(n):return summary[summary.cohort==n].iloc[0]
p=row('persistent_both');a=row('stable_f3');b=row('stable_both')
lines=['# F.6 训练前分支覆盖诊断','',
       '使用 F3-A 既有 5,000 词 + 5,000 字处理器在固定验证集的已保存逐标题非零计数；所有数值是卡方筛选后的实际非零数量。持续错误是 F2、F3-A 两组各五次均错；稳定正确分别给出 F3-A 五次均对和两组均五次对两个口径。后者是前者的子集。','',
       '| 样本组 | 条数 | 词非零均值 / 中位数 | 字非零均值 / 中位数 | 总非零均值 / 中位数 | 两支均零 | 仅词非零 | 仅字非零 | 两支均非零 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for title,x in [('两组持续错误',p),('F3-A 稳定正确',a),('两组稳定正确',b)]:
    lines.append(f'| {title} | {int(x.n)} | {x.word_selected_nnz_mean:.2f} / {x.word_selected_nnz_median:.0f} | {x.char_selected_nnz_mean:.2f} / {x.char_selected_nnz_median:.0f} | {x.combined_nnz_mean:.2f} / {x.combined_nnz_median:.0f} | {int(x.both_zero)} | {int(x.word_only)} | {int(x.char_only)} | {int(x.both_nonzero)} |')
lines += ['',f'与 F3-A 稳定正确组比较，持续错误组词级平均少 {a.word_selected_nnz_mean-p.word_selected_nnz_mean:.2f} 个，字级平均少 {a.char_selected_nnz_mean-p.char_selected_nnz_mean:.2f} 个；字分支非零均值差更大。两组标题类别分布不同，因此该差值不能解释为分支容量不足的因果效应。','',
          '| 样本组 | 类别 | 数量 | 占组比例 | 词非零均值 | 字非零均值 | 全零 | 仅词 | 仅字 |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for title,name in [('两组持续错误','persistent_both'),('F3-A 稳定正确','stable_f3'),('两组稳定正确','stable_both')]:
    for cat in ['社会','财经','文娱']:
        x=cat_summary[(cat_summary.cohort==name)&(cat_summary.category==cat)].iloc[0]
        lines.append(f'| {title} | {cat} | {int(x.n)} | {x.share_of_cohort:.1%} | {x.word_selected_nnz_mean:.2f} | {x.char_selected_nnz_mean:.2f} | {int(x.both_zero)} | {int(x.word_only)} | {int(x.char_only)} |')
lines += ['', '现有诊断支持将分支维数分配作为可检验的方向：字级非零差异和仅词覆盖相对突出；但增加维度是否改善覆盖或验证性能未知，卡方选择与分支重分配同时变化，不能保证提升。应在固定总维度、固定数据和训练设置下对照。不要将全零和单分支覆盖数当成无误标签或模型因果解释。','',
          '完整六类统计、输入哈希与定义见 `coverage_by_category.csv`、`coverage_summary.csv` 和 `coverage_audit.json`。未读取测试集或训练模型。','']
(HERE/'coverage_diagnostic.md').write_text('\n'.join(lines),encoding='utf-8')
print(summary.to_string(index=False));print(cat_summary[cat_summary.category.isin(['社会','财经','文娱'])].to_string(index=False))

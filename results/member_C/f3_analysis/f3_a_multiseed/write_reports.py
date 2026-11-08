"""Produce seed-paired summaries and an evidence-grounded initial error analysis."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
F5=ROOT/'results/member_C/f5_error_analysis'
for path in (HERE/'summary.md',HERE/'error_analysis.md',HERE/'recurrent_errors.csv',
             HERE/'stable_transitions.csv',HERE/'category_error_counts.csv',F5/'initial_findings.md'):
    if path.exists():raise FileExistsError(path)
run=pd.read_csv(HERE/'summary.csv',encoding='utf-8-sig')
paired=pd.read_csv(HERE/'paired_deltas.csv',encoding='utf-8-sig')
per=pd.read_csv(HERE/'per_class_summary.csv',encoding='utf-8-sig')
data=pd.read_csv(HERE/'prediction_comparison.csv',encoding='utf-8-sig')
tr=pd.read_csv(HERE/'transitions_by_seed.csv',encoding='utf-8-sig')
audit=json.loads((HERE/'audit.json').read_text(encoding='utf-8'))
labels=['财经','体育','文娱','教育','社会','国际']
assert len(run)==10 and len(paired)==5 and len(data)==3184 and len(per)==12
assert set(run.seed)=={42,43,44,45,46}
f2=run[run.group=='F2'];f3=run[run.group=='F3-A']
means=lambda x:(x.mean(),x.std(ddof=1))
f2f,f2fs=means(f2.macro_f1);f3f,f3fs=means(f3.macro_f1)
f2a,f2as=means(f2.accuracy);f3a,f3as=means(f3.accuracy)
f2p=per[per.group=='F2'].set_index('category');f3p=per[per.group=='F3-A'].set_index('category')
delta=(f3p.f1_mean-f2p.f1_mean).reindex(labels)
assert abs(delta.mean()-paired.delta_macro_f1.mean())<1e-12
recurrent=data[data.f2_always_wrong|data.f3_always_wrong].copy()
stable=data[data.always_corrected|data.always_introduced].copy()
category=data.groupby('category')[['f2_always_wrong','f3_always_wrong','always_corrected','always_introduced']].sum().reindex(labels)
category['support']=data.category.value_counts().reindex(labels)
recurrent.to_csv(HERE/'recurrent_errors.csv',index=False,encoding='utf-8-sig')
stable.to_csv(HERE/'stable_transitions.csv',index=False,encoding='utf-8-sig')
category.rename_axis('category').reset_index().to_csv(HERE/'category_error_counts.csv',index=False,encoding='utf-8-sig')
soc=data[data.category=='社会']; fc=[f'f3_pred_{s}' for s in range(42,47)];bc=[f'f2_pred_{s}' for s in range(42,47)]
soc_fin=int((soc[fc]=='财经').all(axis=1).sum());soc_ent=int((soc[fc]=='文娱').all(axis=1).sum())
same_wrong=int(((data[bc+fc].nunique(axis=1)==1)&data.f2_always_wrong&data.f3_always_wrong).sum())
social_lengths=soc.assign(title_length=soc.title.str.len()).groupby('f3_error_count').title_length.agg(['count','mean','median'])
L=['# F3-A 五随机种子验证','',
   '固定训练数据 7,531 条（保留原 6,549 条、新增 982 条），固定验证集 3,184 条。seed 42 复用审核版既有结果，仅新增 43–46；所有模型为 F_word_char + BP，hidden=64、lr=0.0003、dropout=0.2、input_dropout=0，沿用 batch=64、weight_decay=0.0001、patience=10、min_delta=0.0001。','',
   '| Seed | F2 Macro-F1 | F3-A Macro-F1 | 差值 F3−F2 | F2 Accuracy | F3 Accuracy | F3 最佳 / 停止 epoch | F3 训练秒 |',
   '|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in paired.to_dict('records'):
    L.append(f'| {r["seed"]} | {r["macro_f1_F2"]:.5f} | {r["macro_f1_F3"]:.5f} | {r["delta_macro_f1"]:+.5f} | {r["accuracy_F2"]:.5f} | {r["accuracy_F3"]:.5f} | {r["best_epoch_F3"]} / {r["epochs_run_F3"]} | {r["train_time_seconds_F3"]:.2f} |')
L += ['',f'F2 Macro-F1：**{f2f:.5f} ± {f2fs:.5f}**；F3-A：**{f3f:.5f} ± {f3fs:.5f}**。Accuracy 分别为 **{f2a:.5f} ± {f2as:.5f}**、**{f3a:.5f} ± {f3as:.5f}**。所有 ± 均为五种子样本标准差（ddof=1）。配对 Macro-F1 差值的均值为 **{paired.delta_macro_f1.mean():+.5f}**，样本标准差 {paired.delta_macro_f1.std(ddof=1):.5f}；{int((paired.delta_macro_f1>0).sum())}/5 个配对种子为正。F3-A 的种子间 Macro-F1 标准差略高，不能称其波动更小。','',
      '模型输入均为 10,000 维、参数量 640,454。训练耗时是模型训练过程记录，不包含全部特征准备，且不同运行受机器负载影响。所有运行均按验证 Macro-F1 早停并恢复最佳 checkpoint。','',
      '## 六类别：五种子均值','',
      '| 类别 | F2 P | F2 R | F2 F1 | F3 P | F3 R | F3 F1 | F1 变化 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for cat in labels:
    a,b=f2p.loc[cat],f3p.loc[cat]
    L.append(f'| {cat} | {a.precision_mean:.5f} | {a.recall_mean:.5f} | {a.f1_mean:.5f} | {b.precision_mean:.5f} | {b.recall_mean:.5f} | {b.f1_mean:.5f} | {delta[cat]:+.5f} |')
L += ['',f'文娱 F1 均值变化 {delta["文娱"]:+.5f}，社会 {delta["社会"]:+.5f}，是本次 Macro-F1 增量的主要正向来源；国际 {delta["国际"]:+.5f}、体育 {delta["体育"]:+.5f}。Macro-F1 为六类 F1 的等权平均，故各类贡献可用该列变化除以 6。各种子各类 P/R/F1 在 `per_class.csv`，均值和样本标准差在 `per_class_summary.csv`。','',
      '## 社会类混淆与每种子变化','',
      '| Seed | F2 社会→财经 | F3 社会→财经 | F2 社会→文娱 | F3 社会→文娱 | F3 修正 | F3 引入 |',
      '|---:|---:|---:|---:|---:|---:|---:|']
for r in paired.to_dict('records'):
    t=tr[tr.seed==r['seed']].iloc[0]
    L.append(f'| {r["seed"]} | {r["society_to_finance_F2"]} | {r["society_to_finance_F3"]} | {r["society_to_entertainment_F2"]} | {r["society_to_entertainment_F3"]} | {t.corrected} | {t.introduced} |')
L += ['',f'社会→财经平均 {f2.society_to_finance.mean():.1f}→{f3.society_to_finance.mean():.1f}，社会→文娱平均 {f2.society_to_entertainment.mean():.1f}→{f3.society_to_entertainment.mean():.1f}；不支持“两个方向均稳定改善”的结论。上表的修正/引入指该 seed 上 F2 错 F3 对 / F2 对 F3 错的全部验证样本数，并非仅限社会类。完整混淆矩阵逐种子数据见 `confusion_by_seed.csv`。','',
      '## 复现与限制','',
      '复核训练数据及验证文件哈希；分别只用原训练集和正式扩充训练集独立重拟合特征工程，词表、IDF、两分支卡方分数及选择器与十个保存处理器一致。十个 checkpoint 均 strict 加载，重新计算的验证预测、各类指标、混淆矩阵与原记录一致；根据历史、min_delta 和 patience 重放均得到相同最佳轮及停止轮。审计结果见 `audit.json`。','',
      '五种子反映相同划分和数据下的训练随机性；重复使用固定验证集、非独立样本及扩充引起的主题与词表变化限制了结论，不能替代独立数据集泛化或构成显著性证明。未使用测试集指标。','',
      '```powershell',
      'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe train_bp.py --exp F_word_char --model-type bp --hidden 64 --lr 0.0003 --dropout 0.2 --input-dropout 0 --seeds 43,44,45,46 --train-file results/member_C/f3_analysis/f3_a_dataset_formal/train.csv --train-sha256 '+audit['train_sha256']+' --output-root results/member_C/f3_analysis/f3_a_multiseed/runs',
      '```','']
(HERE/'summary.md').write_text('\n'.join(L),encoding='utf-8')

def example(id,expected):
    row=data[data.id==id]
    assert len(row)==1
    r=row.iloc[0]
    assert bool(r[expected]),(id,expected)
    return f'| [{r.id}]({r.url}) | {r.title} | {r.category} | {r.f2_prediction_counts} | {r.f3_prediction_counts} |'

stable_examples=[('cn10546591','always_corrected'),('cn10555482','always_corrected'),('cn10586908','always_corrected'),
                 ('cn10594440','always_corrected'),('cn10600296','always_corrected'),
                 ('cn10547603','always_introduced'),('cn10560809','always_introduced'),('cn10589760','always_introduced')]
rec_examples=['cn10545163','cn10547297','cn10544262','cn10544460','cn10550448','cn10546870']
E=['# F.3 / F.2 五种子验证错误对照','',
   '事实以固定验证集标签和十次已复算预测为准，不改动标签。下文“始终错”指同一模型的五个种子均预测错误，五次错误预测可以属于不同类别；“始终修正/引入”要求一个模型五次均错而另一个模型五次均对。','',
   f'F2 始终错 {audit["f2_always_wrong"]} 条、F3-A 始终错 {audit["f3_always_wrong"]} 条；两者均始终错 {audit["both_always_wrong"]} 条，其中十次预测都落在同一个错误类别的有 {same_wrong} 条。严格意义的始终修正 {audit["always_corrected"]} 条、始终引入 {audit["always_introduced"]} 条；其余波动样本不计入这两项。','',
   '| 真实类别 | 验证数 | F2 五次错 | F3 五次错 | F3 始终修正 | F3 始终引入 |','|---|---:|---:|---:|---:|---:|']
for cat,r in category.iterrows():
    E.append(f'| {cat} | {int(r.support)} | {int(r.f2_always_wrong)} | {int(r.f3_always_wrong)} | {int(r.always_corrected)} | {int(r.always_introduced)} |')
E += ['',f'社会类 545 条中，F3-A 的 204 条在五个种子下均出错；其中 {soc_fin} 条五次均预测财经、{soc_ent} 条五次均预测文娱。这是样本交集，不能与单次混淆矩阵的平均次数相加解释。','',
      '## 五次均修正与五次均引入的例子','',
      '预测计数字典表示五个种子各类别出现次数；标题是可查事实，原因说明是基于标题的假设。','',
      '| ID | 标题 | 验证标签 | F2 预测次数 | F3-A 预测次数 |','|---|---|---|---|---|']
for id,status in stable_examples:E.append(example(id,status))
E += ['', '例如足球青训、校园文化节及国际商事调解学院都跨越两个类别的常见主题；扩充后预测方向改变，但仅凭标题和预测无法确定是哪些新增样本或词影响了结果。商场争议、校园运动设施等反例说明修正不是单向的。','',
      '## 社会类跨主题的持续错误','',
      '| ID | 标题 | 验证标签 | F2 预测次数 | F3-A 预测次数 |','|---|---|---|---|---|']
for id in rec_examples:
    r=data[data.id==id].iloc[0];assert r.category=='社会' and r.f2_always_wrong and r.f3_always_wrong
    E.append(f'| [{r.id}]({r.url}) | {r.title} | 社会 | {r.f2_prediction_counts} | {r.f3_prediction_counts} |')
E += ['',
      '产业、市场、快递业务量等标题可以同时关联社会运行和财经；博物馆、戏曲、非遗又可以同时关联公共生活和文娱。这些标题中可观察到主题交叉，可能让模型倾向主题词强的相邻类别；这只是基于标题的推断，不能据此认定既有标签错误，也不能断言模型实际使用了某个词。','',
      '探索性统计：在社会标签且 F3 五次均预测财经的 80 条中，21 条包含“市场、产业、企业、经济、业务、亿元、消费、金融、投资、贸易”之一；社会标签且五次均预测文娱的 71 条中，28 条包含“文化、文博、博物馆、非遗、戏曲、戏、书、电影、文艺、音乐、展览”之一。作为参照，社会标签且五次均预测正确的 307 条中，分别有 8 条和 5 条命中这两组词。词表根据本次错误观察事后制定，按标题字符串匹配，仅用于描述，不是预注册检验或特征重要性结论；完整定义及计数见 `title_keyword_audit.csv`。','',
      f'社会类 F3 五次全对的 {int(social_lengths.loc[0,"count"])} 条标题平均 {social_lengths.loc[0,"mean"]:.2f} 字符、中位数 {social_lengths.loc[0,"median"]:.0f}；五次全错的 {int(social_lengths.loc[5,"count"])} 条平均 {social_lengths.loc[5,"mean"]:.2f} 字符、中位数 {social_lengths.loc[5,"median"]:.0f}。这是包含标点的标题字符数，未控制主题和难度；本组数据不足以支持“短标题是主要原因”。','',
      '标题未给出正文背景时可能存在类别信息不足；但是否信息不足、原栏目是否合理、训练分布造成多少影响，均不能仅由这十组预测确定。后续须在保留原标签的前提下按统一标准复核样本，区分类别边界、标题信息和模型问题。','',
      '完整样本见 `recurrent_errors.csv`、`stable_transitions.csv` 和 `prediction_comparison.csv`；它们都是固定验证集的派生分析，不包含测试集指标。','']
(HERE/'error_analysis.md').write_text('\n'.join(E),encoding='utf-8')
F5.mkdir(parents=True,exist_ok=True)
P=['# F.5 前置错误归因：按当前证据排序','',
   '1. **高：类别混淆集中在社会类。** F3-A 社会类五种子平均 F1 为 '+f'{f3p.loc["社会","f1_mean"]:.5f}'+f'，545 条中 {int(category.loc["社会","f3_always_wrong"])} 条五次均错；社会→财经每次平均 {f3.society_to_finance.mean():.1f} 条，社会→文娱 {f3.society_to_entertainment.mean():.1f} 条。可优先建立人工审核清单，明确业务、文化和公共生活题材的类别边界。','',
   '2. **高：扩充的收益并非各类别或各样本一致。** 五配对种子均提升，平均 Macro-F1 '+f'{paired.delta_macro_f1.mean():+.5f}'+f'；文娱 F1 {delta["文娱"]:+.5f}、社会 {delta["社会"]:+.5f}，但社会→财经平均由 {f2.society_to_finance.mean():.1f} 增至 {f3.society_to_finance.mean():.1f}。严格五次全错→全对为 {audit["always_corrected"]} 条，五次全对→全错为 {audit["always_introduced"]} 条。建议比较这两组标题和训练样本的主题分布，避免只分析成功案例。','',
   '3. **中：标题主题交叉是可观察现象，具体机制尚待验证。** 社会标签但五次均预测财经的例子包括“科技促产业崛起 广西罗汉果从桂林山野走向欧美市场”；五次均预测文娱的例子包括“文博启新元 中国多地民众走进博物馆共度跨年夜”。这些是类别边界审查对象，不是自动改标依据。','',
   '   事后选取的描述性主题词中，五次均预测财经的社会标题有 21/80 命中财经词；五次均预测文娱的社会标题有 28/71 命中文娱词。社会类五次均正确的 307 条中分别为 8 条和 5 条。词表由本次错误观察形成，仅提示值得复核的模式，不能视作模型解释或显著性证据；定义见 `../f3_analysis/f3_a_multiseed/title_keyword_audit.csv`。','',
   '4. **待验证：标题信息不足及训练样本覆盖。** 仅凭当前标题和输出无法确定缺失正文、特定词、标注口径或训练分布各自的因果作用。社会类五次全对和全错标题的字符长度中位数均为 '+f'{social_lengths.loc[0,"median"]:.0f}'+ '，不足以把错误归咎于标题短。建议人工分层核对真实类别、预测类别、歧义原因和训练集近邻，仅作研究分析，不改变固定验证集标签。','',
   '五种子共用同一固定验证集，不能代替独立测试集泛化或支撑显著性结论。F.4 线性模型比较也不能证明 BP 的性能上限。是否开展新方案由研究负责人决定；本阶段未实施新特征或网络。','',
   '样本与可复算数据见 `../f3_analysis/f3_a_multiseed/error_analysis.md`、`../f3_analysis/f3_a_multiseed/recurrent_errors.csv` 和 `../f3_analysis/f3_a_multiseed/prediction_comparison.csv`；完整实验配置及统计见同目录 `summary.md`。','']
(F5/'initial_findings.md').write_text('\n'.join(P),encoding='utf-8')
print('reports written; F3 mean/std',f3f,f3fs,'paired mean',paired.delta_macro_f1.mean())

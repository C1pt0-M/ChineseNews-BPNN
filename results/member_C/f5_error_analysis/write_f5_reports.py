"""Create evidence-bound F.5 reports from the recorded audit tables."""
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
for f in ('summary.md','feature_diagnostics.md','next_experiments.md','audit_manifest.json'):
    if (HERE/f).exists():raise FileExistsError(HERE/f)
read=lambda x:pd.read_csv(HERE/x,encoding='utf-8-sig')
groups=json.loads((HERE/'group_audit.json').read_text(encoding='utf-8'))
meta=json.loads((HERE/'feature_audit.json').read_text(encoding='utf-8'))
stats=read('error_group_statistics.csv');direction=read('group_direction_statistics.csv')
sample=read('manual_review.csv');f=read('feature_per_title.csv');near=read('nearest_training.csv')
cov=read('feature_group_summary.csv');cent=read('class_centroid_similarity.csv')
traincov=read('train_val_coverage.csv');topics=read('added_training_topics.csv')
pred=pd.read_csv(ROOT/'results/member_C/f3_analysis/f3_a_multiseed/prediction_comparison.csv',encoding='utf-8-sig')
assert len(sample)==152 and sample.id.is_unique and len(pred)==3184
assert int((sample.official_body_chars<60).sum())==4
assert set(sample.review_group.value_counts().items())=={('持续错误',60),('稳定修正',30),('稳定退化',32),('种子不稳定',30)}
labels=['财经','体育','文娱','教育','社会','国际']
cohorts=['持续错误','稳定修正','稳定退化','种子不稳定']
row=lambda c:cov[(cov.model=='F3-A')&(cov.cohort==c)].iloc[0]
hard=row('persistent_both');good=row('model_all_correct')
cohort=pred[['id','f2_always_wrong','f3_always_wrong','f3_error_count','category']]
near3=near[(near.model=='F3-A')&(near.role=='nearest_any')].merge(cohort,on='id',validate='one_to_one')
hardnear=near3[near3.f2_always_wrong & near3.f3_always_wrong]
goodnear=near3[near3.f3_error_count==0]
def cross(n):
    n=n[n.train_id.notna()]
    return int((n.train_category!=n.true_category).sum()),len(n)
hc,hn=cross(hardnear);gc,gn=cross(goodnear)
assert (hc,hn,gc,gn)==(448,539,499,2433)
f3f=f[f.model=='F3-A'];social=f3f[f3f.category=='社会']
social_h=social[social.persistent_both];social_g=social[social.model_all_correct]
assert len(social_h)==177 and len(social_g)==307
def example(id):
    a=sample[sample.id==id];assert len(a)==1
    return a.iloc[0]
e1=example('cn10623553');e2=example('cn10577853');e3=example('cn10633625')
assert e1.nearest_train_id=='cn10416785' and e2.nearest_train_id=='cn10373450' and e3.combined_nnz==0

S=['# F.5 错误归因与性能瓶颈分析','',
   '本报告只分析已保存的 F2/F3-A 五种子固定验证预测，未训练模型或改动标签。验证集 3,184 条；F3-A 五种子 Macro-F1 为 0.78734 ± 0.00165，F2 为 0.78369 ± 0.00132。此前五种子重算与数据哈希核对见 `../f3_analysis/f3_a_multiseed/audit.json`。','',
   '## 结果要点','',
   f'1. 两组模型五种子均错的交集为 **548/3,184（{548/3184:.2%}）**；严格五次全错→全对的稳定修正 **30（{30/3184:.2%}）**，五次全对→全错的稳定退化 **32（{32/3184:.2%}）**。至少一组内部五种子预测类别不同的样本有 **288（{288/3184:.2%}）**；该组与前三组可重叠，不能相加。',
   f'2. 持续错误中社会类有 **177/548（{177/548:.1%}）**，而社会类仅占整个验证集 545/3,184（{545/3184:.1%}）。按两组各自五次预测的众数，社会→财经有 65 条、社会→文娱有 64 条；这是样本数而非单模型混淆矩阵的五次总和。',
   f'3. F3-A 持续错误组每条标题平均保留 **{hard.combined_nnz_mean:.2f}** 个非零词字特征，F3-A 五次全对组为 **{good.combined_nnz_mean:.2f}**。二者差异是相关现象，不能单独证明原因。',
   f'4. F3-A 最近训练邻居存在非零相似度的样本中，持续错误组 **{hc}/{hn}（{hc/hn:.1%}）** 的最近邻类别不同于验证标签；五次全对组为 **{gc}/{gn}（{gc/gn:.1%}）**。9 个持续错误标题没有任何非零训练邻居，已从该比例分母排除。',
   '5. 新增样本主题并非均匀：文娱新增 170 条中，分词特征“电影”出现在 29 条；社会新增 159 条中“警方”31 条、“诈骗”22 条。主题覆盖改变与结果变化有关联，但不能据此认定增益的来源。','',
   '## 四组样本与类别分布','',
   '| 验证标签 | 持续错误 | 稳定修正 | 稳定退化 | 种子不稳定 |','|---|---:|---:|---:|---:|']
for cat in labels:
    S.append('| '+cat+' | '+' | '.join(str(int(stats[(stats.group==g)&(stats.category==cat)]['count'].iloc[0])) for g in cohorts)+' |')
S+=['| 合计 | 548 | 30 | 32 | 288 |','',
    '组定义：持续错误是两组各自五次都错；稳定修正是 F2 五次错、F3 五次对；稳定退化相反；种子不稳定是任一组内部五个预测类别不全相同。按此定义，不稳定组与持续错误、稳定修正、稳定退化分别重叠 52、3、1 条。`error_groups.csv` 是一条样本可出现多行的长表；各组类别和占完整验证集比例见 `error_group_statistics.csv`，各组真实类别、两组众数方向及比例见 `group_direction_statistics.csv`。众数平票按 seed 顺序第一次出现的类别处理，原五次投票均保留。','',
    '## 审核与可查例子','',
    '固定种子 42 的哈希排序，持续错误定向覆盖社会→财经/文娱及反向混淆等共 60 条，30 条稳定修正和 32 条稳定退化全部纳入，另按类别均衡抽 30 条与前三组审核样本不重复的不稳定样本。该审核集有目的地过采样混淆方向，不能作为总体随机样本估计标签错误率。','',
    '152 条的官方文章页面与标题均可对应，逐条 URL、页面哈希、原始栏目、预测投票、首段线索、内容审核、词字覆盖和相似训练标题保存在 `manual_review.csv`。其中 69 条记录了人工识别的跨主题线索；其余未标记不表示不存在交叉。4 条页面提取到的是海报、图文标题或来源说明，文本不足以判断正文与类别，已标记需人工进一步确认。文章首段与标题不矛盾不等于真实标签经过独立裁决；没有修改标签。','',
    '| 验证标题（标签 / F3 五次预测） | 最近训练标题（标签） | 余弦 |','|---|---|---:|',
    f'| [{e1.id}]({e1.official_url}) {e1.title}（{e1.validation_label} / {e1.f3_votes}） | [{e1.nearest_train_id}]({e1.nearest_train_url}) {e1.nearest_train_title}（{e1.nearest_train_category}） | {e1.nearest_cosine:.3f} |',
    f'| [{e2.id}]({e2.official_url}) {e2.title}（{e2.validation_label} / {e2.f3_votes}） | [{e2.nearest_train_id}]({e2.nearest_train_url}) {e2.nearest_train_title}（{e2.nearest_train_category}） | {e2.nearest_cosine:.3f} |','',
    '以上展示了公共活动/文化活动、旅客运输/经营统计之间可观察的主题交叉；模型为何选择该类别仍是推断。余弦邻居是特征空间的描述，不是模型逐例的解释，也不自动代表同一事件或数据泄漏。','',
    '## 风险与交付','',
    '训练来自 2025 年，验证来自 2026 年上半年；不同时间与事件本身可能造成表征变化。原 6,549 条训练新闻内部历史近重复和栏目冲突继续保留。相似度、关键词分布和前后模型比较均为描述性关联；未设计验证集专用规则，未更改真实标签，也未读取测试集模型指标。','',
    '详细方法和覆盖分析见 `feature_diagnostics.md`，2–3 个尚未实现的公平对照方案见 `next_experiments.md`；原始预测分组、审核记录及最近训练样本见同目录 CSV。','']
(HERE/'summary.md').write_text('\n'.join(S),encoding='utf-8')

trainall=traincov[(traincov.model=='F3-A')&(traincov.category=='全部')].set_index('split')
def sim(a,b):return float(cent[(cent.model=='F3-A')&(cent.category==a)&(cent.other_category==b)].cosine.iloc[0])
D=['# F.5 特征表达、训练覆盖与近邻诊断','',
   '以已保存的 seed 42 F2 与 F3-A 特征处理器变换各自训练集和同一验证集，不调用 fit；此前已核对每组五个种子处理器一致。词级沿用 jieba 及词 1–2 gram，字级使用字符 2–4 gram，各分支各选择 5,000 维并拼接后行 L2 归一化。所有矩阵为稀疏矩阵。','',
   '## 活跃特征与未见组合','',
   '| F3-A 验证样本组 | 条数 | 词级非零均值 | 字级非零均值 | 总非零均值 | 总非零中位数 | 词/词对从未见过比例均值 | 字符组合从未见过比例均值 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for label,z in [('两组均五次错',hard),('F3 五次全对',good)]:
    D.append(f'| {label} | {int(z.n)} | {z.word_selected_nnz_mean:.2f} | {z.char_selected_nnz_mean:.2f} | {z.combined_nnz_mean:.2f} | {z.combined_nnz_median:.0f} | {z.word_never_seen_fraction_mean:.1%} | {z.char_never_seen_fraction_mean:.1%} |')
D+=['','“从未见过”按每条标题生成的**不重复**词 1–2 gram 或字符 2–4 gram，判断是否在对应训练集任一原始标题中出现，再对标题比例取均值；不等于单字或单词的未登录率。`feature_per_title.csv` 另记“未进入已拟合词表”的比例，后者还包括低频/高频过滤和词表容量限制，不应与真正训练未见混为一谈。被卡方筛除的已知特征也不会计入非零数。','',
    f'社会类内部结果提供了反例：持续错误 {len(social_h)} 条的词组合/字符组合未见比例均值为 {social_h.word_never_seen_fraction.mean():.1%} / {social_h.char_never_seen_fraction.mean():.1%}；F3 五次全对 {len(social_g)} 条为 {social_g.word_never_seen_fraction.mean():.1%} / {social_g.char_never_seen_fraction.mean():.1%}。因此不能概括成“社会类错误主要由未见组合造成”。','',
    '社会类内部持续错误和 F3 五次全对标题的非零总特征均值分别为 10.24 和 12.33，仍有描述性差异。F2 对同一持续错误组和 F2 五次全对组的非零总特征均值分别为 10.47 和 14.23；两组比较并未控制标题长度、主题和其他差异。','',
    f'F3-A 全体验证标题中 {(f3f.combined_nnz==0).sum()}/3,184 的拼接后向量全零，其中 {(f3f[f3f.persistent_both].combined_nnz==0).sum()}/548 属持续错误；全零没有非零近邻。F3-A 训练标题平均保留 {trainall.loc["train","combined_nnz_mean"]:.2f} 个特征，验证标题平均 {trainall.loc["validation","combined_nnz_mean"]:.2f}；各类及词、字分支见 `train_val_coverage.csv`。','',
    f'对相同已选择特征，将训练与验证平均 TF-IDF 权重分别做 L1 归一化，其 Jensen–Shannon 散度为 F2 **{meta["F2"]["train_val_selected_mean_weight_js_divergence_bits"]:.4f} bit**、F3-A **{meta["F3-A"]["train_val_selected_mean_weight_js_divergence_bits"]:.4f} bit**。这是时间不同、选择器也不同的描述指标，没有给定“明显偏移”的阈值，不能直接把差值解释为性能原因。','',
    '## 类别特征重叠与训练邻居','',
    f'F3-A 训练集类别中心为各训练标题最终稀疏向量的均值再归一化；社会/财经中心余弦 **{sim("社会","财经"):.3f}**，社会/文娱 **{sim("社会","文娱"):.3f}**，财经/文娱 **{sim("财经","文娱"):.3f}**。前两者是六类中心中最高的两对，支持在该表示下进一步审查类别交叉，但不是类别混淆的因果证明。完整 6×6 结果见 `class_centroid_similarity.csv`。','',
    f'最近邻是当前保存的词字融合特征上的余弦，不参与分类。持续错误组有非零近邻的 {hn} 条中，{hc} 条最近邻与验证标签不同类；F3 五次全对组为 {gc}/{gn}。仅看社会标签，持续错误 177 条中 141 条最近邻不同类；五次全对且有非零近邻 305 条中 77 条不同类。按各标题的有效非零特征数解释相似度，不能以此自动重标。','',
    f'- 社会验证标题“{e1.title}”的最近训练标题是文娱类“{e1.nearest_train_title}”，余弦 {e1.nearest_cosine:.3f}，两篇分别描述不同地区、不同年度的博物馆日活动。',
    f'- 社会验证标题“{e2.title}”的最近训练标题是财经类“{e2.nearest_train_title}”，余弦 {e2.nearest_cosine:.3f}，两者均涉及春运旅客数量。',
    '- 财经验证标题 [cn10590961](https://www.chinanews.com/cj/2026/03-22/10590961.shtml)“‘阿尔卑斯’矿泉水实则来自吉林长白山？”与体育训练标题 [cn10504364](https://www.chinanews.com/ty/2025/10-25/10504364.shtml)“吉林雪场迎新雪季‘首滑’”余弦为 1.000，但验证标题只剩 1 个词级非零、0 个字级非零。这是特征投影后高度相似，并不表示原新闻相似或模型因某个词作出决定。',
    f'- 文娱验证标题 [{e3.id}]({e3.official_url})“{e3.title}”在 F3-A 的 10,000 维中无非零特征，不能据此构造有意义的最近邻。','',
    '所有标题的最近任意、同类、不同类训练标题、ID、栏目、URL 和相似度见 `nearest_training.csv`；152 条重点审核样本的对应证据在 `manual_review.csv`。','',
    '## 训练集规模、月份与主题','',
    '| 类别 | 原训练数 | 新增数 | F3-A 总数 |','|---|---:|---:|---:|']
counts={'财经':(1109,166),'体育':(1107,166),'文娱':(1134,170),'教育':(1064,160),'社会':(1057,159),'国际':(1078,161)}
for cat in labels:
    a,b=counts[cat];D.append(f'| {cat} | {a} | {b} | {a+b} |')
D+=['','各组各类 1–12 月完整数量见 `training_month_distribution.csv`。例如社会类 1 月由 104 增至 111、3 月由 82 增至 88；教育类新增按冻结数据清单记录。训练窗口未改变。','',
    '在 F3-A 的已选词级特征中，文娱新增标题含分词项“电影”29/170、原文娱训练为 60/1,134；“博物馆”19/170 对 38/1,134。社会新增标题含“警方”31/159 对原社会 16/1,057，“诈骗”22/159 对 3/1,057。财经新增含“外贸”30/166 对原财经 12/1,109。完整类别 Top 25 和分母见 `added_training_topics.csv`。这些是原始/新增样本中的词项文档频率，不是因果归因或模型决策权重。','',
    '文娱 F1 的增加可能与新样本主题及词表变化相关；社会新增样本更多覆盖警方、诈骗和救援，而社会→财经的错分还涉及运输、消费与产业等主题。这是可检验假设：样本是定向审核后选出的，训练分布、IDF、卡方筛选、模型拟合同时变化，现有证据不能分离其作用，也不能保证增加任一主题必然改善。','',
    '## 复现与界限','',
    '从项目根目录执行时，脚本路径均带前缀 `results/member_C/f5_error_analysis/`：先运行 `python results/member_C/f5_error_analysis/build_groups.py`；再运行同目录的 `fetch_review_articles.py`、`extract_review_sources.py`、`feature_audit.py`、`train_val_coverage.py`、`group_directions.py`、`build_manual_review.py` 和 `write_f5_reports.py`。本次使用 `D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe`，脚本拒绝覆盖已有结果；复现时应在独立副本或新输出位置运行。所有处理器只调用 transform。全零样本不输出虚构邻居；近邻同分按训练行顺序确定。','']
(HERE/'feature_diagnostics.md').write_text('\n'.join(D),encoding='utf-8')

N=['# F.5 后续候选对照（仅建议，尚未实现）','',
   '这些方案针对现有分析中可观察的模式提出，不承诺达到 0.85。固定验证集已多次用于研发，后续解释须考虑反复选择偏差；测试集保持封存直到研究设计和最终方案固定。','',
   '## 1. 控制主题覆盖的训练数据对照','',
   '- **问题与依据**：持续错误的最近训练邻居常跨类；社会新增样本偏向警方、诈骗等，社会→财经的平均误分没有下降。现有词字 TF-IDF + BP 无法自动补充未覆盖的、语义接近但类别不同的训练案例。',
   '- **调整**：训练数据策略，保留 BP 和现有特征。只从合法 2025 年候选池按既定规则核验，规划覆盖公共服务、交通、消费等可复核主题，同时保留原 6,549 条和原标签；不为个别验证样本构造补丁。',
   '- **公平对照**：原 F3-A 与候选方案都保持 7,531 条、六类配额和尽可能相同的月份分布；固定采样方案、五配对种子、重新仅用各自训练集拟合特征。若主题调整需要替换新增样本，提前冻结清单并记录近重复审查。',
   '- **资源与方向**：需要额外候选新闻核验与人工类别审查；计算量接近现有多种子 BP，仍属于纯 TF-IDF + BP 研究。结果同时受样本内容变化影响，不能单凭观察声称因果。','',
   '## 2. 对照词字特征覆盖','',
   '- **问题与依据**：持续错误标题有效特征较少，验证集中有 11 条最终向量全零；未见组合并非社会错误的充分解释。现有 TF-IDF 的词表过滤、卡方筛选及固定 n-gram 可能丢失标题中的其他可用线索。',
   '- **调整**：仅提出独立的可选特征对照，例如先单独比较字级加入 1-gram，或单独比较字词低频门槛；每次只改变一个因素，原 F_word_char 和成员 B 模块保持不变。',
   '- **公平对照**：同一冻结 F3-A、同一 BP hidden/lr/训练策略、同一 10,000 输入维度及分支配额、五配对种子；只在训练集拟合词表/IDF/筛选，报告非零覆盖、类别 F1、参数量及耗时。卡方仍可能筛掉新增特征，不能保证消除零向量。',
   '- **资源与方向**：不需额外训练新闻或预训练模型，但候选词表可能增大内存和处理时间；仍是纯 TF-IDF + BP。','',
   '## 3. 冻结式中文预训练表示作为独立扩展','',
   '- **问题与依据**：部分标题共享文化、财经和社会主题，而稀疏词面近邻跨类；TF-IDF 不主动编码上下文语义。预训练表示可能提供不同的语义信息，但现有数据尚未证明其能解决栏目边界。',
   '- **调整**：特征表示增加外部中文预训练编码器，冻结编码器输出后训练相同 BP 分类头，作为独立扩展；它引入**外部预训练知识**，不属于纯 TF-IDF + BP 改进。',
   '- **公平对照**：原训练/验证划分和五种子不变，分别报告 TF-IDF + BP 与冻结编码器 + BP；列明预训练模型版本、语料和时间、可疑数据污染、编码器及 BP 头分别的参数、特征维数、耗时和计算资源。预先固定模型与实验计划，不能用测试集选方案。',
   '- **资源与方向**：需要经过批准的外部模型文件、下载及额外算力/存储；与原研究方向的纯 TF-IDF 口径分开报告。如预训练语料或时间无法审计，应明确局限。','',
   '以上均为下一阶段候选，尚未修改特征、模型、训练配置或数据划分。','']
(HERE/'next_experiments.md').write_text('\n'.join(N),encoding='utf-8')
tracked=['summary.md','feature_diagnostics.md','next_experiments.md','error_groups.csv','manual_review.csv',
         'review_queue.csv','review_source_extract.csv','feature_per_title.csv','nearest_training.csv',
         'error_group_statistics.csv','group_direction_statistics.csv','training_month_distribution.csv',
         'feature_group_summary.csv','class_centroid_similarity.csv','added_training_topics.csv','train_val_coverage.csv']
manifest={'inputs':{'train_sha256':meta['F2']['train_rows'],'f3_train_manifest_sha256':sha256((ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal/manifest.json').read_bytes()).hexdigest(),
                    'val_sha256':sha256((ROOT/'dataset/splits/val.csv').read_bytes()).hexdigest()},
          'source_page_count':152,'source_page_hashes_in':'review_source_extract.csv',
          'outputs_sha256':{n:sha256((HERE/n).read_bytes()).hexdigest() for n in tracked},
          'no_training_or_fit':True,'test_metrics_read':False,'review_sampling_seed':42}
manifest['inputs']['train_sha256']=sha256((ROOT/'dataset/splits/train.csv').read_bytes()).hexdigest()
(HERE/'audit_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('F5 reports written; group and nearest totals verified.')

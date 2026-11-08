"""Produce the F3-A evidence ledger and comparison from verified local artifacts."""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = ['财经','体育','文娱','教育','社会','国际']


def main():
    report = HERE/'f3_a_report.md'
    if report.exists():
        raise FileExistsError(report)
    evidence = pd.read_csv(HERE/'f3_official_evidence_education.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    review = pd.read_csv(HERE/'f3_title_review_decisions.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    edu_review = review[review.category=='教育'][['id','semantic_review','review_note','possible_event_conflict']]
    edu = evidence.merge(edu_review,on='id',how='left',validate='one_to_one').fillna('')
    edu['title_review_status'] = edu.semantic_review.replace('', 'not_reviewed')
    new = pd.read_csv(HERE/'f3_a_dataset/new_sample_evidence.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    edu['included_f3_a'] = edu.id.isin(set(new.id))
    edu.drop(columns=['semantic_review','category_semantic_approval']).to_csv(HERE/'f3_education_final_review.csv',index=False,encoding='utf-8-sig')
    pending = review[review.semantic_review!='title_review_pass'][['id','title','category','url','publish_time','review_note']].copy()
    remaining = edu[edu.title_review_status=='not_reviewed'].copy()
    remaining['review_note'] = '未进入本轮固定标题审阅队列；类别语义待审核'
    remaining = remaining.rename(columns={'article_url':'url'})[['id','title','category','url','publish_time','review_note']]
    pd.concat([pending,remaining],ignore_index=True).to_csv(HERE/'f3_manual_review_needed.csv',index=False,encoding='utf-8-sig')
    v = json.loads((HERE/'f3_a_verification.json').read_text(encoding='utf-8'))
    meta = json.loads((HERE/'f3_a_dataset/manifest.json').read_text(encoding='utf-8'))
    classes = pd.read_csv(HERE/'f3_a_per_class.csv',encoding='utf-8-sig')
    f2, f3 = v['F2'], v['F3-A']
    lines = ['# F.3 官方归档证据与 F3-A 验证实验（初版，训练后发现质量问题）', '',
             '> 更正：本报告的 F3-A 为初版数据和已执行的一次训练，训练后发现 5 条与基准可能描述同一事件、另有 2 条可能属于同一数据发布；该运行仅供诊断。修正候选见 f3_a_dataset_v3，尚未训练。以 f3_official_review_report.md 为准。', '',
             '## 训练准入与证据等级', '',
             '371 条教育候选的时间与来源证据：A 级 **129**，B 级 **242**，未核实 **0**。A 为原文索引时间匹配，B 为重新取得的官方日期归档页逐字段匹配。每条证据的官方链接、原始栏目、标题、分钟及页面 SHA-256 见 `f3_education_final_review.csv`。',
             '', '365 个 2025 年官方归档页已独立重新获取并保存为 `official_archive_recheck/*.html.gz`，对应 JSON 保留 URL、检索时间和页面哈希；本次不将旧解析切片当作独立重核。标题一致采用项目原有空白及不可见字符清理，不接受文章 ID/URL、栏目、日期或分钟冲突。',
             '', '各类均从固定种子、按基准月份分布抽取的 220 条标题队列开展标题级审阅；未知或跨主题标题排除。先保留原规则 Jaccard ≥0.8 去重，并对候选与现有划分及审阅队列额外采用 ≥0.5 的保守筛查；显式识别的相关事件也排除。**下表是本轮已审阅池的合格数量，不是整个 90,742 条候选池的完整合格总数**。标题级审阅不是完整原文事实审核。',
             '', '| 类别 | 本轮审阅池通过 | F3-A 新增 |', '|---|---:|---:|']
    review_pass=review[review.semantic_review=='title_review_pass']
    for label in LABELS:
        lines.append(f'| {label} | {(review_pass.category==label).sum()} | {meta["added_counts"][label]} |')
    lines += ['', '教育候选中 220 条进入本轮标题审阅，168 条通过、52 条待复核；其余 151 条尚未进入标题审阅。F3-A 教育新增 160 条，其中 A 级 51 条、B 级 109 条。所有 982 条新增记录均有重新取得的官方归档逐字段匹配。',
              '', '## 数据与实验配置', '',
              '| 配置 | 训练条数 | 财经 | 体育 | 文娱 | 教育 | 社会 | 国际 |', '|---|---:|---:|---:|---:|---:|---:|---:|',
              '| F2 | 6549 | 1109 | 1107 | 1134 | 1064 | 1057 | 1078 |',
              '| F3-A | 7531 | 1275 | 1273 | 1304 | 1224 | 1216 | 1239 |',
              '', 'F3-A 保留原始训练集全部 6,549 行，追加 982 行。按类别和月份固定 seed 42 哈希排序，以原训练月份分布分配；因通过审阅的池受限，教育、社会月份出现偏差。例如教育 6 月实际新增 18、理想 20，8 月 14、理想 18；社会 1 月 7、理想 16，3 月 6、理想 12。完整逐月数量及哈希见 `f3_a_dataset/manifest.json`。',
              '', '两个运行均使用 F_word_char、BP hidden=64、lr=0.0003、dropout=0.2、input_dropout=0、seed=42，验证集不变；各模型 10,000 维、640,454 参数。对两个训练集独立重新拟合并逐项核对词表、IDF 和卡方选择器。模型按验证 Macro-F1、min_delta=0.0001、patience=10 选取与早停。',
              '', '## 验证集对照', '', '| 配置 | Macro-F1 | Accuracy | 最佳 epoch | 总 epoch | 训练时间（秒） |', '|---|---:|---:|---:|---:|---:|']
    for name in ('F2','F3-A'):
        x=v[name]; lines.append(f'| {name} | {x["macro_f1"]:.5f} | {x["accuracy"]:.5f} | {x["best_epoch"]} | {x["epochs_run"]} | {x["train_time_seconds"]:.2f} |')
    lines += ['', f'F3-A 相对 F2 的 Macro-F1 变化为 **{v["delta_macro_f1"]:+.5f}**。这只是同一验证集上的单随机种子结果，不能判断随机波动或测试集泛化。',
              '', '| 类别 | F2 Precision | F2 Recall | F2 F1 | F3-A Precision | F3-A Recall | F3-A F1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for cat in LABELS:
        x=classes[(classes.model=='F2') & (classes.category==cat)].iloc[0]
        y=classes[(classes.model=='F3-A') & (classes.category==cat)].iloc[0]
        lines.append(f'| {cat} | {x.precision:.5f} | {x.recall:.5f} | {x.f1:.5f} | {y.precision:.5f} | {y.recall:.5f} | {y.f1:.5f} |')
    lines += ['', '| 误分类方向 | F2 | F3-A |', '|---|---:|---:|',
              f'| 社会→财经 | {f2["society_to_finance"]} | {f3["society_to_finance"]} |',
              f'| 社会→文娱 | {f2["society_to_entertainment"]} | {f3["society_to_entertainment"]} |',
              '', '| 配置 | 第 1 轮训练评估损失 / 验证损失 | 最佳轮训练评估损失 / 验证损失 | 最后轮训练评估损失 / 验证损失 |', '|---|---:|---:|---:|']
    for name in ('F2','F3-A'):
        x=v[name]; lines.append(f'| {name} | {x["loss_first"]["train_eval"]:.4f} / {x["loss_first"]["val"]:.4f} | {x["loss_best"]["train_eval"]:.4f} / {x["loss_best"]["val"]:.4f} | {x["loss_last"]["train_eval"]:.4f} / {x["loss_last"]["val"]:.4f} |')
    lines += ['', '训练曲线及验证混淆矩阵分别保留在 `../f2_formal_word_char/F_word_char_s42/` 和 `../f3_formal_a/F_word_char_s42/`。`f3_a_per_class.csv`、`f3_a_verification.json` 保存独立复算的指标。',
              '', '## 风险与限制', '',
              '- 样本按类别关键词形成标题审阅池并经标题级筛查，因此主题分布不等同于无偏随机扩充；社会 1 月和 3 月明显低于基准月份比例，可能影响对规模效应的归因。',
              '- 通过 ≥0.5 的字符三元组相似度筛查不保证不存在同事件的低词面重叠改写；语义类别与来源仍可能存在歧义。风险项保留在 `f3_manual_review_needed.csv` 和 `f3_event_screen.csv`。',
              '- 原训练集内部既有 4 对高相似标题，另有 1 组与恢复数据的跨类别规范化标题冲突；基准未被修改，F2 和 F3-A 均继承原训练集。',
              '- 归档页提供文章链接、栏目、标题、日期和分钟；B 级不等同于单独获取并核实每篇文章原文的实际发布时间。未通过验证集以外的测试集评估。',
              '- 不同规模各自重新拟合 TF-IDF、卡方与 BP，词表、随机梯度路径和训练时间随规模变化；单种子差异不应视为因果结论。',
              '', '## 复核', '',
              '原训练、验证、测试划分的文件哈希与登记一致；未访问测试标签或指标。模型 checkpoint 在验证集的预测、Macro-F1、Accuracy、混淆矩阵及历史早停重放均与保存结果一致。完整命令及相关产物见下方。',
              '', '```powershell',
              'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe train_bp.py --exp F_word_char --seeds 42 --model-type bp --hidden 64 --lr 0.0003 --dropout 0.2 --input-dropout 0 --train-file results/member_C/f3_analysis/f3_a_dataset/train.csv --train-sha256 ' + meta['train_sha256'] + ' --output-root results/member_C/f3_formal_a',
              '```', '']
    report.write_text('\n'.join(lines),encoding='utf-8')
    print('Report written:',report)


if __name__ == '__main__':
    main()

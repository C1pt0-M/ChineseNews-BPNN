"""Document the F.3 reviewed run and joint F.3/F.4 findings from saved metrics."""
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
labels=['财经','体育','文娱','教育','社会','国际']
v=json.loads((HERE/'f34_formal_verification.json').read_text(encoding='utf-8'))
meta=json.loads((HERE/'f3_a_dataset_formal/manifest.json').read_text(encoding='utf-8'))
classes=pd.read_csv(HERE/'f34_formal_per_class.csv',encoding='utf-8-sig')
new=pd.read_csv(HERE/'f3_a_dataset_formal/new_sample_evidence.csv',dtype=str,encoding='utf-8-sig')
dec=pd.read_csv(HERE/'f34_title_review_decisions_v10.csv',dtype=str,encoding='utf-8-sig')
pairs=pd.read_csv(HERE/'f34_pair_review.csv',dtype=str,encoding='utf-8-sig')
classic=json.loads((ROOT/'results/member_C/classic_baselines/verification.json').read_text(encoding='utf-8'))
f2,f3=v['F2'],v['F3-A']
text=['# F.3 正式 F3-A 最终审核与单种子实验','',
      '正式数据版本已冻结，并在独立目录完成 seed 42 训练。原 6,549 行逐字段、逐顺序保留；新增 982 行；固定验证集不变。之前 0.78729 的初版结果因训练后发现数据质量问题，仅为诊断，不能作为正式结论。','',
      '## 数据来源与准入','',
      '恢复数据源是中国新闻网的 2025 年滚动归档，教育候选 371 条既有时间证据 A 129、B 242；定义见 `f3_education_final_review_v3.csv`。本轮额外审阅此前未入队列的 74 条教育标题；教育标题审阅队列共 '+str(sum(dec.category=='教育'))+' 条，其中 '+str(sum((dec.category=='教育')&(dec.semantic_review=='title_review_pass')))+' 条通过标题级筛查。这个通过数不是全部通过最终原文和同事件审核的人数。正式新增教育 160 条，原有证据等级 A '+str(sum((new.category=='教育')&(new.grade=='A')))+'、B '+str(sum((new.category=='教育')&(new.grade=='B')))+'。','',
      '最终 982 条均重新解析经过哈希校验的官方日期归档页，核对 ID、URL、标题、栏目映射与分钟；又直接取得官方文章页面，逐条核对页面标题、发布时间与正文存在。词面数字提示的 11 条经正文复核，处理说明在 `f3_a_dataset_formal/article_audit.csv`。表中 source 字段表示中国新闻网发布平台，页面实际署名来源另列，不将平台误写成全部文章原创机构。','',
      '正式版本对 ID、规范化 URL 和标题进行了与原训练、验证、测试数据的精确排重；读取测试集仅限身份、URL、标题、时间用于排重，不读取测试标签或计算测试指标。Jaccard 0.5 筛查及额外字符相似度筛查均不改变任何标签。进一步检查字符 TF-IDF 相似度：全时段 ≥0.5 的 73 对以及 14 日内 ≥0.3 的 55 对（两组有重合），都保存双方官方正文的区分事实和哈希，分别见 `f34_v10_broad_review.csv` 与 `f34_v10_wide_event_review.csv`。','',
      '按同一案件、政策、发布会或比赛结果可能重复的保守标准排除了多条新增候选，包括同一学位办法、同一大学成立、同一白宫会晤与同一次羽毛球比赛；原有训练集未删。教育补充池及后续排除理由见 `f34_title_review_decisions_v10.csv`，固定 seed 42 按类别和月份分层，完全独立的清单、页面署名、哈希及样本量保存在 `f3_a_dataset_formal/`。','',
      '| 类别 | 原训练 | 新增 | 合计 |','|---|---:|---:|---:|']
for cat in labels:
    added=meta['added_counts'][cat];total=meta['category_counts'][cat]
    text.append(f'| {cat} | {total-added} | {added} | {total} |')
text+=['','教育新增月份分布：'+ '、'.join(f'{int(k)}月 {n}' for k,n in sorted(meta['monthly']['教育']['monthly_actual'].items(),key=lambda x:int(x[0])))+'。完整类别和月份分布、按月配额及偏差保存在 manifest 中。','',
       '正式训练文件 SHA-256：`'+meta['train_sha256']+'`。每条新增样本的 ID、栏目、原始字段与归档证据见 `new_sample_evidence.csv`，文章来源及哈希见 `article_audit.csv`，配额与完整证据文件哈希见 `manifest.json`。','',
       '## 原 v3 的 23 对弱相似记录','',
       '这 23 对均核对双方保留的官方文章正文，均为不同比赛项目、不同独立市场数据发布或不同机构举办的活动；不同日期或不同标题没有被单独当作通过依据。逐对事实、原文 URL 和 SHA-256 在 `f34_pair_review.csv`。其中 cn10516755 虽与对照是不同发布，但其标题写上调 33 个基点、正文写 9 个基点，因数值冲突被排除；其他候选仍需通过后续更宽筛查，不能仅凭本表准入。','',
       '| 新增 ID | 参考 ID | 原文事实与结论 |','|---|---|---|']
for r in pairs.to_dict('records'):
    text.append(f'| [{r["candidate_id"]}]({r["candidate_source"]}) | [{r["reference_id"]}]({r["reference_source"]}) | {r["candidate_body_facts"]}；对照：{r["reference_body_facts"]}。不同事件。{r["admission_note"]} |')
text+=['','## 正式训练与固定验证集对照','',
       '相同 F_word_char：词级与字级各 5,000 维，稀疏拼接后行 L2 归一化；BP hidden=64、lr=0.0003、dropout=0.2、input_dropout=0、seed=42。两个训练集分别仅使用各自训练标题和训练标签拟合词表、IDF 和卡方选择；验证集只做 transform。训练固定 patience=10、min_delta=0.0001。','',
       '| 配置 | 训练数 | Macro-F1 | Accuracy | 最佳 / 总 epoch | 训练秒 | 参数 |','|---|---:|---:|---:|---:|---:|---:|']
for name in ('F2','F3-A'):
    a=v[name];text.append(f'| {name} seed 42 | {a["train_rows"]} | {a["macro_f1"]:.5f} | {a["accuracy"]:.5f} | {a["best_epoch"]} / {a["epochs_run"]} | {a["train_time_seconds"]:.2f} | {a["parameters"]} |')
text+=['',f'F3-A 的验证 Macro-F1 相对 F2 为 **{v["delta_macro_f1"]:+.5f}**。这是同一固定验证集上的单随机种子对照，不足以证明数据规模的独立因果效应或测试集泛化。','',
       '| 类别 | F2 P | F2 R | F2 F1 | F3-A P | F3-A R | F3-A F1 |','|---|---:|---:|---:|---:|---:|---:|']
for cat in labels:
    a=classes[(classes.model=='F2')&(classes.category==cat)].iloc[0]
    b=classes[(classes.model=='F3-A')&(classes.category==cat)].iloc[0]
    text.append(f'| {cat} | {a.precision:.5f} | {a.recall:.5f} | {a.f1:.5f} | {b.precision:.5f} | {b.recall:.5f} | {b.f1:.5f} |')
text+=['','| 误分类方向 | F2 | F3-A |','|---|---:|---:|',
       f'| 社会→财经 | {f2["society_to_finance"]} | {f3["society_to_finance"]} |',
       f'| 社会→文娱 | {f2["society_to_entertainment"]} | {f3["society_to_entertainment"]} |','',
       '| 配置 | 第1轮 train_eval / val loss | 最佳轮 train_eval / val loss | 最后轮 train_eval / val loss |','|---|---:|---:|---:|']
for name in ('F2','F3-A'):
    a=v[name];text.append(f'| {name} | {a["loss_first"]["train_eval"]:.4f} / {a["loss_first"]["val"]:.4f} | {a["loss_best"]["train_eval"]:.4f} / {a["loss_best"]["val"]:.4f} | {a["loss_last"]["train_eval"]:.4f} / {a["loss_last"]["val"]:.4f} |')
text+=['','原训练、验证、测试文件哈希均与版本登记一致。对两个运行独立重新拟合特征处理器，核对词表、IDF、卡方选择器；strict 加载 checkpoint 后，验证预测、Macro-F1、Accuracy 和混淆矩阵与保存记录一致。按历史重放 min_delta 和 patience，最佳模型和早停轮次一致。','',
       '训练历史、特征处理器、checkpoint、验证预测、训练曲线和混淆矩阵见 `../f3_formal_a_reviewed/F_word_char_s42/`；复算结果 `f34_formal_verification.json`，六类明细 `f34_formal_per_class.csv`。','',
       '## 局限','',
       '- 同事件的自然语言改写无法由字符相似度完全穷尽；原训练自身既有 4 对高相似标题，恢复数据还与其有 1 组跨类别标题冲突。基准必须保留，因此这两组对照均继承历史风险。',
       '- 依据教育标题和证据筛选，补充候选并非训练时间窗全部新闻的无偏随机样本；样本规模、主题与重新拟合词表同时改变。',
       '- 这是 seed 42 的单次验证对照；五随机种子尚未运行，不能把 +0.00483 当作稳定增益，也未达到 0.85。',
       '', '## 实际训练命令','', '```powershell',
       'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe train_bp.py --exp F_word_char --model-type bp --hidden 64 --lr 0.0003 --dropout 0.2 --input-dropout 0 --seeds 42 --train-file results/member_C/f3_analysis/f3_a_dataset_formal/train.csv --train-sha256 '+meta['train_sha256']+' --output-root results/member_C/f3_formal_a_reviewed',
       '```','']
report=HERE/'f3_final_review.md'
if report.exists():raise FileExistsError(report)
report.write_text('\n'.join(text),encoding='utf-8')
svm=classic['LinearSVC'];lr=classic['LogisticRegression']
stage=ROOT/'results/member_C/f34_stage_summary.md'
if stage.exists():raise FileExistsError(stage)
stage.write_text('\n'.join(['# F.3 + F.4 阶段结果','',
 f'F.3：正式审核版保留原 6,549 条、增加 982 条，seed 42 验证 Macro-F1 {f3["macro_f1"]:.5f}，相对同 seed F2 {f2["macro_f1"]:.5f} 变化 {v["delta_macro_f1"]:+.5f}；社会→财经 {f2["society_to_finance"]}→{f3["society_to_finance"]}，社会→文娱 {f2["society_to_entertainment"]}→{f3["society_to_entertainment"]}。仅一次运行，主题与词表亦有变化，仍需判断稳定性。详细准入和事件审核见 `f3_analysis/f3_final_review.md`。','',
 f'F.4：训练集内无泄漏五折选参后，LinearSVC 的 C={svm["C"]:g}、验证 Macro-F1={svm["val_macro_f1"]:.5f}；LogisticRegression 的 C={lr["C"]:g}、验证 Macro-F1={lr["val_macro_f1"]:.5f}，均低于 F2 BP seed 42 的 {f2["macro_f1"]:.5f}。详细类别与混淆矩阵见 `classic_baselines/summary.md`。','',
 '两种线性方法在当前特征和 C 网格下未超过 BP，但比较不能证明 BP 已达上限。社会类仍相对困难；类间语义重叠、标注差异、训练特征及样本量可能共同影响结果，现有实验不足以分离各因素。BP 使用验证集早停，而经典模型 C 用训练折选择，也需注意比较口径。','',
 '建议下一阶段先由研究负责人决定是否对审核版扩充数据进行五随机种子复现，并针对社会类错误开展内容与标注审计；现有结果尚不能支持对 0.85 可达性的承诺。未使用测试集指标，未启动其他架构或额外参数搜索。','']),encoding='utf-8')
print('Wrote',report,'and',stage)

"""Reports and figures for the second Member A acceptance audit.

Reads only produced diagnostics and historical fixed-parameter LR results.
Does not train, tune, or run test-set inference.
"""
from pathlib import Path
import json
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from models.plots import _setup_chinese_font
from models.runs import save_json


def table(frame, digits=4):
    def fmt(value):
        if isinstance(value, (float, np.floating)): return f'{value:.{digits}f}'
        return str(value).replace('|', '／').replace('\n', ' ')
    rows=['| '+' | '.join(map(str,frame.columns))+' |', '| '+' | '.join(['---']*len(frame.columns))+' |']
    rows += ['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join(rows)


def generate(root):
    root=Path(root)
    out=root/'A_raw_s42'
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    h=pd.read_csv(out/'history.csv')
    tokens=pd.read_csv(root/'token_diagnostics.csv')
    details=json.loads((root/'error_feature_evidence.json').read_text(encoding='utf-8'))
    near=json.loads((root/'near_duplicate_risk.json').read_text(encoding='utf-8'))
    five=pd.read_csv(root/'five_seed_validation.csv')
    summary=pd.read_csv(root/'five_seed_summary.csv')
    per_class=pd.read_csv(root/'per_class_val.csv')
    lr=json.loads(Path('results/baseline_lr.json').read_text(encoding='utf-8'))
    test_count=int(re.search(r'Ran (\d+) tests', (root/'unit_tests.log').read_text(encoding='utf-8')).group(1))
    figures=root/'figures';figures.mkdir(exist_ok=True)
    _setup_chinese_font()
    qa=[]

    def save(fig,name):
        with warnings.catch_warnings(record=True) as caught:
            fig.canvas.draw()
            fig.savefig(figures/f'{name}.png',dpi=300,bbox_inches='tight')
        renderer=fig.canvas.get_renderer()
        clipped=[]
        for ax in fig.axes:
            for text in [ax.xaxis.label,ax.yaxis.label,ax.title,*ax.get_xticklabels(),*ax.get_yticklabels()]:
                if not text.get_visible() or not text.get_text():continue
                box=text.get_window_extent(renderer)
                if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
                    clipped.append(text.get_text())
        qa.append({'figure':name,'warnings':[str(x.message) for x in caught],'clipped_text':clipped})
        fig.savefig(figures/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)
        with Image.open(figures/f'{name}.png') as im:ImageOps.grayscale(im).save(figures/f'{name}_grayscale.png')

    best=m['training']['best_epoch'];stop=m['training']['epochs_run']
    for name,cols,ylabel in [('train_online_loss',['train_online_loss'],'CrossEntropy Loss'),
            ('eval_loss',['train_eval_loss','val_loss'],'CrossEntropy Loss'),
            ('val_macro_f1',['val_macro_f1'],'Validation Macro-F1')]:
        fig,ax=plt.subplots(figsize=(7.2,3.8),dpi=180,layout='constrained')
        for col,color,marker,ls in zip(cols,['#0072B2','#D55E00'],['o','s'],['-','--']):
            ax.plot(h.epoch,h[col],color=color,marker=marker,ms=3,ls=ls,label=col)
        ax.axvline(best,color='#555',ls=':',label=f'Selected epoch {best}')
        ax.axvline(stop,color='#999',ls='--',label=f'Early stop epoch {stop}')
        ax.set(xlabel='Epoch',ylabel=ylabel,xlim=(.7,stop+.8))
        ax.set_xticks(range(1,stop+1,2))
        if name=='val_macro_f1':
            ax.set_ylim(.72,.77);ax.set_yticks(np.arange(.72,.771,.01))
        else:
            top=np.ceil(h[cols].to_numpy().max()/.2)*.2
            ax.set_ylim(0,top);ax.set_yticks(np.arange(0,top+.01,.2))
        # Keep text away from the early-stop vertical line on the right edge.
        ax.legend(frameon=False,fontsize=8,loc='upper center')
        ax.grid(alpha=.2)
        save(fig,name)
    fig,ax=plt.subplots(figsize=(7.2,4.2),dpi=180,layout='constrained')
    ys=np.arange(len(per_class))
    for col,offset,marker,color in [('precision',-.18,'o','#0072B2'),('recall',0,'s','#D55E00'),('f1',.18,'^','#009E73')]:
        ax.scatter(per_class[col],ys+offset,marker=marker,color=color,label=col,s=34)
    ax.set_yticks(ys,per_class.category)
    ax.set(xlim=(0,1),xlabel='Validation score',ylabel='真实类别',title='B0 逐类别表现（种子 42）')
    ax.grid(axis='x',alpha=.2);ax.legend(frameon=False,loc='lower left')
    save(fig,'per_class_metrics')
    save_json(root/'figure_qa.json',qa)
    assert not any(q['clipped_text'] or q['warnings'] for q in qa),qa
    cases=[]
    for d in details:
        original=d['错误类型']
        if '模型' in original:
            kind='分词/特征表达问题' if d['removed_pure_numbers'] else '暂无法判断'
        elif '标签噪声' in original:kind='疑似标签噪声（待核查）'
        else:kind=original
        cases.append({'ID':d['id'],'标题':d['title'],'真实标签':d['category'],'预测标签':d['predicted_category'],
            '诊断类别':kind,'判断依据及局限':d['简要分析'],
            '保留tokens':' / '.join(d['retained_tokens']), '未覆盖tokens':' / '.join(d['oov_tokens']),
            '非零特征数':d['nonzero_feature_count'],
            '最高权重特征（最多6项）':'; '.join(f"{f['term']}={f['tfidf']:.4f}" for f in d['nonzero_features'][:6])})
    cases=pd.DataFrame(cases)
    cases.to_csv(root/'typical_errors_with_features.csv',index=False,encoding='utf-8-sig')
    prefix='../results/member_A_baseline_recheck/'
    general=Path('reports/member_A_baseline_recheck.md')
    train=tokens[tokens.split=='train'].iloc[0];val=tokens[tokens.split=='val'].iloc[0]
    delta=m['val']['f1_macro']-lr['val']['f1_macro']
    data_text=f'''# 成员 A：数据与特征局限性复核

本轮继续使用数据集 **v1.1.0**、划分 **v1.0.0**。未采集、清洗、抽样、重新划分或改标签。Test 在本轮仅用于只读完整性检查（指纹、ID/URL/标题交集、时间和前缀分组），未进入模型预测或特征覆盖率诊断。

## 数据完整性

校验日志见 [dataset_verification.log]({prefix}dataset_verification.log)，机器结果见 [data_integrity.json]({prefix}data_integrity.json)。12,000 条、六类各 2,000 条，train/val/test 分别 6,549/3,184/2,267 条；模型标签索引与版本登记逐项一致。三对集合的 ID、URL、精确标题交集均为 0。时间边界及四份正式 CSV 指纹检查通过。

## 已知限制与处理决定

| 限制 | 证据与影响边界 | 本轮处理 |
|---|---|---|
| 单一数据来源 | 学到的是中国新闻网标题和频道习惯，不能外推到其他网站 | 保留数据，明确外部效度限制 |
| 文化→文娱代理映射 | 原始标签为文化；含文博、文学、非遗、艺术等。历史150条标题判读中狭义娱乐约22.7%，不是全体比例估计 | 保留类名与索引，使用广义文化定义 |
| 社会频道语义交叉 | 验证集社会↔财经182条、社会↔文娱134条、社会↔教育76条 | 将真实标签视为频道标签，不反向改标签 |
| 标题全局去重可能过度删除 | 历史审计对比标题+归档日期后，目标池179,927→180,650，多723条（约0.402%）；该数字是池差异，不是本12,000条的删样率 | 不重做清洗，不认定为性能下降直接原因 |
| 均衡配额与自然分布不同 | 全集每类2000，教育原始供给较少；子集并非严格均衡 | 不改变配额，不宣称适合自然新闻流 |
| 短标题与上下文缺失 | 无正文、编辑依据或完整事件背景 | 不将标题语义判断写成已核实错标 |
| 历史测试集已接触 | 多组历史配置已测试 | 本次不预测测试集、不据其选择任何配置 |

历史审计曾把“前缀相同”“短时间跨度”和“跨频道路径”用于推测同稿/不同事件，也使用过“硬噪声”等表述；这些均不是正文或编辑级确认。不能将这些启发式判读当作确定错标率、同稿率或“已无语义泄漏”的证明。

## 时间划分与近重复风险

本次重新统计标题前8个字符：跨集合前缀 **{near['cross_split_groups']}组、{near['affected_records']}条**，其中跨集合且跨类别 **{near['cross_split_cross_category_groups']}组**。完整列表见 [cross_split_prefix_groups.csv]({prefix}cross_split_prefix_groups.csv)。前缀相同可能只是栏目模板，也可能是同事件或系列新闻，当前统计不能区分。精确交集为0不代表事件级独立；近重复可能让泛化估计更乐观，无法在未构造事件金标准时确定影响方向和大小。保留时间划分，不改成随机划分，也不移动样本。

## 分词和覆盖率：仅 Train/Validation

{table(tokens,6)}

OOV口径为：当前保留的一元 token 不在训练TF-IDF词表中。`oov_occurrence_rate` 按 token 出现次数加权；`oov_unique_rate` 按唯一 token 类型统计。词表含一元和二元特征，受 min_df/max_df 等筛选，故训练集自身也可能出现“OOV”；验证OOV不是纯粹的“从未在原训练文本出现”的新词率。它衡量未被既定表示覆盖的词项，不能直接等同于错误原因。空token与零特征文档分开计数。

验证 token 出现次数覆盖率为 **{1-val.oov_occurrence_rate:.2%}**，未覆盖率 **{val.oov_occurrence_rate:.2%}**。另按保留token在原训练分词集合中是否出现计算，真正“训练中未见”的出现次数占比为 **{val.unseen_in_train_occurrence_rate:.2%}**，它和词表未覆盖率口径不同。

空token文档为0，但验证集有 **{int(val.zero_feature_documents)}** 条非空标题变成零特征向量，具体ID与保留tokens见 [zero_feature_documents.json]({prefix}zero_feature_documents.json)。这证明存在完全未覆盖输入，但不能将所有错分都解释为词表问题。完整词项计数保存在 train_oov_tokens.csv / val_oov_tokens.csv；数字过滤、专名和错分标题的逐项证据见 [错误分析](member_A_error_analysis.md)。没有改变分词规则或参数。
'''
    Path('reports/member_A_data_limitations.md').write_text(data_text,encoding='utf-8')
    entities=json.loads((root/'named_entity_tokenization.json').read_text(encoding='utf-8'))
    entity_table=pd.DataFrame([{'诊断词串':d['text'],'分词结果':' / '.join(d['retained_tokens']),
        '训练标题含该串':d['train_title_mentions'],'验证标题含该串':d['val_title_mentions'],
        '词表覆盖':str(d['retained_token_in_vocab'])} for d in entities])
    digit=[d for d in details if d['removed_pure_numbers']]
    digit_table=pd.DataFrame([{'ID':d['id'],'标题':d['title'],'过滤纯数字':' / '.join(d['removed_pure_numbers']),
        '实际保留tokens':' / '.join(d['retained_tokens'])} for d in digit])
    error_text=f'''# 成员 A：验证集错误与特征表达诊断

仅使用验证集。20条案例沿用上一轮确定性方向覆盖样本（四个主要方向每个3条、另四个方向每个2条；按时间和ID排序），并给每条附上实际jieba分词、OOV和全部TF-IDF非零特征。预测已与历史最佳checkpoint的已保存验证预测逐条一致核对。取样偏向较早时段，不是随机样本，不估计错误类型占比。

## 分类效果及混淆方向

{table(per_class)}

![逐类别P/R/F1]({prefix}figures/per_class_metrics.png)

![验证混淆矩阵]({prefix}A_raw_s42/confusion_val.png)

社会→财经100、财经→社会82、社会→文娱82、文娱→社会52、教育→社会39、社会→教育37；财经→文娱25、文娱→财经18。社会类别F1最低（0.5755），不能由此认定这些标签错误。

## 20条案例与判断依据

{table(cases[['ID','标题','真实标签','预测标签','诊断类别','判断依据及局限']])}

“暂无法判断”用于无充分特征归因证据的案例，替代笼统的“模型未学好”。分词/特征表达问题仅说明观察到输入表示损失，不等于证明它造成了该条错分。“疑似标签噪声”只是待核查项；例如科研内容可能因高校主体而归入教育，缺少正文和编辑依据时不能认定误标。

## 分词和实际非零特征

{table(cases[['ID','保留tokens','未覆盖tokens','非零特征数','最高权重特征（最多6项）']])}

表中只是TF-IDF输入权重最高的最多6项，**不是神经网络特征贡献或解释权重**。每条全部非零项及原始分词保存在 [error_feature_evidence.json]({prefix}error_feature_evidence.json)，便于复核，未只挑选支持结论的特征。

## 纯数字过滤实例

{table(digit_table)}

这些例子证明部分数值信息在建模前被去除；例如140与785的具体数量消失，后续还可能保留万亿、列车等语义。汉字数字词“四”“万亿”仍符合B0保留规则，不能因isnumeric为真就说它们被删除。另在“57岁家政工”实例中，实际得到“岁家 / 政工”，可以观察到跨词边界的切分风险；这仍不是该次误分类的因果证明。是否保留数字或更换分词能改善分类需要另做受控实验，本轮不实施。

## 专有名词、机构和赛事名称

{table(entity_table)}

以上词串为预先指定的诊断探针；`train_title_mentions`/`val_title_mentions`为标题含完整子串的计数，不是实体识别准确率。没有出现的探针不能伪装成实际新闻案例。分成多个token不必然错误；组合二元词能表达部分局部组合，但长专名、未覆盖片段仍有潜在信息限制。完整证据见 [named_entity_tokenization.json]({prefix}named_entity_tokenization.json)。
'''
    Path('reports/member_A_error_analysis.md').write_text(error_text,encoding='utf-8')
    report=f'''# 成员 A：Baseline 整改验收报告

本轮基准提交为 `{m['provenance']['git_head']}`，任务来源为 docs/成员A_Baseline整改任务书.md。工作范围仅成员A，第1～3章；原 [Baseline正文与算法说明](member_A_baseline_audit.md) 及上一轮所有产物保留不变。本报告提供本轮整改差异与验收证据，数据风险和错误分析分别由链接报告维护，避免混淆历史与本轮实验。

## P0 / P1 / P2 审计发现及处理

| 优先级 | 发现 | 修复与证据 |
|---|---|---|
| P0 | 测试入口缺少SHA256校验 | evaluate_test.load_test_split在读CSV前校验test指纹；仅用临时损坏文件测试阻断，不执行实际测试预测 |
| P0 | 完整性脚本缺少显式URL交集、固定版本和索引检查 | verify_dataset.py补齐；日志通过。未改任何正式CSV |
| P0 | 原记录无本轮源码下的成功训练日志 | 独立目录运行完整B0 s42，训练入口返回0，运行时test.csv读取尝试为0 |
| P1 | 只有2个既有行为测试 | 扩展为{test_count}项，覆盖加权eval、无梯度/参数/RNG变化、Dropout关闭、fit范围、指纹异常、防覆盖、最佳权重恢复、数字诊断与LR加载范围 |
| P1 | 原图未标早停轮次、缺逐类别图 | 本轮图标注选择轮次3和早停13；补充P/R/F1点图 |
| P1 | FeatureBuilder文档过度承诺API强制训练集fit | 修正文档；实际由入口调用边界与集成测试保证 |
| P1 | LR入口即使no-test也经默认loader读取Test | 改为只加载train/val，实际测试读取仅走已加指纹门禁的测试入口；本轮没有重训LR |
| P2 | 未量化分词和词表覆盖、错误归因证据不足 | 新增分词/OOV/空文档诊断、20例完整非零特征，将无证据“模型问题”标为暂无法判断 |
| P2 | 数据局限和交接材料不完整 | 单一来源、文化映射、去重、近重复及历史测试接触均单独记录，并给B/C交接清单 |

## 固定配置与实际复现

TF-IDF维度10864；单隐层256/ReLU/Dropout0.2/6 logits；CrossEntropyLoss和Adam；lr=0.001、weight_decay=0.0001、batch=64、max_epochs=100、patience=10、min_delta=0.0001、clip=5；seed=42；参数量2,782,982。没有新增Softmax，没有改网络、损失、分词或超参数。

损失记录沿用已完成修复：在线loss来自训练过程；train_eval_loss和val_loss在每轮末、无更新间隔、eval/no_grad模式下按样本数加权计算。历史缺失的eval loss没有补造。

本轮最佳epoch **{best}**，共 **{stop}** 轮，验证Macro-F1 **{m['val']['f1_macro']:.6f}**、Accuracy **{m['val']['accuracy']:.6f}**。逐轮online loss、val loss、val F1和val accuracy与历史结果的最大绝对差均为 **0**（声明容差0），验证预测、混淆矩阵、参数量一致。详见 [reproduction_check.json]({prefix}reproduction_check.json) 和 [training.log]({prefix}training.log)。本轮metrics.test为null；训练运行时尝试读取Test次数为0。

## 五随机种子：历史结果重算

{table(five)}

{table(summary,6)}

采用样本标准差ddof=1、n=5；验证Macro-F1为0.7629±0.0024、Accuracy为0.7616±0.0027。seed43在第4轮最优，其余四种子均为第3轮；这说明最优轮次也受训练随机性影响。本轮只重训seed42，五种子均值来自既有原始JSON，不伪称重跑五次。标准差反映固定数据划分下随机初始化/训练波动，不是抽样置信区间。

## 训练过程与过拟合

![Train Online Loss]({prefix}figures/train_online_loss.png)

![Train Eval与Val Loss]({prefix}figures/eval_loss.png)

![验证Macro-F1]({prefix}figures/val_macro_f1.png)

同模式训练eval loss在第1/3/4/13轮分别为1.117927/0.224267/0.135115/0.024436；val loss分别为1.255759/0.687028/0.681630/0.814027。第3轮与第13轮损失差距分别0.462761和0.789591。第4轮之后训练loss继续下降、验证loss持续回升，验证F1由最佳0.762195降至0.744446，支持明显后期过拟合。模型参数多于样本只是风险因素，此处结论由曲线与指标共同支持，不由参数量单独推导。

最佳Macro-F1在第3轮，最低val loss在第4轮，二者衡量不同。早停仅依据F1超过已存最优+min_delta，连续10轮无改善后在第13轮终止，恢复第3轮权重。验证loss上升不是早停条件。时间变化、标题信息和频道交叉也可能贡献泛化差距，不能声称所有差距由某一种原因造成。

## 分类、特征和数据诊断

参见 [验证错误与特征分析](member_A_error_analysis.md)（完整20例、类别图与混淆矩阵）以及 [数据与特征局限](member_A_data_limitations.md)。社会类最难，F1=0.5755；财经0.7047、文娱0.7322。训练平均token数{train.mean_tokens:.3f}，验证{val.mean_tokens:.3f}；验证token出现次数OOV率{val.oov_occurrence_rate:.2%}。这些描述是诊断证据，不能单独证明数字处理或词表覆盖造成性能瓶颈。

网络结构图复用 [上一轮图源](../results/member_A_audit_20261105/A_raw_s42/network.drawio) 和 [PNG](../results/member_A_audit_20261105/A_raw_s42/network.png)，模型定义无变化，无需重复绘图。

## 固定参数Logistic Regression参考

历史LR使用相同时间划分、同一FeatureBuilder('A')、相同训练集fit流程，特征信息为10864维、6549训练行；C=1.0、lbfgs、max_iter=2000，没有系统调参。本轮核查代码和结果元数据，未重新拟合LR或测试。LR验证F1={lr['val']['f1_macro']:.6f}，与BP seed42的差为{delta:.6f}（约{delta*100:.3f}个百分点）。这是单配置参考差异，不能宣称胜过经过充分优化的最强传统分类器，也未进行配对显著性检验。

LR历史测试F1约0.7183、B0五种子历史测试F1约0.7372，仅注明既有记录，不参与本轮选择、诊断模型优化或重新评估。历史LR未保存特征流水线本体，本轮不能独立证明其历史词表逐项相同，只能核对共用代码、数据路径和元信息。

## 修改文件与理由

| 文件 | 目的 |
|---|---|
| evaluate_test.py | 实际Test读取前检查指纹 |
| verify_dataset.py | 补齐版本、索引、URL及规模验收项；重复调用清空检查列表 |
| baseline_lr.py | 训练与--no-test不再加载Test |
| models/features.py | 更正API不能鉴别集合来源的文档承诺，算法不变 |
| models/plots.py、train_bp.py | 标注真实选模与早停轮次，优化过程不变 |
| tests/test_baseline_audit.py | 增加fit边界、损坏指纹、防覆盖和恢复权重测试 |
| tests/run_training_without_test.py | 完整训练期间禁止实际读取test.csv |
| recheck_member_A.py、member_A_recheck_report.py | 独立复现、数据/特征诊断、验收报告和图表 |
| 新增三份报告 | 本轮验收、数据局限、错误证据与交接；保留历史报告 |

## 给成员B/C的交接清单

共同基准：v1.1.0/划分1.0.0、B0上述配置、固定种子集42～46；原始训练/验证文件指纹见data_integrity.json，指标见five_seed_summary.csv，曲线及混淆矩阵见本轮目录。

| 接手人 | 有证据的问题 | 后续应控制的条件与边界 |
|---|---|---|
| B | 10864维对应278万参数；词项覆盖、数字过滤与专名切分存在可观察限制 | 研究特征维度/卡方/SVD的判别信息与成本，字符n-gram或保留数字只能单列补充实验；A不选算法或维度 |
| B | 不同特征表示可能改变表示信息和计算量 | 若解释历史SVD下降需核对维度、标准化等变量；本报告不据历史测试选型，也不给出优劣结论 |
| C | 同状态loss与验证F1支持后期过拟合 | 固定数据、已选特征、其余训练条件，对Input Dropout等候选做受控研究；A不实施改进 |
| C | 单种子成功不足以说明稳定改善 | 相同五种子比较验证F1和泛化差距，报告真实涨跌与方差，不能预设有效 |
| 团队 | 社会频道交叉、单一来源及已接触测试集 | 不把所有错误归因于网络；数据变更需共同决定，独立评估问题需另定方案 |

## 验收、复现入口与不能确认的结论

已实际执行完整数据校验、{test_count}项自动化测试、B0 s42复现以及特征诊断。日志在独立结果根目录；source_snapshot保留训练阶段源码，analysis_source_snapshot保存随后补充的分析脚本和最终测试版本，metrics记录训练git版本和哈希；保护清单对照本轮开始前所有受保护文件，校验结果见protection_check.json。Test完整性读取与Test模型评估不同，本轮没有后者。

| 任务书验收项 | 状态 | 证据 |
|---|---|---|
| 1 固定且可追溯的B0 | 通过 | preflight.json、A_raw_s42/metrics.json、源码快照 |
| 2 数据与划分完整性 | 通过 | dataset_verification.log（41项）、data_integrity.json |
| 3 特征fit范围 | 通过 | 训练入口测试、实际builder.fit_rows=6549；API本身不能辨识数据来源 |
| 4 固定种子重现 | 通过 | reproduction_check.json，声明容差0 |
| 5 五种子统一统计 | 通过 | five_seed_validation.csv、five_seed_summary.csv，ddof=1 |
| 6 同状态同模式损失 | 通过 | history.csv与无参数/梯度/RNG变化测试 |
| 7 过拟合证据 | 通过 | loss与F1图，选模3/早停13 |
| 8 分类困难与方向 | 通过 | 混淆矩阵、逐类别P/R/F1、方向计数 |
| 9 典型案例依据 | 通过 | 20条真实ID、分词和全部非零特征，未确证因果 |
| 10 数据局限与保护 | 通过 | 数据局限报告、protection_check.json |
| 11 自动化测试与完整性 | 通过 | unit_tests.log、dataset_verification.log |
| 12 独立新结果 | 通过 | member_A_baseline_recheck目录、历史文件哈希对照 |
| 13 B/C交接材料 | 通过 | 本报告交接表；未实施B/C改进 |

```powershell
python recheck_member_A.py run       # 新目录；存在则拒绝覆盖
python recheck_member_A.py analyze   # 本轮诊断与报告
```

正文第1～3章与本轮验收证据已具备；仍需按课程格式整合文献引用与图表编号。正文级错标、同事件近重复、特征因果归因、LR最优性能、跨来源泛化，以及任何B/C改进的有效性均未确认。没有这些证据的条目不标作研究结论；本轮没有通过修改数据或参数提高分数。
'''
    general.write_text(report,encoding='utf-8')
    print('Three acceptance reports and four figure sets written.',flush=True)

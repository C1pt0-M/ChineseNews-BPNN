"""Generate member A's report from the isolated, actually executed audit artifacts.

Qualitative cases are title-only readings, not verified label corrections.
"""
from pathlib import Path
import json
import hashlib
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from models.plots import _setup_chinese_font

OUT = Path('results/member_A_audit_20261105/A_raw_s42')
CASES = {
 'cn10544815': ('频道语义交叉', '铁路加开同时涉及公共出行与运输经济；标题无法确定频道侧重。'),
 'cn10545163': ('频道语义交叉', '产业、欧美市场具有财经线索，但源频道按地方社会新闻收录。'),
 'cn10545374': ('模型特征提取不足（候选）', '标题明确出现社交实验，财经关联较弱；可能未充分利用词项组合，须进一步解释模型才可归因。'),
 'cn10546164': ('标题信息不足', '民生蓝图和幸福实景较抽象，标题缺少经济政策、主体与具体事件信息。'),
 'cn10547393': ('频道语义交叉', '入境游同时属于旅游产业和生活社会主题，标题没有给出收入或消费数据。'),
 'cn10548206': ('频道语义交叉', '公司、出版、处罚横跨企业经营、文化与社会监管；预测受处罚事件叙事影响是合理假设。'),
 'cn10544262': ('频道语义交叉', '看戏、文化味是文化线索，真实社会频道侧重群众跨年活动。'),
 'cn10544460': ('频道语义交叉', '博物馆、文博偏文化，民众跨年偏社会，两种主题在同一标题并存。'),
 'cn10546236': ('频道语义交叉', '家政工身份与生活故事偏社会，写书绘画偏文化，短标题兼具两者。'),
 'cn10544388': ('频道语义交叉', '晚会属于文化内容，超高清频道直播又是技术和传播事件；模型可能偏重技术侧词项。'),
 'cn10550352': ('标题信息不足', '藏品提供文化线索，但修辞化表述没有说明博物馆或事件背景。'),
 'cn10550548': ('标题信息不足', '人名和道歉事件缺少身份背景，单凭标题难区分娱乐新闻与一般社会争议。'),
 'cn10544218': ('模型特征提取不足（候选）', '140万亿是重要经济信号，但纯数字过滤会移除140，剩余万亿及贺词未必足以表征财经；未做消融验证。'),
 'cn10546839': ('频道语义交叉', '企业和出海偏财经，国潮娃衣偏文化消费，模型可能突出后者。'),
 'cn10547069': ('频道语义交叉', '文体活动属于文化，消费和部门政策又属于财经语义，频道边界本身交叉。'),
 'cn10547811': ('频道语义交叉', '出版社、百科全书属于文化，数字产品与矩阵又接近企业产品报道。'),
 'cn10550271': ('频道语义交叉', '孩子、引导属于教育，网络梗及生活指导也常见于社会频道。'),
 'cn10556668': ('可能的标签噪声／频道口径差异（待核查）', '标题偏科研技术，教育频道可能依据高校主体归类；标题未交代机构，不能认定原标签错误。'),
 'cn10550222': ('频道语义交叉', '就业与公共服务属于社会议题，也与职业教育相邻；标题缺少学校和培训线索。'),
 'cn10552791': ('模型特征提取不足（候选）', '养宠和消费痛点更接近社会生活，教育线索较弱；不能仅凭标题断言具体致错词。'),
}


def table(df, digits=4):
    def fmt(x):
        if isinstance(x,(float,np.floating)): return f'{x:.{digits}f}'
        return str(x).replace('|','／').replace('\n',' ')
    return '| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+ '\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in df.itertuples(index=False,name=None))


def main():
    h=pd.read_csv(OUT/'history_analysis.csv')
    m=json.loads((OUT/'metrics.json').read_text(encoding='utf-8'))
    check=json.loads((OUT/'reproduction_check.json').read_text(encoding='utf-8'))
    seeds=pd.read_csv(OUT/'historical_five_seeds.csv')
    cases=pd.read_csv(OUT/'error_candidates.csv')
    assert len(cases)==20 and set(cases.id)==set(CASES)
    cases['错误类型']=cases.id.map(lambda x: CASES[x][0])
    cases['简要分析']=cases.id.map(lambda x: CASES[x][1])
    cases.to_csv(OUT/'typical_errors_20.csv',index=False,encoding='utf-8-sig')
    # Three separate exportable figures supplement the combined training plot.
    _setup_chinese_font()
    qa=[]
    for name,cols,ylabel in [
        ('eval_loss',['train_eval_loss','val_loss'],'CrossEntropy Loss'),
        ('online_vs_eval',['train_online_loss','train_eval_loss'],'CrossEntropy Loss'),
        ('val_macro_f1',['val_macro_f1'],'Validation Macro-F1')]:
        fig,ax=plt.subplots(figsize=(7.2,3.7),dpi=300,layout='constrained')
        for c,color,marker,style in zip(cols,['#0072B2','#D55E00'],['o','s'],['-','--']):
            ax.plot(h.epoch,h[c],label=c,color=color,marker=marker,linestyle=style,ms=3,lw=1.6)
        ax.axvline(m['training']['best_epoch'],color='#555555',ls=':',label='selected epoch = 3')
        ax.set(xlabel='Epoch',ylabel=ylabel)
        ax.set_xlim(.7,13.3)
        ax.set_xticks(range(1,14,2))
        if name!='val_macro_f1':
            top=np.ceil(h[cols].to_numpy().max()/.2)*.2
            ax.set_ylim(0,top)
            ax.set_yticks(np.arange(0,top+.01,.2))
        else:
            ax.set_ylim(.72,.77)
            ax.set_yticks(np.arange(.72,.771,.01))
        ax.legend(frameon=False,fontsize=9)
        ax.grid(alpha=.2)
        with warnings.catch_warnings(record=True) as caught:
            fig.canvas.draw()
            fig.savefig(OUT/f'{name}.png',bbox_inches='tight')
        renderer=fig.canvas.get_renderer()
        clipped=[t.get_text() for t in [ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]
                 if t.get_visible() and t.get_text() and not fig.bbox.contains(*t.get_window_extent(renderer).get_points()[0])]
        qa.append({'figure':name,'warnings':[str(w.message) for w in caught], 'clipped_labels':clipped})
        fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
        plt.close(fig)
        with Image.open(OUT/f'{name}.png') as im: ImageOps.grayscale(im).save(OUT/f'{name}_grayscale.png')
    (OUT/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    pc=pd.read_csv(OUT/'per_class_val.csv').rename(columns={'Unnamed: 0':'类别'})
    cm=pd.read_csv(OUT/'confusion_val.csv').rename(columns={'Unnamed: 0':'真实\\预测'})
    dirs=pd.read_csv(OUT/'confusion_directions.csv').head(14)
    link='../results/member_A_audit_20261105/A_raw_s42/'
    hist=h[['epoch','train_online_loss','train_eval_loss','val_loss','eval_loss_gap','val_macro_f1','val_accuracy']]
    summary=pd.DataFrame([{'指标':col,'均值':seeds[col].mean(),'样本标准差(ddof=1)':seeds[col].std(ddof=1)}
                           for col in ['val_accuracy','val_precision_macro','val_recall_macro','val_f1_macro','val_f1_weighted','historical_test_f1_macro']])
    case_table=cases[['id','title','category','predicted_category','错误类型','简要分析']].rename(columns={'id':'样本ID','title':'新闻标题','category':'真实类别','predicted_category':'预测类别'})
    text=f'''# 成员 A：原始 BP 神经网络 Baseline 专项审计与完善

本报告覆盖第 1～3 章。B0 指原始 TF-IDF 输入的既定实验基线（代码实验 ID `A_raw`），不是“未经任何现代训练技术改动的经典 BP”。代码中的 A/B/C 是特征方案编号，与成员 A/B/C 的分工不同。本报告不报告成员 B、C 的研究成果。

## 第 1 章 绪论

### 1.1 研究背景

新闻分类把文本映射到主题类别，可用于信息整理、检索和推荐。本项目使用中文新闻标题研究六分类：标题长度有限，词语歧义、上下文缺失和频道主题交叉会影响识别。TF-IDF 能提供可复现的词项表示；单隐层多层感知机可学习词项的非线性组合，适合作为课程数值实验的基线。

### 1.2 问题提出与研究目标

成员 A 回答三个问题：既定模型的分类表现与随机初始化稳定性如何；训练拟合改善是否伴随验证性能恶化；哪些类别与标题更容易错分。检验依据分别为五种子结果、同一 epoch 末参数状态下的训练/验证损失，以及验证集逐类别指标与错误标题。不能将所有错误归结于网络容量或正则化，也不能用验证集研究的结果声称独立测试结论。

### 1.3 技术路线与范围

采集标题和元数据 → 清洗及六类映射 → 时间划分 → 仅训练集拟合 TF-IDF → 既定 B0 训练 → 验证集选择 checkpoint → 训练曲线和错分分析。后续研究可据此检验高维特征与泛化问题，但具体方案和结果留给其他成员。

## 第 2 章 数据集与原始 BP 算法

### 2.1 数据集

数据来自中国新闻网历史滚动新闻归档，仅使用标题作为模型输入，不使用正文、URL、时间或频道字段作为特征。采集区间为 2025-01-01～2026-10-31。正式数据版本 1.1.0，共 12,000 条，六类各 2,000 条。原始频道“文化”映射为“文娱”，其操作性含义包括文学、艺术、文博、非遗等，不能等同于狭义娱乐。

| 划分 | 时间范围 | 财经 | 体育 | 文娱 | 教育 | 社会 | 国际 | 合计 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | 2025-01-01～2025-12-31 | 1109 | 1107 | 1134 | 1064 | 1057 | 1078 | 6549 |
| val | 2026-01-01～2026-06-30 | 540 | 487 | 503 | 518 | 545 | 591 | 3184 |
| test（版本登记） | 2026-07-01～2026-10-31 | 351 | 406 | 363 | 418 | 398 | 331 | 2267 |

按时间划分用于考察跨时间泛化。全数据平衡不代表每个子集严格平衡；数据来自单一来源，均衡配额也不同于自然新闻流的类别比例。时间变化和同系列近重复可能影响泛化。原审计记录显示 ID、URL 和精确标题跨集合无交集；本次未重做测试集内容分析。

### 2.2 数据预处理

原清洗流程规范空白、换行和不可见字符，剔除空标题、非内容链接及时间异常记录，按 URL 和完全相同标题去重，只保留六类映射，并按日期等距抽样控制每类配额。清洗层保留标题标点；建模时 jieba 精确分词，过滤不含中文或拉丁字母的 token（包括标点和纯数字），纯英文 token 转小写。不额外去停用词。两阶段处理不同，报告不能混写为“原数据已删除全部数字和标点”。本次不改样本或标签。

### 2.3 TF-IDF 特征

训练集拟合词表和 IDF，验证集只 transform。参数为 `min_df=2, max_df=0.95, max_features=20000, ngram_range=(1,2), sublinear_tf=True, norm=l2, smooth_idf=True, lowercase=False`，tokenizer 接收已分词列表。20000 是词表上限，筛选后的实际维度为 10864。

对词项 t 和标题 d，令次数为 c：非零 TF 为 `1+ln(c)`，平滑 IDF 为 `ln((1+N)/(1+df(t)))+1`，两者乘积再对每条标题做 L2 归一化。词项包含一元词和相邻二元词组；未见词不会增加验证集输入维度。标题较短时，稀疏性和未知词可能限制信息表达。本次没有计算未知词率，不能宣称其是主要致错原因。

### 2.4 单隐层 BP 网络原理与定义

本项目使用单隐层多层感知机，通过反向传播计算梯度，并使用 Adam 优化器更新网络参数。BP 是梯度计算机制，Adam 是更新规则；虽然学习率超参数固定为 0.001，Adam 的自适应更新不等于经典固定步长普通梯度下降。

前向过程为 `h=ReLU(W1 x+b1)`，训练时 `h_drop=m*h/(1-p)`，`z=W2 h_drop+b2`，其中 m 是保留概率 0.8 的 Bernoulli 掩码。评估时禁用 Dropout。输出是六个 logits，预测取最大 logit 对应类别。

ReLU 为 `max(0,z)`，提供非线性表达；Dropout 在训练时随机屏蔽隐藏单元，旨在降低共同适应，并不保证消除过拟合。CrossEntropyLoss 对 logits 计算数值稳定的 log-softmax 与负对数似然，`L=-mean(log(exp(z_y)/sum(exp(z_k))))`；模型无需显式再加 softmax。

反向传播按链式法则得到梯度，代码将总梯度范数裁剪到 5。Adam 使用梯度的一阶、二阶矩估计及偏差修正来自适应更新。这里使用 `torch.optim.Adam` 的默认耦合 L2 weight decay：更新计算中加入权重惩罚，覆盖全部传给优化器的参数（包括偏置），不是 AdamW。Weight Decay 旨在约束参数幅度。图中的 Dropout 为已有隐藏层正则化。

![B0 网络结构]({link}network.png)

可编辑图源：[network.drawio]({link}network.drawio)。图源若未导出 PNG，可在 draw.io 打开；Markdown 的结构表示为：

```mermaid
flowchart LR
  X[TF-IDF 10864维] --> H[Linear 10864→256]
  H --> R[ReLU]
  R --> D[Dropout 0.2 仅训练]
  D --> O[Linear 256→6 logits]
  O --> L[CrossEntropyLoss]
```

参数量：第一层 `10864×256+256=2,781,440`；输出层 `256×6+6=1,542`；合计 **2,782,982**。ReLU、Dropout 和损失函数无可训练参数。模型参数数量远多于 6,549 条训练样本，但仅此不能证明过拟合，需要曲线佐证。

### 2.5 固定实验设置与记录方法

| 项目 | 值 |
|---|---|
| 网络 | 10864 → Linear 256 → ReLU → Dropout 0.2 → Linear 6 |
| 损失与优化器 | CrossEntropyLoss；Adam |
| lr / weight_decay | 0.001 / 0.0001 |
| Batch / 最大 epoch | 64 / 100 |
| 早停 | 验证 Macro-F1 超过已存最优值 + 0.0001 才算改善；patience=10 |
| 梯度裁剪 | max_norm=5.0 |
| 历史种子 / 本次种子 | 42、43、44、45、46 / 42 |
| 环境 | Python {m['provenance']['python']}；PyTorch {m['provenance']['torch']}；CPU，{m['provenance']['torch_threads']} 线程 |

Accuracy 为正确数/总数；每类 Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)，Macro 指六类等权平均；Weighted F1 按真实样本量加权。本次只补充测量，不调参。

原来的 `train_loss` 是训练模式下各批次、不同更新状态的样本加权在线损失，现在明确命名 `train_online_loss`。新增 `evaluate_training_set` 在每轮全部更新完成后用 eval/no_grad 在完整训练集分批计算 `train_eval_loss`；验证集紧接着以相同参数状态、相同 eval 模式、同样 512 的评估 batch 和 CrossEntropyLoss 计算 `val_loss`。两者均按样本数加权，不包含 weight decay 惩罚。二者之间无更新；下一轮重新切换 train 模式。

保留 `train_loss=train_online_loss`、`val_f1_macro=val_macro_f1` 作为旧程序兼容别名。历史文件未补造 eval loss。新 `epoch_time` 包含额外训练集评估成本，不能直接与历史计时解释为训练算法变慢。

## 第 3 章 Baseline 实验与问题分析

### 3.1 五随机种子 Baseline

以下为历史 JSON 的直接汇总；没有重跑五种子，也没有运行测试集评估。

{table(seeds)}

{table(summary)}

均值±样本标准差：Validation Macro-F1 **0.7629±0.0024**、Accuracy **0.7616±0.0027**；历史 Test Macro-F1 **0.7372±0.0015**。标准差为 ddof=1，n=5，仅反映固定划分下训练随机性，不是置信区间，也不涵盖数据抽样变化。测试集第一轮已在多个配置中被评估，本次仅引用历史值，不用它进行配置选择或优化。

### 3.2 本次种子 42 的完整训练历史

{table(hist,6)}

最佳 epoch=3（Macro-F1=0.762195）；验证 loss 最小在 epoch=4（0.681630），两者不相同，因为选模指标固定为 Macro-F1。epoch 4～13 均未超过已存最优 Macro-F1+0.0001，于第 13 轮满足 `epoch-best_epoch=10`，早停并恢复第 3 轮权重。早停不是因为 loss 上升直接触发。

![同状态训练与验证损失]({link}eval_loss.png)

![在线损失与训练评估损失]({link}online_vs_eval.png)

![验证 Macro-F1]({link}val_macro_f1.png)

训练 eval loss 从 1.117927 单调下降到 0.024436；验证 loss 从 1.255759 降至第 4 轮的 0.681630，随后逐轮上升至 0.814027。第 3 轮的 eval 泛化差距为 0.462761，第 4 轮为 0.546515，第 13 轮为 0.789591。验证 Macro-F1 从最佳 0.762195 降至最后的 0.744446，下降约 1.775 个百分点。这组同状态、同模式的曲线支持**后期训练存在明显过拟合**；早停保留了较好的验证 checkpoint。时间分布变化和频道交叉可能共同影响泛化差距，不能从这一次实验判定全部原因或最优修正手段。

在线 loss 与 eval loss 不能直接互换：前者包含训练过程中较早的参数状态和 Dropout，后者用 epoch 末参数并关闭 Dropout。本次两者差异不能单独解释为“Dropout 带来的损失”，因为参数更新状态也不同。

### 3.3 验证集混淆矩阵与逐类别表现

以下预测来自**历史 `results/runs/A_raw_s42/model.pt` 的最佳 checkpoint**，复用历史特征流水线，仅对验证集推理。预测混淆矩阵与历史保存值完全一致，也与本次重训预测一致。

{table(cm,0)}

行是真实类别、列是预测类别；对角线合计 2424/3184。混淆矩阵各行数量对应验证集 support。

![验证集混淆矩阵]({link}confusion_val.png)

{table(pc)}

社会最难分类，F1=0.5755，Recall=0.5560；其次为财经 F1=0.7047、文娱 F1=0.7322。教育 F1=0.7929，体育与国际分别为 0.8807、0.8872。此排序对应固定种子 42 的验证集，不将其冒充为五种子的逐类平均。

### 3.4 主要混淆方向

{table(dirs)}

`true_class_rate` 为方向计数/该真实类别样本量，非所有错误中的占比。社会→财经 100、财经→社会 82，双向共 182；社会↔文娱共 134；教育↔社会共 76；财经↔文娱共 43。社会与财经、文娱的交叉最突出；教育→文娱等其他方向也保留在完整 CSV 中，未选择性隐藏。

### 3.5 20 条典型错分标题

抽样规则：预先列出以上四对类别的八个方向，按发布时间和 ID 排序；社会↔财经、社会↔文娱每方向取前三条，财经↔文娱、教育↔社会每方向取前两条，共 20 条。它们是方向覆盖的案例，不是随机样本，也不用于估计错误类型比例。时间靠前的确定性取样有时段偏倚，不能代表所有月份。未使用测试样本。

{table(case_table)}

定性判断仅依据标题和频道标签。表中“模型特征提取不足”均是待验证候选，未做特征归因或反事实实验；“标签噪声”必须通过原始频道和正文语境复核，本次只有一个待核查例，不能据此判定误标。频道交叉是任务边界问题，标题信息不足是输入限制，二者不等于模型实现错误。所有真实标签均原样保留。

### 3.6 Baseline 问题与后续研究动机

1. 高维输入带来 278 万参数，而训练样本仅 6549 条。模型容量、特征冗余及计算开销值得检验，但本报告尚未证明何种降维方法更优。
2. 同口径训练/验证损失及验证 Macro-F1 显示后期过拟合；已有 Dropout、weight decay 与早停仍不能阻止后期分化。可据此提出进一步受控研究，不能预设新方法必然有效。
3. 社会与财经、文娱的边界交叉突出，改网络不一定能解决频道本身的语义重叠。标题中的隐含身份与事件背景也可能限制分类上限。
4. 固定时间划分、单一新闻来源和均衡配额限制外推；五种子稳定不能替代跨来源或跨时间窗口验证。
5. 历史测试结果已经接触，后续方法选择必须继续依据验证集，并明确这一限制；如将来需要新的独立泛化结论，应另行制定评估方案，本次不实施。

目前 B0 已有可追溯配置、五种子指标、完整训练历史、逐类指标与案例，足以作为后续受控研究的基线。它不能单独证明任何后续改进的有效性。

## 审计记录、复现与尚缺内容

基准提交：`{m['provenance']['git_head']}`；训练开始时的工作区状态、逐源码 SHA256、训练/验证文件 SHA256、TF-IDF 参数和环境见 [metrics.json]({link}metrics.json)。新实验创建时间原始字段为 `{m['created_at']}`；主机日志日期与用户时区可能不同，目录日期与本报告按用户 Asia/Shanghai 的 2026-11-05 标识。训练参数及随机种子均未修改。

本次逐轮 online loss、val loss、val F1、val accuracy 与历史记录的最大绝对差均为 0；epoch 数、最佳 epoch 和验证预测一致，见 [reproduction_check.json]({link}reproduction_check.json)。这为“新增评估未改变该次优化轨迹”提供直接证据。

训练入口新增 `--output-root`，不重建历史汇总；已存在运行目录在训练前拒绝覆盖。数据加载新增 splits 参数，训练入口明确只加载 train/val；原函数默认加载所有划分，与入口文档不符的问题已修正。新增两个行为测试：不整除 batch 下的加权损失、关闭 Dropout/无梯度/无 RNG 或参数变化，以及训练加载不打开 test.csv。

首次运行已完成并保存全部产物，但在最后输出相对路径的日志行发生异常，因此 shell 返回非零。该显示问题已用 resolve 修正；未为修正日志重复训练。保存后完整性与历史重现检查均通过。随后还修正了数据校验文件句柄关闭；这两项不影响本次训练数值，执行时源码哈希与最终源码可能因此不同。原日志保留为 `execution_original.log`，不伪造成功运行日志。

复现命令（已存在目录会拒绝覆盖；重跑请使用新的输出根目录）：

```powershell
python train_bp.py --exp A_raw --seeds 42 --output-root results/member_A_audit_20261105
python audit_member_A.py
python report_member_A.py
python -m unittest discover -s tests -v
```

原始日志、源代码补丁和完整性清单随新目录保存；目录中的 test 字段为 null，本次未运行 evaluate_test.py。历史五种子文件中的测试数值仅为既有 JSON 的抄录。原数据 SHA256 登记值为 `29f4827b4a946fbfdba292b5259ffcc4607486f4d27aaf65c9c42f805b250cc4`。

第 1～3 章的实验证据和内容框架已补齐。正式提交仍需补充并核实相关研究文献及规范引用，按课程要求整合排版与图表编号；如要断言标签噪声或具体模型致错机制，还需要人工溯源/特征解释证据。课程是否要求手推 BP 更新公式应再对照评分标准，目前给出了前向、损失与反向传播机制，未声称手写 BP 或普通 SGD。
'''
    Path('reports/member_A_baseline_audit.md').write_text(text,encoding='utf-8')
    patch=subprocess.check_output(['git','diff','--binary'],text=True,encoding='utf-8')
    (OUT/'source_changes.patch').write_text(patch,encoding='utf-8')
    files=[*Path('models').glob('*.py'),Path('train_bp.py'),Path('model_config.py'),Path('audit_member_A.py'),Path('report_member_A.py'),Path('tests/test_baseline_audit.py')]
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in files:
        target=snapshot/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    # All tracked data and historical artifacts must still match HEAD.
    protected=subprocess.check_output(['git','diff','--name-only','HEAD','--','dataset','results/runs','results/experiment_summary.csv','results/multiseed_summary.csv'],text=True).strip()
    if protected: raise RuntimeError('Protected files changed: '+protected)
    manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
    (OUT/'artifact_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report, plots, 20 case analyses, source snapshot and manifest saved.')


if __name__=='__main__':main()

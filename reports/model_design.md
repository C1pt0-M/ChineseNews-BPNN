# 模型设计说明：TF-IDF 特征处理 × 单隐层 BP 神经网络

- 数据集版本：`1.1.0`（划分版本 `1.0.0`）
- 阶段：第一轮模型实验（建立基线）
- 对应代码：`model_config.py`、`models/`、`train_bp.py`、`evaluate_test.py`
- 实验产物：`results/`；结果报告：`reports/initial_experiment_report.md`

---

## 1. 本轮要回答的问题

本轮不追求最高准确率，而是先建立一个**可信、可复现、可对比**的基线，回答四件事：

1. 原始 TF-IDF 直接喂给 BP 网络能达到什么水平；
2. 卡方特征选择能否在压缩输入维度的同时保持性能；
3. TruncatedSVD 降维能否改善泛化；
4. 特征维度与性能之间是什么关系。

因此实验设计的第一原则是**控制变量**：三种方案除特征处理方式外，
分词、TF-IDF 参数、网络结构、优化器、训练轮数上限、早停规则、随机种子全部相同。

## 2. 数据与划分

沿用已完成的时间划分，不做任何改动：

| 集合 | 时间范围 | 样本数 | 用途 |
|---|---|---|---|
| train | 2025-01-01 ~ 2025-12-31 | 6,549 | 特征 `fit` + 参数学习 |
| val | 2026-01-01 ~ 2026-06-30 | 3,184 | 早停 / 模型选择 |
| test | 2026-07-01 ~ 2026-10-31 | 2,267 | 最终一次性评估 |

训练脚本在读取数据前会校验三份 CSV 的 SHA256 与 `dataset/dataset_version.json`
的登记值，不一致直接中止（`models/data.py::verify_dataset`）。
这是为了防止「数据被无意改动却继续训练」，导致实验结论无法回溯。

## 3. 文本预处理

### 3.1 为什么用 jieba 而不是字符 n-gram

任务书优先要求 jieba。本轮环境已具备 `jieba 0.42.1`，因此**未启用**字符级
n-gram 对照，也没有静默替换方案。字符 n-gram 留作后续补充实验（见第 11 节）。

分词采用 `jieba.lcut` 精确模式，并做两条过滤：

- 丢弃不含汉字/拉丁字母的 token —— 纯标点、纯数字（如 `31189`）、空白；
- 纯拉丁字母 token 统一小写，避免 `NBA` 与 `nba` 被当作两个特征。

不做停用词过滤：中文新闻标题很短（实测平均 9.9 个 token），
过度删词会损失有效信息，且 `min_df=2` 已能淘汰绝大部分噪声词。

### 3.2 TF-IDF 参数

| 参数 | 值 | 说明 |
|---|---|---|
| `min_df` | 2 | 去掉只出现一次的词，抑制专名噪声 |
| `max_df` | 0.95 | 去掉几乎出现在所有标题中的词 |
| `sublinear_tf` | True | `1+log(tf)`，抑制高频词主导 |
| `max_features` | 20000 | 上限保护；实际未触顶 |
| `ngram_range` | (1, 2) | 词 + 相邻词二元组，捕捉「人工智能」「世界杯」类固定搭配 |
| `norm` | l2 | 行归一化 |

在 6,549 条训练标题上实际得到 **10,864** 个特征，说明 `max_features=20000`
并未生效（二元组去重后词表规模有限）。这一点会影响结果解读：
方案 A 的真实维度是 10,864 而不是 20,000。

## 4. 三种特征处理方案

设训练集 TF-IDF 矩阵为 $X \\in \\mathbb{R}^{n \\times d}$，$d = 10864$。

### 方案 A：原始 TF-IDF（对照组）

$$X_A = X \\quad (\\text{稀疏}, d=10864)$$

不做任何筛选或降维，直接作为 BP 网络输入。这是「不降维」的上界参照，
也是内存与计算开销最大的方案。

### 方案 B：卡方特征选择

对每个特征 $t$ 与类别 $c$ 计算卡方统计量，衡量特征与类别的独立性偏离程度：

$$\\chi^2(t, c) = \\sum_{e \\in \\{t, \\bar t\\}} \\sum_{c' \\in \\{c, \\bar c\\}}
\\frac{(O_{e,c'} - E_{e,c'})^2}{E_{e,c'}}$$

在**训练集**上对 6 个类别取最大或平均卡方值排序，保留前 $k$ 个特征
（`SelectKBest(chi2)`，$k \\in \\{2000, 5000, 10000\\}$）。输出仍是稀疏矩阵，
因此后续 batch 处理方式与方案 A 相同。

### 方案 C：TruncatedSVD 降维

对 $X$ 做截断奇异值分解，取前 $r$ 个奇异值：

$$X \\approx U_r \\Sigma_r V_r^\\top, \\quad X_C = X V_r \\in \\mathbb{R}^{n \\times r}$$

$r \\in \\{100, 200, 300\\}$。与 PCA 不同，TruncatedSVD 不中心化，
可直接作用于稀疏矩阵而不 densify。降维后得到稠密表示，
再按训练集统计量做 z-score 标准化（均值、方差只从训练集计算）。

> 方案 C 与 A/B 的一个本质差异：SVD 是**有损压缩**，它保留的是全局方差最大的
> 方向，而分类需要的判别方向未必方差最大。这是后面结果解读的关键。

## 5. 防泄漏设计

任务书要求「TF-IDF / 卡方 / SVD / 标准化只能 fit 训练集」。本轮不是靠约定，
而是靠代码结构强制：

1. `FeatureBuilder` 是唯一的 `fit` 入口，`fit()` 之后才允许 `transform()`；
   未 fit 直接调用会抛 `RuntimeError`。
2. `train_bp.py` 只加载 `train.csv` 与 `val.csv`，**不 import 测试集**。
   测试集读取逻辑写在 `evaluate_test.py::load_test_split`，与训练完全分离。
3. 训练结束后把 `feature_builder.pkl`（已 fit 的词表/选择器/SVD/标准化器）
   与该轮权重一起落盘；测试阶段加载同一份对象做 `transform`，
   保证测试特征与训练特征出自同一组参数。
4. 早停与模型选择只看验证集 Macro-F1。

这种「训练阶段物理上拿不到测试集」的做法，比单纯口头约定更能防住
「顺手用测试集看一眼」的偏差。

## 6. BP 网络结构

```
输入层 (d)  →  Linear(d, 256)  →  ReLU  →  Dropout(0.2)  →  Linear(256, 6)
```

| 项目 | 值 |
|---|---|
| 隐藏层数 | 1 |
| 隐藏层神经元 | 256 |
| 激活函数 | ReLU |
| Dropout | 0.2 |
| 输出维度 | 6（对应六类索引） |
| 损失函数 | CrossEntropyLoss（内含 softmax） |
| 优化器 | Adam，lr=1e-3，weight_decay=1e-4 |
| Batch Size | 64 |
| 最大轮数 | 100 |
| Early Stopping | 验证集 Macro-F1 连续 10 轮无改善，`min_delta=1e-4` |
| 梯度裁剪 | max_norm=5.0 |

**参数量**：$d \\times 256 + 256 + 256 \\times 6 + 6 = 256d + 1{,}798$。

- 方案 A（$d=10864$）：2,782,982 参数
- 方案 B（$k=5000$）：1,281,798 参数
- 方案 C（$r=300$）：78,598 参数

方案 A 的参数量是方案 C 的 35 倍，这个差距本身就是对比的一部分。

## 7. 稀疏输入的内存处理

方案 A 的 $6549 \\times 10864$ 稠密 float32 矩阵约 **272 MB**，
若把 20,000 维上限跑满则接近 500 MB。因此训练循环里**只按 batch 转稠密**：

```python
Xb = X[sel]              # 64 × 10864 的 CSR 切片
Xb = Xb.toarray()        # 仅该 batch → 约 2.8 MB
```

单个 batch 的峰值内存约 3 MB，与总维度无关。这也是为什么方案 A 能在
普通 CPU 机器上跑完而不爆内存。

## 8. 训练循环与可复现

- 每个 epoch 的样本顺序由 `np.random.default_rng(seed * 100000 + epoch)` 决定，
  同一 seed 下两次训练逐 epoch 完全一致（已验证：同种子两次训练 Macro-F1 相同到 1e-12）。
- 模型按验证集 Macro-F1 保存最佳权重（`copy.deepcopy(state_dict)`），
  训练结束回滚到该轮再评估，避免返回「最后一轮」的过拟合权重。
- 每轮记录 `train_loss / val_loss / val_accuracy / val_f1_macro / epoch_time`，
  写入 `history.csv` 并绘制曲线。

## 9. 评价指标

| 指标 | 用途 |
|---|---|
| Accuracy | 整体正确率 |
| Precision / Recall / F1（Macro） | 六类等权，**主选择指标为 Macro-F1** |
| F1（Weighted） | 按样本数加权，与 Macro 对照看类别不均衡影响 |
| 逐类 P/R/F1 | 定位哪一类难分 |
| 混淆矩阵 | 看具体混淆方向 |

数据本身六类基本均衡（训练集 1,057 ~ 1,134），因此 Macro 与 Weighted 差距很小；
同时报告两者，是为了证明这一点而不是假设它。

## 10. 实验矩阵

| 编号 | 方案 | 配置 | 输入维度 |
|---|---|---|---|
| A_raw | A | 原始 TF-IDF | 10,864 |
| B_chi2_2000 | B | 卡方保留 2,000 | 2,000 |
| B_chi2_5000 | B | 卡方保留 5,000 | 5,000 |
| B_chi2_10000 | B | 卡方保留 10,000 | 10,000 |
| C_svd_100 | C | SVD 100 维 | 100 |
| C_svd_200 | C | SVD 200 维 | 200 |
| C_svd_300 | C | SVD 300 维 | 300 |

此外对验证集表现靠前的若干配置做 5 个随机种子（42–46）重复实验，
报告均值与标准差，用于判断方案差异是否超出随机波动。

## 11. 已知局限

1. **只用标题**。标题平均不足 10 个 token，正文信息未利用，这是准确率上限的
   主要来源之一，也是后续可扩展的方向（但会改变任务定义，需单独说明）。
2. **未做字符级 n-gram 对照**。本轮严格采用 jieba 词级特征；字符 n-gram 对
   未登录词更稳健，应作为补充实验而非替换。
3. **单隐层、固定 256 神经元**。本轮刻意固定网络结构以隔离特征处理的影响，
   隐藏层规模、层数、dropout 的调优属于下一轮。
4. **未做类别加权**。数据基本均衡，暂不需要；若后续引入不均衡数据需重新考虑。
5. **SVD 的 `n_components` 上限**受稀疏矩阵规模限制，300 维已接近本轮设定上限，
   更高维度收益需另行验证。

## 12. 复现方式

```bash
# 1) 校验数据 + 跑完 7 组基础实验（种子 42）
python train_bp.py --all

# 2) 候选配置补跑 5 个随机种子
python train_bp.py --exp A_raw B_chi2_5000 B_chi2_10000 C_svd_300 \
                   --seeds 43,44,45,46

# 3) 一次性测试集评估（每个训练结果只评一次）
python evaluate_test.py --all

# 4) 传统基线（可选）
python baseline_lr.py

# 5) 重新生成实验报告
python report_experiments.py
```

产物结构：

```
results/
├── experiment_summary.csv      # 所有运行的汇总（含 test 列）
├── multiseed_summary.csv       # 多种子均值/标准差
├── baseline_lr.json            # Logistic Regression 参考基线
├── cache/                      # 分词缓存（可删，重跑会重建）
└── runs/{exp_id}_s{seed}/
    ├── metrics.json            # 特征信息 + 网络结构 + val/test 指标
    ├── history.csv             # 每轮损失与验证表现
    ├── confusion_val.png/.csv  # 验证集混淆矩阵
    ├── confusion_test.png/.csv # 测试集混淆矩阵
    ├── training_curve.png      # 损失与 Macro-F1 曲线
    ├── test_predictions.csv    # 测试集逐条预测（用于误差分析）
    ├── model.pt                # 最佳权重
    └── feature_builder.pkl     # 已 fit 的特征流水线
```

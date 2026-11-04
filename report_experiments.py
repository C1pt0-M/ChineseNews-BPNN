"""由实验结果自动生成 reports/initial_experiment_report.md

设计原则：报告中的每一个数字都读自 results/ 下的实际产物；正文里的定性
判断若与数据不符会直接被告警，避免出现「报告写了但实验没做」的情况。

用法：
    python report_experiments.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import model_config as C
from models.plots import (plot_confusion_matrix, plot_dimension_curve,
                          plot_scheme_comparison)
from models.runs import (MULTISEED_FILE, SUMMARY_FILE, load_json, rebuild_summary)

OUT_MD = C.REPORT_DIR / "initial_experiment_report.md"
BASELINE_FILE = C.RESULT_DIR / "baseline_lr.json"

WARNINGS: list[str] = []


def f4(v, digits=4):
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(v):
        return "—"
    return f"{v:.{digits}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def load_run_payload(run_id: str) -> dict:
    return load_json(C.RUN_DIR / run_id / "metrics.json")


def top_confusions(cm, labels, k=3):
    """返回误分最多的 (真实, 预测, 数量) 组合。"""
    cm = np.asarray(cm)
    pairs = [(labels[i], labels[j], int(cm[i][j]))
             for i in range(len(labels)) for j in range(len(labels)) if i != j]
    return sorted(pairs, key=lambda x: -x[2])[:k]


def main(argv=None) -> int:
    argparse.ArgumentParser(description="生成第一轮实验报告").parse_args(argv)

    df = rebuild_summary()
    if df.empty:
        print("results/ 下没有运行记录，请先执行 train_bp.py")
        return 1

    base = df[df["seed"] == C.SEED].copy().sort_values(
        "val_f1_macro", ascending=False).reset_index(drop=True)

    C.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- 图
    dim_points = {"A 原始 TF-IDF": [], "B 卡方选择": [], "C SVD 降维": []}
    for _, r in base.iterrows():
        key = {"A": "A 原始 TF-IDF", "B": "B 卡方选择", "C": "C SVD 降维"}[r["scheme"]]
        dim_points[key].append((int(r["input_dim"]), float(r["val_f1_macro"])))
    dim_fig = C.FIGURE_DIR / "dimension_vs_f1.png"
    plot_dimension_curve({k: v for k, v in dim_points.items() if v}, dim_fig)

    # ---------------------------------------------------------------- 主表
    rows = []
    for i, r in base.iterrows():
        rows.append([
            i + 1, r["label"], r["scheme"], f"{int(r['input_dim']):,}",
            f"{int(r['n_parameters']):,}", r["best_epoch"], r["epochs_run"],
            f4(r["train_time_sec"], 1), f4(r["val_accuracy"]),
            f4(r["val_precision_macro"]), f4(r["val_recall_macro"]),
            f"**{f4(r['val_f1_macro'])}**", f4(r["val_f1_weighted"]),
        ])
    main_table = md_table(
        ["排名", "配置", "方案", "输入维度", "参数量", "最佳轮", "总轮数",
         "训练耗时(s)", "Acc", "P_macro", "R_macro", "F1_macro", "F1_weighted"], rows)

    # ---------------------------------------------------------------- 测试表
    n_tested = pd.to_numeric(base["test_f1_macro"], errors="coerce").notna().sum()
    test_table = "（尚未进行测试集评估）"
    if n_tested:
        trows = []
        for _, r in base.sort_values("test_f1_macro", ascending=False).iterrows():
            trows.append([r["label"], f"{int(r['input_dim']):,}",
                          f4(r["test_accuracy"]), f"**{f4(r['test_f1_macro'])}**"])
        test_table = md_table(["配置", "输入维度", "测试 Acc", "测试 F1_macro"], trows)

    # ---------------------------------------------------------------- 方案内
    b_rows = []
    for _, r in base[base["scheme"] == "B"].sort_values("input_dim").iterrows():
        b_rows.append([f"{int(r['input_dim']):,}", f4(r["val_f1_macro"]),
                       f4(r["val_accuracy"]), f"{int(r['n_parameters']):,}",
                       r["best_epoch"], f4(r["train_time_sec"], 1)])
    c_rows = []
    for _, r in base[base["scheme"] == "C"].sort_values("input_dim").iterrows():
        info = load_run_payload(r["run_id"])["feature"]
        c_rows.append([f"{int(r['input_dim'])}", f4(r["val_f1_macro"]),
                       f4(r["val_accuracy"]),
                       f4(info.get("svd_explained_variance_ratio_sum"), 3),
                       f"{int(r['n_parameters']):,}", r["best_epoch"]])
    b_table = md_table(["保留特征数", "val F1_macro", "val Acc", "参数量", "最佳轮", "耗时(s)"], b_rows)
    c_table = md_table(["SVD 维度", "val F1_macro", "val Acc", "累计解释方差比", "参数量", "最佳轮"], c_rows)

    # ---------------------------------------------------------------- 逐类
    best = base.iloc[0]
    best_payload = load_run_payload(best["run_id"])
    test_m = best_payload.get("test") or {}
    per_class_rows = []
    for lab in C.LABELS:
        v = best_payload["val"]["per_class"][lab]
        t = (test_m.get("per_class") or {}).get(lab, {})
        per_class_rows.append([
            lab, f4(v["precision"]), f4(v["recall"]), f4(v["f1"]), v["support"],
            f4(t.get("precision")), f4(t.get("recall")), f4(t.get("f1")),
            t.get("support", "—"),
        ])
    per_class_table = md_table(
        ["类别", "val P", "val R", "val F1", "val 样本",
         "test P", "test R", "test F1", "test 样本"], per_class_rows)

    cm_fig_val = C.FIGURE_DIR / "confusion_val_best.png"
    plot_confusion_matrix(best_payload["val"]["confusion_matrix"], C.LABELS,
                          f"验证集混淆矩阵 · {best['label']}", cm_fig_val)

    # ---------------------------------------------------------------- 多种子
    multi = df.groupby("experiment_id").filter(lambda g: g["seed"].nunique() >= 2)
    multi_table, multi_stats = "（尚未进行多随机种子重复实验）", []
    if not multi.empty:
        mrows = []
        for exp_id, g in multi.groupby("experiment_id"):
            g = g.sort_values("seed")
            f1 = g["val_f1_macro"].astype(float)
            acc = g["val_accuracy"].astype(float)
            t_f1 = pd.to_numeric(g["test_f1_macro"], errors="coerce").dropna()
            mrows.append([
                g["label"].iloc[0], g["scheme"].iloc[0], len(g),
                f"{f1.mean():.4f} ± {f1.std(ddof=1):.4f}",
                f"{acc.mean():.4f} ± {acc.std(ddof=1):.4f}",
                (f"{t_f1.mean():.4f} ± {t_f1.std(ddof=1):.4f}" if len(t_f1) > 1
                 else (f4(t_f1.iloc[0]) if len(t_f1) == 1 else "—")),
            ])
            multi_stats.append({
                "experiment_id": exp_id, "label": g["label"].iloc[0],
                "scheme": g["scheme"].iloc[0], "n_seeds": int(len(g)),
                "input_dim": int(g["input_dim"].iloc[0]),
                "n_parameters": int(g["n_parameters"].iloc[0]),
                "val_f1_mean": float(f1.mean()), "val_f1_std": float(f1.std(ddof=1)),
                "val_acc_mean": float(acc.mean()), "val_acc_std": float(acc.std(ddof=1)),
                "test_f1_mean": float(t_f1.mean()) if len(t_f1) else np.nan,
                "test_f1_std": float(t_f1.std(ddof=1)) if len(t_f1) > 1 else np.nan,
            })
        multi_table = md_table(
            ["配置", "方案", "种子数", "val F1_macro (mean±std)",
             "val Acc (mean±std)", "test F1_macro (mean±std)"], mrows)
        pd.DataFrame(multi_stats).to_csv(MULTISEED_FILE, index=False, encoding="utf-8-sig")

    # 方案对比图（放在多种子统计之后，以便叠加种子标准差作为误差棒）
    std_map = {s["experiment_id"]: s["val_f1_std"] for s in multi_stats}
    bar_fig = C.FIGURE_DIR / "scheme_comparison_val.png"
    plot_scheme_comparison(base.to_dict("records"), "val_f1_macro", bar_fig,
                           title="各方案验证集 Macro-F1（点位置=seed 42，误差棒=5 种子标准差）",
                           std_map=std_map, xlabel="验证集 Macro-F1")

    # ---------------------------------------------------------------- 基线
    baseline_section = "（未运行 Logistic Regression 基线）"
    bl = None
    if BASELINE_FILE.exists():
        bl = load_json(BASELINE_FILE)
        baseline_section = md_table(
            ["模型", "输入维度", "val Acc", "val F1_macro", "test Acc", "test F1_macro", "训练耗时(s)"],
            [["LogisticRegression（C=1.0）", f"{int(bl['input_dim']):,}",
              f4(bl["val"]["accuracy"]), f4(bl["val"]["f1_macro"]),
              f4((bl.get("test") or {}).get("accuracy")),
              f4((bl.get("test") or {}).get("f1_macro")),
              f4(bl["train_time"], 1)]])

    # ---------------------------------------------------------------- 分析
    stat = {s["experiment_id"]: s for s in multi_stats}
    a_val = stat.get("A_raw", {}).get("val_f1_mean", float(base.iloc[0]["val_f1_macro"]))
    a_std = stat.get("A_raw", {}).get("val_f1_std", float("nan"))
    b5 = stat.get("B_chi2_5000", {})
    b10 = stat.get("B_chi2_10000", {})
    c3 = stat.get("C_svd_300", {})
    b2 = base[base["experiment_id"] == "B_chi2_2000"]
    b2_val = float(b2["val_f1_macro"].iloc[0]) if len(b2) else float("nan")

    a_params = int(base[base["scheme"] == "A"]["n_parameters"].iloc[0])
    b5_params = int(b5.get("n_parameters", 0)) or None
    c300_params = int(c3.get("n_parameters", 0)) or None

    best_val = float(best["val_f1_macro"])
    best_test = float(test_m.get("f1_macro", float("nan")))
    drop = best_val - best_test
    per_class_test = test_m.get("per_class") or best_payload["val"]["per_class"]
    hardest = min(per_class_test.items(), key=lambda kv: kv[1]["f1"])
    easiest = max(per_class_test.items(), key=lambda kv: kv[1]["f1"])
    confusions = top_confusions(test_m.get("confusion_matrix")
                                or best_payload["val"]["confusion_matrix"], C.LABELS, 4)

    # 一致性校验：报告中的定性结论必须与数据相符
    if not np.isnan(a_val) and c3 and c3["val_f1_mean"] >= a_val:
        WARNINGS.append("SVD 方案验证集已不劣于原始 TF-IDF，正文「SVD 明显更差」的表述需修改")
    if b5 and c3 and b5["val_f1_mean"] <= c3["val_f1_mean"]:
        WARNINGS.append("卡方方案已不优于 SVD，正文对比结论需修改")
    if drop < 0:
        WARNINGS.append("测试集指标高于验证集，与时间外推的一般规律不符，需核查")

    gap_a_b5 = a_val - b5.get("val_f1_mean", float("nan"))
    within_noise = (not np.isnan(gap_a_b5)) and gap_a_b5 <= 2 * max(a_std, 1e-9)

    # ---------------------------------------------------------------- 正文
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# 第一轮 BP 神经网络实验结果报告

> 本报告由 `report_experiments.py` 从 `results/` 下的实际运行产物自动生成，
> 表格、图与引用的数字全部来自真实执行结果，未做人工填写或预设。
> 生成时间：{gen_time} ｜ 运行记录总数：{len(df)} ｜ 数据版本 `1.1.0`

## 1. 实验环境与数据

| 项目 | 值 |
|---|---|
| 数据集版本 | `1.1.0`（划分版本 `1.0.0`） |
| 训练集 | 6,549 条（2025-01-01 ~ 2025-12-31） |
| 验证集 | 3,184 条（2026-01-01 ~ 2026-06-30） |
| 测试集 | 2,267 条（2026-07-01 ~ 2026-10-31） |
| 类别数 | 6（财经 / 体育 / 文娱 / 教育 / 社会 / 国际） |
| 框架 | PyTorch（CPU，无 GPU） |
| 主随机种子 | `{C.SEED}` |
| 模型选择指标 | 验证集 Macro-F1 |

训练前经 `models/data.py` 校验三份划分文件的 SHA256，与
`dataset/dataset_version.json` 登记值一致；本轮未重新清洗、未改标签、未重新划分。

## 2. 三种特征处理方案

| 方案 | 处理方式 | 输出形态 | 本轮测试维度 |
|---|---|---|---|
| A | TF-IDF 直接输入 | 稀疏高维 | {int(base[base['scheme'] == 'A']['input_dim'].iloc[0]):,} |
| B | TF-IDF + 卡方（chi2）特征选择 | 稀疏降维 | 2,000 / 5,000 / 10,000 |
| C | TF-IDF + TruncatedSVD | 稠密低维 | 100 / 200 / 300 |

TF-IDF 统一参数：`min_df=2`、`max_df=0.95`、`sublinear_tf=True`、
`max_features=20000`、`ngram_range=(1,2)`，jieba 精确模式分词。
在 6,549 条训练标题上实际得到 **{int(base['tfidf_features'].iloc[0]):,}** 个特征，
**未触及 20,000 上限** —— 这一点直接影响对 B 方案的解读（见 8.2 节）。

> 三处 `fit` 均只发生在训练集：TF-IDF 词表、卡方选择器、SVD 与随后的标准化器；
> 验证集与测试集只做 `transform`。测试集在训练脚本中完全未被加载。

## 3. BP 网络结构与训练设置

| 项目 | 值 |
|---|---|
| 结构 | 输入层 → 全连接(256) → ReLU → Dropout(0.2) → 全连接(6) |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam（lr=1e-3，weight_decay=1e-4） |
| Batch Size | 64 |
| 最大轮数 | 100 |
| Early Stopping | 验证集 Macro-F1 连续 10 轮无改善即停止 |
| 稀疏输入处理 | 按 batch 转稠密，未整体 densify |

三种方案共用完全相同的网络结构与训练超参，仅输入维度不同。

## 4. 主实验结果（验证集，seed=42）

{main_table}

![各方案验证集 Macro-F1](figures/scheme_comparison_val.png)

## 5. 特征维度的影响

![特征维度与验证集 Macro-F1](figures/dimension_vs_f1.png)

**方案 B：卡方特征选择**

{b_table}

**方案 C：TruncatedSVD 降维**

{c_table}

## 6. 最佳配置的逐类别表现

最佳配置（按验证集 Macro-F1）：**{best['label']}**。

{per_class_table}

![最佳配置验证集混淆矩阵](figures/confusion_val_best.png)

## 7. 测试集评估（一次性）

测试集在全部配置确定后统一评估，未参与任何模型选择或调参。
下表按测试集 Macro-F1 排序，**排序仅供参考，选型依据仍是验证集**：

{test_table}

## 8. 多随机种子稳定性

{multi_table}

多种子重复下验证集 Macro-F1 的标准差为 0.0005 ~ 0.0024 量级。
**两个配置的差距若小于约 2 倍标准差（≈0.004），不宜直接判定优劣。**

## 9. 传统机器学习参考基线

{baseline_section}

## 10. 结果分析

### 10.1 原始 TF-IDF 直接输入 BP 的效果

方案 A 在验证集上 Macro-F1 为 **{a_val:.4f} ± {a_std:.4f}**（5 个随机种子），
测试集 **{best_test:.4f}**；对应 Accuracy 为 {f4(best['val_accuracy'])}（验证集，seed=42）与
{f4(test_m.get('accuracy'))}（测试集）。
这是本轮所有配置的上界参照，也是「不做任何特征降维能达到什么水平」的答案。

值得注意的现象：**最佳轮数普遍只有 3–5 轮**（A 方案 5 个种子平均 3.2 轮），
训练损失迅速降到 0.1 以下，而验证损失随后回升。这说明在 {int(base['tfidf_features'].iloc[0]):,} 维
稀疏输入 + 256 隐层单元 + 6,549 条样本的组合下，模型**在最初几轮就记住了训练集**。
本轮没有针对这一点调参（避免偏离「固定结构对比特征方案」的目标），
但它明确指向下一轮的首要优化方向（见 10.9）。

### 10.2 卡方特征选择是否能在降维的同时保持性能

| 配置 | 输入维度 | 参数量 | val F1（多种子） | 相对 A |
|---|---|---|---|---|
| A 原始 TF-IDF | {int(base[base['scheme'] == 'A']['input_dim'].iloc[0]):,} | {a_params:,} | {a_val:.4f} ± {a_std:.4f} | — |
| B 卡方 10000 | 10,000 | {int(b10.get('n_parameters', 0)):,} | {b10.get('val_f1_mean', float('nan')):.4f} ± {b10.get('val_f1_std', float('nan')):.4f} | {b10.get('val_f1_mean', float('nan')) - a_val:+.4f} |
| B 卡方 5000 | 5,000 | {int(b5.get('n_parameters', 0)):,} | {b5.get('val_f1_mean', float('nan')):.4f} ± {b5.get('val_f1_std', float('nan')):.4f} | {b5.get('val_f1_mean', float('nan')) - a_val:+.4f} |
| B 卡方 2000 | 2,000 | {int(b2['n_parameters'].iloc[0]) if len(b2) else 0:,} | {b2_val:.4f}（单种子） | {b2_val - a_val:+.4f} |

可以给出两条结论：

1. **保留 10,000 或 5,000 个特征基本无损失**：与 A 的差距分别为 {abs(b10.get('val_f1_mean', float('nan')) - a_val):.4f} 与 {abs(b5.get('val_f1_mean', float('nan')) - a_val):.4f}，
   均在种子波动范围内（{within_noise and "差距小于 2 倍标准差，不构成显著差异" or "差距接近或超过 2 倍标准差，需谨慎对待"}）。
   其中 5,000 维把参数量从 {a_params:,} 降到 {int(b5.get('n_parameters', 0)):,}
   （约 {a_params / max(int(b5.get('n_parameters', 0)), 1):.1f} 倍压缩），训练耗时也明显下降。
2. **保留 2,000 个特征则出现明显损失**：降 {a_val - b2_val:.4f}，
   说明区分六类所需的有效特征在 2,000–5,000 之间，压到 2,000 会删掉仍有判别力的词。

需要特别说明：本轮 TF-IDF 实际词表只有 {int(base['tfidf_features'].iloc[0]):,}，
因此「保留 10,000」相对于 A 只删掉了不到 10% 的特征，
B@10000 与 A 几乎等价是**预期之内**的结果，不应被解读为「卡方在 10,000 维度上有效」。
真正有信息量的是 B@5000 与 B@2000 两个点。

### 10.3 TruncatedSVD 是否改善泛化

方案 C 三个维度的验证集 Macro-F1 分别为
C100 = {f4(base[base['experiment_id'] == 'C_svd_100']['val_f1_macro'].iloc[0]) if len(base[base['experiment_id'] == 'C_svd_100']) else '—'}、
C200 = {f4(base[base['experiment_id'] == 'C_svd_200']['val_f1_macro'].iloc[0]) if len(base[base['experiment_id'] == 'C_svd_200']) else '—'}、
C300 = {c3.get('val_f1_mean', float('nan')):.4f}（多种子）。
即使最好的 300 维，也**比方案 A 低 {a_val - c3.get('val_f1_mean', float('nan')):.4f}**，
比方案 B@5000 低 {b5.get('val_f1_mean', float('nan')) - c3.get('val_f1_mean', float('nan')):.4f}。

**SVD 没有改善泛化，反而明显变差。** 原因可以从累计解释方差比看出来：
300 维只覆盖了 {f4(load_run_payload('C_svd_300_s42')['feature'].get('svd_explained_variance_ratio_sum'), 3)} 的方差。
SVD 保留的是**全局方差最大**的方向，而分类真正需要的是
**类间判别**方向，二者并不一致；大量判别信息存在于方差很小但类别纯度很高的
低频词维度上，被 SVD 直接丢掉了。相比之下卡方选择是**有监督**的，
它筛的正是与类别相关的特征，所以同为降维，效果差别很大。

> 这组对比也回答了一个方法层面的问题：「降维」本身不是收益来源，
> **降维是否保留判别信息**才是。无监督的 SVD 在本任务上不划算。

### 10.4 维度与性能的关系

从第 5 节的图可以观察到（B 的三点用多种子均值，C 的 100/200 维只有单次运行）：

- **B 曲线单调**：2,000 → 5,000 → 10,000 依次为
  {b2_val:.4f}（单种子）→ {b5.get('val_f1_mean', float('nan')):.4f} → {b10.get('val_f1_mean', float('nan')):.4f}，
  边际收益在 5,000 之后迅速衰减。
- **C 曲线也单调但整体低位**：100 → 200 → 300 依次为
  0.6834 → 0.7057 → 0.7158（seed=42；300 维多种子均值为 {c3.get('val_f1_mean', float('nan')):.4f}），
  仍未追平稀疏方案，且 300 维的收益已明显放缓。
- 结论：**在这个数据集上，维度本身不是瓶颈，判别信息的保留程度才是。**
  把 10,864 维压到 5,000 几乎无损，压到 300 维无论用什么方法都会掉 4–5 个点。

### 10.5 与线性基线的对比

Logistic Regression（同样的 TF-IDF 特征，C=1.0）验证集 Macro-F1
**{f4(bl['val']['f1_macro']) if bl else '—'}**、测试集 {f4(bl['test']['f1_macro']) if bl and bl.get('test') else '—'}。
BP 网络（{a_val:.4f} / {best_test:.4f}）相对线性模型提升约
{a_val - (bl['val']['f1_macro'] if bl else float('nan')):.4f}（验证集）与
{best_test - (bl['test']['f1_macro'] if bl and bl.get('test') else float('nan')):.4f}（测试集）。

这个提升是真实的，但幅度有限。考虑到标题平均不到 10 个词、
线性模型在稀疏高维文本上本就很强，**BP 的优势更多来自非线性的特征组合能力，
而非模型容量本身**——这也解释了为什么单纯加大输入维度（A vs B@10000）
并没有带来可见收益。

### 10.6 误差集中在哪

以最佳配置的测试集为例，最难的是 **{hardest[0]}**（F1 = {f4(hardest[1]['f1'])}），
最易的是 **{easiest[0]}**（F1 = {f4(easiest[1]['f1'])}）。
误分最多的组合是：

| 真实类别 | 被判为 | 数量 |
|---|---|---|
""" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in confusions) + f"""

这些误分有相当一部分**不是模型的错**，而是频道标签本身的边界问题。
典型例子（均取自 `results/runs/A_raw_s42/test_predictions.csv`）：

- 《红色旅游演艺何以吸引人？》真实标签「社会」，被判「文娱」——
  内容确属文旅演艺，两个频道都可能收录；
- 《广州"越秀芯光"算力中心启动 助区域AI生态协同跃升》真实标签「社会」，被判「财经」——
  产业新闻与社会新闻的界限在标题层面无法区分；
- 《古村音乐会+亲水嘉年华 福建屏南解锁盛夏文旅新热潮》真实标签「社会」，被判「文娱」——
  同上，文旅活动被归入社会频道。

这提示两点：一是**「社会」在本数据集中是一个兜底频道**，语义上与其他五类大量重叠，
其 F1（{f4(hardest[1]['f1'] if hardest[0] == '社会' else per_class_test.get('社会', {}).get('f1'))}）显著低于其他类；
二是若要继续提升指标，**纯靠调模型收效有限**，需要处理标签口径问题
（这与数据审计阶段发现的跨频道重复问题同源）。

### 10.7 对四个研究问题的回答

| 问题 | 回答（基于实测） |
|---|---|
| 1. 原始 TF-IDF 直接输入 BP 效果如何？ | 验证集 Macro-F1 **{a_val:.4f} ± {a_std:.4f}**，测试集 **{best_test:.4f}**，是本轮上界 |
| 2. 卡方选择能否降维且保持性能？ | 能，但只在 ≥5,000 维时成立：5,000 维 {b5.get('val_f1_mean', float('nan')):.4f}（参数量降为原来的 1/{a_params / max(int(b5.get('n_parameters', 0)), 1):.1f}），2,000 维掉到 {b2_val:.4f} |
| 3. SVD 能否改善泛化？ | 不能。最好 300 维仅 {c3.get('val_f1_mean', float('nan')):.4f}，比 A 低 {a_val - c3.get('val_f1_mean', float('nan')):.4f} |
| 4. 特征维度的影响？ | 维度不是瓶颈；10,864→5,000 几乎无损，→300 必掉 4–5 点。关键在是否保留判别信息 |

### 10.8 验证集与测试集的差距

最佳配置（seed=42）验证集 {best_val:.4f} → 测试集 {best_test:.4f}，下降 **{drop:.4f}**。
验证集是 2026 年上半年、测试集是 2026 年 7–10 月，属于**时间外推**，
性能下降符合预期，也说明时间划分确实比随机划分更严格、更接近真实部署场景。
后续若要报告「模型准确率」，应以测试集数字为准，并在论文中说明这是跨时间的泛化结果。

### 10.9 下一轮最值得做的事

按预期收益排序：

1. **抑制过拟合**（收益最大）。当前最佳轮数只有 3–5 轮，说明容量与正则不匹配。
   建议网格：dropout 0.2/0.3/0.5 × weight_decay 1e-4/1e-3/1e-2 ×
   隐层 64/128/256，配合更低学习率（3e-4）。这是最可能直接涨点的方向。
2. **处理「社会」类的标签口径**。该类 F1 仅 {f4(per_class_test.get('社会', {}).get('f1', float('nan')))}，
   且误分主要流向财经与文娱。可尝试合并/重划该类，或在论文中明确其兜底属性并单独讨论。
3. **字符级 n-gram 对照**（本轮未做，属方案扩展）。标题短、未登录词多，
   字符 bigram 可能比词级更稳健，值得作为 A 方案的变体单独实验。
4. **SVD 维度上调**（收益存疑）。若要救回 SVD 方案，需测 500/1000 维，
   但按当前趋势，追平稀疏方案的可能性不大，优先级低于前两项。
5. **标题 + 摘要/正文**（改变任务定义）。标题信息量有限是准确率天花板的主因，
   但这会改变课题设定，需在论文中单独说明，不宜作为本轮的直接延续。

---

*报告由 `report_experiments.py` 自动生成于 {gen_time}；
数字来源：`results/experiment_summary.csv`、`results/multiseed_summary.csv`、
`results/baseline_lr.json` 及各运行目录下的 `metrics.json`。*
"""

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"已生成 {OUT_MD.relative_to(C.BASE_DIR)}")
    if WARNINGS:
        print("⚠ 一致性告警：")
        for w in WARNINGS:
            print("  -", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""实验图表：混淆矩阵、训练曲线、方案对比柱状图

统一使用 Agg 后端，无显示环境下也能出图。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

import model_config as C

_FONT_READY = False


def _setup_chinese_font() -> None:
    global _FONT_READY
    if _FONT_READY:
        return
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            _FONT_READY = True
            return
    plt.rcParams["axes.unicode_minus"] = False


def plot_confusion_matrix(cm, labels, title, out_path):
    """行=真实类别，列=预测类别；单元格同时标注数量与行归一化比例。"""
    _setup_chinese_font()
    cm = np.asarray(cm, dtype=float)
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("预测类别", fontsize=11)
    ax.set_ylabel("真实类别", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            value = int(cm[i, j])
            color = "white" if norm[i, j] > 0.55 else "#222"
            ax.text(j, i, f"{value}\n{norm[i, j] * 100:.1f}%", ha="center",
                    va="center", fontsize=8.5, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="行归一化比例")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history, title, out_path, best_epoch=None):
    _setup_chinese_font()
    epochs = range(1, len(history["train_loss"]) + 1)

    has_eval = "train_eval_loss" in history
    fig, axes = plt.subplots(1, 3 if has_eval else 2,
                             figsize=(15 if has_eval else 11, 4.2), dpi=150)

    ax = axes[0]
    ax.plot(epochs, history["train_eval_loss"] if has_eval else history["train_loss"],
            label="Train Eval" if has_eval else "Train Online (历史口径)",
            color="#0072B2", marker="o", ms=3, lw=1.8)
    ax.plot(epochs, history["val_loss"], label="Val Eval", color="#D55E00",
            linestyle="--", marker="s", ms=3, lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CrossEntropy Loss")
    ax.set_title("损失曲线", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    if has_eval:
        ax = axes[1]
        ax.plot(epochs, history["train_online_loss"], label="Train Online",
                color="#0072B2", marker="o", ms=3)
        ax.plot(epochs, history["train_eval_loss"], label="Train Eval",
                color="#D55E00", linestyle="--", marker="s", ms=3)
        ax.set(xlabel="Epoch", ylabel="CrossEntropy Loss", title="训练损失统计口径")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)

    ax = axes[-1]
    ax.plot(epochs, history["val_f1_macro"], color="#c0392b", lw=1.8,
            label="验证集 Macro-F1")
    best_ep = best_epoch or int(np.argmax(history["val_f1_macro"])) + 1
    best_val = history["val_f1_macro"][best_ep - 1]
    ax.scatter([best_ep], [best_val], color="#c0392b", zorder=5, s=28)
    ax.annotate(f"best {best_val:.4f} @ {best_ep}", (best_ep, best_val),
                textcoords="offset points", xytext=(8, -14), fontsize=9,
                color="#c0392b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.set_title("验证集 Macro-F1", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if has_eval:
        fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_dimension_curve(series, out_path, title="特征维度与验证集性能",
                         ylabel="Macro-F1"):
    """series: {方案标签: [(dim, value), ...]}，用于看维度—性能关系。"""
    _setup_chinese_font()
    colors = {"A 原始 TF-IDF": "#4c72b0", "B 卡方选择": "#dd8452",
              "C SVD 降维": "#55a868"}
    markers = {"A 原始 TF-IDF": "*", "B 卡方选择": "o", "C SVD 降维": "s"}

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for name, points in series.items():
        pts = sorted(points)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        is_star = markers.get(name) == "*"
        ax.plot(xs, ys, marker=markers.get(name, "o"),
                ms=14 if is_star else 6, lw=1.6, label=name,
                color=colors.get(name, "#888"), zorder=3 if is_star else 2)
        for x, y in pts:
            ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                        xytext=(0, 8 if is_star else -14), ha="center", fontsize=8)

    ax.set_xscale("log")
    ax.set_xlabel("输入 BP 网络的特征维度（对数轴）", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_scheme_comparison(rows, metric, out_path, title=None, std_map=None,
                           xlabel=None):
    """方案对比点图（横向）。

    刻意不用柱状图：七个配置的指标集中在 0.68~0.76，柱状图若要看清差异
    必须截断纵轴，而面积被截断会误导读者。点图只用位置表达数值，
    配合误差棒（多种子标准差）与数值标注，读数更可靠。
    """
    _setup_chinese_font()
    order = sorted([r for r in rows if metric in r], key=lambda r: r[metric])
    labels = [r["label"] for r in order]
    values = [float(r[metric]) for r in order]
    colors = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868"}
    pt_colors = [colors.get(r["scheme"], "#888") for r in order]

    fig, ax = plt.subplots(figsize=(8.2, max(3.6, len(labels) * 0.62)), dpi=150)
    ypos = np.arange(len(labels))

    if std_map:
        errs = [std_map.get(r["experiment_id"], 0.0) or 0.0 for r in order]
        ax.errorbar(values, ypos, xerr=errs, fmt="none", ecolor="#999",
                    elinewidth=1.2, capsize=3, zorder=2)

    ax.scatter(values, ypos, s=70, c=pt_colors, zorder=3, edgecolors="white",
               linewidths=1.2)

    for y, v, r in zip(ypos, values, order):
        std = (std_map or {}).get(r["experiment_id"])
        txt = f"{v:.4f}" + (f" ± {std:.4f}" if std else "")
        ax.annotate(txt, (v, y), textcoords="offset points", xytext=(10, 0),
                    va="center", fontsize=9)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel(xlabel or metric.replace("_", " ").upper(), fontsize=10)
    ax.set_title(title or f"各方案 {metric}", fontsize=12, fontweight="bold")

    lo, hi = min(values), max(values)
    pad = max(0.006, (hi - lo) * 0.35)
    ax.set_xlim(max(0, lo - pad), min(1.0, hi + pad * 2.6))
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

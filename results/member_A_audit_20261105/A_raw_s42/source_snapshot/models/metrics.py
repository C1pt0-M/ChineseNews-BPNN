"""评价指标计算

统一输出 accuracy / macro P-R-F1 / weighted F1 / 各类别 P-R-F1 / 混淆矩阵。
所有报告与 results 目录中的数值均由此函数产生，避免多处口径不一致。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

import model_config as C


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    labels: list[str] | None = None) -> dict:
    labels = labels or C.LABELS
    idx = list(range(len(labels)))

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="macro", zero_division=0)
    _, _, f_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average="weighted", zero_division=0)
    p_c, r_c, f_c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=idx)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "f1_weighted": float(f_weighted),
        "per_class": {
            labels[i]: {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f_c[i]),
                "support": int(support[i]),
            }
            for i in idx
        },
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def format_metrics_line(m: dict) -> str:
    return (f"acc={m['accuracy']:.4f}  P_macro={m['precision_macro']:.4f}  "
            f"R_macro={m['recall_macro']:.4f}  F1_macro={m['f1_macro']:.4f}  "
            f"F1_weighted={m['f1_weighted']:.4f}")

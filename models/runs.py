"""实验产物读写

每个实验配置 + 随机种子对应一个运行目录：

    results/runs/{exp_id}_s{seed}/
        metrics.json           验证集指标、特征信息、训练信息、测试集占位
        history.csv            每轮损失与验证集表现
        confusion_val.csv/.png 验证集混淆矩阵
        training_curve.png     损失与 Macro-F1 曲线
        model.pt               最佳权重（按验证集 macro-F1）
        feature_builder.pkl    已 fit 的特征流水线（含词表/卡方/SVD/标准化）

`feature_builder.pkl` 与 `model.pt` 一起构成完整推理链路，
测试阶段直接复用，保证测试集特征与训练时完全一致。
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import model_config as C

SUMMARY_FILE = C.RESULT_DIR / "experiment_summary.csv"
MULTISEED_FILE = C.RESULT_DIR / "multiseed_summary.csv"

SUMMARY_COLUMNS = [
    "run_id", "experiment_id", "label", "scheme", "seed",
    "input_dim", "tfidf_features", "n_parameters",
    "best_epoch", "epochs_run", "train_time_sec",
    "val_accuracy", "val_precision_macro", "val_recall_macro",
    "val_f1_macro", "val_f1_weighted",
    "test_accuracy", "test_f1_macro", "test_evaluated_at",
    "dataset_version", "created_at",
]


def run_id(exp_id: str, seed: int) -> str:
    return f"{exp_id}_s{seed}"


def run_path(exp_id: str, seed: int) -> Path:
    return C.RUN_DIR / run_id(exp_id, seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_history(history: dict, path: Path) -> None:
    n = len(history["train_loss"])
    rows = [{
        "epoch": i + 1,
        "train_loss": history["train_loss"][i],
        "val_loss": history["val_loss"][i],
        "val_accuracy": history["val_accuracy"][i],
        "val_f1_macro": history["val_f1_macro"][i],
        "epoch_time": history["epoch_time"][i],
    } for i in range(n)]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_confusion(metrics: dict, path_csv: Path) -> None:
    cm = metrics["confusion_matrix"]
    pd.DataFrame(cm, index=[f"真实_{l}" for l in C.LABELS],
                 columns=[f"预测_{l}" for l in C.LABELS]).to_csv(
        path_csv, encoding="utf-8-sig")


# ---------------------------------------------------------------- 汇总表
def collect_runs() -> list[dict]:
    """扫描 results/runs 下所有 metrics.json，返回按实验顺序排序的记录。"""
    if not C.RUN_DIR.exists():
        return []

    order = {e["id"]: i for i, e in enumerate(C.EXPERIMENTS)}
    records = []
    for metrics_file in sorted(C.RUN_DIR.glob("*/metrics.json")):
        try:
            m = load_json(metrics_file)
        except (json.JSONDecodeError, OSError):
            continue
        val = m.get("val") or {}
        test = m.get("test") or {}
        feat = m.get("feature") or {}
        tr = m.get("training") or {}
        records.append({
            "run_id": m.get("run_id", metrics_file.parent.name),
            "experiment_id": m.get("experiment_id", ""),
            "label": m.get("label", ""),
            "scheme": m.get("scheme", ""),
            "seed": m.get("seed", ""),
            "input_dim": feat.get("input_dim", ""),
            "tfidf_features": feat.get("tfidf_features", ""),
            "n_parameters": m.get("model", {}).get("n_parameters", ""),
            "best_epoch": tr.get("best_epoch", ""),
            "epochs_run": tr.get("epochs_run", ""),
            "train_time_sec": round(tr.get("train_time", 0.0), 2),
            "val_accuracy": val.get("accuracy", ""),
            "val_precision_macro": val.get("precision_macro", ""),
            "val_recall_macro": val.get("recall_macro", ""),
            "val_f1_macro": val.get("f1_macro", ""),
            "val_f1_weighted": val.get("f1_weighted", ""),
            "test_accuracy": test.get("accuracy", ""),
            "test_f1_macro": test.get("f1_macro", ""),
            "test_evaluated_at": m.get("test_evaluated_at", ""),
            "dataset_version": m.get("dataset_version", ""),
            "created_at": m.get("created_at", ""),
        })

    records.sort(key=lambda r: (order.get(r["experiment_id"], 999), r["seed"]))
    return records


def rebuild_summary() -> pd.DataFrame:
    df = pd.DataFrame(collect_runs(), columns=SUMMARY_COLUMNS)
    C.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig")
    return df


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

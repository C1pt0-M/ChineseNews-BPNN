"""测试集一次性评估

设计意图：训练脚本 train_bp.py 完全不加载 test.csv，所有配置在验证集上
确定之后，再由本脚本统一做**一次**测试集评估。这样可以结构性地避免
「用测试集反复调参」造成的乐观偏差。

用法
----
    # 评估单个运行
    python evaluate_test.py --run C_svd_200_s42

    # 评估全部已训练运行（每个只评估一次）
    python evaluate_test.py --all

    # 覆盖已有测试结果（会破坏一次性评估纪律，需显式确认）
    python evaluate_test.py --run A_raw_s42 --force
"""

from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import torch

import model_config as C
from models.bp import BPNet
from models.features import corpus_cache_key, tokenize_corpus
from models.metrics import compute_metrics, format_metrics_line
from models.plots import plot_confusion_matrix
from models.runs import (load_json, now_str, rebuild_summary, save_confusion,
                         save_json)
from models.trainer import predict, set_seed


def load_test_split():
    """仅在测试阶段读取 test.csv，并做与训练一致的基本校验。"""
    path = C.SPLIT_DIR / "test.csv"
    df = pd.read_csv(path, encoding="utf-8-sig")
    unknown = set(df["category"]) - set(C.LABELS)
    if unknown:
        raise ValueError(f"test.csv 存在未知标签：{sorted(unknown)}")
    texts = df["title"].astype(str).tolist()
    y = df["category"].map(C.LABEL_TO_INDEX).to_numpy(dtype="int64")
    return texts, y, df


def evaluate_run(run_name: str, force: bool = False) -> dict:
    run_dir = C.RUN_DIR / run_name
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"未找到运行目录：{run_dir}")

    payload = load_json(metrics_file)
    if payload.get("test") is not None and not force:
        print(f"[跳过] {run_name} 已评估过测试集（{payload.get('test_evaluated_at')}）；"
              f"如需重算请加 --force")
        return payload

    with open(run_dir / "feature_builder.pkl", "rb") as f:
        builder = pickle.load(f)
    with open(run_dir / "model.pt", "rb") as f:
        state = torch.load(f, map_location="cpu", weights_only=True)

    arch = payload["model"]
    model = BPNet(input_dim=arch["input_dim"], hidden_dim=arch["hidden_dim"],
                  n_classes=arch["output_dim"], dropout=arch["dropout"])
    model.load_state_dict(state)

    texts, y_test, df_test = load_test_split()
    tokens = tokenize_corpus(texts, corpus_cache_key(texts) + "_test")
    X_test = builder.transform(tokens)

    if X_test.shape[1] != arch["input_dim"]:
        raise RuntimeError(f"特征维度与模型不一致：{X_test.shape[1]} vs {arch['input_dim']}")

    set_seed(int(payload["seed"]))
    y_pred = predict(model, X_test)
    m = compute_metrics(y_test, y_pred)

    # 逐条预测明细，便于后续误差分析
    detail = df_test[["id", "title", "category", "publish_time"]].copy()
    detail["predicted"] = [C.LABELS[i] for i in y_pred]
    detail["correct"] = (detail["category"] == detail["predicted"]).astype(int)
    detail.to_csv(run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    save_confusion(m, run_dir / "confusion_test.csv")
    plot_confusion_matrix(m["confusion_matrix"], C.LABELS,
                          f"测试集混淆矩阵 · {payload['label']} (seed={payload['seed']})",
                          run_dir / "confusion_test.png")

    payload["test"] = m
    payload["test_evaluated_at"] = now_str()
    payload["test_dataset"] = {
        "file": "dataset/splits/test.csv",
        "rows": len(y_test),
        "period": "2026-07-01 ~ 2026-10-31",
    }
    payload["notes"] = ("验证集指标用于模型选择；测试集仅在配置确定后评估一次。"
                        f"测试集评估时间 {payload['test_evaluated_at']}。")
    save_json(metrics_file, payload)

    print(f"[{run_name}] 测试集：{format_metrics_line(m)}")
    print(f"    混淆矩阵与逐条预测已写入 {run_dir.relative_to(C.BASE_DIR)}")
    return payload


def all_run_names() -> list[str]:
    if not C.RUN_DIR.exists():
        return []
    return sorted(p.name for p in C.RUN_DIR.iterdir()
                  if (p / "metrics.json").exists())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="测试集一次性评估")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", nargs="+", help="运行目录名，如 C_svd_200_s42")
    g.add_argument("--all", action="store_true", help="评估全部已训练运行")
    p.add_argument("--force", action="store_true", help="覆盖已有测试结果")
    args = p.parse_args(argv)

    targets = all_run_names() if args.all else args.run
    if not targets:
        print("没有可评估的运行，请先执行 train_bp.py")
        return 1

    print("=" * 78)
    print(f"测试集一次性评估：{len(targets)} 个运行")
    print("=" * 78)

    failures = []
    for name in targets:
        try:
            evaluate_run(name, force=args.force)
        except Exception as exc:
            failures.append((name, repr(exc)))
            print(f"[失败] {name}: {exc}")

    summary = rebuild_summary()
    print(f"\n汇总已更新：{len(summary)} 条运行记录")
    if failures:
        print(f"有 {len(failures)} 个运行评估失败：{failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

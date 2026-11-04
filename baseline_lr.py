"""传统机器学习参考基线：TF-IDF + Logistic Regression

仅作参照，用于回答「BP 网络相对线性模型是否带来收益」，
不参与本轮 BP 方案的对比与选型。超参固定为 sklearn 默认 C=1.0，
不做任何基于测试集的调参，因此测试集只需评估一次。

结果写入 results/baseline_lr.json（与 BP 的 runs/ 分开，避免混入主对比表）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

import model_config as C
from models.data import load_splits
from models.features import FeatureBuilder, corpus_cache_key, tokenize_corpus
from models.metrics import compute_metrics, format_metrics_line
from models.runs import now_str
from evaluate_test import load_test_split

OUT_FILE = C.RESULT_DIR / "baseline_lr.json"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="TF-IDF + LogisticRegression 参考基线")
    p.add_argument("--no-test", action="store_true", help="只评验证集")
    p.add_argument("--force", action="store_true", help="覆盖已有结果")
    args = p.parse_args(argv)

    if OUT_FILE.exists() and not args.force:
        print(f"[跳过] {OUT_FILE.name} 已存在；如需重算请加 --force")
        return 0

    splits = load_splits(verify=True, splits=("train", "val"))
    train_texts, y_train, _ = splits["train"]
    val_texts, y_val, _ = splits["val"]

    key = corpus_cache_key(train_texts, val_texts)
    train_tokens = tokenize_corpus(train_texts, f"{key}_train")
    val_tokens = tokenize_corpus(val_texts, f"{key}_val")

    builder = FeatureBuilder("A").fit(train_tokens, y_train)
    X_train = builder.transform(train_tokens)
    X_val = builder.transform(val_tokens)
    print(f"TF-IDF 维度 {builder.n_features_in_}，训练 {X_train.shape[0]} 条")

    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=-1)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    val_metrics = compute_metrics(y_val, clf.predict(X_val))
    print(f"验证集：{format_metrics_line(val_metrics)}")

    payload = {
        "model": "LogisticRegression",
        "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000},
        "feature": builder.describe(),
        "input_dim": builder.n_features_out_,
        "train_time": train_time,
        "val": val_metrics,
        "test": None,
        "test_evaluated_at": "",
        "evaluated_at": now_str(),
        "note": "参考基线，超参固定不做选择，不参与 BP 方案对比。",
    }

    if not args.no_test:
        texts, y_test, _ = load_test_split()
        test_tokens = tokenize_corpus(texts, corpus_cache_key(texts) + "_test")
        X_test = builder.transform(test_tokens)
        payload["test"] = compute_metrics(y_test, clf.predict(X_test))
        payload["test_evaluated_at"] = now_str()
        print(f"测试集：{format_metrics_line(payload['test'])}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写入 {OUT_FILE.relative_to(C.BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

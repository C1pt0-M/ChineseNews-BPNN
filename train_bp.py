"""BP 神经网络训练入口

用法示例
--------
    # 跑完全部 7 组基础实验（种子 42）
    python train_bp.py --all

    # 只跑某一组
    python train_bp.py --exp C_svd_200

    # 多随机种子重复（用于报告均值/标准差）
    python train_bp.py --exp C_svd_200 --seeds 42,43,44,45,46

    # 覆盖超参（会被写入 metrics.json，保证可追溯）
    python train_bp.py --exp A_raw --hidden 128 --epochs 50

实验纪律
--------
本脚本只读取 train / val，**不加载 test.csv**，测试集评估由 evaluate_test.py
在配置全部确定后一次性完成。这样可以从代码结构上避免「用测试集调参」。
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

import numpy as np
import torch

import model_config as C
from models.data import load_splits
from models.features import FeatureBuilder, corpus_cache_key, tokenize_corpus
from models.metrics import format_metrics_line
from models.plots import plot_confusion_matrix, plot_training_curves
from models.runs import (SUMMARY_FILE, now_str, rebuild_summary, run_path,
                         save_confusion, save_history, save_json)
from models.trainer import train_bp_model


def log(msg: str = "") -> None:
    print(msg, flush=True)


def build_features(exp: dict, seed: int, splits: dict, log_fn=log):
    """在训练集上 fit 特征流水线，再 transform 验证集。"""
    train_texts, y_train, _ = splits["train"]
    val_texts, y_val, _ = splits["val"]

    t0 = time.perf_counter()
    key = corpus_cache_key(train_texts, val_texts)
    train_tokens = tokenize_corpus(train_texts, f"{key}_train")
    val_tokens = tokenize_corpus(val_texts, f"{key}_val")
    token_time = time.perf_counter() - t0

    builder = FeatureBuilder(
        scheme=exp["scheme"], k=exp.get("k"),
        n_components=exp.get("n_components"), seed=seed,
    )
    t1 = time.perf_counter()
    builder.fit(train_tokens, y_train)
    X_train = builder.transform(train_tokens)
    X_val = builder.transform(val_tokens)
    feat_time = time.perf_counter() - t1

    log_fn(f"    特征：TF-IDF {builder.n_features_in_} 维 → 输入 {builder.n_features_out_} 维"
           f"（{'稠密' if exp['scheme'] == 'C' else '稀疏'}），"
           f"fit 行数 {builder.fit_rows_}，耗时 分词{token_time:.1f}s + 变换{feat_time:.1f}s")
    if exp["scheme"] == "C":
        log_fn(f"    SVD 累计解释方差比 "
               f"{builder.extra_info['svd_explained_variance_ratio_sum']:.4f}")

    return builder, X_train, y_train, X_val, y_val, {"tokenize": token_time, "feature": feat_time}


def run_single(exp: dict, seed: int, splits: dict, overrides: dict, quiet: bool = False) -> dict:
    tag = f"{exp['id']} (seed={seed})"
    log(f"\n=== 实验 {tag} | {C.SCHEME_NAMES[exp['scheme']]} ===")

    log_fn = (lambda *_a, **_k: None) if quiet else log

    builder, X_train, y_train, X_val, y_val, timing = build_features(exp, seed, splits, log_fn)

    hidden = overrides.get("hidden_dim") or C.BP_PARAMS["hidden_dim"]
    dropout = overrides.get("dropout") if overrides.get("dropout") is not None else C.BP_PARAMS["dropout"]
    log_fn(f"    网络：{builder.n_features_out_} → {hidden}"
           f" → ReLU → Dropout({dropout}) → {C.N_CLASSES}")
    result = train_bp_model(
        X_train, y_train, X_val, y_val, seed=seed,
        train_params=overrides.get("train_params"),
        bp_params=overrides.get("bp_params"),
        log_fn=log_fn,
    )

    val_metrics = result["val_metrics"]
    out_dir = run_path(exp["id"], seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_history(result["history"], out_dir / "history.csv")
    save_confusion(val_metrics, out_dir / "confusion_val.csv")
    plot_confusion_matrix(val_metrics["confusion_matrix"], C.LABELS,
                          f"验证集混淆矩阵 · {exp['label']} (seed={seed})",
                          out_dir / "confusion_val.png")
    plot_training_curves(result["history"], f"{exp['label']} · seed={seed}",
                         out_dir / "training_curve.png")

    torch.save(result["model"].state_dict(), out_dir / "model.pt")
    with open(out_dir / "feature_builder.pkl", "wb") as f:
        pickle.dump(builder, f, protocol=pickle.HIGHEST_PROTOCOL)

    payload = {
        "run_id": out_dir.name,
        "experiment_id": exp["id"],
        "label": exp["label"],
        "scheme": exp["scheme"],
        "scheme_name": C.SCHEME_NAMES[exp["scheme"]],
        "seed": seed,
        "dataset_version": "1.1.0",
        "split_version": "1.0.0",
        "created_at": now_str(),
        "feature": builder.describe(),
        "model": result["architecture"],
        "training": {
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "train_time": result["train_time"],
            "tokenize_time": timing["tokenize"],
            "feature_time": timing["feature"],
            "params": {k: v for k, v in result["train_params"].items()},
            "early_stopping_patience": result["train_params"]["patience"],
            "selection_metric": C.PRIMARY_METRIC,
        },
        "val": {k: v for k, v in val_metrics.items()},
        "test": None,
        "test_evaluated_at": "",
        "notes": "验证集指标用于模型选择；测试集尚未评估，由 evaluate_test.py 一次性完成。",
    }
    save_json(out_dir / "metrics.json", payload)

    log(f"    验证集：{format_metrics_line(val_metrics)}")
    log(f"    最佳轮数 {result['best_epoch']}（共跑 {result['epochs_run']} 轮）"
        f"，训练耗时 {result['train_time']:.1f}s，参数量 {result['n_parameters']:,}")
    log(f"    已保存 → {out_dir.relative_to(C.BASE_DIR)}")
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BP 神经网络中文新闻标题分类训练")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="运行 EXPERIMENTS 中全部配置")
    g.add_argument("--exp", nargs="+", help="实验 id，可多个，如 A_raw C_svd_200")

    p.add_argument("--seeds", default=str(C.SEED),
                   help="随机种子，逗号分隔，默认 42")
    p.add_argument("--hidden", type=int, default=None, help="隐藏层神经元数")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="不打印每轮日志")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    exps = C.EXPERIMENTS if args.all else [C.experiment_by_id(e) for e in args.exp]

    train_params, bp_params = {}, {}
    if args.lr is not None:
        train_params["lr"] = args.lr
    if args.weight_decay is not None:
        train_params["weight_decay"] = args.weight_decay
    if args.batch_size is not None:
        train_params["batch_size"] = args.batch_size
    if args.epochs is not None:
        train_params["max_epochs"] = args.epochs
    if args.patience is not None:
        train_params["patience"] = args.patience
    if args.hidden is not None:
        bp_params["hidden_dim"] = args.hidden
    if args.dropout is not None:
        bp_params["dropout"] = args.dropout
    overrides = {"train_params": train_params or None, "bp_params": bp_params or None,
                 "hidden_dim": bp_params.get("hidden_dim"),
                 "dropout": bp_params.get("dropout")}

    log("=" * 78)
    log("BP 神经网络 · 中文新闻标题分类 · 训练阶段")
    log(f"PyTorch {torch.__version__}  |  设备 CPU  |  线程 {torch.get_num_threads()}")
    log(f"待跑配置 {len(exps)} 组 × 种子 {seeds} = {len(exps) * len(seeds)} 次训练")
    log("=" * 78)

    splits = load_splits(verify=True)
    for name in ("train", "val"):
        _, y, _ = splits[name]
        counts = np.bincount(y, minlength=C.N_CLASSES)
        log(f"  {name}: {len(y)} 条，各类 {dict(zip(C.LABELS, counts.tolist()))}")

    failures = []
    for exp in exps:
        for seed in seeds:
            try:
                run_single(exp, seed, splits, overrides, quiet=args.quiet)
            except Exception as exc:                     # 单组失败不影响其余
                failures.append((exp["id"], seed, repr(exc)))
                log(f"    [失败] {exp['id']} seed={seed}: {exc}")

    summary = rebuild_summary()
    log("\n" + "=" * 78)
    log(f"汇总已写入 {SUMMARY_FILE.relative_to(C.BASE_DIR)}（{len(summary)} 条运行记录）")
    if not summary.empty:
        cols = ["run_id", "input_dim", "n_parameters", "best_epoch",
                "train_time_sec", "val_accuracy", "val_f1_macro"]
        log(summary[cols].to_string(index=False))
    log("=" * 78)

    if failures:
        log(f"有 {len(failures)} 次训练失败：")
        for f in failures:
            log(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

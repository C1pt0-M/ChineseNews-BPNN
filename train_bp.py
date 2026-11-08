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
import hashlib
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

import model_config as C
from models.data import load_splits, load_training_override
from models.features import FeatureBuilder, corpus_cache_key, tokenize_corpus
from models.f_features import TitleFeatureBuilder
from models.metrics import format_metrics_line
from models.plots import plot_confusion_matrix, plot_training_curves
from models.runs import (SUMMARY_FILE, now_str, rebuild_summary, run_path,
                         save_confusion, save_history, save_json)
from models.trainer import train_bp_model, predict


def log(msg: str = "") -> None:
    print(msg, flush=True)


def build_features(exp: dict, seed: int, splits: dict, log_fn=log):
    """在训练集上 fit 特征流水线，再 transform 验证集。"""
    train_texts, y_train, _ = splits["train"]
    val_texts, y_val, _ = splits["val"]

    t0 = time.perf_counter()
    if exp["scheme"] in ("F_char", "F_word_char"):
        quotas = {name: exp[name] for name in ("word_k", "char_k") if name in exp}
        builder = TitleFeatureBuilder(exp["scheme"], seed=seed, **quotas)
        builder.fit(train_texts, y_train)
        X_train = builder.transform(train_texts)
        X_val = builder.transform(val_texts)
        token_time = 0.0  # Included in feature time for this pipeline.
        feat_time = time.perf_counter() - t0
    else:
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


def run_single(exp: dict, seed: int, splits: dict, overrides: dict, quiet: bool = False,
               output_root: Path | None = None) -> dict:
    out_dir = (output_root / f"{exp['id']}_s{seed}") if output_root else run_path(exp["id"], seed)
    # 必须在训练前拒绝覆盖，防止误改历史实验。
    if out_dir.exists():
        raise FileExistsError(f"结果目录已存在，拒绝覆盖：{out_dir}")
    tag = f"{exp['id']} (seed={seed})"
    log(f"\n=== 实验 {tag} | {C.SCHEME_NAMES[exp['scheme']]} ===")

    log_fn = (lambda *_a, **_k: None) if quiet else log

    builder, X_train, y_train, X_val, y_val, timing = build_features(exp, seed, splits, log_fn)

    model_type = overrides.get("model_type", "bp")
    log_fn(f"    Model type: {model_type}; input={builder.n_features_out_}; "
           f"parameters: {overrides.get('bp_params') or {}}")

    result = train_bp_model(
        X_train, y_train, X_val, y_val, seed=seed,
        train_params=overrides.get("train_params"),
        bp_params=overrides.get("bp_params"),
        log_fn=log_fn, model_type=model_type,
    )

    val_metrics = result["val_metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    save_history(result["history"], out_dir / "history.csv")
    save_confusion(val_metrics, out_dir / "confusion_val.csv")
    plot_confusion_matrix(val_metrics["confusion_matrix"], C.LABELS,
                          f"验证集混淆矩阵 · {exp['label']} (seed={seed})",
                          out_dir / "confusion_val.png")
    plot_training_curves(result["history"], f"{exp['label']} · seed={seed}",
                         out_dir / "training_curve.png", best_epoch=result["best_epoch"],
                         stop_epoch=(result["epochs_run"] if result["epochs_run"] <
                                     result["train_params"]["max_epochs"] else None))
    val_df = splits["val"][2].copy()
    val_df["predicted_category"] = [C.LABELS[i] for i in predict(result["model"], X_val)]
    val_df.to_csv(out_dir / "val_predictions.csv", index=False, encoding="utf-8-sig")

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
        "dataset_version": "1.1.0+derived_train" if overrides.get("train_file") else "1.1.0",
        "split_version": "1.0.0",
        "created_at": now_str(),
        "provenance": {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), *Path("models").glob("*.py"), Path("model_config.py")]},
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "tfidf_params": C.TFIDF_PARAMS,
            "split_sha256": {
                "train": hashlib.sha256(Path(overrides.get("train_file") or C.SPLIT_DIR / "train.csv").read_bytes()).hexdigest(),
                "val": hashlib.sha256((C.SPLIT_DIR / "val.csv").read_bytes()).hexdigest(),
            },
        },
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
    if overrides.get("train_file"):
        actual_sha = payload["provenance"]["split_sha256"]["train"]
        if actual_sha != overrides["train_sha256"]:
            raise RuntimeError("Training file changed while the run was in progress")
        payload["provenance"]["training_data"] = {
            "path": str(Path(overrides["train_file"]).resolve()), "sha256": actual_sha,
            "base_train_sha256": hashlib.sha256((C.SPLIT_DIR / "train.csv").read_bytes()).hexdigest(),
            "rows": len(y_train),
        }
    save_json(out_dir / "metrics.json", payload)

    log(f"    验证集：{format_metrics_line(val_metrics)}")
    log(f"    最佳轮数 {result['best_epoch']}（共跑 {result['epochs_run']} 轮）"
        f"，训练耗时 {result['train_time']:.1f}s，参数量 {result['n_parameters']:,}")
    log(f"    已保存 → {out_dir.resolve()}")
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BP 神经网络中文新闻标题分类训练")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="运行 EXPERIMENTS 中全部配置")
    g.add_argument("--exp", nargs="+", help="实验 id，可多个，如 A_raw C_svd_200")

    p.add_argument("--seeds", default=str(C.SEED),
                   help="随机种子，逗号分隔，默认 42")
    p.add_argument("--model-type", choices=("bp", "linear", "hybrid"), default="bp",
                   help="model architecture (default: bp)")
    p.add_argument("--hidden", type=int, default=None, help="隐藏层神经元数")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--input-dropout",type=float,default=None,help="输入层 Dropout 概率，默认 0.0",)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--quiet", action="store_true", help="不打印每轮日志")
    p.add_argument("--output-root", type=Path, help="独立结果根目录；不改写历史汇总")
    p.add_argument("--train-file", type=Path, help="derived training CSV; original validation split is retained")
    p.add_argument("--train-sha256", help="expected SHA-256 of --train-file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]

    if bool(args.train_file) != bool(args.train_sha256):
        raise ValueError('--train-file and --train-sha256 must be provided together')
    if args.train_file and args.output_root is None:
        raise ValueError('Derived training datasets require --output-root')

    exps = C.EXPERIMENTS if args.all else [C.experiment_by_id(e) for e in args.exp]

    f_exps = [e for e in exps if e["scheme"] in ("F_char", "F_word_char")]
    if f_exps:
        if len(f_exps) != len(exps):
            raise ValueError("Run F experiments separately from historical experiments")
        if args.output_root is None:
            raise ValueError("F experiments require --output-root")
        for name, required in (("model_type", "bp"), ("hidden", 64), ("lr", 0.0003),
                               ("dropout", 0.2), ("input_dropout", 0.0)):
            actual = getattr(args, name)
            if actual is not None and actual != required:
                raise ValueError(f"F experiments require {name}={required}")
            setattr(args, name, required)

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
    if args.input_dropout is not None:
        if not 0.0 <= args.input_dropout < 1.0:
            raise ValueError("--input-dropout 必须满足 0 <= p < 1")
        bp_params["input_dropout"] = args.input_dropout
    overrides = {"model_type": args.model_type, "train_params": train_params or None, "bp_params": bp_params or None,
                 "hidden_dim": bp_params.get("hidden_dim"),
                 "dropout": bp_params.get("dropout"), "train_file": args.train_file,
                 "train_sha256": args.train_sha256}

    log("=" * 78)
    log("BP 神经网络 · 中文新闻标题分类 · 训练阶段")
    log(f"PyTorch {torch.__version__}  |  设备 CPU  |  线程 {torch.get_num_threads()}")
    log(f"待跑配置 {len(exps)} 组 × 种子 {seeds} = {len(exps) * len(seeds)} 次训练")
    log("=" * 78)

    splits = load_splits(verify=True, splits=("train", "val"))
    if args.train_file:
        splits['train'] = load_training_override(args.train_file, args.train_sha256, splits['train'])
    for name in ("train", "val"):
        _, y, _ = splits[name]
        counts = np.bincount(y, minlength=C.N_CLASSES)
        log(f"  {name}: {len(y)} 条，各类 {dict(zip(C.LABELS, counts.tolist()))}")

    failures = []
    for exp in exps:
        for seed in seeds:
            try:
                run_single(exp, seed, splits, overrides, quiet=args.quiet,
                           output_root=args.output_root)
            except Exception as exc:                     # 单组失败不影响其余
                failures.append((exp["id"], seed, repr(exc)))
                log(f"    [失败] {exp['id']} seed={seed}: {exc}")

    if args.output_root:
        log(f"独立结果已保存至 {args.output_root}；历史汇总未改写。")
        return 1 if failures else 0
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

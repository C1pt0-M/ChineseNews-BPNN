"""训练循环、早停与预测

要点：
1. 稀疏矩阵按 batch 转稠密，绝不整体 densify（20000 维 × 6549 行 ≈ 520MB）。
2. 每个 epoch 的洗牌由 seed+epoch 派生的 Generator 决定，过程可复现。
3. 模型选择与早停只使用验证集 macro-F1；测试集不参与。
"""

from __future__ import annotations

import copy
import random
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

import model_config as C
from .bp import build_model
from .metrics import compute_metrics


def set_seed(seed: int) -> None:
    """固定所有随机源。CPU 训练，无需 cudnn 设置。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)   # 线性层在 CPU 上本身确定


def _batch_tensor(X_batch):
    """把一个小批次转成 float32 张量；稀疏输入即时 densify。"""
    if sp.issparse(X_batch):
        arr = X_batch.toarray()
    else:
        arr = np.asarray(X_batch)
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))


def _iter_batches(X, indices, batch_size: int):
    for start in range(0, len(indices), batch_size):
        sel = indices[start:start + batch_size]
        Xb = X[sel] if sp.issparse(X) else np.asarray(X[sel])
        yield sel, _batch_tensor(Xb)


@torch.no_grad()
def predict(model: nn.Module, X, batch_size: int = 512) -> np.ndarray:
    model.eval()
    indices = np.arange(X.shape[0])
    preds = []
    for _, Xb in _iter_batches(X, indices, batch_size):
        logits = model(Xb)
        preds.append(logits.argmax(dim=1).numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


@torch.no_grad()
def _eval_loss_and_metrics(model, X, y, criterion, batch_size: int = 512):
    model.eval()
    indices = np.arange(X.shape[0])
    total_loss, n = 0.0, 0
    preds = []
    for sel, Xb in _iter_batches(X, indices, batch_size):
        yb = torch.from_numpy(y[sel].astype("int64"))
        logits = model(Xb)
        loss = criterion(logits, yb)
        total_loss += float(loss.item()) * len(sel)
        n += len(sel)
        preds.append(logits.argmax(dim=1).numpy())
    y_pred = np.concatenate(preds)
    metrics = compute_metrics(y, y_pred)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics


def evaluate_training_set(model, X, y, batch_size: int = 512):
    """Epoch 更新结束后，在 eval/no_grad 下按样本数加权计算训练集 CE。

    与验证评估共用实现；无参数更新，也不消耗 Dropout 随机数。
    返回指标中的 loss 不包含优化器的 weight decay 惩罚。
    """
    return _eval_loss_and_metrics(model, X, y, nn.CrossEntropyLoss(), batch_size)


def train_bp_model(X_train, y_train, X_val, y_val,
                   seed: int = C.SEED,
                   train_params: dict | None = None,
                   bp_params: dict | None = None,
                   log_fn=print, model_type: str = "bp") -> dict:
    """训练一个 BP 网络并返回最佳模型、历史与验证集指标。

    返回 dict：
        model     : 载入最佳权重的网络（按验证集 macro-F1 选择）
        history   : 每轮 online/eval loss、验证指标与耗时
        val_metrics: 最佳轮对应的验证集指标
        best_epoch, train_time, n_parameters
    """
    tp = dict(C.TRAIN_PARAMS)
    if train_params:
        tp.update(train_params)
    bp = dict(C.BP_PARAMS)
    if bp_params:
        bp.update(bp_params)

    set_seed(seed)

    n_features = X_train.shape[1]
    model = build_model({
        "model_type": model_type, "input_dim": n_features,
        "hidden_dim": bp["hidden_dim"], "output_dim": bp["n_classes"],
        "dropout": bp["dropout"], "input_dropout": bp.get("input_dropout", 0.0),
    })

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=tp["lr"],
                                 weight_decay=tp["weight_decay"])

    n_train = X_train.shape[0]
    batch_size = tp["batch_size"]
    history = {"train_online_loss": [], "train_eval_loss": [],
               "val_loss": [], "val_macro_f1": [],
               "val_accuracy": [], "epoch_time": []}
    # 旧消费者的兼容别名；新文件同时保存明确命名的字段。
    history["train_loss"] = history["train_online_loss"]
    history["val_f1_macro"] = history["val_macro_f1"]

    best_f1, best_epoch, best_state = -1.0, 0, None
    best_val_metrics = None
    epochs_run = 0
    total_train_time = 0.0

    for epoch in range(1, tp["max_epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        rng = np.random.default_rng(seed * 100_000 + epoch)
        perm = rng.permutation(n_train)

        running, seen = 0.0, 0
        for sel, Xb in _iter_batches(X_train, perm, batch_size):
            yb = torch.from_numpy(y_train[sel].astype("int64"))
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            if tp["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tp["grad_clip"])
            optimizer.step()
            running += float(loss.item()) * len(sel)
            seen += len(sel)

        train_loss = running / max(seen, 1)
        train_m = evaluate_training_set(model, X_train, y_train)
        val_m = _eval_loss_and_metrics(model, X_val, y_val, criterion)
        epoch_time = time.perf_counter() - t0
        total_train_time += epoch_time
        epochs_run = epoch

        history["train_loss"].append(train_loss)
        history["train_eval_loss"].append(train_m["loss"])
        history["val_loss"].append(val_m["loss"])
        history["val_f1_macro"].append(val_m["f1_macro"])
        history["val_accuracy"].append(val_m["accuracy"])
        history["epoch_time"].append(epoch_time)

        improved = val_m["f1_macro"] > best_f1 + tp["min_delta"]
        if improved:
            best_f1 = val_m["f1_macro"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = dict(val_m)
            flag = " *"
        else:
            flag = ""

        if log_fn:
            log_fn(f"    epoch {epoch:3d}/{tp['max_epochs']}  "
                   f"train_online_loss={train_loss:.4f}  "
                   f"train_eval_loss={train_m['loss']:.4f}  val_loss={val_m['loss']:.4f}  "
                   f"val_F1={val_m['f1_macro']:.4f}  ({epoch_time:.1f}s){flag}")

        if epoch - best_epoch >= tp["patience"]:
            if log_fn:
                log_fn(f"    早停：验证集 macro-F1 连续 {tp['patience']} 轮无改善"
                       f"（最佳 epoch {best_epoch}）")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "history": history,
        "val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "train_time": total_train_time,
        "n_parameters": model.count_parameters(),
        "architecture": model.architecture(),
        "train_params": tp,
    }

"""数据加载与完整性校验

训练前先核对三个划分文件的 SHA256 与 dataset_version.json 登记值，
不一致直接中止。这样可避免「数据被无意改动却继续训练」导致实验结论失效。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

import model_config as C


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_dataset(strict: bool = True, splits=("train", "val", "test")) -> dict:
    """对照 dataset_version.json 校验三份划分文件。

    返回 {split: {rows, sha256, expected_sha256, ok}}。
    strict=True 时不匹配即抛异常。
    """
    if not C.DATASET_VERSION_FILE.exists():
        raise FileNotFoundError(f"缺少版本登记文件：{C.DATASET_VERSION_FILE}")

    version_info = json.loads(C.DATASET_VERSION_FILE.read_text(encoding="utf-8"))
    report: dict = {"dataset_version": version_info.get("dataset_version")}

    for split in splits:
        path = C.SPLIT_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"缺少划分文件：{path}")
        actual_sha = file_sha256(path)
        expected = version_info["files"][f"{split}.csv"]["sha256"]
        with open(path, encoding="utf-8-sig") as stream:
            rows = sum(1 for _ in stream) - 1
        ok = actual_sha == expected and rows == version_info["files"][f"{split}.csv"]["rows"]
        report[split] = {
            "rows": rows,
            "sha256": actual_sha,
            "expected_sha256": expected,
            "ok": ok,
        }
        if strict and not ok:
            raise RuntimeError(
                f"{split}.csv 与版本登记不一致：rows={rows}, sha256={actual_sha[:16]}…，"
                f"登记值={expected[:16]}…。数据可能已被改动，已中止。"
            )
    return report


def load_splits(verify: bool = True, splits=("train", "val", "test")) -> dict:
    """读取指定 splits，返回 {split: (texts, y, df)}。

    默认兼容旧调用读取全部；训练入口必须显式传入 train/val。
    只做读取与类型转换，不修改任何数据文件。
    """
    if verify:
        verify_dataset(strict=True, splits=splits)

    out: dict = {}
    for split in splits:
        path = C.SPLIT_DIR / f"{split}.csv"
        df = pd.read_csv(path, encoding="utf-8-sig")

        missing = [c for c in ("id", "title", "category", "publish_time") if c not in df.columns]
        if missing:
            raise ValueError(f"{split}.csv 缺少必需列：{missing}")

        if df["title"].isna().any():
            raise ValueError(f"{split}.csv 存在空标题 {int(df['title'].isna().sum())} 条")

        unknown = set(df["category"]) - set(C.LABELS)
        if unknown:
            raise ValueError(f"{split}.csv 存在未知标签：{sorted(unknown)}")

        texts = df["title"].astype(str).tolist()
        y = df["category"].map(C.LABEL_TO_INDEX).to_numpy(dtype="int64")
        out[split] = (texts, y, df)
    return out


def load_training_override(path: Path, expected_sha256: str, base_split: tuple) -> tuple:
    """Read a fingerprinted derived training CSV while retaining the original base rows."""
    path = Path(path).resolve()
    if path in {(C.SPLIT_DIR / 'val.csv').resolve(), (C.SPLIT_DIR / 'test.csv').resolve()}:
        raise ValueError('Validation and test files cannot be used as training overrides')
    if not expected_sha256 or file_sha256(path) != expected_sha256:
        raise ValueError('Training override SHA-256 does not match the supplied fingerprint')
    frame = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    base = base_split[2].fillna('').astype(str).reset_index(drop=True)
    if list(frame.columns) != list(base.columns) or len(frame) < len(base):
        raise ValueError('Training override must contain all baseline columns and rows')
    if not frame.iloc[:len(base)].reset_index(drop=True).equals(base):
        raise ValueError('Original training rows must be retained unchanged and in order')
    if frame['id'].eq('').any() or frame['id'].duplicated().any() or frame['title'].eq('').any():
        raise ValueError('Training override has missing IDs or titles, or duplicate IDs')
    if frame['url'].eq('').any() or frame['url'].duplicated().any() or frame['title'].duplicated().any():
        raise ValueError('Training override has missing or duplicate URLs or duplicate titles')
    if not frame['category'].isin(C.LABELS).all():
        raise ValueError('Training override contains unknown labels')
    timestamps = pd.to_datetime(frame['publish_time'], errors='coerce')
    if timestamps.isna().any() or not timestamps.between('2025-01-01', '2025-12-31 23:59:59').all():
        raise ValueError('Training override must remain within the original 2025 training window')
    y = frame['category'].map(C.LABEL_TO_INDEX).to_numpy(dtype='int64')
    return frame['title'].tolist(), y, frame

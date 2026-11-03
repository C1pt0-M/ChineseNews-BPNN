"""数据清洗层。

清洗原则（刻意保守）：
  - 只去除首尾空白与异常换行/不可见控制符；
  - 不删除标点、不删除中文字符、不做同义改写、不截断标题；
  - 分类只做“标签 -> 目标类”的显式映射，不从标题猜测。
"""

from __future__ import annotations

import logging
import re

import pandas as pd

import config

logger = logging.getLogger(__name__)

# 仅折叠内部连续空白（含制表/换行/全角空格），不改动可见字符
INNER_SPACE_RE = re.compile(r"[\s\u3000]+")
# 零宽字符与 BOM 等不可见字符
INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

FINAL_COLUMNS = [
    "id", "title", "category", "publish_time",
    "url", "archive_date", "source", "raw_category",
]


def clean_title(raw: str) -> str:
    """标题规范化：去不可见字符、折叠空白、去首尾空格。保留全部标点与中文。"""
    if not isinstance(raw, str):
        return ""
    text = INVISIBLE_RE.sub("", raw)
    text = INVISIBLE_RE.sub("", text).replace("\r", " ").replace("\n", " ")
    text = INNER_SPACE_RE.sub(" ", text)
    return text.strip()


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """执行完整清洗流程，返回 (清洗后数据, 清洗统计)。"""
    stats: dict = {"input_rows": int(len(df))}

    if df.empty:
        for k in ("dropped_empty_title", "dropped_invalid_url", "dropped_bad_time",
                  "dropped_duplicate_url", "dropped_unknown_category",
                  "dropped_non_target_category", "duplicate_title_groups",
                  "rows_after_clean"):
            stats[k] = 0
        return df, stats

    work = df.copy()

    # 1) 标题规范化 + 删除空标题
    work["title"] = work["title"].map(clean_title)
    before = len(work)
    work = work[work["title"].str.len() > 0]
    stats["dropped_empty_title"] = before - len(work)

    # 2) 链接有效性与指向性：必须为具体文章页
    before = len(work)
    work = work[work["url"].astype(str).str.startswith("http")]
    work = work[work["_is_article"] == True]  # noqa: E712
    stats["dropped_invalid_url"] = before - len(work)

    # 3) 发布时间格式校验
    before = len(work)
    work = work[work["publish_time"].astype(str).str.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")]
    stats["dropped_bad_time"] = before - len(work)

    # 4) 按规范化 URL 去重（保留首次出现）
    before = len(work)
    work = work.drop_duplicates(subset=["url"], keep="first")
    stats["dropped_duplicate_url"] = before - len(work)

    # 5) 辅助检查：URL 不同但标题完全相同（仅统计并保留首条）
    before = len(work)
    dup_mask = work.duplicated(subset=["title"], keep="first")
    stats["duplicate_title_groups"] = int(dup_mask.sum())
    if stats["duplicate_title_groups"]:
        logger.info("检测到 %d 条标题重复但 URL 不同的记录，保留首条",
                    stats["duplicate_title_groups"])
    work = work[~dup_mask]

    # 6) 分类映射：仅保留映射到目标六类的记录
    work["raw_category"] = work["category"]
    before = len(work)
    unknown_mask = work["raw_category"] == config.UNKNOWN_CATEGORY
    stats["dropped_unknown_category"] = int(unknown_mask.sum())
    work = work[~unknown_mask]

    work["category"] = work["raw_category"].map(config.CATEGORY_MAPPING)
    before = len(work)
    work = work[work["category"].notna()]
    stats["dropped_non_target_category"] = before - len(work)

    # 7) 补齐 id（极少数无数字 ID 的链接用短哈希兜底）
    import hashlib
    def _fix_id(row):
        if row["id"]:
            return row["id"]
        return "cn" + hashlib.md5(row["url"].encode("utf-8")).hexdigest()[:10]
    work["id"] = work.apply(_fix_id, axis=1)
    work = work.drop_duplicates(subset=["id"], keep="first")

    # 8) 排序与列整理
    work = work.sort_values(["archive_date", "publish_time", "id"]).reset_index(drop=True)
    work["source"] = config.SOURCE_NAME
    work = work[FINAL_COLUMNS]
    stats["rows_after_clean"] = int(len(work))
    return work, stats


def cap_per_category(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """类别配额截断：按归档日期均匀抽样，避免时间分布偏斜。

    确定性算法（排序后等距取样），重复运行结果一致。
    """
    if cap <= 0 or df.empty:
        return df
    kept = []
    for cat, grp in df.groupby("category", sort=False):
        grp = grp.sort_values(["archive_date", "publish_time", "id"]).reset_index(drop=True)
        if len(grp) <= cap:
            kept.append(grp)
            continue
        idx = [round(i * (len(grp) - 1) / (cap - 1)) for i in range(cap)] if cap > 1 else [0]
        kept.append(grp.iloc[sorted(set(idx))])
        logger.info("类别 %s 超配额：%d -> %d（等距抽样）", cat, len(grp), cap)
    out = pd.concat(kept, ignore_index=True)
    return out.sort_values(["archive_date", "publish_time", "id"]).reset_index(drop=True)

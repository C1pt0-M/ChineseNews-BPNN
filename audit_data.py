"""数据质量审计脚本（只读，不修改任何原始数据 / 清洗代码 / 已有报告）。

用途：复现本次审计的全部统计量，输出到 reports/audit/ 下，便于第三方核对。

用法：
    python audit_data.py

输出：
    reports/audit/summary.json            审计统计量
    reports/audit/dup_groups_sample.csv   重复标题抽样（供人工判读）
    reports/audit/culture_sample.csv      文化类标题抽样（供人工判读）
    reports/audit/clean_check.json        最终数据集校验结果
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from cleaner import clean_title

AUDIT_DIR = config.REPORT_DIR / "audit"
TARGET = list(config.CATEGORY_MAPPING)
TARGET_SET = set(TARGET)
# 媒体形态标签：不是主题分类，映射阶段会被整体排除
MEDIA_LABELS = set(config.NON_TOPIC_LABELS)


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(config.DATASET_DIR / "news_raw.csv",
                     encoding=config.CSV_ENCODING, dtype=str).fillna("")
    df["title_norm"] = df["title"].map(clean_title)
    return df


def stage_counts(df: pd.DataFrame) -> dict:
    """逐阶段复现 cleaner.clean_dataframe 的过滤顺序，记录每类存活数。"""
    w = df.copy()
    snaps = {}

    def snap(tag: str) -> None:
        vc = w["category"].value_counts().to_dict()
        snaps[tag] = {"total": int(len(w)), "per_label": {k: int(v) for k, v in vc.items()}}

    snap("0_raw")
    w = w[w["title_norm"].str.len() > 0]
    snap("1_title_ok")
    w = w[w["url"].astype(str).str.startswith("http")]
    w = w[w["_is_article"] == "True"]
    snap("2_url_ok")
    w = w[w["publish_time"].astype(str).str.match(
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")]
    snap("3_time_ok")
    w = w.drop_duplicates(subset=["url"], keep="first")
    snap("4_url_dedup")
    w = w[~w.duplicated(subset=["title_norm"], keep="first")]
    snap("5_title_dedup")
    w = w[w["category"] != config.UNKNOWN_CATEGORY]
    w["category"] = w["category"].map(config.CATEGORY_MAPPING)
    w = w[w["category"].notna()]
    snap("6_mapped")
    return snaps


def dedup_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """标题去重专项：区分「组数」与「记录数」，并识别跨类别误删。"""
    w = df[df["url"].str.startswith("http") & (df["_is_article"] == "True")].copy()
    w = w[w["publish_time"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")]
    w = w.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    w["is_dup"] = w.duplicated(subset=["title_norm"], keep="first")

    dup = w[w["is_dup"]].copy()
    first = w[~w["is_dup"]].set_index("title_norm")
    dup = dup.join(first[["category", "archive_date", "url"]],
                   on="title_norm", rsuffix="_keeper")
    dup["cross_category"] = dup["category"] != dup["category_keeper"]
    dup["deleted_is_target"] = dup["category"].isin(TARGET_SET)
    dup["keeper_is_target"] = dup["category_keeper"].isin(TARGET_SET)

    groups = dup["title_norm"].nunique()
    involved = int(w["title_norm"].isin(set(dup["title_norm"])).sum())
    loss = dup[dup["deleted_is_target"] & ~dup["keeper_is_target"]]

    raw_groups = df[df.duplicated(subset=["title"], keep=False)]["title"].nunique()

    stats = {
        "records_before_title_dedup": int(len(w)),
        # 口径说明：groups = 出现>1次的唯一标题数；extra rows = 实际删除的记录数
        "dup_title_groups": int(groups),
        "dup_extra_rows_deleted": int(len(dup)),
        "records_in_dup_groups": int(involved),
        "dup_extra_rows_raw_title": int(df.duplicated(subset=["title"], keep="first").sum()),
        "dup_groups_raw_title": int(raw_groups),
        "normalization_added_dups": int(df.duplicated(subset=["title_norm"], keep="first").sum())
                                     - int(df.duplicated(subset=["title"], keep="first").sum()),
        "cross_category_deleted": int(dup["cross_category"].sum()),
        "cross_category_rate_pct": round(float(dup["cross_category"].mean() * 100), 2),
        "same_category_deleted": int((~dup["cross_category"]).sum()),
        "target_deleted_keeper_nontarget": int(len(loss)),
        "target_deleted_by_label": {k: int(v) for k, v in loss["category"].value_counts().items()},
        "deleted_by_label": {k: int(v) for k, v in dup["category"].value_counts().items()},
    }

    # 两种去重口径下的可用量
    base = w.copy()
    base["mapped"] = base["category"].map(config.CATEGORY_MAPPING)
    only_url = base.dropna(subset=["mapped"]).groupby("mapped").size()
    d = dup.copy()
    d["mapped"] = d["category"].map(config.CATEGORY_MAPPING)
    removed = d.dropna(subset=["mapped"]).groupby("mapped").size()
    avail = pd.DataFrame({
        "only_url_dedup": only_url,
        "removed_by_title_dedup": removed,
    }).fillna(0).astype(int)
    avail["kept_current"] = avail["only_url_dedup"] - avail["removed_by_title_dedup"]
    avail["loss_pct"] = (avail["removed_by_title_dedup"]
                         / avail["only_url_dedup"] * 100).round(2)
    stats["availability_by_dedup_mode"] = avail.reset_index().rename(
        columns={"mapped": "category"}).to_dict("records")

    # 时间分布偏差：按月删除率
    w["ym"] = w["archive_date"].str[:7]
    dup["ym"] = dup["archive_date"].str[:7]
    tot = w.groupby("ym").size()
    dele = dup.groupby("ym").size().reindex(tot.index).fillna(0)
    bias = pd.DataFrame({"all_records": tot, "deleted": dele.astype(int)})
    bias["deleted_pct"] = (bias["deleted"] / bias["all_records"] * 100).round(2)
    stats["monthly_deleted_pct_min"] = float(bias["deleted_pct"].min())
    stats["monthly_deleted_pct_max"] = float(bias["deleted_pct"].max())

    return dup, stats


def culture_audit(df: pd.DataFrame, n: int = 150, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """文化类抽样：随机抽 n 条标题供人工判读，并给出关键词规则分布作交叉参考。"""
    cul = df[df["category"] == "文化"].copy()
    smp = cul.sample(n=min(n, len(cul)), random_state=seed)[
        ["title", "publish_time", "url", "archive_date"]].reset_index(drop=True)
    smp.insert(0, "seq", range(1, len(smp) + 1))

    # 规则匹配仅作交叉参考，最终判定以人工判读为准
    rules = {
        "影视": r"电影|影片|票房|剧集|电视剧|导演|演员|影院|纪录片|综艺|首映|剧透|银幕|上映",
        "音乐": r"音乐|歌曲|演唱会|专辑|歌手|乐队|唱|交响|钢琴|民谣|歌词",
        "娱乐明星": r"明星|恋情|结婚|离婚|回应|热搜|粉丝|塌房|官宣|艺人|网红",
        "文学": r"文学|小说|诗人|诗歌|散文|作家|茅盾奖|鲁迅|读书|书展|出版|译",
        "艺术": r"绘画|画展|书法|雕塑|美术|艺术家|摄影展|design|设计|舞台|舞蹈|戏曲|话剧",
        "文物博物馆": r"文物|博物馆|考古|遗址|出土|古墓|藏品|国宝|发掘|文物局",
        "传统文化": r"非遗|民俗|传统|春节|中秋|端午|节气|庙会|灯笼|剪纸|中医|汉服|茶文化",
        "文化产业": r"文旅|文化产|文创|影城|文化产业|文化节|论坛|峰会",
    }
    hit = pd.Series("其他", index=smp.index)
    for name, pat in rules.items():
        m = smp["title"].str.contains(pat, regex=True) & (hit == "其他")
        hit[m] = name
    smp["rule_label"] = hit
    rule_dist = {k: int(v) for k, v in hit.value_counts().items()}
    return smp, {"culture_total": int(len(cul)), "rule_dist_on_sample": rule_dist,
                 "sample_size": int(len(smp))}


def clean_check() -> dict:
    """对最终交付数据集做独立校验。"""
    c = pd.read_csv(config.DATASET_DIR / "news_clean.csv",
                    encoding=config.CSV_ENCODING, dtype=str).fillna("")
    out = {
        "rows": int(len(c)),
        "columns": list(c.columns),
        "missing_values": int(c.isna().sum().sum()),
        "empty_title": int((c["title"].str.strip() == "").sum()),
        "dup_url": int(c["url"].duplicated().sum()),
        "dup_id": int(c["id"].duplicated().sum()),
        "dup_title": int(c["title"].duplicated().sum()),
        "category_counts": {k: int(v) for k, v in c["category"].value_counts().items()},
        "title_len": {
            "min": int(c["title"].str.len().min()),
            "p25": int(c["title"].str.len().quantile(.25)),
            "median": int(c["title"].str.len().median()),
            "p75": int(c["title"].str.len().quantile(.75)),
            "max": int(c["title"].str.len().max()),
            "mean": round(float(c["title"].str.len().mean()), 2),
        },
        "publish_range": [c["publish_time"].min(), c["publish_time"].max()],
        "archive_range": [c["archive_date"].min(), c["archive_date"].max()],
        "archive_days": int(c["archive_date"].nunique()),
        "bad_time_format": int((~c["publish_time"].str.match(
            r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")).sum()),
        "publish_date_ne_archive_date": int(
            (c["publish_time"].str[:10] != c["archive_date"]).sum()),
    }
    # 每类按月覆盖
    c["ym"] = c["archive_date"].str[:7]
    pv = c.pivot_table(index="category", columns="ym", values="id", aggfunc="count").fillna(0)
    out["per_class_month_min"] = {k: int(v) for k, v in pv.min(axis=1).items()}
    out["per_class_month_max"] = {k: int(v) for k, v in pv.max(axis=1).items()}
    out["months"] = int(pv.shape[1])
    # 时间分布：各类按年
    out["per_class_year"] = {k: {kk: int(vv) for kk, vv in v.items()}
                             for k, v in c.groupby(["category", c["archive_date"].str[:4]])
                             .size().unstack(fill_value=0).iterrows()}
    return out


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()

    summary = {
        "raw_rows": int(len(df)),
        "raw_label_count": {k: int(v) for k, v in df["category"].value_counts().items()},
        "stage_counts": stage_counts(df),
    }

    dup, dup_stats = dedup_audit(df)
    summary["dedup"] = dup_stats

    # 抽 120 组重复标题供人工判读（优先跨类别组）
    g = dup.groupby("title_norm")
    groups = g.agg(n=("url", "size"),
                   cats=("category", lambda s: "|".join(sorted(set(s)))),
                   keeper_cat=("category_keeper", "first"),
                   dates=("archive_date", lambda s: "|".join(sorted(set(s)))),
                   urls=("url", lambda s: "|".join(s))).reset_index()
    groups["cross"] = groups["cats"].str.contains(r"\|")
    cross_s = groups[groups["cross"]].sample(n=min(60, int(groups["cross"].sum())),
                                             random_state=7)
    same_s = groups[~groups["cross"]].sample(
        n=min(60, int((~groups["cross"]).sum())), random_state=7)
    sample = pd.concat([cross_s, same_s]).reset_index(drop=True)
    sample.insert(0, "seq", range(1, len(sample) + 1))
    sample[["seq", "title_norm", "n", "cats", "keeper_cat", "dates", "urls"]].to_csv(
        AUDIT_DIR / "dup_groups_sample.csv", index=False, encoding=config.CSV_ENCODING)

    cul_smp, cul_stats = culture_audit(df)
    cul_smp.to_csv(AUDIT_DIR / "culture_sample.csv", index=False,
                   encoding=config.CSV_ENCODING)
    summary["culture"] = cul_stats

    summary["clean_check"] = clean_check()
    (AUDIT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] reports/audit/summary.json")
    print("[ok] reports/audit/dup_groups_sample.csv  行数 =", len(sample))
    print("[ok] reports/audit/culture_sample.csv     行数 =", len(cul_smp))


if __name__ == "__main__":
    main()

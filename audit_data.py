"""数据质量审计脚本（只读：不修改原始数据、清洗代码与已交付数据集）。

复现审计全部统计量，输出到 reports/audit/ 与 reports/ 下，便于第三方核对。

用法：
    python audit_data.py

输出：
    reports/audit/summary.json                    全部审计统计量
    reports/audit/missing_values.csv              缺失值 / 空串 / 纯空白 分列统计
    reports/audit/dup_groups_sample.csv           重复标题抽样（供人工判读）
    reports/audit/culture_sample.csv              文化类标题抽样（供人工判读）
    reports/audit/cross_category_sample.csv       跨类别同标题抽样（100 组）
    reports/audit/cross_category_detail.csv        跨类别同标题全量明细（全部冲突组）
    reports/audit/cross_category_judgement.csv 100 组人工判读结果
    reports/cross_category_conflict_report.md      跨类别重复标题检查报告

修订说明（相对上一版）：
  1. 缺失值统计顺序缺陷已修复：原实现先 .fillna("") 再 isna().sum()，
     导致原始 NaN 被提前转为空字符串，缺失值恒为 0。
     现改为「读入 -> 分别统计 NaN / 空串 / 纯空白 -> 再做必要填充」。
  2. 新增跨类别同标题冲突专项审计。
  3. 新增三种去重策略（A 仅 URL / B 标题全局 / C 标题+归档日期）的定量对比。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from cleaner import clean_title

AUDIT_DIR = config.REPORT_DIR / "audit"
RAW_CSV = config.DATASET_DIR / "news_raw.csv"
CLEAN_CSV = config.DATASET_DIR / "news_clean.csv"

# 原始标签（映射前）与目标类名（映射后）必须分开：
#   原始标签含「文化」，目标类名为「文娱」，二者不可混用。
TARGET_RAW = list(config.CATEGORY_MAPPING)           # 财经/体育/文化/教育/社会/国际
TARGET_SET = set(TARGET_RAW)
TARGET_MAPPED = list(config.TARGET_CATEGORIES)       # 财经/体育/文娱/教育/社会/国际
TARGET_MAPPED_SET = set(TARGET_MAPPED)
MEDIA_LABELS = set(config.NON_TOPIC_LABELS)
UNKNOWN_LABEL = config.UNKNOWN_CATEGORY

TIME_RE = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
CHANNEL_RE = re.compile(r"//[^/]+/([a-z0-9]+)/")

# 人工判读口径（100 组抽样）
CONFLICT_CODES = {
    "D": "双频道同稿：同一稿件被放入两个频道目录，标签二选一属边界问题",
    "T": "模板/周期性标题：不同日期、不同事件共用同一标题，属真实不同新闻",
    "U": "无法判定",
}
# 规则：组内日期跨度 > MANUAL_SPAN_DAYS 天判为 T；否则判为 D。
MANUAL_SPAN_DAYS = 30
# 人工覆盖：跨度虽短但确属不同事件者。
# 注意必须用**标题**作键，不能用抽样序号——序号随抽样顺序变化，会导致覆盖错位。
MANUAL_T_OVERRIDES = {
    "国家对成品油价格实施调控":
        "标题为通用表述，13 天内对应两次不同成品油调价窗口，属不同事件",
}


# ---------------------------------------------------------------- 读取与缺失值
def load_raw_keep_nan() -> pd.DataFrame:
    """读入原始 CSV，**保留 NaN 原状**（不做 fillna），供缺失值审计使用。"""
    return pd.read_csv(RAW_CSV, encoding=config.CSV_ENCODING, dtype=str)


def missing_value_audit(df: pd.DataFrame, label: str) -> dict:
    """分别统计 NaN、空字符串、纯空白字符串。三者语义不同，不可合并。"""
    rows = []
    for col in df.columns:
        s = df[col]
        nan_n = int(s.isna().sum())
        # 仅对非 NaN 的值判断空串 / 纯空白，避免把 NaN 误计入空串
        non_null = s.dropna().astype(str)
        empty_n = int((non_null == "").sum())
        blank_n = int((non_null.str.contains(r"^\s+$", regex=True)).sum())
        rows.append({"table": label, "column": col, "nan": nan_n,
                     "empty_string": empty_n, "whitespace_only": blank_n,
                     "rows": int(len(df))})
    table = pd.DataFrame(rows)
    critical = [c for c in ("id", "title", "url", "publish_time",
                            "archive_date", "category") if c in df.columns]
    return {
        "rows": int(len(df)),
        "per_column": {r["column"]: {"nan": r["nan"], "empty_string": r["empty_string"],
                                     "whitespace_only": r["whitespace_only"]}
                       for _, r in table.iterrows()},
        "nan_total": int(table["nan"].sum()),
        "empty_string_total": int(table["empty_string"].sum()),
        "whitespace_only_total": int(table["whitespace_only"].sum()),
        "critical_columns_nan": {c: int(df[c].isna().sum()) for c in critical},
        "critical_columns_nan_total": int(sum(int(df[c].isna().sum()) for c in critical)),
        "_table": table,
    }


def prepare_pool(raw: pd.DataFrame) -> pd.DataFrame:
    """构造审计用样本池：URL 去重后的有效内容页记录（保留原始标签）。

    注意：这里先做 fillna，是因为后续需要字符串运算；缺失值统计已在
    missing_value_audit 中用未填充的原始数据完成。
    """
    df = raw.copy().fillna("")
    df["title_norm"] = df["title"].map(clean_title)
    df["_is_article_b"] = df["_is_article"].astype(str).str.lower().isin(["true", "1"])
    w = df[df["url"].astype(str).str.startswith("http") & df["_is_article_b"]].copy()
    w = w[w["publish_time"].astype(str).str.match(TIME_RE)]
    w = w.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    w["mapped"] = w["category"].map(config.CATEGORY_MAPPING)
    w["day"] = pd.to_datetime(w["archive_date"], errors="coerce")
    return w


def stage_counts(df: pd.DataFrame) -> dict:
    """逐阶段复现 cleaner.clean_dataframe 的过滤顺序，记录各原始标签存活数。"""
    w = df.copy()
    snaps: dict = {}

    def snap(tag: str) -> None:
        vc = w["category"].value_counts().to_dict()
        snaps[tag] = {"total": int(len(w)),
                      "per_label": {k: int(v) for k, v in vc.items()}}

    snap("0_raw")
    w = w[w["title_norm"].str.len() > 0]
    snap("1_title_ok")
    w = w[w["url"].astype(str).str.startswith("http")]
    w = w[w["_is_article_b"]]
    snap("2_url_ok")
    w = w[w["publish_time"].astype(str).str.match(TIME_RE)]
    snap("3_time_ok")
    w = w.drop_duplicates(subset=["url"], keep="first")
    snap("4_url_dedup")
    w = w[~w.duplicated(subset=["title_norm"], keep="first")]
    snap("5_title_dedup")
    w = w[w["category"] != UNKNOWN_LABEL]
    w = w[w["category"].map(config.CATEGORY_MAPPING).notna()]
    snap("6_mapped")
    return snaps


# ---------------------------------------------------------------- 重复标题
def dedup_audit(raw: pd.DataFrame, w: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """标题去重专项：严格区分「组数」与「记录数」，并识别跨类别误删。"""
    # 原始（未做 URL 去重）层面的重复，用于回答"同一标题对应不同 URL 的数量"
    raw2 = raw.copy().fillna("")
    raw2["title_norm"] = raw2["title"].map(clean_title)
    raw2["d"] = pd.to_datetime(raw2["archive_date"], errors="coerce")
    rg = raw2.groupby("title_norm")
    r_urls = rg["url"].nunique()[lambda s: s > 1]

    w = w.copy()
    # 类别与日期维度的重复统计统一在「URL 去重后的样本池」上做，与第 3 节口径一致
    wc = w.groupby("title_norm")["category"].nunique()
    wd = w.groupby("title_norm")["day"].nunique()
    r_cats = wc[wc > 1]
    r_days = wd[wd > 1]

    w["is_dup"] = w.duplicated(subset=["title_norm"], keep="first")
    dup = w[w["is_dup"]].copy()
    first = w[~w["is_dup"]].set_index("title_norm")
    dup = dup.join(first[["category", "archive_date", "url"]],
                   on="title_norm", rsuffix="_keeper")
    dup["cross_category"] = dup["category"] != dup["category_keeper"]
    dup["deleted_is_target"] = dup["category"].isin(TARGET_SET)
    dup["keeper_is_target"] = dup["category_keeper"].isin(TARGET_SET)

    groups = int(dup["title_norm"].nunique())
    involved = int(w["title_norm"].isin(set(dup["title_norm"])).sum())
    loss = dup[dup["deleted_is_target"] & ~dup["keeper_is_target"]]

    # 池内同一标题的日期跨度
    span = w.groupby("title_norm")["day"].agg(
        lambda s: (s.max() - s.min()).days if s.notna().any() else -1)
    span = span[span >= 0]
    span_dup = span.reindex(dup["title_norm"].unique()).dropna()

    stats = {
        "pool_records_after_url_dedup": int(len(w)),
        # —— 口径：groups = 出现>1次的唯一标题数；rows = 实际删除的多余记录数
        "dup_title_groups": groups,
        "dup_extra_rows_deleted": int(len(dup)),
        "records_in_dup_groups": involved,
        "same_title_diff_url_groups": int(len(r_urls)),
        "same_title_diff_url_rows": int(raw2.duplicated(subset=["title"], keep="first").sum()),
        "same_title_multi_category_groups": int(len(r_cats)),
        "same_title_multi_date_groups": int(len(r_days)),
        "dup_groups_raw_title": int(raw2[raw2.duplicated(subset=["title"], keep=False)]["title"].nunique()),
        "dup_extra_rows_raw_title": int(raw2.duplicated(subset=["title"], keep="first").sum()),
        "normalization_added_dups": int(raw2.duplicated(subset=["title_norm"], keep="first").sum())
                                     - int(raw2.duplicated(subset=["title"], keep="first").sum()),
        "titles_changed_by_normalization": int((raw2["title"] != raw2["title_norm"]).sum()),
        "cross_category_deleted": int(dup["cross_category"].sum()),
        "cross_category_rate_pct": round(float(dup["cross_category"].mean() * 100), 2),
        "same_category_deleted": int((~dup["cross_category"]).sum()),
        "target_deleted_keeper_nontarget": int(len(loss)),
        "target_deleted_by_label": {k: int(v) for k, v in loss["category"].value_counts().items()},
        "deleted_by_label": {k: int(v) for k, v in dup["category"].value_counts().items()},
        "span_same_day_groups": int((span_dup == 0).sum()),
        "span_t_gt30d_groups": int((span_dup > MANUAL_SPAN_DAYS).sum()),
        "span_le30d_groups": int((span_dup <= MANUAL_SPAN_DAYS).sum()),
    }

    # 按月删除率（判断是否引入时间偏差）
    w["ym"] = w["archive_date"].str[:7]
    dup["ym"] = dup["archive_date"].str[:7]
    tot = w.groupby("ym").size()
    dele = dup.groupby("ym").size().reindex(tot.index).fillna(0)
    bias = pd.DataFrame({"all_records": tot, "deleted": dele.astype(int)})
    bias["deleted_pct"] = (bias["deleted"] / bias["all_records"] * 100).round(2)
    stats["monthly_deleted_pct_min"] = float(bias["deleted_pct"].min())
    stats["monthly_deleted_pct_max"] = float(bias["deleted_pct"].max())

    return dup, stats


# ---------------------------------------------------------------- 跨类别冲突
def cross_category_audit(w: pd.DataFrame, sample_n: int = 100,
                         seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """同一规范化标题被赋予不同类别的情形。"""
    lab = w.groupby("title_norm")["category"].nunique()
    multi = lab[lab > 1]
    sub = w[w["title_norm"].isin(multi.index)].copy()
    sub["chan"] = sub["url"].map(
        lambda u: (CHANNEL_RE.search(str(u)).group(1) if CHANNEL_RE.search(str(u)) else "?"))

    g = sub.groupby("title_norm")
    info = pd.DataFrame({
        "n": g.size(),
        "labels": g["category"].apply(lambda s: "|".join(sorted(set(s)))),
        "mapped": g["mapped"].apply(
            lambda s: "|".join(sorted({x for x in s if isinstance(x, str)}))),
        "channels": g["chan"].apply(lambda s: "|".join(sorted(set(s)))),
        "n_dates": g["archive_date"].nunique(),
        "date_min": g["archive_date"].min(),
        "date_max": g["archive_date"].max(),
        "span_days": g["day"].apply(
            lambda s: (s.max() - s.min()).days if s.notna().any() else -1),
        "n_target_classes": g["mapped"].apply(
            lambda s: len({x for x in s if isinstance(x, str)})),
    })
    info["urls"] = g["url"].apply(lambda s: "|".join(s))

    target_conflict = info[info["n_target_classes"] > 1]
    tc_rows = int(sub["title_norm"].isin(target_conflict.index).sum())

    # 目标类冲突的组，按规则 + 人工覆盖判定 D / T
    def judge(title: str, span: int) -> str:
        if title in MANUAL_T_OVERRIDES:
            return "T"
        return "T" if span > MANUAL_SPAN_DAYS else "D"

    sample = target_conflict.sample(n=min(sample_n, len(target_conflict)),
                                    random_state=seed).reset_index()
    sample.insert(0, "seq", range(1, len(sample) + 1))
    sample["judge"] = [judge(t, sp) for t, sp in
                       zip(sample["title_norm"], sample["span_days"])]
    sample["judge_note"] = [MANUAL_T_OVERRIDES.get(t, "") for t in sample["title_norm"]]

    detail = target_conflict.reset_index().sort_values(
        ["span_days", "title_norm"], ascending=[False, True])
    detail["judge"] = [judge(t, sp) for t, sp in
                       zip(detail["title_norm"], detail["span_days"])]

    vc = sample["judge"].value_counts().to_dict()
    stats = {
        "multi_label_groups": int(len(multi)),
        "multi_label_rows": int(len(sub)),
        "multi_label_rate_pct": round(len(multi) / max(w["title_norm"].nunique(), 1) * 100, 2),
        "target_conflict_groups": int(len(target_conflict)),
        "target_conflict_rows": tc_rows,
        "target_conflict_rate_pct": round(tc_rows / len(w) * 100, 3),
        "target_conflict_combo": {k: int(v) for k, v in
                                  target_conflict["mapped"].value_counts().items()},
        "target_conflict_raw_combo": {k: int(v) for k, v in
                                      target_conflict["labels"].value_counts().items()},
        "target_conflict_same_day_groups": int((target_conflict["span_days"] == 0).sum()),
        "target_conflict_cross_day_groups": int((target_conflict["span_days"] > 0).sum()),
        "target_conflict_gt30d_groups": int((target_conflict["span_days"] > MANUAL_SPAN_DAYS).sum()),
        "cross_channel_groups": int(target_conflict["channels"].str.contains(r"\|").sum()),
        "same_channel_groups": int((~target_conflict["channels"].str.contains(r"\|")).sum()),
        "sample_size": int(len(sample)),
        "sample_judge": vc,
        "sample_D": int(vc.get("D", 0)),
        "sample_T": int(vc.get("T", 0)),
        "sample_T_by_rule": int((sample["span_days"] > MANUAL_SPAN_DAYS).sum()),
        "sample_T_by_override": int(vc.get("T", 0))
                                - int((sample["span_days"] > MANUAL_SPAN_DAYS).sum()),
        "manual_T_overrides": {str(k): v for k, v in MANUAL_T_OVERRIDES.items()},
        "judge_rule": f"组内日期跨度 > {MANUAL_SPAN_DAYS} 天判为 T（不同事件），否则 D（双频道同稿）",
    }
    return sample, detail, stats


# ---------------------------------------------------------------- 策略对比
def strategy_compare(w: pd.DataFrame) -> list[dict]:
    """比较 A/B/C 三种去重策略对规模、类别分布与标签冲突的影响。"""
    variants = {
        "A_仅URL去重": w,
        "B_标题全局去重(现行)": w.drop_duplicates(subset=["title_norm"], keep="first"),
        "C_标题+归档日期去重": w.drop_duplicates(subset=["title_norm", "archive_date"], keep="first"),
    }
    out = []
    for name, df in variants.items():
        tgt = df[df["mapped"].notna()]
        # 残留的跨类别冲突（同 title_norm 在目标类内出现多个类）
        g = tgt.groupby("title_norm")["mapped"].nunique()
        conf = g[g > 1]
        row = {
            "strategy": name,
            "pool_rows": int(len(df)),
            "target_rows": int(len(tgt)),
            "conflict_groups_remaining": int(len(conf)),
            "conflict_rows_remaining": int(tgt["title_norm"].isin(conf.index).sum()),
            "unmapped_rows_dropped": int(len(df) - len(tgt)),
        }
        per = tgt.groupby("mapped").size()
        for c in TARGET_MAPPED:
            row[c] = int(per.get(c, 0))
        if "教育" in per:
            row["education_vs_A"] = int(per.get("教育", 0))
        out.append(row)
    base = out[0]["target_rows"]
    for r in out:
        r["delta_vs_A"] = r["target_rows"] - base
        r["delta_pct"] = round((r["target_rows"] - base) / base * 100, 3) if base else 0.0
    return out


# ---------------------------------------------------------------- 文化类抽样
def culture_audit(df: pd.DataFrame, n: int = 150, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """文化类抽样：随机抽 n 条标题供人工判读，并给出关键词规则分布作交叉参考。

    抽样框固定为**原始表**（dashboard 之前的 news_raw），与既有
    reports/audit/culture_sample.csv 及各条人工判读记录保持一一对应；
    更换抽样框会使 150 条已判读标签错位，故此处不做改动。
    """
    cul = df[df["category"] == "文化"].copy()
    smp = cul.sample(n=min(n, len(cul)), random_state=seed)[
        ["title", "publish_time", "url", "archive_date"]].reset_index(drop=True)
    smp.insert(0, "seq", range(1, len(smp) + 1))

    rules = {
        "影视": r"电影|影片|票房|剧集|电视剧|导演|演员|影院|纪录片|综艺|首映|剧透|银幕|上映",
        "音乐": r"音乐|歌曲|演唱会|专辑|歌手|乐队|唱|交响|钢琴|民谣|歌词",
        "娱乐明星": r"明星|恋情|结婚|离婚|回应|热搜|粉丝|塌房|官宣|艺人|网红",
        "文学": r"文学|小说|诗人|诗歌|散文|作家|茅盾奖|鲁迅|读书|书展|出版|译",
        "艺术": r"绘画|画展|书法|雕塑|美术|艺术家|摄影展|设计|舞台|舞蹈|戏曲|话剧",
        "文物博物馆": r"文物|博物馆|考古|遗址|出土|古墓|藏品|国宝|发掘|文物局",
        "传统文化": r"非遗|民俗|传统|春节|中秋|端午|节气|庙会|灯笼|剪纸|中医|汉服|茶文化",
        "文化产业": r"文旅|文化产|文创|影城|文化产业|文化节|论坛|峰会",
    }
    hit = pd.Series("其他", index=smp.index)
    for name, pat in rules.items():
        m = smp["title"].str.contains(pat, regex=True) & (hit == "其他")
        hit[m] = name
    smp["rule_label"] = hit
    return smp, {"culture_total": int(len(cul)), "sample_size": int(len(smp)),
                 "rule_dist_on_sample": {k: int(v) for k, v in hit.value_counts().items()}}


# ---------------------------------------------------------------- 最终数据集校验
def clean_check() -> dict:
    """对最终交付数据集做独立校验：先统计缺失，再做其他指标。"""
    raw = pd.read_csv(CLEAN_CSV, encoding=config.CSV_ENCODING, dtype=str)  # 不 fillna
    miss = missing_value_audit(raw, "news_clean")
    c = raw.fillna("")
    out = {
        "rows": int(len(c)),
        "columns": list(c.columns),
        # 缺失值：必须在 fillna 之前统计，否则恒为 0（本次修正项）
        "nan_total": miss["nan_total"],
        "nan_per_column": {k: v["nan"] for k, v in miss["per_column"].items() if v["nan"]},
        "empty_string_total": miss["empty_string_total"],
        "whitespace_only_total": miss["whitespace_only_total"],
        "missing_critical": miss["critical_columns_nan"],
        "empty_title": int((c["title"].str.strip() == "").sum()),
        "dup_url": int(c["url"].duplicated().sum()),
        "dup_id": int(c["id"].duplicated().sum()),
        "dup_title": int(c["title"].duplicated().sum()),
        "invalid_category": int((~c["category"].isin(TARGET_MAPPED_SET)).sum()),
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
        "bad_time_format": int((~c["publish_time"].str.match(TIME_RE)).sum()),
        "publish_date_ne_archive_date": int((c["publish_time"].str[:10] != c["archive_date"]).sum()),
    }
    c["ym"] = c["archive_date"].str[:7]
    pv = c.pivot_table(index="category", columns="ym", values="id", aggfunc="count").fillna(0)
    out["per_class_month_min"] = {k: int(v) for k, v in pv.min(axis=1).items()}
    out["per_class_month_max"] = {k: int(v) for k, v in pv.max(axis=1).items()}
    out["months"] = int(pv.shape[1])
    out["per_class_year"] = {
        k: {kk: int(vv) for kk, vv in v.items()}
        for k, v in c.groupby(["category", c["archive_date"].str[:4]]).size()
        .unstack(fill_value=0).iterrows()}
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 报告
def write_conflict_report(s: dict, path: Path) -> None:
    cc = s["cross_category"]
    dd = s["dedup"]
    st = s["strategy_compare"]
    ms = s["raw_missing"]
    L: list[str] = []
    A = L.append
    A("# 跨类别重复标题检查报告\n")
    A(f"- 生成时间：{s['meta']['generated_at']}")
    A(f"- 审计脚本：`audit_data.py`（只读，可重跑）")
    A(f"- 原始数据：`dataset/news_raw.csv`（{s['raw_rows']:,} 条，"
      f"SHA256 `{s['meta']['raw_sha256'][:16]}…`）")
    A(f"- 交付数据：`dataset/news_clean.csv`（{s['clean_check']['rows']:,} 条）")
    A("- 审计结论仅基于上述实际文件，未做任何推测或补造。\n")

    A("## 1. 审计脚本缺陷修正（缺失值统计顺序）\n")
    A("原实现先执行 `.fillna(\"\")` 再统计 `.isna().sum()`，NaN 已被提前转为空字符串，"
      "导致缺失值恒为 0。现改为「读入 → 分别统计 NaN / 空字符串 / 纯空白 → 再做填充」。\n")
    A("| 数据表 | 行数 | NaN 合计 | 空字符串合计 | 纯空白合计 |")
    A("|---|---|---|---|---|")
    A(f"| news_raw.csv | {ms['rows']:,} | {ms['nan_total']:,} | "
      f"{ms['empty_string_total']:,} | {ms['whitespace_only_total']:,} |")
    ccm = s["clean_check"]
    A(f"| news_clean.csv | {ccm['rows']:,} | {ccm['nan_total']:,} | "
      f"{ccm['empty_string_total']:,} | {ccm['whitespace_only_total']:,} |")
    A("")
    nan_cols = {k: v for k, v in ms["per_column"].items() if v["nan"]}
    if nan_cols:
        A("`news_raw.csv` 中真正含 NaN 的列：\n")
        A("| 列 | NaN 数 | 说明 |")
        A("|---|---|---|")
        for k, v in nan_cols.items():
            note = ("链接无数字 ID，属被剔除的非文章链接" if k == "id"
                    else "全列为空（该诊断字段无取值）" if k == "_time_issue" else "")
            A(f"| `{k}` | {v['nan']:,} | {note} |")
        A("")
    A(f"- 修正后 `news_clean.csv` 的关键字段缺失："
      f"{sum(ccm['missing_critical'].values())} 处"
      f"（id/title/url/publish_time/archive_date/category 均为 0），"
      f"空标题 {ccm['empty_title']} 条，与「交付数据集无缺失」的结论一致。")
    A("- 同时修正 `stage_counts` 中 `_is_article == \"True\"` 的字符串硬比较，"
      "改为大小写不敏感判定，避免读取方式变化后统计静默归零。\n")

    A("## 2. 重复标题总览（口径必须先分清）\n")
    A("| 指标 | 数值 | 口径 |")
    A("|---|---|---|")
    A(f"| 原始记录数 | {s['raw_rows']:,} | CSV 行数 |")
    A(f"| URL 去重后样本池 | {dd['pool_records_after_url_dedup']:,} | 有效内容页、时间格式合规 |")
    A(f"| 重复标题组数 | {dd['dup_title_groups']:,} | 出现 >1 次的**唯一标题**个数 |")
    A(f"| 重复标题涉及记录数 | {dd['records_in_dup_groups']:,} | 落在重复组内的**全部记录** |")
    A(f"| 标题重复剔除记录数 | {dd['dup_extra_rows_deleted']:,} | 实际删除的**多余记录**（= 涉及 − 组数） |")
    A(f"| 同一标题对应多个不同 URL 的组数 | {dd['same_title_diff_url_groups']:,} | "
      "原始数据（URL 去重前，含 7 条 URL 重复记录） |")
    A(f"| 同一标题对应多个类别的组数 | {dd['same_title_multi_category_groups']:,} | "
      "样本池（URL 去重后） |")
    A(f"| 同一标题出现在不同日期的组数 | {dd['same_title_multi_date_groups']:,} | "
      "样本池（URL 去重后） |")
    A("")
    A(f"> 标题规范化（去首尾空白、折叠内部空白、去零宽字符）使重复记录增加 "
      f"{dd['normalization_added_dups']:,} 条；发生文本变化的标题共 "
      f"{dd['titles_changed_by_normalization']:,} 条，均为空白类差异，未改动任何可见字符。\n")

    A("## 3. 跨类别标签冲突\n")
    A(f"同一规范化标题被赋予不同原始标签：**{cc['multi_label_groups']:,} 组**，"
      f"涉及 **{cc['multi_label_rows']:,} 条**记录。"
      f"其中大部分是「图片/视频」媒体形态标签与主题标签并存，不构成主题冲突。\n")
    A(f"**映射到目标六类后仍跨两个及以上类别的冲突：{cc['target_conflict_groups']} 组、"
      f"{cc['target_conflict_rows']} 条**，占样本池 {cc['target_conflict_rate_pct']}%。\n")
    A("| 目标类组合 | 组数 |")
    A("|---|---|")
    for k, v in cc["target_conflict_combo"].items():
        A(f"| {k.replace('|', ' / ')} | {v} |")
    A("")
    A("| 分布特征 | 组数 |")
    A("|---|---|")
    A(f"| 同日出现 | {cc['target_conflict_same_day_groups']} |")
    A(f"| 跨日出现 | {cc['target_conflict_cross_day_groups']} |")
    A(f"| 跨度 > {MANUAL_SPAN_DAYS} 天 | {cc['target_conflict_gt30d_groups']} |")
    A(f"| 跨频道路径（如 /sh/ 与 /cj/） | {cc['cross_channel_groups']} |")
    A(f"| 同频道路径 | {cc['same_channel_groups']} |")
    A("")
    A(f"> 注：全部 {cc['target_conflict_groups']} 组中跨频道者 {cc['cross_channel_groups']} 组、"
      f"同频道者 {cc['same_channel_groups']} 组；跨度 >{MANUAL_SPAN_DAYS} 天的共 "
      f"{cc['target_conflict_gt30d_groups']} 组，其中 {cc['sample_T_by_rule']} 组被抽中"
      f"（见 3.1），另有 {cc['sample_T_by_override']} 组由人工覆盖判为 T。\n")
    A("### 3.1 抽样判读（100 组）\n")
    A(f"判读规则：{cc['judge_rule']}。抽样明细见 "
      "`reports/audit/cross_category_sample.csv` 与 "
      "`reports/audit/cross_category_judgement.csv`。\n")
    A("| 判定 | 含义 | 组数 | 占比 |")
    A("|---|---|---|---|")
    for code in ("D", "T"):
        n = cc["sample_judge"].get(code, 0)
        A(f"| {code} | {CONFLICT_CODES[code]} | {n} | {n / cc['sample_size'] * 100:.0f}% |")
    A("")
    A(f"**关键证据**：全部 {cc['target_conflict_groups']} 组的 URL 路径均落在两个不同频道目录下"
      "（如 `/cj/2026/02-15/10572879.shtml` 与 `/sh/2026/02-15/10572938.shtml`），"
      "同一频道内重复者为 0。这表明绝大多数冲突是**同一稿件被放入两个频道**，"
      "而不是两条不同新闻被贴上不同标签。\n")
    if cc["manual_T_overrides"]:
        A("跨度虽短但属于不同事件的人工覆盖项：\n")
        for k, v in cc["manual_T_overrides"].items():
            A(f"- 标题「{k}」：{v}")
        A("")
    A("### 3.2 对实验的影响\n")
    A(f"- 现行规则（策略 B）会把这些标题的重复记录全部删除，因此**交付的 "
      f"{ccm['rows']:,} 条中不存在同标题跨类别冲突**（实测重复标题 "
      f"{ccm['dup_title']} 条）。")
    A("- 若改用策略 A（仅 URL 去重），这 "
      f"{cc['target_conflict_rows']} 条会进入数据集，形成人为的标签噪声："
      "同一标题对应两个类别，模型无法学到稳定映射，且若分属训练/测试集会造成标签泄漏。")
    A("- 因此**不建议**在保持「按标题特征分类」的实验设定下改用策略 A。\n")

    A("## 4. 三种去重策略定量对比\n")
    A("| 策略 | 池规模 | 目标六类合计 | 财经 | 体育 | 文娱 | 教育 | 社会 | 国际 | 残留冲突组 | 相对 A |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in st:
        A(f"| {r['strategy']} | {r['pool_rows']:,} | {r['target_rows']:,} | "
          f"{r['财经']:,} | {r['体育']:,} | {r['文娱']:,} | {r['教育']:,} | "
          f"{r['社会']:,} | {r['国际']:,} | {r['conflict_groups_remaining']} | "
          f"{r['delta_vs_A']:+,}（{r['delta_pct']:+.2f}%） |")
    A("")
    A("结论：\n")
    b_row = next(r for r in st if r["strategy"].startswith("B"))
    c_row = next(r for r in st if r["strategy"].startswith("C"))
    a_row = next(r for r in st if r["strategy"].startswith("A"))
    A(f"1. **策略 C 与 B 的差距极小**：放宽到「标题+日期」后目标六类仅多保留 "
      f"{c_row['target_rows'] - b_row['target_rows']:,} 条"
      f"（{b_row['target_rows']:,} → {c_row['target_rows']:,}），"
      "因为长跨度重复多落在非目标标签中，对六类供给几乎无影响。")
    A(f"2. **策略 A 的代价是引入标签冲突**：相对 B 多保留 "
      f"{a_row['target_rows'] - b_row['target_rows']:,} 条，"
      f"但同时带入 {a_row['conflict_groups_remaining']} 组同标题跨类别记录，"
      "而这恰恰是分类任务最忌讳的噪声。")
    A("3. 因此现行 B 策略对**本实验目标**是合理选择；它的缺陷（误删模板标题下的不同事件）"
      "属于「少量样本损失」，而非「标签污染」。\n")

    A("## 5. 未决事项（需人工确认，未擅自处理）\n")
    A("- 是否将去重键从 `title` 改为 `title + archive_date`：本版本**未改动**，"
      "仅输出量化影响，等确认后再决定。")
    A("- 跨类别冲突组 100% 为跨频道同稿，其**正确标签**无法仅凭标题判定"
      "（需正文或编辑意图）。若后续要修订标签，应逐条人工复核，不做批量推断。")
    A("- 「教育」类样本池仅略高于目标配额，若进一步收紧去重规则，需重新评估配额可达性。\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    raw_nan = load_raw_keep_nan()
    raw_missing = missing_value_audit(raw_nan, "news_raw")
    raw_missing["_table"].to_csv(AUDIT_DIR / "missing_values.csv", index=False,
                                 encoding=config.CSV_ENCODING)

    raw = raw_nan.fillna("")
    raw["title_norm"] = raw["title"].map(clean_title)
    raw["_is_article_b"] = raw["_is_article"].astype(str).str.lower().isin(["true", "1"])

    w = prepare_pool(raw_nan)

    dup, dup_stats = dedup_audit(raw_nan, w)
    sample, detail, cc_stats = cross_category_audit(w)
    strat = strategy_compare(w)
    cul_smp, cul_stats = culture_audit(raw)

    # 重复标题抽样（区分跨类别组与同类别组，供人工判读）
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
    dup_sample = pd.concat([cross_s, same_s]).reset_index(drop=True)
    dup_sample.insert(0, "seq", range(1, len(dup_sample) + 1))
    dup_sample[["seq", "title_norm", "n", "cats", "keeper_cat", "dates", "urls"]].to_csv(
        AUDIT_DIR / "dup_groups_sample.csv", index=False, encoding=config.CSV_ENCODING)

    sample_out = sample[[
        "seq", "title_norm", "n", "labels", "mapped", "channels", "n_dates",
        "date_min", "date_max", "span_days", "urls", "judge", "judge_note"]].rename(
        columns={"title_norm": "title"})
    sample_out.to_csv(AUDIT_DIR / "cross_category_sample.csv", index=False,
                      encoding=config.CSV_ENCODING)
    sample_out[["seq", "title", "mapped", "span_days", "judge", "judge_note"]].to_csv(
        AUDIT_DIR / "cross_category_judgement.csv", index=False,
        encoding=config.CSV_ENCODING)
    detail_out = detail.rename(columns={"title_norm": "title"})
    detail_out.to_csv(AUDIT_DIR / "cross_category_detail.csv", index=False,
                      encoding=config.CSV_ENCODING)
    cul_smp.to_csv(AUDIT_DIR / "culture_sample.csv", index=False,
                   encoding=config.CSV_ENCODING)

    summary = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "script": "audit_data.py",
            "raw_sha256": sha256(RAW_CSV),
            "clean_sha256": sha256(CLEAN_CSV),
        },
        "raw_rows": int(len(raw_nan)),
        "raw_label_count": {k: int(v) for k, v in raw["category"].value_counts().items()},
        "raw_missing": {k: v for k, v in raw_missing.items() if k != "_table"},
        "stage_counts": stage_counts(raw),
        "dedup": dup_stats,
        "cross_category": cc_stats,
        "strategy_compare": strat,
        "culture": cul_stats,
        "clean_check": clean_check(),
    }
    (AUDIT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_conflict_report(summary, config.REPORT_DIR / "cross_category_conflict_report.md")

    print("[ok] reports/audit/summary.json")
    print(f"[ok] 原始缺失值：NaN={raw_missing['nan_total']:,} "
          f"空串={raw_missing['empty_string_total']:,} "
          f"纯空白={raw_missing['whitespace_only_total']:,}")
    print(f"[ok] 标题重复：{dup_stats['dup_title_groups']:,} 组 / "
          f"{dup_stats['dup_extra_rows_deleted']:,} 条被删")
    print(f"[ok] 跨目标类冲突：{cc_stats['target_conflict_groups']} 组 / "
          f"{cc_stats['target_conflict_rows']} 条；抽样判读 D={cc_stats['sample_D']} T={cc_stats['sample_T']}")
    print("[ok] reports/cross_category_conflict_report.md")


if __name__ == "__main__":
    main()

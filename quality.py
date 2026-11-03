"""数据质量分析与可视化。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 中文字体
def _setup_chinese_font() -> None:
    """选择系统可用的中文字体，避免图表出现方块。"""
    candidates = ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC",
                  "Source Han Sans SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            logger.info("图表字体: %s", name)
            return
    logger.warning("未找到中文字体，图表中文可能显示异常")


# ---------------------------------------------------------------- 统计
def category_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """按目标六类统计样本量，缺失类别补 0。"""
    counts = df["category"].value_counts() if not df.empty else pd.Series(dtype=int)
    total = int(len(df))
    rows = []
    for cat in config.TARGET_CATEGORIES:
        n = int(counts.get(cat, 0))
        rows.append({
            "category": cat,
            "count": n,
            "percentage": round(n / total * 100, 2) if total else 0.0,
            "target": 2000,
            "gap_to_target": max(0, 2000 - n),
        })
    return pd.DataFrame(rows)


def daily_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby("archive_date").size()


def publish_time_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    ts = pd.to_datetime(df["publish_time"], errors="coerce").dropna()
    if ts.empty:
        return {}
    hour_dist = ts.dt.hour.value_counts().sort_index()
    return {
        "min": str(ts.min()),
        "max": str(ts.max()),
        "mean_date": str(ts.mean().date()),
        "hour_distribution": {int(h): int(c) for h, c in hour_dist.items()},
        "month_distribution": {str(k): int(v) for k, v in
                               ts.dt.to_period("M").value_counts().sort_index().items()},
    }


def duplicate_analysis(raw_df: pd.DataFrame) -> dict:
    """原始数据层面的重复情况。"""
    if raw_df.empty:
        return {"raw_rows": 0, "unique_url": 0, "unique_url_rate": 0.0,
                "duplicate_url_rows": 0, "same_title_diff_url": 0}
    total = len(raw_df)
    uniq = raw_df["url"].nunique()
    dup_rows = total - uniq
    dup_title = int(raw_df.duplicated(subset=["title"], keep="first").sum())
    return {
        "raw_rows": total,
        "unique_url": int(uniq),
        "duplicate_url_rows": int(dup_rows),
        "duplicate_url_rate": round(dup_rows / total * 100, 2) if total else 0.0,
        "same_title_diff_url": dup_title,
        "same_title_diff_url_rate": round(dup_title / total * 100, 2) if total else 0.0,
    }


# ---------------------------------------------------------------- 图表
def plot_category_bar(stat_df: pd.DataFrame, out_path: Path) -> None:
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=150)
    cats = stat_df["category"].tolist()
    counts = stat_df["count"].tolist()

    # 达成目标用红色系强调（中式习惯：正向用红），不足目标用灰蓝
    colors = ["#c0392b" if c >= 2000 else "#5b7c99" for c in counts]
    bars = ax.bar(cats, counts, color=colors, width=0.6, zorder=3)

    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")

    ax.axhline(2000, color="#c0392b", linestyle="--", linewidth=1.2,
               zorder=2, label="目标配额 2,000")
    ax.set_title(f"中国新闻网数据集类别样本分布（总计 {sum(counts):,} 条）",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("新闻类别", fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_ylim(0, max(max(counts) * 1.18, 2300))
    ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("类别分布图已保存: %s", out_path)


def plot_category_availability(available: dict, out_path: Path,
                               target: int = 2000) -> None:
    """抽样前各类实际可用样本量（对数轴），反映站点数据供给能力差异。"""
    _setup_chinese_font()
    cats = sorted(config.TARGET_CATEGORIES, key=lambda c: -available.get(c, 0))
    vals = [available.get(c, 0) for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    colors = ["#c0392b" if v >= target else "#5b7c99" for v in vals]
    bars = ax.barh(cats[::-1], vals[::-1], color=colors[::-1], height=0.6, zorder=3)

    ax.set_xscale("log")
    lo = max(min(vals) * 0.6, 100)
    ax.set_xlim(lo, max(vals) * 2.2)
    for b, v in zip(bars, vals[::-1]):
        ax.text(v * 1.12, b.get_y() + b.get_height() / 2, f"{v:,}",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#222")

    ax.axvline(target, color="#c0392b", linestyle="--", linewidth=1.2,
               zorder=2, label=f"目标配额 {target:,}")
    ax.set_title("抽样前各类别实际可用样本量（对数刻度）",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("可用样本数（对数）", fontsize=11)
    ax.grid(axis="x", linestyle=":", alpha=0.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("类别可用量图已保存: %s", out_path)


def plot_daily_line(daily: pd.Series, out_path: Path) -> None:
    if daily.empty:
        return
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    x = pd.to_datetime(daily.index)
    ax.plot(x, daily.values, color="#c0392b", linewidth=1.1, zorder=3)
    ax.fill_between(x, daily.values, color="#c0392b", alpha=0.12, zorder=2)
    ax.set_title("每日入库新闻数量", fontsize=12, fontweight="bold")
    ax.set_ylabel("条数")
    ax.grid(linestyle=":", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("每日数量图已保存: %s", out_path)


# ---------------------------------------------------------------- 报告
def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info("JSON 已写入: %s", path)


def write_markdown_report(ctx: dict, path: Path) -> None:
    """生成面向课程报告的中文质量报告。"""
    st = ctx["category_stats"]
    q = ctx["clean_stats"]
    dup = ctx["duplicates"]
    pt = ctx["publish_time"]
    daily = ctx["daily"]

    lines: list[str] = []
    A = lines.append
    A("# 中国新闻网新闻标题数据集 —— 采集质量报告\n")
    A(f"- 生成时间：{ctx['generated_at']}")
    A(f"- 数据来源：{config.SOURCE_NAME} 历史滚动新闻归档页")
    A(f"- URL 模板：`{config.ARCHIVE_URL_TEMPLATE}`")
    A(f"- 采集区间：{ctx['date_start']} ~ {ctx['date_end']}（{ctx['n_days_requested']} 天）")
    A(f"- 采集方式：requests + BeautifulSoup，单线程，请求间隔 {ctx['delay']}s\n")

    A("## 1. 请求概况\n")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 请求日期数 | {ctx['n_days_requested']} |")
    A(f"| 成功请求 | {ctx['n_days_ok']} |")
    A(f"| 页面不存在(404) | {ctx['n_days_404']} |")
    A(f"| 请求失败 | {ctx['n_days_failed']} |")
    A(f"| 列表容器缺失(结构异常) | {ctx['n_days_parse_fail']} |")
    A(f"| 原始记录数 | {dup['raw_rows']:,} |")
    A(f"| 清洗后六类合计 | {q['rows_after_clean']:,} |")
    if q.get("cap_per_class", 0) > 0:
        A(f"| 配额抽样后交付样本量 | **{q.get('rows_delivered', 0):,}**"
          f"（每类上限 {q['cap_per_class']:,}） |")
    A("")

    A("## 2. 采集数量分布\n")
    if len(daily) > 31:
        # 长区间按月聚合，避免报告表格过长
        tmp = pd.DataFrame({"d": pd.to_datetime(daily.index), "n": daily.values})
        monthly = tmp.groupby(tmp["d"].dt.to_period("M"))["n"].sum()
        A("采集区间较长，按月份聚合（完整逐日明细见 `crawl_report.json` 的 `days` 字段）。\n")
        A("| 月份 | 原始条数 |")
        A("|---|---|")
        for p, v in monthly.items():
            A(f"| {p} | {int(v):,} |")
        A(f"| **合计** | **{int(daily.sum()):,}** |")
    else:
        A("| 归档日期 | 原始条数 |")
        A("|---|---|")
        for d, c in daily.items():
            A(f"| {d} | {c} |")
        if len(daily):
            A(f"| **合计** | **{int(daily.sum()):,}** |")
    A("")

    A("## 3. 类别样本分布\n")
    if q.get("cap_per_class", 0) > 0:
        A(f"> 六类清洗后合计 {q['rows_after_clean']:,} 条，按每类上限 "
          f"{q['cap_per_class']:,} 条做**按归档日期等距抽样**，交付 {st['count'].sum():,} 条。"
          "抽样按日期均匀覆盖，避免时间分布偏斜；结果确定、可复现。\n")
    A("| 类别 | 样本数 | 占比 | 距目标(2000) |")
    A("|---|---|---|---|")
    for _, r in st.iterrows():
        A(f"| {r['category']} | {r['count']:,} | {r['percentage']}% | {r['gap_to_target']:,} |")
    A(f"| **合计** | **{int(st['count'].sum()):,}** | 100% | — |\n")

    avail = ctx.get("available") or {}
    if avail:
        A("### 3.1 抽样前各类实际可用量\n")
        A("| 类别 | 可用样本数 | 是否满足目标 2,000 |")
        A("|---|---|---|")
        for cat in config.TARGET_CATEGORIES:
            n = int(avail.get(cat, 0))
            A(f"| {cat} | {n:,} | {'满足' if n >= 2000 else '**不足**'} |")
        A("")
        A("> 各类供给能力差异显著（对数刻度见 `category_availability.png`）："
          "社会、财经供给充裕，教育为长尾类别。本文档交付集已按等距抽样平衡至每类 2,000 条；"
          "若实验需要类内更多样本，可调高 `--max-per-class`。\n")

    A("## 4. 原始分类标签分布（映射前）\n")
    A("保留站点原始标签，便于核查映射是否合理。\n")
    A("| 原始标签 | 条数 | 去向 |")
    A("|---|---|---|")
    for cat, n in ctx["raw_label_dist"].items():
        if cat in config.CATEGORY_MAPPING:
            dest = f"映射为「{config.CATEGORY_MAPPING[cat]}」"
        elif cat in config.NON_TOPIC_LABELS:
            dest = "排除（媒体形态标签，非主题）"
        elif cat == config.UNKNOWN_CATEGORY:
            dest = "排除（无明确分类）"
        else:
            dest = "排除（非目标类别）"
        A(f"| {cat} | {n:,} | {dest} |")
    A("")

    A("## 5. 重复与异常\n")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A(f"| 原始记录数 | {dup['raw_rows']:,} |")
    A(f"| 唯一 URL 数 | {dup['unique_url']:,} |")
    A(f"| URL 重复条数 | {dup['duplicate_url_rows']:,}（{dup['duplicate_url_rate']}%） |")
    A(f"| 标题相同但 URL 不同（原始标题） | {dup['same_title_diff_url']:,}（{dup.get('same_title_diff_url_rate', 0)}%） |")
    A(f"| 标题重复实际剔除（规范化后） | {q['duplicate_title_groups']:,} |")
    A(f"| 空标题（剔除） | {q['dropped_empty_title']:,} |")
    A(f"| 无效/非文章链接（剔除） | {q['dropped_invalid_url']:,} |")
    A(f"| 时间格式异常（剔除） | {q['dropped_bad_time']:,} |")
    A(f"| 无明确分类（剔除） | {q['dropped_unknown_category']:,} |")
    A(f"| 非目标类别（剔除） | {q['dropped_non_target_category']:,} |\n")

    A("## 6. 时间分布\n")
    if pt:
        A(f"- 最早发布时间：{pt['min']}")
        A(f"- 最晚发布时间：{pt['max']}")
        A("- 按小时分布：\n")
        A("| 时段 | 条数 |")
        A("|---|---|")
        for h, c in pt["hour_distribution"].items():
            A(f"| {h:02d}:00-{h:02d}:59 | {c:,} |")
    A("")

    A("## 7. 已知限制与注意事项\n")
    for item in ctx["caveats"]:
        A(f"- {item}")
    A("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("质量报告已写入: %s", path)

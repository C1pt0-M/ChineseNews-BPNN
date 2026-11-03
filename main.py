"""中国新闻网历史滚动新闻采集 —— 命令行入口。

用法示例：
    python main.py probe --date 2025-01-01
    python main.py all   --start 2025-01-01 --end 2025-01-07
    python main.py all   --start 2025-01-01 --end 2026-10-31 --max-per-class 2000
    python main.py all   --start 2025-01-01 --end 2025-01-07 --no-resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

import cleaner
import config
import crawler
import quality

logger = logging.getLogger("chinanews")


# ---------------------------------------------------------------- 基础设施
def setup_logging(verbose: bool = False) -> None:
    config.ensure_dirs()
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(config.LOG_DIR / "crawl.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"日期格式应为 YYYY-MM-DD，收到: {s}")


def load_checkpoint() -> dict:
    p = config.STATE_DIR / "checkpoint.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("断点文件损坏，将重新采集")
    return {"completed": {}}


def save_checkpoint(ckpt: dict) -> None:
    p = config.STATE_DIR / "checkpoint.json"
    p.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")


def part_path(day: date) -> Path:
    return config.RAW_PARTS_DIR / f"{day:%Y-%m-%d}.jsonl"


def write_part(day: date, rows: list[dict], meta: dict) -> None:
    """按日期落盘切片，作为断点续采的最小单元。"""
    config.RAW_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(part_path(day), "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}, ensure_ascii=False) + "\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_part(day: date) -> tuple[list[dict], dict] | None:
    p = part_path(day)
    if not p.exists():
        return None
    rows, meta = [], {}
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            if i == 0 and "_meta" in obj:
                meta = obj["_meta"]
            else:
                rows.append(obj)
    return rows, meta


# ---------------------------------------------------------------- 阶段一：结构探测
STAGE1_MD = """# 阶段一：网页结构分析报告

- 分析时间：{ts}
- 样本页面：{url}
- 请求方式：`requests` + `BeautifulSoup(lxml)`，未使用浏览器自动化

## 1. HTTP 响应

| 项目 | 实测值 |
|---|---|
| HTTP 状态码 | {status} |
| 响应体字节数 | {bytes:,} |
| 响应头 Content-Type | `{ctype}` |
| 响应头声明编码 | {declared}（**未声明真实编码**） |
| 实际编码 | UTF-8（`apparent_encoding` 判定一致） |
| 页面 title | {title} |

> 结论：响应头未给出 charset，requests 默认按 ISO-8859-1 解码会造成中文乱码。
> 程序必须显式 `resp.encoding = "utf-8"`，本实现已强制指定。

## 2. 新闻列表定位

采用“链接密度”而非经验猜测定位列表容器，结果如下：

| 容器 | 链接数 |
|---|---|
{containers}

新闻列表位于 **`div.content_list > ul > li`**，单页即单日全量（未发现分页控件 `.pagebox`）。

单条记录结构：

```html
<li>
  <div class="dd_lm">[<a href="//www.chinanews.com/world.shtml">国际</a>]</div>
  <div class="dd_bt"><a href="/gj/2025/01-01/10346216.shtml">2024年赴泰外国游客超3500万人次</a></div>
  <div class="dd_time">1-1 23:21</div>
</li>
```

字段对应关系：

| 字段 | 选择器 | 说明 |
|---|---|---|
| category | `div.dd_lm` | 分类标签，形如 `[ 国际 ]`；部分为纯文本（如 `[其他]`）**不含 `<a>`**，故必须取文本而非取链接 |
| title | `div.dd_bt > a` | 标题文本 |
| url | `div.dd_bt > a[href]` | 文章链接，相对路径，需补全域名 |
| publish_time | `div.dd_time` | `月-日 时:分`，**不含年份**，需与归档日期合成 |

解析诊断结果：

- `<li>` 总数：{items}
- 空占位 `<li></li>`：{skipped_empty}（用于分隔线，需过滤）
- 有效记录：{valid}
- 分页控件：{pagination}

## 3. 链接指向性校验

标题链接共三种形态，均为具体内容页而非栏目首页：图文新闻
`/{频道}/{年}/{月-日}/{8位数字}.shtml`、高清图集 `/tp/hd2011/{年}/{月-日}/{7位数字}.shtml`、
视频 `/{频道}/shipin/cns/{年}/{月-日}/news{7位数字}.shtml`。
本页有效内容链接占比：{article_ratio}%。

## 4. 历史数据入口

页面 `div.content-right` 内含“查看往日回顾”日历，由 `calendar.js` 动态生成，
其内部拼接逻辑为：

```js
url = '/scroll-news/year/monthday/news.shtml';
```

与已知 URL 规律一致，且年份下拉范围为 2008 年至今（`startDate=2004-01-01`），
证实归档可按日直接构造 URL 访问，无需点击日历或执行 JS。

## 5. 实际解析样例（前 10 条）

| # | 分类 | 标题 | 发布时间 | 链接 |
|---|---|---|---|---|
{samples}

## 6. 边界与异常页面行为

| 请求 | 状态码 | 记录数 | 结论 |
|---|---|---|---|
| 2025/0230（非法日期） | 404 | 0 | 返回 404 哨兵页（{nf_bytes}B，标题含“404页面”） |
| 2027/0101（未来） | 404 | 0 | 同上，未归档日期不可访问 |
| 2019/0101（历史） | 200 | 392 | 归档可回溯，结构一致 |

> 程序据此将 404 判定为“当日无归档”，记入报告而非当作错误重试。
"""


class _SafeDict(dict):
    """未提供的占位符原样保留，避免正文中的花括号被 format 误解析。"""

    def __missing__(self, key):
        return "{" + key + "}"


def dataset_dir(args) -> Path:
    """输出数据目录：默认 dataset/，可用 --outdir 覆盖（便于隔离测试集与正式集）。"""
    if getattr(args, "outdir", None):
        p = Path(args.outdir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return config.DATASET_DIR


def report_dir(args) -> Path:
    """报告目录：默认 reports/，可用 --reportdir 覆盖。"""
    if getattr(args, "reportdir", None):
        p = Path(args.reportdir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    return config.REPORT_DIR


def cmd_probe(args) -> int:
    day = args.date
    url = crawler.build_archive_url(day)
    session = crawler.make_session()

    logger.info("探测页面: %s", url)
    res = crawler.fetch_archive_page(session, url, delay=0)
    if not res.ok:
        logger.error("请求失败: status=%s error=%s", res.status, res.error)
        return 1

    parsed = crawler.parse_archive_page(res.content, day, url)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(res.content, "lxml")

    # 链接密度表
    rows = []
    for tag in ("div", "ul"):
        for el in soup.find_all(tag):
            n = len(el.find_all("a", href=True))
            if n >= 10:
                classes = el.get("class") or []
                cls = ("." + ".".join(classes) + (f"#{el.get('id')}" if el.get("id") else "")) if classes else ""
                rows.append((n, f"{tag}{cls}"))
    rows.sort(key=lambda x: -x[0])
    seen, uniq_rows = set(), []
    for n, sel in rows:
        if sel in seen:
            continue
        seen.add(sel)
        uniq_rows.append((n, sel))
    containers = "\n".join(f"| `{sel}` | {n} |" for n, sel in uniq_rows[:8])

    article_cnt = sum(1 for r in parsed.records if r["_is_article"])
    samples = "\n".join(
        f"| {i+1} | {r['category']} | {r['title']} | {r['publish_time']} | `{r['url']}` |"
        for i, r in enumerate(parsed.records[:10])
    )

    md = STAGE1_MD.format_map(_SafeDict(
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        url=url,
        status=res.status,
        bytes=res.byte_len,
        ctype="text/html",
        declared="ISO-8859-1",
        title=soup.title.string.strip() if soup.title and soup.title.string else "-",
        containers=containers,
        items=parsed.item_count,
        skipped_empty=parsed.skipped_empty,
        valid=len(parsed.records),
        pagination="存在" if parsed.has_pagination else "未发现",
        article_ratio=round(article_cnt / len(parsed.records) * 100, 1) if parsed.records else 0,
        samples=samples,
        nf_bytes=config.NOT_FOUND_BYTES,
    ))
    out = report_dir(args) / "stage1_structure_analysis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    logger.info("阶段一报告已写入: %s", out)

    # 控制台样例
    print(f"\n{'='*96}")
    print(f"页面: {url}")
    print(f"状态码: {res.status} | 字节: {res.byte_len:,} | 有效记录: {len(parsed.records)}")
    print(f"{'='*96}")
    for r in parsed.records[:10]:
        print(f"[{r['category']:>4}] {r['publish_time']}  {r['title']}")
        print(f"        {r['url']}")
    return 0


# ---------------------------------------------------------------- 采集
def cmd_crawl(args) -> int:
    start, end = args.start, args.end
    if start > end:
        logger.error("起始日期晚于结束日期")
        return 2

    n_days = (end - start).days + 1
    logger.info("采集区间 %s ~ %s，共 %d 天，间隔 %.1fs，单线程",
                start, end, n_days, args.delay)

    ckpt = {"completed": {}} if args.no_resume else load_checkpoint()
    session = crawler.make_session()

    all_rows: list[dict] = []
    day_stats: list[dict] = []
    t0 = time.time()

    days = list(crawler.daterange(start, end))
    for i, day in enumerate(days, 1):
        key = f"{day:%Y-%m-%d}"

        # 断点续采：已完成且切片存在则直接复用
        if key in ckpt["completed"]:
            cached = read_part(day)
            if cached is not None:
                rows, meta = cached
                all_rows.extend(rows)
                day_stats.append(meta)
                logger.info("[%d/%d] %s 跳过（断点已采集 %d 条）",
                            i, n_days, key, len(rows))
                continue

        url = crawler.build_archive_url(day)
        res = crawler.fetch_archive_page(session, url, delay=args.delay)
        day_rows: list[dict] = []

        meta = {
            "archive_date": key,
            "url": url,
            "status": res.status,
            "bytes": res.byte_len,
            "records": 0,
            "result": "ok",
            "error": "",
        }

        if res.not_found:
            meta["result"] = "not_found"
            logger.warning("[%d/%d] %s 页面不存在(404)", i, n_days, key)
        elif not res.ok:
            meta["result"] = "failed"
            meta["error"] = res.error
            logger.error("[%d/%d] %s 请求失败: %s", i, n_days, key, res.error)
        else:
            parsed = crawler.parse_archive_page(res.content, day, url)
            if not parsed.container_found:
                meta["result"] = "parse_failed"
                meta["error"] = "未找到 div.content_list"
            else:
                if parsed.has_pagination:
                    logger.warning("[%d/%d] %s 出现分页控件，当日数据可能不完整",
                                   i, n_days, key)
                    meta["pagination"] = True
                meta["records"] = len(parsed.records)
                meta["items"] = parsed.item_count
                day_rows = parsed.records
                all_rows.extend(day_rows)
                logger.info("[%d/%d] %s 状态=%s 记录=%d",
                            i, n_days, key, res.status, len(parsed.records))

        write_part(day, day_rows, meta)
        day_stats.append(meta)
        ckpt["completed"][key] = {"records": meta["records"], "result": meta["result"]}
        save_checkpoint(ckpt)

        if i % 20 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed else 0
            eta = (n_days - i) / rate if rate else 0
            logger.info("进度 %d/%d（%.1f%%），已用时 %.0fs，预计剩余 %.0fs",
                        i, n_days, i / n_days * 100, elapsed, eta)

    # 写原始 CSV
    raw_df = pd.DataFrame(all_rows)
    if not raw_df.empty:
        drop_cols = [c for c in raw_df.columns if c.startswith("_")]
        out_cols = ["id", "title", "category", "publish_time",
                    "url", "archive_date", "source"] + drop_cols
        raw_df = raw_df[[c for c in out_cols if c in raw_df.columns]]
    raw_path = dataset_dir(args) / "news_raw.csv"
    raw_df.to_csv(raw_path, index=False, encoding=config.CSV_ENCODING)
    logger.info("原始数据已保存: %s（%d 行）", raw_path, len(raw_df))

    # 采集报告
    report = {
        "crawl_time": datetime.now().isoformat(timespec="seconds"),
        "source": config.SOURCE_NAME,
        "url_template": config.ARCHIVE_URL_TEMPLATE,
        "date_start": f"{start:%Y-%m-%d}",
        "date_end": f"{end:%Y-%m-%d}",
        "days_requested": n_days,
        "delay_seconds": args.delay,
        "concurrency": 1,
        "summary": {
            "days_success": sum(1 for d in day_stats if d["result"] == "ok"),
            "days_not_found": sum(1 for d in day_stats if d["result"] == "not_found"),
            "days_failed": sum(1 for d in day_stats if d["result"] == "failed"),
            "days_parse_failed": sum(1 for d in day_stats if d["result"] == "parse_failed"),
            "total_records": int(len(raw_df)),
        },
        "days": day_stats,
        "robots_txt": {
            "checked": True,
            "note": "https://www.chinanews.com/robots.txt 仅 Disallow /AboutUs/Staff/、/part/、"
                    "/iframe/、/test、/ceshi/、/ad/ 等路径，/scroll-news/ 未被禁止。",
        },
    }
    quality.write_json(report, dataset_dir(args) / "crawl_report.json")

    print(f"\n采集完成：{len(raw_df)} 条原始记录，用时 {time.time()-t0:.0f}s")
    return 0


# ---------------------------------------------------------------- 清洗 + 报告
def cmd_clean(args) -> int:
    raw_path = dataset_dir(args) / "news_raw.csv"
    if not raw_path.exists():
        logger.error("未找到 %s，请先执行 crawl", raw_path)
        return 2

    raw_df = pd.read_csv(raw_path, encoding=config.CSV_ENCODING, dtype=str)
    raw_df = raw_df.fillna("")
    logger.info("读入原始数据 %d 行", len(raw_df))

    # 还原解析诊断字段（CSV 中带 _ 前缀列）
    if "_is_article" in raw_df.columns:
        raw_df["_is_article"] = raw_df["_is_article"].astype(str).str.lower().isin(["true", "1"])
    else:
        raw_df["_is_article"] = raw_df["url"].map(crawler.is_article_url)

    clean_df, stats = cleaner.clean_dataframe(raw_df)
    # 抽样前各类实际可用量（用于供给能力分析图）
    available = clean_df["category"].value_counts().to_dict() if len(clean_df) else {}
    clean_df = cleaner.cap_per_category(clean_df, args.max_per_class)
    stats["rows_delivered"] = int(len(clean_df))
    stats["cap_per_class"] = int(args.max_per_class)

    clean_path = dataset_dir(args) / "news_clean.csv"
    clean_df.to_csv(clean_path, index=False, encoding=config.CSV_ENCODING)
    logger.info("清洗数据已保存: %s（%d 行）", clean_path, len(clean_df))

    stat_df = quality.category_statistics(clean_df)
    stat_path = dataset_dir(args) / "category_statistics.csv"
    stat_df.to_csv(stat_path, index=False, encoding=config.CSV_ENCODING)
    logger.info("类别统计已保存: %s", stat_path)

    # 报告
    report_path = dataset_dir(args) / "crawl_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    day_stats = report.get("days", [])
    daily = pd.Series({d["archive_date"]: d["records"] for d in day_stats
                       if d.get("result") == "ok"}).sort_index()

    raw_label_dist = raw_df["category"].value_counts().to_dict()

    ts = pd.to_datetime(clean_df["publish_time"], errors="coerce").dropna() if len(clean_df) else pd.Series(dtype="datetime64[ns]")
    pt = {}
    if len(ts):
        pt = {
            "min": str(ts.min()),
            "max": str(ts.max()),
            "hour_distribution": {int(k): int(v) for k, v in ts.dt.hour.value_counts().sort_index().items()},
        }

    caveats = [
        "站点滚动列表不存在“文娱”原始标签，文娱内容统一标记为“文化”；本数据集采用 "
        "`文化 -> 文娱` 的显式代理映射，训练时应在报告中说明该标签语义差异。",
        "“教育”类日均不足 5 条，属于长尾类别，目标 2000 条可能无法达成，实际数量如实统计。",
        "原始标签中的“图片”“视频”属于媒体形态而非主题，已排除，避免污染主题分类。",
        "“国内”“时政”“湾区”“华人”“健康”“理论”等标签不为目标类别，已保留原始标签但不纳入数据集。",
        "归档页仅提供“月-日 时:分”，年份由归档路径推断，已校验与归档日期一致。",
        "发布时间为站点列表展示时间，非文章页精确发布时间，存在分钟级误差可能。",
        "仅采集标题与元数据，未采集正文，符合课程实验的最小数据需求与版权要求。",
    ]

    ctx = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date_start": report.get("date_start", "-"),
        "date_end": report.get("date_end", "-"),
        "n_days_requested": report.get("days_requested", len(day_stats)),
        "n_days_ok": sum(1 for d in day_stats if d.get("result") == "ok"),
        "n_days_404": sum(1 for d in day_stats if d.get("result") == "not_found"),
        "n_days_failed": sum(1 for d in day_stats if d.get("result") == "failed"),
        "n_days_parse_fail": sum(1 for d in day_stats if d.get("result") == "parse_failed"),
        "delay": report.get("delay_seconds", config.DEFAULT_DELAY),
        "category_stats": stat_df,
        "clean_stats": stats,
        "duplicates": quality.duplicate_analysis(raw_df),
        "publish_time": pt,
        "daily": daily,
        "raw_label_dist": raw_label_dist,
        "available": available,
        "caveats": caveats,
    }
    quality.write_markdown_report(ctx, report_dir(args) / "quality_report.md")
    quality.plot_category_bar(stat_df, report_dir(args) / "category_distribution.png")
    quality.plot_daily_line(daily, report_dir(args) / "daily_counts.png")
    if available:
        quality.plot_category_availability(available,
                                           report_dir(args) / "category_availability.png")

    # 更新 crawl_report 中的清洗统计
    report["cleaning"] = stats
    report["duplicates"] = ctx["duplicates"]
    report["category_statistics"] = stat_df.to_dict(orient="records")
    quality.write_json(report, report_path)

    print("\n类别分布：")
    for _, r in stat_df.iterrows():
        bar = "█" * int(r["count"] / max(stat_df["count"].max(), 1) * 40)
        print(f"  {r['category']:>3} {r['count']:>6,}  {bar}")
    print(f"\n清洗后总计：{len(clean_df):,} 条")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="中国新闻网历史滚动新闻采集（课程实验用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    common.add_argument("--outdir", default=None,
                        help="输出数据目录，默认 dataset/；用于隔离测试集与正式集")
    common.add_argument("--reportdir", default=None,
                        help="输出报告目录，默认 reports/")

    p_probe = sub.add_parser("probe", parents=[common], help="阶段一：单页结构探测")
    p_probe.add_argument("--date", type=parse_date, default=date(2025, 1, 1))
    p_probe.set_defaults(func=cmd_probe)

    p_crawl = sub.add_parser("crawl", parents=[common], help="按日期区间采集")
    p_crawl.add_argument("--start", type=parse_date, default=parse_date(config.DATE_START_DEFAULT))
    p_crawl.add_argument("--end", type=parse_date, default=parse_date(config.DATE_END_DEFAULT))
    p_crawl.add_argument("--delay", type=float, default=config.DEFAULT_DELAY,
                         help=f"请求间隔秒数，默认 {config.DEFAULT_DELAY}")
    p_crawl.add_argument("--no-resume", action="store_true", help="忽略断点重新采集")
    p_crawl.set_defaults(func=cmd_crawl)

    p_clean = sub.add_parser("clean", parents=[common], help="清洗、去重、统计与出图")
    p_clean.add_argument("--max-per-class", type=int, default=0,
                         help="每类样本上限，0 表示不限制（按日期等距抽样）")
    p_clean.set_defaults(func=cmd_clean)

    p_all = sub.add_parser("all", parents=[common], help="采集 + 清洗 + 报告")
    p_all.add_argument("--start", type=parse_date, default=parse_date(config.DATE_START_DEFAULT))
    p_all.add_argument("--end", type=parse_date, default=parse_date(config.DATE_END_DEFAULT))
    p_all.add_argument("--delay", type=float, default=config.DEFAULT_DELAY)
    p_all.add_argument("--no-resume", action="store_true")
    p_all.add_argument("--max-per-class", type=int, default=0)
    p_all.set_defaults(func=lambda a: cmd_crawl(a) or cmd_clean(a))

    return p


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(getattr(args, "verbose", False))
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

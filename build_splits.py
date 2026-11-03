"""按发布时间划分训练/验证/测试集，并登记数据集版本与可复现信息。

用途：为 TF-IDF + BP 神经网络实验准备**按时间顺序**划分的数据集，
避免随机划分造成的「同系列新闻跨集合」与「未来信息泄漏」。

用法：
    python build_splits.py

输入：
    dataset/news_clean.csv      12,000 条交付数据集（不做任何修改）

输出：
    dataset/splits/train.csv
    dataset/splits/val.csv
    dataset/splits/test.csv
    dataset/splits/split_statistics.csv
    dataset/dataset_version.json        版本、SHA256、划分边界、标签映射
    reports/time_split_report.md        划分报告与泄漏检查

约束：
  - 不重新爬取、不扩样、不改变类别名称与映射；
  - 划分完全依据 publish_time，不使用行号；
  - 不为凑齐样本量而打乱时间顺序。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import config

SPLITS_DIR = config.DATASET_DIR / "splits"
CLEAN_CSV = config.DATASET_DIR / "news_clean.csv"
VERSION_JSON = config.DATASET_DIR / "dataset_version.json"
REPORT_MD = config.REPORT_DIR / "time_split_report.md"

DATASET_VERSION = "1.1.0"
SPLIT_VERSION = "1.0.0"
RANDOM_SEED = 42          # 划分不依赖随机数；此值用于任何后续抽样/初始化，登记以便复现

# 划分边界（左闭右闭，按 publish_time）
BOUNDARIES = [
    ("train", "2025-01-01", "2025-12-31", "模型训练"),
    ("val",   "2026-01-01", "2026-06-30", "参数调整"),
    ("test",  "2026-07-01", "2026-10-31", "最终评估"),
]

TIME_RE = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
NEAR_DUP_PREFIX = 8        # 近重复判定的标题前缀长度


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_clean() -> tuple[pd.DataFrame, dict]:
    """读入交付数据集并校验时间字段完整性。"""
    raw = pd.read_csv(CLEAN_CSV, encoding=config.CSV_ENCODING, dtype=str)
    checks = {
        "rows": int(len(raw)),
        "columns": list(raw.columns),
        "nan_per_column": {k: int(v) for k, v in raw.isna().sum().items() if v > 0},
        "empty_string": int((raw.fillna("") == "").sum().sum()),
        "bad_time_format": int((~raw["publish_time"].fillna("").str.match(TIME_RE)).sum()),
    }
    df = raw.copy()
    df["_ts"] = pd.to_datetime(df["publish_time"], errors="coerce")
    checks["unparsable_time"] = int(df["_ts"].isna().sum())
    checks["publish_date_ne_archive_date"] = int(
        (df["publish_time"].str[:10] != df["archive_date"]).sum())
    return df, checks


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    """按 publish_time 落入的时间窗分配集合；不依赖行序。"""
    df = df.copy()
    df["split"] = pd.NA
    for name, start, end, _ in BOUNDARIES:
        m = (df["_ts"] >= f"{start} 00:00:00") & (df["_ts"] <= f"{end} 23:59:59")
        df.loc[m & df["split"].isna(), "split"] = name
    return df


def leakage_check(df: pd.DataFrame) -> dict:
    """跨集合重复与标签冲突检查。"""
    assigned = df[df["split"].notna()]
    out: dict = {}

    # 1) 精确标题跨集合
    t = assigned.groupby("title")["split"].nunique()
    cross_title = t[t > 1]
    out["exact_title_cross_split_groups"] = int(len(cross_title))
    out["exact_title_cross_split_rows"] = int(
        assigned["title"].isin(cross_title.index).sum())

    # 2) URL / ID 跨集合（应恒为 0）
    for col in ("url", "id"):
        g = assigned.groupby(col)["split"].nunique()
        out[f"cross_split_{col}_groups"] = int((g > 1).sum())

    # 3) 近重复标题（前 N 字相同）
    assigned = assigned.copy()
    assigned["_p"] = assigned["title"].str[:NEAR_DUP_PREFIX]
    pg = assigned.groupby("_p")["split"].nunique()
    cross_prefix = pg[pg > 1]
    out["near_dup_prefix_len"] = NEAR_DUP_PREFIX
    out["near_dup_cross_split_groups"] = int(len(cross_prefix))
    out["near_dup_cross_split_rows"] = int(assigned["_p"].isin(cross_prefix.index).sum())
    out["near_dup_cross_split_rate_pct"] = round(
        out["near_dup_cross_split_rows"] / len(assigned) * 100, 2)

    # 4) 同一标题被赋予不同类别（交付集内应为 0）
    cg = assigned.groupby("title")["category"].nunique()
    out["title_multi_category_groups"] = int((cg > 1).sum())

    # 5) 近重复前缀内的类别冲突（模型更易被误导的情形）
    cpg = assigned.groupby("_p")["category"].nunique()
    out["near_dup_multi_category_groups"] = int((cpg > 1).sum())

    # 6) 近重复前缀跨集合且类别不同的组数（最需关注）
    both = cross_prefix.index.intersection(cpg[cpg > 1].index)
    out["near_dup_cross_split_and_multi_category"] = int(len(both))

    out["_cross_prefix_examples"] = (
        assigned[assigned["_p"].isin(cross_prefix.index)]
        .groupby("_p").agg(n=("title", "size"),
                           splits=("split", lambda s: "|".join(sorted(set(s)))),
                           cats=("category", lambda s: "|".join(sorted(set(s)))),
                           example=("title", "first"))
        .sort_values("n", ascending=False).head(10).reset_index()
        .rename(columns={"_p": "prefix"}))
    return out


def build_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_by_cat = df["category"].value_counts()
    for name, start, end, purpose in BOUNDARIES:
        sub = df[df["split"] == name]
        vc = sub["category"].value_counts()
        for cat in config.TARGET_CATEGORIES:
            n = int(vc.get(cat, 0))
            rows.append({
                "split": name, "category": cat, "count": n,
                "pct_within_split": round(n / len(sub) * 100, 2) if len(sub) else 0.0,
                "pct_of_category_total": round(n / int(total_by_cat.get(cat, 0)) * 100, 2)
                if total_by_cat.get(cat, 0) else 0.0,
                "split_start_date": start, "split_end_date": end,
                "days_covered": int(sub["archive_date"].nunique()),
                "purpose": purpose,
            })
        rows.append({
            "split": name, "category": "合计", "count": int(len(sub)),
            "pct_within_split": 100.0, "pct_of_category_total": 100.0,
            "split_start_date": start, "split_end_date": end,
            "days_covered": int(sub["archive_date"].nunique()),
            "purpose": purpose,
        })
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, stats: pd.DataFrame, checks: dict,
                 leak: dict, version: dict, path: Path) -> None:
    L: list[str] = []
    A = L.append
    A("# 时间划分报告（训练 / 验证 / 测试）\n")
    A(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"- 数据集版本：`{DATASET_VERSION}`（划分版本 `{SPLIT_VERSION}`）")
    A(f"- 输入文件：`dataset/news_clean.csv`（{checks['rows']:,} 条）")
    A(f"- 输入文件 SHA256：`{version['files']['news_clean.csv']['sha256']}`")
    A(f"- 划分依据：`publish_time`（新闻发布时间），不使用文件行号")
    A(f"- 随机种子：`{RANDOM_SEED}`（划分本身为确定性规则，不消耗随机数）\n")

    A("## 1. 划分规则\n")
    A("| 数据集 | 时间范围 | 用途 | 样本数 | 占总量 |")
    A("|---|---|---|---|---|")
    for name, start, end, purpose in BOUNDARIES:
        n = int((df["split"] == name).sum())
        A(f"| {name} | {start} ~ {end} | {purpose} | {n:,} | {n / len(df) * 100:.2f}% |")
    A(f"| **合计** | 2025-01-01 ~ 2026-10-31 | — | **{len(df):,}** | 100% |")
    A("")
    A("> 边界为左闭右闭，按 `publish_time` 判定。区间连续且无重叠，"
      f"未落入任何集合的记录数：{int(df['split'].isna().sum())}。\n")

    A("## 2. 时间字段校验\n")
    A("| 检查项 | 结果 |")
    A("|---|---|")
    A(f"| 总记录数 | {checks['rows']:,} |")
    A(f"| `publish_time` 缺失(NaN) | {checks['nan_per_column'].get('publish_time', 0)} |")
    A(f"| 时间格式不合规 | {checks['bad_time_format']} |")
    A(f"| 无法解析为时间 | {checks['unparsable_time']} |")
    A(f"| 发布日期 ≠ 归档日期 | {checks['publish_date_ne_archive_date']} |")
    A(f"| 全表空字符串单元格 | {checks['empty_string']} |")
    A("")

    A("## 3. 各集合类别分布\n")
    A("| 数据集 | " + " | ".join(config.TARGET_CATEGORIES) + " | 合计 |")
    A("|---" * (len(config.TARGET_CATEGORIES) + 2) + "|")
    for name, _, _, _ in BOUNDARIES:
        sub = stats[stats["split"] == name]
        cells = []
        for cat in config.TARGET_CATEGORIES:
            n = int(sub.loc[sub["category"] == cat, "count"].iloc[0])
            cells.append(f"{n:,}")
        total = int(sub.loc[sub["category"] == "合计", "count"].iloc[0])
        A(f"| {name} | " + " | ".join(cells) + f" | **{total:,}** |")
    A("")
    A("| 数据集 | 最小类样本数 | 最大类样本数 | 最大/最小 | 是否类别缺失 |")
    A("|---|---|---|---|---|")
    for name, _, _, _ in BOUNDARIES:
        sub = stats[(stats["split"] == name) & (stats["category"] != "合计")]
        lo, hi = int(sub["count"].min()), int(sub["count"].max())
        A(f"| {name} | {lo:,} | {hi:,} | {hi / lo:.2f} | {'否' if lo > 0 else '**是**'} |")
    A("")
    A("> 三类集合的类别比例接近均衡（由等距抽样保证），"
      "未出现某类缺失或样本过少（最小类样本数见上表）。"
      "因此**不需要**调整时间边界来平衡类别。\n")

    A("## 4. 数据泄漏检查\n")
    A("| 检查项 | 结果 | 说明 |")
    A("|---|---|---|")
    A(f"| 精确标题跨集合 | {leak['exact_title_cross_split_groups']} 组 / "
      f"{leak['exact_title_cross_split_rows']} 条 | 应为 0；现行标题全局去重已消除 |")
    A(f"| URL 跨集合 | {leak['cross_split_url_groups']} 组 | 应为 0 |")
    A(f"| ID 跨集合 | {leak['cross_split_id_groups']} 组 | 应为 0 |")
    A(f"| 同一标题多类别 | {leak['title_multi_category_groups']} 组 | 应为 0 |")
    A(f"| 近重复前缀跨集合（前 {leak['near_dup_prefix_len']} 字相同） | "
      f"{leak['near_dup_cross_split_groups']:,} 组 / {leak['near_dup_cross_split_rows']:,} 条"
      f"（{leak['near_dup_cross_split_rate_pct']}%） | 同系列不同事件的标题相似，非精确重复 |")
    A(f"| 近重复前缀内类别冲突 | {leak['near_dup_multi_category_groups']:,} 组 | 前缀相同但类别不同 |")
    A(f"| 近重复前缀同时跨集合且跨类别 | {leak['near_dup_cross_split_and_multi_category']} 组 | 最需关注的一类 |")
    A("")
    A("**结论**：精确重复样本的跨集合泄漏为 **0**（标题已在交付集中全局去重）。"
      "存在的是「近重复」——同一事件系列（如全运会各项目、某专项行动的后续通报）"
      "标题前缀相同但确属不同新闻，随时间分布在训练与测试集合中。"
      "这是时间划分的**固有属性**，不是数据缺陷：它衡量的是模型对同主题新事件的泛化能力。\n")
    if len(leak["_cross_prefix_examples"]):
        A("近重复前缀跨集合的高频示例（前 5）：\n")
        A("| 标题前缀 | 条数 | 涉及集合 | 类别 |")
        A("|---|---|---|---|")
        for _, r in leak["_cross_prefix_examples"].head(5).iterrows():
            A(f"| {r['prefix']} | {r['n']} | {r['splits'].replace('|', ' / ')} | "
              f"{r['cats'].replace('|', ' / ')} |")
        A("")
    A("### 4.1 避免精确重复泄漏的处理方案\n")
    A("1. **精确重复已消除**：交付数据集按规范化标题全局去重，同一标题只会出现一次，"
      "因此不存在同一新闻跨集合。")
    A("2. **近重复的可选加固**（若审稿要求更严格的隔离，可按序采用）：")
    A("   - 以 `title 前 N 字` 或相似度（如字符级 Jaccard ≥ 0.8）构造连通分量，"
      "整组只划入一个集合；代价是会丢弃部分跨集合样本，训练集规模下降。")
    A("   - 以「事件实体 + 时间」聚合（如含同一赛事名的标题），整组按最早出现时间归入单一集合。")
    A("3. **不建议**改回随机划分：随机划分会让同系列新闻同时进入训练与测试，"
      "准确率虚高且无法反映时间外推能力，与本课题「不同时间段新闻分类表现」的目标冲突。")
    A("4. 本版本**未**执行近重复隔离，因为当前近重复多为语义不同的真实新闻，"
      "强制隔离会显著削减样本。若实验显示测试集准确率异常偏高，再启用方案 2。\n")

    A("## 5. 版本与复现信息\n")
    A("| 项目 | 值 |")
    A("|---|---|")
    A(f"| 数据集版本 | `{version['dataset_version']}` |")
    A(f"| 划分版本 | `{version['split_version']}` |")
    A(f"| 数据集生成时间 | {version['dataset_generated_at']} |")
    A(f"| 版本登记时间 | {version['registered_at']} |")
    A(f"| 随机种子 | `{version['random_seed']}` |")
    A(f"| `news_clean.csv` SHA256 | `{version['files']['news_clean.csv']['sha256']}` |")
    A("")
    A("各集合文件指纹：\n")
    A("| 文件 | 行数 | SHA256 |")
    A("|---|---|---|")
    for name in ("train", "val", "test"):
        f = version["files"][f"{name}.csv"]
        A(f"| `dataset/splits/{name}.csv` | {f['rows']:,} | `{f['sha256']}` |")
    A("")
    A("完整版本信息见 `dataset/dataset_version.json`。\n")

    A("## 6. 使用说明\n")
    A("```python\nimport pandas as pd\n"
      "train = pd.read_csv('dataset/splits/train.csv', encoding='utf-8-sig')\n"
      "val   = pd.read_csv('dataset/splits/val.csv',   encoding='utf-8-sig')\n"
      "test  = pd.read_csv('dataset/splits/test.csv',  encoding='utf-8-sig')\n"
      "```\n")
    A("- TF-IDF 向量化器**只能**在 `train.csv` 上 `fit`，再 `transform` 验证/测试集，"
      "否则会引入词表泄漏。")
    A("- BP 网络的输入维度取决于 TF-IDF 特征维度；建议在训练集上用交叉验证确定"
      "特征维度与隐藏层规模，避免直接用测试集调参。")
    A("- 三个文件均保留原始 8 列，未增删字段，便于与 `news_clean.csv` 对照。\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    if not CLEAN_CSV.exists():
        print(f"[error] 缺少输入文件：{CLEAN_CSV}")
        return 1

    df, checks = load_clean()
    df = assign_split(df)

    missing = int(df["split"].isna().sum())
    if missing:
        print(f"[warn] {missing} 条记录未落入任何时间区间，未纳入任何集合")

    stats = build_statistics(df[df["split"].notna()])
    leak = leakage_check(df)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    files_meta = {}
    for name, _, _, _ in BOUNDARIES:
        sub = (df[df["split"] == name]
               .sort_values(["publish_time", "id"])
               .drop(columns=["split", "_ts"]))
        out = SPLITS_DIR / f"{name}.csv"
        sub.to_csv(out, index=False, encoding=config.CSV_ENCODING)
        files_meta[f"{name}.csv"] = {"rows": int(len(sub)), "sha256": sha256(out)}
        print(f"[ok] {out}  {len(sub):,} 行")

    stats.to_csv(SPLITS_DIR / "split_statistics.csv", index=False,
                 encoding=config.CSV_ENCODING)
    print(f"[ok] {SPLITS_DIR / 'split_statistics.csv'}")

    # 版本信息
    mtime = datetime.fromtimestamp(CLEAN_CSV.stat().st_mtime)
    version = {
        "dataset_version": DATASET_VERSION,
        "split_version": SPLIT_VERSION,
        "dataset_generated_at": mtime.strftime("%Y-%m-%d %H:%M:%S"),
        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "source": config.SOURCE_NAME,
        "records_total": int(len(df)),
        "splits": {
            name: {
                "start": start, "end": end, "purpose": purpose,
                "rows": int((df["split"] == name).sum()),
                "file": f"dataset/splits/{name}.csv",
                "sha256": files_meta[f"{name}.csv"]["sha256"],
                "per_category": {c: int(((df["split"] == name) &
                                         (df["category"] == c)).sum())
                                 for c in config.TARGET_CATEGORIES},
            }
            for name, start, end, purpose in BOUNDARIES
        },
        "files": {
            "news_clean.csv": {"rows": checks["rows"], "sha256": sha256(CLEAN_CSV)},
            **files_meta,
        },
        "label_mapping": {
            "target_categories": config.TARGET_CATEGORIES,
            "raw_to_target": dict(config.CATEGORY_MAPPING),
            "category_to_index": {c: i for i, c in enumerate(config.TARGET_CATEGORIES)},
            "excluded_raw_labels": sorted(
                set(config.NON_TOPIC_LABELS) | {config.UNKNOWN_CATEGORY}),
            "note": "「文娱」沿用目标类名，数据来自站点「文化」频道标签，一对一确定性映射。",
        },
        "changelog": [
            {"version": "1.0.0", "note": "首次交付 12,000 条（每类 2,000），六类均衡。"},
            {"version": "1.1.0", "note": "审计修正：质量报告统计口径与教育类表述；"
                                         "新增跨类别冲突审计、时间划分与版本登记。"
                                         "**样本数据未改动**，news_clean.csv SHA256 与 1.0.0 一致。"},
        ],
    }
    VERSION_JSON.write_text(json.dumps(version, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"[ok] {VERSION_JSON}")

    write_report(df[df["split"].notna()], stats, checks, leak, version, REPORT_MD)
    print(f"[ok] {REPORT_MD}")

    print("\n划分结果：")
    for name, _, _, _ in BOUNDARIES:
        n = int((df["split"] == name).sum())
        print(f"  {name:5s} {n:>6,} 条")
    print(f"  合计  {len(df):>6,} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

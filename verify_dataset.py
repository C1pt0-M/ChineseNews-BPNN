"""数据集完整性校验（只读，用于交付前自查与后续回归）。

逐项检查十一类约束，全部通过时退出码为 0，否则为 1。
校验对象与报告中的数字**必须一致**，因此本脚本也充当「报告与 CSV 是否同步」的守门人。

用法：
    python verify_dataset.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

import config
import model_config

DATASET = config.DATASET_DIR
SPLITS = DATASET / "splits"
VERSION_JSON = DATASET / "dataset_version.json"
QUALITY_MD = config.REPORT_DIR / "quality_report.md"
SPLIT_MD = config.REPORT_DIR / "time_split_report.md"
CONFLICT_MD = config.REPORT_DIR / "cross_category_conflict_report.md"
CRAWL_JSON = DATASET / "crawl_report.json"

TIME_RE = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
BOUNDARIES = [("train", "2025-01-01", "2025-12-31"),
              ("val", "2026-01-01", "2026-06-30"),
              ("test", "2026-07-01", "2026-10-31")]

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    results.clear()
    # ---------------- 交付数据集
    clean_path = DATASET / "news_clean.csv"
    check("news_clean.csv 存在", clean_path.exists())
    if not clean_path.exists():
        report()
        return 1

    df = pd.read_csv(clean_path, encoding=config.CSV_ENCODING, dtype=str)
    check("样本总量为 12,000", len(df) == 12000, f"实际 {len(df):,}")

    cats = config.TARGET_CATEGORIES
    vc = df["category"].value_counts()
    check("类别名称与目标六类完全一致",
          set(vc.index) == set(cats), f"实际 {sorted(vc.index)}")
    check("每个类别均为 2,000 条",
          all(int(vc.get(c, 0)) == 2000 for c in cats),
          " / ".join(f"{c}={int(vc.get(c, 0))}" for c in cats))

    check("无空标题（含纯空白）",
          int((df["title"].fillna("").str.strip() == "").sum()) == 0)
    check("无无效标签", bool(df["category"].isin(cats).all()))
    check("无重复 ID", int(df["id"].duplicated().sum()) == 0,
          f"重复 {int(df['id'].duplicated().sum())}")
    check("无重复 URL", int(df["url"].duplicated().sum()) == 0,
          f"重复 {int(df['url'].duplicated().sum())}")
    check("无重复标题", int(df["title"].duplicated().sum()) == 0,
          f"重复 {int(df['title'].duplicated().sum())}")
    check("时间字段可正确解析",
          bool(df["publish_time"].str.match(TIME_RE).all())
          and pd.to_datetime(df["publish_time"], errors="coerce").notna().all())
    check("字段完整（8 列）", list(df.columns) == [
        "id", "title", "category", "publish_time", "url",
        "archive_date", "source", "raw_category"], f"实际 {list(df.columns)}")

    # ---------------- 划分数据集
    needed = [SPLITS / f"{n}.csv" for n, _, _ in BOUNDARIES]
    check("划分文件齐备", all(p.exists() for p in needed))
    if all(p.exists() for p in needed):
        parts = {n: pd.read_csv(SPLITS / f"{n}.csv", encoding=config.CSV_ENCODING,
                                dtype=str)
                 for n, _, _ in BOUNDARIES}
        total = sum(len(p) for p in parts.values())
        check("三集合合计等于 12,000", total == 12000, f"实际 {total:,}")

        all_ids = set()
        overlap = 0
        for n, p in parts.items():
            overlap += len(all_ids & set(p["id"]))
            all_ids |= set(p["id"])
        check("三集合 ID 互不交叉", overlap == 0, f"交叉 {overlap}")
        check("三集合并集等于原数据集", all_ids == set(df["id"]),
              f"并集 {len(all_ids):,}")

        merged = pd.concat(parts.values())
        mv = merged["category"].value_counts()
        check("合并后各类仍为 2,000",
              all(int(mv.get(c, 0)) == 2000 for c in cats),
              " / ".join(f"{c}={int(mv.get(c, 0))}" for c in cats))

        okay = True
        detail = []
        for n, start, end in BOUNDARIES:
            ts = pd.to_datetime(parts[n]["publish_time"], errors="coerce")
            in_range = bool(((ts >= f"{start} 00:00:00") &
                             (ts <= f"{end} 23:59:59")).all())
            okay &= in_range
            detail.append(f"{n}:{start}~{end}{'' if in_range else ' 越界'}")
        check("各集合时间边界正确", okay, "；".join(detail))

        inter = 0
        names = list(parts)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                inter += len(set(parts[names[i]]["title"]) & set(parts[names[j]]["title"]))
        check("跨集合无精确重复标题", inter == 0, f"跨集合重复 {inter}")
        url_inter = sum(len(set(parts[a]["url"]) & set(parts[b]["url"]))
                        for i, a in enumerate(names) for b in names[i + 1:])
        check("三集合 URL 互不交叉", url_inter == 0, f"交叉 {url_inter}")
        expected_rows = {"train": 6549, "val": 3184, "test": 2267}
        for name, count in expected_rows.items():
            check(f"{name} 样本数正确", len(parts[name]) == count, str(len(parts[name])))

    # ---------------- 版本信息与指纹
    check("dataset_version.json 存在", VERSION_JSON.exists())
    if VERSION_JSON.exists():
        v = json.loads(VERSION_JSON.read_text(encoding="utf-8"))
        check("数据集固定为 v1.1.0", v.get("dataset_version") == "1.1.0")
        check("划分固定为 v1.0.0", v.get("split_version") == "1.0.0")
        check("版本号已登记", bool(v.get("dataset_version")),
              str(v.get("dataset_version")))
        check("随机种子已登记", v.get("random_seed") is not None,
              str(v.get("random_seed")))
        check("news_clean.csv SHA256 与实际一致",
              v["files"]["news_clean.csv"]["sha256"] == sha256(clean_path))
        for n, _, _ in BOUNDARIES:
            p = SPLITS / f"{n}.csv"
            if p.exists():
                check(f"{n}.csv SHA256 一致",
                      v["files"][f"{n}.csv"]["sha256"] == sha256(p))
        lm = v.get("label_mapping", {})
        check("标签映射完整",
              set(lm.get("target_categories", [])) == set(cats)
              and set(lm.get("category_to_index", {})) == set(cats))
        check("标签索引与模型配置逐项一致",
              lm.get("category_to_index") == model_config.LABEL_TO_INDEX
              and lm.get("target_categories") == model_config.LABELS)
        check("文化到文娱代理映射保持不变",
              lm.get("raw_to_target", {}).get("文化") == "文娱")

    # ---------------- 报告与数据同步
    for name, p in [("质量报告", QUALITY_MD), ("时间划分报告", SPLIT_MD),
                    ("跨类别冲突报告", CONFLICT_MD)]:
        check(f"{name}存在", p.exists(), str(p))

    if CRAWL_JSON.exists() and QUALITY_MD.exists():
        rep = json.loads(CRAWL_JSON.read_text(encoding="utf-8"))
        text = QUALITY_MD.read_text(encoding="utf-8")
        rows_after = int(rep.get("cleaning", {}).get("rows_after_clean", -1))
        delivered = int(rep.get("cleaning", {}).get("rows_delivered", -1))
        check("报告中清洗后合计与 crawl_report 一致",
              f"{rows_after:,}" in text, f"crawl_report={rows_after:,}")
        check("报告中交付样本量与 crawl_report 一致",
              f"{delivered:,}" in text, f"crawl_report={delivered:,}")
        check("报告未残留「教育可能无法达成」的过时表述",
              "可能无法达成" not in text)

    if SPLIT_MD.exists():
        text = SPLIT_MD.read_text(encoding="utf-8")
        stats_csv = SPLITS / "split_statistics.csv"
        if stats_csv.exists():
            st = pd.read_csv(stats_csv, encoding=config.CSV_ENCODING)
            okay = all(f"{int(st.loc[(st['split'] == n) & (st['category'] == '合计'), 'count'].iloc[0]):,}" in text
                       for n, _, _ in BOUNDARIES)
            check("划分报告样本数与 split_statistics.csv 一致", okay)

    return report()


def report() -> int:
    width = max(len(n) for n, _, _ in results) + 2
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        flag = "PASS" if ok else "FAIL"
        line = f"[{flag}] {name.ljust(width)}"
        if detail and not ok:
            line += f"  <- {detail}"
        elif detail and ok:
            line += f"  ({detail})"
        print(line)
    print(f"\n合计 {len(results)} 项，通过 {passed} 项，失败 {len(results) - passed} 项")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

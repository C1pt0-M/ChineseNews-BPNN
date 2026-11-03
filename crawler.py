"""请求与解析层。

职责边界：
  - 构造归档页 URL
  - 带超时/重试/限速的 HTTP 请求
  - 将归档页 HTML 解析为结构化新闻记录
不涉及清洗、去重、落盘。
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# 内容页 URL 判定：以数字 ID + .shtml 结尾。
# 普通文章为 8 位数字（/gj/2025/01-01/10346216.shtml）；
# 高清图集为 7 位数字（/tp/hd2011/2025/01-05/1136408.shtml）；
# 视频页为 news 前缀 + 7 位数字（/sh/shipin/cns/2025/01-01/news1009460.shtml）。
# 三者均为具体内容页，而非栏目首页（如 /world.shtml）。
ARTICLE_URL_RE = re.compile(r"/(?:news)?(\d{7,10})\.shtml$")
# 媒体形态路径标记，用于区分图集与视频
GALLERY_PATH_MARK = "/tp/"
VIDEO_PATH_MARK = "/shipin/"
CATEGORY_BRACKET_RE = re.compile(r"^[\[【]\s*|\s*[\]】]$")


# ---------------------------------------------------------------- 数据模型
@dataclass
class FetchResult:
    url: str
    status: int | None = None
    ok: bool = False
    not_found: bool = False
    content: str = ""
    byte_len: int = 0
    error: str = ""
    attempts: int = 0


@dataclass
class ParseResult:
    records: list[dict] = field(default_factory=list)
    item_count: int = 0          # 列表项总数（含空占位）
    skipped_empty: int = 0       # 空 <li> 占位
    skipped_no_title: int = 0    # 无标题链接
    has_pagination: bool = False  # 是否出现分页控件
    container_found: bool = False


# ---------------------------------------------------------------- URL 工具
def build_archive_url(day: date) -> str:
    """按日期构造历史归档页 URL。"""
    return config.ARCHIVE_URL_TEMPLATE.format(year=f"{day:%Y}", mmdd=f"{day:%m%d}")


def normalize_url(href: str, base: str = "https://www.chinanews.com") -> str:
    """规范化新闻链接：补全协议/域名、去除查询串与锚点、统一尾斜杠。

    用于去重键生成，不改变原始链接语义。
    """
    if not href:
        return ""
    href = href.strip()
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = base + href
    elif not href.startswith("http"):
        href = base + "/" + href.lstrip("/")

    parts = urlsplit(href)
    host = parts.netloc.lower()
    if host.endswith("chinanews.com.cn"):
        host = host.replace("chinanews.com.cn", "chinanews.com")
    path = re.sub(r"/+", "/", parts.path)
    if path.endswith("/index.shtml"):
        path = path[: -len("index.shtml")]
    return urlunsplit(("https", host, path, "", ""))


def is_article_url(url: str) -> bool:
    """判断链接是否指向具体新闻文章（而非栏目首页/专题页）。"""
    return bool(ARTICLE_URL_RE.search(urlsplit(url).path))


def extract_article_id(url: str) -> str:
    m = ARTICLE_URL_RE.search(urlsplit(url).path)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- 请求
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(config.REQUEST_HEADERS)
    return s


def fetch_archive_page(
    session: requests.Session,
    url: str,
    delay: float = config.DEFAULT_DELAY,
) -> FetchResult:
    """抓取单个归档页，含有限次重试与指数退避。

    仅对网络异常与 5xx 重试；404 视为“当日无归档”，不重试。
    """
    res = FetchResult(url=url)
    for attempt in range(1, config.MAX_RETRIES + 1):
        res.attempts = attempt
        try:
            resp = session.get(url, timeout=config.TIMEOUT)
            res.status = resp.status_code
            if resp.status_code == 404:
                res.not_found = True
                _sleep(delay)
                return res
            if resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()

            # 显式指定 UTF-8：站点未在响应头声明编码
            resp.encoding = config.FORCED_ENCODING
            res.content = resp.text
            res.byte_len = len(resp.content)
            res.ok = True
            _sleep(delay)
            return res

        except (requests.Timeout, requests.ConnectionError) as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            logger.warning("请求失败(%d/%d) %s -> %s",
                           attempt, config.MAX_RETRIES, url, res.error)
        except requests.HTTPError as exc:
            res.error = f"HTTPError: {exc}"
            logger.warning("HTTP 异常(%d/%d) %s -> %s",
                           attempt, config.MAX_RETRIES, url, res.error)
        except requests.RequestException as exc:
            res.error = f"{type(exc).__name__}: {exc}"
            logger.warning("请求异常(%d/%d) %s -> %s",
                           attempt, config.MAX_RETRIES, url, res.error)

        if attempt < config.MAX_RETRIES:
            time.sleep(config.RETRY_BACKOFF ** attempt)

    _sleep(delay)
    return res


def _sleep(delay: float) -> None:
    """请求间隔：固定延迟叠加小幅随机抖动，降低访问节奏的机械性。"""
    if delay and delay > 0:
        time.sleep(delay + random.uniform(0, 0.3))


# ---------------------------------------------------------------- 解析
def normalize_category(raw_text: str) -> str:
    """将 [ 社会 ] / [其他] 之类的标签规范化为裸分类名。

    保留站点原始用词，不做语义推断；无法识别时返回 'unknown'。
    """
    if not raw_text:
        return config.UNKNOWN_CATEGORY
    text = CATEGORY_BRACKET_RE.sub("", raw_text.strip())
    text = text.replace("[", "").replace("]", "").replace("【", "").replace("】", "")
    text = re.sub(r"\s+", "", text)
    return text or config.UNKNOWN_CATEGORY


def build_publish_time(archive_date: date, raw_time: str) -> tuple[str, str]:
    """将 '1-1 23:21' 与归档日期合成为标准时间戳。

    返回 (publish_time, issue)。issue 非空表示时间与归档日期不一致。
    归档页仅给出“月-日 时:分”，年份取自归档路径。
    """
    raw_time = (raw_time or "").strip()
    m = re.match(r"^(\d{1,2})\s*-\s*(\d{1,2})\s+(\d{1,2}):(\d{2})$", raw_time)
    if not m:
        return "", f"无法解析时间格式: {raw_time!r}"

    month, day, hour, minute = (int(g) for g in m.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31 and hour <= 23 and minute <= 59):
        return "", f"时间数值越界: {raw_time!r}"
    if (month, day) != (archive_date.month, archive_date.day):
        return "", f"时间与归档日不一致: 页面={month}-{day} 归档={archive_date:%m-%d}"

    try:
        ts = date(archive_date.year, month, day)
    except ValueError:
        return "", f"非法日期: {archive_date.year}-{month}-{day}"
    return f"{ts:%Y-%m-%d} {hour:02d}:{minute:02d}:00", ""


def parse_archive_page(html: str, archive_day: date, page_url: str) -> ParseResult:
    """从归档页 HTML 中提取全部新闻记录。

    结构（2025-01 实测）：
      div.content_list > ul > li
        ├── div.dd_lm     分类标签，如 [ 社会 ]（可能是纯文本，无 <a>）
        ├── div.dd_bt > a 标题及文章链接
        └── div.dd_time   发布时间，如 1-1 23:21
      另有大量空 <li></li> 作为分隔占位，需过滤。
    """
    out = ParseResult()
    soup = BeautifulSoup(html, "lxml")

    container = soup.select_one(config.SEL_LIST_CONTAINER)
    if container is None:
        logger.error("未找到列表容器 %s：%s", config.SEL_LIST_CONTAINER, page_url)
        return out
    out.container_found = True
    out.has_pagination = soup.select_one(config.SEL_PAGINATION) is not None

    for li in container.select(config.SEL_ITEM):
        out.item_count += 1

        title_link = li.select_one(config.SEL_TITLE_LINK)
        if title_link is None or not title_link.get("href"):
            out.skipped_empty += 1
            continue

        title = title_link.get_text(strip=True)
        if not title:
            out.skipped_no_title += 1
            continue

        raw_href = title_link["href"].strip()
        url = normalize_url(raw_href)

        cat_node = li.select_one(config.SEL_CATEGORY)
        category = normalize_category(
            cat_node.get_text(" ", strip=True) if cat_node else ""
        )

        time_node = li.select_one(config.SEL_TIME)
        raw_time = time_node.get_text(strip=True) if time_node else ""
        publish_time, issue = build_publish_time(archive_day, raw_time)

        article_id = extract_article_id(url)

        out.records.append({
            "id": f"cn{article_id}" if article_id else "",
            "title": title,
            "category": category,
            "publish_time": publish_time,
            "url": url,
            "archive_date": f"{archive_day:%Y-%m-%d}",
            "source": config.SOURCE_NAME,
            # 以下为解析诊断字段，最终 CSV 前会剔除
            "_raw_category": cat_node.get_text(" ", strip=True) if cat_node else "",
            "_raw_time": raw_time,
            "_is_article": is_article_url(url),
            "_is_gallery": GALLERY_PATH_MARK in urlsplit(url).path,
            "_is_video": VIDEO_PATH_MARK in urlsplit(url).path,
            "_time_issue": issue,
            "_page_url": page_url,
        })

    return out


def daterange(start: date, end: date) -> Iterable[date]:
    """按日迭代（含首尾）。"""
    cur = start
    while cur <= end:
        yield cur
        cur = date.fromordinal(cur.toordinal() + 1)

"""中国新闻网历史滚动新闻采集 —— 全局配置

所有结构性常量集中在此，便于站点改版后单点维护。
选择器均基于 2025-01 实际页面验证得出（见 reports/stage1_structure_analysis.md），
非经验猜测。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
STATE_DIR = BASE_DIR / "state"
RAW_PARTS_DIR = STATE_DIR / "raw_parts"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

# ---------------------------------------------------------------- 站点
# 该模板与站点自身 calendar.js 中的跳转逻辑一致：
#   url = '/scroll-news/year/monthday/news.shtml'
ARCHIVE_URL_TEMPLATE = "https://www.chinanews.com/scroll-news/{year}/{mmdd}/news.shtml"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 站点响应头未声明 charset，实测为 UTF-8；必须显式覆盖，
# 否则 requests 会按 ISO-8859-1 解码导致中文乱码。
FORCED_ENCODING = "utf-8"

TIMEOUT = 25              # 单次请求超时（秒）
MAX_RETRIES = 3           # 失败重试次数上限
RETRY_BACKOFF = 2.0       # 重试退避基数（指数退避）
DEFAULT_DELAY = 1.2       # 相邻请求间隔（秒），单线程限速

# ---------------------------------------------------------------- 解析选择器
SEL_LIST_CONTAINER = "div.content_list"
SEL_ITEM = "li"
SEL_CATEGORY = "div.dd_lm"
SEL_TITLE_LINK = "div.dd_bt > a"
SEL_TIME = "div.dd_time"
SEL_PAGINATION = ".pagebox"          # 若出现说明单日内容被分页
SEL_CALENDAR = "#Calendar_Cnl_0"     # 往日回顾日历控件的锚点

# 404 哨兵页特征：状态码 404 + 固定体积 6534B + 标题含“404页面”
NOT_FOUND_TITLE_MARK = "404页面"
NOT_FOUND_BYTES = 6534

# ---------------------------------------------------------------- 数据字段
SOURCE_NAME = "中国新闻网"

DATE_START_DEFAULT = "2025-01-01"
DATE_END_DEFAULT = "2026-10-31"

# ---------------------------------------------------------------- 分类映射
# 目标六类。注意：站点滚动列表并不存在“文娱”标签，
# 其文娱内容统一归入“文化”频道标签，故 文化 -> 文娱 是显式代理映射。
TARGET_CATEGORIES = ["财经", "体育", "文娱", "教育", "社会", "国际"]

CATEGORY_MAPPING = {
    "财经": "财经",
    "体育": "体育",
    "文化": "文娱",   # 代理映射：站点无“文娱”原始标签
    "教育": "教育",
    "社会": "社会",
    "国际": "国际",
}

# 原始标签中属于“媒体形态”而非“主题”的，不参与主题分类
NON_TOPIC_LABELS = {"图片", "视频"}

UNKNOWN_CATEGORY = "unknown"

CSV_ENCODING = "utf-8-sig"


def ensure_dirs() -> None:
    """创建运行所需目录。"""
    for d in (DATASET_DIR, STATE_DIR, RAW_PARTS_DIR, REPORT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)

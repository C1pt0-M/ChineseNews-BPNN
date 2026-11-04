"""特征处理：jieba 分词 → TF-IDF →（卡方选择 | TruncatedSVD）

三种方案共用同一份 TF-IDF，仅在其后接不同的变换，保证对比公平：

    A：TF-IDF                        → 稀疏矩阵（高维）
    B：TF-IDF → SelectKBest(chi2)    → 稀疏矩阵（低维）
    C：TF-IDF → TruncatedSVD         → 稠密矩阵（低维），可选 z-score

FeatureBuilder.fit() 创建词表及变换器，transform() 未 fit 直接抛错。
API 无法辨识传入文本属于哪个集合；只在训练集 fit 由训练入口保证，
并通过入口集成测试核查，不能将方法参数名视为数据泄漏防护。
"""

from __future__ import annotations

import hashlib
import pickle
import re
import warnings

# jieba 依赖旧的 pkg_resources，Python 3.12 下会刷屏弃用警告，与实验无关
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import jieba
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler

import model_config as C

jieba.setLogLevel(60)   # 关闭 jieba 首次加载词典的调试输出

# 至少包含一个汉字或拉丁字母的 token 才保留：
# 过滤标点、纯数字、全角符号等无区分度的碎片
_KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z]")
_LATIN_RE = re.compile(r"^[A-Za-z]+$")


def tokenize(text: str) -> list[str]:
    """中文标题分词（jieba 精确模式）。

    规则：
      - 去首尾空白，空串返回空列表
      - 丢弃不含汉字/字母的 token（标点、纯数字、空格）
      - 纯拉丁 token 统一小写，避免 "NBA"/"nba" 被当作两个词
    """
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    tokens = []
    for tok in jieba.lcut(text):
        tok = tok.strip()
        if not tok or not _KEEP_RE.search(tok):
            continue
        if _LATIN_RE.match(tok):
            tok = tok.lower()
        tokens.append(tok)
    return tokens


def tokenize_corpus(texts: list[str], cache_key: str | None = None) -> list[list[str]]:
    """对整份语料分词，可选磁盘缓存。

    分词结果是纯函数输出，可按语料哈希缓存，避免多组实验、多随机种子
    重复分词（12,000 条标题约数秒，属于可省的开销）。
    """
    if cache_key is None:
        return [tokenize(t) for t in texts]

    cache_path = C.CACHE_DIR / f"tokens_{cache_key}.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        # 长度校验：防止缓存与当前语料错位
        if len(cached) == len(texts):
            return cached

    tokens = [tokenize(t) for t in texts]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(tokens, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tokens


def _identity_tokens(tokens):
    """TF-IDF 的分析器入口：输入已是 token 列表，直接透传。

    分词在 FeatureBuilder 之外完成（可缓存、可复用），因此这里不能再调用
    jieba，否则会把「列表」当字符串处理而得到空词表。
    """
    return tokens


def corpus_cache_key(*text_lists: list[str]) -> str:
    h = hashlib.sha256()
    for texts in text_lists:
        h.update("\n".join(texts).encode("utf-8"))
        h.update(b"||")
    return h.hexdigest()[:16]


class FeatureBuilder:
    """统一的特征流水线，fit 只允许在训练集上调用。

    参数
    ----
    scheme : "A" | "B" | "C"
    k      : 方案 B 保留的特征数
    n_components : 方案 C 的 SVD 维度
    seed   : SVD 随机种子
    """

    def __init__(self, scheme: str, k: int | None = None,
                 n_components: int | None = None, seed: int = C.SEED,
                 tfidf_params: dict | None = None):
        if scheme not in ("A", "B", "C"):
            raise ValueError(f"未知方案：{scheme}")
        if scheme == "B" and not k:
            raise ValueError("方案 B 必须指定 k")
        if scheme == "C" and not n_components:
            raise ValueError("方案 C 必须指定 n_components")

        self.scheme = scheme
        self.k = k
        self.n_components = n_components
        self.seed = seed
        self.tfidf_params = dict(tfidf_params or C.TFIDF_PARAMS)
        # 输入已是 jieba 分好的 token 列表，故用恒等分析器 + 关闭默认正则
        self.tfidf_params["tokenizer"] = _identity_tokens
        self.tfidf_params["preprocessor"] = None
        self.tfidf_params["token_pattern"] = None

        self.vectorizer: TfidfVectorizer | None = None
        self.selector: SelectKBest | None = None
        self.svd: TruncatedSVD | None = None
        self.scaler: StandardScaler | None = None

        self.n_features_in_: int | None = None    # TF-IDF 原始维度
        self.n_features_out_: int | None = None   # 最终输入 BP 的维度
        self.fit_rows_: int | None = None
        self.extra_info: dict = {}

    # ------------------------------------------------------------ fit
    def fit(self, train_tokens: list[list[str]], y_train: np.ndarray) -> "FeatureBuilder":
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)
        X = self.vectorizer.fit_transform(train_tokens)
        self.n_features_in_ = X.shape[1]
        self.fit_rows_ = X.shape[0]

        if self.scheme == "B":
            k = min(self.k, X.shape[1])
            self.selector = SelectKBest(chi2, k=k).fit(X, y_train)
            X = self.selector.transform(X)
            self.extra_info["chi2_k_requested"] = self.k
            self.extra_info["chi2_k_effective"] = int(k)

        elif self.scheme == "C":
            n_comp = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
            self.svd = TruncatedSVD(n_components=n_comp, random_state=self.seed)
            Xd = self.svd.fit_transform(X)
            if C.SVD_STANDARDIZE:
                self.scaler = StandardScaler().fit(Xd)
                Xd = self.scaler.transform(Xd)
            self.extra_info["svd_components_effective"] = int(n_comp)
            self.extra_info["svd_explained_variance_ratio_sum"] = float(
                self.svd.explained_variance_ratio_.sum()
            )
            X = Xd

        self.n_features_out_ = X.shape[1]
        return self

    # ------------------------------------------------------------ transform
    def transform(self, tokens: list[list[str]]):
        if self.vectorizer is None:
            raise RuntimeError("FeatureBuilder 尚未 fit，禁止在未 fit 的情况下 transform")

        X = self.vectorizer.transform(tokens)
        if self.scheme == "B":
            X = self.selector.transform(X)
        elif self.scheme == "C":
            Xd = self.svd.transform(X)
            if self.scaler is not None:
                Xd = self.scaler.transform(Xd)
            X = Xd
        return X

    def fit_transform(self, train_tokens, y_train):
        self.fit(train_tokens, y_train)
        return self.transform(train_tokens)

    # ------------------------------------------------------------ 诊断
    def describe(self) -> dict:
        info = {
            "scheme": self.scheme,
            "tfidf_features": self.n_features_in_,
            "input_dim": self.n_features_out_,
            "fit_rows": self.fit_rows_,
            "sparse": self.scheme in ("A", "B"),
        }
        info.update(self.extra_info)
        return info


def is_sparse(X) -> bool:
    return sp.issparse(X)

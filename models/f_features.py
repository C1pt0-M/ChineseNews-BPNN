"""Opt-in title-only character and word/character sparse feature pipelines."""
from __future__ import annotations

import scipy.sparse as sp
import sklearn
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import normalize

import model_config as C
from .features import FeatureBuilder, tokenize_corpus


def _vectorizer_params(vectorizer):
    """Record explicit and library-default settings in JSON-compatible form."""
    result = {}
    for key, value in vectorizer.get_params(deep=False).items():
        if callable(value):
            value = f"{value.__module__}.{value.__qualname__}"
        result[key] = value
    return result


class TitleFeatureBuilder:
    """Fits vocabularies, IDF and selectors only on supplied training titles."""
    raw_text_input = True

    def __init__(self, scheme, seed=C.SEED, word_k=None, char_k=None):
        if scheme not in ("F_char", "F_word_char"):
            raise ValueError(f"Unknown title feature scheme: {scheme}")
        for name, value in (("word_k", word_k), ("char_k", char_k)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if scheme == "F_char" and word_k is not None:
            raise ValueError("word_k applies only to F_word_char")
        self.scheme = scheme
        self.char_params = dict(C.CHAR_TFIDF_PARAMS)
        self.char_k = char_k if char_k is not None else (10000 if scheme == "F_char" else 5000)
        self.word = (FeatureBuilder("B", k=word_k if word_k is not None else 5000, seed=seed)
                     if scheme == "F_word_char" else None)
        self.char_vectorizer = None
        self.char_selector = None

    def fit(self, train_texts, y_train):
        self.char_vectorizer = TfidfVectorizer(**self.char_params)
        chars = self.char_vectorizer.fit_transform(train_texts)
        self.char_selector = SelectKBest(chi2, k=min(self.char_k, chars.shape[1])).fit(chars, y_train)
        if self.word is not None:
            self.word.fit(tokenize_corpus(train_texts), y_train)
        self.fit_rows_ = len(train_texts)
        self.char_dim_ = int(self.char_selector.get_support().sum())
        self.word_dim_ = self.word.n_features_out_ if self.word is not None else 0
        self.n_features_in_ = chars.shape[1] + (self.word.n_features_in_ if self.word else 0)
        self.n_features_out_ = self.char_dim_ + self.word_dim_
        return self

    def transform(self, texts):
        if self.char_selector is None:
            raise RuntimeError("TitleFeatureBuilder has not been fitted")
        chars = self.char_selector.transform(self.char_vectorizer.transform(texts))
        if self.word is None:
            return chars.tocsr()
        words = self.word.transform(tokenize_corpus(texts))
        return normalize(sp.hstack((words, chars), format="csr"), norm="l2", copy=False)

    def describe(self):
        return {
            "scheme": self.scheme, "input_source": "title", "fit_rows": self.fit_rows_,
            "sparse": True, "tfidf_features": self.n_features_in_, "input_dim": self.n_features_out_,
            "library_versions": {"scikit_learn": sklearn.__version__, "jieba": jieba.__version__},
            "word": None if self.word is None else {
                "tokenization": "models.features.tokenize: jieba precise mode; filter non-Chinese/non-Latin tokens; lowercase Latin",
                "tfidf_params": _vectorizer_params(self.word.vectorizer),
                "chi2_k_requested": self.word.k, "tfidf_features": self.word.n_features_in_,
                "selected_features": self.word_dim_,
            },
            "char": {"tfidf_params": _vectorizer_params(self.char_vectorizer), "chi2_k_requested": self.char_k,
                     "tfidf_features": len(self.char_vectorizer.vocabulary_), "selected_features": self.char_dim_},
            "normalization": ("each TF-IDF branch L2 before chi2; concatenated selected features row L2"
                              if self.word else "character TF-IDF row L2 before chi2; no post-selection normalization"),
        }

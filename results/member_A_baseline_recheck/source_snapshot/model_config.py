"""模型与实验阶段全局配置

与采集阶段的 config.py 分离：config.py 管站点与清洗，本文件管建模实验。
所有可调超参集中在此，train_bp.py / evaluate_test.py / report_experiments.py
均从此处读取，保证「同一套参数」贯穿训练、评估与报告，避免报告与代码脱节。
"""

from pathlib import Path

# ---------------------------------------------------------------- 路径
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
SPLIT_DIR = DATASET_DIR / "splits"
RESULT_DIR = BASE_DIR / "results"
RUN_DIR = RESULT_DIR / "runs"
CACHE_DIR = RESULT_DIR / "cache"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DATASET_VERSION_FILE = DATASET_DIR / "dataset_version.json"

# ---------------------------------------------------------------- 复现
SEED = 42
# 多随机种子重复实验（仅用于最终最佳配置）
MULTI_SEEDS = [42, 43, 44, 45, 46]

# ---------------------------------------------------------------- 标签
# 与 dataset_version.json 的 category_to_index 完全一致，顺序即输出层索引顺序
LABELS = ["财经", "体育", "文娱", "教育", "社会", "国际"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}
N_CLASSES = len(LABELS)

# ---------------------------------------------------------------- 文本与 TF-IDF
# 中文标题需分词；标点与纯数字 token 在 tokenize 阶段已过滤
TFIDF_PARAMS = {
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "max_features": 20000,
    "ngram_range": (1, 2),
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
    "token_pattern": None,   # 使用自定义 tokenizer，关闭默认正则
    "lowercase": False,      # 中文无大小写；英文 token 在 tokenize 内统一小写
}

# ---------------------------------------------------------------- 特征方案
# A 原始 TF-IDF；B + 卡方选择；C + TruncatedSVD
CHI2_K_OPTIONS = [2000, 5000, 10000]
SVD_COMPONENTS_OPTIONS = [100, 200, 300]
SVD_STANDARDIZE = True   # SVD 后做 z-score，均值/方差只从训练集计算

# ---------------------------------------------------------------- BP 网络
BP_PARAMS = {
    "hidden_dim": 256,
    "dropout": 0.2,
    "n_classes": N_CLASSES,
}

TRAIN_PARAMS = {
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,          # 验证集 macro-F1 连续 10 轮无改善即停止
    "min_delta": 1e-4,       # 判定「改善」的最小幅度
    "grad_clip": 5.0,
}

# ---------------------------------------------------------------- 指标
METRIC_NAMES = ["accuracy", "precision_macro", "recall_macro",
                "f1_macro", "f1_weighted"]
PRIMARY_METRIC = "f1_macro"   # 模型选择主指标

# ---------------------------------------------------------------- 实验清单
# 每条为一个待比较配置；id 用于结果目录与报告表
EXPERIMENTS = [
    {"id": "A_raw",         "scheme": "A", "label": "A 原始 TF-IDF",            "k": None,   "n_components": None},
    {"id": "B_chi2_2000",   "scheme": "B", "label": "B 卡方 2000",             "k": 2000,   "n_components": None},
    {"id": "B_chi2_5000",   "scheme": "B", "label": "B 卡方 5000",             "k": 5000,   "n_components": None},
    {"id": "B_chi2_10000",  "scheme": "B", "label": "B 卡方 10000",            "k": 10000,  "n_components": None},
    {"id": "C_svd_100",     "scheme": "C", "label": "C SVD 100",               "k": None,   "n_components": 100},
    {"id": "C_svd_200",     "scheme": "C", "label": "C SVD 200",               "k": None,   "n_components": 200},
    {"id": "C_svd_300",     "scheme": "C", "label": "C SVD 300",               "k": None,   "n_components": 300},
]

SCHEME_NAMES = {
    "A": "原始 TF-IDF（稀疏高维，直接输入 BP）",
    "B": "TF-IDF + 卡方特征选择（稀疏降维）",
    "C": "TF-IDF + TruncatedSVD（稠密低维）",
}


def experiment_by_id(exp_id: str) -> dict:
    """按 id 取实验配置，找不到时抛错而非静默返回空配置。"""
    for exp in EXPERIMENTS:
        if exp["id"] == exp_id:
            return exp
    raise KeyError(f"未知实验 id: {exp_id}，可选：{[e['id'] for e in EXPERIMENTS]}")

"""建模阶段代码包

模块划分：
    data      —— 读取时间划分文件并校验 SHA256（防止误用被改动的数据）
    features  —— 分词、TF-IDF、卡方选择、TruncatedSVD、标准化
    bp        —— 单隐层 BP 神经网络定义
    trainer   —— 训练循环、早停、指标计算与落盘
"""

from .data import load_splits, verify_dataset
from .features import FeatureBuilder, tokenize
from .bp import BPNet
from .trainer import train_bp_model, predict, set_seed

__all__ = [
    "load_splits",
    "verify_dataset",
    "FeatureBuilder",
    "tokenize",
    "BPNet",
    "train_bp_model",
    "predict",
    "set_seed",
]


"""单隐层 BP 神经网络

原始结构：
    输入层(d) → 全连接(hidden) → ReLU → Dropout → 输出层(6)

改进结构：
    输入层(d) → Input Dropout → 全连接(hidden)
    → ReLU → Dropout → 输出层(6)

input_dropout=0.0 时保持原始 BP 的行为。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import model_config as C


class BPNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = C.BP_PARAMS["hidden_dim"],
        n_classes: int = C.N_CLASSES,
        dropout: float = C.BP_PARAMS["dropout"],
        input_dropout: float = 0.0,
    ):
        super().__init__()

        if not 0.0 <= input_dropout < 1.0:
            raise ValueError("input_dropout 必须满足 0 <= p < 1")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.input_dropout_p = input_dropout

        # 新增：输入层随机特征遮蔽
        self.input_drop = nn.Dropout(p=input_dropout)

        # 原始 BP 网络结构保持不变
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅在训练模式下随机遮蔽输入特征
        x = self.input_drop(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

    def count_parameters(self) -> int:
        """可训练参数总量。"""
        return sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

    def architecture(self) -> dict:
        return {
            "model_type": "bp",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "activation": "ReLU",
            "dropout": self.dropout_p,
            "input_dropout": self.input_dropout_p,
            "output_dim": self.fc2.out_features,
            "n_parameters": self.count_parameters(),
        }


class LinearNet(nn.Module):
    """A single affine classifier, with no hidden layer or dropout."""
    def __init__(self, input_dim: int, n_classes: int = C.N_CLASSES):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict:
        return {"model_type": "linear", "input_dim": self.linear.in_features,
                "output_dim": self.linear.out_features, "dropout": 0.0,
                "n_parameters": self.count_parameters()}


class HybridNet(nn.Module):
    """Add a linear classifier's logits to the original BP branch's logits."""
    def __init__(self, input_dim: int, hidden_dim: int = C.BP_PARAMS["hidden_dim"],
                 n_classes: int = C.N_CLASSES, dropout: float = C.BP_PARAMS["dropout"],
                 input_dropout: float = 0.0):
        super().__init__()
        self.bp = BPNet(input_dim, hidden_dim, n_classes, dropout, input_dropout)
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bp(x) + self.linear(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict:
        arch = self.bp.architecture()
        arch.update(model_type="hybrid", n_parameters=self.count_parameters())
        return arch


def build_model(architecture: dict) -> nn.Module:
    """Rebuild a model from metadata; missing legacy fields retain BP defaults."""
    model_type = architecture.get("model_type", "bp")
    input_dim = architecture["input_dim"]
    n_classes = architecture["output_dim"]
    if model_type == "linear":
        return LinearNet(input_dim, n_classes)
    if model_type in ("bp", "hybrid"):
        cls = BPNet if model_type == "bp" else HybridNet
        return cls(input_dim=input_dim, hidden_dim=architecture["hidden_dim"],
                   n_classes=n_classes, dropout=architecture["dropout"],
                   input_dropout=architecture.get("input_dropout", 0.0))
    raise ValueError(f"Unknown model type: {model_type}")

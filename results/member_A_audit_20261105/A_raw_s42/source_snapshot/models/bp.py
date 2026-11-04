"""单隐层 BP 神经网络

结构（与任务书一致）：

    输入层(d) → 全连接(hidden) → ReLU → Dropout → 输出层(6)

对三种特征方案使用完全相同的结构与超参，仅输入维度 d 不同，
因此实验差异只来自特征处理方式。
"""

from __future__ import annotations

import torch
import torch.nn as nn

import model_config as C


class BPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = C.BP_PARAMS["hidden_dim"],
                 n_classes: int = C.N_CLASSES, dropout: float = C.BP_PARAMS["dropout"]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))

    def count_parameters(self) -> int:
        """可训练参数总量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "activation": "ReLU",
            "dropout": self.dropout_p,
            "output_dim": self.fc2.out_features,
            "n_parameters": self.count_parameters(),
        }

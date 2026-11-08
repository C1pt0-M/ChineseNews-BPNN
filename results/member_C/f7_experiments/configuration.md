# F.7 配置与结果位置

## 共同条件

- 训练集：已审核 F3-A，7,531 条，SHA-256 `7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880`。
- 验证集：原固定验证集，3,184 条，SHA-256 `9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0`；种子为 42–46。
- 输入仅为标题：词级沿用原 jieba 分词及词级 1–2 gram TF-IDF，最多选择 7,000 维；字级 2–4 gram TF-IDF，最多选择 3,000 维；各分支的卡方选择器、词表和 IDF 仅在训练集拟合。两支稀疏拼接后逐行 L2 归一化，实际分别为 7,000、3,000 维。
- 两支 TF-IDF 均使用 `min_df=2`、`max_df=0.95`、`sublinear_tf=True`、IDF 和 L2；词级最多保留 20,000 个 TF-IDF 候选，字级 TF-IDF 不设候选数上限。完整参数、实际选择数与库版本见每次运行的 `metrics.json` 和 `feature_builder.pkl`。
- 单隐藏层 BP：输入 Dropout 0 → Linear(10,000, hidden) → ReLU → Dropout 0.2 → Linear(hidden, 6)。64 宽度 640,454 参数，256 宽度 2,561,798 参数。
- PyTorch 2.4.1+cu121 的 CPU 路径、14 个 Torch 线程；Adam，batch 64，weight decay 0.0001，梯度裁剪 5.0，最多 100 轮。按验证 Macro-F1 和 `min_delta=0.0001` 选择最佳 checkpoint，连续 10 轮无改善早停。
- 其余训练参数与实现沿用 F6-B；F.7 入口调用原 `train_bp.run_single`，从而保留原入口的 F 系列配置保护，不修改原训练函数或网络。

## 分组与文件

| 组 | hidden | learning rate | 结果位置 |
|---|---:|---:|---|
| F7-0 | 64 | 0.0003 | seed=42：`results/member_C/f6_experiments/runs/F6_B_s42/`；seed=43–46：`results/member_C/f6_experiments/f6_b_multiseed/runs/F6_B_s{seed}/` |
| F7-1 | 256 | 0.0003 | `results/member_C/f7_experiments/runs/F7_1_s{seed}/` |
| F7-2 | 64 | 0.001 | `results/member_C/f7_experiments/runs/F7_2_s{seed}/` |
| F7-3 | 256 | 0.001 | `results/member_C/f7_experiments/runs/F7_3_s{seed}/` |

每次运行均保存 `metrics.json`、`history.csv`、`model.pt`、`feature_builder.pkl`、`val_predictions.csv`、训练曲线及混淆矩阵。F7-0 为经逐条复核的历史产物，没有重新训练。

实验命令与聚合表见同目录 `summary.md`；完整哈希、严格重建和预测核验见 `audit.json`。原固定验证集既用于早停也用于比较，且 F6-B 曾按 seed=42 预选，因此本实验不等于独立测试。

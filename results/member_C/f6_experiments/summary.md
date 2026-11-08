# F.6 seed=42 特征配额受控对照

固定 F3-A 训练集 7,531 条及原验证集 3,184 条，BP hidden=64、学习率 0.0003、dropout=0.2、input dropout=0。训练集与验证集 SHA-256、特征配置、保存模型与预测和早停经 `audit.json` 所述检查。三次均仅是 seed=42，性能差异不能视作多种子稳定结果。

实际训练命令：

```powershell
python train_bp.py --exp F6_A F6_B --seeds 42 --model-type bp --hidden 64 --lr 0.0003 --dropout 0.2 --input-dropout 0 --train-file results/member_C/f3_analysis/f3_a_dataset_formal/train.csv --train-sha256 7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880 --output-root results/member_C/f6_experiments/runs
```

## 验证指标

| scheme | word_dim | char_dim | macro_f1 | accuracy | best_epoch | epochs_run | train_seconds | society_to_finance | society_to_entertainment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 5000 | 5000 | 0.78950 | 0.78737 | 9 | 19 | 29.44 | 88 | 75 |
| F6-A | 3000 | 7000 | 0.78768 | 0.78518 | 19 | 29 | 53.66 | 89 | 75 |
| F6-B | 7000 | 3000 | 0.79114 | 0.78926 | 11 | 21 | 35.99 | 96 | 76 |

模型参数量均为 640,454。训练耗时不包含特征拟合时间，基准与本次实验的耗时来自不同运行，硬件负载等条件不一定一致，不能严格比较；完整记录见 `summary.csv`。

## 相对于 seed=42 基准的差值

F1 与 Accuracy 的数值为绝对差，混淆方向为误分类条数差；正数表示增加。

| scheme | macro_f1 | accuracy | social_f1 | finance_f1 | entertainment_f1 | social_to_finance | social_to_entertainment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F6-A | -0.00182 | -0.00220 | -0.00139 | +0.00965 | -0.02024 | +1 | +0 |
| F6-B | +0.00164 | +0.00188 | +0.00651 | -0.00038 | -0.00299 | +8 | +1 |

## 六类别 F1

| category | Baseline | F6-A | F6-B |
| --- | --- | --- | --- |
| 体育 | 0.91161 | 0.91064 | 0.91569 |
| 国际 | 0.89177 | 0.89655 | 0.89119 |
| 教育 | 0.83821 | 0.83547 | 0.84139 |
| 文娱 | 0.77456 | 0.75432 | 0.77156 |
| 社会 | 0.60685 | 0.60545 | 0.61336 |
| 财经 | 0.71403 | 0.72367 | 0.71365 |

## 固定样本组的非零特征覆盖

持续错误是原 F2 与原 F3-A 各五种子全部错误；stable_f3 为 F3-A 五种子全对；stable_both 为两组均五种子全对。分组固定不随新实验结果改变。

| scheme | cohort | n | word_nnz_mean | char_nnz_mean | total_nnz_mean | both_zero | word_only | char_only |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | persistent_both | 548 | 5.088 | 5.263 | 10.350 | 9 | 26 | 0 |
| Baseline | stable_f3 | 2435 | 6.046 | 8.320 | 14.365 | 2 | 19 | 8 |
| Baseline | stable_both | 2342 | 6.083 | 8.455 | 14.538 | 2 | 16 | 7 |
| F6-A | persistent_both | 548 | 4.093 | 6.026 | 10.119 | 12 | 17 | 6 |
| F6-A | stable_f3 | 2435 | 5.154 | 9.342 | 14.496 | 5 | 9 | 16 |
| F6-A | stable_both | 2342 | 5.196 | 9.485 | 14.681 | 5 | 7 | 13 |
| F6-B | persistent_both | 548 | 5.701 | 4.066 | 9.766 | 8 | 41 | 0 |
| F6-B | stable_f3 | 2435 | 6.667 | 6.742 | 13.409 | 1 | 43 | 3 |
| F6-B | stable_both | 2342 | 6.701 | 6.856 | 13.556 | 1 | 38 | 3 |

相对于基准，固定持续错误样本组的分支非零均值及全零样本数差值：

| scheme | word_delta | char_delta | total_delta | zero_delta |
| --- | --- | --- | --- | --- |
| F6-A | -0.995 | +0.763 | -0.232 | +3 |
| F6-B | +0.613 | -1.197 | -0.584 | -1 |

社会、财经、文娱逐类覆盖见 `coverage_comparison.csv`；逐标题数据见 `coverage_per_title.csv`。精确率、召回率与混淆矩阵见 `per_class.csv`、`confusion_matrices.csv`。选中特征覆盖变化是描述性结果，无法单独确定性能变化原因。

本脚本没有拟合任何特征处理器，没有训练模型，也没有访问测试集。训练时仅用训练集拟合的证据是训练实现、记录的训练文件哈希和处理器 fit_rows；本脚本未再次独立拟合。

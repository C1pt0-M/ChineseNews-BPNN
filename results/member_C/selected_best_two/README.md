# 各实验配置的前两次结果

按各配置单次 Macro-F1 从高到低排序；分数相同按种子从小到大。每个配置最多取两次；图和历史指标从原目录复制，复制后逐文件核对 SHA-256。未训练模型，未改动原实验。
图中所称“E 系列 Input Dropout”在项目中的实际目录是 `results/member_C/c3_multi/`（0、0.05、0.10、0.20）；它不是 E1/E2 的线性或混合分支实验。
F2、F3、F6、F7、F8 依据验证集 Macro-F1；`final_test` 依据已有最终评估的测试 Macro-F1，仅为事后展示。该测试集以前已被查看，不能用此排序重新选择模型或声称盲测。
F6-A 仅 seed 42 一次，无法提供第二次。最高单次结果不等于五种子均值，也不代表模型稳健性；各组完整种子结果与均值分别见 `all_runs.csv` 和 `group_summary.csv`。

## 每阶段整体前两次

| 阶段 | 名次 | 配置 | seed | 数据集 | Macro-F1 | Accuracy |
|---|---:|---|---:|---|---:|---:|
| E_input_dropout | 1 | dropout_010 | 44 | validation | 0.76142 | 0.75911 |
| E_input_dropout | 2 | dropout_0 | 46 | validation | 0.76133 | 0.75911 |
| F2 | 1 | F_word_char | 42 | validation | 0.78467 | 0.78329 |
| F2 | 2 | F_word_char | 45 | validation | 0.78446 | 0.78204 |
| F3 | 1 | F3_A_formal | 42 | validation | 0.78950 | 0.78737 |
| F3 | 2 | F3_A_formal | 43 | validation | 0.78809 | 0.78580 |
| F6 | 1 | F6_B_7000_3000 | 45 | validation | 0.79213 | 0.78989 |
| F6 | 2 | F6_B_7000_3000 | 44 | validation | 0.79145 | 0.78926 |
| F7 | 1 | F7-0 | 45 | validation | 0.79213 | 0.78989 |
| F7 | 2 | F7-2 | 44 | validation | 0.79206 | 0.78989 |
| F8 | 1 | F8-2 | 43 | validation | 0.84008 | 0.84045 |
| F8 | 2 | F8-2 | 46 | validation | 0.83702 | 0.83700 |
| final_test | 1 | F8-2 | 43 | previously_viewed_test | 0.82135 | 0.82223 |
| final_test | 2 | F8-2 | 46 | previously_viewed_test | 0.81999 | 0.82047 |

## 每个配置的前两次

| 阶段 | 配置 | 名次 | seed | Macro-F1 | Accuracy | 文件夹 |
|---|---|---:|---:|---:|---:|---|
| E_input_dropout | dropout_0 | 1 | 46 | 0.76133 | 0.75911 | [E_input_dropout/dropout_0/rank1_seed46](E_input_dropout/dropout_0/rank1_seed46/) |
| E_input_dropout | dropout_0 | 2 | 45 | 0.76066 | 0.75722 | [E_input_dropout/dropout_0/rank2_seed45](E_input_dropout/dropout_0/rank2_seed45/) |
| E_input_dropout | dropout_005 | 1 | 46 | 0.76100 | 0.75911 | [E_input_dropout/dropout_005/rank1_seed46](E_input_dropout/dropout_005/rank1_seed46/) |
| E_input_dropout | dropout_005 | 2 | 44 | 0.76050 | 0.75785 | [E_input_dropout/dropout_005/rank2_seed44](E_input_dropout/dropout_005/rank2_seed44/) |
| E_input_dropout | dropout_010 | 1 | 44 | 0.76142 | 0.75911 | [E_input_dropout/dropout_010/rank1_seed44](E_input_dropout/dropout_010/rank1_seed44/) |
| E_input_dropout | dropout_010 | 2 | 46 | 0.76054 | 0.75879 | [E_input_dropout/dropout_010/rank2_seed46](E_input_dropout/dropout_010/rank2_seed46/) |
| E_input_dropout | dropout_020 | 1 | 42 | 0.76035 | 0.75879 | [E_input_dropout/dropout_020/rank1_seed42](E_input_dropout/dropout_020/rank1_seed42/) |
| E_input_dropout | dropout_020 | 2 | 45 | 0.75912 | 0.75660 | [E_input_dropout/dropout_020/rank2_seed45](E_input_dropout/dropout_020/rank2_seed45/) |
| F2 | A_raw | 1 | 42 | 0.76698 | 0.76759 | [F2/A_raw/rank1_seed42](F2/A_raw/rank1_seed42/) |
| F2 | A_raw | 2 | 46 | 0.76625 | 0.76476 | [F2/A_raw/rank2_seed46](F2/A_raw/rank2_seed46/) |
| F2 | B_chi2_10000 | 1 | 42 | 0.76612 | 0.76539 | [F2/B_chi2_10000/rank1_seed42](F2/B_chi2_10000/rank1_seed42/) |
| F2 | B_chi2_10000 | 2 | 45 | 0.76548 | 0.76476 | [F2/B_chi2_10000/rank2_seed45](F2/B_chi2_10000/rank2_seed45/) |
| F2 | D3 | 1 | 44 | 0.76388 | 0.76162 | [F2/D3/rank1_seed44](F2/D3/rank1_seed44/) |
| F2 | D3 | 2 | 42 | 0.76287 | 0.76131 | [F2/D3/rank2_seed42](F2/D3/rank2_seed42/) |
| F2 | F_char | 1 | 42 | 0.77804 | 0.77387 | [F2/F_char/rank1_seed42](F2/F_char/rank1_seed42/) |
| F2 | F_char | 2 | 45 | 0.77734 | 0.77324 | [F2/F_char/rank2_seed45](F2/F_char/rank2_seed45/) |
| F2 | F_word_char | 1 | 42 | 0.78467 | 0.78329 | [F2/F_word_char/rank1_seed42](F2/F_word_char/rank1_seed42/) |
| F2 | F_word_char | 2 | 45 | 0.78446 | 0.78204 | [F2/F_word_char/rank2_seed45](F2/F_word_char/rank2_seed45/) |
| F3 | F3_A_formal | 1 | 42 | 0.78950 | 0.78737 | [F3/F3_A_formal/rank1_seed42](F3/F3_A_formal/rank1_seed42/) |
| F3 | F3_A_formal | 2 | 43 | 0.78809 | 0.78580 | [F3/F3_A_formal/rank2_seed43](F3/F3_A_formal/rank2_seed43/) |
| F6 | F6_A_3000_7000 | 1 | 42 | 0.78768 | 0.78518 | [F6/F6_A_3000_7000/rank1_seed42](F6/F6_A_3000_7000/rank1_seed42/) |
| F6 | F6_B_7000_3000 | 1 | 45 | 0.79213 | 0.78989 | [F6/F6_B_7000_3000/rank1_seed45](F6/F6_B_7000_3000/rank1_seed45/) |
| F6 | F6_B_7000_3000 | 2 | 44 | 0.79145 | 0.78926 | [F6/F6_B_7000_3000/rank2_seed44](F6/F6_B_7000_3000/rank2_seed44/) |
| F6 | F6_baseline_5000_5000 | 1 | 42 | 0.78950 | 0.78737 | [F6/F6_baseline_5000_5000/rank1_seed42](F6/F6_baseline_5000_5000/rank1_seed42/) |
| F6 | F6_baseline_5000_5000 | 2 | 43 | 0.78809 | 0.78580 | [F6/F6_baseline_5000_5000/rank2_seed43](F6/F6_baseline_5000_5000/rank2_seed43/) |
| F7 | F7-0 | 1 | 45 | 0.79213 | 0.78989 | [F7/F7-0/rank1_seed45](F7/F7-0/rank1_seed45/) |
| F7 | F7-0 | 2 | 44 | 0.79145 | 0.78926 | [F7/F7-0/rank2_seed44](F7/F7-0/rank2_seed44/) |
| F7 | F7-1 | 1 | 45 | 0.79048 | 0.78769 | [F7/F7-1/rank1_seed45](F7/F7-1/rank1_seed45/) |
| F7 | F7-1 | 2 | 44 | 0.78911 | 0.78769 | [F7/F7-1/rank2_seed44](F7/F7-1/rank2_seed44/) |
| F7 | F7-2 | 1 | 44 | 0.79206 | 0.78989 | [F7/F7-2/rank1_seed44](F7/F7-2/rank1_seed44/) |
| F7 | F7-2 | 2 | 45 | 0.79189 | 0.78989 | [F7/F7-2/rank2_seed45](F7/F7-2/rank2_seed45/) |
| F7 | F7-3 | 1 | 45 | 0.78816 | 0.78643 | [F7/F7-3/rank1_seed45](F7/F7-3/rank1_seed45/) |
| F7 | F7-3 | 2 | 42 | 0.78799 | 0.78706 | [F7/F7-3/rank2_seed42](F7/F7-3/rank2_seed42/) |
| F8 | F8-0 | 1 | 45 | 0.79213 | 0.78989 | [F8/F8-0/rank1_seed45](F8/F8-0/rank1_seed45/) |
| F8 | F8-0 | 2 | 44 | 0.79145 | 0.78926 | [F8/F8-0/rank2_seed44](F8/F8-0/rank2_seed44/) |
| F8 | F8-1 | 1 | 42 | 0.82785 | 0.82820 | [F8/F8-1/rank1_seed42](F8/F8-1/rank1_seed42/) |
| F8 | F8-1 | 2 | 44 | 0.82575 | 0.82632 | [F8/F8-1/rank2_seed44](F8/F8-1/rank2_seed44/) |
| F8 | F8-2 | 1 | 43 | 0.84008 | 0.84045 | [F8/F8-2/rank1_seed43](F8/F8-2/rank1_seed43/) |
| F8 | F8-2 | 2 | 46 | 0.83702 | 0.83700 | [F8/F8-2/rank2_seed46](F8/F8-2/rank2_seed46/) |
| final_test | F8-0 | 1 | 44 | 0.77204 | 0.77062 | [final_test/F8-0/rank1_seed44](final_test/F8-0/rank1_seed44/) |
| final_test | F8-0 | 2 | 42 | 0.77181 | 0.77018 | [final_test/F8-0/rank2_seed42](final_test/F8-0/rank2_seed42/) |
| final_test | F8-1 | 1 | 44 | 0.81223 | 0.81341 | [final_test/F8-1/rank1_seed44](final_test/F8-1/rank1_seed44/) |
| final_test | F8-1 | 2 | 43 | 0.81207 | 0.81341 | [final_test/F8-1/rank2_seed43](final_test/F8-1/rank2_seed43/) |
| final_test | F8-2 | 1 | 43 | 0.82135 | 0.82223 | [final_test/F8-2/rank1_seed43](final_test/F8-2/rank1_seed43/) |
| final_test | F8-2 | 2 | 46 | 0.81999 | 0.82047 | [final_test/F8-2/rank2_seed46](final_test/F8-2/rank2_seed46/) |

`selected_metrics.json` 是该次的指标摘要；`metrics.json` 是原训练记录（对最终评估而言，它记录的是验证集而非测试集）。`training_curve` 为训练/验证曲线；`confusion_val` 是验证混淆矩阵；最终评估的测试混淆图使用 `confusion_test.png`。
各类别 F1 在 `per_class_val.csv` 或 `per_class_test.csv`，训练历史在 `history.csv`。F8-0、F7-0 等复用既有运行，因此不同阶段可能引用同一 checkpoint 和图。原始路径及哈希见 `manifest.json`。

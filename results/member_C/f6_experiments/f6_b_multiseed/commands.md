# F6-B 实验与复核命令

工作目录为项目根目录。训练实际使用的 Python 为 `D:\WorkSpace\PATH\Miniconda\envs\vision\python.exe`；训练前已核对正式训练文件和原验证文件的 SHA-256。

```powershell
$env:PYTHONUTF8='1'
D:\WorkSpace\PATH\Miniconda\envs\vision\python.exe train_bp.py --exp F6_B --seeds 43,44,45,46 --model-type bp --hidden 64 --lr 0.0003 --dropout 0.2 --input-dropout 0 --train-file results/member_C/f3_analysis/f3_a_dataset_formal/train.csv --train-sha256 7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880 --output-root results/member_C/f6_experiments/f6_b_multiseed/runs
```

seed=42 使用已保存的 `results/member_C/f6_experiments/runs/F6_B_s42`，并未重新训练。再次执行上述命令会因运行目录已存在而拒绝覆盖。

汇总脚本不训练、不拟合特征或读取测试集。再次复核时，请为 `--output-dir` 指定项目内新的空目录；脚本拒绝覆盖已有汇总文件。

```powershell
$env:PYTHONUTF8='1'
D:\WorkSpace\PATH\Miniconda\envs\vision\python.exe results/member_C/f6_experiments/f6_b_multiseed/analyze.py --output-dir results/member_C/f6_experiments/f6_b_multiseed/another_audit
```

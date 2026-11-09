# F.8 复核与复现

从项目根目录运行。项目原有的 `vision` 环境提供 PyTorch、NumPy、SciPy、scikit-learn、pandas、jieba 等依赖；完整实测版本和审计时源码哈希见 `environment.json`，语义编码配置和模型文件哈希见 `cache/manifest.json`。源码哈希在训练后审计时记录，而非训练时自动存证。

额外依赖安装至项目内独立目录，不修改系统级 Python 环境：

```powershell
$env:PYTHONUTF8='1'
D:\WorkSpace\PATH\Miniconda\envs\vision\python.exe -m pip install --target results/member_C/f8_experiments/vendor --no-deps --no-cache-dir -r results/member_C/f8_experiments/requirements_f8.txt
D:\WorkSpace\PATH\Miniconda\envs\vision\python.exe results/member_C/f8_experiments/download_model.py
```

下载脚本固定 BAAI/bge-small-zh-v1.5 的官方 revision `7999e1d3359715c523056ef9478215996d62a620`。模型文件及语义缓存已存在时，不必再次下载或编码。`pipeline.py encode` 和训练入口均拒绝覆盖同名缓存或运行目录；要复跑审计，请指定项目内新的空目录：

```powershell
D:\WorkSpace\PATH\Miniconda\envs\vision\python.exe results/member_C/f8_experiments/analyze_f8.py --output-dir results/member_C/f8_experiments/another_audit
```

实际种子 42 与 43–46 的训练命令记录在 `summary.md`。模型下载、输入缓存、处理器和 BP checkpoint 均在 F.8 独立目录或通过明确记录的历史 F7-0 路径引用；不访问测试集。

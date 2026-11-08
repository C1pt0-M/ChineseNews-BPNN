"""Recompute and audit F.7 validation results, then report paired factorial effects."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import model_config as C
from models.bp import build_model
from models.trainer import predict

SEEDS = (42, 43, 44, 45, 46)
GROUPS = {'F7-0': (64, .0003), 'F7-1': (256, .0003),
          'F7-2': (64, .001), 'F7-3': (256, .001)}
TRAIN = ROOT / 'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv'
VAL = ROOT / 'dataset/splits/val.csv'
TRAIN_HASH = '7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
VAL_HASH = '9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0'
OUTPUTS = ('per_seed.csv', 'summary.csv', 'per_class.csv', 'per_class_summary.csv', 'loss_diagnostics.csv',
           'confusion_matrices.csv', 'paired_effects.csv', 'effect_summary.csv', 'audit.json', 'summary.md')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def table(df, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join('---' for _ in cols) + ' |']
    lines.extend('| ' + ' | '.join(str(row[c]) for c in cols) + ' |' for _, row in df.iterrows())
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=HERE)
    out = ap.parse_args().output_dir.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (out / name).exists():
            raise FileExistsError(f'Refusing to overwrite {out / name}')
    assert digest(TRAIN) == TRAIN_HASH and digest(VAL) == VAL_HASH
    assert json.loads((TRAIN.parent / 'manifest.json').read_text(encoding='utf8'))['train_sha256'] == TRAIN_HASH
    val = pd.read_csv(VAL, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    assert len(val) == 3184 and val.id.is_unique
    y = val.category.map(C.LABEL_TO_INDEX).to_numpy(dtype=int)
    baseline = pd.read_csv(ROOT / 'results/member_C/f6_experiments/f6_b_multiseed/per_seed.csv', dtype=str)
    baseline = baseline[baseline.group == 'F6-B'].set_index('seed')
    assert set(baseline.index) == {str(s) for s in SEEDS}
    torch.set_num_threads(4)
    runs, classes, confusions, hashes = [], [], [], {}
    expected_params, reference_feat, vocab_ref, runtime_ref = None, None, {}, None
    for group, (hidden, lr) in GROUPS.items():
        for seed in SEEDS:
            folder = ROOT / baseline.loc[str(seed), 'run_path'] if group == 'F7-0' else HERE / 'runs' / f"{group.replace('-', '_')}_s{seed}"
            required = ['metrics.json', 'model.pt', 'feature_builder.pkl', 'history.csv', 'val_predictions.csv', 'training_curve.png', 'confusion_val.png']
            assert all((folder / f).is_file() for f in required), f'Missing artifacts: {folder}'
            m = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
            feat, arch, training = m['feature'], m['model'], m['training']
            p = training['params']
            runtime = tuple(m['provenance'][k] for k in ('python', 'torch', 'torch_threads'))
            if runtime_ref is None:
                runtime_ref = runtime
            assert runtime == runtime_ref
            assert m['seed'] == seed and m['provenance']['split_sha256'] == {'train': TRAIN_HASH, 'val': VAL_HASH}
            assert feat['scheme'] == 'F_word_char' and feat['input_source'] == 'title' and feat['sparse']
            assert feat['fit_rows'] == 7531 and feat['input_dim'] == 10000
            assert feat['word']['selected_features'] == feat['word']['chi2_k_requested'] == 7000
            assert feat['char']['selected_features'] == feat['char']['chi2_k_requested'] == 3000
            assert arch.get('model_type', 'bp') == 'bp' and arch['input_dim'] == 10000 and arch['output_dim'] == 6
            assert arch['hidden_dim'] == hidden and arch['dropout'] == .2 and arch.get('input_dropout', 0) == 0
            assert arch['activation'] == 'ReLU' and arch['n_parameters'] == 10007 * hidden + 6
            assert p['lr'] == lr and p['patience'] == 10 and p['min_delta'] == .0001
            constant_params = {k: v for k, v in p.items() if k != 'lr'}
            fp = (feat['word']['tfidf_params'], feat['char']['tfidf_params'], feat['normalization'])
            if expected_params is None:
                expected_params, reference_feat = constant_params, fp
            assert constant_params == expected_params and fp == reference_feat
            with (folder / 'feature_builder.pkl').open('rb') as f:
                builder = pickle.load(f)
            assert builder.fit_rows_ == 7531 and builder.word_dim_ == 7000 and builder.char_dim_ == 3000
            for name, vec, selector in [('word', builder.word.vectorizer, builder.word.selector),
                                        ('char', builder.char_vectorizer, builder.char_selector)]:
                if name not in vocab_ref:
                    vocab_ref[name] = (dict(vec.vocabulary_), vec.idf_.copy(), selector.get_support().copy())
                vocabulary, idf, support = vocab_ref[name]
                assert vec.vocabulary_ == vocabulary
                np.testing.assert_allclose(vec.idf_, idf, rtol=0, atol=1e-14)
                np.testing.assert_array_equal(selector.get_support(), support)
            X = builder.transform(val.title.tolist())
            assert sparse.isspmatrix_csr(X) and X.shape == (3184, 10000)
            model = build_model(arch)
            model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
            assert model.count_parameters() == arch['n_parameters']
            pred = predict(model, X)
            saved = pd.read_csv(folder / 'val_predictions.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
            for col in ('id', 'title', 'category'):
                assert saved[col].tolist() == val[col].tolist()
            assert np.array_equal(pred, saved.predicted_category.map(C.LABEL_TO_INDEX).to_numpy(dtype=int))
            f1 = f1_score(y, pred, labels=np.arange(6), average='macro', zero_division=0)
            acc = accuracy_score(y, pred)
            cm = confusion_matrix(y, pred, labels=np.arange(6))
            assert abs(f1 - m['val']['f1_macro']) < 1e-12 and abs(acc - m['val']['accuracy']) < 1e-12
            assert cm.tolist() == m['val']['confusion_matrix']
            history = pd.read_csv(folder / 'history.csv')
            best_score, best_epoch = -1., 0
            for epoch, score in zip(history.epoch, history.val_f1_macro):
                if score > best_score + p['min_delta']:
                    best_score, best_epoch = score, int(epoch)
            assert best_epoch == training['best_epoch'] and abs(best_score - f1) < 1e-12
            assert len(history) == training['epochs_run']
            stopped = len(history) - best_epoch == p['patience']
            assert stopped or len(history) == p['max_epochs']
            best_history = history.loc[history.epoch == best_epoch].iloc[0]
            last_history = history.iloc[-1]
            prec, recall, fs, support = precision_recall_fscore_support(y, pred, labels=np.arange(6), zero_division=0)
            for i, cat in enumerate(C.LABELS):
                for metric, value in [('precision', prec[i]), ('recall', recall[i]), ('f1', fs[i])]:
                    assert abs(m['val']['per_class'][cat][metric] - value) < 1e-12
                classes.append(dict(group=group, seed=seed, category=cat, precision=prec[i], recall=recall[i], f1=fs[i], support=int(support[i])))
                for j, target in enumerate(C.LABELS):
                    confusions.append(dict(group=group, seed=seed, true_category=cat, predicted_category=target, count=int(cm[i,j])))
            runs.append(dict(group=group, seed=seed, hidden=hidden, lr=lr, macro_f1=f1, accuracy=acc,
                             parameters=arch['n_parameters'], best_epoch=best_epoch, epochs_run=len(history),
                             early_stopped=stopped, train_seconds=training['train_time'], feature_seconds=training['feature_time'],
                             best_train_loss=best_history.train_eval_loss, best_val_loss=best_history.val_loss,
                             last_train_loss=last_history.train_eval_loss, last_val_loss=last_history.val_loss,
                             society_to_finance=int(cm[4,0]), society_to_entertainment=int(cm[4,2]), path=str(folder.relative_to(ROOT))))
            hashes[f'{group} seed={seed}'] = {f: digest(folder / f) for f in required}
            print(f'{group} seed={seed}: strict checkpoint and predictions verified, macro-F1={f1:.8f}', flush=True)
    runs, classes, confusions = pd.DataFrame(runs), pd.DataFrame(classes), pd.DataFrame(confusions)
    summary = runs.groupby('group', sort=False).agg(n=('seed','size'), hidden=('hidden','first'), lr=('lr','first'),
        f1_mean=('macro_f1','mean'), f1_std=('macro_f1','std'), accuracy_mean=('accuracy','mean'), accuracy_std=('accuracy','std'),
        parameters=('parameters','first'), train_seconds_mean=('train_seconds','mean'), best_epoch_mean=('best_epoch','mean'),
        society_to_finance_mean=('society_to_finance','mean'), society_to_entertainment_mean=('society_to_entertainment','mean')).reset_index()
    class_summary = classes.groupby(['group','category'], sort=False).agg(f1_mean=('f1','mean'), f1_std=('f1','std'),
        precision_mean=('precision','mean'), recall_mean=('recall','mean')).reset_index()
    loss_summary = runs.groupby('group', sort=False)[['best_train_loss','best_val_loss','last_train_loss','last_val_loss']].mean().reset_index()
    pivot = runs.pivot(index='seed', columns='group', values='macro_f1')
    effects = pd.DataFrame(index=pivot.index)
    effects['hidden_at_lr0003'] = pivot['F7-1'] - pivot['F7-0']
    effects['hidden_at_lr001'] = pivot['F7-3'] - pivot['F7-2']
    effects['lr_at_hidden64'] = pivot['F7-2'] - pivot['F7-0']
    effects['lr_at_hidden256'] = pivot['F7-3'] - pivot['F7-1']
    effects['interaction'] = effects['hidden_at_lr001'] - effects['hidden_at_lr0003']
    effects['hidden_marginal'] = (effects['hidden_at_lr0003'] + effects['hidden_at_lr001']) / 2
    effects['lr_marginal'] = (effects['lr_at_hidden64'] + effects['lr_at_hidden256']) / 2
    effects['F7_3_minus_F7_0'] = pivot['F7-3'] - pivot['F7-0']
    effect_summary = pd.DataFrame([{'effect': c, 'mean': effects[c].mean(), 'std_sample': effects[c].std(ddof=1),
                                    'positive': int((effects[c] > 0).sum())} for c in effects.columns])
    acc_pivot = runs.pivot(index='seed', columns='group', values='accuracy')
    effects['accuracy_hidden_at_lr0003'] = acc_pivot['F7-1'] - acc_pivot['F7-0']
    effects['accuracy_hidden_at_lr001'] = acc_pivot['F7-3'] - acc_pivot['F7-2']
    effects['accuracy_lr_at_hidden64'] = acc_pivot['F7-2'] - acc_pivot['F7-0']
    effects['accuracy_lr_at_hidden256'] = acc_pivot['F7-3'] - acc_pivot['F7-1']
    effects['accuracy_interaction'] = effects['accuracy_hidden_at_lr001'] - effects['accuracy_hidden_at_lr0003']
    effects['accuracy_F7_3_minus_F7_0'] = acc_pivot['F7-3'] - acc_pivot['F7-0']
    effects = effects.reset_index()
    outputs = [('per_seed.csv', runs), ('summary.csv', summary), ('per_class.csv', classes),
               ('per_class_summary.csv', class_summary), ('loss_diagnostics.csv', loss_summary), ('confusion_matrices.csv', confusions),
               ('paired_effects.csv', effects), ('effect_summary.csv', effect_summary)]
    for name, df in outputs:
        df.to_csv(out / name, index=False, encoding='utf-8-sig')
    audit = {'train_sha256': TRAIN_HASH, 'val_sha256': VAL_HASH, 'artifact_sha256': hashes,
             'checkpoint_strict_load_and_predictions_match': True, 'metrics_and_confusions_recomputed': True,
             'early_stopping_recomputed': True, 'vocabulary_idf_selector_same': True,
             'python_torch_threads': runtime_ref,
             'feature_fit_evidence': 'training implementation, frozen file hashes and fit_rows; audit does not independently refit',
             'audit_fit_train_or_test_evaluation': False}
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    disp = summary.copy()
    for c in ('f1_mean','f1_std','accuracy_mean','accuracy_std','train_seconds_mean','best_epoch_mean',
              'society_to_finance_mean','society_to_entertainment_mean'):
        disp[c] = disp[c].map(lambda v: f'{v:.5f}')
    per = runs[['group','seed','macro_f1','accuracy','best_epoch','epochs_run']].copy()
    for c in ('macro_f1','accuracy'): per[c] = per[c].map(lambda v: f'{v:.5f}')
    cdisp = class_summary.pivot(index='category', columns='group', values='f1_mean').reset_index()
    for g in GROUPS: cdisp[g] = cdisp[g].map(lambda v: f'{v:.5f}')
    edisp = effect_summary.copy()
    for c in ('mean','std_sample'): edisp[c] = edisp[c].map(lambda v: f'{v:+.5f}' if c == 'mean' else f'{v:.5f}')
    ldisp = loss_summary.copy()
    for c in ldisp.columns[1:]: ldisp[c] = ldisp[c].map(lambda v: f'{v:.5f}')
    common = 'Adam、weight_decay=0.0001、batch=64、max_epochs=100、patience=10、min_delta=0.0001、grad_clip=5.0；ReLU，隐藏 dropout=0.2，input dropout=0。'
    lines = ['# F.7 BP 参数 2×2 对照', '',
        '四组均使用 F3-A 正式训练集 7,531 条、原验证集 3,184 条和词 7,000 + 字 3,000 的 F6-B 特征。F7-0 复用 F6-B 五种子；其他三组由 F.7 独立入口调用原 run_single 和 trainer。',
        'BP 为单隐层：输入 → 线性层（宽度 hidden）→ ReLU → Dropout → 六类输出线性层；层数不变。' + common, '',
        '## 已有实验口径', '',
        '| 实验 | 训练样本 | 输入 | 隐藏宽度 | 学习率 |', '|---|---:|---|---:|---:|',
        '| B0（A_raw） | 6549 | 词级原始 TF-IDF，10864 维 | 256 | 0.001 |',
        '| F3-A | 7531 | 词 5000 + 字 5000 | 64 | 0.0003 |',
        '| F6-B / F7-0 | 7531 | 词 7000 + 字 3000 | 64 | 0.0003 |',
        'B0 与 F3-A/F6-B 同时改变了训练数据、特征及部分参数，不能用其差值单独归因。B0 信息来自 `results/member_A_baseline_recheck/A_raw_s42/metrics.json`；F3-A/F6-B 分别来自既有正式运行记录。', '',
        '## 四组汇总（样本标准差，n=5）', '', table(disp, list(disp.columns)), '',
        '训练耗时不含特征处理，跨运行的机器负载可能不同；曲线及 6×6 混淆图保存在每个运行目录。', '',
        '## 五种子完整指标', '', table(per, list(per.columns)), '',
        '## 六类别平均 F1', '', table(cdisp, ['category'] + list(GROUPS)), '',
        '## 同种子因素对比（Macro-F1）', '',
        '`hidden_at_lr0003` = F7-1 − F7-0；`hidden_at_lr001` = F7-3 − F7-2；`lr_at_hidden64` = F7-2 − F7-0；`lr_at_hidden256` = F7-3 − F7-1。',
        '交互 = (F7-3 − F7-2) − (F7-1 − F7-0)；边际效应是对应两个条件差的平均。正值表示前者更高。', '',
        table(edisp, list(edisp.columns)), '',
        '逐种子配对差、类别 Precision/Recall/F1 和混淆矩阵分别见 `paired_effects.csv`、`per_class.csv`、`confusion_matrices.csv`。', '',
        '## 训练与验证损失（五种子均值）', '',
        '最佳 epoch 指验证 Macro-F1 的最佳 epoch；末轮为早停时的最后 epoch。', '',
        table(ldisp, list(ldisp.columns)), '',
        '## 复现与证据边界', '',
        '```powershell', "$env:PYTHONUTF8='1'", 'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/f7_experiments/run_f7.py', '```',
        '入口支持 `--groups` 和 `--seeds` 用于缺失运行；任何已存在的目标运行目录都会在训练前拒绝覆盖。审计脚本 `analyze_f7.py --output-dir <项目内新目录>` 可独立复核。',
        '20 次运行已核对文件哈希、分支维数、训练配置、词表/IDF/选择器、严格加载模型、逐条验证预测、指标和早停。仅训练拟合的证据为训练代码、文件哈希和 fit_rows；审计没有重新拟合。',
        'F6-B 原先据 seed=42 选出，四组继续共享同一验证集并用于早停和比较；这些结果不是独立测试，五个种子不等于五个独立数据集，差值和交互均为描述性估计。CPU 确定性算法未强制开启，不保证跨环境逐位复现。未使用测试集进行选择或评估。', '',
        '## F.8 候选问题', '',
        '在确定是否采用更大隐藏层或更高学习率之前，可检查主要混淆类别的配对预测转变与训练、验证损失曲线，评估容量、收敛与过拟合迹象；另需在不更改既有验证标签的条件下规划独立泛化验证。此处仅提出问题，不启动新实验。', '']
    ev = effect_summary.set_index('effect')['mean']
    loss = loss_summary.set_index('group')
    lines += ['## 基于本轮结果的技术观察', '',
        f"- 隐藏宽度从 64 增至 256：学习率 0.0003 时，Macro-F1 的平均配对差为 {ev['hidden_at_lr0003']:+.5f}；学习率 0.001 时为 {ev['hidden_at_lr001']:+.5f}。这只描述当前固定数据与五个种子。",
        f"- 学习率从 0.0003 增至 0.001：宽度 64 时平均差为 {ev['lr_at_hidden64']:+.5f}，宽度 256 时为 {ev['lr_at_hidden256']:+.5f}；交互差中之差为 {ev['interaction']:+.5f}。种子数少，不能仅凭数值断言交互显著。",
        f"- F7-3 在最佳 Macro-F1 轮的训练/验证损失均值为 {loss.loc['F7-3','best_train_loss']:.5f}/{loss.loc['F7-3','best_val_loss']:.5f}；到停止轮分别为 {loss.loc['F7-3','last_train_loss']:.5f}/{loss.loc['F7-3','last_val_loss']:.5f}。训练损失下降、验证损失上升是后续轮出现过拟合的迹象，不证明单一成因。早停按 Macro-F1 选模，而不是按最小验证损失。", '',
        'F.8 可具体考察固定 64 隐藏单元和 F6-B 特征时，改变单一正则化强度是否缩小训练与验证损失差距；应先固定比较水平、种子和选模方式，并在训练集内部做参数选择。另可按社会→财经与相反方向逐标题对齐预测转变，区分决策边界变化和类别 F1 的总体变化。以上仅为候选问题，不在本轮训练。', '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf8')
    print('F7 analysis completed', flush=True)

if __name__ == '__main__':
    main()

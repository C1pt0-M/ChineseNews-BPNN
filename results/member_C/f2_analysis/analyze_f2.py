"""Audit and summarize F.2 validation experiments; never read test data."""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import model_config as C
from models.bp import build_model
from models.data import load_splits
from models.features import FeatureBuilder, tokenize_corpus
from models.f_features import TitleFeatureBuilder
from models.trainer import predict

OUT = ROOT / 'results/member_C/f2_analysis'
GROUPS = {
    'D3': ('B_chi2_5000', ROOT / 'results/member_C/diagnostics_multi/hidden64_lr0003'),
    'A_raw': ('A_raw', ROOT / 'results/member_C/f1_raw_multi'),
    'B_chi2_10000': ('B_chi2_10000', ROOT / 'results/member_C/f2_formal_word10000'),
    'F_char': ('F_char', ROOT / 'results/member_C/f2_formal_char'),
    'F_word_char': ('F_word_char', ROOT / 'results/member_C/f2_formal_word_char'),
}
OLD_B42 = ROOT / 'results/member_C/f1_chi2_10000_hidden64/B_chi2_10000_s42'
SEEDS = (42, 43, 44, 45, 46)
PARAMS = {'lr': .0003, 'weight_decay': .0001, 'batch_size': 64, 'max_epochs': 100,
          'patience': 10, 'min_delta': .0001, 'grad_clip': 5.0}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def equal_features(actual, reference, name):
    def check_word(a, b):
        check(a.vectorizer.vocabulary_ == b.vectorizer.vocabulary_, f'{name}: word vocabulary')
        np.testing.assert_allclose(a.vectorizer.idf_, b.vectorizer.idf_, rtol=0, atol=1e-14)
        check((a.selector is None) == (b.selector is None), f'{name}: word selector')
        if a.selector is not None:
            np.testing.assert_array_equal(a.selector.get_support(), b.selector.get_support())
            np.testing.assert_allclose(a.selector.scores_, b.selector.scores_, rtol=1e-12, atol=1e-12, equal_nan=True)
    if isinstance(reference, TitleFeatureBuilder):
        check(actual.char_vectorizer.vocabulary_ == reference.char_vectorizer.vocabulary_, f'{name}: char vocabulary')
        np.testing.assert_allclose(actual.char_vectorizer.idf_, reference.char_vectorizer.idf_, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(actual.char_selector.get_support(), reference.char_selector.get_support())
        np.testing.assert_allclose(actual.char_selector.scores_, reference.char_selector.scores_, rtol=1e-12, atol=1e-12, equal_nan=True)
        if reference.word is not None:
            check_word(actual.word, reference.word)
    else:
        check_word(actual, reference)


def main():
    splits = load_splits(verify=True, splits=('train', 'val'))
    train, y_train, _ = splits['train']
    val, y_val, val_df = splits['val']
    val_tokens = tokenize_corpus(val)
    train_tokens = tokenize_corpus(train)
    hashes = {s: hashlib.sha256((C.SPLIT_DIR / (s + '.csv')).read_bytes()).hexdigest() for s in ('train', 'val')}
    labels = np.arange(len(C.LABELS))
    rows, classes = [], []
    for group, (experiment, root) in GROUPS.items():
        if group in ('F_char', 'F_word_char'):
            ref = TitleFeatureBuilder(group).fit(train, y_train)
            x_val = ref.transform(val)
        else:
            ref = FeatureBuilder('A' if group == 'A_raw' else 'B',
                                 k=None if group == 'A_raw' else (5000 if group == 'D3' else 10000)).fit(train_tokens, y_train)
            x_val = ref.transform(val_tokens)
        check(sp.issparse(x_val), f'{group}: expected sparse features')
        for seed in SEEDS:
            path = OLD_B42 if group == 'B_chi2_10000' and seed == 42 else root / f'{experiment}_s{seed}'
            for f in ('metrics.json', 'history.csv', 'model.pt', 'val_predictions.csv', 'feature_builder.pkl'):
                check((path / f).is_file(), f'{path}: missing {f}')
            d = json.loads((path / 'metrics.json').read_text(encoding='utf-8-sig'))
            arch, traininfo = d['model'], d['training']
            check(d['experiment_id'] == experiment and d['seed'] == seed, f'{path}: identity')
            check(d['provenance']['split_sha256'] == hashes, f'{path}: split hashes')
            source = d['provenance']['source_sha256']
            for module in ('data.py', 'features.py', 'metrics.py'):
                key = next((k for k in source if k.replace('\\', '/').endswith('/' + module)), None)
                current = hashlib.sha256((ROOT / 'models' / module).read_bytes()).hexdigest()
                check(key is not None and source[key] == current, f'{path}: {module} source fingerprint')
            if group in ('F_char', 'F_word_char'):
                key = next((k for k in source if k.replace('\\', '/').endswith('/f_features.py')), None)
                current = hashlib.sha256((ROOT / 'models/f_features.py').read_bytes()).hexdigest()
                check(key is not None and source[key] == current, f'{path}: F feature source fingerprint')
            check(d.get('test') is None, f'{path}: test metrics are present')
            check(traininfo['params'] == PARAMS and traininfo['selection_metric'] == 'f1_macro', f'{path}: training configuration')
            check(arch.get('model_type', 'bp') == 'bp' and arch['hidden_dim'] == 64 and arch['dropout'] == .2 and arch.get('input_dropout', 0) == 0 and arch['output_dim'] == 6, f'{path}: model configuration')
            check(arch['input_dim'] == x_val.shape[1] and d['feature']['fit_rows'] == len(train), f'{path}: feature dimension / fit rows')
            with (path / 'feature_builder.pkl').open('rb') as f:
                builder = pickle.load(f)
            equal_features(builder, ref, str(path))
            actual_x = builder.transform(val if getattr(builder, 'raw_text_input', False) else val_tokens)
            diff = actual_x - x_val
            check(diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-12, f'{path}: transform mismatch')
            model = build_model(arch)
            model.load_state_dict(torch.load(path / 'model.pt', map_location='cpu', weights_only=True), strict=True)
            check(model.count_parameters() == arch['n_parameters'], f'{path}: parameter count')
            preds = predict(model, actual_x)
            details = pd.read_csv(path / 'val_predictions.csv', encoding='utf-8-sig')
            check(len(details) == len(val_df) and details['id'].astype(str).tolist() == val_df['id'].astype(str).tolist(), f'{path}: validation ids')
            check(details['category'].tolist() == val_df['category'].tolist() and details['title'].tolist() == val_df['title'].tolist(), f'{path}: validation text/labels')
            stored_pred = details['predicted_category'].map(C.LABEL_TO_INDEX).to_numpy()
            check(np.array_equal(preds, stored_pred), f'{path}: checkpoint predictions')
            macro, acc = f1_score(y_val, preds, labels=labels, average='macro', zero_division=0), accuracy_score(y_val, preds)
            check(abs(macro-d['val']['f1_macro']) < 1e-12 and abs(acc-d['val']['accuracy']) < 1e-12, f'{path}: metrics')
            cm = confusion_matrix(y_val, preds, labels=labels)
            check(cm.tolist() == d['val']['confusion_matrix'], f'{path}: confusion matrix')
            cm_file = path / 'confusion_val.csv'
            check(cm_file.is_file(), f'{path}: missing confusion_val.csv')
            np.testing.assert_array_equal(pd.read_csv(cm_file, index_col=0, encoding='utf-8-sig').to_numpy(), cm)
            h = pd.read_csv(path / 'history.csv')
            check(h['epoch'].tolist() == list(range(1, len(h) + 1)), f'{path}: epoch sequence')
            best, best_epoch = -1.0, 0
            for epoch, score in zip(h['epoch'], h['val_f1_macro']):
                if score > best + PARAMS['min_delta']:
                    best, best_epoch = score, epoch
            check(len(h) == traininfo['epochs_run'] and best_epoch == traininfo['best_epoch'], f'{path}: best epoch')
            check(abs(best-macro) < 1e-12, f'{path}: restored best model')
            check(len(h) == PARAMS['max_epochs'] or len(h)-best_epoch == PARAMS['patience'], f'{path}: early stopping')
            p, r, f, support = precision_recall_fscore_support(y_val, preds, labels=labels, zero_division=0)
            for i, label in enumerate(C.LABELS):
                saved = d['val']['per_class'][label]
                check(max(abs(saved['precision']-p[i]), abs(saved['recall']-r[i]), abs(saved['f1']-f[i])) < 1e-12, f'{path}: per-class metrics')
                classes.append(dict(group=group, seed=seed, category=label, precision=p[i], recall=r[i], f1=f[i], support=int(support[i])))
            social = C.LABEL_TO_INDEX['\u793e\u4f1a']
            rows.append(dict(group=group, experiment=experiment, seed=seed, macro_f1=macro, accuracy=acc,
                             input_dim=arch['input_dim'], n_parameters=arch['n_parameters'], best_epoch=best_epoch,
                             epochs_run=len(h), society_support=int(cm[social].sum()),
                             society_to_finance=int(cm[social, C.LABEL_TO_INDEX['\u8d22\u7ecf']]),
                             society_to_entertainment=int(cm[social, C.LABEL_TO_INDEX['\u6587\u5a31']]),
                             run_path=str(path.relative_to(ROOT))))
            print('OK', group, seed, f'{macro:.6f}', flush=True)

    runs, perclass = pd.DataFrame(rows), pd.DataFrame(classes)
    runs.to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    perclass.to_csv(OUT / 'per_class.csv', index=False, encoding='utf-8-sig', float_format='%.12g')
    # The human-readable report is generated after all audits pass.
    lines = ['# F.2 验证集对照结果', '', '每组均使用 seed 42–46；Macro-F1 样本标准差使用 ddof=1。Accuracy 和分类别指标为五种子算术平均，混淆数为每个验证集的五种子平均。',
             '', '## 表 1：总体分类性能', '', '| 配置 | 42 | 43 | 44 | 45 | 46 | Macro-F1 均值 | 样本标准差 | Accuracy 均值 | 参数量 | 最佳 epoch 均值 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g in GROUPS:
        t = runs[runs.group == g].sort_values('seed')
        v = t.macro_f1.to_numpy()
        lines.append('| ' + ' | '.join([g] + [f'{x:.5f}' for x in v] + [f'{v.mean():.5f}', f'{v.std(ddof=1):.5f}', f'{t.accuracy.mean():.5f}', str(t.n_parameters.iloc[0]), f'{t.best_epoch.mean():.1f}']) + ' |')
    lines += ['', '## 表 2：各类别平均表现', '', '| 配置 | 类别 | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for g in GROUPS:
        for label in C.LABELS:
            t = perclass[(perclass.group == g) & (perclass.category == label)]
            lines.append(f'| {g} | {label} | {t.precision.mean():.5f} | {t.recall.mean():.5f} | {t.f1.mean():.5f} |')
    lines += ['', '社会类混淆：下表为每个 545 条社会类验证样本中，预测为对应类别的五种子平均数和比例。', '', '| 配置 | 社会→财经 均数 | 比例 | 社会→文娱 均数 | 比例 |', '|---|---:|---:|---:|---:|']
    for g in GROUPS:
        t = runs[runs.group == g]
        a, b, n = t.society_to_finance.mean(), t.society_to_entertainment.mean(), t.society_support.iloc[0]
        lines.append(f'| {g} | {a:.1f} | {a/n:.2%} | {b:.1f} | {b/n:.2%} |')
    lines += ['', '## 表 3：特征方案', '', '| 配置 | 输入维数 | 特征与筛选 | 归一化 | 参数量 |', '|---|---:|---|---|---:|',
              '| D3 | 5,000 | 词级 TF-IDF，chi2 选 5,000 | TF-IDF 行 L2 后筛选，无筛选后归一化 | 320,454 |',
              '| A_raw | 10,864 | 词级 TF-IDF，无筛选 | TF-IDF 行 L2 | 695,750 |',
              '| B_chi2_10000 | 10,000 | 词级 TF-IDF，chi2 选 10,000 | TF-IDF 行 L2 后筛选，无筛选后归一化 | 640,454 |',
              '| F_char | 10,000 | 字符 2–4 gram TF-IDF，chi2 选 10,000 | TF-IDF 行 L2 后筛选，无筛选后归一化 | 640,454 |',
              '| F_word_char | 10,000 | 词级与字符 2–4 gram 各 chi2 选 5,000 | 各 TF-IDF 行 L2、筛选后拼接再逐行 L2 | 640,454 |',
              '', '## 配置与完整性', '',
              '- BP hidden=64、lr=0.0003、hidden dropout=0.2、input dropout=0，Adam weight decay=0.0001、batch=64；验证 Macro-F1 早停 patience=10、min_delta=0.0001。',
              '- 训练/验证各 6,549/3,184 条；25 个运行的划分 SHA-256 与当前版本一致。校验预测文件的 ID、标题、真实标签和保存的模型预测，并独立重算 Macro-F1、Accuracy、分类别指标和混淆矩阵。',
              '- 从训练集重新拟合词表、IDF 和 chi2，与每个保存的处理器逐项比较；验证集仅 transform。此检查和代码审阅支持训练集拟合边界，不能单凭产物证明历史运行时的全部行为。',
              '- 根据 history 和 min_delta 重放最佳 epoch 选择并验证早停；已保存 checkpoint 对验证集的预测与最佳轮指标一致。训练入口只加载 train/val，未运行测试集评估。',
              '- 词级 10,000 维 seed 42 复用 `results/member_C/f1_chi2_10000_hidden64`；seed 43–46 使用新目录。旧运行与新运行的入口源文件哈希不同，已核对特征、模型、训练参数、数据哈希和预测，属于跨版本复用。各运行来源见 summary.csv。',
              '- 原始词级和 D3 对照使用既有五种子结果。各随机种子独立训练；这里的标准差反映随机种子波动，所有结果均为同一验证集上的指标。',
              '- D3 五种子根据现存预测复算的均值为 0.76256，与任务背景给出的 0.76258 相差 0.00002；表中使用可复核的实际结果，差异来源尚未确定。',
              '', '## 客观发现', '']
    d3 = runs[runs.group == 'D3'].macro_f1.mean()
    for g in ('B_chi2_10000', 'F_char', 'F_word_char'):
        score = runs[runs.group == g].macro_f1.mean()
        lines.append(f'- {g} 的平均 Macro-F1 为 {score:.5f}，相对 D3 为 {score-d3:+.5f}。')
    lines += ['', '原始结果和每运行的训练曲线、混淆矩阵保留在 summary.csv 所列运行目录。', '']
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()

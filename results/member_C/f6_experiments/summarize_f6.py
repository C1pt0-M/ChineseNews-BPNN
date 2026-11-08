"""Audit F6 saved runs and compare validation coverage; never fit or train."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import model_config as C
from models.bp import build_model
from models.features import tokenize_corpus
from models.trainer import predict

LABELS = list(C.LABELS)
TRAIN = ROOT / 'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv'
VAL = ROOT / 'dataset/splits/val.csv'
MANIFEST = TRAIN.parent / 'manifest.json'
PRED_COMPARISON = ROOT / 'results/member_C/f3_analysis/f3_a_multiseed/prediction_comparison.csv'
RUNS = [
    ('Baseline', ROOT / 'results/member_C/f3_formal_a_reviewed/F_word_char_s42', 5000, 5000),
    ('F6-A', HERE / 'runs/F6_A_s42', 3000, 7000),
    ('F6-B', HERE / 'runs/F6_B_s42', 7000, 3000),
]
OUTPUTS = ('summary.csv', 'per_class.csv', 'coverage_comparison.csv',
           'coverage_per_title.csv', 'confusion_matrices.csv', 'summary.md', 'audit.json')
EXPECTED_TRAIN = '7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
EXPECTED_VAL = '9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0'


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def table(frame, columns):
    rows = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    rows.extend('| ' + ' | '.join(str(row[c]) for c in columns) + ' |' for _, row in frame.iterrows())
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE,
                        help='output directory inside this project (default: F6 experiment directory)')
    out = parser.parse_args().output_dir.resolve()
    try:
        out.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError('Output directory must be inside the project') from exc
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (out / name).exists():
            raise FileExistsError(f'Will not overwrite {out / name}')
    assert digest(TRAIN) == EXPECTED_TRAIN
    assert json.loads(MANIFEST.read_text(encoding='utf-8'))['train_sha256'] == EXPECTED_TRAIN
    assert digest(VAL) == EXPECTED_VAL
    train, val, previous = read_csv(TRAIN), read_csv(VAL), read_csv(PRED_COMPARISON)
    assert len(train) == 7531 and len(val) == 3184 and val.id.is_unique
    assert previous.id.tolist() == val.id.tolist()
    assert previous.title.tolist() == val.title.tolist() and previous.category.tolist() == val.category.tolist()
    assert set(val.category) == set(LABELS)
    y = val.category.map(C.LABEL_TO_INDEX).to_numpy(dtype=int)
    persistent = (previous.f2_error_count == '5') & (previous.f3_error_count == '5')
    stable_f3 = previous.f3_error_count == '0'
    stable_both = stable_f3 & (previous.f2_error_count == '0')
    assert (int(persistent.sum()), int(stable_f3.sum()), int(stable_both.sum())) == (548, 2435, 2342)
    tokens = tokenize_corpus(val.title.tolist())
    torch.set_num_threads(4)
    all_rows, all_classes, all_confusions, title_coverage, coverage = [], [], [], [], []
    reference_params = reference_feature_params = None
    hashes = {}
    for label, folder, wk, ck in RUNS:
        required = ['metrics.json', 'model.pt', 'feature_builder.pkl', 'history.csv', 'val_predictions.csv']
        assert all((folder / name).is_file() for name in required), f'{label}: missing run artifact'
        m = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        feat, arch, params = m['feature'], m['model'], m['training']['params']
        assert m['seed'] == 42 and m['test'] is None
        assert m['provenance']['split_sha256'] == {'train': EXPECTED_TRAIN, 'val': EXPECTED_VAL}
        assert feat['scheme'] == 'F_word_char' and feat['input_source'] == 'title' and feat['sparse']
        assert feat['fit_rows'] == 7531 and feat['input_dim'] == 10000
        assert feat['word']['selected_features'] == feat['word']['chi2_k_requested'] == wk
        assert feat['char']['selected_features'] == feat['char']['chi2_k_requested'] == ck
        assert arch.get('model_type', 'bp') == 'bp' and arch['input_dim'] == 10000
        assert arch['output_dim'] == 6 and arch['hidden_dim'] == 64 and arch['dropout'] == .2
        assert arch.get('input_dropout', 0) == 0 and arch['n_parameters'] == 640454
        assert params['lr'] == .0003 and params['patience'] == 10 and params['min_delta'] == .0001
        features_comparable = (feat['word']['tfidf_params'], feat['char']['tfidf_params'], feat['normalization'])
        if reference_params is None:
            reference_params, reference_feature_params = params, features_comparable
        assert params == reference_params and features_comparable == reference_feature_params
        with (folder / 'feature_builder.pkl').open('rb') as f:
            b = pickle.load(f)
        assert b.fit_rows_ == 7531 and b.word_dim_ == wk and b.char_dim_ == ck and b.n_features_out_ == 10000
        w = b.word.selector.transform(b.word.vectorizer.transform(tokens)).tocsr()
        c = b.char_selector.transform(b.char_vectorizer.transform(val.title.tolist())).tocsr()
        X = b.transform(val.title.tolist()).tocsr()
        assert sp.issparse(w) and sp.issparse(c) and sp.issparse(X)
        assert w.shape == (3184, wk) and c.shape == (3184, ck) and X.shape == (3184, 10000)
        wn, cn, xn = (np.diff(x.indptr) for x in (w, c, X))
        assert np.array_equal(wn + cn, xn)
        model = build_model(arch)
        model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
        assert model.count_parameters() == arch['n_parameters']
        pred = predict(model, X)
        saved = read_csv(folder / 'val_predictions.csv')
        for col in ('id', 'title', 'category'):
            assert saved[col].tolist() == val[col].tolist(), f'{label}: inconsistent {col}'
        assert np.array_equal(pred, saved.predicted_category.map(C.LABEL_TO_INDEX).to_numpy(dtype=int))
        if label == 'Baseline':
            assert saved.predicted_category.tolist() == previous.f3_pred_42.tolist()
        f1 = f1_score(y, pred, labels=np.arange(6), average='macro', zero_division=0)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred, labels=np.arange(6))
        assert abs(f1 - m['val']['f1_macro']) < 1e-12 and abs(acc - m['val']['accuracy']) < 1e-12
        assert cm.tolist() == m['val']['confusion_matrix']
        hist = pd.read_csv(folder / 'history.csv')
        best_score, best_epoch = -1., 0
        for epoch, score in zip(hist.epoch, hist.val_f1_macro):
            if score > best_score + params['min_delta']:
                best_score, best_epoch = score, int(epoch)
        assert best_epoch == m['training']['best_epoch'] and abs(best_score - f1) < 1e-12
        assert len(hist) == m['training']['epochs_run']
        stopped = len(hist) - best_epoch == params['patience']
        assert stopped or len(hist) == params['max_epochs']
        p, r, fs, support = precision_recall_fscore_support(y, pred, labels=np.arange(6), zero_division=0)
        for i, category in enumerate(LABELS):
            s = m['val']['per_class'][category]
            assert abs(s['precision'] - p[i]) < 1e-12 and abs(s['recall'] - r[i]) < 1e-12 and abs(s['f1'] - fs[i]) < 1e-12
            all_classes.append(dict(scheme=label, category=category, precision=p[i], recall=r[i], f1=fs[i], support=int(support[i])))
            for j, target in enumerate(LABELS):
                all_confusions.append(dict(scheme=label, true_category=category, predicted_category=target, count=int(cm[i, j])))
        all_rows.append(dict(scheme=label, seed=42, word_dim=wk, char_dim=ck, input_dim=X.shape[1],
                             train_rows=7531, val_rows=3184, macro_f1=f1, accuracy=acc, best_epoch=best_epoch,
                             epochs_run=len(hist), early_stopped=stopped, n_parameters=arch['n_parameters'],
                             train_seconds=m['training']['train_time'], feature_seconds=m['training']['feature_time'],
                             society_to_finance=int(cm[LABELS.index('社会'), LABELS.index('财经')]),
                             society_to_entertainment=int(cm[LABELS.index('社会'), LABELS.index('文娱')]),
                             run_path=str(folder.relative_to(ROOT))))
        per_title = pd.DataFrame({'scheme': label, 'id': val.id, 'category': val.category,
                                  'word_nnz': wn, 'char_nnz': cn, 'total_nnz': xn,
                                  'persistent_both': persistent, 'stable_f3': stable_f3, 'stable_both': stable_both})
        title_coverage.append(per_title)
        cohorts = [('all', np.ones(len(val), dtype=bool)), ('persistent_both', persistent),
                   ('stable_f3', stable_f3), ('stable_both', stable_both)]
        for group, mask in cohorts:
            for cat in ('全部', '社会', '财经', '文娱'):
                part = per_title[mask if cat == '全部' else (mask & (val.category == cat))]
                word, char = part.word_nnz.to_numpy(), part.char_nnz.to_numpy()
                coverage.append(dict(scheme=label, cohort=group, category=cat, n=len(part),
                    word_nnz_mean=part.word_nnz.mean(), word_nnz_median=part.word_nnz.median(),
                    char_nnz_mean=part.char_nnz.mean(), char_nnz_median=part.char_nnz.median(),
                    total_nnz_mean=part.total_nnz.mean(), total_nnz_median=part.total_nnz.median(),
                    both_zero=int(((word == 0) & (char == 0)).sum()),
                    word_only=int(((word > 0) & (char == 0)).sum()),
                    char_only=int(((word == 0) & (char > 0)).sum())))
        hashes[label] = {name: digest(folder / name) for name in required}
        print(f'{label}: verified F1={f1:.8f}; strict checkpoint predictions match', flush=True)
    runs = pd.DataFrame(all_rows)
    classes = pd.DataFrame(all_classes)
    cov = pd.DataFrame(coverage)
    title = pd.concat(title_coverage, ignore_index=True)
    hashes.update(train=EXPECTED_TRAIN, validation=EXPECTED_VAL, prediction_comparison=digest(PRED_COMPARISON))
    audit = {'sha256': hashes, 'checkpoint_strict_load_and_prediction_match': True,
             'saved_validation_metrics_recomputed': True, 'test_data_read': False,
             'feature_fit_called': False, 'training_executed': False,
             'fit_scope_evidence': 'run metadata split hash and fit_rows, builder fit_rows; no independent refit in this audit',
             'cohorts': {'persistent_both': 'F2 and F3-A each wrong in all five prior seeds',
                         'stable_f3': 'F3-A correct in all five prior seeds',
                         'stable_both': 'F2 and F3-A each correct in all five prior seeds'}}
    runs.to_csv(out / 'summary.csv', index=False, encoding='utf-8-sig')
    classes.to_csv(out / 'per_class.csv', index=False, encoding='utf-8-sig')
    cov.to_csv(out / 'coverage_comparison.csv', index=False, encoding='utf-8-sig')
    title.to_csv(out / 'coverage_per_title.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(all_confusions).to_csv(out / 'confusion_matrices.csv', index=False, encoding='utf-8-sig')
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    display = runs.copy()
    for col in ('macro_f1', 'accuracy'):
        display[col] = display[col].map(lambda x: f'{x:.5f}')
    display['train_seconds'] = display['train_seconds'].map(lambda x: f'{x:.2f}')
    cdisplay = classes.pivot(index='category', columns='scheme', values='f1').reset_index()
    for col in ('Baseline', 'F6-A', 'F6-B'):
        cdisplay[col] = cdisplay[col].map(lambda x: f'{x:.5f}')
    kdisplay = cov[(cov.cohort.isin(['persistent_both', 'stable_f3', 'stable_both'])) & (cov.category == '全部')].copy()
    for col in ('word_nnz_mean', 'char_nnz_mean', 'total_nnz_mean'):
        kdisplay[col] = kdisplay[col].map(lambda x: f'{x:.3f}')
    base = runs.set_index('scheme').loc['Baseline']
    f1_by_class = classes.pivot(index='category', columns='scheme', values='f1')
    deltas = []
    for _, x in runs[runs.scheme != 'Baseline'].iterrows():
        deltas.append({'scheme': x.scheme, 'macro_f1': f'{x.macro_f1-base.macro_f1:+.5f}',
                       'accuracy': f'{x.accuracy-base.accuracy:+.5f}',
                       'social_f1': f'{f1_by_class.loc["社会", x.scheme]-f1_by_class.loc["社会", "Baseline"]:+.5f}',
                       'finance_f1': f'{f1_by_class.loc["财经", x.scheme]-f1_by_class.loc["财经", "Baseline"]:+.5f}',
                       'entertainment_f1': f'{f1_by_class.loc["文娱", x.scheme]-f1_by_class.loc["文娱", "Baseline"]:+.5f}',
                       'social_to_finance': f'{int(x.society_to_finance-base.society_to_finance):+d}',
                       'social_to_entertainment': f'{int(x.society_to_entertainment-base.society_to_entertainment):+d}'})
    deltas = pd.DataFrame(deltas)
    cov_base = cov[(cov.cohort == 'persistent_both') & (cov.category == '全部')].set_index('scheme')
    coverage_deltas = []
    for scheme in ('F6-A', 'F6-B'):
        v, z = cov_base.loc[scheme], cov_base.loc['Baseline']
        coverage_deltas.append({'scheme': scheme, 'word_delta': f'{v.word_nnz_mean-z.word_nnz_mean:+.3f}',
                                'char_delta': f'{v.char_nnz_mean-z.char_nnz_mean:+.3f}',
                                'total_delta': f'{v.total_nnz_mean-z.total_nnz_mean:+.3f}',
                                'zero_delta': f'{int(v.both_zero-z.both_zero):+d}'})
    coverage_deltas = pd.DataFrame(coverage_deltas)
    lines = ['# F.6 seed=42 特征配额受控对照', '',
             '固定 F3-A 训练集 7,531 条及原验证集 3,184 条，BP hidden=64、学习率 0.0003、dropout=0.2、input dropout=0。训练集与验证集 SHA-256、特征配置、保存模型与预测和早停经 `audit.json` 所述检查。三次均仅是 seed=42，性能差异不能视作多种子稳定结果。', '',
             '实际训练命令：', '',
             '```powershell',
             'python train_bp.py --exp F6_A F6_B --seeds 42 --model-type bp --hidden 64 --lr 0.0003 --dropout 0.2 --input-dropout 0 --train-file results/member_C/f3_analysis/f3_a_dataset_formal/train.csv --train-sha256 7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880 --output-root results/member_C/f6_experiments/runs',
             '```', '',
             '## 验证指标', '', table(display, ['scheme', 'word_dim', 'char_dim', 'macro_f1', 'accuracy', 'best_epoch', 'epochs_run', 'train_seconds', 'society_to_finance', 'society_to_entertainment']), '',
             '模型参数量均为 640,454。训练耗时不包含特征拟合时间，基准与本次实验的耗时来自不同运行，硬件负载等条件不一定一致，不能严格比较；完整记录见 `summary.csv`。', '',
             '## 相对于 seed=42 基准的差值', '',
             'F1 与 Accuracy 的数值为绝对差，混淆方向为误分类条数差；正数表示增加。', '',
             table(deltas, ['scheme', 'macro_f1', 'accuracy', 'social_f1', 'finance_f1', 'entertainment_f1', 'social_to_finance', 'social_to_entertainment']), '',
             '## 六类别 F1', '', table(cdisplay, ['category', 'Baseline', 'F6-A', 'F6-B']), '',
             '## 固定样本组的非零特征覆盖', '',
             '持续错误是原 F2 与原 F3-A 各五种子全部错误；stable_f3 为 F3-A 五种子全对；stable_both 为两组均五种子全对。分组固定不随新实验结果改变。', '',
             table(kdisplay, ['scheme', 'cohort', 'n', 'word_nnz_mean', 'char_nnz_mean', 'total_nnz_mean', 'both_zero', 'word_only', 'char_only']), '',
             '相对于基准，固定持续错误样本组的分支非零均值及全零样本数差值：', '',
             table(coverage_deltas, ['scheme', 'word_delta', 'char_delta', 'total_delta', 'zero_delta']), '',
             '社会、财经、文娱逐类覆盖见 `coverage_comparison.csv`；逐标题数据见 `coverage_per_title.csv`。精确率、召回率与混淆矩阵见 `per_class.csv`、`confusion_matrices.csv`。选中特征覆盖变化是描述性结果，无法单独确定性能变化原因。', '',
             '本脚本没有拟合任何特征处理器，没有训练模型，也没有访问测试集。训练时仅用训练集拟合的证据是训练实现、记录的训练文件哈希和处理器 fit_rows；本脚本未再次独立拟合。', '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('F6 summary completed', flush=True)


if __name__ == '__main__':
    main()

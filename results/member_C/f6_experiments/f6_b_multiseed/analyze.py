"""Audit the ten frozen validation runs and produce paired F6-B statistics.

No feature fitting or training is performed.  All outputs are written to a new
directory; no historical experiment is modified.
"""
import argparse
from hashlib import sha256
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
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import model_config as C
from models.bp import build_model
from models.trainer import predict

TRAIN = ROOT / 'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv'
VAL = ROOT / 'dataset/splits/val.csv'
PREVIOUS = ROOT / 'results/member_C/f3_analysis/f3_a_multiseed/summary.csv'
EXPECTED_TRAIN = '7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
EXPECTED_VAL = '9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0'
SEEDS = (42, 43, 44, 45, 46)
OUTPUTS = ('per_seed.csv', 'paired.csv', 'aggregate.csv', 'per_class.csv',
           'per_class_aggregate.csv', 'confusion_matrices.csv', 'confusion_aggregate.csv',
           'audit.json', 'summary.md')


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')


def md_table(df, cols):
    lines = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    lines.extend('| ' + ' | '.join(str(row[c]) for c in cols) + ' |' for _, row in df.iterrows())
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=HERE)
    out = p.parse_args().output_dir.resolve()
    out.relative_to(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS:
        if (out / name).exists():
            raise FileExistsError(f'Refusing to overwrite {out / name}')
    assert sha(TRAIN) == EXPECTED_TRAIN and sha(VAL) == EXPECTED_VAL
    manifest = json.loads((TRAIN.parent / 'manifest.json').read_text(encoding='utf8'))
    assert manifest['train_sha256'] == EXPECTED_TRAIN
    train, val, prior = csv(TRAIN), csv(VAL), csv(PREVIOUS)
    assert len(train) == 7531 and len(val) == 3184 and val.id.is_unique
    assert set(val.category) == set(C.LABELS)
    y = val.category.map(C.LABEL_TO_INDEX).to_numpy(dtype=int)
    baseline = prior[prior.group == 'F3-A'].set_index('seed')
    assert set(baseline.index) == {str(s) for s in SEEDS}
    torch.set_num_threads(4)
    all_runs, all_classes, all_confusions, hashes = [], [], [], {}
    expected_params = expected_feature_params = None
    reference_vocab, reference_support = {}, {}

    for group, word_dim, char_dim in [('F3-A baseline', 5000, 5000), ('F6-B', 7000, 3000)]:
        for seed in SEEDS:
            if group == 'F3-A baseline':
                folder = ROOT / Path(baseline.loc[str(seed), 'run_path'])
            elif seed == 42:
                folder = HERE.parent / 'runs/F6_B_s42'
            else:
                folder = HERE / f'runs/F6_B_s{seed}'
            required = ['metrics.json', 'history.csv', 'model.pt', 'feature_builder.pkl', 'val_predictions.csv']
            assert all((folder / n).is_file() for n in required), f'Missing artifacts: {folder}'
            m = json.loads((folder / 'metrics.json').read_text(encoding='utf8'))
            feat, arch, training = m['feature'], m['model'], m['training']
            params = training['params']
            assert m['seed'] == seed
            assert m['provenance']['split_sha256'] == {'train': EXPECTED_TRAIN, 'val': EXPECTED_VAL}
            assert feat['scheme'] == 'F_word_char' and feat['input_source'] == 'title' and feat['sparse']
            assert feat['fit_rows'] == 7531 and feat['input_dim'] == 10000
            assert feat['word']['selected_features'] == feat['word']['chi2_k_requested'] == word_dim
            assert feat['char']['selected_features'] == feat['char']['chi2_k_requested'] == char_dim
            assert arch.get('model_type', 'bp') == 'bp' and arch['input_dim'] == 10000
            assert arch['output_dim'] == 6 and arch['hidden_dim'] == 64 and arch['dropout'] == .2
            assert arch.get('input_dropout', 0) == 0 and arch['n_parameters'] == 640454
            assert params['lr'] == .0003 and params['patience'] == 10 and params['min_delta'] == .0001
            feature_params = (feat['word']['tfidf_params'], feat['char']['tfidf_params'], feat['normalization'])
            if expected_params is None:
                expected_params, expected_feature_params = params, feature_params
            assert params == expected_params and feature_params == expected_feature_params
            with (folder / 'feature_builder.pkl').open('rb') as f:
                builder = pickle.load(f)
            assert builder.fit_rows_ == 7531 and builder.word_dim_ == word_dim and builder.char_dim_ == char_dim
            assert builder.n_features_out_ == 10000
            for part, vectorizer, selector in [('word', builder.word.vectorizer, builder.word.selector),
                                                ('char', builder.char_vectorizer, builder.char_selector)]:
                if part not in reference_vocab:
                    reference_vocab[part] = (dict(vectorizer.vocabulary_), vectorizer.idf_.copy())
                vocab, idf = reference_vocab[part]
                assert vectorizer.vocabulary_ == vocab
                np.testing.assert_allclose(vectorizer.idf_, idf, rtol=0, atol=1e-14)
                key = (group, part)
                support = selector.get_support()
                if key not in reference_support:
                    reference_support[key] = support.copy()
                assert np.array_equal(support, reference_support[key])
            X = builder.transform(val.title.tolist())
            assert sparse.isspmatrix_csr(X) and X.shape == (3184, 10000)
            model = build_model(arch)
            model.load_state_dict(torch.load(folder / 'model.pt', map_location='cpu', weights_only=True), strict=True)
            assert model.count_parameters() == 640454
            pred = predict(model, X)
            saved = csv(folder / 'val_predictions.csv')
            for col in ('id', 'title', 'category'):
                assert saved[col].tolist() == val[col].tolist(), f'{folder} mismatch {col}'
            assert np.array_equal(pred, saved.predicted_category.map(C.LABEL_TO_INDEX).to_numpy(dtype=int))
            f1 = f1_score(y, pred, labels=np.arange(6), average='macro', zero_division=0)
            acc = accuracy_score(y, pred)
            cm = confusion_matrix(y, pred, labels=np.arange(6))
            assert abs(f1 - m['val']['f1_macro']) < 1e-12 and abs(acc - m['val']['accuracy']) < 1e-12
            assert cm.tolist() == m['val']['confusion_matrix']
            history = pd.read_csv(folder / 'history.csv')
            best_score, best_epoch = -1., 0
            for epoch, score in zip(history.epoch, history.val_f1_macro):
                if score > best_score + params['min_delta']:
                    best_score, best_epoch = score, int(epoch)
            assert best_epoch == training['best_epoch'] and abs(best_score - f1) < 1e-12
            assert len(history) == training['epochs_run']
            stopped = len(history) - best_epoch == params['patience']
            assert stopped or len(history) == params['max_epochs']
            precision, recall, fs, supports = precision_recall_fscore_support(
                y, pred, labels=np.arange(6), zero_division=0)
            for i, cat in enumerate(C.LABELS):
                saved_class = m['val']['per_class'][cat]
                for metric, value in [('precision', precision[i]), ('recall', recall[i]), ('f1', fs[i])]:
                    assert abs(saved_class[metric] - value) < 1e-12
                all_classes.append(dict(group=group, seed=seed, category=cat, precision=precision[i],
                                        recall=recall[i], f1=fs[i], support=int(supports[i])))
                for j, target in enumerate(C.LABELS):
                    all_confusions.append(dict(group=group, seed=seed, true_category=cat,
                                               predicted_category=target, count=int(cm[i, j])))
            all_runs.append(dict(group=group, seed=seed, macro_f1=f1, accuracy=acc, best_epoch=best_epoch,
                                 epochs_run=len(history), early_stopped=stopped, word_dim=word_dim, char_dim=char_dim,
                                 n_parameters=arch['n_parameters'], train_seconds=training['train_time'],
                                 society_to_finance=int(cm[C.LABELS.index('社会'), C.LABELS.index('财经')]),
                                 society_to_entertainment=int(cm[C.LABELS.index('社会'), C.LABELS.index('文娱')]),
                                 run_path=str(folder.relative_to(ROOT))))
            hashes[f'{group} seed={seed}'] = {n: sha(folder / n) for n in required}
            print(f'{group} seed={seed}: strict checkpoint and predictions verified; F1={f1:.8f}', flush=True)

    runs = pd.DataFrame(all_runs)
    classes, confusion = pd.DataFrame(all_classes), pd.DataFrame(all_confusions)
    base = runs[runs.group == 'F3-A baseline'].set_index('seed')
    cand = runs[runs.group == 'F6-B'].set_index('seed')
    paired = pd.DataFrame({'seed': SEEDS})
    for metric in ('macro_f1', 'accuracy', 'society_to_finance', 'society_to_entertainment'):
        paired[f'baseline_{metric}'] = [base.loc[s, metric] for s in SEEDS]
        paired[f'f6b_{metric}'] = [cand.loc[s, metric] for s in SEEDS]
        paired[f'delta_{metric}'] = paired[f'f6b_{metric}'] - paired[f'baseline_{metric}']
    agg = runs.groupby('group', sort=False).agg(
        n=('seed', 'size'), macro_f1_mean=('macro_f1', 'mean'), macro_f1_std_sample=('macro_f1', 'std'),
        accuracy_mean=('accuracy', 'mean'), accuracy_std_sample=('accuracy', 'std'),
        society_to_finance_mean=('society_to_finance', 'mean'),
        society_to_entertainment_mean=('society_to_entertainment', 'mean')).reset_index()
    class_agg = classes.groupby(['group', 'category'], sort=False).agg(
        precision_mean=('precision', 'mean'), recall_mean=('recall', 'mean'),
        f1_mean=('f1', 'mean'), f1_std_sample=('f1', 'std')).reset_index()
    conf_agg = confusion.groupby(['group', 'true_category', 'predicted_category'], sort=False).agg(
        count_mean=('count', 'mean'), count_std_sample=('count', 'std')).reset_index()
    for name, df in [('per_seed.csv', runs), ('paired.csv', paired), ('aggregate.csv', agg),
                     ('per_class.csv', classes), ('per_class_aggregate.csv', class_agg),
                     ('confusion_matrices.csv', confusion), ('confusion_aggregate.csv', conf_agg)]:
        df.to_csv(out / name, index=False, encoding='utf-8-sig')
    audit = {'train_sha256': EXPECTED_TRAIN, 'validation_sha256': EXPECTED_VAL, 'artifact_sha256': hashes,
             'strict_checkpoint_and_all_saved_predictions_match': True, 'metrics_and_confusions_recomputed': True,
             'best_epoch_and_early_stopping_checked': True, 'features_sparse': True,
             'same_vectorizer_vocab_idf_and_group_selector_support': True,
             'no_test_evaluation': True, 'no_fitting_or_training_in_this_audit': True,
             'training_only_fit_evidence': 'training implementation, fit_rows and frozen data hashes; no independent refit'}
    (out / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf8')
    da = agg.copy()
    for c in da.columns[2:]: da[c] = da[c].map(lambda x: f'{x:.5f}')
    dp = paired.copy()
    for c in dp.columns:
        if 'macro_f1' in c or 'accuracy' in c: dp[c] = dp[c].map(lambda x: f'{x:.5f}')
    dc = class_agg.pivot(index='category', columns='group', values='f1_mean').reset_index()
    for c in ('F3-A baseline', 'F6-B'): dc[c] = dc[c].map(lambda x: f'{x:.5f}')
    top = conf_agg[conf_agg.true_category != conf_agg.predicted_category].pivot(
        index=['true_category', 'predicted_category'], columns='group', values='count_mean').reset_index()
    top['max_mean'] = top[['F3-A baseline', 'F6-B']].max(axis=1)
    top = top.sort_values(['max_mean', 'true_category', 'predicted_category'], ascending=[False, True, True]).head(8)
    top['delta'] = top['F6-B'] - top['F3-A baseline']
    for c in ('F3-A baseline', 'F6-B', 'delta'): top[c] = top[c].map(lambda x: f'{x:+.2f}' if c == 'delta' else f'{x:.2f}')
    diff = paired.delta_macro_f1.to_numpy()
    dacc = paired.delta_accuracy.to_numpy()
    lines = ['# F6-B 五随机种子验证', '',
             'F6-B 根据 seed=42 初步结果预先选出；比较沿用同一固定验证集，不是独立测试。训练文件 7,531 条、验证集 3,184 条，固定数据和原早停配置；五种子样本标准差的分母为 n-1。', '',
             '## 整体指标', '', md_table(da, list(da.columns)), '',
             '## 同种子配对', '', md_table(dp, list(dp.columns)), '',
             f'五个配对 Macro-F1 差值平均 {np.mean(diff):+.5f}；{sum(diff > 0)}/5 个种子提高，{sum(diff < 0)}/5 个降低，{sum(diff == 0)}/5 个相等。Accuracy 平均配对差值 {np.mean(dacc):+.5f}。', '',
             '## 六类别五种子平均 F1', '', md_table(dc, ['category', 'F3-A baseline', 'F6-B']), '',
             '## 主要误分类方向：五种子平均条数', '',
             '以下选取两组中平均误分类数量较大的八个方向；差值为 F6-B 减基准。', '',
             md_table(top, ['true_category', 'predicted_category', 'F3-A baseline', 'F6-B', 'delta']), '',
             '逐种子类别 Precision、Recall、F1 见 `per_class.csv`；逐种子 6×6 混淆矩阵和五种子方向均值见 `confusion_matrices.csv`、`confusion_aggregate.csv`。', '',
             '10 组模型均已严格重建并逐条复算验证预测、指标和混淆；逐组核查实际维数、模型参数、数据哈希、最佳轮和早停记录。处理器仅训练集拟合依据训练代码与记录；本次审计不重复拟合。具体文件哈希见 `audit.json`。', '',
             '没有读取测试集或运行测试集模型评估；没有根据本结果继续搜索配额。单个固定验证集上的五种子波动不能替代独立泛化验证。', '']
    (out / 'summary.md').write_text('\n'.join(lines), encoding='utf8')
    print('F6-B paired multiseed audit completed', flush=True)


if __name__ == '__main__':
    main()

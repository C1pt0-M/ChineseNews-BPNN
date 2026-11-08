"""Leakage-controlled word/character TF-IDF classic baselines on train/val only."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

import model_config as C
from models.data import load_splits
from models.f_features import TitleFeatureBuilder
from models.metrics import compute_metrics
from models.plots import plot_confusion_matrix
from models.runs import save_confusion, save_json

CS = (0.01, 0.1, 1.0, 10.0)
KINDS = ('LinearSVC', 'LogisticRegression')
F2_RUN = C.BASE_DIR / 'results/member_C/f2_formal_word_char/F_word_char_s42'


def estimator(kind, c, seed):
    if kind == 'LinearSVC':
        return LinearSVC(C=c, random_state=seed, dual='auto', max_iter=20000, tol=1e-4)
    if kind == 'LogisticRegression':
        return LogisticRegression(C=c, solver='lbfgs', random_state=seed, max_iter=3000, tol=1e-4)
    raise ValueError(f'Unknown classifier: {kind}')


def fit_checked(classifier, X, y):
    if not sp.issparse(X):
        raise TypeError('Feature matrix must remain sparse')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(X, y)
    return classifier


def cross_validate(texts, y, seed=42, n_splits=5, builder_factory=TitleFeatureBuilder, log_fn=print):
    """Fit all feature state inside each fold; share fold transforms only across classifiers."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    items = np.asarray(texts, dtype=object)
    observations = []
    for fold, (tr, va) in enumerate(cv.split(items, y), 1):
        t0 = time.perf_counter()
        builder = builder_factory('F_word_char', seed=seed)
        builder.fit(items[tr].tolist(), y[tr])
        X_tr = builder.transform(items[tr].tolist())
        X_va = builder.transform(items[va].tolist())
        if not sp.issparse(X_tr) or not sp.issparse(X_va):
            raise TypeError('CV features must remain sparse')
        feature_seconds = time.perf_counter() - t0
        for kind in KINDS:
            for c in CS:
                t1 = time.perf_counter()
                model = fit_checked(estimator(kind, c, seed), X_tr, y[tr])
                score = f1_score(y[va], model.predict(X_va), labels=np.arange(C.N_CLASSES),
                                 average='macro', zero_division=0)
                observations.append({'fold': fold, 'classifier': kind, 'C': c,
                                     'macro_f1': float(score), 'fit_seconds': time.perf_counter()-t1,
                                     'feature_fit_seconds': feature_seconds,
                                     'train_rows': len(tr), 'validation_rows': len(va),
                                     'train_indices_sha256': hashlib.sha256(tr.astype('<i8').tobytes()).hexdigest(),
                                     'validation_indices_sha256': hashlib.sha256(va.astype('<i8').tobytes()).hexdigest(),
                                     'feature_fit_rows': builder.fit_rows_, 'input_dim': X_tr.shape[1]})
        if log_fn:
            log_fn(f'fold {fold}/{n_splits} complete: fit={len(tr)}, validate={len(va)}, dim={X_tr.shape[1]}')
    return observations


def select_c(observations, kind):
    """Highest mean fold Macro-F1; a tie selects the smallest C."""
    means = {c: float(np.mean([r['macro_f1'] for r in observations if r['classifier'] == kind and r['C'] == c]))
             for c in CS}
    if any(not np.isfinite(v) for v in means.values()):
        raise ValueError('Incomplete cross-validation scores')
    return max(CS, key=lambda c: (means[c], -c)), means


def load_f2_builder(train_texts, y_train):
    """Verify original provenance and independently refit to establish feature identity."""
    payload = json.loads((F2_RUN/'metrics.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256((C.SPLIT_DIR/'train.csv').read_bytes()).hexdigest()
    if payload['provenance']['split_sha256']['train'] != digest or payload['feature']['fit_rows'] != len(y_train):
        raise RuntimeError('F2 feature provenance does not match the current train split')
    with (F2_RUN/'feature_builder.pkl').open('rb') as f:
        builder = pickle.load(f)
    fresh = TitleFeatureBuilder('F_word_char', seed=42).fit(train_texts, y_train)
    for old, new in ((builder.char_vectorizer, fresh.char_vectorizer), (builder.word.vectorizer, fresh.word.vectorizer)):
        if old.vocabulary_ != new.vocabulary_:
            raise RuntimeError('F2 feature vocabulary differs from fresh training-only fit')
        np.testing.assert_allclose(old.idf_, new.idf_, rtol=0, atol=1e-14)
    for old, new in ((builder.char_selector, fresh.char_selector), (builder.word.selector, fresh.word.selector)):
        np.testing.assert_array_equal(old.get_support(), new.get_support())
        np.testing.assert_allclose(old.scores_, new.scores_, rtol=1e-12, atol=1e-12, equal_nan=True)
    return builder


def main(argv=None):
    parser = argparse.ArgumentParser(description='Train leakage-controlled LinearSVC and LogisticRegression baselines')
    parser.add_argument('--output-root', type=Path, default=C.BASE_DIR/'results/member_C/classic_baselines')
    args = parser.parse_args(argv)
    out = args.output_root
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    out.mkdir(parents=True)
    splits = load_splits(verify=True, splits=('train','val'))
    train_text, y_train, _ = splits['train']
    val_text, y_val, val_df = splits['val']
    if len(train_text) != 6549 or len(val_text) != 3184:
        raise RuntimeError('Unexpected train/validation split sizes')
    observations = cross_validate(train_text, y_train)
    pd.DataFrame(observations).to_csv(out/'cv_folds.csv', index=False, encoding='utf-8-sig')
    chosen = {kind: select_c(observations, kind) for kind in KINDS}
    feature_start = time.perf_counter()
    builder = load_f2_builder(train_text, y_train)
    X_train = builder.transform(train_text)
    X_val = builder.transform(val_text)
    assert sp.issparse(X_train) and sp.issparse(X_val) and X_train.shape[1] == X_val.shape[1] == 10000
    features_seconds = time.perf_counter()-feature_start
    with (out/'feature_builder.pkl').open('xb') as f:
        pickle.dump(builder, f, protocol=pickle.HIGHEST_PROTOCOL)
    scores = {}
    for kind in KINDS:
        model_dir = out/kind
        model_dir.mkdir()
        c, means = chosen[kind]
        t0 = time.perf_counter()
        model = fit_checked(estimator(kind, c, 42), X_train, y_train)
        train_seconds = time.perf_counter()-t0
        pred = model.predict(X_val)
        metrics = compute_metrics(y_val, pred)
        details = val_df.copy()
        details['predicted_category'] = [C.LABELS[i] for i in pred]
        details.to_csv(model_dir/'val_predictions.csv', index=False, encoding='utf-8-sig')
        with (model_dir/'model.pkl').open('xb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        save_confusion(metrics, model_dir/'confusion_val.csv')
        plot_confusion_matrix(metrics['confusion_matrix'], C.LABELS, f'Validation · {kind}', model_dir/'confusion_val.png')
        scores[kind] = {'classifier': kind, 'best_C': c, 'cv_macro_f1': means[c],
                        'cv_mean_by_C': {str(k): v for k,v in means.items()},
                        'final_fit_seconds': train_seconds, 'params': model.get_params(), 'val': metrics}
        save_json(model_dir/'metrics.json', scores[kind])
        print(f'{kind}: C={c}, CV={means[c]:.6f}, val F1={metrics["f1_macro"]:.6f}, accuracy={metrics["accuracy"]:.6f}', flush=True)
    result = {'seed':42, 'cv': {'n_splits':5,'stratified':True,'shuffle':True,'selection':'mean_macro_f1',
                              'tie_break':'smaller C','C_values':list(CS),'fold_feature_fit':'fold training rows only'},
              'train_rows':len(train_text), 'val_rows':len(val_text), 'feature':builder.describe(),
              'feature_fit_source':str(F2_RUN/'feature_builder.pkl'), 'feature_identity_verified_by_refit':True,
              'feature_prepare_seconds_including_refit':features_seconds,
              'split_sha256':{s: hashlib.sha256((C.SPLIT_DIR/f'{s}.csv').read_bytes()).hexdigest() for s in ('train','val')},
              'versions':{'python':platform.python_version(),'scikit_learn':sklearn.__version__}, 'models':scores,
              'test_evaluated':False}
    save_json(out/'results.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

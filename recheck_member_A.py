"""Member A acceptance run. No test inference, tuning, or dataset mutation.

python recheck_member_A.py run
python recheck_member_A.py analyze
The run command refuses an existing output root. Analyze writes only this
round's diagnostic artifacts; previous audit directories are never touched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import jieba
import numpy as np
import pandas as pd

import model_config as C
from models.data import file_sha256, load_splits, verify_dataset
from models.features import FeatureBuilder, tokenize_corpus, tokenize, _KEEP_RE
from models.runs import save_json

ROOT = Path('results/member_A_baseline_recheck')


def removed_numeric_tokens(raw_tokens):
    """Report only numeric pieces that the actual B0 keep rule removes.

    Chinese numeric words such as 四/万亿 are retained by B0 and must not be
    mistaken for removed tokens merely because str.isnumeric() is true.
    """
    return [t.strip() for t in raw_tokens if t.strip().isnumeric() and not _KEEP_RE.search(t)]


def capture(command, name):
    env = dict(os.environ, PYTHONUTF8='1')
    with open(ROOT / name, 'w', encoding='utf-8') as log:
        log.write('COMMAND: ' + ' '.join(command) + '\n')
        log.flush()
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    if code:
        raise RuntimeError(f'{name} failed with exit {code}; see log')
    print(f'{name}: PASS', flush=True)


def protected_files():
    names = subprocess.check_output(['git', 'ls-files', 'dataset', 'results', 'reports'], text=True).splitlines()
    return {p: file_sha256(Path(p)) for p in names if Path(p).is_file()}


def run():
    ROOT.mkdir(parents=True, exist_ok=False)
    save_json(ROOT/'preflight.json', {
        'started_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
        'protected_sha256': protected_files(),
        'b0': {'tfidf': C.TFIDF_PARAMS, 'network': C.BP_PARAMS, 'training': C.TRAIN_PARAMS,
               'seed': 42, 'expected_input_dim': 10864, 'expected_parameters': 2782982},
        'package_versions': {p: importlib.metadata.version(p) for p in
                            ['torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'jieba', 'matplotlib']},
    })
    source = [Path(p) for p in ['train_bp.py', 'evaluate_test.py', 'verify_dataset.py',
                               'baseline_lr.py', 'model_config.py', 'recheck_member_A.py']]
    source += list(Path('models').glob('*.py')) + list(Path('tests').glob('*.py'))
    for path in source:
        target = ROOT/'source_snapshot'/path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    capture([sys.executable, 'verify_dataset.py'], 'dataset_verification.log')
    capture([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'unit_tests.log')
    capture([sys.executable, 'tests/run_training_without_test.py', '--exp', 'A_raw',
             '--seeds', '42', '--output-root', str(ROOT)], 'training.log')


def analyze():
    out = ROOT/'A_raw_s42'
    version = json.loads(C.DATASET_VERSION_FILE.read_text(encoding='utf-8'))
    fingerprints = verify_dataset(splits=('train', 'val', 'test'))  # integrity only, no model
    assert version['dataset_version']=='1.1.0' and version['split_version']=='1.0.0'
    assert version['label_mapping']['category_to_index']==C.LABEL_TO_INDEX
    splits=load_splits(splits=('train','val'))
    # Token diagnostics are freshly computed (no cache). Builder is train-only fit.
    train_tokens=tokenize_corpus(splits['train'][0])
    val_tokens=tokenize_corpus(splits['val'][0])
    builder=FeatureBuilder('A').fit(train_tokens,splits['train'][1])
    X=builder.transform(train_tokens);V=builder.transform(val_tokens)
    assert X.shape==(6549,10864) and V.shape==(3184,10864)
    vocab=builder.vectorizer.vocabulary_
    terms=builder.vectorizer.get_feature_names_out()
    training_token_types={t for doc in train_tokens for t in doc}
    summaries=[]
    zero_documents=[]
    for name,tokens,matrix in [('train',train_tokens,X),('val',val_tokens,V)]:
        occurrence=Counter(t for doc in tokens for t in doc)
        oov={t:n for t,n in occurrence.items() if t not in vocab}
        unseen={t:n for t,n in occurrence.items() if t not in training_token_types}
        summaries.append({'split':name,'documents':len(tokens),'mean_tokens':float(np.mean([len(d) for d in tokens])),
            'empty_token_documents':sum(not d for d in tokens),
            'empty_token_ratio':sum(not d for d in tokens)/len(tokens),
            'zero_feature_documents':int((matrix.getnnz(axis=1)==0).sum()),
            'token_occurrences':sum(occurrence.values()),'unique_tokens':len(occurrence),
            'oov_token_occurrences':sum(oov.values()),
            'oov_occurrence_rate':sum(oov.values())/sum(occurrence.values()),
            'oov_unique_rate':len(oov)/len(occurrence),'tfidf_features':len(vocab),
            'unseen_in_train_occurrence_rate':sum(unseen.values())/sum(occurrence.values()),
            'unseen_in_train_unique_rate':len(unseen)/len(occurrence)})
        for ix in np.flatnonzero(matrix.getnnz(axis=1)==0):
            zero_documents.append({'split':name,**splits[name][2].iloc[ix].to_dict(),
                                   'retained_tokens':tokens[ix]})
        pd.DataFrame(sorted(oov.items(),key=lambda x:(-x[1],x[0])),columns=['token','count']).to_csv(ROOT/f'{name}_oov_tokens.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summaries).to_csv(ROOT/'token_diagnostics.csv',index=False,encoding='utf-8-sig')
    save_json(ROOT/'zero_feature_documents.json',zero_documents)
    # Cross-split checks read test solely for integrity, not prediction or feature diagnosis.
    frames={s:pd.read_csv(C.SPLIT_DIR/f'{s}.csv',encoding='utf-8-sig') for s in ['train','val','test']}
    overlaps=[]
    for a,b in [('train','val'),('train','test'),('val','test')]:
        for field in ['id','url','title']:
            n=len(set(frames[a][field]) & set(frames[b][field]))
            assert n==0
            overlaps.append({'left':a,'right':b,'field':field,'intersection':n})
    save_json(ROOT/'data_integrity.json',{'fingerprints':fingerprints,'exact_intersections':overlaps,
              'label_mapping':version['label_mapping'],'dataset_version':version['dataset_version'],
              'split_version':version['split_version']})
    # Prefix is a risk proxy, not a verified same-event or plagiarism detector.
    all_rows=pd.concat([df.assign(split=s) for s,df in frames.items()],ignore_index=True)
    all_rows['prefix']=all_rows.title.str[:8]
    prefixes=all_rows.groupby('prefix').agg(records=('id','size'),split_count=('split','nunique'),category_count=('category','nunique'))
    cross=prefixes[prefixes.split_count>1]
    cross.to_csv(ROOT/'cross_split_prefix_groups.csv',encoding='utf-8-sig')
    save_json(ROOT/'near_duplicate_risk.json',{'definition':'first 8 title characters, exact prefix; integrity risk only',
        'cross_split_groups':len(cross),'affected_records':int(cross.records.sum()),
        'cross_split_cross_category_groups':int((cross.category_count>1).sum()),
        'not_proof_of_same_event':True})
    cases=pd.read_csv('results/member_A_audit_20261105/A_raw_s42/typical_errors_20.csv',encoding='utf-8-sig')
    predictions=pd.read_csv(out/'val_predictions.csv',encoding='utf-8-sig').set_index('id')
    details=[]
    for row in cases.to_dict(orient='records'):
        ix=splits['val'][2].index[splits['val'][2].id==row['id']][0]
        tokens=val_tokens[ix]
        features=[{'term':str(terms[j]),'tfidf':float(w)} for j,w in zip(V[ix].indices,V[ix].data)]
        features.sort(key=lambda d:(-d['tfidf'],d['term']))
        raw=jieba.lcut(row['title'])
        row.update(raw_jieba_tokens=raw,retained_tokens=tokens,
                   removed_pure_numbers=removed_numeric_tokens(raw),
                   oov_tokens=[t for t in tokens if t not in vocab],nonzero_features=features,
                   nonzero_feature_count=len(features))
        assert row['predicted_category']==predictions.loc[row['id'],'predicted_category']
        details.append(row)
    save_json(ROOT/'error_feature_evidence.json',details)
    # Named entities: fixed diagnostic strings and actual occurrences in train/val.
    probes=['中国人民银行','北京大学','中国科学院','美加墨世界杯','全运会','SNH48','丝芭','纤维芯片']
    save_json(ROOT/'named_entity_tokenization.json',[
        {'text':t,'jieba_tokens':jieba.lcut(t),'retained_tokens':tokenize(t),
         'retained_token_in_vocab':{s:s in vocab for s in tokenize(t)},
         'train_title_mentions':sum(t in x for x in splits['train'][0]),
         'val_title_mentions':sum(t in x for x in splits['val'][0])} for t in probes])
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    old=json.loads((C.RUN_DIR/'A_raw_s42/metrics.json').read_text(encoding='utf-8'))
    history=pd.read_csv(out/'history.csv');historic=pd.read_csv(C.RUN_DIR/'A_raw_s42/history.csv')
    previous=pd.read_csv('results/member_A_audit_20261105/A_raw_s42/historical_best_val_predictions.csv',encoding='utf-8-sig')
    differences={col:float((history[col]-historic[col]).abs().max()) for col in ['train_loss','val_loss','val_f1_macro','val_accuracy']}
    same_prediction=predictions.loc[previous.id,'predicted_category'].tolist()==previous.predicted_category.tolist()
    check={'best_epoch_equal':m['training']['best_epoch']==old['training']['best_epoch'],
           'epochs_run_equal':len(history)==len(historic),'max_absolute_history_difference':differences,
           'history_tolerance':0.0,'validation_predictions_equal':same_prediction,
           'confusion_equal':m['val']['confusion_matrix']==old['val']['confusion_matrix'],
           'parameters_equal':m['model']['n_parameters']==old['model']['n_parameters']==2782982,
           'no_test_read_attempts':'Runtime test.csv read attempts: 0' in (ROOT/'training.log').read_text(encoding='utf-8'),
           'test_metrics':m['test']}
    assert all(check[k] for k in ['best_epoch_equal','epochs_run_equal','validation_predictions_equal','confusion_equal','parameters_equal','no_test_read_attempts'])
    assert len(history)==len(historic) and max(differences.values())==0 and m['test'] is None
    save_json(ROOT/'reproduction_check.json',check)
    five=[]
    for seed in C.MULTI_SEEDS:
        saved=json.loads((C.RUN_DIR/f'A_raw_s{seed}/metrics.json').read_text(encoding='utf-8'))
        five.append({'seed':seed,'best_epoch':saved['training']['best_epoch'],'epochs_run':saved['training']['epochs_run'],
                     **{f'val_{k}':saved['val'][k] for k in C.METRIC_NAMES}})
    five=pd.DataFrame(five);five.to_csv(ROOT/'five_seed_validation.csv',index=False,encoding='utf-8-sig')
    summary=[{'metric':c,'mean':five[c].mean(),'sample_std_ddof1':five[c].std(ddof=1),'n':5}
             for c in five if c.startswith('val_')]
    pd.DataFrame(summary).to_csv(ROOT/'five_seed_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(m['val']['per_class']).T.rename_axis('category').to_csv(ROOT/'per_class_val.csv',encoding='utf-8-sig')
    from member_A_recheck_report import generate
    generate(ROOT)
    for p in [Path('recheck_member_A.py'),Path('member_A_recheck_report.py'),
              *Path('tests').glob('*.py')]:
        target=ROOT/'analysis_source_snapshot'/p
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    before=json.loads((ROOT/'preflight.json').read_text(encoding='utf-8'))['protected_sha256']
    changed=[p for p,digest in before.items() if not Path(p).is_file() or file_sha256(Path(p))!=digest]
    assert not changed,changed
    save_json(ROOT/'protection_check.json',{'checked_files':len(before),'changed_files':changed,
             'original_dataset_and_historical_results_unchanged':True})
    save_json(ROOT/'artifact_manifest.json',{str(p.relative_to(ROOT)):file_sha256(p) for p in ROOT.rglob('*')
        if p.is_file() and p.name!='artifact_manifest.json'})
    print('Diagnostics, reproduction, reports and protected-file checks: PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('phase',choices=['run','analyze'])
    args=parser.parse_args()
    if args.phase=='run':run()
    else:analyze()

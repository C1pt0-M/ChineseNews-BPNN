"""Independently verify F2/F3-A validation predictions and saved feature pipelines."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import model_config as C
from models.bp import build_model
from models.data import load_splits, load_training_override
from models.f_features import TitleFeatureBuilder
from models.trainer import predict


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    output = HERE / 'f3_a_verification.json'
    if output.exists():
        raise FileExistsError(output)
    ds = HERE / 'f3_a_dataset'
    manifest = read_json(ds/'manifest.json')
    splits = load_splits(verify=True, splits=('train','val'))
    derived = load_training_override(ds/'train.csv', manifest['train_sha256'], splits['train'])
    assert len(derived[0]) == 7531 and len(splits['train'][0]) == 6549
    new = pd.read_csv(ds/'new_sample_evidence.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    evidence = pd.read_csv(HERE/'f3_official_evidence_all.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    selected = new.merge(evidence[['id','grade','archive_source','matching_fields']], on='id', suffixes=('_selection','_official'), validate='one_to_one')
    assert len(selected) == 982 and selected['grade_official'].isin(['A','B']).all()
    assert selected.archive_source_official.eq('official_page_retrieved').all() and selected.matching_fields_official.ne('').all()
    assert selected.semantic_review.eq('title_review_pass').all() and selected.possible_event_conflict.eq('False').all()
    assert sha256((ds/'new_sample_evidence.csv').read_bytes()).hexdigest() == manifest['new_sample_evidence_sha256']
    assert sha256((HERE/'f3_official_evidence_all.csv').read_bytes()).hexdigest() == manifest['official_evidence_sha256']
    expected_added = {'财经':166,'体育':166,'文娱':170,'教育':160,'社会':159,'国际':161}
    assert dict(Counter(new.category)) == expected_added
    val_text, y_val, val_df = splits['val']
    refs = {
        'F2': (ROOT/'results/member_C/f2_formal_word_char/F_word_char_s42', splits['train']),
        'F3-A': (ROOT/'results/member_C/f3_formal_a/F_word_char_s42', derived),
    }
    results, class_rows = {}, []
    for name, (path, training) in refs.items():
        d = read_json(path/'metrics.json')
        arch, tp = d['model'], d['training']['params']
        assert d['seed'] == 42 and arch.get('model_type','bp') == 'bp' and arch['hidden_dim'] == 64
        assert arch['dropout'] == .2 and arch.get('input_dropout',0) == 0 and tp['lr'] == .0003
        assert tp['patience'] == 10 and tp['min_delta'] == .0001 and d['test'] is None
        assert d['feature']['fit_rows'] == len(training[0])
        expected = sha256((ds/'train.csv' if name == 'F3-A' else ROOT/'dataset/splits/train.csv').read_bytes()).hexdigest()
        assert d['provenance']['split_sha256']['train'] == expected
        assert d['provenance']['split_sha256']['val'] == sha256((ROOT/'dataset/splits/val.csv').read_bytes()).hexdigest()
        with (path/'feature_builder.pkl').open('rb') as f:
            builder = pickle.load(f)
        reconstructed = TitleFeatureBuilder('F_word_char', seed=42).fit(training[0], training[1])
        assert builder.char_vectorizer.vocabulary_ == reconstructed.char_vectorizer.vocabulary_
        np.testing.assert_allclose(builder.char_vectorizer.idf_, reconstructed.char_vectorizer.idf_, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(builder.char_selector.get_support(), reconstructed.char_selector.get_support())
        assert builder.word.vectorizer.vocabulary_ == reconstructed.word.vectorizer.vocabulary_
        np.testing.assert_allclose(builder.word.vectorizer.idf_, reconstructed.word.vectorizer.idf_, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(builder.word.selector.get_support(), reconstructed.word.selector.get_support())
        x = builder.transform(val_text)
        model = build_model(arch)
        model.load_state_dict(torch.load(path/'model.pt', map_location='cpu', weights_only=True), strict=True)
        pred = predict(model, x)
        details = pd.read_csv(path/'val_predictions.csv', encoding='utf-8-sig')
        assert details.id.astype(str).tolist() == val_df.id.astype(str).tolist()
        assert np.array_equal(pred, details.predicted_category.map(C.LABEL_TO_INDEX).to_numpy())
        f = f1_score(y_val,pred,labels=np.arange(6),average='macro',zero_division=0)
        acc = accuracy_score(y_val,pred)
        assert abs(f-d['val']['f1_macro'])<1e-12 and abs(acc-d['val']['accuracy'])<1e-12
        cm = confusion_matrix(y_val,pred,labels=np.arange(6))
        assert cm.tolist() == d['val']['confusion_matrix']
        hist = pd.read_csv(path/'history.csv')
        best_score, best_epoch = -1., 0
        for epoch, score in zip(hist.epoch, hist.val_f1_macro):
            if score > best_score + tp['min_delta']:
                best_score, best_epoch = score, int(epoch)
        assert best_epoch == d['training']['best_epoch'] and abs(best_score-f)<1e-12
        assert len(hist) == d['training']['epochs_run'] and (len(hist)-best_epoch==tp['patience'] or len(hist)==tp['max_epochs'])
        p, r, fclass, support = precision_recall_fscore_support(y_val,pred,labels=np.arange(6),zero_division=0)
        for i, label in enumerate(C.LABELS):
            assert abs(fclass[i]-d['val']['per_class'][label]['f1']) < 1e-12
            class_rows.append(dict(model=name,category=label,precision=float(p[i]),recall=float(r[i]),f1=float(fclass[i]),support=int(support[i])))
        results[name] = {'train_rows':len(training[0]),'macro_f1':f,'accuracy':acc,'best_epoch':best_epoch,
                         'epochs_run':len(hist),'train_time_seconds':d['training']['train_time'],
                         'parameters':arch['n_parameters'],'input_dim':arch['input_dim'],
                         'society_to_finance':int(cm[4,0]),'society_to_entertainment':int(cm[4,2]),
                         'loss_first':{'train_eval':float(hist.train_eval_loss.iloc[0]),'val':float(hist.val_loss.iloc[0])},
                         'loss_best':{'train_eval':float(hist.train_eval_loss.iloc[best_epoch-1]),'val':float(hist.val_loss.iloc[best_epoch-1])},
                         'loss_last':{'train_eval':float(hist.train_eval_loss.iloc[-1]),'val':float(hist.val_loss.iloc[-1])}}
    results['delta_macro_f1'] = results['F3-A']['macro_f1']-results['F2']['macro_f1']
    output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(class_rows).to_csv(HERE/'f3_a_per_class.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

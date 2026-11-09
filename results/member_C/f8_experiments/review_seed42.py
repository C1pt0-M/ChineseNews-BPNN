"""Pre-multiseed seed-42 audit: no training, fitting, or test evaluation."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch

import pipeline as p
from models.bp import build_model
from models.trainer import predict

def main():
    target = p.HERE/'seed42_review.json'
    if target.exists():
        raise FileExistsError('Refusing to overwrite existing seed-42 review')
    splits, base = p.splits_and_baseline()
    vectors, manifest = p.load_vectors(splits)
    y = splits['val'][1]
    items = {}
    for group in ('F8_1','F8_2'):
        folder = p.HERE/'runs'/f'{group}_s42'
        m = json.loads((folder/'metrics.json').read_text(encoding='utf8'))
        Xtr, Xval, feature = p.make_features(group, splits, vectors, base)
        assert feature == m['feature'] and m['training']['params'] == base['training']['params']
        assert m['data_sha256'] == {'train':p.TRAIN_SHA,'val':p.VAL_SHA}
        assert m['semantic_cache_manifest_sha256'] == p.digest(p.CACHE/'manifest.json')
        assert m['model']['n_parameters'] == p.bp_parameter_count(Xtr.shape[1])
        model = build_model(m['model'])
        model.load_state_dict(torch.load(folder/'model.pt', map_location='cpu', weights_only=True), strict=True)
        pred = predict(model,Xval)
        import pandas as pd
        saved = pd.read_csv(folder/'val_predictions.csv', dtype=str, encoding='utf-8-sig')
        assert saved.id.tolist() == splits['val'][2].id.astype(str).tolist()
        assert np.array_equal(pred, saved.predicted_category.map(p.C.LABEL_TO_INDEX).to_numpy())
        f1 = f1_score(y,pred,average='macro')
        acc = accuracy_score(y,pred)
        assert abs(f1-m['val']['f1_macro'])<1e-12 and abs(acc-m['val']['accuracy'])<1e-12
        hist=pd.read_csv(folder/'history.csv')
        best, epoch = -1., 0
        for e, v in zip(hist.epoch,hist.val_f1_macro):
            if v > best + m['training']['params']['min_delta']:
                best, epoch = v, int(e)
        assert epoch == m['training']['best_epoch'] and abs(best-f1)<1e-12
        assert len(hist)==m['training']['epochs_run'] and len(hist)-epoch==m['training']['params']['patience']
        assert np.isfinite(hist.select_dtypes(include=[np.number]).to_numpy()).all()
        cm=confusion_matrix(y,pred,labels=np.arange(6))
        assert cm.tolist()==m['val']['confusion_matrix']
        bestrow=hist.loc[hist.epoch==epoch].iloc[0]
        items[group]={'macro_f1':f1,'accuracy':acc,'best_epoch':epoch,'epochs_run':len(hist),
            'input_dim':Xtr.shape[1], 'parameters':m['model']['n_parameters'],
            'best_train_loss':float(bestrow.train_eval_loss), 'best_val_loss':float(bestrow.val_loss),
            'final_train_loss':float(hist.iloc[-1].train_eval_loss), 'final_val_loss':float(hist.iloc[-1].val_loss),
            'society_to_finance':int(cm[4,0]), 'society_to_entertainment':int(cm[4,2])}
    base_f1=base['val']['f1_macro']
    more = all(v['macro_f1'] > base_f1 for v in items.values())
    report={'baseline_f1':base_f1, 'runs':items, 'cache_revision':manifest['revision'],
        'strict_checkpoint_predictions_match':True,'finite_history':True,'further_validation_justified':more,
        'decision':'Both semantic and fused exploratory Macro-F1 exceed seed-42 baseline; proceed with seeds 43–46 only if all integrity checks pass. This selection uses the validation set and is not independent evidence.',
        'test_access':False}
    p.save_json(target,report)
    print(json.dumps(report,ensure_ascii=False,indent=2), flush=True)
    if not more: raise RuntimeError('Seed-42 did not justify the preset follow-up gate')

if __name__ == '__main__': main()

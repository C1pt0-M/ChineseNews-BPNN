"""Verify saved final predictions and metrics without rerunning model inference."""
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'results/member_C/f8_experiments'))
import pipeline as p
import evaluate_fixed as evaluation

def main():
    target=HERE/'verification.json'
    if target.exists():raise FileExistsError(f'Refusing to overwrite {target}')
    plan=json.loads((HERE/'preflight.json').read_text(encoding='utf8'))
    audit=json.loads((HERE/'evaluation.json').read_text(encoding='utf8'))
    evaluation.validate_locked(plan)
    assert p.digest(HERE/'preflight.json')==audit['selection_preflight_sha256']
    assert p.digest(HERE/'test_embeddings.npz')==audit['test_embeddings_sha256']
    test=pd.read_csv(evaluation.TEST,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    saved=pd.read_csv(HERE/'predictions.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    for c in ('id','title','category','publish_time'): assert saved[c].tolist()==test[c].tolist()
    y=test.category.map(p.C.LABEL_TO_INDEX).to_numpy(dtype=int)
    with np.load(HERE/'test_embeddings.npz',allow_pickle=False) as z:
        assert np.array_equal(z['ids'],test.id.to_numpy(dtype=str))
        assert np.array_equal(z['titles'],test.title.to_numpy(dtype=str))
        assert np.array_equal(z['labels'],y)
        assert z['vectors'].shape==(2267,512)
        assert np.isfinite(z['vectors']).all() and np.allclose(np.linalg.norm(z['vectors'],axis=1),1,atol=1e-5)
    runs=pd.read_csv(HERE/'per_seed.csv')
    classes=pd.read_csv(HERE/'per_class.csv')
    confusions=pd.read_csv(HERE/'confusion_matrices.csv')
    paired=pd.read_csv(HERE/'paired.csv')
    summary=pd.read_csv(HERE/'summary.csv').set_index('group')
    classes_summary=pd.read_csv(HERE/'per_class_summary.csv')
    validation=pd.read_csv(HERE/'validation_comparison.csv')
    computed={}
    for group in evaluation.GROUPS:
        computed[group]={}
        for seed in evaluation.SEEDS:
            pred=saved[f'{group}_pred_{seed}'].map(p.C.LABEL_TO_INDEX).to_numpy(dtype=int)
            macro=f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)
            acc=accuracy_score(y,pred)
            precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=np.arange(6),zero_division=0)
            cm=confusion_matrix(y,pred,labels=np.arange(6))
            record=runs[(runs.group==group)&(runs.seed==seed)]
            assert len(record)==1
            record=record.iloc[0]
            for key,value in [('accuracy',acc),('macro_f1',macro),('macro_precision',precision.mean()),('macro_recall',recall.mean())]:
                assert abs(record[key]-value)<1e-12
            for i,cat in enumerate(p.C.LABELS):
                row=classes[(classes.group==group)&(classes.seed==seed)&(classes.category==cat)]
                assert len(row)==1
                row=row.iloc[0]
                for key,value in [('precision',precision[i]),('recall',recall[i]),('f1',f1[i]),('support',support[i])]:
                    assert abs(row[key]-value)<1e-12
                for j,dest in enumerate(p.C.LABELS):
                    cell=confusions[(confusions.group==group)&(confusions.seed==seed)&
                                     (confusions.true_category==cat)&(confusions.predicted_category==dest)]
                    assert len(cell)==1 and int(cell.iloc[0]['count'])==int(cm[i,j])
            vr=validation[(validation.group==group)&(validation.seed==seed)]
            assert len(vr)==1
            vr=vr.iloc[0]
            assert abs(vr.validation_macro_f1-plan['locked_runs'][f'{group} seed={seed}']['validation_macro_f1'])<1e-12
            assert abs(vr.test_minus_val_f1-(macro-vr.validation_macro_f1))<1e-12
            assert (HERE/'plots'/f'{group}_s{seed}.png').is_file()
            computed[group][seed]={'macro_f1':macro,'accuracy':acc,
                'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
                'society_to_finance':int(cm[4,0]),'society_to_entertainment':int(cm[4,2])}
        for metric,col in [('macro_f1','macro_f1'),('accuracy','accuracy'),
                           ('macro_precision','precision_macro'),('macro_recall','recall_macro')]:
            vals=[computed[group][s][metric] for s in evaluation.SEEDS]
            assert abs(summary.loc[group,f'{col}_mean']-statistics.mean(vals))<1e-12
            assert abs(summary.loc[group,f'{col}_std']-statistics.stdev(vals))<1e-12
        for cat in p.C.LABELS:
            vals=classes[(classes.group==group)&(classes.category==cat)].f1.tolist()
            row=classes_summary[(classes_summary.group==group)&(classes_summary.category==cat)]
            assert len(row)==1 and abs(row.iloc[0].f1_mean-statistics.mean(vals))<1e-12
            assert abs(row.iloc[0].f1_std-statistics.stdev(vals))<1e-12
    for seed in evaluation.SEEDS:
        row=paired[paired.seed==seed]
        assert len(row)==1
        row=row.iloc[0]
        for metric in ('macro_f1','accuracy','macro_precision','macro_recall','society_to_finance','society_to_entertainment'):
            for a,b in [('F8-1','F8-0'),('F8-2','F8-0'),('F8-2','F8-1')]:
                assert abs(row[f'{a}_minus_{b}_{metric}']-(computed[a][seed][metric]-computed[b][seed][metric]))<1e-12
    evaluation.validate_locked(plan)
    files=['predictions.csv','per_seed.csv','summary.csv','per_class.csv','per_class_summary.csv',
           'confusion_matrices.csv','paired.csv','validation_comparison.csv','evaluation.json','summary.md','test_embeddings.npz']
    result={'test_predictions_recomputed_from_checkpoint':False,
        'verified_from_saved_predictions_only':True,'rows':len(test),'models':15,
        'test_data_hash':p.digest(evaluation.TEST),'preflight_hash':p.digest(HERE/'preflight.json'),
        'all_checkpoint_and_feature_hashes_unchanged':True,'saved_test_ids_titles_labels_and_embedding_order_match':True,
        'all_metrics_classes_confusions_and_paired_statistics_match':True,'all_15_confusion_plots_exist':True,
        'output_sha256':{f:p.digest(HERE/f) for f in files}}
    p.save_json(target,result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

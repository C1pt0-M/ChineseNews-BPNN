"""Recompute F2/F3-A seed-paired validation metrics and trace recurrent errors."""
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
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
import model_config as C
from models.bp import build_model
from models.data import load_splits, load_training_override
from models.f_features import TitleFeatureBuilder
from models.trainer import predict

SEEDS=(42,43,44,45,46)
LABELS=list(C.LABELS)
F2=ROOT/'results/member_C/f2_formal_word_char'
F3_42=ROOT/'results/member_C/f3_formal_a_reviewed/F_word_char_s42'
F3_REST=HERE/'runs'
DATA=ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal'


def transitions(true, old, new):
    """Per-seed disjoint correctness transitions, with no outcome reweighting."""
    a=np.asarray(old)==np.asarray(true)
    b=np.asarray(new)==np.asarray(true)
    return {'both_correct':int((a & b).sum()),'corrected':int((~a & b).sum()),
            'introduced':int((a & ~b).sum()),'both_wrong':int((~a & ~b).sum())}


def check_builder(a,b):
    for x,y in ((a.word.vectorizer,b.word.vectorizer),(a.char_vectorizer,b.char_vectorizer)):
        assert x.vocabulary_==y.vocabulary_
        np.testing.assert_allclose(x.idf_,y.idf_,rtol=0,atol=1e-14)
    for x,y in ((a.word.selector,b.word.selector),(a.char_selector,b.char_selector)):
        np.testing.assert_array_equal(x.get_support(),y.get_support())
        np.testing.assert_allclose(x.scores_,y.scores_,rtol=1e-12,atol=1e-12,equal_nan=True)


def path_for(group,seed):
    return F2/f'F_word_char_s{seed}' if group=='F2' else (F3_42 if seed==42 else F3_REST/f'F_word_char_s{seed}')


def main():
    targets=['summary.csv','paired_deltas.csv','per_class.csv','per_class_summary.csv',
             'confusion_by_seed.csv','prediction_comparison.csv','transitions_by_seed.csv','audit.json']
    for file in targets:
        if (HERE/file).exists():raise FileExistsError(HERE/file)
    torch.set_num_threads(4)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    train_hash=sha256((DATA/'train.csv').read_bytes()).hexdigest()
    assert train_hash==manifest['train_sha256']=='7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
    splits=load_splits(verify=True,splits=('train','val'))
    derived=load_training_override(DATA/'train.csv',train_hash,splits['train'])
    val_text,y,val_frame=splits['val']
    val_hash=sha256((C.SPLIT_DIR/'val.csv').read_bytes()).hexdigest()
    assert len(y)==3184 and len(derived[0])==7531 and len(splits['train'][0])==6549
    runs=[];classes=[];confusions=[];predictions={};baseline_params=None;baseline_feature=None
    for group,training in (('F2',splits['train']),('F3-A',derived)):
        # Independent training-only refit catches accidental reuse or leakage.
        fresh=TitleFeatureBuilder('F_word_char',seed=42).fit(training[0],training[1])
        for seed in SEEDS:
            p=path_for(group,seed)
            required=['metrics.json','history.csv','model.pt','feature_builder.pkl','val_predictions.csv']
            assert all((p/x).is_file() for x in required),(group,seed,'missing artifact')
            m=json.loads((p/'metrics.json').read_text(encoding='utf-8'))
            arch=m['model'];params=m['training']['params'];feat=m['feature']
            expected_hash=sha256((C.SPLIT_DIR/'train.csv').read_bytes()).hexdigest() if group=='F2' else train_hash
            assert m['seed']==seed and m['test'] is None and m['provenance']['split_sha256']=={'train':expected_hash,'val':val_hash}
            assert arch.get('model_type','bp')=='bp' and arch['hidden_dim']==64 and arch['dropout']==.2
            assert arch.get('input_dropout',0)==0 and arch['input_dim']==10000 and arch['output_dim']==6 and arch['n_parameters']==640454
            assert params['lr']==.0003 and params['patience']==10 and params['min_delta']==.0001
            assert feat['scheme']=='F_word_char' and feat['fit_rows']==len(training[0]) and feat['sparse'] and feat['input_source']=='title'
            assert feat['word']['selected_features']==feat['char']['selected_features']==5000
            comparable=(feat['word']['tfidf_params'],feat['char']['tfidf_params'],feat['normalization'])
            if baseline_params is None: baseline_params=params;baseline_feature=comparable
            assert params==baseline_params and comparable==baseline_feature
            with (p/'feature_builder.pkl').open('rb') as f:builder=pickle.load(f)
            check_builder(builder,fresh)
            X=builder.transform(val_text)
            model=build_model(arch)
            model.load_state_dict(torch.load(p/'model.pt',map_location='cpu',weights_only=True),strict=True)
            pred=predict(model,X)
            saved=pd.read_csv(p/'val_predictions.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
            assert saved.id.tolist()==val_frame.id.astype(str).tolist()
            assert saved.title.tolist()==val_frame.title.astype(str).tolist()
            assert saved.category.tolist()==val_frame.category.astype(str).tolist()
            assert np.array_equal(pred,saved.predicted_category.map(C.LABEL_TO_INDEX).to_numpy())
            f1=f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)
            acc=accuracy_score(y,pred);cm=confusion_matrix(y,pred,labels=np.arange(6))
            assert abs(f1-m['val']['f1_macro'])<1e-12 and abs(acc-m['val']['accuracy'])<1e-12
            assert cm.tolist()==m['val']['confusion_matrix']
            hist=pd.read_csv(p/'history.csv');best_score=-1.;best_epoch=0
            for epoch,score in zip(hist.epoch,hist.val_f1_macro):
                if score>best_score+params['min_delta']: best_score,best_epoch=score,int(epoch)
            assert best_epoch==m['training']['best_epoch'] and abs(best_score-f1)<1e-12
            assert len(hist)==m['training']['epochs_run']
            stopped=len(hist)-best_epoch==params['patience']
            assert stopped or len(hist)==params['max_epochs']
            pvals,rvals,fvals,support=precision_recall_fscore_support(y,pred,labels=np.arange(6),zero_division=0)
            for i,cat in enumerate(LABELS):
                saved_class=m['val']['per_class'][cat]
                assert abs(saved_class['precision']-pvals[i])<1e-12
                assert abs(saved_class['recall']-rvals[i])<1e-12
                assert abs(saved_class['f1']-fvals[i])<1e-12
                classes.append(dict(group=group,seed=seed,category=cat,precision=pvals[i],recall=rvals[i],f1=fvals[i],support=int(support[i])))
                for j,cat2 in enumerate(LABELS):
                    confusions.append(dict(group=group,seed=seed,true_category=cat,predicted_category=cat2,count=int(cm[i,j])))
            predictions[(group,seed)]=pred
            runs.append(dict(group=group,seed=seed,train_rows=len(training[0]),val_rows=len(y),macro_f1=f1,accuracy=acc,
                             best_epoch=best_epoch,epochs_run=len(hist),early_stopped=stopped,early_stop_epoch=len(hist) if stopped else None,
                             n_parameters=arch['n_parameters'],input_dim=arch['input_dim'],train_time_seconds=m['training']['train_time'],
                             society_to_finance=int(cm[4,0]),society_to_entertainment=int(cm[4,2]),run_path=str(p.relative_to(ROOT))))
            print('verified',group,seed,round(f1,6),flush=True)
    run=pd.DataFrame(runs);per=pd.DataFrame(classes);cf=pd.DataFrame(confusions)
    paired=run[run.group=='F2'].merge(run[run.group=='F3-A'],on='seed',suffixes=('_F2','_F3'),validate='one_to_one')
    paired['delta_macro_f1']=paired.macro_f1_F3-paired.macro_f1_F2
    paired['delta_accuracy']=paired.accuracy_F3-paired.accuracy_F2
    transition=[]
    for seed in SEEDS:
        transition.append({'seed':seed,**transitions(y,predictions[('F2',seed)],predictions[('F3-A',seed)])})
    frame=val_frame[['id','title','category','publish_time','url']].copy()
    for group,prefix in (('F2','f2'),('F3-A','f3')):
        for seed in SEEDS:frame[f'{prefix}_pred_{seed}']=[LABELS[i] for i in predictions[(group,seed)]]
        cols=[f'{prefix}_pred_{seed}' for seed in SEEDS]
        frame[f'{prefix}_error_count']=(frame[cols].to_numpy()!=frame.category.to_numpy()[:,None]).sum(axis=1)
        frame[f'{prefix}_prediction_counts']=[json.dumps(dict(Counter(row)),ensure_ascii=False,sort_keys=True) for row in frame[cols].to_numpy()]
    frame['f2_always_wrong']=frame.f2_error_count.eq(5);frame['f3_always_wrong']=frame.f3_error_count.eq(5)
    frame['always_corrected']=frame.f2_error_count.eq(5)&frame.f3_error_count.eq(0)
    frame['always_introduced']=frame.f2_error_count.eq(0)&frame.f3_error_count.eq(5)
    assert len(frame)==len(y) and frame.id.is_unique
    agg=per.groupby(['group','category'],sort=False).agg(
        precision_mean=('precision','mean'),precision_std=('precision','std'),recall_mean=('recall','mean'),
        recall_std=('recall','std'),f1_mean=('f1','mean'),f1_std=('f1','std'),support=('support','first')).reset_index()
    pvt=agg.pivot(index='category',columns='group',values='f1_mean')
    assert abs(float((pvt['F3-A']-pvt['F2']).mean())-float(paired.delta_macro_f1.mean()))<1e-12
    audit={'train_sha256':train_hash,'val_sha256':val_hash,'base_train_sha256':sha256((C.SPLIT_DIR/'train.csv').read_bytes()).hexdigest(),
           'seeds':list(SEEDS),'reused_f3_seed42':str(F3_42.relative_to(ROOT)),'heldout_test_metrics_used':False,
           'all_checkpoint_predictions_recomputed':True,'strict_model_loads':10,'independent_feature_refits':2,
           'features_match_refits_all_seeds':True,'saved_metrics_and_early_stopping_verified':True,
           'f2_always_wrong':int(frame.f2_always_wrong.sum()),'f3_always_wrong':int(frame.f3_always_wrong.sum()),
           'both_always_wrong':int((frame.f2_always_wrong&frame.f3_always_wrong).sum()),
           'always_corrected':int(frame.always_corrected.sum()),'always_introduced':int(frame.always_introduced.sum())}
    for name,df in [('summary.csv',run),('paired_deltas.csv',paired),('per_class.csv',per),('per_class_summary.csv',agg),
                    ('confusion_by_seed.csv',cf),('prediction_comparison.csv',frame),('transitions_by_seed.csv',pd.DataFrame(transition))]:
        df.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    (HERE/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()

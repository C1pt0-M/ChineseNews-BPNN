"""Freeze data selection, checkpoint inventory, and hashes before final evaluation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
F8=ROOT/'results/member_C/f8_experiments'
sys.path.insert(0,str(F8))
import pipeline as p
from models.bp import build_model

TEST=ROOT/'dataset/splits/test.csv'
TEST_SHA='626cc1cd09bbdd74b7a49bb4570c2efde3c59089e31ace918665352fda694509'
SEEDS=(42,43,44,45,46)

def main():
    output=HERE/'preflight.json'
    if output.exists(): raise FileExistsError(f'Refusing to overwrite {output}')
    splits,base=p.splits_and_baseline()
    assert p.digest(TEST)==TEST_SHA
    v=json.loads((p.C.DATASET_VERSION_FILE).read_text(encoding='utf8'))
    assert v['splits']['test']['sha256']==TEST_SHA and v['splits']['test']['rows']==2267
    frame={name:splits[name][2].fillna('').astype(str) for name in ('train','val')}
    frame['test']=pd.read_csv(TEST,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    assert len(frame['test'])==2267
    assert set(frame['test'].category)==set(p.C.LABELS)
    temporal={}
    for n,df in frame.items():
        for col in ('id','url','title'):
            assert not df[col].eq('').any() and df[col].is_unique, f'{n}: duplicate/missing {col}'
        dates=pd.to_datetime(df.publish_time,errors='coerce')
        assert not dates.isna().any()
        temporal[n]={'minimum':str(dates.min()),'maximum':str(dates.max()),
                     'class_counts':df.category.value_counts().to_dict(),'rows':len(df)}
    assert pd.to_datetime(frame['test'].publish_time).between('2026-07-01','2026-10-31 23:59:59').all()
    intersections={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        intersections[f'{a}_{b}']={k:len(set(frame[a][k]) & set(frame[b][k])) for k in ('id','url','title')}
        assert not any(intersections[f'{a}_{b}'].values())
    seven=pd.read_csv(ROOT/'dataset/test_7days/news_clean.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    assert len(seven)==1854
    seven_counts={'rows':len(seven),'class_counts':seven.category.value_counts().to_dict(),
        'id_overlap_with_f3a_train':len(set(seven.id)&set(frame['train'].id)),
        'title_overlap_with_f3a_train':len(set(seven.title)&set(frame['train'].title)),
        'unmatched_class_counts':seven[~seven.id.isin(frame['train'].id)].category.value_counts().to_dict(),
        'period':'2025-01-01 through 2025-01-07',
        'decision':'Not a documented blind independent set: overlaps F3-A training, restricted to seven days, severe class imbalance, previously used in F3 resource review.'}
    assert seven_counts['id_overlap_with_f3a_train']==123
    semantic=json.loads((p.CACHE/'manifest.json').read_text(encoding='utf8'))
    assert semantic['revision']==p.REVISION and semantic['encoder_trainable_parameters']==0
    for name,sha in semantic['files_sha256'].items(): assert p.digest(p.MODEL/name)==sha
    assert p.digest(p.MODEL/'model.safetensors')==p.WEIGHTS_SHA
    prev=json.loads((F8/'audit.json').read_text(encoding='utf8'))
    tfidf_audit=json.loads((ROOT/'results/member_C/f6_experiments/f6_b_multiseed/audit.json').read_text(encoding='utf8'))
    baseline=pd.read_csv(ROOT/'results/member_C/f6_experiments/f6_b_multiseed/per_seed.csv',dtype=str)
    baseline=baseline[baseline.group=='F6-B'].set_index('seed')
    assert set(baseline.index)=={str(s) for s in SEEDS}
    locked={}
    for group,dimension in [('F8-0',10000),('F8-1',512),('F8-2',10512)]:
        for seed in SEEDS:
            folder=ROOT/baseline.loc[str(seed),'run_path'] if group=='F8-0' else F8/'runs'/f"{group.replace('-','_')}_s{seed}"
            m=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
            previous=prev['artifact_sha256'][f'{group} seed={seed}']
            for name in ('metrics.json','model.pt','history.csv','val_predictions.csv'):
                assert p.digest(folder/name)==previous[name], f'Changed historical {folder/name}'
            arch=m['model']
            assert arch['model_type']=='bp' and arch['input_dim']==dimension and arch['hidden_dim']==64
            assert arch['dropout']==.2 and arch['input_dropout']==0 and arch['output_dim']==6
            assert arch['n_parameters']==p.bp_parameter_count(dimension)
            assert m['training']['params']==base['training']['params'] and m['seed']==seed
            source_sha=m['provenance']['split_sha256'] if group=='F8-0' else m['data_sha256']
            assert source_sha=={'train':p.TRAIN_SHA,'val':p.VAL_SHA}
            if group!='F8-0':
                assert m['semantic_cache_manifest_sha256']==p.digest(p.CACHE/'manifest.json')
                assert m['frozen_encoder_parameters']==semantic['frozen_encoder_parameters']
            model=build_model(arch)
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
            assert model.count_parameters()==arch['n_parameters']
            if group=='F8-0':
                processor=folder/'feature_builder.pkl'
                expected=tfidf_audit['artifact_sha256'][f'F6-B seed={seed}']['feature_builder.pkl']
                assert p.digest(processor)==expected
            else: processor=p.BASE/'feature_builder.pkl' if group=='F8-2' else None
            if group=='F8-2': assert p.digest(processor)==m['feature']['tfidf_processor_sha256']
            locked[f'{group} seed={seed}']={'path':str(folder.relative_to(ROOT)),
                'checkpoint_sha256':p.digest(folder/'model.pt'),'metrics_sha256':p.digest(folder/'metrics.json'),
                'feature_processor':str(processor.relative_to(ROOT)) if processor else None,
                'feature_processor_sha256':p.digest(processor) if processor else None,
                'input_dim':dimension,'bp_parameters':arch['n_parameters'],
                'validation_macro_f1':m['val']['f1_macro']}
    plan={'selection_fixed_before_test_prediction':True,
        'selection_basis':'User fixed all three schemes and all five seeds before evaluation; no test-driven selection.',
        'selected_data':'dataset/splits/test.csv','data_sha256':{'train':p.TRAIN_SHA,'val':p.VAL_SHA,'test':TEST_SHA},
        'test_status':'Previously evaluated in historical experiments; not a strict independent blind test.',
        'test_history_sources':['README.md:423','reports/member_A_baseline_audit.md:110'],
        'alternative_review':seven_counts,'temporal_and_class_distribution':temporal,
        'exact_cross_split_intersections':intersections,
        'near_duplicate_limit':'Exact ID/URL/title checks do not exclude paraphrases or multiple reports of the same event.',
        'encoder_pretraining_overlap':'Unknown whether external BGE pretraining data overlap evaluation titles.',
        'encoder_revision':semantic['revision'],'encoder_manifest_sha256':p.digest(p.CACHE/'manifest.json'),
        'frozen_encoder_parameters':semantic['frozen_encoder_parameters'],
        'locked_runs':locked,'test_model_predictions_generated_in_preflight':False}
    p.save_json(output,plan)
    print(json.dumps({'selected_data':plan['selected_data'],'test_status':plan['test_status'],
        'candidate_test_7days':seven_counts,'rows':{k:len(v) for k,v in frame.items()},
        'intersections':intersections,'locked_checkpoints':len(locked)},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

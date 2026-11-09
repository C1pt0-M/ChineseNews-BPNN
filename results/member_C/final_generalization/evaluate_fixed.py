"""Evaluate all 15 pre-locked checkpoints once on the previously viewed test set."""
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'results/member_C/f8_experiments'))
import pipeline as p
from models.bp import build_model
from models.plots import plot_confusion_matrix
from models.trainer import predict

TEST=ROOT/'dataset/splits/test.csv'
SEEDS=(42,43,44,45,46)
GROUPS=('F8-0','F8-1','F8-2')
OUTPUTS=('test_embeddings.npz','predictions.csv','per_seed.csv','summary.csv','per_class.csv',
         'per_class_summary.csv','confusion_matrices.csv','paired.csv','validation_comparison.csv',
         'evaluation.json','summary.md')

def table(df,cols):
    lines=['| '+' | '.join(cols)+' |','| '+' | '.join('---' for _ in cols)+' |']
    lines += ['| '+' | '.join(str(row[c]) for c in cols)+' |' for _,row in df.iterrows()]
    return '\n'.join(lines)

def validate_locked(plan):
    assert plan['selection_fixed_before_test_prediction'] and not plan['test_model_predictions_generated_in_preflight']
    assert plan['selected_data']=='dataset/splits/test.csv'
    for path,key in ((p.TRAIN,'train'),(p.VAL,'val'),(TEST,'test')):
        assert p.digest(path)==plan['data_sha256'][key],f'Changed dataset {key}'
    manifest=json.loads((p.CACHE/'manifest.json').read_text(encoding='utf8'))
    assert p.digest(p.CACHE/'manifest.json')==plan['encoder_manifest_sha256']
    assert manifest['revision']==p.REVISION==plan['encoder_revision']
    for name,h in manifest['files_sha256'].items(): assert p.digest(p.MODEL/name)==h
    for k,info in plan['locked_runs'].items():
        folder=ROOT/info['path']
        assert p.digest(folder/'model.pt')==info['checkpoint_sha256'],f'Changed checkpoint {k}'
        assert p.digest(folder/'metrics.json')==info['metrics_sha256'],f'Changed metrics {k}'
        if info['feature_processor']:
            assert p.digest(ROOT/info['feature_processor'])==info['feature_processor_sha256']
    return manifest

def encode_test(texts,manifest):
    import transformers,tokenizers,huggingface_hub,safetensors
    from transformers import AutoModel,AutoTokenizer
    versions={'torch':torch.__version__,'transformers':transformers.__version__,
              'tokenizers':tokenizers.__version__,'huggingface_hub':huggingface_hub.__version__,
              'safetensors':safetensors.__version__,'numpy':np.__version__}
    assert versions==manifest['versions'], f'Encoder software versions changed: {versions}'
    assert manifest['device']=='cuda' and torch.cuda.is_available(), 'Original CUDA encoder unavailable'
    assert torch.cuda.get_device_name(0)==manifest['device_name'], 'Original GPU model changed'
    device=torch.device('cuda')
    t0=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(p.MODEL,local_files_only=True)
    model,info=AutoModel.from_pretrained(p.MODEL,local_files_only=True,output_loading_info=True)
    assert not info['missing_keys'] and not info['mismatched_keys'] and not info['error_msgs']
    model.requires_grad_(False).eval()
    assert not any(v.requires_grad for v in model.parameters())
    assert sum(v.numel() for v in model.parameters())==manifest['frozen_encoder_parameters']
    assert model.config.hidden_size==512
    model.to(device)
    torch.cuda.synchronize()
    load_seconds=time.perf_counter()-t0
    conf=manifest['tokenizer']
    assert conf=={'padding':True,'truncation':True,'max_length':512,'batch_size':32}
    pieces=[]
    truncated=0
    t0=time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(texts),32):
            batch=texts[start:start+32]
            lengths=tokenizer(batch,truncation=False,return_length=True)['length']
            truncated+=sum(v>512 for v in lengths)
            encoded=tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='pt')
            output=model(**{k:v.to(device) for k,v in encoded.items()}).last_hidden_state[:,0,:]
            pieces.append(torch.nn.functional.normalize(output,p=2,dim=1).cpu().numpy().astype(np.float32,copy=False))
    vectors=np.concatenate(pieces)
    torch.cuda.synchronize()
    seconds=time.perf_counter()-t0
    assert vectors.shape==(len(texts),512) and np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-5)
    del model
    return vectors,{'device':str(device),'device_name':manifest['device_name'],'dtype':'float32',
        'model_load_seconds':load_seconds,'batch_encode_seconds':seconds,'batch_size':32,
        'titles_truncated':truncated,'encoder_frozen_parameters':manifest['frozen_encoder_parameters'],
        'encoder_trainable_parameters':0,'versions':versions}

def main():
    for name in OUTPUTS:
        if (HERE/name).exists(): raise FileExistsError(f'Refusing to overwrite existing evaluation: {name}')
    if (HERE/'plots').exists(): raise FileExistsError('Refusing to overwrite plot directory')
    plan=json.loads((HERE/'preflight.json').read_text(encoding='utf8'))
    manifest=validate_locked(plan)
    df=pd.read_csv(TEST,dtype=str,keep_default_na=False,encoding='utf-8-sig')
    assert len(df)==2267 and df.id.is_unique and set(df.category)==set(p.C.LABELS)
    texts=df.title.tolist()
    y=df.category.map(p.C.LABEL_TO_INDEX).to_numpy(dtype='int64')
    t0=time.perf_counter()
    bge,encoder_cost=encode_test(texts,manifest)
    print(f'Encoded {len(df)} fixed test titles once in {encoder_cost["batch_encode_seconds"]:.2f}s',flush=True)
    tfidf,tfidf_seconds={},{}
    for seed in SEEDS:
        info=plan['locked_runs'][f'F8-0 seed={seed}']
        with (ROOT/info['feature_processor']).open('rb') as f: builder=pickle.load(f)
        assert builder.fit_rows_==7531 and builder.word_dim_==7000 and builder.char_dim_==3000
        t0=time.perf_counter()
        z=builder.transform(texts).tocsr()
        tfidf_seconds[str(seed)]=time.perf_counter()-t0
        assert z.shape==(len(df),10000) and sp.isspmatrix_csr(z)
        norms=np.sqrt(np.asarray(z.multiply(z).sum(axis=1)).ravel())
        assert np.allclose(norms[norms>0],1,atol=1e-5)
        tfidf[seed]=z
    base=tfidf[42]
    for seed in SEEDS:
        difference=(tfidf[seed]-base).tocsr()
        assert difference.nnz==0 or np.max(np.abs(difference.data))<1e-12
    fused=sp.hstack((base.astype(np.float32),sp.csr_matrix(bge)),format='csr',dtype=np.float32)
    assert fused.shape==(len(df),10512)
    torch.set_num_threads(4)
    preds=pd.DataFrame({'id':df.id,'title':df.title,'category':df.category,'publish_time':df.publish_time})
    records,classes,confusions=[],[],[]
    t0=time.perf_counter()
    for group in GROUPS:
        for seed in SEEDS:
            info=plan['locked_runs'][f'{group} seed={seed}']
            folder=ROOT/info['path']
            m=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
            if group=='F8-2':
                assert m['feature']['tfidf_processor_sha256']==p.digest(p.BASE/'feature_builder.pkl')
            X=tfidf[seed] if group=='F8-0' else bge if group=='F8-1' else fused
            assert X.shape==(len(df),info['input_dim'])
            model=build_model(m['model'])
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
            assert model.count_parameters()==info['bp_parameters']
            original={k:v.detach().clone() for k,v in model.state_dict().items()}
            t1=time.perf_counter()
            pred=predict(model,X)
            prediction_seconds=time.perf_counter()-t1
            assert all(torch.equal(original[k],v) for k,v in model.state_dict().items())
            assert len(pred)==len(df)
            preds[f'{group}_pred_{seed}']=[p.C.LABELS[i] for i in pred]
            macro=f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)
            acc=accuracy_score(y,pred)
            prec,rec,fs,support=precision_recall_fscore_support(y,pred,labels=np.arange(6),zero_division=0)
            matrix=confusion_matrix(y,pred,labels=np.arange(6))
            for i,cat in enumerate(p.C.LABELS):
                classes.append(dict(group=group,seed=seed,category=cat,precision=prec[i],recall=rec[i],f1=fs[i],support=int(support[i])))
                for j,dest in enumerate(p.C.LABELS):
                    confusions.append(dict(group=group,seed=seed,true_category=cat,predicted_category=dest,count=int(matrix[i,j])))
            records.append(dict(group=group,seed=seed,accuracy=acc,macro_f1=macro,
                macro_precision=float(np.mean(prec)),macro_recall=float(np.mean(rec)),
                bp_parameters=info['bp_parameters'],encoder_parameters=0 if group=='F8-0' else manifest['frozen_encoder_parameters'],
                validation_macro_f1=m['val']['f1_macro'],validation_accuracy=m['val']['accuracy'],
                test_minus_val_f1=macro-m['val']['f1_macro'],test_minus_val_accuracy=acc-m['val']['accuracy'],
                society_to_finance=int(matrix[4,0]),society_to_entertainment=int(matrix[4,2]),
                finance_to_society=int(matrix[0,4]),entertainment_to_society=int(matrix[2,4]),
                bp_prediction_seconds=prediction_seconds,checkpoint_path=info['path']))
            print(f'{group} seed={seed}: fixed model evaluated on {len(df)} titles',flush=True)
    total_predict_seconds=time.perf_counter()-t0
    # Recheck frozen inputs after inference, before committing any output.
    validate_locked(plan)
    results=pd.DataFrame(records)
    class_df=pd.DataFrame(classes)
    conf_df=pd.DataFrame(confusions)
    agg=results.groupby('group',sort=False).agg(n=('seed','size'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_std=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),
        precision_macro_mean=('macro_precision','mean'),precision_macro_std=('macro_precision','std'),
        recall_macro_mean=('macro_recall','mean'),recall_macro_std=('macro_recall','std'),
        validation_f1_mean=('validation_macro_f1','mean'),test_minus_val_f1_mean=('test_minus_val_f1','mean'),
        bp_parameters=('bp_parameters','first'),encoder_parameters=('encoder_parameters','first'),
        society_to_finance_mean=('society_to_finance','mean'),society_to_entertainment_mean=('society_to_entertainment','mean'),
        finance_to_society_mean=('finance_to_society','mean'),entertainment_to_society_mean=('entertainment_to_society','mean'),
        bp_prediction_seconds_mean=('bp_prediction_seconds','mean')).reset_index()
    class_agg=class_df.groupby(['group','category'],sort=False).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),
        precision_mean=('precision','mean'),recall_mean=('recall','mean'),support=('support','first')).reset_index()
    paired=pd.DataFrame({'seed':SEEDS})
    for metric in ('macro_f1','accuracy','macro_precision','macro_recall','society_to_finance','society_to_entertainment'):
        pivot=results.pivot(index='seed',columns='group',values=metric)
        for g in GROUPS: paired[f'{g}_{metric}']=[pivot.loc[s,g] for s in SEEDS]
        for a,b in [('F8-1','F8-0'),('F8-2','F8-0'),('F8-2','F8-1')]:
            paired[f'{a}_minus_{b}_{metric}']=paired[f'{a}_{metric}']-paired[f'{b}_{metric}']
    validation=results[['group','seed','validation_macro_f1','macro_f1','test_minus_val_f1',
                         'validation_accuracy','accuracy','test_minus_val_accuracy']].copy()
    encoder_cache=HERE/'test_embeddings.npz'
    np.savez_compressed(encoder_cache,vectors=bge,ids=df.id.to_numpy(dtype=str),titles=np.asarray(texts,dtype=str),labels=y)
    for name,obj in [('predictions.csv',preds),('per_seed.csv',results),('summary.csv',agg),
                     ('per_class.csv',class_df),('per_class_summary.csv',class_agg),
                     ('confusion_matrices.csv',conf_df),('paired.csv',paired),('validation_comparison.csv',validation)]:
        obj.to_csv(HERE/name,index=False,encoding='utf-8-sig')
    (HERE/'plots').mkdir()
    for group in GROUPS:
        for seed in SEEDS:
            part=conf_df[(conf_df.group==group)&(conf_df.seed==seed)]
            mat=part['count'].to_numpy().reshape(6,6).tolist()
            plot_confusion_matrix(mat,p.C.LABELS,f'{group} seed={seed} final test',HERE/'plots'/f'{group}_s{seed}.png')
    audit={'data_source':'dataset/splits/test.csv','test_previously_viewed':True,
        'selection_preflight_sha256':p.digest(HERE/'preflight.json'),'test_sha256':plan['data_sha256']['test'],
        'encoder_revision':manifest['revision'],'test_embeddings_sha256':p.digest(encoder_cache),
        'encoder_cost':encoder_cost,'tfidf_transform_seconds_by_seed':tfidf_seconds,
        'bp_prediction_seconds_total':total_predict_seconds,'bp_inference_cpu_threads':4,
        'same_test_rows_labels_order_for_15_runs':True,'strict_model_load':True,
        'model_state_unchanged_after_inference':True,'all_locked_input_hashes_unchanged':True,
        'fit_or_train_performed':False,'test_labels_used_to_change_models_or_settings':False,
        'history_test_set_previously_viewed':True,'external_encoder_pretraining_overlap_unknown':True}
    p.save_json(HERE/'evaluation.json',audit)
    show=agg[['group','macro_f1_mean','macro_f1_std','accuracy_mean','accuracy_std','precision_macro_mean',
              'recall_macro_mean','validation_f1_mean','test_minus_val_f1_mean','bp_parameters','encoder_parameters',
              'society_to_finance_mean','society_to_entertainment_mean']].copy()
    for col in show.columns:
        if col not in ('group','bp_parameters','encoder_parameters','society_to_finance_mean','society_to_entertainment_mean'):
            show[col]=show[col].map(lambda v:f'{v:.5f}')
    per=results[['group','seed','macro_f1','accuracy','macro_precision','macro_recall','validation_macro_f1','test_minus_val_f1']].copy()
    for c in per.columns[2:]:per[c]=per[c].map(lambda v:f'{v:.5f}')
    cp=class_agg.pivot(index='category',columns='group',values='f1_mean').reset_index()
    for g in GROUPS:cp[g]=cp[g].map(lambda v:f'{v:.5f}')
    pair=paired[['seed','F8-1_minus_F8-0_macro_f1','F8-2_minus_F8-0_macro_f1','F8-2_minus_F8-1_macro_f1']].copy()
    for c in pair.columns[1:]: pair[c]=pair[c].map(lambda v:f'{v:+.5f}')
    lines=['# F.8 固定方案最终测试集评估','','## 数据和证据界限','',
        '评估前已固定 F8-0、F8-1、F8-2 和 42–46 五个 checkpoint。现有 `test_7days` 仅覆盖 2025 年 7 天，与正式训练集有 123 条 ID 重合；其余样本教育类仅 2 条，且该资料已在 F.3 数据审查中使用，不能作为经确认的独立盲测。未发现其他符合要求、未参与开发的本地标注集。',
        '因此使用既有 `dataset/splits/test.csv`：2,267 条，2026-07-01 至 2026-10-31；该测试集此前已在其他实验中查看，**不是严格独立盲测**。数据来源为中新网，同一清洗及六类映射，不能代表所有来源或自然类别比例。',
        '训练、验证、测试集的 ID、URL、标题均无精确交集；不排除同事件改写或语义近重复。BGE 外部预训练语料与测试集是否重叠未知。预先锁定记录和各输入哈希见 `preflight.json`。','',
        '## 五种子汇总（样本标准差）','',table(show,list(show.columns)),'',
        '## 逐种子指标及与验证集比较','',table(per,list(per.columns)),'',
        '## 六类别平均 F1','',table(cp,['category']+list(GROUPS)),'',
        '## 同种子测试集 Macro-F1 差值','',table(pair,list(pair.columns)),'',
        '完整 Accuracy、Macro-Precision、Macro-Recall、各类 Precision/Recall/F1、混淆矩阵和社会/财经/文娱双向混淆数见对应 CSV。逐个模型矩阵图位于 `plots/`，逐标题预测见 `predictions.csv`。','',
        '## 模型与计算开销','',
        f'F8-0 的 BP 为 640,454 参数；F8-1 为 33,222 参数；F8-2 为 673,222 参数。语义组另需冻结 BGE 编码器 {manifest["frozen_encoder_parameters"]:,} 参数。',
        f'本次使用 {encoder_cost["device_name"]}、float32，编码器加载 {encoder_cost["model_load_seconds"]:.2f}s，批量编码 {len(df)} 条测试标题 {encoder_cost["batch_encode_seconds"]:.2f}s；截断 {encoder_cost["titles_truncated"]} 条。时间不包括模型下载、训练、NPZ 写入，设备与负载不同无法直接比较。TF-IDF 转换时间和 4 线程 CPU 上各 BP 批量预测时间见 `evaluation.json`、`per_seed.csv`。','',
        '## 可支持结论与限制','',
        '可报告固定三个方案在这份已被历史实验查看的时间划分测试集上的表现和同种子差值，以及验证—测试之间的变化；五种子标准差仅反映初始化差异，不能作为独立样本误差或显著性结论。F8 语义方案包含外部预训练知识，不能解释成纯 TF-IDF 或 BP 结构改进。',
        '15 个 checkpoint 严格加载，输入行及维数一致；在推理前后检查权重和锁定文件哈希。未训练、拟合、改动参数或根据结果改变方案。冻结编码器、词表、选择器和模型路径可从 `preflight.json`、`evaluation.json` 追溯。','',
        '## 实际命令','', '```powershell',"$env:PYTHONUTF8='1'",
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/final_generalization/preflight.py',
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/final_generalization/evaluate_fixed.py',
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/final_generalization/verify_outputs.py',
        '```','所有入口拒绝覆盖同名输出；审计脚本只检查保存的预测，不重新执行模型测试。','']
    (HERE/'summary.md').write_text('\n'.join(lines),encoding='utf8')
    print('Fixed evaluation completed; no model selection performed.',flush=True)

if __name__=='__main__':main()

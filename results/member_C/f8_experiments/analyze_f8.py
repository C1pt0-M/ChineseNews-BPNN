"""Audit frozen BGE/F6-B runs and produce paired validation analyses."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
import torch

import pipeline as p
from models.bp import build_model
from models.trainer import predict

SEEDS = (42,43,44,45,46)
GROUPS = ('F8-0','F8-1','F8-2')
OUTPUTS = ('per_seed.csv','summary.csv','per_class.csv','per_class_summary.csv',
    'confusion_matrices.csv','paired.csv','persistent_summary.csv','persistent_per_seed.csv',
    'per_title.csv','audit.json','summary.md')

def table(df, columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    lines.extend('| '+' | '.join(str(row[c]) for c in columns)+' |' for _,row in df.iterrows())
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=p.HERE)
    args=ap.parse_args()
    out=args.output_dir.resolve()
    out.relative_to(p.ROOT)
    out.mkdir(parents=True,exist_ok=True)
    for name in OUTPUTS:
        if (out/name).exists(): raise FileExistsError(f'Refusing to overwrite {out/name}')
    splits, base = p.splits_and_baseline()
    vectors, manifest = p.load_vectors(splits)
    X1, V1, f1 = p.make_features('F8_1',splits,vectors,base)
    X2, V2, f2 = p.make_features('F8_2',splits,vectors,base)
    X0, V0 = X2[:,:10000].tocsr(), V2[:,:10000].tocsr()
    features = {'F8-0':(X0,V0), 'F8-1':(X1,V1), 'F8-2':(X2,V2)}
    y, frame = splits['val'][1], splits['val'][2]
    baseline=pd.read_csv(p.ROOT/'results/member_C/f6_experiments/f6_b_multiseed/per_seed.csv', dtype=str)
    baseline=baseline[baseline.group=='F6-B'].set_index('seed')
    assert set(baseline.index)=={str(s) for s in SEEDS}
    historic=pd.read_csv(p.ROOT/'results/member_C/f3_analysis/f3_a_multiseed/prediction_comparison.csv', dtype=str, encoding='utf-8-sig')
    assert historic.id.tolist()==frame.id.astype(str).tolist()
    assert historic.title.tolist()==frame.title.astype(str).tolist()
    assert historic.category.tolist()==frame.category.astype(str).tolist()
    persistent=(historic.f2_error_count=='5') & (historic.f3_error_count=='5')
    assert persistent.sum()==548
    torch.set_num_threads(4)
    run_rows, class_rows, confusion_rows, hashes = [],[],[],{}
    preds={g:[] for g in GROUPS}
    for group in GROUPS:
        X, V = features[group]
        for seed in SEEDS:
            folder = p.ROOT / baseline.loc[str(seed),'run_path'] if group=='F8-0' else p.HERE/'runs'/f"{group.replace('-','_')}_s{seed}"
            names=['metrics.json','model.pt','history.csv','val_predictions.csv','training_curve.png','confusion_val.png']
            assert all((folder/n).is_file() for n in names), f'missing artifacts: {folder}'
            m=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
            assert m['seed']==seed and m['training']['params']==base['training']['params']
            expected_sha=m['provenance']['split_sha256'] if group=='F8-0' else m['data_sha256']
            assert expected_sha=={'train':p.TRAIN_SHA,'val':p.VAL_SHA}
            arch=m['model']
            assert arch['model_type']=='bp' and arch['hidden_dim']==64 and arch['dropout']==.2 and arch['input_dropout']==0
            assert arch['input_dim']==X.shape[1] and arch['output_dim']==6
            assert arch['n_parameters']==p.bp_parameter_count(X.shape[1])
            if group != 'F8-0':
                assert m['feature']==(f1 if group=='F8-1' else f2)
                assert m['semantic_cache_manifest_sha256']==p.digest(p.CACHE/'manifest.json')
                assert m['frozen_encoder_parameters']==manifest['frozen_encoder_parameters']
            model=build_model(arch)
            model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True),strict=True)
            assert model.count_parameters()==arch['n_parameters']
            pred=predict(model,V)
            saved=pd.read_csv(folder/'val_predictions.csv', dtype=str, encoding='utf-8-sig')
            for col in ('id','title','category'):
                assert saved[col].tolist()==frame[col].astype(str).tolist()
            assert np.array_equal(pred,saved.predicted_category.map(p.C.LABEL_TO_INDEX).to_numpy())
            macro=f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)
            acc=accuracy_score(y,pred)
            cm=confusion_matrix(y,pred,labels=np.arange(6))
            assert abs(macro-m['val']['f1_macro'])<1e-12 and abs(acc-m['val']['accuracy'])<1e-12
            assert cm.tolist()==m['val']['confusion_matrix']
            hist=pd.read_csv(folder/'history.csv')
            assert np.isfinite(hist.select_dtypes(include=[np.number]).to_numpy()).all()
            best,epoch=-1.,0
            for e,score in zip(hist.epoch,hist.val_f1_macro):
                if score>best+m['training']['params']['min_delta']: best,epoch=score,int(e)
            assert epoch==m['training']['best_epoch'] and abs(best-macro)<1e-12
            assert len(hist)==m['training']['epochs_run']
            assert len(hist)-epoch==m['training']['params']['patience'] or len(hist)==m['training']['params']['max_epochs']
            prec,rec,fs,supp=precision_recall_fscore_support(y,pred,labels=np.arange(6),zero_division=0)
            for i,cat in enumerate(p.C.LABELS):
                for metric,val in [('precision',prec[i]),('recall',rec[i]),('f1',fs[i])]:
                    assert abs(m['val']['per_class'][cat][metric]-val)<1e-12
                class_rows.append(dict(group=group,seed=seed,category=cat,precision=prec[i],recall=rec[i],f1=fs[i],support=int(supp[i])))
                for j,dest in enumerate(p.C.LABELS):
                    confusion_rows.append(dict(group=group,seed=seed,true_category=cat,predicted_category=dest,count=int(cm[i,j])))
            bestrow=hist.loc[hist.epoch==epoch].iloc[0]
            run_rows.append(dict(group=group,seed=seed,macro_f1=macro,accuracy=acc,bp_parameters=arch['n_parameters'],
                encoder_parameters=0 if group=='F8-0' else manifest['frozen_encoder_parameters'], input_dim=X.shape[1],
                best_epoch=epoch,epochs_run=len(hist),train_seconds=m['training']['train_time'],
                best_train_loss=float(bestrow.train_eval_loss),best_val_loss=float(bestrow.val_loss),
                final_train_loss=float(hist.iloc[-1].train_eval_loss),final_val_loss=float(hist.iloc[-1].val_loss),
                society_to_finance=int(cm[4,0]),society_to_entertainment=int(cm[4,2]),path=str(folder.relative_to(p.ROOT))))
            hashes[f'{group} seed={seed}']={n:p.digest(folder/n) for n in names}
            preds[group].append(pred)
            print(f'{group} seed={seed}: strict checkpoint, predictions and metrics verified; F1={macro:.8f}',flush=True)
    run,classes,confusion=pd.DataFrame(run_rows),pd.DataFrame(class_rows),pd.DataFrame(confusion_rows)
    agg=run.groupby('group',sort=False).agg(n=('seed','size'),f1_mean=('macro_f1','mean'),f1_std=('macro_f1','std'),
        accuracy_mean=('accuracy','mean'),accuracy_std=('accuracy','std'),bp_parameters=('bp_parameters','first'),
        encoder_parameters=('encoder_parameters','first'),best_epoch_mean=('best_epoch','mean'),
        train_seconds_mean=('train_seconds','mean'),society_to_finance_mean=('society_to_finance','mean'),
        society_to_entertainment_mean=('society_to_entertainment','mean'),
        best_train_loss_mean=('best_train_loss','mean'),best_val_loss_mean=('best_val_loss','mean')).reset_index()
    class_agg=classes.groupby(['group','category'],sort=False).agg(f1_mean=('f1','mean'),f1_std=('f1','std'),
        precision_mean=('precision','mean'),recall_mean=('recall','mean')).reset_index()
    paired=pd.DataFrame({'seed':SEEDS})
    for metric in ('macro_f1','accuracy','society_to_finance','society_to_entertainment'):
        pivot=run.pivot(index='seed',columns='group',values=metric)
        for g in GROUPS: paired[f'{g}_{metric}']=[pivot.loc[s,g] for s in SEEDS]
        for a,b in [('F8-1','F8-0'),('F8-2','F8-0'),('F8-2','F8-1')]:
            paired[f'{a}_minus_{b}_{metric}']=paired[f'{a}_{metric}']-paired[f'{b}_{metric}']
    correct={g:(np.stack(preds[g])==y[None,:]) for g in GROUPS}
    f70_persistent=~correct['F8-0'].any(axis=0)
    title=pd.DataFrame({'id':frame.id.astype(str),'title':frame.title.astype(str),'category':frame.category.astype(str),
                        'f2_f3_persistent':persistent,'f70_persistent':f70_persistent})
    per_seed_persistent, cohort_rows = [],[]
    for g in GROUPS:
        title[f'{g}_correct_count']=correct[g].sum(axis=0)
        for i,seed in enumerate(SEEDS):
            title[f'{g}_pred_{seed}']=[p.C.LABELS[k] for k in preds[g][i]]
            per_seed_persistent.append(dict(group=g,seed=seed,cohort='F2_and_F3_all_wrong',n=int(persistent.sum()),
                correct=int(correct[g][i,persistent].sum()),wrong=int((~correct[g][i,persistent]).sum())))
        for label,mask in [('F2_and_F3_all_wrong',persistent.to_numpy()),('F7_0_all_wrong',f70_persistent)]:
            corr=correct[g][:,mask]
            cohort_rows.append(dict(group=g,cohort=label,n=int(mask.sum()),all_five_wrong=int((~corr.any(axis=0)).sum()),
                all_five_correct=int(corr.all(axis=0).sum()),at_least_one_correct=int(corr.any(axis=0).sum()),
                correct_predictions_mean=float(corr.sum(axis=1).mean())))
    cohorts=pd.DataFrame(cohort_rows)
    tables={'per_seed.csv':run,'summary.csv':agg,'per_class.csv':classes,'per_class_summary.csv':class_agg,
        'confusion_matrices.csv':confusion,'paired.csv':paired,'persistent_summary.csv':cohorts,
        'persistent_per_seed.csv':pd.DataFrame(per_seed_persistent),'per_title.csv':title}
    for name,df in tables.items(): df.to_csv(out/name,index=False,encoding='utf-8-sig')
    audit={'train_sha256':p.TRAIN_SHA,'val_sha256':p.VAL_SHA,'cache_manifest_sha256':p.digest(p.CACHE/'manifest.json'),
        'encoder_revision':manifest['revision'],'strict_checkpoint_and_prediction_match':True,
        'metrics_confusions_and_early_stopping_recomputed':True,'id_title_label_order_verified':True,
        'cache_and_encoder_file_hashes_verified':True,'artifact_sha256':hashes,
        'fit_scope_evidence':'F6-B training code and frozen training hashes; BGE inference mode and frozen parameters; audit does not independently refit',
        'pretraining_dataset_overlap_with_heldout_unknown':True,'test_access':False}
    p.save_json(out/'audit.json',audit)
    show=agg[['group','f1_mean','f1_std','accuracy_mean','accuracy_std','bp_parameters','encoder_parameters',
              'best_epoch_mean','train_seconds_mean','society_to_finance_mean','society_to_entertainment_mean']].copy()
    for col in ('f1_mean','f1_std','accuracy_mean','accuracy_std'): show[col]=show[col].map(lambda v:f'{v:.5f}')
    for col in ('best_epoch_mean','train_seconds_mean','society_to_finance_mean','society_to_entertainment_mean'):
        show[col]=show[col].map(lambda v:f'{v:.2f}')
    cp=class_agg.pivot(index='category',columns='group',values='f1_mean').reset_index()
    for g in GROUPS: cp[g]=cp[g].map(lambda v:f'{v:.5f}')
    per=run[['group','seed','macro_f1','accuracy','best_epoch','epochs_run','train_seconds']].copy()
    for col in ('macro_f1','accuracy'): per[col]=per[col].map(lambda v:f'{v:.5f}')
    per['train_seconds']=per.train_seconds.map(lambda v:f'{v:.2f}')
    pair=paired[['seed','F8-1_minus_F8-0_macro_f1','F8-2_minus_F8-0_macro_f1','F8-2_minus_F8-1_macro_f1']].copy()
    for c in pair.columns[1:]:pair[c]=pair[c].map(lambda v:f'{v:+.5f}')
    d10=paired['F8-1_minus_F8-0_macro_f1']; d20=paired['F8-2_minus_F8-0_macro_f1']; d21=paired['F8-2_minus_F8-1_macro_f1']
    encode=manifest['splits']['train']['seconds']+manifest['splits']['val']['seconds']
    lines=['# F.8 冻结 BGE 语义特征实验','','## 配置与外部知识','',
        '固定 F3-A 训练集 7,531 条与原验证集 3,184 条；BP hidden=64、lr=0.0003，其余训练、早停配置与 F7-0 相同。F8-0 复用 F7-0，F8-1 为 512 维语义，F8-2 为 10,000 维 F6-B 特征加 512 维语义。种子均为 42–46。',
        'BGE 是外部预训练知识，不能将差异解释为纯 TF-IDF 或 BP 网络结构改进。编码器全程冻结，不参与 BP 训练；预训练语料与项目验证集是否重叠无法从模型卡确认。',
        '模型 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)，固定版本 `7999e1d3359715c523056ef9478215996d62a620`，MIT 许可证；引用 Xiao et al., [C-Pack](https://arxiv.org/abs/2309.07597) (2023)。使用 `[CLS]` 最终隐藏状态、逐行 L2，分类不添加检索指令。',
        'F8-2 的 TF-IDF 块和 BGE 块各自逐行 L2，直接按 1:1 拼接，不进行拼接后归一化或融合权重搜索；TF-IDF 全零行的融合向量范数为 1，其余一般为 √2。512 维语义模型权重及特征缓存哈希和依赖版本见 `cache/manifest.json`。', '',
        '## 五种子总体结果（样本标准差）','',table(show,list(show.columns)),'',
        '## 每种子结果','',table(per,list(per.columns)),'',
        '## 配对 Macro-F1 差值','',table(pair,list(pair.columns)),'',
        f'F8-1 − F8-0 平均 {d10.mean():+.5f}，{(d10>0).sum()}/5 为正；F8-2 − F8-0 平均 {d20.mean():+.5f}，{(d20>0).sum()}/5 为正；F8-2 − F8-1 平均 {d21.mean():+.5f}，{(d21>0).sum()}/5 为正。','',
        '## 六类别平均 F1','',table(cp,['category']+list(GROUPS)),'',
        '## 持续错误样本','',
        'F2 和 F3-A 各五种子均错的 548 条来自既有 F.5 定义；F7-0 持续错误表示 F7-0 五种子均错。三组在同一固定集合上统计。','',
        table(cohorts,list(cohorts.columns)),'',
        '逐标题 ID、标题、真实类别、各组五次预测见 `per_title.csv`；逐种子困难样本和各类别 Precision/Recall/F1、混淆矩阵见对应 CSV。','',
        '## 资源、核验与限制','',
        f'冻结编码器参数 {manifest["frozen_encoder_parameters"]:,}，GPU {manifest["device_name"]}；一次性批量编码训练与验证标题分别 {manifest["splits"]["train"]["seconds"]:.2f}s 和 {manifest["splits"]["val"]["seconds"]:.2f}s，合计 {encode:.2f}s。该计时不包括模型下载、模型加载、缓存序列化，也不等于线上推理延迟。所有标题截断数为 {manifest["splits"]["train"]["titles_truncated"]+manifest["splits"]["val"]["titles_truncated"]}。',
        '表中训练耗时仅指 BP 每轮训练和验证，不包括 BGE 提取、下载、模型加载或缓存后特征拼接；机器负载与历史运行可能不同，不能单凭秒数判断部署成本。缓存读入及特征拼接耗时记录于各运行 metrics.json，特征转换在同一调用内为多个种子共享。',
        '15 个 checkpoint 已 strict=True 重建，复算验证预测、指标、混淆与早停；缓存哈希、ID、标题、标签与顺序已核对。训练代码未拟合 BGE，TF-IDF 处理器仅由训练集拟合。未评估测试集。',
        'F8 五种子在 seed=42 初步结果后决定执行，且模型早停和方案比较均使用同一验证集；五种子检验初始化波动，不能代替独立泛化测试或排除外部预训练语料重叠。','',
        '## 实际命令','', '```powershell',"$env:PYTHONUTF8='1'",
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/f8_experiments/pipeline.py encode',
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/f8_experiments/pipeline.py train --groups F8_1 F8_2 --seeds 42',
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/f8_experiments/pipeline.py train --groups F8_1 F8_2 --seeds 43 44 45 46',
        'D:\\WorkSpace\\PATH\\Miniconda\\envs\\vision\\python.exe results/member_C/f8_experiments/analyze_f8.py --output-dir results/member_C/f8_experiments/another_audit',
        '```','以上输出目录已有结果，原训练入口会拒绝覆盖；审计复跑请使用新的项目内空目录。','']
    (out/'summary.md').write_text('\n'.join(lines),encoding='utf8')
    print('F8 audit and report completed',flush=True)

if __name__=='__main__':main()

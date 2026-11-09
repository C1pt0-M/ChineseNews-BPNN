"""Copy the two highest-scoring saved runs for each requested configuration.

All ranks are retrospective; no checkpoint is trained, evaluated or selected for deployment.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import model_config as C

STAGES=('E_input_dropout','F2','F3','F6','F7','F8','final_test')
LABELS=list(C.LABELS)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''): h.update(x)
    return h.hexdigest()

def csv(path): return pd.read_csv(ROOT/path,encoding='utf-8-sig')

def json_read(path): return json.loads(path.read_text(encoding='utf8'))

def metric_from_saved_prediction(path):
    df=pd.read_csv(path,encoding='utf-8-sig',dtype=str,keep_default_na=False)
    assert set(df.category).issubset(LABELS) and set(df.predicted_category).issubset(LABELS)
    return (f1_score(df.category,df.predicted_category,labels=LABELS,average='macro',zero_division=0),
            accuracy_score(df.category,df.predicted_category))

def main():
    for name in (*STAGES,'all_runs.csv','selected_runs.csv','group_summary.csv','stage_top_two.csv','README.md','manifest.json'):
        if (HERE/name).exists(): raise FileExistsError(f'Refusing to overwrite {HERE/name}')
    records=[]
    def add(stage,group,seed,source,score,accuracy,split='validation'):
        folder=ROOT/Path(str(source))
        m=json_read(folder/'metrics.json')
        seed=int(seed)
        assert int(m['seed'])==seed, f'Wrong seed: {folder}'
        if split=='validation':
            computed,acc=metric_from_saved_prediction(folder/'val_predictions.csv')
            assert abs(computed-float(score))<1e-9 and abs(acc-float(accuracy))<1e-9, f'Metrics/prediction mismatch: {folder}'
            assert abs(computed-float(m['val']['f1_macro']))<1e-9
        records.append({'stage':stage,'configuration':group,'seed':seed,'split':split,
                        'macro_f1':float(score),'accuracy':float(accuracy),
                        'parameters':int(m['model']['n_parameters']),
                        'best_epoch':int(m['training']['best_epoch']),
                        'source_run':str(folder.relative_to(ROOT))})
    # The image calls this E; the actual Input Dropout experiment is C3.
    for group,subdir,drop in [('dropout_0','b1',0.),('dropout_005','p005',.05),
                              ('dropout_010','p010',.10),('dropout_020','p020',.20)]:
        folders=sorted((ROOT/'results/member_C/c3_multi'/subdir).glob('*/metrics.json'))
        assert len(folders)==5, f'Incomplete Input Dropout runs: {subdir}'
        for file in folders:
            m=json_read(file)
            assert abs(m['model'].get('input_dropout',0)-drop)<1e-12
            add('E_input_dropout',group,m['seed'],file.parent.relative_to(ROOT),m['val']['f1_macro'],m['val']['accuracy'])
    f2=csv('results/member_C/f2_analysis/summary.csv')
    assert set(f2.group)=={'D3','A_raw','B_chi2_10000','F_char','F_word_char'}
    for _,r in f2.iterrows(): add('F2',r['group'],r.seed,r.run_path,r.macro_f1,r.accuracy)
    f3=csv('results/member_C/f3_analysis/f3_a_multiseed/summary.csv')
    for _,r in f3[f3.group=='F3-A'].iterrows():
        add('F3','F3_A_formal',r.seed,r.run_path,r.macro_f1,r.accuracy)
        add('F6','F6_baseline_5000_5000',r.seed,r.run_path,r.macro_f1,r.accuracy)
    f6=csv('results/member_C/f6_experiments/summary.csv')
    for _,r in f6[f6.scheme=='F6-A'].iterrows(): add('F6','F6_A_3000_7000',r.seed,r.run_path,r.macro_f1,r.accuracy)
    f6multi=csv('results/member_C/f6_experiments/f6_b_multiseed/per_seed.csv')
    for _,r in f6multi[f6multi.group=='F6-B'].iterrows(): add('F6','F6_B_7000_3000',r.seed,r.run_path,r.macro_f1,r.accuracy)
    f7=csv('results/member_C/f7_experiments/per_seed.csv')
    for _,r in f7.iterrows(): add('F7',r['group'],r.seed,r['path'],r.macro_f1,r.accuracy)
    f8=csv('results/member_C/f8_experiments/per_seed.csv')
    for _,r in f8.iterrows(): add('F8',r['group'],r.seed,r['path'],r.macro_f1,r.accuracy)
    final=csv('results/member_C/final_generalization/per_seed.csv')
    predictions=csv('results/member_C/final_generalization/predictions.csv')
    actual=pd.read_csv(ROOT/'dataset/splits/test.csv',encoding='utf-8-sig',dtype=str,keep_default_na=False)
    assert predictions.id.astype(str).tolist()==actual.id.tolist() and predictions.category.tolist()==actual.category.tolist()
    for _,r in final.iterrows():
        yhat=predictions[f"{r['group']}_pred_{int(r.seed)}"]
        f1=f1_score(actual.category,yhat,labels=LABELS,average='macro',zero_division=0)
        acc=accuracy_score(actual.category,yhat)
        assert abs(f1-r.macro_f1)<1e-12 and abs(acc-r.accuracy)<1e-12
        add('final_test',r['group'],r.seed,r.checkpoint_path,f1,acc,'previously_viewed_test')
    data=pd.DataFrame(records)
    for (stage,group),part in data.groupby(['stage','configuration']):
        assert part.seed.is_unique
        expected=1 if (stage,group)==('F6','F6_A_3000_7000') else 5
        assert len(part)==expected, f'Unexpected seed count: {stage} {group}'
    ranked=data.sort_values(['stage','configuration','macro_f1','seed'],ascending=[True,True,False,True]).copy()
    ranked['rank']=ranked.groupby(['stage','configuration']).cumcount()+1
    chosen=ranked[ranked['rank']<=2].copy()
    selected_outputs=[]
    file_manifest=[]
    test_classes=csv('results/member_C/final_generalization/per_class.csv')
    test_conf=csv('results/member_C/final_generalization/confusion_matrices.csv')
    for _,r in chosen.iterrows():
        dest=HERE/r.stage/r.configuration/f"rank{r['rank']}_seed{r.seed}"
        dest.mkdir(parents=True,exist_ok=False)
        source=ROOT/r.source_run
        copies=['metrics.json','history.csv','training_curve.png','training_curve.svg','confusion_val.png','confusion_val.csv']
        for name in copies:
            src=source/name
            if not src.exists(): continue
            assert src.is_file()
            shutil.copy2(src,dest/name)
            assert digest(src)==digest(dest/name)
            file_manifest.append({'file':str((dest/name).relative_to(HERE)),'source':str(src.relative_to(ROOT)),'sha256':digest(src)})
        if r['split']=='previously_viewed_test':
            src=ROOT/'results/member_C/final_generalization/plots'/f'{r.configuration}_s{r.seed}.png'
            assert src.is_file()
            shutil.copy2(src,dest/'confusion_test.png')
            assert digest(src)==digest(dest/'confusion_test.png')
            file_manifest.append({'file':str((dest/'confusion_test.png').relative_to(HERE)),'source':str(src.relative_to(ROOT)),'sha256':digest(src)})
            pc=test_classes[(test_classes.group==r.configuration)&(test_classes.seed==r.seed)]
            cm=test_conf[(test_conf.group==r.configuration)&(test_conf.seed==r.seed)]
            assert len(pc)==6 and len(cm)==36
            pc.to_csv(dest/'per_class_test.csv',index=False,encoding='utf-8-sig')
            cm.to_csv(dest/'confusion_test.csv',index=False,encoding='utf-8-sig')
            result=final[(final.group==r.configuration)&(final.seed==r.seed)]
            assert len(result)==1
            summary=json.loads(result.to_json(orient='records',force_ascii=False))[0]
        else:
            m=json_read(source/'metrics.json')
            per=m['val']['per_class']
            pd.DataFrame([{'category':k,**v} for k,v in per.items()]).to_csv(dest/'per_class_val.csv',index=False,encoding='utf-8-sig')
            summary={'group':r.configuration,'seed':int(r.seed),'split':'validation','val':m['val'],
                     'model':m['model'],'training':m['training']}
        (dest/'selected_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
        selected_outputs.append(str(dest.relative_to(HERE)))
    chosen['output_directory']=selected_outputs
    ranked.to_csv(HERE/'all_runs.csv',index=False,encoding='utf-8-sig')
    chosen.to_csv(HERE/'selected_runs.csv',index=False,encoding='utf-8-sig')
    summaries=ranked.groupby(['stage','configuration','split'],sort=True).agg(n=('seed','size'),
        mean_macro_f1=('macro_f1','mean'),std_sample=('macro_f1','std'),
        min_macro_f1=('macro_f1','min'),max_macro_f1=('macro_f1','max')).reset_index()
    summaries.to_csv(HERE/'group_summary.csv',index=False,encoding='utf-8-sig')
    stage_top=data.sort_values(['stage','macro_f1','seed','configuration'],ascending=[True,False,True,True]).copy()
    stage_top['rank_in_stage']=stage_top.groupby('stage').cumcount()+1
    stage_top=stage_top[stage_top.rank_in_stage<=2]
    stage_top.to_csv(HERE/'stage_top_two.csv',index=False,encoding='utf-8-sig')
    manifest={'selection':'Each configuration ranked by saved per-run Macro-F1 descending; ties use lower seed.',
        'result_scope':'F2/F3/F6/F7/F8 and C3 Input Dropout use validation; final_test uses previously viewed test only for retrospective display.',
        'not_model_selection':'Final test ranking is descriptive, not a basis for choosing or retraining a model.',
        'e_label':'Screenshot calls it E; actual Input Dropout experiment is results/member_C/c3_multi.',
        'f6_a_limit':'Only seed 42 is available; no second run is fabricated.',
        'group_count':len(summaries),'source_run_count':len(ranked),'selected_run_count':len(chosen),'copied_originals':file_manifest}
    (HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    lines=['# 各实验配置的前两次结果','',
        '按各配置单次 Macro-F1 从高到低排序；分数相同按种子从小到大。每个配置最多取两次；图和历史指标从原目录复制，复制后逐文件核对 SHA-256。未训练模型，未改动原实验。',
        '图中所称“E 系列 Input Dropout”在项目中的实际目录是 `results/member_C/c3_multi/`（0、0.05、0.10、0.20）；它不是 E1/E2 的线性或混合分支实验。',
        'F2、F3、F6、F7、F8 依据验证集 Macro-F1；`final_test` 依据已有最终评估的测试 Macro-F1，仅为事后展示。该测试集以前已被查看，不能用此排序重新选择模型或声称盲测。',
        'F6-A 仅 seed 42 一次，无法提供第二次。最高单次结果不等于五种子均值，也不代表模型稳健性；各组完整种子结果与均值分别见 `all_runs.csv` 和 `group_summary.csv`。','',
        '## 每阶段整体前两次','', '| 阶段 | 名次 | 配置 | seed | 数据集 | Macro-F1 | Accuracy |', '|---|---:|---|---:|---|---:|---:|']
    for _,r in stage_top.iterrows():
        lines.append(f"| {r.stage} | {r.rank_in_stage} | {r.configuration} | {r.seed} | {r['split']} | {r.macro_f1:.5f} | {r.accuracy:.5f} |")
    lines += ['', '## 每个配置的前两次','',
        '| 阶段 | 配置 | 名次 | seed | Macro-F1 | Accuracy | 文件夹 |', '|---|---|---:|---:|---:|---:|---|']
    for _,r in chosen.iterrows():
        target=r.output_directory.replace('\\','/')
        lines.append(f"| {r.stage} | {r.configuration} | {r['rank']} | {r.seed} | {r.macro_f1:.5f} | {r.accuracy:.5f} | [{target}]({target}/) |")
    lines += ['', '`selected_metrics.json` 是该次的指标摘要；`metrics.json` 是原训练记录（对最终评估而言，它记录的是验证集而非测试集）。`training_curve` 为训练/验证曲线；`confusion_val` 是验证混淆矩阵；最终评估的测试混淆图使用 `confusion_test.png`。',
        '各类别 F1 在 `per_class_val.csv` 或 `per_class_test.csv`，训练历史在 `history.csv`。F8-0、F7-0 等复用既有运行，因此不同阶段可能引用同一 checkpoint 和图。原始路径及哈希见 `manifest.json`。','']
    (HERE/'README.md').write_text('\n'.join(lines),encoding='utf8')
    print(f'Collected {len(chosen)} selected runs from {len(summaries)} configurations; audited {len(ranked)} saved runs.',flush=True)

if __name__=='__main__':main()

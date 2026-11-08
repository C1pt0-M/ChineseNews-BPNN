"""Recheck saved classic models against validation predictions and document the comparison."""
from hashlib import sha256
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(ROOT))
import model_config as C
from models.data import load_splits


def main():
    report=HERE/'summary.md'
    if report.exists():
        raise FileExistsError(report)
    result=json.loads((HERE/'results.json').read_text(encoding='utf-8'))
    cv=pd.read_csv(HERE/'cv_folds.csv',encoding='utf-8-sig')
    assert len(cv)==40 and cv.groupby(['classifier','C']).size().eq(5).all()
    folds=cv.groupby('fold').first()
    assert (folds.train_rows+folds.validation_rows==6549).all()
    assert (folds.feature_fit_rows==folds.train_rows).all()
    labels=load_splits(verify=True,splits=('val',))['val']
    texts,y,frame=labels
    with (HERE/'feature_builder.pkl').open('rb') as f: builder=pickle.load(f)
    f2=ROOT/'results/member_C/f2_formal_word_char/F_word_char_s42'
    with (f2/'feature_builder.pkl').open('rb') as f: orig=pickle.load(f)
    X=builder.transform(texts); ref=orig.transform(texts); diff=X-ref
    assert X.shape==(3184,10000) and (diff.nnz==0 or np.max(np.abs(diff.data))<1e-12)
    baseline=json.loads((f2/'metrics.json').read_text(encoding='utf-8'))
    assert baseline['provenance']['split_sha256']['val']==sha256((C.SPLIT_DIR/'val.csv').read_bytes()).hexdigest()
    results={}; class_rows=[]
    for name in ('LinearSVC','LogisticRegression'):
        data=result['models'][name]
        means=cv[cv.classifier==name].groupby('C').macro_f1.mean().to_dict()
        best=sorted(means,key=lambda c:(-means[c],c))[0]
        assert best==data['best_C'] and abs(means[best]-data['cv_macro_f1'])<1e-12
        with (HERE/name/'model.pkl').open('rb') as f: model=pickle.load(f)
        pred=model.predict(X)
        saved=pd.read_csv(HERE/name/'val_predictions.csv',encoding='utf-8-sig')
        assert saved.id.astype(str).tolist()==frame.id.astype(str).tolist()
        assert np.array_equal(pred,saved.predicted_category.map(C.LABEL_TO_INDEX).to_numpy())
        score=f1_score(y,pred,labels=np.arange(6),average='macro',zero_division=0)
        acc=accuracy_score(y,pred); cm=confusion_matrix(y,pred,labels=np.arange(6))
        assert abs(score-data['val']['f1_macro'])<1e-12 and abs(acc-data['val']['accuracy'])<1e-12
        assert cm.tolist()==data['val']['confusion_matrix']
        assert np.array_equal(pd.read_csv(HERE/name/'confusion_val.csv',index_col=0,encoding='utf-8-sig').to_numpy(),cm)
        p,r,f,s=precision_recall_fscore_support(y,pred,labels=np.arange(6),zero_division=0)
        for i,label in enumerate(C.LABELS):
            assert abs(f[i]-data['val']['per_class'][label]['f1'])<1e-12
            class_rows.append({'model':name,'category':label,'precision':p[i],'recall':r[i],'f1':f[i],'support':s[i]})
        results[name]={'C':best,'cv_mean':means[best], 'val_macro_f1':score,'accuracy':acc,
                       'society_to_finance':int(cm[4,0]),'society_to_entertainment':int(cm[4,2]),
                       'fit_seconds':data['final_fit_seconds']}
    for label,v in baseline['val']['per_class'].items():
        class_rows.append({'model':'BP_seed42','category':label,**v})
    pd.DataFrame(class_rows).to_csv(HERE/'per_class_comparison.csv',index=False,encoding='utf-8-sig')
    (HERE/'verification.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# F.4 经典线性分类器对照', '',
           '训练集 6,549 条，固定验证集 3,184 条；只使用标题。五折 StratifiedKFold，shuffle=True、seed=42；每折的词级和字级 TF-IDF、IDF、卡方筛选器仅在该折训练部分拟合。两个分类器共用该折特征以保证对照，C 在 0.01、0.1、1、10 中按折平均 Macro-F1 选择，平分时取较小 C。',
           '', 'CV 选 C 后，复用 F2 seed 42 保存在完整训练集上拟合的特征处理器；已独立重新拟合并核对词表、IDF、选择器及验证特征一致。模型接收相同的稀疏 5,000 词 + 5,000 字、拼接后逐行 L2 归一化特征。固定验证集不参与参数选择。',
           '', '| 模型 | C | 5 折 Macro-F1 | 固定验证 Macro-F1 | Accuracy | 末次拟合秒数 |', '|---|---:|---:|---:|---:|---:|',
           f'| BP F_word_char seed 42 | — | — | {baseline["val"]["f1_macro"]:.5f} | {baseline["val"]["accuracy"]:.5f} | {baseline["training"]["train_time"]:.2f} |']
    for name in ('LinearSVC','LogisticRegression'):
        a=results[name]; lines.append(f'| {name} | {a["C"]:g} | {a["cv_mean"]:.5f} | {a["val_macro_f1"]:.5f} | {a["accuracy"]:.5f} | {a["fit_seconds"]:.3f} |')
    lines += ['', '拟合时间为最终模型单次拟合，不含特征准备、CV 或验证预测；BP 时间按原记录为全部训练 epoch 总时间，计时口径不同。BP 五种子验证 Macro-F1 均值为 0.78369、样本标准差 0.00132；与单次经典模型不是同一随机重复口径。',
              '', '在本次固定验证集上，LinearSVC 和 LogisticRegression 分别比 BP seed 42 低 0.00927 和 0.01211 Macro-F1。两者社会类 F1 分别为 0.57637、0.57252，低于 BP 的 0.59639；LinearSVC 的教育类 F1 略高于 BP。两个线性模型的整体排序一致，但类别变化并不完全相同。',
              '', '## 训练集五折参数选择', '', '| C | LinearSVC Macro-F1 | LogisticRegression Macro-F1 |', '|---:|---:|---:|']
    for c in (.01,.1,1.,10.):
        a=cv[(cv.classifier=='LinearSVC')&(cv.C==c)].macro_f1.mean();b=cv[(cv.classifier=='LogisticRegression')&(cv.C==c)].macro_f1.mean()
        lines.append(f'| {c:g} | {a:.5f} | {b:.5f} |')
    classes=pd.DataFrame(class_rows)
    lines += ['', '## 六类别指标', '', '| 模型 | 类别 | Precision | Recall | F1 |', '|---|---|---:|---:|---:|']
    for name in ('BP_seed42','LinearSVC','LogisticRegression'):
        for label in C.LABELS:
            a=classes[(classes.model==name)&(classes.category==label)].iloc[0]
            lines.append(f'| {name} | {label} | {a.precision:.5f} | {a.recall:.5f} | {a.f1:.5f} |')
    lines += ['', '## 社会类混淆', '', '| 模型 | 社会→财经 | 社会→文娱 |', '|---|---:|---:|',
              f'| BP seed 42 | {baseline["val"]["confusion_matrix"][4][0]} | {baseline["val"]["confusion_matrix"][4][2]} |']
    for name in ('LinearSVC','LogisticRegression'):
        a=results[name];lines.append(f'| {name} | {a["society_to_finance"]} | {a["society_to_entertainment"]} |')
    lines += ['', '完整验证混淆矩阵与逐条预测分别位于各模型子目录；折级结果在 `cv_folds.csv`，完整配置和指标在 `results.json`。保存的特征处理器和模型重新加载后均能复现验证预测。',
              '', '## 固定验证集混淆矩阵', '', '行是真实类别，列是预测类别。类别顺序为财经、体育、文娱、教育、社会、国际。']
    for name in ('LinearSVC','LogisticRegression'):
        cm=result['models'][name]['val']['confusion_matrix']
        lines += ['', '| '+name+' | '+' | '.join(C.LABELS)+' |', '|---|---:|---:|---:|---:|---:|---:|']
        lines += ['| '+label+' | '+' | '.join(map(str,row))+' |' for label,row in zip(C.LABELS,cm)]
    lines += ['', '## 解释边界', '',
              '在这套词字融合特征、C 网格和固定划分下比较模型；折内拟合避免了词表、IDF 和监督特征选择跨折泄漏。单次验证集结果不能证明 BP 或线性模型已经达到性能上限，也不能单凭差异判断模型能力与类别噪声各自的贡献。未访问测试集指标。',
              '', '比较还存在验证集使用方式差异：BP 使用验证 Macro-F1 早停并选择 checkpoint，经典模型只用训练集内部交叉验证选择 C，随后在固定验证集评估一次。', '']
    report.write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

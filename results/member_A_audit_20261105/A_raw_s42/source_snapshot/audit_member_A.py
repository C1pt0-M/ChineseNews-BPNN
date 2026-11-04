"""Analyze only B0 train/validation artifacts; never evaluates the test set.

Run after: python train_bp.py --exp A_raw --seeds 42 --output-root results/member_A_audit_20261105
"""
from pathlib import Path
import argparse
import json
import hashlib
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import torch

import model_config as C
from models.bp import BPNet
from models.data import load_splits
from models.features import tokenize_corpus
from models.trainer import predict
from models.runs import save_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, default=Path('results/member_A_audit_20261105/A_raw_s42'))
    args = parser.parse_args()
    out = args.run
    m = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    history = pd.read_csv(out / 'history.csv')
    history['eval_loss_gap'] = history.val_loss - history.train_eval_loss
    history.to_csv(out / 'history_analysis.csv', index=False, encoding='utf-8-sig')
    # Existing best checkpoint supplies all classification-error evidence.
    old = C.RUN_DIR / 'A_raw_s42'
    splits = load_splits(splits=('train', 'val'))
    with open(old / 'feature_builder.pkl', 'rb') as f:
        builder = pickle.load(f)
    X = builder.transform(tokenize_corpus(splits['val'][0]))
    model = BPNet(10864)
    model.load_state_dict(torch.load(old / 'model.pt', map_location='cpu', weights_only=True))
    pred = predict(model, X)
    frame = splits['val'][2].copy()
    frame['predicted_category'] = [C.LABELS[i] for i in pred]
    frame['is_error'] = frame.category != frame.predicted_category
    frame.to_csv(out / 'historical_best_val_predictions.csv', index=False, encoding='utf-8-sig')
    from models.metrics import compute_metrics
    metrics = compute_metrics(splits['val'][1], pred)
    saved = json.loads((old / 'metrics.json').read_text(encoding='utf-8'))
    assert metrics['confusion_matrix'] == saved['val']['confusion_matrix']
    per_class = pd.DataFrame(metrics['per_class']).T
    per_class.to_csv(out / 'per_class_val.csv', encoding='utf-8-sig')
    errors = frame[frame.is_error]
    directions = errors.groupby(['category', 'predicted_category']).size().reset_index(name='count')
    directions['true_class_rate'] = directions.apply(lambda r: r['count'] / (frame.category == r.category).sum(), axis=1)
    directions = directions.sort_values('count', ascending=False)
    directions.to_csv(out / 'confusion_directions.csv', index=False, encoding='utf-8-sig')
    pairs = [('社会','财经'),('财经','社会'),('社会','文娱'),('文娱','社会'),
             ('财经','文娱'),('文娱','财经'),('教育','社会'),('社会','教育')]
    # 3 examples for each of the 4 dominant directions, 2 for each remaining direction = 20.
    examples = pd.concat([errors[(errors.category == a) & (errors.predicted_category == b)]
                          .sort_values(['publish_time','id']).head(3 if i < 4 else 2)
                          for i,(a,b) in enumerate(pairs)])
    examples.to_csv(out / 'error_candidates.csv', index=False, encoding='utf-8-sig')
    old_h = pd.read_csv(old / 'history.csv')
    comparison = {'same_epochs': len(old_h) == len(history),
                  'historical_prediction_matches_metrics': True,
                  'new_predictions_equal_historical': frame.predicted_category.tolist() == pd.read_csv(out / 'val_predictions.csv').predicted_category.tolist(),
                  'max_abs_history_difference': {col: float(np.max(np.abs(old_h[col] - history[col])))
                     if len(old_h) == len(history) else None
                     for col in ('train_loss','val_loss','val_f1_macro','val_accuracy')},
                  'best_epoch': m['training']['best_epoch'], 'epochs_run': len(history),
                  'historical_model_sha256': hashlib.sha256((old/'model.pt').read_bytes()).hexdigest(),
                  'historical_feature_sha256': hashlib.sha256((old/'feature_builder.pkl').read_bytes()).hexdigest()}
    save_json(out / 'reproduction_check.json', comparison)
    records=[]
    for seed in C.MULTI_SEEDS:
        h=json.loads((C.RUN_DIR/f'A_raw_s{seed}'/'metrics.json').read_text(encoding='utf-8'))
        records.append({'seed':seed,'best_epoch':h['training']['best_epoch'],
                        **{f'val_{k}':h['val'][k] for k in C.METRIC_NAMES},
                        'historical_test_f1_macro':h['test']['f1_macro']})
    pd.DataFrame(records).to_csv(out/'historical_five_seeds.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(comparison,ensure_ascii=False,indent=2))
    print(examples[['id','title','category','predicted_category']].to_json(orient='records',force_ascii=False,indent=2))
    # Editable network diagram; diagram uses a single palette so no semantic color legend needed.
    doc=ET.Element('mxfile',host='app.diagrams.net')
    diagram=ET.SubElement(doc,'diagram',name='B0 network',id='b0')
    root=ET.SubElement(ET.SubElement(diagram,'mxGraphModel',page='1',pageWidth='1500',pageHeight='300'),'root')
    ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
    labels=['TF-IDF input\n10864','Linear\n10864 → 256','ReLU\nmax(0,z)','Dropout p=0.2\ntraining only','Linear\n256 → 6','6 logits\nCrossEntropyLoss']
    for i,label in enumerate(labels):
        cell=ET.SubElement(root,'mxCell',id=f'n{i}',value=label,style='rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=16;',vertex='1',parent='1')
        ET.SubElement(cell,'mxGeometry',x=str(30+i*240),y='60',width='200',height='90',attrib={'as':'geometry'})
        if i:
            edge=ET.SubElement(root,'mxCell',id=f'e{i}',edge='1',parent='1',source=f'n{i-1}',target=f'n{i}',style='edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;')
            ET.SubElement(edge,'mxGeometry',relative='1',attrib={'as':'geometry'})
    ET.ElementTree(doc).write(out/'network.drawio',encoding='utf-8',xml_declaration=True)


if __name__ == '__main__':
    main()

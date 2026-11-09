"""Pinned frozen BGE extraction and F.8 BP training on the fixed F3-A split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / 'vendor'))
os.environ['HF_HOME'] = str(HERE / '.hf_home')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

import numpy as np
import scipy.sparse as sp
import torch
import model_config as C
from models.data import load_splits, load_training_override
from models.plots import plot_confusion_matrix, plot_training_curves
from models.runs import now_str, save_confusion, save_history, save_json
from models.trainer import predict, train_bp_model

TRAIN = ROOT / 'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv'
VAL = ROOT / 'dataset/splits/val.csv'
BASE = ROOT / 'results/member_C/f6_experiments/runs/F6_B_s42'
TRAIN_SHA = '7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
VAL_SHA = '9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0'
REVISION = '7999e1d3359715c523056ef9478215996d62a620'
WEIGHTS_SHA = '354763b9b1357bc9c44f62c6be2276321081ed2567773608c0d0785b61d5a026'
MODEL = HERE / 'bge_model'
CACHE = HERE / 'cache'

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def bp_parameter_count(input_dim, hidden_dim=64, n_classes=6):
    return input_dim * hidden_dim + hidden_dim + hidden_dim * n_classes + n_classes

def splits_and_baseline():
    if digest(TRAIN) != TRAIN_SHA or digest(VAL) != VAL_SHA:
        raise RuntimeError('Frozen train/validation SHA mismatch')
    baseline = json.loads((BASE / 'metrics.json').read_text(encoding='utf8'))
    if baseline['provenance']['split_sha256'] != {'train': TRAIN_SHA, 'val': VAL_SHA}:
        raise RuntimeError('F7-0 dataset mismatch')
    a, f = baseline['model'], baseline['feature']
    if (a['model_type'] != 'bp' or a['hidden_dim'] != 64 or a['dropout'] != .2
        or a['input_dropout'] != 0 or a['input_dim'] != 10000
        or f['word']['selected_features'] != 7000 or f['char']['selected_features'] != 3000
        or f['fit_rows'] != 7531 or baseline['training']['params']['lr'] != .0003):
        raise RuntimeError('F7-0 feature or network configuration mismatch')
    splits = load_splits(verify=True, splits=('train', 'val'))
    splits['train'] = load_training_override(TRAIN, TRAIN_SHA, splits['train'])
    for name in ('train', 'val'):
        texts, labels, df = splits[name]
        if not df.id.is_unique or list(df.title.astype(str)) != texts:
            raise RuntimeError(f'{name}: invalid IDs or title order')
        if not np.array_equal(df.category.map(C.LABEL_TO_INDEX).to_numpy(dtype='int64'), labels):
            raise RuntimeError(f'{name}: label mismatch')
    if (len(splits['train'][1]), len(splits['val'][1])) != (7531, 3184):
        raise RuntimeError('Unexpected split sizes')
    return splits, baseline

def encode():
    import transformers, tokenizers, huggingface_hub, safetensors
    from transformers import AutoModel, AutoTokenizer
    splits, _ = splits_and_baseline()
    paths = [CACHE / 'train.npz', CACHE / 'val.npz', CACHE / 'manifest.json']
    if any(p.exists() for p in paths):
        raise FileExistsError('Semantic cache already exists; refusing to overwrite')
    required = ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','README.md']
    if not all((MODEL / n).is_file() for n in required):
        raise FileNotFoundError('Pinned model files missing; run download_model.py first')
    if digest(MODEL/'model.safetensors') != WEIGHTS_SHA:
        raise RuntimeError('Pinned BGE weights SHA-256 mismatch')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model, info = AutoModel.from_pretrained(MODEL, local_files_only=True, output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
        raise RuntimeError(f'Model loading had missing/mismatched weights: {info}')
    model.requires_grad_(False)
    model.eval()
    if any(p.requires_grad for p in model.parameters()) or model.config.hidden_size != 512:
        raise RuntimeError('Encoder not frozen or unexpected embedding dimension')
    encoder_parameters = sum(p.numel() for p in model.parameters())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    CACHE.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name in ('train','val'):
        texts, labels, df = splits[name]
        pieces = []
        truncated = 0
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(texts), 32):
                lengths = tokenizer(texts[start:start+32], truncation=False, return_length=True)['length']
                truncated += sum(length > 512 for length in lengths)
                data = tokenizer(texts[start:start+32], padding=True, truncation=True,
                                 max_length=512, return_tensors='pt')
                data = {k: v.to(device) for k, v in data.items()}
                result = model(**data).last_hidden_state[:,0,:]
                result = torch.nn.functional.normalize(result, p=2, dim=1)
                pieces.append(result.cpu().numpy().astype(np.float32, copy=False))
        embeddings = np.concatenate(pieces)
        seconds = time.perf_counter() - t0
        if embeddings.shape != (len(texts),512) or not np.isfinite(embeddings).all():
            raise RuntimeError('Invalid BGE output')
        if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5):
            raise RuntimeError('BGE output normalization mismatch')
        path = CACHE / f'{name}.npz'
        np.savez_compressed(path, vectors=embeddings, ids=df.id.astype(str).to_numpy(dtype=str),
                            titles=np.asarray(texts, dtype=str), labels=labels)
        entries[name] = {'rows': len(texts), 'seconds': seconds, 'titles_truncated': truncated, 'cache_sha256': digest(path),
                         'source_sha256': TRAIN_SHA if name == 'train' else VAL_SHA}
        print(f'{name}: encoded {len(texts)} titles in {seconds:.2f}s', flush=True)
    manifest = {'model': 'BAAI/bge-small-zh-v1.5', 'revision': REVISION,
        'model_source': 'https://huggingface.co/BAAI/bge-small-zh-v1.5', 'license': 'MIT',
        'citation': 'Xiao et al., C-Pack: Packaged Resources To Advance General Chinese Embedding, arXiv:2309.07597 (2023)',
        'files_sha256': {n:digest(MODEL/n) for n in required}, 'hidden_size': 512,
        'frozen_encoder_parameters': encoder_parameters, 'encoder_trainable_parameters': 0,
        'pooling': 'last_hidden_state CLS token, row L2 normalized', 'instruction': None,
        'tokenizer': {'padding': True, 'truncation': True, 'max_length': 512, 'batch_size': 32},
        'device': str(device), 'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'dtype': 'float32', 'versions': {'torch':torch.__version__, 'transformers':transformers.__version__,
        'tokenizers':tokenizers.__version__, 'huggingface_hub':huggingface_hub.__version__,
        'safetensors':safetensors.__version__, 'numpy':np.__version__}, 'splits':entries}
    save_json(CACHE / 'manifest.json', manifest)
    print('Frozen encoder parameters:', encoder_parameters, flush=True)

def load_vectors(splits):
    manifest = json.loads((CACHE / 'manifest.json').read_text(encoding='utf8'))
    if manifest['revision'] != REVISION or manifest['encoder_trainable_parameters'] != 0:
        raise RuntimeError('Wrong or unfrozen encoder cache')
    for name, expected in manifest['files_sha256'].items():
        if digest(MODEL/name) != expected:
            raise RuntimeError(f'Encoder file mismatch: {name}')
    values = {}
    for name in ('train','val'):
        path = CACHE / f'{name}.npz'
        if digest(path) != manifest['splits'][name]['cache_sha256']:
            raise RuntimeError(f'{name} cache hash mismatch')
        texts, labels, frame = splits[name]
        with np.load(path, allow_pickle=False) as saved:
            if not (np.array_equal(saved['ids'], frame.id.astype(str).to_numpy(dtype=str))
                    and np.array_equal(saved['titles'], np.asarray(texts, dtype=str))
                    and np.array_equal(saved['labels'], labels)):
                raise RuntimeError(f'{name}: cached ID/title/label order differs from split')
            x = saved['vectors']
        if x.shape != (len(texts),512) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5):
            raise RuntimeError(f'{name}: invalid cached embeddings')
        values[name] = x
    return values, manifest

def make_features(group, splits, vectors, baseline):
    if group == 'F8_1':
        return vectors['train'], vectors['val'], {'scheme':'bge', 'input_dim':512,
            'normalization':'BGE CLS row L2', 'semantic_dim':512}
    if group != 'F8_2':
        raise ValueError(group)
    with (BASE/'feature_builder.pkl').open('rb') as f:
        builder = pickle.load(f)
    if builder.fit_rows_ != 7531 or builder.word_dim_ != 7000 or builder.char_dim_ != 3000:
        raise RuntimeError('F7-0 TF-IDF processor mismatch')
    tfidf = {name:builder.transform(splits[name][0]).tocsr() for name in ('train','val')}
    out = {}
    for name in ('train','val'):
        t, s = tfidf[name], vectors[name]
        if t.shape != (len(s), 10000) or not sp.isspmatrix_csr(t):
            raise RuntimeError('TF-IDF shape or sparse type mismatch')
        norms = np.sqrt(np.asarray(t.multiply(t).sum(axis=1)).ravel())
        if not np.allclose(norms[norms > 0],1,atol=1e-5):
            raise RuntimeError('TF-IDF block normalization mismatch')
        out[name] = sp.hstack((t.astype(np.float32), sp.csr_matrix(s)), format='csr', dtype=np.float32)
        if out[name].shape != (len(s),10512):
            raise RuntimeError('Fusion shape mismatch')
    return out['train'], out['val'], {'scheme':'f6b_plus_bge', 'input_dim':10512,
        'tfidf_dim':10000,'semantic_dim':512,'tfidf_processor_sha256':digest(BASE/'feature_builder.pkl'),
        'normalization':'F6-B selected concatenated TF-IDF row L2; BGE CLS row L2; concatenate at unit block weights, no post-fusion normalization'}

def train(groups, seeds):
    splits, baseline = splits_and_baseline()
    vectors, manifest = load_vectors(splits)
    for group in groups:
        for seed in seeds:
            if (HERE/'runs'/f'{group}_s{seed}').exists():
                raise FileExistsError(f'Refusing to overwrite {group} seed {seed}')
    for group in groups:
        t0 = time.perf_counter()
        X_train, X_val, feature = make_features(group, splits, vectors, baseline)
        transform_seconds = time.perf_counter() - t0
        _, y_train, _ = splits['train']
        _, y_val, frame = splits['val']
        for seed in seeds:
            print(f'=== {group} seed={seed}; train={X_train.shape}, validation={X_val.shape} ===', flush=True)
            t0 = time.perf_counter()
            result = train_bp_model(X_train, y_train, X_val, y_val, seed=seed,
                train_params=dict(baseline['training']['params']),
                bp_params={'hidden_dim':64,'dropout':.2,'input_dropout':0}, model_type='bp',
                log_fn=lambda s: print(s,flush=True))
            run_seconds = time.perf_counter()-t0
            expected = bp_parameter_count(X_train.shape[1])
            if result['architecture']['n_parameters'] != expected:
                raise RuntimeError('BP parameter count mismatch')
            out = HERE/'runs'/f'{group}_s{seed}'
            out.mkdir(parents=True, exist_ok=False)
            save_history(result['history'], out/'history.csv')
            save_confusion(result['val_metrics'], out/'confusion_val.csv')
            plot_confusion_matrix(result['val_metrics']['confusion_matrix'], C.LABELS,
                f'{group} seed={seed}', out/'confusion_val.png')
            plot_training_curves(result['history'], f'{group} seed={seed}', out/'training_curve.png',
                best_epoch=result['best_epoch'], stop_epoch=result['epochs_run'])
            output = frame.copy()
            output['predicted_category'] = [C.LABELS[i] for i in predict(result['model'], X_val)]
            output.to_csv(out/'val_predictions.csv', index=False, encoding='utf-8-sig')
            torch.save(result['model'].state_dict(), out/'model.pt')
            payload = {'run_id':out.name, 'seed':seed, 'created_at':now_str(),
                'data_sha256':{'train':TRAIN_SHA,'val':VAL_SHA}, 'feature':feature,
                'semantic_cache_manifest_sha256':digest(CACHE/'manifest.json'), 'model':result['architecture'],
                'frozen_encoder_parameters':manifest['frozen_encoder_parameters'],
                'training':{'params':result['train_params'],'best_epoch':result['best_epoch'],
                    'epochs_run':result['epochs_run'], 'train_time':result['train_time'],
                    'training_wall_seconds':run_seconds, 'cached_feature_transform_seconds':transform_seconds,
                    'selection_metric':'f1_macro'}, 'val':result['val_metrics'], 'test':None}
            save_json(out/'metrics.json', payload)
            print(f"{group} seed={seed}: validation Macro-F1={result['val_metrics']['f1_macro']:.6f}; saved {out}", flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('encode','train'))
    p.add_argument('--groups', nargs='+', choices=('F8_1','F8_2'), default=['F8_1','F8_2'])
    p.add_argument('--seeds', nargs='+', type=int, choices=range(42,47), default=[42])
    args=p.parse_args()
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('duplicate groups or seeds')
    if args.action == 'encode': encode()
    else: train(args.groups,args.seeds)

if __name__ == '__main__':
    main()

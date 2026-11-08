"""Run the fixed F.7 factorial through the existing run_single training path."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import model_config as C
from models.data import load_splits, load_training_override
from train_bp import run_single

TRAIN = ROOT / 'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv'
VAL = ROOT / 'dataset/splits/val.csv'
BASE = ROOT / 'results/member_C/f6_experiments/runs/F6_B_s42/metrics.json'
TRAIN_HASH = '7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
VAL_HASH = '9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0'
GROUPS = {'F7_1': (256, .0003), 'F7_2': (64, .001), 'F7_3': (256, .001)}

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    p.add_argument('--seeds', nargs='+', type=int, choices=range(42, 47), default=list(range(42, 47)))
    args = p.parse_args()
    if len(set(args.groups)) != len(args.groups) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Duplicate groups or seeds')
    if digest(TRAIN) != TRAIN_HASH or digest(VAL) != VAL_HASH:
        raise RuntimeError('Frozen data SHA-256 mismatch')
    baseline = json.loads(BASE.read_text(encoding='utf8'))
    feature = baseline['feature']
    arch = baseline['model']
    if (baseline['provenance']['split_sha256'] != {'train': TRAIN_HASH, 'val': VAL_HASH}
        or feature['word']['selected_features'] != 7000 or feature['char']['selected_features'] != 3000
        or feature['fit_rows'] != 7531 or arch['hidden_dim'] != 64 or arch['dropout'] != .2
        or arch.get('input_dropout', 0) != 0 or baseline['training']['params']['lr'] != .0003):
        raise RuntimeError('F6-B baseline configuration mismatch')
    if C.experiment_by_id('F6_B').get('word_k') != 7000 or C.experiment_by_id('F6_B').get('char_k') != 3000:
        raise RuntimeError('F6-B feature configuration changed')
    # A changed model, trainer, feature pipeline or entry point invalidates direct reuse.
    for name in ('models/bp.py', 'models/data.py', 'models/features.py', 'models/f_features.py',
                 'models/trainer.py', 'models/metrics.py', 'train_bp.py', 'model_config.py'):
        candidates = {str(ROOT / name), name, name.replace('/', '\\')}
        recorded = baseline['provenance']['source_sha256']
        old = next((recorded[k] for k in candidates if k in recorded), None)
        if old is None or digest(ROOT / name) != old:
            raise RuntimeError(f'F6-B source mismatch: {name}')
    for group in args.groups:
        for seed in args.seeds:
            path = HERE / 'runs' / f'{group}_s{seed}'
            if path.exists():
                raise FileExistsError(f'Refusing to overwrite {path}')
    splits = load_splits(verify=True, splits=('train', 'val'))
    splits['train'] = load_training_override(TRAIN, TRAIN_HASH, splits['train'])
    for group in args.groups:
        hidden, lr = GROUPS[group]
        exp = dict(C.experiment_by_id('F6_B'), id=group, label=f'{group} F6-B features')
        params = dict(baseline['training']['params'], lr=lr)
        overrides = {'model_type': 'bp', 'train_params': params,
                     'bp_params': {'hidden_dim': hidden, 'dropout': .2, 'input_dropout': 0.},
                     'hidden_dim': hidden, 'dropout': .2,
                     'train_file': TRAIN, 'train_sha256': TRAIN_HASH}
        for seed in args.seeds:
            print(f'F7 starting {group} seed {seed}', flush=True)
            run_single(exp, seed, splits, overrides, output_root=HERE / 'runs')

if __name__ == '__main__':
    main()

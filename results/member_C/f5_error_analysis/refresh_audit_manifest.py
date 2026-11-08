"""Reconcile fresh F.5 report and input digests after editorial verification."""
from hashlib import sha256
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
dest=HERE/'audit_manifest.json'
m=json.loads(dest.read_text(encoding='utf-8'))
digest=lambda p:sha256(p.read_bytes()).hexdigest()
paths=[ROOT/'dataset/splits/train.csv',ROOT/'dataset/splits/val.csv',
       ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv',
       ROOT/'results/member_C/f2_formal_word_char/F_word_char_s42/feature_builder.pkl',
       ROOT/'results/member_C/f3_formal_a_reviewed/F_word_char_s42/feature_builder.pkl',
       ROOT/'models/features.py',ROOT/'models/f_features.py']
for seed in range(42,47):
    paths.append(ROOT/f'results/member_C/f2_formal_word_char/F_word_char_s{seed}/val_predictions.csv')
    paths.append((ROOT/'results/member_C/f3_formal_a_reviewed/F_word_char_s42' if seed==42 else
                  ROOT/f'results/member_C/f3_analysis/f3_a_multiseed/runs/F_word_char_s{seed}')/'val_predictions.csv')
m['input_file_sha256']={str(p.relative_to(ROOT)):digest(p) for p in paths}
for name in m['outputs_sha256']:m['outputs_sha256'][name]=digest(HERE/name)
m['analysis_scripts_sha256']={p.name:digest(p) for p in sorted(HERE.glob('*.py'))}
dest.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
print('Updated local F.5 evidence manifest:',len(paths),'inputs,',len(m['outputs_sha256']),'outputs')

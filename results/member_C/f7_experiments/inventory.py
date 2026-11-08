"""Inspect run metadata without accessing test-set predictions or metrics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRAIN_HASH = '7631c1433e411af51b6d01509739ec716987d028cd785392cdbd951915032880'
VAL_HASH = '9113aca8af798ec25d7c8f0062a00603b4fcf2e9afdb42077f8ea0d7ac68d2d0'
GROUPS = {(64, .0003): 'F7-0', (256, .0003): 'F7-1',
          (64, .001): 'F7-2', (256, .001): 'F7-3'}

def main():
    for p in (ROOT / 'results').rglob('metrics.json'):
        try:
            m = json.loads(p.read_text(encoding='utf8'))
            feat, model = m['feature'], m['model']
            split = m['provenance']['split_sha256']
            pair = (model['hidden_dim'], m['training']['params']['lr'])
            if (pair not in GROUPS or m.get('seed') not in range(42, 47)
                or split.get('train') != TRAIN_HASH or split.get('val') != VAL_HASH
                or model.get('model_type', 'bp') != 'bp' or feat.get('scheme') != 'F_word_char'
                or (feat.get('word') or {}).get('selected_features') != 7000
                or (feat.get('char') or {}).get('selected_features') != 3000):
                continue
            print(GROUPS[pair], m['seed'], p.relative_to(ROOT))
        except (KeyError, TypeError, OSError, ValueError):
            continue

if __name__ == '__main__':
    main()

"""Recheck education candidates and list baseline conflicts without using test labels."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SRC = ROOT / 'results/member_C/f3_recovery/audit_recovery.py'
spec = importlib.util.spec_from_file_location('audit_recovery', SRC)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def main():
    out = Path(__file__).resolve().parent
    source = ROOT / 'results/member_C/f3_recovery/analysis/candidate_manifest.csv'
    version_path = ROOT / 'dataset/dataset_version.json'
    version = json.loads(version_path.read_text(encoding='utf-8'))
    fields = ['id', 'title', 'url', 'publish_time']
    refs = {}
    for split in ('train', 'val', 'test'):
        path = ROOT / 'dataset/splits' / (split + '.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == version['files'][split+'.csv']['sha256']
        refs[split] = audit.enrich(audit.read(path, fields + (['category'] if split == 'train' else [])))
    c = audit.enrich(audit.read(source))
    education = c[c.category == '\u6559\u80b2'].copy()
    assert len(education) == 371 and c.id.is_unique
    counts = {}
    for split, ref in refs.items():
        idx = audit.NearIndex()
        for title in ref._title_key:
            idx.add(title)
        counts[split] = {
            'id_exact': int(education.id.isin(set(ref.id)).sum()),
            'url_exact': int(education._url_key.isin(set(ref._url_key)).sum()),
            'normalized_title_exact': int(education._title_key.isin(set(ref._title_key)).sum()),
            'near_rows': sum(bool(idx.matches(t)) for t in education._title_key),
        }
    idx = audit.NearIndex()
    for title in c._title_key:
        idx.add(title)
    all_id = c.id.tolist()
    internal = sum(any(all_id[pos] != row.id for pos, _ in idx.matches(row['_title_key']))
                   for _, row in education.iterrows())
    train = refs['train'].reset_index(drop=True)
    idx = audit.NearIndex()
    for t in train._title_key:
        idx.add(t)
    pairs = []
    for i, row in train.iterrows():
        for j, score in idx.matches(row['_title_key']):
            if i < j:
                other = train.iloc[j]
                pairs.append({'id_a': row.id, 'title_a': row.title, 'url_a': row.url, 'category_a': row.category,
                              'id_b': other.id, 'title_b': other.title, 'url_b': other.url, 'category_b': other.category,
                              'jaccard': score})
    pd.DataFrame(pairs, columns=['id_a', 'title_a', 'url_a', 'category_a', 'id_b', 'title_b', 'url_b', 'category_b', 'jaccard']).to_csv(
        out / 'baseline_near_conflicts.csv', index=False, encoding='utf-8-sig')
    result = {'education_rows': len(education), 'education_against_splits': counts,
              'education_near_other_candidates': int(internal), 'baseline_near_pairs': len(pairs),
              'baseline_near_involved': len({x[k] for x in pairs for k in ('id_a', 'id_b')}),
              'candidate_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'rule': 'normalized character trigram-set Jaccard >= 0.8',
              'test_fields': fields}
    (out / 'duplicate_recheck.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

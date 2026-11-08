"""Flag possible same-event titles more conservatively than the prior 0.8 threshold."""
from collections import Counter
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
src = ROOT / 'results/member_C/f3_recovery/audit_recovery.py'
spec = importlib.util.spec_from_file_location('prior_audit', src)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def matches(index, text, threshold=.5):
    target = audit.grams(text)
    candidates = Counter(pos for g in target for pos in index.index.get(g, ()))
    result = []
    for pos, intersection in candidates.items():
        score = intersection / (len(target) + len(index.values[pos]) - intersection)
        if score >= threshold:
            result.append((pos, score))
    return result


def main():
    out = Path(__file__).resolve().parent
    dest = out / 'f3_event_screen.csv'
    if dest.exists():
        raise FileExistsError(dest)
    q = audit.enrich(pd.read_csv(out / 'f3_title_review_queue.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')).reset_index(drop=True)
    flags, pairs = set(), []
    fields = ['id', 'title', 'url', 'publish_time']
    for split in ('train', 'val', 'test'):
        ref = audit.enrich(audit.read(ROOT / 'dataset/splits' / (split + '.csv'), fields)).reset_index(drop=True)
        idx = audit.NearIndex()
        for t in ref._title_key:
            idx.add(t)
        for _, row in q.iterrows():
            for pos, score in matches(idx, row['_title_key']):
                r = ref.iloc[pos]
                flags.add(row.id)
                pairs.append({'candidate_id': row.id, 'candidate_title': row.title, 'reference_split': split,
                              'reference_id': r.id, 'reference_title': r.title, 'jaccard': score})
    idx = audit.NearIndex()
    for i, row in q.iterrows():
        for pos, score in matches(idx, row['_title_key']):
            r = q.iloc[pos]
            flags.update((row.id, r.id))
            pairs.append({'candidate_id': row.id, 'candidate_title': row.title, 'reference_split': 'review_queue',
                          'reference_id': r.id, 'reference_title': r.title, 'jaccard': score})
        idx.add(row['_title_key'])
    pd.DataFrame(pairs, columns=['candidate_id', 'candidate_title', 'reference_split', 'reference_id', 'reference_title', 'jaccard']).to_csv(dest, index=False, encoding='utf-8-sig')
    q['possible_event_conflict'] = q.id.isin(flags)
    q.drop(columns=['_title_key', '_url_key']).to_csv(out / 'f3_title_review_with_event_flags.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'pair_rows': len(pairs), 'flagged_by_category': q.groupby('category').possible_event_conflict.sum().to_dict(),
                      'unflagged_by_category': q.groupby('category').possible_event_conflict.apply(lambda x: int((~x).sum())).to_dict()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

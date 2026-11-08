"""Build a reviewable F3-A candidate from the fixed, title-reviewed queue."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import config

spec = importlib.util.spec_from_file_location('review_queue', HERE / 'prepare_review_queue.py')
queue_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue_module)

ADDED = {'财经':166, '体育':166, '文娱':170, '教育':160, '社会':159, '国际':161}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main(output_name='f3_a_dataset', decision_name='f3_title_review_decisions.csv'):
    out = HERE / output_name
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite: {out}')
    base_path = ROOT / 'dataset/splits/train.csv'
    base = pd.read_csv(base_path, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    version = json.loads((ROOT/'dataset/dataset_version.json').read_text(encoding='utf-8'))
    assert len(base) == 6549 and base.id.is_unique and digest(base_path) == version['files']['train.csv']['sha256']
    decisions = pd.read_csv(HERE/decision_name, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    evidence = pd.read_csv(HERE/'f3_official_evidence_all.csv', dtype=str, encoding='utf-8-sig', keep_default_na=False)
    pool = decisions.merge(evidence[['id','grade','archive_source','archive_url','archive_sha256','matching_fields']],
                           on='id', validate='one_to_one')
    pool = pool[(pool.semantic_review == 'title_review_pass') & pool.grade.isin(['A','B']) &
                (pool.archive_source == 'official_page_retrieved')].copy()
    assert pool.id.is_unique and set(pool.id).isdisjoint(set(base.id))
    selected, plans = [], {}
    for category in config.TARGET_CATEGORIES:
        subset = pool[pool.category == category].copy()
        subset['month'] = subset.publish_time.str[5:7].astype(int)
        weights = base[base.category == category].publish_time.str[5:7].astype(int).value_counts().to_dict()
        available = subset.month.value_counts().to_dict()
        quotas = queue_module.allocate(ADDED[category], weights, available)
        ideal = queue_module.allocate(ADDED[category], weights, {i: ADDED[category] for i in range(1,13)})
        pieces = [subset[subset.month == month].sort_values('rank').head(count) for month,count in quotas.items()]
        chosen = pd.concat(pieces, ignore_index=True)
        assert len(chosen) == ADDED[category]
        selected.append(chosen)
        plans[category] = {'available': len(subset), 'monthly_actual': quotas, 'monthly_unconstrained': ideal}
    new = pd.concat(selected, ignore_index=True).sort_values(['category','publish_time','id']).reset_index(drop=True)
    assert len(new) == 982 and new.id.is_unique
    assert new.publish_time.between('2025-01-01 00:00:00', '2025-12-31 23:59:59').all()
    assert new.raw_category.map(config.CATEGORY_MAPPING).equals(new.category)
    train = pd.concat([base, new[base.columns]], ignore_index=True)
    assert len(train) == 7531 and train.id.is_unique and train.iloc[:len(base)].equals(base)
    out.mkdir(parents=True)
    train_path = out/'train.csv'
    train.to_csv(train_path, encoding='utf-8-sig', index=False)
    new.to_csv(out/'new_sample_evidence.csv', encoding='utf-8-sig', index=False)
    (out/'new_ids.txt').write_text('\n'.join(new.id) + '\n', encoding='utf-8')
    meta = {'name':'F3-A', 'rows':len(train), 'base_rows':len(base), 'new_rows':len(new), 'seed':42,
            'train_sha256':digest(train_path), 'base_train_sha256':digest(base_path),
            'candidate_manifest_sha256':digest(ROOT/'results/member_C/f3_recovery/analysis/candidate_manifest.csv'),
            'title_decisions_sha256':digest(HERE/decision_name),
            'official_evidence_sha256':digest(HERE/'f3_official_evidence_all.csv'),
            'new_sample_evidence_sha256':digest(out/'new_sample_evidence.csv'),
            'new_ids_sha256':digest(out/'new_ids.txt'),
            'category_counts':dict(Counter(train.category)), 'added_counts':dict(Counter(new.category)),
            'monthly':plans, 'provenance':'base original train + 2025 recovered candidate; independent archive page evidence',
            'selection':'Fixed seed 42 hash rank, title review, conservative Jaccard 0.5 event screen plus additional reviewed exclusions when present; stratified by baseline category/month; capacity-aware',
            'limitations':'title-level semantic review, possible same-event paraphrases, constrained monthly quotas'}
    (out/'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

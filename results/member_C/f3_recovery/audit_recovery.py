"""Audit a completed isolated recovery; never train or evaluate a model."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cleaner
import config

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = HERE / 'analysis'
SPLITS = ROOT / 'dataset/splits'
LABELS = config.TARGET_CATEGORIES


def title_key(title):
    return ''.join(x for x in unicodedata.normalize('NFKC', cleaner.clean_title(title)).casefold() if x.isalnum())


def url_key(url):
    u = urlsplit(url.strip())
    return urlunsplit((u.scheme.lower(), u.netloc.lower(), u.path, '', ''))


def grams(text):
    return frozenset(text[i:i+3] for i in range(len(text)-2)) if len(text) >= 3 else frozenset([text])


class NearIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.values = []

    def add(self, text):
        g = grams(text)
        index = len(self.values)
        self.values.append(g)
        for key in g:
            self.index[key].append(index)

    def matches(self, text):
        target = grams(text)
        counts = Counter(i for k in target for i in self.index.get(k, ()))
        return [(i, n / (len(target) + len(self.values[i]) - n)) for i, n in counts.items()
                if n / (len(target) + len(self.values[i]) - n) >= 0.80]


def enrich(df):
    d = df.copy()
    d['_title_key'] = d.title.map(title_key)
    d['_url_key'] = d.url.map(url_key)
    return d


def distribution(df):
    counts = df.category.value_counts()
    return {x: int(counts.get(x, 0)) for x in LABELS}


def read(path, usecols=None):
    return pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False, usecols=usecols)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, fields, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUT.exists():
        raise FileExistsError(f'Refusing to overwrite audit directory: {OUT}')
    report = json.loads((DATA / 'crawl_report.json').read_text(encoding='utf-8'))
    if report['days_requested'] != 365 or report['summary']['days_success'] != 365:
        raise RuntimeError('Full-year coverage is incomplete; inspect crawl report before analysis')
    recovered = enrich(read(DATA / 'news_clean.csv'))
    raw = read(DATA / 'news_raw.csv')
    assert recovered.publish_time.between('2025-01-01 00:00:00', '2025-12-31 23:59:59').all()
    assert recovered.category.isin(LABELS).all()
    assert recovered.raw_category.map(config.CATEGORY_MAPPING).equals(recovered.category)
    assert (recovered.publish_time.str[:10] == recovered.archive_date).all()
    train = enrich(read(SPLITS / 'train.csv'))
    held_cols = ['id', 'title', 'url', 'publish_time']
    refs = {'train': train, 'val': enrich(read(SPLITS / 'val.csv', held_cols)),
            'test': enrich(read(SPLITS / 'test.csv', held_cols))}
    version = json.loads((ROOT / 'dataset/dataset_version.json').read_text(encoding='utf-8'))
    for name in refs:
        assert digest(SPLITS / f'{name}.csv') == version['files'][f'{name}.csv']['sha256']
    OUT.mkdir(parents=True)

    print(f'Auditing {len(recovered)} cleaned rows', flush=True)
    exact_masks, overlap_counts = {}, {}
    exact_union = pd.Series(False, index=recovered.index)
    for name, ref in refs.items():
        masks = {col: recovered[col].isin(set(ref[col])) for col in ('id', '_url_key', '_title_key')}
        exact_masks[name] = masks['id'] | masks['_url_key'] | masks['_title_key']
        exact_union |= exact_masks[name]
        overlap_counts[name] = {col: int(mask.sum()) for col, mask in masks.items()}
        overlap_counts[name]['any'] = int(exact_masks[name].sum())

    # All original training rows remain untouched, including any disagreements.
    train_maps = {key: {value: idx for idx, value in enumerate(train[key])}
                  for key in ('id', '_url_key', '_title_key')}
    train_label_conflicts = []
    for _, row in recovered.loc[exact_masks['train']].iterrows():
        positions = {train_maps[key][row[key]] for key in train_maps if row[key] in train_maps[key]}
        for position in positions:
            original = train.iloc[position]
            if row.category != original.category:
                train_label_conflicts.append({'recovered_id': row.id, 'train_id': original.id,
                                              'recovered_category': row.category, 'train_category': original.category,
                                              'recovered_title': row.title, 'train_title': original.title})
    write_rows(OUT / 'train_label_conflicts.csv', ['recovered_id', 'train_id', 'recovered_category',
                                                   'train_category', 'recovered_title', 'train_title'], train_label_conflicts)

    train_by_id = train.set_index('id', drop=False)
    metadata_conflicts = []
    for _, row in recovered.loc[recovered.id.isin(set(train.id))].iterrows():
        original = train_by_id.loc[row.id]
        changed = [key for key in ('title', 'url', 'publish_time', 'category') if row[key] != original[key]]
        if changed:
            metadata_conflicts.append({'id': row.id, 'changed_fields': '|'.join(changed),
                                       'train_title': original.title, 'recovered_title': row.title,
                                       'train_time': original.publish_time, 'recovered_time': row.publish_time,
                                       'train_url': original.url, 'recovered_url': row.url})
    write_rows(OUT / 'train_metadata_conflicts.csv', ['id', 'changed_fields', 'train_title', 'recovered_title',
                                                      'train_time', 'recovered_time', 'train_url', 'recovered_url'], metadata_conflicts)

    raw_target = raw.loc[raw.category.isin(config.CATEGORY_MAPPING)].copy()
    raw_target['mapped'] = raw_target.category.map(config.CATEGORY_MAPPING)
    raw_target['clean_title'] = raw_target.title.map(cleaner.clean_title)
    raw_groups = raw_target.groupby('clean_title').agg(
        rows=('id', 'size'), mapped_count=('mapped', 'nunique'),
        categories=('mapped', lambda x: '|'.join(sorted(set(x)))))
    raw_groups = raw_groups[raw_groups.mapped_count > 1].reset_index()
    raw_groups.to_csv(OUT / 'raw_title_label_conflicts.csv', index=False, encoding='utf-8-sig')

    unseen = recovered.loc[~exact_union].copy().reset_index(drop=True)
    indices = {}
    for name, df in refs.items():
        index = NearIndex()
        for title in df._title_key:
            index.add(title)
        indices[name] = index
    candidates = []
    near_count = Counter()
    columns = ['candidate_id', 'candidate_title', 'candidate_category', 'reference_split', 'reference_id', 'reference_title', 'jaccard']
    with (OUT / 'near_reference_conflicts.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for j, row in unseen.iterrows():
            found = False
            for name, idx in indices.items():
                matches = idx.matches(row['_title_key'])
                if matches:
                    near_count[name] += 1
                    found = True
                for pos, score in matches:
                    ref = refs[name].iloc[pos]
                    writer.writerow(dict(candidate_id=row.id, candidate_title=row.title,
                                         candidate_category=row.category, reference_split=name,
                                         reference_id=ref.id, reference_title=ref.title, jaccard=f'{score:.8f}'))
            if not found:
                candidates.append(j)
            if (j + 1) % 20000 == 0:
                print(f'Reference near-duplicate scan {j+1}/{len(unseen)}', flush=True)
    filtered = unseen.loc[candidates].reset_index(drop=True)

    # Preserve all conflicts; conservatively exclude both endpoints rather than silently select one.
    index, involved, pair_count = NearIndex(), set(), 0
    with (OUT / 'near_candidate_conflicts.csv').open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id_a', 'title_a', 'id_b', 'title_b', 'jaccard'])
        writer.writeheader()
        for j, row in filtered.iterrows():
            for prev, score in index.matches(row['_title_key']):
                pair_count += 1
                involved.update((prev, j))
                writer.writerow({'id_a': filtered.at[prev, 'id'], 'title_a': filtered.at[prev, 'title'],
                                 'id_b': row.id, 'title_b': row.title, 'jaccard': f'{score:.8f}'})
            index.add(row['_title_key'])
            if (j + 1) % 20000 == 0:
                print(f'Candidate near-duplicate scan {j+1}/{len(filtered)}', flush=True)
    conservative = filtered.iloc[[i for i in range(len(filtered)) if i not in involved]].copy()
    conservative.drop(columns=['_title_key', '_url_key']).to_csv(OUT / 'candidate_manifest.csv', index=False, encoding='utf-8-sig')
    print('Candidate scan complete', flush=True)

    base_counts, available = distribution(train), distribution(conservative)
    scale = min((base_counts[c] + available[c]) / base_counts[c] for c in LABELS)
    two = all(available[c] >= base_counts[c] for c in LABELS)
    four = all(available[c] >= 3*base_counts[c] for c in LABELS)
    raw_flags = {
        'time_issues': int(raw['_time_issue'].ne('').sum()),
        'bad_publish_time': int(pd.to_datetime(raw['publish_time'], errors='coerce').isna().sum()),
        'publish_date_differs_from_archive_date': int((raw.publish_time.str[:10] != raw.archive_date).sum()),
        'duplicate_id_extra': int(raw.id.duplicated().sum()),
        'duplicate_url_extra': int(raw.url.duplicated().sum()),
        'duplicate_clean_title_extra': int(raw.title.map(cleaner.clean_title).duplicated().sum()),
    }
    stats = {
        'provenance': {'dataset_sha256': digest(DATA / 'news_clean.csv'), 'raw_sha256': digest(DATA / 'news_raw.csv'),
                       'source': config.SOURCE_NAME, 'crawl_summary': report['summary'],
                       'coverage': [report['date_start'], report['date_end']]},
        'raw': {'rows': len(raw), **raw_flags},
        'clean': {'rows': len(recovered), 'categories': distribution(recovered),
                  'date_min': recovered.publish_time.min(), 'date_max': recovered.publish_time.max(),
                  'monthly_counts': recovered.groupby(recovered.publish_time.str[:7]).size().to_dict(),
                  'duplicates': {k: int(recovered[k].duplicated().sum()) for k in ('id', 'url', 'title', '_url_key', '_title_key')}},
        'train': {'rows': len(train), 'categories': base_counts, 'hash': digest(SPLITS / 'train.csv')},
        'exact_overlap_rows': overlap_counts,
        'train_exact_label_conflicts': len(train_label_conflicts),
        'train_id_metadata_conflicts': len(metadata_conflicts),
        'raw_title_cross_category_groups': len(raw_groups),
        'train_ids_missing_in_recovered_clean': int((~train.id.isin(set(recovered.id))).sum()),
        'unseen_after_exact': len(unseen), 'near_reference_rows': dict(near_count),
        'after_reference_near': len(filtered), 'internal_near_pairs': pair_count,
        'internal_near_involved_rows': len(involved),
        'conservative_candidate_count': len(conservative), 'conservative_categories': available,
        'total_after_conservative_filter': len(train) + len(conservative),
        'balanced_2x_possible': two, 'balanced_4x_possible': four, 'balanced_scale_upper_bound': scale,
        'near_definition': 'NFKC/casefold, alphanumeric only, trigram-set Jaccard >= 0.80; conservative removal of both candidate endpoints',
        'publication_time_limitation': 'Archive list gives month/day/hour/minute; year is inferred from archive path, not verified against each article page.'
    }
    (OUT / 'audit_results.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

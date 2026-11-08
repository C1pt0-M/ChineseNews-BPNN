"""Read-only F.3 data feasibility audit. Test data is used only for duplicate screening."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import sys
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cleaner
import crawler
import model_config as C

OUT = Path(__file__).resolve().parent


def load(path, columns=None):
    return pd.read_csv(ROOT / path, encoding='utf-8-sig', dtype=str, keep_default_na=False, usecols=columns)


def title_key(s):
    return ''.join(c for c in unicodedata.normalize('NFKC', cleaner.clean_title(s)).casefold() if c.isalnum())


def url_key(s):
    p = urlsplit(s.strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, '', ''))


def add_keys(df):
    d = df.copy()
    d['title_key'] = d.title.map(title_key)
    d['url_key'] = d.url.map(url_key)
    return d


def count_dupes(df):
    return {k: int(df[k].duplicated().sum()) for k in ('id', 'url', 'url_key', 'title', 'title_key')}


def shingles(text):
    return frozenset(text[i:i+3] for i in range(len(text)-2)) if len(text) >= 3 else frozenset([text])


class NearIndex:
    def __init__(self, titles):
        self.grams = []
        self.index = defaultdict(list)
        for title in titles:
            grams = shingles(title)
            pos = len(self.grams)
            self.grams.append(grams)
            for gram in grams:
                self.index[gram].append(pos)

    def matches(self, title, threshold=.8):
        target = shingles(title)
        counts = Counter(pos for gram in target for pos in self.index.get(gram, ()))
        return [pos for pos, overlap in counts.items()
                if overlap / (len(target) + len(self.grams[pos]) - overlap) >= threshold]


def main():
    paths = {'train': 'dataset/splits/train.csv', 'val': 'dataset/splits/val.csv', 'test': 'dataset/splits/test.csv',
             'clean': 'dataset/news_clean.csv', 'week_clean': 'dataset/test_7days/news_clean.csv',
             'week_raw': 'dataset/test_7days/news_raw.csv'}
    parts = {s: add_keys(load(paths[s], ['id', 'title', 'url', 'publish_time'] + (['category'] if s == 'train' else [])))
             for s in ('train', 'val', 'test')}
    base, val, test = parts['train'], parts['val'], parts['test']
    clean = add_keys(load(paths['clean']))
    week = add_keys(load(paths['week_clean']))
    # Reapply the project's cleaners to the only locally available raw sample.
    raw = pd.read_csv(ROOT / paths['week_raw'], encoding='utf-8-sig', keep_default_na=False,
                      dtype={'id': str})
    rebuilt, clean_stats = cleaner.clean_dataframe(raw)
    rebuilt_ids = set(rebuilt.id.astype(str))
    week_ids = set(week.id)
    in_window = clean[clean.publish_time.between('2025-01-01 00:00:00', '2025-12-31 23:59:59')]
    missing_in_train = in_window[~in_window.id.isin(set(base.id))]
    missing_in_clean = base[~base.id.isin(set(in_window.id))]
    base_by_id = base.set_index('id')
    clean_by_id = in_window.set_index('id')
    columns = ['title', 'url', 'publish_time', 'category']
    identity_consistent = (not len(missing_in_train) and not len(missing_in_clean)
                           and base_by_id[columns].sort_index().equals(clean_by_id[columns].sort_index()))
    assert identity_consistent, 'Train split and time-windowed clean source do not match.'
    assert rebuilt_ids == week_ids, 'The raw seven-day file does not reproduce the clean ID set.'

    screen = week.copy()
    in_time = screen.publish_time.between('2025-01-01 00:00:00', '2025-12-31 23:59:59')
    target = screen.category.isin(C.LABELS)
    screen = screen[in_time & target].copy()
    overlaps = {}
    for name, other in parts.items():
        overlaps[name] = {field: int(screen[field].isin(set(other[field])).sum())
                          for field in ('id', 'url', 'url_key', 'title', 'title_key')}
    exact = pd.Series(False, index=screen.index)
    for other in parts.values():
        for field in ('id', 'url_key', 'title_key'):
            exact |= screen[field].isin(set(other[field]))
    candidates = screen[~exact].copy()
    references = {name: NearIndex(d.title_key) for name, d in parts.items()}
    near_flags = {name: [bool(index.matches(t)) for t in candidates.title_key]
                  for name, index in references.items()}
    for name, flags in near_flags.items():
        candidates['near_' + name] = flags
    no_near = candidates[~candidates[['near_train', 'near_val', 'near_test']].any(axis=1)].copy()
    near_index = NearIndex(no_near.title_key)
    pairs = {(i, j) for i, t in enumerate(no_near.title_key) for j in near_index.matches(t) if i < j}
    in_pairs = {i for pair in pairs for i in pair}
    # Conservative deterministic upper bound: reject all rows involved in any near-duplicate pair.
    conservative = no_near.iloc[[i for i in range(len(no_near)) if i not in in_pairs]]
    baseline_heldout = {}
    for name in ('val', 'test'):
        ref = references[name]
        baseline_heldout[name] = sum(bool(ref.matches(t)) for t in base.title_key)
    base_index = references['train']
    base_pairs = {(i, j) for i, t in enumerate(base.title_key) for j in base_index.matches(t) if i < j}
    # Existing audit snapshots contain a small raw-culture sample, not the complete raw pool.
    # Its labels come from the documented raw 文化 channel, not from rule_label.
    culture_path = 'reports/audit/culture_sample.csv'
    culture = load(culture_path)
    culture_all_count = len(culture)
    culture = culture[culture.publish_time.between('2025-01-01 00:00:00', '2025-12-31 23:59:59')].copy()
    culture['id'] = culture.url.map(lambda u: 'cn' + crawler.extract_article_id(u) if crawler.extract_article_id(u) else '')
    culture['article'] = culture.url.map(crawler.is_article_url)
    culture = add_keys(culture)
    evidence = culture[culture.article & culture.id.ne('') & culture.title_key.ne('')].copy()
    before = len(evidence)
    prior = [*parts.values(), week]
    already = pd.Series(False, index=evidence.index)
    for other in prior:
        for field in ('id', 'url_key', 'title_key'):
            already |= evidence[field].isin(set(other[field]))
    evidence = evidence[~already].copy()
    evidence_refs = [*references.values(), NearIndex(week.title_key)]
    audit_near = [any(ref.matches(t) for ref in evidence_refs) for t in evidence.title_key]
    remaining_culture = evidence.loc[[not x for x in audit_near]].copy()
    audit_self_index = NearIndex(remaining_culture.title_key)
    audit_pairs = {(i, j) for i, t in enumerate(remaining_culture.title_key)
                   for j in audit_self_index.matches(t) if i < j}
    prefix_cross = {}
    for name in ('val', 'test'):
        prefix_cross[name] = int(base.title.str[:8].isin(set(parts[name].title.str[:8])).sum())
    def distribution(d):
        return {c: int((d.category == c).sum()) for c in C.LABELS}
    digest = {name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for name, path in paths.items()}
    digest['culture_audit_sample'] = hashlib.sha256((ROOT / culture_path).read_bytes()).hexdigest()
    version = json.loads((ROOT / 'dataset/dataset_version.json').read_text(encoding='utf-8'))
    for key in ('train', 'val', 'test'):
        assert digest[key] == version['files'][key + '.csv']['sha256'], f'{key} hash mismatch'
    assert digest['clean'] == version['files']['news_clean.csv']['sha256'], 'clean hash mismatch'
    result = {
        'scope': 'local project only; held-out files used solely for id/url/title/time duplicate screening',
        'sha256': digest,
        'base': {'rows': len(base), 'date_min': base.publish_time.min(), 'date_max': base.publish_time.max(),
                 'categories': distribution(base), 'duplicates_extra_rows': count_dupes(base)},
        'clean_2025_rows': len(in_window), 'clean_2025_not_in_train': len(missing_in_train),
        'train_not_in_clean_2025': len(missing_in_clean), 'train_source_fields_match': identity_consistent,
        'week': {'raw_rows': len(raw), 'reconstructed_clean_rows': len(rebuilt), 'clean_rows': len(week),
                 'raw_clean_reproduces_id_set': rebuilt_ids == week_ids,
                 'date_min': week.publish_time.min(), 'date_max': week.publish_time.max(),
                 'categories': distribution(week), 'clean_stats': clean_stats,
                 'duplicates_extra_rows': count_dupes(week)},
        'exact_overlap_rows_by_key': overlaps,
        'excluded_any_exact_identity_rows': int(exact.sum()),
        'unseen_exact_candidate_count': len(candidates), 'unseen_exact_candidate_categories': distribution(candidates),
        'near_definition': 'NFKC casefold; retain Unicode alphanumeric; set of character trigrams; Jaccard >= 0.80',
        'candidate_near_rows_by_reference': {name: int(sum(flags)) for name, flags in near_flags.items()},
        'candidate_count_after_reference_near_screen': len(no_near),
        'candidate_categories_after_reference_near_screen': distribution(no_near),
        'within_candidate_near_pairs': len(pairs), 'within_candidate_near_involved_rows': len(in_pairs),
        'conservative_available_count': len(conservative), 'conservative_available_categories': distribution(conservative),
        'upper_bound_without_near_screen': len(candidates),
        'max_total_before_near_screen': len(base) + len(candidates),
        'base_train_rows_near_heldout': baseline_heldout,
        'base_train_near_internal_pairs': len(base_pairs),
        'base_train_near_internal_involved_rows': len({i for pair in base_pairs for i in pair}),
        'base_train_rows_sharing_exact_8char_prefix_with_heldout': prefix_cross,
        'raw_culture_audit_sample': {
            'all_source_rows': culture_all_count, 'rows_in_2025': len(culture), 'article_url_and_id_rows_in_2025': before,
            'remaining_after_exact_screen_against_splits_and_week': len(evidence),
            'remaining_after_near_screen_against_splits_and_week': len(remaining_culture),
            'remaining_near_internal_pairs': len(audit_pairs),
            'qualification': 'provisional; raw snapshot lacks the full raw pool and original row-level cleaning flags',
        },
    }
    (OUT / 'audit_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Match candidates to independently retrieved official daily pages and retained source slices."""
from collections import defaultdict
from datetime import date
from hashlib import sha256
import gzip
import json
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cleaner
import config
import crawler

OUT = Path(__file__).resolve().parent
PAGES = OUT / 'official_archive_recheck'
RECOVERY = ROOT / 'results/member_C/f3_recovery'
ONLINE = ROOT / 'results/member_C/f3_experiments/education_online_evidence.jsonl'
MINUTE_RE = re.compile(r'^\d{1,2}\s*-\s*\d{1,2}\s+\d{1,2}:\d{2}$')


def check_record(candidate, records, page_day):
    """Return matching rows and conflicts; ambiguous duplicates never auto-pass."""
    possible = [r for r in records if r['url'] == candidate['url'] or r['id'] == candidate['id']]
    accepted, conflicts = [], []
    for r in possible:
        fields = {
            'article_identity': r['url'] == candidate['url'] or (bool(r['id']) and r['id'] == candidate['id']),
            'no_identity_conflict': r['url'] == candidate['url'] and r['id'] == candidate['id'],
            'title': cleaner.clean_title(r['title']) == cleaner.clean_title(candidate['title']),
            'category': config.CATEGORY_MAPPING.get(r['category']) == candidate['category'] and r['category'] == candidate['raw_category'],
            'minute': bool(MINUTE_RE.fullmatch(r.get('_raw_time', '').strip())) and
                      r.get('_time_issue', '') == '' and r['publish_time'][:16] == candidate['publish_time'][:16],
            'archive_date': r['archive_date'] == page_day == candidate['archive_date'] and page_day.startswith('2025-'),
        }
        if all(fields.values()):
            accepted.append((r, fields))
        else:
            conflicts.append((r, fields))
    return accepted, conflicts


def main():
    dest = OUT / 'f3_official_evidence_all.csv'
    education_dest = OUT / 'f3_official_evidence_education.csv'
    if dest.exists() or education_dest.exists():
        raise FileExistsError('Evidence output already exists; do not overwrite')
    candidate = pd.read_csv(RECOVERY / 'analysis/candidate_manifest.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    online = [json.loads(line) for line in ONLINE.read_text(encoding='utf-8').splitlines()]
    a_results = {r['id']: r for r in online if r['status'] == 'matched' and r.get('evidence')}
    grouped = {k: g.to_dict('records') for k, g in candidate.groupby('archive_date')}
    output = []
    for day, rows in sorted(grouped.items()):
        expected_url = crawler.build_archive_url(date.fromisoformat(day))
        p, meta = PAGES / (day + '.html.gz'), PAGES / (day + '.json')
        source = 'none'
        records = []
        source_hash = ''
        if p.exists() and meta.exists():
            m = json.loads(meta.read_text(encoding='utf-8'))
            html = gzip.decompress(p.read_bytes())
            assert m['requested_url'] == m['final_url'] == expected_url and m['http_status'] == 200
            assert m['html_sha256'] == sha256(html).hexdigest()
            parsed = crawler.parse_archive_page(html.decode('utf-8'), date.fromisoformat(day), expected_url)
            assert parsed.container_found and not parsed.has_pagination
            records, source, source_hash = parsed.records, 'official_page_retrieved', m['html_sha256']
        else:
            path = RECOVERY / 'state/raw_parts' / (day + '.jsonl')
            lines = path.read_text(encoding='utf-8').splitlines()
            m = json.loads(lines[0])['_meta']
            assert m['url'] == expected_url and m['archive_date'] == day and m['result'] == 'ok' and m['status'] == 200
            records, source, source_hash = [json.loads(x) for x in lines[1:]], 'retained_parsed_slice_not_independent', sha256(path.read_bytes()).hexdigest()
        lookup = defaultdict(list)
        for r in records:
            lookup[('id', r['id'])].append(r)
            lookup[('url', r['url'])].append(r)
        for row in rows:
            possible = {id(r): r for r in lookup[('id', row['id'])] + lookup[('url', row['url'])]}.values()
            matched, conflicts = check_record(row, possible, day)
            a = a_results.get(row['id'])
            a_ok = bool(a and a['url'] == row['url'] and a['expected'] == row['publish_time'])
            b_ok = len(matched) == 1 and not conflicts
            if conflicts or len(matched) > 1:
                grade, reason = 'unverified', 'conflicting or ambiguous archive entries'
            elif a_ok:
                grade, reason = 'A', 'official article indexed timestamp matches'
            elif b_ok:
                grade, reason = 'B', 'official archive fields match'
            else:
                grade, reason = 'unverified', 'no matching official archive entry'
            r = matched[0][0] if b_ok else {}
            output.append({'id': row['id'], 'title': row['title'], 'category': row['category'],
                           'raw_category': row['raw_category'], 'publish_time': row['publish_time'],
                           'article_url': row['url'], 'archive_url': expected_url,
                           'grade': grade, 'reason': reason, 'archive_source': source,
                           'archive_sha256': source_hash, 'official_title': r.get('title', ''),
                           'official_category': r.get('category', ''), 'official_raw_time': r.get('_raw_time', ''),
                           'article_timestamp_evidence': a.get('evidence', '') if a_ok else '',
                           'matching_fields': 'article URL and ID; normalized title; raw and mapped category; exact minute; date' if b_ok else '',
                           'category_semantic_approval': 'pending'})
    df = pd.DataFrame(output)
    assert len(df) == len(candidate) and df.id.is_unique
    df.to_csv(dest, index=False, encoding='utf-8-sig')
    edu = df[df.category == '\u6559\u80b2']
    assert len(edu) == 371
    edu.to_csv(education_dest, index=False, encoding='utf-8-sig')
    print('all', len(df), df.grade.value_counts().to_dict(), 'education', edu.grade.value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()

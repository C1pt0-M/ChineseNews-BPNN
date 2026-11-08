"""Read-only candidate provenance and stratified title review; no training or test labels."""
from pathlib import Path
import json
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import config
import cleaner

RECOVERY = ROOT / 'results/member_C/f3_recovery'
OUT = Path(__file__).resolve().parent
EDU = '\u6559\u80b2'
EXPECTED = {'\u8d22\u7ecf', '\u4f53\u80b2', '\u6587\u5a31', EDU, '\u793e\u4f1a', '\u56fd\u9645'}


def main():
    out1, out2, out3 = [OUT / n for n in ('education_evidence.csv', 'other_categories_sample.csv', 'quality_counts.json')]
    candidate = pd.read_csv(RECOVERY / 'analysis/candidate_manifest.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    raw = pd.read_csv(RECOVERY / 'data/news_raw.csv', dtype=str, keep_default_na=False, encoding='utf-8-sig')
    keys = ['id', 'url', 'publish_time', 'archive_date']
    fields = keys + ['title', 'category', 'source', '_raw_category', '_raw_time', '_time_issue', '_page_url', '_is_article']
    raw = raw[fields].rename(columns={'title': 'raw_title', 'category': 'raw_label', 'source': 'raw_source'})
    merged = candidate.merge(raw, how='left', on=keys, validate='one_to_one', indicator=True)
    assert len(merged) == len(candidate) == 90742
    merged['raw_record_matches'] = ((merged['_merge'] == 'both') &
                                    (merged.raw_title.map(cleaner.clean_title) == merged.title) &
                                    (merged.raw_label == merged.raw_category) &
                                    (merged.raw_source == merged.source) &
                                    (merged['_is_article'].str.lower() == 'true') &
                                    (merged['_time_issue'] == ''))
    merged['mapping_matches'] = merged.raw_category.map(config.CATEGORY_MAPPING) == merged.category
    url_dates = merged.url.str.extract(r'/((?:19|20)\d{2})/(\d{2})-(\d{2})/')[0].fillna('')
    parsed = merged.url.str.extract(r'/((?:19|20)\d{2})/(\d{2})-(\d{2})/')
    merged['url_date'] = parsed.fillna('').agg('-'.join, axis=1).where(url_dates.ne(''), '')
    merged['url_date_matches_archive'] = merged.url_date.eq(merged.archive_date) & merged.url_date.ne('')
    merged['month'] = merged.publish_time.str[:7]
    online_path = OUT / 'education_online_evidence.jsonl'
    online = [json.loads(x) for x in online_path.read_text(encoding='utf-8').splitlines()] if online_path.exists() else []
    by_id = {x['id']: x for x in online}
    if online:
        assert len(online) == len(by_id) == 371
    def page_status(row):
        item = by_id.get(row.id, {})
        if item.get('url') != row.url or item.get('expected') != row.publish_time:
            return False
        return item.get('status') == 'matched' and bool(item.get('evidence'))
    merged['article_page_date_verified'] = merged.apply(page_status, axis=1)
    merged['article_page_status'] = merged.article_page_date_verified.map(
        {True: 'official indexed page date and minute match', False: 'no verified indexed page date; direct access unavailable'})
    merged['category_review_status'] = 'original channel label; semantic review pending'
    education = merged[merged.category == EDU].copy()
    assert len(education) == 371 and set(merged.category) == EXPECTED
    cols = ['id', 'title', 'category', 'raw_category', 'source', 'publish_time', 'archive_date', 'url',
            'raw_record_matches', 'mapping_matches', '_raw_category', '_raw_time', '_page_url', 'url_date',
            'url_date_matches_archive', 'article_page_date_verified', 'article_page_status', 'category_review_status']
    education[cols].to_csv(out1, index=False, encoding='utf-8-sig')
    others = merged[merged.category != EDU].copy()
    sample = others.groupby(['category', 'month'], group_keys=False).sample(n=1, random_state=42)
    sample.sort_values(['category', 'month'])[cols].to_csv(out2, index=False, encoding='utf-8-sig')
    counts = {
        'candidate_rows': len(merged), 'education_rows': len(education),
        'candidate_raw_record_mismatch': int((~merged.raw_record_matches).sum()),
        'candidate_mapping_mismatch': int((~merged.mapping_matches).sum()),
        'education_url_date_match': int(education.url_date_matches_archive.sum()),
        'education_url_date_mismatch_or_missing': int((~education.url_date_matches_archive).sum()),
        'education_independently_verified_article_date': int(education.article_page_date_verified.sum()),
        'education_semantically_reviewed_and_approved': 0,
        'monthly_candidate_counts': {k: {m: int(v) for m, v in g.groupby('month').size().items()}
                                     for k, g in merged.groupby('category')},
        'other_stratified_sample_rows': len(sample),
        'note': 'URL path date is corroborating evidence only, not proof of publication date; approval requires independent review.',
    }
    out3.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

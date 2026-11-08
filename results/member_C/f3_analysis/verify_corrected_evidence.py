"""Verify every v3 addition directly against the saved official HTML snapshot."""
from datetime import date
from hashlib import sha256
import gzip
import importlib.util
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('official_evidence', HERE/'compile_official_evidence.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def main():
    ds = HERE/'f3_a_dataset_v3'
    m = json.loads((ds/'manifest.json').read_text(encoding='utf-8'))
    new = pd.read_csv(ds/'new_sample_evidence.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    count = 0
    for day, group in new.groupby('archive_date'):
        path, meta = HERE/'official_archive_recheck'/f'{day}.html.gz', HERE/'official_archive_recheck'/f'{day}.json'
        payload = gzip.decompress(path.read_bytes())
        info = json.loads(meta.read_text(encoding='utf-8'))
        assert sha256(payload).hexdigest() == info['html_sha256']
        expected = official.crawler.build_archive_url(date.fromisoformat(day))
        assert info['requested_url'] == info['final_url'] == expected and info['http_status'] == 200
        page = official.crawler.parse_archive_page(payload.decode('utf-8'),date.fromisoformat(day),expected)
        assert page.container_found and not page.has_pagination
        for row in group.to_dict('records'):
            matches, conflicts = official.check_record(row, page.records, day)
            assert len(matches) == 1 and not conflicts, row['id']
            assert row['archive_sha256'] == info['html_sha256']
            count += 1
    assert count == 982 and sha256((ds/'train.csv').read_bytes()).hexdigest() == m['train_sha256']
    output = {'selected_count':count, 'archive_days':int(new.archive_date.nunique()),
              'train_sha256':m['train_sha256'], 'test_labels_used':False,
              'evidence':'each selected record re-parsed from hash-verified official HTML and matched URL, ID, title, category and minute'}
    (HERE/'f3_a_v3_evidence_verification.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(output,ensure_ascii=False))


if __name__ == '__main__':
    main()

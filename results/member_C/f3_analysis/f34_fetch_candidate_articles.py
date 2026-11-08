"""Retrieve the v3 candidate articles politely for an additional source audit.

Stops on access restrictions, never replaces prior responses, and can resume.
"""
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import config
import crawler


def main():
    out = HERE / 'f34_official_articles'
    rows = pd.read_csv(HERE/'f3_a_f34_review_candidate_v9/new_sample_evidence.csv', dtype=str, encoding='utf-8-sig')
    flags=pd.concat([pd.read_csv(HERE/name,dtype=str,encoding='utf-8-sig') for name in
                     ('f34_v9_broad_flags.csv','f34_v9_wide_event_flags.csv')],ignore_index=True)
    ids=set(flags.candidate_id)|set(flags.reference_id)
    parts=[rows[['id','url']]]
    for split in ('train','val','test'):
        # Only article identities and URLs are read from heldout splits.
        ref=pd.read_csv(ROOT/'dataset/splits'/f'{split}.csv',usecols=['id','url'],dtype=str,encoding='utf-8-sig')
        parts.append(ref[ref.id.isin(ids)])
    rows=pd.concat(parts,ignore_index=True).drop_duplicates('id')
    rows['audit_priority']=[0 if i in ids else 1 for i in rows.id]
    rows=rows.sort_values(['audit_priority','id'])
    session = crawler.make_session()
    robots = session.get('https://www.chinanews.com/robots.txt', timeout=config.TIMEOUT)
    if robots.status_code in (401, 403, 429):
        raise RuntimeError(f'Robots request restricted: HTTP {robots.status_code}')
    robots.raise_for_status()
    parser = RobotFileParser(); parser.parse(robots.text.splitlines())
    for i, row in enumerate(rows.itertuples(), 1):
        url = row.url
        if urlsplit(url).hostname != 'www.chinanews.com' or not parser.can_fetch(config.REQUEST_HEADERS['User-Agent'], url):
            raise RuntimeError(f'Unexpected or robots-disallowed URL: {url}')
        page, meta = out/(row.id+'.html.gz'), out/(row.id+'.json')
        if page.exists() and meta.exists():
            info = json.loads(meta.read_text(encoding='utf-8'))
            assert info['url'] == url and sha256(gzip.decompress(page.read_bytes())).hexdigest() == info['sha256']
            continue
        if page.exists() or meta.exists():
            raise RuntimeError(f'Incomplete previous article files: {row.id}')
        response = session.get(url, timeout=config.TIMEOUT)
        if response.status_code in (401, 403, 429):
            raise RuntimeError(f'Access/rate limit HTTP {response.status_code}: {url}')
        response.raise_for_status()
        if urlsplit(response.url).hostname != 'www.chinanews.com':
            raise RuntimeError(f'Unexpected redirect: {response.url}')
        info = {'id':row.id,'url':url,'final_url':response.url,'status':response.status_code,
                'sha256':sha256(response.content).hexdigest(),'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
        with page.open('xb') as f: f.write(gzip.compress(response.content, mtime=0))
        with meta.open('x',encoding='utf-8') as f: json.dump(info,f,ensure_ascii=False,indent=2)
        if i % 25 == 0 or i == 1:
            print(f'v3 official articles {i}/{len(rows)}', flush=True)
        time.sleep(config.DEFAULT_DELAY + .15)
    print('V3 official article retrieval complete.',len(rows),flush=True)


if __name__ == '__main__':
    main()

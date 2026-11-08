"""Fetch only the official article pages needed to review the 23 flagged pairs."""
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import config
import crawler

HERE = Path(__file__).resolve().parent
OUT = HERE/'f34_official_articles'


def main():
    pairs = pd.read_csv(HERE/'f3_a_v3_event_review.csv',dtype=str,encoding='utf-8-sig')
    ids = set(pairs.candidate_id) | set(pairs.reference_id)
    new = pd.read_csv(HERE/'f3_a_dataset_v3/new_sample_evidence.csv',dtype=str,encoding='utf-8-sig')
    train = pd.read_csv(ROOT/'dataset/splits/train.csv',dtype=str,encoding='utf-8-sig')
    data = pd.concat([new,train],ignore_index=True)
    data = data[data.id.isin(ids)].drop_duplicates('id')
    assert set(data.id) == ids
    OUT.mkdir(parents=True,exist_ok=True)
    session = crawler.make_session()
    robots = session.get('https://www.chinanews.com/robots.txt', timeout=config.TIMEOUT)
    if robots.status_code in (401,403,429):
        raise RuntimeError(f'Robots request restricted: HTTP {robots.status_code}')
    robots.raise_for_status()
    parser = RobotFileParser(); parser.parse(robots.text.splitlines())
    robots_file = OUT/'robots.txt'
    if not robots_file.exists():
        robots_file.write_bytes(robots.content)
    for i, row in enumerate(data.itertuples(),1):
        url = row.url
        if urlsplit(url).hostname != 'www.chinanews.com' or not parser.can_fetch(config.REQUEST_HEADERS['User-Agent'],url):
            raise RuntimeError(f'Unexpected or robots-disallowed URL: {url}')
        page, meta = OUT/(row.id+'.html.gz'),OUT/(row.id+'.json')
        if page.exists() and meta.exists():
            info=json.loads(meta.read_text(encoding='utf-8'))
            assert sha256(gzip.decompress(page.read_bytes())).hexdigest() == info['sha256']
            continue
        if page.exists() or meta.exists():
            raise RuntimeError(f'Incomplete previous article files: {row.id}')
        response=session.get(url,timeout=config.TIMEOUT)
        if response.status_code in (401,403,429):
            raise RuntimeError(f'Access/rate limit HTTP {response.status_code} on {row.id}')
        response.raise_for_status()
        if urlsplit(response.url).hostname != 'www.chinanews.com':
            raise RuntimeError(f'Unexpected redirect: {response.url}')
        info={'id':row.id,'url':url,'final_url':response.url,'status':response.status_code,
              'sha256':sha256(response.content).hexdigest(),'retrieved_at_utc':datetime.now(timezone.utc).isoformat()}
        with page.open('xb') as f: f.write(gzip.compress(response.content,mtime=0))
        with meta.open('x',encoding='utf-8') as f: json.dump(info,f,ensure_ascii=False,indent=2)
        if i%5==0 or i==1: print(f'official article {i}/{len(data)}: {row.id}',flush=True)
        time.sleep(config.DEFAULT_DELAY+.15)
    print('Official event article collection finished.',len(data),flush=True)


if __name__=='__main__':
    main()

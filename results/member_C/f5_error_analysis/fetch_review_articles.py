"""Acquire official source pages for the review queue at the project crawler's rate."""
from datetime import datetime,timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import config
import crawler

rows=pd.read_csv(HERE/'review_queue.csv',dtype=str,encoding='utf-8-sig')
assert len(rows)==152 and rows.id.is_unique
out=HERE/'official_article_evidence';out.mkdir(exist_ok=True)
session=crawler.make_session()
robots=session.get('https://www.chinanews.com/robots.txt',timeout=config.TIMEOUT)
if robots.status_code in (401,403,429):raise RuntimeError('access restricted at robots.txt')
robots.raise_for_status()
parser=RobotFileParser();parser.parse(robots.text.splitlines())
robots_path=out/'robots.txt'
if not robots_path.exists():robots_path.write_bytes(robots.content)
for i,row in enumerate(rows.itertuples(),1):
    if urlsplit(row.url).hostname!='www.chinanews.com' or not parser.can_fetch(config.REQUEST_HEADERS['User-Agent'],row.url):
        raise RuntimeError('unexpected or robots-disallowed URL: '+row.url)
    page=out/f'{row.id}.html.gz';meta=out/f'{row.id}.json'
    if page.exists() and meta.exists():
        x=json.loads(meta.read_text(encoding='utf-8'))
        assert x['url']==row.url and x['sha256']==sha256(gzip.decompress(page.read_bytes())).hexdigest()
        continue
    prior=ROOT/'results/member_C/f3_analysis/f34_official_articles'
    if (prior/f'{row.id}.html.gz').exists() and (prior/f'{row.id}.json').exists():
        x=json.loads((prior/f'{row.id}.json').read_text(encoding='utf-8'))
        raw=gzip.decompress((prior/f'{row.id}.html.gz').read_bytes())
        assert x['url']==row.url and x['sha256']==sha256(raw).hexdigest()
        x['evidence_origin']='retained earlier official-page snapshot'
    else:
        r=session.get(row.url,timeout=config.TIMEOUT)
        if r.status_code in (401,403,429):raise RuntimeError(f'access/rate restriction HTTP {r.status_code}: {row.url}')
        r.raise_for_status()
        if urlsplit(r.url).hostname!='www.chinanews.com':raise RuntimeError('unexpected redirect '+r.url)
        raw=r.content
        x={'id':row.id,'url':row.url,'final_url':r.url,'status':r.status_code,'sha256':sha256(raw).hexdigest(),
           'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),'evidence_origin':'new official-page retrieval'}
        time.sleep(config.DEFAULT_DELAY+.15)
    with page.open('xb') as f:f.write(gzip.compress(raw,mtime=0))
    with meta.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
    if i%20==0 or i==1:print('official pages:',i,'/',len(rows),flush=True)
print('official page acquisition complete',len(rows),flush=True)

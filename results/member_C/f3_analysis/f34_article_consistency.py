"""Provisional checks of available official article snapshots; flags require review."""
import gzip
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
import pandas as pd
from bs4 import BeautifulSoup

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default='f3_a_f34_review_candidate')
parser.add_argument('--output',default='f34_article_consistency_provisional.csv')
args=parser.parse_args()
new=pd.read_csv(HERE/args.dataset/'new_sample_evidence.csv',dtype=str,encoding='utf-8-sig').fillna('')
rows=[]
def norm(text): return re.sub(r'[^\w\u4e00-\u9fff]', '', unicodedata.normalize('NFKC',text)).lower()
for row in new.itertuples():
    p=HERE/'f34_official_articles'/f'{row.id}.html.gz';m=p.with_name(row.id+'.json')
    if not p.exists() or not m.exists(): continue
    info=json.loads(m.read_text(encoding='utf-8'));raw=gzip.decompress(p.read_bytes())
    assert sha256(raw).hexdigest()==info['sha256']
    s=BeautifulSoup(raw,'html.parser');h=s.select_one('h1');b=s.select_one('.left_zw');t=s.select_one('.content_left_time')
    headline=h.get_text(' ',strip=True) if h else ''
    body=' '.join(b.stripped_strings) if b else ''
    # Missing numbers are only a review cue: dates, shorthand and paraphrases vary.
    nums=re.findall(r'(?<![a-zA-Z])\d+(?:\.\d+)?',unicodedata.normalize('NFKC',row.title))
    missing=[n for n in nums if n not in unicodedata.normalize('NFKC',body)]
    published=t.get_text(' ',strip=True) if t else ''
    found=re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*(\d{1,2}):(\d{2})',published)
    credited=re.search(r'来源\s*[:：]\s*(.*?)(?:\s*大字体|\s*小字体|\s*分享到|$)',published)
    article_time='-'.join([found.group(1),found.group(2).zfill(2),found.group(3).zfill(2)])+' '+found.group(4).zfill(2)+':'+found.group(5) if found else ''
    rows.append({'id':row.id,'category':row.category,'title':row.title,'url':row.url,
                 'article_title':headline,'title_matches':norm(row.title)==norm(headline),
                 'candidate_time':row.publish_time[:16],'article_time':article_time,
                 'time_matches':article_time==row.publish_time[:16],
                 'candidate_source_field':row.source,'article_credited_source':credited.group(1).strip() if credited else '',
                 'body_characters':len(body),'title_numbers_missing_in_body':'|'.join(missing),
                 'article_sha256':info['sha256']})
out=HERE/args.output
pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
r=pd.DataFrame(rows)
print('available:',len(r),'title mismatch:',int((~r.title_matches).sum()),'time mismatch or missing:',int((~r.time_matches).sum()),'no body:',int((r.body_characters==0).sum()),'missing title number:',int(r.title_numbers_missing_in_body.ne('').sum()))
print(r[(~r.title_matches)|(~r.time_matches)|(r.body_characters==0)|(r.title_numbers_missing_in_body.ne(''))][['id','category','title','candidate_time','article_time','title_numbers_missing_in_body']].to_string(index=False))

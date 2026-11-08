"""Extract traceable source metadata and brief first-paragraph review cues."""
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
from bs4 import BeautifulSoup
import pandas as pd

HERE=Path(__file__).resolve().parent
q=pd.read_csv(HERE/'review_queue.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
out=HERE/'review_source_extract.csv'
if out.exists():raise FileExistsError(out)
norm=lambda x:re.sub(r'[^\w\u4e00-\u9fff]','',unicodedata.normalize('NFKC',x)).lower()
rows=[]
for r in q.to_dict('records'):
    page=HERE/'official_article_evidence'/f'{r["id"]}.html.gz'
    meta=HERE/'official_article_evidence'/f'{r["id"]}.json'
    if not page.exists() or not meta.exists():
        rows.append({'id':r['id'],'official_status':'未取得','official_title':'','title_match':False,
                     'official_time':'','official_source':'','body_chars':0,'brief_source_text':'','sha256':''})
        continue
    m=json.loads(meta.read_text(encoding='utf-8'));raw=gzip.decompress(page.read_bytes())
    assert m['url']==r['url'] and sha256(raw).hexdigest()==m['sha256']
    soup=BeautifulSoup(raw,'html.parser');h=soup.select_one('h1');b=soup.select_one('.left_zw');t=soup.select_one('.content_left_time')
    title=h.get_text(' ',strip=True) if h else ''
    full=' '.join(b.stripped_strings) if b else ''
    timestamp=t.get_text(' ',strip=True) if t else ''
    src=re.search(r'来源\s*[:：]\s*(.*?)(?:\s*大字体|\s*小字体|\s*分享到|$)',timestamp)
    rows.append({'id':r['id'],'official_status':'已取官方页面' if b else '正文未提取','official_title':title,
                 'title_match':norm(r['title'])==norm(title),'official_time':timestamp,
                 'official_source':src.group(1).strip() if src else '',
                 'body_chars':len(full),'brief_source_text':full[:160],
                 'sha256':m['sha256'],'evidence_origin':m.get('evidence_origin','')})
source=pd.DataFrame(rows);source.to_csv(out,index=False,encoding='utf-8-sig')
print('sources:',len(source),'body available:',int((source.body_chars>0).sum()),'titles exact:',int(source.title_match.sum()))
print(source[~source.title_match][['id','official_title']].to_string(index=False))

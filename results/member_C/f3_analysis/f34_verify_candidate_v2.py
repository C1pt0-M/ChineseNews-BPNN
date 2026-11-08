"""Re-parse official day archives for every v2 provisional addition."""
from datetime import date
import argparse
from hashlib import sha256
import gzip
import importlib.util
import json
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
spec=importlib.util.spec_from_file_location('official_evidence', HERE/'compile_official_evidence.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
parser=argparse.ArgumentParser()
parser.add_argument('--dataset',default='f3_a_f34_review_candidate_v2')
parser.add_argument('--output',default='f34_candidate_v2_archive_verification.csv')
args=parser.parse_args()
ds=HERE/args.dataset
meta=json.loads((ds/'manifest.json').read_text(encoding='utf-8'))
new=pd.read_csv(ds/'new_sample_evidence.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
base=pd.read_csv(ROOT/'dataset/splits/train.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
train=pd.read_csv(ds/'train.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
assert len(base)==6549 and len(new)==982 and len(train)==7531
assert train.iloc[:len(base)].reset_index(drop=True).equals(base)
assert sha256((ds/'train.csv').read_bytes()).hexdigest()==meta['train_sha256']
verified=[]
for day,group in new.groupby('archive_date'):
    p, m = HERE/'official_archive_recheck'/f'{day}.html.gz', HERE/'official_archive_recheck'/f'{day}.json'
    raw=gzip.decompress(p.read_bytes()); info=json.loads(m.read_text(encoding='utf-8'))
    assert sha256(raw).hexdigest()==info['html_sha256']
    url=official.crawler.build_archive_url(date.fromisoformat(day))
    assert info['requested_url']==info['final_url']==url and info['http_status']==200
    page=official.crawler.parse_archive_page(raw.decode('utf-8'),date.fromisoformat(day),url)
    assert page.container_found and not page.has_pagination
    for row in group.to_dict('records'):
        matches,conflicts=official.check_record(row,page.records,day)
        assert len(matches)==1 and not conflicts and row['archive_sha256']==info['html_sha256'],row['id']
        verified.append({'id':row['id'],'category':row['category'],'date':day,'archive_url':url,
                         'archive_sha256':info['html_sha256'],'match':'url id title category minute'})
assert len(verified)==982
out=HERE/args.output
if out.exists(): raise FileExistsError(out)
pd.DataFrame(verified).to_csv(out,index=False,encoding='utf-8-sig')
print('982 official archive records verified; train SHA-256:',meta['train_sha256'])

"""Extract source text for the flagged-pair review; retain original HTML as evidence."""
import gzip
import json
from pathlib import Path
from hashlib import sha256
from bs4 import BeautifulSoup
import pandas as pd

HERE = Path(__file__).resolve().parent
PAIRS = pd.read_csv(HERE / 'f3_a_v3_event_review.csv', dtype=str, encoding='utf-8-sig')
OUT = HERE / 'f34_official_articles'
for article_id in sorted(set(PAIRS.candidate_id) | set(PAIRS.reference_id)):
    meta = json.loads((OUT / (article_id + '.json')).read_text(encoding='utf-8'))
    raw = gzip.decompress((OUT / (article_id + '.html.gz')).read_bytes())
    assert sha256(raw).hexdigest() == meta['sha256']
    soup = BeautifulSoup(raw, 'html.parser')
    body = soup.select_one('.left_zw') or soup.select_one('#content') or soup.select_one('article')
    print('\n###', article_id, meta['url'])
    print(' '.join((body or soup).stripped_strings)[:2600])

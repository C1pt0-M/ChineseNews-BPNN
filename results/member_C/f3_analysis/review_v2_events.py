"""Conservatively hold back two potentially overlapping tax-data reports."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
source = HERE / 'f3_title_review_decisions_v2.csv'
target = HERE / 'f3_title_review_decisions_v3.csv'
if target.exists():
    raise FileExistsError(target)
rows = pd.read_csv(source, dtype=str, encoding='utf-8-sig', keep_default_na=False)
for key in ('cn10499216', 'cn10499965'):
    mask = rows.id.eq(key)
    assert mask.sum() == 1
    rows.loc[mask, 'semantic_review'] = 'exclude_pending'
    rows.loc[mask, 'review_note'] = '同一前三季度税收数据发布的相关报道，事件独立性需核实。'
rows.to_csv(target, index=False, encoding='utf-8-sig')
print('Held back two related tax-data articles.')

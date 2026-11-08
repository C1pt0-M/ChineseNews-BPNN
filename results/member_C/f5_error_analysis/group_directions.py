"""Summarize modal prediction directions in each overlapping error cohort."""
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
f=pd.read_csv(HERE/'error_groups.csv',dtype=str,encoding='utf-8-sig')
totals=f.groupby('group').size().to_dict()
counts=f.groupby(['group','category','f2_mode','f3_mode'],dropna=False).size().reset_index(name='count')
counts['group_total']=counts.group.map(totals)
counts['share_of_group']=counts['count']/counts.group_total
counts['share_of_validation']=counts['count']/3184
out=HERE/'group_direction_statistics.csv'
if out.exists():raise FileExistsError(out)
counts.to_csv(out,index=False,encoding='utf-8-sig')
print('direction groups:',len(counts))

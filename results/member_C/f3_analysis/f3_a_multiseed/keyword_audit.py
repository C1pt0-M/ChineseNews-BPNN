"""Post-hoc descriptive title keyword counts; never used in model training."""
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
data=pd.read_csv(HERE/'prediction_comparison.csv',encoding='utf-8-sig')
soc=data[data.category=='社会']
pred=[f'f3_pred_{s}' for s in range(42,47)]
subsets={'五次均预测财经':soc[(soc[pred]=='财经').all(axis=1)],
         '五次均预测文娱':soc[(soc[pred]=='文娱').all(axis=1)],
         '五次均正确':soc[soc.f3_error_count==0]}
patterns={'财经主题词':'市场|产业|企业|经济|业务|亿元|消费|金融|投资|贸易',
          '文娱主题词':'文化|文博|博物馆|非遗|戏曲|戏|书|电影|文艺|音乐|展览'}
rows=[]
for group,frame in subsets.items():
    for label,pattern in patterns.items():
        n=int(frame.title.str.contains(pattern,regex=True).sum())
        rows.append({'group':group,'pattern_name':label,'pattern':pattern,'group_size':len(frame),'matched_titles':n,'proportion':n/len(frame)})
out=HERE/'title_keyword_audit.csv'
if out.exists():raise FileExistsError(out)
pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
print(pd.DataFrame(rows).to_string(index=False))

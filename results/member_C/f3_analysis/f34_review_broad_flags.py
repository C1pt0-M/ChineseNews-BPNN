"""Assess supplementary high-similarity pairs against retained source bodies."""
import gzip
import json
from hashlib import sha256
from pathlib import Path
import re
import pandas as pd
from bs4 import BeautifulSoup

HERE=Path(__file__).resolve().parent
flags=pd.read_csv(HERE/'f34_candidate_v2_broad_flags.csv',dtype=str,encoding='utf-8-sig')
facts={
 'cn10420836':'教育部与财政部 2025 特岗通知，拟招聘 2.1 万人',
 'cn10625768':'2026 特岗通知，拟招聘 8954 人并列明学历和年龄条件',
 'cn10489314':'驻印尼大使王鲁彤在雅加达举办 76 周年招待会',
 'cn10706317':'驻登巴萨总领事张志昇与巴厘省省长出席 77 周年招待会',
 'cn10398372':'2025 年第 30 届教育展与留学论坛，北京开幕，发布 2025 数据报告',
 'cn10595920':'2026 年第 31 届教育展和第 22 届论坛，计划巡回五城并发布 2026 报告',
 'cn10450271':'预警覆盖内蒙古、四川、西藏、青海；7 月 18 日至 19 日',
 'cn10700332':'预警覆盖四川、重庆、湖北、西藏；9 月 20 日至 21 日',
 'cn10480816':'盛李豪男子气步枪决赛银牌，索拉佐 255 环夺冠',
 'cn10481848':'姚千寻女子气手枪决赛银牌，埃莎·辛格 242.6 环夺冠',
 'cn10347647':'上半年教师资格笔试报名 1 月 7 日开始',
 'cn10440788':'下半年教师资格笔试报名 7 月 4 日开始',
 'cn10532449':'A 股收盘上证 3889 点、成交额约 20922 亿元',
 'cn10681921':'A 股收盘上证 3905 点、成交额约 18793 亿元',
 'cn10430827':'河北蓝色山洪预警，涉及张家口、承德部分地区',
 'cn10658310':'河北橙色山洪预警，涉及秦皇岛青龙和承德宽城等',
 'cn10488788':'第九届平遥影展向野上照代和冯小刚授荣誉',
 'cn10703513':'第十届平遥影展向许鞍华授荣誉',
 'cn10489658':'驻欧盟使团及驻比利时使馆在布鲁塞尔联合主办招待会',
 'cn10415397':'央行 4 月金融统计报告，前四个月贷款增加 10.06 万亿元',
 'cn10383194':'央行 2 月金融统计报告，前两个月贷款增加 6.14 万亿元',
 'cn10509151':'教育部办公厅 2026 年特殊类型招生通知，教学厅〔2025〕7 号',
 'cn10370852':'教育部 2025 年普通高校招生通知，教学〔2025〕1 号',
 'cn10491148':'新疆自贸试验区外贸与企业数据报道',
 'cn10487510':'新疆特色村及乡村振兴，报道吉木萨尔小分子村农民画',
 'cn10363974':'白宫及国务卿回应特朗普关于接管加沙的言论',
 'cn10364713':'哈马斯回应同一关于接管加沙的言论，涉及同一原始事件背景',
 'cn10491523':'财政部公布 2025 年 8 月新增地方政府债券 5715 亿元',
 'cn10721842':'财政部公布 2026 年 9 月新增地方政府债券 7335 亿元',
 'cn10468891':'大广高速泰赣段中巴故障，26 人滞留，救援在吉安',
 'cn10410622':'泉南高速石吉段客车故障，50 人滞留，救援在泰和',
 'cn10438537':'2025 暑期档票房超过 15 亿元，榜首为碟中谍8',
 'cn10365657':'2025 年度票房超过 150 亿元，哪吒之魔童闹海占比超过 51%',
 'cn10464063':'广东前 7 个月进出口 5.4 万亿元、增长 4.3%',
 'cn10479911':'广东前 8 个月进出口 6.21 万亿元、增长 4.2%',
 'cn10521235':'2026 届毕业生秋招：京津冀和四川等 4 场招聘活动',
 'cn10401704':'2025 届毕业生春招：水利生态环保、建筑地产等 5 场活动',
}
cache={}
def article(i):
    if i not in cache:
        d=HERE/'f34_official_articles'; info=json.loads((d/f'{i}.json').read_text(encoding='utf-8'))
        raw=gzip.decompress((d/f'{i}.html.gz').read_bytes())
        assert sha256(raw).hexdigest()==info['sha256']
        soup=BeautifulSoup(raw,'html.parser');body=soup.select_one('.left_zw')
        if body is None: raise ValueError('No body '+i)
        cache[i]=(info,' '.join(body.stripped_strings))
    return cache[i]
out=[]
for p in flags.to_dict('records'):
    a,b=p['candidate_id'],p['reference_id']; ai,ab=article(a);bi,bb=article(b)
    if '中间价' in p['candidate_title']:
        av=re.search(r'\d\.\d{4}',p['candidate_title']);bv=re.search(r'\d\.\d{4}',p['reference_title'])
        assert av and bv and av.group() in ab and bv.group() in bb and av.group()!=bv.group(),(a,b)
        fa='中国外汇交易中心公告中间价 '+av.group();fb='中国外汇交易中心公告中间价 '+bv.group()
    elif '逆回购' in p['candidate_title']:
        av=re.search(r'(\d+)亿元',p['candidate_title']);bv=re.search(r'(\d+)亿元',p['reference_title'])
        assert av and bv and av.group() in ab and bv.group() in bb and av.group()!=bv.group(),(a,b)
        fa='央行公布逆回购金额 '+av.group();fb='央行公布逆回购金额 '+bv.group()
    else:
        fa,fb=facts[a],facts[b]
    unclear=a=='cn10363974'
    out.append({**p,'assessment':'同一原始事件背景，暂排除候选' if unclear else '不同事件或不同独立公告',
                'candidate_body_fact':fa,'reference_body_fact':fb,'candidate_url':ai['url'],'reference_url':bi['url'],
                'candidate_sha256':ai['sha256'],'reference_sha256':bi['sha256'],
                'quality_note':'该候选正文未完整支持标题数字，另行排除' if a in ('cn10405444','cn10429463') else ''})
assert len(out)==95
path=HERE/'f34_candidate_v2_broad_review.csv'
if path.exists():raise FileExistsError(path)
pd.DataFrame(out).to_csv(path,index=False,encoding='utf-8-sig')
print('reviewed',len(out),'pairs; marked same-event risk',sum(r['assessment'].startswith('同一') for r in out))

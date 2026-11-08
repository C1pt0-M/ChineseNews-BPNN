"""Document official-body facts for all flagged pairs in the current review candidate."""
import gzip
import argparse
from hashlib import sha256
import json
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--version',default='v9')
args=parser.parse_args()
FACTS={
 'cn10522084':'央行公布7天期逆回购3564亿元',
 'cn10479494':'射击世界杯混合团体赛：胡凯姚千寻夺冠、彭鑫露盛李豪获银',
 'cn10402193':'北京国际电影节开幕并启动中国电影消费年，评委会亮相',
 'cn10406180':'北京电影节举行电影投融资论坛，讨论金融支持电影',
 'cn10435016':'海南三亚设立洛桑旅游大学，初设酒店管理等四专业',
 'cn10435520':'浙江宁波设立东方理工大学，陈十一任首任校长',
 'cn10376989':'哪吒电影幕后创作展在成都举行，集中展示角色手稿',
 'cn10383647':'哪吒电影全球票房达150.21亿元，进入全球票房前五',
 'cn10512406':'旅游团从广州出发，赴湾区观看篮球、乒乓球和跳水',
 'cn10432762':'四川珙县发生4.8级地震，震中北纬28.26、东经104.87',
 'cn10437942':'四川珙县发生4.5级地震，震中北纬28.41、东经104.83',
 'cn10513284':'惠州举办全运滑板比赛，进行碗池和街式半决赛',
 'cn10403807':'哈尔滨海关发布黑龙江一季度货物进出口835.7亿元',
 'cn10399198':'海关总署发布全国一季度货物进出口10.3万亿元',
 'cn10502075':'全国前三季度GDP同比增长5.2%，高技术制造业增加值增长9.6%',
 'cn10502577':'上海发布前三季度地区GDP同比增长5.5%',
 'cn10450931':'广东水利厅发布江门、茂名等地山洪风险预警',
 'cn10458467':'银川水务及气象部门发布贺兰山东麓等地山洪预警',
 'cn10498650':'红十字会从哈马斯接收四具被扣押人员遗体并移交以方',
 'cn10502157':'以方接收一具遗体，经鉴定为塔尔·海米；万斯拟访以',
 'cn10512977':'广州全运开幕式展示武术、舞狮及粤港澳文化',
 'cn10510027':'全运男子篮球成年组，澳门首次参赛负于福建',
 'cn10354936':'国家2024全年GDP初步核算1349084亿元、增长5%',
 'cn10358044':'国新办工信部发布会介绍工业增加值增长5.8%',
 'cn10514524':'广东激流回旋赛场皮艇及划艇预赛，吕陆辉等晋级',
 'cn10537262':'云顶自由式滑雪空中技巧赛，孔凡钰102.17分获女子金牌',
 'cn10533017':'云顶自由式滑雪U型场地赛，谷爱凌91.75分获女子冠军',
 'cn10516676':'全运男排U18组，浙江以3:0胜江苏夺冠',
 'cn10511834':'澳门全运乒乓球混双首轮争夺16强席位',
 'cn10517389':'全运18日赛程预告：羽毛球半决赛、排球及田径400米',
 'cn10516776':'全运17日赛程预告：张展硕游泳收官及体操决赛',
 'cn10359421':'广东17名运动员首次参加亚冬会冰上项目',
 'cn10364663':'中国邮政发行第九届亚冬会纪念邮票',
 'cn10477460':'平遥创投揭晓17个入围剧本',
 'cn10480690':'平遥一角公布最佳学生短片评审团',
 'cn10389892':'河北永清2.2级地震，震源17千米',
 'cn10393764':'日本九州附近海域6.2级地震，震源30千米',
 'cn10513221':'东莞全运举重女子45公斤级，赵金兰与赵金红包揽冠亚军',
 'cn10465691':'中国驻印尼使馆在雅加达举办抗战胜利纪念座谈会',
 'cn10457557':'探访江苏阜宁新四军军部旧址及茅草屋',
 'cn10513329':'全运11日赛程预告：游泳比赛和跳水收官',
 'cn10513883':'全运开幕式在广州点燃主火炬，报道粤港澳生活圈',
 'cn10367601':'宁夏泾源2025巡回赛148名选手，以高山滑雪大回转为主',
 'cn10568777':'宁夏泾源第三届巡回赛170余名选手，设高山及单板12小项',
 'cn10428300':'国家外汇管理局公布5月末储备32853亿美元，月增36亿美元',
 'cn10364671':'国家外汇管理局公布1月末储备32090亿美元，月增67亿美元',
}
# Reuse earlier documented body facts where possible.
for name in ('f34_pair_review.csv','f34_candidate_v2_broad_review.csv'):
    for r in pd.read_csv(HERE/name,dtype=str,encoding='utf-8-sig').fillna('').to_dict('records'):
        FACTS.setdefault(r['candidate_id'],r['candidate_body_facts'] if 'candidate_body_facts' in r else r['candidate_body_fact'])
        FACTS.setdefault(r['reference_id'],r['reference_body_facts'] if 'reference_body_facts' in r else r['reference_body_fact'])

def metadata(i):
    d=HERE/'f34_official_articles';m=json.loads((d/f'{i}.json').read_text(encoding='utf-8'))
    raw=gzip.decompress((d/f'{i}.html.gz').read_bytes());assert sha256(raw).hexdigest()==m['sha256']
    s=BeautifulSoup(raw,'html.parser');assert s.select_one('.left_zw')
    return m

for name in ('wide_event','broad'):
    f=pd.read_csv(HERE/f'f34_{args.version}_{name}_flags.csv',dtype=str,encoding='utf-8-sig')
    rows=[]
    for r in f.to_dict('records'):
        a,b=r['candidate_id'],r['reference_id'];am,bm=metadata(a),metadata(b)
        if a not in FACTS or b not in FACTS: raise ValueError('Missing reviewed facts: '+a+' '+b)
        rows.append({**r,'decision':'不同事件或独立数据发布','candidate_body_fact':FACTS[a],
                     'reference_body_fact':FACTS[b],'candidate_url':am['url'],'reference_url':bm['url'],
                     'candidate_sha256':am['sha256'],'reference_sha256':bm['sha256']})
    out=HERE/f'f34_{args.version}_{name}_review.csv'
    if out.exists():raise FileExistsError(out)
    pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
    print(name,len(rows),'pair records')

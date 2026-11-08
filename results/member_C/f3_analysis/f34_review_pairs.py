"""Write the manually assessed 23-pair review, citing retained official articles."""
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
FACTS = {
 'cn10417676':'成都都江堰棍网球赛，14 支队伍参赛，成都大学队分获成年男女组冠军',
 'cn10413615':'北京残疾人乒乓球锦标赛暨东京聋奥选拔赛，24 个省区市 227 人参赛',
 'cn10480816':'宁波男子气步枪决赛，索拉佐 255 环夺冠，盛李豪获得银牌',
 'cn10481848':'宁波女子气手枪决赛，埃莎·辛格 242.6 环夺冠，姚千寻 242.5 环获银牌',
 'cn10511212':'广东国际划船中心赛艇预赛，包括四人双桨等 8 个小项',
 'cn10514829':'广州飞碟训练中心射击比赛，设多向和双向飞碟等项目',
 'cn10517813':'广东奥体中心女子 5000 米决赛，梁田田 15 分 09 秒 85 获冠军',
 'cn10518656':'广东奥体中心女子跳远决赛，谈梦怡以 6 米 54 获金牌',
 'cn10519163':'广东奥体中心女子 10000 米决赛，梁田田以 32 分 23 秒 40 获金牌',
 'cn10518615':'广东奥体中心女子铁饼决赛，冯彬第四投 65 米 46 夺冠',
 'cn10519086':'羽毛球女子单打决赛，王祉怡以 2:1 战胜陈雨菲',
 'cn10519108':'广东奥体中心女子 1500 米决赛，李春辉 4 分 06 秒 28 获金牌',
 'cn10515580':'羽毛球成年组女子团体决赛，浙江队以 3:0 战胜江苏队',
 'cn10517838':'广东奥体中心女子 400 米决赛，刘英兰 51 秒 75 夺冠',
 'cn10489314':'雅加达，中国驻印尼大使王鲁彤主办，印尼政府代表阿古斯等出席',
 'cn10489658':'布鲁塞尔，驻欧盟使团与驻比使馆联合主办，欧盟和比利时代表出席',
 'cn10476331':'外汇交易中心 9 月 4 日公布中间价 7.1052，上调 56 个基点',
 'cn10478328':'外汇交易中心 9 月 8 日公布中间价 7.1029，上调 35 个基点',
 'cn10480354':'外汇交易中心 9 月 11 日公布中间价 7.1034，上调 28 个基点',
 'cn10516755':'外汇交易中心 11 月 17 日公布中间价 7.0816；原文正文为上调 9 个基点，候选标题为上调 33，冲突须排除',
 'cn10514692':'外汇交易中心 11 月 13 日公布中间价 7.0865，下调 32 个基点',
 'cn10537820':'外汇交易中心 12 月 22 日公布中间价 7.0572，下调 22 个基点',
 'cn10538515':'外汇交易中心 12 月 23 日公布中间价 7.0523，上调 49 个基点',
 'cn10540596':'外汇交易中心 12 月 26 日公布中间价 7.0358，上调 34 个基点',
 'cn10543482':'外汇交易中心 12 月 31 日公布中间价 7.0288，上调 60 个基点',
 'cn10352777':'外汇交易中心 1 月 14 日公布中间价 7.1878，上调 7 个基点',
 'cn10353509':'外汇交易中心 1 月 15 日公布中间价 7.1883，下调 5 个基点',
 'cn10495907':'外汇交易中心 10 月 10 日公布中间价 7.1048，上调 54 个基点',
 'cn10499165':'外汇交易中心 10 月 16 日公布中间价 7.0968，上调 27 个基点',
}

def main():
    pairs = pd.read_csv(HERE/'f3_a_v3_event_review.csv', dtype=str, encoding='utf-8-sig')
    rows=[]
    for p in pairs.to_dict('records'):
        a,b=p['candidate_id'],p['reference_id']
        am=json.loads((HERE/'f34_official_articles'/f'{a}.json').read_text(encoding='utf-8'))
        bm=json.loads((HERE/'f34_official_articles'/f'{b}.json').read_text(encoding='utf-8'))
        rows.append({**p, 'official_decision':'不同事件', 'candidate_body_facts':FACTS[a],
                     'reference_body_facts':FACTS[b], 'candidate_source':am['url'], 'reference_source':bm['url'],
                     'candidate_html_sha256':am['sha256'], 'reference_html_sha256':bm['sha256'],
                     'admission_note':'候选标题与正文基点数冲突，排除候选' if a=='cn10516755' else '本对事件独立；其他准入项仍须复核'})
    assert len(rows)==23
    pd.DataFrame(rows).to_csv(HERE/'f34_pair_review.csv',index=False,encoding='utf-8-sig')
    print('reviewed pairs:',len(rows),'distinct; cn10516755 excluded due to article inconsistency')

if __name__ == '__main__': main()

"""Join the stratified review with official source cues and training coverage evidence."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import config

# Manually inspected title/content cues; these are review notes, not prediction rules.
CROSS={
 'cn10582858':'养老民生与科技产业','cn10606554':'直播消费、产品造假与社会治理',
 'cn10628625':'消费市场与食品质量问题','cn10562572':'边检服务与国际商务',
 'cn10577853':'春运交通与客流统计','cn10567733':'机场客流货运与公众出行',
 'cn10580186':'农业销售与村民就业增收','cn10646935':'文化创意设计与公共活动',
 'cn10648823':'博物馆文物与中美民间交流','cn10554983':'非遗展览与公共文化活动',
 'cn10546870':'老人遗赠与法律社会问题','cn10636755':'文化遗产、考古与学术讨论',
 'cn10637181':'旅游产业与林区文旅活动','cn10640940':'科学家纪念与馆藏史料',
 'cn10623553':'博物馆文化活动与公众参与','cn10572159':'遗址文化与科技文旅',
 'cn10600113':'人才招聘与地方发展','cn10585090':'生态权益与金融贷款',
 'cn10577647':'能源产业项目与基层发展','cn10574897':'民营医药企业与海外贸易',
 'cn10630947':'商业服务与家庭健康','cn10595719':'产业扶持与公共政策',
 'cn10630723':'史学家藏书与大学馆藏','cn10633625':'非遗工艺与教育体验',
 'cn10580924':'海关执法与文物保护','cn10566399':'跨地域教育资源与公共服务',
 'cn10614502':'博物馆展览与教育人物','cn10595384':'大学生交流与青年文化活动',
 'cn10598817':'高校古生物科研与公共科学新闻','cn10605645':'企业招聘与高校毕业生',
 'cn10621341':'非遗艺术与课堂教学','cn10625028':'公益跑步与文化地标',
 'cn10601008':'科研失信与监管通报','cn10635480':'外国政策与金融市场',
 'cn10615796':'教育人物与高考制度','cn10602812':'国际会谈与市场反应',
 'cn10628595':'体育用品与产业发展','cn10586127':'境外活动与文化交流',
 'cn10641784':'节日文化与消费市场','cn10634532':'儿童图书与出版市场',
 'cn10590283':'高校科研与新材料技术','cn10607219':'教师育人与展示活动',
 'cn10602030':'音乐会文化活动与城市公共空间',
 'cn10555482':'足球训练与学校教育','cn10600296':'外贸订单与世界杯体育赛事',
 'cn10624782':'音乐内容与版权争议','cn10586908':'大学数学教育与文化节活动',
 'cn10594440':'商事调解与高校招生','cn10614306':'大学教师科研问题与社会调查',
 'cn10595092':'学校教育与地震安全演练','cn10560809':'中小学教育与冰雪运动设施',
 'cn10602291':'大学教育与网络攻防竞赛','cn10619752':'高校管理与隐私事件',
 'cn10618127':'宠物消费经济与民生生活','cn10589760':'国产邮轮制造与交通产业',
 'cn10574165':'国际议题与货币政策','cn10641284':'高考与诈骗防范',
 'cn10552355':'体育赛事与消费经济','cn10632967':'外国旅游与经济数据',
 'cn10606030':'航班消费与公共维权','cn10639895':'产业发展与职业教育',
 'cn10636703':'国际教育活动与出国留学','cn10598503':'中学交流与工业发展',
 'cn10601056':'跨国黄金储备与金融市场','cn10614747':'机场出入境与交通数据',
 'cn10623479':'奥运承办与城市经济','cn10595512':'中小学春假与地方旅游',
 'cn10637781':'食品消费与市场监管','cn10643783':'地方民俗与龙舟运动',
}
UNDERSPECIFIED={'cn10576284','cn10642935','cn10618187','cn10572743','cn10608818',
                'cn10573788','cn10612833','cn10569791','cn10587860','cn10643911',
                'cn10576766','cn10637407','cn10598630'}
src=pd.read_csv(HERE/'review_source_extract.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
q=pd.read_csv(HERE/'review_queue.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
ft=pd.read_csv(HERE/'feature_per_title.csv',encoding='utf-8-sig')
nt=pd.read_csv(HERE/'nearest_training.csv',encoding='utf-8-sig')
assert len(q)==len(src)==152 and q.id.is_unique and src.id.is_unique
assert q.raw_category.map(config.CATEGORY_MAPPING).equals(q.category)
assert src.title_match.eq('True').all()
d=q.merge(src,on='id',validate='one_to_one')
small=ft[ft.model=='F3-A'].combined_nnz.quantile(.25)
train_n=nt[(nt.model=='F3-A')&(nt.role=='nearest_any')].drop(columns=['model','role','true_category'])
train_other=nt[(nt.model=='F3-A')&(nt.role=='nearest_other')][['id','cosine']].rename(columns={'cosine':'nearest_other_cosine'})
train_same=nt[(nt.model=='F3-A')&(nt.role=='nearest_same')][['id','cosine']].rename(columns={'cosine':'nearest_same_cosine'})
f3=ft[ft.model=='F3-A'][['id','word_selected_nnz','char_selected_nnz','combined_nnz',
                        'word_never_seen_fraction','char_never_seen_fraction','word_vocab_miss_fraction','char_vocab_miss_fraction']]
d=d.merge(f3,on='id',validate='one_to_one').merge(train_n,on='id',validate='one_to_one').merge(train_same,on='id',validate='one_to_one').merge(train_other,on='id',validate='one_to_one')
rows=[]
for r in d.to_dict('records'):
    id=r['id'];n=int(r['body_chars']);cross=CROSS.get(id,'')
    weak=n<60
    raw='归档原栏目“'+r['raw_category']+'”映射为“'+r['category']+'”；官网标题一致。'
    if weak:source_review=raw+'页面文字为海报、图片说明或来源信息，无法核实正文与栏目是否一致。'
    elif cross:source_review=raw+'官网首段与标题主题相关；'+cross+'，栏目语义边界仍需人工判断。'
    else:source_review=raw+'官网首段可读，与标题未见明显矛盾；栏目是否唯一合适仍需按标注口径人工判断。'
    info=('标题有跨主题线索，单凭标题不能确认唯一栏目' if cross else
          '标题指代或内容较笼统，需要正文和类别口径' if id in UNDERSPECIFIED else
          '标题包含事件或主体；是否足以确定唯一栏目仍需按标注口径判断')
    signal=('保留特征总数处于全体验证标题下四分位' if r['combined_nnz']<=small else '保留特征总数高于全体验证标题下四分位')
    if r['nearest_other_cosine']>r['nearest_same_cosine']+1e-12:signal+='；最相似不同类训练标题高于同类近邻'
    else:signal+='；同类近邻不低于不同类近邻'
    rows.append({'review_group':r['review_group'],'id':id,'title':r['title'],'validation_label':r['category'],
                 'raw_category':r['raw_category'],'f2_votes':r['f2_votes'],'f3_votes':r['f3_votes'],
                 'observed_cross_topic':cross if cross else '本次未确认明显跨主题线索；不排除其他解释',
                 'title_sufficiency_assessment':info,'official_url':r['url'],'official_page_sha256':r['sha256'],
                 'official_original_source':r['official_source'],'official_body_chars':n,
                 'official_first_passage':r['brief_source_text'],'source_and_category_review':source_review,
                 'word_selected_nnz':int(r['word_selected_nnz']),'char_selected_nnz':int(r['char_selected_nnz']),
                 'combined_nnz':int(r['combined_nnz']),'word_never_seen_fraction':r['word_never_seen_fraction'],
                 'char_never_seen_fraction':r['char_never_seen_fraction'],'nearest_train_id':r['train_id'],
                 'nearest_train_title':r['train_title'],'nearest_train_category':r['train_category'],
                 'nearest_train_url':r['train_url'],'nearest_cosine':r['cosine'],
                 'nearest_same_cosine':r['nearest_same_cosine'],'nearest_other_cosine':r['nearest_other_cosine'],
                 'coverage_observation':signal,
                 'uncertainty':'词面与近邻只显示关联，不证明模型使用该词或训练覆盖不足造成错误；不修改标签。'})
out=HERE/'manual_review.csv'
if out.exists():raise FileExistsError(out)
pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
print(json.dumps({'rows':len(rows),'manual_cross_topic_notes':sum(bool(CROSS.get(r['id'])) for r in rows),
                  'limited_official_text':sum(int(r['official_body_chars'])<60 for r in rows),
                  'f3_combined_nnz_lower_quartile':small},ensure_ascii=False))

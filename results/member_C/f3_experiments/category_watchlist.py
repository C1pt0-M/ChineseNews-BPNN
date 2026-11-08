"""Record title-level review flags without changing source labels."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
education = pd.read_csv(HERE / 'education_evidence.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
other = pd.read_csv(HERE / 'other_categories_sample.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
rows = pd.concat([education, other], ignore_index=True).set_index('id')
flags = {
    'cn10524314': '教育栏目标题聚焦精品消费活动，教育关联仅凭标题不明，需查原文。',
    'cn10535509': '教育栏目标题聚焦板球友谊赛，体育与教育边界需查原文。',
    'cn10461496': '教育栏目标题聚焦关键矿物及低碳转型，教育关联需查原文。',
    'cn10466701': '教育栏目标题聚焦奥运场馆，是否与教育相关需查原文。',
    'cn10369695': '财经栏目标题聚焦竖屏剧，影视内容与产业报道边界需查原文。',
    'cn10378994': '社会栏目标题涉及留学生文化寻访，社会、教育与文化边界需查原文。',
    'cn10382582': '国际栏目标题涉及气候模型研究，国际新闻与科学研究边界需查原文。',
}
result = []
for key, reason in flags.items():
    r = rows.loc[key]
    result.append({'id': key, 'category': r['category'], 'title': r['title'],
                   'url': r['url'], 'publish_time': r['publish_time'], 'review_note': reason,
                   'decision': '待原文复核；未修改标签'})
pd.DataFrame(result).to_csv(HERE / 'category_watchlist.csv', index=False, encoding='utf-8-sig')

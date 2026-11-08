"""Prepare a reproducible, month-stratified title-review queue; no labels are changed."""
from hashlib import sha256
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import config

PATTERNS = {
    '财经': r'央行|人民银行|银行业|存贷款|利率|汇率|人民币|外汇|股票|股市|A股|证券|债券|基金|保险业|财政收入|税收|进出口|外贸|经济增长|国内生产总值|GDP|经济运行|工业增加值|金融市场|金融机构|上市公司|经济数据|宏观经济|货币政策|通货膨胀|居民消费价格',
    '体育': r'乒乓球|羽毛球|篮球|足球|排球|网球|马拉松|田径|滑雪|游泳|奥运会|亚运会|全运会|中超|女足|男足|国足|国乒|网球公开赛|世界锦标赛|世锦赛|锦标赛|职业联赛|运动员|世界杯|体育赛事',
    '文娱': r'电影|影片|电视剧|戏曲|京剧|博物馆|文物|考古|书法|歌剧|交响乐|图书|古籍|书展|文学|作家|文化遗产|非遗|舞蹈|艺术展|电影节|音乐会|音乐剧|艺术家|剧院|文艺演出',
    '教育': r'教育部|教育厅|教育局|教育政策|教育改革|教育强国|义务教育|基础教育|职业教育|高等教育|中小学|小学|中学|高中|幼儿园|学前教育|教师|师生|学生|高考|中考|考研|考生|招生|录取分数线|大学|高校|学校|教授|留学生|留学|校园|课堂|课程|教学|入学|升学|毕业生|校长|班主任|课间|学籍',
    '社会': r'警方|公安|民警|消防|救援|火灾|车祸|交通事故|暴雨预警|地震|诈骗|法院|检察院|救助|铁路客流|旅客列车|交通安全|道路交通|道路塌方|山洪|台风预警|应急救援|刑事案件|居民供暖|医疗救助',
    '国际': r'总统|总理|外交部|外交部长|外长|联合国|安理会|议会|俄乌|乌克兰|白宫|国防部|停火|驻.*使馆|领事馆|国会|政府首脑|外交关系|外交会谈|外交部长|美军|以色列军方|巴勒斯坦|哈马斯|欧盟委员会|联合声明|北约|国际原子能机构',
}


def allocate(total, weights, capacities):
    quotas = {month: 0 for month in range(1, 13)}
    for _ in range(total):
        candidates = [m for m in quotas if quotas[m] < capacities.get(m, 0)]
        if not candidates:
            raise ValueError('insufficient title-review pool')
        # Sequential highest proportional shortfall; deterministic tie-break by month.
        m = max(candidates, key=lambda x: (weights.get(x, 0) * total / sum(weights.values()) - quotas[x], -x))
        quotas[m] += 1
    return quotas


def main():
    target = ROOT / 'results/member_C/f3_analysis/f3_title_review_queue.csv'
    if target.exists():
        raise FileExistsError(target)
    candidate = pd.read_csv(ROOT / 'results/member_C/f3_recovery/analysis/candidate_manifest.csv',
                            encoding='utf-8-sig', dtype=str, keep_default_na=False)
    train = pd.read_csv(ROOT / 'dataset/splits/train.csv', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    rows = []
    for category in config.TARGET_CATEGORIES:
        pattern = re.compile(PATTERNS[category], re.IGNORECASE)
        pool = candidate[(candidate.category == category) & candidate.title.str.contains(pattern, regex=True)].copy()
        pool['month'] = pool.publish_time.str[5:7].astype(int)
        pool['rank'] = pool.id.map(lambda value: sha256(('f3-review-v1-seed42:' + value).encode()).hexdigest())
        base = train[train.category == category].publish_time.str[5:7].astype(int).value_counts().to_dict()
        quotas = allocate(220, base, pool.month.value_counts().to_dict())
        for month, quota in quotas.items():
            rows.extend(pool[pool.month == month].sort_values('rank').head(quota).to_dict('records'))
        print(category, 'anchor pool', len(pool), 'review queue', sum(quotas.values()), flush=True)
    df = pd.DataFrame(rows)
    df['semantic_review'] = 'pending'
    df['review_note'] = ''
    df.sort_values(['category', 'publish_time', 'id']).to_csv(target, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()

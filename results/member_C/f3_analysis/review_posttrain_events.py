"""Record conservative decisions for additional same-event candidates."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
REJECT = {
    'cn10412253': '同日同一全运群众赛事的开赛与首金报道，事件独立性存疑。',
    'cn10528120': '与基准均描述同日中国队战胜日本队的成都混团世界杯比赛。',
    'cn10436828': '与基准均描述伊朗接受美国停火方案。',
    'cn10448859': '与基准均描述教育部严控中小学暑期作业总量。',
    'cn10538729': '同日标题高度相近，是否为同一双高计划报道尚不确定。',
}


def main():
    flags = pd.read_csv(HERE/'f3_a_posttrain_event_flags.csv',dtype=str,encoding='utf-8-sig')
    found = set(flags.candidate_id) & set(REJECT)
    assert found == set(REJECT)
    flags['decision'] = flags.candidate_id.map(lambda x: 'exclude_pending' if x in REJECT else 'different_details_or_date')
    flags['review_note'] = flags.candidate_id.map(lambda x: REJECT.get(x, '标题所述人物、项目、地区或日期不同；仅基于标题判断，保留语义风险。'))
    flags.to_csv(HERE/'f3_a_posttrain_event_review.csv',index=False,encoding='utf-8-sig')
    decisions = pd.read_csv(HERE/'f3_title_review_decisions.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    for key, note in REJECT.items():
        mask = decisions.id == key
        assert mask.sum() == 1
        decisions.loc[mask, 'semantic_review'] = 'exclude_pending'
        decisions.loc[mask, 'review_note'] = note
    decisions.to_csv(HERE/'f3_title_review_decisions_v2.csv',index=False,encoding='utf-8-sig')
    print(f'Recorded {len(REJECT)} candidates for exclusion; preserved the original data and model.')


if __name__ == '__main__':
    main()

"""Verify and document the corrected, untrained F3-A dataset."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXCLUDED = {'cn10412253','cn10528120','cn10436828','cn10448859','cn10538729','cn10499216','cn10499965'}


def main():
    target = HERE/'f3_education_final_review_v3.csv'
    if (HERE/'f3_a_v3_event_review.csv').exists():
        raise FileExistsError('Corrected review already finalized')
    dataset = HERE/'f3_a_dataset_v3'
    meta = json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    new = pd.read_csv(dataset/'new_sample_evidence.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    full = pd.read_csv(dataset/'train.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    base = pd.read_csv(ROOT/'dataset/splits/train.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    assert sha256((dataset/'train.csv').read_bytes()).hexdigest() == meta['train_sha256']
    assert len(full) == 7531 and len(new) == 982 and len(base) == 6549
    assert full.iloc[:len(base)].reset_index(drop=True).equals(base)
    assert not (set(new.id) & EXCLUDED) and new.id.is_unique
    expected = {'财经':166,'体育':166,'文娱':170,'教育':160,'社会':159,'国际':161}
    assert dict(Counter(new.category)) == expected
    assert new.grade.isin(['A','B']).all() and new.archive_source.eq('official_page_retrieved').all()
    assert new.semantic_review.eq('title_review_pass').all() and new.possible_event_conflict.eq('False').all()
    reviews = pd.read_csv(HERE/'f3_title_review_decisions_v3.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    edu = pd.read_csv(HERE/'f3_official_evidence_education.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
    fields = ['id','semantic_review','review_note','possible_event_conflict']
    joined = edu.merge(reviews[reviews.category=='教育'][fields],on='id',how='left',validate='one_to_one').fillna('')
    joined['title_review_status'] = joined.semantic_review.replace('', 'not_reviewed')
    joined['included_f3_a_v3'] = joined.id.isin(set(new.id))
    joined.drop(columns=['semantic_review','category_semantic_approval']).to_csv(target,index=False,encoding='utf-8-sig')
    pending = reviews[reviews.semantic_review!='title_review_pass'][['id','title','category','url','publish_time','review_note']].copy()
    remaining = joined[joined.title_review_status=='not_reviewed'].copy()
    remaining['review_note'] = '未进入本轮固定标题审阅队列；类别语义待审核'
    remaining = remaining.rename(columns={'article_url':'url'})[['id','title','category','url','publish_time','review_note']]
    pd.concat([pending,remaining],ignore_index=True).to_csv(HERE/'f3_manual_review_needed_v3.csv',index=False,encoding='utf-8-sig')
    flags = pd.read_csv(HERE/'f3_a_v3_event_flags.csv',dtype=str,encoding='utf-8-sig')
    previous = pd.read_csv(HERE/'f3_a_posttrain_event_review.csv',dtype=str,encoding='utf-8-sig')
    flags = flags.merge(previous[['candidate_id','reference_id','decision','review_note']],on=['candidate_id','reference_id'],how='left',validate='one_to_one')
    different_sports = flags.candidate_id.eq('cn10417676') & flags.reference_id.eq('cn10413615')
    flags.loc[different_sports, 'decision'] = 'different_details_or_date'
    flags.loc[different_sports, 'review_note'] = '标题分别为棍网球与残疾人乒乓球赛事，项目及日期不同。'
    assert len(flags)==23 and flags.decision.eq('different_details_or_date').all()
    flags.to_csv(HERE/'f3_a_v3_event_review.csv',index=False,encoding='utf-8-sig')
    print('Corrected dataset:',len(full),'rows; education grades:',new[new.category=='教育'].grade.value_counts().to_dict(),
          '; weak flags reviewed:',len(flags),'; no new model training.')


if __name__ == '__main__':
    main()

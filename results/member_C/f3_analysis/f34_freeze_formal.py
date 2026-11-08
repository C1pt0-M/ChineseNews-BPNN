"""Freeze reviewed candidate with independent evidence, hashes and heldout exclusion."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
import config

src=HERE/'f3_a_f34_review_candidate_v10'
dst=HERE/'f3_a_dataset_formal'
if dst.exists():raise FileExistsError(dst)
sha=lambda p:sha256(p.read_bytes()).hexdigest()
version=json.loads((ROOT/'dataset/dataset_version.json').read_text(encoding='utf-8'))
for split in ('train','val','test'):
    assert sha(ROOT/'dataset/splits'/f'{split}.csv')==version['files'][f'{split}.csv']['sha256']
base=pd.read_csv(ROOT/'dataset/splits/train.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
train=pd.read_csv(src/'train.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
new=pd.read_csv(src/'new_sample_evidence.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
article=pd.read_csv(HERE/'f34_candidate_v10_article_review.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
archive=pd.read_csv(HERE/'f34_candidate_v10_archive_verification.csv',dtype=str,keep_default_na=False,encoding='utf-8-sig')
meta=json.loads((src/'manifest.json').read_text(encoding='utf-8'))
assert len(base)==6549 and len(train)==7531 and len(new)==len(article)==len(archive)==982
assert train.iloc[:6549].reset_index(drop=True).equals(base)
assert sha(src/'train.csv')==meta['train_sha256']
assert set(new.id)==set(article.id)==set(archive.id)
assert new.id.is_unique and new.url.is_unique and new.title.is_unique
assert new.raw_category.map(config.CATEGORY_MAPPING).equals(new.category)
assert new.semantic_review.eq('title_review_pass').all() and new.grade.isin(['A','B']).all()
assert new.publish_time.between('2025-01-01 00:00:00','2025-12-31 23:59:59').all()
assert article.title_matches.eq('True').all() and article.time_matches.eq('True').all()
assert article.body_characters.astype(int).gt(0).all()
for name,expected in [('wide_event',55),('broad',73)]:
    flags=pd.read_csv(HERE/f'f34_v10_{name}_flags.csv',dtype=str,encoding='utf-8-sig')
    reviewed=pd.read_csv(HERE/f'f34_v10_{name}_review.csv',dtype=str,encoding='utf-8-sig')
    assert len(flags)==len(reviewed)==expected
    assert set(zip(flags.candidate_id,flags.reference_id))==set(zip(reviewed.candidate_id,reviewed.reference_id))
    assert reviewed.decision.eq('不同事件或独立数据发布').all()
spec=importlib.util.spec_from_file_location('audit',ROOT/'results/member_C/f3_recovery/audit_recovery.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
en=audit.enrich(new)
for split in ('train','val','test'):
    ref=audit.enrich(pd.read_csv(ROOT/'dataset/splits'/f'{split}.csv',usecols=['id','url','title'],dtype=str,keep_default_na=False,encoding='utf-8-sig'))
    for column in ('id','_url_key','_title_key'):
        assert set(en[column]).isdisjoint(ref[column]),(split,column)
for column in ('_url_key','_title_key'): assert en[column].is_unique
notes={
 'cn10348087':'正文9029人，支持标题逾9000人',
 'cn10415518':'正文中文数字“三枚金牌”与标题一致',
 'cn10467203':'正文成年男女各12队、青年男女各8队，共40队',
 'cn10512448':'正文14252人，支持标题逾1.4万人',
 'cn10347099':'正文政策名称使用中文数字“二十条”',
 'cn10382443':'正文用中文数字“八年制”',
 'cn10450271':'正文列出内蒙古、四川、西藏、青海四省区',
 'cn10488337':'正文实际成交47.2万元，标题取整为47万元',
 'cn10383314':'正文称今年前两个月，官方发布时间为2025年',
 'cn10434614':'正文85.96亿元，标题取近86亿元',
 'cn10538872':'正文年内增加5572.58亿元，支持超过5500亿元',
}
missing=article[article.title_numbers_missing_in_body.ne('')]
assert set(missing.id)==set(notes),set(missing.id)^set(notes)
assert new.category.value_counts().to_dict()=={'财经':166,'体育':166,'文娱':170,'教育':160,'社会':159,'国际':161}
dst.mkdir()
for name in ('train.csv','new_sample_evidence.csv','new_ids.txt'):shutil.copy2(src/name,dst/name)
article=article.merge(new[['id','grade','archive_url','archive_sha256','raw_category']],on='id',validate='one_to_one')
article['numeric_review_note']=article.id.map(notes).fillna('原文字面数量未触发差异提示')
article.to_csv(dst/'article_audit.csv',index=False,encoding='utf-8-sig')
meta.update({'name':'F3-A formal reviewed', 'status':'frozen', 'parent_candidate':str(src.relative_to(ROOT)),
             'article_audit_sha256':sha(dst/'article_audit.csv'),
             'review_files':{x:sha(HERE/x) for x in ['f34_pair_review.csv','f34_candidate_v10_archive_verification.csv',
                              'f34_v10_wide_event_review.csv','f34_v10_broad_review.csv','f34_title_review_decisions_v10.csv']},
             'recorded_source':'source field denotes official platform; article_audit.csv also records original credited media',
             'dedup':'exact id/url/title and reviewed flagged title similarities against train/val/test; no heldout labels',
             'limitations':'title similarity thresholds cannot prove absence of semantically rewritten same-event reports; base duplicates retained'})
(dst/'manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print('formal frozen:',dst,'SHA256:',meta['train_sha256'])

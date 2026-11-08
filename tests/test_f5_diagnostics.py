"""Independent checks of F.5 cohort and sparse nearest-neighbor evidence."""
from pathlib import Path
import pickle
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/member_C/f5_error_analysis'


class F5AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred=pd.read_csv(ROOT/'results/member_C/f3_analysis/f3_a_multiseed/prediction_comparison.csv',encoding='utf-8-sig')
        cls.groups=pd.read_csv(OUT/'error_groups.csv',encoding='utf-8-sig')

    def test_rebuild_disjoint_definitions_and_overlap(self):
        d=self.pred
        sets={g:set(x.id) for g,x in self.groups.groupby('group')}
        self.assertEqual(sets['持续错误'],set(d[(d.f2_error_count==5)&(d.f3_error_count==5)].id))
        self.assertEqual(sets['稳定修正'],set(d[(d.f2_error_count==5)&(d.f3_error_count==0)].id))
        self.assertEqual(sets['稳定退化'],set(d[(d.f2_error_count==0)&(d.f3_error_count==5)].id))
        f2=[f'f2_pred_{s}' for s in range(42,47)];f3=[f'f3_pred_{s}' for s in range(42,47)]
        unstable=set(d[(d[f2].nunique(axis=1)>1)|(d[f3].nunique(axis=1)>1)].id)
        self.assertEqual(sets['种子不稳定'],unstable)
        self.assertEqual([len(sets[x]) for x in ('持续错误','稳定修正','稳定退化','种子不稳定')],[548,30,32,288])

    def test_unseen_is_subset_of_vocabulary_missing(self):
        f=pd.read_csv(OUT/'feature_per_title.csv',encoding='utf-8-sig')
        self.assertEqual(len(f),6368)
        for branch in ('word','char'):
            valid=f[f[f'{branch}_never_seen_fraction'].notna()]
            self.assertTrue((valid[f'{branch}_never_seen_fraction']<=valid[f'{branch}_vocab_miss_fraction']+1e-12).all())
            self.assertTrue((valid[f'{branch}_selected_nnz']<=valid[f'{branch}_tfidf_nnz']).all())

    def test_real_nearest_and_zero_vector(self):
        train=pd.read_csv(ROOT/'results/member_C/f3_analysis/f3_a_dataset_formal/train.csv',encoding='utf-8-sig')
        val=pd.read_csv(ROOT/'dataset/splits/val.csv',encoding='utf-8-sig')
        with (ROOT/'results/member_C/f3_formal_a_reviewed/F_word_char_s42/feature_builder.pkl').open('rb') as f:b=pickle.load(f)
        ids=['cn10590961','cn10633625']
        selected=val.set_index('id').loc[ids]
        scores=cosine_similarity(b.transform(selected.title.tolist()),b.transform(train.title.tolist()))
        n=pd.read_csv(OUT/'nearest_training.csv',encoding='utf-8-sig')
        for i,id in enumerate(ids):
            for role in ('nearest_any','nearest_same','nearest_other'):
                allowed=np.ones(len(train),dtype=bool)
                if role=='nearest_same':allowed=train.category.to_numpy()==selected.iloc[i].category
                if role=='nearest_other':allowed=train.category.to_numpy()!=selected.iloc[i].category
                expected=float(scores[i,allowed].max())
                r=n[(n.id==id)&(n.model=='F3-A')&(n.role==role)].iloc[0]
                self.assertAlmostEqual(expected,r.cosine,places=12)
                if expected==0:self.assertTrue(pd.isna(r.train_id))


if __name__=='__main__':unittest.main()

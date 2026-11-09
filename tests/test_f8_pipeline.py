"""F.8 semantic alignment, sparse fusion, and block-normalization checks."""
from pathlib import Path
import importlib.util
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'results/member_C/f8_experiments/pipeline.py'
spec = importlib.util.spec_from_file_location('f8_pipeline_tests', path)
f8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f8)

class F8Tests(unittest.TestCase):
    def test_bp_parameter_counts(self):
        from models.bp import BPNet
        for width, expected in ((512, 33222), (10512, 673222)):
            self.assertEqual(f8.bp_parameter_count(width), expected)
            self.assertEqual(BPNet(width, hidden_dim=64).count_parameters(), expected)

    def test_fusion_uses_unit_blocks_and_preserves_sparse_rows(self):
        class FakeBuilder:
            fit_rows_, word_dim_, char_dim_ = 7531, 7000, 3000
            def transform(self, texts):
                return sp.csr_matrix((np.ones(len(texts)), (np.arange(len(texts)), np.zeros(len(texts)))), shape=(len(texts),10000))
        splits = {'train':(['a','b'], np.array([0,1]), None), 'val':(['c'], np.array([2]), None)}
        vectors = {'train':np.eye(512, dtype=np.float32)[:2], 'val':np.eye(512, dtype=np.float32)[2:3]}
        with patch.object(f8.pickle, 'load', return_value=FakeBuilder()):
            train, val, config = f8.make_features('F8_2', splits, vectors, {})
        self.assertTrue(sp.isspmatrix_csr(train))
        self.assertEqual(train.shape, (2,10512))
        self.assertEqual(val.shape, (1,10512))
        np.testing.assert_allclose(train[:,10000:].toarray(), vectors['train'])
        np.testing.assert_allclose(np.sqrt(np.asarray(train.multiply(train).sum(axis=1)).ravel()), np.sqrt(2))
        self.assertIn('no post-fusion normalization', config['normalization'])
        sem_train, sem_val, _ = f8.make_features('F8_1', splits, vectors, {})
        np.testing.assert_array_equal(sem_train, vectors['train'])
        np.testing.assert_array_equal(sem_val, vectors['val'])

    def test_cache_rejects_any_id_title_or_label_mismatch(self):
        vectors = np.eye(512, dtype=np.float32)[:2]
        frame = pd.DataFrame({'id':['x','y'], 'title':['甲','乙'], 'category':['财经','体育']})
        labels = np.array([0,1], dtype='int64')
        splits = {'train':(frame.title.tolist(), labels, frame), 'val':(frame.title.tolist(), labels, frame)}
        meta = {'revision':f8.REVISION, 'encoder_trainable_parameters':0, 'files_sha256':{},
                'splits':{'train':{'cache_sha256':'expected'},'val':{'cache_sha256':'expected'}}}
        class FakeNpz:
            def __enter__(self):
                return {'vectors':vectors, 'ids':np.array(['x','y']), 'titles':np.array(['甲','乙']), 'labels':labels}
            def __exit__(self, *_): return False
        with patch.object(Path, 'read_text', return_value=json.dumps(meta)), patch.object(f8, 'digest', return_value='expected'), patch.object(f8.np, 'load', side_effect=lambda *_a, **_k:FakeNpz()):
                output, _ = f8.load_vectors(splits)
                np.testing.assert_array_equal(output['train'], vectors)
                bad = frame.copy(); bad.loc[0,'id'] = 'not-x'
                with self.assertRaisesRegex(RuntimeError, 'order differs'):
                    f8.load_vectors({'train':(bad.title.tolist(),labels,bad), 'val':splits['val']})
                with self.assertRaisesRegex(RuntimeError, 'order differs'):
                    f8.load_vectors({'train':(frame.title.tolist(),labels[::-1],frame), 'val':splits['val']})

if __name__ == '__main__':
    unittest.main()

import pickle
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

import model_config as C
from models.f_features import TitleFeatureBuilder
from models.features import FeatureBuilder, tokenize_corpus
from train_bp import build_features


class FeatureComparisonTests(unittest.TestCase):
    def setUp(self):
        self.train = ['alpha news market', 'alpha news market', 'beta sports game',
                      'beta sports game', 'gamma school class', 'gamma school class']
        self.y = np.array([0, 0, 1, 1, 2, 2])
        self.val = ['zzzzzz news market']

    def test_char_and_fusion_fit_train_only_sparse_and_pickle(self):
        for scheme in ('F_char', 'F_word_char'):
            with self.subTest(scheme=scheme):
                splits = {'train': (self.train, self.y, None),
                          'val': (self.val, np.array([0]), None)}
                builder, X, _, V, _, _ = build_features(C.experiment_by_id(scheme), 42, splits,
                                                         log_fn=lambda *_: None)
                self.assertEqual(builder.fit_rows_, len(self.train))
                self.assertEqual(builder.char_vectorizer.analyzer, 'char')
                self.assertEqual(builder.char_vectorizer.ngram_range, (2, 4))
                self.assertNotIn('zz', builder.char_vectorizer.vocabulary_)
                self.assertTrue(sparse.isspmatrix_csr(X) and sparse.isspmatrix_csr(V))
                self.assertEqual(X.shape[1], builder.char_dim_ + builder.word_dim_)
                self.assertEqual(V.shape[1], X.shape[1])
                self.assertEqual(builder.char_selector.n_features_in_, len(builder.char_vectorizer.vocabulary_))
                vocab = dict(builder.char_vectorizer.vocabulary_)
                idf = builder.char_vectorizer.idf_.copy()
                mask = builder.char_selector.get_support().copy()
                restored = pickle.loads(pickle.dumps(builder))
                np.testing.assert_allclose(restored.transform(self.val).toarray(), V.toarray())
                self.assertEqual(vocab, builder.char_vectorizer.vocabulary_)
                np.testing.assert_array_equal(idf, builder.char_vectorizer.idf_)
                np.testing.assert_array_equal(mask, builder.char_selector.get_support())
                info = builder.describe()
                self.assertEqual(info['input_dim'], X.shape[1])
                if scheme == 'F_word_char':
                    self.assertNotIn('zzzzzz', builder.word.vectorizer.vocabulary_)
                    np.testing.assert_allclose(np.sqrt(X.multiply(X).sum(axis=1)).A.ravel(), 1.0)
                    reference = FeatureBuilder('B', k=5000).fit(tokenize_corpus(self.train), self.y)
                    np.testing.assert_allclose(reference.transform(tokenize_corpus(self.val)).toarray(),
                                               builder.word.transform(tokenize_corpus(self.val)).toarray())
                    self.assertEqual(info['word']['selected_features'], builder.word_dim_)

    def test_historical_experiments_remain_opt_in_separate(self):
        self.assertEqual(len(C.EXPERIMENTS), 7)
        for name, k in [('A_raw', None), ('B_chi2_5000', 5000), ('B_chi2_10000', 10000)]:
            self.assertEqual(C.experiment_by_id(name)['k'], k)
        self.assertNotIn('F_char', [e['id'] for e in C.EXPERIMENTS])

    def test_d3_defaults_and_output_root_are_required(self):
        from train_bp import main
        with self.assertRaisesRegex(ValueError, 'output-root'):
            main(['--exp', 'F_char'])
        with self.assertRaisesRegex(ValueError, 'hidden=64'):
            main(['--exp', 'F_char', '--output-root', 'unused', '--hidden', '128'])

if __name__ == '__main__':
    unittest.main()

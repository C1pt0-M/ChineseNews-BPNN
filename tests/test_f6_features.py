"""F.6 allocations preserve the original train-only, sparse feature pipeline."""
import pickle
import unittest

import numpy as np
from scipy import sparse

import model_config as C
from models.f_features import TitleFeatureBuilder
from train_bp import build_features, main, parse_args


class F6FeatureTests(unittest.TestCase):
    def setUp(self):
        self.train = [
            "alpha market stocks", "alpha market stocks",
            "beta sports team", "beta sports team",
            "gamma school teacher", "gamma school teacher",
        ]
        self.y = np.array([0, 0, 1, 1, 2, 2])
        self.val = ["zzzzzz alpha market", "beta sports team"]

    def test_defaults_and_historical_configs(self):
        old = TitleFeatureBuilder("F_word_char")
        self.assertEqual((old.word.k, old.char_k), (5000, 5000))
        self.assertEqual(TitleFeatureBuilder("F_char").char_k, 10000)
        self.assertEqual(C.experiment_by_id("F_word_char")["k"], 5000)
        self.assertEqual(C.experiment_by_id("B_chi2_5000")["k"], 5000)
        self.assertEqual(C.experiment_by_id("B_chi2_10000")["k"], 10000)
        self.assertEqual([e['id'] for e in C.EXPERIMENTS], [
            'A_raw', 'B_chi2_2000', 'B_chi2_5000', 'B_chi2_10000',
            'C_svd_100', 'C_svd_200', 'C_svd_300',
        ])
        self.assertTrue(parse_args(["--all"]).all)
        for name in ("F6_A", "F6_B"):
            with self.assertRaisesRegex(ValueError, "output-root"):
                main(["--exp", name])

    def test_configs_build_from_training_only(self):
        splits = {'train': (self.train, self.y, None),
                  'val': (self.val, np.array([0, 1]), None)}
        for name, word_k, char_k in [('F6_A', 3000, 7000), ('F6_B', 7000, 3000)]:
            with self.subTest(name=name):
                builder, train, _, val, _, _ = build_features(
                    C.experiment_by_id(name), 42, splits, log_fn=lambda *_: None)
                info = builder.describe()
                self.assertEqual((builder.word.k, builder.char_k), (word_k, char_k))
                self.assertEqual(info['word']['chi2_k_requested'], word_k)
                self.assertEqual(info['char']['chi2_k_requested'], char_k)
                self.assertEqual(builder.fit_rows_, len(self.train))
                self.assertEqual(builder.word.fit_rows_, len(self.train))
                self.assertNotIn('zzzzzz', builder.word.vectorizer.vocabulary_)
                self.assertNotIn('zz', builder.char_vectorizer.vocabulary_)
                self.assertEqual(builder.word_dim_, min(word_k, builder.word.n_features_in_))
                self.assertEqual(builder.char_dim_, min(char_k, len(builder.char_vectorizer.vocabulary_)))
                self.assertEqual(train.shape, (len(self.train), builder.word_dim_ + builder.char_dim_))
                self.assertEqual(val.shape, (len(self.val), train.shape[1]))
                self.assertTrue(sparse.isspmatrix_csr(train) and sparse.isspmatrix_csr(val))
                norms = np.sqrt(train.multiply(train).sum(axis=1)).A.ravel()
                np.testing.assert_allclose(norms[norms > 0], 1.0)
                restored = pickle.loads(pickle.dumps(builder))
                np.testing.assert_allclose(restored.transform(self.val).toarray(), val.toarray())

    def test_small_allocations_selection_and_transform_are_immutable(self):
        # Small caps exercise active chi2 selection, rather than just vocabulary clipping.
        for word_k, char_k in [(2, 3), (3, 2)]:
            with self.subTest(word_k=word_k, char_k=char_k):
                builder = TitleFeatureBuilder('F_word_char', word_k=word_k, char_k=char_k)
                builder.fit(self.train, self.y)
                self.assertEqual((builder.word_dim_, builder.char_dim_), (word_k, char_k))
                state = (dict(builder.word.vectorizer.vocabulary_),
                         builder.word.vectorizer.idf_.copy(), builder.word.selector.get_support().copy(),
                         dict(builder.char_vectorizer.vocabulary_),
                         builder.char_vectorizer.idf_.copy(), builder.char_selector.get_support().copy())
                transformed = builder.transform(self.val)
                self.assertEqual(state[0], builder.word.vectorizer.vocabulary_)
                np.testing.assert_array_equal(state[1], builder.word.vectorizer.idf_)
                np.testing.assert_array_equal(state[2], builder.word.selector.get_support())
                self.assertEqual(state[3], builder.char_vectorizer.vocabulary_)
                np.testing.assert_array_equal(state[4], builder.char_vectorizer.idf_)
                np.testing.assert_array_equal(state[5], builder.char_selector.get_support())
                restored = pickle.loads(pickle.dumps(builder))
                np.testing.assert_allclose(restored.transform(self.val).toarray(), transformed.toarray())


if __name__ == '__main__':
    unittest.main()

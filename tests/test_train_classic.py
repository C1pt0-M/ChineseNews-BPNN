import unittest

import numpy as np
from scipy import sparse

import train_classic as classic


class RecordingBuilder:
    instances = []

    def __init__(self, scheme, seed):
        self.scheme, self.seed, self.transforms = scheme, seed, []
        self.__class__.instances.append(self)

    def fit(self, texts, labels):
        self.fit_texts = list(texts)
        self.fit_rows_ = len(texts)
        return self

    def transform(self, texts):
        self.transforms.append(list(texts))
        # Features are derived only from synthetic text, not validation labels.
        return sparse.csr_matrix([[1., float(t.startswith('odd'))] for t in texts])


class ClassicTests(unittest.TestCase):
    def test_cv_fits_features_only_on_each_training_fold(self):
        RecordingBuilder.instances = []
        texts = [f'odd{i}' if i % 2 else f'even{i}' for i in range(24)]
        y = np.array([i % 2 for i in range(24)])
        rows = classic.cross_validate(texts, y, n_splits=3, builder_factory=RecordingBuilder, log_fn=None)
        self.assertEqual(len(rows), 3 * 2 * 4)
        self.assertEqual(len(RecordingBuilder.instances), 3)
        for builder in RecordingBuilder.instances:
            self.assertEqual(builder.fit_texts, builder.transforms[0])
            self.assertTrue(set(builder.fit_texts).isdisjoint(builder.transforms[1]))
            self.assertEqual(set(builder.fit_texts) | set(builder.transforms[1]), set(texts))
            self.assertEqual(builder.fit_rows_, 16)
        self.assertEqual({r['feature_fit_rows'] for r in rows}, {16})

    def test_selection_uses_cv_means_and_ties_choose_smaller_c(self):
        rows = [{'classifier':'LinearSVC','C':c,'macro_f1':v}
                for c, v in [(0.01, .2), (0.1, .7), (1., .7), (10., .6)] for _ in range(3)]
        self.assertEqual(classic.select_c(rows, 'LinearSVC')[0], 0.1)

    def test_dense_features_are_rejected(self):
        with self.assertRaises(TypeError):
            classic.fit_checked(classic.estimator('LinearSVC', 1., 42), np.ones((3, 2)), np.array([0, 1, 0]))


if __name__ == '__main__':
    unittest.main()

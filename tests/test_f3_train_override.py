"""Checks for fingerprinted derived training data without reading held-out labels."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

import model_config as C
from models.data import load_training_override
from train_bp import parse_args


class TrainingOverrideTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'train.csv'
        self.base = pd.DataFrame([
            {'id': f'cn{i}', 'title': f'title{i}', 'category': label,
             'publish_time': '2025-04-01 12:00:00', 'url': f'https://example.com/{i}'}
            for i, label in enumerate(C.LABELS)
        ])
        self.base_split = (self.base.title.tolist(), None, self.base)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, frame):
        frame.to_csv(self.path, index=False, encoding='utf-8-sig')
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def test_fingerprinted_override_retains_baseline_and_labels(self):
        row = pd.DataFrame([{'id':'cn99', 'title':'new title', 'category':C.LABELS[0],
                             'publish_time':'2025-12-31 23:59:59', 'url':'https://example.com/99'}])
        full = pd.concat([self.base, row], ignore_index=True)
        digest = self.save(full)
        texts, labels, actual = load_training_override(self.path, digest, self.base_split)
        self.assertEqual(len(texts), 7)
        self.assertEqual(labels.tolist(), [0, 1, 2, 3, 4, 5, 0])
        self.assertTrue(actual.iloc[:6].reset_index(drop=True).equals(self.base))
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            load_training_override(self.path, '0'*64, self.base_split)

    def test_rejects_changed_baseline_future_and_duplicates(self):
        cases = []
        changed = self.base.copy(); changed.loc[0, 'title'] = 'changed'; cases.append(changed)
        future = pd.concat([self.base, self.base.iloc[[0]].assign(id='new', title='new',
                            url='https://example.com/new', publish_time='2026-01-01 00:00:00')]); cases.append(future)
        duplicate = pd.concat([self.base, self.base.iloc[[0]]]); cases.append(duplicate)
        for frame in cases:
            with self.subTest(rows=len(frame), title=frame.iloc[0].title):
                with self.assertRaises(ValueError):
                    load_training_override(self.path, self.save(frame), self.base_split)

    def test_heldout_paths_forbidden_and_cli_default_unchanged(self):
        for split in ('val', 'test'):
            with self.assertRaisesRegex(ValueError, 'cannot be used'):
                load_training_override(C.SPLIT_DIR / (split + '.csv'), 'x', self.base_split)
        args = parse_args(['--exp', 'F_word_char'])
        self.assertIsNone(args.train_file)
        self.assertIsNone(args.train_sha256)


if __name__ == '__main__':
    unittest.main()

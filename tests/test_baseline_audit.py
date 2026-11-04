"""Behavioral checks for the loss instrumentation and data boundary."""
import unittest
import tempfile
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from models.data import load_splits
from models.trainer import evaluate_training_set
import model_config as C


class AuditTests(unittest.TestCase):
    def test_eval_loss_is_weighted_no_grad_and_rng_neutral(self):
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(4, 6), nn.Dropout(0.9))
        X = np.arange(28, dtype=np.float32).reshape(7, 4) / 10
        y = np.array([0, 1, 2, 3, 4, 5, 0])
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        model.train()
        actual = evaluate_training_set(model, X, y, batch_size=3)["loss"]
        with torch.no_grad():
            expected = nn.functional.cross_entropy(model(torch.from_numpy(X)), torch.from_numpy(y))
        self.assertAlmostEqual(actual, expected.item(), places=6)
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, value in model.state_dict().items():
            self.assertTrue(torch.equal(before[k], value))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_training_loader_does_not_open_test(self):
        import builtins
        original = builtins.open

        def guarded(path, *args, **kwargs):
            if str(path).replace('\\', '/').endswith('/test.csv'):
                raise AssertionError("Training opened test.csv")
            return original(path, *args, **kwargs)

        with patch('builtins.open', guarded):
            self.assertEqual(set(load_splits(splits=('train', 'val'))), {'train', 'val'})

    def test_fingerprint_mismatch_blocks_loading_and_test_entry(self):
        from models.data import verify_dataset
        from evaluate_test import load_test_split
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = b'id,title,category,publish_time\n1,abc,x,2026-01-01\n'
            version = root / 'version.json'
            version.write_text(json.dumps({'files': {
                f'{s}.csv': {'rows': 1, 'sha256': hashlib.sha256(original).hexdigest()}
                for s in ('train', 'test')}}))
            for split in ('train', 'test'):
                (root / f'{split}.csv').write_bytes(original.replace(b'abc', b'xyz'))
            with patch.object(C, 'SPLIT_DIR', root), patch.object(C, 'DATASET_VERSION_FILE', version):
                with self.assertRaises(RuntimeError):
                    verify_dataset(splits=('train',))
                with patch('models.data.pd.read_csv') as read:
                    with self.assertRaises(RuntimeError):
                        load_splits(splits=('train',))
                    read.assert_not_called()
                with patch('evaluate_test.pd.read_csv') as read:
                    with self.assertRaises(RuntimeError):
                        load_test_split()
                    read.assert_not_called()

    def test_existing_run_rejected_before_features_or_training(self):
        from train_bp import run_single
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'A_raw_s42'
            old.mkdir()
            marker = old / 'history.csv'
            marker.write_bytes(b'original result')
            with patch('train_bp.build_features') as features, patch('train_bp.train_bp_model') as train:
                with self.assertRaises(FileExistsError):
                    run_single(C.experiment_by_id('A_raw'), 42, {}, {}, output_root=root)
                features.assert_not_called()
                train.assert_not_called()
            self.assertEqual(marker.read_bytes(), b'original result')

    def test_training_entry_fits_features_only_on_train(self):
        from models.features import FeatureBuilder
        from train_bp import build_features
        fitted = []

        class ObservedBuilder(FeatureBuilder):
            def fit(self, tokens, labels):
                fitted.append([list(t) for t in tokens])
                return super().fit(tokens, labels)

        train = ['trainonly shared', 'trainonly shared', 'another shared', 'another shared']
        val = ['validationonly shared']
        splits = {'train': (train, np.array([0, 0, 1, 1]), None),
                  'val': (val, np.array([0]), None)}
        with patch('train_bp.FeatureBuilder', ObservedBuilder), patch(
                'train_bp.tokenize_corpus', side_effect=lambda texts, key: [s.split() for s in texts]):
            builder, X, _, V, _, _ = build_features(C.experiment_by_id('A_raw'), 42, splits, log_fn=lambda *_: None)
        self.assertEqual(fitted, [[s.split() for s in train]])
        self.assertNotIn('validationonly', builder.vectorizer.vocabulary_)
        before = builder.vectorizer.idf_.copy()
        vocab = dict(builder.vectorizer.vocabulary_)
        builder.transform([['brandnew', 'shared']])
        np.testing.assert_array_equal(before, builder.vectorizer.idf_)
        self.assertEqual(vocab, builder.vectorizer.vocabulary_)
        self.assertEqual(X.shape[1], V.shape[1])

    def test_numeric_diagnostics_match_actual_filter(self):
        from recheck_member_A import removed_numeric_tokens
        self.assertEqual(removed_numeric_tokens(['140', '万亿', '四', '785', 'SNH48']), ['140', '785'])

    def test_lr_no_test_uses_train_val_loader(self):
        import baseline_lr
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                baseline_lr, 'OUT_FILE', Path(tmp) / 'lr.json'), patch(
                'baseline_lr.load_splits', side_effect=RuntimeError('stop before fitting')) as loader:
            with self.assertRaisesRegex(RuntimeError, 'stop before fitting'):
                baseline_lr.main(['--no-test'])
            loader.assert_called_once_with(verify=True, splits=('train', 'val'))

    def test_early_stopping_restores_macro_f1_checkpoint(self):
        from models.trainer import train_bp_model
        states = []
        scores = iter([.5, .7, .69])

        def validation(model, *_args, **_kwargs):
            states.append({k: v.clone() for k, v in model.state_dict().items()})
            return {'f1_macro': next(scores), 'loss': 1 / len(states), 'accuracy': .5}

        X = np.arange(20, dtype=np.float32).reshape(5, 4) / 10
        y = np.arange(5, dtype=np.int64)
        with patch('models.trainer.evaluate_training_set', return_value={'loss': .3}), patch(
                'models.trainer._eval_loss_and_metrics', side_effect=validation):
            result = train_bp_model(X, y, X, y, seed=42,
                                    bp_params={'hidden_dim': 4},
                                    train_params={'max_epochs': 8, 'patience': 1}, log_fn=None)
        self.assertEqual(result['best_epoch'], 2)
        self.assertEqual(result['epochs_run'], 3)
        for key, value in result['model'].state_dict().items():
            self.assertTrue(torch.equal(value, states[1][key]))
        # Validation loss continued to fall; selection must still follow F1.
        self.assertEqual(result['val_metrics']['f1_macro'], .7)


if __name__ == '__main__':
    unittest.main()

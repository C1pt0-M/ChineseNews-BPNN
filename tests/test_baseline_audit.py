"""Behavioral checks for the loss instrumentation and data boundary."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from models.data import load_splits
from models.trainer import evaluate_training_set


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


if __name__ == '__main__':
    unittest.main()

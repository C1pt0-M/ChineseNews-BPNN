import io
import subprocess
import sys
import unittest

import torch
from torch import nn

from models.bp import BPNet, build_model


class ModelArchitectureTests(unittest.TestCase):
    def arch(self, kind):
        return dict(model_type=kind, input_dim=5000, output_dim=6,
                    hidden_dim=64, dropout=0.2, input_dropout=0.0)

    def test_shapes_counts_and_roundtrip(self):
        x = torch.randn(3, 5000)
        for kind, count in [('linear', 30006), ('bp', 320454), ('hybrid', 350460)]:
            with self.subTest(kind=kind):
                model = build_model(self.arch(kind)).eval()
                self.assertEqual(tuple(model(x).shape), (3, 6))
                self.assertEqual(model.count_parameters(), count)
                self.assertEqual(model.architecture()['n_parameters'], count)
                buf = io.BytesIO()
                torch.save(model.state_dict(), buf)
                buf.seek(0)
                restored = build_model(model.architecture()).eval()
                restored.load_state_dict(torch.load(buf, weights_only=True), strict=True)
                torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)

    def test_hybrid_branches_receive_gradients(self):
        model = build_model(self.arch('hybrid'))
        nn.CrossEntropyLoss()(model(torch.randn(3, 5000)), torch.tensor([0, 1, 2])).backward()
        for branch in (model.bp, model.linear):
            self.assertTrue(all(p.grad is not None and torch.count_nonzero(p.grad) > 0
                                for p in branch.parameters()))

    def test_legacy_bp_checkpoint_and_defaults(self):
        torch.manual_seed(42)
        old = BPNet(5000, 64, 6, 0.2).eval()
        legacy = dict(input_dim=5000, hidden_dim=64, output_dim=6, dropout=0.2)
        restored = build_model(legacy).eval()
        restored.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(restored.input_drop.p, 0.0)
        self.assertEqual(set(old.state_dict()), {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'})
        x = torch.randn(3, 5000)
        torch.testing.assert_close(old(x), restored(x), rtol=0, atol=0)
        torch.testing.assert_close(old(x), old.fc2(old.drop(old.act(old.fc1(x)))), rtol=0, atol=0)

    def test_cli_help(self):
        result = subprocess.run([sys.executable, 'train_bp.py', '--help'],
                                capture_output=True, check=True)
        self.assertIn(b'--model-type {bp,linear,hybrid}', result.stdout)


if __name__ == '__main__':
    unittest.main()

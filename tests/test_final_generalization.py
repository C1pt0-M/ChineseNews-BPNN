"""Pre-test checks do not evaluate any checkpoint on test data."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
F8=ROOT/'results/member_C/f8_experiments'
FINAL=ROOT/'results/member_C/final_generalization'
sys.path.insert(0,str(F8))
spec=importlib.util.spec_from_file_location('final_eval_tests',FINAL/'evaluate_fixed.py')
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

class FinalEvaluationTests(unittest.TestCase):
    def test_locked_artifact_manifest_is_complete(self):
        import json
        plan=json.loads((FINAL/'preflight.json').read_text(encoding='utf8'))
        self.assertEqual(len(plan['locked_runs']),15)
        manifest=evaluation.validate_locked(plan)
        self.assertEqual(manifest['encoder_trainable_parameters'],0)
        self.assertEqual(manifest['hidden_size'],512)

    @unittest.skipUnless(torch.cuda.is_available(),'Original experiment used CUDA')
    def test_encoder_matches_existing_train_cache(self):
        p=evaluation.p
        splits,_=p.splits_and_baseline()
        vectors,manifest=p.load_vectors(splits)
        observed,info=evaluation.encode_test(splits['train'][0][:32],manifest)
        np.testing.assert_allclose(observed,vectors['train'][:32],rtol=0,atol=1e-5)
        self.assertEqual(info['encoder_trainable_parameters'],0)
        self.assertEqual(observed.shape,(32,512))

if __name__=='__main__': unittest.main()

"""Checks the paired error accounting used by the F.3 report."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('f3_analysis',ROOT/'results/member_C/f3_analysis/f3_a_multiseed/analyze.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class PairedErrorTests(unittest.TestCase):
    def test_transition_partition_and_net_change(self):
        truth=[0,1,2,0,1,2]
        old=[0,0,1,0,1,0]
        new=[0,1,2,1,1,1]
        got=module.transitions(truth,old,new)
        self.assertEqual(got,{'both_correct':2,'corrected':2,'introduced':1,'both_wrong':1})
        self.assertEqual(sum(got.values()),len(truth))
        self.assertEqual(got['corrected']-got['introduced'],sum(a==b for a,b in zip(truth,new))-sum(a==b for a,b in zip(truth,old)))


if __name__=='__main__':unittest.main()

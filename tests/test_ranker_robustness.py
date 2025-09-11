import os, sys, unittest
PROJECT_DIR = os.path.dirname(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from probability_ranker import ProbabilityRanker

class TestRankerRobustness(unittest.TestCase):
    def test_missing_payoffs_do_not_crash(self):
        r = ProbabilityRanker()
        # One opp is missing profit_if_no; another has both
        opps = [
            { 'profit_if_yes': 10.0, 'polymarket': {'yes_price': 0.6} },
            { 'profit_if_yes': 5.0, 'profit_if_no': 4.0, 'polymarket': {'yes_price': 0.4} },
        ]
        out = r.rank_opportunities(opps)
        self.assertEqual(len(out), 2)
        for o in out:
            self.assertIn('metrics', o)
            self.assertIn('probabilities', o)

    def test_missing_spot_graceful(self):
        r = ProbabilityRanker()
        opp = { 'profit_if_yes': 2.0, 'profit_if_no': 1.0, 'polymarket': {'yes_price': 0.55} }
        out = r.rank_opportunities([opp])
        self.assertEqual(len(out), 1)
        self.assertIn('blended', out[0]['probabilities'])
        self.assertGreaterEqual(out[0]['probabilities']['blended'], 0.0)
        self.assertLessEqual(out[0]['probabilities']['blended'], 1.0)
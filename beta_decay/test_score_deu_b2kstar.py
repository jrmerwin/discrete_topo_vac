"""Run before registration; uses frozen v0.1 inputs and synthetic v0.2 data only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import contextlib
import io
import numpy as np
import pandas as pd
import score_deu_b2kstar as score

ROOT = Path(__file__).resolve().parent


class Regression(unittest.TestCase):
    def test_v01_byte_identical(self):
        for fits, fixture in ((False, 'v01'), (True, 'v01_with_fits')):
            with tempfile.TemporaryDirectory() as tmp:
                command = [sys.executable, str(ROOT / 'score_deu_b2kstar.py'), '--measurements',
                           str(ROOT / 'regression_fixtures/measurements_v01.csv'), '--out-dir', tmp]
                if fits:
                    command += ['--fits', str(ROOT / 'regression_fixtures/fits_v01.csv')]
                subprocess.run(command, check=True, capture_output=True)
                for name in ('primary_scored.csv', 'cp_null_scored.csv', 'fit_scored.csv', 'summary.json'):
                    self.assertEqual((Path(tmp) / name).read_bytes(), (ROOT / 'regression_fixtures' / fixture / name).read_bytes(), name)
                summary = json.loads((Path(tmp) / 'summary.json').read_text())
                self.assertEqual(summary['n_primary_matches'], 5)
                self.assertAlmostEqual(summary['directional_stouffer_z'], 5.883356999671099, places=13)


class SyntheticDiscriminators(unittest.TestCase):
    def test_pending_null_skipped_once_even_with_no_data(self):
        pending = dict(id='absorptive_parts_vs_dispersive', null_mode='ld_baseline', status='PENDING_AUTHOR_NUMBERS')
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = score.score_cp_nulls(pd.DataFrame(), dict(null_predictions=[pending]))
        self.assertTrue(result.empty)
        self.assertEqual(stream.getvalue(), 'Skipped null absorptive_parts_vs_dispersive: PENDING_AUTHOR_NUMBERS\n')

    def test_ld_null_uses_separate_baseline(self):
        df = pd.DataFrame([dict(dataset='synthetic', observable='S7', q2_low=1, q2_high=2,
                                value=5, err_total=3, sm_value=1, sm_err=4,
                                baseline_ld=4, baseline_ld_err=4)])
        card = dict(null_predictions=[dict(id='ld', observable_regex='^S7$', null_mode='ld_baseline')])
        self.assertAlmostEqual(score.score_cp_nulls(df, card).iloc[0].z_to_null, .2)
        df['baseline_ld_err'] = np.nan
        self.assertTrue(score.score_cp_nulls(df, card).empty)

    def test_ld_is_separate_and_missing_values_are_unscored(self):
        df = pd.DataFrame([dict(dataset='synthetic', observable='S7', q2_low=1, q2_high=2,
                                value=5, err_total=3, sm_value=1, sm_err=4,
                                baseline_ld=4, baseline_ld_err=4)])
        result = score.score_ld_comparison(df)
        self.assertAlmostEqual(result.iloc[0].z_to_ld, .2)
        self.assertEqual(df.iloc[0].sm_value, 1)
        df['baseline_ld'] = ''
        self.assertTrue(score.score_ld_comparison(df).empty)

    def test_lepton_matching(self):
        pred = dict(q2_low=1, q2_high=2, lepton='e')
        self.assertFalse(score.bins_match(pd.Series(dict(q2_low=1, q2_high=2, lepton='')), pred))
        self.assertTrue(score.bins_match(pd.Series(dict(q2_low=1, q2_high=2, lepton='e')), pred))
        del pred['lepton']
        self.assertTrue(score.bins_match(pd.Series(dict(q2_low=1, q2_high=2, lepton='e')), pred))

    def test_sm_null_uses_combined_error_and_skips_missing_baseline(self):
        df = pd.DataFrame([dict(dataset='synthetic', observable='S7', q2_low=1, q2_high=2,
                                value=5, err_total=3, sm_value=1, sm_err=4)])
        card = dict(null_predictions=[dict(id='absorptive_parts_zero', observable_regex='^S7$', null_mode='sm')])
        result = score.score_cp_nulls(df, card)
        self.assertAlmostEqual(result.iloc[0].z_to_null, .8)
        df['sm_value'] = np.nan
        self.assertTrue(score.score_cp_nulls(df, card).empty)

    def test_two_sided_fit_does_not_fail_signed_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fits.csv'
            pd.DataFrame([dict(dataset='synthetic', parameter='lfu', value=3, err_total=1)]).to_csv(path, index=False)
            card = dict(fit_predictions=[dict(id='lfu', parameter='lfu', target=0, null_mode='two_sided')])
            result = score.score_fit_results(path, card)
            self.assertTrue(result.iloc[0].danger)
            summary = score.summarize_fits(result)
            summary.update(directional_stouffer_z=3, sign_hit_rate=1, cp_danger_count=0)
            self.assertEqual(score.classify(summary), 'amber')
            summary['cp_danger_count'] = 1
            self.assertEqual(score.classify(summary), 'red')

    def test_flatness_slope_and_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bins.csv'
            rows = [dict(dataset='synthetic', parameter='c9', q2_low=i, q2_high=i+1,
                         helicity=h, value=i + offset, err_total=1, notes='synthetic')
                    for h, offset in [('K*0', 0), ('K*par', 10)] for i in range(3)]
            pd.DataFrame(rows).to_csv(path, index=False)
            card = dict(flatness_predictions=[dict(id='flat', parameter='c9', group_by='helicity', test='chi2_to_weighted_mean')])
            result = score.score_flatness(path, card)
            self.assertEqual(list(result.dof), [2, 2, 1])
            self.assertAlmostEqual(result.iloc[0].chi2, 2)
            self.assertAlmostEqual(result.iloc[0].slope_z, np.sqrt(2))
            self.assertAlmostEqual(result.iloc[-1].chi2, 150)
            self.assertEqual(score.summarize_flatness(result)['flat_danger_count'], 1)
            pd.DataFrame(rows[:1]).to_csv(path, index=False)
            single = score.score_flatness(path, card)
            self.assertTrue(single.p_value.isna().all())
            self.assertFalse(single.danger.any())


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(Regression))
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticDiscriminators))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not result.wasSuccessful())

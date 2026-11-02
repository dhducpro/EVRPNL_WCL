"""Regression checks for paired reporting and campaign validation."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from verify_key_results import (
    FILES, RESULTS, ROOT, validate_numeric, load_manifest, load_starts,
    code_fingerprint, source_matches,
)
from evrp_dwc.result_statistics import (
    instance_summary, pair_ablation, matched_ablation_summary,
)


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if (RESULTS / 'run_info.json').is_file():
            cls.frames = {name: pd.read_csv(RESULTS / name / filename)
                          for name, filename in FILES.items()}
            cls.starts = load_starts(RESULTS, cls.frames['ablation_frvcp'])

    def test_mean_of_percentages_is_not_ratio_of_means(self):
        full = pd.DataFrame(dict(instance=['small', 'large'], ablation=['baseline']*2,
                                 best_cost=[10., 100.]))
        variant = full.assign(ablation='removed', best_cost=[11., 98.])
        paired = pair_ablation(pd.concat([full, variant]), keys=['instance'])
        self.assertAlmostEqual(paired.cost_change_pct.mean(), 4.)
        self.assertLess(variant.best_cost.mean() / full.best_cost.mean() - 1, 0.)

    def test_ci_uses_instances(self):
        frame = pd.DataFrame(dict(instance=['a']*5 + ['b']*5,
                                  reduction_pct=[0.]*5 + [10.]*5))
        summary = instance_summary(frame)
        self.assertEqual(summary['n_instances'], 2)
        self.assertAlmostEqual(summary['mean'], 5.)
        self.assertGreater(summary['ci95'], 60.)

    def test_missing_full_pair_fails(self):
        frame = pd.DataFrame(dict(instance=['a', 'b'], ablation=['baseline', 'removed'],
                                  best_cost=[10., 11.]))
        with self.assertRaisesRegex(ValueError, 'no matched Full'):
            pair_ablation(frame, keys=['instance'])

    def test_code_fingerprint_ignores_documentation(self):
        source = '"""Module description."""\ndef objective(x):\n    """Cost."""\n    return x + 1\n'
        documented = '# Routing objective\n' + source.replace('Cost.', 'Route cost.')
        self.assertEqual(code_fingerprint(source), code_fingerprint(documented))
        self.assertNotEqual(code_fingerprint(source), code_fingerprint(source.replace('x + 1', 'x + 2')))

    def test_distribution_source_rejects_logic_changes(self):
        records = json.loads((ROOT / 'data/source_fingerprints.json').read_text(encoding='utf-8'))
        for name, record in records.items():
            source_path = ROOT / 'src/evrp_dwc' / name
            self.assertTrue(source_matches(source_path, record['campaign_sha256']))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / name
                path.write_text(source_path.read_text(encoding='utf-8') + '\nraise RuntimeError("changed logic")\n', encoding='utf-8')
                self.assertFalse(source_matches(path, record['campaign_sha256']))

    @unittest.skipUnless((RESULTS / 'run_info.json').is_file(), 'Import the result ZIP first')
    def test_ablation_pair_counts(self):
        result = matched_ablation_summary(self.frames['ablation_frvcp'], self.starts).set_index('setting')
        self.assertEqual(result.loc['no_shaking', 'n_start_pairs'],
                         5 * result.loc['no_shaking', 'n_overlap'])
        self.assertEqual(result.loc['no_candidate_filtering', 'n_overlap'], 20)
        self.assertEqual(result.loc['no_filter_no_extended', 'n_start_pairs'], 100)

    @unittest.skipUnless((RESULTS / 'run_info.json').is_file(), 'Import the result ZIP first')
    def test_incomplete_grid_fails(self):
        frames = {k: v.copy() for k, v in self.frames.items()}
        frames['omega_sensitivity'] = frames['omega_sensitivity'].iloc[1:]
        with self.assertRaisesRegex(ValueError, 'experimental grid'):
            validate_numeric(frames, RESULTS)

    @unittest.skipUnless((RESULTS / 'run_info.json').is_file(), 'Import the result ZIP first')
    def test_wrong_source_or_budget_fails(self):
        for field, value, message in [('max_moves_per_iteration', 1200, 'protocol'),
                                      ('search_sha256', '0'*64, 'Source hash')]:
            info = load_manifest(RESULTS)
            info['protocol'][field] = value
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path/'run_info.json').write_text(json.dumps(info))
                with self.assertRaisesRegex(ValueError, message):
                    validate_numeric(self.frames, path)


if __name__ == '__main__':
    unittest.main()

import math
import unittest
import sys
import os

import numpy as np

# Add root to path to import cross_space_logs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cross_space_logs as csl


def _refine_block(mode):
    """A fully-populated per-mode refinement block, shaped like cross_space_projection._refine_block."""
    return {
        'refine_enabled':             True,
        'refine_mode':                mode,
        'refine_best_epoch':          100,
        'refine_best_val_total':      0.01,
        'refine_val_selection':       'balanced_val',
        'proj_anchor_loss_before':    0.04,
        'proj_anchor_loss_after':     0.08,
        'mae_micro_old_oncsv_before': 1.10,
        'mae_macro_old_oncsv_before': 1.05,
        'mae_micro_old_oncsv_after':  0.90,
        'mae_macro_old_oncsv_after':  1.00,
        'mae_micro_new_test_before':  1.15,
        'mae_macro_new_test_before':  1.15,
        'mae_micro_new_test_after':   1.16,
        'mae_macro_new_test_after':   1.16,
        'refine_old_model_csv':       'test',
        'refine_new_eval_split':      'val',
        'config': {
            'epochs': 150, 'lr_projector': 1e-4, 'lr_linear': 1e-4,
            'lambda_B': 1e-4, 'lambda_A': 1e-3, 'optimizer': 'adamw',
            'weight_decay': 0, 'loss': 'mse', 'batch_size': 64,
        },
        'per_epoch_metrics': {},
        'new_test_eval': {
            'labels':       np.array([0., 1., 2., 4.], dtype=np.float32),
            'preds_before': np.array([0.1, 1.1, 2.1, 3.9], dtype=np.float32),
            'preds_after':  np.array([0.2, 1.2, 2.2, 3.8], dtype=np.float32),
        },
    }


def _linear_projector_block():
    """A minimal linear_projector bundle (only the fields the summary columns read)."""
    return {
        'config': {
            'epochs': 750, 'lr': 1e-5, 'batch_size': 64, 'optimizer': 'adamw',
            'weight_decay': 0, 'normalize_embeddings': False, 'loss': 'mse',
        },
        'best_epoch': 100,
    }


def _grid_data(refinement=None, refinements=None):
    """A minimal grid-format pkl dict (trial_params present) for _collect_summary_row(s)."""
    d = {
        'trial_number': 0,
        'seed': 42,
        'trial_params': {
            'num_anchors': 250,
            'anchor_selection_type': 'balance_class_random',
            'csv_anchor_selection': 'train',
            'old_model_csv': 'test',
            'interpolation_similarity': 'linear',
            'weighting_method': 'none',
            'rbf_sigma': 1.0,
        },
        'metrics': {'mae': 0.75, 'ccc': 0.5, 'runtime_min': 1.0},
        'new_model_tensors': {
            'predictions': np.array([0., 1., 2., 3.], dtype=np.float32),
            'labels':      np.array([0., 1., 2., 4.], dtype=np.float32),
        },
        'old_model_tensors': {
            'predictions': np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        },
        'linear_projector': _linear_projector_block(),
    }
    if refinement is not None:
        d['refinement'] = refinement
    if refinements is not None:
        d['refinements'] = refinements
    return d


def _standalone_data(refinement=None, refinements=None):
    """A minimal standalone/subtrial pkl dict (config_cross_space_projection, no trial_params)."""
    d = {
        'seed': 42,
        'config_cross_space_projection': {
            'num_anchors': 250,
            'anchor_selection_type': 'balance_class_random',
            'csv_anchor_selection': 'train',
            'old_model_csv': 'test',
            'interpolation_similarity': 'linear',
            'weighting_method': 'none',
            'rbf_sigma': 1.0,
            'new_model_pth': 'new_model.pt',
            'old_model_pth': 'old_model.pt',
        },
        'metrics': {'mae': 0.75, 'ccc': 0.5, 'runtime_min': 1.0},
        'new_model_tensors': {
            'predictions': np.array([0., 1., 2., 3.], dtype=np.float32),
            'labels':      np.array([0., 1., 2., 4.], dtype=np.float32),
        },
        'old_model_tensors': {
            'predictions': np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32),
        },
        'linear_projector': _linear_projector_block(),
    }
    if refinement is not None:
        d['refinement'] = refinement
    if refinements is not None:
        d['refinements'] = refinements
    return d


class TestRefinementItems(unittest.TestCase):
    def test_plural_key(self):
        d = {'refinements': {'linear_only': {'a': 1}, 'projector_linear': {'b': 2}}}
        items = csl._refine_items(d)
        self.assertEqual([m for m, _ in items], ['linear_only', 'projector_linear'])

    def test_singular_key(self):
        d = {'refinement': {'refine_mode': 'linear_only'}}
        items = csl._refine_items(d)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0][0], 'linear_only')

    def test_no_refinement(self):
        self.assertEqual(csl._refine_items({}), [])


class TestCollectSummaryRows(unittest.TestCase):
    def setUp(self):
        # These best-effort title helpers lazily import the heavy cross_space_projection
        # stack and touch model-checkpoint paths; stub them out — irrelevant to refinement columns.
        self._orig_old = csl._resolve_old_dataset
        self._orig_new = csl._resolve_new_dataset
        csl._resolve_old_dataset = lambda *a, **k: None
        csl._resolve_new_dataset = lambda *a, **k: None

    def tearDown(self):
        csl._resolve_old_dataset = self._orig_old
        csl._resolve_new_dataset = self._orig_new

    def test_multi_mode_yields_one_row_per_mode(self):
        d = _grid_data(refinements={
            'linear_only':      _refine_block('linear_only'),
            'projector_linear': _refine_block('projector_linear'),
        })
        rows = csl._collect_summary_rows(d, 'dummy.pkl')
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['refine_mode'] for r in rows}, {'linear_only', 'projector_linear'})
        for r in rows:
            self.assertTrue(r['refine_enabled'])
            # Refinement-sourced columns populated (previously empty for --refinement 3).
            self.assertAlmostEqual(r['srctest_mae_micro_before'], 1.10, places=5)
            self.assertAlmostEqual(r['mae_micro_old_oncsv_after'], 0.90, places=5)
            self.assertTrue(math.isfinite(r['newtest_mae_micro_before']))
            self.assertAlmostEqual(r['ref_lr_projector'], 1e-4, places=8)
            self.assertTrue(math.isfinite(r['refine_best_epoch_frac']))

    def test_single_mode_backward_compatible(self):
        d = _grid_data(refinement=_refine_block('linear_only'))
        rows = csl._collect_summary_rows(d, 'dummy.pkl')
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertTrue(r['refine_enabled'])
        self.assertEqual(r['refine_mode'], 'linear_only')
        self.assertAlmostEqual(r['srctest_mae_micro_before'], 1.10, places=5)

    def test_no_refinement_defaults(self):
        d = _grid_data()
        rows = csl._collect_summary_rows(d, 'dummy.pkl')
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertFalse(r['refine_enabled'])
        self.assertIsNone(r['refine_mode'])
        self.assertTrue(math.isnan(r['srctest_mae_micro_before']))
        # srctest *_old always comes from the direct old-model MAE, not the refinement block.
        self.assertTrue(math.isfinite(r['srctest_mae_micro_old']))


class TestAggregatedSummaryRows(unittest.TestCase):
    def setUp(self):
        self._orig_old = csl._resolve_old_dataset
        self._orig_new = csl._resolve_new_dataset
        csl._resolve_old_dataset = lambda *a, **k: None
        csl._resolve_new_dataset = lambda *a, **k: None

    def tearDown(self):
        csl._resolve_old_dataset = self._orig_old
        csl._resolve_new_dataset = self._orig_new

    def test_subtrial_expands_per_mode_with_ident_columns(self):
        d = _standalone_data(refinements={
            'linear_only':      _refine_block('linear_only'),
            'projector_linear': _refine_block('projector_linear'),
        })
        rows = csl._aggregated_summary_rows(d, 'dummy.pkl', subtrial_index=3, n_subtrials=25)
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertEqual(r['subtrial_index'], 3)
            self.assertEqual(r['n_subtrials'], 25)
            self.assertEqual(r['new_model_pth'], 'new_model.pt')
            self.assertTrue(r['refine_enabled'])
            self.assertAlmostEqual(r['srctest_mae_micro_before'], 1.10, places=5)

    def test_aggregated_summary_row_singular_wrapper(self):
        # Aggregated (pooled) pkl carries no refinement stage -> exactly one row, empty refine cols.
        d = _standalone_data()
        row = csl._aggregated_summary_row(d, 'dummy.pkl', 'AGGREGATE', 25)
        self.assertEqual(row['subtrial_index'], 'AGGREGATE')
        self.assertFalse(row['refine_enabled'])


def _refine_block_after(mode, micro_after, macro_after):
    """A _refine_block with the source after-refinement MAE overridden (for aggregation tests)."""
    rb = _refine_block(mode)
    rb['mae_micro_old_oncsv_after'] = micro_after
    rb['mae_macro_old_oncsv_after'] = macro_after
    return rb


def _sub_rows_two_subtrials():
    """Per-subtrial x per-mode summary rows for 2 subtrials (linear_only + projector_linear).

    The source after-refinement MAE varies per subtrial so mean/std are non-trivial:
      linear_only      after micro: [0.90, 0.94]  -> mean 0.92, std(ddof=1) ~0.0283
      projector_linear after micro: [0.80, 0.84]  -> mean 0.82
    Every other stage column is constant (from _refine_block / the fixed tensors).
    """
    rows = []
    for idx, (lo_micro, pl_micro) in enumerate([(0.90, 0.80), (0.94, 0.84)]):
        d = _standalone_data(refinements={
            'linear_only':      _refine_block_after('linear_only', lo_micro, 1.00),
            'projector_linear': _refine_block_after('projector_linear', pl_micro, 1.00),
        })
        rows.extend(csl._aggregated_summary_rows(d, 'dummy.pkl', subtrial_index=idx, n_subtrials=2))
    return rows


class TestAggregateSubtrialRows(unittest.TestCase):
    def setUp(self):
        self._orig_old = csl._resolve_old_dataset
        self._orig_new = csl._resolve_new_dataset
        csl._resolve_old_dataset = lambda *a, **k: None
        csl._resolve_new_dataset = lambda *a, **k: None

    def tearDown(self):
        csl._resolve_old_dataset = self._orig_old
        csl._resolve_new_dataset = self._orig_new

    def test_two_rows_per_mode_with_identifier_columns(self):
        rows = csl._aggregate_subtrial_rows(_sub_rows_two_subtrials(), n_subtrials=2)
        self.assertEqual(len(rows), 4)  # (mean + std) x 2 modes
        by_key = {(r['refine_mode'], r['subtrial_index']): r for r in rows}
        self.assertEqual(
            set(by_key),
            {('linear_only', 'AGGREGATE_MEAN'), ('linear_only', 'AGGREGATE_STD'),
             ('projector_linear', 'AGGREGATE_MEAN'), ('projector_linear', 'AGGREGATE_STD')},
        )
        for r in rows:
            self.assertEqual(r['n_subtrials'], 2)

    def test_mean_and_std_of_stage_column(self):
        rows = csl._aggregate_subtrial_rows(_sub_rows_two_subtrials(), n_subtrials=2)
        lo_mean = next(r for r in rows if r['refine_mode'] == 'linear_only'
                       and r['subtrial_index'] == 'AGGREGATE_MEAN')
        lo_std  = next(r for r in rows if r['refine_mode'] == 'linear_only'
                       and r['subtrial_index'] == 'AGGREGATE_STD')
        # srctest_mae_micro_after == the per-mode source after-refinement MAE.
        self.assertAlmostEqual(lo_mean['srctest_mae_micro_after'], 0.92, places=6)
        self.assertAlmostEqual(lo_std['srctest_mae_micro_after'],
                               float(np.std([0.90, 0.94], ddof=1)), places=6)
        pl_mean = next(r for r in rows if r['refine_mode'] == 'projector_linear'
                       and r['subtrial_index'] == 'AGGREGATE_MEAN')
        self.assertAlmostEqual(pl_mean['srctest_mae_micro_after'], 0.82, places=6)

    def test_single_subtrial_std_is_zero(self):
        # One subtrial -> a group of size 1 -> std defined as 0.0 (not NaN).
        rows = csl._aggregate_subtrial_rows(_sub_rows_two_subtrials()[:2], n_subtrials=1)
        std_rows = [r for r in rows if r['subtrial_index'] == 'AGGREGATE_STD']
        self.assertTrue(std_rows)
        for r in std_rows:
            self.assertEqual(r['srctest_mae_micro_after'], 0.0)

    def test_non_numeric_columns_preserved(self):
        rows = csl._aggregate_subtrial_rows(_sub_rows_two_subtrials(), n_subtrials=2)
        mean_row = next(r for r in rows if r['refine_mode'] == 'linear_only'
                        and r['subtrial_index'] == 'AGGREGATE_MEAN')
        self.assertEqual(mean_row['new_model_pth'], 'new_model.pt')
        self.assertEqual(mean_row['interpolation_similarity'], 'linear')


class TestPerModeStageMeans(unittest.TestCase):
    def setUp(self):
        self._orig_old = csl._resolve_old_dataset
        self._orig_new = csl._resolve_new_dataset
        csl._resolve_old_dataset = lambda *a, **k: None
        csl._resolve_new_dataset = lambda *a, **k: None

    def tearDown(self):
        csl._resolve_old_dataset = self._orig_old
        csl._resolve_new_dataset = self._orig_new

    def test_stage_means_per_mode(self):
        stages_by_mode = csl._per_mode_stage_means(_sub_rows_two_subtrials())
        self.assertEqual(set(stages_by_mode), {'linear_only', 'projector_linear'})
        lo = stages_by_mode['linear_only']
        self.assertEqual(set(lo), {'old', 'projected', 'refined', 'preserve_before', 'preserve_after'})
        # old = |old_preds - labels| micro = 0.5; projected(before) = 1.10; refined(after) = 0.92.
        self.assertAlmostEqual(lo['old'][0], 0.5, places=6)
        self.assertAlmostEqual(lo['projected'][0], 1.10, places=6)
        self.assertAlmostEqual(lo['refined'][0], 0.92, places=6)
        self.assertAlmostEqual(stages_by_mode['projector_linear']['refined'][0], 0.82, places=6)

    def test_no_refinement_omits_projected_and_refined(self):
        # No refinement stage -> before/after source columns are NaN -> those stages dropped,
        # but the 'old' stage (direct old-model MAE) is still present.
        d = _standalone_data()
        sub_rows = csl._aggregated_summary_rows(d, 'dummy.pkl', subtrial_index=0, n_subtrials=1)
        stages_by_mode = csl._per_mode_stage_means(sub_rows)
        self.assertEqual(len(stages_by_mode), 1)
        stages = next(iter(stages_by_mode.values()))
        self.assertIn('old', stages)
        self.assertNotIn('refined', stages)
        self.assertNotIn('projected', stages)


class TestPerModeStageStds(unittest.TestCase):
    def setUp(self):
        self._orig_old = csl._resolve_old_dataset
        self._orig_new = csl._resolve_new_dataset
        csl._resolve_old_dataset = lambda *a, **k: None
        csl._resolve_new_dataset = lambda *a, **k: None

    def tearDown(self):
        csl._resolve_old_dataset = self._orig_old
        csl._resolve_new_dataset = self._orig_new

    def test_std_keys_match_mean_keys(self):
        # The std companion must expose exactly the same modes/stages as the means so the
        # dashboard can pair each mean cell with its std.
        sub_rows = _sub_rows_two_subtrials()
        means = csl._per_mode_stage_means(sub_rows)
        stds  = csl._per_mode_stage_stds(sub_rows)
        self.assertEqual(set(stds), set(means))
        for mode in means:
            self.assertEqual(set(stds[mode]), set(means[mode]))

    def test_std_of_stage_column(self):
        stds = csl._per_mode_stage_stds(_sub_rows_two_subtrials())
        # linear_only source after-refinement micro MAE: [0.90, 0.94] -> ddof=1 std.
        self.assertAlmostEqual(stds['linear_only']['refined'][0],
                               float(np.std([0.90, 0.94], ddof=1)), places=6)

    def test_single_subtrial_std_is_zero(self):
        # A lone subtrial -> ddof=1 std is NaN -> reported as 0.0, but the stage stays present.
        stds = csl._per_mode_stage_stds(_sub_rows_two_subtrials()[:2])
        self.assertIn('refined', stds['linear_only'])
        self.assertEqual(stds['linear_only']['refined'][0], 0.0)


if __name__ == '__main__':
    unittest.main()

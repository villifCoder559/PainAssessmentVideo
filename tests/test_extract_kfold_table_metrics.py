import io
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from extract_kfold_test_table import extract_table, main


def write_results(path: Path, criterion: str = 'L1Loss()') -> None:
  data = {
    'config': {'criterion': criterion},
    'results': {
      'k0_cross_val_final': {
        'test': {
          'test_l1_error': 1.0,
          'test_loss_per_class': np.array([0.5, 1.5]),
          'test_accuracy': 0.5,
          'test_accuracy_per_class': np.array([0.75, 0.25]),
          'test_unique_y': np.array([0, 1]),
          'test_count_y': np.array([2, 2]),
          'test_count_subject_ids': np.array([2, 2]),
        },
      },
      'k1_cross_val_final': {
        'test': {
          'test_l1_error': 0.625,
          'test_loss_per_class': np.array([0.25, 0.75]),
          'test_accuracy': 0.75,
          'test_accuracy_per_class': np.array([1.0, 0.5]),
          'test_unique_y': np.array([0, 1]),
          'test_count_y': np.array([1, 3]),
          'test_count_subject_ids': np.array([1, 3]),
        },
      },
    },
  }
  with path.open('wb') as file:
    pickle.dump(data, file)


def raw_metrics():
  return {
    'k0_cross_val_final': {
      'raw_mae': 0.9,
      'raw_mae_per_class': {0: 0.4, 1: 1.4},
      'recomputed_l1': 1.0,
      'recomputed_accuracy': 0.6,
      'recomputed_accuracy_per_class': np.array([0.8, 0.4]),
      'n_samples': 4,
    },
    'k1_cross_val_final': {
      'raw_mae': 0.6,
      'raw_mae_per_class': {0: 0.2, 1: 0.7},
      'recomputed_l1': 0.625,
      'recomputed_accuracy': 0.8,
      'recomputed_accuracy_per_class': np.array([0.9, 0.7]),
      'n_samples': 4,
    },
  }


class TestMetricSelection(unittest.TestCase):
  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp_dir.cleanup)
    self.pkl_path = Path(self.temp_dir.name) / 'k_fold_results.pkl'
    write_results(self.pkl_path)

  def test_accuracy_report_uses_percentages_and_summary_rows(self):
    table = extract_table(str(self.pkl_path), metric='accuracy')

    self.assertEqual(table.columns.tolist(), [
      'fold', 'test_accuracy_pct',
      'accuracy_class_0_pct', 'n_class_0',
      'accuracy_class_1_pct', 'n_class_1',
      'n_samples', 'n_subjects',
    ])
    self.assertEqual(table['fold'].tolist(), ['k0', 'k1', 'mean', 'std'])
    np.testing.assert_allclose(table['test_accuracy_pct'][:3], [50.0, 75.0, 62.5])
    np.testing.assert_allclose(table['accuracy_class_0_pct'][:3], [75.0, 100.0, 87.5])
    np.testing.assert_allclose(table['accuracy_class_1_pct'][:3], [25.0, 50.0, 37.5])
    self.assertAlmostEqual(table.loc[3, 'test_accuracy_pct'], 17.6776695297)
    self.assertEqual(table.loc[:1, 'n_samples'].tolist(), [4.0, 4.0])
    self.assertTrue(pd.isna(table.loc[2, 'n_samples']))

  def test_raw_accuracy_uses_recomputed_overall_and_per_class_values(self):
    with mock.patch('extract_kfold_test_table.recompute_raw_fold_metrics', return_value=raw_metrics()):
      table = extract_table(str(self.pkl_path), raw=True, metric='accuracy')

    self.assertEqual(table.columns.tolist(), [
      'fold', 'test_accuracy_pct', 'test_accuracy_raw_pct',
      'accuracy_class_0_pct', 'n_class_0',
      'accuracy_class_1_pct', 'n_class_1',
      'n_samples', 'n_subjects',
    ])
    np.testing.assert_allclose(table.loc[:1, 'test_accuracy_pct'], [50.0, 75.0])
    np.testing.assert_allclose(table.loc[:1, 'test_accuracy_raw_pct'], [60.0, 80.0])
    np.testing.assert_allclose(table.loc[:1, 'accuracy_class_0_pct'], [80.0, 90.0])
    np.testing.assert_allclose(table.loc[:1, 'accuracy_class_1_pct'], [40.0, 70.0])

  def test_mae_remains_the_default_metric(self):
    default_table = extract_table(str(self.pkl_path))
    explicit_table = extract_table(str(self.pkl_path), metric='mae')

    pd.testing.assert_frame_equal(default_table, explicit_table)

  def test_unknown_metric_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "metric must be 'mae' or 'accuracy'"):
      extract_table(str(self.pkl_path), metric='rmse')

  def test_l1_criterion_warning_is_limited_to_mae_reports(self):
    non_l1_path = self.pkl_path.parent / 'non_l1_results.pkl'
    write_results(non_l1_path, criterion='MSELoss()')

    mae_output = io.StringIO()
    with redirect_stdout(mae_output):
      extract_table(str(non_l1_path), metric='mae')
    accuracy_output = io.StringIO()
    with redirect_stdout(accuracy_output):
      extract_table(str(non_l1_path), metric='accuracy')

    self.assertIn('not L1Loss', mae_output.getvalue())
    self.assertNotIn('not L1Loss', accuracy_output.getvalue())

  def test_cli_creates_metric_specific_default_output_files(self):
    cases = [
      (None, False, 'test_table_mae.csv'),
      ('mae', False, 'test_table_mae.csv'),
      ('mae', True, 'test_table_mae_raw.csv'),
      ('accuracy', False, 'test_table_accuracy.csv'),
      ('accuracy', True, 'test_table_accuracy_raw.csv'),
    ]
    with mock.patch('extract_kfold_test_table.recompute_raw_fold_metrics', return_value=raw_metrics()):
      for metric, use_raw, filename in cases:
        with self.subTest(metric=metric, raw=use_raw):
          output_path = self.pkl_path.parent / filename
          output_path.unlink(missing_ok=True)
          argv = ['extract_kfold_test_table.py', '--pkl', str(self.pkl_path)]
          if metric is not None:
            argv.extend(('--metric', metric))
          if use_raw:
            argv.append('--raw')
          with mock.patch.object(sys, 'argv', argv):
            main()
          self.assertTrue(output_path.is_file())


if __name__ == '__main__':
  unittest.main()

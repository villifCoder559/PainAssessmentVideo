import io
import pickle
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from custom.targets import TargetSpec  # Load torch before scoped sys.modules patches.
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

  def test_macro_mae_averages_classes_and_summarizes_folds(self):
    for use_raw, expected in [
      (False, [1.0, 0.5, 0.75, 0.3535533906]),
      (True, [0.9, 0.45, 0.675, 0.3181980515]),
    ]:
      with self.subTest(raw=use_raw), mock.patch(
        'extract_kfold_test_table.recompute_raw_fold_metrics', return_value=raw_metrics()
      ):
        table = extract_table(str(self.pkl_path), raw=use_raw)
        self.assertIn('test_MAE_macro', table.columns)
        np.testing.assert_allclose(table['test_MAE_macro'], expected)

  def test_macro_mae_uses_present_classes_and_single_fold_std_is_nan(self):
    with self.pkl_path.open('rb') as file:
      data = pickle.load(file)
    test = data['results']['k1_cross_val_final']['test']
    test.update({
      'test_unique_y': np.array([1]),
      'test_loss_per_class': np.array([0.75]),
      'test_count_y': np.array([3]),
      'test_count_subject_ids': np.array([3]),
      'test_l1_error': 0.75,
    })
    with self.pkl_path.open('wb') as file:
      pickle.dump(data, file)
    table = extract_table(str(self.pkl_path))
    self.assertIn('test_MAE_macro', table.columns)
    np.testing.assert_allclose(table['test_MAE_macro'][:3], [1.0, 0.75, 0.875])

    del data['results']['k0_cross_val_final']
    with self.pkl_path.open('wb') as file:
      pickle.dump(data, file)
    table = extract_table(str(self.pkl_path)).set_index('fold')
    self.assertEqual(table.loc['mean', 'test_MAE_macro'], 0.75)
    self.assertTrue(pd.isna(table.loc['std', 'test_MAE_macro']))

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
          csv = pd.read_csv(output_path)
          if metric == 'accuracy':
            self.assertNotIn('test_MAE_macro', csv.columns)
          else:
            self.assertIn('test_MAE_macro', csv.columns)
            expected = [0.9, 0.45, 0.675, 0.3182] if use_raw else [1.0, 0.5, 0.75, 0.3536]
            np.testing.assert_allclose(csv['test_MAE_macro'], expected)

  def test_merged_predictions_cli(self):
    for use_raw, custom_output in [(False, False), (False, True), (True, True)]:
      with self.subTest(raw=use_raw, custom_output=custom_output):
        self.run_prediction_export(use_raw, custom_output)

  def test_export_rejects_missing_predictions_and_duplicate_ids(self):
    for problem, message in [('missing', 'no logged prediction'), ('duplicate', 'duplicate sample_id')]:
      with self.subTest(problem=problem):
        with self.assertRaisesRegex((RuntimeError, ValueError), message):
          self.run_prediction_export(False, True, problem)
        self.assertFalse((self.pkl_path.parent / 'summary_predictions.csv').exists())

  def run_prediction_export(self, use_raw, custom_output, problem=None):
    with self.pkl_path.open('rb') as file:
      data = pickle.load(file)
    data['model_advanced_params'] = {'head': 'HEAD', 'head_params': {}}
    data['config'].update(concatenate_temp_dim=0, concatenate_quadrants=0)
    for key, result in data['results'].items():
      fold = key.split('_')[0]
      result['best_model'] = {'fold_sub_fold_idx': (0, 0), 'best_model_idx': 2}
      folder = self.pkl_path.parent / 'train_HEAD' / f'{fold}_cross_val'
      checkpoint = folder / f'{fold}_cross_val_sub_0' / 'best_model_ep_2.pt'
      checkpoint.parent.mkdir(parents=True, exist_ok=True)
      checkpoint.touch()
      pd.DataFrame({
        'sample_id': [2, 1] if problem != 'duplicate' else [1, 1],
        'sample_name': ['002', '001'], 'class_id': [1, 0], 'subject_id': [8, 9],
      }).to_csv(folder / 'test_cleaned.csv', sep='\t', index=False)
    with self.pkl_path.open('wb') as file:
      pickle.dump(data, file)

    calls = []
    class FakeModel:
      def __init__(self, **kwargs):
        pass

      def test_pretrained_model(self, **kwargs):
        calls.append(kwargs['csv_path'])
        return {
          'history_test_sample_predictions': {1: [0.123456789]} if problem == 'missing'
            else {1: [0.123456789], 2: [0.876543211]},
          'test_l1_error': 0.1, 'test_accuracy': 1.0,
          'test_accuracy_per_class': np.ones(2), 'test_loss_per_subject': np.zeros(2),
          'test_accuracy_per_subject': np.ones(2), 'test_unique_subject_ids': [8, 9],
        }

    helper = types.ModuleType('custom.helper')
    helper.init_log_cross_attention = helper.init_log_video_embeddings = lambda: None
    helper.step_shift = 100
    model = types.ModuleType('custom.model')
    model.Model_Advanced = FakeModel
    output = self.pkl_path.parent / ('summary.csv' if custom_output else 'test_table_mae.csv')
    argv = ['extract_kfold_test_table.py', '--pkl', str(self.pkl_path), '--export-predictions']
    if custom_output:
      argv.extend(['--out', str(output)])
    if use_raw:
      argv.append('--raw')
    with mock.patch.dict(sys.modules, {'custom.helper': helper, 'custom.model': model}), \
         mock.patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
      main()
    merged = pd.read_csv(output.with_name(output.stem + '_predictions.csv'), dtype={'sample_name': str})
    self.assertEqual(merged.columns.tolist(), ['sample_id', 'sample_name', 'class_id', 'subject_id', 'fold', 'prediction'])
    self.assertEqual(merged['fold'].tolist(), ['k0', 'k0', 'k1', 'k1'])
    self.assertEqual(merged['sample_name'].tolist(), ['001', '002', '001', '002'])
    self.assertEqual(merged['subject_id'].tolist(), [9, 8, 9, 8])
    np.testing.assert_allclose(merged['prediction'], [0.123456789, 0.876543211] * 2, rtol=0, atol=1e-12)
    self.assertEqual(len(calls), 2)
    summary = pd.read_csv(output)
    self.assertEqual('test_MAE_raw' in summary, use_raw)


if __name__ == '__main__':
  unittest.main()

import csv
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statistical_compare_kfold_results as comparison


def make_results(subject_order=(0, 1), model_type='MODEL_A'):
  """Build two aligned synthetic k-fold result dictionaries."""
  ids = np.array([10, 11])[list(subject_order)]
  return {
    'config': {
      'model_type': model_type,
      'path_csv_dataset': ['UNBC/starting_point/samples.csv'],
    },
    'model_advanced_params': {'head': 'HEAD'},
    'results': {
      'k0_cross_val_final': {
        'test': {
          'test_unique_subject_ids': ids,
          'test_loss_per_subject': np.array([1.0, 2.0])[list(subject_order)],
          'test_accuracy_per_subject': np.array([0.8, 0.6])[list(subject_order)],
          'test_l1_error': 1.5,
          'test_accuracy': 0.7,
        },
      },
      'k1_cross_val_final': {
        'test': {
          'test_unique_subject_ids': np.array([12, 13]),
          'test_loss_per_subject': np.array([1.5, 1.0]),
          'test_accuracy_per_subject': np.array([0.7, 0.9]),
          'test_l1_error': 1.25,
          'test_accuracy': 0.8,
        },
      },
      'k0_cross_val_sub_0': {},
    },
  }


def shifted_results(model_type='MODEL_B'):
  """Build a second result dictionary where model zero is uniformly better."""
  data = make_results(subject_order=(1, 0), model_type=model_type)
  for result in data['results'].values():
    if 'test' not in result:
      continue
    test = result['test']
    test['test_loss_per_subject'] += 0.2
    test['test_accuracy_per_subject'] -= 0.1
    test['test_l1_error'] += 0.2
    test['test_accuracy'] -= 0.1
  return data


def fresh_metrics(data, mae_delta=0.0):
  """Convert stored tests into the checkpoint-recomputation contract."""
  metrics = {}
  for key, result in data['results'].items():
    if 'final' not in key:
      continue
    test = result['test']
    metrics[key] = {
      'recomputed_l1': test['test_l1_error'] + mae_delta,
      'recomputed_accuracy': test['test_accuracy'],
      'recomputed_loss_per_subject': test['test_loss_per_subject'] + mae_delta,
      'recomputed_accuracy_per_subject': test['test_accuracy_per_subject'],
      'recomputed_subject_ids': test['test_unique_subject_ids'],
    }
  return metrics


def make_video_results(history, model_type='MODEL_A', fold='k0_cross_val_final'):
  """Build one final fold containing stored per-video predictions."""
  return {
    'config': {'model_type': model_type},
    'model_advanced_params': {'head': 'HEAD'},
    'results': {fold: {'test': {'history_test_sample_predictions': history}}},
  }


def write_test_csv(pkl_path, fold, rows):
  """Write the fold CSV layout used by real k-fold result directories."""
  fold_name = fold.split('_')[0]
  path = Path(pkl_path).parent / 'train_HEAD' / f'{fold_name}_cross_val' / 'test_cleaned.csv'
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=('sample_id', 'class_id'), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)


class TestKfoldComparison(unittest.TestCase):
  def test_aligns_subjects_and_reports_positive_effect_for_better_model_zero(self):
    paired = comparison.extract_paired_values(make_results(), shifted_results(), 'subject')
    rows = comparison.compare_paired_values(
      paired, analysis_level='subject', measure='both', metric_source='stored'
    )
    by_metric = {row['metric']: row for row in rows}

    self.assertTrue(np.allclose(paired['mae'][1] - paired['mae'][0], 0.2))
    self.assertAlmostEqual(by_metric['mae']['mean_effect_pkl0_better'], 0.2)
    self.assertAlmostEqual(by_metric['accuracy']['mean_effect_pkl0_better'], 0.1)
    self.assertEqual(by_metric['mae']['metric_source'], 'stored')
    self.assertIsNotNone(by_metric['mae']['ttest_p_holm'])
    self.assertIsNotNone(by_metric['mae']['permutation_p_holm'])

  def test_model_level_uses_one_pair_per_final_fold(self):
    paired = comparison.extract_paired_values(make_results(), shifted_results(), 'model')
    rows = comparison.compare_paired_values(
      paired, analysis_level='model', measure='mae', metric_source='stored'
    )

    self.assertEqual(len(paired['mae'][0]), 2)
    self.assertEqual(rows[0]['analysis_level'], 'model')
    self.assertEqual(rows[0]['analysis_role'], 'exploratory')
    self.assertIsNone(rows[0]['ttest_p_holm'])

  def test_reports_hand_calculated_paired_t_statistic(self):
    paired = {
      'mae': (
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([1.1, 2.2, 3.3, 4.4]),
      ),
    }

    row = comparison.compare_paired_values(
      paired, analysis_level='video', measure='mae', metric_source='stored'
    )[0]

    self.assertAlmostEqual(row['ttest_statistic'], 3.872983346207417, places=12)
    self.assertAlmostEqual(row['ttest_p_raw'], 0.030466291662170977, places=12)
    self.assertTrue(row['ttest_significant'])
    self.assertEqual(row['analysis_role'], 'primary')

  def test_constant_nonzero_differences_do_not_emit_precision_warning(self):
    paired = {'mae': (np.array([1.0, 2.0]), np.array([1.2, 2.2]))}

    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter('always')
      row = comparison.compare_paired_values(
        paired, analysis_level='model', measure='mae', metric_source='stored'
      )[0]

    self.assertEqual(caught, [])
    self.assertEqual(row['ttest_statistic'], float('inf'))
    self.assertEqual(row['ttest_p_raw'], 0.0)

  def test_video_permutation_suppresses_scipy_exact_space_overflow_warning(self):
    def scipy_permutation_warning(*args, **kwargs):
      warnings.warn('overflow encountered in scalar power', RuntimeWarning)
      return mock.Mock(statistic=0.25, pvalue=0.5)

    with mock.patch.object(
      comparison.stats, 'permutation_test', side_effect=scipy_permutation_warning
    ), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        comparison._test_difference(np.array([-0.5, 0.25, 1.0]), 'video')

    messages = [str(item.message) for item in caught]
    self.assertFalse(any('overflow encountered in scalar power' in message for message in messages))

  def test_rejects_mismatched_subjects(self):
    data_1 = shifted_results()
    data_1['results']['k0_cross_val_final']['test']['test_unique_subject_ids'][0] = 99

    with self.assertRaisesRegex(ValueError, 'subject IDs differ'):
      comparison.extract_paired_values(make_results(), data_1, 'subject')

  def test_identical_values_return_unit_p_values(self):
    data = make_results()
    rows = comparison.compare_paired_values(
      comparison.extract_paired_values(data, make_results(), 'subject'),
      analysis_level='subject',
      measure='both',
      metric_source='stored',
    )

    for row in rows:
      self.assertEqual(row['ttest_p_raw'], 1.0)
      self.assertEqual(row['permutation_p_raw'], 1.0)
      self.assertEqual(row['wilcoxon_p_raw'], 1.0)

  def test_measure_filters_result_rows(self):
    rows = comparison.compare_paired_values(
      comparison.extract_paired_values(make_results(), shifted_results(), 'subject'),
      analysis_level='subject',
      measure='accuracy',
      metric_source='stored',
    )
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp, 'comparison.csv')
      comparison.write_rows(output, rows)
      with output.open(newline='') as handle:
        written = list(csv.DictReader(handle))

    self.assertEqual(len(written), 1)
    self.assertEqual(written[0]['metric'], 'accuracy')

  def test_default_output_contains_models_level_measure_and_source(self):
    output = comparison.resolve_output_path(
      None,
      make_results(model_type='VIDEOMAE_v2_S'),
      shifted_results(model_type='DFER'),
      'video',
      'both',
      recompute=True,
    )

    self.assertEqual(
      output,
      Path('statistical_analysis_folder/unbc/VIDEOMAE_v2_S_vs_DFER_video_both_recomputed.csv'),
    )

  def test_default_output_requires_dataset_metadata(self):
    data_0 = make_results()
    data_1 = shifted_results()
    del data_0['config']['path_csv_dataset']

    with self.assertRaisesRegex(ValueError, 'path_csv_dataset'):
      comparison.resolve_output_path(
        None, data_0, data_1, 'subject', 'mae', recompute=False
      )

  def test_explicit_output_is_preserved(self):
    output = comparison.resolve_output_path(
      'chosen/report.csv', make_results(), shifted_results(), 'subject', 'mae', False
    )

    self.assertEqual(output, Path('chosen/report.csv'))

  def test_cli_requires_analysis_level_and_defaults_measure_to_both(self):
    with self.assertRaises(SystemExit):
      comparison.parse_args(['--pkl_path_0', 'zero.pkl', '--pkl_path_1', 'one.pkl'])

    args = comparison.parse_args([
      '--pkl_path_0', 'zero.pkl', '--pkl_path_1', 'one.pkl',
      '--analysis_level', 'video',
    ])
    self.assertEqual(args.analysis_level, 'video')
    self.assertEqual(args.measure, 'both')
    self.assertIsNone(args.out)

  def test_video_level_aligns_histories_by_sample_id_and_computes_metrics(self):
    data_0 = make_video_results({2: [0.4], 1: [1.6]})
    data_1 = make_video_results({1: [1.2], 2: [0.6]}, model_type='MODEL_B')
    rows = [{'sample_id': 1, 'class_id': 1}, {'sample_id': 2, 'class_id': 0}]
    with tempfile.TemporaryDirectory() as tmp:
      paths = (Path(tmp, 'zero', 'k_fold_results.pkl'), Path(tmp, 'one', 'k_fold_results.pkl'))
      for path in paths:
        write_test_csv(path, 'k0_cross_val_final', rows)

      paired = comparison.extract_paired_values(
        data_0, data_1, 'video', pkl_paths=paths
      )

    self.assertTrue(np.allclose(paired['mae'][0], [0.6, 0.4]))
    self.assertTrue(np.allclose(paired['mae'][1], [0.2, 0.6]))
    self.assertTrue(np.array_equal(paired['accuracy'][0], [1.0, 1.0]))
    self.assertTrue(np.array_equal(paired['accuracy'][1], [1.0, 0.0]))

  def test_video_level_requires_recompute_when_histories_are_missing(self):
    data_0 = make_video_results(None)
    data_1 = make_video_results(None, model_type='MODEL_B')

    with self.assertRaisesRegex(ValueError, '--recompute'):
      comparison.extract_paired_values(
        data_0, data_1, 'video', pkl_paths=('zero.pkl', 'one.pkl')
      )

  def test_video_level_rejects_different_sample_ids(self):
    data_0 = make_video_results({1: [0.1], 2: [0.2]})
    data_1 = make_video_results({1: [0.1], 3: [0.2]}, model_type='MODEL_B')
    rows_0 = [{'sample_id': 1, 'class_id': 0}, {'sample_id': 2, 'class_id': 0}]
    rows_1 = [{'sample_id': 1, 'class_id': 0}, {'sample_id': 3, 'class_id': 0}]
    with tempfile.TemporaryDirectory() as tmp:
      paths = (Path(tmp, 'zero', 'k_fold_results.pkl'), Path(tmp, 'one', 'k_fold_results.pkl'))
      write_test_csv(paths[0], 'k0_cross_val_final', rows_0)
      write_test_csv(paths[1], 'k0_cross_val_final', rows_1)

      with self.assertRaisesRegex(ValueError, 'video IDs differ'):
        comparison.extract_paired_values(data_0, data_1, 'video', pkl_paths=paths)

  def test_video_level_rejects_different_ground_truth(self):
    data_0 = make_video_results({1: [0.1], 2: [0.2]})
    data_1 = make_video_results({1: [0.1], 2: [0.2]}, model_type='MODEL_B')
    rows_0 = [{'sample_id': 1, 'class_id': 0}, {'sample_id': 2, 'class_id': 0}]
    rows_1 = [{'sample_id': 1, 'class_id': 1}, {'sample_id': 2, 'class_id': 0}]
    with tempfile.TemporaryDirectory() as tmp:
      paths = (Path(tmp, 'zero', 'k_fold_results.pkl'), Path(tmp, 'one', 'k_fold_results.pkl'))
      write_test_csv(paths[0], 'k0_cross_val_final', rows_0)
      write_test_csv(paths[1], 'k0_cross_val_final', rows_1)

      with self.assertRaisesRegex(ValueError, 'ground-truth labels differ'):
        comparison.extract_paired_values(data_0, data_1, 'video', pkl_paths=paths)

  def test_recomputed_sample_arrays_feed_video_analysis_without_csv(self):
    stored_0 = make_video_results(None)
    stored_1 = make_video_results(None, model_type='MODEL_B')
    fresh_0 = {
      'k0_cross_val_final': {
        'recomputed_l1': 0.5,
        'recomputed_accuracy': 0.5,
        'recomputed_loss_per_subject': np.array([0.5]),
        'recomputed_accuracy_per_subject': np.array([0.5]),
        'recomputed_subject_ids': np.array([10]),
        'recomputed_sample_ids': np.array([1, 2]),
        'recomputed_sample_predictions': np.array([1.6, 0.4]),
        'recomputed_sample_labels': np.array([1, 0]),
      },
    }
    fresh_1 = {
      'k0_cross_val_final': {
        **fresh_0['k0_cross_val_final'],
        'recomputed_sample_predictions': np.array([1.2, 0.6]),
      },
    }

    recomputed_0 = comparison.with_recomputed_metrics(stored_0, fresh_0)
    recomputed_1 = comparison.with_recomputed_metrics(stored_1, fresh_1)
    paired = comparison.extract_paired_values(
      recomputed_0, recomputed_1, 'video', pkl_paths=('missing0.pkl', 'missing1.pkl')
    )

    self.assertTrue(np.allclose(paired['mae'][0], [0.6, 0.4]))
    self.assertTrue(np.allclose(paired['mae'][1], [0.2, 0.6]))

  def test_recomputed_values_replace_stored_values_and_create_detailed_audit(self):
    stored = make_results()
    fresh = fresh_metrics(stored, mae_delta=0.01)

    audit = comparison.build_sanity_rows(0, stored, fresh)
    replaced = comparison.with_recomputed_metrics(stored, fresh)

    self.assertEqual(len(audit), 12)
    self.assertEqual(sum(not row['matches'] for row in audit), 6)
    self.assertTrue(any(row['subject_id'] == 10 and row['metric'] == 'mae' for row in audit))
    self.assertAlmostEqual(
      replaced['results']['k0_cross_val_final']['test']['test_l1_error'],
      1.51,
    )

  def test_run_comparison_uses_fresh_metrics_and_writes_sanity_csv(self):
    data_0 = make_results()
    data_1 = shifted_results()
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp, 'result.csv')
      with mock.patch.object(comparison, 'load_pickle', side_effect=[data_0, data_1]), \
           mock.patch.object(
             comparison,
             'recompute_fold_metrics',
             side_effect=[fresh_metrics(data_0), fresh_metrics(data_1)],
           ):
        rows, sanity_rows = comparison.run_comparison(
          'zero.pkl', 'one.pkl', output, analysis_level='subject', measure='both', recompute=True
        )

      self.assertTrue(output.is_file())
      self.assertTrue(Path(tmp, 'result_sanity.csv').is_file())
      self.assertTrue(all(row['metric_source'] == 'recomputed' for row in rows))
      self.assertEqual(len(sanity_rows), 24)


if __name__ == '__main__':
  unittest.main()

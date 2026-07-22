import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import compare_kfold_results as comparison


def make_results(subject_order=(0, 1)):
  """Build two aligned synthetic k-fold result dictionaries."""
  ids = np.array([10, 11])[list(subject_order)]
  return {
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


def shifted_results():
  """Build a second result dictionary where model zero is uniformly better."""
  data = make_results(subject_order=(1, 0))
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


class TestKfoldComparison(unittest.TestCase):
  def test_aligns_subjects_and_reports_positive_effect_for_better_model_zero(self):
    paired = comparison.extract_paired_values(make_results(), shifted_results())
    rows = comparison.compare_paired_values(paired, metric_source='stored')
    by_key = {(row['analysis_level'], row['metric']): row for row in rows}

    self.assertTrue(np.allclose(paired['subject']['mae'][1] - paired['subject']['mae'][0], 0.2))
    self.assertAlmostEqual(by_key[('subject', 'mae')]['mean_effect_pkl0_better'], 0.2)
    self.assertAlmostEqual(by_key[('subject', 'accuracy')]['mean_effect_pkl0_better'], 0.1)
    self.assertEqual(by_key[('subject', 'mae')]['metric_source'], 'stored')
    self.assertIsNotNone(by_key[('subject', 'mae')]['permutation_p_holm'])
    self.assertIsNone(by_key[('fold', 'mae')]['permutation_p_holm'])

  def test_rejects_mismatched_subjects(self):
    data_1 = shifted_results()
    data_1['results']['k0_cross_val_final']['test']['test_unique_subject_ids'][0] = 99

    with self.assertRaisesRegex(ValueError, 'subject IDs differ'):
      comparison.extract_paired_values(make_results(), data_1)

  def test_identical_values_return_unit_p_values(self):
    data = make_results()
    rows = comparison.compare_paired_values(
      comparison.extract_paired_values(data, make_results()),
      metric_source='stored',
    )

    for row in rows:
      self.assertEqual(row['permutation_p_raw'], 1.0)
      self.assertEqual(row['wilcoxon_p_raw'], 1.0)

  def test_writes_four_result_rows(self):
    rows = comparison.compare_paired_values(
      comparison.extract_paired_values(make_results(), shifted_results()),
      metric_source='stored',
    )
    with tempfile.TemporaryDirectory() as tmp:
      output = Path(tmp, 'comparison.csv')
      comparison.write_rows(output, rows)
      with output.open(newline='') as handle:
        written = list(csv.DictReader(handle))

    self.assertEqual(len(written), 4)
    self.assertEqual({row['analysis_level'] for row in written}, {'subject', 'fold'})

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
        rows, sanity_rows = comparison.run_comparison('zero.pkl', 'one.pkl', output, recompute=True)

      self.assertTrue(output.is_file())
      self.assertTrue(Path(tmp, 'result_sanity.csv').is_file())
      self.assertTrue(all(row['metric_source'] == 'recomputed' for row in rows))
      self.assertEqual(len(sanity_rows), 24)


if __name__ == '__main__':
  unittest.main()

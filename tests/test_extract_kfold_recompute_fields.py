import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from extract_kfold_test_table import recompute_raw_fold_metrics


class FakeModel:
  """Return deterministic test metrics without loading a real checkpoint."""

  def __init__(self, **kwargs):
    self.kwargs = kwargs

  def test_pretrained_model(self, **kwargs):
    return {
      'history_test_sample_predictions': {1: [0.25], 2: [1.75]},
      'test_l1_error': 0.25,
      'test_accuracy': 0.5,
      'test_loss_per_subject': np.array([0.2, 0.3]),
      'test_accuracy_per_subject': np.array([0.4, 0.6]),
      'test_unique_subject_ids': np.array([10, 11]),
    }


class TestRecomputedMetricContract(unittest.TestCase):
  def test_returns_fold_and_subject_l1_and_accuracy(self):
    fake_helper = types.ModuleType('custom.helper')
    fake_helper.init_log_cross_attention = lambda: None
    fake_helper.init_log_video_embeddings = lambda: None
    fake_helper.LOG_HISTORY_SAMPLE = False
    fake_helper.step_shift = 100
    fake_model_module = types.ModuleType('custom.model')
    fake_model_module.Model_Advanced = FakeModel

    with tempfile.TemporaryDirectory() as tmp:
      run_dir = Path(tmp)
      fold_dir = run_dir / 'train_HEAD' / 'k0_cross_val'
      checkpoint = fold_dir / 'k0_cross_val_sub_0' / 'best_model_ep_2.pt'
      checkpoint.parent.mkdir(parents=True)
      checkpoint.touch()
      pd.DataFrame({'sample_id': [1, 2], 'class_id': [0, 2]}).to_csv(
        fold_dir / 'test_cleaned.csv', sep='\t', index=False
      )
      data = {
        'model_advanced_params': {'head': 'HEAD', 'head_params': {}},
        'config': {
          'criterion': object(),
          'concatenate_temp_dim': 0,
          'concatenate_quadrants': 0,
          'CCC_loss': None,
        },
        'results': {
          'k0_cross_val_final': {
            'best_model': {'fold_sub_fold_idx': (0, 0), 'best_model_idx': 2},
          },
        },
      }
      with mock.patch.dict(sys.modules, {
        'custom.helper': fake_helper,
        'custom.model': fake_model_module,
      }):
        result = recompute_raw_fold_metrics(
          str(run_dir / 'k_fold_results.pkl'), data, ['k0_cross_val_final']
        )['k0_cross_val_final']

    self.assertEqual(result['recomputed_l1'], 0.25)
    self.assertEqual(result['recomputed_accuracy'], 0.5)
    self.assertTrue(np.array_equal(result['recomputed_loss_per_subject'], [0.2, 0.3]))
    self.assertTrue(np.array_equal(result['recomputed_accuracy_per_subject'], [0.4, 0.6]))
    self.assertTrue(np.array_equal(result['recomputed_subject_ids'], [10, 11]))


if __name__ == '__main__':
  unittest.main()

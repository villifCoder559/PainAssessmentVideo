import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cross_space_fake_projection import (
  _projector_from_state,
  _random_projector_from_state,
  ReplayError,
  discover_results,
  generate_fake_embeddings,
  group_results,
  main as replay_main,
  replay_result,
  run,
)

# cross_space_projection imports custom.helper, whose profiling Manager opens a
# local socket at import time. These unit tests do not exercise that profiler.
with mock.patch('multiprocessing.Manager') as manager:
  manager.return_value.dict.return_value = {}
  from cross_space_projection import (
    _aggregate_model_combo_pkls,
    _evaluate_projection_inputs,
    _fake_embeddings,
    _fake_projection_suffix,
    _matched_gaussian_embeddings,
    _prediction_frame,
    _run_trial,
    _validate_fake_projection,
    _yaml_to_argv,
    LINEAR_PROJECTOR_CONFIG,
    REFINEMENT_CONFIG,
  )

from cross_space_logs import (
  _fake_replay_labels,
  generate_logs,
  plot_fake_vs_real_dashboard,
)


class FakeEmbeddingTest(unittest.TestCase):
  def test_matched_gaussian_remains_the_default(self):
    real = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)

    np.testing.assert_array_equal(
      _fake_embeddings(real, seed=7),
      _matched_gaussian_embeddings(real, seed=7),
    )

  def test_matched_gaussian_is_deterministic_without_mutating_input(self):
    real = np.array([
      [1.0, 5.0, -2.0],
      [3.0, 5.0,  0.0],
      [5.0, 5.0,  2.0],
    ], dtype=np.float32)
    original = real.copy()

    fake = _matched_gaussian_embeddings(real, seed=7)

    expected = np.random.default_rng(7).normal(
      loc=real.mean(axis=0), scale=real.std(axis=0), size=real.shape,
    ).astype(np.float32)
    np.testing.assert_array_equal(fake, expected)
    np.testing.assert_array_equal(real, original)
    self.assertEqual(fake.dtype, np.float32)
    self.assertEqual(fake.shape, real.shape)
    np.testing.assert_array_equal(fake[:, 1], np.full(3, 5.0, dtype=np.float32))

  def test_prediction_frame_preserves_sample_and_label_order(self):
    frame = _prediction_frame(
      np.array([20, 10]),
      np.array([4.0, 1.0], dtype=np.float32),
      np.array([[3.5], [2.0]], dtype=np.float32),
    )

    self.assertEqual(frame.columns.tolist(), ['sample_id', 'label', 'prediction'])
    self.assertEqual(frame.to_dict('list'), {
      'sample_id': [20, 10], 'label': [4.0, 1.0], 'prediction': [3.5, 2.0],
    })

  def test_standard_normal_matches_seeded_numpy_draw(self):
    real = np.zeros((3, 4), dtype=np.float64)

    fake = _fake_embeddings(real, seed=19, distribution='standard_normal')

    expected = np.random.default_rng(19).standard_normal(real.shape).astype(np.float32)
    np.testing.assert_array_equal(fake, expected)
    self.assertEqual(fake.dtype, np.float32)
    self.assertEqual(fake.shape, real.shape)

  def test_fake_generation_does_not_change_global_numpy_rng_state(self):
    real = np.zeros((2, 3), dtype=np.float32)
    np.random.seed(123)
    before = np.random.get_state()

    _fake_embeddings(real, seed=7, distribution='matched_gaussian')
    _fake_embeddings(real, seed=7, distribution='standard_normal')

    after = np.random.get_state()
    self.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    self.assertEqual(before[2:], after[2:])

  def test_paired_evaluation_uses_fake_as_headline_and_real_as_baseline(self):
    import torch

    real = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.array([0.0, 1.0], dtype=np.float32)
    anchors = {
      'embeddings': np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    }
    new_anchors = {
      'embeddings': np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    }
    linear = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
      linear.weight[:] = torch.tensor([[0.5, 0.5]])

    headline, baseline = _evaluate_projection_inputs(
      real_embeddings=real,
      labels=labels,
      classify_linear=linear,
      label_denorm=1.0,
      interpolation_similarity='l2',
      rbf_sigma=0.2,
      old_model_anchors=anchors,
      new_model_anchors_aligned=new_anchors,
      fake_projection=True,
      seed=11,
    )

    np.testing.assert_allclose(baseline['predictions'].reshape(-1), labels, atol=1e-5)
    np.testing.assert_array_equal(
      headline['source_embeddings'], _matched_gaussian_embeddings(real, 11))
    self.assertEqual(headline['embeddings'].shape, real.shape)
    self.assertEqual(headline['metrics']['mae'],
                     float(np.mean(np.abs(headline['predictions'].reshape(-1) - labels))))
    np.testing.assert_array_equal(real, [[0.0, 0.0], [1.0, 1.0]])

  def test_paired_evaluation_reuses_learned_projector(self):
    import torch

    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    projector = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
      projector.weight[:] = torch.eye(2)
      classifier.weight[:] = torch.tensor([[1.0, 0.0]])

    headline, baseline = _evaluate_projection_inputs(
      real_embeddings=real,
      labels=np.array([1.0, 3.0], dtype=np.float32),
      classify_linear=classifier,
      label_denorm=1.0,
      interpolation_similarity='linear',
      rbf_sigma=1.0,
      old_model_anchors=None,
      new_model_anchors_aligned=None,
      projector=projector,
      fake_projection=True,
      seed=3,
    )

    np.testing.assert_allclose(baseline['embeddings'], real)
    np.testing.assert_allclose(baseline['predictions'].reshape(-1), [1.0, 3.0])
    np.testing.assert_allclose(headline['embeddings'],
                               _matched_gaussian_embeddings(real, 3))

  def test_paired_evaluation_uses_selected_distribution(self):
    import torch

    real = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    projector = torch.nn.Linear(2, 2, bias=False)
    classifier = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
      projector.weight[:] = torch.eye(2)
      classifier.weight[:] = torch.tensor([[1.0, 0.0]])

    headline, _ = _evaluate_projection_inputs(
      real_embeddings=real,
      labels=np.array([1.0, 3.0], dtype=np.float32),
      classify_linear=classifier,
      label_denorm=1.0,
      interpolation_similarity='linear',
      rbf_sigma=1.0,
      old_model_anchors=None,
      new_model_anchors_aligned=None,
      projector=projector,
      fake_projection=True,
      fake_projection_distribution='standard_normal',
      seed=3,
    )

    np.testing.assert_allclose(
      headline['source_embeddings'],
      np.random.default_rng(3).standard_normal(real.shape).astype(np.float32),
    )


class FakeProjectionConfigTest(unittest.TestCase):
  def test_yaml_true_emits_flag_and_false_does_not(self):
    self.assertEqual(_yaml_to_argv({'fake_projection': True}), ['--fake_projection'])
    self.assertEqual(_yaml_to_argv({'fake_projection': False}), [])
    with self.assertRaisesRegex(ValueError, 'must be a boolean'):
      _yaml_to_argv({'fake_projection': 'true'})

  def test_fake_projection_requires_positive_anchors(self):
    _validate_fake_projection(False, [0, -1])
    _validate_fake_projection(True, [1, 10])
    with self.assertRaisesRegex(ValueError, 'requires every num_anchors value to be > 0'):
      _validate_fake_projection(True, [10, 0])

  def test_yaml_forwards_distribution_and_rejects_invalid_value(self):
    self.assertEqual(
      _yaml_to_argv({'fake_projection_distribution': 'standard_normal'}),
      ['--fake_projection_distribution', 'standard_normal'],
    )
    with self.assertRaisesRegex(ValueError, 'fake_projection_distribution'):
      _yaml_to_argv({'fake_projection_distribution': 'uniform'})

  def test_cli_rejects_invalid_distribution(self):
    result = subprocess.run(
      [
        sys.executable, os.path.join(os.path.dirname(__file__), '..', 'cross_space_projection.py'),
        '--new_model_pth', 'new.pth', '--old_model_pth', 'old.pth',
        '--num_anchors', '1', '--csv_anchor_selection', 'train',
        '--old_model_csv', 'test', '--fake_projection_distribution', 'uniform',
      ],
      capture_output=True,
      text=True,
    )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("invalid choice: 'uniform'", result.stderr)

  def test_fake_projection_suffix_distinguishes_standard_normal_only(self):
    self.assertEqual(_fake_projection_suffix(False, 'standard_normal'), '')
    self.assertEqual(_fake_projection_suffix(True, 'matched_gaussian'), '_fake')
    self.assertEqual(
      _fake_projection_suffix(True, 'standard_normal'), '_fake_standard_normal')


class FakeProjectionAggregationTest(unittest.TestCase):
  def test_aggregation_without_flag_keeps_legacy_schema(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = os.path.join(tmp, 'subtrial.pkl')
      labels = np.array([1.0, 2.0], dtype=np.float32)
      sample_ids = np.array([10, 11], dtype=np.int64)
      data = {
        'config_cross_space_projection': {},
        'metrics': {'mae': 0.0, 'ccc': 1.0},
        'old_model_tensors': {
          'predictions': labels, 'labels': labels, 'sample_ids': sample_ids,
        },
        'new_model_tensors': {
          'predictions': labels, 'labels': labels, 'sample_ids': sample_ids,
        },
      }
      with open(path, 'wb') as f:
        pickle.dump(data, f)
      records = [{
        'new_idx': 0, 'old_idx': 0, 'new_model_pth': 'new',
        'old_model_pth': 'old', 'pkl_path': path,
      }]

      out_dir = os.path.join(tmp, 'aggregate')
      out_pkl = _aggregate_model_combo_pkls(records, out_dir, argparse.Namespace())

      with open(out_pkl, 'rb') as f:
        result = pickle.load(f)
      self.assertNotIn('real_projection', result)
      self.assertNotIn('fake_projection', result['config_cross_space_projection'])
      self.assertFalse(os.path.exists(os.path.join(out_dir, 'predictions_real.csv')))

  def test_aggregation_pools_real_and_fake_predictions_separately(self):
    with tempfile.TemporaryDirectory() as tmp:
      records = []
      for i, (fake, real) in enumerate((([9.0, 8.0], [1.0, 2.0]),
                                        ([7.0, 6.0], [3.0, 4.0]))):
        path = os.path.join(tmp, f'subtrial_{i}.pkl')
        labels = np.array([1.0, 2.0], dtype=np.float32)
        sample_ids = np.array([10 + 2 * i, 11 + 2 * i], dtype=np.int64)
        data = {
          'config_cross_space_projection': {
            'fake_projection': True,
            'num_anchors': 2,
            'anchor_selection_type': 'random',
            'csv_anchor_selection': 'train',
            'old_model_csv': 'test',
            'interpolation_similarity': 'cos',
            'mlp_activation': 'gelu',
            'mlp_num_layers': 1,
            'weighting_method': 'rbf',
            'rbf_sigma': 1.0,
          },
          'metrics': {'mae': 1.0, 'ccc': 0.0},
          'old_model_tensors': {
            'predictions': labels, 'labels': labels, 'sample_ids': sample_ids,
          },
          'new_model_tensors': {
            'predictions': np.asarray(fake, dtype=np.float32),
            'labels': labels,
            'sample_ids': sample_ids,
          },
          'real_projection': {
            'predictions': np.asarray(real, dtype=np.float32),
            'metrics': {'mae': 0.0, 'ccc': 1.0},
          },
        }
        with open(path, 'wb') as f:
          pickle.dump(data, f)
        records.append({
          'new_idx': i, 'old_idx': i,
          'new_model_pth': f'new{i}', 'old_model_pth': f'old{i}', 'pkl_path': path,
        })

      out_dir = os.path.join(tmp, 'aggregate')
      out_pkl = _aggregate_model_combo_pkls(records, out_dir, argparse.Namespace())

      with open(out_pkl, 'rb') as f:
        result = pickle.load(f)
      np.testing.assert_array_equal(
        result['new_model_tensors']['predictions'], [9.0, 8.0, 7.0, 6.0])
      np.testing.assert_array_equal(
        result['real_projection']['predictions'], [1.0, 2.0, 3.0, 4.0])
      for name, predictions in (
        ('predictions_fake.csv', [9.0, 8.0, 7.0, 6.0]),
        ('predictions_real.csv', [1.0, 2.0, 3.0, 4.0]),
      ):
        frame = np.genfromtxt(os.path.join(out_dir, name), delimiter=',', names=True)
        np.testing.assert_array_equal(frame['prediction'], predictions)
        np.testing.assert_array_equal(frame['label'], [1.0, 2.0, 1.0, 2.0])

  def test_aggregation_pools_paired_modes_into_deterministic_fake_pkl(self):
    with tempfile.TemporaryDirectory() as tmp:
      records = []
      for index in range(2):
        path = Path(tmp) / f'subtrial_{index}.pkl'
        labels = np.array([0., 1.], dtype=np.float32)
        sample_ids = np.array([2 * index, 2 * index + 1])
        real = labels.copy()
        fake = labels + index + 1
        data = {
          'config_cross_space_projection': {
            'fake_projection': True,
            'fake_projection_distribution': 'matched_gaussian',
          },
          'fake_projection_metadata': {'distribution': 'matched_gaussian', 'seed': 42},
          'fake_projection_evaluations': {'linear_only': {
            'sample_ids': sample_ids, 'labels': labels,
            'real_predictions': real, 'fake_predictions': fake,
            'real_metrics': {}, 'fake_metrics': {},
          }},
          'metrics': {'mae': float(index + 1), 'ccc': 0.},
          'real_projection': {'predictions': real, 'metrics': {}},
          'old_model_tensors': {
            'predictions': labels, 'labels': labels, 'sample_ids': sample_ids,
          },
          'new_model_tensors': {
            'predictions': fake, 'labels': labels, 'sample_ids': sample_ids,
          },
        }
        with path.open('wb') as stream:
          pickle.dump(data, stream)
        records.append({
          'new_idx': index, 'old_idx': index, 'new_model_pth': f'new{index}',
          'old_model_pth': f'old{index}', 'pkl_path': str(path),
        })

      output = _aggregate_model_combo_pkls(
        records, str(Path(tmp) / 'aggregate'), argparse.Namespace(),
        output_filename='results_fake.pkl')

      self.assertEqual(Path(output).name, 'results_fake.pkl')
      with open(output, 'rb') as stream:
        aggregate = pickle.load(stream)
      evaluation = aggregate['fake_projection_evaluations']['linear_only']
      np.testing.assert_array_equal(evaluation['sample_ids'], [0, 1, 2, 3])
      np.testing.assert_array_equal(evaluation['real_predictions'], [0., 1., 0., 1.])
      np.testing.assert_array_equal(evaluation['fake_predictions'], [1., 2., 2., 3.])
      self.assertEqual(evaluation['real_metrics']['mae_micro'], 0.)
      self.assertEqual(evaluation['fake_metrics']['mae_micro'], 1.5)


class FakeProjectionTrialTest(unittest.TestCase):
  def test_trial_keeps_real_cache_and_writes_paired_outputs(self):
    import torch

    real = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    old_tensors = {
      'embeddings': real.copy(),
      'labels': np.array([0.0, 1.0], dtype=np.float32),
      'sample_ids': np.array([10, 11], dtype=np.int64),
      'predictions': np.array([0.0, 1.0], dtype=np.float32),
    }
    anchor_key = ('train', 2, 'random')
    anchors = {'embeddings': real.copy()}
    anchor_cache = {
      anchor_key: {
        'old': anchors,
        'new': {'embeddings': real.copy()},
        'projectors': {},
        'refine_distance': {},
      },
    }
    tensor_cache = {'test': {'old_tensors': old_tensors, 'old_tensors_csv': 'unused.csv'}}
    linear = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
      linear.weight[:] = torch.tensor([[0.5, 0.5]])
    model = SimpleNamespace(head=SimpleNamespace(linear=linear))
    params = {
      'num_anchors': 2,
      'anchor_selection_type': 'random',
      'csv_anchor_selection': 'train',
      'old_model_csv': 'test',
      'interpolation_similarity': 'l2',
      'mlp_activation': 'gelu',
      'mlp_num_layers': 1,
      'weighting_method': 'rbf',
      'rbf_sigma': 0.2,
      'projector_config': 'projector',
      'refinement_config': 'refinement',
      'refine_mode': 'none',
    }

    with tempfile.TemporaryDirectory() as trial_dir:
      mae = _run_trial(
        params, 0, anchor_cache, tensor_cache, model,
        {'config': {'normalize_labels': 0}}, trial_dir, 123,
        {'projector': dict(LINEAR_PROJECTOR_CONFIG)},
        {'refinement': dict(REFINEMENT_CONFIG)},
        fake_projection=True,
        fake_projection_distribution='standard_normal',
      )

      with open(os.path.join(trial_dir, 'results.pkl'), 'rb') as f:
        result = pickle.load(f)
      self.assertEqual(mae, result['metrics']['mae'])
      self.assertEqual(result['fake_projection_distribution'], 'standard_normal')
      self.assertIn('real_projection', result)
      np.testing.assert_allclose(result['real_projection']['predictions'].reshape(-1),
                                 [0.0, 1.0], atol=1e-5)
      np.testing.assert_array_equal(old_tensors['embeddings'], real)
      self.assertTrue(os.path.isfile(os.path.join(trial_dir, 'predictions_fake.csv')))
      self.assertTrue(os.path.isfile(os.path.join(trial_dir, 'predictions_real.csv')))


def _mae_pair(predictions, labels):
  predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
  labels = np.asarray(labels, dtype=np.float32).reshape(-1)
  per_class = [
    np.abs(predictions[np.round(labels).astype(int) == cls]
           - labels[np.round(labels).astype(int) == cls]).mean()
    for cls in np.unique(np.round(labels).astype(int))
  ]
  return float(np.abs(predictions - labels).mean()), float(np.mean(per_class))


def _ccc(labels, predictions):
  labels = np.asarray(labels, dtype=np.float64).reshape(-1)
  predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
  denominator = (labels.var() + predictions.var()
                 + (labels.mean() - predictions.mean()) ** 2)
  if denominator == 0:
    return 1.0 if np.allclose(labels, predictions) else float('nan')
  return float(2 * np.mean((labels - labels.mean())
                           * (predictions - predictions.mean())) / denominator)


def _write_replay_fixture(folder, modes=('linear_only',), *, distance=False,
                          grid=False, broken=False):
  folder = Path(folder)
  folder.mkdir(parents=True, exist_ok=True)
  embeddings = np.array([[0., 0.], [1., 1.], [2., 0.], [3., 1.]], dtype=np.float32)
  sample_ids = np.arange(10, 14, dtype=np.int64)

  projector = torch.nn.Linear(2, 2)
  with torch.no_grad():
    projector.weight.copy_(torch.eye(2))
    projector.bias.zero_()
  projector_pth = folder / 'projector.pt'
  torch.save(projector.state_dict(), projector_pth)

  heads = {}
  for name, weight in (
      ('before', [1., 0.]),
      ('linear_only', [.5, .5]),
      ('projector_linear', [0., 1.])):
    head = torch.nn.Linear(2, 1)
    with torch.no_grad():
      head.weight.copy_(torch.tensor([weight]))
      head.bias.zero_()
    pth = folder / f'head_{name}.pt'
    torch.save(head.state_dict(), pth)
    heads[name] = (head, pth)

  if distance:
    projected = np.ones_like(embeddings)
  else:
    projected = embeddings.copy()
  with torch.no_grad():
    before_predictions = heads['before'][0](torch.from_numpy(projected)).numpy().reshape(-1)

  labels = np.array([0., 1., 1., 2.], dtype=np.float32)
  refinements = {}
  for mode in modes:
    with torch.no_grad():
      after_predictions = heads[mode][0](torch.from_numpy(projected)).numpy().reshape(-1)
    before_micro, before_macro = _mae_pair(before_predictions, labels)
    after_micro, after_macro = _mae_pair(after_predictions, labels)
    refinements[mode] = {
      'refine_mode': mode,
      'mae_micro_old_oncsv_before': before_micro,
      'mae_macro_old_oncsv_before': before_macro,
      'mae_micro_old_oncsv_after': after_micro,
      'mae_macro_old_oncsv_after': after_macro,
      'projector_before_pth': (str(projector_pth) if mode == 'projector_linear' else None),
      'projector_after_pth': (str(projector_pth) if mode == 'projector_linear' else None),
      'linear_before_pth': str(heads['before'][1]),
      'linear_after_pth': str(folder / 'missing.pt') if broken else str(heads[mode][1]),
      'config': {'mode': mode},
      'new_test_eval': {
        'split': 'test',
        'labels': labels.copy(),
        'preds_before': before_predictions.copy(),
        'preds_after': after_predictions.copy(),
      },
    }

  headline_mode = modes[0] if len(modes) == 1 else 'before'
  headline_head = heads[headline_mode][0]
  with torch.no_grad():
    headline_logits = headline_head(torch.from_numpy(projected)).numpy()
  headline_predictions = headline_logits.reshape(-1)
  headline_micro, _ = _mae_pair(headline_predictions, labels)
  data = {
    'seed': 9,
    'config_cross_space_projection': {
      'uid': 7,
      'out_dir': str(folder),
      'old_tensors_csv_path': str(folder / 'test.csv'),
      'num_anchors': 1 if distance else 4,
      'anchor_selection_type': 'random',
      'csv_anchor_selection': 'train',
      'old_model_csv': 'test',
      'interpolation_similarity': 'l2' if distance else 'linear',
      'weighting_method': 'rbf' if distance else 'none',
      'rbf_sigma': .2,
    },
    'old_model_tensors': {
      'embeddings': embeddings,
      'predictions': labels.copy(),
      'labels': labels,
      'sample_ids': sample_ids,
    },
    'new_model_tensors': {
      'embeddings': projected,
      'logits': headline_logits.astype(np.float32),
      'predictions': headline_predictions.astype(np.float32),
      'labels': labels,
      'sample_ids': sample_ids,
      'weights': np.zeros((len(labels), 0), dtype=np.float32),
    },
    'metrics': {'mae': headline_micro, 'ccc': _ccc(labels, headline_predictions)},
  }
  if distance and not grid:
    data['old_model_anchors_embeddings'] = {
      'embeddings': np.array([[0., 0.]], dtype=np.float32),
      'sample_ids': np.array([1]),
    }
    data['new_model_anchors_embeddings'] = {
      'embeddings': np.array([[1., 1.]], dtype=np.float32),
      'sample_ids': np.array([1]),
    }
  if not distance:
    data['linear_projector'] = {
      'kind': 'linear', 'config': {}, 'norm_stats': None,
      'ckpt_path': str(projector_pth),
    }
  if len(refinements) == 1:
    data['refinement'] = refinements[modes[0]]
  else:
    data['refinements'] = refinements
  if grid:
    data['trial_number'] = 0
    data['trial_params'] = dict(data.pop('config_cross_space_projection'))

  csv_path = folder / 'test.csv'
  pd.DataFrame({'sample_id': sample_ids, 'subject_id': sample_ids,
                'class_id': labels.astype(int)}).to_csv(csv_path, sep='\t', index=False)
  result_path = folder / ('results.pkl' if grid else 'results_7.pkl')
  with open(result_path, 'wb') as stream:
    pickle.dump(data, stream)
  return result_path


class RetrospectiveFakeProjectionTest(unittest.TestCase):
  def test_cli_forwards_fast_mode_for_both_controls(self):
    for control in ('fake_embeddings', 'fake_adapter'):
      with self.subTest(control=control), mock.patch(
          'cross_space_fake_projection.run', return_value=0) as called:
        self.assertEqual(replay_main([
          'experiment', '--control', control, '--fast-mode']), 0)
        called.assert_called_once_with(
          'experiment', None, 42, control=control, adapter_seeds=None, fast_mode=True)

  def test_fast_mode_filters_metadata_before_replay_and_aggregation(self):
    modes = ('linear_only', 'projector_linear')
    for control, distribution in (
        ('fake_embeddings', 'standard_normal'),
        ('fake_embeddings', 'matched_gaussian'),
        ('fake_adapter', None)):
      with self.subTest(control=control, distribution=distribution), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index, family in enumerate((
            'linear', 'mlp', 'autoencoder', 'linear_close', 'procrustes',
            'l2', 'cosine', 'l1', 'unknown')):
          source = _write_replay_fixture(root / f'trial{index:04d}', modes=modes)
          data = pickle.loads(source.read_bytes())
          cfg = data['config_cross_space_projection']
          cfg['interpolation_similarity'] = family
          if family == 'mlp':
            data['trial_params'] = dict(cfg)
            cfg['interpolation_similarity'] = 'l2'
          if family == 'autoencoder':
            del cfg['interpolation_similarity']
            data['linear_projector']['kind'] = family
          source.write_bytes(pickle.dumps(data))
        originals = {path: path.read_bytes() for path in discover_results(root)}
        with mock.patch('builtins.print') as printed:
          self.assertEqual(run(root, distribution=distribution, control=control,
                               adapter_seeds=(7, 8), fast_mode=True), 0)
        self.assertTrue(any('3 of 9' in str(call) for call in printed.call_args_list))
        output_root = root / ('fake_adapter_random_init_fast' if control == 'fake_adapter'
                              else f'fake_projection_{distribution}_fast')
        replay_root = output_root / 'seed_7' if control == 'fake_adapter' else output_root
        leaves = sorted(path for path in replay_root.rglob('results_7.pkl'))
        self.assertEqual([path.parent.name for path in leaves],
                         ['trial0000', 'trial0001', 'trial0002'])
        aggregate = pickle.loads((replay_root / 'aggregated_fake' / 'results_fake.pkl').read_bytes())
        self.assertEqual(len(aggregate['subtrial_pkls']), 3)
        self.assertEqual(set(aggregate['fake_projection_evaluations']), set(modes))
        summary_name = ('aggregated_summary_fake_adapter_fast.csv' if control == 'fake_adapter'
                        else f'aggregated_summary_fake_{distribution}_fast.csv')
        summary = pd.read_csv(root / summary_name)
        self.assertEqual(set(summary['refinement_mode']), set(modes))
        self.assertEqual(set(summary['status']), {'success'})
        self.assertEqual(set(summary['success_count']), {3})
        self.assertFalse((root / 'aggregated_summary_fake.csv').exists())
        self.assertFalse((root / 'aggregated_summary_fake_adapter.csv').exists())
        if control == 'fake_adapter':
          seeds = pd.read_csv(root / 'fake_adapter_seed_results_fast.csv')
          self.assertEqual(set(seeds['fake_projection_seed']), {7, 8})
          self.assertEqual(len(seeds), 12)
          self.assertTrue((root / 'fake_adapter_seed_variability_fast.png').is_file())
          self.assertFalse((root / 'fake_adapter_seed_results.csv').exists())
        self.assertEqual(discover_results(root), list(originals))
        for path, original in originals.items():
          self.assertEqual(path.read_bytes(), original)

  def test_fast_mode_reports_no_eligible_and_unreadable_artifacts(self):
    for control in ('fake_embeddings', 'fake_adapter'):
      with self.subTest(control=control), tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = _write_replay_fixture(root, distance=True)
        summary_path = root / ('aggregated_summary_fake_adapter_fast.csv'
          if control == 'fake_adapter' else 'aggregated_summary_fake_matched_gaussian_fast.csv')
        self.assertEqual(run(root, control=control, fast_mode=True), 1)
        summary = pd.read_csv(summary_path)
        self.assertEqual(summary['summary_row'].tolist(), ['ERROR'])
        self.assertIn('No eligible', summary.iloc[0]['replay_error'])
        source.write_bytes(b'not a pickle')
        self.assertEqual(run(root, control=control, fast_mode=True), 1)
        summary = pd.read_csv(summary_path)
        self.assertEqual(summary['failure_count'].tolist(), [1])
        self.assertIn(str(source), summary.iloc[0]['source_pkl_path'])
        self.assertNotIn('No eligible', summary.iloc[0]['replay_error'])
        if control == 'fake_adapter':
          seeds = pd.read_csv(root / 'fake_adapter_seed_results_fast.csv')
          self.assertEqual(seeds['fake_projection_seed'].tolist(), [42, 43, 44, 45, 46])

  def test_fake_adapter_reporting_labels_do_not_call_real_embeddings_fake(self):
    self.assertEqual(
      _fake_replay_labels({'control': 'fake_adapter'}),
      ('Trained adapter', 'Random adapter', 'Random adapter control'),
    )
    self.assertEqual(
      _fake_replay_labels({'distribution': 'standard_normal'}),
      ('Real embeddings', 'Fake embeddings', 'Fake embedding control'),
    )

  def test_random_projector_is_seeded_fresh_and_preserves_global_rng(self):
    with tempfile.TemporaryDirectory() as tmp:
      learned = torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
      path = Path(tmp) / 'projector.pt'
      torch.save(learned.state_dict(), path)
      before = torch.random.get_rng_state().clone()

      first = _random_projector_from_state(path, activation='relu', seed=42)
      second = _random_projector_from_state(path, activation='relu', seed=42)
      third = _random_projector_from_state(path, activation='relu', seed=43)

      self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
      for left, right in zip(first.parameters(), second.parameters()):
        self.assertTrue(torch.equal(left, right))
      self.assertTrue(any(
        not torch.equal(left, right)
        for left, right in zip(first.parameters(), third.parameters())))
      self.assertTrue(any(
        not torch.equal(random, saved)
        for random, saved in zip(first.parameters(), learned.parameters())))

  def test_random_projector_rejects_checkpoint_without_linear_weights(self):
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'invalid_projector.pt'
      torch.save({'running_mean': torch.zeros(2)}, path)

      with self.assertRaisesRegex(ReplayError, 'no linear weights'):
        _random_projector_from_state(path, seed=42)

  def test_fake_adapter_replay_uses_real_embeddings_and_fresh_projector(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'trial')
      output = Path(tmp) / 'fake_adapter' / source.name

      rows = replay_result(
        source, output, control='fake_adapter', seed=42)

      with output.open('rb') as stream:
        replayed = pickle.load(stream)
      evaluation = replayed['fake_projection_evaluations']['linear_only']
      self.assertEqual(replayed['fake_projection_control'], 'fake_adapter')
      self.assertEqual(
        replayed['fake_projection_metadata']['initialization'], 'pytorch_default')
      self.assertNotIn('fake_source_embeddings', replayed)
      self.assertNotIn('control_source_embeddings', evaluation)
      np.testing.assert_array_equal(evaluation['real_predictions'], [0., 1., 1., 2.])
      self.assertFalse(np.array_equal(
        evaluation['fake_predictions'], evaluation['real_predictions']))
      self.assertEqual(rows[0]['fake_projection_control'], 'fake_adapter')
      self.assertTrue((output.parent /
                       'predictions_random_adapter_before_refinement.csv').is_file())
      self.assertTrue((output.parent /
                       'predictions_random_adapter_after_refinement_linear_only.csv').is_file())
      self.assertFalse((output.parent / 'predictions_fake.csv').exists())

  def test_fake_adapter_rejects_distance_projection_without_learned_weights(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'distance', distance=True)

      with self.assertRaisesRegex(ReplayError, 'no learned adapter weights'):
        replay_result(
          source, Path(tmp) / 'fake_adapter' / source.name,
          control='fake_adapter', seed=42)

  def test_rebuilds_mlp_autoencoder_and_plain_linear_projector_state_dicts(self):
    modules = {
      'mlp': torch.nn.Sequential(
        torch.nn.Linear(2, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2)),
      'autoencoder': torch.nn.Sequential(
        torch.nn.Linear(2, 1), torch.nn.GELU(), torch.nn.Linear(1, 1),
        torch.nn.GELU(), torch.nn.Linear(1, 2)),
      'procrustes': torch.nn.Linear(2, 2),
      'linear_close': torch.nn.Linear(2, 2),
    }
    inputs = torch.tensor([[1., 2.], [3., 4.]])
    with tempfile.TemporaryDirectory() as tmp:
      for name, module in modules.items():
        path = Path(tmp) / f'{name}.pt'
        torch.save(module.state_dict(), path)
        rebuilt = _projector_from_state(
          path, activation='relu' if name == 'mlp' else 'gelu')
        with torch.no_grad():
          np.testing.assert_allclose(
            rebuilt(inputs).numpy(), module(inputs).numpy(), rtol=1e-6, atol=1e-6)

  def test_discovery_groups_cv_inputs_and_excludes_generated_artifacts(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      wanted = []
      for experiment in ('cv_a', 'cv_b'):
        path = root / experiment / 'trial0001_x' / 'results.pkl'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'original')
        wanted.append(path)
      for path in (
          root / 'logs_x' / 'results.pkl',
          root / 'precomputed' / 'results.pkl',
          root / 'aggregated_old' / 'results_1.pkl',
          root / 'fake_projection_matched_gaussian' / 'trial' / 'results.pkl',
          root / 'fake_adapter_random_init' / 'seed_42' / 'trial' / 'results.pkl',
          root / 'cv_a' / 'trial0002_x' / 'results_fake.pkl'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'generated')

      found = discover_results(root)

      self.assertEqual(found, wanted)
      grouped = group_results(root, found)
      self.assertEqual(list(grouped), [root / 'cv_a', root / 'cv_b'])

  def test_cli_accepts_fake_adapter_seed_list_and_rejects_mixed_options(self):
    with mock.patch('cross_space_fake_projection.run', return_value=0) as called:
      self.assertEqual(replay_main([
        'experiment', '--control', 'fake_adapter', '--adapter-seeds', '7', '8',
      ]), 0)
    called.assert_called_once_with(
      'experiment', None, 42, control='fake_adapter', adapter_seeds=(7, 8), fast_mode=False)

    with self.assertRaises(SystemExit):
      replay_main([
        'experiment', '--control', 'fake_adapter',
        '--distribution', 'standard_normal',
      ])
    with self.assertRaises(SystemExit):
      replay_main([
        'experiment', '--control', 'fake_embeddings', '--adapter-seeds', '7',
      ])

  def test_run_averages_adapter_seeds_before_writing_single_trial_summary(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      source = _write_replay_fixture(root)

      self.assertEqual(run(
        root, control='fake_adapter', adapter_seeds=(7, 8)), 0)

      seed_metrics = []
      for seed in (7, 8):
        output = root / 'fake_adapter_random_init' / f'seed_{seed}' / source.name
        self.assertTrue(output.is_file())
        with output.open('rb') as stream:
          data = pickle.load(stream)
        seed_metrics.append(data['fake_projection_evaluations']['linear_only'][
          'fake_mae_micro'])
      summary = pd.read_csv(root / 'aggregated_summary_fake_adapter.csv')
      seed_results = pd.read_csv(root / 'fake_adapter_seed_results.csv')
      self.assertEqual(len(summary), 1)
      self.assertEqual(len(seed_results), 2)
      self.assertEqual(seed_results['fake_projection_seed'].tolist(), [7, 8])
      self.assertIn('random_adapter_mae_micro', seed_results)
      self.assertIn('random_adapter_mae_micro', summary)
      self.assertTrue((root / 'fake_adapter_seed_variability.png').is_file())
      self.assertEqual(summary.loc[0, 'adapter_seed_count'], 2)
      self.assertEqual(summary.loc[0, 'adapter_seeds'], '7;8')
      self.assertAlmostEqual(
        summary.loc[0, 'fake_mae_micro'], float(np.mean(seed_metrics)))
      self.assertAlmostEqual(
        summary.loc[0, 'random_adapter_mae_micro'], float(np.mean(seed_metrics)))

  def test_fake_adapter_cv_aggregate_keeps_control_metadata_and_csv_names(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / 'cv'
      _write_replay_fixture(root / 'trial0001')
      _write_replay_fixture(root / 'trial0002')

      self.assertEqual(run(
        root, control='fake_adapter', adapter_seeds=(7,)), 0)

      aggregate_dir = (root / 'fake_adapter_random_init' / 'seed_7' /
                       'aggregated_fake')
      aggregate = pickle.loads((aggregate_dir / 'results_fake.pkl').read_bytes())
      self.assertEqual(aggregate['fake_projection_control'], 'fake_adapter')
      self.assertEqual(
        aggregate['config_cross_space_projection']['fake_projection_control'],
        'fake_adapter')
      self.assertNotIn('fake_projection_distribution', aggregate)
      self.assertTrue((aggregate_dir / 'predictions_random_adapter.csv').is_file())
      self.assertFalse((aggregate_dir / 'predictions_fake.csv').exists())
      config_log = (aggregate_dir / 'config_logging.txt').read_text()
      self.assertIn('fake_projection_control: fake_adapter', config_log)
      self.assertNotIn('fake_projection_distribution: matched_gaussian', config_log)

  def test_fake_adapter_logs_do_not_replay_trained_adapter_for_embedding_plots(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      source = _write_replay_fixture(root / 'trial')
      output = root / 'fake_adapter' / source.name
      replay_result(source, output, control='fake_adapter', seed=42)

      with mock.patch(
        'cross_space_logs._refined_projected_embeddings', return_value=None,
      ) as refined_embeddings, mock.patch(
        'cross_space_logs._projected_before_refinement_embeddings', return_value=None,
      ) as projected_embeddings, mock.patch(
        'cross_space_logs.plot_projector_diagnostics',
      ) as projector_diagnostics, mock.patch(
        'cross_space_logs.plot_refinement_diagnostics',
      ) as refinement_diagnostics, mock.patch(
        'cross_space_logs.log_embedding_reconstruction',
      ) as embedding_reconstruction:
        generate_logs(str(output), skip_umap=True, out_dir_override=root / 'logs')

      refined_embeddings.assert_not_called()
      projected_embeddings.assert_not_called()
      projector_diagnostics.assert_not_called()
      refinement_diagnostics.assert_not_called()
      self.assertEqual(embedding_reconstruction.call_count, 1)
      self.assertIsNotNone(
        embedding_reconstruction.call_args.kwargs['projected_override'])

  def test_run_reports_progress_for_each_result(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      source = root / 'results.pkl'
      source.write_bytes(b'original')
      row = {
        'refinement_mode': 'linear_only',
        'source_pkl_path': str(source.resolve()),
        'fake_pkl_path': str(root / 'fake.pkl'),
        'replay_error': '',
        'status': 'success',
      }
      with mock.patch('cross_space_fake_projection.tqdm', create=True,
                      side_effect=lambda iterable, **_: iterable) as progress, \
           mock.patch('cross_space_fake_projection.replay_result', return_value=[row]):
        self.assertEqual(run(root), 0)

      progress.assert_called_once_with(
        [source.resolve()], desc='Testing fake projections', unit='test')

  def test_replays_single_mode_without_changing_source(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'trial')
      original = source.read_bytes()
      output = Path(tmp) / 'fake' / source.name

      rows = replay_result(source, output, distribution='standard_normal', seed=42)

      self.assertEqual(source.read_bytes(), original)
      with open(output, 'rb') as stream:
        replayed = pickle.load(stream)
      expected = np.random.default_rng(42).standard_normal((4, 2)).astype(np.float32)
      np.testing.assert_array_equal(replayed['fake_source_embeddings'], expected)
      self.assertEqual(replayed['fake_projection_metadata']['distribution'], 'standard_normal')
      evaluation = replayed['fake_projection_evaluations']['linear_only']
      np.testing.assert_allclose(evaluation['real_predictions'], [0., 1., 1., 2.])
      self.assertEqual(evaluation['real_metrics']['mae_micro'], 0.)
      np.testing.assert_array_equal(
        replayed['new_model_tensors']['predictions'].reshape(-1),
        evaluation['fake_predictions'],
      )
      before_path = output.parent / 'predictions_fake_before_refinement.csv'
      alias_path = output.parent / 'predictions_fake.csv'
      after_path = output.parent / 'predictions_fake_after_refinement_linear_only.csv'
      self.assertTrue(before_path.is_file())
      self.assertTrue(alias_path.is_file())
      self.assertTrue(after_path.is_file())
      before = pd.read_csv(before_path)
      alias = pd.read_csv(alias_path)
      after = pd.read_csv(after_path)
      self.assertEqual(before.columns.tolist(), ['sample_id', 'label', 'prediction'])
      pd.testing.assert_frame_equal(alias, before)
      np.testing.assert_array_equal(before['sample_id'], evaluation['sample_ids'])
      np.testing.assert_array_equal(before['label'], evaluation['labels'])
      np.testing.assert_allclose(before['prediction'], evaluation['fake_before_predictions'])
      np.testing.assert_allclose(after['prediction'], evaluation['fake_predictions'])
      self.assertEqual(rows[0]['status'], 'success')

  def test_reports_real_new_test_head_mae_before_after_and_delta(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'trial')
      output = Path(tmp) / 'fake' / source.name

      with mock.patch('builtins.print') as printed:
        rows = replay_result(source, output)

      with output.open('rb') as stream:
        replayed = pickle.load(stream)
      metrics = replayed['fake_projection_evaluations']['linear_only'][
        'new_test_head_metrics']
      self.assertEqual(metrics, {
        'before': {'mae_micro': .5, 'mae_macro': .5},
        'after': {'mae_micro': 0., 'mae_macro': 0.},
        'delta': {'mae_micro': -.5, 'mae_macro': -.5},
      })
      for key, expected in (
          ('new_test_head_mae_micro_before', .5),
          ('new_test_head_mae_macro_before', .5),
          ('new_test_head_mae_micro_after', 0.),
          ('new_test_head_mae_macro_after', 0.),
          ('new_test_head_mae_micro_delta', -.5),
          ('new_test_head_mae_macro_delta', -.5)):
        self.assertEqual(rows[0][key], expected)
      self.assertTrue(any(
        'new-model real test head MAE' in str(call)
        for call in printed.call_args_list))

  def test_missing_real_new_test_predictions_rejects_mode(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'trial')
      with source.open('rb') as stream:
        data = pickle.load(stream)
      del data['refinement']['new_test_eval']
      with source.open('wb') as stream:
        pickle.dump(data, stream)

      with self.assertRaisesRegex(ReplayError, 'new_test_eval'):
        replay_result(source, Path(tmp) / 'fake' / source.name)

  def test_multi_mode_headline_is_fake_before_refinement(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(
        Path(tmp) / 'trial', modes=('linear_only', 'projector_linear'))
      output = Path(tmp) / 'fake' / source.name

      replay_result(source, output)

      with open(output, 'rb') as stream:
        replayed = pickle.load(stream)
      fake = replayed['fake_source_embeddings']
      np.testing.assert_allclose(
        replayed['new_model_tensors']['predictions'].reshape(-1), fake[:, 0])
      self.assertEqual(set(replayed['fake_projection_evaluations']),
                       {'linear_only', 'projector_linear'})
      for mode, evaluation in replayed['fake_projection_evaluations'].items():
        frame = pd.read_csv(
          output.parent / f'predictions_fake_after_refinement_{mode}.csv')
        np.testing.assert_array_equal(frame['sample_id'], evaluation['sample_ids'])
        np.testing.assert_array_equal(frame['label'], evaluation['labels'])
        np.testing.assert_allclose(frame['prediction'], evaluation['fake_predictions'])

  def test_standalone_distance_uses_anchors_but_grid_without_anchors_fails(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'standalone', distance=True)
      output = Path(tmp) / 'fake' / source.name
      replay_result(source, output)
      with open(output, 'rb') as stream:
        replayed = pickle.load(stream)
      np.testing.assert_allclose(
        replayed['fake_projection_evaluations']['linear_only']['real_predictions'], 1.)

      grid = _write_replay_fixture(Path(tmp) / 'grid', distance=True, grid=True)
      with self.assertRaisesRegex(ReplayError, 'anchors'):
        replay_result(grid, Path(tmp) / 'fake_grid' / grid.name)

  def test_real_replay_metric_mismatch_rejects_output(self):
    with tempfile.TemporaryDirectory() as tmp:
      source = _write_replay_fixture(Path(tmp) / 'trial')
      with source.open('rb') as stream:
        data = pickle.load(stream)
      data['refinement']['mae_micro_old_oncsv_after'] += .1
      with source.open('wb') as stream:
        pickle.dump(data, stream)
      output = Path(tmp) / 'fake' / source.name
      with self.assertRaisesRegex(ReplayError, 'real replay validation failed'):
        replay_result(source, output)
      self.assertFalse(output.exists())

  def test_cli_aggregates_partial_groups_and_all_failed_exits_nonzero(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _write_replay_fixture(root / 'cv' / 'trial0001_ok')
      _write_replay_fixture(root / 'cv' / 'trial0002_bad', broken=True)

      self.assertEqual(replay_main([str(root)]), 0)

      summary = pd.read_csv(root / 'aggregated_summary_fake.csv')
      self.assertEqual(set(summary['summary_row']), {'MEAN', 'STD'})
      self.assertEqual(summary['status'].unique().tolist(), ['partial'])
      self.assertEqual(summary['success_count'].unique().tolist(), [1])
      self.assertEqual(summary['failure_count'].unique().tolist(), [1])
      self.assertTrue((root / 'fake_projection_matched_gaussian' / 'cv'
                       / 'aggregated_fake' / 'results_fake.pkl').is_file())

    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _write_replay_fixture(root / 'trial', broken=True)
      self.assertEqual(replay_main([str(root)]), 1)
      summary = pd.read_csv(root / 'aggregated_summary_fake.csv')
      self.assertEqual(summary['summary_row'].tolist(), ['ERROR'])

  def test_one_trial_cv_folder_still_gets_mean_std_and_aggregate(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _write_replay_fixture(root / 'cv' / 'trial0001')
      self.assertEqual(replay_main([str(root / 'cv')]), 0)
      summary = pd.read_csv(root / 'cv' / 'aggregated_summary_fake.csv')
      self.assertEqual(summary['summary_row'].tolist(), ['MEAN', 'STD'])
      self.assertEqual(summary['new_test_head_mae_micro_before'].tolist(), [.5, 0.])
      self.assertEqual(summary['new_test_head_mae_micro_after'].tolist(), [0., 0.])
      self.assertEqual(summary['new_test_head_mae_micro_delta'].tolist(), [-.5, 0.])
      self.assertTrue((root / 'cv' / 'fake_projection_matched_gaussian'
                       / 'aggregated_fake' / 'results_fake.pkl').is_file())
      aggregate_dir = (root / 'cv' / 'fake_projection_matched_gaussian'
                       / 'aggregated_fake')
      aggregate = pickle.loads((aggregate_dir / 'results_fake.pkl').read_bytes())
      evaluation = aggregate['fake_projection_evaluations']['linear_only']
      before = pd.read_csv(aggregate_dir / 'predictions_fake_before_refinement.csv')
      pd.testing.assert_frame_equal(
        pd.read_csv(aggregate_dir / 'predictions_fake.csv'), before)
      np.testing.assert_array_equal(before['sample_id'], evaluation['sample_ids'])
      np.testing.assert_array_equal(before['label'], evaluation['labels'])
      after = pd.read_csv(
        aggregate_dir / 'predictions_fake_after_refinement_linear_only.csv')
      np.testing.assert_allclose(after['prediction'], evaluation['fake_predictions'])

  def test_aggregate_skips_mode_csv_with_partial_sample_coverage(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      modes = ('linear_only', 'projector_linear')
      _write_replay_fixture(root / 'cv' / 'trial0001', modes=modes)
      partial = _write_replay_fixture(root / 'cv' / 'trial0002', modes=modes)
      data = pickle.loads(partial.read_bytes())
      data['refinements']['projector_linear']['linear_after_pth'] = str(
        partial.parent / 'missing.pt')
      partial.write_bytes(pickle.dumps(data))

      self.assertEqual(replay_main([str(root / 'cv')]), 0)

      aggregate_dir = (root / 'cv' / 'fake_projection_matched_gaussian'
                       / 'aggregated_fake')
      self.assertTrue((aggregate_dir /
                       'predictions_fake_after_refinement_linear_only.csv').is_file())
      self.assertFalse((aggregate_dir /
                        'predictions_fake_after_refinement_projector_linear.csv').exists())

  def test_mode_dashboards_use_after_refinement_predictions_and_metrics(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      source = _write_replay_fixture(
        root / 'trial', modes=('linear_only', 'projector_linear'))
      output = root / 'fake' / source.name
      replay_result(source, output)
      replayed = pickle.loads(output.read_bytes())

      with mock.patch('cross_space_logs.plot_dashboard') as dashboard:
        generate_logs(str(output), skip_umap=True, out_dir_override=root / 'logs')

      mode_calls = {
        call.kwargs.get('filename_suffix', ''): call
        for call in dashboard.call_args_list
      }
      self.assertIn('', mode_calls)
      for mode, evaluation in replayed['fake_projection_evaluations'].items():
        call = mode_calls[f'_{mode}']
        expected = evaluation['fake_predictions']
        np.testing.assert_allclose(call.args[0], expected)
        self.assertAlmostEqual(call.args[4], float(np.mean(np.abs(expected - evaluation['labels']))))
        self.assertAlmostEqual(call.args[5], _ccc(evaluation['labels'], expected))
        self.assertEqual(call.kwargs['projected_stage_name'], 'Refined (after refinement)')

  def test_real_aggregate_mode_dashboard_requires_aligned_subtrials(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      subtrial = _write_replay_fixture(
        root / 'subtrial', modes=('linear_only', 'projector_linear'))
      record = {
        'new_idx': 0, 'old_idx': 0, 'new_model_pth': 'new',
        'old_model_pth': 'old', 'pkl_path': str(subtrial),
      }
      aggregate_path = Path(_aggregate_model_combo_pkls(
        [record], str(root / 'aggregate'), argparse.Namespace()))

      with mock.patch('cross_space_logs.plot_dashboard') as dashboard:
        generate_logs(str(aggregate_path), skip_umap=True, out_dir_override=root / 'logs_ok')
      mode_call = next(
        call for call in dashboard.call_args_list
        if call.kwargs.get('filename_suffix') == '_linear_only')
      aggregate = pickle.loads(aggregate_path.read_bytes())
      np.testing.assert_allclose(
        mode_call.args[0], aggregate['new_model_tensors']['labels'])
      self.assertEqual(mode_call.args[4], 0.)
      self.assertEqual(mode_call.args[5], 1.)

      subtrial_data = pickle.loads(subtrial.read_bytes())
      subtrial_data['new_model_tensors']['sample_ids'][0] = 999
      subtrial.write_bytes(pickle.dumps(subtrial_data))
      with mock.patch('cross_space_logs.plot_dashboard') as dashboard, \
           mock.patch('builtins.print') as printed:
        generate_logs(str(aggregate_path), skip_umap=True, out_dir_override=root / 'logs_bad')
      self.assertFalse(any(
        call.kwargs.get('filename_suffix') == '_linear_only'
        for call in dashboard.call_args_list))
      self.assertTrue(any('misaligned' in str(call).lower()
                          for call in printed.call_args_list))

  def test_real_single_mode_aggregate_skips_unavailable_baseline_dashboard(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      subtrial = _write_replay_fixture(root / 'subtrial')
      aggregate_path = Path(_aggregate_model_combo_pkls([{
        'new_idx': 0, 'old_idx': 0, 'new_model_pth': 'new',
        'old_model_pth': 'old', 'pkl_path': str(subtrial),
      }], str(root / 'aggregate'), argparse.Namespace()))
      subtrial_data = pickle.loads(subtrial.read_bytes())
      subtrial_data['new_model_tensors']['sample_ids'][0] = 999
      subtrial.write_bytes(pickle.dumps(subtrial_data))

      with mock.patch('cross_space_logs.plot_dashboard') as dashboard, \
           mock.patch('builtins.print') as printed:
        generate_logs(str(aggregate_path), skip_umap=True, out_dir_override=root / 'logs')

      dashboard.assert_not_called()
      self.assertTrue(any('baseline dashboard' in str(call).lower()
                          for call in printed.call_args_list))

  def test_fake_vs_real_dashboard_is_written_for_one_mode(self):
    evaluation = {
      'labels': np.array([0., 0., 1., 1.], dtype=np.float32),
      'real_predictions': np.array([0., .2, .8, 1.], dtype=np.float32),
      'fake_predictions': np.array([.5, .6, .4, .5], dtype=np.float32),
      'real_metrics': {'mae_micro': .1, 'mae_macro': .1, 'ccc': .9},
      'fake_metrics': {'mae_micro': .55, 'mae_macro': .55, 'ccc': 0.},
    }
    with tempfile.TemporaryDirectory() as tmp:
      path = plot_fake_vs_real_dashboard(evaluation, tmp, mode='linear_only')
      self.assertEqual(Path(path).name, 'fake_vs_real_dashboard.png')
      self.assertTrue(Path(path).is_file())

  def test_aggregate_write_failure_does_not_turn_successful_replays_into_exit_failure(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _write_replay_fixture(root / 'cv' / 'trial0001')
      _write_replay_fixture(root / 'cv' / 'trial0002')
      with mock.patch('cross_space_fake_projection._aggregate',
                      side_effect=RuntimeError('aggregate disk error')):
        code = replay_main([str(root)])
      self.assertEqual(code, 0)
      summary = pd.read_csv(root / 'aggregated_summary_fake.csv')
      self.assertEqual(summary['status'].unique().tolist(), ['partial'])
      self.assertEqual(summary['success_count'].unique().tolist(), [2])

  def test_leaf_and_aggregate_fake_pkls_load_through_logs(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      _write_replay_fixture(root / 'cv' / 'trial0001')
      _write_replay_fixture(root / 'cv' / 'trial0002')
      self.assertEqual(replay_main([str(root)]), 0)
      leaf = root / 'fake_projection_matched_gaussian' / 'cv' / 'trial0001' / 'results_7.pkl'
      aggregate = (root / 'fake_projection_matched_gaussian' / 'cv'
                   / 'aggregated_fake' / 'results_fake.pkl')

      leaf_logs, _ = generate_logs(
        str(leaf), skip_umap=True, out_dir_override=root / 'leaf_logs')
      aggregate_logs, _ = generate_logs(
        str(aggregate), skip_umap=True, out_dir_override=root / 'aggregate_logs')

      self.assertTrue((Path(leaf_logs) / 'fake_vs_real_dashboard.png').is_file())
      self.assertTrue((Path(aggregate_logs) / 'fake_vs_real_dashboard.png').is_file())


if __name__ == '__main__':
  unittest.main()

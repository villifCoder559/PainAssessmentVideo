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
  ReplayError,
  discover_results,
  generate_fake_embeddings,
  group_results,
  main as replay_main,
  replay_result,
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

from cross_space_logs import generate_logs, plot_fake_vs_real_dashboard


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
          root / 'cv_a' / 'trial0002_x' / 'results_fake.pkl'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'generated')

      found = discover_results(root)

      self.assertEqual(found, wanted)
      grouped = group_results(root, found)
      self.assertEqual(list(grouped), [root / 'cv_a', root / 'cv_b'])

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
      self.assertEqual(rows[0]['status'], 'success')

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
      self.assertTrue((root / 'cv' / 'fake_projection_matched_gaussian'
                       / 'aggregated_fake' / 'results_fake.pkl').is_file())

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

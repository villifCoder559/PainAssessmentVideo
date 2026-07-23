#!/usr/bin/env python3
"""Replay saved cross-space refinements on deterministic fake source embeddings."""
import argparse
import copy
import pickle
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


SEED = 42
DISTRIBUTIONS = ('matched_gaussian', 'standard_normal')
PROJECTOR_KINDS = {'linear', 'mlp', 'procrustes', 'linear_close', 'autoencoder'}
SUMMARY_METRICS = (
  'real_mae_micro', 'real_mae_macro', 'real_ccc',
  'fake_mae_micro', 'fake_mae_macro', 'fake_ccc',
  'fake_minus_real_mae_micro', 'fake_minus_real_mae_macro',
  'fake_minus_real_ccc',
  'new_test_head_mae_micro_before', 'new_test_head_mae_macro_before',
  'new_test_head_mae_micro_after', 'new_test_head_mae_macro_after',
  'new_test_head_mae_micro_delta', 'new_test_head_mae_macro_delta',
)
CONFIG_KEYS = (
  'trial_number', 'uid', 'seed', 'new_model_pth', 'old_model_pth',
  'num_anchors', 'anchor_selection_type',
  'csv_anchor_selection', 'old_model_csv', 'interpolation_similarity',
  'mlp_activation', 'mlp_num_layers', 'weighting_method', 'rbf_sigma',
  'temperature', 'projector_config', 'refinement_config', 'refine_mode',
)


class ReplayError(RuntimeError):
  """An artifact cannot be replayed exactly from its saved state."""


def generate_fake_embeddings(embeddings, distribution='matched_gaussian', seed=SEED):
  """Return a deterministic float32 fake array with the input shape."""
  real = np.asarray(embeddings, dtype=np.float32)
  rng = np.random.default_rng(seed)
  if distribution == 'matched_gaussian':
    fake = rng.normal(real.mean(axis=0), real.std(axis=0), real.shape)
  elif distribution == 'standard_normal':
    fake = rng.standard_normal(real.shape)
  else:
    raise ValueError(f'Unknown distribution {distribution!r}; choose from {DISTRIBUTIONS}')
  return fake.astype(np.float32)


def _is_source_result(path):
  name = path.name.lower()
  if not (name == 'results.pkl' or (name.startswith('results_') and name.endswith('.pkl'))):
    return False
  if 'fake' in name:
    return False
  blocked = ('aggregated', 'logs', 'precomputed', 'fake_projection_')
  return not any(any(part.lower().startswith(prefix) for prefix in blocked)
                 for part in path.parts)


def discover_results(input_path):
  """Discover original result PKLs beneath INPUT in deterministic path order."""
  root = Path(input_path).resolve()
  if root.is_file():
    return [root] if _is_source_result(root) else []
  if not root.is_dir():
    raise FileNotFoundError(f'Input does not exist or is not a directory: {root}')
  return sorted((path for path in root.rglob('*.pkl') if _is_source_result(path)),
                key=lambda path: str(path))


def group_results(input_path, result_paths):
  """Group trial/subtrial results by their enclosing CV/search experiment."""
  root = Path(input_path).resolve()
  if root.is_file():
    root = root.parent
  grouped = {}
  run_prefixes = ('trial', 'cross_space_projection_subtrial_', 'subtrial_', 'fold')
  for result in sorted(map(Path, result_paths), key=lambda path: str(path)):
    parent = result.resolve().parent
    if parent == root:
      group = root
    elif parent.name.lower().startswith(run_prefixes):
      group = parent.parent
    else:
      group = parent
    grouped.setdefault(group, []).append(result.resolve())
  return dict(sorted(grouped.items(), key=lambda item: str(item[0])))


def _metrics(predictions, labels):
  predictions = np.asarray(predictions, dtype=np.float32).reshape(-1)
  labels = np.asarray(labels, dtype=np.float32).reshape(-1)
  if predictions.shape != labels.shape or predictions.size == 0:
    raise ReplayError(
      f'Prediction/label shape mismatch: {predictions.shape} vs {labels.shape}')
  errors = np.abs(predictions - labels)
  classes = np.round(labels).astype(int)
  macro = np.mean([errors[classes == value].mean() for value in np.unique(classes)])
  x, y = labels.astype(np.float64), predictions.astype(np.float64)
  denominator = x.var() + y.var() + (x.mean() - y.mean()) ** 2
  ccc = (1.0 if np.allclose(x, y) else float('nan')) if denominator == 0 else (
    2.0 * np.mean((x - x.mean()) * (y - y.mean())) / denominator)
  return {
    'mae_micro': float(errors.mean()),
    'mae_macro': float(macro),
    'ccc': float(ccc),
  }


def _new_test_head_metrics(block):
  """Return before/after/delta MAE from saved strict-test head predictions."""
  test = block.get('new_test_eval')
  if not isinstance(test, dict) or test.get('split') != 'test':
    raise ReplayError('Missing strict-test refinement new_test_eval block')
  missing = [key for key in ('labels', 'preds_before', 'preds_after')
             if test.get(key) is None]
  if missing:
    raise ReplayError(f'new_test_eval missing fields: {missing}')
  before = _metrics(test['preds_before'], test['labels'])
  after = _metrics(test['preds_after'], test['labels'])
  return {
    'before': {key: before[key] for key in ('mae_micro', 'mae_macro')},
    'after': {key: after[key] for key in ('mae_micro', 'mae_macro')},
    'delta': {key: after[key] - before[key] for key in ('mae_micro', 'mae_macro')},
  }


def _refinement_items(data):
  plural = data.get('refinements')
  if isinstance(plural, dict) and plural:
    return [(str(mode), block) for mode, block in plural.items()]
  block = data.get('refinement')
  if not isinstance(block, dict) or not block:
    return []
  params = data.get('trial_params') or data.get('config_cross_space_projection') or {}
  mode = block.get('refine_mode') or params.get('refine_mode')
  if not mode:
    mode = ('projector_linear' if block.get('projector_after_pth') else 'linear_only')
  return [(str(mode), block)]


def _prediction_frame(evaluation, key):
  sample_ids = np.asarray(evaluation['sample_ids']).reshape(-1)
  labels = np.asarray(evaluation['labels']).reshape(-1)
  predictions = np.asarray(evaluation[key]).reshape(-1)
  if not (len(sample_ids) == len(labels) == len(predictions)):
    raise ReplayError(
      f'Prediction CSV fields are misaligned: {len(sample_ids)}, '
      f'{len(labels)}, {len(predictions)}')
  return pd.DataFrame({
    'sample_id': sample_ids, 'label': labels, 'prediction': predictions,
  })


def _write_fake_prediction_csvs(data, pkl_path):
  """Write before-refinement and per-mode after-refinement fake predictions."""
  evaluations = data.get('fake_projection_evaluations') or {}
  if not evaluations:
    return

  if data.get('aggregated'):
    parts = []
    base = Path(pkl_path).resolve().parent
    for relative in data.get('subtrial_pkls') or []:
      with (base / relative).resolve().open('rb') as stream:
        subtrial = pickle.load(stream)
      sub_evaluations = subtrial.get('fake_projection_evaluations') or {}
      if not sub_evaluations:
        raise ReplayError(f'Aggregate subtrial has no successful fake replay: {relative}')
      parts.append(_prediction_frame(
        next(iter(sub_evaluations.values())), 'fake_before_predictions'))
    before = pd.concat(parts, ignore_index=True)
  else:
    before = _prediction_frame(
      next(iter(evaluations.values())), 'fake_before_predictions')

  output_dir = Path(pkl_path).resolve().parent
  expected = data.get('new_model_tensors') or {}
  if data.get('aggregated') and (
      not np.array_equal(before['sample_id'], np.asarray(expected['sample_ids']).reshape(-1))
      or not np.allclose(before['label'], np.asarray(expected['labels']).reshape(-1))):
    raise ReplayError('Aggregate before-refinement CSV rows are misaligned')
  before.to_csv(output_dir / 'predictions_fake_before_refinement.csv', index=False)
  before.to_csv(output_dir / 'predictions_fake.csv', index=False)
  for mode, evaluation in evaluations.items():
    frame = _prediction_frame(evaluation, 'fake_predictions')
    if data.get('aggregated') and (
        len(frame) != len(before)
        or not np.array_equal(frame['sample_id'], before['sample_id'])
        or not np.allclose(frame['label'], before['label'])):
      continue
    frame.to_csv(
      output_dir / f'predictions_fake_after_refinement_{mode}.csv', index=False)


def _config(data):
  return data.get('trial_params') or data.get('config_cross_space_projection') or {}


def _resolve_path(saved_path, data, source_pkl):
  if not saved_path:
    raise ReplayError('Required checkpoint path is missing from the result artifact')
  saved = Path(saved_path).expanduser()
  if saved.is_file():
    return saved.resolve()
  source_dir = Path(source_pkl).resolve().parent
  if not saved.is_absolute() and (source_dir / saved).is_file():
    return (source_dir / saved).resolve()
  saved_root = (data.get('config_cross_space_projection') or {}).get('out_dir')
  if saved_root:
    try:
      relative = saved.resolve().relative_to(Path(saved_root).expanduser().resolve())
      rebased = source_dir / relative
      if rebased.is_file():
        return rebased.resolve()
    except (OSError, ValueError):
      pass
  parts = list(saved.parts)
  if 'precomputed' in parts:
    suffix = Path(*parts[parts.index('precomputed'):])
    for parent in (source_dir, *source_dir.parents):
      candidate = parent / suffix
      if candidate.is_file():
        return candidate.resolve()
  raise ReplayError(f'Checkpoint not found: {saved_path}')


def _load_state(path):
  try:
    state = torch.load(path, map_location='cpu', weights_only=True)
  except TypeError:  # torch < 2.0
    state = torch.load(path, map_location='cpu')
  if isinstance(state, dict) and 'state_dict' in state:
    state = state['state_dict']
  if not isinstance(state, dict):
    raise ReplayError(f'Checkpoint is not a state_dict: {path}')
  return state


def _linear_from_state(path):
  state = _load_state(path)
  if 'weight' not in state:
    raise ReplayError(f'Linear checkpoint has no weight tensor: {path}')
  out_features, in_features = state['weight'].shape
  module = torch.nn.Linear(int(in_features), int(out_features), bias='bias' in state)
  module.load_state_dict(state)
  return module.eval()


def _projector_from_state(path, activation='gelu'):
  state = _load_state(path)
  if 'weight' in state:
    return _linear_from_state(path)
  weight_keys = sorted(
    (key for key in state if key.endswith('.weight')),
    key=lambda key: int(key.split('.')[0]) if key.split('.')[0].isdigit() else key,
  )
  if not weight_keys:
    raise ReplayError(f'Projector checkpoint has no linear weights: {path}')
  activations = {
    'gelu': torch.nn.GELU, 'relu': torch.nn.ReLU, 'silu': torch.nn.SiLU,
    'leaky_relu': torch.nn.LeakyReLU,
  }
  if activation not in activations:
    raise ReplayError(f'Unknown saved projector activation: {activation!r}')
  layers = []
  for index, key in enumerate(weight_keys):
    out_features, in_features = state[key].shape
    prefix = key[:-len('.weight')]
    layers.append(torch.nn.Linear(
      int(in_features), int(out_features), bias=f'{prefix}.bias' in state))
    if index + 1 < len(weight_keys):
      layers.append(activations[activation]())
  module = torch.nn.Sequential(*layers)
  module.load_state_dict(state)
  return module.eval()


def _apply_projector(module, embeddings, norm_stats):
  values = np.asarray(embeddings, dtype=np.float32)
  if norm_stats is not None:
    values = ((values - np.asarray(norm_stats['old_mean'], dtype=np.float32))
              / np.asarray(norm_stats['old_std'], dtype=np.float32))
  with torch.no_grad():
    projected = module(torch.from_numpy(values.astype(np.float32))).numpy()
  if norm_stats is not None:
    projected = (projected * np.asarray(norm_stats['new_std'], dtype=np.float32)
                 + np.asarray(norm_stats['new_mean'], dtype=np.float32))
  return projected.astype(np.float32)


def _softmax(values):
  values = values - values.max(axis=1, keepdims=True)
  values = np.exp(values)
  return values / values.sum(axis=1, keepdims=True)


def _distance_weights(embeddings, anchors, metric, sigma):
  from scipy.spatial.distance import cdist

  embeddings = np.asarray(embeddings, dtype=np.float32)
  anchors = np.asarray(anchors, dtype=np.float32)
  if anchors.shape[0] == 0:
    raise ReplayError('Saved anchor array is empty')
  if metric == 'cos':
    left = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    right = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    distance = 1.0 - left @ right.T
  elif metric == 'geodesic':
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    direct_anchors = cdist(anchors, anchors)
    k = min(5, len(anchors) - 1)
    if k < 1:
      distance = cdist(embeddings, anchors)
    else:
      graph = np.full(direct_anchors.shape, np.inf)
      np.fill_diagonal(graph, 0.)
      neighbours = np.argsort(direct_anchors, axis=1)[:, 1:k + 1]
      rows = np.repeat(np.arange(len(anchors)), k)
      cols = neighbours.reshape(-1)
      graph[rows, cols] = direct_anchors[rows, cols]
      graph[cols, rows] = direct_anchors[rows, cols]
      anchor_paths = shortest_path(csr_matrix(graph), method='D', directed=False)
      direct = cdist(embeddings, anchors)
      nearest = np.argpartition(direct, k - 1, axis=1)[:, :k]
      distance = np.full(direct.shape, np.inf)
      for column in range(k):
        neighbour = nearest[:, column]
        candidate = direct[np.arange(len(direct)), neighbour, None] + anchor_paths[neighbour]
        np.minimum(distance, candidate, out=distance)
      distance[~np.isfinite(distance)] = direct[~np.isfinite(distance)]
  elif metric == 'l_inf':
    distance = cdist(embeddings, anchors, metric='chebyshev')
  elif metric in ('l1', 'l2'):
    distance = cdist(embeddings, anchors, metric='minkowski', p={'l1': 1, 'l2': 2}[metric])
  else:
    raise ReplayError(f'Unsupported distance projection: {metric!r}')
  return _softmax(-(distance ** 2) / (2.0 * float(sigma) ** 2)).astype(np.float32)


def _projection(data, source_pkl, block, mode, stage, embeddings):
  cfg = _config(data)
  kind = str(cfg.get('interpolation_similarity') or
             (data.get('linear_projector') or {}).get('kind') or '').lower()
  bundle = data.get('linear_projector') or {}
  if kind in PROJECTOR_KINDS or bundle:
    path = block.get(f'projector_{stage}_pth') if mode == 'projector_linear' else None
    path = path or bundle.get('ckpt_path')
    checkpoint = _resolve_path(path, data, source_pkl)
    activation = (bundle.get('config') or {}).get('mlp_activation') or cfg.get('mlp_activation') or 'gelu'
    projector = _projector_from_state(checkpoint, activation)
    projected = _apply_projector(projector, embeddings, bundle.get('norm_stats'))
    return projected, np.zeros((len(projected), 0), dtype=np.float32)

  old_anchors = (data.get('old_model_anchors_embeddings') or {}).get('embeddings')
  new_anchors = (data.get('new_model_anchors_embeddings') or {}).get('embeddings')
  if old_anchors is None or new_anchors is None:
    raise ReplayError(
      f'{kind or "distance"} projection requires saved anchors; grid-distance '
      'artifacts without anchors cannot be replayed exactly')
  weights = _distance_weights(
    embeddings, old_anchors, kind, cfg.get('rbf_sigma', 1.0))
  return (weights @ np.asarray(new_anchors, dtype=np.float32)).astype(np.float32), weights


def _label_denorm(data):
  tensors = data.get('new_model_tensors') or {}
  logits, predictions = tensors.get('logits'), tensors.get('predictions')
  if logits is not None and predictions is not None:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    mask = np.abs(logits) > 1e-8
    if logits.size == predictions.size and mask.any():
      return float(np.median(predictions[mask] / logits[mask]))
  cfg = (data.get('new_model_config') or {}).get('config') or {}
  return float(cfg.get('max_label')) if cfg.get('normalize_labels') and cfg.get('max_label') else 1.0


def _evaluate(data, source_pkl, block, mode, stage, embeddings, labels):
  projected, weights = _projection(data, source_pkl, block, mode, stage, embeddings)
  head = _linear_from_state(_resolve_path(block.get(f'linear_{stage}_pth'), data, source_pkl))
  with torch.no_grad():
    logits = head(torch.from_numpy(projected)).numpy().astype(np.float32)
  predictions = (logits.reshape(-1) * _label_denorm(data)).astype(np.float32)
  return {
    'source_embeddings': np.asarray(embeddings, dtype=np.float32),
    'projected_embeddings': projected,
    'weights': weights,
    'logits': logits,
    'predictions': predictions,
    'metrics': _metrics(predictions, labels),
  }


def _validate_real(mode, saved, actual):
  expected = {
    'mae_micro': saved.get('mae_micro_old_oncsv_after'),
    'mae_macro': saved.get('mae_macro_old_oncsv_after'),
  }
  missing = [name for name, value in expected.items() if value is None]
  if missing:
    raise ReplayError(f'{mode}: missing saved mode-specific metrics: {missing}')
  mismatches = [
    f'{name} saved={float(expected[name]):.8g} replay={actual[name]:.8g}'
    for name in expected
    if not np.isclose(float(expected[name]), actual[name], rtol=1e-4, atol=1e-4)
  ]
  if mismatches:
    raise ReplayError(f'{mode}: real replay validation failed: ' + '; '.join(mismatches))


def _identifier_row(data):
  cfg = _config(data)
  row = {key: cfg.get(key) for key in CONFIG_KEYS}
  row['trial_number'] = data.get('trial_number', row.get('trial_number'))
  row['uid'] = data.get('uid', row.get('uid'))
  row['seed'] = data.get('seed', row.get('seed'))
  return row


def replay_result(source_pkl, output_pkl, distribution='matched_gaussian', seed=SEED):
  """Replay every saved refinement mode and write one compatible fake-result PKL."""
  source_pkl, output_pkl = Path(source_pkl).resolve(), Path(output_pkl).resolve()
  with source_pkl.open('rb') as stream:
    source = pickle.load(stream)
  items = _refinement_items(source)
  if not items:
    raise ReplayError('No saved final refinement modes were found')

  old = source.get('old_model_tensors') or {}
  real_embeddings = np.asarray(old.get('embeddings'), dtype=np.float32)
  labels = np.asarray(
    (source.get('new_model_tensors') or {}).get('labels', old.get('labels')),
    dtype=np.float32,
  ).reshape(-1)
  sample_ids = np.asarray(
    (source.get('new_model_tensors') or {}).get('sample_ids', old.get('sample_ids')),
    dtype=np.int64,
  ).reshape(-1)
  if real_embeddings.ndim != 2 or len(real_embeddings) != len(labels):
    raise ReplayError('Saved source embeddings/labels are missing or misaligned')
  fake_embeddings = generate_fake_embeddings(real_embeddings, distribution, seed)

  output = copy.deepcopy(source)
  evaluations, stage_results, rows, errors = {}, {}, {}, []
  for mode, block in items:
    try:
      head_metrics = _new_test_head_metrics(block)
      real_before = _evaluate(source, source_pkl, block, mode, 'before', real_embeddings, labels)
      fake_before = _evaluate(source, source_pkl, block, mode, 'before', fake_embeddings, labels)
      real_after = _evaluate(source, source_pkl, block, mode, 'after', real_embeddings, labels)
      fake_after = _evaluate(source, source_pkl, block, mode, 'after', fake_embeddings, labels)
      _validate_real(mode, block, real_after['metrics'])
    except Exception as exc:
      errors.append((mode, str(exc)))
      continue

    evaluation = {
      'sample_ids': sample_ids.copy(),
      'labels': labels.copy(),
      'real_predictions': real_after['predictions'],
      'fake_predictions': fake_after['predictions'],
      'real_before_predictions': real_before['predictions'],
      'fake_before_predictions': fake_before['predictions'],
      'real_metrics': real_after['metrics'],
      'fake_metrics': fake_after['metrics'],
      'new_test_head_metrics': head_metrics,
    }
    for prefix, metric in (('real', real_after['metrics']), ('fake', fake_after['metrics'])):
      evaluation.update({f'{prefix}_{name}': value for name, value in metric.items()})
    evaluation.update({
      f'fake_minus_real_{name}': fake_after['metrics'][name] - real_after['metrics'][name]
      for name in ('mae_micro', 'mae_macro', 'ccc')
    })
    evaluations[mode] = evaluation
    stage_results[mode] = (real_before, fake_before, real_after, fake_after)
    row = {
      **_identifier_row(source),
      'refinement_mode': mode,
      'distribution': distribution,
      'fake_projection_seed': seed,
      'source_pkl_path': str(source_pkl),
      'fake_pkl_path': str(output_pkl),
      'replay_error': '',
      'status': 'success',
    }
    for prefix, metric in (('real', real_after['metrics']), ('fake', fake_after['metrics'])):
      for name, value in metric.items():
        row[f'{prefix}_{name}'] = value
    for name in ('mae_micro', 'mae_macro', 'ccc'):
      row[f'fake_minus_real_{name}'] = fake_after['metrics'][name] - real_after['metrics'][name]
    for stage, metrics in head_metrics.items():
      for name, value in metrics.items():
        row[f'new_test_head_{name}_{stage}'] = value
    rows[mode] = row
    print(
      f'  [{mode}] new-model real test head MAE — '
      f"micro: {head_metrics['before']['mae_micro']:.4f} → "
      f"{head_metrics['after']['mae_micro']:.4f} "
      f"(Δ {head_metrics['delta']['mae_micro']:+.4f})  |  "
      f"macro: {head_metrics['before']['mae_macro']:.4f} → "
      f"{head_metrics['after']['mae_macro']:.4f} "
      f"(Δ {head_metrics['delta']['mae_macro']:+.4f})")

    target = ((output.get('refinements') or {}).get(mode)
              if isinstance(output.get('refinements'), dict) else output.get('refinement'))
    if isinstance(target, dict):
      for stage, real_result, fake_result in (
          ('before', real_before, fake_before), ('after', real_after, fake_after)):
        for metric in ('mae_micro', 'mae_macro'):
          key = f'{metric}_old_oncsv_{stage}'
          target[f'real_{key}'] = target.get(key)
          target[f'fake_{key}'] = fake_result['metrics'][metric]
          target[key] = fake_result['metrics'][metric]

  if not evaluations:
    detail = '; '.join(f'{mode}: {error}' for mode, error in errors)
    raise ReplayError(detail or 'No refinement mode replay succeeded')

  multi_mode = len(items) > 1
  headline_mode = next(iter(evaluations))
  real_before, fake_before, real_after, fake_after = stage_results[headline_mode]
  real_headline, fake_headline = ((real_before, fake_before) if multi_mode
                                  else (real_after, fake_after))
  new_tensors = output.setdefault('new_model_tensors', {})
  new_tensors.update({
    'embeddings': fake_headline['projected_embeddings'],
    'weights': fake_headline['weights'],
    'logits': fake_headline['logits'],
    'predictions': fake_headline['predictions'],
    'labels': labels.copy(),
    'sample_ids': sample_ids.copy(),
  })
  output['metrics'] = {
    **(output.get('metrics') or {}),
    'mae': fake_headline['metrics']['mae_micro'],
    'mae_micro': fake_headline['metrics']['mae_micro'],
    'mae_macro': fake_headline['metrics']['mae_macro'],
    'ccc': fake_headline['metrics']['ccc'],
  }
  output['real_projection'] = {
    'predictions': real_headline['predictions'],
    'metrics': real_headline['metrics'],
  }
  output['fake_source_embeddings'] = fake_embeddings
  output['fake_projection_evaluations'] = evaluations
  output['fake_projection_metadata'] = {
    'distribution': distribution,
    'seed': int(seed),
    'source_pkl': str(source_pkl),
    'errors': {mode: error for mode, error in errors},
  }
  output['fake_projection_distribution'] = distribution
  output['fake_projection_seed'] = int(seed)
  cfg = output.get('config_cross_space_projection')
  if isinstance(cfg, dict):
    cfg.update({
      'out_dir': str(output_pkl.parent),
      'fake_projection': True,
      'fake_projection_distribution': distribution,
      'fake_projection_seed': int(seed),
      'fake_projection_source_pkl': str(source_pkl),
    })
  output_pkl.parent.mkdir(parents=True, exist_ok=True)
  with output_pkl.open('wb') as stream:
    pickle.dump(output, stream)
  _write_fake_prediction_csvs(output, output_pkl)

  for mode, error in errors:
    rows[mode] = {
      **_identifier_row(source), 'refinement_mode': mode,
      'distribution': distribution, 'fake_projection_seed': seed,
      'source_pkl_path': str(source_pkl), 'fake_pkl_path': str(output_pkl),
      'replay_error': error, 'status': 'error',
    }
  return list(rows.values())


def _failure_rows(source_pkl, distribution, seed, error):
  try:
    with Path(source_pkl).open('rb') as stream:
      data = pickle.load(stream)
    modes = [mode for mode, _ in _refinement_items(data)] or ['ERROR']
    identifiers = _identifier_row(data)
  except Exception:
    modes, identifiers = ['ERROR'], {}
  return [{
    **identifiers,
    'refinement_mode': mode,
    'distribution': distribution,
    'fake_projection_seed': seed,
    'source_pkl_path': str(Path(source_pkl).resolve()),
    'fake_pkl_path': '',
    'replay_error': str(error),
    'status': 'error',
  } for mode in modes]


def _common_value(rows, key):
  values = [row.get(key) for row in rows if row.get(key) is not None]
  if not values:
    return None
  rendered = list(dict.fromkeys(map(str, values)))
  return values[0] if len(rendered) == 1 else ';'.join(rendered)


def _summary_rows(input_root, grouped, attempt_rows):
  all_paths = [path for paths in grouped.values() for path in paths]
  single = len(all_paths) == 1 and all_paths[0].parent == input_root
  summaries = []
  for group, paths in grouped.items():
    group_rows = [row for row in attempt_rows if Path(row['source_pkl_path']) in paths]
    group_name = str(group.relative_to(input_root)) if group != input_root else input_root.name
    modes = list(dict.fromkeys(row['refinement_mode'] for row in group_rows)) or ['ERROR']
    for mode in modes:
      selected = [row for row in group_rows if row['refinement_mode'] == mode]
      good = [row for row in selected if row['status'] in ('success', 'partial')]
      bad = [row for row in selected if row['status'] == 'error']
      partial = any(row['status'] == 'partial' for row in selected)
      status = 'partial' if good and (bad or partial) else 'success' if good else 'error'
      base = {
        'experiment': group_name,
        'refinement_mode': mode,
        'distribution': _common_value(selected, 'distribution'),
        'fake_projection_seed': _common_value(selected, 'fake_projection_seed'),
        'source_pkl_path': ';'.join(row['source_pkl_path'] for row in selected),
        'fake_pkl_path': ';'.join(row['fake_pkl_path'] for row in good),
        'replay_error': '; '.join(
          row['replay_error'] for row in selected if row.get('replay_error')),
        'status': status,
        'success_count': len(good),
        'failure_count': len(bad),
      }
      base.update({key: _common_value(selected, key) for key in CONFIG_KEYS})
      if not good:
        summaries.append({**base, 'summary_row': 'ERROR'})
        continue
      if single:
        summaries.append({**base, **{key: good[0].get(key) for key in SUMMARY_METRICS},
                          'summary_row': 'RESULT'})
        continue
      for label, reducer in (('MEAN', np.mean), ('STD', lambda values: np.std(values, ddof=1)
                                                   if len(values) > 1 else 0.0)):
        metrics = {
          key: float(reducer([row[key] for row in good]))
          for key in SUMMARY_METRICS
        }
        summaries.append({**base, **metrics, 'summary_row': label})
  return summaries


def _aggregate(group, output_pkls, output_dir):
  from cross_space_projection import _aggregate_model_combo_pkls

  records = []
  for index, pkl_path in enumerate(output_pkls):
    with Path(pkl_path).open('rb') as stream:
      data = pickle.load(stream)
    cfg = _config(data)
    records.append({
      'new_idx': index, 'old_idx': index,
      'new_model_pth': cfg.get('new_model_pth', f'new_{index}'),
      'old_model_pth': cfg.get('old_model_pth', f'old_{index}'),
      'pkl_path': str(pkl_path),
    })
  output = Path(_aggregate_model_combo_pkls(
    records, str(output_dir), argparse.Namespace(), output_filename='results_fake.pkl'))
  with output.open('rb') as stream:
    data = pickle.load(stream)
  _write_fake_prediction_csvs(data, output)
  return output


def run(input_path, distribution='matched_gaussian', seed=SEED):
  """Run discovery, replay, CV aggregation, and summary writing. Return an exit code."""
  input_root = Path(input_path).resolve()
  if input_root.is_file():
    input_root = input_root.parent
  sources = discover_results(input_path)
  grouped = group_results(input_path, sources)
  fake_root = input_root / f'fake_projection_{distribution}'
  attempt_rows, output_by_source, generated = [], {}, []

  for source in tqdm(sources, desc='Testing fake projections', unit='test'):
    output = fake_root / source.relative_to(input_root)
    try:
      rows = replay_result(source, output, distribution, seed)
      attempt_rows.extend(rows)
      if any(row['status'] == 'success' for row in rows):
        output_by_source[source] = output
        generated.append(output)
    except Exception as exc:
      attempt_rows.extend(_failure_rows(source, distribution, seed, exc))

  for group, paths in grouped.items():
    successful = [output_by_source[path] for path in paths if path in output_by_source]
    is_cv_group = len(paths) > 1 or any(path.parent != group for path in paths)
    if successful and is_cv_group:
      relative = group.relative_to(input_root)
      aggregate_dir = fake_root / relative / 'aggregated_fake'
      try:
        generated.append(_aggregate(group, successful, aggregate_dir))
      except Exception as exc:
        for row in attempt_rows:
          if Path(row['source_pkl_path']) in paths and row['status'] == 'success':
            row['replay_error'] = f'aggregation failed: {exc}'
            row['status'] = 'partial'

  summaries = _summary_rows(input_root, grouped, attempt_rows)
  if not summaries:
    summaries = [{
      'experiment': input_root.name, 'summary_row': 'ERROR', 'status': 'error',
      'distribution': distribution, 'fake_projection_seed': seed,
      'success_count': 0, 'failure_count': 0,
      'replay_error': 'No original results.pkl/results_*.pkl files found',
    }]
  summary_path = input_root / 'aggregated_summary_fake.csv'
  pd.DataFrame(summaries).to_csv(summary_path, index=False)

  print(f'Generated: {summary_path}')
  for path in generated:
    print(f'Generated: {path}')
    print('Logs: python3 cross_space_logs.py --pkl_path '
          f'{shlex.quote(str(path))} --skip_umap')
  return 0 if output_by_source else 1


def main(argv=None):
  parser = argparse.ArgumentParser(
    description='Replay saved cross-space refinements on deterministic fake embeddings.')
  parser.add_argument('input', help='Single trial, CV experiment, or higher root folder')
  parser.add_argument('--distribution', choices=DISTRIBUTIONS, default='matched_gaussian')
  args = parser.parse_args(argv)
  try:
    return run(args.input, args.distribution, SEED)
  except (FileNotFoundError, ValueError) as exc:
    parser.error(str(exc))


if __name__ == '__main__':
  sys.exit(main())

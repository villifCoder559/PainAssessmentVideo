#!/usr/bin/env python3
"""Compare paired subject-independent k-fold results from two pickle files."""

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
from scipy import stats


RESULT_FIELDS = [
  'analysis_level',
  'analysis_role',
  'metric',
  'n_pairs',
  'metric_source',
  'pkl_0_mean',
  'pkl_1_mean',
  'mean_effect_pkl0_better',
  'permutation_statistic',
  'permutation_p_raw',
  'permutation_p_holm',
  'permutation_significant',
  'wilcoxon_statistic',
  'wilcoxon_p_raw',
  'wilcoxon_p_holm',
  'wilcoxon_significant',
]
SANITY_FIELDS = [
  'pkl_index',
  'fold',
  'analysis_level',
  'subject_id',
  'metric',
  'stored_value',
  'recomputed_value',
  'absolute_difference',
  'tolerance',
  'matches',
]


def _final_keys(data):
  """Return naturally sorted result keys containing ``final``."""
  keys = [key for key in data.get('results', {}) if 'final' in key.lower()]
  if not keys:
    raise ValueError('No result keys containing "final" were found')
  return sorted(keys, key=lambda key: int(re.search(r'\d+', key).group()) if re.search(r'\d+', key) else key)


def _numeric_array(value, label):
  """Convert a stored tensor or sequence to a finite one-dimensional float array."""
  if hasattr(value, 'detach'):
    value = value.detach().cpu().numpy()
  array = np.asarray(value, dtype=float).reshape(-1)
  if not np.all(np.isfinite(array)):
    raise ValueError(f'{label} contains non-finite values')
  return array


def extract_paired_values(data_0, data_1):
  """Extract aligned subject- and fold-level MAE and accuracy pairs."""
  keys_0 = _final_keys(data_0)
  keys_1 = _final_keys(data_1)
  if keys_0 != keys_1:
    raise ValueError(f'Final fold keys differ: {keys_0} != {keys_1}')

  subject_values = {'mae': [[], []], 'accuracy': [[], []]}
  fold_values = {'mae': [[], []], 'accuracy': [[], []]}
  seen_subjects = [set(), set()]

  for key in keys_0:
    tests = [data_0['results'][key]['test'], data_1['results'][key]['test']]
    ids = [np.asarray(test['test_unique_subject_ids']).reshape(-1) for test in tests]
    if len(set(ids[0].tolist())) != len(ids[0]) or len(set(ids[1].tolist())) != len(ids[1]):
      raise ValueError(f'{key}: duplicate subject IDs within a fold')
    if set(ids[0].tolist()) != set(ids[1].tolist()):
      raise ValueError(f'{key}: subject IDs differ between pickle files')

    order = sorted(ids[0].tolist())
    indices = [{subject_id: idx for idx, subject_id in enumerate(fold_ids.tolist())} for fold_ids in ids]
    for model_idx in (0, 1):
      duplicates = seen_subjects[model_idx].intersection(order)
      if duplicates:
        raise ValueError(f'Subject IDs occur in multiple final folds: {sorted(duplicates)}')
      seen_subjects[model_idx].update(order)

      fields = {
        'mae': 'test_loss_per_subject',
        'accuracy': 'test_accuracy_per_subject',
      }
      for metric, field in fields.items():
        values = _numeric_array(tests[model_idx][field], f'{key}.{field}')
        if len(values) != len(ids[model_idx]):
          raise ValueError(f'{key}.{field} length does not match subject IDs')
        subject_values[metric][model_idx].extend(values[indices[model_idx][subject_id]] for subject_id in order)

    for metric, field in {'mae': 'test_l1_error', 'accuracy': 'test_accuracy'}.items():
      for model_idx in (0, 1):
        value = float(tests[model_idx][field])
        if not np.isfinite(value):
          raise ValueError(f'{key}.{field} is not finite')
        fold_values[metric][model_idx].append(value)

  return {
    'subject': {metric: tuple(np.asarray(values, dtype=float) for values in models)
                for metric, models in subject_values.items()},
    'fold': {metric: tuple(np.asarray(values, dtype=float) for values in models)
             for metric, models in fold_values.items()},
  }


def holm_adjust(p_values):
  """Return Holm family-wise-error adjusted p-values in input order."""
  values = np.asarray(p_values, dtype=float)
  order = np.argsort(values)
  adjusted_sorted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
  adjusted = np.empty_like(values)
  adjusted[order] = np.minimum(adjusted_sorted, 1.0)
  return adjusted


def _test_difference(differences, level):
  """Run the paired mean sign-flip test and Wilcoxon test on differences."""
  if np.all(differences == 0):
    return float(np.mean(differences)), 1.0, 0.0, 1.0
  permutation = stats.permutation_test(
    (differences,),
    lambda values: np.mean(values),
    permutation_type='samples',
    n_resamples=99999 if level == 'subject' else np.inf,
    alternative='two-sided',
    vectorized=False,
    random_state=0,
  )
  wilcoxon = stats.wilcoxon(differences, alternative='two-sided', method='auto')
  return (
    float(permutation.statistic),
    float(permutation.pvalue),
    float(wilcoxon.statistic),
    float(wilcoxon.pvalue),
  )


def compare_paired_values(paired, metric_source):
  """Build four statistical comparison rows from paired metric arrays."""
  rows = []
  for level in ('subject', 'fold'):
    for metric in ('mae', 'accuracy'):
      values_0, values_1 = paired[level][metric]
      differences = values_1 - values_0 if metric == 'mae' else values_0 - values_1
      permutation_stat, permutation_p, wilcoxon_stat, wilcoxon_p = _test_difference(differences, level)
      rows.append({
        'analysis_level': level,
        'analysis_role': 'primary' if level == 'subject' else 'exploratory',
        'metric': metric,
        'n_pairs': len(differences),
        'metric_source': metric_source,
        'pkl_0_mean': float(np.mean(values_0)),
        'pkl_1_mean': float(np.mean(values_1)),
        'mean_effect_pkl0_better': float(np.mean(differences)),
        'permutation_statistic': permutation_stat,
        'permutation_p_raw': permutation_p,
        'permutation_p_holm': None,
        'permutation_significant': None,
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_p_raw': wilcoxon_p,
        'wilcoxon_p_holm': None,
        'wilcoxon_significant': None,
      })

  subject_rows = [row for row in rows if row['analysis_level'] == 'subject']
  for prefix in ('permutation', 'wilcoxon'):
    adjusted = holm_adjust([row[f'{prefix}_p_raw'] for row in subject_rows])
    for row, p_value in zip(subject_rows, adjusted):
      row[f'{prefix}_p_holm'] = float(p_value)
      row[f'{prefix}_significant'] = bool(p_value < 0.05)
  return rows


def write_rows(path, rows, fieldnames=RESULT_FIELDS):
  """Write dictionaries to a CSV file using a fixed column order."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def load_pickle(path):
  """Load one trusted experiment pickle."""
  import custom.helper  # Initialize its multiprocessing manager before unpickling model classes.
  with open(path, 'rb') as handle:
    return pickle.load(handle)


def recompute_fold_metrics(pkl_path, data):
  """Re-run every final fold checkpoint and return fresh evaluation metrics."""
  import torch
  from extract_kfold_test_table import recompute_raw_fold_metrics

  if not torch.cuda.is_available():
    raise RuntimeError('--recompute requires an available CUDA device')
  features = data.get('model_advanced_params', {}).get('features_folder_saving_path')
  if not features or not Path(features).exists():
    raise FileNotFoundError(f'Cached feature path is unavailable: {features}')
  return recompute_raw_fold_metrics(pkl_path, data, _final_keys(data))


def _subject_metric_map(test, ids_field, value_field, label):
  """Return a validated subject-ID-to-value mapping for one metric."""
  ids = np.asarray(test[ids_field]).reshape(-1)
  values = _numeric_array(test[value_field], label)
  if len(ids) != len(values) or len(set(ids.tolist())) != len(ids):
    raise ValueError(f'{label} does not align with unique subject IDs')
  return dict(zip(ids.tolist(), values.tolist()))


def build_sanity_rows(pkl_index, stored, recomputed, tolerance=1e-6):
  """Build detailed stored-versus-fresh fold and subject audit rows."""
  rows = []
  keys = _final_keys(stored)
  if set(keys) != set(recomputed):
    raise ValueError('Recomputed fold keys do not match stored final fold keys')
  for key in keys:
    test = stored['results'][key]['test']
    fresh = recomputed[key]
    for metric, stored_field, fresh_field in (
      ('mae', 'test_l1_error', 'recomputed_l1'),
      ('accuracy', 'test_accuracy', 'recomputed_accuracy'),
    ):
      stored_value = float(test[stored_field])
      fresh_value = float(fresh[fresh_field])
      difference = abs(stored_value - fresh_value)
      rows.append({
        'pkl_index': pkl_index,
        'fold': key,
        'analysis_level': 'fold',
        'subject_id': None,
        'metric': metric,
        'stored_value': stored_value,
        'recomputed_value': fresh_value,
        'absolute_difference': difference,
        'tolerance': tolerance,
        'matches': bool(difference <= tolerance),
      })

    stored_ids = np.asarray(test['test_unique_subject_ids']).reshape(-1)
    fresh_ids = np.asarray(fresh['recomputed_subject_ids']).reshape(-1)
    if set(stored_ids.tolist()) != set(fresh_ids.tolist()):
      raise ValueError(f'{key}: recomputed subject IDs differ from stored IDs')
    for metric, stored_field, fresh_field in (
      ('mae', 'test_loss_per_subject', 'recomputed_loss_per_subject'),
      ('accuracy', 'test_accuracy_per_subject', 'recomputed_accuracy_per_subject'),
    ):
      stored_map = _subject_metric_map(test, 'test_unique_subject_ids', stored_field, f'{key}.{stored_field}')
      fresh_test = {
        'ids': fresh['recomputed_subject_ids'],
        'values': fresh[fresh_field],
      }
      fresh_map = _subject_metric_map(fresh_test, 'ids', 'values', f'{key}.{fresh_field}')
      for subject_id in sorted(stored_map):
        difference = abs(stored_map[subject_id] - fresh_map[subject_id])
        rows.append({
          'pkl_index': pkl_index,
          'fold': key,
          'analysis_level': 'subject',
          'subject_id': subject_id,
          'metric': metric,
          'stored_value': stored_map[subject_id],
          'recomputed_value': fresh_map[subject_id],
          'absolute_difference': difference,
          'tolerance': tolerance,
          'matches': bool(difference <= tolerance),
        })
  return rows


def with_recomputed_metrics(stored, recomputed):
  """Return a shallow copy whose final test metrics are freshly evaluated."""
  updated = dict(stored)
  updated['results'] = dict(stored['results'])
  for key, fresh in recomputed.items():
    result = dict(stored['results'][key])
    test = dict(result['test'])
    test.update({
      'test_l1_error': fresh['recomputed_l1'],
      'test_accuracy': fresh['recomputed_accuracy'],
      'test_loss_per_subject': fresh['recomputed_loss_per_subject'],
      'test_accuracy_per_subject': fresh['recomputed_accuracy_per_subject'],
      'test_unique_subject_ids': fresh['recomputed_subject_ids'],
    })
    result['test'] = test
    updated['results'][key] = result
  return updated


def run_comparison(pkl_path_0, pkl_path_1, output, recompute=False):
  """Load inputs, optionally re-evaluate checkpoints, and write comparison reports."""
  data = [load_pickle(pkl_path_0), load_pickle(pkl_path_1)]
  sanity_rows = []
  source = 'stored'
  if recompute:
    fresh = [
      recompute_fold_metrics(pkl_path_0, data[0]),
      recompute_fold_metrics(pkl_path_1, data[1]),
    ]
    for index in (0, 1):
      sanity_rows.extend(build_sanity_rows(index, data[index], fresh[index]))
      data[index] = with_recomputed_metrics(data[index], fresh[index])
    sanity_path = Path(output).with_name(f'{Path(output).stem}_sanity{Path(output).suffix}')
    write_rows(sanity_path, sanity_rows, SANITY_FIELDS)
    mismatches = [row for row in sanity_rows if not row['matches']]
    maximum = max((row['absolute_difference'] for row in sanity_rows), default=0.0)
    print(f'Sanity check: {len(mismatches)} mismatches; maximum absolute difference={maximum:.6g}')
    if mismatches:
      print('WARNING: stored and recomputed metrics differ; statistical tests use recomputed values.')
    source = 'recomputed'

  rows = compare_paired_values(extract_paired_values(data[0], data[1]), metric_source=source)
  write_rows(output, rows)
  return rows, sanity_rows


def main():
  """Parse arguments, compare two pickle files, and write the result CSV."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pkl_path_0')
  parser.add_argument('--pkl_path_1')
  parser.add_argument('--out', default='statistical_comparison.csv')
  parser.add_argument('--recompute', action='store_true',
                      help='Re-run best checkpoints, audit stored metrics, and test fresh metrics')
  args = parser.parse_args()

  rows, _ = run_comparison(args.pkl_path_0, args.pkl_path_1, args.out, recompute=args.recompute)
  for row in rows:
    effect = row['mean_effect_pkl0_better'] * (100 if row['metric'] == 'accuracy' else 1)
    unit = ' pp' if row['metric'] == 'accuracy' else ''
    print(f"{row['analysis_level']:7} {row['metric']:8} effect={effect:.6f}{unit} "
          f"permutation_p={row['permutation_p_raw']:.6g} wilcoxon_p={row['wilcoxon_p_raw']:.6g}")
  print(f'\nSaved to {args.out}')


if __name__ == '__main__':
  main()

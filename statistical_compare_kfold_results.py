#!/usr/bin/env python3
"""Compare paired subject-independent k-fold results from two pickle files."""

import argparse
import csv
import pickle
import re
import warnings
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
  'ttest_statistic',
  'ttest_p_raw',
  'ttest_p_holm',
  'ttest_significant',
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


def _fold_csv_path(pkl_path, data, fold_key):
  """Locate the cleaned or original test CSV for one final fold."""
  head = data.get('model_advanced_params', {}).get('head')
  if head is None:
    raise ValueError('model_advanced_params.head is required for stored video analysis')
  fold = fold_key.split('_')[0]
  folder = Path(pkl_path).parent / f'train_{head}' / f'{fold}_cross_val'
  for filename in ('test_cleaned.csv', 'test.csv'):
    candidate = folder / filename
    if candidate.is_file():
      return candidate
  raise FileNotFoundError(f'{fold_key}: no test_cleaned.csv or test.csv found under {folder}')


def _video_maps(test, data, pkl_path, fold_key):
  """Return validated sample-ID mappings for predictions and labels."""
  embedded = ('test_sample_ids', 'test_sample_predictions', 'test_sample_labels')
  present = [field in test for field in embedded]
  if any(present):
    if not all(present):
      raise ValueError(f'{fold_key}: recomputed video fields are incomplete')
    ids = _numeric_array(test['test_sample_ids'], f'{fold_key}.test_sample_ids').astype(int)
    predictions = _numeric_array(
      test['test_sample_predictions'], f'{fold_key}.test_sample_predictions'
    )
    labels = _numeric_array(test['test_sample_labels'], f'{fold_key}.test_sample_labels')
    if len(ids) != len(predictions) or len(ids) != len(labels):
      raise ValueError(f'{fold_key}: recomputed video fields have unequal lengths')
    if len(set(ids.tolist())) != len(ids):
      raise ValueError(f'{fold_key}: duplicate video IDs')
    return dict(zip(ids.tolist(), predictions.tolist())), dict(zip(ids.tolist(), labels.tolist()))

  history = test.get('history_test_sample_predictions')
  if not history:
    raise ValueError(
      f'{fold_key}: stored per-video predictions are unavailable; rerun with --recompute'
    )
  predictions = {}
  for raw_sample_id, stored_values in history.items():
    sample_id = int(raw_sample_id)
    if sample_id in predictions:
      raise ValueError(f'{fold_key}: duplicate video IDs')
    values = _numeric_array(stored_values, f'{fold_key}.history[{sample_id}]')
    if len(values) != 1:
      raise ValueError(f'{fold_key}: video {sample_id} must have exactly one test prediction')
    predictions[sample_id] = float(values[0])

  if pkl_path is None:
    raise ValueError('pkl_paths are required to load stored video ground truth')
  csv_path = _fold_csv_path(pkl_path, data, fold_key)
  labels = {}
  with csv_path.open(newline='') as handle:
    reader = csv.DictReader(handle, delimiter='\t')
    if not reader.fieldnames or not {'sample_id', 'class_id'}.issubset(reader.fieldnames):
      raise ValueError(f'{csv_path}: sample_id and class_id columns are required')
    for row in reader:
      sample_id = int(row['sample_id'])
      if sample_id in labels:
        raise ValueError(f'{fold_key}: duplicate video ID {sample_id} in {csv_path}')
      labels[sample_id] = float(row['class_id'])
  missing = set(predictions) - set(labels)
  if missing:
    raise ValueError(f'{fold_key}: ground truth is missing for video IDs {sorted(missing)[:5]}')
  return predictions, {sample_id: labels[sample_id] for sample_id in predictions}


def _extract_video_values(data_0, data_1, keys, pkl_paths):
  """Extract aligned per-video absolute errors and correctness indicators."""
  if pkl_paths is None:
    pkl_paths = (None, None)
  values_by_metric = {'mae': [[], []], 'accuracy': [[], []]}
  seen_ids = set()
  for key in keys:
    tests = [data_0['results'][key]['test'], data_1['results'][key]['test']]
    maps = [
      _video_maps(tests[index], (data_0, data_1)[index], pkl_paths[index], key)
      for index in (0, 1)
    ]
    predictions = [item[0] for item in maps]
    labels = [item[1] for item in maps]
    if set(predictions[0]) != set(predictions[1]):
      raise ValueError(f'{key}: video IDs differ between pickle files')
    order = sorted(predictions[0])
    duplicates = seen_ids.intersection(order)
    if duplicates:
      raise ValueError(f'Video IDs occur in multiple final folds: {sorted(duplicates)[:5]}')
    seen_ids.update(order)
    if any(labels[0][sample_id] != labels[1][sample_id] for sample_id in order):
      raise ValueError(f'{key}: ground-truth labels differ between pickle files')

    maximum_class = max(labels[0][sample_id] for sample_id in order)
    for model_idx in (0, 1):
      for sample_id in order:
        prediction = predictions[model_idx][sample_id]
        label = labels[model_idx][sample_id]
        rounded = np.copysign(np.floor(abs(prediction) + 0.5), prediction)
        rounded = np.clip(rounded, 0, maximum_class)
        values_by_metric['mae'][model_idx].append(abs(prediction - label))
        values_by_metric['accuracy'][model_idx].append(float(rounded == label))
  return {
    metric: tuple(np.asarray(values, dtype=float) for values in models)
    for metric, models in values_by_metric.items()
  }


def extract_paired_values(data_0, data_1, analysis_level, pkl_paths=None):
  """Extract aligned MAE and accuracy pairs for one analysis level."""
  if analysis_level not in ('subject', 'model', 'video'):
    raise ValueError('analysis_level must be subject, model, or video')

  keys_0 = _final_keys(data_0)
  keys_1 = _final_keys(data_1)
  if keys_0 != keys_1:
    raise ValueError(f'Final fold keys differ: {keys_0} != {keys_1}')
  if analysis_level == 'video':
    return _extract_video_values(data_0, data_1, keys_0, pkl_paths)

  values_by_metric = {'mae': [[], []], 'accuracy': [[], []]}
  seen_subjects = [set(), set()]

  for key in keys_0:
    tests = [data_0['results'][key]['test'], data_1['results'][key]['test']]
    if analysis_level == 'subject':
      ids = [np.asarray(test['test_unique_subject_ids']).reshape(-1) for test in tests]
      if len(set(ids[0].tolist())) != len(ids[0]) or len(set(ids[1].tolist())) != len(ids[1]):
        raise ValueError(f'{key}: duplicate subject IDs within a fold')
      if set(ids[0].tolist()) != set(ids[1].tolist()):
        raise ValueError(f'{key}: subject IDs differ between pickle files')

      order = sorted(ids[0].tolist())
      for model_idx in (0, 1):
        duplicates = seen_subjects[model_idx].intersection(order)
        if duplicates:
          raise ValueError(f'Subject IDs occur in multiple final folds: {sorted(duplicates)}')
        seen_subjects[model_idx].update(order)
        indices = {subject_id: idx for idx, subject_id in enumerate(ids[model_idx].tolist())}
        for metric, field in {
          'mae': 'test_loss_per_subject',
          'accuracy': 'test_accuracy_per_subject',
        }.items():
          values = _numeric_array(tests[model_idx][field], f'{key}.{field}')
          if len(values) != len(ids[model_idx]):
            raise ValueError(f'{key}.{field} length does not match subject IDs')
          values_by_metric[metric][model_idx].extend(
            values[indices[subject_id]] for subject_id in order
          )
    else:
      for metric, field in {'mae': 'test_l1_error', 'accuracy': 'test_accuracy'}.items():
        for model_idx in (0, 1):
          value = float(tests[model_idx][field])
          if not np.isfinite(value):
            raise ValueError(f'{key}.{field} is not finite')
          values_by_metric[metric][model_idx].append(value)

  return {
    metric: tuple(np.asarray(values, dtype=float) for values in models)
    for metric, models in values_by_metric.items()
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
  """Run paired t, mean sign-flip, and Wilcoxon tests on differences."""
  if len(differences) < 2:
    raise ValueError('Paired analyses require at least two pairs')
  if np.all(differences == 0):
    return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0
  if np.allclose(differences, differences[0], rtol=1e-12, atol=1e-15):
    ttest_statistic = float(np.copysign(np.inf, differences[0]))
    ttest_pvalue = 0.0
  else:
    ttest = stats.ttest_1samp(differences, popmean=0.0)
    ttest_statistic = float(ttest.statistic)
    ttest_pvalue = float(ttest.pvalue)
  with warnings.catch_warnings():
    warnings.filterwarnings(
      'ignore', message='overflow encountered in scalar power', category=RuntimeWarning
    )
    permutation = stats.permutation_test(
      (differences,),
      lambda values: np.mean(values),
      permutation_type='samples',
      n_resamples=np.inf if level == 'model' else 99999,
      alternative='two-sided',
      vectorized=False,
      random_state=0,
    )
  wilcoxon = stats.wilcoxon(differences, alternative='two-sided', method='auto')
  return (
    ttest_statistic,
    ttest_pvalue,
    float(permutation.statistic),
    float(permutation.pvalue),
    float(wilcoxon.statistic),
    float(wilcoxon.pvalue),
  )


def compare_paired_values(paired, analysis_level, measure, metric_source):
  """Build statistical comparison rows from paired metric arrays."""
  if analysis_level not in ('subject', 'model', 'video'):
    raise ValueError('analysis_level must be subject, model, or video')
  if measure not in ('mae', 'accuracy', 'both'):
    raise ValueError('measure must be mae, accuracy, or both')

  rows = []
  metrics = ('mae', 'accuracy') if measure == 'both' else (measure,)
  for metric in metrics:
    values_0, values_1 = paired[metric]
    if len(values_0) != len(values_1):
      raise ValueError(f'{metric} paired arrays have different lengths')
    differences = values_1 - values_0 if metric == 'mae' else values_0 - values_1
    (ttest_stat, ttest_p, permutation_stat, permutation_p,
     wilcoxon_stat, wilcoxon_p) = _test_difference(differences, analysis_level)
    rows.append({
        'analysis_level': analysis_level,
        'analysis_role': 'exploratory' if analysis_level == 'model' else 'primary',
        'metric': metric,
        'n_pairs': len(differences),
        'metric_source': metric_source,
        'pkl_0_mean': float(np.mean(values_0)),
        'pkl_1_mean': float(np.mean(values_1)),
        'mean_effect_pkl0_better': float(np.mean(differences)),
        'ttest_statistic': ttest_stat,
        'ttest_p_raw': ttest_p,
        'ttest_p_holm': None,
        'ttest_significant': bool(ttest_p < 0.05) if analysis_level != 'subject' else None,
        'permutation_statistic': permutation_stat,
        'permutation_p_raw': permutation_p,
        'permutation_p_holm': None,
        'permutation_significant': None,
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_p_raw': wilcoxon_p,
        'wilcoxon_p_holm': None,
        'wilcoxon_significant': None,
      })

  if analysis_level == 'subject':
    for prefix in ('ttest', 'permutation', 'wilcoxon'):
      adjusted = holm_adjust([row[f'{prefix}_p_raw'] for row in rows])
      for row, p_value in zip(rows, adjusted):
        row[f'{prefix}_p_holm'] = float(p_value)
        row[f'{prefix}_significant'] = bool(p_value < 0.05)
  else:
    for prefix in ('permutation', 'wilcoxon'):
      for row in rows:
        row[f'{prefix}_significant'] = bool(row[f'{prefix}_p_raw'] < 0.05)
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


def _model_name(data, pkl_path):
  """Return a filesystem-safe model name from result metadata or its folder."""
  value = data.get('config', {}).get('model_type')
  if value is None:
    value = data.get('model_advanced_params', {}).get('model_type')
  if value is None:
    value = Path(pkl_path).parent.name
  if hasattr(value, 'name'):
    value = value.name
  name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._-')
  return name or 'model'


def _dataset_name(data):
  """Return a filesystem-safe dataset name from result metadata."""
  csv_path = data.get('config', {}).get('path_csv_dataset')
  if not csv_path:
    raise ValueError('config.path_csv_dataset is required to infer the dataset name')
  csv_path = "/".join(csv_path) if isinstance(csv_path, (list, tuple)) else csv_path
  if 'unbc' in csv_path.lower():
    return 'unbc'
  elif 'xite' in csv_path.lower():
    return 'xite'
  elif 'parta' in csv_path.lower():
    return 'biovid'
  elif 'mint' in csv_path.lower():
    return 'mint'
  else:
    raise ValueError(f'Cannot infer dataset name from {csv_path}')

def resolve_output_path(output, data_0, data_1, analysis_level, measure, recompute,
                        pkl_paths=('model_0.pkl', 'model_1.pkl')):
  """Resolve an explicit output or build the default descriptive CSV path."""
  if output:
    return Path(output)
  model_0 = _model_name(data_0, pkl_paths[0])
  model_1 = _model_name(data_1, pkl_paths[1])
  dataset_0 = _dataset_name(data_0)
  dataset_1 = _dataset_name(data_1)
  if dataset_0 != dataset_1:
    raise ValueError(f'Dataset names differ: {dataset_0} != {dataset_1}')
  source = 'recomputed' if recompute else 'stored'
  filename = f'{model_0}_vs_{model_1}_{analysis_level}_{measure}_{source}.csv'
  return Path('statistical_analysis_folder', dataset_0, filename)


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
    sample_fields = (
      'recomputed_sample_ids',
      'recomputed_sample_predictions',
      'recomputed_sample_labels',
    )
    if all(field in fresh for field in sample_fields):
      test.update({
        'test_sample_ids': fresh['recomputed_sample_ids'],
        'test_sample_predictions': fresh['recomputed_sample_predictions'],
        'test_sample_labels': fresh['recomputed_sample_labels'],
      })
    result['test'] = test
    updated['results'][key] = result
  return updated


def run_comparison(pkl_path_0, pkl_path_1, output, analysis_level, measure='both', recompute=False):
  """Load inputs, optionally re-evaluate checkpoints, and write comparison reports."""
  data = [load_pickle(pkl_path_0), load_pickle(pkl_path_1)]
  output = resolve_output_path(
    output, data[0], data[1], analysis_level, measure, recompute,
    pkl_paths=(pkl_path_0, pkl_path_1),
  )
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

  paired = extract_paired_values(
    data[0], data[1], analysis_level, pkl_paths=(pkl_path_0, pkl_path_1)
  )
  rows = compare_paired_values(
    paired, analysis_level=analysis_level, measure=measure, metric_source=source
  )
  write_rows(output, rows)
  print(f'\nSaved to {output}')
  return rows, sanity_rows


def parse_args(argv=None):
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--pkl_path_0', required=True)
  parser.add_argument('--pkl_path_1', required=True)
  parser.add_argument('--analysis_level', required=True, choices=('subject', 'model', 'video'))
  parser.add_argument('--measure', choices=('mae', 'accuracy', 'both'), default='both')
  parser.add_argument('--out', default=None)
  parser.add_argument('--recompute', action='store_true',
                      help='Re-run best checkpoints, audit stored metrics, and test fresh metrics')
  return parser.parse_args(argv)


def main():
  """Parse arguments, compare two pickle files, and write the result CSV."""
  args = parse_args()

  rows, _ = run_comparison(
    args.pkl_path_0,
    args.pkl_path_1,
    args.out,
    analysis_level=args.analysis_level,
    measure=args.measure,
    recompute=args.recompute,
  )
  for row in rows:
    effect = row['mean_effect_pkl0_better'] * (100 if row['metric'] == 'accuracy' else 1)
    unit = ' pp' if row['metric'] == 'accuracy' else ''
    print(f"{row['analysis_level']:7} {row['metric']:8} effect={effect:.6f}{unit} "
          f"permutation_p={row['permutation_p_raw']:.6g} wilcoxon_p={row['wilcoxon_p_raw']:.6g}")


if __name__ == '__main__':
  main()

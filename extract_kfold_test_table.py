import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd


def to_numpy(x):
  """
  Convert a tensor/array/list to a numpy array.

  Args:
    x: torch.Tensor, np.ndarray or list.

  Returns:
    np.ndarray version of x.
  """
  if hasattr(x, 'detach'):
    return x.detach().cpu().numpy()
  return np.asarray(x)


def recompute_raw_fold_metrics(pkl_path: str, data: dict, final_keys: list, ffsp_override: str = None) -> dict:
  """
  Re-run each fold's best checkpoint on its test set to get raw (un-rounded,
  de-normalized) per-sample predictions and compute sample-aggregated MAE.

  Args:
    pkl_path:      Path to k_fold_results.pkl (checkpoints live next to it under train_<HEAD>/).
    data:          Loaded pkl dict (needs 'results', 'config', 'model_advanced_params').
    final_keys:    Ordered list of 'k<i>_cross_val_final' result keys.
    ffsp_override: Optional replacement for features_folder_saving_path.

  Returns:
    Dict keyed by fold with raw MAE plus freshly evaluated fold/subject L1
    and accuracy metrics.
  """
  import custom.helper as helper
  from custom.model import Model_Advanced

  helper.init_log_cross_attention()
  helper.init_log_video_embeddings()
  helper.LOG_HISTORY_SAMPLE = True

  params = data['model_advanced_params']
  params['head_params']['skip_init_weights'] = True
  if ffsp_override is not None:
    params['features_folder_saving_path'] = ffsp_override
  model = Model_Advanced(**params)

  cfg = data['config']
  run_dir = os.path.dirname(os.path.abspath(pkl_path))
  train_dir = os.path.join(run_dir, f"train_{params['head']}")

  out = {}
  for key in final_keys:
    fold = key.split('_')[0]
    bm = data['results'][key]['best_model']
    sub_idx = bm['fold_sub_fold_idx'][1]
    epoch = bm['best_model_idx']
    fold_dir = os.path.join(train_dir, f'{fold}_cross_val')
    ckpt = os.path.join(fold_dir, f'{fold}_cross_val_sub_{sub_idx}', f'best_model_ep_{epoch}.pt')
    if not os.path.isfile(ckpt):
      raise FileNotFoundError(f'Checkpoint not found for {fold}: {ckpt}')

    csv_path = os.path.join(fold_dir, 'test_cleaned.csv')
    if not os.path.isfile(csv_path):
      df_test = pd.read_csv(os.path.join(fold_dir, 'test.csv'), sep='\t', dtype={'sample_name': str})
      df_test = df_test[df_test['sample_id'] <= helper.step_shift]
      df_test.to_csv(csv_path, index=False, sep='\t')

    test_args = {
      'path_model_weights': ckpt,
      'state_dict': None,
      'csv_path': csv_path,
      'criterion': cfg['criterion'],
      'is_test': True,
      'concatenate_temporal': cfg['concatenate_temp_dim'],
      'concatenate_quadrants': cfg['concatenate_quadrants'],
      'CCC_loss': cfg['CCC_loss'],
    }
    kwargs = {k: v for k, v in cfg.items() if k not in test_args.keys()}
    print(f'\n=== Re-running {fold} best model (sub_{sub_idx}, epoch {epoch}) ===')
    dict_test = model.test_pretrained_model(**test_args, **kwargs)

    history = dict_test['history_test_sample_predictions']
    preds = {sid: epochs[0] for sid, epochs in history.items()}
    df_test = pd.read_csv(csv_path, sep='\t', dtype={'sample_name': str})
    labels = dict(zip(df_test['sample_id'].astype(int), df_test['class_id'].astype(int)))
    missing = set(labels) - set(preds)
    if missing:
      raise RuntimeError(f'{fold}: {len(missing)} test samples have no logged prediction: {sorted(missing)[:5]}...')

    err = {sid: abs(preds[sid] - labels[sid]) for sid in labels}
    per_class = {}
    for c in sorted(set(labels.values())):
      sids = [sid for sid, y in labels.items() if y == c]
      per_class[c] = float(np.mean([err[sid] for sid in sids]))
    out[key] = {
      'raw_mae': float(np.mean(list(err.values()))),
      'raw_mae_per_class': per_class,
      'recomputed_l1': float(dict_test['test_l1_error']),
      'recomputed_accuracy': float(dict_test['test_accuracy']),
      'recomputed_loss_per_subject': to_numpy(dict_test['test_loss_per_subject']).astype(float),
      'recomputed_accuracy_per_subject': to_numpy(dict_test['test_accuracy_per_subject']).astype(float),
      'recomputed_subject_ids': to_numpy(dict_test['test_unique_subject_ids']),
      'n_samples': len(labels),
    }
  if hasattr(model, 'free_gpu_memory'):
    model.free_gpu_memory()
  return out


def extract_table(pkl_path: str, raw: bool = False, ffsp_override: str = None) -> pd.DataFrame:
  """
  Build a per-fold test summary table from a k_fold_results.pkl file.

  Args:
    pkl_path:      Path to k_fold_results.pkl produced by train_model.py.
    raw:           If True, re-run each fold's best checkpoint to compute per-class MAE
                   from raw per-sample predictions (same pipeline as test_l1_error).
    ffsp_override: Optional replacement for the cached-features folder path.

  Returns:
    DataFrame with one row per fold (columns: fold, test_MAE, MAE_class_<c>,
    n_class_<c>, n_samples, n_subjects; plus test_MAE_raw when raw=True)
    plus mean/std summary rows.
  """
  with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

  criterion = str(data.get('config', {}).get('criterion', ''))
  if 'L1Loss' not in criterion:
    print(f"WARNING: criterion is {criterion!r}, not L1Loss -> per-class columns are per-class loss, not MAE.")

  final_keys = sorted(
    [k for k in data['results'] if re.fullmatch(r'k\d+_cross_val_final', k)],
    key=lambda k: int(re.match(r'k(\d+)', k).group(1)),
  )
  raw_metrics = recompute_raw_fold_metrics(pkl_path, data, final_keys, ffsp_override) if raw else None

  rows = []
  for key in final_keys:
    test = data['results'][key]['test']
    fold = key.split('_')[0]
    mae = float(test['test_l1_error'])
    classes = to_numpy(test['test_unique_y']).astype(int)
    count_y = to_numpy(test['test_count_y']).astype(int)
    loss_per_class = to_numpy(test['test_loss_per_class']).astype(float)
    count_subjects = to_numpy(test['test_count_subject_ids']).astype(int)

    n_samples = int(count_y.sum())
    n_subjects = len(count_subjects)
    if n_samples != int(count_subjects.sum()):
      print(f"WARNING {fold}: sum(test_count_y)={n_samples} != sum(test_count_subject_ids)={int(count_subjects.sum())}")

    if raw_metrics is not None:
      rm = raw_metrics[key]
      if rm['n_samples'] != n_samples:
        print(f"WARNING {fold}: re-run test set has {rm['n_samples']} samples, pkl says {n_samples}")
      if abs(rm['recomputed_l1'] - mae) > 1e-6:
        print(f"WARNING {fold}: recomputed test_l1_error {rm['recomputed_l1']:.6f} != stored {mae:.6f}")
      row = {'fold': fold, 'test_MAE': mae, 'test_MAE_raw': rm['raw_mae']}
      per_class = rm['raw_mae_per_class']
    else:
      weighted = float((loss_per_class * count_y).sum() / count_y.sum())
      if abs(weighted - mae) > 1e-3:
        print(f"INFO {fold}: weighted per-class MAE {weighted:.4f} vs test_l1_error {mae:.4f} (diff {abs(weighted - mae):.4f})")
      row = {'fold': fold, 'test_MAE': mae, 'test_MAE_weighted': weighted}
      per_class = dict(zip(classes.tolist(), loss_per_class.tolist()))

    for c, n in zip(classes, count_y):
      row[f'MAE_class_{c}'] = float(per_class[int(c)])
      row[f'n_class_{c}'] = int(n)
    row['n_samples'] = n_samples
    row['n_subjects'] = n_subjects
    rows.append(row)

  df = pd.DataFrame(rows)
  extra_col = 'test_MAE_raw' if raw else 'test_MAE_weighted'
  mae_cols = ['test_MAE', extra_col] + [c for c in df.columns if c.startswith('MAE_class_')]
  mean_row = {'fold': 'mean', **df[mae_cols].mean().to_dict()}
  std_row = {'fold': 'std', **df[mae_cols].std().to_dict()}
  return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def main():
  """
  Parse CLI arguments, extract the per-fold test table and write it to CSV.

  Args:
    None (reads --pkl and --out from the command line).

  Returns:
    None. Writes the CSV and prints the table.
  """
  parser = argparse.ArgumentParser(description='Extract per-fold test metrics from k_fold_results.pkl into a CSV.')
  parser.add_argument('--pkl', required=True, help='Path to k_fold_results.pkl')
  parser.add_argument('--out', default=None, help='Output CSV path (default: test_table.csv next to the pkl)')
  parser.add_argument('--raw', action='store_true',
                      help='Re-run each fold best checkpoint to compute per-class MAE from raw per-sample predictions')
  parser.add_argument('--ffsp', default=None, help='Override features_folder_saving_path for re-inference')
  args = parser.parse_args()

  default_name = 'test_table_raw.csv' if args.raw else 'test_table.csv'
  out = args.out or os.path.join(os.path.dirname(args.pkl), default_name)
  df = extract_table(args.pkl, raw=args.raw, ffsp_override=args.ffsp)
  df.to_csv(out, index=False, float_format='%.4f')
  print(df.to_string(index=False))
  print(f"\nSaved to {out}")


if __name__ == '__main__':
  main()

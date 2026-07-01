#!/usr/bin/env python3
"""
Aggregate a folder of per-seed cross_space_projection runs into a cross-seed summary
plus before-vs-after refinement significance tests.

Intended for a "significance test" folder holding one standalone run per seed (the same
config repeated across seeds), e.g.:
  Cross_projection/significance_test/<exp>/cross_space_projection_seed<N>_..._<uid>/results_<uid>.pkl

The seed is read from each run's pkl ('seed' key, written by newer runs) and falls back
to the seed<N> token in the folder name (older runs that predate seed persistence).

Outputs, written into the aggregated folder:
  seed_summary.csv        — one row per seed (headline mae/ccc + per-mode before/after
                            refinement MAE), followed by MEAN and STD rows.
  significance_tests.csv  — one row per (refinement mode, metric): mean+/-std before/after,
                            the mean after-before delta, and paired-test p-values
                            (paired t-test + Wilcoxon signed-rank) across seeds.

The --compare mode instead takes TWO such folders (e.g. two interpolation_similarity configs
repeated across the same seeds) and tests whether their after-refinement metrics differ. Runs
are paired by seed; for every refinement mode present in both folders and every metric in
_AFTER_METRICS (srctest/newtest micro+macro after), it writes one row with type_A/type_B (each
folder's interpolation_similarity, 'linear'/'mlp'), mean+/-std per folder, the mean B-A delta, and
paired-test p-values (paired t-test + Wilcoxon). The table is written as comparison_<A>_vs_<B>.csv
into a statistical_comparison/ subfolder of BOTH compared folders.

Usage:
  python3 cross_space_aggregate_seed_runs.py --parent_folder <folder> [<folder> ...]
  python3 cross_space_aggregate_seed_runs.py --compare <folder_A> <folder_B>
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

# Refinement metrics aggregated/tested per mode. Each has a _before and _after variant in
# every refinement block: 'old_oncsv' is the projected source split (refinement target),
# 'new_test' is the new model's own test split (preservation check).
_REFINE_METRICS = (
  'mae_micro_old_oncsv',
  'mae_macro_old_oncsv',
  'mae_micro_new_test',
  'mae_macro_new_test',
)

# After-refinement metrics compared across two folders (see compare_folders). Each is the
# post-refinement value: 'srctest_*' is the projected source split (the refinement target),
# 'newtest_*' is the new model's own TEST split (recomputed from per-sample preds, matching
# summary.csv's newtest_mae_*_after rather than the stored val-split scalars).
_AFTER_METRICS = (
  'srctest_mae_micro_after',
  'srctest_mae_macro_after',
  'newtest_mae_micro_after',
  'newtest_mae_macro_after',
)


def _parse_seed_from_name(folder_name):
  """
  Extract the integer seed from a run folder name's `seed<N>` token.

  Args:
    folder_name (str): Basename of a run folder, e.g.
      'cross_space_projection_seed42_K250_..._<uid>'.

  Returns:
    int | None: The parsed seed, or None when no `seed<N>` token is present.
  """
  m = re.search(r'_seed(\d+)_', folder_name)
  return int(m.group(1)) if m else None


def _normalize_refinements(data):
  """
  Return a {mode: refinement_block} mapping regardless of the pkl's refinement schema.

  --refinement 3 runs store a 'refinements' dict keyed by mode; --refinement 1/2 runs
  store a single 'refinement' block (its mode taken from refine_mode). Runs without
  refinement yield an empty mapping.

  Args:
    data (dict): Loaded results pkl contents.

  Returns:
    dict[str, dict]: mode name → that mode's refinement block.
  """
  if isinstance(data.get('refinements'), dict):
    return data['refinements']
  block = data.get('refinement')
  if isinstance(block, dict):
    return {block.get('refine_mode', 'refinement'): block}
  return {}


def _collect_run_record(pkl_path, seed):
  """
  Build one flat per-seed record from a single run's results pkl.

  Args:
    pkl_path (str): Path to a results_*.pkl (or results.pkl) file.
    seed     (int): The seed attributed to this run.

  Returns:
    dict: Keys 'seed', 'mae', 'ccc', and for every refinement mode/metric the flat
      columns '<mode>__<metric>_before' / '<mode>__<metric>_after' (missing values
      become NaN).
  """
  with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
  m = data.get('metrics') or {}
  row = {
    'seed': seed,
    'mae':  float(m.get('mae', np.nan)),
    'ccc':  float(m.get('ccc', np.nan)),
  }
  for mode, block in _normalize_refinements(data).items():
    for metric in _REFINE_METRICS:
      for when in ('before', 'after'):
        key = f'{metric}_{when}'
        val = block.get(key)
        row[f'{mode}__{key}'] = float(val) if val is not None else np.nan
  return row


def _paired_significance(before, after):
  """
  Run paired before-vs-after significance tests across seeds for one metric.

  Args:
    before (np.ndarray): Per-seed before values, shape (n,).
    after  (np.ndarray): Per-seed after values, shape (n,), paired with `before`.

  Returns:
    dict: n, mean/std for before and after, the mean/std of the (after - before) delta,
      and paired-test p-values (ttest_p from scipy.stats.ttest_rel, wilcoxon_p from
      scipy.stats.wilcoxon). p-values are NaN when a test is undefined (e.g. all deltas
      zero, or n < 1).
  """
  before = np.asarray(before, dtype=np.float64)
  after  = np.asarray(after,  dtype=np.float64)
  mask = np.isfinite(before) & np.isfinite(after)
  before, after = before[mask], after[mask]
  delta = after - before
  n = int(before.size)

  ttest_p = wilcoxon_p = np.nan
  if n >= 2:
    try:
      ttest_p = float(stats.ttest_rel(after, before).pvalue)
    except Exception:
      ttest_p = np.nan
    if np.any(delta != 0):
      try:
        wilcoxon_p = float(stats.wilcoxon(after, before).pvalue)
      except Exception:
        wilcoxon_p = np.nan
  return {
    'n':           n,
    'mean_before': float(np.mean(before)) if n else np.nan,
    'std_before':  float(np.std(before, ddof=1)) if n >= 2 else np.nan,
    'mean_after':  float(np.mean(after)) if n else np.nan,
    'std_after':   float(np.std(after, ddof=1)) if n >= 2 else np.nan,
    'mean_delta':  float(np.mean(delta)) if n else np.nan,   # after - before; <0 = improvement (MAE)
    'std_delta':   float(np.std(delta, ddof=1)) if n >= 2 else np.nan,
    'ttest_p':     ttest_p,
    'wilcoxon_p':  wilcoxon_p,
  }


def _micro_macro_mae(preds, labels):
  """
  Compute micro- and macro-averaged MAE (local replica of
  cross_space_logs._compute_global_mae, kept here to avoid importing that heavy module).

  Args:
    preds  (array-like): Float predictions, shape (N,) or (N, 1).
    labels (array-like): Float ground-truth labels, shape (N,).

  Returns:
    tuple[float, float]: (micro_mae, macro_mae). micro = mean |pred-label| over all samples
      (each sample equal weight); macro = mean of per-class MAEs over round(labels) classes
      (each class equal weight). Both NaN when no/ mismatched samples are given.
  """
  preds  = np.asarray(preds,  dtype=np.float64).reshape(-1)
  labels = np.asarray(labels, dtype=np.float64).reshape(-1)
  if preds.size == 0 or preds.size != labels.size:
    return np.nan, np.nan
  micro = float(np.mean(np.abs(preds - labels)))
  labels_int = np.round(labels).astype(int)
  per_class = [float(np.mean(np.abs(preds[labels_int == c] - labels[labels_int == c])))
               for c in np.unique(labels_int)]
  macro = float(np.mean(per_class)) if per_class else np.nan
  return micro, macro


def _after_metrics(block):
  """
  Extract the four after-refinement metrics (see _AFTER_METRICS) from one refinement block.

  srctest values are read straight from the block. The newtest values are recomputed on the
  new model's own TEST split from the per-sample predictions in block['new_test_eval']
  (matching summary.csv's newtest_mae_{micro,macro}_after), falling back to the stored
  val-split mae_{micro,macro}_new_test_after scalars when new_test_eval is unavailable
  (older pkls).

  Args:
    block (dict): A refinement block (pkl 'refinement', or one mode of 'refinements').

  Returns:
    dict[str, float]: One entry per name in _AFTER_METRICS; missing values become NaN.
  """
  def _f(v):
    return float(v) if v is not None else np.nan
  out = {
    'srctest_mae_micro_after': _f(block.get('mae_micro_old_oncsv_after')),
    'srctest_mae_macro_after': _f(block.get('mae_macro_old_oncsv_after')),
  }
  nte = block.get('new_test_eval') or {}
  preds, labels = nte.get('preds_after'), nte.get('labels')
  if preds is not None and labels is not None:
    micro, macro = _micro_macro_mae(preds, labels)
  else:
    micro, macro = _f(block.get('mae_micro_new_test_after')), _f(block.get('mae_macro_new_test_after'))
  out['newtest_mae_micro_after'] = micro
  out['newtest_mae_macro_after'] = macro
  return out


def _collect_after_record(pkl_path, seed):
  """
  Build one flat per-seed record of after-refinement metrics from a single run's results pkl.

  Args:
    pkl_path (str): Path to a results_*.pkl (or results.pkl) file.
    seed     (int): The seed attributed to this run.

  Returns:
    dict: 'seed', 'interpolation_similarity' (the run's projector type, 'linear'/'mlp', read
      from config_cross_space_projection; None when absent), plus a flat '<mode>__<metric>'
      column for every refinement mode and every name in _AFTER_METRICS.
  """
  with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
  cfg = data.get('config_cross_space_projection') or {}
  row = {'seed': seed, 'interpolation_similarity': cfg.get('interpolation_similarity')}
  for mode, block in _normalize_refinements(data).items():
    for name, val in _after_metrics(block).items():
      row[f'{mode}__{name}'] = val
  return row


def _find_run_pkls(folder):
  """
  Find every per-seed run pkl directly under a folder of standalone seed runs.

  Matches both the standalone (`results_*.pkl`) and grid (`results.pkl`) filenames so the
  same folder layout works for either; one pkl per immediate seed<N> sub-folder.

  Args:
    folder (str): Folder holding the per-seed run sub-folders.

  Returns:
    list[tuple[str, int]]: (pkl_path, seed) pairs, sorted by seed, skipping sub-folders
      whose seed cannot be parsed or that contain no results pkl.
  """
  out = []
  for sub in sorted(glob.glob(os.path.join(folder, '*'))):
    if not os.path.isdir(sub):
      continue
    seed = _parse_seed_from_name(os.path.basename(sub))
    if seed is None:
      continue
    pkls = sorted(glob.glob(os.path.join(sub, 'results_*.pkl'))) \
        or sorted(glob.glob(os.path.join(sub, 'results.pkl')))
    if pkls:
      out.append((pkls[0], seed))
  return sorted(out, key=lambda t: t[1])


def aggregate_folder(folder):
  """
  Aggregate one folder of per-seed runs and write seed_summary.csv + significance_tests.csv.

  Args:
    folder (str): Folder holding the per-seed run sub-folders.

  Returns:
    None.
  """
  runs = _find_run_pkls(folder)
  if not runs:
    print(f'[aggregate] No per-seed runs found under {folder}')
    return
  rows = []
  for pkl_path, seed in runs:
    try:
      rows.append(_collect_run_record(pkl_path, seed))
    except Exception as exc:
      print(f'[WARN] skipping {pkl_path}: {exc}')
  if not rows:
    print(f'[aggregate] No readable runs under {folder}')
    return

  df = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
  seeds = df['seed'].tolist()
  print(f'[aggregate] {folder}\n  {len(seeds)} seeds: {seeds}')

  # --- seed_summary.csv: per-seed rows + MEAN / STD rows over the numeric columns ---
  num_cols = [c for c in df.columns if c != 'seed']
  mean_row = {'seed': 'MEAN', **df[num_cols].mean().to_dict()}
  std_row  = {'seed': 'STD',  **df[num_cols].std(ddof=1).to_dict()}
  summary = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
  summary_path = os.path.join(folder, 'seed_summary.csv')
  summary.to_csv(summary_path, index=False)

  # --- significance_tests.csv: paired before-vs-after per (mode, metric) ---
  modes = sorted({c.split('__')[0] for c in df.columns if '__' in c})
  sig_rows = []
  for mode in modes:
    for metric in _REFINE_METRICS:
      bcol, acol = f'{mode}__{metric}_before', f'{mode}__{metric}_after'
      if bcol not in df or acol not in df:
        continue
      sig_rows.append({
        'mode': mode, 'metric': metric,
        **_paired_significance(df[bcol].to_numpy(), df[acol].to_numpy()),
      })
  sig_df = pd.DataFrame(sig_rows)
  sig_path = os.path.join(folder, 'significance_tests.csv')
  sig_df.to_csv(sig_path, index=False)

  # --- console headline ---
  print(f'  headline MAE: {df["mae"].mean():.4f} +/- {df["mae"].std(ddof=1):.4f}'
        f'   CCC: {df["ccc"].mean():.4f} +/- {df["ccc"].std(ddof=1):.4f}')
  if not sig_df.empty:
    with pd.option_context('display.width', 200, 'display.max_columns', 20,
                           'display.float_format', lambda v: f'{v:.4g}'):
      print(sig_df.to_string(index=False))
  print(f'  wrote: {summary_path}\n         {sig_path}')


def _folder_after_df(folder):
  """
  Build a per-seed DataFrame of after-refinement metrics for one folder of standalone seed runs.

  Args:
    folder (str): Folder holding the per-seed cross_space_projection_seed<N>_... sub-folders.

  Returns:
    pd.DataFrame | None: One row per seed with a 'seed' column and '<mode>__<metric>' columns,
      sorted by seed; None when no readable runs are found. Duplicate seeds (unexpected) are
      reduced to the first occurrence with a warning.
  """
  runs = _find_run_pkls(folder)
  if not runs:
    print(f'[compare] No per-seed runs found under {folder}')
    return None
  rows = []
  for pkl_path, seed in runs:
    try:
      rows.append(_collect_after_record(pkl_path, seed))
    except Exception as exc:
      print(f'[WARN] skipping {pkl_path}: {exc}')
  if not rows:
    print(f'[compare] No readable runs under {folder}')
    return None
  df = pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
  if df['seed'].duplicated().any():
    dups = sorted(set(df.loc[df['seed'].duplicated(keep=False), 'seed'].tolist()))
    print(f'[WARN] {folder}: duplicate seeds {dups}; keeping first run per seed')
    df = df.drop_duplicates('seed', keep='first').reset_index(drop=True)
  return df


def compare_folders(folder_a, folder_b):
  """
  Compare two folders of per-seed runs and write a paired-significance table (CSV + console).

  Runs are paired by seed. For every refinement mode present in BOTH folders and every metric
  in _AFTER_METRICS, a paired t-test and Wilcoxon signed-rank test are run with folder A as the
  baseline and folder B as the comparison, so mean_delta_B_minus_A < 0 means folder B has the
  lower MAE (the better config). Tests reuse _paired_significance(before=A, after=B).

  Each row also carries 'type_A'/'type_B': the folder's interpolation_similarity ('linear'/'mlp',
  read from config_cross_space_projection), so it is clear which config A and B are without
  decoding the folder names.

  The resulting table is written as comparison_<A>_vs_<B>.csv (from each folder's basename) into
  a 'statistical_comparison' subfolder of BOTH folders, so each keeps a copy of any comparison it
  participated in.

  Args:
    folder_a (str): Baseline folder (the "A" config, e.g. one interpolation_similarity).
    folder_b (str): Comparison folder (the "B" config, the other interpolation_similarity).

  Returns:
    None.
  """
  def _folder_type(df):
    """Return the folder's single interpolation_similarity value (first non-null), or NaN."""
    vals = df['interpolation_similarity'].dropna() if 'interpolation_similarity' in df else []
    return vals.iloc[0] if len(vals) else np.nan

  df_a, df_b = _folder_after_df(folder_a), _folder_after_df(folder_b)
  if df_a is None or df_b is None:
    return
  type_a, type_b = _folder_type(df_a), _folder_type(df_b)

  modes_a = {c.split('__')[0] for c in df_a.columns if '__' in c}
  modes_b = {c.split('__')[0] for c in df_b.columns if '__' in c}
  common_modes = sorted(modes_a & modes_b)
  print(f'[compare] A: {folder_a}\n          B: {folder_b}')
  print(f'  A seeds: {df_a["seed"].tolist()}\n  B seeds: {df_b["seed"].tolist()}')
  if not common_modes:
    print(f'[compare] No refinement mode shared by both folders '
          f'(A modes: {sorted(modes_a)}; B modes: {sorted(modes_b)}). Nothing to compare.')
    return

  sa, sb = df_a.set_index('seed'), df_b.set_index('seed')
  common_seeds = sorted(set(sa.index) & set(sb.index))
  only_a, only_b = sorted(set(sa.index) - set(sb.index)), sorted(set(sb.index) - set(sa.index))
  if only_a or only_b:
    print(f'[WARN] seed sets differ; pairing on {len(common_seeds)} common seeds '
          f'(A-only: {only_a}, B-only: {only_b})')
  if len(common_seeds) < 2:
    print(f'[compare] Only {len(common_seeds)} common seed(s); paired tests need >=2 '
          f'(p-values will be NaN).')

  rows = []
  for mode in common_modes:
    for metric in _AFTER_METRICS:
      col = f'{mode}__{metric}'
      if col not in df_a.columns or col not in df_b.columns:
        continue
      a_vals = sa.loc[common_seeds, col].to_numpy()
      b_vals = sb.loc[common_seeds, col].to_numpy()
      s = _paired_significance(a_vals, b_vals)  # before=A, after=B -> delta = B - A
      rows.append({
        'mode':                 mode,
        'metric':               metric,
        'type_A':               type_a,
        'type_B':               type_b,
        'n':                    s['n'],
        'mean_A':               s['mean_before'],
        'std_A':                s['std_before'],
        'mean_B':               s['mean_after'],
        'std_B':                s['std_after'],
        'mean_delta_B_minus_A': s['mean_delta'],   # <0 = B lower MAE = B better
        'std_delta':            s['std_delta'],
        'ttest_p':              s['ttest_p'],
        'wilcoxon_p':           s['wilcoxon_p'],
      })

  cmp_df = pd.DataFrame(rows)
  name_a = os.path.basename(os.path.normpath(folder_a))
  name_b = os.path.basename(os.path.normpath(folder_b))
  filename = f'comparison_{name_a}_vs_{name_b}.csv'
  out_paths = []
  for folder in (folder_a, folder_b):
    dest_dir = os.path.join(folder, 'statistical_comparison')
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, filename)
    cmp_df.to_csv(out_path, index=False)
    out_paths.append(out_path)
  if not cmp_df.empty:
    with pd.option_context('display.width', 200, 'display.max_columns', 20,
                           'display.float_format', lambda v: f'{v:.4g}'):
      print(cmp_df.to_string(index=False))
  print('  wrote: ' + '\n         '.join(out_paths))


def main(argv=None):
  """
  CLI entry point: either aggregate each --parent_folder, or --compare two folders.

  Args:
    argv (list[str] | None): Argument list to parse (defaults to sys.argv[1:]).

  Returns:
    int: Process exit code (0 on success; argparse exits 2 on a usage error).
  """
  parser = argparse.ArgumentParser(
    description='Aggregate per-seed cross_space_projection runs into a cross-seed '
                'summary plus before-vs-after refinement significance tests, or compare '
                'two folders of per-seed runs (e.g. two interpolation_similarity configs) '
                'with paired across-seed significance tests.'
  )
  mode = parser.add_mutually_exclusive_group(required=True)
  mode.add_argument(
    '--parent_folder', type=str, nargs='+',
    help='Experiment folder(s) holding the per-seed '
         'cross_space_projection_seed<N>_... run sub-folders. Each is aggregated '
         'in place, writing seed_summary.csv + significance_tests.csv into it.',
  )
  mode.add_argument(
    '--compare', type=str, nargs=2, metavar=('FOLDER_A', 'FOLDER_B'),
    help='Two folders of per-seed runs to compare (A=baseline, B=comparison). Runs are '
         'paired by seed and tested per shared refinement mode / after-refinement metric '
         '(paired t-test + Wilcoxon). The result is written as '
         'comparison_<A>_vs_<B>.csv into a statistical_comparison/ subfolder of BOTH folders.',
  )
  args = parser.parse_args(argv)
  if args.compare:
    compare_folders(args.compare[0], args.compare[1])
  else:
    for folder in args.parent_folder:
      aggregate_folder(folder)
  return 0


if __name__ == '__main__':
  sys.exit(main())

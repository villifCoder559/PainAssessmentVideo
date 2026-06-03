#!/usr/bin/env python3
"""
Two-stage grid search orchestrator for cross_space_projection.py.

Stage 1: run a broad grid over (num_anchors, anchor_selection_type,
         interpolation_similarity, weighting_method, rbf_sigma) with
         csv_anchor_selection=train and old_model_csv=val.
Stage 2: pick the top-5 trials by val MAE (ascending) and re-run them with
         old_model_csv=test, csv_anchor_selection=train.

Stage outputs are routed into separate subfolders by passing out_root to
run_optuna / cross_space_projection, so the working directory never changes and
relative model paths keep resolving against the project root.
"""
import argparse
import itertools
import os
import pickle
import time
from collections import Counter

import pandas as pd

import cross_space_projection as csp
from cross_space_logs import generate_logs, generate_logs_search


_INTERP_ANCHOR_WEIGHTED = ('cos', 'l1', 'l2', 'l_inf')
_INTERP_CLOSED_FORM     = ('linear', 'mlp', 'procrustes', 'linear_close')
_ANCHOR_FREE_SIZES      = (0, -1)


def _projector_key(interp, mlp_activation):
  """
  Cache/dedup key for a projector trial — mirrors
  cross_space_projection._projector_key so 'mlp' activations stay distinct while
  'linear'/'procrustes'/'linear_close' collapse across the (irrelevant) activation axis.

  Args:
    interp         (str): interpolation_similarity value.
    mlp_activation (str): Activation name; only consulted when interp == 'mlp'.

  Returns:
    str: 'mlp_<activation>' when interp == 'mlp', else interp unchanged.
  """
  return f'mlp_{mlp_activation}' if interp == 'mlp' else interp


def _parse_args():
  """
  Parse orchestrator CLI flags. Mirror the grid-axis flag names of
  cross_space_projection.py so existing invocations transfer verbatim.

  Returns:
    argparse.Namespace: Parsed args. List axes carry one or more sweep values.
  """
  p = argparse.ArgumentParser(
    description='Two-stage cross-space projection grid search (val sweep → top-5 test re-run).'
  )
  p.add_argument('--new_model_pth', type=str, required=True)
  p.add_argument('--old_model_pth', type=str, required=True)
  p.add_argument('--num_anchors', type=int, nargs='+', required=True)
  p.add_argument('--anchor_selection_type', type=str, nargs='+', default=['random'],
                 choices=['random', 'balance_class_random',
                          'balance_subject_random', 'balance_class_subject'])
  p.add_argument('--interpolation_similarity', type=str, nargs='+', default=['cos'],
                 choices=['cos', 'l1', 'l2', 'l_inf', 'linear', 'mlp', 'procrustes', 'linear_close'])
  p.add_argument('--mlp_activation', type=str, nargs='+', default=['gelu'],
                 choices=['gelu', 'relu', 'silu', 'leaky_relu'],
                 help="Activation(s) for the 'mlp' projector; sweepable, ignored for non-mlp.")
  p.add_argument('--weighting_method', type=str, nargs='+', default=['rbf'],
                 choices=['rbf', 'none'])
  p.add_argument('--rbf_sigma', type=float, nargs='+', default=[1.0])
  p.add_argument('--top_k', type=int, default=5,
                 help='Number of top val trials to re-run on test (default: 5).')
  p.add_argument('--run_tag', type=str, default=None,
                 help='Optional label embedded in the parent output directory.')
  p.add_argument('--refinement', type=int, choices=[0, 1],
                 default=int(csp.REFINEMENT_CONFIG['enabled']),
                 help='Enable (1) / disable (0) the post-projection refinement stage (see '
                      'cross_space_projection.REFINEMENT_CONFIG). Default off. Propagates to both '
                      'Stage-1 (run_optuna) and Stage-2 (cross_space_projection) runs.')
  return p.parse_args()


def _make_parent_dir(run_tag, uid):
  """
  Build the top-level orchestrator output directory.

  Args:
    run_tag (str | None): Optional label included in the folder name.
    uid     (int):        Unique timestamp suffix.

  Returns:
    str: Absolute path to the freshly created parent dir.
  """
  tag = run_tag if run_tag else 'two_stage'
  parent = os.path.join(os.getcwd(), 'Cross_projection_two_stage', f'{tag}_{uid}')
  os.makedirs(parent, exist_ok=True)
  return parent


def _classify_combo(combo):
  """
  Validate a single raw grid combo against the downstream rule table.

  Args:
    combo (dict): One cartesian-product point with the seven sweep axes.

  Returns:
    tuple[str, str]: ('valid', '') or ('invalid', human-readable reason).
  """
  interp = combo['interpolation_similarity']
  wm     = combo['weighting_method']
  na     = combo['num_anchors']
  if interp in _INTERP_ANCHOR_WEIGHTED and wm == 'none':
    return 'invalid', f"interp={interp!r} requires weighting_method='rbf'"
  if interp in _INTERP_CLOSED_FORM and wm != 'none':
    return 'invalid', f"interp={interp!r} requires weighting_method='none'"
  if interp in _INTERP_CLOSED_FORM and na in _ANCHOR_FREE_SIZES:
    return 'invalid', f"interp={interp!r} requires num_anchors > 0"
  return 'valid', ''


def _effective_config(combo):
  """
  Collapse axes that the downstream trial would ignore into a single dedup key.

  Mirrors the three caches in cross_space_projection.run_optuna
  (_zero_anchor_mae_cache, _neg_one_anchor_mae_cache, _closed_form_mae_cache),
  so two combos sharing this key are guaranteed to produce identical MAE.

  Args:
    combo (dict): One raw cartesian-product point with the seven sweep axes.

  Returns:
    tuple: Hashable effective key. First element is a regime tag
      ('NA' = anchor-free, 'CF' = closed-form, 'AW' = anchor-weighted).
  """
  na     = combo['num_anchors']
  interp = combo['interpolation_similarity']
  if na in _ANCHOR_FREE_SIZES:
    return ('NA', na, combo['old_model_csv'])
  if interp in _INTERP_CLOSED_FORM:
    # mlp_activation only matters for 'mlp'; _projector_key collapses it otherwise.
    return ('CF', combo['csv_anchor_selection'], na,
            combo['anchor_selection_type'], combo['old_model_csv'],
            _projector_key(interp, combo.get('mlp_activation')))
  return ('AW', combo['csv_anchor_selection'], na,
          combo['anchor_selection_type'], combo['old_model_csv'],
          interp, combo['weighting_method'], combo['rbf_sigma'])


def _expand_effective_grid(args):
  """
  Enumerate the Stage-1 cartesian product and classify every raw combo.

  Stage 1 always uses csv_anchor_selection=['train'] and old_model_csv=['val'],
  so those axes are fixed here regardless of the orchestrator args.

  Args:
    args (argparse.Namespace): Orchestrator args (sweep axes only).

  Returns:
    tuple[list[dict], list[dict], list[tuple[dict, str]]]:
      (valid, redundant, invalid). 'valid' is one combo per effective key
      (first-seen wins); 'redundant' is the rest that share an existing key;
      'invalid' carries the offending combo and its human-readable reason.
  """
  axes = [
    ('num_anchors',              args.num_anchors),
    ('anchor_selection_type',    args.anchor_selection_type),
    ('csv_anchor_selection',     ['train']),
    ('old_model_csv',            ['val']),
    ('interpolation_similarity', args.interpolation_similarity),
    ('mlp_activation',           args.mlp_activation),
    ('weighting_method',         args.weighting_method),
    ('rbf_sigma',                args.rbf_sigma),
  ]
  keys = [k for k, _ in axes]
  valid, redundant, invalid = [], [], []
  seen = set()
  for prod in itertools.product(*[v for _, v in axes]):
    combo = dict(zip(keys, prod))
    status, reason = _classify_combo(combo)
    if status == 'invalid':
      invalid.append((combo, reason))
      continue
    eff = _effective_config(combo)
    if eff in seen:
      redundant.append(combo)
      continue
    seen.add(eff)
    valid.append(combo)
  return valid, redundant, invalid


def _dry_run_report(valid, redundant, invalid):
  """
  Print a pre-flight breakdown of the expanded grid before any GPU work.

  Args:
    valid     (list[dict]):                 First-seen-per-effective-key combos.
    redundant (list[dict]):                 Cache-hit duplicates.
    invalid   (list[tuple[dict, str]]):     Pruned combos with reasons.
  """
  total = len(valid) + len(redundant) + len(invalid)
  print(f'[grid_two_stage] Dry-run: {total} grid points → '
        f'{len(valid)} valid, {len(redundant)} redundant, {len(invalid)} invalid')
  if invalid:
    for reason, n in Counter(r for _, r in invalid).most_common():
      print(f'  [invalid x{n}] {reason}')


def _axis_union(valid, key):
  """
  Collect the unique values of one axis across all valid combos, order-preserving.

  Args:
    valid (list[dict]): Output of _expand_effective_grid.
    key   (str):        Axis name.

  Returns:
    list: Distinct values in first-seen order. Empty if no valid combos.
  """
  seen, out = set(), []
  for c in valid:
    v = c[key]
    if v not in seen:
      seen.add(v)
      out.append(v)
  return out


def _run_stage_val(args, stage_val_dir, valid):
  """
  Launch the Stage-1 broad sweep via cross_space_projection.run_optuna.

  The output root is routed under stage_val_dir by passing out_root, so the
  working directory stays put and relative model paths keep resolving.

  Per-axis lists are reduced to the union of values actually used by `valid`
  combos. The cartesian product run_optuna iterates over may still contain a
  few invalid or cache-hit points; its per-trial pruning at
  cross_space_projection.py:1787-1813 handles those cleanly (no pkl written).

  Args:
    args          (argparse.Namespace): Orchestrator args.
    stage_val_dir (str):                Stage-1 subfolder under the parent.
    valid         (list[dict]):         Pre-pruned effective-config list from
                                        _expand_effective_grid.

  Returns:
    str: Absolute path to the search root (the dir containing per-trial
         subfolders and the upcoming summary.csv).
  """
  ns = argparse.Namespace(
    new_model_pth            = args.new_model_pth,
    old_model_pth            = args.old_model_pth,
    num_anchors              = _axis_union(valid, 'num_anchors'),
    anchor_selection_type    = _axis_union(valid, 'anchor_selection_type'),
    csv_anchor_selection     = ['train'],
    old_model_csv            = ['val'],
    interpolation_similarity = _axis_union(valid, 'interpolation_similarity'),
    mlp_activation           = _axis_union(valid, 'mlp_activation'),
    weighting_method         = _axis_union(valid, 'weighting_method'),
    rbf_sigma                = _axis_union(valid, 'rbf_sigma'),
    n_trials                 = None,
    optuna_sampler           = 'grid',
    run_tag                  = (args.run_tag or 'stage_val'),
  )
  print(f'[grid_two_stage] Stage 1: broad sweep on val → {stage_val_dir}')
  print(f'  axis unions: num_anchors={ns.num_anchors}, '
        f'anchor_selection_type={ns.anchor_selection_type}, '
        f'interp={ns.interpolation_similarity}, '
        f'weighting={ns.weighting_method}, rbf_sigma={ns.rbf_sigma}')
  return csp.run_optuna(ns, out_root=stage_val_dir)


def _pick_top_k(summary_csv, k):
  """
  Read a Stage-1 summary.csv and return the K rows with the lowest MAE,
  deduplicated by effective config so Stage 2 never re-runs identical work.

  Args:
    summary_csv (str): Path to summary.csv written by generate_logs_search.
    k           (int): Number of distinct effective configs to keep.

  Returns:
    pd.DataFrame: K rows sorted by MAE ascending, columns trimmed to the five
      swept hyperparameter axes plus mae and ccc.
  """
  df = pd.read_csv(summary_csv)
  df = df.sort_values('mae', ascending=True, kind='mergesort').reset_index(drop=True)
  df['_effective_key'] = df.apply(lambda r: _effective_config({
    'num_anchors':              int(r['num_anchors']),
    'anchor_selection_type':    str(r['anchor_selection_type']),
    'csv_anchor_selection':     str(r['csv_anchor_selection']),
    'old_model_csv':            str(r['old_model_csv']),
    'interpolation_similarity': str(r['interpolation_similarity']),
    'mlp_activation':           str(r.get('mlp_activation', 'gelu')),
    'weighting_method':         str(r['weighting_method']),
    'rbf_sigma':                float(r['rbf_sigma']),
  }), axis=1)
  df = df.drop_duplicates(subset='_effective_key', keep='first').head(k).reset_index(drop=True)
  return df[['num_anchors', 'anchor_selection_type', 'interpolation_similarity',
             'mlp_activation', 'weighting_method', 'rbf_sigma', 'mae', 'ccc']]


def _run_stage_test(args, top_k_df, stage_test_dir):
  """
  Re-run each top-K Stage-1 trial on the test split via cross_space_projection.

  Args:
    args           (argparse.Namespace): Orchestrator args (for model paths
                                          and run_tag).
    top_k_df       (pd.DataFrame):       Output of _pick_top_k.
    stage_test_dir (str):                Stage-2 subfolder under the parent.

  Returns:
    list[str]: Paths to the per-trial results pkls written by Stage 2.
  """
  pkls = []
  base_tag = args.run_tag or 'stage_test'
  print(f'[grid_two_stage] Stage 2: top-{len(top_k_df)} test re-runs → {stage_test_dir}')
  for i, row in top_k_df.iterrows():
    ns = argparse.Namespace(
      new_model_pth            = args.new_model_pth,
      old_model_pth            = args.old_model_pth,
      num_anchors              = int(row['num_anchors']),
      anchor_selection_type    = str(row['anchor_selection_type']),
      csv_anchor_selection     = 'train',
      old_model_csv            = 'test',
      interpolation_similarity = str(row['interpolation_similarity']),
      mlp_activation           = str(row['mlp_activation']),
      weighting_method         = str(row['weighting_method']),
      rbf_sigma                = float(row['rbf_sigma']),
      run_tag                  = f'{base_tag}_top{i + 1}',
    )
    print(f'  [test {i + 1}/{len(top_k_df)}] params={dict(row)}')
    pkls.append(csp.cross_space_projection(ns, out_root=stage_test_dir))
  return pkls


def _write_summary_test(test_pkls, out_csv):
  """
  Aggregate Stage-2 results pkls into a single summary CSV.

  Each pkl stores 'metrics' (mae, ccc) and 'config_cross_space_projection'
  (full scalar param set). One row per pkl. Stage-2 pkls don't carry
  'trial_params', so we cannot reuse cross_space_logs._collect_summary_row.

  Args:
    test_pkls (list[str]): Stage-2 result pkl paths.
    out_csv   (str):       Destination CSV path.
  """
  rows = []
  for rank, pkl in enumerate(test_pkls, start=1):
    with open(pkl, 'rb') as f:
      d = pickle.load(f)
    cfg = d['config_cross_space_projection']
    m   = d['metrics']
    rows.append({
      'val_rank':                rank,
      'num_anchors':             cfg['num_anchors'],
      'anchor_selection_type':   cfg['anchor_selection_type'],
      'csv_anchor_selection':    cfg['csv_anchor_selection'],
      'old_model_csv':           cfg['old_model_csv'],
      'interpolation_similarity': cfg['interpolation_similarity'],
      'mlp_activation':          cfg.get('mlp_activation'),
      'weighting_method':        cfg['weighting_method'],
      'rbf_sigma':               cfg['rbf_sigma'],
      'mae':                     m['mae'],
      'ccc':                     m['ccc'],
      'pkl_path':                pkl,
    })
  pd.DataFrame(rows).to_csv(out_csv, index=False)
  print(f'[grid_two_stage] Wrote test summary: {out_csv}')


def main():
  """
  Orchestrate the two-stage grid search end-to-end.
  """
  args = _parse_args()
  # Propagate the refinement toggle to the shared pipeline global read by both stages.
  csp.REFINEMENT_CONFIG['enabled'] = bool(args.refinement)

  valid, redundant, invalid = _expand_effective_grid(args)
  _dry_run_report(valid, redundant, invalid)
  if not valid:
    raise RuntimeError(
      f'No valid trials in sweep ({len(invalid)} invalid, {len(redundant)} redundant). '
      f'Check axis compatibility: '
      f'num_anchors={args.num_anchors}, '
      f'interpolation_similarity={args.interpolation_similarity}, '
      f'weighting_method={args.weighting_method}, '
      f'rbf_sigma={args.rbf_sigma}.'
    )

  uid    = int(time.time())
  parent = _make_parent_dir(args.run_tag, uid)
  print(f'[grid_two_stage] Parent dir: {parent}')

  stage_val_dir  = os.path.join(parent, 'stage_val')
  stage_test_dir = os.path.join(parent, 'stage_test')
  os.makedirs(stage_val_dir,  exist_ok=True)
  os.makedirs(stage_test_dir, exist_ok=True)

  # --- Stage 1: broad sweep on val ---
  sweep_root = _run_stage_val(args, stage_val_dir, valid)
  print(f'[grid_two_stage] Stage 1 root: {sweep_root}')

  # Build summary.csv + top-K plots (only the top-K trials get plotted).
  generate_logs_search(sweep_root, plot_only_top_k=args.top_k)

  summary_csv = os.path.join(sweep_root, 'summary.csv')
  if not os.path.isfile(summary_csv):
    raise RuntimeError(
      f'Stage 1 produced no summary.csv at {summary_csv}. '
      f'Dry-run expected {len(valid)} valid trials; likely all failed at runtime. '
      f'Inspect Stage 1 trial dirs under {sweep_root}.'
    )

  top_k_df = _pick_top_k(summary_csv, args.top_k)
  top_k_csv = os.path.join(parent, f'top{args.top_k}_val.csv')
  top_k_df.to_csv(top_k_csv, index=False)
  print(f'[grid_two_stage] Top-{args.top_k} val trials → {top_k_csv}')

  # --- Stage 2: test re-run for each top-K tuple ---
  test_pkls = _run_stage_test(args, top_k_df, stage_test_dir)
  for pkl in test_pkls:
    generate_logs(pkl)

  _write_summary_test(test_pkls, os.path.join(parent, 'summary_test.csv'))
  print(f'[grid_two_stage] Done. Two-stage grid in: {parent}')


if __name__ == '__main__':
  main()

#!/usr/bin/env python3
"""
Diagnostic plots for cross_space_projection.py outputs.

Loads a pkl file produced by cross_space_projection.py and writes all plots
to {out_dir}/logs/.

Plot filenames encode the pipeline stage (old / projected / refined) and, for comparisons,
the two stages being differenced (e.g. _projected_vs_old, _refined_vs_projected).

Stage semantics (important for refinement runs): 'projected' always means *after
projection, before refinement*. Because cross_space_projection's report_after_refinement
overwrites new_model_tensors with the after-refinement preds/embeddings on a single-mode
run, the projected stage is recomputed here from the refinement checkpoints (via
_refinement_predictions / _projected_before_refinement_embeddings) so the projected plots
are never silently the refined ones. Refinement runs additionally emit '_refined' variants
of the prediction-level plots (predictions_histogram_refined_vs_old,
prediction_scatter_refined_vs_old, prediction_by_class_boxplot_refined_vs_old,
mae_per_subject_refined), of embedding_norm_cosine_per_class (_refined_vs_old), and of the
embedding-reconstruction outputs (stage tag _projected / _refined in their filenames).
The reported headline MAE/CCC (console + summary.csv) stay the after-refinement numbers.

Plots generated:
  1.  predictions_histogram_projected_vs_old.png — prediction distributions (bin 0.1) + ground truth
  2a. mae_per_class_old_bar.png       — bar: old model MAE per pain class
  2b. mae_per_class_old_box.png       — box: old model per-sample error per class
  2c. mae_per_class_projected_bar.png — bar: projected MAE per pain class
  2d. mae_per_class_projected_box.png — box: projected per-sample error per class
  2e. mae_per_class_refined_bar.png / _box.png — refined-model MAE per class (refinement runs only)
  3a. mae_per_subject_old.png       — single bar: old model MAE per subject
  3b. mae_per_subject_projected.png — single bar: projected MAE per subject
  8a. mae_improvement_per_class_projected_vs_old.png   — bar: old_mae - projected_mae per class
  8a'.mae_per_class_compare_projected_vs_old.png       — grouped 2-bar companion: old_mae
                                      and projected_mae drawn side by side per class
  8b. mae_improvement_per_subject_projected_vs_old.png — bar: old_mae - projected_mae per subject
  8c. mae_improvement_per_class_newtest_refined_vs_original.png — bar: new-model test-set
                                      MAE_original - MAE_refined per class
                                      (green=refinement lowered error; refinement runs only)
  8c'.mae_per_class_compare_newtest_refined_vs_original.png — grouped 2-bar companion:
                                      before vs after MAE per class (refinement runs only)
  8d. mae_improvement_per_class_refined_vs_projected.png — projected_mae - refined_mae per class
  8d'.mae_per_class_compare_refined_vs_projected.png    — grouped 2-bar companion:
                                      projected vs refined MAE per class (refinement runs only)
  8e. mae_improvement_per_class_refined_vs_old.png       — old_mae - refined_mae per class
  8e'.mae_per_class_compare_refined_vs_old.png          — grouped 2-bar companion:
                                      old vs refined MAE per class (refinement runs only)
  8f. mae_improvement_per_class_combined.png — 1x3 panel combining all three source-split
                                      per-class improvements: (old - projected_before),
                                      (projected_before - projected_after), and
                                      (old - projected_after). Panel 1 fills the previously
                                      missing old-vs-before-refinement comparison
                                      (refinement runs only)
  4.  Confusion matrices (rounded predictions vs ground truth; each skipped when its
      num_classes > 15). Titles name the stage, dataset, and split:
        confusion_matrix_old.png              — old-model dataset / source set: old model (original)
        confusion_matrix_projected.png        — old-model dataset / source set: after projection
        confusion_matrix_refined.png          — old-model dataset / source set: after refinement
                                                (refinement runs only)
        confusion_matrix_newtest_original.png — new-model dataset / held-out split: new model
                                                (original, before refinement) (refinement runs only)
        confusion_matrix_newtest_refined.png  — new-model dataset / held-out split: new model
                                                (after refinement) (refinement runs only)
        confusion_matrix_all_stages.png       — combined 2×3 comparison of all of the above
                                                (stages absent for the run are blanked)
  5.  umap_all_projected.png / umap_all_refined.png — 1×2 UMAP: colored by label and by subject
      umap_space_comparison_*.png — 2×2 UMAP comparing aligned old-model test
                                      embeddings before and after projection, colored by
                                      label and by subject
  6.  anchor_weights.png            — weight entropy histogram + top-20 anchor usage
  7.  anchor_umap.png               — old vs new anchor embeddings in UMAP space
  9.  anchor_norm_comparison.png    — 3-panel: scatter (old_norm vs new_norm), overlaid
                                      histograms, and ratio (new/old) histogram
      anchor_norm_comparison.csv    — per-anchor table: idx, label, old_norm, new_norm,
                                      ratio, delta
  10. weight_rank_distribution.png  — 2-panel boxplot (linear + symlog): weight value at
                                      each rank position (0=most impactful, top_n-1=least
                                      shown) across all N samples
  11. dashboard.png                 — combined panel; metrics table lists per-stage MAE
                                      (old/projected/refined) micro/macro + preserve before/after.
                                      For an aggregate (pooled multi-model) run the per-mode
                                      dashboard_<mode>.png renders each MAE as 'mean ± std' across
                                      subtrials instead of a bare mean.
  12. refinement_training_curves_train_vs_val.png — per-epoch train-vs-val loss (total/B/A) +
                                      held-out val MAE (micro/macro) for source-B and preserve-A
                                      (refinement runs only)
  13. refinement_mae_before_vs_after.png — before/after MAE bars + projector anchor drift
                                      (refinement runs only)

Multi-mode refinement (--refinement 3, pkl key 'refinements'):
  Every refinement PNG above is emitted once per mode with a '_<mode>' suffix before the
  extension (e.g. refinement_training_curves_train_vs_val_linear_only.png,
  mae_per_class_refined_projector_linear_bar.png, umap_all_refined_linear_only.png), and a
  per-mode dashboard_<mode>.png is written alongside the projected-only base dashboard.png.
  dashboard_refinement_comparison.png adds a single figure contrasting the modes (per-stage
  MAE table + grouped per-class old−refined improvement bars). Single-mode runs
  (--refinement 1/2) keep the legacy unsuffixed filenames and the combined dashboard.png.
  Grid sweeps treat refine_mode as a sweep axis → mae_by_refine_mode.png at the search level.
  Training-recipe sweeps additionally emit mae_by_<recipe>_per_interp.png at the search level
  (one file per swept projector lp_* / refinement ref_* hyperparameter — lr, batch_size,
  optimizer, weight_decay, epochs, normalize_embeddings, loss — with one MAE bar row per
  interpolation_similarity value).

Usage:
  python3 cross_space_logs.py --pkl_path <path>
  python3 cross_space_logs.py --pkl_path <folder> --plot_only_top_k 5
  python3 cross_space_logs.py --pkl_path <folder> --plot_trials 3 7 12
  python3 cross_space_logs.py --pkl_path <root_folder> --subtrial_idx 2_3 4_1
  python3 cross_space_logs.py --pkl_path <root_folder> --only_aggregated
"""
import argparse
import glob
import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import pandas as pd
import torch
import umap
from torchmetrics.classification import MulticlassConfusionMatrix

from custom.tools import concordance_ccc, plot_confusion_matrix
from new_plot_tsne_post_head import plot_reducted_embeddings


# ── split-impact UMAP config ──────────────────────────────────────────────────
# Which new-model split is overlaid with the projected embeddings to judge whether
# the projected points fall inside the distribution of the embeddings actually used
# for training/validation/testing. Hard-coded here on purpose.
SPLIT_TO_COMPARE     = 'train'   # 'train' | 'val' | 'test'
SPLIT_SUBSAMPLE_FRAC = 0.8       # 0..1 — fraction of the split set kept before UMAP
SPLIT_SUBSAMPLE_SEED = 42        # reproducible subsample of the split set


# ── internal helpers ─────────────────────────────────────────────────────────

def _load_pkl(pkl_path):
  """
  Load a pickle file from disk.

  Args:
    pkl_path (str): Path to the pkl file.

  Returns:
    dict: Deserialized pickle contents.
  """
  with open(pkl_path, 'rb') as f:
    return pickle.load(f)


def _trial_number_from_pkl_path(pkl_path):
  """
  Parse the Optuna trial number from a grid-search results.pkl path.

  Relies on the cross_space_projection.py layout where each trial lives in a
  directory named 'trial{number:04d}_{tag}'. Reads only the path string; does
  not open the pkl.

  Args:
    pkl_path (str): Path to a .../trialNNNN_<tag>/results.pkl file.

  Returns:
    int | None: The parsed trial number, or None if the parent directory name
      does not match the 'trialNNNN_' convention.
  """
  name = os.path.basename(os.path.dirname(pkl_path))
  if not name.startswith('trial') or '_' not in name:
    return None
  head = name[len('trial'):name.index('_')]
  return int(head) if head.isdigit() else None


def _rebase_path(p, saved_root, actual_root):
  """
  Rewrite a single path so a path stored under saved_root points under actual_root.

  Used to repair the absolute paths a standalone pkl embeds at creation time after
  its run folder has been moved. Paths that are not located under saved_root (e.g.
  external model checkpoints / feature folders) are returned unchanged, since
  moving the run folder does not move those.

  Args:
    p           (str): The stored path to rewrite.
    saved_root  (str): The out_dir recorded in the pkl at creation time.
    actual_root (str): The directory the pkl is actually loaded from.

  Returns:
    str: The rebased path if p == saved_root or p is under saved_root; otherwise p
      unchanged.
  """
  ap     = os.path.abspath(p)
  asaved = os.path.abspath(saved_root)
  if ap == asaved:
    return actual_root
  if ap.startswith(asaved + os.sep):
    return os.path.join(actual_root, os.path.relpath(ap, asaved))
  return p


def _rebase_standalone_paths(data, pkl_path):
  """
  Repair stale out_dir-relative paths in a moved standalone pkl, in memory.

  A standalone pkl is saved inside its run's out_dir as results_<uid>.pkl and
  embeds absolute paths (out_dir, old_tensors_csv_path, anchors_csv_path, the
  linear-projector ckpt_path, the refinement *_pth keys) captured at creation
  time. When the run folder is moved those paths break. This rebases every such
  path so it points at the directory the pkl was actually loaded from, leaving
  external paths (model checkpoints, feature folders) untouched. The pkl on disk
  is never rewritten. Grid-format pkls (which resolve paths from pkl_path) and
  unmoved standalone pkls are left unchanged.

  Args:
    data     (dict): Deserialized pkl contents (mutated in place when rebased).
    pkl_path (str):  Path the pkl was loaded from; its directory is the ground-
      truth out_dir.

  Returns:
    dict: The (possibly mutated) data dict.
  """
  if _detect_format(data) != 'standalone':
    return data
  cfg   = data.get('config_cross_space_projection') or {}
  saved = cfg.get('out_dir')
  if not saved:
    return data
  actual = os.path.dirname(os.path.abspath(pkl_path))
  if os.path.abspath(saved) == actual:
    return data

  print(f'[cross_space_logs] pkl moved: rebasing paths {saved} -> {actual}')
  # Rebase against the original `saved` root for every key. cfg['out_dir'] is
  # rewritten last so the other keys still resolve against the original root.
  targets = [
    (cfg,  'old_tensors_csv_path'),
    (cfg,  'anchors_csv_path'),
    (data, 'old_tensors_csv_path'),
    (data, 'anchors_csv_path'),
    (data.get('linear_projector') or {}, 'ckpt_path'),
    (data.get('refinement') or {}, 'projector_before_pth'),
    (data.get('refinement') or {}, 'projector_after_pth'),
    (data.get('refinement') or {}, 'linear_before_pth'),
    (data.get('refinement') or {}, 'linear_after_pth'),
    (cfg,  'out_dir'),
  ]
  for container, key in targets:
    val = container.get(key)
    if isinstance(val, str) and val:
      container[key] = _rebase_path(val, saved, actual)
  return data


def _get_subject_map(csv_path):
  """
  Build a sample_id → subject_id lookup from a cross-projection CSV.

  Args:
    csv_path (str): Path to tab-separated CSV with columns sample_id and subject_id.

  Returns:
    dict[int, int]: Mapping from sample_id to subject_id.
  """
  df = pd.read_csv(csv_path, sep='\t', dtype={'sample_name': str})
  return dict(zip(df['sample_id'].astype(int), df['subject_id'].astype(int)))


def _round_preds(preds, num_classes: int):
  """
  Squeeze, round to nearest integer, and clip to [0, num_classes - 1].

  Args:
    preds       (np.ndarray): Float predictions, shape (N,) or (N, 1).
    num_classes (int):        Number of classes; determines the upper clip bound.

  Returns:
    np.ndarray: Integer predictions, shape (N,), dtype int64.
  """
  arr = np.asarray(preds, dtype=np.float32).squeeze()
  return np.clip(np.round(arr), 0, num_classes - 1).astype(np.int64)


def _mae_per_group(preds, labels, group_ids):
  """
  Compute mean absolute error per group.

  Args:
    preds     (np.ndarray): Shape (N,), float predictions.
    labels    (np.ndarray): Shape (N,), float ground truth.
    group_ids (np.ndarray): Shape (N,), integer group identifier per sample.

  Returns:
    dict[int, tuple[float, int]]: {group_id: (mae, count)} sorted by group_id.
  """
  abs_err = np.abs(preds - labels)
  result = {}
  for gid in np.unique(group_ids):
    mask = group_ids == gid
    result[int(gid)] = (float(np.mean(abs_err[mask])), int(mask.sum()))
  return result


def _raw_errors_per_group(preds, labels, group_ids):
  """
  Return per-sample absolute errors grouped by group_id.

  Args:
    preds     (np.ndarray): Shape (N,), float predictions.
    labels    (np.ndarray): Shape (N,), float ground truth.
    group_ids (np.ndarray): Shape (N,), integer group identifier.

  Returns:
    tuple[list[int], list[np.ndarray]]:
      - Sorted list of group ids.
      - One 1-D float array of absolute errors per group (same order).
  """
  abs_err = np.abs(np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32))
  gids    = sorted(int(g) for g in np.unique(group_ids))
  return gids, [abs_err[group_ids == gid] for gid in gids]


def _compute_global_mae(preds, labels):
  """
  Compute micro-averaged and macro-averaged MAE over all classes.

  Args:
    preds  (np.ndarray): Shape (N,), float predictions.
    labels (np.ndarray): Shape (N,), float ground-truth labels.

  Returns:
    tuple[float, float]: (micro_mae, macro_mae).
      micro_mae: mean |pred - label| over all samples (each sample equal weight).
      macro_mae: mean of per-class MAEs (each class equal weight, ignores class size).
  """
  labels_int = np.round(labels).astype(int)
  micro = float(np.mean(np.abs(preds - labels)))
  per_class = _mae_per_group(preds, labels, labels_int)
  macro = float(np.mean([v[0] for v in per_class.values()]))
  return micro, macro


def _round_clamp_like_training(preds, num_classes: int):
  """
  Round predictions half-away-from-zero and clamp to [0, num_classes - 1].

  Reproduces exactly the post-processing the training/test loop applies before it
  computes `test_l1_error` (head.py: copysign/floor rounding to avoid IEEE banker's
  rounding, then a clamp to the valid class range). Use this so cross-space MAE can be
  compared like-for-like against the training-pipeline `test_l1_error`.

  Args:
    preds       (np.ndarray): Float predictions in the real label scale, shape (N,) or (N, 1).
    num_classes (int):        Number of classes; upper clamp bound is num_classes - 1
      (equals the training clamp bound unique_val_classes.max() since
      num_classes = int(round(labels).max()) + 1).

  Returns:
    np.ndarray: Rounded+clamped predictions, shape (N,), dtype float32.
  """
  arr     = np.asarray(preds, dtype=np.float32).squeeze()
  rounded = np.copysign(np.floor(np.abs(arr) + 0.5), arr)   # avoid banker's rounding
  return np.clip(rounded, 0, num_classes - 1).astype(np.float32)


def _compute_rounded_mae(preds, labels, num_classes: int):
  """
  Micro- and macro-averaged MAE after rounding+clamping preds like the training test loop.

  Args:
    preds       (np.ndarray): Shape (N,), float predictions in the real label scale.
    labels      (np.ndarray): Shape (N,), float ground-truth labels.
    num_classes (int):        Number of classes (see _round_clamp_like_training).

  Returns:
    tuple[float, float]: (micro_mae, macro_mae) computed on the rounded+clamped preds.
  """
  return _compute_global_mae(_round_clamp_like_training(preds, num_classes), labels)


def _improvement(a, b):
  """
  Improvement from value a to value b for a lower-is-better metric (MAE).

  Args:
    a (float): Baseline metric value (the earlier stage).
    b (float): Later-stage metric value.

  Returns:
    tuple[float, float]: (abs_delta, pct_delta) where
      abs_delta = a - b  (positive => error reduced => improvement),
      pct_delta = (a - b) / a * 100  (NaN when a is 0 or non-finite).
  """
  d = a - b
  pct = (d / a * 100.0) if (np.isfinite(a) and a != 0) else float('nan')
  return d, pct


def _single_bar(ax, groups, vals, ylabel, title, color):
  """
  Render a bar chart with one bar per group.

  Args:
    ax     (matplotlib.axes.Axes): Axes to draw on.
    groups (list): Group labels for the x-axis.
    vals   (list[float]): One value per group.
    ylabel (str): Y-axis label.
    title  (str): Plot title.
    color  (str): Bar colour.
  """
  x = np.arange(len(groups))
  bars = ax.bar(x, vals, color=color, alpha=0.85)
  for bar in bars:
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + 0.02,
      f'{bar.get_height():.2f}',
      ha='center', va='bottom', fontsize=7,
    )
  ax.set_xticks(x)
  ax.set_xticklabels([str(g) for g in groups])
  ax.set_ylabel(ylabel)
  ax.set_title(title)


def _draw_mae_bar(ax, groups, vals, ylabel, title, color):
  """
  Draw a labelled MAE bar chart into a single pre-existing axes.

  Args:
    ax     (matplotlib.axes.Axes): Axes for the bar chart.
    groups (list): Group labels for the x-axis.
    vals   (list[float]): Mean value per group (bar heights).
    ylabel (str): Y-axis label.
    title  (str): Plot title.
    color  (str): Bar fill color.
  """
  x      = np.arange(len(groups))
  margin = 0.6
  xlim   = (x[0] - margin, x[-1] + margin) if len(x) > 0 else (-0.6, 0.6)

  bars = ax.bar(x, vals, color=color, alpha=0.85, edgecolor='white', linewidth=0.7)
  ax.set_xlim(*xlim)
  ax.set_xticks(x)
  ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='center')
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(axis='y', alpha=0.3)
  val_range = max(vals) - min(vals) if len(vals) > 1 else abs(vals[0]) if vals else 0
  offset    = val_range * 0.02 + 1e-6
  for bar in bars:
    h = bar.get_height()
    ax.text(
      bar.get_x() + bar.get_width() / 2, h + offset,
      f'{h:.2f}', ha='center', va='bottom', fontsize=7,
    )


def _draw_mae_improvement_bar(ax, groups, diffs, xlabel, title,
                              ylabel='MAE improvement (old - new)'):
  """
  Draw a signed MAE-improvement bar chart into a single pre-existing axes.

  Each bar is the per-group difference old_mae - new_mae: positive (the new
  model has lower error) is drawn green, negative (worse) is drawn red. NaN
  diffs — a group present for only one model — are skipped entirely. The
  signed value is printed at the bar tip, above positive bars and below
  negative ones.

  Args:
    ax     (matplotlib.axes.Axes): Axes to draw on.
    groups (list): Group labels for the x-axis (class or subject ids).
    diffs  (list[float]): old_mae - new_mae per group; NaN for missing groups.
    xlabel (str): X-axis label.
    title  (str): Plot title.
    ylabel (str): Y-axis label describing the signed difference convention.
  """
  x      = np.arange(len(groups))
  margin = 0.6
  xlim   = (x[0] - margin, x[-1] + margin) if len(x) > 0 else (-0.6, 0.6)

  finite = [d for d in diffs if np.isfinite(d)]
  val_range = (max(finite) - min(finite)) if len(finite) > 1 else (abs(finite[0]) if finite else 1.0)
  offset    = val_range * 0.03 + 1e-6

  for xi, d in zip(x, diffs):
    if not np.isfinite(d):
      continue
    color = '#2ca02c' if d > 0 else '#d62728'
    ax.bar(xi, d, color=color, alpha=0.85, edgecolor='white', linewidth=0.7)
    if d > 0:
      ax.text(xi, d + offset, f'{d:+.3f}', ha='center', va='bottom', fontsize=7)
    else:
      ax.text(xi, d - offset, f'{d:+.3f}', ha='center', va='top', fontsize=7)

  ax.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.6)
  ax.set_xlim(*xlim)
  ax.set_xticks(x)
  ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='center')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(axis='y', alpha=0.3)


def _draw_grouped_mae_bar(ax, groups, vals_a, vals_b, label_a, label_b,
                          color_a, color_b, ylabel, title):
  """
  Draw a grouped (two-bar-per-group) MAE bar chart into a single pre-existing axes.

  For each group two bars are drawn side by side so the two stages' MAE can be
  compared directly — an alternative reading of the signed improvement bar chart
  drawn by _draw_mae_improvement_bar. A bar with a non-finite height (a group
  present in only one series) is skipped for its value label, matching how the
  improvement helper skips NaN diffs.

  Args:
    ax      (matplotlib.axes.Axes): Axes to draw on.
    groups  (list): Group labels for the x-axis (class ids).
    vals_a  (list[float]): First-series mean MAE per group (the baseline stage).
    vals_b  (list[float]): Second-series mean MAE per group (the later stage).
    label_a (str): Legend label for the first series.
    label_b (str): Legend label for the second series.
    color_a (str): Bar fill colour for the first series.
    color_b (str): Bar fill colour for the second series.
    ylabel  (str): Y-axis label.
    title   (str): Plot title.
  """
  x      = np.arange(len(groups))
  width  = 0.4
  margin = 0.6
  xlim   = (x[0] - margin, x[-1] + margin) if len(x) > 0 else (-0.6, 0.6)

  for xpos, vals, color, label in (
    (x - width / 2, vals_a, color_a, label_a),
    (x + width / 2, vals_b, color_b, label_b),
  ):
    bars = ax.bar(xpos, vals, width=width, color=color, alpha=0.85,
                  edgecolor='white', linewidth=0.7, label=label)

  ax.set_xlim(*xlim)
  ax.set_xticks(x)
  ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='center')
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(axis='y', alpha=0.3)
  ax.legend(fontsize=8)


def _draw_mae_box(ax, groups, raw_by_group, ylabel, color, title=None):
  """
  Draw a per-group box plot of per-sample values into a single pre-existing axes.

  Args:
    ax           (matplotlib.axes.Axes): Axes for the box plot.
    groups       (list): Group labels for the x-axis.
    raw_by_group (list[np.ndarray]): Per-sample value arrays, one per group.
    ylabel       (str): Y-axis label.
    color        (str): Box fill color.
    title        (str | None): Optional plot title.
  """
  x      = np.arange(len(groups))
  margin = 0.6
  xlim   = (x[0] - margin, x[-1] + margin) if len(x) > 0 else (-0.6, 0.6)

  face_rgba      = list(mcolors.to_rgba(color))
  face_rgba[3]   = 0.5
  box_data = [
    arr if (len(arr) > 0 and not np.all(np.isnan(arr))) else np.array([np.nan])
    for arr in raw_by_group
  ]
  bp = ax.boxplot(
    box_data,
    positions=x,
    widths=0.5,
    patch_artist=True,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                    markerfacecolor=color, markeredgecolor='none'),
    medianprops=dict(color='#222222', linewidth=1.5),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
  )
  for patch in bp['boxes']:
    patch.set_facecolor(face_rgba)
    patch.set_edgecolor(color)
  ax.set_xlim(*xlim)
  ax.set_xticks(x)
  ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='right')
  ax.set_ylabel(ylabel)
  ax.grid(axis='y', alpha=0.3)
  if title is not None:
    ax.set_title(title)


def _draw_bar_boxplot(ax_bar, ax_box, groups, vals, raw_by_group, ylabel, title, color):
  """
  Draw a stacked bar (top) + box plot (bottom) into two pre-existing axes.

  Args:
    ax_bar       (matplotlib.axes.Axes): Axes for the bar chart (top).
    ax_box       (matplotlib.axes.Axes): Axes for the box plot (bottom).
    groups       (list): Group labels for the x-axis.
    vals         (list[float]): Mean value per group (bar heights).
    raw_by_group (list[np.ndarray]): Per-sample value arrays, one per group.
    ylabel       (str): Y-axis label for both subplots.
    title        (str): Title for the bar subplot.
    color        (str): Bar fill and box fill color.
  """
  _draw_mae_bar(ax_bar, groups, vals, ylabel, title, color)
  _draw_mae_box(ax_box, groups, raw_by_group, ylabel, color)


def _pairwise_cosine_sim_per_class(emb, labels_int):
  """
  Compute all pairwise cosine similarities within each class.

  Args:
    emb        (np.ndarray): Shape (N, D), float32 embedding matrix.
    labels_int (np.ndarray): Shape (N,), integer class labels.

  Returns:
    dict[int, np.ndarray]: Maps class id to a 1-D array of upper-triangle
      pairwise cosine similarities.  Classes with fewer than 2 samples get
      a single-element array [1.0].
  """
  result = {}
  for cls in np.unique(labels_int):
    cls_emb = emb[labels_int == cls]
    if len(cls_emb) < 2:
      result[int(cls)] = np.array([1.0], dtype=np.float32)
      continue
    norms   = np.linalg.norm(cls_emb, axis=1, keepdims=True)
    normed  = cls_emb / (norms + 1e-8)
    cos_mat = normed @ normed.T
    idx     = np.triu_indices(len(cls_emb), k=1)
    result[int(cls)] = cos_mat[idx].astype(np.float32)
  return result


def _compute_umap(embeddings):
  """
  Fit UMAP and return 2-D projections.

  Args:
    embeddings (np.ndarray): Shape (N, D), float32.

  Returns:
    np.ndarray: Shape (N, 2).
  """
  reducer = umap.UMAP(random_state=42)
  return reducer.fit_transform(embeddings.astype(np.float32))


def _weight_entropy(weights):
  """
  Compute Shannon entropy of each row in a weight matrix.

  Args:
    weights (np.ndarray): Shape (N, K), rows sum to 1.

  Returns:
    np.ndarray: Shape (N,), entropy (nats) per sample.
  """
  return -np.sum(weights * np.log(weights + 1e-9), axis=1)


def _write_summary_csv(row, out_dir):
  """
  Write a single-row summary CSV into out_dir/summary.csv.

  Args:
    row     (dict): Flat dict of hyperparameters and metrics.
    out_dir (str):  Directory to write the CSV into.
  """
  _write_summary_rows([row], out_dir)


def _write_summary_rows(rows, out_dir):
  """
  Write one-or-more summary rows into out_dir/summary.csv.

  Args:
    rows    (list[dict]): Flat rows of hyperparameters and metrics (one per refinement mode
      for a --refinement 3 run; a single row otherwise).
    out_dir (str):        Directory to write the CSV into.
  """
  path = os.path.join(out_dir, 'summary.csv')
  pd.DataFrame(rows).to_csv(path, index=False)
  print(f'Saved: {path}')


def _detect_format(data):
  """
  Detect whether a loaded pkl dict is a grid-search trial or a standalone run.

  Args:
    data (dict): Deserialized pkl contents.

  Returns:
    str: 'grid' if trial_params key is present, else 'standalone'.
  """
  return 'grid' if 'trial_params' in data else 'standalone'


def _anchor_count_from_data(data):
  """
  Real number of anchors actually used by a single run, from its anchor embeddings.

  The configured `num_anchors` is only a budget; anchor selection caps each stratum at the
  samples available (see cross_space_projection._allocate_balanced), so the real count is
  often lower. The aligned `old_model_anchors_embeddings` are exactly the anchors the
  projector was fit on, making their length the canonical "real" count.

  Args:
    data (dict): Deserialized run pkl contents.

  Returns:
    int | None: len(old_model_anchors_embeddings['sample_ids']), or None when the pkl
      carries no anchor embeddings (grid trials without them, or num_anchors in {0, -1}).
  """
  sid = (data.get('old_model_anchors_embeddings') or {}).get('sample_ids')
  return int(len(sid)) if sid is not None else None


def _real_anchor_freq_from_subtrials(data, pkl_path):
  """
  Frequency of the real anchor count across an aggregate's subtrials: {real_count: n_subtrials}.

  The pooled aggregate pkl stores no anchor embeddings, and the base dashboard is drawn
  before the subtrial pkls are (heavily) loaded, so the count is read cheaply from each
  subtrial's sibling anchors.csv (rows = header + one line per selected anchor). Subtrial
  pkl paths are stored relative to this aggregate pkl's directory in data['subtrial_pkls'].

  Args:
    data     (dict): Aggregated pkl contents (must carry 'subtrial_pkls').
    pkl_path (str):  Path the aggregate pkl was loaded from (anchors.csv paths resolve
      relative to its directory).

  Returns:
    dict | None: {real_anchor_count: number_of_subtrials}, sorted by count ascending;
      None when no subtrial anchors.csv could be read.
  """
  base = os.path.dirname(os.path.abspath(pkl_path))
  counts = []
  for rel in data.get('subtrial_pkls') or []:
    csvp = os.path.join(os.path.dirname(os.path.join(base, rel)), 'anchors.csv')
    try:
      with open(csvp) as f:
        counts.append(sum(1 for _ in f) - 1)
    except OSError:
      continue
  return dict(sorted(Counter(counts).items())) if counts else None


# Refinement-stage columns surfaced from the optional 'refinement' pkl block written by
# cross_space_projection._run_refinement_stage. Absent / non-refinement runs receive
# defaults so summary.csv keeps a stable schema across mixed sweeps.
_REFINEMENT_SUMMARY_KEYS = (
  'refine_enabled', 'refine_best_epoch',
  'refine_val_selection', 'refine_best_val_total',
  'proj_anchor_loss_before', 'proj_anchor_loss_after',
  'mae_micro_old_oncsv_before', 'mae_macro_old_oncsv_before',
  'mae_micro_old_oncsv_after',  'mae_macro_old_oncsv_after',
  'mae_micro_new_test_before', 'mae_macro_new_test_before',
  'mae_micro_new_test_after',  'mae_macro_new_test_after',
  'refine_old_model_csv', 'refine_new_eval_split',
  'projector_before_pth', 'projector_after_pth',
  'linear_before_pth', 'linear_after_pth',
)


def _refine_items(data):
  """
  Resolve a pkl's refinement stage(s) into a uniform (mode, block) list.

  --refinement 3 stores several blocks under the plural 'refinements' key (one per
  mode: linear_only / projector_linear); --refinement 1/2 store a single block under
  'refinement'. This mirrors the plotting path's idiom so summary-row builders can be
  mode-aware without caring which schema a pkl uses.

  Args:
    data (dict): Deserialized pkl contents.

  Returns:
    list[tuple[str, dict]]: [(mode, block), ...]. The plural blocks are keyed by mode;
      the singular block's mode is read from its 'refine_mode' field ('' if absent).
      Empty list when the pkl has no refinement stage.
  """
  multi = data.get('refinements')
  if multi:
    return list(multi.items())
  single = data.get('refinement')
  if single:
    return [(single.get('refine_mode') or '', single)]
  return []


def _refinement_columns(data, refine_block=None):
  """
  Extract the flat refinement-stage columns from a pkl dict for summary.csv.

  Args:
    data         (dict): Deserialized pkl contents (may or may not have a 'refinement' block).
    refine_block (dict | None): A specific per-mode refinement block (one entry of the
      plural 'refinements') to read instead of the singular data['refinement']. When None,
      falls back to data.get('refinement') (backward-compatible for grid/single-mode pkls).

  Returns:
    dict: One entry per _REFINEMENT_SUMMARY_KEYS. With no refinement block,
      'refine_enabled' is False and numeric/path fields default to NaN/None.
  """
  ref = (refine_block if refine_block is not None else data.get('refinement')) or {}
  out = {}
  for k in _REFINEMENT_SUMMARY_KEYS:
    if k in ref:
      out[k] = ref[k]
    elif k == 'refine_enabled':
      out[k] = False
    elif k.endswith('_pth') or k in ('refine_old_model_csv', 'refine_new_eval_split',
                                     'refine_val_selection'):
      out[k] = None
    else:
      out[k] = float('nan')
  return out


# Swept projector / refinement recipe fields surfaced into summary.csv as lp_* / ref_*
# columns (from data['linear_projector']['config'] / data['refinement']['config']), so
# a recipe sweep is comparable at a glance. Absent blocks → None (stable schema).
_LP_SUMMARY_FIELDS = (
  'lr', 'batch_size', 'optimizer', 'weight_decay', 'epochs', 'normalize_embeddings', 'loss',
  'encoder_ratio',
)
_REF_SUMMARY_FIELDS = (
  'lr_projector', 'lr_linear', 'lambda_B', 'lambda_A', 'optimizer',
  'weight_decay', 'epochs', 'loss', 'batch_size',
)


def _recipe_columns(data, refine_block=None):
  """
  Extract the swept projector / refinement recipe fields as flat lp_* / ref_* columns.

  Args:
    data         (dict): Deserialized pkl contents (may lack the 'linear_projector' /
      'refinement' blocks for non-projector / non-refinement runs).
    refine_block (dict | None): A specific per-mode refinement block to read the ref_*
      recipe from instead of the singular data['refinement']. When None, falls back to
      data.get('refinement'). The lp_* recipe always comes from data['linear_projector']
      (the projector is shared across refinement modes).

  Returns:
    dict: One 'lp_<field>' per _LP_SUMMARY_FIELDS and one 'ref_<field>' per
      _REF_SUMMARY_FIELDS; None where the corresponding block/field is absent.
  """
  lp  = (data.get('linear_projector') or {}).get('config') or {}
  ref = ((refine_block if refine_block is not None else data.get('refinement')) or {}).get('config') or {}
  out = {f'lp_{f}':  lp.get(f)  for f in _LP_SUMMARY_FIELDS}
  out.update({f'ref_{f}': ref.get(f) for f in _REF_SUMMARY_FIELDS})
  return out


def _best_epoch_columns(data, refine_block=None):
  """
  Best-epoch-vs-configured-epochs columns for every trained stage.

  Lets a sweep show at a glance whether each stage's epoch budget is enough: a
  fraction near 1.0 means the best (val-selected) checkpoint was at the end of
  training, i.e. the stage was still improving when epochs ran out. The projector
  (1st stage) best epoch is read from data['linear_projector']; the totals
  (lp_epochs / ref_epochs) and refine_best_epoch already come from _recipe_columns /
  _refinement_columns, so the fractions are consistent with those columns. Closed-form
  projectors (procrustes / closed_form, best_epoch=1) make the projector fraction
  meaningless — it is still emitted; the 'kind'/interpolation_similarity columns
  disambiguate.

  Args:
    data         (dict): Deserialized pkl contents.
    refine_block (dict | None): A specific per-mode refinement block to read
      refine_best_epoch / config.epochs from instead of the singular data['refinement'].
      When None, falls back to data.get('refinement'). The projector (1st stage) always
      comes from data['linear_projector'].

  Returns:
    dict: lp_best_epoch (int | None), lp_best_epoch_frac (float),
      refine_best_epoch_frac (float). Fraction is NaN where the stage is absent,
      the total is missing/0, or best_epoch <= 0.
  """
  nan = float('nan')

  def _frac(best, total):
    """Return best/total, or NaN when either is missing/0 or best <= 0."""
    if best is None or total in (None, 0) or best <= 0:
      return nan
    try:
      return float(best) / float(total)
    except (TypeError, ZeroDivisionError):
      return nan

  lp  = data.get('linear_projector') or {}
  ref = (refine_block if refine_block is not None else data.get('refinement')) or {}
  lp_best  = lp.get('best_epoch')
  lp_tot   = (lp.get('config') or {}).get('epochs')
  ref_best = ref.get('refine_best_epoch')
  ref_tot  = (ref.get('config') or {}).get('epochs')
  return {
    'lp_best_epoch':          lp_best,
    'lp_best_epoch_frac':     _frac(lp_best, lp_tot),
    'refine_best_epoch_frac': _frac(ref_best, ref_tot),
  }


def _collect_summary_row(data, pkl_path, refine_block=None, refine_mode=None):
  """
  Extract hyperparameters and metrics from a grid-format pkl into a flat dict.

  Args:
    data         (dict): Grid-format pkl contents (must have trial_params and metrics).
    pkl_path     (str):  Path to the pkl file (unused, kept for signature consistency).
    refine_block (dict | None): A specific per-mode refinement block (one entry of the
      plural 'refinements') to source every refinement column from, instead of the
      singular data['refinement']. When None, falls back to data.get('refinement') — the
      backward-compatible path for grid / single-mode (--refinement 1/2) pkls. Callers
      that iterate the modes of a --refinement 3 pkl pass one block per row (see
      _collect_summary_rows).
    refine_mode  (str | None): Mode label for the 'refine_mode' column when refine_block
      is supplied; falls back to trial_params['refine_mode'] when None.

  Returns:
    dict: Flat row with trial_number, seed (the run's RNG seed, None for pkls saved
      before seed was persisted), the trial hyperparams (including the mlp axes
      mlp_activation / mlp_num_layers), mae, ccc, refinement columns, the per-stage
      best-epoch-vs-budget columns (lp_best_epoch, lp_best_epoch_frac,
      refine_best_epoch_frac — a fraction near 1.0 flags too few epochs),
      the clear before/after comparison block (srctest_* / newtest_*), the pairwise
      improvement deltas (srctest_*_delta_*/_pct_*, newtest_*_delta_*/_pct_*), and the
      merged general_mae_* block summarizing per-step (before/after) improvement vs
      the original models, averaged equally across both splits (per-metric
      general_mae_{micro,macro}_{delta,pct}_{before,after}) and across both
      splits + micro/macro (grand general_mae_{delta,pct}_{before,after}).
  """
  p     = data['trial_params']
  m     = data['metrics']
  new_t = data['new_model_tensors']
  old_t = data['old_model_tensors']
  new_preds = np.asarray(new_t['predictions'], dtype=np.float32).squeeze()
  old_preds = np.asarray(old_t['predictions'], dtype=np.float32).squeeze()
  lbl       = np.asarray(new_t['labels'],      dtype=np.float32)
  mae_micro_new, mae_macro_new = _compute_global_mae(new_preds, lbl)
  mae_micro_old, mae_macro_old = _compute_global_mae(old_preds, lbl)
  row = {
    'trial_number':             data['trial_number'],
    'seed':                     data.get('seed'),
    'num_anchors':              p['num_anchors'],
    'num_anchors_real':         _anchor_count_from_data(data),
    'anchor_selection_type':    p['anchor_selection_type'],
    'csv_anchor_selection':     p['csv_anchor_selection'],
    'old_model_csv':            p['old_model_csv'],
    'interpolation_similarity': p['interpolation_similarity'],
    'mlp_activation':           p.get('mlp_activation'),
    'mlp_num_layers':           p.get('mlp_num_layers'),
    'weighting_method':         p['weighting_method'],
    'temperature':              p.get('temperature'),
    'rbf_sigma':                p['rbf_sigma'],
    'projector_config':         p.get('projector_config'),
    'refinement_config':        p.get('refinement_config'),
    'refine_mode':              refine_mode if refine_mode is not None else p.get('refine_mode'),
    'mae':                      m['mae'],
    'ccc':                      m['ccc'],
    'runtime_min':              m.get('runtime_min'),
    'mae_micro':                mae_micro_new,
    'mae_macro':                mae_macro_new,
    'mae_micro_old':            mae_micro_old,
    'mae_macro_old':            mae_macro_old,
  }
  fake_meta = data.get('fake_projection_metadata') or {}
  if data.get('fake_projection_evaluations'):
    row.update({
      'fake_projection': True,
      'fake_projection_distribution': fake_meta.get(
        'distribution', data.get('fake_projection_distribution')),
      'fake_projection_seed': fake_meta.get('seed', data.get('fake_projection_seed')),
    })
  row.update(_refinement_columns(data, refine_block=refine_block))
  row.update(_recipe_columns(data, refine_block=refine_block))
  row.update(_best_epoch_columns(data, refine_block=refine_block))
  # Resolved dataset names (best-effort, never raise) so summary plots can name the
  # source / new-model dataset in titles. _collect_summary_row is grid-only.
  row['src_dataset'] = _resolve_old_dataset(data, 'grid', pkl_path)
  row['new_dataset'] = _resolve_new_dataset(data, 'grid', pkl_path)

  # ── Clear, grouped before/after refinement comparison columns ──
  # These restate the existing (obscurely-named) refinement metrics under
  # self-documenting names so the three-stage comparison reads at a glance.
  # 'srctest_*' is the projected source/old split (= --old_model_csv; run it as
  # 'test'); 'newtest_*' is the new (target) model's own test split.
  ref = (refine_block if refine_block is not None else data.get('refinement')) or {}
  nan = float('nan')
  # Source/projected split stage values: old (old model on its own split) / before
  # (projection with the original head) / after (projection with the refined head/
  # projector). Per metric we also emit the three pairwise improvement deltas
  # (old-before, old-after, before-after) and their relative % (see _improvement):
  # positive => MAE went down => improvement.
  src = {
    'micro': (mae_micro_old, ref.get('mae_micro_old_oncsv_before', nan),
              ref.get('mae_micro_old_oncsv_after', nan)),
    'macro': (mae_macro_old, ref.get('mae_macro_old_oncsv_before', nan),
              ref.get('mae_macro_old_oncsv_after', nan)),
  }
  for m, (old_v, bef_v, aft_v) in src.items():
    row[f'srctest_mae_{m}_old']    = old_v
    row[f'srctest_mae_{m}_before'] = bef_v
    row[f'srctest_mae_{m}_after']  = aft_v
    for tag, a, b in (('old_before',   old_v, bef_v),
                      ('old_after',    old_v, aft_v),
                      ('before_after', bef_v, aft_v)):
      d, pct = _improvement(a, b)
      row[f'srctest_mae_{m}_delta_{tag}'] = d
      row[f'srctest_mae_{m}_pct_{tag}']   = pct

  # New (target) model's own test split: before (native new model) / after (refined
  # head). No "old" stage here (no old model on the target test). Computed from the
  # per-sample predictions in new_test_eval (real label scale, see _linear_preds);
  # the before->after delta is the refinement preserve check on the real test set.
  nte = ref.get('new_test_eval')
  if nte:
    lbl_nt = np.asarray(nte['labels'],       dtype=np.float32)
    pre_b  = np.asarray(nte['preds_before'], dtype=np.float32).reshape(-1)
    pre_a  = np.asarray(nte['preds_after'],  dtype=np.float32).reshape(-1)
    nb_micro, nb_macro = _compute_global_mae(pre_b, lbl_nt)
    na_micro, na_macro = _compute_global_mae(pre_a, lbl_nt)
    # Rounded+clamped siblings (match the training `test_l1_error` definition). This new-model
    # own-test split is the most direct analog of summary.csv's all_test_l1_error.
    nt_classes = int(np.round(lbl_nt).max()) + 1
    nb_micro_r, nb_macro_r = _compute_rounded_mae(pre_b, lbl_nt, nt_classes)
    na_micro_r, na_macro_r = _compute_rounded_mae(pre_a, lbl_nt, nt_classes)
  else:
    nb_micro = nb_macro = na_micro = na_macro = nan
    nb_micro_r = nb_macro_r = na_micro_r = na_macro_r = nan
  for m, bef_v, aft_v, bef_r, aft_r in (
      ('micro', nb_micro, na_micro, nb_micro_r, na_micro_r),
      ('macro', nb_macro, na_macro, nb_macro_r, na_macro_r)):
    row[f'newtest_mae_{m}_before'] = bef_v
    row[f'newtest_mae_{m}_after']  = aft_v
    d, pct = _improvement(bef_v, aft_v)
    row[f'newtest_mae_{m}_delta_before_after'] = d
    row[f'newtest_mae_{m}_pct_before_after']   = pct
    row[f'newtest_mae_{m}_before_rounded'] = bef_r
    row[f'newtest_mae_{m}_after_rounded']  = aft_r

  # ── General (merged srctest+newtest) MAE improvement vs the original models ──
  # Headline measures of overall improvement/degradation relative to the original
  # models ("old results": old model on the source split = srctest *_old; native
  # new model on its own test split = newtest *_before). For each refinement step
  # (before / after) and metric (micro / macro), each split's improvement vs its
  # own original baseline is taken via _improvement (positive => MAE dropped =>
  # improvement) and the two splits are averaged equally. The per-metric
  # general_mae_{micro,macro}_* columns average the 2 splits; the grand
  # general_mae_* columns additionally average across micro+macro (a single overall
  # scalar per step). newtest is unchanged at the 'before' step, so its 'before'
  # improvement is 0 by construction; a fixed split count keeps 'before' and
  # 'after' comparable. A stage/metric absent (e.g. non-refinement runs) is dropped
  # from the average; NaN only when nothing is finite.
  metric_stages = {
    'micro': (src['micro'][0], src['micro'][1], src['micro'][2], nb_micro, na_micro),
    'macro': (src['macro'][0], src['macro'][1], src['macro'][2], nb_macro, na_macro),
  }
  grand = {'before': {'delta': [], 'pct': []}, 'after': {'delta': [], 'pct': []}}
  for metric, (s_old, s_bef, s_aft, n_bef, n_aft) in metric_stages.items():
    for step, s_stage, n_stage in (('before', s_bef, n_bef), ('after', s_aft, n_aft)):
      s_d, s_pct = _improvement(s_old, s_stage)  # srctest baseline = old model
      n_d, n_pct = _improvement(n_bef, n_stage)  # newtest baseline = native new model
      deltas = [v for v in (s_d, n_d)     if np.isfinite(v)]
      pcts   = [v for v in (s_pct, n_pct) if np.isfinite(v)]
      row[f'general_mae_{metric}_delta_{step}'] = float(np.mean(deltas)) if deltas else nan
      row[f'general_mae_{metric}_pct_{step}']   = float(np.mean(pcts))   if pcts   else nan
      grand[step]['delta'].extend(deltas)
      grand[step]['pct'].extend(pcts)
  for step in ('before', 'after'):
    gd, gp = grand[step]['delta'], grand[step]['pct']
    row[f'general_mae_delta_{step}'] = float(np.mean(gd)) if gd else nan
    row[f'general_mae_pct_{step}']   = float(np.mean(gp)) if gp else nan
  return row


def _collect_row_task(pkl_path):
  """
  Load one trial pkl and extract its summary row (worker for the parallel
  'Collecting metrics' phase of generate_logs_search).

  Args:
    pkl_path (str): Path to a trial's results.pkl.

  Returns:
    dict: The _collect_summary_row row, with '_pkl_path' set to pkl_path.
  """
  data = _load_pkl(pkl_path)
  row  = _collect_summary_row(data, pkl_path)
  row['_pkl_path'] = pkl_path
  return row


def _collect_summary_rows(data, pkl_path):
  """
  Build one summary row per refinement mode present in a pkl (grid-schema columns).

  --refinement 3 stores two refinement blocks (linear_only / projector_linear) under the
  plural 'refinements' key, each with its own srctest_* / newtest_* / ref_* metrics. This
  emits one _collect_summary_row per mode (the 'refine_mode' column distinguishes them),
  so every mode's measures land in the CSV. A single-mode ('refinement') or no-refinement
  pkl yields exactly one row, identical to the pre-existing _collect_summary_row output.

  Args:
    data     (dict): pkl contents (grid-format, or standalone with trial_params injected).
    pkl_path (str):  Path to the pkl (forwarded to _collect_summary_row).

  Returns:
    list[dict]: One flat row per refinement mode (>= 1 row).
  """
  items = _refine_items(data)
  if not items:
    return [_collect_summary_row(data, pkl_path)]
  return [_collect_summary_row(data, pkl_path, refine_block=block, refine_mode=(mode or None))
          for mode, block in items]


def _synth_trial_params_from_cfg(cfg):
  """
  Build a grid-style trial_params dict from a standalone run's config block.

  Standalone (and aggregated) pkls store the hyperparameters under
  config_cross_space_projection rather than the grid format's trial_params. This maps the
  former onto the keys _collect_summary_row reads, so a grid-schema summary row can be built
  for a standalone subtrial. Keys absent from a standalone config (projector_config /
  refinement_config / refine_mode / temperature) default to None.

  Args:
    cfg (dict): A run's config_cross_space_projection mapping.

  Returns:
    dict: trial_params with the keys _collect_summary_row reads.
  """
  return {
    'num_anchors':              cfg.get('num_anchors'),
    'anchor_selection_type':    cfg.get('anchor_selection_type'),
    'csv_anchor_selection':     cfg.get('csv_anchor_selection'),
    'old_model_csv':            cfg.get('old_model_csv'),
    'interpolation_similarity': cfg.get('interpolation_similarity'),
    'mlp_activation':           cfg.get('mlp_activation'),
    'mlp_num_layers':           cfg.get('mlp_num_layers'),
    'weighting_method':         cfg.get('weighting_method'),
    'temperature':              cfg.get('temperature'),
    'rbf_sigma':                cfg.get('rbf_sigma'),
    'projector_config':         cfg.get('projector_config'),
    'refinement_config':        cfg.get('refinement_config'),
    'refine_mode':              cfg.get('refine_mode'),
  }


def _aggregated_summary_rows(data, pkl_path, subtrial_index, n_subtrials):
  """
  Build the grid-schema summary rows for a standalone-format pkl — one per refinement mode.

  Reuses _collect_summary_rows by first injecting a synthesized trial_params + a trial_number,
  then prepends identifier columns (subtrial_index, new_model_pth, old_model_pth, n_subtrials)
  to every row so each is traceable to its model pair. A --refinement 3 subtrial yields one
  row per mode (linear_only / projector_linear); single-mode / no-refinement pkls yield one.
  Used for both the per-subtrial rows and the pooled aggregate row, so summary_per_subtrial.csv
  and the pooled summary.csv share columns.

  Args:
    data           (dict): Loaded standalone/aggregated pkl contents.
    pkl_path       (str):  Path the pkl was loaded from (forwarded to _collect_summary_rows).
    subtrial_index (int | str): Identifier for these rows ('AGGREGATE' for the pooled row).
    n_subtrials    (int): Number of subtrials pooled (same on every row for stable columns).

  Returns:
    list[dict]: One dict per refinement mode (>= 1), each = identifier columns followed by
      every _collect_summary_row column.
  """
  cfg = data.get('config_cross_space_projection') or {}
  d = dict(data)
  d.setdefault('trial_params', _synth_trial_params_from_cfg(cfg))
  d.setdefault('trial_number', subtrial_index)
  ident = {
    'subtrial_index': subtrial_index,
    'new_model_pth':  cfg.get('new_model_pth'),
    'old_model_pth':  cfg.get('old_model_pth'),
    'n_subtrials':    n_subtrials,
  }
  return [{**ident, **row} for row in _collect_summary_rows(d, pkl_path)]


def _aggregated_summary_row(data, pkl_path, subtrial_index, n_subtrials):
  """
  Build a single grid-schema summary row from a standalone-format pkl.

  Thin wrapper over _aggregated_summary_rows returning its first row — used where exactly
  one row is expected (the pooled 'AGGREGATE' row, whose pkl carries no refinement stage).

  Args:
    data           (dict): Loaded standalone/aggregated pkl contents.
    pkl_path       (str):  Path the pkl was loaded from.
    subtrial_index (int | str): Identifier for this row ('AGGREGATE' for the pooled row).
    n_subtrials    (int): Number of subtrials pooled (same on every row for stable columns).

  Returns:
    dict: identifier columns followed by every _collect_summary_row column.
  """
  return _aggregated_summary_rows(data, pkl_path, subtrial_index, n_subtrials)[0]


# Per-stage source/preserve MAE column pairs (micro, macro) that the dashboard metrics table
# consumes as mae_stages. Keys mirror plot_dashboard's recognized stages; the source stages
# map onto the srctest_* comparison block and the preserve stages onto newtest_* (see
# _collect_summary_row). Used to roll a pooled cross-validation aggregate up from the
# per-subtrial rows without needing the (pooled) refined predictions.
_STAGE_MAE_COLUMNS = {
  'old':             ('srctest_mae_micro_old',    'srctest_mae_macro_old'),
  'projected':       ('srctest_mae_micro_before', 'srctest_mae_macro_before'),
  'refined':         ('srctest_mae_micro_after',  'srctest_mae_macro_after'),
  'preserve_before': ('newtest_mae_micro_before', 'newtest_mae_macro_before'),
  'preserve_after':  ('newtest_mae_micro_after',  'newtest_mae_macro_after'),
}


def _aggregate_subtrial_rows(sub_rows, n_subtrials):
  """
  Roll the per-subtrial x per-mode summary rows up into per-mode mean + std rows.

  A cross-validation aggregate's pooled pkl carries no refinement metrics, so its single
  AGGREGATE summary row leaves every refinement column empty. This instead averages the
  fully-populated per-subtrial rows (grouped by refine_mode), emitting for each mode a MEAN
  row and a STD row that share summary_per_subtrial.csv's exact schema — so the aggregate
  summary.csv carries the same srctest_* / newtest_* / refine_* columns, now filled.

  Args:
    sub_rows    (list[dict]): Per-subtrial x per-mode rows (from _aggregated_summary_rows);
      each carries a 'refine_mode' (may be None) and the numeric stage/refinement columns.
    n_subtrials (int): Number of subtrials pooled (stamped onto every emitted row).

  Returns:
    list[dict]: Two rows per refine_mode — subtrial_index 'AGGREGATE_MEAN' (numeric columns =
      mean across the mode's subtrials) and 'AGGREGATE_STD' (sample std, ddof=1; 0.0 for a
      single-subtrial group). Non-numeric columns (paths, hyperparameter labels, refine_mode)
      are carried through unchanged (constant within a mode). Empty list when sub_rows is empty.
  """
  if not sub_rows:
    return []
  df = pd.DataFrame(sub_rows)
  numeric_cols     = df.select_dtypes(include=[np.number]).columns
  non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
  out = []
  for _mode, grp in df.groupby('refine_mode', dropna=False):
    means = grp[numeric_cols].mean()
    stds  = grp[numeric_cols].std(ddof=1)
    if len(grp) == 1:                       # ddof=1 std of a lone sample is NaN → report 0.0
      stds = stds.fillna(0.0)
    base = {c: grp.iloc[0][c] for c in non_numeric_cols}
    for tag, stat in (('AGGREGATE_MEAN', means), ('AGGREGATE_STD', stds)):
      row = {**base, **stat.to_dict()}
      row['subtrial_index'] = tag
      row['n_subtrials']    = n_subtrials
      out.append(row)
  return out


def _per_mode_stage_reduce(sub_rows, reduce_fn):
  """
  Per-mode dashboard mae_stages built by reducing the per-subtrial stage columns.

  Rolls the per-subtrial rows up (grouped by refine_mode) into the (micro, macro) MAE tuples
  the dashboard metrics table renders: source old/projected/refined plus the new-model
  preserve before/after. A stage whose columns are entirely NaN for a mode (e.g. no
  refinement ran) is omitted so single-mode / no-refinement aggregates degrade gracefully.

  Stage presence is decided by the *mean* (i.e. data availability), independent of reduce_fn,
  so the mean and std roll-ups expose exactly the same stage keys — a lone subtrial's ddof=1
  std is NaN yet its stage stays present (the std reducer reports 0.0 for it).

  Args:
    sub_rows  (list[dict]): Per-subtrial x per-mode rows (from _aggregated_summary_rows).
    reduce_fn (callable): pandas Series -> float reducer applied to each stage column
      (e.g. Series.mean or a ddof=1 std).

  Returns:
    dict[str, dict]: {refine_mode: {stage: (micro, macro)}}. refine_mode may be None
      (no-refinement aggregate). Empty dict when sub_rows is empty.
  """
  if not sub_rows:
    return {}
  df = pd.DataFrame(sub_rows)
  out = {}
  for mode, grp in df.groupby('refine_mode', dropna=False):
    stages = {}
    for stage, (mi_col, ma_col) in _STAGE_MAE_COLUMNS.items():
      if mi_col not in grp.columns or ma_col not in grp.columns:
        continue
      if not (np.isfinite(grp[mi_col].mean()) or np.isfinite(grp[ma_col].mean())):
        continue                                        # pandas mean skips NaN
      stages[stage] = (reduce_fn(grp[mi_col]), reduce_fn(grp[ma_col]))
    out[mode] = stages
  return out


def _per_mode_stage_means(sub_rows):
  """
  Per-mode dashboard mae_stages holding the mean of the per-subtrial stage columns.

  Args:
    sub_rows (list[dict]): Per-subtrial x per-mode rows (from _aggregated_summary_rows).

  Returns:
    dict[str, dict]: {refine_mode: {stage: (mae_micro_mean, mae_macro_mean)}}. refine_mode may
      be None (no-refinement aggregate). Empty dict when sub_rows is empty.
  """
  return _per_mode_stage_reduce(sub_rows, lambda s: s.mean())


def _per_mode_stage_stds(sub_rows):
  """
  Per-mode dashboard mae_stages holding the sample std (ddof=1) of the per-subtrial columns.

  Companion to _per_mode_stage_means with identical keys, so the aggregate dashboard can
  render each per-stage MAE as 'mean ± std'. A single-subtrial group's ddof=1 std is NaN and
  is reported as 0.0, matching the AGGREGATE_STD convention in _aggregate_subtrial_rows.

  Args:
    sub_rows (list[dict]): Per-subtrial x per-mode rows (from _aggregated_summary_rows).

  Returns:
    dict[str, dict]: {refine_mode: {stage: (mae_micro_std, mae_macro_std)}}. refine_mode may
      be None (no-refinement aggregate). Empty dict when sub_rows is empty.
  """
  def _std(s):
    v = s.std(ddof=1)
    return 0.0 if not np.isfinite(v) else float(v)
  return _per_mode_stage_reduce(sub_rows, _std)


def _extract_linear_bundle(data):
  """
  Return the projector training bundle from a pkl dict, if present.

  interpolation_similarity='linear', 'mlp' and 'procrustes' all write their bundle
  under the same 'linear_projector' key (historical naming kept for log/plot
  compat); a 'kind' field inside the bundle disambiguates them. The bundle is
  absent for runs where no projector was trained: num_anchors in {0, -1} or
  interpolation_similarity not in {'linear', 'mlp', 'procrustes'}.

  Args:
    data (dict): Deserialized pkl contents.

  Returns:
    dict | None: The bundle with keys 'config', 'norm_stats', 'best_epoch',
      'best_val_mse', 'ckpt_path', 'metrics', 'splits', 'kind' ('linear', 'mlp'
      or 'procrustes'), and optionally 'procrustes_params'. Returns None if the
      key is absent.
  """
  return data.get('linear_projector')


# ── plot functions ───────────────────────────────────────────────────────────

def plot_predictions_histogram(new_preds, old_preds, labels, out_dir,
                               run_label: str = '', axes=None,
                               new_name: str = 'Projected (new model)',
                               out_filename: str = 'predictions_histogram_projected_vs_old.png'):
  """
  Side-by-side histograms of new and old model predictions with KDE overlay.

  Bars inside the ground-truth label range are blue; bars outside are orange.
  Dotted vertical lines mark the label min/max. A KDE curve is overlaid on each
  panel.

  Args:
    new_preds  (np.ndarray): Shape (N,), new-model float predictions (the left panel;
      the projected/before-refinement stage by default, or refined when overridden).
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth labels.
    out_dir    (str): Directory where the plot is saved (ignored when axes provided).
    run_label  (str): Optional run identity string appended to plot titles.
    axes       (array-like[Axes] | None): Two pre-existing axes for dashboard
               embedding. When None a new figure is created and saved.
    new_name   (str): Label for the left (new-model) panel title.
    out_filename (str): Basename of the saved PNG (ignored when axes provided).
  """
  label_lo = float(labels.min())
  label_hi = float(labels.max())
  step     = 0.1
  suffix   = f' | {run_label}' if run_label else ''

  standalone = axes is None
  if standalone:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  else:
    fig = axes[0].figure

  for ax, preds, name in [
    (axes[0], new_preds, new_name),
    (axes[1], old_preds, 'Old model'),
  ]:
    pred_lo = float(preds.min())
    pred_hi = float(preds.max())
    lo = min(label_lo, pred_lo)
    hi = max(label_hi, pred_hi)

    bins              = np.arange(lo, hi + step, step)
    counts, edges     = np.histogram(preds, bins=bins)
    centers           = (edges[:-1] + edges[1:]) / 2.0
    bar_colors        = [
      '#DD8452' if (c < label_lo or c > label_hi) else '#4C72B0'
      for c in centers
    ]

    in_mask  = np.array([c == '#4C72B0' for c in bar_colors])
    oob_mask = ~in_mask

    ax.bar(centers[in_mask],  counts[in_mask],  width=step,
           color='#4C72B0', edgecolor='white', linewidth=0.6, alpha=0.75, label='In range')
    if oob_mask.any():
      ax.bar(centers[oob_mask], counts[oob_mask], width=step,
             color='#DD8452', edgecolor='white', linewidth=0.6, alpha=0.75, label='Out of range')

    if len(preds) > 1:
      kde    = stats.gaussian_kde(preds)
      x_fine = np.linspace(lo, hi, 500)
      ax.plot(x_fine, kde(x_fine) * len(preds) * step,
              color='#C44E52', linewidth=1.8, label='KDE')

    for val in (label_lo, label_hi):
      ax.axvline(val, linestyle=':', color='#555555', linewidth=1.5)

    ax.set_xlabel('Labels')
    ax.set_ylabel('Count')
    ax.set_title(f'Prediction distribution — {name}{suffix}')
    ax.set_xlim(lo - step, hi + step)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

  if standalone:
    plt.tight_layout()
    path = os.path.join(out_dir, out_filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_mae_per_class(new_preds, old_preds, labels, out_dir, run_label: str = ''):
  """
  Four separate single-plot figures of MAE per pain class: a bar chart and a
  box plot for each of the old model and the projected model.

  The bar chart shows mean MAE per class; the box plot shows the full per-sample
  absolute error distribution per class.

  Args:
    new_preds  (np.ndarray): Shape (N,), projected model predictions.
    old_preds  (np.ndarray): Shape (N,), old model predictions.
    labels     (np.ndarray): Shape (N,), ground-truth labels.
    out_dir    (str): Output directory.
    run_label  (str): Optional run identity string appended to plot titles.
  """
  labels_int = np.round(labels).astype(int)
  old_mae = _mae_per_group(old_preds, labels, labels_int)
  new_mae = _mae_per_group(new_preds, labels, labels_int)
  groups   = sorted(set(old_mae) | set(new_mae))
  old_vals = [old_mae.get(g, (float('nan'), 0))[0] for g in groups]
  new_vals = [new_mae.get(g, (float('nan'), 0))[0] for g in groups]

  _, old_raw = _raw_errors_per_group(old_preds, labels, labels_int)
  _, new_raw = _raw_errors_per_group(new_preds, labels, labels_int)

  suffix = f' | {run_label}' if run_label else ''

  for vals, raw_errors, color, name, prefix in [
    (old_vals, old_raw, 'steelblue',  'Old model',       'mae_per_class_old'),
    (new_vals, new_raw, 'darkorange', 'Projected (new)', 'mae_per_class_projected'),
  ]:
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_mae_bar(ax, groups, vals, 'MAE', f'MAE per pain class — {name}{suffix}', color)
    ax.set_xlabel('Labels')
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f'{prefix}_bar.png')
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {bar_path}')

    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_mae_box(
      ax, groups, raw_errors, 'MAE', color,
      title=f'MAE per pain class — {name}{suffix}',
    )
    ax.set_xlabel('Labels')
    plt.tight_layout()
    box_path = os.path.join(out_dir, f'{prefix}_box.png')
    fig.savefig(box_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {box_path}')


def plot_mae_improvement_per_class(new_preds, old_preds, labels, out_dir,
                                   run_label: str = '', ax=None):
  """
  Bar chart of MAE improvement (old_mae - new_mae) per pain class.

  Positive bars (the projected model lowered the error for that class) are
  green; negative bars (it got worse) are red. The signed difference is
  printed at each bar tip.

  Args:
    new_preds  (np.ndarray): Shape (N,), projected model predictions.
    old_preds  (np.ndarray): Shape (N,), old model predictions.
    labels     (np.ndarray): Shape (N,), ground-truth labels.
    out_dir    (str): Output directory (ignored when ax is provided).
    run_label  (str): Optional run identity string appended to the plot title.
    ax         (matplotlib.axes.Axes | None): Pre-existing axes for dashboard
               embedding. When None a new figure is created and saved.
  """
  labels_int = np.round(labels).astype(int)
  old_mae = _mae_per_group(old_preds, labels, labels_int)
  new_mae = _mae_per_group(new_preds, labels, labels_int)
  groups  = sorted(set(old_mae) | set(new_mae))
  diffs   = [
    old_mae.get(g, (float('nan'), 0))[0] - new_mae.get(g, (float('nan'), 0))[0]
    for g in groups
  ]
  suffix  = f' | {run_label}' if run_label else ''

  standalone = ax is None
  if standalone:
    fig, ax = plt.subplots(figsize=(14, 5))

  _draw_mae_improvement_bar(
    ax, groups, diffs, 'Labels',
    f'MAE improvement per pain class (old - new){suffix}',
  )

  if standalone:
    plt.tight_layout()
    path = os.path.join(out_dir, 'mae_improvement_per_class_projected_vs_old.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')

    # Companion 2-bar comparison: old vs projected MAE side by side per class.
    old_vals = [old_mae.get(g, (float('nan'), 0))[0] for g in groups]
    new_vals = [new_mae.get(g, (float('nan'), 0))[0] for g in groups]
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_grouped_mae_bar(
      ax, groups, old_vals, new_vals, 'Old model', 'Projected (new)',
      'steelblue', 'darkorange', 'MAE',
      f'MAE per pain class — old vs projected{suffix}',
    )
    ax.set_xlabel('Labels')
    plt.tight_layout()
    path = os.path.join(out_dir, 'mae_per_class_compare_projected_vs_old.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_mae_per_subject(new_preds, old_preds, labels, sample_ids, subject_map, out_dir,
                         run_label: str = '', new_name: str = 'Projected (new)',
                         new_filename: str = 'mae_per_subject_projected.png',
                         emit_old: bool = True):
  """
  Single-bar figures of MAE per subject for the new model (and, optionally, the old model).

  Args:
    new_preds   (np.ndarray): Shape (N,), new-model predictions (projected/before-
      refinement by default, or refined when overridden).
    old_preds   (np.ndarray): Shape (N,), old model predictions.
    labels      (np.ndarray): Shape (N,), ground-truth labels.
    sample_ids  (np.ndarray): Shape (N,), int sample IDs.
    subject_map (dict[int, int]): Mapping from sample_id to subject_id.
    out_dir     (str): Output directory.
    run_label   (str): Optional run identity string appended to plot titles.
    new_name    (str): Title label for the new-model bar.
    new_filename (str): Basename of the new-model PNG.
    emit_old    (bool): When True also (re)emit mae_per_subject_old.png. Set False for
      refined variants so the unchanged old-model bar is not rewritten.
  """
  subj_ids = np.array([subject_map.get(int(sid), -1) for sid in sample_ids])
  old_mae  = _mae_per_group(old_preds, labels, subj_ids)
  new_mae  = _mae_per_group(new_preds, labels, subj_ids)
  groups   = sorted(set(old_mae) | set(new_mae))
  old_vals = [old_mae.get(g, (float('nan'), 0))[0] for g in groups]
  new_vals = [new_mae.get(g, (float('nan'), 0))[0] for g in groups]
  suffix = f' | {run_label}' if run_label else ''

  series = [(new_vals, 'darkorange', new_name, new_filename)]
  if emit_old:
    series.insert(0, (old_vals, 'steelblue', 'Old model', 'mae_per_subject_old.png'))
  for vals, color, name, filename in series:
    fig, ax = plt.subplots(figsize=(12, 7))
    _single_bar(ax, groups, vals, 'MAE', f'MAE per subject — {name}{suffix}', color)
    ax.set_xlabel('Subject ID')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_mae_improvement_per_subject(new_preds, old_preds, labels, sample_ids,
                                     subject_map, out_dir, run_label: str = ''):
  """
  Bar chart of MAE improvement (old_mae - new_mae) per subject.

  Positive bars (the projected model lowered the error for that subject) are
  green; negative bars (it got worse) are red. The signed difference is
  printed at each bar tip.

  Args:
    new_preds   (np.ndarray): Shape (N,), projected model predictions.
    old_preds   (np.ndarray): Shape (N,), old model predictions.
    labels      (np.ndarray): Shape (N,), ground-truth labels.
    sample_ids  (np.ndarray): Shape (N,), int sample IDs.
    subject_map (dict[int, int]): Mapping from sample_id to subject_id.
    out_dir     (str): Output directory.
    run_label   (str): Optional run identity string appended to the plot title.
  """
  subj_ids = np.array([subject_map.get(int(sid), -1) for sid in sample_ids])
  old_mae  = _mae_per_group(old_preds, labels, subj_ids)
  new_mae  = _mae_per_group(new_preds, labels, subj_ids)
  groups   = sorted(set(old_mae) | set(new_mae))
  diffs    = [
    old_mae.get(g, (float('nan'), 0))[0] - new_mae.get(g, (float('nan'), 0))[0]
    for g in groups
  ]
  suffix   = f' | {run_label}' if run_label else ''

  fig, ax = plt.subplots(figsize=(12, 7))
  _draw_mae_improvement_bar(
    ax, groups, diffs, 'Subject ID',
    f'MAE improvement per subject (old - new){suffix}',
  )
  plt.tight_layout()
  path = os.path.join(out_dir, 'mae_improvement_per_subject_projected_vs_old.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_confusion_matrix_cross(new_preds, labels, out_dir, num_classes: int,
                                run_label: str = '', ax=None,
                                title: str = 'Confusion matrix',
                                out_filename: str = 'confusion_matrix_projected.png'):
  """
  Confusion matrix of rounded predictions vs ground-truth labels.

  Predictions are rounded to the nearest integer and clipped to [0, num_classes - 1].

  Args:
    new_preds    (np.ndarray): Shape (N,), float predictions for the stage being plotted.
    labels       (np.ndarray): Shape (N,), ground-truth float labels.
    out_dir      (str): Output directory (ignored when ax is provided).
    num_classes  (int): Number of distinct pain classes inferred from the labels.
    run_label    (str): Optional run identity string appended to the title.
    ax           (matplotlib.axes.Axes | None): Pre-existing axes for dashboard /
                 combined-figure embedding. When None a standalone figure is saved.
    title        (str): Title text (the pipeline stage / dataset / split). The run_label
                 suffix is appended automatically. Defaults to the legacy 'Confusion matrix'.
    out_filename (str): Basename of the saved PNG (ignored when ax is provided). Defaults to
                 the legacy 'confusion_matrix_projected.png'.
  """
  preds_int  = torch.tensor(_round_preds(new_preds, num_classes),                                    dtype=torch.long)
  labels_int = torch.tensor(np.clip(np.round(labels), 0, num_classes - 1).astype(np.int64), dtype=torch.long)
  cm = MulticlassConfusionMatrix(num_classes=num_classes)
  cm.update(preds_int, labels_int)
  cm_arr = cm.compute().cpu().numpy()[:num_classes, :num_classes].astype(int)
  # run_label goes on its own line so the (already multi-line) stage/dataset/split title
  # stays narrow enough to render without overlapping neighbouring panels.
  suffix = f'\n{run_label}' if run_label else ''

  standalone = ax is None
  if standalone:
    fig, ax = plt.subplots(figsize=(7, 5))

  sns.heatmap(
    cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax,
    linewidths=0.5, linecolor='lightgray', annot_kws={'size': 7},
  )
  ax.set_title(f'{title}{suffix}', fontsize=9, fontweight='bold')
  ax.set_xlabel('Predicted', fontsize=8)
  ax.set_ylabel('True', fontsize=8)
  ax.tick_params(axis='x', rotation=45, labelsize=7)
  ax.tick_params(axis='y', rotation=0,  labelsize=7)

  if standalone:
    fig.tight_layout()
    path = os.path.join(out_dir, out_filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_confusion_matrices_combined(panels, out_dir, run_label: str = '',
                                     out_filename: str = 'confusion_matrix_all_stages.png'):
  """
  Combined comparison figure of the pipeline-stage confusion matrices.

  Lays out a 2x3 grid: row 0 holds the three old-model-dataset (source-set) stages
  (original / after projection / after refinement) and row 1 holds the two
  new-model-dataset stages (original / after refinement). Each present panel is drawn
  by plot_confusion_matrix_cross with ax=. Panel titles are the short stage/dataset/split
  strings (the run identity lives only in the figure suptitle, so per-panel titles stay
  narrow); the separate per-stage PNGs carry the longer 'Confusion matrix — ...' titles.
  Panels with no data (e.g. refinement stages on a non-refinement run) render a blanked
  axis carrying the panel title plus a note, mirroring the dashboard's num_classes>15
  fallback.

  Args:
    panels       (list[dict]): One dict per grid cell, in row-major order (length up to 6;
      the unused row-1 third cell may be omitted). Each dict has:
        'title'       (str): Short panel title (stage / dataset / split), no run identity.
        'preds'       (np.ndarray | None): Shape (N,), predictions; None => blanked panel.
        'labels'      (np.ndarray | None): Shape (N,), ground-truth labels.
        'num_classes' (int | None):        Class count for this panel.
        'note'        (str, optional):     Reason shown on a blanked panel.
    out_dir      (str): Output directory.
    run_label    (str): Optional run identity string shown in the figure suptitle.
    out_filename (str): Basename of the saved PNG.
  """
  fig = plt.figure(figsize=(18, 11))
  gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)
  positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
  suffix = f' | {run_label}' if run_label else ''

  for panel, (r, c) in zip(panels, positions):
    ax = fig.add_subplot(gs[r, c])
    preds       = panel.get('preds')
    labels      = panel.get('labels')
    num_classes = panel.get('num_classes')
    drawable = (
      preds is not None and labels is not None
      and num_classes is not None and num_classes <= 15
    )
    if drawable:
      # run_label='' here: the per-panel title carries only the stage/dataset/split.
      plot_confusion_matrix_cross(
        preds, labels, out_dir, num_classes,
        run_label='', ax=ax, title=panel['title'],
      )
    else:
      note = panel.get('note', 'not available')
      if num_classes is not None and num_classes > 15:
        note = f'num_classes={num_classes} > 15'
      ax.axis('off')
      ax.set_title(panel['title'], fontsize=9, fontweight='bold')
      ax.text(0.5, 0.5, f'({note})', ha='center', va='center', fontsize=10)

  fig.suptitle(f'Confusion matrices across pipeline stages{suffix}',
               fontsize=13, fontweight='bold')
  path = os.path.join(out_dir, out_filename)
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')


def plot_umap(embeddings, labels, sample_ids, subject_map, out_dir, run_label: str = '',
              filename_suffix: str = ''):
  """
  Compute UMAP on projected embeddings and plot colored by label and by subject.

  Args:
    embeddings      (np.ndarray): Shape (N, D), projected embedding matrix.
    labels          (np.ndarray): Shape (N,), ground-truth labels.
    sample_ids      (np.ndarray): Shape (N,), int sample IDs.
    subject_map     (dict[int, int]): Mapping from sample_id to subject_id.
    out_dir         (str): Output directory.
    run_label       (str): Optional run identity string appended to plot titles.
    filename_suffix (str): Optional suffix appended to the PNG basename (before
      '.png'), e.g. '_refined' to distinguish the after-refinement variant.
  """
  print('Computing UMAP for projected embeddings...')
  reduced  = _compute_umap(embeddings)
  subj_ids = np.array([subject_map.get(int(sid), -1) for sid in sample_ids])
  n_subj   = len(np.unique(subj_ids))
  cmap_subj = 'tab10' if n_subj <= 10 else ('tab20' if n_subj <= 20 else 'nipy_spectral')
  suffix = f' | {run_label}' if run_label else ''

  v_min, v_max = float(labels.min()), float(labels.max())
  fig, axes = plt.subplots(1, 2, figsize=(20, 8))
  sc = axes[0].scatter(
    reduced[:, 0], reduced[:, 1], c=labels,
    cmap='jet', vmin=v_min, vmax=v_max, s=10, alpha=0.7,
  )
  plt.colorbar(sc, ax=axes[0], label='Labels')
  axes[0].set_title(f'UMAP — projected embeddings (by label){suffix}')
  axes[0].set_xlabel('UMAP 1')
  axes[0].set_ylabel('UMAP 2')
  plot_reducted_embeddings(
    reduced_embeddings=reduced, labels=subj_ids,
    output_folder=out_dir, title=f'UMAP — projected embeddings (by subject){suffix}',
    group_by='subjects', cmap=cmap_subj, save_plot=False, ax=axes[1], reduction_name='UMAP',
  )
  plt.tight_layout()
  path = os.path.join(out_dir, f'umap_all{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_umap_space_comparison(old_embeddings, projected_embeddings, labels, sample_ids,
                               old_sample_ids, subject_map, out_dir, stage_title, filename_suffix,
                               run_label: str = ''):
  """Compare aligned source-test samples before and after projection in a 2x2 UMAP."""
  if old_embeddings is None or projected_embeddings is None:
    print('[WARN] UMAP space comparison: embeddings unavailable — skipped.')
    return
  old_embeddings = np.asarray(old_embeddings, dtype=np.float32)
  projected_embeddings = np.asarray(projected_embeddings, dtype=np.float32)
  labels = np.asarray(labels, dtype=np.float32).reshape(-1)
  sample_ids = np.asarray(sample_ids).reshape(-1)
  old_sample_ids = np.asarray(old_sample_ids).reshape(-1)
  counts = (
    len(old_embeddings), len(projected_embeddings), len(labels),
    len(sample_ids), len(old_sample_ids),
  )
  if len(set(counts)) != 1:
    print(f'[WARN] UMAP space comparison: sample count mismatch {counts} — skipped.')
    return
  if not np.array_equal(old_sample_ids, sample_ids):
    print('[WARN] UMAP space comparison: sample ID mismatch — skipped.')
    return

  print(f'Computing old-vs-projected UMAP comparison ({stage_title})...')
  try:
    old_reduced = _compute_umap(old_embeddings)
    projected_reduced = _compute_umap(projected_embeddings)
  except Exception as exc:
    print(f'[WARN] UMAP space comparison failed: {exc} — skipped.')
    return
  subject_ids = np.array([subject_map.get(int(sid), -1) for sid in sample_ids])
  n_subjects = len(np.unique(subject_ids))
  subject_cmap = (
    'tab10' if n_subjects <= 10 else ('tab20' if n_subjects <= 20 else 'nipy_spectral')
  )

  fig, axes = plt.subplots(2, 2, figsize=(20, 16))
  spaces = (
    ('Old model feature space', old_reduced),
    ('Projected new-model feature space', projected_reduced),
  )
  for col, (space_title, reduced) in enumerate(spaces):
    label_scatter = axes[0, col].scatter(
      reduced[:, 0], reduced[:, 1], c=labels, cmap='jet',
      vmin=float(labels.min()), vmax=float(labels.max()), s=10, alpha=0.7,
    )
    fig.colorbar(label_scatter, ax=axes[0, col], label='Pain label')
    axes[0, col].set_title(f'{space_title} — by pain label')

    subject_scatter = axes[1, col].scatter(
      reduced[:, 0], reduced[:, 1], c=subject_ids, cmap=subject_cmap,
      vmin=float(subject_ids.min()), vmax=float(subject_ids.max()), s=10, alpha=0.7,
    )
    fig.colorbar(subject_scatter, ax=axes[1, col], label='Subject')
    axes[1, col].set_title(f'{space_title} — by subject')

  for ax in axes.flat:
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
  suffix = f' | {run_label}' if run_label else ''
  fig.suptitle(f'Old-model test embeddings: old vs projected space — {stage_title}{suffix}',
               fontsize=14, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.96))
  path = os.path.join(out_dir, f'umap_space_comparison{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def _umap_comparison_embedding(mode, projected_embedding, refined_embeddings_by_mode):
  """Return the mode's embedding matrix; linear-only refinement leaves it unchanged."""
  return (projected_embedding if mode == 'linear_only'
          else refined_embeddings_by_mode.get(mode))


def plot_umap_split_impact(projected_emb, projected_labels, split_emb, split_labels,
                           split_name, out_dir, run_label: str = '', filename_suffix: str = '',
                           new_dataset: str = None, src_dataset: str = None):
  """
  Overlay the projected embeddings on the new model's real <split> embeddings in UMAP.

  Three panels, all colored by label (jet):
    1. UMAP fit on BOTH (split + projected), all points shown. Projected points use a
       distinct marker ('x') from the split points ('o') so the overlap/impact is visible.
    2. UMAP fit on BOTH, but only the projected points are drawn.
    3. UMAP fit on the split set alone.

  The split set arrives already subsampled (the rows are subsampled to SPLIT_SUBSAMPLE_FRAC
  before extraction in _load_split_embeddings); this function does no further subsampling.

  Args:
    projected_emb    (np.ndarray): Shape (P, D), projected new-model embeddings.
    projected_labels (np.ndarray): Shape (P,),  labels of the projected samples.
    split_emb        (np.ndarray): Shape (S, D), real new-model embeddings of the split.
    split_labels     (np.ndarray): Shape (S,),  labels of the split samples.
    split_name       (str): Split identity ('train'/'val'/'test'), used in titles/filename.
    out_dir          (str): Output directory.
    run_label        (str): Optional run identity string appended to plot titles.
    filename_suffix  (str): Optional suffix appended to the PNG basename (before
      '.png'), e.g. '_refined' to distinguish the after-refinement variant.
    new_dataset      (str | None): Resolved new/target dataset name (the real split's
      dataset). Names the 'real' cloud so it is not misattributed to the source; None ⇒
      generic 'new model'.
    src_dataset      (str | None): Resolved old/source dataset name (the projected cloud's
      origin). Names the 'projected' cloud; None ⇒ generic 'source'.
  """
  new_ds_lbl = new_dataset or 'new model'
  src_ds_lbl = src_dataset or 'source'
  projected_emb = np.asarray(projected_emb, dtype=np.float32)
  split_sub     = np.asarray(split_emb,     dtype=np.float32)
  projected_labels = np.asarray(projected_labels, dtype=np.float32).reshape(-1)
  split_lab        = np.asarray(split_labels,     dtype=np.float32).reshape(-1)

  if projected_emb.shape[1] != split_sub.shape[1]:
    print(f'[WARN] split-impact UMAP: dim mismatch projected={projected_emb.shape[1]} '
          f'vs {split_name}={split_sub.shape[1]} — skipped.')
    return

  # UMAP needs more samples than its default neighborhood; guard tiny splits.
  if split_sub.shape[0] < 5:
    print(f'[WARN] split-impact UMAP: only {split_sub.shape[0]} {split_name} points '
          f'— too few; skipped.')
    return

  print(f'Computing split-impact UMAP ({split_name}: {split_sub.shape[0]} pts, '
        f'projected: {projected_emb.shape[0]} pts)...')

  # --- Fit UMAP on the joint set and on the split alone ---
  combined = np.vstack([split_sub, projected_emb])
  is_proj  = np.zeros(combined.shape[0], dtype=bool)
  is_proj[split_sub.shape[0]:] = True
  reduced_both  = _compute_umap(combined)
  reduced_split = _compute_umap(split_sub)

  all_labels   = np.concatenate([split_lab, projected_labels])
  v_min, v_max = float(all_labels.min()), float(all_labels.max())
  suffix = f' | {run_label}' if run_label else ''
  frac_note = f'split subsample={SPLIT_SUBSAMPLE_FRAC:g}'

  fig, axes = plt.subplots(1, 3, figsize=(30, 8))

  # Panel 1: fit on both, all points, marker distinguishes group.
  axes[0].scatter(
    reduced_both[~is_proj, 0], reduced_both[~is_proj, 1], c=split_lab,
    cmap='jet', vmin=v_min, vmax=v_max, s=12, alpha=0.7, marker='o',
    label=f'{split_name} (real, {new_ds_lbl})',
  )
  sc0 = axes[0].scatter(
    reduced_both[is_proj, 0], reduced_both[is_proj, 1], c=projected_labels,
    cmap='jet', vmin=v_min, vmax=v_max, s=28, alpha=0.8, marker='x',
    label=f'projected ({src_ds_lbl})',
  )
  plt.colorbar(sc0, ax=axes[0], label='Labels')
  axes[0].legend(loc='best', framealpha=0.9)
  axes[0].set_title(
    f'UMAP fit on BOTH (projected {src_ds_lbl} + {new_ds_lbl} {split_name}) — all points\n'
    f'colored by label · {frac_note}{suffix}'
  )

  # Panel 2: fit on both, projected points only.
  sc1 = axes[1].scatter(
    reduced_both[is_proj, 0], reduced_both[is_proj, 1], c=projected_labels,
    cmap='jet', vmin=v_min, vmax=v_max, s=20, alpha=0.8, marker='x',
  )
  plt.colorbar(sc1, ax=axes[1], label='Labels')
  axes[1].set_title(
    f'UMAP fit on BOTH (projected {src_ds_lbl} + {new_ds_lbl} {split_name}) — projected only\n'
    f'colored by label · {frac_note}{suffix}'
  )
  axes[1].set_xlim(axes[0].get_xlim())
  axes[1].set_ylim(axes[0].get_ylim())

  # Panel 3: fit on the split alone.
  sc2 = axes[2].scatter(
    reduced_split[:, 0], reduced_split[:, 1], c=split_lab,
    cmap='jet', vmin=v_min, vmax=v_max, s=12, alpha=0.7, marker='o',
  )
  plt.colorbar(sc2, ax=axes[2], label='Labels')
  axes[2].set_title(
    f'UMAP fit on {new_ds_lbl} {split_name} only\n'
    f'colored by label · {frac_note}{suffix}'
  )

  for ax in axes:
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
  plt.tight_layout()
  path = os.path.join(out_dir, f'umap_split_impact_{split_name}{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_anchor_weights(weights, out_dir, run_label: str = ''):
  """
  Two-subplot figure: per-sample weight entropy histogram and top-20 anchor usage bar.

  Args:
    weights    (np.ndarray): Shape (N, K), interpolation weights from projection step.
    out_dir    (str): Output directory.
    run_label  (str): Optional run identity string appended to plot titles.
  """
  if weights.shape[1] == 0:
    print('[plot_anchor_weights] Skipped — num_anchors=0 or -1 (no interpolation weights)')
    return
  entropy  = _weight_entropy(weights)
  mean_w   = weights.mean(axis=0)
  top20_idx = np.argsort(mean_w)[-20:][::-1]
  top20_w   = mean_w[top20_idx]
  suffix = f' | {run_label}' if run_label else ''

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))

  axes[0].hist(entropy, bins=30, color='slateblue', alpha=0.8, edgecolor='black')
  axes[0].axvline(float(entropy.mean()), color='red', linestyle='--',
                  label=f'mean = {entropy.mean():.2f}')
  axes[0].set_xlabel('Weight entropy (nats)')
  axes[0].set_ylabel('Count')
  axes[0].set_title(f'Per-sample weight entropy distribution{suffix}')
  axes[0].legend()

  axes[1].bar(range(20), top20_w, color='teal', alpha=0.85)
  axes[1].set_xticks(range(20))
  axes[1].set_xticklabels([str(i) for i in top20_idx], rotation=45, ha='right', fontsize=8)
  axes[1].set_xlabel('Anchor index')
  axes[1].set_ylabel('Mean weight')
  axes[1].set_title(f'Top-20 most-used anchors (by mean weight across all samples){suffix}')

  plt.tight_layout()
  path = os.path.join(out_dir, 'anchor_weights.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_weight_rank_distribution(weights, out_dir, run_label: str = '', top_n: int = 30):
  """
  Box plot of interpolation-weight values at each rank position (most → least impactful).

  For each sample the K weights are sorted descending; the value at rank j is the
  j-th largest weight for that sample. Box plots at each rank position show the
  distribution of that rank's weight value across all N samples, revealing whether
  the projection is dominated by one anchor (concentrated) or spread across many (flat).
  Two panels are produced side-by-side: linear y-scale (left) and symlog y-scale
  (right), to expose both the dominant weights and the tail structure.

  Args:
    weights   (np.ndarray): Shape (N, K), interpolation weights per sample.
    out_dir   (str): Directory where the plot is saved.
    run_label (str): Optional run identity string appended to plot titles.
    top_n     (int): Number of rank positions to display (capped at K). Default 30.
  """
  K = weights.shape[1]
  if K == 0:
    print('[plot_weight_rank_distribution] Skipped — no anchors')
    return

  n_show   = min(top_n, K)
  sorted_w = np.sort(np.abs(weights), axis=1)[:, ::-1][:, :n_show]
  box_data = [sorted_w[:, j] for j in range(n_show)]

  suffix = f' | {run_label}' if run_label else ''
  title  = f'Weight rank distribution (top {n_show} of {K}){suffix}'
  color  = 'slateblue'
  face_rgba    = list(mcolors.to_rgba(color))
  face_rgba[3] = 0.5

  fig, axes = plt.subplots(1, 2, figsize=(max(14, n_show * 0.5), 6))
  for ax, log_y in [(axes[0], False), (axes[1], True)]:
    bp = ax.boxplot(
      box_data,
      positions=np.arange(n_show),
      widths=0.6,
      patch_artist=True,
      showfliers=False,
      medianprops=dict(color='#222222', linewidth=1.5),
      whiskerprops=dict(linewidth=0.8),
      capprops=dict(linewidth=0.8),
    )
    for patch in bp['boxes']:
      patch.set_facecolor(face_rgba)
      patch.set_edgecolor(color)
    ax.set_xticks(np.arange(n_show))
    ax.set_xticklabels([str(j) for j in range(n_show)], rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('Rank (0 = most impactful)')
    ax.set_ylabel('|Weight| value')
    ax.set_title(f'{title} — {"symlog" if log_y else "linear"}')
    ax.grid(axis='y', alpha=0.3)
    if log_y:
      ax.set_yscale('symlog', linthresh=1e-4)

  plt.tight_layout()
  path = os.path.join(out_dir, 'weight_rank_distribution.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_anchor_umap(old_anchors_emb, new_anchors_emb, anchor_labels, out_dir, run_label: str = ''):
  """
  Side-by-side UMAP of old and new anchor embeddings, colored by label.

  Each embedding set is reduced independently (dimensions differ), so they share
  the same color scale but not the same coordinate space.

  Args:
    old_anchors_emb (np.ndarray): Shape (K, D_old), old model anchor embeddings.
    new_anchors_emb (np.ndarray): Shape (K, D_new), new model anchor embeddings.
    anchor_labels   (np.ndarray): Shape (K,), labels for anchors.
    out_dir         (str): Output directory.
    run_label       (str): Optional run identity string appended to plot titles.
  """
  print('Computing UMAP for anchor embeddings...')
  reduced_old = _compute_umap(old_anchors_emb)
  reduced_new = _compute_umap(new_anchors_emb)
  suffix = f' | {run_label}' if run_label else ''

  v_min, v_max = float(anchor_labels.min()), float(anchor_labels.max())
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))
  for ax, reduced, title in [
    (axes[0], reduced_old, 'Old model anchor space'),
    (axes[1], reduced_new, 'New (projected) anchor space'),
  ]:
    sc = ax.scatter(
      reduced[:, 0], reduced[:, 1], c=anchor_labels,
      cmap='jet', vmin=v_min, vmax=v_max, s=60, alpha=0.85,
      edgecolors='black', linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax, label='Labels')
    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

  plt.suptitle(f'Anchor embeddings in UMAP space — old vs new model{suffix}')
  plt.tight_layout()
  path = os.path.join(out_dir, 'anchor_umap.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_anchor_norm_comparison(old_anchors_emb, new_anchors_emb, anchor_labels,
                                out_dir, run_label: str = ''):
  """
  Compare the L2 norm of each anchor between the old and new model spaces and
  save the per-anchor data as a CSV.

  Three-panel figure (1 × 3):
    Left:   Scatter — old_norm (x) vs new_norm (y), one point per anchor,
            colored by pain class, identity line for reference.
    Middle: Overlaid histograms — old_norm vs new_norm distributions.
    Right:  Histogram of the ratio new_norm / old_norm; vertical line at 1.

  CSV saved alongside the plot: anchor_norm_comparison.csv, with columns
  anchor_idx, label, old_norm, new_norm, ratio, delta.

  Args:
    old_anchors_emb (np.ndarray): Shape (K, D_old), old model anchor embeddings.
    new_anchors_emb (np.ndarray): Shape (K, D_new), new model anchor embeddings.
    anchor_labels   (np.ndarray): Shape (K,), label per anchor.
    out_dir         (str): Output directory.
    run_label       (str): Optional run identity string appended to plot titles.
  """
  old_emb = np.asarray(old_anchors_emb, dtype=np.float32)
  new_emb = np.asarray(new_anchors_emb, dtype=np.float32)
  labels  = np.asarray(anchor_labels,   dtype=np.float32)

  old_norm = np.linalg.norm(old_emb, axis=1)
  new_norm = np.linalg.norm(new_emb, axis=1)
  ratio    = new_norm / np.clip(old_norm, 1e-12, None)
  delta    = new_norm - old_norm

  suffix = f' | {run_label}' if run_label else ''

  # ── CSV ──────────────────────────────────────────────────────────────────────
  df_csv = pd.DataFrame({
    'anchor_idx': np.arange(len(old_norm)),
    'label':      labels,
    'old_norm':   old_norm,
    'new_norm':   new_norm,
    'ratio':      ratio,
    'delta':      delta,
  })
  csv_path = os.path.join(out_dir, 'anchor_norm_comparison.csv')
  df_csv.to_csv(csv_path, index=False)
  print(f'Saved: {csv_path}')

  # ── figure ───────────────────────────────────────────────────────────────────
  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  # Panel 0: scatter old_norm vs new_norm, colored by pain level (continuous jet)
  ax = axes[0]
  v_min, v_max = float(labels.min()), float(labels.max())
  sc = ax.scatter(
    old_norm, new_norm,
    c=labels, cmap='jet', vmin=v_min, vmax=v_max,
    s=12, alpha=0.5, edgecolors='none',
  )
  plt.colorbar(sc, ax=ax, label='Labels')
  lo = float(min(old_norm.min(), new_norm.min()))
  hi = float(max(old_norm.max(), new_norm.max()))
  ax.plot([lo, hi], [lo, hi], '--', color='black', linewidth=0.9, alpha=0.7, label='y = x')
  ax.set_xlabel('old model ||anchor||')
  ax.set_ylabel('new model ||anchor||')
  ax.set_title(f'Anchor L2 norm — old vs new{suffix}')
  ax.legend(fontsize=8, loc='best', framealpha=0.7)
  ax.grid(alpha=0.3)

  # Panel 1: overlaid histograms of old_norm and new_norm
  ax = axes[1]
  bins = min(80, max(20, len(old_norm) // 30))
  ax.hist(old_norm, bins=bins, alpha=0.55, color='#2ca02c', label='old model', edgecolor='white')
  ax.hist(new_norm, bins=bins, alpha=0.55, color='#d62728', label='new model', edgecolor='white')
  ax.axvline(float(old_norm.mean()), color='#2ca02c', linestyle='--', linewidth=1.4,
             label=f'mean old = {old_norm.mean():.2f}')
  ax.axvline(float(new_norm.mean()), color='#d62728', linestyle='--', linewidth=1.4,
             label=f'mean new = {new_norm.mean():.2f}')
  ax.set_xlabel('L2 norm')
  ax.set_ylabel('Count')
  ax.set_title(f'Anchor norm distributions{suffix}')
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  # Panel 2: histogram of ratio new_norm / old_norm
  ax = axes[2]
  finite_ratio = ratio[np.isfinite(ratio)]
  ax.hist(finite_ratio, bins=bins, color='#9467bd', alpha=0.8, edgecolor='white')
  ax.axvline(1.0, color='black', linestyle='--', linewidth=1.2, label='ratio = 1')
  ax.axvline(float(finite_ratio.mean()), color='#e377c2', linestyle='--', linewidth=1.4,
             label=f'mean = {finite_ratio.mean():.3f}')
  ax.set_xlabel('new_norm / old_norm')
  ax.set_ylabel('Count')
  ax.set_title(f'Anchor norm ratio (new / old){suffix}')
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)

  plt.tight_layout()
  path = os.path.join(out_dir, 'anchor_norm_comparison.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_prediction_scatter(new_preds, old_preds, labels, out_dir,
                            run_label: str = '', axes=None,
                            new_name: str = 'Projected (new model)',
                            out_filename: str = 'prediction_scatter_projected_vs_old.png'):
  """
  Side-by-side scatter of model predictions vs ground-truth labels.

  Left panel: new model. Right panel: old model.
  Each panel includes a red y=x reference line and a text annotation with
  MAE and CCC.

  Args:
    new_preds  (np.ndarray): Shape (N,), new-model float predictions (projected/before-
      refinement by default, or refined when overridden).
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth labels.
    out_dir    (str): Directory where the plot is saved (ignored when axes provided).
    run_label  (str): Optional run identity string appended to plot titles.
    axes       (array-like[Axes] | None): Two pre-existing axes for dashboard
               embedding. When None a new figure is created and saved.
    new_name   (str): Title label for the left (new-model) panel.
    out_filename (str): Basename of the saved PNG (ignored when axes provided).
  """
  suffix     = f' | {run_label}' if run_label else ''
  standalone = axes is None
  if standalone:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  else:
    fig = axes[0].figure

  for ax, preds, name in [
    (axes[0], new_preds, new_name),
    (axes[1], old_preds, 'Old model'),
  ]:
    mae_val = float(np.mean(np.abs(preds - labels)))
    ccc_val = float(concordance_ccc(labels, preds))

    ax.scatter(labels, preds, alpha=0.45, s=18, color='#4C72B0',
               edgecolors='white', linewidths=0.3)

    lo = float(min(labels.min(), preds.min())) - 0.5
    hi = float(max(labels.max(), preds.max())) + 0.5
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.2, label='y = x')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.text(0.04, 0.95, f'MAE={mae_val:.3f}\nCCC={ccc_val:.3f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax.set_title(f'Predicted vs Ground Truth — {name}\n{suffix}', fontsize=10, fontweight='bold')
    ax.set_xlabel('True label', fontsize=9)
    ax.set_ylabel('Predicted value', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

  if standalone:
    plt.tight_layout()
    path = os.path.join(out_dir, out_filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_prediction_by_class_boxplot(new_preds, old_preds, labels, out_dir, run_label: str = '',
                                     new_name: str = 'Projected (new model)',
                                     out_filename: str = 'prediction_by_class_boxplot_projected_vs_old.png'):
  """
  1×2 box plot of raw float predictions grouped by ground-truth pain class.

  Left panel: new model. Right panel: old model.
  A short red reference segment at y = class_id marks perfect calibration.

  Args:
    new_preds  (np.ndarray): Shape (N,), new-model float predictions (projected/before-
      refinement by default, or refined when overridden).
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth labels.
    out_dir    (str): Output directory.
    run_label  (str): Optional run identity string appended to plot titles.
    new_name   (str): Title label for the left (new-model) panel.
    out_filename (str): Basename of the saved PNG.
  """
  class_ids  = sorted(int(c) for c in np.unique(np.round(labels).astype(int)))
  labels_int = np.round(labels).astype(int)
  suffix     = f' | {run_label}' if run_label else ''
  x          = np.arange(len(class_ids))
  margin     = 0.6
  xlim       = (x[0] - margin, x[-1] + margin)

  color      = '#4C72B0'
  ref_color  = '#C44E52'
  face_rgba  = list(mcolors.to_rgba(color))
  face_rgba[3] = 0.5

  box_kwargs = dict(
    positions=x, widths=0.5, patch_artist=True, showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                    markerfacecolor=color, markeredgecolor='none'),
    medianprops=dict(color='#222222', linewidth=1.5),
    whiskerprops=dict(linewidth=1.0),
    capprops=dict(linewidth=1.0),
  )

  fig, axes = plt.subplots(1, 2, figsize=(12, 7))

  for ax, preds, model_name in [
    (axes[0], new_preds, new_name),
    (axes[1], old_preds, 'Old model'),
  ]:
    box_data = []
    for cid in class_ids:
      mask = labels_int == cid
      arr  = preds[mask].astype(np.float64)
      box_data.append(arr if arr.size > 0 else np.array([np.nan]))

    bp = ax.boxplot(box_data, **box_kwargs)
    for patch in bp['boxes']:
      patch.set_facecolor(face_rgba)
      patch.set_edgecolor(color)

    for i, cid in enumerate(class_ids):
      ax.plot([x[i] - 0.38, x[i] + 0.38], [cid, cid],
              color=ref_color, linewidth=2.0, zorder=5)

    ax.set_xlim(*xlim)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in class_ids], rotation=45, ha='right', fontsize=8)
    ax.set_title(f'{model_name}{suffix}', fontsize=10, fontweight='bold')
    ax.set_xlabel('True pain class', fontsize=9)
    ax.set_ylabel('Predicted value', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

  plt.tight_layout()
  path = os.path.join(out_dir, out_filename)
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


# ── linear-projector training-diagnostics plots ──────────────────────────────

def _projector_curve_arrays(metrics):
  """
  Convert metrics['train']/metrics['val'] lists into per-key numpy arrays.

  Args:
    metrics (dict): Bundle from linear_projector['metrics']. Expected keys
      'train' and 'val', each a list of dicts with at least 'epoch', 'mse',
      'cos' (and 'mae' when produced by the updated training script).

  Returns:
    tuple[dict, dict, bool]:
      - train arrays:  {'epoch': (E,), 'mse': (E,), 'mae': (E,) or None, 'cos': (E,)}
      - val   arrays:  same shape
      - has_mae: True iff every train/val entry carries an 'mae' key.
  """
  def _stack(rows, key):
    return np.asarray([r[key] for r in rows], dtype=np.float32) if rows else np.zeros(0, dtype=np.float32)

  train_rows = metrics.get('train', []) or []
  val_rows   = metrics.get('val',   []) or []
  has_mae    = bool(train_rows) and bool(val_rows) \
               and all('mae' in r for r in train_rows) \
               and all('mae' in r for r in val_rows)

  tr = {
    'epoch': _stack(train_rows, 'epoch'),
    'mse':   _stack(train_rows, 'mse'),
    'cos':   _stack(train_rows, 'cos'),
    'mae':   _stack(train_rows, 'mae') if has_mae else None,
  }
  va = {
    'epoch': _stack(val_rows, 'epoch'),
    'mse':   _stack(val_rows, 'mse'),
    'cos':   _stack(val_rows, 'cos'),
    'mae':   _stack(val_rows, 'mae') if has_mae else None,
  }
  return tr, va, has_mae


def _plot_metric_curve(ax, tr_epochs, tr_vals, va_epochs, va_vals,
                      best_epoch, test_value, ylabel, title, log_y=False):
  """
  Draw a single train/val curve plus a star at the test metric of the saved
  best checkpoint.

  The star sits at (best_epoch, test_value) — i.e. the test-set evaluation of
  the checkpoint chosen by val MSE. It is intentionally NOT drawn at the val
  value at best_epoch: that would conflate the selection criterion with the
  reported generalization performance.

  Args:
    ax         (matplotlib.axes.Axes): Axes to draw on.
    tr_epochs  (np.ndarray): Training-epoch indices, shape (E,).
    tr_vals    (np.ndarray): Training-metric values, shape (E,).
    va_epochs  (np.ndarray): Validation-epoch indices, shape (E,).
    va_vals    (np.ndarray): Validation-metric values, shape (E,).
    best_epoch (int):       Epoch index (1-based) of the saved checkpoint.
    test_value (float | None): Test-set metric for this curve at the best
      checkpoint. When None or non-finite, no star is drawn (we deliberately
      do not fall back to the val value).
    ylabel     (str):       Y-axis label.
    title      (str):       Subplot title.
    log_y      (bool):      If True, use a symlog y-scale (useful for MSE).
  """
  ax.plot(tr_epochs, tr_vals, '-', color='#1f77b4', label='train', linewidth=1.5)
  ax.plot(va_epochs, va_vals, '-', color='#d62728', label='val',   linewidth=1.5)

  if (best_epoch is not None and best_epoch > 0
      and test_value is not None and np.isfinite(test_value)):
    ax.scatter([best_epoch], [float(test_value)], marker='*', s=180,
               color='#2ca02c', edgecolor='black', linewidth=0.7,
               zorder=5,
               label=f'test @ best ep {best_epoch} = {float(test_value):.4f}')

  ax.set_xlabel('Epoch')
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(alpha=0.3)
  ax.legend(loc='best', fontsize=8)
  if log_y:
    ax.set_yscale('symlog', linthresh=1e-4)


def _format_projector_config_text(linear_bundle):
  """
  Build a human-readable, monospace-friendly summary of the projector run.

  Includes the LINEAR_PROJECTOR_CONFIG hyperparameters, derived I/O shapes,
  best epoch / val MSE, and the one-shot test metrics.

  Args:
    linear_bundle (dict): Output of _extract_linear_bundle.

  Returns:
    str: Multi-line text block.
  """
  cfg    = linear_bundle.get('config', {}) or {}
  metrics_test = (linear_bundle.get('metrics') or {}).get('test') or {}
  splits = linear_bundle.get('splits') or {}
  norm   = linear_bundle.get('norm_stats') or {}

  d_new = None
  if 'train' in splits and splits['train'].get('projected') is not None:
    d_new = int(np.asarray(splits['train']['projected']).shape[1])
  d_old = None
  if norm.get('old_mean') is not None:
    d_old = int(np.asarray(norm['old_mean']).shape[-1])

  kind = (linear_bundle.get('kind') or 'linear').lower()
  if kind == 'procrustes':
    header = '── PROCRUSTES (closed form) ──'
    cfg_keys = ('normalize_embeddings', 'split_ratios', 'device')
  else:
    header = '── MLP_PROJECTOR_CONFIG ──' if kind == 'mlp' else '── LINEAR_PROJECTOR_CONFIG ──'
    cfg_keys = ('lr', 'batch_size', 'optimizer', 'weight_decay', 'epochs',
                'normalize_embeddings', 'loss', 'split_ratios', 'device',
                'num_workers')
    if kind == 'mlp':
      cfg_keys = ('mlp_activation',) + cfg_keys
  rows = [header]
  for k in cfg_keys:
    if k in cfg:
      rows.append(f'{k:<22s} {cfg[k]}')
  proc = linear_bundle.get('procrustes_params') or {}
  if proc:
    rows.append('')
    rows.append('── procrustes params ──')
    if 'scale' in proc:
      rows.append(f'scale                  {float(proc["scale"]):.6f}')
    if 'R' in proc and proc['R'] is not None:
      rows.append(f'R.shape                {tuple(np.asarray(proc["R"]).shape)}')
    if 'sigma' in proc and proc['sigma'] is not None:
      sig = np.asarray(proc['sigma'])
      rows.append(f'sigma (top 5)          {np.round(sig[:5], 4).tolist()}')

  rows.append('')
  rows.append('── samples ──')
  split_mode = linear_bundle.get('split_mode')
  if split_mode is not None:
    rows.append(f'split_mode             {split_mode}')
  for name in ('train', 'val', 'test'):
    split = splits.get(name) or {}
    arr = (split.get('labels')
           if split.get('labels') is not None
           else split.get('projected')
           if split.get('projected') is not None
           else split.get('sample_ids'))
    if arr is not None:
      rows.append(f'n_{name:<20s} {int(np.asarray(arr).shape[0])}')

  rows.append('')
  rows.append('── shapes ──')
  rows.append(f'd_old → d_new          {d_old} → {d_new}')

  rows.append('')
  sel_name     = (linear_bundle.get('best_val_metric_name') or 'mse').lower()
  sel_val      = linear_bundle.get('best_val_metric',
                                   linear_bundle.get('best_val_mse'))
  best_epoch   = linear_bundle.get('best_epoch')
  best_val_mse = linear_bundle.get('best_val_mse')
  rows.append(f'── best (val {sel_name.upper()}) ──')
  rows.append(f'best_epoch             {best_epoch}')
  if sel_val is not None:
    rows.append(f'best_val_{sel_name:<13s} {float(sel_val):.6f}')
  # Always also show MSE for cross-run comparability when the selection metric
  # was something else.
  if sel_name != 'mse' and best_val_mse is not None:
    rows.append(f'best_val_mse           {float(best_val_mse):.6f}')

  if metrics_test:
    rows.append('')
    rows.append('── test (best ckpt) ──')
    for k in ('mse', 'mae', 'cos'):
      if k in metrics_test:
        rows.append(f'test_{k:<17s} {float(metrics_test[k]):.6f}')

  return '\n'.join(rows)


def plot_projector_training_curves(linear_bundle, out_dir, run_label: str = ''):
  """
  Render a 2×2 figure with train/val curves for MSE, MAE and cosine similarity,
  plus a text block listing the projector's hyperparameters and final metrics.

  A green star marks the best epoch (lowest val MSE) on each metric subplot.
  When per-epoch MAE is missing (older pkls produced before MAE-per-epoch was
  tracked in cross_space_projection.py), the MAE subplot displays a placeholder
  message instead of crashing.

  Args:
    linear_bundle (dict): Output of _extract_linear_bundle. Must include
      'metrics' (with 'train' and 'val' lists), 'best_epoch', and 'config'.
    out_dir       (str):  Directory in which to write the PNG.
    run_label     (str):  Suptitle suffix identifying the run.
  """
  metrics = linear_bundle.get('metrics') or {}
  tr, va, has_mae = _projector_curve_arrays(metrics)
  best_epoch = linear_bundle.get('best_epoch')
  test_metrics = metrics.get('test') or {}

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))

  _plot_metric_curve(
    axes[0, 0], tr['epoch'], tr['mse'], va['epoch'], va['mse'],
    best_epoch, test_metrics.get('mse'),
    ylabel='MSE', title='MSE (train vs val, ★ = test @ best ckpt)', log_y=True,
  )

  if has_mae:
    _plot_metric_curve(
      axes[0, 1], tr['epoch'], tr['mae'], va['epoch'], va['mae'],
      best_epoch, test_metrics.get('mae'),
      ylabel='MAE', title='MAE (train vs val, ★ = test @ best ckpt)', log_y=False,
    )
  else:
    axes[0, 1].axis('off')
    axes[0, 1].text(
      0.5, 0.5,
      'Per-epoch MAE not available in this pkl.\n'
      '(Produced before MAE-per-epoch logging was added.)',
      ha='center', va='center', fontsize=10, color='#555555',
      transform=axes[0, 1].transAxes,
    )
    axes[0, 1].set_title('MAE (train vs val)')

  _plot_metric_curve(
    axes[1, 0], tr['epoch'], tr['cos'], va['epoch'], va['cos'],
    best_epoch, test_metrics.get('cos'),
    ylabel='cosine similarity',
    title='Cosine similarity (train vs val, ★ = test @ best ckpt)',
    log_y=False,
  )

  axes[1, 1].axis('off')
  axes[1, 1].text(
    0.0, 1.0, _format_projector_config_text(linear_bundle),
    ha='left', va='top', family='monospace', fontsize=9,
    transform=axes[1, 1].transAxes,
  )

  fig.suptitle(f'Linear projector training — {run_label}', fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.97))
  path = os.path.join(out_dir, 'projector_training_curves_train_vs_val.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_projector_train_val_gap(linear_bundle, out_dir, run_label: str = ''):
  """
  Plot the validation-minus-training gap for MSE (and MAE when available)
  across epochs, to make over/under-fitting visually obvious.

  A horizontal dashed line at zero marks parity; a star marks the best epoch.

  Args:
    linear_bundle (dict): Output of _extract_linear_bundle.
    out_dir       (str):  Directory in which to write the PNG.
    run_label     (str):  Suptitle suffix identifying the run.
  """
  if (linear_bundle.get('kind') or '').lower() == 'procrustes':
    print('[plot_projector_train_val_gap] Skipped — closed-form procrustes has no epoch curve.')
    return
  metrics = linear_bundle.get('metrics') or {}
  tr, va, has_mae = _projector_curve_arrays(metrics)
  best_epoch = linear_bundle.get('best_epoch')

  if tr['epoch'].size == 0 or va['epoch'].size == 0:
    print('[WARN] projector train/val gap: empty metrics — skipping.')
    return

  fig, ax = plt.subplots(1, 1, figsize=(10, 5))
  ax.plot(va['epoch'], va['mse'] - tr['mse'], '-', color='#1f77b4',
          label='val − train (MSE)', linewidth=1.5)
  if has_mae:
    ax.plot(va['epoch'], va['mae'] - tr['mae'], '-', color='#ff7f0e',
            label='val − train (MAE)', linewidth=1.5)

  ax.axhline(0.0, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

  if best_epoch is not None and best_epoch > 0:
    idx = np.argmin(np.abs(va['epoch'] - best_epoch))
    ax.scatter([va['epoch'][idx]], [(va['mse'] - tr['mse'])[idx]],
               marker='*', s=180, color='#2ca02c', edgecolor='black',
               linewidth=0.7, zorder=5, label=f'best (ep {best_epoch})')

  ax.set_xlabel('Epoch')
  ax.set_ylabel('val − train')
  ax.set_title('Train/val gap (positive = val worse than train)')
  ax.grid(alpha=0.3)
  ax.legend(loc='best', fontsize=9)
  fig.suptitle(f'Projector train/val gap — {run_label}', fontsize=12, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.95))
  path = os.path.join(out_dir, 'projector_train_val_gap.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_projector_weight_analysis(linear_bundle, out_dir, run_label: str = ''):
  """
  Visualize the learned linear projector's weight matrix and its singular
  value spectrum to expose rank collapse or sparsity patterns.

  Generates three side-by-side panels:
    1. Heatmap of W (shape d_new × d_old), centered at 0.
    2. Histogram of |W| values.
    3. Singular value spectrum on log y, annotated with the effective rank
       (s.sum())² / (s²).sum().

  Args:
    linear_bundle (dict): Output of _extract_linear_bundle. Must include
      'ckpt_path' pointing to a torch state_dict with a 'weight' tensor.
    out_dir       (str):  Directory in which to write the PNG.
    run_label     (str):  Suptitle suffix identifying the run.
  """
  ckpt_path = linear_bundle.get('ckpt_path')
  if not ckpt_path or not os.path.isfile(ckpt_path):
    print(f'[WARN] projector weight analysis: checkpoint not found at {ckpt_path} — skipping.')
    return

  state = torch.load(ckpt_path, map_location='cpu')
  if 'weight' not in state:
    print('[WARN] projector weight analysis: state_dict has no "weight" key — skipping.')
    return
  W = state['weight'].detach().cpu().numpy().astype(np.float32)

  fig, axes = plt.subplots(1, 3, figsize=(18, 5))

  vmax = float(np.abs(W).max()) if W.size else 1.0
  sns.heatmap(
    W, ax=axes[0],
    cmap='RdBu_r', center=0.0, vmin=-vmax, vmax=vmax,
    cbar_kws={'shrink': 0.7},
  )
  axes[0].set_title(f'W heatmap  shape={W.shape}')
  axes[0].set_xlabel('d_old')
  axes[0].set_ylabel('d_new')

  axes[1].hist(np.abs(W).ravel(), bins=80, color='#4c72b0', edgecolor='white')
  axes[1].set_title('Histogram of |W|')
  axes[1].set_xlabel('|weight|')
  axes[1].set_ylabel('count')
  axes[1].grid(alpha=0.3)

  s = np.linalg.svd(W, compute_uv=False)
  s_sq_sum = float((s ** 2).sum())
  eff_rank = float((s.sum() ** 2) / s_sq_sum) if s_sq_sum > 0 else 0.0
  axes[2].plot(np.arange(1, len(s) + 1), s, marker='o', markersize=3,
               linewidth=1.2, color='#2ca02c')
  axes[2].set_yscale('log')
  axes[2].set_xlabel('index')
  axes[2].set_ylabel('singular value')
  axes[2].set_title(f'SVD spectrum  (eff. rank ≈ {eff_rank:.1f} / {len(s)})')
  axes[2].grid(alpha=0.3, which='both')

  fig.suptitle(f'Linear projector weight analysis — {run_label}', fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.95))
  path = os.path.join(out_dir, 'projector_weight_analysis.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_projector_norm_comparison(linear_bundle, out_dir, run_label: str = ''):
  """
  Compare the L2-norms of projected vs target embeddings on the test split.

  Left panel: scatter of ||target|| (x) vs ||projected|| (y) per sample, with
  an identity line for reference.
  Right panel: overlayed histograms of both norm distributions.

  Args:
    linear_bundle (dict): Output of _extract_linear_bundle. Must contain a
      'splits' dict whose 'test' entry has 'projected' (N, D) and
      'target' (N, D) numpy arrays.
    out_dir       (str):  Directory in which to write the PNG.
    run_label     (str):  Suptitle suffix identifying the run.
  """
  splits = linear_bundle.get('splits') or {}
  test   = splits.get('test')
  if not test or test.get('projected') is None or test.get('target') is None:
    print('[WARN] projector norm comparison: missing test projected/target — skipping.')
    return

  proj = np.asarray(test['projected'], dtype=np.float32)
  tgt  = np.asarray(test['target'],    dtype=np.float32)
  if proj.shape != tgt.shape or proj.ndim != 2:
    print(f'[WARN] projector norm comparison: shape mismatch {proj.shape} vs {tgt.shape} — skipping.')
    return

  proj_n = np.linalg.norm(proj, axis=1)
  tgt_n  = np.linalg.norm(tgt,  axis=1)

  if proj_n.size >= 2 and tgt_n.size >= 2:
    pearson_r = float(stats.pearsonr(tgt_n, proj_n)[0])
  else:
    pearson_r = float('nan')
  mean_ratio = float(np.mean(proj_n / np.clip(tgt_n, 1e-12, None)))

  fig, axes = plt.subplots(1, 2, figsize=(12, 5))

  axes[0].scatter(tgt_n, proj_n, s=10, alpha=0.5, color='#4c72b0')
  lo = float(min(tgt_n.min(), proj_n.min()))
  hi = float(max(tgt_n.max(), proj_n.max()))
  axes[0].plot([lo, hi], [lo, hi], '--', color='black', linewidth=0.8, alpha=0.7, label='y = x')
  axes[0].set_xlabel('||target||')
  axes[0].set_ylabel('||projected||')
  axes[0].set_title(f'Embedding norms (test)  r={pearson_r:.3f}  mean(proj/tgt)={mean_ratio:.3f}')
  axes[0].grid(alpha=0.3)
  axes[0].legend(loc='best', fontsize=8)

  bins = 60
  axes[1].hist(tgt_n,  bins=bins, alpha=0.55, color='#2ca02c', label='||target||',    edgecolor='white')
  axes[1].hist(proj_n, bins=bins, alpha=0.55, color='#d62728', label='||projected||', edgecolor='white')
  axes[1].set_xlabel('L2 norm')
  axes[1].set_ylabel('count')
  axes[1].set_title('Norm distributions (test)')
  axes[1].grid(alpha=0.3)
  axes[1].legend(loc='best', fontsize=9)

  fig.suptitle(f'Projector norm comparison — {run_label}', fontsize=12, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.95))
  path = os.path.join(out_dir, 'projector_norm_comparison.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_projector_diagnostics(linear_bundle, out_dir, run_label: str = ''):
  """
  Convenience wrapper: emit all four projector-training diagnostic plots.

  Skips gracefully (with a console warning) if linear_bundle is None — useful
  for callers that don't want to repeat the check themselves.

  Args:
    linear_bundle (dict | None): Output of _extract_linear_bundle.
    out_dir       (str):         Directory in which to write the PNGs.
    run_label     (str):         Suptitle suffix identifying the run.
  """
  if linear_bundle is None:
    print('[cross_space_logs] No linear_projector in pkl — skipping projector-training plots.')
    return
  plot_projector_training_curves(linear_bundle, out_dir, run_label=run_label)
  plot_projector_train_val_gap(linear_bundle, out_dir, run_label=run_label)
  try:
    plot_projector_weight_analysis(linear_bundle, out_dir, run_label=run_label)
  except Exception as exc:
    print(f'[WARN] projector weight analysis failed: {exc}')
  plot_projector_norm_comparison(linear_bundle, out_dir, run_label=run_label)


# ── refinement-training inspection ────────────────────────────────────────────

def _refine_to_float(v):
  """
  Coerce a possibly-None / non-numeric value to float.

  Args:
    v: Any value (float, int, str, None, ...).

  Returns:
    float: float(v), or NaN when v is None or cannot be parsed as a number.
  """
  try:
    return float(v)
  except (TypeError, ValueError):
    return float('nan')


def _newtest_preserve_mae(block, which):
  """
  Preserve MAE on the new model's TEST split for the 'before'/'after' refinement stage.

  Recomputes (micro, macro) from the per-sample predictions in block['new_test_eval'] (the
  new model's own TEST split) so the value matches summary.csv's
  newtest_mae_{micro,macro}_{which} columns. The stored mae_{micro,macro}_new_test_{which}
  scalars carry the same name but are computed on new_eval_split (default 'val'); they are
  used only as a fallback when new_test_eval is unavailable (e.g. older pkls).

  Args:
    block: A refinement block (pkl 'refinement' or 'refinements'[mode]). dict.
    which: Refinement stage, 'before' or 'after'. str.

  Returns:
    tuple[float, float]: (micro_mae, macro_mae) on the test split, or the stored val-split
      scalars / NaNs when new_test_eval is missing.
  """
  nte = (block or {}).get('new_test_eval') or {}
  preds = nte.get(f'preds_{which}')
  lbls  = nte.get('labels')
  if preds is not None and lbls is not None:
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    lbls  = np.asarray(lbls,  dtype=np.float32).reshape(-1)
    if preds.size and preds.size == lbls.size:
      return _compute_global_mae(preds, lbls)
  return (_refine_to_float(block.get(f'mae_micro_new_test_{which}')),
          _refine_to_float(block.get(f'mae_macro_new_test_{which}')))


def _refinement_curve_arrays(per_epoch):
  """
  Convert the refinement per-epoch metric list into per-key numpy arrays.

  Args:
    per_epoch (list[dict] | None): Each dict carries at least 'epoch',
      'loss_total', 'loss_B', 'loss_A' (real-label MSE scale), as written by
      cross_space_projection._refine_projector_and_linear. Runs with held-out
      validation also carry 'val_total', 'val_B', 'val_A' and the per-epoch
      'val_mae_{micro,macro}_{B,A}' real-label MAEs.

  Returns:
    dict: float32 arrays keyed by 'epoch', the three 'loss_*' train losses, the
      three 'val_*' validation losses, the four 'val_mae_*' arrays, plus a
      boolean 'has_val'. All arrays are empty (size 0) when per_epoch is
      missing/empty; the val_* arrays are empty when 'has_val' is False (old
      train-only pkls).
  """
  rows = per_epoch or []

  def _stack(key):
    return (np.asarray([_refine_to_float(r.get(key)) for r in rows], dtype=np.float32)
            if rows else np.zeros(0, dtype=np.float32))

  has_val = bool(rows) and all('val_total' in r for r in rows)
  out = {
    'epoch':      _stack('epoch'),
    'loss_total': _stack('loss_total'),
    'loss_B':     _stack('loss_B'),
    'loss_A':     _stack('loss_A'),
    'has_val':    has_val,
  }
  for key in ('val_total', 'val_B', 'val_A',
              'val_mae_micro_B', 'val_mae_macro_B',
              'val_mae_micro_A', 'val_mae_macro_A'):
    out[key] = _stack(key) if has_val else np.zeros(0, dtype=np.float32)
  return out


def _format_refinement_config_text(refine_block):
  """
  Build a monospace-friendly summary of the refinement run for a text panel.

  Lists the REFINEMENT_CONFIG hyperparameters, the selection criterion and kept
  epoch (with its held-out val_total when validation selection was used), the
  projector anchor-loss drift, and the before→after MAE pairs.

  Args:
    refine_block (dict): The pkl 'refinement' block (standalone or Optuna-trial
      format). Reads 'config', 'refine_best_epoch', 'refine_val_selection',
      'refine_best_val_total', 'proj_anchor_loss_*' and the four
      'mae_*_*_before/after' scalars.

  Returns:
    str: Multi-line text block.
  """
  def _f(v):
    v = _refine_to_float(v)
    return 'n/a' if not np.isfinite(v) else f'{v:.6f}'

  cfg = refine_block.get('config', {}) or {}
  rows = ['── REFINEMENT_CONFIG ──']
  for k in ('lr_projector', 'lr_linear', 'lambda_B', 'lambda_A', 'optimizer',
            'loss', 'epochs', 'batch_size', 'weight_decay',
            'refine_split', 'refine_val_split', 'new_eval_split',
            'refine_val_min_keep_frac'):
    if k in cfg:
      rows.append(f'{k:<24s} {cfg[k]}')

  selection = refine_block.get('refine_val_selection')
  rows.append('')
  rows.append('── best epoch (selection) ──')
  rows.append(f'selection_used         {selection or "train_loss"}')
  rows.append(f'best_epoch             {refine_block.get("refine_best_epoch")}')
  if refine_block.get('refine_best_val_total') is not None:
    rows.append(f'best_val_total         {_f(refine_block.get("refine_best_val_total"))}')

  rows.append('')
  rows.append('── projector anchor loss ──')
  rows.append(f'before                 {_f(refine_block.get("proj_anchor_loss_before"))}')
  rows.append(f'after                  {_f(refine_block.get("proj_anchor_loss_after"))}')

  rows.append('')
  rows.append('── MAE (before → after) ──')
  # new_test rows recomputed on the new model's TEST split (new_test_eval), matching
  # summary.csv's newtest_mae_* (the stored mae_*_new_test_* scalars are on the val split).
  nt_b = _newtest_preserve_mae(refine_block, 'before')
  nt_a = _newtest_preserve_mae(refine_block, 'after')
  for label, vb, va in (
    ('old_oncsv micro', refine_block.get('mae_micro_old_oncsv_before'), refine_block.get('mae_micro_old_oncsv_after')),
    ('old_oncsv macro', refine_block.get('mae_macro_old_oncsv_before'), refine_block.get('mae_macro_old_oncsv_after')),
    ('new_test  micro', nt_b[0], nt_a[0]),
    ('new_test  macro', nt_b[1], nt_a[1]),
  ):
    rows.append(f'{label:<16s} {_f(vb)} → {_f(va)}')

  return '\n'.join(rows)


def _plot_refine_loss_curve(ax, epochs, vals, best_epoch, ylabel, title,
                            log_y=True, val_vals=None):
  """
  Draw a refinement loss curve (train, optionally with a validation overlay) and
  a star at the best epoch.

  When val_vals is given the run selected its kept checkpoint by held-out
  validation loss, so the best-epoch star sits on the VAL curve at
  (best_epoch, val@best_epoch). With val_vals=None (old train-only pkls) the
  star sits on the train curve instead, matching the legacy behavior.

  Args:
    ax         (matplotlib.axes.Axes): Axes to draw on.
    epochs     (np.ndarray): Epoch indices, shape (E,).
    vals       (np.ndarray): Train loss values, shape (E,).
    best_epoch (int | None): 1-based epoch of the kept state.
    ylabel     (str):        Y-axis label.
    title      (str):        Subplot title.
    log_y      (bool):       If True, use a symlog y-scale.
    val_vals   (np.ndarray | None): Validation loss values, shape (E,). When
      present, drawn as a red overlay and used for the best-epoch star.
  """
  ax.plot(epochs, vals, '-', color='#1f77b4', linewidth=1.5, label='train')
  star_vals = vals
  if val_vals is not None and val_vals.size:
    ax.plot(epochs, val_vals, '-', color='#d62728', linewidth=1.5, label='val')
    star_vals = val_vals
  if best_epoch is not None and best_epoch > 0 and epochs.size:
    idx = int(np.argmin(np.abs(epochs - best_epoch)))
    star_label = (f'best (val) ep {best_epoch} = {float(star_vals[idx]):.4f}'
                  if val_vals is not None and val_vals.size
                  else f'best ep {best_epoch} = {float(star_vals[idx]):.4f}')
    ax.scatter([epochs[idx]], [star_vals[idx]], marker='*', s=180, color='#2ca02c',
               edgecolor='black', linewidth=0.7, zorder=5, label=star_label)
  ax.set_xlabel('Epoch')
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  ax.grid(alpha=0.3)
  ax.legend(loc='best', fontsize=8)
  if log_y:
    ax.set_yscale('symlog', linthresh=1e-4)


def _plot_refine_val_mae_curve(ax, epochs, micro, macro, best_epoch, title):
  """
  Draw per-epoch held-out validation MAE (micro + macro) for one refinement term.

  Both traces are in the real label scale (lower = better). A vertical dashed
  line marks the kept (best-validation) epoch.

  Args:
    ax         (matplotlib.axes.Axes): Axes to draw on.
    epochs     (np.ndarray): Epoch indices, shape (E,).
    micro      (np.ndarray): Per-epoch micro MAE, shape (E,).
    macro      (np.ndarray): Per-epoch macro MAE, shape (E,).
    best_epoch (int | None): 1-based epoch of the kept state (vertical marker).
    title      (str):        Subplot title.
  """
  ax.plot(epochs, micro, '-', color='#1f77b4', linewidth=1.5, label='val MAE micro')
  ax.plot(epochs, macro, '-', color='#ff7f0e', linewidth=1.5, label='val MAE macro')
  if best_epoch is not None and best_epoch > 0 and epochs.size:
    ax.axvline(best_epoch, color='#2ca02c', linestyle='--', linewidth=1.0,
               label=f'best ep {best_epoch}')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('MAE (real label scale)')
  ax.set_title(title)
  ax.grid(alpha=0.3)
  ax.legend(loc='best', fontsize=8)


def plot_refinement_training_curves(refine_block, out_dir, run_label: str = '',
                                    filename_suffix: str = '',
                                    src_dataset: str = None, new_dataset: str = None):
  """
  Render a 3×2 figure of the refinement per-epoch metrics.

  Panels:
    (0,0) loss_total, (0,1) loss_B (source/model-B label term),
    (1,0) loss_A (new-anchor preserve term) — each a symlog train curve with a
          held-out val overlay (when present) and a green star at the kept epoch.
    (1,1) held-out val MAE (micro+macro) for the source (B) term.
    (2,0) held-out val MAE (micro+macro) for the preserve (A) term.
    (2,1) monospace config/summary block.
  Runs without held-out validation (old pkls) fall back to train-only loss
  curves and blank the two MAE panels.

  Args:
    refine_block (dict): The pkl 'refinement' block; must carry 'per_epoch_metrics'.
    out_dir      (str):  Directory in which to write the PNG.
    run_label    (str):  Suptitle suffix identifying the run.
    src_dataset  (str | None): Resolved old/source (model-B) dataset name, named on the
      source-B panels; None ⇒ generic 'model B'.
    new_dataset  (str | None): Resolved new/target (preserve-A) dataset name, named on the
      preserve-A panels; None ⇒ generic 'new-model'.
  """
  src_ds_lbl = src_dataset or 'model B'
  new_ds_lbl = new_dataset or 'new-model'
  cur = _refinement_curve_arrays(refine_block.get('per_epoch_metrics'))
  if cur['epoch'].size == 0:
    print('[WARN] refinement training curves: empty per-epoch metrics — skipping.')
    return
  best_epoch = refine_block.get('refine_best_epoch')
  has_val    = cur['has_val']
  vt = cur['val_total'] if has_val else None
  vb = cur['val_B']     if has_val else None
  va = cur['val_A']     if has_val else None

  fig, axes = plt.subplots(3, 2, figsize=(15, 15))
  _plot_refine_loss_curve(
    axes[0, 0], cur['epoch'], cur['loss_total'], best_epoch,
    ylabel='loss_total', title='Total loss  (λ_B·loss_B + λ_A·loss_A)', log_y=True, val_vals=vt)
  _plot_refine_loss_curve(
    axes[0, 1], cur['epoch'], cur['loss_B'], best_epoch,
    ylabel='loss_B', title=f'Source term loss_B  ({src_ds_lbl} labels)', log_y=True, val_vals=vb)
  _plot_refine_loss_curve(
    axes[1, 0], cur['epoch'], cur['loss_A'], best_epoch,
    ylabel='loss_A', title='Preserve term loss_A  (new anchors)', log_y=True, val_vals=va)

  if has_val:
    _plot_refine_val_mae_curve(
      axes[1, 1], cur['epoch'], cur['val_mae_micro_B'], cur['val_mae_macro_B'],
      best_epoch, title=f'Val MAE — source B (held-out {src_ds_lbl} val)')
    _plot_refine_val_mae_curve(
      axes[2, 0], cur['epoch'], cur['val_mae_micro_A'], cur['val_mae_macro_A'],
      best_epoch, title=f'Val MAE — preserve A (held-out {new_ds_lbl} val)')
  else:
    for ax in (axes[1, 1], axes[2, 0]):
      ax.axis('off')
      ax.text(0.5, 0.5, 'No held-out validation\nrecorded for this run.',
              ha='center', va='center', fontsize=10, color='#555555')

  axes[2, 1].axis('off')
  axes[2, 1].text(
    0.0, 1.0, _format_refinement_config_text(refine_block),
    ha='left', va='top', family='monospace', fontsize=9,
    transform=axes[2, 1].transAxes,
  )

  fig.suptitle(f'Refinement training — {run_label}', fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.97))
  path = os.path.join(out_dir, f'refinement_training_curves_train_vs_val{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_refinement_before_after(refine_block, out_dir, run_label: str = '',
                                 filename_suffix: str = '',
                                 src_dataset: str = None, new_dataset: str = None):
  """
  Render before-vs-after grouped bars showing the "improve B without hurting A"
  effect of refinement.

  Left panel: MAE (micro & macro) for old-model-on-old_model_csv (source, B) and
  new-model-on-test (target, A), before vs after, with Δ=after−before annotated.
  Right panel: projector anchor embedding-MSE before vs after (its own scale).
  Metric groups whose before & after are both NaN are skipped with a warning.

  Args:
    refine_block (dict): The pkl 'refinement' block.
    out_dir      (str):  Directory in which to write the PNG.
    run_label    (str):  Suptitle suffix identifying the run.
    src_dataset  (str | None): Resolved old/source dataset name, named on the source-csv
      bars; None ⇒ generic 'old'.
    new_dataset  (str | None): Resolved new/target dataset name, named on the new-test bars;
      None ⇒ generic 'new'.
  """
  src_ds_lbl = src_dataset or 'old'
  new_ds_lbl = new_dataset or 'new'
  # new-test groups: recompute on the new model's TEST split (new_test_eval) so the bars
  # match summary.csv's newtest_mae_* rather than the val-split stored scalars.
  nt_b = _newtest_preserve_mae(refine_block, 'before')
  nt_a = _newtest_preserve_mae(refine_block, 'after')
  mae_groups = (
    (f'{src_ds_lbl}-csv\nmicro',  refine_block.get('mae_micro_old_oncsv_before'), refine_block.get('mae_micro_old_oncsv_after')),
    (f'{src_ds_lbl}-csv\nmacro',  refine_block.get('mae_macro_old_oncsv_before'), refine_block.get('mae_macro_old_oncsv_after')),
    (f'{new_ds_lbl}-test\nmicro', nt_b[0], nt_a[0]),
    (f'{new_ds_lbl}-test\nmacro', nt_b[1], nt_a[1]),
  )
  labels, before_vals, after_vals = [], [], []
  for lab, vb, va in mae_groups:
    b = _refine_to_float(vb)
    a = _refine_to_float(va)
    if not (np.isfinite(b) or np.isfinite(a)):
      print(f'[WARN] refinement before/after: {lab.replace(chr(10), " ")} both NaN — skipping group.')
      continue
    labels.append(lab)
    before_vals.append(b)
    after_vals.append(a)

  fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

  if labels:
    x = np.arange(len(labels))
    w = 0.38
    axes[0].bar(x - w / 2, before_vals, w, label='before', color='#4c72b0', edgecolor='white')
    axes[0].bar(x + w / 2, after_vals,  w, label='after',  color='#dd8452', edgecolor='white')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE before vs after refinement')
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].legend(loc='best', fontsize=9)
    for i in range(len(labels)):
      d = after_vals[i] - before_vals[i]
      top = np.nanmax([before_vals[i], after_vals[i], 0.0])
      axes[0].annotate(
        f'Δ={d:+.4f}', (x[i], top), textcoords='offset points', xytext=(0, 4),
        ha='center', fontsize=8, color='#2ca02c' if d <= 0 else '#d62728')
  else:
    axes[0].axis('off')
    axes[0].text(0.5, 0.5, 'No finite MAE before/after values.',
                 ha='center', va='center', fontsize=10, color='#555555')

  pab = _refine_to_float(refine_block.get('proj_anchor_loss_before'))
  paa = _refine_to_float(refine_block.get('proj_anchor_loss_after'))
  if np.isfinite(pab) or np.isfinite(paa):
    x2 = np.arange(2)
    axes[1].bar(x2, [pab, paa], 0.5, color=['#4c72b0', '#dd8452'], edgecolor='white')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(['before', 'after'])
    axes[1].set_ylabel('anchor embedding-MSE')
    axes[1].set_title(f'Projector anchor loss (drift)  Δ={paa - pab:+.6f}')
    axes[1].grid(alpha=0.3, axis='y')
  else:
    axes[1].axis('off')
    axes[1].text(0.5, 0.5, 'No finite projector anchor loss.',
                 ha='center', va='center', fontsize=10, color='#555555')

  fig.suptitle(f'Refinement before vs after — {run_label}', fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.95))
  path = os.path.join(out_dir, f'refinement_mae_before_vs_after{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_refinement_newtest_mae_improvement_per_class(refine_block, out_dir, run_label: str = '',
                                                      filename_suffix: str = ''):
  """
  Signed per-class MAE improvement of the new model on its own test split, before
  vs after refinement.

  Refinement fine-tunes a copy of the new model's head.linear, which can change how
  the new model scores its own (original-distribution) test set. Each bar is the
  per-class difference MAE_before - MAE_after on that test split: positive (green)
  means refinement lowered the error for that class, negative (red) means it got
  worse. The new test set lives in the new space already, so no projector is
  involved — only head.linear differs between before and after. The per-sample
  predictions are read from the pkl block written by
  cross_space_projection._run_refinement_stage; this plot does no recomputation.

  Args:
    refine_block (dict | None): The pkl 'refinement' block; must carry a
      'new_test_eval' sub-dict with 'labels', 'preds_before', 'preds_after'.
    out_dir      (str): Output directory.
    run_label    (str): Optional run identity string appended to the plot title.
  """
  nt = (refine_block or {}).get('new_test_eval')
  if not nt:
    print('[cross_space_logs] No new_test_eval in refinement block — '
          'skipping new-test per-class improvement plot.')
    return

  labels = np.asarray(nt['labels'],       dtype=np.float32).reshape(-1)
  before = np.asarray(nt['preds_before'], dtype=np.float32).reshape(-1)
  after  = np.asarray(nt['preds_after'],  dtype=np.float32).reshape(-1)

  labels_int = np.round(labels).astype(int)
  before_mae = _mae_per_group(before, labels, labels_int)
  after_mae  = _mae_per_group(after,  labels, labels_int)
  groups = sorted(set(before_mae) | set(after_mae))
  diffs  = [
    before_mae.get(g, (float('nan'), 0))[0] - after_mae.get(g, (float('nan'), 0))[0]
    for g in groups
  ]
  suffix = f' | {run_label}' if run_label else ''
  split  = nt.get('split', 'test')

  fig, ax = plt.subplots(figsize=(14, 5))
  _draw_mae_improvement_bar(
    ax, groups, diffs, 'Labels',
    f'New-model {split}-set MAE improvement per class (before - after refinement){suffix}',
    ylabel='MAE improvement (before - after)',
  )
  plt.tight_layout()
  path = os.path.join(out_dir, f'mae_improvement_per_class_newtest_refined_vs_original{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')

  # Companion 2-bar comparison: before vs after MAE side by side per class.
  before_vals = [before_mae.get(g, (float('nan'), 0))[0] for g in groups]
  after_vals  = [after_mae.get(g, (float('nan'), 0))[0] for g in groups]
  fig, ax = plt.subplots(figsize=(14, 5))
  _draw_grouped_mae_bar(
    ax, groups, before_vals, after_vals, 'Before refinement', 'After refinement',
    '#E69F00', '#0072B2', 'MAE',
    f'New-model {split}-set MAE per class — before vs after refinement{suffix}',
  )
  ax.set_xlabel('Labels')
  plt.tight_layout()
  path = os.path.join(out_dir, f'mae_per_class_compare_newtest_refined_vs_original{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_refinement_diagnostics(refine_block, out_dir, run_label: str = '', filename_suffix: str = '',
                                newtest_run_label: str = None,
                                src_dataset: str = None, new_dataset: str = None):
  """
  Convenience wrapper: emit all refinement-stage diagnostic plots.

  Skips gracefully (with a console message) when refine_block is absent or has no
  per-epoch metrics — so non-refinement pkls pass through untouched.

  Args:
    refine_block (dict | None): The pkl 'refinement' block (or None).
    out_dir      (str):         Directory in which to write the PNGs.
    run_label    (str):         Suptitle suffix identifying the run.
    filename_suffix (str):      Optional per-mode PNG suffix (e.g. '_linear_only'),
      forwarded to every plot. Default '' ⇒ legacy filenames unchanged.
    newtest_run_label (str | None): Run label used only for the new-model-test
      per-class plot. The '*newtest*' plots are computed on the new model's own
      test split, so they must carry the new model's dataset tag rather than the
      source one in run_label. None ⇒ fall back to run_label.
    src_dataset (str | None): Resolved old/source dataset name, forwarded to the
      training-curves / before-after plots to name their source side. None ⇒ generic.
    new_dataset (str | None): Resolved new/target dataset name, forwarded to name the
      preserve / new-test side. None ⇒ generic.
  """
  if not refine_block or not refine_block.get('per_epoch_metrics'):
    print('[cross_space_logs] No refinement block in pkl — skipping refinement-training plots.')
    return
  try:
    plot_refinement_training_curves(refine_block, out_dir, run_label=run_label,
                                    filename_suffix=filename_suffix,
                                    src_dataset=src_dataset, new_dataset=new_dataset)
  except Exception as exc:
    print(f'[WARN] refinement training curves failed: {exc}')
  try:
    plot_refinement_before_after(refine_block, out_dir, run_label=run_label,
                                 filename_suffix=filename_suffix,
                                 src_dataset=src_dataset, new_dataset=new_dataset)
  except Exception as exc:
    print(f'[WARN] refinement before/after plot failed: {exc}')
  try:
    plot_refinement_newtest_mae_improvement_per_class(
      refine_block, out_dir,
      run_label=newtest_run_label if newtest_run_label is not None else run_label,
      filename_suffix=filename_suffix)
  except Exception as exc:
    print(f'[WARN] refinement new-test per-class improvement plot failed: {exc}')


# ── search-level summary plots ────────────────────────────────────────────────

def _annotate_bar_counts(ax, x, counts, y_top):
  """
  Print the per-bar trial count ('n=NN') near the base inside each bar.

  The label is placed just above the x-axis (well inside a typical MAE bar) in
  white bold so it reads on the steelblue fill and never collides with the mean
  value printed at the bar tip.

  Args:
    ax     (matplotlib.axes.Axes): Axes the bars were drawn on.
    x      (np.ndarray): Shape (G,), x positions of the bars (one per count).
    counts (array-like[int]): Number of trials aggregated into each bar (same order as x).
    y_top  (float): Shared upper y-limit; the label sits at a small fraction of it
      above the axis so it falls inside the bar near its base.
  """
  y = y_top * 0.025
  for xi, n in zip(x, counts):
    ax.text(xi, y, f'n={int(n)}', ha='center', va='bottom',
            fontsize=7, color='white', fontweight='bold')


def _draw_mae_summary_bar(ax, vals, means, yerr_lo, yerr_hi, y_top, xlabel, title,
                          counts=None):
  """
  Draw a mean-MAE bar chart with min/max whiskers into a single pre-existing axes.

  Renders the recurring search-summary block: one steelblue bar per value, a
  black [min, max] error whisker per bar, the mean printed at each bar tip, and a
  shared y-axis top so multiple such axes are directly comparable. When counts are
  supplied the per-bar trial count is also printed inside each bar near its base.

  Args:
    ax      (matplotlib.axes.Axes): Axes to draw on.
    vals    (list): Group labels for the x-axis (one per bar).
    means   (np.ndarray): Shape (G,), mean MAE per value (bar heights).
    yerr_lo (np.ndarray): Shape (G,), mean - min per value (lower whisker length).
    yerr_hi (np.ndarray): Shape (G,), max - mean per value (upper whisker length).
    y_top   (float): Fixed upper y-limit shared across plots for comparability.
    xlabel  (str): X-axis label.
    title   (str): Plot title.
    counts  (array-like[int] | None): Number of trials per bar (same order as vals).
      When provided, 'n=NN' is annotated near the base of each bar.
  """
  x    = np.arange(len(vals))
  bars = ax.bar(x, means, color='steelblue', alpha=0.85)
  ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi],
              fmt='none', color='black', capsize=5, linewidth=1.5)
  for bar in bars:
    ax.text(
      bar.get_x() + bar.get_width() / 2,
      bar.get_height() + y_top * 0.01,
      f'{bar.get_height():.3f}',
      ha='center', va='bottom', fontsize=8,
    )
  ax.set_ylim(0, y_top)
  ax.set_xticks(x)
  ax.set_xticklabels([str(v) for v in vals], rotation=30, ha='right')
  ax.set_xlabel(xlabel)
  ax.set_ylabel('MAE')
  ax.set_title(title)
  if counts is not None:
    _annotate_bar_counts(ax, x, counts, y_top)


def plot_hyperparam_mae_summary(df, out_dir):
  """
  For each hyperparameter with ≥ 2 distinct values, write mae_by_<param>.png.

  Bar height = mean MAE per value; error whiskers span [min, max].
  Y-axis top is fixed to df['mae'].max() * 1.05 across all plots so they
  share the same scale and are directly comparable.

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics and hyperparams.
    out_dir (str): Directory to write plots into.
  """
  hp_cols = [
    'num_anchors', 'anchor_selection_type', 'csv_anchor_selection', 'old_model_csv',
    'interpolation_similarity', 'mlp_num_layers', 'weighting_method', 'temperature',
    'rbf_sigma', 'refine_mode',
  ]
  active = [c for c in hp_cols if c in df.columns and df[c].nunique() >= 2]
  if not active:
    return

  y_top = float(df['mae'].max()) * 1.05

  for col in active:
    grp = (
      df.groupby(col)['mae']
      .agg(mean='mean', lo='min', hi='max', n='count')
      .reset_index()
      .sort_values(col)
    )
    vals    = grp[col].tolist()
    means   = grp['mean'].to_numpy()
    yerr_lo = (grp['mean'] - grp['lo']).to_numpy()
    yerr_hi = (grp['hi']   - grp['mean']).to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(vals) * 1.2), 5))
    x = np.arange(len(vals))
    bars = ax.bar(x, means, color='steelblue', alpha=0.85)
    ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi],
                fmt='none', color='black', capsize=5, linewidth=1.5)
    for bar in bars:
      ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + y_top * 0.01,
        f'{bar.get_height():.3f}',
        ha='center', va='bottom', fontsize=8,
      )
    _annotate_bar_counts(ax, x, grp['n'].to_numpy(), y_top)
    ax.set_ylim(0, y_top)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vals], rotation=30, ha='right')
    ax.set_xlabel(col)
    ax.set_ylabel('MAE')
    ax.set_title(f'MAE by {col}  (bar = mean, whiskers = min / max)')

    plt.tight_layout()
    path = os.path.join(out_dir, f'mae_by_{col}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_temperature_anchors_heatmap(df, out_dir):
  """
  Grid of heatmaps: temperature (x-axis) × num_anchors (y-axis), faceted by
  weighting_method (rows) and interpolation_similarity (columns).

  Cell colour encodes mean MAE; empty combinations are shown as white.
  Skipped entirely when temperature or num_anchors has < 2 distinct values.

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics and hyperparams.
    out_dir (str): Directory to write the plot into.
  """
  needed = {'temperature', 'num_anchors', 'weighting_method', 'interpolation_similarity', 'mae'}
  if not needed.issubset(df.columns):
    return
  if df['temperature'].nunique() < 2 or df['num_anchors'].nunique() < 2:
    return

  wm_vals   = sorted(df['weighting_method'].unique())
  is_vals   = sorted(df['interpolation_similarity'].unique())
  temp_vals = sorted(df['temperature'].unique())
  anc_vals  = sorted(df['num_anchors'].unique())

  vmin = float(df['mae'].min())
  vmax = float(df['mae'].max())
  n_rows, n_cols = len(wm_vals), len(is_vals)

  cell_w = max(5, len(temp_vals) * 0.8 + 1.5)
  cell_h = max(4, len(anc_vals)  * 0.6 + 1.5)
  fig, axes = plt.subplots(n_rows, n_cols,
                           figsize=(n_cols * cell_w, n_rows * cell_h),
                           squeeze=False, constrained_layout=True)

  im_ref = None
  for ri, wm in enumerate(wm_vals):
    for ci, is_ in enumerate(is_vals):
      ax = axes[ri][ci]
      subset = df[(df['weighting_method'] == wm) & (df['interpolation_similarity'] == is_)]
      pivot = (
        subset.groupby(['num_anchors', 'temperature'])['mae']
        .mean()
        .unstack('temperature')
        .reindex(index=anc_vals, columns=temp_vals)
      )
      data = pivot.to_numpy(dtype=float)

      im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                     origin='lower')
      im_ref = im

      ax.set_xticks(range(len(temp_vals)))
      ax.set_xticklabels([f'{t:.3g}' for t in temp_vals], rotation=45, ha='right')
      ax.set_yticks(range(len(anc_vals)))
      ax.set_yticklabels([str(a) for a in anc_vals])
      ax.set_xlabel('temperature')
      ax.set_ylabel('num_anchors')
      ax.set_title(f'weighting={wm} / interp_sim={is_}')

      for r in range(len(anc_vals)):
        for c in range(len(temp_vals)):
          val = data[r, c]
          if not np.isnan(val):
            ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                    fontsize=7, color='black')

  if im_ref is not None:
    fig.colorbar(im_ref, ax=axes, label='MAE', shrink=0.6)

  plt.suptitle(
    'MAE heatmap: temperature × num_anchors\n'
    '(rows = weighting_method, cols = interpolation_similarity)',
    fontsize=11,
  )
  path = os.path.join(out_dir, 'heatmap_temperature_anchors.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')


def plot_scale_interp_heatmap(df, out_dir):
  """
  For each of temperature and rbf_sigma (if ≥ 2 distinct values), write one PNG
  showing how MAE varies jointly with that scale parameter and interpolation_similarity.

  Layout per file: rows = weighting_method, cols = num_anchors.
  Inside each subplot: x-axis = interpolation_similarity, y-axis = scale parameter.
  Cell colour encodes mean MAE; empty combinations are shown as white.

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics and hyperparams.
    out_dir (str): Directory to write plots into.
  """
  base_needed = {'interpolation_similarity', 'weighting_method', 'num_anchors', 'mae'}
  if not base_needed.issubset(df.columns):
    return

  vmin = float(df['mae'].min())
  vmax = float(df['mae'].max())

  wm_vals  = sorted(df['weighting_method'].unique())
  anc_vals = sorted(df['num_anchors'].unique())
  is_vals  = sorted(df['interpolation_similarity'].unique())

  for scale_col in ('temperature', 'rbf_sigma'):
    if scale_col not in df.columns or df[scale_col].nunique() < 2:
      continue

    scale_vals = sorted(df[scale_col].unique())
    n_rows, n_cols = len(wm_vals), len(anc_vals)

    cell_w = max(5, len(is_vals)    * 0.9 + 1.5)
    cell_h = max(4, len(scale_vals) * 0.6 + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h),
                             squeeze=False, constrained_layout=True)

    im_ref = None
    for ri, wm in enumerate(wm_vals):
      for ci, anc in enumerate(anc_vals):
        ax = axes[ri][ci]
        subset = df[(df['weighting_method'] == wm) & (df['num_anchors'] == anc)]
        pivot = (
          subset.groupby([scale_col, 'interpolation_similarity'])['mae']
          .mean()
          .unstack('interpolation_similarity')
          .reindex(index=scale_vals, columns=is_vals)
        )
        data = pivot.to_numpy(dtype=float)

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                       origin='lower')
        im_ref = im

        ax.set_xticks(range(len(is_vals)))
        ax.set_xticklabels(is_vals, rotation=45, ha='right')
        ax.set_yticks(range(len(scale_vals)))
        ax.set_yticklabels([f'{v:.3g}' for v in scale_vals])
        ax.set_xlabel('interpolation_similarity')
        ax.set_ylabel(scale_col)
        ax.set_title(f'weighting={wm} / num_anchors={anc}')

        for r in range(len(scale_vals)):
          for c in range(len(is_vals)):
            val = data[r, c]
            if not np.isnan(val):
              ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                      fontsize=7, color='black')

    if im_ref is not None:
      fig.colorbar(im_ref, ax=axes, label='MAE', shrink=0.6)

    plt.suptitle(
      f'MAE heatmap: {scale_col} × interpolation_similarity\n'
      '(rows = weighting_method, cols = num_anchors)',
      fontsize=11,
    )
    path = os.path.join(out_dir, f'heatmap_{scale_col}_interpolation_similarity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_anchor_interp_heatmap(df, out_dir):
  """
  Write one PNG showing how MAE varies jointly with anchor_selection_type and
  interpolation_similarity.

  Mirrors plot_scale_interp_heatmap but uses the categorical anchor_selection_type
  column on the y-axis (in the slot a scale parameter like rbf_sigma would occupy).
  Layout: rows = weighting_method, cols = num_anchors. Inside each subplot:
  x-axis = interpolation_similarity, y-axis = anchor_selection_type. Cell colour
  encodes mean MAE (mean over every other swept param, including rbf_sigma /
  temperature); empty combinations are shown as white. Skipped when
  anchor_selection_type has < 2 distinct values (nothing to compare).

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics and hyperparams.
    out_dir (str): Directory to write the plot into.
  """
  base_needed = {'interpolation_similarity', 'weighting_method', 'num_anchors',
                 'anchor_selection_type', 'mae'}
  if not base_needed.issubset(df.columns):
    return
  if df['anchor_selection_type'].nunique() < 2:
    return

  vmin = float(df['mae'].min())
  vmax = float(df['mae'].max())

  wm_vals   = sorted(df['weighting_method'].unique())
  anc_vals  = sorted(df['num_anchors'].unique())
  is_vals   = sorted(df['interpolation_similarity'].unique())
  asel_vals = sorted(df['anchor_selection_type'].unique())

  n_rows, n_cols = len(wm_vals), len(anc_vals)

  cell_w = max(5, len(is_vals)   * 0.9 + 1.5)
  cell_h = max(4, len(asel_vals) * 0.6 + 1.5)
  fig, axes = plt.subplots(n_rows, n_cols,
                           figsize=(n_cols * cell_w, n_rows * cell_h),
                           squeeze=False, constrained_layout=True)

  im_ref = None
  for ri, wm in enumerate(wm_vals):
    for ci, anc in enumerate(anc_vals):
      ax = axes[ri][ci]
      subset = df[(df['weighting_method'] == wm) & (df['num_anchors'] == anc)]
      pivot = (
        subset.groupby(['anchor_selection_type', 'interpolation_similarity'])['mae']
        .mean()
        .unstack('interpolation_similarity')
        .reindex(index=asel_vals, columns=is_vals)
      )
      data = pivot.to_numpy(dtype=float)

      im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                     origin='lower')
      im_ref = im

      ax.set_xticks(range(len(is_vals)))
      ax.set_xticklabels(is_vals, rotation=45, ha='right')
      ax.set_yticks(range(len(asel_vals)))
      ax.set_yticklabels([str(v) for v in asel_vals])
      ax.set_xlabel('interpolation_similarity')
      ax.set_ylabel('anchor_selection_type')
      ax.set_title(f'weighting={wm} / num_anchors={anc}')

      for r in range(len(asel_vals)):
        for c in range(len(is_vals)):
          val = data[r, c]
          if not np.isnan(val):
            ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                    fontsize=7, color='black')

  if im_ref is not None:
    fig.colorbar(im_ref, ax=axes, label='MAE', shrink=0.6)

  plt.suptitle(
    'MAE heatmap: anchor_selection_type × interpolation_similarity\n'
    '(rows = weighting_method, cols = num_anchors)',
    fontsize=11,
  )
  path = os.path.join(out_dir, 'heatmap_anchor_selection_type_interpolation_similarity.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')


def plot_refinement_lambda_heatmap(df, out_dir):
  """
  For the refinement sweep, write one PNG per after-refinement MAE metric showing
  how that MAE varies jointly with the refinement loss weights lambda_A and lambda_B.

  Mirrors plot_scale_interp_heatmap but for the refinement recipe knobs:
  layout per file is rows = weighting_method, cols = ref_lr_projector. Inside each
  subplot: x-axis = ref_lambda_B, y-axis = ref_lambda_A. Cell colour encodes mean
  MAE for that metric (mean over every other swept param); empty combinations are
  shown as white. One file is emitted for each of srctest_mae_micro_after and
  newtest_mae_micro_after. Each file uses its own metric's [min, max] colour range.
  The title's third line names the dataset and split the metric refers to (e.g.
  'BIOVID · test split'), resolved from the src_dataset / new_dataset /
  refine_new_eval_split columns; missing columns fall back to generic labels.

  Only refinement rows are considered (ref_lambda_A / ref_lambda_B / ref_lr_projector
  non-null). Returns silently for non-refinement sweeps (those columns are all-None),
  when the recipe columns are absent, or when neither lambda axis has >= 2 distinct
  values (nothing to compare).

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics and lp_*/ref_*
      recipe columns.
    out_dir (str): Directory to write plots into.
  """
  base_needed = {'weighting_method', 'ref_lr_projector', 'ref_lambda_A', 'ref_lambda_B'}
  if not base_needed.issubset(df.columns):
    return

  ref_df = df.dropna(subset=['ref_lambda_A', 'ref_lambda_B', 'ref_lr_projector'])
  if ref_df.empty:
    return
  if ref_df['ref_lambda_A'].nunique() < 2 and ref_df['ref_lambda_B'].nunique() < 2:
    return

  wm_vals = sorted(ref_df['weighting_method'].unique())
  lr_vals = sorted(ref_df['ref_lr_projector'].unique())
  la_vals = sorted(ref_df['ref_lambda_A'].unique())
  lb_vals = sorted(ref_df['ref_lambda_B'].unique())

  def _uniq_tag(col, fallback):
    """Join a column's unique non-null values with '/', or return fallback when empty/absent."""
    if col not in ref_df.columns:
      return fallback
    vals = sorted({str(v) for v in ref_df[col].dropna().unique()})
    return '/'.join(vals) if vals else fallback

  for metric in ('srctest_mae_micro_after', 'newtest_mae_micro_after'):
    if metric not in ref_df.columns or ref_df[metric].notna().sum() == 0:
      continue

    vmin = float(ref_df[metric].min())
    vmax = float(ref_df[metric].max())
    n_rows, n_cols = len(wm_vals), len(lr_vals)

    cell_w = max(5, len(lb_vals) * 0.9 + 1.5)
    cell_h = max(4, len(la_vals) * 0.6 + 1.5)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h),
                             squeeze=False, constrained_layout=True)

    im_ref = None
    for ri, wm in enumerate(wm_vals):
      for ci, lr in enumerate(lr_vals):
        ax = axes[ri][ci]
        subset = ref_df[(ref_df['weighting_method'] == wm) & (ref_df['ref_lr_projector'] == lr)]
        pivot = (
          subset.groupby(['ref_lambda_A', 'ref_lambda_B'])[metric]
          .mean()
          .unstack('ref_lambda_B')
          .reindex(index=la_vals, columns=lb_vals)
        )
        data = pivot.to_numpy(dtype=float)

        im = ax.imshow(data, aspect='auto', cmap='RdYlGn_r', vmin=vmin, vmax=vmax,
                       origin='lower')
        im_ref = im

        ax.set_xticks(range(len(lb_vals)))
        ax.set_xticklabels([f'{v:.3g}' for v in lb_vals], rotation=45, ha='right')
        ax.set_yticks(range(len(la_vals)))
        ax.set_yticklabels([f'{v:.3g}' for v in la_vals])
        ax.set_xlabel('lambda_B')
        ax.set_ylabel('lambda_A')
        ax.set_title(f'weighting={wm} / lr_projector={lr:.3g}')

        for r in range(len(la_vals)):
          for c in range(len(lb_vals)):
            val = data[r, c]
            if not np.isnan(val):
              ax.text(c, r, f'{val:.3f}', ha='center', va='center',
                      fontsize=7, color='black')

    if im_ref is not None:
      fig.colorbar(im_ref, ax=axes, label=metric, shrink=0.6)

    # srctest is the source/old set run as 'test'; newtest is the new model's own
    # eval split (refine_new_eval_split). Name the dataset · split the metric refers to.
    if metric.startswith('srctest'):
      set_tag = f"{_uniq_tag('src_dataset', 'source dataset')} · test split"
    else:
      set_tag = f"{_uniq_tag('new_dataset', 'new-model dataset')} · " \
                f"{_uniq_tag('refine_new_eval_split', 'test')} split"

    plt.suptitle(
      'MAE heatmap: lambda_A × lambda_B (after refinement)\n'
      '(rows = weighting_method, cols = lr_projector)\n'
      f'{metric} — {set_tag}',
      fontsize=11,
    )
    path = os.path.join(out_dir, f'heatmap_refinement_lambda_{metric}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_mae_anchors_per_interp_sim(df, out_dir):
  """
  For each distinct interpolation_similarity value, write one bar-chart PNG
  showing MAE as a function of num_anchors.

  One subplot row per weighting_method value (or a single row when the column
  is absent). Bar height = mean MAE; error whiskers span [min, max].
  Y-axis top is fixed to df['mae'].max() * 1.05 across all subplots and all
  output files so every PNG shares the same scale and is directly comparable.
  Skipped when num_anchors has < 2 distinct values.

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics and hyperparams.
    out_dir (str): Directory to write the plots into.
  """
  needed = {'num_anchors', 'interpolation_similarity', 'mae'}
  if not needed.issubset(df.columns):
    return
  if df['num_anchors'].nunique() < 2:
    return

  anc_vals = sorted(df['num_anchors'].unique())
  is_vals  = sorted(df['interpolation_similarity'].unique())
  wm_vals  = sorted(df['weighting_method'].unique()) if 'weighting_method' in df.columns else [None]
  y_top    = float(df['mae'].max()) * 1.05

  for is_ in is_vals:
    subset_is = df[df['interpolation_similarity'] == is_]
    n_rows    = len(wm_vals)
    fig, axes = plt.subplots(
      n_rows, 1,
      figsize=(max(8, len(anc_vals) * 1.2), 5 * n_rows),
      squeeze=False,
    )

    for ri, wm in enumerate(wm_vals):
      ax = axes[ri][0]
      subset = subset_is[subset_is['weighting_method'] == wm] if wm is not None else subset_is
      title  = (f'weighting={wm} | interp_sim={is_}') if wm is not None else f'interp_sim={is_}'

      grp = (
        subset.groupby('num_anchors')['mae']
        .agg(mean='mean', lo='min', hi='max', n='count')
        .reset_index()
        .sort_values('num_anchors')
      )
      vals    = grp['num_anchors'].tolist()
      means   = grp['mean'].to_numpy()
      yerr_lo = (grp['mean'] - grp['lo']).to_numpy()
      yerr_hi = (grp['hi']   - grp['mean']).to_numpy()

      x    = np.arange(len(vals))
      bars = ax.bar(x, means, color='steelblue', alpha=0.85)
      ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi],
                  fmt='none', color='black', capsize=5, linewidth=1.5)
      for bar in bars:
        ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + y_top * 0.01,
          f'{bar.get_height():.3f}',
          ha='center', va='bottom', fontsize=8,
        )
      _annotate_bar_counts(ax, x, grp['n'].to_numpy(), y_top)
      ax.set_ylim(0, y_top)
      ax.set_xticks(x)
      ax.set_xticklabels([str(v) for v in vals])
      ax.set_xlabel('num_anchors')
      ax.set_ylabel('MAE')
      ax.set_title(title)

    is_str = str(is_).replace('/', '_').replace(' ', '_')
    plt.tight_layout()
    path = os.path.join(out_dir, f'mae_anchors_interp_{is_str}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_recipe_mae_per_interp_sim(df, out_dir):
  """
  For each swept training-recipe hyperparameter, write one bar-chart PNG showing
  MAE as a function of that hyperparameter, faceted by interpolation_similarity.

  Covers both the projector recipe (lp_* columns, e.g. lp_lr, lp_batch_size,
  lp_optimizer, lp_weight_decay, lp_epochs, lp_normalize_embeddings, lp_loss) and
  the refinement recipe (ref_* columns). Only columns present in df with >= 2
  distinct non-null values are plotted, so a sweep that touched only the projector
  recipe emits only lp_* files (and vice-versa). One subplot row per
  interpolation_similarity value (mlp / linear / procrustes / ...) gives a complete
  view of each recipe knob for each interpolation choice. Bar height = mean MAE;
  error whiskers span [min, max]. Y-axis top is fixed to df['mae'].max() * 1.05
  across all subplots and files so every PNG shares the same scale.

  Interpolation values with no rows for a given column (e.g. procrustes trains no
  projector, so its lp_* fields are all None) are dropped from that file's rows.

  Args:
    df      (pd.DataFrame): Summary DataFrame with trial metrics, the lp_*/ref_*
      recipe columns and an interpolation_similarity column.
    out_dir (str): Directory to write the plots into.
  """
  if df.empty or 'mae' not in df.columns or 'interpolation_similarity' not in df.columns:
    return

  recipe_cols = (
    [f'lp_{f}'  for f in _LP_SUMMARY_FIELDS] +
    [f'ref_{f}' for f in _REF_SUMMARY_FIELDS]
  )
  active = [c for c in recipe_cols if c in df.columns and df[c].nunique(dropna=True) >= 2]
  if not active:
    return

  is_vals = sorted(df['interpolation_similarity'].dropna().unique())
  y_top   = float(df['mae'].max()) * 1.05

  for col in active:
    sub_all        = df[df[col].notna()]
    interp_present = [iv for iv in is_vals
                      if not sub_all[sub_all['interpolation_similarity'] == iv].empty]
    if not interp_present:
      continue

    n_vals    = sub_all[col].nunique()
    n_rows    = len(interp_present)
    fig, axes = plt.subplots(
      n_rows, 1,
      figsize=(max(8, n_vals * 1.2), 5 * n_rows),
      squeeze=False,
    )

    for ri, iv in enumerate(interp_present):
      ax     = axes[ri][0]
      subset = sub_all[sub_all['interpolation_similarity'] == iv]
      grp = (
        subset.groupby(col)['mae']
        .agg(mean='mean', lo='min', hi='max', n='count')
        .reset_index()
        .sort_values(col)
      )
      vals    = grp[col].tolist()
      means   = grp['mean'].to_numpy()
      yerr_lo = (grp['mean'] - grp['lo']).to_numpy()
      yerr_hi = (grp['hi']   - grp['mean']).to_numpy()
      _draw_mae_summary_bar(
        ax, vals, means, yerr_lo, yerr_hi, y_top,
        xlabel=col, title=f'interp_sim={iv}', counts=grp['n'].to_numpy(),
      )

    plt.tight_layout()
    path = os.path.join(out_dir, f'mae_by_{col}_per_interp.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _emit_summary_plots(df, out_dir):
  """
  Run the full set of search-level summary plots for df into out_dir.

  Single source of truth for which summary plots are produced, shared by the
  per-folder search_summary/ emitter (generate_search_summary_plots) and the
  cross-folder global_summary/ block (generate_logs_multi) so both always emit
  the identical plot set — including the recipe MAE plots and any plot added
  here later. Creating the directory is the caller's responsibility.

  Args:
    df      (pd.DataFrame): Summary DataFrame (all trials) with metrics,
      hyperparameter and lp_*/ref_* recipe columns.
    out_dir (str): Existing directory to write the plots into.
  """
  plot_hyperparam_mae_summary(df, out_dir)
  plot_temperature_anchors_heatmap(df, out_dir)
  plot_scale_interp_heatmap(df, out_dir)
  plot_anchor_interp_heatmap(df, out_dir)
  plot_refinement_lambda_heatmap(df, out_dir)
  plot_mae_anchors_per_interp_sim(df, out_dir)
  plot_recipe_mae_per_interp_sim(df, out_dir)


def generate_search_summary_plots(df, search_dir):
  """
  Write all search-level summary plots into <search_dir>/search_summary/.

  Args:
    df         (pd.DataFrame): Full summary DataFrame (all trials, sorted by MAE).
    search_dir (str): Root folder of the grid search.

  Returns:
    str: Path to the search_summary directory.
  """
  summary_dir = os.path.join(search_dir, 'search_summary')
  os.makedirs(summary_dir, exist_ok=True)
  print(f'[cross_space_logs] Writing search summary plots to {summary_dir}')
  _emit_summary_plots(df, summary_dir)
  return summary_dir


def plot_fake_vs_real_dashboard(evaluation, out_dir, mode=None, filename_suffix=''):
  """Plot paired real/fake replay predictions on shared axes and return the PNG path."""
  labels = np.asarray(evaluation['labels'], dtype=np.float32).reshape(-1)
  real = np.asarray(evaluation['real_predictions'], dtype=np.float32).reshape(-1)
  fake = np.asarray(evaluation['fake_predictions'], dtype=np.float32).reshape(-1)
  classes = np.round(labels).astype(int)
  limits = [float(min(labels.min(), real.min(), fake.min())),
            float(max(labels.max(), real.max(), fake.max()))]
  if limits[0] == limits[1]:
    limits = [limits[0] - .5, limits[1] + .5]

  fig, axes = plt.subplots(2, 2, figsize=(15, 12))
  for ax, predictions, name, color in (
      (axes[0, 0], real, 'Real replay', '#4C72B0'),
      (axes[0, 1], fake, 'Fake replay', '#DD8452')):
    ax.scatter(labels, predictions, alpha=.65, color=color, edgecolor='white', linewidth=.3)
    ax.plot(limits, limits, '--', color='#555555', linewidth=1)
    ax.set(xlim=limits, ylim=limits, xlabel='Ground truth', ylabel='Prediction', title=name)
    ax.grid(alpha=.25)

  bins = np.linspace(limits[0], limits[1], 20)
  axes[1, 0].hist(real, bins=bins, alpha=.65, label='Real replay', color='#4C72B0')
  axes[1, 0].hist(fake, bins=bins, alpha=.65, label='Fake replay', color='#DD8452')
  axes[1, 0].set(title='Prediction distributions', xlabel='Prediction', ylabel='Count')
  axes[1, 0].legend()
  axes[1, 0].grid(axis='y', alpha=.25)

  class_ids = sorted(np.unique(classes))
  real_mae = [float(np.abs(real[classes == cls] - labels[classes == cls]).mean())
              for cls in class_ids]
  fake_mae = [float(np.abs(fake[classes == cls] - labels[classes == cls]).mean())
              for cls in class_ids]
  x = np.arange(len(class_ids))
  axes[1, 1].bar(x - .2, real_mae, .4, label='Real replay', color='#4C72B0')
  axes[1, 1].bar(x + .2, fake_mae, .4, label='Fake replay', color='#DD8452')
  axes[1, 1].set(title='MAE per class', xlabel='Class', ylabel='MAE',
                 xticks=x, xticklabels=[str(cls) for cls in class_ids])
  axes[1, 1].legend()
  axes[1, 1].grid(axis='y', alpha=.25)

  rm, fm = evaluation['real_metrics'], evaluation['fake_metrics']
  title = f'Fake vs real projection replay{f" — {mode}" if mode else ""}'
  subtitle = (
    f"Real: micro MAE {rm['mae_micro']:.4f}, macro MAE {rm['mae_macro']:.4f}, CCC {rm['ccc']:.4f}"
    f"   |   Fake: micro MAE {fm['mae_micro']:.4f}, macro MAE {fm['mae_macro']:.4f}, "
    f"CCC {fm['ccc']:.4f}")
  fig.suptitle(f'{title}\n{subtitle}', fontsize=13, fontweight='bold')
  fig.tight_layout(rect=(0, 0, 1, .94))
  path = os.path.join(out_dir, f'fake_vs_real_dashboard{filename_suffix}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')
  return path


def plot_dashboard(new_preds, old_preds, labels, num_classes, mae, ccc, out_dir,
                   run_label: str = '', mae_macro=None, mae_macro_old=None,
                   mae_stages=None, mae_stages_std=None, filename_suffix: str = '',
                   projected_stage_name: str = 'Projected',
                   src_dataset: str = None, new_dataset: str = None,
                   real_anchors=None, config_anchors=None):
  """
  Combined dashboard PNG with all key diagnostic plots for a single run.

  Layout (3 rows × 3 cols via GridSpec):
    Row 0: confusion_matrix | mae_per_class_new (bar+box) | prediction_scatter
    Row 1: pred_by_class projected | pred_by_class old | metrics table
    Row 2: prediction_histogram (full width, 3 columns)

  The confusion-matrix cell is left empty when num_classes > 15.

  The MAE per class bar+box cell uses a nested GridSpec (2 sub-rows).

  Args:
    new_preds     (np.ndarray): Shape (N,), projected model float predictions.
    old_preds     (np.ndarray): Shape (N,), old model float predictions.
    labels        (np.ndarray): Shape (N,), ground-truth labels.
    num_classes   (int): Number of distinct pain classes.
    mae           (float): Micro-averaged MAE for the projected model.
    ccc           (float): Global CCC for the projected model.
    out_dir       (str): Output directory.
    run_label     (str): Optional run identity string appended to the suptitle.
    mae_macro     (float | None): Macro-averaged MAE for the projected model.
    mae_macro_old (float | None): Macro-averaged MAE for the old model.
    mae_stages    (dict | None): Optional per-stage MAE block for a richer metrics
      table. Recognized keys (all optional, each a (micro, macro) tuple): 'old',
      'projected', 'refined' (source/cross-domain side) and 'preserve_before',
      'preserve_after' (new-model-on-test side). When None the table shows only
      the Old + Projected rows derived from new_preds/old_preds (legacy behavior).
    mae_stages_std (dict | None): Optional per-stage MAE *std* companion to mae_stages
      (same {stage: (micro, macro)} shape). Aggregate-only: when supplied, each source/
      preserve MAE cell renders as 'mean ± std'. None ⇒ plain mean cells (single-run runs).
    filename_suffix (str): Optional suffix before '.png' (e.g. '_linear_only') so a
      multi-mode run can emit one dashboard per mode. Default '' ⇒ dashboard.png.
    projected_stage_name (str): Stage label for the new-model panels (passed as
      new_preds). For refinement runs the caller passes the before-refinement stage
      so the panels match the standalone projected plots.
    src_dataset (str | None): Resolved old/source dataset name, named on the source
      (cross-domain) metrics-table section. None ⇒ generic wording.
    new_dataset (str | None): Resolved new/target dataset name, named on the preserve
      (new-model test) metrics-table section. None ⇒ generic wording.
    real_anchors (int | dict | None): Real number of anchors actually used, shown in the
      metrics table against the configured budget. An int for a single run; for an
      aggregate a {real_count: n_subtrials} frequency dict (counts vary per subtrial).
      None ⇒ the anchor row is omitted.
    config_anchors (int | None): Configured num_anchors budget, shown alongside real_anchors.
  """
  suffix     = f' | {run_label}' if run_label else ''
  active_stage = ('refined' if projected_stage_name.lower().startswith('refined')
                  else 'projected')
  labels_int = np.round(labels).astype(int)
  class_ids  = sorted(int(c) for c in np.unique(labels_int))

  # Pre-compute MAE per class (new model) for the bar+box cell
  new_mae_dict            = _mae_per_group(new_preds, labels, labels_int)
  groups                  = sorted(new_mae_dict)
  new_vals                = [new_mae_dict.get(g, (float('nan'), 0))[0] for g in groups]
  _, new_raw              = _raw_errors_per_group(new_preds, labels, labels_int)

  fig = plt.figure(figsize=(26, 28))
  gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.5, wspace=0.38)

  # ── Row 0, Col 0: confusion matrix (skipped when num_classes > 15) ──────────
  if num_classes <= 15:
    plot_confusion_matrix_cross(
      new_preds, labels, out_dir, num_classes,
      run_label=run_label, ax=fig.add_subplot(gs[0, 0]),
    )
  else:
    ax_cm = fig.add_subplot(gs[0, 0])
    ax_cm.axis('off')
    ax_cm.text(
      0.5, 0.5, f'Confusion matrix skipped\n(num_classes={num_classes} > 15)',
      ha='center', va='center', fontsize=10,
    )

  # ── Row 0, Col 1: MAE per class — new model (bar + box) ─────────────────────
  inner_mae = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=gs[0, 1], height_ratios=[2, 1], hspace=0.08,
  )
  _draw_bar_boxplot(
    fig.add_subplot(inner_mae[0]),
    fig.add_subplot(inner_mae[1]),
    groups, new_vals, new_raw,
    'MAE', f'MAE per class — {projected_stage_name}{suffix}', 'darkorange',
  )
  fig.axes[-1].set_xlabel('Labels', fontsize=8)

  # ── Row 0, Col 2: prediction scatter (nested 1×2) ───────────────────────────
  inner_scatter = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[0, 2], wspace=0.35,
  )
  plot_prediction_scatter(
    new_preds, old_preds, labels, out_dir,
    run_label=run_label, new_name=projected_stage_name,
    axes=[fig.add_subplot(inner_scatter[0]), fig.add_subplot(inner_scatter[1])],
  )

  # ── Row 1, Col 0: predictions by class — raw ────────────────────────────────
  ax_raw = fig.add_subplot(gs[1, 0])
  color  = '#4C72B0'
  face_rgba = list(mcolors.to_rgba(color))
  face_rgba[3] = 0.5
  x = np.arange(len(class_ids))
  box_data = []
  for cid in class_ids:
    mask = labels_int == cid
    arr  = new_preds[mask].astype(np.float64)
    box_data.append(arr if arr.size > 0 else np.array([np.nan]))
  bp = ax_raw.boxplot(
    box_data, positions=x, widths=0.5, patch_artist=True,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                    markerfacecolor=color, markeredgecolor='none'),
    medianprops=dict(color='#222222', linewidth=1.5),
    whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0),
  )
  for patch in bp['boxes']:
    patch.set_facecolor(face_rgba)
    patch.set_edgecolor(color)
  for i, cid in enumerate(class_ids):
    ax_raw.plot([x[i] - 0.38, x[i] + 0.38], [cid, cid],
                color='#C44E52', linewidth=2.0, zorder=5)
  ax_raw.set_xticks(x)
  ax_raw.set_xticklabels([str(c) for c in class_ids], rotation=45, ha='right', fontsize=7)
  ax_raw.set_title(f'Pred by class — {projected_stage_name}{suffix}', fontsize=9, fontweight='bold')
  ax_raw.set_xlabel('True pain class', fontsize=8)
  ax_raw.set_ylabel('Predicted value', fontsize=8)
  ax_raw.grid(axis='y', alpha=0.3)

  # ── Row 1, Col 1: predictions by class — old model ──────────────────────────
  ax_old = fig.add_subplot(gs[1, 1])
  box_data_old = []
  for cid in class_ids:
    mask = labels_int == cid
    arr  = old_preds[mask].astype(np.float64)
    box_data_old.append(arr if arr.size > 0 else np.array([np.nan]))
  bp2 = ax_old.boxplot(
    box_data_old, positions=x, widths=0.5, patch_artist=True,
    showfliers=True,
    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                    markerfacecolor=color, markeredgecolor='none'),
    medianprops=dict(color='#222222', linewidth=1.5),
    whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0),
  )
  for patch in bp2['boxes']:
    patch.set_facecolor(face_rgba)
    patch.set_edgecolor(color)
  for i, cid in enumerate(class_ids):
    ax_old.plot([x[i] - 0.38, x[i] + 0.38], [cid, cid],
                color='#C44E52', linewidth=2.0, zorder=5)
  ax_old.set_xticks(x)
  ax_old.set_xticklabels([str(c) for c in class_ids], rotation=45, ha='right', fontsize=7)
  ax_old.set_title(f'Pred by class — Old model{suffix}', fontsize=9, fontweight='bold')
  ax_old.set_xlabel('True pain class', fontsize=8)
  ax_old.set_ylabel('Predicted value', fontsize=8)
  ax_old.grid(axis='y', alpha=0.3)

  # ── Row 1, Col 2: metrics table ──────────────────────────────────────────────
  ax_tbl = fig.add_subplot(gs[1, 2])
  ax_tbl.axis('off')
  mae_micro_old = float(np.mean(np.abs(old_preds - labels)))

  def _pair(v):
    return f'{float(v):.4f}' if (v is not None and np.isfinite(float(v))) else '—'

  stages     = mae_stages or {}
  stages_std = mae_stages_std or {}

  def _cell(stage_key, mean_tuple):
    """'micro / macro' cell, appending '± std' per value when mae_stages_std supplies it."""
    std_tuple = stages_std.get(stage_key)
    def _fmt(m, s):
      if m is None or not np.isfinite(float(m)):
        return '—'
      base = f'{float(m):.4f}'
      if s is not None and np.isfinite(float(s)):
        return f'{base} ± {float(s):.4f}'
      return base
    s0 = std_tuple[0] if std_tuple is not None else None
    s1 = std_tuple[1] if std_tuple is not None else None
    return f'{_fmt(mean_tuple[0], s0)} / {_fmt(mean_tuple[1], s1)}'

  # Source (cross-domain) MAE per stage. Fall back to the scalar args when a stage
  # is not supplied in mae_stages so the legacy Old+Projected view still renders.
  src_old       = stages.get('old',       (mae_micro_old, mae_macro_old))
  src_projected = stages.get('projected', (mae, mae_macro))
  src_refined   = stages.get('refined')
  src_section = f'── source (cross-domain: {src_dataset}) ──' if src_dataset else '── source (cross-domain) ──'
  rows_data = [[src_section, '']]
  rows_data.append(['MAE micro / macro (old)',       _cell('old', src_old)])
  rows_data.append(['MAE micro / macro (projected)', _cell('projected', src_projected)])
  if src_refined is not None:
    rows_data.append(['MAE micro / macro (refined)', _cell('refined', src_refined)])
  # Rounded+clamped MAE (matches the training `test_l1_error` definition) shown alongside
  # the continuous values for direct comparison against summary.csv all_test_l1_error.
  old_r = _compute_rounded_mae(old_preds, labels, num_classes)
  new_r = _compute_rounded_mae(new_preds, labels, num_classes)
  rows_data.append(['MAE micro / macro rounded (old)',       f'{_pair(old_r[0])} / {_pair(old_r[1])}'])
  rows_data.append([f'MAE micro / macro rounded ({active_stage})',
                    f'{_pair(new_r[0])} / {_pair(new_r[1])}'])
  # Preserve (new-model-on-test) MAE before vs after refinement, when available.
  pre_b = stages.get('preserve_before')
  pre_a = stages.get('preserve_after')
  if pre_b is not None or pre_a is not None:
    preserve_section = (f'── preserve ({new_dataset} test) ──' if new_dataset
                        else '── preserve (new-model test) ──')
    rows_data.append([preserve_section, ''])
    if pre_b is not None:
      rows_data.append(['MAE micro / macro (before)', _cell('preserve_before', pre_b)])
    if pre_a is not None:
      rows_data.append(['MAE micro / macro (after)',  _cell('preserve_after', pre_a)])
  rows_data.append(['── overall ──', ''])
  rows_data.append([f'CCC ({active_stage})', f'{ccc:.4f}'])
  rows_data.append(['N samples',       str(len(labels))])
  rows_data.append(['N classes',       str(num_classes)])
  # Real anchors actually used (int, or {count: n_subtrials} for an aggregate) vs the
  # configured num_anchors budget — the two diverge when few source samples are available.
  if real_anchors is not None or config_anchors is not None:
    cfg_str = str(config_anchors) if config_anchors is not None else '—'
    rows_data.append(['N anchors (real / config)',
                      f'{real_anchors if real_anchors is not None else "—"} / {cfg_str}'])

  tbl = ax_tbl.table(
    cellText=rows_data, colLabels=['Metric', 'micro / macro'],
    loc='center', cellLoc='center',
  )
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(9)
  tbl.scale(1.2, 1.55)
  for col in range(2):
    tbl[(0, col)].set_facecolor('#4C72B0')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')
  for r, row in enumerate(rows_data, start=1):
    is_section = str(row[0]).startswith('──')
    for col in range(2):
      cell = tbl[(r, col)]
      if is_section:
        cell.set_facecolor('#dbe4ff')
        cell.set_text_props(fontweight='bold')
      else:
        cell.set_facecolor('#eef2ff' if r % 2 == 0 else 'white')
  ax_tbl.set_title('Metrics Summary (per stage)', fontsize=10, fontweight='bold', pad=8)

  # ── Row 2: prediction histogram (full width, nested 1×2) ────────────────────
  inner_hist = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[2, :], wspace=0.3,
  )
  plot_predictions_histogram(
    new_preds, old_preds, labels, out_dir,
    run_label=run_label, new_name=projected_stage_name,
    axes=[fig.add_subplot(inner_hist[0]), fig.add_subplot(inner_hist[1])],
  )

  # ── Row 3: MAE improvement per class (full width) ───────────────────────────
  plot_mae_improvement_per_class(
    new_preds, old_preds, labels, out_dir,
    run_label=run_label, ax=fig.add_subplot(gs[3, :]),
  )

  fig.suptitle(f'Dashboard{suffix}', fontsize=15, fontweight='bold', y=1.002)
  path = os.path.join(out_dir, f'dashboard{filename_suffix}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')


def plot_refinement_modes_comparison(refine_items, refine_preds_by_mode, old_preds, labels,
                                     base_stages, out_dir, run_label: str = '',
                                     src_dataset: str = None, new_dataset: str = None):
  """
  Combined cross-mode comparison for a --refinement 3 run: one figure contrasting every
  refinement mode (linear_only / projector_linear) against the projected baseline.

  Top panel: a metrics table, one row per stage (old, projected, and each refined mode),
  with source-domain MAE (micro/macro) and the new-model preserve MAE (micro/macro;
  'projected' shows the native before-refinement preserve, each mode shows its after).
  Bottom panel: grouped per-class MAE-improvement bars (old − refined) — one bar per mode
  per class, so the modes' per-class gains are directly comparable.

  Modes missing reconstructed predictions still appear in the table (with '—') but are
  omitted from the grouped bars.

  Args:
    refine_items         (list[tuple[str, dict]]): (mode, refinement-block) pairs.
    refine_preds_by_mode (dict[str, tuple]): mode → (before_preds, after_preds), real scale.
    old_preds            (np.ndarray): Shape (N,), old-model predictions (source baseline).
    labels               (np.ndarray): Shape (N,), ground-truth labels.
    base_stages          (dict): {'old': (micro,macro), 'projected': (micro,macro)} baseline.
    out_dir              (str): Output directory.
    run_label            (str): Optional run identity string appended to the titles.
    src_dataset          (str | None): Resolved old/source dataset name, named in the
      'src MAE' table column headers. None ⇒ generic 'src'.
    new_dataset          (str | None): Resolved new/target dataset name, named in the
      'preserve' table column headers. None ⇒ generic 'preserve'.
  """
  old_preds  = np.asarray(old_preds, dtype=np.float32).reshape(-1)
  labels     = np.asarray(labels,    dtype=np.float32).reshape(-1)
  labels_int = np.round(labels).astype(int)
  modes  = [m for m, _ in refine_items]
  suffix = f' | {run_label}' if run_label else ''

  def _fmt(v):
    return f'{float(v):.4f}' if (v is not None and np.isfinite(float(v))) else '—'

  # --- metrics table: one row per stage ---
  src_col = f'src MAE ({src_dataset})' if src_dataset else 'src MAE'
  pre_col = f'preserve ({new_dataset})' if new_dataset else 'preserve'
  col_labels = ['Stage', f'{src_col} micro', f'{src_col} macro',
                f'{pre_col} micro', f'{pre_col} macro']
  o = base_stages.get('old',       (float('nan'), float('nan')))
  p = base_stages.get('projected', (float('nan'), float('nan')))
  rows = [['old', _fmt(o[0]), _fmt(o[1]), '—', '—']]
  # 'projected' preserve = native new model before any refinement (same for every mode);
  # take it from the first block that carries it.
  pb_micro = pb_macro = float('nan')
  for _m, _blk in refine_items:
    _v = _newtest_preserve_mae(_blk, 'before')
    if any(np.isfinite(x) for x in _v):
      pb_micro, pb_macro = _v
      break
  rows.append(['projected (before)', _fmt(p[0]), _fmt(p[1]), _fmt(pb_micro), _fmt(pb_macro)])
  for _mode, _blk in refine_items:
    rp = refine_preds_by_mode.get(_mode)
    src_micro, src_macro = (_compute_global_mae(np.asarray(rp[1]).reshape(-1), labels)
                            if rp is not None else (float('nan'), float('nan')))
    pa = _newtest_preserve_mae(_blk, 'after')
    rows.append([f'refined: {_mode}', _fmt(src_micro), _fmt(src_macro), _fmt(pa[0]), _fmt(pa[1])])

  # --- grouped per-class improvement (old − refined) per mode ---
  old_mae   = _mae_per_group(old_preds, labels, labels_int)
  class_ids = sorted(old_mae)
  per_mode_diffs = {}
  for _mode in modes:
    rp = refine_preds_by_mode.get(_mode)
    if rp is None:
      continue
    after_mae = _mae_per_group(np.asarray(rp[1]).reshape(-1), labels, labels_int)
    per_mode_diffs[_mode] = [
      old_mae.get(c, (float('nan'), 0))[0] - after_mae.get(c, (float('nan'), 0))[0]
      for c in class_ids
    ]

  fig = plt.figure(figsize=(20, 12))
  gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 1.4], hspace=0.3)

  ax_tbl = fig.add_subplot(gs[0])
  ax_tbl.axis('off')
  tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(10)
  tbl.scale(1.1, 1.7)
  for c in range(len(col_labels)):
    tbl[(0, c)].set_facecolor('#4C72B0')
    tbl[(0, c)].set_text_props(color='white', fontweight='bold')
  for r in range(1, len(rows) + 1):
    for c in range(len(col_labels)):
      tbl[(r, c)].set_facecolor('#eef2ff' if r % 2 == 0 else 'white')
  ax_tbl.set_title(f'Refinement mode comparison — metrics{suffix}',
                   fontsize=12, fontweight='bold', pad=10)

  ax_bar = fig.add_subplot(gs[1])
  if per_mode_diffs and class_ids:
    x = np.arange(len(class_ids))
    n = len(per_mode_diffs)
    w = 0.8 / max(n, 1)
    cmap = plt.get_cmap('tab10')
    for i, (_mode, diffs) in enumerate(per_mode_diffs.items()):
      ax_bar.bar(x + (i - (n - 1) / 2) * w, diffs, w, label=_mode, color=cmap(i),
                 alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_bar.axhline(0.0, color='black', linewidth=0.8, alpha=0.6)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([str(c) for c in class_ids], rotation=45, ha='right')
    ax_bar.set_xlabel('True pain class')
    ax_bar.set_ylabel('MAE improvement (old − refined)')
    ax_bar.set_title(f'Per-class MAE improvement vs old model, by refinement mode{suffix}',
                     fontsize=12, fontweight='bold')
    ax_bar.legend(title='mode', fontsize=9)
    ax_bar.grid(axis='y', alpha=0.3)
  else:
    ax_bar.axis('off')
    ax_bar.text(0.5, 0.5, 'No reconstructed refined predictions available.',
                ha='center', va='center', fontsize=11, color='#555555')

  path = os.path.join(out_dir, 'dashboard_refinement_comparison.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')


# ── embedding-reconstruction diagnostic ──────────────────────────────────────

def _resolve_scope_sample_ids(data, fmt):
  """
  Read the sample-id scope and split label from a loaded pkl.

  Args:
    data (dict): Deserialized pkl contents.
    fmt  (str):  'grid' or 'standalone' (output of _detect_format).

  Returns:
    tuple[np.ndarray, str]:
      - sample_ids: Shape (N,), int64. Full content of
        data['new_model_tensors']['sample_ids'].
      - split_name: Value of old_model_csv (e.g. 'test', 'val', 'train',
        'all', 'exc_train'). Used for filenames and titles.
  """
  sample_ids = np.asarray(
    data['new_model_tensors']['sample_ids'], dtype=np.int64,
  )
  if fmt == 'grid':
    split_name = str(data['trial_params']['old_model_csv'])
  else:
    split_name = str(data['config_cross_space_projection'].get('old_model_csv', 'unknown'))
  return sample_ids, split_name


def _fetch_real_embeddings_from_linear(linear_bundle, target_ids):
  """
  Pull real new-model embeddings out of the stored linear-projector splits.

  Args:
    linear_bundle (dict): Output of _extract_linear_bundle. Must contain a
      'splits' dict with 'train'/'val'/'test' entries holding 'target' and
      'sample_ids' arrays.
    target_ids    (Iterable[int]): Sample ids of interest.

  Returns:
    dict[int, np.ndarray]: {sample_id: target_vector_1d}. Empty when
      linear_bundle is None or its splits are missing.
  """
  if linear_bundle is None:
    return {}
  splits = linear_bundle.get('splits') or {}
  out = {}
  target_set = set(int(s) for s in target_ids)
  for split_name in ('train', 'val', 'test'):
    split = splits.get(split_name)
    if not split:
      continue
    sids = np.asarray(split.get('sample_ids'), dtype=np.int64).reshape(-1)
    tgt  = np.asarray(split.get('target'),     dtype=np.float32)
    if sids.size == 0 or tgt.size == 0:
      continue
    if tgt.ndim != 2 or tgt.shape[0] != sids.shape[0]:
      print(f'[emb_recon] linear splits[{split_name!r}]: shape mismatch '
            f'sample_ids={sids.shape} target={tgt.shape} — skipping split.')
      continue
    for sid, vec in zip(sids, tgt):
      sid_int = int(sid)
      if sid_int in target_set and sid_int not in out:
        out[sid_int] = vec
  return out


def _load_split_embeddings(data, fmt, pkl_path, split_name, out_dir):
  """
  Get the NEW MODEL's real embeddings + labels for one of its own splits.

  The split is the new model's actual ``<split>.csv`` (train/val/test) resolved
  relative to the new-model checkpoint — NOT the projector's splits stored in the
  pkl (those are the anchors / val.csv subsets, a different set). To keep cost
  bounded the CSV rows are randomly subsampled to SPLIT_SUBSAMPLE_FRAC *before*
  extraction, then real embeddings are produced by head-only inference using the
  new model's NATIVE features folder. Results are cached to a safetensors file so
  re-runs skip the (expensive) extraction.

  Args:
    data       (dict): Deserialized pkl contents.
    fmt        (str):  'grid' or 'standalone'.
    pkl_path   (str):  Path to the loaded pkl (used by grid resolution).
    split_name (str):  'train', 'val', or 'test'.
    out_dir    (str):  Directory for the safetensors cache and the temp CSV.

  Returns:
    tuple[np.ndarray, np.ndarray] | None: (embeddings (S, D) float32, labels (S,)
      float32) for the subsampled split, or None when it cannot be produced.
  """
  from safetensors.numpy import load_file as st_load, save_file as st_save

  cache_path = os.path.join(
    out_dir, f'split_impact_emb_{split_name}_f{SPLIT_SUBSAMPLE_FRAC:g}.safetensors')
  if os.path.isfile(cache_path):
    try:
      cached = st_load(cache_path)
      emb    = np.asarray(cached['embeddings'], dtype=np.float32)
      labels = np.asarray(cached['labels'],     dtype=np.float32).reshape(-1)
      print(f'[split-impact] loaded cached {split_name!r} embeddings '
            f'({emb.shape[0]} samples) from {cache_path}')
      return emb, labels
    except Exception as exc:
      print(f'[split-impact] cache read failed ({exc}); re-extracting.')

  try:
    from cross_space_projection import (
      _build_model, _extract_embeddings, _load_config, _resolve_split_csv,
    )
    new_model_pth = _resolve_new_model_pth(data, fmt, pkl_path)
    if new_model_pth is None:
      print(f'[split-impact] no new-model checkpoint resolvable — cannot extract '
            f'{split_name!r}.')
      return None
    csv_path = _resolve_split_csv(new_model_pth, split_name)
    if not os.path.isfile(csv_path):
      print(f'[split-impact] split CSV not found: {csv_path} — cannot extract '
            f'{split_name!r}.')
      return None

    # --- Subsample CSV rows BEFORE extraction to manage cost ---
    df = pd.read_csv(csv_path, sep='\t',
                     dtype={'sample_name': str, 'subject_name': str})
    if SPLIT_SUBSAMPLE_FRAC < 1.0:
      df = df.sample(frac=SPLIT_SUBSAMPLE_FRAC, random_state=SPLIT_SUBSAMPLE_SEED)
    if len(df) < 5:
      print(f'[split-impact] only {len(df)} {split_name!r} rows after subsampling '
            f'(frac={SPLIT_SUBSAMPLE_FRAC}) — too few; skipped.')
      return None
    temp_csv = os.path.join(out_dir, f'_split_impact_{split_name}_sub.csv')
    df.to_csv(temp_csv, sep='\t', index=False)

    config_model = _load_config(new_model_pth)
    model        = _build_model(config_model)
    print(f'[split-impact] extracting real {split_name!r} embeddings for {len(df)} '
          f'subsampled rows from {csv_path}')
    raw = _extract_embeddings(model, new_model_pth, temp_csv, config_model)  # native features
    emb    = np.asarray(raw['embeddings'], dtype=np.float32)
    labels = np.asarray(raw['labels'],     dtype=np.float32).reshape(-1)
    if emb.size == 0:
      return None

    try:
      st_save({'embeddings': emb, 'labels': labels,
               'sample_ids': np.asarray(raw['sample_ids'], dtype=np.int64)}, cache_path)
      print(f'[split-impact] cached {split_name!r} embeddings → {cache_path}')
    except Exception as exc:
      print(f'[split-impact] cache write failed ({exc}); continuing.')
    return emb, labels
  except Exception as exc:
    print(f'[split-impact] extraction of {split_name!r} failed: {exc}')
    return None


def _apply_projector_ckpt(data, old_emb, ckpt_path, stage_label='refinement'):
  """
  Rebuild a projector from a checkpoint and apply it to the source embeddings.

  Shared core behind the before-/after-refinement embedding helpers. Rebuilds the
  projector module from ckpt_path using the 'linear_projector' bundle's kind/config,
  then applies it to old_emb the same way the original projector was applied at
  projection time. The regressor (head.linear) is irrelevant here: this returns
  embeddings, not predictions, so only the projector matters.

  Args:
    data        (dict): Deserialized pkl. Needs a 'linear_projector' bundle
      (norm_stats/kind/config).
    old_emb     (np.ndarray): Raw old-model source embeddings. Shape (N, D_old) —
      the same matrix the original projector consumed
      (old_model_tensors['embeddings']).
    ckpt_path   (str): Path to the projector state_dict (.pt) to load.
    stage_label (str): Human label used only in [WARN] messages (e.g.
      'after-refinement', 'before-refinement').

  Returns:
    np.ndarray | None: Projected embeddings, shape (N, D_new), sample-aligned with
      old_emb. None with a [WARN] when the checkpoint / bundle is missing or the
      projector cannot be rebuilt/applied.
  """
  if not ckpt_path or not os.path.isfile(ckpt_path):
    print(f'[WARN] {stage_label} embeddings: projector checkpoint not found ({ckpt_path}) — skipped.')
    return None
  bundle = _extract_linear_bundle(data)
  if bundle is None:
    print(f'[WARN] {stage_label} embeddings: no linear_projector bundle in pkl — skipped.')
    return None

  try:
    norm_stats = bundle.get('norm_stats')
    old_emb = np.asarray(old_emb, dtype=np.float32)
    d_old = int(old_emb.shape[1])
    if norm_stats is not None and norm_stats.get('new_mean') is not None:
      d_new = int(np.asarray(norm_stats['new_mean']).shape[-1])
    else:
      d_new = int(np.asarray(data['new_model_tensors']['embeddings']).shape[1])
    kind = (bundle.get('kind') or 'linear').lower()
    if kind not in ('linear', 'mlp', 'autoencoder'):
      kind = 'linear'  # procrustes / linear_close use an nn.Linear under the hood
    activation = (bundle.get('config') or {}).get('mlp_activation')
    num_layers = int((bundle.get('config') or {}).get('mlp_num_layers') or 1)
    encoder_ratio = int((bundle.get('config') or {}).get('encoder_ratio') or 4)

    from cross_space_projection import _build_projector_network, _apply_linear_projector
    projector = _build_projector_network(d_old, d_new, kind, activation, num_layers, encoder_ratio)
    projector.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    projector.eval()
    return _apply_linear_projector(projector, norm_stats, old_emb)
  except Exception as exc:
    print(f'[WARN] {stage_label} embeddings: failed to apply projector — {exc}')
    return None


def _refined_projected_embeddings(data, old_emb, refine_block=None):
  """
  Re-project the source embeddings through the REFINED projector for after-
  refinement UMAPs.

  Thin wrapper over _apply_projector_ckpt using the refinement checkpoint
  data['refinement']['projector_after_pth'].

  Args:
    data    (dict): Deserialized pkl. Needs a 'linear_projector' bundle
      (norm_stats/kind/config); the refinement block is supplied via refine_block.
    old_emb (np.ndarray): Raw old-model source embeddings. Shape (N, D_old) —
      the same matrix the original projector consumed (old_model_tensors['embeddings']).
    refine_block (dict | None): The refinement block to use (with 'projector_after_pth').
      Defaults to data.get('refinement'); pass one entry of data['refinements'] to
      reconstruct a specific mode in a multi-mode (--refinement 3) run.

  Returns:
    np.ndarray | None: Refined projected embeddings, shape (N, D_new), sample-aligned
      with old_emb. None (silently) for non-refinement runs, or with a [WARN] when the
      checkpoint / bundle is missing or the projector cannot be rebuilt/applied.
  """
  ref = refine_block if refine_block is not None else data.get('refinement')
  if not ref:
    return None  # non-refinement run — nothing to do
  return _apply_projector_ckpt(
    data, old_emb, ref.get('projector_after_pth'), stage_label='after-refinement')


def _projected_before_refinement_embeddings(data, old_emb, refine_block=None):
  """
  Re-project the source embeddings through the BEFORE-refinement projector.

  This is the true "after projection, before refinement" embedding stage. In
  projector_linear mode the projector is refined, so the before-refinement
  projector lives at refine_block['projector_before_pth']. In linear_only mode the
  projection is frozen (no projector_before_pth); we fall back to the trained
  projector from the 'linear_projector' bundle ('ckpt_path'), which by construction
  equals the projection used for the stored embeddings.

  Args:
    data    (dict): Deserialized pkl. Needs a 'linear_projector' bundle.
    old_emb (np.ndarray): Raw old-model source embeddings. Shape (N, D_old).
    refine_block (dict | None): Refinement block (with 'projector_before_pth').
      Defaults to data.get('refinement'); pass one entry of data['refinements'] for
      a specific mode in a multi-mode (--refinement 3) run.

  Returns:
    np.ndarray | None: Before-refinement projected embeddings, shape (N, D_new),
      sample-aligned with old_emb. None (silently) for non-refinement runs, or with a
      [WARN] when the checkpoint / bundle is missing or the projector cannot be applied.
  """
  ref = refine_block if refine_block is not None else data.get('refinement')
  if not ref:
    return None  # non-refinement run — nothing to do
  bundle = _extract_linear_bundle(data)
  ckpt = ref.get('projector_before_pth') or (bundle or {}).get('ckpt_path')
  return _apply_projector_ckpt(
    data, old_emb, ckpt, stage_label='before-refinement')


def _label_denorm(data):
  """
  Recover the label denormalization scale used to map linear logits to real labels.

  predictions = logits * label_denorm exactly at projection time, so the constant
  scale is recoverable from the stored tensors (works for both standalone and grid
  pkls, where new_model_config is absent). Falls back to the new-model training
  config (max_label when normalize_labels) and finally to 1.0.

  Args:
    data (dict): Deserialized pkl contents.

  Returns:
    float: The label_denorm multiplier (1.0 when labels were not normalized).
  """
  nt = data.get('new_model_tensors') or {}
  logits = nt.get('logits')
  preds  = nt.get('predictions')
  if logits is not None and preds is not None:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    preds  = np.asarray(preds,  dtype=np.float64).reshape(-1)
    if logits.size and logits.size == preds.size:
      mask = np.abs(logits) > 1e-8
      if mask.any():
        return float(np.median(preds[mask] / logits[mask]))
  cfg = (data.get('new_model_config') or {}).get('config', {}) or {}
  if bool(cfg.get('normalize_labels', 0)) and cfg.get('max_label'):
    return float(cfg['max_label'])
  return 1.0


def _apply_linear_ckpt(emb, linear_pth, denorm):
  """
  Classify embeddings with a checkpointed plain nn.Linear head.

  Rebuilds an nn.Linear (dims read from the checkpoint's weight shape), applies it to
  `emb`, and scales the logits by `denorm` to real label units.

  Args:
    emb        (np.ndarray): Input embeddings (linear input). Shape (N, D).
    linear_pth (str): Path to the head.linear state_dict (.pt).
    denorm     (float): Multiplier mapping logits to the real label scale.

  Returns:
    np.ndarray: Predictions in real label scale. Shape (N,).
  """
  sd = torch.load(linear_pth, map_location='cpu')
  out_f, in_f = sd['weight'].shape
  linear = torch.nn.Linear(int(in_f), int(out_f))
  linear.load_state_dict(sd)
  linear.eval()
  with torch.no_grad():
    logits = linear(torch.as_tensor(emb, dtype=torch.float32)).cpu().numpy()
  logits = logits.squeeze(-1) if logits.ndim > 1 else logits
  return (logits * float(denorm)).astype(np.float32).reshape(-1)


def _predict_with_ckpts(old_emb, proj_pth, linear_pth, norm_stats,
                        d_old, d_new, kind, activation, denorm, num_layers=1,
                        encoder_ratio=4):
  """
  Project old-space embeddings then classify with a checkpointed linear head.

  Rebuilds the projector (via cross_space_projection._build_projector_network), applies
  it in the same order as projection time, then classifies with _apply_linear_ckpt.

  Args:
    old_emb       (np.ndarray): Raw old-model embeddings. Shape (N, D_old).
    proj_pth      (str): Path to the projector state_dict (.pt).
    linear_pth    (str): Path to the head.linear state_dict (.pt).
    norm_stats    (dict | None): Projector normalization stats from the bundle.
    d_old         (int): Old embedding dim (projector input).
    d_new         (int): New embedding dim (projector output / linear input).
    kind          (str): Projector kind ('linear' | 'mlp' | 'autoencoder').
    activation    (str | None): Activation for kind in ('mlp', 'autoencoder').
    denorm        (float): Multiplier mapping logits to the real label scale.
    num_layers    (int): Depth of the 'mlp' projector (ignored for 'linear'/'autoencoder';
      default 1). Must match the depth the checkpoint was trained with or load_state_dict fails.
    encoder_ratio (int): Bottleneck divisor for kind == 'autoencoder' (ignored otherwise;
      default 4). Must match the ratio the checkpoint was trained with or load_state_dict fails.

  Returns:
    np.ndarray: Predictions in real label scale. Shape (N,).
  """
  from cross_space_projection import _build_projector_network, _apply_linear_projector
  projector = _build_projector_network(d_old, d_new, kind, activation, num_layers, encoder_ratio)
  projector.load_state_dict(torch.load(proj_pth, map_location='cpu'))
  projector.eval()
  projected = _apply_linear_projector(projector, norm_stats, old_emb)
  return _apply_linear_ckpt(projected, linear_pth, denorm)


def _refinement_predictions(data, old_emb, refine_block=None):
  """
  Recompute pre-refinement and after-refinement predictions on the source samples.

  Both are rebuilt from the refinement checkpoints so the result is correct regardless of
  REFINEMENT_CONFIG's report_after_refinement (which may already have overwritten
  new_model_tensors).

  Two paths:
  - linear_only mode (--refinement 1, projector_{before,after}_pth both None): the
    projection is frozen, so new_model_tensors['embeddings'] already holds the
    before-refinement projection. Both stages just apply their linear_{before,after}_pth
    head to those stored embeddings — no projector and no 'linear_projector' bundle are
    needed, which is what makes this work for anchor-interpolation sims (l2/rbf/cosine/…)
    that never have a projector bundle.
  - projector_linear mode (--refinement 2): the projector is refined too, so each stage
    rebuilds its own projector_{before,after}_pth (via the bundle's kind/norm_stats) and
    applies it to old_emb before the linear head.

  Args:
    data    (dict): Deserialized pkl. For projector_linear it needs a 'linear_projector'
      bundle; for linear_only only the refinement linear checkpoints are required.
    old_emb (np.ndarray): Raw old-model source embeddings. Shape (N, D_old). Used by the
      projector_linear path; ignored by the linear_only path.
    refine_block (dict | None): The refinement block to reconstruct (linear_{before,after}_pth
      and, for projector_linear, projector_{before,after}_pth). Defaults to
      data.get('refinement'); pass one entry of data['refinements'] for a specific mode
      in a multi-mode (--refinement 3) run.

  Returns:
    tuple[np.ndarray, np.ndarray] | None: (before_preds, after_preds), each shape
      (N,) in real label scale. None (silently) for non-refinement runs, or with a
      [WARN] when checkpoints / bundle are missing or recomputation fails.
  """
  ref = refine_block if refine_block is not None else data.get('refinement')
  if not ref:
    return None  # non-refinement run

  linear_before = ref.get('linear_before_pth')
  linear_after  = ref.get('linear_after_pth')
  lin_missing = [k for k, p in (('linear_before_pth', linear_before),
                                ('linear_after_pth', linear_after))
                 if not p or not os.path.isfile(p)]
  if lin_missing:
    print(f'[WARN] after-refinement MAE-per-class: missing checkpoint(s) {lin_missing} — skipped.')
    return None

  # ── linear_only path: frozen projection → apply linear heads to stored embeddings. ──
  is_linear_only = not ref.get('projector_before_pth') and not ref.get('projector_after_pth')
  if is_linear_only:
    try:
      emb    = np.asarray(data['new_model_tensors']['embeddings'], dtype=np.float32)
      denorm = _label_denorm(data)
      before_preds = _apply_linear_ckpt(emb, linear_before, denorm)
      after_preds  = _apply_linear_ckpt(emb, linear_after,  denorm)
      return before_preds, after_preds
    except Exception as exc:
      print(f'[WARN] after-refinement MAE-per-class: failed to recompute predictions — {exc}')
      return None

  # ── projector_linear path: rebuild the per-stage projector then classify. ──
  bundle = _extract_linear_bundle(data)
  if bundle is None:
    print('[WARN] after-refinement MAE-per-class: no linear_projector bundle in pkl — skipped.')
    return None

  # The projector checkpoints exist only in projector_linear mode; a stage that did not
  # refine the projector falls back to the bundle's trained projector (before == after
  # projection for that stage; only the linear head differs).
  proj_before = ref.get('projector_before_pth') or bundle.get('ckpt_path')
  proj_after  = ref.get('projector_after_pth')  or bundle.get('ckpt_path')
  proj_missing = [k for k, p in (('projector(before)', proj_before),
                                 ('projector(after)', proj_after))
                  if not p or not os.path.isfile(p)]
  if proj_missing:
    print(f'[WARN] after-refinement MAE-per-class: missing checkpoint(s) {proj_missing} — skipped.')
    return None

  try:
    norm_stats = bundle.get('norm_stats')
    old_emb = np.asarray(old_emb, dtype=np.float32)
    d_old = int(old_emb.shape[1])
    if norm_stats is not None and norm_stats.get('new_mean') is not None:
      d_new = int(np.asarray(norm_stats['new_mean']).shape[-1])
    else:
      d_new = int(np.asarray(data['new_model_tensors']['embeddings']).shape[1])
    kind = (bundle.get('kind') or 'linear').lower()
    if kind not in ('linear', 'mlp', 'autoencoder'):
      kind = 'linear'  # procrustes / linear_close use an nn.Linear under the hood
    activation = (bundle.get('config') or {}).get('mlp_activation')
    num_layers = int((bundle.get('config') or {}).get('mlp_num_layers') or 1)
    encoder_ratio = int((bundle.get('config') or {}).get('encoder_ratio') or 4)
    denorm = _label_denorm(data)

    before_preds = _predict_with_ckpts(
      old_emb, proj_before, linear_before,
      norm_stats, d_old, d_new, kind, activation, denorm, num_layers, encoder_ratio)
    after_preds = _predict_with_ckpts(
      old_emb, proj_after, linear_after,
      norm_stats, d_old, d_new, kind, activation, denorm, num_layers, encoder_ratio)
    return before_preds, after_preds
  except Exception as exc:
    print(f'[WARN] after-refinement MAE-per-class: failed to recompute predictions — {exc}')
    return None


def _aggregate_refinement_modes(data, pkl_path):
  modes = list((data.get('fake_projection_evaluations') or {}).keys())
  if modes:
    return modes
  base = os.path.dirname(os.path.abspath(pkl_path))
  for rel in data.get('subtrial_pkls') or []:
    try:
      sub_path = os.path.join(base, rel)
      return [mode for mode, _ in _refine_items(_load_pkl(sub_path))]
    except Exception:
      continue
  return []


def _aggregate_refinement_predictions(data, pkl_path, mode):
  """Return aligned pooled (before, after) predictions for one aggregate mode."""
  expected = data.get('new_model_tensors') or {}
  expected_ids = np.asarray(expected.get('sample_ids')).reshape(-1)
  expected_labels = np.asarray(expected.get('labels'), dtype=np.float32).reshape(-1)
  base = os.path.dirname(os.path.abspath(pkl_path))

  if data.get('fake_projection_evaluations'):
    evaluation = data['fake_projection_evaluations'].get(mode)
    if evaluation is None:
      print(f'[WARN] aggregate refinement predictions ({mode}) unavailable — dashboard skipped.')
      return None
    after = np.asarray(evaluation['fake_predictions'], dtype=np.float32).reshape(-1)
    after_ids = np.asarray(evaluation['sample_ids']).reshape(-1)
    after_labels = np.asarray(evaluation['labels'], dtype=np.float32).reshape(-1)
    before_parts, id_parts, label_parts = [], [], []
    for rel in data.get('subtrial_pkls') or []:
      sub_path = os.path.join(base, rel)
      try:
        sub = _load_pkl(sub_path)
        part = (sub.get('fake_projection_evaluations') or {})[mode]
        before_parts.append(np.asarray(
          part['fake_before_predictions'], dtype=np.float32).reshape(-1))
        id_parts.append(np.asarray(part['sample_ids']).reshape(-1))
        label_parts.append(np.asarray(part['labels'], dtype=np.float32).reshape(-1))
      except Exception as exc:
        print(f'[WARN] aggregate refinement predictions ({mode}) incomplete '
              f'at {sub_path}: {exc} — dashboard skipped.')
        return None
    before = np.concatenate(before_parts) if before_parts else np.array([], dtype=np.float32)
    pooled_ids = np.concatenate(id_parts) if id_parts else np.array([], dtype=np.int64)
    pooled_labels = (np.concatenate(label_parts) if label_parts
                     else np.array([], dtype=np.float32))
  else:
    before_parts, after_parts, id_parts, label_parts = [], [], [], []
    for rel in data.get('subtrial_pkls') or []:
      sub_path = os.path.join(base, rel)
      try:
        sub = _rebase_standalone_paths(_load_pkl(sub_path), sub_path)
        block = dict(_refine_items(sub))[mode]
        predictions = _refinement_predictions(
          sub, np.asarray(sub['old_model_tensors']['embeddings'], dtype=np.float32),
          refine_block=block)
        if predictions is None:
          raise ValueError('checkpoint predictions unavailable')
        before, after = predictions
        tensors = sub['new_model_tensors']
        before_parts.append(np.asarray(before, dtype=np.float32).reshape(-1))
        after_parts.append(np.asarray(after, dtype=np.float32).reshape(-1))
        id_parts.append(np.asarray(tensors['sample_ids']).reshape(-1))
        label_parts.append(np.asarray(tensors['labels'], dtype=np.float32).reshape(-1))
      except Exception as exc:
        print(f'[WARN] aggregate refinement predictions ({mode}) incomplete '
              f'at {sub_path}: {exc} — dashboard skipped.')
        return None
    before = np.concatenate(before_parts) if before_parts else np.array([], dtype=np.float32)
    after = np.concatenate(after_parts) if after_parts else np.array([], dtype=np.float32)
    pooled_ids = np.concatenate(id_parts) if id_parts else np.array([], dtype=np.int64)
    pooled_labels = (np.concatenate(label_parts) if label_parts
                     else np.array([], dtype=np.float32))
    after_ids, after_labels = pooled_ids, pooled_labels

  aligned = (
    len(before) == len(after) == len(expected_ids) == len(expected_labels)
    and np.array_equal(pooled_ids, expected_ids)
    and np.allclose(pooled_labels, expected_labels)
    and np.array_equal(after_ids, expected_ids)
    and np.allclose(after_labels, expected_labels)
  )
  if not aligned:
    print(f'[WARN] aggregate refinement predictions ({mode}) are incomplete or '
          'misaligned with aggregate tensors — dashboard skipped.')
    return None
  return before, after


def plot_refinement_mae_per_class(after_preds, before_preds, old_preds, labels,
                                  out_dir, run_label: str = '', filename_suffix: str = ''):
  """
  Per-class MAE diagnostics for the refined model, mirroring plot_mae_per_class /
  plot_mae_improvement_per_class.

  Writes five PNGs (filename_suffix inserted before the trailing _bar/_box/.png so a
  multi-mode run, --refinement 3, can keep both modes side-by-side):
    - mae_per_class_refined{sfx}_bar.png / {sfx}_box.png: refined-model MAE per class.
    - mae_improvement_per_class_refined_vs_projected{sfx}.png: projected − refined.
    - mae_improvement_per_class_refined_vs_old{sfx}.png: old model − refined.
    - mae_improvement_per_class_combined{sfx}.png: 1×3 panel with all three source-split
      improvements — (old − before), (before − after), (old − after). Panel 1 is the
      old-vs-before-refinement comparison not emitted as its own standalone PNG.
  Positive improvement bars (green) mean the later stage lowered the error.

  Args:
    after_preds  (np.ndarray): Shape (N,), refined (after-refinement) predictions.
    before_preds (np.ndarray): Shape (N,), pre-refinement predictions.
    old_preds    (np.ndarray): Shape (N,), old-model predictions (Plot C baseline).
    labels       (np.ndarray): Shape (N,), ground-truth labels.
    out_dir      (str): Output directory.
    run_label    (str): Optional run identity string appended to plot titles.
    filename_suffix (str): Optional per-mode suffix (e.g. '_linear_only') inserted into
      every PNG name. Default '' ⇒ legacy filenames unchanged.
  """
  after_preds  = np.asarray(after_preds,  dtype=np.float32).reshape(-1)
  before_preds = np.asarray(before_preds, dtype=np.float32).reshape(-1)
  old_preds    = np.asarray(old_preds,    dtype=np.float32).reshape(-1)
  labels       = np.asarray(labels,       dtype=np.float32).reshape(-1)
  labels_int   = np.round(labels).astype(int)
  suffix = f' | {run_label}' if run_label else ''

  after_mae  = _mae_per_group(after_preds,  labels, labels_int)
  before_mae = _mae_per_group(before_preds, labels, labels_int)
  old_mae    = _mae_per_group(old_preds,    labels, labels_int)

  # Plot A — refined model MAE per class (bar + box).
  groups   = sorted(after_mae)
  after_vals = [after_mae.get(g, (float('nan'), 0))[0] for g in groups]
  _, after_raw = _raw_errors_per_group(after_preds, labels, labels_int)

  fig, ax = plt.subplots(figsize=(14, 5))
  _draw_mae_bar(ax, groups, after_vals, 'MAE',
                f'MAE per pain class — Refined (after){suffix}', 'seagreen')
  ax.set_xlabel('Labels')
  plt.tight_layout()
  path = os.path.join(out_dir, f'mae_per_class_refined{filename_suffix}_bar.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')

  fig, ax = plt.subplots(figsize=(14, 5))
  _draw_mae_box(ax, groups, after_raw, 'MAE', 'seagreen',
                title=f'MAE per pain class — Refined (after){suffix}')
  ax.set_xlabel('Labels')
  plt.tight_layout()
  path = os.path.join(out_dir, f'mae_per_class_refined{filename_suffix}_box.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')

  # Plots B & C — improvement per class (baseline − after); positive = refined better.
  # Each comparison also emits a companion 2-bar plot (baseline vs refined side by side).
  for base_mae, fname, title, cmp_fname, base_label, base_color, cmp_title in (
    (before_mae, f'mae_improvement_per_class_refined_vs_projected{filename_suffix}.png',
     f'MAE improvement per pain class (projected − refined){suffix}',
     f'mae_per_class_compare_refined_vs_projected{filename_suffix}.png',
     'Projected (before)', 'darkorange',
     f'MAE per pain class — projected vs refined{suffix}'),
    (old_mae, f'mae_improvement_per_class_refined_vs_old{filename_suffix}.png',
     f'MAE improvement per pain class (old − refined){suffix}',
     f'mae_per_class_compare_refined_vs_old{filename_suffix}.png',
     'Old model', '#E69F00',
     f'MAE per pain class — old vs refined{suffix}'),
  ):
    imp_groups = sorted(set(base_mae) | set(after_mae))
    diffs = [
      base_mae.get(g, (float('nan'), 0))[0] - after_mae.get(g, (float('nan'), 0))[0]
      for g in imp_groups
    ]
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_mae_improvement_bar(ax, imp_groups, diffs, 'Labels', title)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')

    base_vals  = [base_mae.get(g,  (float('nan'), 0))[0] for g in imp_groups]
    after_vals = [after_mae.get(g, (float('nan'), 0))[0] for g in imp_groups]
    fig, ax = plt.subplots(figsize=(14, 5))
    _draw_grouped_mae_bar(
      ax, imp_groups, base_vals, after_vals, base_label, 'Refined (after)',
      base_color, '#0072B2', 'MAE', cmp_title,
    )
    ax.set_xlabel('Labels')
    plt.tight_layout()
    path = os.path.join(out_dir, cmp_fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')

  # Combined 1×3 panel — all three source-split per-class improvements on a shared
  # x-axis. Panel 1 (old − before) is the comparison that has no standalone PNG.
  combined_groups = sorted(set(old_mae) | set(before_mae) | set(after_mae))
  panels = (
    (old_mae,    before_mae,
     'MAE improvement per pain class (old − projected before refinement)'),
    (before_mae, after_mae,
     'MAE improvement per pain class (projected before − after refinement)'),
    (old_mae,    after_mae,
     'MAE improvement per pain class (old − refined)'),
  )
  fig, axes = plt.subplots(1, 3, figsize=(28, 5))
  for ax, (base_mae, later_mae, title) in zip(axes, panels):
    diffs = [
      base_mae.get(g, (float('nan'), 0))[0] - later_mae.get(g, (float('nan'), 0))[0]
      for g in combined_groups
    ]
    _draw_mae_improvement_bar(ax, combined_groups, diffs, 'Labels', f'{title}{suffix}')
  plt.tight_layout()
  path = os.path.join(out_dir, f'mae_improvement_per_class_combined{filename_suffix}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def _model_pth_from_config(config_path, search_root, key, *, run_local=False):
  """
  Read a model checkpoint path from a cross_space_projection --config YAML.

  Legacy --config paths are tried as recorded and relative to the search root.
  Snapshot metadata is run-local, so relative paths are resolved only against the
  durable search root.

  Args:
    config_path (str): YAML path from best_config.txt.
    search_root (str): The grid-search root dir.
    key (str): YAML key to read (new_model_pth or old_model_pth).
    run_local (bool): Resolve a relative path strictly within search_root.

  Returns:
    str | None: The YAML checkpoint value, or None when the YAML/key is missing.
  """
  import yaml
  if not config_path:
    return None
  if run_local and config_path and not os.path.isabs(config_path):
    candidates = (os.path.join(search_root, config_path),)
  else:
    candidates = (config_path, os.path.join(search_root, config_path))
  for candidate in candidates:
    if candidate and os.path.isfile(candidate):
      try:
        with open(candidate) as f:
          ycfg = yaml.safe_load(f) or {}
        return ycfg.get(key)
      except Exception as exc:
        print(f'[cross_space_logs] failed to read {key} from {candidate}: {exc}')
        return None
  return None


def _new_model_pth_from_config(config_path, search_root):
  """Backward-compatible wrapper for reading new_model_pth from a launch YAML."""
  return _model_pth_from_config(config_path, search_root, 'new_model_pth')


def _model_pth_from_best_config(search_root, key):
  """Resolve a grid checkpoint via explicit CLI, snapshot, then legacy YAML."""
  cfg_txt = os.path.join(search_root, 'best_config.txt')
  if not os.path.isfile(cfg_txt):
    return None
  config_path = None
  snapshot_path = None
  explicit_flag = f'--{key}'
  with open(cfg_txt) as f:
    for line in f:
      if line.startswith('script_cmd:'):
        tokens = line.split()
        for i, tok in enumerate(tokens):
          if tok == explicit_flag and i + 1 < len(tokens):
            return tokens[i + 1]
          if tok == '--config' and i + 1 < len(tokens):
            config_path = tokens[i + 1]
      elif line.startswith('config_snapshot:'):
        snapshot_path = line.partition(':')[2].strip()
  if snapshot_path:
    snapshot_value = _model_pth_from_config(
      snapshot_path, search_root, key, run_local=True)
    if snapshot_value is not None:
      return snapshot_value
  if config_path:
    return _model_pth_from_config(config_path, search_root, key)
  return None


def _resolve_new_model_pth(data, fmt, pkl_path):
  """
  Best-effort lookup of the path to the new model's checkpoint.

  Standalone pkls store it under config_cross_space_projection. Grid trial
  pkls do not — for those we parse the search root's best_config.txt to
  recover --new_model_pth from the saved script_cmd line. Searches launched from
  a YAML (--config <yaml>) carry no --new_model_pth token, so the run-local
  config_snapshot is consulted before the referenced legacy YAML path.

  Args:
    data     (dict): Deserialized pkl contents.
    fmt      (str):  'grid' or 'standalone'.
    pkl_path (str):  Path to the loaded pkl (used to walk up to the search root).

  Returns:
    str | None: Path to the new model checkpoint, or None if it cannot be resolved.
  """
  if fmt == 'standalone':
    return data.get('config_cross_space_projection', {}).get('new_model_pth')

  search_root = os.path.dirname(os.path.dirname(pkl_path))
  return _model_pth_from_best_config(search_root, 'new_model_pth')


def _resolve_new_features_path(data, fmt, new_model_pth, pkl_path=None):
  """
  Find the new model's safetensors feature folder for the diagnostic's bulk
  head-only extraction.

  When the projection ran across datasets (old domain ≠ new model's native
  domain), the new model needs a features folder that matches the OLD
  domain — same selection logic as cross_space_projection.py:1237-1238.

  Args:
    data           (dict): Deserialized pkl contents.
    fmt            (str):  'grid' or 'standalone'.
    new_model_pth  (str | None): Path to new model checkpoint (used in grid
      fallback when new_model_config isn't in the pkl).
    pkl_path       (str | None): Path to the loaded trial pkl. In grid mode the
      search root (which holds best_config.txt) is derived from it as
      os.path.dirname(os.path.dirname(pkl_path)).

  Returns:
    str | None: Absolute path to the features folder for the new model on
      the OLD domain, or None if it cannot be resolved.
  """
  from cross_space_projection import (
    _detect_dataset, _get_features_path, _load_config,
  )

  old_tensors_csv_path = data.get('old_tensors_csv_path')
  if fmt == 'standalone' and old_tensors_csv_path is None:
    old_tensors_csv_path = data.get('config_cross_space_projection', {}).get('old_tensors_csv_path')
  if fmt == 'grid' and old_tensors_csv_path is None:
    # grid trials don't carry old_tensors_csv_path; reconstruct from precomputed/
    old_csv_split = data['trial_params']['old_model_csv']
    search_root   = data.get('_search_root')
    old_tensors_csv_path = old_tensors_csv_path or None  # filled by caller if available

  # Resolve the old-domain features path to know which dataset to match.
  if fmt == 'standalone':
    old_features_path = (data.get('old_model_config') or {}) \
      .get('model_advanced_params', {}).get('features_folder_saving_path')
    new_features_native = (data.get('new_model_config') or {}) \
      .get('model_advanced_params', {}).get('features_folder_saving_path')
  else:
    if new_model_pth is None:
      return None
    new_cfg = _load_config(new_model_pth)
    new_features_native = new_cfg['model_advanced_params']['features_folder_saving_path']
    old_model_pth = data.get('config_cross_space_projection', {}).get('old_model_pth')
    if old_model_pth is None:
      old_model_pth = _resolve_old_model_pth(data, fmt, pkl_path)
    if old_model_pth is None:
      return None
    old_cfg = _load_config(old_model_pth)
    old_features_path = old_cfg['model_advanced_params']['features_folder_saving_path']

  if new_features_native is None or old_features_path is None:
    return None

  try:
    old_dataset = _detect_dataset(old_features_path)
    return _get_features_path(new_features_native, old_dataset)
  except ValueError as exc:
    print(f'[emb_recon] could not resolve new-model features for old domain: {exc}')
    return None


def _resolve_old_model_pth(data, fmt, pkl_path):
  """
  Best-effort lookup of the path to the old/source model's checkpoint.

  Mirrors _resolve_new_model_pth for the old model: standalone pkls store it under
  config_cross_space_projection; grid trial pkls recover --old_model_pth from the
  search root's best_config.txt script_cmd, falling back first to the run-local
  config_snapshot and then to the referenced --config YAML's top-level
  old_model_pth key.

  Args:
    data     (dict): Deserialized pkl contents.
    fmt      (str):  'grid' or 'standalone'.
    pkl_path (str):  Path to the loaded pkl (grid: walked up to the search root).

  Returns:
    str | None: Path to the old model checkpoint, or None if it cannot be resolved.
  """
  if fmt == 'standalone':
    return (data.get('config_cross_space_projection') or {}).get('old_model_pth')
  if not pkl_path:
    return None
  search_root = os.path.dirname(os.path.dirname(pkl_path))
  return _model_pth_from_best_config(search_root, 'old_model_pth')


@lru_cache(maxsize=None)
def _features_folder_from_ckpt(model_pth):
  """
  Resolve a checkpoint's native features folder from its k_fold_results.pkl, cached.

  _load_config deserializes the full k_fold_results.pkl (which embeds torch
  tensors, ~seconds per load), and every trial of a sweep points at the same one
  or two checkpoints — so the result is memoized per checkpoint path for the
  lifetime of the process.

  Args:
    model_pth (str): Absolute path to the .pt/.pth model checkpoint.

  Returns:
    str: The config's model_advanced_params['features_folder_saving_path'].
  """
  from cross_space_projection import _load_config
  return _load_config(model_pth)['model_advanced_params']['features_folder_saving_path']


def _resolve_old_dataset(data, fmt, pkl_path):
  """
  Infer the old/source model's dataset name for use in plot titles.

  Tries, in order, the most reliable path sources and returns the first that
  _detect_dataset (cross_space_projection) maps to a known dataset keyword. Used
  to annotate plot titles with the source dataset; never raises so a failure to
  resolve simply drops the dataset name from the title.

  Args:
    data     (dict): Deserialized pkl contents.
    fmt      (str):  'grid' or 'standalone'.
    pkl_path (str):  Path the trial pkl was loaded from (grid: its grandparent is
      the search root holding best_config.txt).

  Returns:
    str | None: Uppercase dataset name ('UNBC' | 'BIOVID' | 'AGEDB' | 'CAER' |
      'MORPH') or None if no source path yields a known dataset keyword.
  """
  from cross_space_projection import _detect_dataset

  cfg        = data.get('config_cross_space_projection') or {}
  candidates = []

  # 1. Old model checkpoint path (its directory name encodes the dataset, e.g.
  #    UNBC_OPI_.../...) and its native features folder.
  old_model_pth = _resolve_old_model_pth(data, fmt, pkl_path)
  if old_model_pth:
    candidates.append(old_model_pth)
    try:
      candidates.append(_features_folder_from_ckpt(old_model_pth))
    except Exception:
      pass

  # 2. Standalone pkls also carry the old model's resolved config inline.
  if fmt == 'standalone':
    feat = (data.get('old_model_config') or {}) \
      .get('model_advanced_params', {}).get('features_folder_saving_path')
    if feat:
      candidates.append(feat)

  # 3. Fallback: the label-CSV paths often contain a dataset keyword (e.g. partA/...).
  for key in ('old_tensors_csv_path', 'anchors_csv_path'):
    val = data.get(key) or cfg.get(key)
    if val:
      candidates.append(val)

  for path in candidates:
    try:
      return _detect_dataset(path)
    except ValueError:
      continue
  return None


def _resolve_new_dataset(data, fmt, pkl_path):
  """
  Infer the new/target model's dataset name for use in plot titles.

  Mirror of _resolve_old_dataset but for the new model: its own test split (the
  '*newtest*' refinement plots) lives in the new model's native domain, so those
  titles must name this dataset rather than the source one. Tries, in order, the
  most reliable path sources and returns the first that _detect_dataset
  (cross_space_projection) maps to a known dataset keyword. Never raises so a
  failure to resolve simply drops the dataset name from the title.

  Args:
    data     (dict): Deserialized pkl contents.
    fmt      (str):  'grid' or 'standalone'.
    pkl_path (str):  Path the trial pkl was loaded from (grid: its grandparent is
      the search root holding best_config.txt).

  Returns:
    str | None: Uppercase dataset name ('UNBC' | 'BIOVID' | 'AGEDB' | 'CAER' |
      'MORPH') or None if no source path yields a known dataset keyword.
  """
  from cross_space_projection import _detect_dataset

  candidates = []

  # 1. New model checkpoint path (its directory name encodes the dataset) and its
  #    native features folder.
  new_model_pth = _resolve_new_model_pth(data, fmt, pkl_path)
  if new_model_pth:
    candidates.append(new_model_pth)
    try:
      candidates.append(_features_folder_from_ckpt(new_model_pth))
    except Exception:
      pass

  # 2. Standalone pkls also carry the new model's resolved config inline.
  if fmt == 'standalone':
    feat = (data.get('new_model_config') or {}) \
      .get('model_advanced_params', {}).get('features_folder_saving_path')
    if feat:
      candidates.append(feat)

  for path in candidates:
    try:
      return _detect_dataset(path)
    except ValueError:
      continue
  return None


def _fetch_real_embeddings_via_pipeline(data, fmt, pkl_path, target_ids,
                                        out_dir, split_name):
  """
  Bulk-extract real new-model embeddings via head-only inference and cache them.

  Builds the new model from its k_fold_results.pkl, writes a temporary
  augmentation-free CSV restricted to ``target_ids``, calls
  cross_space_projection._extract_embeddings once, and persists the result
  as ``logs/real_embeddings_<split>.safetensors`` for subsequent re-runs.

  Args:
    data        (dict): Deserialized pkl contents.
    fmt         (str):  'grid' or 'standalone'.
    pkl_path    (str):  Path to the loaded pkl (used in grid fallback paths).
    target_ids  (Iterable[int]): Sample ids whose real embeddings we need.
    out_dir     (str):  Directory used to write the safetensors cache and
      the temporary input CSV.
    split_name  (str):  Used to name the cache file (real_embeddings_<split>.safetensors).

  Returns:
    dict[int, np.ndarray]: {sample_id: real_vector_1d}. Empty on failure.
  """
  from safetensors.numpy import load_file as st_load, save_file as st_save

  target_set = set(int(s) for s in target_ids)
  if not target_set:
    return {}

  cache_path = os.path.join(out_dir, f'real_embeddings_{split_name}.safetensors')
  if os.path.isfile(cache_path):
    try:
      cached = st_load(cache_path)
      sids = cached.get('sample_ids')
      embs = cached.get('embeddings')
      if sids is not None and embs is not None:
        cached_map = {int(s): embs[i] for i, s in enumerate(sids)}
        if target_set.issubset(cached_map.keys()):
          print(f'[emb_recon] cache hit ({len(cached_map)} samples) → {cache_path}')
          return {sid: cached_map[sid] for sid in target_set}
        print(f'[emb_recon] cache present but missing '
              f'{len(target_set - cached_map.keys())} ids — re-extracting.')
    except Exception as exc:
      print(f'[emb_recon] cache read failed ({exc}) — re-extracting.')

  # We need to load the new model + the old-domain CSV that lists target ids.
  new_model_pth = _resolve_new_model_pth(data, fmt, pkl_path)
  if new_model_pth is None:
    print('[emb_recon] cannot resolve new_model_pth — head-only extraction skipped.')
    return {}

  # Locate the old-domain tensors CSV (has the rows we want, with sample metadata)
  old_tensors_csv_path = data.get('old_tensors_csv_path')
  if old_tensors_csv_path is None and fmt == 'standalone':
    old_tensors_csv_path = data.get('config_cross_space_projection', {}).get('old_tensors_csv_path')
  if old_tensors_csv_path is None and fmt == 'grid':
    search_root  = os.path.dirname(os.path.dirname(pkl_path))
    old_csv_split = data['trial_params']['old_model_csv']
    candidate = os.path.join(search_root, 'precomputed', f'old_tensors_{old_csv_split}.csv')
    if os.path.isfile(candidate):
      old_tensors_csv_path = candidate
  if old_tensors_csv_path is None or not os.path.isfile(old_tensors_csv_path):
    print(f'[emb_recon] old_tensors_csv_path missing ({old_tensors_csv_path!r}) — '
          'head-only extraction skipped.')
    return {}

  # Subset that CSV to target_ids and write to a temp CSV for the extractor.
  df_all = pd.read_csv(old_tensors_csv_path, sep='\t', dtype={'sample_name': str})
  df_sub = df_all[df_all['sample_id'].astype(int).isin(target_set)].copy()
  if df_sub.empty:
    print('[emb_recon] no rows in old_tensors_csv matched target_ids — skipping.')
    return {}
  temp_csv = os.path.join(out_dir, f'_emb_recon_input_{split_name}.csv')
  df_sub.to_csv(temp_csv, sep='\t', index=False)

  # Build new model and run head-only extraction with the right features folder.
  from cross_space_projection import _build_model, _extract_embeddings, _load_config
  new_config   = _load_config(new_model_pth)
  new_model    = _build_model(new_config)
  features_override = _resolve_new_features_path(data, fmt, new_model_pth, pkl_path=pkl_path)

  # Cross-domain guard: when the override is unresolved the extractor falls back
  # to the new model's NATIVE features folder. With a foreign (old-domain) CSV
  # those rows point at sample ids that do not exist there (e.g. AgeDB ids under
  # a MORPH folder), failing mid-DataLoader with a confusing "File missing".
  # Probe the first row against the native folder and skip cleanly if absent.
  if features_override is None:
    native = new_config['model_advanced_params']['features_folder_saving_path']
    first = df_sub.iloc[0]
    probe = os.path.join(native, str(first['subject_name']),
                         f"{first['sample_name']}.safetensors")
    if not os.path.exists(probe):
      print(f'[emb_recon] cannot resolve new-model features for the old domain; '
            f'native path {native!r} does not contain the requested samples '
            f'(probe missing: {probe}) — skipping head-only extraction.')
      return {}

  print(f'[emb_recon] extracting real embeddings for {len(df_sub)} samples '
        f'(features_override={features_override})')
  raw = _extract_embeddings(
    new_model, new_model_pth, temp_csv, new_config,
    features_path_override=features_override,
  )

  sids = np.asarray(raw['sample_ids'], dtype=np.int64)
  embs = np.asarray(raw['embeddings'], dtype=np.float32)
  try:
    st_save({'sample_ids': sids, 'embeddings': embs}, cache_path)
    print(f'[emb_recon] cached real embeddings → {cache_path}')
  except Exception as exc:
    print(f'[emb_recon] failed to save cache ({exc}) — continuing without cache.')

  return {int(s): embs[i] for i, s in enumerate(sids) if int(s) in target_set}


def _compute_reconstruction_metrics(real, projected):
  """
  Per-sample distance metrics between two aligned embedding matrices.

  Args:
    real      (np.ndarray): Shape (N, D), float — real new-model embeddings.
    projected (np.ndarray): Shape (N, D), float — projected/reconstructed embeddings.

  Returns:
    dict[str, np.ndarray]:
      'l1'       (N,) — sum of |real - projected| per row.
      'l2'       (N,) — Euclidean norm of (real - projected) per row.
      'cos_sim'  (N,) — row-wise cosine similarity, clipped to [-1, 1].
      'cos_dist' (N,) — 1 - cos_sim.
  """
  real      = np.asarray(real,      dtype=np.float32)
  projected = np.asarray(projected, dtype=np.float32)
  diff = real - projected
  l1   = np.sum(np.abs(diff), axis=1)
  l2   = np.linalg.norm(diff, axis=1)

  real_norm = np.linalg.norm(real,      axis=1)
  proj_norm = np.linalg.norm(projected, axis=1)
  denom     = np.clip(real_norm * proj_norm, 1e-12, None)
  cos_sim   = np.clip(np.sum(real * projected, axis=1) / denom, -1.0, 1.0)
  cos_dist  = 1.0 - cos_sim
  return {'l1': l1, 'l2': l2, 'cos_sim': cos_sim, 'cos_dist': cos_dist}


_EMB_RECON_METRICS = (
  ('l1',       'L1 distance'),
  ('l2',       'L2 distance'),
  ('cos_sim',  'Cosine similarity'),
  ('cos_dist', 'Cosine distance'),
)


def plot_embedding_reconstruction_histograms(metrics_df, out_dir,
                                             run_label='', split_name=''):
  """
  2×2 histogram+KDE figure for L1, L2, cos_sim and cos_dist.

  Each panel shows a 60-bin histogram, a KDE overlay (scaled to count units),
  vertical mean & median reference lines, and an N text annotation.

  Args:
    metrics_df  (pd.DataFrame): One row per sample with at least the four
      columns 'l1', 'l2', 'cos_sim', 'cos_dist'.
    out_dir     (str): Output directory.
    run_label   (str): Run identity string appended to the suptitle.
    split_name  (str): Split key (test/val/train/all/...); used in the
      output filename.
  """
  suffix = f' | {run_label}' if run_label else ''
  fig, axes = plt.subplots(2, 2, figsize=(14, 9))
  for ax, (col, title) in zip(axes.flat, _EMB_RECON_METRICS):
    vals = metrics_df[col].to_numpy(dtype=np.float32)
    if vals.size == 0:
      ax.axis('off')
      ax.text(0.5, 0.5, f'No data for {col}', ha='center', va='center')
      continue
    bins = 60
    counts, edges = np.histogram(vals, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    width   = edges[1] - edges[0]
    ax.bar(centers, counts, width=width, color='#4C72B0',
           alpha=0.75, edgecolor='white', linewidth=0.4)

    if vals.size > 1 and float(vals.std()) > 0:
      try:
        kde    = stats.gaussian_kde(vals)
        x_fine = np.linspace(float(vals.min()), float(vals.max()), 500)
        ax.plot(x_fine, kde(x_fine) * vals.size * width,
                color='#C44E52', linewidth=1.6, label='KDE')
      except Exception:
        pass

    mean_v   = float(vals.mean())
    median_v = float(np.median(vals))
    ax.axvline(mean_v,   color='#2ca02c', linestyle='--', linewidth=1.2,
               label=f'mean = {mean_v:.4f}')
    ax.axvline(median_v, color='#d62728', linestyle=':',  linewidth=1.2,
               label=f'median = {median_v:.4f}')

    ax.set_title(f'{title}  (N={vals.size})', fontsize=10, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('count')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='best')

  fig.suptitle(f'Embedding reconstruction distances — split={split_name}{suffix}',
               fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.96))
  path = os.path.join(out_dir, f'emb_recon_histograms_{split_name}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_embedding_reconstruction_per_class(metrics_df, out_dir,
                                            run_label='', split_name=''):
  """
  2×2 per-class box-plot figure for L1, L2, cos_sim and cos_dist.

  Samples are grouped by their rounded ground-truth label. Reuses
  `_draw_mae_box` for the box rendering so styling matches the existing
  per-class diagnostics.

  Args:
    metrics_df  (pd.DataFrame): One row per sample with columns 'label',
      'l1', 'l2', 'cos_sim', 'cos_dist'.
    out_dir     (str): Output directory.
    run_label   (str): Run identity string appended to the suptitle.
    split_name  (str): Split key (test/val/train/all/...); used in the
      output filename.
  """
  suffix = f' | {run_label}' if run_label else ''
  labels_int = np.round(metrics_df['label'].to_numpy(dtype=np.float32)).astype(int)
  classes    = sorted(int(c) for c in np.unique(labels_int))

  fig, axes = plt.subplots(2, 2, figsize=(16, 10))
  for ax, (col, title) in zip(axes.flat, _EMB_RECON_METRICS):
    vals = metrics_df[col].to_numpy(dtype=np.float32)
    raw_by_class = [vals[labels_int == c] for c in classes]
    mean_v = float(vals.mean()) if vals.size else float('nan')
    _draw_mae_box(
      ax, classes, raw_by_class, col, '#4C72B0',
      title=f'{title} per pain class  (mean={mean_v:.4f})',
    )
    ax.set_xlabel('Labels')

  fig.suptitle(f'Embedding reconstruction per class — split={split_name}{suffix}',
               fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.96))
  path = os.path.join(out_dir, f'emb_recon_per_class_box_{split_name}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_embedding_norm_cosine_per_class(new_emb, old_emb, labels, out_dir,
                                         run_label: str = '',
                                         new_stage_name: str = 'Projected (before refinement)',
                                         out_filename: str = 'embedding_norm_cosine_per_class_projected_vs_old.png'):
  """
  2×2 box-plot figure showing L2 norm and pairwise cosine similarity per class
  for both the new-model and old-model embeddings.

  Layout:
    Row 0, Col 0: L2 norm per class — new-model embeddings
    Row 0, Col 1: L2 norm per class — old embeddings
    Row 1, Col 0: Pairwise cosine similarity per class — new-model embeddings
    Row 1, Col 1: Pairwise cosine similarity per class — old embeddings

  Cosine similarity is computed over all unique intra-class pairs, giving a
  sense of how cohesive each class is in the embedding space.

  Args:
    new_emb        (np.ndarray): Shape (N, D), float32 new-model embeddings (the
      before-refinement projection by default, or refined when overridden).
    old_emb        (np.ndarray): Shape (N, D), float32 old model embeddings.
    labels         (np.ndarray): Shape (N,), float ground-truth class labels.
    out_dir        (str): Output directory.
    run_label      (str): Optional run identity string appended to the suptitle.
    new_stage_name (str): Stage label for the new-model column titles / suptitle.
    out_filename   (str): Basename of the saved PNG.
  """
  suffix     = f' | {run_label}' if run_label else ''
  labels_int = np.round(labels).astype(int)
  classes    = sorted(int(c) for c in np.unique(labels_int))
  x          = np.arange(len(classes))

  # --- collect per-class arrays for each metric + model ---
  def _l2_per_class(emb):
    norms = np.linalg.norm(emb, axis=1)
    return {cls: norms[labels_int == cls] for cls in classes}

  l2_new  = _l2_per_class(new_emb)
  l2_old  = _l2_per_class(old_emb)
  cos_new = _pairwise_cosine_sim_per_class(new_emb, labels_int)
  cos_old = _pairwise_cosine_sim_per_class(old_emb, labels_int)

  fig, axes = plt.subplots(2, 2, figsize=(16, 10))
  fig.suptitle(f'Embedding L2 norm & cosine similarity per class — new={new_stage_name}{suffix}',
               fontsize=13, fontweight='bold')

  specs = [
    (axes[0, 0], l2_new,  'L2 norm', new_stage_name,  '#4C72B0'),
    (axes[0, 1], l2_old,  'L2 norm', 'Old model',       '#DD8452'),
    (axes[1, 0], cos_new, 'Cosine similarity (intra-class)', new_stage_name, '#4C72B0'),
    (axes[1, 1], cos_old, 'Cosine similarity (intra-class)', 'Old model',  '#DD8452'),
  ]

  for ax, per_class_dict, ylabel, model_label, color in specs:
    box_data = []
    for cls in classes:
      arr = np.asarray(per_class_dict.get(cls, [np.nan]), dtype=np.float64)
      box_data.append(arr if arr.size > 0 else np.array([np.nan]))
    face_rgba      = list(mcolors.to_rgba(color))
    face_rgba[3]   = 0.5
    bp = ax.boxplot(
      box_data, positions=x, widths=0.5, patch_artist=True,
      showfliers=True,
      flierprops=dict(marker='o', markersize=2, alpha=0.4,
                      markerfacecolor=color, markeredgecolor='none'),
      medianprops=dict(color='#222222', linewidth=1.5),
      whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0),
    )
    for patch in bp['boxes']:
      patch.set_facecolor(face_rgba)
      patch.set_edgecolor(color)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Class', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{ylabel} — {model_label}', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

  fig.tight_layout()
  path = os.path.join(out_dir, out_filename)
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'Saved: {path}')


def log_embedding_reconstruction(data, fmt, pkl_path, out_dir, run_label='',
                                 projected_override=None, stage_tag=''):
  """
  Compare projected (reconstructed) vs real new-model embeddings.

  Uses linear_projector splits when available as a fast path (no model load)
  and falls back to head-only bulk extraction via the new model's safetensors
  cache for the remaining sample ids. Writes a per-sample CSV plus two
  large-N-friendly plots (histograms+KDE and per-class box plots).

  Args:
    data       (dict): Deserialized pkl contents.
    fmt        (str):  'grid' or 'standalone'.
    pkl_path   (str):  Path to the loaded pkl (used to walk to search-root
      artefacts in grid mode).
    out_dir    (str):  Directory in which to write the CSV and PNGs.
    run_label  (str):  Run identity string appended to plot titles.
    projected_override (np.ndarray | None): Projected embeddings to reconstruct,
      row-aligned with new_model_tensors['sample_ids']. When None the stored
      new_model_tensors['embeddings'] are used (which, for refinement runs, are the
      after-refinement projection). Pass the before-refinement projection here to
      reconstruct that stage instead.
    stage_tag  (str):  Stage suffix woven into the CSV/PNG names and titles (e.g.
      '_projected', '_refined') so before/after outputs do not overwrite each other.
  """
  scope_ids, split_name = _resolve_scope_sample_ids(data, fmt)
  if scope_ids.size == 0:
    print('[emb_recon] new_model_tensors has no sample_ids — skipping.')
    return

  new_t      = data['new_model_tensors']
  projected  = np.asarray(
    new_t['embeddings'] if projected_override is None else projected_override,
    dtype=np.float32)
  sids_all   = np.asarray(new_t['sample_ids'],  dtype=np.int64)
  labels_all = np.asarray(new_t['labels'],      dtype=np.float32)
  proj_map   = {int(sid): projected[i] for i, sid in enumerate(sids_all)}
  label_map  = {int(sid): float(labels_all[i]) for i, sid in enumerate(sids_all)}

  linear_bundle = _extract_linear_bundle(data)
  real_from_linear = _fetch_real_embeddings_from_linear(linear_bundle, scope_ids)
  print(f'[emb_recon] linear-projector fast path covered '
        f'{len(real_from_linear)} / {len(scope_ids)} sample ids')

  missing_ids = [int(s) for s in scope_ids if int(s) not in real_from_linear]
  if missing_ids:
    real_from_pipeline = _fetch_real_embeddings_via_pipeline(
      data, fmt, pkl_path, missing_ids, out_dir, split_name,
    )
  else:
    real_from_pipeline = {}

  rows = []
  for sid in scope_ids:
    sid_i = int(sid)
    if sid_i in real_from_linear:
      real_vec = real_from_linear[sid_i]
      source   = 'linear_projector_stored'
    elif sid_i in real_from_pipeline:
      real_vec = real_from_pipeline[sid_i]
      source   = 'head_extracted'
    else:
      continue
    proj_vec = proj_map.get(sid_i)
    if proj_vec is None or proj_vec.shape != real_vec.shape:
      continue
    rows.append((sid_i, real_vec, proj_vec, source))

  if not rows:
    print('[emb_recon] no aligned (real, projected) pairs — skipping plots/CSV.')
    return

  sids_aligned = np.array([r[0] for r in rows], dtype=np.int64)
  real_mat     = np.stack([r[1] for r in rows], axis=0).astype(np.float32)
  proj_mat     = np.stack([r[2] for r in rows], axis=0).astype(np.float32)
  sources      = [r[3] for r in rows]

  metrics = _compute_reconstruction_metrics(real_mat, proj_mat)

  # Optional subject_id column when a subject map can be built.
  subject_map = {}
  csv_for_subjects = data.get('old_tensors_csv_path')
  if csv_for_subjects is None and fmt == 'standalone':
    csv_for_subjects = data.get('config_cross_space_projection', {}).get('old_tensors_csv_path')
  if csv_for_subjects is None and fmt == 'grid':
    search_root  = os.path.dirname(os.path.dirname(pkl_path))
    old_csv_split = data['trial_params']['old_model_csv']
    cand = os.path.join(search_root, 'precomputed', f'old_tensors_{old_csv_split}.csv')
    if os.path.isfile(cand):
      csv_for_subjects = cand
  if csv_for_subjects and os.path.isfile(csv_for_subjects):
    try:
      subject_map = _get_subject_map(csv_for_subjects)
    except Exception as exc:
      print(f'[emb_recon] subject map unavailable ({exc}).')

  df = pd.DataFrame({
    'sample_id':  sids_aligned,
    'subject_id': [subject_map.get(int(s), -1) for s in sids_aligned],
    'label':      [label_map.get(int(s), float('nan')) for s in sids_aligned],
    'split':      split_name,
    'l1':         metrics['l1'],
    'l2':         metrics['l2'],
    'cos_sim':    metrics['cos_sim'],
    'cos_dist':   metrics['cos_dist'],
    'source':     sources,
  })

  name_tag = f'{split_name}{stage_tag}'
  csv_path = os.path.join(out_dir, f'embedding_reconstruction_{name_tag}.csv')
  df.to_csv(csv_path, index=False)
  print(f'Saved: {csv_path}')

  src_counts = df['source'].value_counts().to_dict()
  print(f'[emb_recon] {name_tag} source counts: {src_counts}')
  print(f'[emb_recon] {name_tag} means — L1={df["l1"].mean():.4f}  L2={df["l2"].mean():.4f}  '
        f'cos_sim={df["cos_sim"].mean():.4f}  cos_dist={df["cos_dist"].mean():.4f}')

  plot_embedding_reconstruction_histograms(df, out_dir, run_label=run_label,
                                           split_name=name_tag)
  plot_embedding_reconstruction_per_class(df, out_dir, run_label=run_label,
                                          split_name=name_tag)


# ── search-folder aggregation ────────────────────────────────────────────────

def generate_logs_search(search_dir, plot_only_top_k=None, plot_trials=None, skip_umap=False):
  """
  Process all results.pkl files found recursively under a grid-search root folder.

  For each pkl: generates per-trial diagnostic plots in a logs/ sub-directory.
  After all trials: writes a summary.csv at the search root, sorted by MAE ascending.

  When plot_only_top_k is set, metrics are collected for all trials (summary.csv
  covers the full sweep) but diagnostic plots are generated only for the top K
  trials by MAE ascending.

  When plot_trials is set it takes precedence over plot_only_top_k and triggers a
  fast path: only the listed trials are loaded and plotted. The requested trial
  numbers are resolved to their pkls from the 'trialNNNN_' directory names (with a
  per-pkl fallback for directories that don't match that convention), so the rest
  of the sweep is never read. summary.csv and the search_summary/ plots are NOT
  written in this mode. A trial number that is not present under search_dir aborts
  with a ValueError.

  Args:
    search_dir      (str): Root folder of a cross_space_projection grid search.
    plot_only_top_k (int | None): If set, limit plot generation to the K best
      trials by MAE. If None, plots are generated for every trial.
    plot_trials     (list[int] | None): If set, load and plot only the trials whose
      trial_number appears in this list, overriding plot_only_top_k and skipping
      summary.csv / search_summary/. None leaves the top-K / plot-all behaviour
      unchanged.
    skip_umap       (bool): Forwarded to generate_logs for each trial to skip the
      slow UMAP plots. Default False.

  Returns:
    str: Path to search_dir.
  """
  pkl_paths = sorted(glob.glob(
    os.path.join(search_dir, '**', 'results.pkl'), recursive=True,
  ))
  if not pkl_paths:
    print(f'[cross_space_logs] No results.pkl found under {search_dir}')
    return search_dir

  print(f'[cross_space_logs] Found {len(pkl_paths)} trial(s) under {search_dir}')

  # Fast path: when explicit trials are requested, resolve them from the
  # 'trialNNNN_' directory names and load/plot only those, skipping the full-sweep
  # metric collection, summary.csv, and search_summary/ plots entirely.
  if plot_trials is not None:
    # Map trial_number -> pkl_path from directory names without opening any pkl.
    by_number = {}
    unparsed  = []
    for pkl_path in pkl_paths:
      n = _trial_number_from_pkl_path(pkl_path)
      if n is None:
        unparsed.append(pkl_path)
      else:
        by_number.setdefault(n, pkl_path)

    requested = list(dict.fromkeys(plot_trials))

    # Fallback: only for directories whose name didn't parse, open the pkl to read
    # its trial_number, stopping as soon as every requested trial is resolved.
    missing = [t for t in requested if t not in by_number]
    if missing and unparsed:
      for pkl_path in unparsed:
        if not missing:
          break
        try:
          data = _load_pkl(pkl_path)
        except Exception as exc:
          print(f'[WARN] Skipping {pkl_path}: {exc}')
          continue
        n = data.get('trial_number')
        if n is not None:
          by_number.setdefault(n, pkl_path)
          missing = [t for t in requested if t not in by_number]

    if missing:
      raise ValueError(
        f'--plot_trials: trial number(s) {missing} not found under {search_dir}. '
        f'Available: {sorted(by_number)}'
      )

    print(f'[cross_space_logs] Plotting {len(requested)} requested trial(s): {requested}')
    for t in tqdm(requested, desc='Plotting selected trials', unit='trial'):
      try:
        generate_logs(by_number[t], skip_umap=skip_umap)
      except Exception as exc:
        print(f'[WARN] Skipping {by_number[t]}: {exc}')

    return search_dir

  # Collect metrics for every trial first (without plotting) whenever plotting is
  # restricted to a subset, so summary.csv still covers the full sweep. (plot_trials
  # is handled by the fast path above, so it is always None here.)
  collect_all = plot_only_top_k is not None

  if not collect_all:
    rows = []
    for pkl_path in tqdm(pkl_paths, desc='Processing trials', unit='trial'):
      try:
        _, row = generate_logs(pkl_path, skip_umap=skip_umap)
        if row is not None:
          rows.append(row)
      except Exception as exc:
        print(f'[WARN] Skipping {pkl_path}: {exc}')
  else:
    # Phase 1: collect metrics for all trials without generating plots. The pkl
    # loads are NFS-I/O-bound, so a small thread pool overlaps the reads; rows are
    # consumed in submission order to keep the output identical to the sequential
    # loop.
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
      futures = [(p, pool.submit(_collect_row_task, p)) for p in pkl_paths]
      for pkl_path, fut in tqdm(futures, desc='Collecting metrics', unit='trial'):
        try:
          rows.append(fut.result())
        except Exception as exc:
          print(f'[WARN] Skipping {pkl_path}: {exc}')

  if rows:
    df = pd.DataFrame(rows).sort_values('mae').reset_index(drop=True)
    csv_path = os.path.join(search_dir, 'summary.csv')
    df.drop(columns=['_pkl_path'], errors='ignore').to_csv(csv_path, index=False)
    print(f'Saved summary: {csv_path}')

    generate_search_summary_plots(df, search_dir)

    if plot_only_top_k is not None:
      k = min(plot_only_top_k, len(df))
      print(f'[cross_space_logs] Plotting top {k} trial(s) by MAE...')
      for pkl_path in tqdm(df.head(k)['_pkl_path'], desc='Plotting top-K trials', unit='trial'):
        try:
          generate_logs(pkl_path, skip_umap=skip_umap)
        except Exception as exc:
          print(f'[WARN] Skipping {pkl_path}: {exc}')

  return search_dir


def generate_logs_subtrials(container_dir, plot_only_top_k=None):
  """
  Process a container folder of standalone model-pair subtrials, grid-style.

  cross_space_projection's model-combos mode (multiple --new_model_pth/--old_model_pth)
  writes one standalone run per (new, old) pair under
  <container>/cross_space_projection_subtrial_<i>_<j>_.../results_<uid>.pkl, plus a pooled
  <container>/aggregated_<uid>/results_<uid>.pkl. This emulates generate_logs_search for that
  layout: it writes a root summary.csv over the subtrials (sorted by MAE ascending) and
  generates per-subtrial diagnostic plots in each subtrial's logs/ (UMAPs forced off, the
  slow plots). The aggregated_* folder is excluded — it pools the subtrials and is left
  untouched. No search_summary/ plots are emitted (all subtrials share the same hypers).

  Args:
    container_dir   (str): Folder holding the subtrial_* run folders (and one aggregated_*).
    plot_only_top_k (int | None): If set, generate per-subtrial logs/ only for the K best
      subtrials by MAE ascending (summary.csv still covers all). If None, plots every subtrial.

  Returns:
    str | None: container_dir if any subtrial pkls were found and processed; None when the
      folder holds no standalone subtrials (so the caller can fall back to the grid path).
  """
  pkls = sorted(glob.glob(os.path.join(container_dir, '*', 'results_*.pkl')))
  sub_pkls = [p for p in pkls
              if not os.path.basename(os.path.dirname(p)).startswith('aggregated')]
  if not sub_pkls:
    return None

  print(f'[cross_space_logs] Found {len(sub_pkls)} subtrial(s) under {container_dir}')
  n_sub = len(sub_pkls)

  rows = []
  for idx, pkl_path in enumerate(sub_pkls):
    try:
      # One row per refinement mode (--refinement 3 → 2 rows/subtrial); all share _pkl_path.
      for row in _aggregated_summary_rows(_load_pkl(pkl_path), pkl_path, idx, n_sub):
        row['_pkl_path'] = pkl_path
        rows.append(row)
    except Exception as exc:
      print(f'[WARN] Skipping {pkl_path}: {exc}')

  if rows:
    df = pd.DataFrame(rows).sort_values('mae').reset_index(drop=True)
    csv_path = os.path.join(container_dir, 'summary.csv')
    df.drop(columns=['_pkl_path'], errors='ignore').to_csv(csv_path, index=False)
    print(f'Saved summary: {csv_path}')

    # One plot pass per subtrial pkl (dedupe the per-mode rows), best MAE first.
    unique_pkls = df['_pkl_path'].drop_duplicates()
    if plot_only_top_k is not None:
      k = min(plot_only_top_k, len(unique_pkls))
      print(f'[cross_space_logs] Plotting top {k} subtrial(s) by MAE...')
      to_plot = unique_pkls.head(k)
    else:
      to_plot = unique_pkls
    for pkl_path in tqdm(to_plot, desc='Plotting subtrials', unit='subtrial'):
      try:
        generate_logs(pkl_path, skip_umap=True)
      except Exception as exc:
        print(f'[WARN] Skipping {pkl_path}: {exc}')

  return container_dir


# ── subtrial-index entry point ───────────────────────────────────────────────

def _find_subtrial_pkls(root_dirs, subtrial_indices):
  """Find direct results pkls for exact ``i_j`` subtrial pairs below roots."""
  indices = list(dict.fromkeys(os.fspath(index) for index in subtrial_indices))
  if not indices or any(
      len(parts := index.split('_')) != 2 or not all(part.isdigit() for part in parts)
      for index in indices
  ):
    raise ValueError('--subtrial_idx values must use the DIGITS_DIGITS form (for example 2_3).')

  roots = [os.path.abspath(os.fspath(root)) for root in root_dirs]
  invalid = [root for root in roots if not os.path.isdir(root)]
  if invalid:
    raise ValueError(f'--subtrial_idx root is not a directory: {invalid[0]}')

  prefixes = {index: f'cross_space_projection_subtrial_{index}_' for index in indices}
  matches, found_indices = set(), set()
  for root in roots:
    for current_dir, dirnames, filenames in os.walk(root):
      name = os.path.basename(os.path.normpath(current_dir))
      index = next((key for key, prefix in prefixes.items() if name.startswith(prefix)), None)
      if index is None:
        continue
      found = [
        os.path.abspath(os.path.join(current_dir, filename))
        for filename in filenames
        if filename.startswith('results_') and filename.endswith('.pkl')
      ]
      if found:
        matches.update(found)
        found_indices.add(index)
      dirnames[:] = []

  if not matches:
    raise ValueError(
      f'No subtrial results pkls found for {indices} under: {", ".join(roots)}'
    )
  return sorted(matches), [index for index in indices if index not in found_indices]


def generate_logs_subtrial_indices(root_dirs, subtrial_indices, skip_umap=False,
                                   only_projector_plots=False):
  """Run the normal single-pkl logger for selected subtrial pairs below roots."""
  pkl_paths, missing = _find_subtrial_pkls(root_dirs, subtrial_indices)
  if missing:
    print(f'[WARN] --subtrial_idx pair(s) not found under any root: {missing}')
  print(f'[cross_space_logs] Found {len(pkl_paths)} matching subtrial pkl(s)')

  processed = []
  for pkl_path in tqdm(pkl_paths, desc='Plotting selected subtrials', unit='subtrial'):
    try:
      generate_logs(
        pkl_path,
        skip_umap=skip_umap,
        only_projector_plots=only_projector_plots,
      )
      processed.append(pkl_path)
    except Exception as exc:
      print(f'[WARN] Skipping subtrial pkl {pkl_path}: {exc}')

  if not processed:
    raise RuntimeError('No subtrial logs were generated successfully.')
  return processed


# ── aggregated-only entry point ──────────────────────────────────────────────

def _find_aggregated_pkls(root_dir):
  """Find every pkl located in or below an ``aggregated*`` directory.

  Matching is recursive and applies to directory basenames, including
  ``root_dir`` itself. Returned paths are absolute, sorted, and deduplicated.

  Args:
    root_dir (str | os.PathLike): Root directory to search.

  Returns:
    list[str]: Matching absolute pkl paths in deterministic order.

  Raises:
    ValueError: If root_dir is not a directory or contains no matching pkls.
  """
  root = os.path.abspath(os.fspath(root_dir))
  if not os.path.isdir(root):
    raise ValueError(f'--only_aggregated root is not a directory: {root_dir}')

  root_is_aggregated = os.path.basename(os.path.normpath(root)).startswith('aggregated')
  matches = set()
  for current_dir, _, filenames in os.walk(root):
    rel_dir = os.path.relpath(current_dir, root)
    rel_parts = [] if rel_dir == os.curdir else rel_dir.split(os.sep)
    inside_aggregated = root_is_aggregated or any(
      part.startswith('aggregated') for part in rel_parts
    )
    if not inside_aggregated:
      continue
    for filename in filenames:
      if filename.endswith('.pkl'):
        matches.add(os.path.abspath(os.path.join(current_dir, filename)))

  if not matches:
    raise ValueError(
      f'No .pkl files found under an aggregated* directory in: {root_dir}'
    )
  return sorted(matches)


def generate_logs_aggregated(root_dirs, skip_umap=False):
  """Generate normal logs only for aggregate pkls below one or more roots.

  Each matching pkl is processed through :func:`generate_logs`. The summary rows
  emitted by those runs are then combined into ``aggregated_summary.csv`` at the
  corresponding search root, with ``source_pkl`` identifying their source.

  Args:
    root_dirs (list[str | os.PathLike]): Roots searched independently.
    skip_umap (bool): Forwarded to each single-pkl logging call.

  Returns:
    list[str]: Input root paths after successful processing.

  Raises:
    ValueError: If any root is invalid or has no aggregate pkls.
    RuntimeError: If every matching pkl for a root fails to produce a summary.
  """
  roots = [os.fspath(root) for root in root_dirs]
  # Discover everything first so an invalid/empty root fails before any outputs
  # are generated for the other roots.
  pkls_by_root = [(root, _find_aggregated_pkls(root)) for root in roots]

  failed_roots = []
  for root, pkl_paths in pkls_by_root:
    root_abs = os.path.abspath(root)
    parent_counts = Counter(os.path.dirname(path) for path in pkl_paths)
    print(
      f'[cross_space_logs] Found {len(pkl_paths)} aggregated pkl(s) under {root}'
    )
    summary_frames = []
    for pkl_path in tqdm(pkl_paths, desc='Processing aggregated pkls', unit='pkl'):
      try:
        log_kwargs = {'skip_umap': skip_umap}
        parent = os.path.dirname(pkl_path)
        if parent_counts[parent] > 1:
          stem = os.path.splitext(os.path.basename(pkl_path))[0]
          log_kwargs['out_dir_override'] = os.path.join(parent, f'logs_{stem}')
        out_dir, _ = generate_logs(pkl_path, **log_kwargs)
        summary_path = os.path.join(out_dir, 'summary.csv')
        frame = pd.read_csv(summary_path)
        if frame.empty:
          raise ValueError(f'generated summary has no rows: {summary_path}')
        frame['source_pkl'] = os.path.relpath(pkl_path, root_abs)
        ordered = ['source_pkl'] + [c for c in frame.columns if c != 'source_pkl']
        summary_frames.append(frame[ordered])
      except Exception as exc:
        print(f'[WARN] Skipping aggregated pkl {pkl_path}: {exc}')

    if not summary_frames:
      failed_roots.append(root)
      continue

    combined = pd.concat(summary_frames, ignore_index=True)
    combined_path = os.path.join(root_abs, 'aggregated_summary.csv')
    combined.to_csv(combined_path, index=False)
    print(f'Saved aggregated summary: {combined_path}')

  if failed_roots:
    joined = ', '.join(failed_roots)
    raise RuntimeError(
      f'No aggregated summaries were generated successfully under: {joined}'
    )

  return roots


# ── multi-folder entry point ─────────────────────────────────────────────────

def generate_logs_multi(search_dirs, plot_only_top_k=None, top_k_scope='global'):
  """
  Process multiple grid-search root folders in a single pass.

  Runs per-folder analysis (per-trial diagnostic plots, per-folder summary.csv,
  per-folder search_summary/ plots) and then writes a cross-folder global summary
  into a global_summary/ subdirectory inside every input folder.

  When plot_only_top_k is set, top_k_scope controls how the limit is applied:
    'global'   — rank all trials from every folder together and plot only the K
                 trials with the lowest MAE overall (a folder may get 0 plots).
    'per_path' — rank trials within each input folder independently and plot up
                 to K per folder, so every folder receives its own best trials.
  In both scopes, summary CSVs and global plots still cover all trials.

  Args:
    search_dirs     (list[str]): Paths to grid-search root folders to process.
    plot_only_top_k (int | None): If set, generate per-trial plots only for the
      K best trials by MAE (scope set by top_k_scope). Summary CSVs and global
      plots still cover all trials. If None, plots every trial.
    top_k_scope     (str): 'global' (default) ranks across all folders combined;
      'per_path' applies the top-K limit within each folder. Ignored when
      plot_only_top_k is None.

  Returns:
    list[str]: List of search_dir paths processed.
  """
  # Phase 1: collect metrics from all folders
  all_rows = []
  for search_dir in search_dirs:
    pkl_paths = sorted(glob.glob(
      os.path.join(search_dir, '**', 'results.pkl'), recursive=True,
    ))
    if not pkl_paths:
      print(f'[cross_space_logs] No results.pkl found under {search_dir}')
      continue
    print(f'[cross_space_logs] Found {len(pkl_paths)} trial(s) under {search_dir}')
    for pkl_path in tqdm(pkl_paths, desc=f'Collecting {os.path.basename(search_dir)}', unit='trial'):
      try:
        data = _load_pkl(pkl_path)
        if _detect_format(data) != 'grid':
          print(f'[WARN] Skipping non-grid pkl: {pkl_path}')
          continue
        row = _collect_summary_row(data, pkl_path)
        row['_pkl_path']   = pkl_path
        row['_search_dir'] = search_dir
        all_rows.append(row)
      except Exception as exc:
        print(f'[WARN] Skipping {pkl_path}: {exc}')

  if not all_rows:
    print('[cross_space_logs] No grid-format trials found across all folders.')
    return search_dirs

  global_df = pd.DataFrame(all_rows).sort_values('mae').reset_index(drop=True)
  print(f'[cross_space_logs] Total trials collected: {len(global_df)}')

  # Phase 2: determine which trials to plot
  if plot_only_top_k is not None:
    if top_k_scope == 'per_path':
      # global_df is sorted by MAE ascending, so head(k) per folder keeps ranking.
      plot_paths = set()
      for search_dir in search_dirs:
        folder_df = global_df[global_df['_search_dir'] == search_dir]
        if folder_df.empty:
          continue
        k = min(plot_only_top_k, len(folder_df))
        plot_paths.update(folder_df.head(k)['_pkl_path'])
        print(f'[cross_space_logs] Plotting top {k} trial(s) by MAE for {search_dir}...')
    else:
      k = min(plot_only_top_k, len(global_df))
      plot_paths = set(global_df.head(k)['_pkl_path'])
      print(f'[cross_space_logs] Plotting top {k} trial(s) by MAE (global ranking)...')
  else:
    plot_paths = set(global_df['_pkl_path'])

  # Phase 3: per-trial diagnostic plots for selected trials
  for pkl_path in tqdm(sorted(plot_paths), desc='Plotting selected trials', unit='trial'):
    try:
      generate_logs(pkl_path)
    except Exception as exc:
      print(f'[WARN] Skipping plot for {pkl_path}: {exc}')

  # Phase 4: per-folder summaries
  for search_dir in search_dirs:
    folder_mask = global_df['_search_dir'] == search_dir
    folder_df   = global_df[folder_mask].copy()
    if folder_df.empty:
      continue
    folder_clean = folder_df.drop(columns=['_pkl_path', '_search_dir'])
    csv_path = os.path.join(search_dir, 'summary.csv')
    folder_clean.to_csv(csv_path, index=False)
    print(f'Saved per-folder summary: {csv_path}')
    generate_search_summary_plots(folder_clean, search_dir)

  # Phase 5: global summary into each folder
  clean_global_df = global_df.drop(columns=['_pkl_path', '_search_dir'])
  for search_dir in search_dirs:
    global_out_dir = os.path.join(search_dir, 'global_summary')
    os.makedirs(global_out_dir, exist_ok=True)
    csv_path = os.path.join(global_out_dir, 'global_summary.csv')
    clean_global_df.to_csv(csv_path, index=False)
    print(f'Saved global summary: {csv_path}')
    print(f'[cross_space_logs] Writing global summary plots to {global_out_dir}')
    _emit_summary_plots(clean_global_df, global_out_dir)

  return search_dirs


# ── entry point ──────────────────────────────────────────────────────────────

def _resolve_logs_out_dir(data, fmt, pkl_path, out_dir_override=None):
  """Resolve the logs directory for one pkl, honoring an explicit override."""
  if out_dir_override is not None:
    return os.path.abspath(os.fspath(out_dir_override))
  if fmt == 'grid':
    return os.path.join(os.path.dirname(pkl_path), 'logs')
  cfg = data['config_cross_space_projection']
  return os.path.join(cfg['out_dir'], 'logs')


def generate_logs(pkl_path, plot_only_top_k=None, only_projector_plots=False,
                  plot_trials=None, skip_umap=False, out_dir_override=None):
  """
  Load a cross_space_projection pkl and write all diagnostic plots.

  Accepts either a path to a single pkl file or a directory. A directory holding
  grid-search trials (**/results.pkl) is delegated to generate_logs_search; a
  directory holding standalone model-pair subtrials (subtrial_*/results_<uid>.pkl)
  is delegated to generate_logs_subtrials.

  For grid-search trial pkls (containing trial_params) the subject-map CSV is
  resolved from <search_root>/precomputed/old_tensors_<old_model_csv>.csv and
  the anchor-UMAP plot is skipped (anchor embeddings are absent in that format).

  Args:
    pkl_path            (str): Path to a results pkl, or a grid-search / subtrial-container dir.
    plot_only_top_k     (int | None): When pkl_path is a directory, limits plot
      generation to the top K trials/subtrials by MAE ascending. Ignored for single files.
    only_projector_plots (bool): If True, emit only the linear-projector
      training-diagnostic plots (projector_training_curves, train_val_gap,
      weight_analysis, norm_comparison) and skip everything else. Useful for
      regenerating these on existing pkls without redoing UMAPs / heavy work.
    plot_trials         (list[int] | None): When pkl_path is a directory, loads and
      plots only the trials whose trial_number is listed, overriding plot_only_top_k
      and skipping summary.csv / search_summary/ for speed. Ignored for single files.
    skip_umap           (bool): If True, skip all UMAP plots (the slow ones:
      umap_all, umap_split_impact, anchor_umap). Default False. Forced True by
      generate_logs_subtrials when processing a subtrial-container folder.
    out_dir_override    (str | os.PathLike | None): Explicit logs directory for
      a single pkl. Used by aggregated-only batches when sibling pkls would
      otherwise overwrite one another. None preserves the standard location.

  Returns:
    tuple[str, dict | None]:
      - Path to the logs directory (or search_dir when given a directory).
      - Summary row dict for grid-format pkls; None for standalone runs.
  """
  if os.path.isdir(pkl_path):
    has_grid = bool(glob.glob(os.path.join(pkl_path, '**', 'results.pkl'), recursive=True))
    if not has_grid:
      res = generate_logs_subtrials(pkl_path, plot_only_top_k=plot_only_top_k)
      if res is not None:
        return res, None
    return generate_logs_search(
      pkl_path, plot_only_top_k=plot_only_top_k, plot_trials=plot_trials, skip_umap=skip_umap,
    ), None

  data = _load_pkl(pkl_path)
  data = _rebase_standalone_paths(data, pkl_path)
  fmt  = _detect_format(data)
  fake_evaluations = data.get('fake_projection_evaluations') or {}
  fake_metadata = data.get('fake_projection_metadata') or {}
  is_fake_replay = bool(fake_evaluations)
  # Aggregated (multi-model subtrial) pkls pool per-sample predictions across models but
  # drop embeddings (different spaces can't be pooled): skip every embedding-based plot
  # (UMAP / split-impact / anchor-UMAP / norm-cosine / reconstruction) and instead emit a
  # per-subtrial summary CSV. Prediction-based plots run normally on the pooled preds.
  is_aggregated = bool(data.get('aggregated'))

  if only_projector_plots:
    out_dir = _resolve_logs_out_dir(data, fmt, pkl_path, out_dir_override)
    if fmt == 'grid':
      uid       = data.get('uid') or os.path.basename(os.path.dirname(os.path.dirname(pkl_path))).split('_')[-1]
      run_label = f"Trial #{data['trial_number']} | UID: {uid}"
    else:
      cfg       = data['config_cross_space_projection']
      run_label = f"UID: {cfg['uid']}"
    os.makedirs(out_dir, exist_ok=True)
    print(f'[cross_space_logs] (projector-only) Output: {out_dir}')
    src_csv = (data['trial_params']['old_model_csv'] if fmt == 'grid'
               else cfg.get('old_model_csv'))
    src_dataset = _resolve_old_dataset(data, fmt, pkl_path)
    src_tag = ' · '.join(str(p) for p in (src_dataset, src_csv) if p)
    if src_tag:
      run_label = f'{run_label}\n{src_tag}'
    plot_projector_diagnostics(_extract_linear_bundle(data), out_dir, run_label=run_label)
    return out_dir, None

  if fmt == 'grid':
    out_dir       = _resolve_logs_out_dir(data, fmt, pkl_path, out_dir_override)
    search_root   = os.path.dirname(os.path.dirname(pkl_path))
    uid           = data.get('uid') or os.path.basename(search_root).split('_')[-1]
    run_label     = f"Trial #{data['trial_number']} | UID: {uid}"
    num_anchors_val = data['trial_params']['num_anchors']
    old_model_csv = data['trial_params']['old_model_csv']
    csv_path      = os.path.join(search_root, 'precomputed', f'old_tensors_{old_model_csv}.csv')
    subject_map   = _get_subject_map(csv_path)
    mae           = data['metrics']['mae']
    ccc           = data['metrics']['ccc']
    summary_row   = _collect_summary_row(data, pkl_path)
  else:
    cfg             = data['config_cross_space_projection']
    out_dir         = _resolve_logs_out_dir(data, fmt, pkl_path, out_dir_override)
    run_label       = f"UID: {cfg['uid']}"
    num_anchors_val = cfg.get('num_anchors')
    subject_map     = _get_subject_map(cfg['old_tensors_csv_path'])
    summary_row     = {
      'uid':                      cfg.get('uid'),
      'num_anchors':              cfg.get('num_anchors'),
      'num_anchors_real':         _anchor_count_from_data(data),
      'anchor_selection_type':    cfg.get('anchor_selection_type'),
      'old_model_csv':            cfg.get('old_model_csv'),
      'interpolation_similarity': cfg.get('interpolation_similarity'),
      'mlp_activation':           cfg.get('mlp_activation'),
      'mlp_num_layers':           cfg.get('mlp_num_layers'),
      'weighting_method':         cfg.get('weighting_method'),
      'temperature':              cfg.get('temperature'),
      'rbf_sigma':                cfg.get('rbf_sigma'),
    }

  if num_anchors_val == 0:
    run_label += ' | K=0 (identity)'
  elif num_anchors_val == -1:
    run_label += ' | K=-1 (original_video)'
  if is_fake_replay:
    distribution = fake_metadata.get(
      'distribution', data.get('fake_projection_distribution', 'unknown'))
    fake_seed = fake_metadata.get('seed', data.get('fake_projection_seed'))
    run_label += f' | FAKE {distribution}' + (f' (seed {fake_seed})' if fake_seed is not None else '')
    summary_row.update({
      'fake_projection': True,
      'fake_projection_distribution': distribution,
      'fake_projection_seed': fake_seed,
    })

  os.makedirs(out_dir, exist_ok=True)
  print(f'[cross_space_logs] Output: {out_dir}')
  for mode, evaluation in fake_evaluations.items():
    suffix = f'_{mode}' if len(fake_evaluations) > 1 else ''
    plot_fake_vs_real_dashboard(evaluation, out_dir, mode=mode, filename_suffix=suffix)

  # Source-set CSV name, surfaced into the confusion-matrix titles so the dataset is explicit.
  src_csv = (data['trial_params']['old_model_csv'] if fmt == 'grid'
             else cfg.get('old_model_csv'))

  # Source dataset · split tag (e.g. 'BIOVID · test'), appended on its own title line to
  # every per-run plot EXCEPT the confusion matrices (whose titles already carry it). The
  # dataset name is best-effort; when unresolved the tag is the split alone.
  src_dataset = _resolve_old_dataset(data, fmt, pkl_path)
  src_tag     = ' · '.join(str(p) for p in (src_dataset, src_csv) if p)

  def _with_src(lbl):
    """Append the 'dataset · split' source tag on its own line below a run label."""
    return f'{lbl}\n{src_tag}' if src_tag else lbl

  # New/target model dataset name, used for the new-model-test plots ('*newtest*' +
  # the new-model confusion matrices) whose data lives in the new space, not the
  # source one. Best-effort; None ⇒ titles fall back to the generic 'new-model dataset'.
  new_dataset     = _resolve_new_dataset(data, fmt, pkl_path)
  new_dataset_lbl = new_dataset or 'new-model dataset'

  new_t      = data['new_model_tensors']
  old_t      = data['old_model_tensors']
  new_preds   = np.asarray(new_t['predictions'], dtype=np.float32).squeeze()
  old_preds   = np.asarray(old_t['predictions'], dtype=np.float32).squeeze()
  labels      = np.asarray(new_t['labels'],      dtype=np.float32)
  sample_ids  = np.asarray(new_t['sample_ids'],  dtype=np.int64)
  weights     = np.asarray(new_t['weights'],      dtype=np.float32)
  num_classes = int(np.round(labels).max()) + 1

  # Real anchors actually used vs the configured budget, surfaced in the dashboard metrics
  # table. A single run reports an int (from its aligned anchor embeddings); an aggregate
  # reports a {real_count: n_subtrials} frequency dict read from the subtrials' anchors.csv.
  config_anchors = num_anchors_val
  real_anchors   = (_real_anchor_freq_from_subtrials(data, pkl_path) if is_aggregated
                    else _anchor_count_from_data(data))

  if fmt == 'standalone':
    mae = float(np.mean(np.abs(new_preds - labels)))
    ccc = concordance_ccc(labels, new_preds)
    summary_row['mae'] = mae
    summary_row['ccc'] = ccc
    summary_row['runtime_min'] = data['metrics'].get('runtime_min')

  mae_micro_new, mae_macro_new = _compute_global_mae(new_preds, labels)
  mae_micro_old, mae_macro_old = _compute_global_mae(old_preds, labels)
  summary_row['mae_micro']     = mae_micro_new
  summary_row['mae_macro']     = mae_macro_new
  summary_row['mae_micro_old'] = mae_micro_old
  summary_row['mae_macro_old'] = mae_macro_old
  # Rounded+clamped siblings: same predictions/split, but post-processed like the training
  # test loop (head.py) so they reproduce the pipeline's `test_l1_error` for direct comparison.
  mae_micro_new_r, mae_macro_new_r = _compute_rounded_mae(new_preds, labels, num_classes)
  mae_micro_old_r, mae_macro_old_r = _compute_rounded_mae(old_preds, labels, num_classes)
  summary_row['mae_micro_rounded']     = mae_micro_new_r
  summary_row['mae_macro_rounded']     = mae_macro_new_r
  summary_row['mae_micro_old_rounded'] = mae_micro_old_r
  summary_row['mae_macro_old_rounded'] = mae_macro_old_r
  summary_row.update(_refinement_columns(data))
  summary_row.update(_recipe_columns(data))
  summary_row.update(_best_epoch_columns(data))
  print(f'Global metrics — MAE micro: {mae_micro_new:.4f}  MAE macro: {mae_macro_new:.4f}  CCC: {ccc:.4f}')
  print(f'Global metrics (rounded+clamped) — MAE micro: {mae_micro_new_r:.4f}  '
        f'MAE macro: {mae_macro_new_r:.4f}')

  # ── Resolve the refinement stage(s) up front ────────────────────────────────
  # --refinement 3 writes several blocks under 'refinements' (keyed by mode); 1/2 write a
  # single 'refinement' block. With report_after_refinement, a SINGLE-mode run overwrites
  # new_model_tensors with the AFTER-refinement preds/embeddings, so the true "projected"
  # (after projection, before refinement) stage must be recomputed from the refinement
  # checkpoints; multi-mode / no-refine leave the stored tensors as the pure projection.
  refine_items = _refine_items(data)                          # [(mode, block), ...]
  multi_refine = len(refine_items) > 1

  def _mode_sfx(mode):
    """Per-mode PNG suffix: '' for a single-mode run, '_<mode>' under --refinement 3."""
    return f'_{mode}' if multi_refine else ''

  def _ref_title(mode):
    """Per-mode UMAP / per-class title label appended to run_label."""
    return (f'{run_label} | {mode} (after refinement)' if multi_refine
            else f'{run_label} | after refinement')

  old_emb_src = (None if is_aggregated else
                 np.asarray(data.get('fake_source_embeddings'), dtype=np.float32)
                 if is_fake_replay else np.asarray(old_t['embeddings'], dtype=np.float32))

  # Recompute per-mode (before, after) predictions + after-refinement embeddings once;
  # reused by the projected/refined plots, UMAPs, dashboards and comparison below.
  refine_preds_by_mode = {}    # mode -> (before_preds, after_preds)
  refined_emb_by_mode  = {}    # mode -> after-refinement projected embeddings
  for _mode, _block in refine_items:
    try:
      if is_fake_replay:
        _evaluation = fake_evaluations[_mode]
        _rp = (
          np.asarray(_evaluation['fake_before_predictions'], dtype=np.float32).reshape(-1),
          np.asarray(_evaluation['fake_predictions'], dtype=np.float32).reshape(-1),
        )
      else:
        _rp = _refinement_predictions(data, old_emb_src, refine_block=_block)
      if _rp is not None:
        refine_preds_by_mode[_mode] = _rp
    except Exception as exc:
      print(f'[WARN] refinement predictions ({_mode or "refinement"}) failed: {exc}')
    try:
      _re = _refined_projected_embeddings(data, old_emb_src, refine_block=_block)
      if _re is not None:
        refined_emb_by_mode[_mode] = _re
    except Exception as exc:
      print(f'[WARN] refined embeddings ({_mode or "refinement"}) failed: {exc}')

  aggregate_modes = []
  aggregate_refine_preds = {}
  if is_aggregated:
    aggregate_modes = _aggregate_refinement_modes(data, pkl_path)
    for _mode in aggregate_modes:
      _rp = _aggregate_refinement_predictions(data, pkl_path, _mode)
      if _rp is not None:
        aggregate_refine_preds[_mode] = _rp

  # The "projected" stage = after projection, before refinement. For a single-mode run
  # this is recomputed (stored tensors are after-refinement); otherwise the stored
  # tensors already hold the pure projection.
  single_block = refine_items[0][1] if len(refine_items) == 1 else None
  proj_preds = new_preds
  proj_emb   = None if is_aggregated else np.asarray(new_t['embeddings'], dtype=np.float32)
  if refine_items:
    _rp = refine_preds_by_mode.get(refine_items[0][0])
    if _rp is not None:
      proj_preds = _rp[0]
  if aggregate_refine_preds:
    proj_preds = next(iter(aggregate_refine_preds.values()))[0]
  if single_block is not None:
    _be = _projected_before_refinement_embeddings(data, old_emb_src, refine_block=single_block)
    if _be is not None:
      proj_emb = _be

  mae_micro_proj, mae_macro_proj = _compute_global_mae(proj_preds, labels)
  ccc_proj = float(concordance_ccc(labels, proj_preds))
  # Dashboard "Projected" panels are before-refinement when a refinement stage exists.
  has_refinement = bool(refine_items or aggregate_modes)
  proj_stage_name = ('Projected (before refinement)' if has_refinement
                     else 'Projected (new model)')
  projected_dashboard_available = not (
    (refine_items and not refine_preds_by_mode)
    or (aggregate_modes and not aggregate_refine_preds)
  )

  # ── Projected (before-refinement) plots ─────────────────────────────────────
  plot_predictions_histogram(proj_preds, old_preds, labels, out_dir, run_label=_with_src(run_label))
  plot_mae_per_class(proj_preds, old_preds, labels, out_dir, run_label=_with_src(run_label))
  plot_mae_per_subject(proj_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label=_with_src(run_label))
  plot_mae_improvement_per_class(proj_preds, old_preds, labels, out_dir, run_label=_with_src(run_label))
  plot_mae_improvement_per_subject(proj_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label=_with_src(run_label))
  # ── Confusion matrices (old-model dataset / source set) ─────────────────────
  # Stage-explicit titles name the step, the dataset, and the split. Cases 1-2 here are
  # mode-independent (the old model itself and the before-refinement projection on the
  # full source set); the after-refinement matrix (case 3) and the new-model-dataset
  # matrices (cases 4-5) are emitted per refinement mode in the loop below.
  # Titles are two-line: line 1 = pipeline step, line 2 = dataset · split. The standalone
  # PNGs use the full 'Confusion matrix — ...' titles (cm_*); the combined figure reuses the
  # short stage/dataset/split panel titles (pt_*) since its suptitle already carries the run.
  src_set_lbl   = f'{src_dataset or "old-model dataset"} · source set [{src_csv}]'
  proj_step     = 'After projection (before refinement)' if refine_items else 'After projection (new model)'
  cm_old_title  = f'Confusion matrix — Old model (original)\n{src_set_lbl}'
  cm_proj_title = f'Confusion matrix — {proj_step}\n{src_set_lbl}'
  cm_refined_title = f'Confusion matrix — After refinement\n{src_set_lbl}'
  pt_old  = f'Old model (original)\n{src_set_lbl}'
  pt_proj = f'{proj_step}\n{src_set_lbl}'
  pt_ref  = f'After refinement\n{src_set_lbl}'
  if num_classes <= 15:
    plot_confusion_matrix_cross(old_preds, labels, out_dir, num_classes, run_label=run_label,
                                title=cm_old_title, out_filename='confusion_matrix_old.png')
    plot_confusion_matrix_cross(proj_preds, labels, out_dir, num_classes, run_label=run_label,
                                title=cm_proj_title)
  else:
    print(f'Skipped confusion matrix: num_classes={num_classes} > 15')

  # No-refinement runs get the combined comparison figure here (cases 3-5 blanked).
  if not refine_items:
    panels = [
      {'title': pt_old,  'preds': old_preds,  'labels': labels, 'num_classes': num_classes},
      {'title': pt_proj, 'preds': proj_preds, 'labels': labels, 'num_classes': num_classes},
      {'title': pt_ref,  'preds': None, 'labels': None, 'num_classes': None,
       'note': 'no refinement stage'},
      {'title': f'New model (original)\n{new_dataset_lbl}',
       'preds': None, 'labels': None, 'num_classes': None, 'note': 'no refinement stage'},
      {'title': f'New model (after refinement)\n{new_dataset_lbl}',
       'preds': None, 'labels': None, 'num_classes': None, 'note': 'no refinement stage'},
    ]
    plot_confusion_matrices_combined(panels, out_dir, run_label=run_label)
  # Embedding-space plots (UMAP + split-impact): pooled aggregates have no embeddings, skip.
  # skip_umap additionally suppresses these (the slow plots) regardless of format.
  split_data = None
  if not is_aggregated and not skip_umap:
    plot_umap(proj_emb, labels, sample_ids, subject_map, out_dir, run_label=_with_src(run_label),
              filename_suffix='_projected')
    plot_umap_space_comparison(
      old_emb_src, proj_emb, labels, sample_ids, old_t['sample_ids'], subject_map, out_dir,
      stage_title=('After projection (before refinement)' if has_refinement
                   else 'After projection (new-model space)'),
      filename_suffix='_projected', run_label=_with_src(run_label),
    )
    try:
      split_data = _load_split_embeddings(data, fmt, pkl_path, SPLIT_TO_COMPARE, out_dir)
      if split_data is not None:
        s_emb, s_lab = split_data
        plot_umap_split_impact(
          np.asarray(proj_emb, dtype=np.float32), labels,
          s_emb, s_lab, SPLIT_TO_COMPARE, out_dir, run_label=_with_src(run_label),
          filename_suffix='_projected',
          new_dataset=new_dataset, src_dataset=src_dataset,
        )
      else:
        print(f'[WARN] split-impact UMAP: could not load {SPLIT_TO_COMPARE!r} '
              f'embeddings — skipped.')
    except Exception as exc:
      print(f'[WARN] split-impact UMAP failed: {exc}')

  plot_prediction_scatter(proj_preds, old_preds, labels, out_dir, run_label=_with_src(run_label))
  plot_prediction_by_class_boxplot(proj_preds, old_preds, labels, out_dir, run_label=_with_src(run_label))
  if not is_aggregated:
    try:
      plot_embedding_norm_cosine_per_class(proj_emb, old_emb_src, labels, out_dir, run_label=_with_src(run_label))
    except Exception as exc:
      print(f'[WARN] Failed to plot embedding norm/cosine: {exc}')

  # ── Refinement (after-refinement) plots, one set per mode ───────────────────
  # UMAPs + per-class MAE + the refined-variant prediction/embedding plots. Single-mode
  # keeps the legacy unsuffixed filenames; --refinement 3 appends a '_<mode>' suffix.
  for _mode, _block in refine_items:
    rl   = _ref_title(_mode)
    rl_src = _with_src(rl)   # rl + source dataset·split line, for the non-CM plots
    msfx = _mode_sfx(_mode)
    refined_emb = refined_emb_by_mode.get(_mode)
    comparison_emb = _umap_comparison_embedding(
      _mode, proj_emb, refined_emb_by_mode)
    if not skip_umap:
      if comparison_emb is None:
        print(f'[WARN] UMAP space comparison ({_mode or "refinement"}): '
              'projected embeddings unavailable — skipped.')
      else:
        comparison_title = (
          'After linear_only refinement (head refined; projector unchanged)'
          if _mode == 'linear_only' else f'After {_mode} refinement'
        )
        try:
          plot_umap_space_comparison(
            old_emb_src, comparison_emb, labels, sample_ids, old_t['sample_ids'],
            subject_map, out_dir,
            stage_title=comparison_title,
            filename_suffix=f'_refined_{_mode}' if _mode else '_refined',
            run_label=rl_src,
          )
        except Exception as exc:
          print(f'[WARN] UMAP space comparison ({_mode or "refinement"}) failed: {exc}')
    if refined_emb is not None:
      if not skip_umap:
        try:
          plot_umap(refined_emb, labels, sample_ids, subject_map, out_dir,
                    run_label=rl_src, filename_suffix=f'_refined{msfx}')
          if split_data is not None:
            s_emb, s_lab = split_data
            plot_umap_split_impact(refined_emb, labels, s_emb, s_lab, SPLIT_TO_COMPARE,
                                   out_dir, run_label=rl_src, filename_suffix=f'_refined{msfx}',
                                   new_dataset=new_dataset, src_dataset=src_dataset)
          else:
            print(f'[WARN] after-refinement split-impact UMAP: {SPLIT_TO_COMPARE!r} split unavailable — skipped.')
        except Exception as exc:
          print(f'[WARN] after-refinement UMAP ({_mode or "refinement"}) failed: {exc}')
      try:
        plot_embedding_norm_cosine_per_class(
          refined_emb, old_emb_src, labels, out_dir, run_label=rl_src,
          new_stage_name='Refined (after refinement)',
          out_filename=f'embedding_norm_cosine_per_class_refined_vs_old{msfx}.png')
      except Exception as exc:
        print(f'[WARN] refined embedding norm/cosine ({_mode or "refinement"}) failed: {exc}')

    _rp = refine_preds_by_mode.get(_mode)
    if _rp is not None:
      before_preds, after_preds = _rp
      try:
        plot_refinement_mae_per_class(after_preds, before_preds, old_preds, labels, out_dir,
                                      run_label=rl_src, filename_suffix=msfx)
      except Exception as exc:
        print(f'[WARN] after-refinement MAE-per-class ({_mode or "refinement"}) failed: {exc}')
      # Refined-variant prediction-level plots mirroring the projected ones above.
      try:
        plot_predictions_histogram(
          after_preds, old_preds, labels, out_dir, run_label=rl_src,
          new_name='Refined (after refinement)',
          out_filename=f'predictions_histogram_refined_vs_old{msfx}.png')
        plot_prediction_scatter(
          after_preds, old_preds, labels, out_dir, run_label=rl_src,
          new_name='Refined (after refinement)',
          out_filename=f'prediction_scatter_refined_vs_old{msfx}.png')
        plot_prediction_by_class_boxplot(
          after_preds, old_preds, labels, out_dir, run_label=rl_src,
          new_name='Refined (after refinement)',
          out_filename=f'prediction_by_class_boxplot_refined_vs_old{msfx}.png')
        plot_mae_per_subject(
          after_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label=rl_src,
          new_name='Refined (after refinement)',
          new_filename=f'mae_per_subject_refined{msfx}.png', emit_old=False)
      except Exception as exc:
        print(f'[WARN] refined prediction-level plots ({_mode or "refinement"}) failed: {exc}')

    # ── Confusion matrices: after-refinement source (case 3) + new-model dataset
    #    original/refined (cases 4-5), plus a per-mode combined comparison figure that
    #    also carries the mode-independent old/projected source matrices built above.
    src_after = _rp[1] if _rp is not None else None
    if src_after is not None and num_classes <= 15:
      plot_confusion_matrix_cross(
        src_after, labels, out_dir, num_classes, run_label=rl,
        title=cm_refined_title, out_filename=f'confusion_matrix_refined{msfx}.png')

    nte = (_block or {}).get('new_test_eval')
    nt_labels = nt_preds_before = nt_preds_after = None
    nt_classes = None
    cm_nt_orig_title = f'Confusion matrix — New model (original)\n{new_dataset_lbl}'
    cm_nt_ref_title  = f'Confusion matrix — New model (after refinement)\n{new_dataset_lbl}'
    pt_nt_orig = f'New model (original)\n{new_dataset_lbl}'
    pt_nt_ref  = f'New model (after refinement)\n{new_dataset_lbl}'
    if nte:
      nt_labels       = np.asarray(nte['labels'],       dtype=np.float32).reshape(-1)
      nt_preds_before = np.asarray(nte['preds_before'], dtype=np.float32).reshape(-1)
      nt_preds_after  = np.asarray(nte['preds_after'],  dtype=np.float32).reshape(-1)
      nt_classes      = int(np.round(nt_labels).max()) + 1
      new_set_lbl     = f"{new_dataset_lbl} · {nte.get('split', 'test')} split"
      cm_nt_orig_title = f'Confusion matrix — New model (original)\n{new_set_lbl}'
      cm_nt_ref_title  = f'Confusion matrix — New model (after refinement)\n{new_set_lbl}'
      pt_nt_orig = f'New model (original)\n{new_set_lbl}'
      pt_nt_ref  = f'New model (after refinement)\n{new_set_lbl}'
      if nt_classes <= 15:
        # 'original' = the unrefined new model, so use the base run_label (not rl, which
        # carries the contradictory '| after refinement' tag used by the after-stage plots).
        plot_confusion_matrix_cross(
          nt_preds_before, nt_labels, out_dir, nt_classes, run_label=run_label,
          title=cm_nt_orig_title, out_filename=f'confusion_matrix_newtest_original{msfx}.png')
        plot_confusion_matrix_cross(
          nt_preds_after, nt_labels, out_dir, nt_classes, run_label=rl,
          title=cm_nt_ref_title, out_filename=f'confusion_matrix_newtest_refined{msfx}.png')
      else:
        print(f'Skipped new-model confusion matrices: num_classes={nt_classes} > 15')

    panels = [
      {'title': pt_old,     'preds': old_preds,  'labels': labels, 'num_classes': num_classes},
      {'title': pt_proj,    'preds': proj_preds, 'labels': labels, 'num_classes': num_classes},
      {'title': pt_ref,     'preds': src_after,  'labels': labels, 'num_classes': num_classes,
       'note': 'refinement predictions unavailable'},
      {'title': pt_nt_orig, 'preds': nt_preds_before, 'labels': nt_labels,
       'num_classes': nt_classes, 'note': 'no new_test_eval block'},
      {'title': pt_nt_ref,  'preds': nt_preds_after,  'labels': nt_labels,
       'num_classes': nt_classes, 'note': 'no new_test_eval block'},
    ]
    plot_confusion_matrices_combined(
      panels, out_dir, run_label=rl,
      out_filename=f'confusion_matrix_all_stages{msfx}.png')

  try:
    plot_anchor_weights(weights, out_dir, run_label=_with_src(run_label))
  except Exception as exc:
    print(f'[WARN] Failed to plot anchor weights: {exc}')
  try:
    plot_weight_rank_distribution(weights, out_dir, run_label=_with_src(run_label))
  except Exception as exc:
    print(f'[WARN] Failed to plot weight rank distribution: {exc}')
  # Per-stage MAE block for the dashboard metrics table: source Old/Projected/Refined plus
  # the new-domain preserve before/after. Reuses the cached per-mode refine_preds_by_mode +
  # the preserve scalars already in each refinement pkl block (no recomputation).
  base_stages = {
    'old':       (mae_micro_old, mae_macro_old),
    'projected': (mae_micro_proj, mae_macro_proj),
  }

  def _preserve_pair(block, which):
    """(micro, macro) preserve MAE for 'before'/'after' on the new model's TEST split, or None if both NaN."""
    pr = _newtest_preserve_mae(block, which)
    return pr if any(np.isfinite(v) for v in pr) else None

  def _stages_for(mode, block):
    """Per-stage MAE dict for one refinement mode (Old/Projected/Refined + preserve before/after)."""
    st = dict(base_stages)
    rp = refine_preds_by_mode.get(mode)
    if rp is not None:
      before_preds, after_preds = rp
      st['projected'] = _compute_global_mae(before_preds, labels)
      st['refined']   = _compute_global_mae(after_preds,  labels)
    pb, pa = _preserve_pair(block, 'before'), _preserve_pair(block, 'after')
    if pb is not None:
      st['preserve_before'] = pb
    if pa is not None:
      st['preserve_after'] = pa
    return st

  # dashboard.png stays the projected-before-refinement baseline. Each mode dashboard
  # uses that mode's after-refinement predictions for every active panel and metric.
  if projected_dashboard_available:
    plot_dashboard(
      proj_preds, old_preds, labels, num_classes, mae_micro_proj, ccc_proj, out_dir,
      run_label=_with_src(run_label), mae_macro=mae_macro_proj,
      mae_macro_old=mae_macro_old, mae_stages=base_stages,
      projected_stage_name=proj_stage_name, src_dataset=src_dataset,
      new_dataset=new_dataset, real_anchors=real_anchors,
      config_anchors=config_anchors)
  else:
    print('[WARN] projected-before-refinement predictions are unavailable; '
          'baseline dashboard skipped.')
  for _mode, _block in refine_items:
    _rp = refine_preds_by_mode.get(_mode)
    if _rp is None:
      print(f'[WARN] refinement dashboard ({_mode or "refinement"}) has incomplete '
            'prediction coverage — skipped.')
      continue
    _after = _rp[1]
    _micro, _macro = _compute_global_mae(_after, labels)
    _ccc = float(concordance_ccc(labels, _after))
    plot_dashboard(
      _after, old_preds, labels, num_classes, _micro, _ccc, out_dir,
      run_label=_with_src(f'{run_label} | {_mode}'), mae_macro=_macro,
      mae_macro_old=mae_macro_old, mae_stages=_stages_for(_mode, _block),
      filename_suffix=f'_{_mode}', projected_stage_name='Refined (after refinement)',
      src_dataset=src_dataset, new_dataset=new_dataset,
      real_anchors=real_anchors, config_anchors=config_anchors)
  if multi_refine:
    try:
      plot_refinement_modes_comparison(refine_items, refine_preds_by_mode, old_preds, labels,
                                       base_stages, out_dir, run_label=_with_src(run_label),
                                       src_dataset=src_dataset, new_dataset=new_dataset)
    except Exception as exc:
      print(f'[WARN] refinement modes comparison failed: {exc}')

  plot_projector_diagnostics(_extract_linear_bundle(data), out_dir, run_label=_with_src(run_label))

  for _mode, _block in refine_items:
    base_rl = f'{run_label} | {_mode}' if multi_refine else run_label
    # The '*newtest*' per-class plot is computed on the new model's own test split,
    # so it carries the new-model 'dataset · split' tag instead of the source one.
    nt_split    = ((_block or {}).get('new_test_eval') or {}).get('split', 'test')
    newtest_tag = f'{new_dataset_lbl} · {nt_split} split'
    plot_refinement_diagnostics(
      _block, out_dir,
      run_label=_with_src(base_rl),
      filename_suffix=_mode_sfx(_mode),
      newtest_run_label=f'{base_rl}\n{newtest_tag}',
      src_dataset=src_dataset, new_dataset=new_dataset,
    )

  # Embedding reconstruction (projected vs real new-model embeddings). The projected
  # stage is the before-refinement projection; refinement runs additionally get one
  # after-refinement variant per mode. Stage tags keep the CSV/PNG names distinct.
  # Skipped for aggregates (no pooled embeddings).
  if not is_aggregated:
    try:
      log_embedding_reconstruction(data, fmt, pkl_path, out_dir, run_label=_with_src(run_label),
                                   projected_override=proj_emb,
                                   stage_tag=('_projected' if refine_items else ''))
    except Exception as exc:
      print(f'[WARN] embedding reconstruction diagnostic failed: {exc}')
  for _mode, _block in refine_items:
    refined_emb = refined_emb_by_mode.get(_mode)
    if refined_emb is None:
      continue
    try:
      log_embedding_reconstruction(data, fmt, pkl_path, out_dir, run_label=_with_src(_ref_title(_mode)),
                                   projected_override=refined_emb,
                                   stage_tag=f'_refined{_mode_sfx(_mode)}')
    except Exception as exc:
      print(f'[WARN] refined embedding reconstruction ({_mode or "refinement"}) failed: {exc}')

  if fmt == 'standalone' and not skip_umap and data.get('old_model_anchors_embeddings') is not None:
    plot_anchor_umap(
      data['old_model_anchors_embeddings']['embeddings'],
      data['new_model_anchors_embeddings']['embeddings'],
      data['old_model_anchors_embeddings']['labels'],
      out_dir,
      run_label=_with_src(run_label),
    )
    try:
      plot_anchor_norm_comparison(
        data['old_model_anchors_embeddings']['embeddings'],
        data['new_model_anchors_embeddings']['embeddings'],
        data['old_model_anchors_embeddings']['labels'],
        out_dir,
        run_label=_with_src(run_label),
      )
    except Exception as exc:
      print(f'[WARN] Failed to plot anchor norm comparison: {exc}')

  if is_aggregated:
    # The pooled aggregated pkl stores only predictions/labels (no per-mode refinement metrics —
    # see _aggregate_model_combo_pkls), so its lone AGGREGATE summary row and the dashboard's
    # stage table can only reach the old + projected stages. Recover the refined (and preserve)
    # stages by rolling the fully-populated per-subtrial rows up per refinement mode: each
    # subtrial pkl is loaded from its own results pkl (paths stored relative to this pkl's dir).
    n_sub = data.get('n_subtrials')
    base = os.path.dirname(os.path.abspath(pkl_path))
    sub_rows = []
    for idx, rel in enumerate(data.get('subtrial_pkls') or []):
      sub_path = os.path.join(base, rel)
      try:
        # One row per refinement mode (--refinement 3 → 2 rows/subtrial).
        sub_rows.extend(_aggregated_summary_rows(_load_pkl(sub_path), sub_path, idx, n_sub))
      except Exception as exc:
        print(f'[WARN] subtrial {idx} summary failed ({sub_path}): {exc}')
    if sub_rows:
      sub_csv = os.path.join(out_dir, 'summary_per_subtrial.csv')
      pd.DataFrame(sub_rows).to_csv(sub_csv, index=False)
      print(f'Saved: {sub_csv}  ({len(sub_rows)} rows)')
      # summary.csv: per-mode MEAN + STD rows (cross-validation aggregate) with every
      # srctest_* / newtest_* / refine_* column filled, replacing the single empty AGGREGATE row.
      _write_summary_rows(_aggregate_subtrial_rows(sub_rows, n_sub), out_dir)
      # Per-mode dashboards carry cross-validation mean ± sample std in the stage table,
      # while every active panel uses the aligned pooled after-refinement predictions.
      # dashboard.png remains the pooled before-refinement baseline.
      stds_by_mode = _per_mode_stage_stds(sub_rows)
      for _mode, _stages in _per_mode_stage_means(sub_rows).items():
        if 'refined' not in _stages:
          continue
        _rp = aggregate_refine_preds.get(_mode)
        if _rp is None:
          print(f'[WARN] aggregate refinement dashboard ({_mode or "refined"}) has '
                'incomplete prediction coverage — skipped.')
          continue
        _after = _rp[1]
        _micro, _macro = _compute_global_mae(_after, labels)
        _ccc = float(concordance_ccc(labels, _after))
        _sfx = f'_{_mode}' if _mode else '_refined'
        _lbl = f'{run_label} | {_mode or "refined"} (aggregate mean ± std of {n_sub} subtrials)'
        plot_dashboard(_after, old_preds, labels, num_classes, _micro, _ccc,
                       out_dir, run_label=_with_src(_lbl), mae_macro=_macro,
                       mae_macro_old=mae_macro_old, mae_stages=_stages,
                       mae_stages_std=stds_by_mode.get(_mode), filename_suffix=_sfx,
                       projected_stage_name='Refined (after refinement)',
                       src_dataset=src_dataset, new_dataset=new_dataset,
                       real_anchors=real_anchors, config_anchors=config_anchors)
    else:
      # No subtrial rows resolvable → fall back to the single (empty-refinement) AGGREGATE row.
      _write_summary_csv(_aggregated_summary_row(data, pkl_path, 'AGGREGATE', n_sub), out_dir)
  elif multi_refine:
    # --refinement 3: one summary row per mode, refinement columns re-sourced per block
    # (the base summary_row's singular-key refinement columns are empty for this schema).
    out_rows = []
    for _mode, _block in refine_items:
      _r = dict(summary_row)
      _r['refine_mode'] = _mode or None
      _r.update(_refinement_columns(data, refine_block=_block))
      _r.update(_recipe_columns(data, refine_block=_block))
      _r.update(_best_epoch_columns(data, refine_block=_block))
      out_rows.append(_r)
    _write_summary_rows(out_rows, out_dir)
  else:
    _write_summary_csv(summary_row, out_dir)
  print(f'Done. All logs in {out_dir}')
  return out_dir, summary_row


def main(argv=None):
  """Parse command-line arguments and dispatch the requested logging mode."""
  parser = argparse.ArgumentParser(
    description='Generate diagnostic plots from a cross_space_projection pkl file or grid-search folder.'
  )
  parser.add_argument(
    '--pkl_path', type=str, nargs='+', required=True,
    help=(
      'One or more paths to a results pkl produced by cross_space_projection.py, '
      'OR grid-search root folder(s), OR a subtrial-container folder (holding '
      'cross_space_projection_subtrial_*/results_<uid>.pkl runs from a multi-model '
      'combos run). A container folder produces per-subtrial logs/ + a root summary.csv '
      '(UMAPs auto-skipped) and leaves the aggregated_* folder untouched. '
      'Use --only_aggregated to recursively process only pkls below aggregated* folders. '
      'Use --subtrial_idx i_j [i_j ...] to recursively process exact model-pair '
      'subtrials below the supplied folder roots. '
      'Single path: behaviour is unchanged — pkl or folder processed as before. '
      'Multiple paths: per-folder analysis is run for each folder and a merged '
      'global_summary/ (global_summary.csv + hyperparameter plots) is written '
      'into every input folder.'
    ),
  )
  parser.add_argument(
    '--plot_only_top_k', type=int, default=5,
    help=(
      'When pkl_path is a folder, generate diagnostic plots only for the '
      'top K trials ranked by MAE ascending. Metrics are still collected '
      'for all trials and summary.csv covers the full sweep. '
      'Ignored when pkl_path is a single pkl file or --subtrial_idx is used.'
    ),
  )
  parser.add_argument(
    '--top_k_scope', type=str, choices=['global', 'per_path'], default='global',
    help=(
      'Scope of --plot_only_top_k when multiple pkl_path folders are given. '
      "'global' (default) ranks all trials across folders together and plots "
      "the K best overall; 'per_path' plots up to K best trials within each "
      'folder. Ignored for a single pkl_path.'
    ),
  )
  parser.add_argument(
    '--only_projector_plots', action='store_true', default=False,
    help=(
      'Emit only the linear-projector training-diagnostic plots and skip '
      'everything else (UMAPs, confusion matrix, dashboard, etc.). Applies '
      'to a single pkl file or every pkl selected by --subtrial_idx.'
    ),
  )
  parser.add_argument(
    '--skip_umap', action='store_true', default=False,
    help=(
      'Skip all UMAP plots (umap_all, umap_split_impact, anchor_umap) — the slow '
      'ones. Default off. Forced on when --pkl_path is a subtrial-container folder '
      '(per-subtrial logs/ are always generated without UMAPs).'
    ),
  )
  parser.add_argument(
    '--plot_trials', type=int, nargs='+', default=None,
    help=(
      'Explicit list of trial numbers (the trial_number field / "Trial #N") to '
      'generate per-trial diagnostic plots for. Overrides --plot_only_top_k and '
      'takes a fast path: only the listed trials are loaded and plotted, so the '
      'rest of the sweep is never read and summary.csv / search_summary/ are NOT '
      '(re)written. Only valid when a SINGLE grid-search folder is given as '
      '--pkl_path (errors for a single pkl file, ignored with a warning when '
      'multiple folders are passed). Aborts if any listed trial number is not '
      'found in the folder.'
    ),
  )
  parser.add_argument(
    '--subtrial_idx', type=str, nargs='+', default=None,
    help=(
      'One or more exact model-pair indices in i_j form (for example 2_3 4_1). '
      'Treat every --pkl_path value as a root folder, recursively find matching '
      'cross_space_projection_subtrial_<i>_<j>_*/results_<uid>.pkl files, and '
      'run the normal single-pkl logging workflow for each. Honors --skip_umap '
      'and --only_projector_plots. Incompatible with --plot_trials and '
      '--only_aggregated; top-K options are ignored.'
    ),
  )
  parser.add_argument(
    '--only_aggregated', action='store_true', default=False,
    help=(
      'Treat every --pkl_path value as a root folder, recursively find all .pkl '
      'files contained in aggregated* directories, and run the usual single-pkl '
      'logging workflow for each. Writes aggregated_summary.csv at each root. '
      'Incompatible with --plot_trials and --only_projector_plots; top-K options '
      'are ignored.'
    ),
  )
  args = parser.parse_args(argv)

  if args.subtrial_idx is not None:
    if args.plot_trials is not None:
      parser.error('--plot_trials cannot be used with --subtrial_idx.')
    if args.only_aggregated:
      parser.error('--only_aggregated cannot be used with --subtrial_idx.')
    try:
      generate_logs_subtrial_indices(
        args.pkl_path,
        args.subtrial_idx,
        skip_umap=args.skip_umap,
        only_projector_plots=args.only_projector_plots,
      )
    except (ValueError, RuntimeError) as exc:
      parser.error(str(exc))
    return

  if args.only_aggregated:
    if args.plot_trials is not None:
      parser.error('--plot_trials cannot be used with --only_aggregated.')
    if args.only_projector_plots:
      parser.error('--only_projector_plots cannot be used with --only_aggregated.')
    try:
      generate_logs_aggregated(args.pkl_path, skip_umap=args.skip_umap)
    except (ValueError, RuntimeError) as exc:
      parser.error(str(exc))
    return

  if len(args.pkl_path) > 1:
    if args.plot_trials is not None:
      print('[WARN] --plot_trials is ignored when multiple --pkl_path folders are given.')
    if args.only_projector_plots:
      for p in args.pkl_path:
        generate_logs(p, only_projector_plots=True, skip_umap=args.skip_umap)
    else:
      generate_logs_multi(
        args.pkl_path,
        plot_only_top_k=args.plot_only_top_k,
        top_k_scope=args.top_k_scope,
      )
  else:
    single = args.pkl_path[0]
    if args.plot_trials is not None and not os.path.isdir(single):
      parser.error('--plot_trials is only valid when --pkl_path is a grid-search folder.')
    generate_logs(
      single,
      plot_only_top_k=args.plot_only_top_k,
      only_projector_plots=args.only_projector_plots,
      plot_trials=args.plot_trials,
      skip_umap=args.skip_umap,
    )


if __name__ == '__main__':
  main()

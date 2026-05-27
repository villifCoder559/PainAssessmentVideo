#!/usr/bin/env python3
"""
Diagnostic plots for cross_space_projection.py outputs.

Loads a pkl file produced by cross_space_projection.py and writes all plots
to {out_dir}/logs/.

Plots generated:
  1.  predictions_histogram.png     — prediction distributions (bin 0.1) + ground truth
  2a. mae_per_class_old_bar.png       — bar: old model MAE per pain class
  2b. mae_per_class_old_box.png       — box: old model per-sample error per class
  2c. mae_per_class_projected_bar.png — bar: projected MAE per pain class
  2d. mae_per_class_projected_box.png — box: projected per-sample error per class
  3a. mae_per_subject_old.png       — single bar: old model MAE per subject
  3b. mae_per_subject_projected.png — single bar: projected MAE per subject
  8a. mae_improvement_per_class.png   — bar: old_mae - new_mae per class (green=better, red=worse)
  8b. mae_improvement_per_subject.png — bar: old_mae - new_mae per subject (green=better, red=worse)
  4.  confusion_matrix.png          — new model rounded predictions vs ground truth
                                      (skipped when num_classes > 15)
  5.  umap_all.png                  — 1×2 UMAP: colored by label and by subject
  6.  anchor_weights.png            — weight entropy histogram + top-20 anchor usage
  7.  anchor_umap.png               — old vs new anchor embeddings in UMAP space

Usage:
  python3 cross_space_logs.py --pkl_path <path>
  python3 cross_space_logs.py --pkl_path <folder> --plot_only_top_k 5
"""
import argparse
import glob
import os
import pickle
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
  ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='right')
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


def _draw_mae_improvement_bar(ax, groups, diffs, xlabel, title):
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
  ax.set_xticklabels([str(g) for g in groups], rotation=45, ha='right')
  ax.set_xlabel(xlabel)
  ax.set_ylabel('MAE improvement (old - new)')
  ax.set_title(title)
  ax.grid(axis='y', alpha=0.3)


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
  path = os.path.join(out_dir, 'summary.csv')
  pd.DataFrame([row]).to_csv(path, index=False)
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


def _collect_summary_row(data, pkl_path):
  """
  Extract hyperparameters and metrics from a grid-format pkl into a flat dict.

  Args:
    data     (dict): Grid-format pkl contents (must have trial_params and metrics).
    pkl_path (str):  Path to the pkl file (unused, kept for signature consistency).

  Returns:
    dict: Flat row with trial_number, 8 hyperparams, mae, ccc.
  """
  p = data['trial_params']
  m = data['metrics']
  return {
    'trial_number':          data['trial_number'],
    'num_anchors':           p['num_anchors'],
    'anchor_selection_type': p['anchor_selection_type'],
    'csv_anchor_selection':  p['csv_anchor_selection'],
    'old_model_csv':         p['old_model_csv'],
    'interpolation_similarity': p['interpolation_similarity'],
    'weighting_method':      p['weighting_method'],
    'temperature':           p.get('temperature'),
    'rbf_sigma':             p['rbf_sigma'],
    'mae':                   m['mae'],
    'ccc':                   m['ccc'],
  }


def _extract_linear_bundle(data):
  """
  Return the closed-form-projector training bundle from a pkl dict, if present.

  Both interpolation_similarity='linear' and 'procrustes' write their bundle under
  the same 'linear_projector' key (historical naming kept for log/plot compat); a
  'kind' field inside the bundle disambiguates them. The bundle is absent for
  runs where no projector was trained: num_anchors in {0, -1} or
  interpolation_similarity not in {'linear', 'procrustes'}.

  Args:
    data (dict): Deserialized pkl contents.

  Returns:
    dict | None: The bundle with keys 'config', 'norm_stats', 'best_epoch',
      'best_val_mse', 'ckpt_path', 'metrics', 'splits', 'kind' ('linear' or
      'procrustes'), and optionally 'procrustes_params'. Returns None if the
      key is absent.
  """
  return data.get('linear_projector')


# ── plot functions ───────────────────────────────────────────────────────────

def plot_predictions_histogram(new_preds, old_preds, labels, out_dir,
                               run_label: str = '', axes=None):
  """
  Side-by-side histograms of new and old model predictions with KDE overlay.

  Bars inside the ground-truth label range are blue; bars outside are orange.
  Dotted vertical lines mark the label min/max. A KDE curve is overlaid on each
  panel.

  Args:
    new_preds  (np.ndarray): Shape (N,), projected model float predictions.
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth pain labels.
    out_dir    (str): Directory where the plot is saved (ignored when axes provided).
    run_label  (str): Optional run identity string appended to plot titles.
    axes       (array-like[Axes] | None): Two pre-existing axes for dashboard
               embedding. When None a new figure is created and saved.
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
    (axes[0], new_preds, 'Projected (new model)'),
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

    ax.set_xlabel('Pain level')
    ax.set_ylabel('Count')
    ax.set_title(f'Prediction distribution — {name}{suffix}')
    ax.set_xlim(lo - step, hi + step)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

  if standalone:
    plt.tight_layout()
    path = os.path.join(out_dir, 'predictions_histogram.png')
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
    ax.set_xlabel('Pain level')
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
    ax.set_xlabel('Pain level')
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
    ax, groups, diffs, 'Pain level',
    f'MAE improvement per pain class (old - new){suffix}',
  )

  if standalone:
    plt.tight_layout()
    path = os.path.join(out_dir, 'mae_improvement_per_class.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_mae_per_subject(new_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label: str = ''):
  """
  Two separate single-bar figures of MAE per subject: one for the old model and
  one for the projected model.

  Args:
    new_preds   (np.ndarray): Shape (N,), projected model predictions.
    old_preds   (np.ndarray): Shape (N,), old model predictions.
    labels      (np.ndarray): Shape (N,), ground-truth labels.
    sample_ids  (np.ndarray): Shape (N,), int sample IDs.
    subject_map (dict[int, int]): Mapping from sample_id to subject_id.
    out_dir     (str): Output directory.
    run_label   (str): Optional run identity string appended to plot titles.
  """
  subj_ids = np.array([subject_map.get(int(sid), -1) for sid in sample_ids])
  old_mae  = _mae_per_group(old_preds, labels, subj_ids)
  new_mae  = _mae_per_group(new_preds, labels, subj_ids)
  groups   = sorted(set(old_mae) | set(new_mae))
  old_vals = [old_mae.get(g, (float('nan'), 0))[0] for g in groups]
  new_vals = [new_mae.get(g, (float('nan'), 0))[0] for g in groups]
  suffix = f' | {run_label}' if run_label else ''

  for vals, color, name, filename in [
    (old_vals, 'steelblue',  'Old model',       'mae_per_subject_old.png'),
    (new_vals, 'darkorange', 'Projected (new)', 'mae_per_subject_projected.png'),
  ]:
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
  path = os.path.join(out_dir, 'mae_improvement_per_subject.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_confusion_matrix_cross(new_preds, labels, out_dir, num_classes: int,
                                run_label: str = '', ax=None):
  """
  Confusion matrix of rounded projected predictions vs ground-truth labels.

  Predictions are rounded to the nearest integer and clipped to [0, num_classes - 1].

  Args:
    new_preds   (np.ndarray): Shape (N,), float projected predictions.
    labels      (np.ndarray): Shape (N,), ground-truth float labels.
    out_dir     (str): Output directory (ignored when ax is provided).
    num_classes (int): Number of distinct pain classes inferred from the labels.
    run_label   (str): Optional run identity string appended to plot titles.
    ax          (matplotlib.axes.Axes | None): Pre-existing axes for dashboard
                embedding. When None the plot is saved as confusion_matrix.png.
  """
  preds_int  = torch.tensor(_round_preds(new_preds, num_classes),                                    dtype=torch.long)
  labels_int = torch.tensor(np.clip(np.round(labels), 0, num_classes - 1).astype(np.int64), dtype=torch.long)
  cm = MulticlassConfusionMatrix(num_classes=num_classes)
  cm.update(preds_int, labels_int)
  cm_arr = cm.compute().cpu().numpy()[:num_classes, :num_classes].astype(int)
  suffix = f' | {run_label}' if run_label else ''

  standalone = ax is None
  if standalone:
    fig, ax = plt.subplots(figsize=(5, 4))

  sns.heatmap(
    cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax,
    linewidths=0.5, linecolor='lightgray', annot_kws={'size': 7},
  )
  ax.set_title(f'Confusion matrix{suffix}', fontsize=10, fontweight='bold')
  ax.set_xlabel('Predicted', fontsize=8)
  ax.set_ylabel('True', fontsize=8)
  ax.tick_params(axis='x', rotation=45, labelsize=7)
  ax.tick_params(axis='y', rotation=0,  labelsize=7)

  if standalone:
    fig.tight_layout()
    path = os.path.join(out_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def plot_umap(embeddings, labels, sample_ids, subject_map, out_dir, run_label: str = ''):
  """
  Compute UMAP on projected embeddings and plot colored by label and by subject.

  Args:
    embeddings  (np.ndarray): Shape (N, D), projected embedding matrix.
    labels      (np.ndarray): Shape (N,), ground-truth pain labels.
    sample_ids  (np.ndarray): Shape (N,), int sample IDs.
    subject_map (dict[int, int]): Mapping from sample_id to subject_id.
    out_dir     (str): Output directory.
    run_label   (str): Optional run identity string appended to plot titles.
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
  plt.colorbar(sc, ax=axes[0], label='Pain level')
  axes[0].set_title(f'UMAP — projected embeddings (by pain label){suffix}')
  axes[0].set_xlabel('UMAP 1')
  axes[0].set_ylabel('UMAP 2')
  plot_reducted_embeddings(
    reduced_embeddings=reduced, labels=subj_ids,
    output_folder=out_dir, title=f'UMAP — projected embeddings (by subject){suffix}',
    group_by='subjects', cmap=cmap_subj, save_plot=False, ax=axes[1], reduction_name='UMAP',
  )
  plt.tight_layout()
  path = os.path.join(out_dir, 'umap_all.png')
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


def plot_anchor_umap(old_anchors_emb, new_anchors_emb, anchor_labels, out_dir, run_label: str = ''):
  """
  Side-by-side UMAP of old and new anchor embeddings, colored by pain label.

  Each embedding set is reduced independently (dimensions differ), so they share
  the same color scale but not the same coordinate space.

  Args:
    old_anchors_emb (np.ndarray): Shape (K, D_old), old model anchor embeddings.
    new_anchors_emb (np.ndarray): Shape (K, D_new), new model anchor embeddings.
    anchor_labels   (np.ndarray): Shape (K,), pain labels for anchors.
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
    plt.colorbar(sc, ax=ax, label='Pain level')
    ax.set_title(title)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

  plt.suptitle(f'Anchor embeddings in UMAP space — old vs new model{suffix}')
  plt.tight_layout()
  path = os.path.join(out_dir, 'anchor_umap.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_prediction_scatter(new_preds, old_preds, labels, out_dir,
                            run_label: str = '', axes=None):
  """
  Side-by-side scatter of model predictions vs ground-truth labels.

  Left panel: projected (new) model. Right panel: old model.
  Each panel includes a red y=x reference line and a text annotation with
  MAE and CCC.

  Args:
    new_preds  (np.ndarray): Shape (N,), projected model float predictions.
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth pain labels.
    out_dir    (str): Directory where the plot is saved (ignored when axes provided).
    run_label  (str): Optional run identity string appended to plot titles.
    axes       (array-like[Axes] | None): Two pre-existing axes for dashboard
               embedding. When None a new figure is created and saved.
  """
  suffix     = f' | {run_label}' if run_label else ''
  standalone = axes is None
  if standalone:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
  else:
    fig = axes[0].figure

  for ax, preds, name in [
    (axes[0], new_preds, 'Projected (new model)'),
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
    ax.set_title(f'Predicted vs Ground Truth — {name}{suffix}', fontsize=10, fontweight='bold')
    ax.set_xlabel('True label', fontsize=9)
    ax.set_ylabel('Predicted value', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

  if standalone:
    plt.tight_layout()
    path = os.path.join(out_dir, 'prediction_scatter.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_prediction_by_class_boxplot(new_preds, old_preds, labels, out_dir, run_label: str = ''):
  """
  1×2 box plot of raw float predictions grouped by ground-truth pain class.

  Left panel: projected (new) model. Right panel: old model.
  A short red reference segment at y = class_id marks perfect calibration.

  Args:
    new_preds  (np.ndarray): Shape (N,), projected model float predictions.
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth pain labels.
    out_dir    (str): Output directory.
    run_label  (str): Optional run identity string appended to plot titles.
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
    (axes[0], new_preds, 'Projected (new model)'),
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
  path = os.path.join(out_dir, 'prediction_by_class_boxplot.png')
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
    header = '── LINEAR_PROJECTOR_CONFIG ──'
    cfg_keys = ('lr', 'batch_size', 'optimizer', 'weight_decay', 'epochs',
                'normalize_embeddings', 'loss', 'split_ratios', 'device',
                'num_workers')
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
  path = os.path.join(out_dir, 'projector_training_curves.png')
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


# ── search-level summary plots ────────────────────────────────────────────────

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
    'interpolation_similarity', 'weighting_method', 'temperature', 'rbf_sigma',
  ]
  active = [c for c in hp_cols if c in df.columns and df[c].nunique() >= 2]
  if not active:
    return

  y_top = float(df['mae'].max()) * 1.05

  for col in active:
    grp = (
      df.groupby(col)['mae']
      .agg(mean='mean', lo='min', hi='max')
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
                           squeeze=False)

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
  # plt.tight_layout()
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
                             squeeze=False)

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
    # plt.tight_layout()
    path = os.path.join(out_dir, f'heatmap_{scale_col}_interpolation_similarity.png')
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
        .agg(mean='mean', lo='min', hi='max')
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
  plot_hyperparam_mae_summary(df, summary_dir)
  plot_temperature_anchors_heatmap(df, summary_dir)
  plot_scale_interp_heatmap(df, summary_dir)
  plot_mae_anchors_per_interp_sim(df, summary_dir)
  return summary_dir


def plot_dashboard(new_preds, old_preds, labels, num_classes, mae, ccc, out_dir,
                   run_label: str = ''):
  """
  Combined dashboard PNG with all key diagnostic plots for a single run.

  Layout (3 rows × 3 cols via GridSpec):
    Row 0: confusion_matrix | mae_per_class_new (bar+box) | prediction_scatter
    Row 1: pred_by_class projected | pred_by_class old | metrics table
    Row 2: prediction_histogram (full width, 3 columns)

  The confusion-matrix cell is left empty when num_classes > 15.

  The MAE per class bar+box cell uses a nested GridSpec (2 sub-rows).

  Args:
    new_preds   (np.ndarray): Shape (N,), projected model float predictions.
    old_preds   (np.ndarray): Shape (N,), old model float predictions.
    labels      (np.ndarray): Shape (N,), ground-truth pain labels.
    num_classes (int): Number of distinct pain classes.
    mae         (float): Global MAE for the projected model.
    ccc         (float): Global CCC for the projected model.
    out_dir     (str): Output directory.
    run_label   (str): Optional run identity string appended to the suptitle.
  """
  suffix     = f' | {run_label}' if run_label else ''
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
    'MAE', f'MAE per class — Projected{suffix}', 'darkorange',
  )
  fig.axes[-1].set_xlabel('Pain level', fontsize=8)

  # ── Row 0, Col 2: prediction scatter (nested 1×2) ───────────────────────────
  inner_scatter = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[0, 2], wspace=0.35,
  )
  plot_prediction_scatter(
    new_preds, old_preds, labels, out_dir,
    run_label=run_label,
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
  ax_raw.set_title(f'Pred by class — Projected{suffix}', fontsize=9, fontweight='bold')
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
  rows_data = [
    ['MAE (projected)', f'{mae:.4f}'],
    ['CCC (projected)', f'{ccc:.4f}'],
    ['MAE (old)',       f'{float(np.mean(np.abs(old_preds - labels))):.4f}'],
    ['N samples',       str(len(labels))],
    ['N classes',       str(num_classes)],
  ]
  tbl = ax_tbl.table(
    cellText=rows_data, colLabels=['Metric', 'Value'],
    loc='center', cellLoc='center',
  )
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(9)
  tbl.scale(1.2, 1.8)
  for col in range(2):
    tbl[(0, col)].set_facecolor('#4C72B0')
    tbl[(0, col)].set_text_props(color='white', fontweight='bold')
  for row in range(1, len(rows_data) + 1):
    bg = '#eef2ff' if row % 2 == 0 else 'white'
    for col in range(2):
      tbl[(row, col)].set_facecolor(bg)
  ax_tbl.set_title('Metrics Summary', fontsize=10, fontweight='bold', pad=8)

  # ── Row 2: prediction histogram (full width, nested 1×2) ────────────────────
  inner_hist = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=gs[2, :], wspace=0.3,
  )
  plot_predictions_histogram(
    new_preds, old_preds, labels, out_dir,
    run_label=run_label,
    axes=[fig.add_subplot(inner_hist[0]), fig.add_subplot(inner_hist[1])],
  )

  # ── Row 3: MAE improvement per class (full width) ───────────────────────────
  plot_mae_improvement_per_class(
    new_preds, old_preds, labels, out_dir,
    run_label=run_label, ax=fig.add_subplot(gs[3, :]),
  )

  fig.suptitle(f'Dashboard{suffix}', fontsize=15, fontweight='bold', y=1.002)
  path = os.path.join(out_dir, 'dashboard.png')
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


def _resolve_new_model_pth(data, fmt, pkl_path):
  """
  Best-effort lookup of the path to the new model's checkpoint.

  Standalone pkls store it under config_cross_space_projection. Grid trial
  pkls do not — for those we parse the search root's best_config.txt to
  recover --new_model_pth from the saved script_cmd line.

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
  cfg_txt = os.path.join(search_root, 'best_config.txt')
  if not os.path.isfile(cfg_txt):
    return None
  with open(cfg_txt) as f:
    for line in f:
      if not line.startswith('script_cmd:'):
        continue
      tokens = line.split()
      for i, tok in enumerate(tokens):
        if tok == '--new_model_pth' and i + 1 < len(tokens):
          return tokens[i + 1]
  return None


def _resolve_new_features_path(data, fmt, new_model_pth):
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
      # Try parsing search root's best_config.txt
      search_root = os.path.dirname(os.path.dirname(
        data.get('_pkl_path') or ''))  # populated by caller when available
      cfg_txt = os.path.join(search_root, 'best_config.txt') if search_root else ''
      old_model_pth = None
      if cfg_txt and os.path.isfile(cfg_txt):
        with open(cfg_txt) as f:
          for line in f:
            if line.startswith('script_cmd:'):
              tokens = line.split()
              for i, tok in enumerate(tokens):
                if tok == '--old_model_pth' and i + 1 < len(tokens):
                  old_model_pth = tokens[i + 1]
                  break
              break
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
  features_override = _resolve_new_features_path(data, fmt, new_model_pth)

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

  Samples are grouped by their rounded ground-truth pain label. Reuses
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
    ax.set_xlabel('Pain level')

  fig.suptitle(f'Embedding reconstruction per class — split={split_name}{suffix}',
               fontsize=13, fontweight='bold')
  plt.tight_layout(rect=(0, 0, 1, 0.96))
  path = os.path.join(out_dir, f'emb_recon_per_class_box_{split_name}.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def log_embedding_reconstruction(data, fmt, pkl_path, out_dir, run_label=''):
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
  """
  scope_ids, split_name = _resolve_scope_sample_ids(data, fmt)
  if scope_ids.size == 0:
    print('[emb_recon] new_model_tensors has no sample_ids — skipping.')
    return

  new_t      = data['new_model_tensors']
  projected  = np.asarray(new_t['embeddings'], dtype=np.float32)
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

  csv_path = os.path.join(out_dir, f'embedding_reconstruction_{split_name}.csv')
  df.to_csv(csv_path, index=False)
  print(f'Saved: {csv_path}')

  src_counts = df['source'].value_counts().to_dict()
  print(f'[emb_recon] source counts: {src_counts}')
  print(f'[emb_recon] means — L1={df["l1"].mean():.4f}  L2={df["l2"].mean():.4f}  '
        f'cos_sim={df["cos_sim"].mean():.4f}  cos_dist={df["cos_dist"].mean():.4f}')

  plot_embedding_reconstruction_histograms(df, out_dir, run_label=run_label,
                                           split_name=split_name)
  plot_embedding_reconstruction_per_class(df, out_dir, run_label=run_label,
                                          split_name=split_name)


# ── search-folder aggregation ────────────────────────────────────────────────

def generate_logs_search(search_dir, plot_only_top_k=None):
  """
  Process all results.pkl files found recursively under a grid-search root folder.

  For each pkl: generates per-trial diagnostic plots in a logs/ sub-directory.
  After all trials: writes a summary.csv at the search root, sorted by MAE ascending.

  When plot_only_top_k is set, metrics are collected for all trials (summary.csv
  covers the full sweep) but diagnostic plots are generated only for the top K
  trials by MAE ascending.

  Args:
    search_dir      (str): Root folder of a cross_space_projection grid search.
    plot_only_top_k (int | None): If set, limit plot generation to the K best
      trials by MAE. If None, plots are generated for every trial.

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

  if plot_only_top_k is None:
    rows = []
    for pkl_path in tqdm(pkl_paths, desc='Processing trials', unit='trial'):
      try:
        _, row = generate_logs(pkl_path)
        if row is not None:
          rows.append(row)
      except Exception as exc:
        print(f'[WARN] Skipping {pkl_path}: {exc}')
  else:
    # Phase 1: collect metrics for all trials without generating plots.
    rows = []
    for pkl_path in tqdm(pkl_paths, desc='Collecting metrics', unit='trial'):
      try:
        data = _load_pkl(pkl_path)
        row = _collect_summary_row(data, pkl_path)
        row['_pkl_path'] = pkl_path
        rows.append(row)
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
          generate_logs(pkl_path)
        except Exception as exc:
          print(f'[WARN] Skipping {pkl_path}: {exc}')

  return search_dir


# ── multi-folder entry point ─────────────────────────────────────────────────

def generate_logs_multi(search_dirs, plot_only_top_k=None):
  """
  Process multiple grid-search root folders in a single pass.

  Runs per-folder analysis (per-trial diagnostic plots, per-folder summary.csv,
  per-folder search_summary/ plots) and then writes a cross-folder global summary
  into a global_summary/ subdirectory inside every input folder.

  When plot_only_top_k is set, the top-K limit is applied globally across all
  folders combined: only the K trials with the lowest MAE — regardless of which
  folder they belong to — receive per-trial diagnostic plots.

  Args:
    search_dirs     (list[str]): Paths to grid-search root folders to process.
    plot_only_top_k (int | None): If set, generate per-trial plots only for the
      K trials with the lowest MAE across all folders. Summary CSVs and global
      plots still cover all trials. If None, plots every trial.

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
    plot_hyperparam_mae_summary(clean_global_df, global_out_dir)
    plot_temperature_anchors_heatmap(clean_global_df, global_out_dir)
    plot_scale_interp_heatmap(clean_global_df, global_out_dir)
    plot_mae_anchors_per_interp_sim(clean_global_df, global_out_dir)

  return search_dirs


# ── entry point ──────────────────────────────────────────────────────────────

def generate_logs(pkl_path, plot_only_top_k=None, only_projector_plots=False):
  """
  Load a cross_space_projection pkl and write all diagnostic plots.

  Accepts either a path to a single pkl file or a grid-search root directory.
  When given a directory, delegates to generate_logs_search.

  For grid-search trial pkls (containing trial_params) the subject-map CSV is
  resolved from <search_root>/precomputed/old_tensors_<old_model_csv>.csv and
  the anchor-UMAP plot is skipped (anchor embeddings are absent in that format).

  Args:
    pkl_path            (str): Path to a results pkl, or a grid-search root directory.
    plot_only_top_k     (int | None): When pkl_path is a directory, limits plot
      generation to the top K trials by MAE ascending. Ignored for single files.
    only_projector_plots (bool): If True, emit only the linear-projector
      training-diagnostic plots (projector_training_curves, train_val_gap,
      weight_analysis, norm_comparison) and skip everything else. Useful for
      regenerating these on existing pkls without redoing UMAPs / heavy work.

  Returns:
    tuple[str, dict | None]:
      - Path to the logs directory (or search_dir when given a directory).
      - Summary row dict for grid-format pkls; None for standalone runs.
  """
  if os.path.isdir(pkl_path):
    return generate_logs_search(pkl_path, plot_only_top_k=plot_only_top_k), None

  data = _load_pkl(pkl_path)
  fmt  = _detect_format(data)

  if only_projector_plots:
    if fmt == 'grid':
      out_dir   = os.path.join(os.path.dirname(pkl_path), 'logs')
      uid       = data.get('uid') or os.path.basename(os.path.dirname(os.path.dirname(pkl_path))).split('_')[-1]
      run_label = f"Trial #{data['trial_number']} | UID: {uid}"
    else:
      cfg       = data['config_cross_space_projection']
      out_dir   = os.path.join(cfg['out_dir'], 'logs')
      run_label = f"UID: {cfg['uid']}"
    os.makedirs(out_dir, exist_ok=True)
    print(f'[cross_space_logs] (projector-only) Output: {out_dir}')
    plot_projector_diagnostics(_extract_linear_bundle(data), out_dir, run_label=run_label)
    return out_dir, None

  if fmt == 'grid':
    out_dir       = os.path.join(os.path.dirname(pkl_path), 'logs')
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
    out_dir         = os.path.join(cfg['out_dir'], 'logs')
    run_label       = f"UID: {cfg['uid']}"
    num_anchors_val = cfg.get('num_anchors')
    subject_map     = _get_subject_map(cfg['old_tensors_csv_path'])
    summary_row     = {
      'uid':                      cfg.get('uid'),
      'num_anchors':              cfg.get('num_anchors'),
      'anchor_selection_type':    cfg.get('anchor_selection_type'),
      'old_model_csv':            cfg.get('old_model_csv'),
      'interpolation_similarity': cfg.get('interpolation_similarity'),
      'weighting_method':         cfg.get('weighting_method'),
      'temperature':              cfg.get('temperature'),
      'rbf_sigma':                cfg.get('rbf_sigma'),
    }

  if num_anchors_val == 0:
    run_label += ' | K=0 (identity)'
  elif num_anchors_val == -1:
    run_label += ' | K=-1 (original_video)'

  os.makedirs(out_dir, exist_ok=True)
  print(f'[cross_space_logs] Output: {out_dir}')

  new_t      = data['new_model_tensors']
  old_t      = data['old_model_tensors']
  new_preds   = np.asarray(new_t['predictions'], dtype=np.float32).squeeze()
  old_preds   = np.asarray(old_t['predictions'], dtype=np.float32).squeeze()
  labels      = np.asarray(new_t['labels'],      dtype=np.float32)
  sample_ids  = np.asarray(new_t['sample_ids'],  dtype=np.int64)
  weights     = np.asarray(new_t['weights'],      dtype=np.float32)
  num_classes = int(np.round(labels).max()) + 1

  if fmt == 'standalone':
    mae = float(np.mean(np.abs(new_preds - labels)))
    ccc = concordance_ccc(labels, new_preds)
    summary_row['mae'] = mae
    summary_row['ccc'] = ccc
  print(f'Global metrics — MAE: {mae:.4f}  CCC: {ccc:.4f}')

  plot_predictions_histogram(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_mae_per_class(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_mae_per_subject(new_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label=run_label)
  plot_mae_improvement_per_class(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_mae_improvement_per_subject(new_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label=run_label)
  if num_classes <= 15:
    plot_confusion_matrix_cross(new_preds, labels, out_dir, num_classes, run_label=run_label)
  else:
    print(f'Skipped confusion matrix: num_classes={num_classes} > 15')
  plot_umap(new_t['embeddings'], labels, sample_ids, subject_map, out_dir, run_label=run_label)
  try:
    plot_anchor_weights(weights, out_dir, run_label=run_label)
  except Exception as exc:
    print(f'[WARN] Failed to plot anchor weights: {exc}')
  plot_prediction_scatter(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_prediction_by_class_boxplot(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_dashboard(new_preds, old_preds, labels, num_classes, mae, ccc, out_dir, run_label=run_label)

  plot_projector_diagnostics(_extract_linear_bundle(data), out_dir, run_label=run_label)

  try:
    log_embedding_reconstruction(data, fmt, pkl_path, out_dir, run_label=run_label)
  except Exception as exc:
    print(f'[WARN] embedding reconstruction diagnostic failed: {exc}')

  if fmt == 'standalone' and data.get('old_model_anchors_embeddings') is not None:
    plot_anchor_umap(
      data['old_model_anchors_embeddings']['embeddings'],
      data['new_model_anchors_embeddings']['embeddings'],
      data['old_model_anchors_embeddings']['labels'],
      out_dir,
      run_label=run_label,
    )

  _write_summary_csv(summary_row, out_dir)
  print(f'Done. All logs in {out_dir}')
  return out_dir, summary_row


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Generate diagnostic plots from a cross_space_projection pkl file or grid-search folder.'
  )
  parser.add_argument(
    '--pkl_path', type=str, nargs='+', required=True,
    help=(
      'One or more paths to a results pkl produced by cross_space_projection.py, '
      'OR grid-search root folder(s). '
      'Single path: behaviour is unchanged — pkl or folder processed as before. '
      'Multiple paths: per-folder analysis is run for each folder and a merged '
      'global_summary/ (global_summary.csv + hyperparameter plots) is written '
      'into every input folder.'
    ),
  )
  parser.add_argument(
    '--plot_only_top_k', type=int, default=3,
    help=(
      'When pkl_path is a folder, generate diagnostic plots only for the '
      'top K trials ranked by MAE ascending. Metrics are still collected '
      'for all trials and summary.csv covers the full sweep. '
      'Ignored when pkl_path is a single pkl file.'
    ),
  )
  parser.add_argument(
    '--only_projector_plots', action='store_true', default=False,
    help=(
      'Emit only the linear-projector training-diagnostic plots and skip '
      'everything else (UMAPs, confusion matrix, dashboard, etc.). Only '
      'applies when pkl_path is a single pkl file.'
    ),
  )
  args = parser.parse_args()
  if len(args.pkl_path) > 1:
    if args.only_projector_plots:
      for p in args.pkl_path:
        generate_logs(p, only_projector_plots=True)
    else:
      generate_logs_multi(args.pkl_path, plot_only_top_k=args.plot_only_top_k)
  else:
    generate_logs(
      args.pkl_path[0],
      plot_only_top_k=args.plot_only_top_k,
      only_projector_plots=args.only_projector_plots,
    )

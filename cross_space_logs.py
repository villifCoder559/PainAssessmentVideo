#!/usr/bin/env python3
"""
Diagnostic plots for cross_space_projection.py outputs.

Loads a pkl file produced by cross_space_projection.py and writes all plots
to {out_dir}/logs/.

Plots generated:
  1.  predictions_histogram.png     — prediction distributions (bin 0.1) + ground truth
  2a. mae_per_class_old.png         — single bar: old model MAE per pain class
  2b. mae_per_class_projected.png   — single bar: projected MAE per pain class
  3a. mae_per_subject_old.png       — single bar: old model MAE per subject
  3b. mae_per_subject_projected.png — single bar: projected MAE per subject
  4.  confusion_matrix.png          — new model rounded predictions vs ground truth
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
import matplotlib.pyplot as plt
import numpy as np
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


def _round_preds(preds):
  """
  Squeeze, round to nearest integer, and clip to [0, 9].

  Args:
    preds (np.ndarray): Float predictions, shape (N,) or (N, 1).

  Returns:
    np.ndarray: Integer predictions, shape (N,), dtype int64.
  """
  arr = np.asarray(preds, dtype=np.float32).squeeze()
  return np.clip(np.round(arr), 0, 9).astype(np.int64)


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
    'temperature':           p['temperature'],
    'rbf_sigma':             p['rbf_sigma'],
    'mae':                   m['mae'],
    'ccc':                   m['ccc'],
  }


# ── plot functions ───────────────────────────────────────────────────────────

def plot_predictions_histogram(new_preds, old_preds, labels, out_dir, run_label: str = ''):
  """
  Side-by-side histograms of new and old model predictions overlaid with ground truth.

  Args:
    new_preds  (np.ndarray): Shape (N,), projected model float predictions.
    old_preds  (np.ndarray): Shape (N,), old model float predictions.
    labels     (np.ndarray): Shape (N,), ground-truth pain labels.
    out_dir    (str): Directory where the plot is saved.
    run_label  (str): Optional run identity string appended to plot titles.
  """
  lo = float(min(labels.min(), new_preds.min(), old_preds.min()))
  hi = float(max(labels.max(), new_preds.max(), old_preds.max()))
  bins_fine   = np.arange(lo, hi + 0.11, 0.1)
  bins_coarse = np.arange(lo, hi + 1.1,  1.0)
  suffix = f' | {run_label}' if run_label else ''

  fig, axes = plt.subplots(1, 2, figsize=(14, 5))
  for ax, preds, color, name in [
    (axes[0], new_preds, 'darkorange', 'Projected (new model)'),
    (axes[1], old_preds, 'steelblue',  'Old model'),
  ]:
    ax.hist(preds,  bins=bins_fine,   alpha=0.7, label=name,           color=color)
    ax.hist(labels, bins=bins_coarse, alpha=0.4, label='Ground truth', color='green', edgecolor='darkgreen')
    ax.set_xlabel('Pain level')
    ax.set_ylabel('Count')
    ax.set_title(f'Prediction distribution — {name}{suffix}')
    ax.legend()

  plt.tight_layout()
  path = os.path.join(out_dir, 'predictions_histogram.png')
  fig.savefig(path, dpi=150)
  plt.close(fig)
  print(f'Saved: {path}')


def plot_mae_per_class(new_preds, old_preds, labels, out_dir, run_label: str = ''):
  """
  Two separate single-bar figures of MAE per pain class (0–9): one for the old
  model and one for the projected model.

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
  suffix = f' | {run_label}' if run_label else ''

  for vals, color, name, filename in [
    (old_vals, 'steelblue',  'Old model',       'mae_per_class_old.png'),
    (new_vals, 'darkorange', 'Projected (new)', 'mae_per_class_projected.png'),
  ]:
    fig, ax = plt.subplots(figsize=(12, 5))
    _single_bar(ax, groups, vals, 'MAE', f'MAE per pain class — {name}{suffix}', color)
    ax.set_xlabel('Pain level')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
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
  width    = max(12, len(groups) * 0.9)
  suffix = f' | {run_label}' if run_label else ''

  for vals, color, name, filename in [
    (old_vals, 'steelblue',  'Old model',       'mae_per_subject_old.png'),
    (new_vals, 'darkorange', 'Projected (new)', 'mae_per_subject_projected.png'),
  ]:
    fig, ax = plt.subplots(figsize=(width, 5))
    _single_bar(ax, groups, vals, 'MAE', f'MAE per subject — {name}{suffix}', color)
    ax.set_xlabel('Subject ID')
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved: {path}')


def plot_confusion_matrix_cross(new_preds, labels, out_dir, run_label: str = ''):
  """
  Confusion matrix of rounded projected predictions vs ground-truth labels.

  Predictions are rounded to the nearest integer and clipped to [0, 9].

  Args:
    new_preds  (np.ndarray): Shape (N,), float projected predictions.
    labels     (np.ndarray): Shape (N,), ground-truth float labels.
    out_dir    (str): Output directory.
    run_label  (str): Optional run identity string appended to plot titles.
  """
  preds_int  = torch.tensor(_round_preds(new_preds),                       dtype=torch.long)
  labels_int = torch.tensor(np.clip(np.round(labels), 0, 9).astype(np.int64), dtype=torch.long)
  cm = MulticlassConfusionMatrix(num_classes=10)
  cm.update(preds_int, labels_int)
  suffix = f' | {run_label}' if run_label else ''
  path = os.path.join(out_dir, 'confusion_matrix.png')
  plot_confusion_matrix(
    cm,
    title=f'Confusion matrix — projected predictions (rounded){suffix}',
    saving_path=path,
  )
  plt.close('all')
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
  plt.tight_layout()
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

def generate_logs(pkl_path, plot_only_top_k=None):
  """
  Load a cross_space_projection pkl and write all diagnostic plots.

  Accepts either a path to a single pkl file or a grid-search root directory.
  When given a directory, delegates to generate_logs_search.

  For grid-search trial pkls (containing trial_params) the subject-map CSV is
  resolved from <search_root>/precomputed/old_tensors_<old_model_csv>.csv and
  the anchor-UMAP plot is skipped (anchor embeddings are absent in that format).

  Args:
    pkl_path        (str): Path to a results pkl, or a grid-search root directory.
    plot_only_top_k (int | None): When pkl_path is a directory, limits plot
      generation to the top K trials by MAE ascending. Ignored for single files.

  Returns:
    tuple[str, dict | None]:
      - Path to the logs directory (or search_dir when given a directory).
      - Summary row dict for grid-format pkls; None for standalone runs.
  """
  if os.path.isdir(pkl_path):
    return generate_logs_search(pkl_path, plot_only_top_k=plot_only_top_k), None

  data = _load_pkl(pkl_path)
  fmt  = _detect_format(data)

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
  new_preds  = np.asarray(new_t['predictions'], dtype=np.float32).squeeze()
  old_preds  = np.asarray(old_t['predictions'], dtype=np.float32).squeeze()
  labels     = np.asarray(new_t['labels'],      dtype=np.float32)
  sample_ids = np.asarray(new_t['sample_ids'],  dtype=np.int64)
  weights    = np.asarray(new_t['weights'],      dtype=np.float32)

  if fmt == 'standalone':
    mae = float(np.mean(np.abs(new_preds - labels)))
    ccc = concordance_ccc(labels, new_preds)
    summary_row['mae'] = mae
    summary_row['ccc'] = ccc
  print(f'Global metrics — MAE: {mae:.4f}  CCC: {ccc:.4f}')

  plot_predictions_histogram(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_mae_per_class(new_preds, old_preds, labels, out_dir, run_label=run_label)
  plot_mae_per_subject(new_preds, old_preds, labels, sample_ids, subject_map, out_dir, run_label=run_label)
  plot_confusion_matrix_cross(new_preds, labels, out_dir, run_label=run_label)
  plot_umap(new_t['embeddings'], labels, sample_ids, subject_map, out_dir, run_label=run_label)
  plot_anchor_weights(weights, out_dir, run_label=run_label)

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
    '--plot_only_top_k', type=int, default=None,
    help=(
      'When pkl_path is a folder, generate diagnostic plots only for the '
      'top K trials ranked by MAE ascending. Metrics are still collected '
      'for all trials and summary.csv covers the full sweep. '
      'Ignored when pkl_path is a single pkl file.'
    ),
  )
  args = parser.parse_args()
  if len(args.pkl_path) > 1:
    generate_logs_multi(args.pkl_path, plot_only_top_k=args.plot_only_top_k)
  else:
    generate_logs(args.pkl_path[0], plot_only_top_k=args.plot_only_top_k)

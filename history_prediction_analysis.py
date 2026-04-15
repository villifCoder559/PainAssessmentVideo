#!/usr/bin/env python3
"""
Analyze per-sample prediction histories from a k_fold_results.pkl file.

For each fold in data['results'], generates:
  - Ranked bar chart of top-K best and worst predicted samples (by MAE)      [feature 2]
  - Error heatmap  (samples × epochs, |pred − gt|)                           [feature 6]
  - Per-sample prediction trajectory with GT line (--sample_id)              [feature 5]
  - Per-sample signed error trajectory            (--sample_id)              [feature 5]
  - Prediction distribution histogram at a specific epoch (--epoch)          [feature 7]
  - Annotated top-K and worst-K concatenated videos (--generate_video)       [feature 3]

All analyses can be restricted to an epoch sub-range with --from_to.

Usage:
  python3 history_prediction_analysis.py --pkl path/to/k_fold_results.pkl [options]
"""

import argparse
import os
import pickle
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom import tools
from plot_test_results_from_pkl import plot_prediction_histogram


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
  """
  Parse command-line arguments.

  Returns:
    argparse.Namespace: Parsed arguments.
  """
  parser = argparse.ArgumentParser(
    description='Analyze per-sample prediction histories from a k_fold_results.pkl file.'
  )
  parser.add_argument(
    '--pkl', required=True,
    help='Path to k_fold_results.pkl.'
  )
  parser.add_argument(
    '--top_k', type=int, default=10,
    help='Number of best/worst samples for bar plot and video (default: 10).'
  )
  parser.add_argument(
    '--from_to', type=int, nargs=2, metavar=('FROM', 'TO'),
    help='Epoch range [FROM, TO] inclusive — restricts ALL analyses to this window.'
  )
  parser.add_argument(
    '--sample_id', type=int,
    help='Plot prediction trajectory and signed error trajectory for this sample ID.'
  )
  parser.add_argument(
    '--epoch', type=int,
    help='Plot prediction distribution histogram at this epoch index.'
  )
  parser.add_argument(
    '--generate_video', action='store_true',
    help='Concatenate annotated top-K and worst-K videos.'
  )
  parser.add_argument(
    '--video_ext', default='.mp4',
    help='Video file extension (default: .mp4).'
  )
  return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_data(pkl_path: str) -> dict:
  """
  Load and validate a k_fold_results pickle file.

  Args:
    pkl_path (str): Path to the .pkl file.

  Returns:
    dict: Full data dictionary.

  Raises:
    ValueError: If 'results' key is missing or empty.
  """
  with open(pkl_path, 'rb') as f:
    data = pickle.load(f)
  if 'results' not in data or not data['results']:
    raise ValueError(f"PKL has no 'results' or it is empty: {pkl_path}")
  return data


def resolve_path(path_list: list) -> str:
  """
  Reconstruct a path from a list of components stored in the config.

  The config stores paths as split lists (e.g. ['partA', 'video', 'file.csv']).
  They are resolved relative to the current working directory, consistent
  with how train_model.py is invoked from the project root.

  Args:
    path_list (list[str]): Path components from data['config'].

  Returns:
    str: Joined absolute path.
  """
  return os.path.join(os.getcwd(), *path_list)


def load_gt(csv_path: str) -> dict:
  """
  Load ground-truth labels and sample metadata from the CSV.

  Args:
    csv_path (str): Path to the tab-separated CSV with columns:
                    sample_id, class_id, subject_name, sample_name.

  Returns:
    dict[int, dict]: sample_id → {'class_id': int, 'subject_name': str, 'sample_name': str}.
  """
  df = pd.read_csv(csv_path, sep='\t')
  gt = {}
  for _, row in df.iterrows():
    sid = int(row['sample_id'])
    gt[sid] = {
      'class_id':    int(row['class_id']),
      'subject_name': str(row['subject_name']),
      'sample_name':  str(row['sample_name']),
    }
  return gt


def _save_fig(fig: plt.Figure, out_path: str) -> None:
  """
  Save a figure to disk and close it.

  Args:
    fig      (plt.Figure): Matplotlib figure.
    out_path (str):        Destination .png path.
  """
  fig.savefig(out_path, bbox_inches='tight', dpi=150)
  plt.close(fig)
  print(f'  Saved: {out_path}')


# ---------------------------------------------------------------------------
# Epoch filtering
# ---------------------------------------------------------------------------

def apply_epoch_filter(history: dict, from_ep: int, to_ep: int) -> dict:
  """
  Restrict the prediction history to a given epoch range.

  Args:
    history  (dict[int, torch.Tensor]): sample_id → Tensor(num_epochs,).
    from_ep  (int): First epoch index (inclusive).
    to_ep    (int): Last  epoch index (inclusive).

  Returns:
    dict[int, torch.Tensor]: New dict with tensors sliced to [from_ep : to_ep+1].
  """
  return {sid: t[from_ep : to_ep + 1] for sid, t in history.items()}


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------

def compute_mae(history: dict, gt: dict) -> dict:
  """
  Compute mean absolute error per sample across (filtered) epochs.

  Args:
    history (dict[int, torch.Tensor]): sample_id → Tensor(num_epochs,).
    gt      (dict[int, dict]):          sample_id → {'class_id': int, ...}.

  Returns:
    dict[int, float]: sample_id → mean |pred − gt| over all epochs.
                      Samples absent from gt are skipped.
  """
  mae = {}
  for sid, t in history.items():
    if sid not in gt:
      continue
    preds     = t.numpy().astype(float) if isinstance(t, torch.Tensor) else np.array(t, dtype=float)
    gt_val    = float(gt[sid]['class_id'])
    mae[sid]  = float(np.mean(np.abs(preds - gt_val)))
  return mae


# ---------------------------------------------------------------------------
# Feature 2: Best / worst bar chart
# ---------------------------------------------------------------------------

def plot_bar_best_worst(mae_dict: dict, gt: dict, top_k: int, fold_out_dir: str) -> None:
  """
  Horizontal bar chart with top-K best (lowest MAE) and worst (highest MAE) samples.

  Args:
    mae_dict     (dict[int, float]): sample_id → MAE.
    gt           (dict[int, dict]):  sample_id → metadata.
    top_k        (int):              Number of samples per group.
    fold_out_dir (str):              Output directory.
  """
  sorted_sids = sorted(mae_dict, key=mae_dict.get)
  best_sids   = sorted_sids[:top_k]
  worst_sids  = sorted_sids[-top_k:][::-1]  # highest MAE first

  def _labels_values(sids):
    labels = [f"{gt[s]['sample_name']}  (gt={gt[s]['class_id']})" for s in sids]
    values = [mae_dict[s] for s in sids]
    return labels, values

  best_labels,  best_vals  = _labels_values(best_sids)
  worst_labels, worst_vals = _labels_values(worst_sids)

  fig, (ax_best, ax_worst) = plt.subplots(
    2, 1,
    figsize=(10, max(6, top_k * 0.55 * 2)),
  )

  for ax, labels, values, color, title in [
    (ax_best,  best_labels,  best_vals,  '#4C72B0', f'Top-{top_k} Best Predicted Samples (lowest MAE)'),
    (ax_worst, worst_labels, worst_vals, '#DD8452', f'Top-{top_k} Worst Predicted Samples (highest MAE)'),
  ]:
    ax.barh(range(len(labels)), values, color=color, edgecolor='white', alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

  fig.tight_layout(pad=2.0)
  _save_fig(fig, os.path.join(fold_out_dir, 'bar_best_worst.png'))


# ---------------------------------------------------------------------------
# Feature 5: Sample trajectory
# ---------------------------------------------------------------------------

def plot_sample_trajectory(
  sample_id:    int,
  history:      dict,
  gt:           dict,
  epoch_offset: int,
  fold_out_dir: str,
) -> None:
  """
  Two plots for a single sample: prediction trajectory and signed error trajectory.

  Plot A — prediction across epochs with a horizontal GT line.
  Plot B — signed error (pred − gt) across epochs with a zero line.

  Args:
    sample_id    (int):                        Target sample ID.
    history      (dict[int, torch.Tensor]):    sample_id → Tensor(num_epochs,).
    gt           (dict[int, dict]):            sample_id → metadata.
    epoch_offset (int):                        First epoch in the (filtered) history
                                               (used as x-axis starting point).
    fold_out_dir (str):                        Output directory.
  """
  if sample_id not in history:
    print(f'  Warning: sample_id {sample_id} not in history — skipping trajectory.')
    return
  if sample_id not in gt:
    print(f'  Warning: sample_id {sample_id} not in GT — skipping trajectory.')
    return

  t         = history[sample_id]
  preds     = t.numpy().astype(float) if isinstance(t, torch.Tensor) else np.array(t, dtype=float)
  gt_val    = float(gt[sample_id]['class_id'])
  nm        = gt[sample_id]['sample_name']
  epochs    = np.arange(epoch_offset, epoch_offset + len(preds))

  # Plot A: prediction trajectory
  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(epochs, preds, linewidth=1.2, color='#4C72B0', label='Prediction')
  ax.axhline(gt_val, linestyle='--', color='#C44E52', linewidth=1.5, label=f'GT = {gt_val}')
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Predicted Value')
  ax.set_title(f'Sample {nm} — Prediction Trajectory', fontweight='bold')
  ax.legend(fontsize=9)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  _save_fig(fig, os.path.join(fold_out_dir, f'sample_{sample_id}_trajectory.png'))

  # Plot B: signed error trajectory
  error = preds - gt_val
  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(epochs, error, linewidth=1.2, color='#5A9E6F', label='Error (pred − gt)')
  ax.fill_between(epochs, error, 0, alpha=0.15, color='#5A9E6F')
  ax.axhline(0, linestyle='--', color='#555555', linewidth=1.2)
  ax.set_xlabel('Epoch')
  ax.set_ylabel('Error (pred − gt)')
  ax.set_title(f'Sample {nm} — Error Trajectory', fontweight='bold')
  ax.legend(fontsize=9)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  _save_fig(fig, os.path.join(fold_out_dir, f'sample_{sample_id}_error_traj.png'))


# ---------------------------------------------------------------------------
# Feature 6: Error heatmap
# ---------------------------------------------------------------------------

def plot_error_heatmap(history: dict, gt: dict, fold_out_dir: str) -> None:
  """
  Heatmap of |pred − gt| with rows sorted by mean MAE (best at top).

  Rows = samples, cols = epochs.
  When the number of epochs exceeds 100 and no --from_to was applied, epochs
  are subsampled to at most 100 evenly-spaced columns so the figure stays readable.

  Args:
    history      (dict[int, torch.Tensor]): sample_id → Tensor(num_epochs,).
    gt           (dict[int, dict]):          sample_id → metadata.
    fold_out_dir (str):                      Output directory.
  """
  common_sids = [sid for sid in history if sid in gt]
  if not common_sids:
    print('  Warning: no common sample IDs between history and GT — skipping heatmap.')
    return

  n_epochs    = len(next(iter(history.values())))
  n_samples   = len(common_sids)
  mat         = np.zeros((n_samples, n_epochs), dtype=float)

  for i, sid in enumerate(common_sids):
    t         = history[sid]
    preds     = t.numpy().astype(float) if isinstance(t, torch.Tensor) else np.array(t, dtype=float)
    mat[i, :] = np.abs(preds - float(gt[sid]['class_id']))

  # Sort rows by mean MAE ascending (best at top)
  order      = np.argsort(mat.mean(axis=1))
  mat        = mat[order]
  row_sids   = [common_sids[i] for i in order]
  row_labels = [gt[sid]['sample_name'] for sid in row_sids]

  # Subsample columns when too wide
  max_cols    = 100
  col_indices = np.arange(n_epochs)
  if n_epochs > max_cols:
    step        = n_epochs // max_cols
    col_indices = np.arange(0, n_epochs, step)[:max_cols]
    mat         = mat[:, col_indices]
    print(f'  Heatmap: {n_epochs} epochs → {len(col_indices)} columns (step {step}).')

  fig_h = max(6.0, min(40.0, n_samples * 0.15))
  fig, ax = plt.subplots(figsize=(14, fig_h))

  sns.heatmap(
    mat,
    ax=ax,
    cmap='viridis',
    yticklabels=row_labels if n_samples <= 200 else False,
    xticklabels=False,
    cbar_kws={'label': '|pred − gt|'},
  )

  # x-tick labels: ~10 evenly spaced epoch numbers
  n_ticks        = min(10, mat.shape[1])
  tick_col_pos   = np.linspace(0, mat.shape[1] - 1, n_ticks, dtype=int)
  tick_labels    = [str(col_indices[p]) for p in tick_col_pos]
  ax.set_xticks(tick_col_pos + 0.5)
  ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)

  if n_samples <= 200:
    label_size = max(4, min(8, int(200 / n_samples)))
    ax.tick_params(axis='y', labelsize=label_size)

  ax.set_xlabel('Epoch')
  ax.set_ylabel('Sample  (sorted by mean MAE ↑ best at top)')
  ax.set_title('Error Heatmap  |pred − gt|', fontweight='bold')
  fig.tight_layout()
  _save_fig(fig, os.path.join(fold_out_dir, 'error_heatmap.png'))


# ---------------------------------------------------------------------------
# Feature 7: Epoch histogram
# ---------------------------------------------------------------------------

def plot_epoch_histogram(
  epoch_idx:    int,
  history:      dict,
  csv_path:     str,
  fold_out_dir: str,
) -> None:
  """
  Prediction distribution histogram for one epoch, reusing plot_prediction_histogram.

  Args:
    epoch_idx    (int):                     Target epoch index.
    history      (dict[int, torch.Tensor]): sample_id → Tensor(num_epochs,).
    csv_path     (str):                     Resolved CSV path (for class range colouring).
    fold_out_dir (str):                     Output directory.
  """
  max_ep = len(next(iter(history.values()))) - 1
  if epoch_idx > max_ep:
    print(f'  Warning: --epoch {epoch_idx} exceeds max epoch {max_ep} — skipping histogram.')
    return

  preds_arr = np.array(
    [
      (t[epoch_idx].item() if isinstance(t, torch.Tensor) else float(t[epoch_idx]))
      for t in history.values()
    ],
    dtype=float,
  )

  # Minimal data stub so plot_prediction_histogram can read the CSV class range
  data_stub = {'csv_path': csv_path}
  plot_prediction_histogram(
    data=data_stub,
    results={},
    out_dir=fold_out_dir,
    preds_arr=preds_arr,
    title_suffix=f' — Epoch {epoch_idx}',
    filename_suffix=f'_epoch_{epoch_idx}',
  )
  print(f'  Saved: {os.path.join(fold_out_dir, f"prediction_histogram_epoch_{epoch_idx}.png")}')


# ---------------------------------------------------------------------------
# Feature 3: Annotated video
# ---------------------------------------------------------------------------

def _read_video_frames(video_path: str) -> tuple:
  """
  Read all frames from a video file in RGB format.

  Args:
    video_path (str): Path to the video file.

  Returns:
    tuple[list[np.ndarray], float]: (RGB frames, source FPS).
                                     Returns ([], 25.0) if the file cannot be opened.
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print(f'  Warning: cannot open video {video_path}')
    return [], 25.0
  fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  cap.release()
  return frames, fps


def _annotate_frame(frame: np.ndarray, text: str, bar_height: int = 50) -> np.ndarray:
  """
  Append a black bar at the bottom of a frame and write white annotation text.

  Args:
    frame      (np.ndarray): RGB frame. Shape: (H, W, 3).
    text       (str):        Annotation string.
    bar_height (int):        Height of the appended black bar in pixels.

  Returns:
    np.ndarray: RGB frame with annotation bar. Shape: (H + bar_height, W, 3).
  """
  annotated  = cv2.copyMakeBorder(frame, 0, bar_height, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
  bgr        = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
  font_scale = max(0.4, frame.shape[1] / 1200.0)
  y_text     = frame.shape[0] + bar_height - 12
  cv2.putText(
    bgr, text, (8, y_text),
    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
    (255, 255, 255), 1, cv2.LINE_AA,
  )
  return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def generate_annotated_video(
  sample_ids:         list,
  history:            dict,
  gt:                 dict,
  video_dataset_path: list,
  video_ext:          str,
  output_path:        str,
) -> None:
  """
  Build a concatenated annotated video for the given samples.

  For each sample, all source frames are extracted and a black bar with
  the sample name, mean prediction, and GT is appended at the bottom.
  The annotated clips are written sequentially into a single output video.

  Args:
    sample_ids         (list[int]):              Ordered sample IDs to include.
    history            (dict[int, torch.Tensor]): sample_id → Tensor(num_epochs,).
    gt                 (dict[int, dict]):          sample_id → metadata.
    video_dataset_path (list[str]):               Path components to the video root.
    video_ext          (str):                     Video file extension (e.g. '.mp4').
    output_path        (str):                     Destination .mp4 path.
  """
  all_frames = []
  src_fps    = 25.0
  video_root = resolve_path(video_dataset_path)

  for sid in sample_ids:
    if sid not in gt:
      print(f'  Warning: sample_id {sid} not in GT — skipping.')
      continue

    meta       = gt[sid]
    video_path = os.path.join(video_root, meta['subject_name'], meta['sample_name'] + video_ext)
    if not os.path.isfile(video_path):
      print(f'  Warning: video not found at {video_path} — skipping.')
      continue

    if sid in history:
      t         = history[sid]
      preds     = t.numpy().astype(float) if isinstance(t, torch.Tensor) else np.array(t, dtype=float)
      pred_mean = float(preds.mean())
    else:
      pred_mean = float('nan')

    annotation = (
      f"{meta['sample_name']}   "
      f"pred: {pred_mean:.2f}   "
      f"GT: {meta['class_id']}"
    )

    frames, fps = _read_video_frames(video_path)
    src_fps     = fps
    for frame in frames:
      all_frames.append(_annotate_frame(frame, annotation))

  if not all_frames:
    print(f'  Warning: no frames collected — skipping {output_path}')
    return

  tools.generate_video_from_list_frame(
    list_frame=all_frames,
    path_video_output=output_path,
    fps=int(round(src_fps)),
    already_bgr=False,
  )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
  """
  Entry point: parse arguments, iterate over folds, produce outputs.
  """
  args     = parse_args()
  data     = load_data(args.pkl)
  pkl_dir  = os.path.dirname(os.path.abspath(args.pkl))
  base_out = os.path.join(pkl_dir, 'prediction_analysis')

  csv_path   = resolve_path(data['config']['path_csv_dataset'])
  video_path = data['config']['path_video_dataset']

  if not os.path.isfile(csv_path):
    print(f'Warning: CSV not found at {csv_path} — GT labels unavailable.')
    gt = {}
  else:
    gt = load_gt(csv_path)
    print(f'Loaded {len(gt)} GT samples from {csv_path}')

  for fold_key, fold_data in data['results'].items():
    print(f'\n=== Fold: {fold_key} ===')

    tv = fold_data.get('train_val')
    if tv is None:
      print('  Skipping: no train_val data.')
      continue
    raw_history = tv.get('history_val_sample_predictions')
    if raw_history is None:
      print('  Skipping: history_val_sample_predictions is None.')
      continue

    # Epoch filter
    if args.from_to:
      from_ep, to_ep = args.from_to
      history        = apply_epoch_filter(raw_history, from_ep, to_ep)
      epoch_offset   = from_ep
      print(f'  Epoch filter: [{from_ep}, {to_ep}]  ({to_ep - from_ep + 1} epochs)')
    else:
      history      = raw_history
      epoch_offset = 0

    fold_out_dir = os.path.join(base_out, fold_key)
    os.makedirs(fold_out_dir, exist_ok=True)

    if not gt:
      print('  Skipping: GT labels unavailable.')
      continue

    mae_dict = compute_mae(history, gt)
    if not mae_dict:
      print('  Skipping: no overlapping sample IDs between history and GT.')
      continue
    print(f'  {len(mae_dict)} samples with MAE computed.')

    # Feature 2
    plot_bar_best_worst(mae_dict, gt, args.top_k, fold_out_dir)

    # Feature 6
    plot_error_heatmap(history, gt, fold_out_dir)

    # Feature 5 (optional)
    if args.sample_id is not None:
      plot_sample_trajectory(args.sample_id, history, gt, epoch_offset, fold_out_dir)

    # Feature 7 (optional)
    if args.epoch is not None:
      plot_epoch_histogram(args.epoch, history, csv_path, fold_out_dir)

    # Feature 3 (optional)
    if args.generate_video:
      sorted_sids = sorted(mae_dict, key=mae_dict.get)
      top_k_ids   = sorted_sids[:args.top_k]
      worst_k_ids = sorted_sids[-args.top_k:][::-1]

      print(f'  Generating top-{args.top_k} video …')
      generate_annotated_video(
        sample_ids=top_k_ids,
        history=history,
        gt=gt,
        video_dataset_path=video_path,
        video_ext=args.video_ext,
        output_path=os.path.join(fold_out_dir, 'top_k_video.mp4'),
      )
      print(f'  Generating worst-{args.top_k} video …')
      generate_annotated_video(
        sample_ids=worst_k_ids,
        history=history,
        gt=gt,
        video_dataset_path=video_path,
        video_ext=args.video_ext,
        output_path=os.path.join(fold_out_dir, 'worst_k_video.mp4'),
      )

  print('\nDone.')


if __name__ == '__main__':
  main()

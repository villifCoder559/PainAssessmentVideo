#!/usr/bin/env python3
"""
Analyze k-fold cross-validation log pkl files.

Generates per-fold summary figures, aggregated figures (mean ± std), and
aggregated confusion matrices from a k_fold_results.pkl produced by train_model.py.

Usage:
  python3 analyze_kfold_pkl.py path/to/k_fold_results.pkl
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from plot_test_results_from_pkl import plot_confusion_matrix as _plot_cm_heatmap
from custom.tools import plot_with_std

CLASS_LABELS = ['0', '1', '2', '3', '4']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_get(d, *keys, default=None):
  """
  Safely navigate a nested dict/list with multiple keys.

  Args:
    d:       Root container.
    *keys:   Sequence of keys/indices to traverse.
    default: Value returned on any missing key or type error.

  Returns:
    The nested value or default.
  """
  for k in keys:
    try:
      d = d[k]
    except (KeyError, IndexError, TypeError):
      return default
  return d


def to_numpy(x):
  """
  Convert a tensor, list, or scalar to a 1-D (or N-D) numpy array.

  Args:
    x: Any numeric container supported by PyTorch or numpy.

  Returns:
    numpy.ndarray or None if x is None.
  """
  if x is None:
    return None
  try:
    return x.detach().cpu().numpy()
  except AttributeError:
    return np.array(x)


def extract_cm_array(cm_obj):
  """
  Extract an integer numpy matrix from a MulticlassConfusionMatrix or ndarray.

  Args:
    cm_obj: torchmetrics.MulticlassConfusionMatrix or numpy.ndarray.

  Returns:
    numpy.ndarray of shape (C, C) with integer counts, or None.
  """
  if cm_obj is None:
    return None
  try:
    return cm_obj.compute().detach().cpu().numpy().astype(int)
  except AttributeError:
    pass
  if isinstance(cm_obj, np.ndarray):
    return cm_obj.astype(int)
  return None


def extract_pearson_r(val):
  """
  Extract the scalar r value from a Pearson output that may be a float or tuple.

  Args:
    val: float, or tuple/list where the first element is r.

  Returns:
    float r, or np.nan on failure.
  """
  if val is None:
    return np.nan
  if isinstance(val, (tuple, list)):
    val = val[0]
  try:
    return float(val)
  except (TypeError, ValueError):
    return np.nan


def _save_fig(fig, path):
  """Save figure to path and close it to free memory."""
  os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
  fig.savefig(path, bbox_inches='tight', dpi=150)
  plt.close(fig)


def _label_unavailable(ax, label='Data unavailable'):
  """Write a placeholder in an axis when data is missing."""
  ax.text(0.5, 0.5, label, ha='center', va='center',
          transform=ax.transAxes, color='gray', fontsize=10)
  ax.set_axis_off()


# ---------------------------------------------------------------------------
# Per-fold figure
# ---------------------------------------------------------------------------

def plot_fold(fold_key, fold_data, out_dir):
  """
  Generate a 6-panel summary figure for one cross-validation fold.

  Panels:
    [0,0] test_loss_per_class — one line per class across epochs
    [0,1] test accuracy per class at best epoch (bar chart)
    [0,2] train_losses across epochs with best-epoch marker
    [1,0] train confusion matrix at best epoch
    [1,1] test confusion matrix at best epoch
    [1,2] RMSE and Pearson r across epochs (twin y-axis)

  Args:
    fold_key:  String key identifying this fold.
    fold_data: Dict with key 'train_val' containing training/test logs.
    out_dir:   Directory where fold_summary.png is saved.

  Returns:
    Dict with keys best_epoch, test_l1, test_acc, rmse, pearson_r.
  """
  os.makedirs(out_dir, exist_ok=True)

  tv = safe_get(fold_data, 'train_val', default={})
  te = safe_get(tv, 'test_as_eval', default={})

  train_losses   = to_numpy(safe_get(tv, 'train_losses'))
  test_l1        = to_numpy(safe_get(te, 'test_list_l1_error'))
  test_loss_pc   = safe_get(te, 'test_loss_per_class')        # ndarray (E, C)
  test_acc       = to_numpy(safe_get(te, 'list_test_accuracy'))
  test_acc_pc    = safe_get(te, 'list_test_accuracy_per_class')  # list[Tensor[C]]
  test_rmse      = to_numpy(safe_get(te, 'test_list_rmse'))
  test_pearson_raw = safe_get(te, 'list_test_pearson_correlation')
  train_cms      = safe_get(tv, 'train_confusion_matricies', default={})
  test_cms       = safe_get(te, 'test_confusion_matricies', default=[])

  best_ep = int(np.argmin(test_l1)) if test_l1 is not None and len(test_l1) > 0 else 0
  n_epochs = len(train_losses) if train_losses is not None else 0
  epochs_x = np.arange(n_epochs)

  test_pearson_r = None
  if test_pearson_raw is not None:
    test_pearson_r = np.array([extract_pearson_r(v) for v in test_pearson_raw])

  fig, axes = plt.subplots(2, 3, figsize=(18, 10))
  fig.suptitle(f'Fold: {fold_key}  |  best epoch = {best_ep}', fontsize=13, fontweight='bold')

  # [0,0] test_loss_per_class across epochs
  ax = axes[0, 0]
  try:
    if test_loss_pc is not None and len(test_loss_pc) > 0:
      ep_x = np.arange(len(test_loss_pc))
      for c in range(test_loss_pc.shape[1]):
        ax.plot(ep_x, test_loss_pc[:, c], label=f'Class {c}')
      ax.axvline(best_ep, color='black', linestyle='--', alpha=0.6, label=f'Best ep {best_ep}')
      ax.set_xlabel('Epoch')
      ax.set_ylabel('Loss')
      ax.set_title('Test Loss per Class (epochs)')
      ax.legend(fontsize=7)
    else:
      _label_unavailable(ax, 'test_loss_per_class unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'test_loss_per_class error\n{exc}')

  # [0,1] test accuracy per class at best epoch
  ax = axes[0, 1]
  try:
    if test_acc_pc is not None and len(test_acc_pc) > best_ep:
      acc_at_best = to_numpy(test_acc_pc[best_ep])
      ax.bar(CLASS_LABELS[:len(acc_at_best)], acc_at_best, color='#4C72B0')
      ax.set_ylim(0, 1)
      ax.set_xlabel('Class')
      ax.set_ylabel('Accuracy')
      ax.set_title(f'Test Accuracy per Class @ epoch {best_ep}')
    else:
      _label_unavailable(ax, 'test_accuracy_per_class unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'test_accuracy_per_class error\n{exc}')

  # [0,2] train losses across epochs
  ax = axes[0, 2]
  try:
    if train_losses is not None and len(train_losses) > 0:
      ax.plot(epochs_x, train_losses, color='tab:red', label='Train loss')
      ax.axvline(best_ep, color='black', linestyle='--', alpha=0.6, label=f'Best ep {best_ep}')
      ax.set_xlabel('Epoch')
      ax.set_ylabel('Loss')
      ax.set_title('Train Loss (epochs)')
      ax.legend()
    else:
      _label_unavailable(ax, 'train_losses unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'train_losses error\n{exc}')

  # [1,0] train confusion matrix at best epoch
  ax = axes[1, 0]
  try:
    cm_obj = train_cms.get(str(best_ep))
    cm_arr = extract_cm_array(cm_obj)
    if cm_arr is not None:
      _plot_cm_heatmap(cm_arr, CLASS_LABELS, out_dir=out_dir, normalized=False,
                       ax=ax, title_suffix=f'\nTrain @ epoch {best_ep}')
    else:
      _label_unavailable(ax, 'train confusion matrix unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'train CM error\n{exc}')

  # [1,1] test confusion matrix at best epoch
  ax = axes[1, 1]
  try:
    cm_obj = test_cms[best_ep] if len(test_cms) > best_ep else None
    cm_arr = extract_cm_array(cm_obj)
    if cm_arr is not None:
      _plot_cm_heatmap(cm_arr, CLASS_LABELS, out_dir=out_dir, normalized=False,
                       ax=ax, title_suffix=f'\nTest @ epoch {best_ep}')
    else:
      _label_unavailable(ax, 'test confusion matrix unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'test CM error\n{exc}')

  # [1,2] RMSE + Pearson r (twin y-axis)
  ax = axes[1, 2]
  try:
    rmse_x = np.arange(len(test_rmse)) if test_rmse is not None else epochs_x
    has_rmse    = test_rmse is not None and len(test_rmse) > 0
    has_pearson = test_pearson_r is not None and len(test_pearson_r) > 0
    if has_rmse:
      ax.plot(rmse_x, test_rmse, color='tab:blue', label='RMSE')
      ax.set_xlabel('Epoch')
      ax.set_ylabel('RMSE', color='tab:blue')
      ax.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = None
    if has_pearson:
      ax2 = ax.twinx()
      ax2.plot(rmse_x[:len(test_pearson_r)], test_pearson_r, color='tab:orange', label='Pearson r')
      ax2.set_ylabel('Pearson r', color='tab:orange')
      ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax.set_title('RMSE & Pearson r (epochs)')
    if not has_rmse and not has_pearson:
      _label_unavailable(ax, 'RMSE/Pearson unavailable')
    else:
      lines = ax.get_lines() + (ax2.get_lines() if ax2 else [])
      ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
  except Exception as exc:
    _label_unavailable(ax, f'RMSE/Pearson error\n{exc}')

  fig.tight_layout()
  _save_fig(fig, os.path.join(out_dir, 'fold_summary.png'))

  best_l1      = float(test_l1[best_ep])        if test_l1      is not None and len(test_l1)      > best_ep else float('nan')
  best_acc     = float(test_acc[best_ep])       if test_acc     is not None and len(test_acc)     > best_ep else float('nan')
  best_rmse    = float(test_rmse[best_ep])      if test_rmse    is not None and len(test_rmse)    > best_ep else float('nan')
  best_pearson = float(test_pearson_r[best_ep]) if test_pearson_r is not None and len(test_pearson_r) > best_ep else float('nan')

  print(f"Fold {fold_key} | best_epoch={best_ep} | test_L1={best_l1:.4f} | "
        f"test_acc={best_acc:.4f} | RMSE={best_rmse:.4f} | Pearson_r={best_pearson:.4f}")

  return {'best_epoch': best_ep, 'test_l1': best_l1, 'test_acc': best_acc,
          'rmse': best_rmse, 'pearson_r': best_pearson}


# ---------------------------------------------------------------------------
# Aggregated arrays
# ---------------------------------------------------------------------------

def build_aggregated(fold_results):
  """
  Stack per-epoch metric arrays across all folds, truncated to min epoch count.

  Args:
    fold_results: Dict mapping fold_key → fold_data.

  Returns:
    Dict containing:
      fold_keys    — sorted list of fold keys
      min_ep       — common epoch length used for truncation
      test_l1      — (F, E) or None
      train_loss   — (F, E) or None
      train_acc    — (F, E) or None
      test_acc     — (F, E) or None
      test_rmse    — (F, E) or None
      test_pearson — (F, E) or None
      best_epochs  — (F,) argmin of test_l1 per fold
  """
  keys = sorted(fold_results.keys())

  min_ep = None
  for fk in keys:
    tl = safe_get(fold_results[fk], 'train_val', 'train_losses')
    if tl is not None:
      n = len(tl)
      min_ep = n if min_ep is None else min(min_ep, n)
  if min_ep is None:
    min_ep = 0

  def _stack(extractor):
    arrays = []
    for fk in keys:
      arr = extractor(fold_results[fk])
      if arr is None:
        return None
      arr = np.array(arr).flatten()[:min_ep]
      arrays.append(arr)
    return np.stack(arrays) if arrays else None

  test_l1    = _stack(lambda fd: to_numpy(safe_get(fd, 'train_val', 'test_as_eval', 'test_list_l1_error')))
  train_loss = _stack(lambda fd: to_numpy(safe_get(fd, 'train_val', 'train_losses')))
  train_acc  = _stack(lambda fd: to_numpy(safe_get(fd, 'train_val', 'list_train_accuracy')))
  test_acc   = _stack(lambda fd: to_numpy(safe_get(fd, 'train_val', 'test_as_eval', 'list_test_accuracy')))
  test_rmse  = _stack(lambda fd: to_numpy(safe_get(fd, 'train_val', 'test_as_eval', 'test_list_rmse')))

  pearson_arrays = []
  for fk in keys:
    raw = safe_get(fold_results[fk], 'train_val', 'test_as_eval', 'list_test_pearson_correlation')
    if raw is None:
      pearson_arrays = None
      break
    pearson_arrays.append(np.array([extract_pearson_r(v) for v in raw])[:min_ep])
  test_pearson = np.stack(pearson_arrays) if pearson_arrays else None

  def _best_ep(fk):
    l1 = to_numpy(safe_get(fold_results[fk], 'train_val', 'test_as_eval', 'test_list_l1_error'))
    return int(np.argmin(l1)) if l1 is not None and len(l1) > 0 else 0

  best_epochs = np.array([_best_ep(fk) for fk in keys])

  return {
    'fold_keys':    keys,
    'min_ep':       min_ep,
    'test_l1':      test_l1,
    'train_loss':   train_loss,
    'train_acc':    train_acc,
    'test_acc':     test_acc,
    'test_rmse':    test_rmse,
    'test_pearson': test_pearson,
    'best_epochs':  best_epochs,
  }


# ---------------------------------------------------------------------------
# Aggregated figure
# ---------------------------------------------------------------------------

def plot_aggregated(agg, out_dir):
  """
  Generate a 4-panel aggregated figure showing mean ± std across folds.

  Panels:
    [0,0] Mean test L1 error ± std with best-epoch annotation
    [0,1] Mean train loss ± std with best-epoch marker
    [1,0] Mean train vs test accuracy ± std
    [1,1] Mean RMSE ± std (left) and mean Pearson r ± std (right, twin axis)

  Args:
    agg:     Dict returned by build_aggregated.
    out_dir: Directory to save aggregated_summary.png.
  """
  os.makedirs(out_dir, exist_ok=True)
  min_ep   = agg['min_ep']
  epochs_x = np.arange(min_ep)
  mean_best = int(np.round(np.mean(agg['best_epochs']))) if agg['best_epochs'] is not None and min_ep > 0 else None

  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  fig.suptitle('Aggregated Cross-Validation Summary', fontsize=13, fontweight='bold')

  # [0,0] mean test L1 error
  ax = axes[0, 0]
  y_axis_plot_range = [0, 2]  # Adjust as needed based on expected L1 error range
  try:
    if agg['test_l1'] is not None and min_ep > 0:
      mean = agg['test_l1'].mean(0)
      std  = agg['test_l1'].std(0)
      plot_with_std(ax, epochs_x, mean, std, 'Epoch', 'Test L1 Error',
                    'Mean Test L1 Error ± std', color='tab:blue',y_lim=y_axis_plot_range)
      if mean_best is not None and mean_best < min_ep:
        best_val = mean[mean_best]
        ax.axvline(mean_best, color='red', linestyle='--', alpha=0.7)
        ax.annotate(
          f'ep={mean_best}\nL1={best_val:.3f}',
          xy=(mean_best, best_val),
          xytext=(mean_best + max(1, min_ep * 0.05), best_val + std[mean_best] * 0.5),
          fontsize=8, color='red',
          arrowprops=dict(arrowstyle='->', color='red'),
        )
    else:
      _label_unavailable(ax, 'test_l1 unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'test_l1 error\n{exc}')

  # [0,1] mean train loss
  ax = axes[0, 1]
  try:
    if agg['train_loss'] is not None and min_ep > 0:
      mean = agg['train_loss'].mean(0)
      std  = agg['train_loss'].std(0)
      plot_with_std(ax, epochs_x, mean, std, 'Epoch', 'Train Loss',
                    'Mean Train Loss ± std', color='tab:red', y_lim=y_axis_plot_range)
      if mean_best is not None and mean_best < min_ep:
        ax.axvline(mean_best, color='black', linestyle='--', alpha=0.5,
                   label=f'Mean best ep {mean_best}')
        ax.legend(fontsize=8)
    else:
      _label_unavailable(ax, 'train_loss unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'train_loss error\n{exc}')

  # [1,0] mean train vs test accuracy
  ax = axes[1, 0]
  try:
    has_tr = agg['train_acc'] is not None and min_ep > 0
    has_te = agg['test_acc']  is not None and min_ep > 0
    if has_tr:
      plot_with_std(ax, epochs_x, agg['train_acc'].mean(0), agg['train_acc'].std(0),
                    'Epoch', 'Accuracy', 'Mean Train vs Test Accuracy ± std',
                    color='tab:green', legend_label_mean='Train acc', legend_label_std='Train std')
    if has_te:
      plot_with_std(ax, epochs_x, agg['test_acc'].mean(0), agg['test_acc'].std(0),
                    'Epoch', 'Accuracy', 'Mean Train vs Test Accuracy ± std',
                    color='tab:orange', legend_label_mean='Test acc', legend_label_std='Test std')
    if not has_tr and not has_te:
      _label_unavailable(ax, 'accuracy unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'accuracy error\n{exc}')

  # [1,1] mean RMSE + Pearson r (twin axis)
  ax = axes[1, 1]
  try:
    has_rmse    = agg['test_rmse']    is not None and min_ep > 0
    has_pearson = agg['test_pearson'] is not None and min_ep > 0
    if has_rmse:
      plot_with_std(ax, epochs_x, agg['test_rmse'].mean(0), agg['test_rmse'].std(0),
                    'Epoch', 'RMSE', 'Mean RMSE & Pearson r ± std',
                    color='tab:blue', legend_label_mean='RMSE', legend_label_std='RMSE std')
    if has_pearson:
      ax2 = ax.twinx()
      pr_mean = np.nanmean(agg['test_pearson'], axis=0)
      pr_std  = np.nanstd(agg['test_pearson'],  axis=0)
      ax2.plot(epochs_x, pr_mean, color='tab:orange', label='Pearson r')
      ax2.fill_between(epochs_x, pr_mean - pr_std, pr_mean + pr_std,
                       color='tab:orange', alpha=0.2, label='Pearson std')
      ax2.set_ylabel('Pearson r', color='tab:orange')
      ax2.tick_params(axis='y', labelcolor='tab:orange')
    if not has_rmse and not has_pearson:
      _label_unavailable(ax, 'RMSE/Pearson unavailable')
  except Exception as exc:
    _label_unavailable(ax, f'RMSE/Pearson error\n{exc}')

  fig.tight_layout()
  out_path = os.path.join(out_dir, 'aggregated_summary.png')
  _save_fig(fig, out_path)
  print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Aggregated confusion matrix
# ---------------------------------------------------------------------------

def plot_aggregated_cm(fold_results, best_epochs, out_dir):
  """
  Sum raw confusion matrices across folds at each fold's best epoch, then save
  both raw-count and row-normalized versions.

  Args:
    fold_results: Dict mapping fold_key → fold_data.
    best_epochs:  Array of best-epoch indices per fold (same order as sorted keys).
    out_dir:      Directory to save the aggregated CM PNGs.
  """
  os.makedirs(out_dir, exist_ok=True)
  keys = sorted(fold_results.keys())
  aggregated = None

  for fk, best_ep in zip(keys, best_epochs):
    te       = safe_get(fold_results[fk], 'train_val', 'test_as_eval', default={})
    test_cms = safe_get(te, 'test_confusion_matricies', default=[])
    cm_obj   = test_cms[best_ep] if len(test_cms) > best_ep else None
    cm_arr   = extract_cm_array(cm_obj)
    if cm_arr is None:
      print(f"  Warning: no test CM for fold {fk} at epoch {best_ep}")
      continue
    aggregated = cm_arr if aggregated is None else aggregated + cm_arr

  if aggregated is None:
    print("  Warning: could not build aggregated confusion matrix — no data.")
    return

  suffix = '\nAggregated (sum across folds)'
  _plot_cm_heatmap(aggregated, CLASS_LABELS, out_dir=out_dir, normalized=False,
                   title_suffix=suffix, filename_suffix='_aggregated')
  _plot_cm_heatmap(aggregated, CLASS_LABELS, out_dir=out_dir, normalized=True,
                   title_suffix=suffix, filename_suffix='_aggregated')
  print(f"Saved aggregated confusion matrices to {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
  """
  CLI entry point: load pkl, generate per-fold figures and aggregated plots.
  """
  parser = argparse.ArgumentParser(description='Analyze k-fold cross-validation log pkl.')
  parser.add_argument('--pkl_path', type=str, help='Path to k_fold_results.pkl')
  args = parser.parse_args()

  pkl_path = Path(args.pkl_path).resolve()
  if not pkl_path.exists():
    sys.exit(f"Error: file not found: {pkl_path}")

  with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

  results = safe_get(data, 'results')
  if results is None:
    sys.exit("Error: pkl has no 'results' key.")

  fold_keys = sorted(k for k in results if 'final' not in k)
  if not fold_keys:
    sys.exit("Error: no fold keys found (excluding 'final').")

  out_root = pkl_path.parent / 'plot_loss'
  os.makedirs(out_root, exist_ok=True)
  print(f"\nOutput root: {out_root}")
  print(f"Found {len(fold_keys)} fold(s): {fold_keys}\n")

  summaries = {}
  for fk in fold_keys:
    fold_out = out_root / fk
    summaries[fk] = plot_fold(fk, results[fk], str(fold_out))

  agg = build_aggregated({fk: results[fk] for fk in fold_keys})
  plot_aggregated(agg, str(out_root))
  plot_aggregated_cm({fk: results[fk] for fk in fold_keys}, agg['best_epochs'], str(out_root))

  print("\n=== Per-Fold Summary ===")
  for fk, s in summaries.items():
    print(f"  {fk}: best_ep={s['best_epoch']} | L1={s['test_l1']:.4f} | "
          f"acc={s['test_acc']:.4f} | RMSE={s['rmse']:.4f} | r={s['pearson_r']:.4f}")


if __name__ == '__main__':
  main()

import argparse
import os
import numpy as np
import pandas as pd
import torch
import umap
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm
import custom.tools as tools

WARN_COLOR = '\033[93m'
WARN_RESET = '\033[0m'
PLOT_COLORS = ('#1f77b4', '#d62728')


def pool_features(features: torch.Tensor, pooling: str) -> torch.Tensor:
  """
  Pool a single sample's feature tensor into one or more UMAP points.

  Args:
    features: Raw feature tensor loaded from safetensors. Shape: (B, T, S, S, C)
      where B=num_chunks, T=frames, S=spatial side, C=embedding dim.
    pooling:  One of 'mean', 'max', 'min' (collapse all of B,T,S,S into a single
      point) or 'none' (collapse only the spatial dims S,S, keeping B*T points).

  Returns:
    Pooled tensor. Shape: (C,) for mean/max/min, or (B*T, C) for 'none'.
  """
  if pooling == 'mean':
    return features.mean(dim=(0, 1, 2, 3))
  elif pooling == 'max':
    return features.amax(dim=(0, 1, 2, 3))
  elif pooling == 'min':
    return features.amin(dim=(0, 1, 2, 3))
  elif pooling == 'none':
    b, t, _, _, c = features.shape
    return features.mean(dim=(2, 3)).reshape(b * t, c)
  raise ValueError(f'Unknown pooling: {pooling}')


def load_sample_points(feature_path: str, subject_name: str, sample_name: str, pooling: str):
  """
  Load and pool the features for a single CSV row.

  Args:
    feature_path: Root folder containing {subject_name}/{sample_name}.safetensors.
    subject_name: Subject folder name (CSV 'subject_name' column).
    sample_name:  Sample file stem (CSV 'sample_name' column).
    pooling:      Pooling strategy, see pool_features.

  Returns:
    Pooled tensor (see pool_features), or None if the safetensors file is missing.
  """
  path = os.path.join(feature_path, subject_name, sample_name + '.safetensors')
  if not os.path.isfile(path):
    print(f'{WARN_COLOR}Warning: missing safetensors file, skipping sample: {path}{WARN_RESET}')
    return None
  dict_data = tools.load_dict_data(path)
  return pool_features(dict_data['features'], pooling)


def build_points_and_labels(df_subject: pd.DataFrame, feature_path: str, pooling: str):
  """
  Build the (points, labels) arrays for one subject's label-filtered samples.

  Args:
    df_subject:   CSV rows for a single subject, already filtered to the 2 selected labels.
    feature_path: Root folder containing {subject_name}/{sample_name}.safetensors.
    pooling:      Pooling strategy, see pool_features.

  Returns:
    Tuple (embeddings, labels): embeddings has shape (N, C), labels has shape (N,)
    holding the class_id each point was pooled from.
  """
  points, labels = [], []
  for _, row in df_subject.iterrows():
    pooled = load_sample_points(feature_path, row['subject_name'], row['sample_name'], pooling)
    if pooled is None:
      continue
    pooled = np.atleast_2d(pooled.numpy())
    points.append(pooled)
    labels.extend([row['class_id']] * pooled.shape[0])
  if not points:
    return np.empty((0,)), np.empty((0,))
  return np.vstack(points), np.array(labels)


def compute_inter_class_distance(embeddings: np.ndarray, labels: np.ndarray, label_a, label_b) -> float:
  """
  Mean pairwise distance between the two label groups.

  Args:
    embeddings: Points to measure. Shape: (N, D).
    labels:     Class id per point. Shape: (N,).
    label_a:    First class id.
    label_b:    Second class id.

  Returns:
    Mean distance between every point of label_a and every point of label_b.
  """
  return cdist(embeddings[labels == label_a], embeddings[labels == label_b]).mean()


def compute_metrics(embeddings_2d: np.ndarray, labels: np.ndarray, label_a, label_b) -> dict:
  """
  Compute cluster compactness/separation metrics on a 2D UMAP embedding.

  Args:
    embeddings_2d: UMAP output. Shape: (N, 2).
    labels:        Class id per point. Shape: (N,).
    label_a:       First class id.
    label_b:       Second class id.

  Returns:
    Dict with per-label intra-class distance, inter-class distance, silhouette
    score and Davies-Bouldin index.
  """
  intra = tools.pairwise_distance(embeddings_2d, labels)
  metrics = {f'intra-class dist (label {k})': float(v) for k, v in intra.items()}
  metrics['inter-class dist'] = float(compute_inter_class_distance(embeddings_2d, labels, label_a, label_b))
  metrics['silhouette score'] = float(tools.get_silhouette_score(embeddings_2d, labels))
  metrics['davies-bouldin index'] = float(tools.get_davies_bouldin_index(embeddings_2d, labels))
  return metrics


def has_enough_data(labels: np.ndarray, label_a, label_b) -> bool:
  """
  Check whether there is enough data to run UMAP and the compactness metrics.

  Args:
    labels:  Class id per point. Shape: (N,).
    label_a: First class id.
    label_b: Second class id.

  Returns:
    True if both labels are present with at least 2 points each and at least
    4 points in total.
  """
  count_a = np.sum(labels == label_a)
  count_b = np.sum(labels == label_b)
  return count_a >= 2 and count_b >= 2 and (count_a + count_b) >= 4


def run_umap(embeddings: np.ndarray) -> np.ndarray:
  """
  Fit UMAP on the pooled high-dimensional features and reduce to 2D.

  Args:
    embeddings: Pooled feature vectors. Shape: (N, C).

  Returns:
    2D embedding. Shape: (N, 2).
  """
  n_neighbors = max(1, min(15, embeddings.shape[0] - 1))
  reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
  return reducer.fit_transform(embeddings)


def plot_umap_with_metrics(embeddings_2d, labels, metrics, label_a, label_b, subject_id, output_path):
  """
  Plot the UMAP scatter next to a text panel of compactness metrics and save it.

  Args:
    embeddings_2d: UMAP output. Shape: (N, 2).
    labels:        Class id per point. Shape: (N,).
    metrics:       Dict of metric name -> value, see compute_metrics.
    label_a:       First class id.
    label_b:       Second class id.
    subject_id:    Subject id, used in the plot title.
    output_path:   Path (including .png) to save the figure to.
  """
  fig, (ax_scatter, ax_text) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [3, 1]})

  for color, label in zip(PLOT_COLORS, (label_a, label_b)):
    idx = labels == label
    ax_scatter.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=str(label), color=color, s=30)
  ax_scatter.legend(title='class_id')
  ax_scatter.set_xlabel('UMAP Dimension 1')
  ax_scatter.set_ylabel('UMAP Dimension 2')
  ax_scatter.set_title(f'Subject {subject_id} — UMAP (labels {label_a} vs {label_b})')

  ax_text.axis('off')
  text_lines = [f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v}' for k, v in metrics.items()]
  ax_text.text(0.0, 1.0, '\n'.join(text_lines), transform=ax_text.transAxes, va='top', ha='left', fontsize=10, family='monospace')

  plt.tight_layout()
  plt.savefig(output_path)
  plt.close(fig)


def process_subject(df_labels: pd.DataFrame, subject_id, feature_path: str, umap_folder: str, label_a, label_b, pooling: str):
  """
  Run the full UMAP + metrics + plot pipeline for a single subject.

  Args:
    df_labels:    CSV rows already filtered to the 2 selected labels (all subjects).
    subject_id:   Subject id to process.
    feature_path: Root folder containing {subject_name}/{sample_name}.safetensors.
    umap_folder:  Output folder for the saved plot.
    label_a:      First class id.
    label_b:      Second class id.
    pooling:      Pooling strategy, see pool_features.

  Returns:
    Dict of the computed metrics prefixed with a 'subject_id' key, or None if the
    subject was skipped (missing one label or too few points).
  """
  df_subject = df_labels[df_labels['subject_id'] == subject_id]
  embeddings, labels = build_points_and_labels(df_subject, feature_path, pooling)

  if not has_enough_data(labels, label_a, label_b):
    print(f'{WARN_COLOR}Warning: subject {subject_id} skipped — missing one of labels '
          f'{label_a}/{label_b} or too few points (n={len(labels)}){WARN_RESET}')
    return None

  embeddings_2d = run_umap(embeddings)
  metrics = compute_metrics(embeddings_2d, labels, label_a, label_b)
  print(f'Subject {subject_id}: {metrics}')

  output_path = os.path.join(umap_folder, f'umap_subject{subject_id}_labels{label_a}_{label_b}.png')
  plot_umap_with_metrics(embeddings_2d, labels, metrics, label_a, label_b, subject_id, output_path)

  return {'subject_id': subject_id, **metrics}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Compute and plot UMAP embeddings of backbone features restricted to two class labels.')
  parser.add_argument('--feature_path', type=str, required=True, help='Root folder containing {subject_name}/{sample_name}.safetensors')
  parser.add_argument('--csv', type=str, required=True, help='Path to samples.csv (tab-separated)')
  parser.add_argument('--labels', type=int, nargs=2, required=True, help='The two class_id values to compare')
  parser.add_argument('--umap_folder', type=str, required=True, help='Output folder for saved plots')
  parser.add_argument('--subject_id', type=int, default=-1, help='Subject id to plot. -1 = loop over every subject present')
  parser.add_argument('--pooling', type=str, choices=['mean', 'max', 'min', 'none'], default='mean', help='Pooling strategy applied to [B,T,S,S,C] features')
  args = parser.parse_args()

  df = pd.read_csv(args.csv, sep='\t', dtype={'sample_name': str, 'subject_name': str})
  df_labels = df[df['class_id'].isin(args.labels)]
  label_a, label_b = args.labels

  subject_ids = [args.subject_id] if args.subject_id != -1 else sorted(df_labels['subject_id'].unique())

  os.makedirs(args.umap_folder, exist_ok=True)
  summary_rows = []
  for subject_id in tqdm(subject_ids, desc='Processing subjects'):
    row = process_subject(df_labels, subject_id, args.feature_path, args.umap_folder, label_a, label_b, args.pooling)
    if row is not None:
      summary_rows.append(row)

  if summary_rows:
    summary_path = os.path.join(args.umap_folder, 'umap_summary.csv')
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f'Saved summary of {len(summary_rows)} subjects to {summary_path}')
  else:
    print(f'{WARN_COLOR}Warning: no subjects produced metrics, summary CSV not written{WARN_RESET}')

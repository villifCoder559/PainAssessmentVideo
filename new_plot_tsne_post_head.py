"""Scatter plot of 2-D reduced embeddings (t-SNE/UMAP), used by cross_space_logs.py."""
import os
import time

import matplotlib.pyplot as plt
import numpy as np


def plot_reducted_embeddings(reduced_embeddings, labels, output_folder, save_plot=True,v_max=None,v_min=None, title="t-SNE Visualization", group_by="pain",cmap='viridis', output_path=None,ax = None, reduction_name="t-SNE", show_colorbar=True):
  if ax is None:
    fig,ax = plt.subplots(figsize=(12, 8))
  else:
    fig = ax.figure

  unique_labels = np.unique(labels)
  if group_by == 'labels': # continuous colormap
    normalized_unique_labels = (unique_labels - np.min(unique_labels)) / (np.max(unique_labels) - np.min(unique_labels))
    colormap = plt.get_cmap(cmap)
    colors = colormap(normalized_unique_labels)
    colors = [colormap(label) for label in normalized_unique_labels]
  else: # categorical colormap
    if cmap == 'nipy_spectral':
      colormap = plt.get_cmap(cmap, len(unique_labels))
    else:
      colormap = plt.get_cmap(cmap)
    colors = [colormap(i) for i in range(len(unique_labels))]
    
  for color, label in zip(colors, unique_labels):
    idx = labels == label
    ax.scatter(reduced_embeddings[idx, 0],
               reduced_embeddings[idx, 1],
               label=str(label),
               marker='o',
               color=color,
               s=30)
  if len(unique_labels) <= 20:  # Only show legend if not too many labels
    ax.legend(title=group_by)
  # if colormap is continuous, add colorbar
  if group_by == 'labels' and show_colorbar:
    norm = plt.Normalize(vmin=v_min if v_min is not None else np.min(labels), vmax=v_max if v_max is not None else np.max(labels))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(group_by)
  ax.set_xlabel(f"{reduction_name} Dimension 1")
  ax.set_ylabel(f"{reduction_name} Dimension 2")
  ax.set_title(title)

  
  if not save_plot:
    return
  plt.tight_layout()
  os.makedirs(output_folder, exist_ok=True)
  if output_path is None:
    timestamp = int(time.time())
    output_path = os.path.join(output_folder, f'{reduction_name.lower()}_plot_{group_by}_{timestamp}.png')
  plt.savefig(output_path)
  plt.close()
  print(f"{reduction_name} plot saved to {output_path}")

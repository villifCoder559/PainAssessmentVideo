import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from openTSNE import TSNE
import pandas as pd
from pathlib import Path

def load_data(pkl_file):
  with open(pkl_file, 'rb') as f:
    data = pickle.load(f)
  return data

def plot_tsne(embeddings, labels, output_folder, title="t-SNE Visualization", group_by="labels",cmap='viridis'):
  tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
  reduced_embeddings = tsne.fit(embeddings)

  plt.figure(figsize=(12, 8))

  scatter = plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    c=labels, cmap=cmap, s=10
  )
  plt.colorbar(scatter, label=group_by.capitalize())

  plt.title(title)
  plt.xlabel("t-SNE Dimension 1")
  plt.ylabel("t-SNE Dimension 2")

  os.makedirs(output_folder, exist_ok=True)
  plt.savefig(os.path.join(output_folder, f"{group_by}_{title}_tsne_plot.png"))
  plt.close()

# NOTE: pkl_file extracted from log_cross_attention_from_model.py

def main():
  parser = argparse.ArgumentParser(description="Plot t-SNE from embeddings in a pickle file from log_cross_attention_from_model.py")
  parser.add_argument("--pkl_file", type=str, required=True, help="Path to the pickle file containing embeddings and labels.")
  parser.add_argument("--output_folder", type=str, required=True, help="Folder to save the t-SNE plot.")
  parser.add_argument("--title", type=str, default="t-SNE Visualization", help="Title of the plot.")
  parser.add_argument("--group_by", type=str, choices=["labels", "subjects"], default="labels", help="Group visualization by labels or subjects.")
  parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for the plot.")
  args = parser.parse_args()

  data = load_data(args.pkl_file)

  if not isinstance(data, dict) or 'embeddings' not in data or 'labels' not in data:
    raise ValueError("Pickle file must contain a dictionary with 'embeddings' and 'labels' keys.")

  log_path_folder = os.path.join(*Path(data['pkl_file']).parts[:-1], f'log_tsne_plots')
  os.makedirs(log_path_folder, exist_ok=True)
  
  # Determine labels based on grouping choice
  if args.group_by == "labels":
    labels = np.array(data['labels'])
  elif args.group_by == "subjects":
    sample_ids = np.array(data['sample_ids'])
    df = pd.read_csv(data['csv_path'], sep='\t', dtype={'sample_name': str})
    id_to_subject = dict(zip(df['sample_id'], df['subject_id']))
    labels = np.array([id_to_subject[sample_id] for sample_id in sample_ids])
  else:
    raise ValueError("group_by must be either 'labels' or 'subjects'.")
  
  embeddings = np.array(data['embeddings'])
  plot_tsne(embeddings = embeddings,
            labels = labels,
            output_folder = log_path_folder,
            title = args.title,
            group_by = args.group_by,
            cmap = args.cmap)

if __name__ == "__main__":
  main()
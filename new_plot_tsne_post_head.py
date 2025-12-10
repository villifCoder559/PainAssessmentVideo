import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from openTSNE import TSNE
import pandas as pd
from pathlib import Path
import custom.helper as helper  
from custom.dataset import customDataset
import time


def load_data(pkl_file):
  with open(pkl_file, 'rb') as f:
    data = pickle.load(f)
  return data

def get_custom_ds(data):
  feats_path = data['config_model']['model_advanced_params']['features_folder_saving_path']

  config_dict = os.path.join(feats_path, "config_dict.pkl")
  if 'raid' in config_dict:
    config_dict = config_dict.replace('raid','dune')
  
  with open(config_dict, 'rb') as f:
    config = pickle.load(f)
  # Retrieve list frames for the sample id
  if 'caer' in feats_path.lower():
    config['video_extension'] = '.avi'
  custom_ds = customDataset(**config)
  return custom_ds

def plot_tsne(embeddings, labels, output_folder, v_max=None,v_min=None, title="t-SNE Visualization", group_by="pain",cmap='viridis', output_path=None):
  tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
  reduced_embeddings = tsne.fit(embeddings)

  fig,ax = plt.subplots(figsize=(12, 8))
  unique_labels = np.unique(labels)
  if group_by == 'labels': # continuous colormap
    normalized_unique_labels = (unique_labels - np.min(unique_labels)) / (np.max(unique_labels) - np.min(unique_labels))
    colormap = plt.get_cmap(cmap)
    colors = colormap(normalized_unique_labels)
    colors = [colormap(label) for label in normalized_unique_labels]
  else: # categorical colormap
    colormap = plt.get_cmap(cmap)
    colors = [colormap(i) for i in range(len(unique_labels))]
    
  for color, label in zip(colors, unique_labels):
    idx = labels == label
    ax.scatter(reduced_embeddings[idx, 0],
               reduced_embeddings[idx, 1],
               label=str(label), 
               color=color,
               s=30)
  ax.legend(title=group_by)
  plt.title(title)
  plt.xlabel("t-SNE Dimension 1")
  plt.ylabel("t-SNE Dimension 2")
  
  plt.tight_layout()
  os.makedirs(output_folder, exist_ok=True)
  if output_path is None:
    timestamp = int(time.time())
    output_path = os.path.join(output_folder, f'tsne_plot_{group_by}_{timestamp}.png')
  plt.savefig(output_path)
  plt.close()
  print(f"t-SNE plot saved to {output_path}")
  dict_tsne = {
    'embeddings_2d': reduced_embeddings,
    'labels': labels
  }
  return dict_tsne

# NOTE: pkl_file extracted from log_cross_attention_from_model.py

def prepare_data_for_tsne(pkl_file, group_by, cmap):
  if isinstance(pkl_file, dict):
    data = pkl_file
  elif isinstance(pkl_file, str) and os.path.isfile(pkl_file):  
    data = load_data(pkl_file)
  else:
    raise ValueError("pkl_file must be a path to a pickle file or a dictionary.")

  if not isinstance(data, dict) or 'embeddings' not in data['video_embeddings'] or 'labels' not in data['video_embeddings']:
    raise ValueError("Pickle file must contain a dictionary with 'embeddings' and 'labels' keys.")

  # Determine labels based on grouping choice
  df = pd.read_csv(data['csv_path'], sep='\t', dtype={'sample_name': str})
  if group_by == "labels":
    labels = np.array(data['video_embeddings']['labels'])
  elif group_by == "subjects":
    sample_ids = np.array(data['video_embeddings']['sample_ids'])
    id_to_subject = dict(zip(df['sample_id'], df['subject_id']))
    labels = np.array([id_to_subject[sample_id] for sample_id in sample_ids])
    nr_subjects = len(set(df['subject_id']))
    if cmap == 'jet':  # only change if default
      if nr_subjects <= 10:
        cmap = 'tab10'  # better for categorical data
      elif nr_subjects <= 20:
        cmap = 'tab20'  
      else:
        cmap = 'tab20c'  # good for many categories
  else:
    raise ValueError("group_by must be either 'labels' or 'subjects'.")
  # Create a reference for readability
  embeddings_data = data['video_embeddings']['embeddings']
  embeddings = [
      desc.cpu().numpy() 
      for batch_list in embeddings_data 
      for desc in batch_list
  ]
  embeddings = np.array(embeddings)
  list_sample_ids = np.array(data['video_embeddings']['sample_ids'])
  
  # Filter out augmented samples if any
  custom_ds = get_custom_ds(data) # set the helper.step_shift
  valid_indices = [i for i, sample_id in enumerate(list_sample_ids) if sample_id <= helper.step_shift]
  del custom_ds
  labels = labels[valid_indices]
  embeddings = embeddings[valid_indices]
  list_sample_ids = list_sample_ids[valid_indices]
  
  # Adjust title
  # group_by = 'pain' if group_by.lower() == 'labels' else 'subject_id'
  title = f"t-SNE grouped by {group_by} - tot samples: {len(list_sample_ids)} - subject_ids: {set(df['subject_id'])}"
  
  return embeddings, labels, title, cmap

def run_tsne_and_plot(pkl_file, group_by, cmap, log_path_folder, png_output_name=None):
  embeddings, labels, title, cmap = prepare_data_for_tsne(pkl_file, group_by, cmap)  
  dict_tsne = plot_tsne(embeddings = embeddings,
            labels = labels,
            output_folder = log_path_folder,
            title = title,
            output_path=png_output_name,
            group_by = group_by,
            cmap = cmap)
  dict_tsne['title'] = title
  dict_tsne['group_by'] = group_by
  return dict_tsne


def main():
  parser = argparse.ArgumentParser(description="Plot t-SNE from embeddings in a pickle file from log_cross_attention_from_model.py")
  parser.add_argument("--pkl_file", type=str, required=True, help="Path to the pickle file containing embeddings and labels.")
  parser.add_argument("--group_by", type=str, choices=["labels", "subjects"], default="labels", help="Group visualization by labels or subjects.")
  parser.add_argument("--cmap", type=str, default="jet", help="Colormap for the plot.") # jet, tab20, viridis, etc
  args = parser.parse_args()
  
  log_path_folder = str(Path(args.pkl_file).parent)
  run_tsne_and_plot(args.pkl_file, args.group_by, args.cmap, log_path_folder)
  

if __name__ == "__main__":
  main()
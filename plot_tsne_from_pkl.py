import argparse
import os
import tqdm
import pandas as pd
import torch
import numpy as np
from openTSNE import TSNE
import time
from pathlib import Path
import pickle
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def get_data_from_pkl(xattn_path):
  with open(xattn_path, 'rb') as f:
    data = pickle.load(f)
  if data['config_model']['config']['concatenate_quadrants']:
    raise NotImplementedError("Concatenate quadrants not implemented")
  run_id = Path(xattn_path).parts[-4].split('_')[0]
  return data, run_id

def apply_tsne(embeddings, perplexity=30, random_state=42, n_jobs=2):
  tsne = TSNE(
      n_components=2,
      perplexity=perplexity,
      random_state=random_state,
      n_jobs=n_jobs
  )
  tsne_results = tsne.fit(embeddings)
  return tsne_results

def plot_tsne(tsne_results, labels, legends, save_path, title):
  unique_labels = np.unique(labels)
  if len(unique_labels) != len(legends):
    raise ValueError("Number of unique labels does not match number of legends")
  colors = plt.colormaps['Set1'].resampled(len(unique_labels))
  plt.figure(figsize=(10, 8))
  for i, (label, legend) in enumerate(zip(unique_labels, legends)):
    mask = labels == label
    plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], label=legend, alpha=0.7, color=colors(i))
  plt.title(title)
  plt.tight_layout()
  plt.legend()
  plt.savefig(save_path)
  plt.close()

def generate_labels_from_video_lengths(video_original_lengths, threshold):
  lst_labels = []
  for len_video in video_original_lengths:
    lst_labels.extend([0 if i < threshold else 1 for i in range(len_video)])
  return np.array(lst_labels)
  
def generate_video_from_folder(folder_path, output_video_path, fps=2):
  images_path = [img for img in os.listdir(folder_path) if img.endswith(".png")]
  images_path.sort()
  frame = cv2.imread(os.path.join(folder_path, images_path[0]))
  height, width, _ = frame.shape
  video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
  for image_path in tqdm.tqdm(images_path,desc="Generating video from images"):
    frame = cv2.imread(os.path.join(folder_path, image_path))
    if frame.shape[0] != height or frame.shape[1] != width:
      frame = cv2.resize(frame, (width, height))
    video.write(frame)
  video.release()
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Plot t-SNE from PKL embeddings')
  parser.add_argument('--pkl_path', type=str, required=True, help='Path to the PKL file containing embeddings')
  parser.add_argument('--plot_per_sample_tsne', type=int, default=1, help='Whether to plot per-sample t-SNE (1) or not (0)')
  parser.add_argument('--plot_per_subject_tsne', type=int, default=1, help='Whether to plot per-subject t-SNE (1) or not (0)')
  parser.add_argument('--plot_overall_tsne', type=int, default=1, help='Whether to plot overall t-SNE (1) or not (0)')
  dict_args = vars(parser.parse_args())
  
  data, run_id = get_data_from_pkl(dict_args['pkl_path'])
  uid = int(time.time())
  dict_video_embeddings = data['video_embeddings'] # keys: embeddings(List of tensors batch), predictions(List), labels(List), sample_ids(List),enable(Bool)
  for k in dict_video_embeddings.keys():
    if k not in ['enable']:
      if k in ['embeddings']:
        dict_video_embeddings[k] = np.concatenate(dict_video_embeddings[k], axis=0)
      else:
        dict_video_embeddings[k] = np.array(dict_video_embeddings[k])
  # embeddings: (nr_samples, embedding_dim), predictions: (nr_samples,), labels: (nr_samples,), sample_ids: (nr_samples,)
  
  csv_path = data['csv_path']
  df = pd.read_csv(csv_path, sep='\t', dtype={'sample_name': str})
  
  target_class = 4
  FPS = 25
  fpc = 16 # frames per chunk
  threshold = 4 # emb[:threshold] no pain, emb[threshold:] pain
  
  list_sample_ids = df[df['class_id'] == target_class]['sample_id'].tolist()
  logs_global_folder = os.path.join(*Path(dict_args['pkl_path']).parts[:-1], f"tsne_logs_{uid}",f'tsne_plots_{"split_chunks" if data["config_logging"]["split_chunks"] else ""}')
  os.makedirs(logs_global_folder, exist_ok=True)
  
  # Split chunks case -> one video has multiple embeddings
  if data['config_logging']['split_chunks']:
    
    ###### CASE: Plot per sample #######
    if dict_args['plot_per_sample_tsne']:
      folder_path = os.path.join(logs_global_folder, "per_sample_tsne")
      os.makedirs(folder_path, exist_ok=True)
      for sample_id in tqdm.tqdm(list_sample_ids, desc="Processing samples for t-SNE"):
        sample_name = df[df['sample_id'] == sample_id]['sample_name'].values[0]
        title = f't-SNE for Sample {sample_name} ({sample_id}) - threshold {threshold}'
        mask = dict_video_embeddings['sample_ids'] == sample_id
        embeddings_sample = dict_video_embeddings['embeddings'][mask]
        legends = [f'No Pain (0,{threshold*fpc/FPS:.2f}) sec', f'Pain ({threshold*fpc/FPS:.2f}, {(embeddings_sample.shape[0])*fpc/FPS:.2f}) sec']
        labels = np.array([0 if i < threshold else 1 for i in range(embeddings_sample.shape[0])])
        tsne_results = apply_tsne(embeddings_sample)
        sil_score = silhouette_score(tsne_results, labels)
        save_path = os.path.join(folder_path, f'{sample_name}_id_{sample_id}_score_{sil_score:.2f}.png')
        plot_tsne(tsne_results, labels, legends, save_path, title)
      generate_video_from_folder(folder_path, os.path.join(folder_path, 'per_sample_tsne_video.mp4'), fps=4)
      
    ###### CASE: Plot per subject #######
    if dict_args['plot_per_subject_tsne']:
      folder_path = os.path.join(logs_global_folder, "per_subject_tsne")
      os.makedirs(folder_path, exist_ok=True)
      list_subject_ids = df[df['class_id'] == target_class]['subject_id'].unique().tolist()
      for subject_id in tqdm.tqdm(list_subject_ids, desc="Processing subjects for t-SNE"):
        subject_sample_ids = df[(df['class_id'] == target_class) & (df['subject_id'] == subject_id)]['sample_id'].tolist()
        title = f't-SNE for Subject ID {subject_id} - threshold {threshold} - total samples {len(subject_sample_ids)} - pain class {target_class}'
        mask = np.isin(dict_video_embeddings['sample_ids'], subject_sample_ids)
        embeddings_subject = dict_video_embeddings['embeddings'][mask]
        video_original_lengths = np.unique(dict_video_embeddings['sample_ids'][mask], return_counts=True)[1]
        print(f'Unique videos lengths considered for subject {subject_id} t-SNE: {np.unique(video_original_lengths, return_counts=True)}')
        labels = generate_labels_from_video_lengths(video_original_lengths, threshold)
        legends = [f'No Pain (0,{threshold*fpc/FPS:.2f}) sec', f'Pain ({threshold*fpc/FPS:.2f}, {(max(video_original_lengths))*fpc/FPS:.2f}) sec']
        tsne_results = apply_tsne(embeddings_subject)
        sil_score = silhouette_score(tsne_results, labels)
        subject_name = df[df['subject_id'] == subject_id]['subject_name'].values[0]
        save_path = os.path.join(folder_path, f'{subject_name}_id_{subject_id}_score_{sil_score:.2f}.png')
        plot_tsne(tsne_results, labels, legends, save_path, title)
      generate_video_from_folder(folder_path, os.path.join(folder_path, 'per_subject_tsne_video.mp4'), fps=2)
        
        
    ###### CASE: Plot overall #######
    if dict_args['plot_overall_tsne']:
      print("Plotting overall t-SNE...")
      folder_path = os.path.join(logs_global_folder, "overall_tsne")
      os.makedirs(folder_path, exist_ok=True)
      mask = np.isin(dict_video_embeddings['sample_ids'], list_sample_ids)
      embeddings_overall = dict_video_embeddings['embeddings'][mask]
      title = f'Overall t-SNE - threshold {threshold} - total samples {embeddings_overall.shape[0]} - pain class {target_class}'
      video_original_lengths = np.unique(dict_video_embeddings['sample_ids'][mask], return_counts=True)[1]
      print(f'Unique videos lengths considered for overall t-SNE: {np.unique(video_original_lengths,return_counts=True)}')
      labels = generate_labels_from_video_lengths(video_original_lengths, threshold)
      legends = [f'No Pain (0,{threshold*fpc/FPS:.2f}) sec', f'Pain ({threshold*fpc/FPS:.2f}, {(max(video_original_lengths))*fpc/FPS:.2f}) sec']
      tsne_results = apply_tsne(embeddings_overall, n_jobs=4)
      sil_score = silhouette_score(tsne_results, labels)
      save_path = os.path.join(folder_path, f'overall_tsne_score_{sil_score:.2f}.png')
      plot_tsne(tsne_results, labels, legends, save_path, title)
      
  else:
    raise NotImplementedError("Non-split chunks case not implemented for video embeddings.")
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import re
from openTSNE import TSNE
import umap
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, cdist
import custom.helper as helper
from custom.dataset import customDataset
import time
import tqdm
import shutil

def load_data(pkl_file):
  if isinstance(pkl_file, dict):
    data = pkl_file
  elif isinstance(pkl_file, str) and os.path.isfile(pkl_file):  
    with open(pkl_file, 'rb') as f:
      data = pickle.load(f)
  else:
    raise ValueError("pkl_file must be a path to a pickle file or a dictionary.")
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


def plot_reducted_embeddings(reduced_embeddings, labels, output_folder, save_plot=True,v_max=None,v_min=None, title="t-SNE Visualization", group_by="pain",cmap='viridis', output_path=None,ax = None, reduction_name="t-SNE"):
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
              #  edgecolors=color,
              #  facecolors='none',
               s=30)
  if len(unique_labels) <= 20:  # Only show legend if not too many labels
    ax.legend(title=group_by)
  # if colormap is continuous, add colorbar
  if group_by == 'labels':
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

# NOTE: pkl_file extracted from log_cross_attention_from_model.py
def determine_labels(data, group_by, return_subject_ids=False, top_n_subjects=None):
  df = pd.read_csv(data['csv_path'], sep='\t', dtype={'sample_name': str})
  if group_by == "labels":
    labels = np.array(data['video_embeddings']['labels'])
  elif group_by == "subjects":
    sample_ids = np.array(data['video_embeddings']['sample_ids'])
    id_to_subject = dict(zip(df['sample_id'], df['subject_id']))
    labels = np.array([id_to_subject[sample_id] for sample_id in sample_ids])
  else:
    raise ValueError("group_by must be either 'labels' or 'subjects'.")
  if return_subject_ids:
    return labels, np.array(df['subject_id'])
  return labels

def filter_top_n_subjects(embeddings, labels, n_subjects):
  """
  Filter embeddings and labels to keep only the top N subjects with the most samples.
  
  Args:
    embeddings: numpy array of embeddings
    labels: numpy array of subject labels
    n_subjects: number of top subjects to keep
    
  Returns:
    filtered_embeddings, filtered_labels, selected_subjects
  """
  if n_subjects is None or n_subjects <= 0:
    return embeddings, labels, np.unique(labels)
  
  # Count samples per subject
  unique_subjects, counts = np.unique(labels, return_counts=True)
  
  # Get the indices of the top N subjects by count
  top_indices = np.argsort(counts)[-n_subjects:]
  top_subjects = unique_subjects[top_indices]
  
  # Create mask for samples from top subjects
  mask = np.isin(labels, top_subjects)
  
  filtered_embeddings = embeddings[mask]
  filtered_labels = labels[mask]
  
  return filtered_embeddings, filtered_labels, top_subjects

def compute_intra_inter_distances(reduced_embeddings, labels):
  """
  Compute per-label intra-class and inter-class mean pairwise Euclidean distances.

  Args:
    reduced_embeddings: 2D reduced embeddings. Shape: (N, 2).
    labels:             Array of labels for each point. Shape: (N,).

  Returns:
    List of dicts with keys: label, intra_distance, inter_distance, sample_count.
  """
  unique_labels = np.unique(labels)
  results = []
  for label in unique_labels:
    mask = labels == label
    cluster_points = reduced_embeddings[mask]
    other_points = reduced_embeddings[~mask]
    # Intra: mean pairwise distance within this label
    intra = float(np.mean(pdist(cluster_points))) if len(cluster_points) > 1 else 0.0
    # Inter: mean distance from this label's points to all other points
    inter = float(np.mean(cdist(cluster_points, other_points))) if len(other_points) > 0 else 0.0
    results.append({
      'label': label,
      'intra_distance': intra,
      'inter_distance': inter,
      'sample_count': len(cluster_points)
    })
  return results

def save_distance_csv(results, pkl_file_path, csv_name, group_by, reduction_method, model_name="unknown_model"):
  """
  Save intra/inter-class distance results to a CSV file.

  Args:
    results:          List of dicts from compute_intra_inter_distances.
    pkl_file_path:    Path to the source pkl file (used to find the output folder).
    csv_name:         Name of the source CSV file (e.g. train_cleaned.csv).
    group_by:         Label type used ('labels' or 'subjects').
    reduction_method: Reduction method name ('tsne' or 'umap').
    model_name:       Name of the model (used in the output filename).
  """
  try:
    if not isinstance(pkl_file_path, str):
      return

    abs_pkl_path = os.path.abspath(pkl_file_path)
    parts = Path(abs_pkl_path).parts

    # Find k{i}_cross_val_sub_{j} ancestor folder
    gallery_root = None
    for i, part in enumerate(parts):
      if re.fullmatch(r'k\d+_cross_val_sub_\d+', part):
        gallery_root = Path(*parts[:i+1])
        break

    if gallery_root is None:
      print("Could not find k*_cross_val_sub_* folder to save distance CSV.")
      return

    # Extract main folder id from path (numeric prefix like 1773225230603)
    main_folder_id = ""
    for part in parts:
      match = re.match(r'^(\d{10,})_', part)
      if match:
        main_folder_id = match.group(1)

    # Create output directory
    output_dir = gallery_root / "reduction_intra_inter_distance"
    os.makedirs(output_dir, exist_ok=True)

    # Build DataFrame with per-label rows
    df = pd.DataFrame(results)
    df['csv_file'] = csv_name
    df['main_folder_id'] = main_folder_id
    df['reduction_method'] = reduction_method

    # Append overall mean row
    mean_row = pd.DataFrame([{
      'label': 'OVERALL_MEAN',
      'intra_distance': df['intra_distance'].mean(),
      'inter_distance': df['inter_distance'].mean(),
      'sample_count': df['sample_count'].sum(),
      'csv_file': csv_name,
      'main_folder_id': main_folder_id,
      'reduction_method': reduction_method
    }])
    df = pd.concat([df, mean_row], ignore_index=True)

    # Save
    output_path = output_dir / f"{model_name}_{csv_name}_distances_{group_by}_{reduction_method}.csv"
    df.to_csv(output_path, index=False, sep='\t')
    print(f"Distance CSV saved to {output_path}")

  except Exception as e:
    print(f"Error saving distance CSV: {e}")

def set_cmap(data, group_by, cmap):
  df = pd.read_csv(data['csv_path'], sep='\t', dtype={'sample_name': str})
  if group_by == "subjects":
    nr_subjects = len(set(df['subject_id']))
    if cmap is None or cmap == 'jet':  # only change if default
      if nr_subjects <= 10:
        cmap = 'tab10'
      elif nr_subjects <= 20:
        cmap = 'tab20'  
      else:
        cmap = 'nipy_spectral'  # good for many categories
  elif group_by == "labels":
    if cmap is None:
      cmap = 'jet'  # good for continuous data
  
  return cmap

def get_valid_indices(data, list_sample_ids):
  custom_ds = get_custom_ds(data) # set the helper.step_shift
  valid_indices = [i for i, sample_id in enumerate(list_sample_ids) if sample_id <= helper.step_shift]
  del custom_ds
  return valid_indices

def compute_valid_tsne_embeddings(data, return_valid_indices=False):
  list_sample_ids = np.array(data['video_embeddings']['sample_ids'])
  valid_indices = get_valid_indices(data, list_sample_ids)
  
  embeddings_data = data['video_embeddings']['embeddings']
  embeddings = [
      desc.cpu().numpy() 
      for batch_list in embeddings_data 
      for desc in batch_list
  ]
  embeddings = np.array(embeddings)
  
  # Filter out augmented samples if any
  embeddings = embeddings[valid_indices]
  
  # Compute t-SNE
  tsne = TSNE(n_components=2, random_state=42, n_jobs=1)
  tsne_embeddings = tsne.fit(embeddings)
  if return_valid_indices:
    return tsne_embeddings, valid_indices, list_sample_ids[valid_indices]
  return tsne_embeddings

def compute_valid_umap_embeddings(data, return_valid_indices=False):
  list_sample_ids = np.array(data['video_embeddings']['sample_ids'])
  valid_indices = get_valid_indices(data, list_sample_ids)
  
  embeddings_data = data['video_embeddings']['embeddings']
  embeddings = [
      desc.cpu().numpy() 
      for batch_list in embeddings_data 
      for desc in batch_list
  ]
  embeddings = np.array(embeddings)
  
  # Filter out augmented samples if any
  embeddings = embeddings[valid_indices]
  
  # Compute UMAP
  reducer = umap.UMAP(random_state=42)
  umap_embeddings = reducer.fit_transform(embeddings)
  if return_valid_indices:
    return umap_embeddings, valid_indices, list_sample_ids[valid_indices]
  return umap_embeddings

def create_gallery_link(pkl_file_path, plot_path):
  try:
    if not isinstance(pkl_file_path, str):
      return

    # Find the folder matching pattern k{i}_cross_val_sub_{j}
    abs_pkl_path = os.path.abspath(pkl_file_path)
    parts = Path(abs_pkl_path).parts
    
    gallery_root = None
    for i, part in enumerate(parts):
      if re.fullmatch(r'k\d+_cross_val_sub_\d+', part):
        gallery_root = Path(*parts[:i+1])
        break
          # break
            
    if gallery_root:
      gallery_dir = gallery_root / "tsne_plots_gallery"
      os.makedirs(gallery_dir, exist_ok=True)
      
      plot_name = os.path.basename(plot_path)
      link_path = gallery_dir / plot_name
      # create a copy of the plot in the gallery directory
      
      if os.path.lexists(link_path):
        os.remove(link_path)
      
      shutil.copy2(plot_path, link_path)
      print(f"Copied plot to gallery: {link_path} -> {plot_path}")
      # os.symlink(plot_path, link_path)
      # print(f"Created gallery link: {link_path} -> {plot_path}")
    else:
      print("Could not find pth_idx_folder pattern (k..._cross_val_sub_...) to create gallery link.")

  except Exception as e:
    print(f"Error creating gallery link: {e}")

def run_tsne_and_plot(pkl_file, group_by, cmap,
                      png_output_name=None,reduced_embeddings=None,save_plot=True,ax=None,
                      add_title_info="",log_path_folder=None, reduction_method="tsne", top_n_subjects=None,
                      compute_distances=False):
  
  # print(f'Log path folder: {log_path_folder}')
  if not isinstance(pkl_file, dict):
    log_path_folder = str(Path(pkl_file).parent)
    data = load_data(pkl_file)
  else:
    data = pkl_file
  
  if reduction_method == 'umap':
    reduction_title = "UMAP"
  else:
    reduction_title = "t-SNE"

  # Extract model name
  model_name = "unknown_model"
  if 'model_pth_path' in data:
      model_name = Path(data['model_pth_path']).stem
  
  if reduced_embeddings is None:
    if reduction_method == 'umap':
      reduced_embeddings, valid_indices, _ = compute_valid_umap_embeddings(data, return_valid_indices=True)
    else:
      reduced_embeddings, valid_indices, _ = compute_valid_tsne_embeddings(data, return_valid_indices=True)
  else:
    reduced_embeddings, valid_indices = reduced_embeddings
  csv_name = os.path.basename(data['csv_path'])
  
  # Handle group_by "all"
  if group_by == "all":
    fig,axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # Plot Subjects (Left)
    labels_subj, subject_ids_subj = determine_labels(data, "subjects", return_subject_ids=True)
    labels_subj = labels_subj[valid_indices]
    subject_ids_subj = subject_ids_subj[valid_indices]
    
    # Filter by top N subjects if specified
    if top_n_subjects and top_n_subjects > 0:
      reduced_emb_subj, labels_subj, top_subjects = filter_top_n_subjects(reduced_embeddings, labels_subj, top_n_subjects)
    else:
      reduced_emb_subj = reduced_embeddings
      top_subjects = np.unique(labels_subj)
    
    cmap_subj = set_cmap(data, "subjects", cmap)
    
    title_subj = f"{model_name} - {csv_name} - {reduction_title} (Subjects)\ntot samples: {len(labels_subj)} - subject_ids: {set(top_subjects) if len(set(top_subjects)) <= 20 else f'tot ({len(set(top_subjects))})'}" + add_title_info
    
    plot_reducted_embeddings(reduced_embeddings=reduced_emb_subj,
                              labels=labels_subj,
                              output_folder=log_path_folder,
                              title=title_subj,
                              group_by="subjects",
                              save_plot=False,
                              cmap=cmap_subj,
                              ax=axes[0],
                              reduction_name=reduction_title)
    
    # Plot Labels (Right)
    labels_lbl, subject_ids_lbl = determine_labels(data, "labels", return_subject_ids=True)
    labels_lbl = labels_lbl[valid_indices]
    
    # Filter by top N subjects if specified (for consistent filtering)
    if top_n_subjects and top_n_subjects > 0:
      reduced_emb_lbl, labels_lbl, _ = filter_top_n_subjects(reduced_embeddings, labels_lbl, top_n_subjects)
    else:
      reduced_emb_lbl = reduced_embeddings
    
    cmap_lbl = set_cmap(data, "labels", cmap)
    
    title_lbl = f"{model_name} - {csv_name} - {reduction_title} (Labels)"
    
    plot_reducted_embeddings(reduced_embeddings=reduced_emb_lbl,
                              labels=labels_lbl,
                              output_folder=log_path_folder,
                              title=title_lbl,
                              group_by="labels",
                              save_plot=False,
                              cmap=cmap_lbl,
                              ax=axes[1],
                              reduction_name=reduction_title)
    
    plt.tight_layout()
    if save_plot:
      os.makedirs(log_path_folder, exist_ok=True)
      timestamp = int(time.time())
      final_output_name = png_output_name
      if final_output_name is None:
        final_output_name = os.path.join(log_path_folder, f'{model_name}_{csv_name}_{reduction_title.lower()}_plot_all_{timestamp}.png')
      else:
        # Ensure model name is in filename if user provided one, or prepending to it
        if model_name not in os.path.basename(final_output_name):
            dirname = os.path.dirname(final_output_name)
            basename = os.path.basename(final_output_name)
            final_output_name = os.path.join(dirname, f"{model_name}_{csv_name}_{basename}")

      plt.savefig(final_output_name)
      plt.close()
      print(f"{reduction_title} plot (all) saved to {final_output_name}")
      if not isinstance(pkl_file, dict):
        create_gallery_link(pkl_file, final_output_name)

    # Compute and save distances for both group types
    if compute_distances and not isinstance(pkl_file, dict):
      results_subj = compute_intra_inter_distances(reduced_emb_subj, labels_subj)
      save_distance_csv(results_subj, pkl_file, csv_name, "subjects", reduction_method, model_name=model_name)
      results_lbl = compute_intra_inter_distances(reduced_emb_lbl, labels_lbl)
      save_distance_csv(results_lbl, pkl_file, csv_name, "labels", reduction_method, model_name=model_name)
    return


  # Standard cases (single plot)
  labels, subject_ids = determine_labels(data, group_by, return_subject_ids=True)
  labels = labels[valid_indices]
  subject_ids = subject_ids[valid_indices]
  
  # Filter by top N subjects if specified and grouping by subjects
  reduced_emb = reduced_embeddings
  if group_by == "subjects" and top_n_subjects and top_n_subjects > 0:
    reduced_emb, labels, top_subjects = filter_top_n_subjects(reduced_embeddings, labels, top_n_subjects)
    subject_ids = np.array([s for s in top_subjects])
  else:
    top_subjects = np.unique(labels) if group_by == "subjects" else None
  
  cmap = set_cmap(data, group_by, cmap)
  if add_title_info:
    title = f"{model_name} - {csv_name} - {reduction_title} tot samples: {len(labels)} - subject_ids: {set(subject_ids) if group_by == 'labels' or (top_subjects is None or len(top_subjects) <= 20) else f'tot ({len(set(subject_ids))})'}" + add_title_info
  else:
    title = f"{model_name} - {csv_name} - {reduction_title} grouped by {group_by} - tot samples: {len(labels)} - subject_ids: {set(subject_ids) if group_by == 'labels' or (top_subjects is None or len(top_subjects) <= 20) else f'tot ({len(set(subject_ids))})'}" + add_title_info
  
  # Prepend model name to output filename if not present
  if png_output_name is None:
    timestamp = int(time.time())
    png_output_name = os.path.join(log_path_folder, f'{model_name}_{csv_name}_{reduction_title.lower()}_plot_{group_by}_{timestamp}.png')
  else:
    if model_name not in os.path.basename(png_output_name):
      dirname = os.path.dirname(png_output_name)
      basename = os.path.basename(png_output_name)
      png_output_name = os.path.join(dirname, f"{model_name}_{csv_name}_{basename}")

  plot_reducted_embeddings(reduced_embeddings = reduced_emb,
            labels = labels,
            output_folder = log_path_folder,
            title = title,
            output_path=png_output_name,
            group_by = group_by,
            save_plot=save_plot,
            cmap = cmap,
            ax = ax,
            reduction_name = reduction_title)

  if save_plot and not isinstance(pkl_file, dict):
    create_gallery_link(pkl_file, png_output_name)

  # Compute and save distances for the specified group_by
  if compute_distances and not isinstance(pkl_file, dict):
    results = compute_intra_inter_distances(reduced_emb, labels)
    save_distance_csv(results, pkl_file, csv_name, group_by, reduction_method, model_name=model_name)

def main():
  parser = argparse.ArgumentParser(description="Plot t-SNE or UMAP from embeddings in a pickle file from log_cross_attention_from_model.py")
  parser.add_argument("--pkl_file", type=str, required=True, help="Path to the pickle file containing embeddings and labels.")
  parser.add_argument("--group_by", type=str, choices=["labels", "subjects", "all"], default="labels", help="Group visualization by labels or subjects.")
  parser.add_argument("--cmap", type=str, default=None, help="Colormap for the plot.") # jet, tab20, viridis, etc
  parser.add_argument("--reduction_method", type=str, choices=["tsne", "umap"], default="umap", help="Dimensionality reduction method to use.")
  parser.add_argument("--top_n_subjects", type=int, default=None, help="Number of subjects with the most samples to plot. If not specified, all subjects are plotted.")
  parser.add_argument("--compute_distances", action="store_true", help="Compute and save intra/inter-class distances to CSV.")
  args = parser.parse_args()
  list_pkl_files = []
  if os.path.isdir(args.pkl_file):
    root_folder = args.pkl_file
    print(f"Searching for pkl files in folder: {root_folder}")
    for root, dirs, files in os.walk(root_folder):
      for file in files:
        if file.endswith(".pkl") and 'xattn_embeds_' in file:
          list_pkl_files.append(os.path.join(root, file))
    print(f"Found {len(list_pkl_files)} pkl files.")
    
    for pkl_path in tqdm.tqdm(list_pkl_files, desc="Processing pkl files..."):
      if args.group_by == "all":
        run_tsne_and_plot(pkl_path, "all", None, reduction_method=args.reduction_method, top_n_subjects=args.top_n_subjects, compute_distances=args.compute_distances)
      else:
        # Call once if group_by is specific, or twice if previously "all" but handled differently
        # Since "all" is now handled inside run_tsne_and_plot as a single plot:
        run_tsne_and_plot(pkl_path, args.group_by, args.cmap, reduction_method=args.reduction_method, top_n_subjects=args.top_n_subjects, compute_distances=args.compute_distances)

  else:
    if args.group_by == "all":
      run_tsne_and_plot(args.pkl_file, "all", args.cmap, reduction_method=args.reduction_method, top_n_subjects=args.top_n_subjects, compute_distances=args.compute_distances)
    else:
      run_tsne_and_plot(args.pkl_file, args.group_by, args.cmap, reduction_method=args.reduction_method, top_n_subjects=args.top_n_subjects, compute_distances=args.compute_distances)

if __name__ == "__main__":
  main()
import argparse
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import silhouette_score

import custom.tsne_cuda_tools as tsne_tools
import custom.helper as helper
import safetensors.torch
import matplotlib.pyplot as plt

def save_config_file(dict_args: Dict[str, Any]) -> None:
  """
  Save the t-SNE configuration as both txt and pickle files.
  """
  save_dir = os.path.dirname(dict_args['path_save_plot'])
  os.makedirs(save_dir, exist_ok=True)
  txt_path = os.path.join(dict_args['path_save_plot'] + '_config.txt')
  pkl_path = os.path.join(dict_args['path_save_plot'] + '_config.pkl')

  with open(txt_path, 'w') as f_txt:
    f_txt.write("Arguments used for t-SNE:\n")
    for key, value in dict_args.items():
      f_txt.write(f"{key}: {value}\n")

  with open(pkl_path, 'wb') as f_pkl:
    pickle.dump(dict_args, f_pkl)

# =============================================================================
# t-SNE computation
# =============================================================================

def compute_tsne(dict_args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  Compute t-SNE embeddings for either 'whole' or 'aggregated' feature sets.
  Returns:
    new_dict_data: dict with updated 'features' as torch.Tensor of shape (...,2)
    config: t-SNE configuration details
  """
  feat_type = dict_args['type_feat']

  if feat_type == 'whole':
    df = pd.read_csv(dict_args['csv_path'], sep='\t')
    # Apply filters
    for key in ['sample_id', 'subject_id', 'pain_levels']:
      val = dict_args.get(key)
      if val is not None:
        print(f"Filtering {key} with values: {val}")
        col = 'list_labels' if key == 'pain_levels' else key
        df = df[df[col].isin(val)]

    X_tsne, config, raw_data, orig_dict = tsne_tools.compute_tsne(
      X=None,
      df_filtered=df,
      apply_pca_before_tsne=dict_args['apply_pca_before_tsne'],
      pca_nr_batch=dict_args['nr_batch'],
      perplexity=dict_args['perplexity'],
      random_seed=dict_args['random_seed'],
      n_jobs=dict_args['n_jobs'],
      mode=dict_args['mode'],
      root_folder_feats=dict_args['feat_path'],
    )
    unique_list_sample_id = np.unique(raw_data['list_sample_id'])
    
    for sample_id in unique_list_sample_id:
      mask = orig_dict['list_sample_id'] == sample_id
      shape = orig_dict['features'][mask].shape[1:-1]
      if 'real_shape' not in raw_data:
        raw_data['real_shape'] = []
      raw_data['real_shape'].append(shape)
      
    reshaped_reduced_feats = _reshape_features(raw_data, X_tsne)
    orig_dict['features'] = reshaped_reduced_feats
    return orig_dict,config

  elif feat_type == 'aggregated':
    data = tsne_tools.load_dict_data(dict_args['feat_path'])
    # Filtering
    if not dict_args['no_filter_tsne']:
      filters = [
        ('list_labels', dict_args['pain_levels']),
        ('list_sample_id', dict_args['sample_id']),
        ('list_subject_id', dict_args['subject_id']),
      ]
      for key, val in filters:
        if val is not None:
          data = tsne_tools.filter_dict_features_per_key(data, val, key=key)

    feats = data['features'].reshape(-1, data['features'].shape[-1])
    
    # Flat the dict
    flatten_data = {}
    tsne_tools.flat_dict_data_(data, flatten_data)
    unique_list_sample_id = np.unique(data['list_sample_id'])
    
    # Save original shape
    for sample_id in unique_list_sample_id:
      mask = data['list_sample_id'] == sample_id
      shape = data['features'][mask].shape[1:-1]
      if 'real_shape' not in data:
        flatten_data['real_shape'] = []
      flatten_data['real_shape'].append(shape)
    
    # Transform into np
    for k,v in flatten_data.items():
      if k != 'real_shape':
        flatten_data[k] = np.concatenate(v, axis=0)
    
    X_tsne, config, _, _ = tsne_tools.compute_tsne(
      feats,
      pca_nr_batch=dict_args['nr_batch'],
      perplexity=dict_args['perplexity'],
      apply_pca_before_tsne=dict_args['apply_pca_before_tsne'],
      random_seed=dict_args['random_seed'],
      n_jobs=dict_args['n_jobs'],
      mode=dict_args['mode'],
    )
    reshaped_reduced_feats = _reshape_features(flatten_data, X_tsne)
    data['features'] = reshaped_reduced_feats
    return data, config
      

  else:
    raise ValueError(f"Unknown type_feat: {feat_type}")


def _reshape_features(raw_data: Dict[str, Any], X_tsne: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  Reconstruct the tsne embeddings to match the original feature shape.
  Returns new_data with torch.Tensor features and the raw config dict.
  """
  unique_ids = np.unique(raw_data['list_sample_id'])
  reshaped_list = []
  for count,uid in enumerate(unique_ids):
    mask = raw_data['list_sample_id'] == uid
    nr_samples = mask.sum()
    if 'real_shape' in raw_data:
      feat_shape = raw_data['real_shape'][count]
    else:
      raise ValueError("No 'real_shape' found in raw_data. Ensure the data is preprocessed correctly.")
    # feat_shape = raw_data['features'][mask].shape[1:-1]
    # count = np.prod(feat_shape)
    # mask = np.repeat(mask,np.prod(feat_shape))
    flat = X_tsne[mask]
    reshaped = flat.reshape(-1, *feat_shape, 2)
    reshaped_list.append(reshaped)

  all_feats = np.concatenate(reshaped_list, axis=0)
  # new_data = raw_data.copy()
  # new_data['features'] = torch.tensor(all_feats)
  return all_feats

# =============================================================================
# Plotting utilities
# =============================================================================


def get_labels_from_dict(reduced_feature: np.ndarray, threshold: int, key: str) -> np.ndarray:
  # unique_samples = np.unique(reduced_feature)
  list_labels = []
  # for sample in unique_samples:
  mask = np.prod(reduced_feature.shape[:-1])
  seq_len = mask.sum()
  binary_labels = np.arange(seq_len, dtype=int)  # Ensure threshold is not larger than the sequence length
  adapted_threshold = (threshold*seq_len) / reduced_feature.shape[0]
  binary_labels = (binary_labels >= adapted_threshold).astype(int)
    
  list_labels.append(binary_labels)
  return np.concatenate(list_labels, axis=0)
    
def _generic_plot_binary(
  dict_data: Dict[str, Any], # original dict with features reduced
  dict_args: Dict[str, Any],
  group_key: str,
  tsne_label: str,
  fig_size=None,
) -> List[Dict[str, Any]]:
  """
  Generic function to plot binary t-SNE results per grouping key.
  group_key: one of 'list_sample_id', 'list_subject_id', 'list_labels'
  tsne_label: string identifier for titles and file names.
  """
  # df_meta = pd.read_csv(dict_args['csv_path'], sep='\t')
  df_meta = dict_args['df_csv_path']
  entries = []

  groups = np.unique(dict_data[group_key])
  for gid in tqdm.tqdm(groups, desc=f'Plotting {tsne_label}s'):
    mask = dict_data[group_key] == gid
    X = dict_data['features'][mask]
    # labels = get_labels_from_dict(np.array(dict_data['features'][mask]), dict_args['thresh_binary'], group_key)
    labels = np.zeros(shape=(X.shape[:-1]),dtype=bool)
    labels[dict_args['thresh_binary']:] = True
    # total = np.prod(X.shape[:-1]
    # threshold = dict_args['thresh_binary'] * np.prod(X.shape[1:-1])
    # labels = (np.arange(total) >= threshold).astype(int)
    sil = 0.0
    if dict_args['thresh_binary'] < labels.shape[0]:
      sil = silhouette_score(
        X=X.reshape(-1, 2),
        labels=labels.reshape(-1),
      )
    #   continue  # Skip if threshold is larger than the number of samples
    # name = df_meta[df_meta[group_key.replace('list_', '')] == gid]
    # name = name.iloc[0]['sample_name'] if 'sample_name' in name else str(gid)
    if group_key == 'list_sample_id':
      name = df_meta[df_meta['sample_id'] == gid]['sample_name'].values[0]
      title = f"{name}_{tsne_label}_{X.shape[:-1]}_sil_{sil:.4f}_threshold_{dict_args['thresh_binary']}"
    else:  
      name = f"perClass_{gid}" if group_key == 'list_labels' else str(gid)
      title = f"subjectid_{dict_args['subject_id']}_PA{dict_args['pain_levels']}_{name}_{tsne_label}_{X.shape[:-1]}_sil_{sil:.4f}_threshold_{dict_args['thresh_binary']}"
    
    list_additional_desc_legend = []
    for idx in range(2):
      list_additional_desc_legend.append(f' ({idx*dict_args["stride_dataset"]/dict_args["fps_dataset"]*dict_args["thresh_binary"]}, {(idx+1)*dict_args["stride_dataset"]/25*dict_args["thresh_binary"]}) sec')
      
    if dict_args['plot_results']:
      img = tsne_tools.plot_tsne(
        X_tsne=X.reshape(-1, 2),
        labels=labels.reshape(-1),
        ax=dict_args['ax'] if 'ax' in dict_args else None,
        saving_path=dict_args['path_save_plot'] if not dict_args['root_video_path'] else None,
        title=title,
        legend_label='Movement',
        list_additiona_desc_legend=list_additional_desc_legend,
        tot_labels=2,
        cmap='bwr',
        fig_size=fig_size
      )

    entry = {
      f'{tsne_label}_id': gid,
      'name': name,
      'silhouette_score': sil,
      'threshold': dict_args['thresh_binary'],
      **dict_args,
    }
    entries.append(entry)
    save_config_file(dict_args)
  if img is not None:
    return img
  else:
    return entries

def plot_per_sample_binary(dict_data: Dict[str, Any], dict_args: Dict[str, Any], fig_size=None) -> List[Dict[str, Any]]:
  """Plot binary t-SNE results per sample."""
  if dict_args['filter_plot_sample']:
    dict_data = tsne_tools.filter_dict_features_per_key(
      dict_data,
      dict_args['filter_plot_sample'],
      key='list_sample_id'
    )
  return _generic_plot_binary(dict_data, dict_args, 'list_sample_id', 'sample', fig_size)

def plot_per_subject_binary(dict_data: Dict[str, Any], dict_args: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Plot binary t-SNE results per subject."""
  if dict_args['filter_plot_subject']:
    dict_data = tsne_tools.filter_dict_features_per_key(
      dict_data,
      dict_args['filter_plot_subject'],
      key='list_subject_id'
    )
  return _generic_plot_binary(dict_data, dict_args, 'list_subject_id', 'subject')

def plot_per_class_binary(dict_data: Dict[str, Any], dict_args: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Plot binary t-SNE results per class (pain level)."""
  if dict_args['filter_plot_class']:
    dict_data = tsne_tools.filter_dict_features_per_key(
      dict_data,
      dict_args['filter_plot_class'],
      key='list_labels'
    )
  return _generic_plot_binary(dict_data, dict_args, 'list_labels', 'class')

# =============================================================================
# CLI interface
# =============================================================================

def get_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="t-SNE analysis script")
  parser.add_argument("--feat_path", type=str, required=True, help="Path to the feature file.")
  parser.add_argument("--perplexity", type=int, default=30, help="Perplexity for t-SNE.")
  parser.add_argument("--apply_pca_before_tsne", type=int, default=100, help="Number of PCA components to apply before t-SNE. 0 means no PCA.")
  parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility. Works only for CPU implementation.")
  parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for CPU implementation.")
  parser.add_argument("--csv_path", type=str, default=None, help="Path to the CSV file.")
  parser.add_argument("--nr_batch", type=int, default=0, help="Number of batch to divide the dataset. If 0, it will use the entire dataset.")
  parser.add_argument("--all_tsne_feat_path", type=str, default=None, help="Path to the all t-SNE features file. If provided, it will use this file instead of computing t-SNE if level_tsne is all.")
  parser.add_argument("--type_feat", type=str, default='aggregated', choices=['aggregated', 'whole'], help='Type of features to use. aggregated (saved in one dict), whole(save in more dicts)')
  parser.add_argument("--plot_per_class_subject", action='store_true', help='Plot per class subject t-SNE results. If not set, it will plot per sample or per subject.')
  parser.add_argument("--plot_per_class_all", action='store_true', help='Plot per class t-SNE results. If not set, it will plot per sample or per subject.')
  parser.add_argument("--plot_per_sample_binary", action='store_true', help='Plot per sample binary t-SNE results.')
  parser.add_argument("--no_filter_tsne", action='store_true', help='Do not filter the data before computing t-SNE. If set, it will use all the data.')
  parser.add_argument("--filter_plot_sample", type=int, nargs='*', default=None, help="Sample ID to filter the data for plotting per sample.")
  parser.add_argument("--filter_plot_subject", type=int, nargs='*', default=None, help="Subject ID to filter the data for plotting per subject.")
  parser.add_argument("--filter_plot_class", type=int, nargs='*', default=None, help="Pain level to filter the data for plotting per class.")
  parser.add_argument("--save_tsne", action='store_true',help="Save tsne in {path_save_plot }/tsne_saved_features")
  parser.add_argument("--root_video_path", type=str, default=None, help="Set the video root folder and generate a video with plot")
  
  parser.add_argument("--plot_modality", type=str, default='sample_id', choices=['sample_id', 'subject_id', 'class'], help='Modality for t-SNE computation. sample_id: per sample, subject_id: per subject, class: per class (pain level)')
  parser.add_argument("--pain_levels", type=int,nargs='*', default=None, help="Pain level for filtering data.")
  parser.add_argument("--sample_id", type=int, nargs='*', default=None, help="Sample ID to filter the data.")
  parser.add_argument("--subject_id", type=int, nargs='*', default=None, help="Subject ID to filter the data.")
  parser.add_argument('--mode', type=str, default='open', choices=['open', 'skl'], help='Mode for t-SNE computation. open: OpenTSNE, skl: Scikit-learn')
  parser.add_argument('--from_', type=int, default=0, help='Starting index for the features. Default is 0.')
  parser.add_argument('--plot_results', action='store_true', help='Plot the t-SNE results')
  parser.add_argument('--df_result_path', type=int, default=None, help='Path to the DataFrame file to attach the t-SNE results')
  
  parser.add_argument('--path_save_plot', type=str, help='Path to save the tsne plot')
  parser.add_argument('--thresh_binary',type=int, default=4, help='Threshold chunks for binary classification. Index starting from 0')
  parser.add_argument('--stride_dataset', type=int, default=None, help='Stride of the dataset in frames')
  parser.add_argument('--fps_dataset', type=int, default=25, help='Frames per second of the dataset (default: 25)')
  parser.add_argument('--gp', action='store_true', help='Use global path for saving plots and csv files')  
  return parser


def start(dict_args: Dict[str, Any],plot_fn, df=None) -> None:
  if dict_args['all_tsne_feat_path']:
    dict_data = tsne_tools.load_dict_data(dict_args['all_tsne_feat_path'])
    config = None
  else:  
    
    # If tsne already computed and saved in saving_tsne_folder use that
    saving_tsne_folder = os.path.join(dict_args['path_save_plot'],'tsne_saved_features')
    sample_folder_path = os.path.join(saving_tsne_folder,f"sample_id_{dict_args['sample_id'][0]}.pkl")
    if os.path.isfile(sample_folder_path):
      with open(sample_folder_path,'rb') as f:
        dict_data = pickle.load(f)
        config = dict_data.pop('dict_config')
    else:
      dict_data, config = compute_tsne(dict_args)
    
    # Save results
    dict_args['tsne_config'] = config
    if dict_args['save_tsne'] and not os.path.isfile(sample_folder_path):
      os.makedirs(saving_tsne_folder, exist_ok=True)
      with open(os.path.join(saving_tsne_folder,f"sample_id_{dict_data['list_sample_id'][0]}.pkl"),'wb') as f:
        tmp = dict_data.copy()
        tmp['dict_config'] = dict_args
        pickle.dump(tmp,f)
        
  # Video generation case
  if dict_args['root_video_path']:
    list_rgb_plot = []
    len_video_seq = list(range(dict_data['features'].shape[0]))
    
    # Generate sequential plots to use for the video
    for count in tqdm.tqdm(len_video_seq,desc='Generating plots...'):
      tmp_dict = {}
      for k,v in dict_data.items():
        tmp_dict[k] = np.array(v[:count+1])
      # append the partial plt
      list_rgb_plot.append(plot_fn(tmp_dict,dict_args,(10,8)))
    video_path = os.path.join( dict_args['root_video_path'], str(df['subject_name'].iloc[0]),str(df['sample_name'].iloc[0])+'.mp4')
    
    tsne_tools.generate_video_from_list_video_path(video_path=video_path,
                                                   list_clip_ranges=dict_data['list_frames'],
                                                   list_rgb_image_plot=list_rgb_plot,
                                                   output_fps=4,
                                                   sample_id=dict_args['sample_id'][0],
                                                   y_gt=df['class_id'].iloc[0],
                                                   saving_path=os.path.join(dict_args['path_save_plot'],'tsne_video')
                                                   )
  else:
    results = plot_fn(dict_data, dict_args)
  # Save DataFrame if requested
  if dict_args.get('df_result_path') and not dict_args['root_video_path']:
    df_out = pd.DataFrame(results)
    df_out.to_csv(dict_args['df_result_path'], sep='\t', index=False)
    
    
def main() -> None:
  parser = get_parser()
  args = parser.parse_args()
  dict_args = vars(args)

  # Global paths
  if dict_args.get('gp'):
    for key in ['path_save_plot', 'csv_path', 'feat_path']:
      if dict_args.get(key):
        dict_args[key] = helper.GLOBAL_PATH.get_global_path(dict_args[key])

  df = pd.read_csv(dict_args['csv_path'], sep='\t')
  
  dict_args['df_csv_path'] = df
  modality = dict_args['plot_modality']

  if modality == 'sample_id':
    plot_fn = plot_per_sample_binary
  elif modality == 'subject_id':
    plot_fn = plot_per_subject_binary
  elif modality == 'class':
    plot_fn = plot_per_class_binary
  else:
    raise ValueError(f"Unsupported modality: {modality}")
  if dict_args['plot_per_class_subject']:
    list_unique_subjects = np.unique(df['subject_id'])
    list_unique_labels = np.unique(df['class_id'])
    for subject in tqdm.tqdm(list_unique_subjects,desc='Plotting per class subject'):
      for label in list_unique_labels:
        dict_args['subject_id'] = [subject]
        dict_args['pain_levels'] = [label]
        start(dict_args, plot_fn)
  elif dict_args['plot_per_class_all']:
    list_unique_labels = np.unique(df['class_id'])
    dict_args['path_save_plot'] = os.path.join(dict_args['path_save_plot'], 'plot_per_class_all')
    for label in tqdm.tqdm(list_unique_labels, desc='Plotting per class'):
      dict_args['pain_levels'] = [label]
      dict_args['subject_id'] = None
      dict_args['sample_id'] = None
      start(dict_args, plot_fn)
  elif dict_args['plot_per_sample_binary']:
    if dict_args['from_'] > 0:
      df = df.iloc[dict_args['from_']:]
    if dict_args['sample_id']:
      df = df[df['sample_id'].isin(dict_args['sample_id'])]
    list_unique_samples = np.unique(df['sample_id'])
    dict_args['path_save_plot'] = os.path.join(dict_args['path_save_plot'], 'plot_per_sample')
    for sample in tqdm.tqdm(list_unique_samples, desc='Plotting per sample'):
      dict_args['sample_id'] = [sample]
      dict_args['subject_id'] = None
      dict_args['pain_levels'] = None
      start(dict_args, plot_fn, df)
      if dict_args['all_tsne_feat_path']:
        break  # using precomputed features I plot everything at once 
  else:
    start(dict_args, plot_fn)
if __name__ == '__main__':
  main()

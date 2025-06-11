from openTSNE import TSNE as CPU_tsne
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA,TruncatedSVD
from sklearn.manifold import TSNE as sklearn_tsne
import pandas as pd
import time
import numpy as np
import warnings
import tqdm
import torch
import safetensors.torch
import os
from matplotlib import pyplot as plt
import cv2

def load_dict_data(saving_folder_path):
  """
  Load the dictionary containing numpy and torch elements from the specified path.

  Args:
    saving_path (str): Path to load the dictionary data.

  Returns:
    dict: Dictionary containing the loaded data.
  """
  if not ".safetensors" in saving_folder_path:
    dict_data = {}
    list_dir = os.listdir(saving_folder_path)
    for file in tqdm.tqdm(list_dir,desc="Loading files"):
      if file.endswith(".pt"):
        dict_data[file[:-3]] = torch.load(os.path.join(saving_folder_path, file),weights_only=True)
      elif file.endswith(".npy"):
        dict_data[file[:-4]] = np.load(os.path.join(saving_folder_path, file))
      else:
        print(f"Unsupported file format: {file}")
  else:
    dict_data = safetensors.torch.load_file(saving_folder_path, device='cpu')

  return dict_data



# def compute_tsne_whole(df_filtered,root_folder_feats,apply_pca_before_tsne, pca_nr_batch,perplexity=30, random_seed=42,n_jobs=1,mode='open'):
#   def get_X_batch(batch, df_filtered, root_folder_feats):
#     X_batch = []
#     for el in batch:
#       subject_name = df_filtered.iloc[el]['subject_name']
#       sample_name = df_filtered.iloc[el]['sample_name']
#       path_features = os.path.join(root_folder_feats, subject_name, sample_name + '.safetensors')
#       features = load_dict_data(path_features)  
#     X_batch = np.concatenate(X_batch, axis=0)
#     return X_batch  
    
#   config = {}
#   reduced_chunks = []
#   pca = IncrementalPCA(n_components=apply_pca_before_tsne, batch_size=pca_nr_batch)
#   print('Partial fitting PCA')
#   nr_samples = df_filtered.shape[0]
#   array_split = np.array_split(nr_samples, pca_nr_batch)
#   for batch in tqdm.tqdm(array_split,desc='Partial PCA fitting'):
#     X_batch = get_X_batch(batch, df_filtered, root_folder_feats)
#     pca.partial_fit(X_batch)
#   for batch in tqdm.tqdm(array_split,desc='Partial PCA transform'):
#     X_batch = get_X_batch(batch, df_filtered, root_folder_feats)
#     X_batch = pca.transform(X_batch)
#     reduced_chunks.append(X_batch)
    
#   X = np.concatenate(reduced_chunks, axis=0)
#   print(f'Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
#     # config['pca_explained_variance_ratio'] = np.sum(pca.explained_variance_ratio_)
  
#   # if mode == 'cpu':
#   print(f'Computing CPU t-SNE for {X.shape} features')
#   tsne_time = time.time()
#   if mode == 'open':
#     tsne = CPU_tsne(perplexity=perplexity,random_state=random_seed,n_jobs=n_jobs)
#     X_tsne = tsne.fit(X)
#   elif mode == 'skl':
#     perplexity = min(perplexity,X.shape[0]-1)
#     tsne = sklearn_tsne(perplexity=perplexity,random_state=random_seed,n_jobs=n_jobs)
#     X_tsne = tsne.fit_transform(X)
#   else:
#     raise ValueError("Invalid mode. Choose 'open' for openTSNE or 'skl' for sklearn t-SNE.")
#   print(f't-sne cpu time: {time.time() - tsne_time:.4f} seconds')
#   config = config | tsne.get_params()
#   if apply_pca_before_tsne:
#     config['pca_explained_variance_ratio'] = np.sum(pca.explained_variance_ratio_)
#   return X_tsne,config

# def compute_tsne(X, pca_nr_batch=0,perplexity=30, apply_pca_before_tsne=0,random_seed=42,n_jobs=1,mode='open'):
#   config = {}
#   reduced_chunks = []
#   if apply_pca_before_tsne:
#     if pca_nr_batch != 0:
#       pca = IncrementalPCA(n_components=apply_pca_before_tsne, batch_size=pca_nr_batch)
#       print('Partial fitting PCA')
#       array_split = np.array_split(X, pca_nr_batch)
#       for X_batch in tqdm.tqdm(array_split,desc='Partial PCA fitting'):
#         pca.partial_fit(X_batch)
#       for X_batch in tqdm.tqdm(array_split,desc='Partial PCA transform'):
#         X_reduced = pca.transform(X_batch)
#         reduced_chunks.append(X_reduced)
#       X = np.concatenate(reduced_chunks, axis=0)
#     else:
#       pca = PCA(n_components=apply_pca_before_tsne)
#       print(f'PCA for {X.shape} features')
#       pca_time = time.time()
#       X = pca.fit_transform(X)
#       print(f'PCA time: {time.time() - pca_time:.4f} seconds')
    
#     print(f'Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
#     # config['pca_explained_variance_ratio'] = np.sum(pca.explained_variance_ratio_)
  
#   # if mode == 'cpu':
#   print(f'Computing CPU t-SNE for {X.shape} features')
#   tsne_time = time.time()
#   if mode == 'open':
#     tsne = CPU_tsne(perplexity=perplexity,random_state=random_seed,n_jobs=n_jobs)
#     X_tsne = tsne.fit(X)
#   elif mode == 'skl':
#     perplexity = min(perplexity,X.shape[0]-1)
#     tsne = sklearn_tsne(perplexity=perplexity,random_state=random_seed,n_jobs=n_jobs)
#     X_tsne = tsne.fit_transform(X)
#   else:
#     raise ValueError("Invalid mode. Choose 'open' for openTSNE or 'skl' for sklearn t-SNE.")
#   print(f't-sne cpu time: {time.time() - tsne_time:.4f} seconds')
#   config = config | tsne.get_params()
#   if apply_pca_before_tsne:
#     config['pca_explained_variance_ratio'] = np.sum(pca.explained_variance_ratio_)
#   return X_tsne,config
#   # elif mode == 'cuda':
#   #   raise NotImplementedError("CUDA t-SNE is not implemented in this version. Please use CPU mode.")
#   #   # Importing tsnecuda here to avoid issues
#   #   # from tsnecuda import TSNE as CUDA_tsne
#   #   # print(f'Computing CUDA t-SNE for {X.shape} features')
#   #   # warnings.warn("Random seed is not available for CUDA t-SNE. Results may vary between runs.")
#   #   # tsne = CUDA_tsne(perplexity=perplexity)
#   #   # tsne_time = time.time()
#   #   # X_tsne = tsne.fit_transform(X)
#   #   # print(f't-sne cuda time: {time.time() - tsne_time:.4f} seconds')
#   #   # config = None  
#   # else:
#   #   raise ValueError("Invalid mode. Choose 'cpu'")
  
  
def flat_dict_data_(old_dict,new_dict):
  for k,v in old_dict.items():
    if k != 'features':
      if k not in new_dict:
        new_dict[k] = []
      new_dict[k].append(np.repeat(v, np.prod(old_dict['features'].shape[1:-1]), axis=0))
        
        
def compute_tsne(
  X=None,
  df_filtered=None,
  root_folder_feats=None,
  apply_pca_before_tsne=0,
  pca_nr_batch=0,
  perplexity=30,
  random_seed=42,
  n_jobs=1,
  mode='open'
):
  def get_X_batch(batch, df_filtered, root_folder_feats):
    X_batch = []
    for el in batch:
      subject_name = df_filtered.iloc[el]['subject_name']
      sample_name = df_filtered.iloc[el]['sample_name']
      path_features = os.path.join(root_folder_feats, subject_name, sample_name + '.safetensors')
      features = load_dict_data(path_features)  # Should return a numpy array
      X_batch.append(features)
    # X_batch = np.concatenate(X_batch, axis=0)
    dict_data = {}
    for partial_dict_data in X_batch:
      for k,v in partial_dict_data.items():
        dict_data.setdefault(k, []).append(v)
    for k,v in dict_data.items():
      dict_data[k] = np.concatenate(v, axis=0)
    return dict_data

    
  
  config = {}
  reduced_chunks = []
  raw_data = {}
  dict_ = None
  # WHOLE feats
  if X is None:
    if apply_pca_before_tsne:
      assert df_filtered is not None and root_folder_feats is not None, "Must provide df_filtered and root_folder_feats if X is None"
      nr_samples = df_filtered.shape[0]
      array_split = np.array_split(np.arange(nr_samples), pca_nr_batch)
      array_split = [batch for batch in array_split if len(batch) > 0]  # Filter out empty batches
      print('Partial fitting PCA')
      pca = IncrementalPCA(n_components=apply_pca_before_tsne,
                          batch_size=pca_nr_batch)
      
      # Partial fit PCA on each batch
      for batch in tqdm.tqdm(array_split, desc='Partial PCA fitting'):
        batch_dict_data = get_X_batch(batch, df_filtered, root_folder_feats)
        # Flatten the dictionary data to match the features shape
        flat_dict_data_(batch_dict_data, raw_data)
        X_batch = batch_dict_data['features'].reshape(-1,batch_dict_data['features'].shape[-1])  # Ensure correct shape
        pca.partial_fit(X_batch)
        
      # Concatenate the features from all batches
      for k,v in raw_data.items():
        raw_data[k] = np.concatenate(v, axis=0)
        
      # Transform the features 
      for batch in tqdm.tqdm(array_split, desc='Partial PCA transform'):
        batch_dict_data = get_X_batch(batch, df_filtered, root_folder_feats)
        X_batch = batch_dict_data['features'].reshape(-1,batch_dict_data['features'].shape[-1])
        X_reduced = pca.transform(X_batch)
        reduced_chunks.append(X_reduced)
      X = np.concatenate(reduced_chunks, axis=0)
      
    else:
      assert df_filtered is not None and root_folder_feats is not None, "Must provide df_filtered and root_folder_feats if X is None"
      nr_samples = df_filtered.shape[0]
      if nr_samples == 1:
        # sample_id = df_filtered['sample_id']
        subject_name = df_filtered['subject_name'].values[0]
        sample_name = df_filtered['sample_name'].values[0]
        path_features = os.path.join(root_folder_feats, subject_name, sample_name + '.safetensors')
        dict_ = load_dict_data(path_features) 
        flat_dict_data_(dict_, raw_data)
        # raw_data = dict_.copy()
        for k,v in raw_data.items():
          raw_data[k] = np.concatenate(v, axis=0)
        # raw_data['features']= dict_['features']
        X = dict_['features'].reshape(-1,dict_['features'].shape[-1])

        
  # AGGREGATED feats
  elif apply_pca_before_tsne:
    if pca_nr_batch != 0:
      pca = IncrementalPCA(n_components=apply_pca_before_tsne, batch_size=pca_nr_batch)
      print('Partial fitting PCA')
      array_split = np.array_split(X, pca_nr_batch)
      for X_batch in tqdm.tqdm(array_split, desc='Partial PCA fitting'):
        pca.partial_fit(X_batch)
      for X_batch in tqdm.tqdm(array_split, desc='Partial PCA transform'):
        X_reduced = pca.transform(X_batch)
        reduced_chunks.append(X_reduced)
      X = np.concatenate(reduced_chunks, axis=0)
    else:
      pca = PCA(n_components=apply_pca_before_tsne)
      print(f'PCA for {X.shape} features')
      pca_time = time.time()
      X = pca.fit_transform(X)
      print(f'PCA time: {time.time() - pca_time:.4f} seconds')

  # Get cumulative explained variance ratio if PCA is applied
  if apply_pca_before_tsne:
    print(f'Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
    config['pca_explained_variance_ratio'] = np.sum(pca.explained_variance_ratio_)

  print(f'Computing CPU t-SNE for {X.shape} features')
  tsne_time = time.time()
  if mode == 'open':
    tsne = CPU_tsne(perplexity=perplexity, random_state=random_seed, n_jobs=n_jobs)
    X_tsne = tsne.fit(X)
  elif mode == 'skl':
    perplexity = min(perplexity, X.shape[0] - 1)
    tsne = sklearn_tsne(perplexity=perplexity, random_state=random_seed, n_jobs=n_jobs)
    X_tsne = tsne.fit_transform(X)
  else:
    raise ValueError("Invalid mode. Choose 'open' for openTSNE or 'skl' for sklearn t-SNE.")
  print(f't-sne cpu time: {time.time() - tsne_time:.4f} seconds')

  config = config | tsne.get_params()
  return X_tsne, config, raw_data, dict_


def filter_dict_features_per_key(dict_data, filter_array,key='list_subject_id'):
  mask = np.isin(dict_data[key], filter_array)
  for k,v in dict_data.items():
    dict_data[k] = v[mask]
  return dict_data

def filter_dict_features_per_csv(dict_data, csv_path):
  df = pd.read_csv(csv_path, sep='\t')
  list_sample_id = np.array(df['sample_id'].values)
  mask = np.isin(dict_data['list_sample_id'], list_sample_id)
  for k,v in dict_data.items():
    dict_data[k] = v[mask]
  return dict_data

def plot_per_subject(dict_data,labels_key,path_save_plot,max_elements_per_plot=20,ax=None,**kwargs):
  X_tsne = dict_data['features']
  list_key = dict_data[labels_key]
  limit_key_per_plot = max_elements_per_plot
  unique_key_id = np.unique(list_key)
  print(f'unique_subject_id: {unique_key_id.shape[0]}')
  sbj_chunk_array = np.array_split(unique_key_id, unique_key_id.shape[0]//limit_key_per_plot+1)
  for sbj_chunk in sbj_chunk_array:
    mask = np.isin(list_key, sbj_chunk)
    plot_tsne(X_tsne=X_tsne[mask],
                    labels=list_key[mask],
                    saving_path=path_save_plot,
                    title=f'Plot_{sbj_chunk}',
                    legend_label='Subject ID',
                    ax=ax,
                    cmap='tab20',)
  print(f'Plots saved in {path_save_plot}_per_subject')
  
  
  
def plot_tsne(X_tsne,
  labels,
  cmap='copper',
  tot_labels=None,
  legend_label='',
  title='',
  cluster_measure='',
  list_additiona_desc_legend=None,
  saving_path=None,
  chunk_interval=None,
  axis_scale=None,
  last_point_bigger=False,
  plot_trajectory=False,
  stride_windows=None,
  clip_length=None,
  list_axis_name=None,
  ax=None,
  fig_size=None,
  return_ax=False):
  """
  Plot a 2D t-SNE embedding, return either:
    - an RGB array of the rendered plot, or
    - the Axes object for further in-code modifications, or
    - save to disk and return the path.
  """
  marker_dict = {
    0: 'o',  # circle
    1: '^',  # triangle
  }
  unique_labels = np.unique(labels)
  n_colors = tot_labels if tot_labels is not None else len(unique_labels)
  color_map = plt.cm.get_cmap(cmap, n_colors)
  color_dict = {val: color_map(i) for i, val in enumerate(unique_labels)}

  # decide whether to create a new figure/ax or use the one passed in
  if ax is None:
    fig, ax = plt.subplots(figsize=fig_size if fig_size is not None else (10, 8))
  else:
    fig = ax.figure

  # apply axis bounds if given
  if axis_scale is not None:
    ax.set_xlim(axis_scale['min_x'], axis_scale['max_x'])
    ax.set_ylim(axis_scale['min_y'], axis_scale['max_y'])

  # optionally enlarge just the last point
  sizes = None
  if last_point_bigger:
    sizes = np.full(X_tsne.shape[0], 50)
    sizes[-1] = 200

  # plot each label
  i=0
  for val in unique_labels:
    idx = (labels == val)
    # special “clip” labeling logic
    if clip_length is not None and legend_label == 'clip':
      idx_color = 0 if stride_windows * val < 48 else max(color_dict.keys())
      label = (
        f'{legend_label} [{stride_windows * val}, '
        f'{clip_length + stride_windows * val - 1}]'
        + (f' ({cluster_measure[idx_color]:.2f})'
           if cluster_measure and idx_color < len(cluster_measure) else '')
      )
      c = color_dict[idx_color]
    else:
      label = (
        f'{legend_label} {val}'
        + (f'{chunk_interval[i]}' if chunk_interval is not None else '')
        + (f' ({cluster_measure[val]:.2f})' if cluster_measure else '')
      )
      if list_additiona_desc_legend is not None:
        label += f' {list_additiona_desc_legend[i]}' if i < len(list_additiona_desc_legend) else ''
      i+=1
      c = color_dict[val]
    marker = marker_dict.get(val, 'o')  # default to circle if not found
    ax.scatter(
      X_tsne[idx, 0],
      X_tsne[idx, 1],
      color=c,
      label=label,
      alpha=0.7,
      facecolors='none',
      # edgecolors='face',
      marker=marker,
      s=sizes[idx] if sizes is not None else 50
    )

  # optional trajectory line
  if plot_trajectory:
    ax.plot(
      X_tsne[:, 0],
      X_tsne[:, 1],
      linestyle='--',
      color=color_dict[0],
      label='Trajectory',
      alpha=0.7
    )

  # axis labels & legend/title
  if list_axis_name:
    ax.set_xlabel(list_axis_name[0])
    ax.set_ylabel(list_axis_name[1])
  else:
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')

  ax.legend()
  ax.set_title(f'{title} (Colored by {legend_label})')

  # --- OUTPUT BRANCHES ---

  # 1) SAVE TO DISK
  if saving_path is not None:
    # ensure .png extension
    if not saving_path.lower().endswith('.png'):
      os.makedirs(saving_path, exist_ok=True)
      saving_path = os.path.join(saving_path, f'{title}_{legend_label}.png')
    fig.savefig(saving_path)
    if not return_ax:
      plt.close(fig)
    return saving_path

  # 2) RETURN AXES FOR FURTHER IN-CODE USE
  if return_ax:
    # leave the figure open so the caller can continue to modify or close it
    return ax

  # 3) RENDER TO RGB ARRAY
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = buf.reshape(h, w, 3)
  plt.close(fig)
  return img


def generate_video_from_list_video_path(video_path, list_clip_ranges, sample_id, y_gt, saving_path, list_rgb_image_plot,output_fps=4):
  """
  Generate a video by extracting specific frames from a list of videos.

  Args:
    list_video_path (list): List of paths to the input video files.
    list_frames (list): List of lists, where each sublist contains the frame indices to extract from the corresponding video.
    idx_list_frames (list): List of indices corresponding to the frames to be extracted.
    list_sample_id (list): List of sample IDs corresponding to the videos.
    list_y_gt (list): List of ground truth labels corresponding to the videos.
    saving_path (str): Path to save the generated video.
    list_image_path (list, optional): List of paths to images to be merged with the video frames. Defaults to None.
  """
  print('Generating video...')
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError(f"Error: Unable to open video file: {video_path}")

  frames_to_process = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames_to_process.append(cv2.resize(frame,(600,600)))    
  cap.release()
  
  frames_to_process = np.stack(frames_to_process,axis=0)
  frame_width = int(frames_to_process.shape[2])
  frame_height = int(frames_to_process.shape[1])
  video_frame_size = (frame_width, frame_height)
  
  # Suppose that each image in the list has the same width and height
  rgb_plot_height, rgb_plot_width = list_rgb_image_plot[0].shape[:2]
  frame_size = (frame_width + rgb_plot_width, max(frame_height, rgb_plot_height))
  plot_size = (rgb_plot_width,rgb_plot_height)
  # if out is None:
  os.makedirs(saving_path,exist_ok=True)
  out = cv2.VideoWriter(os.path.join(saving_path, f'video_{sample_id}.mp4'), fourcc, output_fps, frame_size)
  
  # list_rgb_image_plot = np.concatenate(list_rgb_image_plot,axis=0)
  reframed_image_plot = np.zeros(shape=(frames_to_process.shape[0],
                                        plot_size[1],
                                        plot_size[0],
                                        3)).astype(np.uint8)
  count = 0
  for frame_range in list_clip_ranges:
    reframed_image_plot[frame_range] = list_rgb_image_plot[count]
    count += 1
  
  # shape -> [nr_frames,combined_row,combined_cols,3]
  delay_plot = 16 # frames of delay
  combined_frame = np.zeros(shape=(frames_to_process.shape[0] + delay_plot,
                                   frame_size[1],
                                   frame_size[0],
                                   3)).astype(np.uint8)
  combined_frame[:combined_frame.shape[0]-delay_plot,frame_size[1]-video_frame_size[1]:,:video_frame_size[0],:] = frames_to_process
  combined_frame[delay_plot:,:plot_size[1], video_frame_size[0]:video_frame_size[0] + plot_size[0],:] = reframed_image_plot
  
  overlay_text = [
      # f'Sample ID  : {sample_id}',
      f'Pain class : {y_gt}',
      # f'Clip num.  : {list_clip_ranges}',
      # f'Frame range: [{list_frames[0]},{list_frames[-1]}]'
  ]
  count = 0
  for frame in combined_frame:
    for i, text in enumerate(overlay_text):
      cv2.putText(frame, text, (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Chunk: {(count//16)+1}', (50, 50 + (i+1) * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Frame: {count}', (50, 50 + (i+2) * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    count += 1
    out.write(frame)
  
  out.release()
  print(f"Generated video saved to folder {saving_path}")

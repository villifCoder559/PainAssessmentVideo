from openTSNE import TSNE as CPU_tsne
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA,TruncatedSVD
import pandas as pd
import time
import numpy as np
import warnings
import tqdm
import tools

def compute_tsne(X, pca_nr_batch=0,perplexity=30, apply_pca_before_tsne=0,random_seed=42,n_jobs=1,mode='cpu'):
  config = {}
  reduced_chunks = []
  if apply_pca_before_tsne:
    if pca_nr_batch != 0:
      pca = IncrementalPCA(n_components=apply_pca_before_tsne, batch_size=pca_nr_batch)
      print('Partial fitting PCA')
      array_split = np.array_split(X, pca_nr_batch)
      for X_batch in tqdm.tqdm(array_split,desc='Partial PCA fitting'):
        pca.partial_fit(X_batch)
      for X_batch in tqdm.tqdm(array_split,desc='Partial PCA transform'):
        X_reduced = pca.transform(X_batch)
        reduced_chunks.append(X_reduced)
      X = np.concatenate(reduced_chunks, axis=0)
    else:
      pca = PCA(n_components=apply_pca_before_tsne)
      print(f'PCA for {X.shape} features')
      pca_time = time.time()
      X = pca.fit_transform(X)
      print(f'PCA time: {time.time() - pca_time:.4f} seconds')
    
    print(f'Cumulative explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}')
  
  if mode == 'cpu':
    print(f'Computing CPU t-SNE for {X.shape} features')
    tsne = CPU_tsne(perplexity=perplexity,random_state=random_seed,n_jobs=n_jobs,metric='cosine')
    tsne_time = time.time()
    X_tsne = tsne.fit(X)
    print(f't-sne cpu time: {time.time() - tsne_time:.4f} seconds')
    config = tsne.get_params()
    if apply_pca_before_tsne:
      config['pca'] = np.sum(pca.explained_variance_ratio_)
  elif mode == 'cuda':
    # Importing tsnecuda here to avoid issues
    from tsnecuda import TSNE as CUDA_tsne
    print(f'Computing CUDA t-SNE for {X.shape} features')
    warnings.warn("Random seed is not available for CUDA t-SNE. Results may vary between runs.")
    tsne = CUDA_tsne(perplexity=perplexity)
    tsne_time = time.time()
    X_tsne = tsne.fit_transform(X)
    print(f't-sne cuda time: {time.time() - tsne_time:.4f} seconds')
    config = None  
  else:
    raise ValueError("Invalid mode. Choose 'cpu' or 'cuda'")
  
  return X_tsne,config


def filter_dict_features_per_key(dict_data, filter_array,key='list_subject_id'):
  mask = np.isin(dict_data[key], filter_array)
  for k,v in dict_data.items():
    dict_data[k] = v[mask]
  return dict_data

def filter_dict_features_per_csv(dict_data, csv_path):
  df = pd.read_csv(csv_path)
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
    tools.plot_tsne(X_tsne=X_tsne[mask],
                    labels=list_key[mask],
                    saving_path=path_save_plot,
                    title=f'Plot_{sbj_chunk}',
                    legend_label='Subject ID',
                    ax=ax,
                    cmap='tab20',)
  print(f'Plots saved in {path_save_plot}_per_subject')
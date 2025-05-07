import custom.tsne_cuda_tools as tsne_tools
import argparse
import custom.tools as tools
import numpy as np
import pandas as pd
import torch
import os


# def save_dict_data(dict_data, output_path):

def main(dict_args):
  dict_data = tools.load_dict_data(dict_args['feat_path'])

  # Filter based on the CSV file
  if dict_args['csv_path'] is not None:
    dict_data = tsne_tools.filter_dict_features_per_csv(dict_data, dict_args['csv_path'])
  
  # Filter based on the pain levels
  if dict_args['pain_levels'] is not None:
    dict_data = tsne_tools.filter_dict_features_per_key(dict_data, dict_args['pain_levels'], key='list_labels')
  
  # Filter based on the first subject chunk size
  if dict_args['sbj_chunk_size'] > 1:
    sbj_array = np.unique(dict_data['list_subject_id'])
    sbj_array = np.array_split(sbj_array, dict_args['sbj_chunk_size'])
    print(f'Number of subjects: {len(sbj_array)}')
    dict_data = tsne_tools.filter_dict_features_per_key(dict_data, sbj_array[0], key='list_subject_id')
  
  original_shape = dict_data['features'].shape
  # Perform t-SNE on the filtered data
  if dict_args['temporal_as_emb']:
    dict_data['features'] = dict_data['features'].reshape(dict_data['features'].shape[0], -1)
    # _,count_chunks_per_video = np.unique(dict_data['list_sample_id'], return_counts=True)
    # dict_data['list_subject_id'] = dict_data['list_subject_id'] * 8
    # dict_data['list_sample_id'] = dict_data['list_sample_id'] * 8
    # dict_data['list_labels'] = dict_data['list_labels'] * 8
  else:
    dict_data['features'] = dict_data['features'].reshape(dict_data['features'].shape[0] * dict_data['features'].shape[1], -1)
  
  # for mode in dict_args['mode']:
  X_tsne,config = tsne_tools.compute_tsne(dict_data['features'], 
                          pca_nr_batch=dict_args['nr_batch'],
                          perplexity=dict_args['perplexity'], 
                          apply_pca_before_tsne=dict_args['apply_pca_before_tsne'],
                          random_seed=dict_args['random_seed'], # NOT applied for CUDA
                          n_jobs=dict_args['n_jobs'], 
                          mode=dict_args['mode'])
  
  if not dict_args['temporal_as_emb']:
    X_tsne = X_tsne.reshape(original_shape[0], -1, X_tsne.shape[1])
    print(f'X_tsne shape: {X_tsne.shape}')
  if dict_args['output_path'] is not None:
    # Save the t-SNE results
    new_dict_data = dict_data.copy()
    new_dict_data['features'] = torch.tensor(X_tsne)
    tools.save_dict_data(new_dict_data, dict_args['output_path'])
    
    # Save the t-SNE configuration
    with open(os.path.join(dict_args['output_path'],'_config.txt'), 'w') as f:
      f.write("Arguments used for t-SNE:\n")
      for key, value in dict_args.items():
        f.write(f"{key}: {value}\n")
      f.write("CPU t-SNE configuration:\n")
      for key, value in config.items():
        f.write(f"{key}: {value}\n")
      
      
      


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare t-SNE results between CUDA and CPU implementations.")
    parser.add_argument("--feat_path", type=str, required=True, help="Path to the feature file.")
    parser.add_argument("--output_path", type=str,default=None, help="Path to save the comparison results.")
    parser.add_argument("--perplexity", type=int, default=30, help="Perplexity for t-SNE.")
    parser.add_argument("--apply_pca_before_tsne", type=int, default=100, help="Number of PCA components to apply before t-SNE. 0 means no PCA.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility. Works only for CPU implementation.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for CPU implementation.")
    parser.add_argument("--csv_path", type=str, default=None, help="Path to the CSV file.")
    parser.add_argument("--sbj_chunk_size", type=int, default=1, help="Chunk size for subject data. If 1, it will use all subjects, it will use the first chunk of subjects.")
    parser.add_argument("--pain_levels", type=int,nargs='*', default=None, help="Pain level for filtering data.")
    parser.add_argument("--mode", type=str, default='cpu',choices=['cuda','cpu'], help="Mode for t-SNE computation.")
    parser.add_argument("--temporal_as_emb", type=int, default=0, help="If 1 reshape(dict_data['features'].shape[0], -1), else reshape(dict_data['features'].shape[0] * dict_data['features'].shape[1], -1)")
    parser.add_argument("--nr_batch", type=int, default=0, help="Number of batch to divide the dataset. If 0, it will use the entire dataset.")
    
    dict_args = vars(parser.parse_args())
    for key, value in dict_args.items():
      print(f"{key}: {value}")
    main(dict_args)
    

import custom.tsne_cuda_tools as tsne_tools
import argparse
import custom.tools as tools
import numpy as np
import pandas as pd
import torch
import os
import pickle
import extract_and_compute_tsne 
import custom.helper as helper

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
  print(f'feature shape: {dict_data["features"].shape}')
  # Perform t-SNE on the filtered data
  if dict_args['only_last_dim_as_embedding']:
    # list_real_shape = []
    # list_sample_id = []
    # for feat,sample_id in zip(dict_data['features'],dict_data['list_sample_id']):
    #   list_real_shape.append(feat.shape)
    #   reps = np.prod(feat.shape[:-1])
    #   list_sample_id.append(np.repeat(np.array(sample_id), reps))
    # list_real_shape = np.array(list_real_shape)
    # list_sample_id = np.array(list_sample_id)
   feats = dict_data['features'].reshape(-1,dict_data['features'].shape[-1])
    # _,count_chunks_per_video = np.unique(dict_data['list_sample_id'], return_counts=True)
    # dict_data['list_subject_id'] = dict_data['list_subject_id'] * 8
    # dict_data['list_sample_id'] = dict_data['list_sample_id'] * 8
    # dict_data['list_labels'] = dict_data['list_labels'] * 8
  else:
    feats = dict_data['features'].reshape(dict_data['features'].shape[0], -1)
  
  # for mode in dict_args['mode']:
  X_tsne,config,_ = tsne_tools.compute_tsne(feats, 
                          pca_nr_batch=dict_args['nr_batch'],
                          perplexity=dict_args['perplexity'], 
                          apply_pca_before_tsne=dict_args['apply_pca_before_tsne'],
                          random_seed=dict_args['random_seed'], # NOT applied for CUDA
                          n_jobs=dict_args['n_jobs'], 
                          mode=dict_args['mode'])
  # reconstruct the original shape if only last dimension is considered as embedding
  if dict_args['only_last_dim_as_embedding']:
    list_reshaped_feats = []
    unique_sample_ids = np.unique(dict_data['list_sample_id'])
    for sample_id in unique_sample_ids: # both list have -> (Batch, *shape_array)
      mask = np.isin(dict_data['list_sample_id'], sample_id)
      real_shape = dict_data['features'][mask].shape[1:-1]  # Get the shape of the features excluding the last dimension
      mask = np.repeat(mask,np.prod(real_shape))
      tmp = np.concatenate(X_tsne[mask])
      tmp = tmp.reshape(-1, *real_shape, 2)  # Reshape to the original shape, 2 is the t-SNE dimension
      list_reshaped_feats.append(tmp)
  list_reshaped_feats = np.concatenate(list_reshaped_feats, axis=0)
  # if not dict_args['temporal_as_emb']:
  #   X_tsne = X_tsne.reshape(original_shape[0], -1, X_tsne.shape[1])
  #   print(f'X_tsne shape: {X_tsne.shape}')
  if dict_args['output_path'] is not None:
    # Save the t-SNE results
    new_dict_data = dict_data.copy()
    assert np.all(list_reshaped_feats.shape[:-1]==new_dict_data['features'].shape[:-1]), f"Original shape and new t-SNE shapeed feats do not match. New shape: {list_reshaped_feats.shape[:-1]}, original shape: {new_dict_data['features'].shape[:-1]}"
    new_dict_data['features'] = torch.tensor(list_reshaped_feats)
    
    tools.save_dict_data(dict_data=new_dict_data, 
                         saving_folder_path=dict_args['output_path'],
                         save_as_safetensors=dict_args['save_as_safetensors'],)
    out_folder = os.path.dirname(dict_args['output_path'])
    # Save the t-SNE configuration
    with open(os.path.join(out_folder,'_config.txt'), 'w') as f:
      f.write("Arguments used for t-SNE:\n")
      for key, value in dict_args.items():
        f.write(f"{key}: {value}\n")
      f.write("CPU t-SNE configuration:\n")
      for key, value in config.items():
        f.write(f"{key}: {value}\n")
    with open(os.path.join(out_folder,'_config.pkl'), 'wb') as f:
      pickle.dump(dict_args, f)
      
    print(f"t-SNE results saved to {dict_args['output_path']}")

      
      
      


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Compare t-SNE results between CUDA and CPU implementations.")
  parser.add_argument("--feat_path", type=str, required=True, help="Path to the feature file.")
  parser.add_argument("--output_path", type=str,default=None, help="Path to save the comparison results.")
  parser.add_argument("--perplexity", type=int, default=30, help="Perplexity for t-SNE.")
  parser.add_argument("--apply_pca_before_tsne", type=int, default=100, help="Number of PCA components to apply before t-SNE. 0 means no PCA.")
  parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility. Works only for CPU implementation.")
  parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs for CPU implementation.")
  parser.add_argument("--csv_path", type=str, default=None, help="Path to the CSV file.")
  parser.add_argument("--sbj_chunk_size", type=int, default=1, help="Chunk size for subject data. If 1, it will use all subjects, else it will use the first chunk of subjects.")
  parser.add_argument("--pain_levels", type=int,nargs='*', default=None, help="Pain level for filtering data.")
  parser.add_argument("--mode", type=str, default='open',choices=['skl','open'], help="Mode for t-SNE computation.")
  parser.add_argument("--only_last_dim_as_embedding", type=int, default=1, help="Consider only the last dimension as embedding. If 1, it will reshape the features to (-1, emb.shape[-1]).")
  parser.add_argument("--nr_batch", type=int, default=0, help="Number of batch to divide the dataset. If 0, it will use the entire dataset.")
  parser.add_argument("--type_feat", type=str, default='aggregated', choices=['aggregated', 'whole'], help='Type of features to use. aggregated (saved in one dict), whole(save in more dicts)')
  parser.add_argument('--gp', action='store_true', help='Use global path for saving plots and csv files')  

  dict_args = vars(parser.parse_args())
  if dict_args['gp']:
    for key in ['feat_path', 'output_path', 'csv_path']:
      if dict_args.get(key):
        dict_args[key] = helper.GLOBAL_PATH.get_global_path(dict_args[key])
        
  dict_args['output_path'] = os.path.join(dict_args['output_path'],os.path.basename(dict_args['feat_path']).replace('.safetensors',''),os.path.basename(dict_args['feat_path'])) 
  dict_args['save_as_safetensors'] = True 
  
  assert dict_args['feat_path'] != dict_args['output_path'], "Output path cannot be the same as feature path. Please provide a different output path."
  for key, value in dict_args.items():
    print(f"{key}: {value}")
  if dict_args['type_feat'] == 'whole':
    new_dict, old_dict = extract_and_compute_tsne.compute_tsne(dict_args)
    tools.save_dict_data(dict_data=new_dict, 
                      saving_folder_path=dict_args['output_path'],
                      save_as_safetensors=dict_args['save_as_safetensors'],)
  else:
    main(dict_args)
  
  
    

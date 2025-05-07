import custom.tools as tools
import custom.tsne_cuda_tools as tsne_tools
import os
import numpy as np
import argparse
import shutil
import tqdm
import custom.helper as helper
from sklearn.metrics import silhouette_score, davies_bouldin_score


def main(dict_args):
  
  dict_data = tools.load_dict_data(saving_folder_path=dict_args['path_tsne_feat'])
  
  if len(dict_data['features'].shape) == 3: # [nr_chunks, T, 2]
    temporal_dim = dict_data['features'].shape[1]
    dict_data['features'] = dict_data['features'].reshape(-1, dict_data['features'].shape[2])
    for k,v in dict_data.items():
      if k == 'features':
        continue
      dict_data[k] = np.repeat(v, temporal_dim, axis=0) 
    print(f'X_tsne shape: {dict_data["features"].shape}')
    
  if dict_args['filter_class'] is not None:
    dict_data = tsne_tools.filter_dict_features_per_key(dict_data,
                                                        filter_array=np.array(dict_args['filter_class']).astype(int),
                                                        key='list_labels')

  if 'sbj' in dict_args['plot_type']:
    plot_per_subject(dict_data,dict_args)
  if 'sam_bin' in dict_args['plot_type']:
    plot_per_sample_binary(dict_data,dict_args)
    
  if 'sbj_bin' in dict_args['plot_type'] :
    if dict_args['filter_class'] is not None and len(dict_args['filter_class']) == 1:
      plot_per_subject_binary(dict_data,dict_args)
    else:
      print('Plotting per subject binary not possible. Please check the filter_class argument.')
      
def plot_per_subject(dict_data,dict_args,ax=None):
  X_tsne = dict_data['features']
  list_key = dict_data[dict_args['labels_key']]
  limit_key_per_plot = dict_args['max_elements_per_plot']
  unique_key_id = np.unique(list_key)
  print(f'unique_subject_id: {unique_key_id.shape[0]}')
  sbj_chunk_array = np.array_split(unique_key_id, unique_key_id.shape[0]//limit_key_per_plot+1)
  for sbj_chunk in sbj_chunk_array:
    mask = np.isin(list_key, sbj_chunk)
    tools.plot_tsne(X_tsne=X_tsne[mask],
                    labels=list_key[mask],
                    saving_path=dict_args['path_save_plot']+'_per_subject',
                    title=f'Plot_{sbj_chunk}',
                    legend_label='Subject ID',
                    ax=ax,
                    cmap='tab20',)
  print(f'Plots saved in {dict_args["path_save_plot"]}_per_subject')
  save_config_file(path_save_plot=dict_args['path_save_plot']+'_per_subject',
                   dict_args=dict_args)

def plot_per_sample_binary(dict_data,dict_args): # do the same for subjects
  unique_sample_id = np.unique(dict_data['list_sample_id'])
  for sample_id in tqdm.tqdm(unique_sample_id,desc='Plotting samples'):
    mask = np.isin(dict_data['list_sample_id'], sample_id)
    # new_labels = np.array(dict_data['list_labels'])[mask]
    tot_chunks = mask.sum()
    binary_labels = np.array([dict_data['list_labels'][mask][0] if i >= dict_args['thresh_binary'] else 0 for i in range(tot_chunks)])
    title = f"sbj_{dict_data['list_subject_id'][mask][0]}_chunks_{tot_chunks}_PA_0vs{int(dict_data['list_labels'][mask][0])}" +('_stoic_' if dict_data['list_subject_id'][mask][0] in helper.stoic_subjects else "")
    tools.plot_tsne(X_tsne=dict_data['features'][mask],
                    labels=binary_labels,
                    saving_path=dict_args['path_save_plot']+'_per_sample_binary',
                    title=title,
                    legend_label='Pain Level',
                    tot_labels=max(binary_labels),
                    cmap='tab10')
  print(f'Plots saved in {dict_args["path_save_plot"]}_per_sample_binary')
  save_config_file(path_save_plot=dict_args['path_save_plot']+'_per_sample_binary',
                   dict_args=dict_args)

def plot_per_subject_binary(dict_data,dict_args):
  unique_subject_id = np.unique(dict_data['list_subject_id'])
  for subject_id in tqdm.tqdm(unique_subject_id,desc='Plotting subjects'):
    mask = np.isin(dict_data['list_subject_id'], subject_id)
    _,nr_chunks_per_video = np.unique(dict_data['list_sample_id'][mask], return_counts=True)
    
    # For each video I enumerate the chunks and the generate the binary labels based on the threshold
    binary_labels = np.array([np.arange(i) for i in nr_chunks_per_video]).reshape(-1)
    mask_0_label = binary_labels < dict_args['thresh_binary']
    binary_labels[mask_0_label] = 0
    binary_labels[~mask_0_label] = int(dict_args['filter_class'][0])
    silh_score = silhouette_score(X=dict_data['features'][mask], labels=binary_labels)
    title = f"sbj_{subject_id}_chunks_{np.unique(nr_chunks_per_video)}_PA_0vs{int(dict_args['filter_class'][0])}_silhScore_{silh_score:.4}" + ('_stoic_' if subject_id in helper.stoic_subjects else "")
    tools.plot_tsne(X_tsne=dict_data['features'][mask],
                    labels=binary_labels,
                    saving_path=dict_args['path_save_plot']+'_per_subject_binary',
                    title=title,
                    legend_label='Pain Level',
                    tot_labels=int(dict_args['filter_class'][0]),
                    cmap='tab10')
  print(f'Plots saved in {dict_args["path_save_plot"]}_per_subject_binary')
  save_config_file(path_save_plot=dict_args['path_save_plot']+'_per_subject_binary',
                   dict_args=dict_args)
  
  
def save_config_file(path_save_plot,dict_args):
  # Save config file
  config_file_tsne = os.path.join(dict_args['path_tsne_feat'], '_config.txt')
  if os.path.exists(config_file_tsne):
    shutil.copyfile(config_file_tsne, os.path.join(path_save_plot, '_config.txt'))
    
  with open(os.path.join(path_save_plot,'_config.txt'), 'a') as f:
    f.write("\n\nArguments used for PLOT t-SNE:\n")
    for key, value in dict_args.items():
      f.write(f"{key}: {value}\n")
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Compute tsne plot')
  parser.add_argument('--path_tsne_feat', type=str, help='Path to the folder containing the tsne data extracted')
  parser.add_argument('--path_save_plot', type=str, help='Path to save the tsne plot')
  parser.add_argument('--filter_class', type=str, nargs='*', default=None, help='Filter the data based on the class')
  parser.add_argument('--labels_key', type=str, choices=['list_labels','list_subject_id'],default='list_subject_id', help='Key to use for labels in the dictionary')
  parser.add_argument('--max_elements_per_plot', type=int, default=20, help='Maximum number of elements per plot')
  parser.add_argument('--thresh_binary',type=int, default=4, help='Threshold chunks for binary classification. Index starting from 0')
  parser.add_argument('--plot_type', type=str, nargs='*', choices=['sbj','sbj_bin','sam_bin'], default=['sbj'], help='Type of plot to generate')
  
  args = parser.parse_args()
  dict_args = vars(args)
  
  main(dict_args)
  
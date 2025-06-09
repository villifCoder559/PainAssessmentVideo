import custom.tools as tools
import custom.tsne_cuda_tools as tsne_tools
import custom.helper as helper
import os
import numpy as np
import argparse
import shutil
import tqdm
import custom.helper as helper
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import pickle

def main(dict_args):
  
  dict_data = tools.load_dict_data(saving_folder_path=dict_args['path_tsne_feat'])
  # Use the last dimension of the features as the number of channels
  if len(dict_data['features'].shape) == 3: # [nr_chunks, T, 2]
    # nr_chunks = dict_data['features'].shape[0]
    temporal_dim = dict_data['features'].shape[1]
    dict_data['features'] = dict_data['features'].reshape(-1, dict_data['features'].shape[2])
    for k,v in dict_data.items():
      # make consistent the shape with "features" for all keys in the dict_data (List_labels, List_sample_id, List_subject_id)
      if k == 'features':
        continue
      dict_data[k] = np.repeat(v, temporal_dim, axis=0) 
    print(f'X_tsne shape: {dict_data["features"].shape}')
    
  if dict_args['filter_class'] is not None:
    dict_data = tsne_tools.filter_dict_features_per_key(dict_data,
                                                        filter_array=np.array(dict_args['filter_class']).astype(int),
                                                        key='list_labels')
  # plot per subject -> plot all videos from a chunk of subjects together
  if 'sbj' in dict_args['plot_type']:
    plot_per_subject(dict_data,dict_args)
  
  # plot per sample binary where dict_args['thresh_binary'] is the threshold to consider for the 2 classes
  if 'sam_bin' in dict_args['plot_type']:
    plot_per_sample_binary(dict_data,dict_args)
  
  # plot per subject binary where dict_args['thresh_binary'] is the threshold to consider for the 2 classes
  # and dict_args['filter_class'] is the class to consider
  if 'sbj_bin' in dict_args['plot_type'] :
    if dict_args['filter_class'] is not None and len(dict_args['filter_class']) == 1:
      plot_per_subject_binary(dict_data,dict_args)
    else:
      print('Plotting per subject binary not possible. Please check the filter_class argument.')

      
def plot_per_subject(dict_data,dict_args,ax=None):
  dict_args['path_save_plot'] = dict_args['path_save_plot'] + '_sbj' + dict_args['labels_key'] + '_max_' + str(dict_args['max_elements_per_plot'])
  X_tsne = dict_data['features'] # [nr_chunks, T, S, S, 2] 
  list_key = dict_data[dict_args['labels_key']]
  limit_key_per_plot = dict_args['max_elements_per_plot']
  unique_key_id = np.unique(list_key)
  # nr_chunks=X_tsne.shape[0]
  temporal_dim = X_tsne.shape[1]
  space_dim = X_tsne.shape[2]
  if temporal_dim > 1:
    list_key = np.repeat(list_key, temporal_dim, axis=0)
  if space_dim > 1:
    list_key = np.repeat(list_key, space_dim*2, axis=0)
  
  print(f'unique_subject_id: {unique_key_id.shape[0]}')
  sbj_chunk_array = np.array_split(unique_key_id, unique_key_id.shape[0]//limit_key_per_plot+1)
  for sbj_chunk in sbj_chunk_array:
    mask = np.isin(list_key, sbj_chunk)
    tools.plot_tsne(X_tsne=X_tsne.reshape(-1,X_tsne.shape[-1])[mask],
                    labels=list_key[mask],
                    saving_path=dict_args['path_save_plot'],
                    title=f'Plot_{sbj_chunk}_all_classes_all_videos',
                    legend_label='Subject ID',
                    ax=ax,
                    cmap='tab20',)
  print(f'Plots saved in {dict_args["path_save_plot"]}')
  save_config_file(dict_args=dict_args)
  
def plot_per_sample_binary(dict_data,dict_args): # do the same for subjects
  unique_sample_id = np.unique(dict_data['list_sample_id'])
  dict_args['path_save_plot'] = dict_args['path_save_plot'] + '_sam_bin' + dict_args['thresh_binary'] + f'stride_{dict_args["stride_dataset"]}' 
  df = pd.read_csv(dict_args['csv_path'],sep='\t')
  for sample_id in tqdm.tqdm(unique_sample_id,desc='Plotting samples'):
    mask = np.isin(dict_data['list_sample_id'], sample_id)
    X = dict_data['features'][mask]
    # new_labels = np.array(dict_data['list_labels'])[mask]
    tot_chunks = np.prod(X.shape[:-1])  # Total number of chunks
    # thresh_binary is cosidered the number of chunks given the feats [num_chunks, T, S, S, 2] -> So if T != 1 or S != 1 I have to change the threshold to be consistent 
    updated_threshold = np.prod([dict_args['thresh_binary'], *X.shape[1:-1]])
    binary_labels = np.array([1 if i >= updated_threshold else 0 for i in range(tot_chunks)])
    sil_score = silhouette_score(X=X.reshape(-1,X.shape[-1]), labels=binary_labels)
    # binary_labels = np.array([dict_data['list_labels'][mask][0] if i >= dict_args['thresh_binary'] else 0 for i in range(tot_chunks)])
    sample_name = dict_data['list_sample_id'][mask][0].item()
    if dict_data['list_sample_id'][mask][0] in df['sample_id'].values:
      sample_name = df[df['sample_id'] == sample_name]['sample_name'].values[0]
    else:
      sample_name = dict_data['list_sample_id'][mask][0]
    title = f"{sample_name}_feat_{list(dict_data['features'][mask].shape[:-1])}"+ \
            ('_stoic_' if dict_data['list_subject_id'][mask][0] in helper.stoic_subjects else "") + \
            f"_sil_score_{sil_score:.4f}"
    # Add labels windows in seconds, consider Biovid FPS=25 and chunks of 16 frames
    stride = dict_args['stride_dataset'] 
    assert stride is not None, "Please provide the stride_dataset argument"
    chunk_time = stride / dict_args['fps_dataset']  # 16 frames at 25 FPS
    list_additiona_desc_legend = [
      f' (0, {chunk_time*dict_args["thresh_binary"]:.1f}) sec',
      f' ({chunk_time*dict_args["thresh_binary"]:.1f}, {chunk_time*X.shape[0]:.0f}) sec',
    ]
    tools.plot_tsne(X_tsne=X.reshape(-1,X.shape[-1]),
                    labels=binary_labels,
                    saving_path=dict_args['path_save_plot'],
                    title=title,
                    legend_label='Movement',
                    list_additiona_desc_legend=list_additiona_desc_legend,
                    tot_labels=2,
                    cmap='bwr')
  print(f'Plots saved in {dict_args["path_save_plot"]}')
  save_config_file(dict_args=dict_args)

def plot_per_subject_binary(dict_data,dict_args):
  unique_subject_id = np.unique(dict_data['list_subject_id'])
  dict_args['path_save_plot'] = dict_args['path_save_plot'] + '_sbj_bin' + f'filter_{dict_args["filter_class"][0]}' + f'_thresh_{dict_args["thresh_binary"]}' + f'_stride_{dict_args["stride_dataset"]}'
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
                    saving_path=dict_args['path_save_plot'],
                    title=title,
                    legend_label='Pain Level',
                    tot_labels=int(dict_args['filter_class'][0]),
                    cmap='tab10')
  print(f'Plots saved in {dict_args["path_save_plot"]}')
  save_config_file(dict_args=dict_args)
  

def save_config_file(dict_args):
  # Save the t-SNE configuration
  os.makedirs(dict_args['path_save_plot'], exist_ok=True)
  with open(os.path.join(dict_args['path_save_plot'], '_config.txt'), 'w') as f:
    f.write("Arguments used for t-SNE:\n")
    for key, value in dict_args.items():
      f.write(f"{key}: {value}\n")
  with open(os.path.join(dict_args['path_save_plot'], '_config.pkl'), 'wb') as f:
    pickle.dump(dict_args, f)
  print(f"Saved configuration file to {dict_args['path_save_plot']}_config.txt and _config.pkl")
    
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Compute tsne plot')
  parser.add_argument('--path_tsne_feat', type=str, help='Path to the folder containing the tsne data extracted')
  parser.add_argument('--path_save_plot', type=str, help='Path to save the tsne plot')
  parser.add_argument('--filter_class', type=str, nargs='*', default=None, help='Filter the data based on the class')
  parser.add_argument('--labels_key', type=str, choices=['list_labels','list_subject_id'],default='list_labels', help='Key to use for labels in the dictionary')
  parser.add_argument('--max_elements_per_plot', type=int, default=20, help='Maximum number of elements per plot')
  parser.add_argument('--thresh_binary',type=int, default=4, help='Threshold chunks for binary classification. Index starting from 0')
  parser.add_argument('--plot_type', type=str, nargs='*', choices=['sbj','sbj_bin','sam_bin'], default=['sbj'], help='Type of plot to generate')
  parser.add_argument('--csv_path', type=str, default="partA/starting_point/samples.csv", help='Path to save the csv file with the t-SNE coordinates')
  parser.add_argument('--stride_dataset', type=int, default=None, help='Stride of the dataset in frames')
  parser.add_argument('--fps_dataset', type=int, default=25, help='Frames per second of the dataset (default: 25)')
  parser.add_argument('--gp', action='store_true', help='Add /equilibrium/fvilli/PainAssessmentVideo to paths')
  args = parser.parse_args()
  dict_args = vars(args)
  # dict_args['path_save_plot'] = os.path.join(dict_args['path_save_plot'], dict_args['plot_type'][0],)
  
  if dict_args['gp']:
    dict_args['path_tsne_feat'] = helper.GLOBAL_PATH.get_global_path(dict_args['path_tsne_feat'])
    dict_args['path_save_plot'] = helper.GLOBAL_PATH.get_global_path(dict_args['path_save_plot'])
    dict_args['csv_path'] = helper.GLOBAL_PATH.get_global_path(dict_args['csv_path'])
  
  # save_config_file(dict_args=dict_args)
  main(dict_args)
  
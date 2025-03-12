import custom.tools as tools
import custom.scripts as scripts
import os
import time
import matplotlib.pyplot as plt
# OK finish the video clip, start to create plot list_same_clip_positions many people
# try to combine plot and video in one
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
import tqdm
import pickle
from collections import defaultdict

def calculate_wcss(embeddings, labels):
  """Within-Cluster Sum of Squares implementation"""
  clusters = np.unique(labels)
  wcss = 0
  for cluster in clusters:
    cluster_points = embeddings[labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    wcss += np.sum((cluster_points - centroid)**2)
  return wcss

def pairwise_distance(embeddings, labels):
  """Mean intra-cluster distance calculation"""
  clusters = np.unique(labels)
  avg_distances = {}
  for cluster in clusters:
    cluster_points = embeddings[labels == cluster]
    if len(cluster_points) > 1:
        distances = pdist(cluster_points)
        avg_distances[cluster] = np.mean(distances)
  return avg_distances

def cluster_radius(embeddings, labels):
  """Maximum distance from centroid implementation"""
  clusters = np.unique(labels)
  radii = {}
  for cluster in clusters:
    cluster_points = embeddings[labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    radii[cluster] = np.max(distances)
    
  return radii

# Built-in metrics from scikit-learn
def get_silhouette_score(embeddings, labels):
    return silhouette_score(embeddings, labels)

def get_davies_bouldin_index(embeddings, labels):
    return davies_bouldin_score(embeddings, labels)
  
def generate_unique_plot(list_tsne_plot_per_folder_feature):
  fig, axs = plt.subplots(1, len(list_tsne_plot_per_folder_feature), figsize=(15, 5))
  for i, ax in enumerate(list_tsne_plot_per_folder_feature):
    axs[i]=ax
  plt.tight_layout()
  plt.savefig(os.path.join(folder_tsne_results,f'plot_merged.png'))
  # return fig,axs

def nested_dict():
  return defaultdict(nested_dict)

# feat_mean = False
# stoic = True
# folder_name = 'tsne_Results_mean' if feat_mean else 'tsne_Results_temporal'
# folder_name = f'{folder_name}_stoic' if stoic else f'{folder_name}_highSensitive'
folder_name = 'tsne_Results_per_subj_new_comp'
folder_tsne_results = os.path.join(folder_name,f'test_{str(int(time.time()))}')
# folder_path_features = os.path.join('partA','video','features','samples_16_frontalized')
list_folder_path_features = [os.path.join('partA','video','features','samples_16_frontalized_new'),
                            #  os.path.join('partA','video','features','samples_16_aligned_cropped'),
                             os.path.join('partA','video','features','samples_16_frontalized')]

if not os.path.exists(folder_tsne_results):
    os.makedirs(folder_tsne_results)

stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
all_subjects = list(range(1,88))
# if stoic:
#   subject_id_list = stoic_subjects
# else:
#   subject_id_list = [sbj for sbj in all_subjects if sbj not in stoic_subjects]
# all_subjects = [list(range(0,20)),list(range(20,40)),list(range(40,60)),list(range(60,80)),list(range(80,88))]


subject_id_list = all_subjects
plot = True

count = 0
# all_subjects = subject_id_list
# for sub in subject_id_list:
list_tsne_plot_per_folder_feature = []  
clip_list = [0,1,2,3,4,5,6,7]
class_list = [4]
# sample_id_list = [84]
dict_results = {}
sliding_windows =  16
status = [False] # feature mean applied
legend_label = 'clip' # can be clip, subject and class
nested_dict_results = nested_dict()
for count_subject_id_list,chunk_sbj in enumerate(subject_id_list):
  dict_chunk_sbj_result={}
  fig, axes = plt.subplots(len(list_folder_path_features), len(status), figsize=(30, 30))
  axes.flatten()
  for j,feat_mean in enumerate(status):
    # feat_mean = apply_mean
    for i,folder_path_features in enumerate(list_folder_path_features):
      dict_plot = tools.plot_and_generate_video(folder_path_features=folder_path_features,
                                      folder_path_tsne_results=folder_tsne_results,
                                      subject_id_list=[chunk_sbj] if isinstance(chunk_sbj,int) else chunk_sbj,
                                      clip_list=clip_list,
                                      legend_label=legend_label,
                                      class_list=class_list,
                                      sliding_windows=sliding_windows,
                                      # plot_only_sample_id_list=sample_id_list,
                                      plot_third_dim_time=False,
                                      create_video=False,
                                      apply_pca_before_tsne=True,
                                      tsne_n_component=2,
                                      sort_elements=True,
                                      cmap='tab20',
                                      feat_mean=feat_mean,
                                      csv_path='partA/starting_point/samples_exc_no_detection.csv'
                                      ) # copper, tab20
      folder_type_feat = folder_path_features.split('/')[-1]
      dict_plot['saving_path'] = None
      if legend_label == 'clip':
        # deep copy the labels
        labels = dict_plot['labels'].copy()
        
        bool_mask = labels >= 3
        labels[bool_mask] = 0  # set 0 for the first 3 clips (no pain)
        labels[~bool_mask] = 7 # set 1 for the others (pain)
        sil_score = silhouette_score(dict_plot['X_tsne'], labels)
        wcss_score = calculate_wcss(dict_plot['X_tsne'], labels)
        db_score = davies_bouldin_score(dict_plot['X_tsne'], labels)
        dict_plot['cmap'] = 'brg' 
        nested_dict_results[chunk_sbj][f'mean' if feat_mean else 'temporal'][folder_type_feat] = {'sil_score':sil_score,'wcss_score':wcss_score,'db_score':db_score,'cls':class_list,'clip':clip_list}
        # dict_chunk_sbj_result[f'{"mean" if feat_mean else "temporal"}_{folder_type_feat}'] = {'sil_score':sil_score,'wcss_score':wcss_score,'db_score':db_score}
        # dict_subject_result[f'{folder_type_feat}'] = {'sil_score':sil_score,'wcss_score':wcss_score,'db_score':db_score}
        # dict_results[chunk_sbj] = {f'{folder_type_feat}':{'sil_score':sil_score,'wcss_score':wcss_score,'db_score':db_score}}
        cluster_measure = [sil_score]
        # ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=color_dict[0 if stride_windows*val < 48 else max(color_dict.keys())], label=label, alpha=0.7, s=sizes[idx] if sizes is not None else 50)
      else:
        # wcss = calculate_wcss(dict_plot['X_tsne'], dict_plot['labels'])
        # sil_score = silhouette_score(dict_plot['X_tsne'], dict_plot['labels'])
        # db_score = davies_bouldin_score(dict_plot['X_tsne'], dict_plot['labels'])
        pairwise_distances = pairwise_distance(dict_plot['X_tsne'], dict_plot['labels'])
        cluster_measure = pairwise_distances
        # radii = cluster_radius(dict_plot['X_tsne'], dict_plot['labels'])
        for k,v in pairwise_distances.items():
          nested_dict_results[f'group_{count_subject_id_list}'][f'id_{k}']['mean' if feat_mean else 'temporal'][folder_type_feat] = v
      if feat_mean:
        dict_plot['title'] = f'MEAN_{dict_plot["title"]}'
      # else:
      #   dict_plot['title'] = f'{dict_plot["title"]}'
      if chunk_sbj in stoic_subjects:
        dict_plot['title'] = f'STOIC_{dict_plot["title"]}'
      if plot:
        tools.plot_tsne(**dict_plot,
                        ax=axes[i*len(status)+j],
                        return_ax=True,
                        cluster_measure=cluster_measure)
      print(f'Processed {i+1}/{len(list_folder_path_features)}')
      # list_tsne_plot_per_folder_feature.append(tools.plot_tsne(**dict_plot,ax=axes[i],return_ax=True))
    # plt.savefig(os.path.join(folder_tsne_results,f'plot_merged.png'))
    
    #save plot
  if legend_label == 'clip':
    dict_results[chunk_sbj] = dict_chunk_sbj_result
  plot_name = f'STOIC_plot_{chunk_sbj}.png' if chunk_sbj in stoic_subjects else f'plot_{chunk_sbj}.png'
  plot_name = f'MEAN_{plot_name}' if feat_mean else plot_name
  plt.savefig(os.path.join(folder_tsne_results,plot_name))
  plt.close(fig)
  count+=1
  # print(f'Processed {count}/{len(subject_id_list)}')
  print(f'Processed {count}/{len(subject_id_list)}')
  
# save configuration in folder_tsne_results
if dict_results:
  with open(os.path.join(folder_tsne_results,'dict_results.pkl'),'wb') as f:
    pickle.dump(dict_results,f)
  print(f'Dict results saved in {os.path.join(folder_tsne_results,"dict_results.pkl")}')
if nested_dict_results:
  with open(os.path.join(folder_tsne_results,'nested_dict_results.pkl'),'wb') as f:
    pickle.dump(nested_dict_results,f)
  print(f'Nested dict results saved in {os.path.join(folder_tsne_results,"nested_dict_results.pkl")}')
  
with open(os.path.join(folder_tsne_results,'config.txt'),'w') as f:
  f.write(f'subject_id_list: {all_subjects}\n')
  f.write(f'clip_list: {clip_list}\n')
  f.write(f'class_list: {class_list}\n')
  f.write(f'sliding_windows: {sliding_windows}\n')
  f.write(f'legend_label: {legend_label}\n')
  f.write(f'plot_third_dim_time: False\n')
  f.write(f'apply_pca_before_tsne: True\n')
  f.write(f'tsne_n_component: 2\n')
  f.write(f'sort_elements: True\n')
  f.write(f'cmap: copper\n')
  f.write(f'folder_path_features: {list_folder_path_features}\n')
  f.write(f'folder_path_tsne_results: {folder_tsne_results}\n')
  f.write(f'create_video: False\n')
print(f'Configuration saved in {os.path.join(folder_tsne_results,"config.txt")}')
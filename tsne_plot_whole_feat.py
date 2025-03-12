import custom.tools as tools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tqdm
import os
import time
def generate_groups(csv_path, subjects_per_group=20,pain_classes = [0,1,2,3,4]):
  """Generate a group of subjects from the dataframe"""
  df = pd.read_csv(csv_path,sep='\t')
  df = df[df['class_id'].isin(pain_classes)]
  df_filtered = df.filter(['subject_name','sample_name'])
  dict_results_per_sbj = {}
  list_subjects = df_filtered['subject_name'].unique().tolist()
  for sbj in list_subjects:
    dict_results_per_sbj[sbj] = df_filtered[df_filtered['subject_name'] == sbj]['sample_name'].tolist()
  
  return dict_results_per_sbj
 
def perform_tsne_analysis_whole(dict_results_per_sbj,dict_pooling, root_folder_feature, folder_saving_path):
  list_records = []
  for subject,list_sample in tqdm.tqdm(dict_results_per_sbj.items()):
    list_features = []
    list_labels = []
    list_count_frame = []
    
    for sample in list_sample:
      feature = torch.load(root_folder_feature / subject / sample / 'features.pt')
      labels = torch.load(root_folder_feature / subject / sample / 'list_labels.pt')
      count_frame = torch.arange(0,feature.shape[0])
      list_features.append(feature)
      list_labels.append(labels)
      list_count_frame.append(count_frame)
    X = torch.concat(list_features,dim=0)
    if dict_pooling is not None:
      if dict_pooling['type'] == 'mean':
        X = X.mean(dim=dict_pooling['dim'],keepdim=True)
      elif dict_pooling['type'] == 'max':
        X = X.max(dim=dict_pooling['dim'],keepdim=True)
      elif dict_pooling['type'] == 'sum':
        X = X.sum(dim=dict_pooling['dim'],keepdim=True)
      else:
        raise ValueError(f'Unknown pooling type: {dict_pooling["type"]}. Can be mean, max, or sum')
      
    labels = torch.concat(list_labels,dim=0)
    count_frame = torch.concat(list_count_frame,dim=0)
    labels[count_frame <= 2] = 0 # no movement
    labels[count_frame >= 3] = 1 # movement
    # Apply the t-SNE
    
    X_tsne = tools.compute_tsne(X=X,
                                labels=labels,
                                apply_pca_before_tsne=True,
                                tsne_n_component=2)
    sil_score = tools.silhouette_score(X_tsne, labels)
    wcss_score = tools.calculate_wcss(X_tsne, labels)
    db_score = tools.davies_bouldin_score(X_tsne, labels)
    list_records.append({'subject':subject,
                         'pooling_type':dict_pooling['type'] if dict_pooling is not None else None,
                         'pooling_dim':dict_pooling['dim'] if dict_pooling is not None else None,
                         'shape_features':X.shape,
                         'pain_class': labels.unique().tolist(),
                         'sil_score':sil_score,
                         'wcss_score':wcss_score,
                         'db_score':db_score})
    # tools.plot_tsne(X_tsne=X_tsne,
    #                 labels=labels,
    #                 title=f'{subject}----sil_{sil_score:.2f}_wcss_{wcss_score:.2f}_db_{db_score:.2f}',
    #                 saving_path=folder_saving_path,
    #                 cmap='brg',
    #                 legend_label='clip')  
  return list_records

if __name__ == '__main__':
  # Load the data
  stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
  start = time.time()
  root_folder_feature = Path('/media/villi/TOSHIBA EXT/samples_16_whole')
  csv_path = Path('partA/starting_point/samples_exc_no_detection.csv')
  dict_groups = generate_groups(csv_path, pain_classes=[4])
  list_test = []
  df = pd.read_csv(csv_path,sep='\t')
  filter_df = df.filter(['subject_name','subject_id']).drop_duplicates()
  filter_df['stoic'] = filter_df['subject_id'].isin(stoic_subjects)
  filter_df.to_csv('tsne_whole/stoic.csv',index=False)
  folder_saving_path = 'tsne_whole'
  list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
                                                   root_folder_feature=root_folder_feature,
                                                   folder_saving_path=folder_saving_path,
                                                   dict_pooling=None, # shape [n_samples,temporal,space,space,embedding]
                                                   ))
  list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
                                                   root_folder_feature=root_folder_feature,
                                                   folder_saving_path=folder_saving_path,
                                                   dict_pooling={'type':'mean','dim':(2,3)},
                                                   ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'sum','dim':(2,3)},
  #                                                  ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'max','dim':(2,3)},
  #                                                  ))
  list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
                                                   root_folder_feature=root_folder_feature,
                                                   folder_saving_path=folder_saving_path,
                                                   dict_pooling={'type':'mean','dim':1},
                                                   ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'sum','dim':1},
  #                                                  ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'max','dim':1},
  #                                                  ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'mean','dim':(1,2,3)},
  #                                                  ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'sum','dim':(1,2,3)},
  #                                                  ))
  # list_test.append(perform_tsne_analysis_whole(dict_results_per_sbj=dict_groups,
  #                                                  root_folder_feature=root_folder_feature,
  #                                                  folder_saving_path=folder_saving_path,
  #                                                  dict_pooling={'type':'max','dim':(1,2,3)},
  #                                                  ))
  merged_records = {}
  for test in list_test:
    for record in test:
      if record['subject'] not in merged_records:
        merged_records[record['subject']] = {'subject':record['subject'],'pain_class':record['pain_class']}
      merged_records[record['subject']].update({
        f'sil_score_{record["pooling_type"]}_{record["pooling_dim"]}':record['sil_score'],
        f'wcss_score_{record["pooling_type"]}_{record["pooling_dim"]}':record['wcss_score'],
        f'db_score_{record["pooling_type"]}_{record["pooling_dim"]}':record['db_score'],
        f'shape_features_{record["pooling_type"]}_{record["pooling_dim"]}':record['shape_features']
      })
  # Order the records putting the shape features last
  # group sil_score, wcss_score, db_score
  columns = list(merged_records.values())[0].keys()
  columns = ['subject'] + \
            [column for column in columns if 'sil_score' in column] + \
            [column for column in columns if 'wcss_score' in column] + \
            [column for column in columns if 'db_score' in column] + \
            [column for column in columns if 'shape_features' in column] + \
            ['pain_class']
  merged_records = [{key:record[key] for key in columns} for record in merged_records.values()]
  # Save the records
  df = pd.DataFrame(merged_records)
  os.makedirs(folder_saving_path,exist_ok=True)
  df.to_csv(os.path.join(folder_saving_path,'results.csv'),index=False)
  end = time.time()
  print(f'Elapsed time: {np.round((end-start)/3600,2)} hours')
  print(f'Results saved in {os.path.join(folder_saving_path,"results.csv")}')
  
                              
  
  


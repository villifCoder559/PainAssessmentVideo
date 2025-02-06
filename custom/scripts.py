from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY,GLOBAL_PATH
from custom.model import Model_Advanced
from transformers import AutoImageProcessor
from custom.head import HeadSVR, HeadGRU
import time
from custom.tools import NpEncoder
import custom.tools as tools
from custom.helper import HEAD, MODEL_TYPE, EMBEDDING_REDUCTION, SAMPLE_FRAME_STRATEGY
import torch.nn as nn
import torch.optim as optim
import dataframe_image as dfi
import pandas as pd
import torch
import numpy as np
import time
import json
from sklearn.model_selection import StratifiedGroupKFold
import os
# import wandb

def get_dict_all_features_from_model(sliding_windows,subject_id_list,classes,folder_path_tsne_results):
  def get_model_advanced(stride_window_in_video=16):
    model_type = MODEL_TYPE.VIDEOMAE_v2_S
    pooling_embedding_reduction = EMBEDDING_REDUCTION.MEAN_SPATIAL
    pooling_clips_reduction = CLIPS_REDUCTION.NONE
    sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
    # path_dict ={
    #   'all' : os.path.join('partA','starting_point','samples.csv'),
      # 'train' : os.path.join('partA','starting_point','train_21.csv'),
      # 'val' : os.path.join('partA','starting_point','val_26.csv'),
      # 'test' : os.path.join('partA','starting_point','test_5.csv')
    # }
    path_dataset = os.path.join('partA','video','video')  
    path_cvs_dataset = os.path.join('partA','starting_point','samples.csv')
    head = HEAD.GRU
    # if head == 'GRU':
    params = {
      'hidden_size': 1024,
      'num_layers': 1,
      'dropout': 0.0,
      'input_size': 384 * 8 # can be 384  (small), 768  (base), 1408  (large) [temporal_dim considered as input sequence for GRU]
                        # can be 384*8(small), 768*8(base), 1408*8(large) [temporal_dim considered feature in GRU] 
    }
    
    preprocess = AutoImageProcessor.from_pretrained(os.path.join("local_model_directory","preprocessor_config.json"))
    features_folder_saving_path = os.path.join('partA','video','features',f'{os.path.split(path_cvs_dataset)[-1][:-4]}_{stride_window_in_video}') # get the name of the csv file
    
    model_advanced = Model_Advanced(model_type=model_type,
                              path_dataset=path_dataset,
                              embedding_reduction=pooling_embedding_reduction,
                              clips_reduction=pooling_clips_reduction,
                              sample_frame_strategy=sample_frame_strategy,
                              stride_window=stride_window_in_video,
                              path_labels=path_cvs_dataset,
                              preprocess=preprocess,
                              batch_size_training=1,
                              batch_size_feat_extraction=1,
                              head=head.value,
                              head_params=params,
                              download_if_unavailable=False,
                              features_folder_saving_path=features_folder_saving_path
                              )
    return model_advanced
  
  model_advanced = get_model_advanced(stride_window_in_video=sliding_windows)
  csv_array,cols = tools.get_array_from_csv(model_advanced.dataset.path_labels) # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  print(csv_array)
  idx_subjects = np.any([csv_array[:,0].astype(int) == id for id in subject_id_list],axis=0)
  idx_classes = np.any([csv_array[:,2].astype(int) == id for id in classes],axis=0)
  new_csv_array = csv_array[np.logical_and(idx_subjects,idx_classes)]
  print(f'new_csv_array {new_csv_array.shape}')
  path_new_csv=tools.save_csv_file(cols=cols,
                      csv_array=new_csv_array,
                      saving_path=folder_path_tsne_results,
                      sliding_windows=sliding_windows)
  dict_all_features = model_advanced.extract_features(csv_path=path_new_csv,read_from_memory=False)
  dict_all_features['features'] = dict_all_features['features'].detach().cpu()
  return dict_all_features

# TODO: select the right stride window for each video when read data from SSD
def plot_and_generate_video(folder_path_features,folder_path_tsne_results,subject_id_list,clip_list,class_list,sliding_windows,legend_label,create_video=True,
                            plot_only_sample_id_list=None,tsne_n_component=2,plot_third_dim_time=False,apply_pca_before_tsne=False,cmap='copper',
                            sort_elements=True,axis_dict=None):

  # if sliding_windows != 16 and sliding_windows!=4:
  #   dict_all_features = get_dict_all_features_from_model(sliding_windows=sliding_windows,
  #                                                        classes=class_list,
  #                                                        subject_id_list=subject_id_list,
  #                                                        folder_path_tsne_results=folder_path_tsne_results)
  # else:
  dict_all_features = tools.load_dict_data(folder_path_features)
  # print(dict_all_features.keys())
  print(f'dict_all_features["list_subject_id"] shape {dict_all_features["list_subject_id"].shape}')
  time_start = time.time()
  idx_subjects = np.any([dict_all_features['list_subject_id'] == id for id in subject_id_list],axis=0)
  idx_class = np.any([dict_all_features['list_labels'] == id for id in class_list],axis=0)
  filter_idx = np.logical_and(idx_subjects,idx_class)
  if plot_only_sample_id_list is not None:
    print(f'Warning: Using sample id will ignore subject_id_list and class_list')
    filter_idx = np.any([dict_all_features['list_sample_id'] == id for id in plot_only_sample_id_list],axis=0)
  # Filter for clip_list
  _, list_count_clips = np.unique(dict_all_features['list_sample_id'],return_counts=True)
  arange_clip = range(max(list_count_clips)) # suppose clip_list is ordered
  clip_list_array = np.array([True if i in clip_list else False for i in arange_clip]) 
  filter_clip = np.concatenate([clip_list_array[:end] for end in list_count_clips])
  filter_idx = np.logical_and(filter_idx, filter_clip)
  
  list_frames = []
  list_sample_id = []
  list_subject_id = []
  list_video_path = []
  list_feature = []
  list_idx_list_frames = []
  list_y_gt = []

  list_frames=dict_all_features['list_frames'][filter_idx]
  list_sample_id=dict_all_features['list_sample_id'][filter_idx]
  list_video_path=dict_all_features['list_path'][filter_idx]
  list_feature=dict_all_features['features'][filter_idx]
  # print(f'length list_feature {len(list_feature)}')
  list_y_gt=dict_all_features['list_labels'][filter_idx]
  list_subject_id=dict_all_features['list_subject_id'][filter_idx]
  # print(f'list_sample_id {list_sample_id}')
  list_idx_list_frames=np.concatenate([np.arange(end) for end in list_count_clips])[filter_idx]
  
  if sort_elements:
    class_bool_idxs = [list_y_gt == i for i in class_list]
    list_frames = torch.cat([list_frames[bool_idx] for bool_idx in class_bool_idxs])
    list_sample_id = torch.cat([list_sample_id[bool_idx] for bool_idx in class_bool_idxs])
    list_video_path = np.concatenate([list_video_path[bool_idx] for bool_idx in class_bool_idxs])
    list_feature = torch.cat([list_feature[bool_idx] for bool_idx in class_bool_idxs])
    list_y_gt = torch.cat([list_y_gt[bool_idx] for bool_idx in class_bool_idxs])
    list_subject_id = torch.cat([list_subject_id[bool_idx] for bool_idx in class_bool_idxs])
    list_idx_list_frames = np.concatenate([list_idx_list_frames[bool_idx] for bool_idx in class_bool_idxs])
  
  print('Elasped time to get all features: ',time.time()-time_start)
  print(f'list_frames {list_frames.shape}')
  print(f'list_sample_id {list_sample_id.shape}')
  print(f'list_video_path {list_video_path.shape}')
  print(f'list_feature {list_feature.shape}')
  print(f'list_idx_list_frames {list_idx_list_frames.shape}')
  print(f'list_y_gt {list_y_gt.shape}')
  
  tsne_plot_path = os.path.join(folder_path_tsne_results,f'tsne_plot_{sliding_windows}_{legend_label}')
  
  X_tsne = tools.compute_tsne(X=list_feature,
                           plot=False,
                           saving_path=os.path.join(folder_path_tsne_results,'dummy'),
                           tsne_n_component=tsne_n_component,
                           apply_pca_before_tsne=apply_pca_before_tsne)
  # add 3th dimension to X_tsne
  list_axis_name = None
  if tsne_n_component == 2 and plot_third_dim_time:
    X_tsne = np.concatenate([X_tsne,np.expand_dims(list_idx_list_frames,axis=1)],axis=1)
    X_tsne = X_tsne[:,[2,0,1]] 
    list_axis_name = ['nr_clip','t-SNE_x','t-SNE_y']
  if axis_dict is None:
    if X_tsne.shape[1] == 2:
      min_x,min_y = X_tsne.min(axis=0)
      max_x,max_y = X_tsne.max(axis=0)
      axis_dict = {'min_x':min_x-3,'min_y':min_y-3,'max_x':max_x+3,'max_y':max_y+3}
    else:
      min_x,min_y,min_z = X_tsne.min(axis=0)
      max_x,max_y,max_z = X_tsne.max(axis=0)
      axis_dict = {'min_x':min_x-3,'min_y':min_y-3,'min_z':min_z-3,'max_x':max_x+3,'max_y':max_y+3,'max_z':max_z+3}
  
  print(f'axis_dict {axis_dict}')
  if legend_label == 'clip':
    labels_to_plot = list_idx_list_frames
  elif legend_label == 'subject':
    labels_to_plot = list_subject_id
  elif legend_label == 'class':
    labels_to_plot = list_y_gt
  else:
    raise ValueError('legend_label must be one of the following: "clip", "subject", "class"') 
  
  # labels_to_plot = list_idx_list_frames
  # print(f'clip_length {dict_all_features["list_frames"][filter_idx].shape[1]}')
  if not os.path.exists(tsne_plot_path):
    os.makedirs(tsne_plot_path)
  if plot_only_sample_id_list is None:
    if len(subject_id_list) > 1:
      title_plot = f'sliding_{sliding_windows}_tot-subjects_{len(subject_id_list)}__clips_{clip_list}__classes_{(class_list)}'
    else:
      title_plot = f'sliding_{sliding_windows}__clips_{clip_list}__classes_{(np.unique(list_y_gt))}__subjectID_{np.unique(list_subject_id)}'
  else:
    title_plot = f'sliding_{sliding_windows}_sample_id_{plot_only_sample_id_list}__clips_{len(clip_list)}__classes_{(np.unique(list_y_gt))}__subjectID_{np.unique(list_subject_id)}'
  print('START ONLY PLOT TSNE')
  tools.plot_tsne(X_tsne=X_tsne,
                       labels=labels_to_plot,
                       saving_path=tsne_plot_path,
                       title=title_plot,
                       legend_label=legend_label,
                       plot_trajectory = True if plot_only_sample_id_list is not None else False,
                       clip_length=dict_all_features['list_frames'][filter_idx].shape[1],
                       stride_windows=sliding_windows,
                       axis_scale=axis_dict,
                       list_axis_name=list_axis_name,
                       cmap=cmap)  
  print('END ONLY PLOT TSNE')
  with open(os.path.join(folder_path_tsne_results,'config.txt'),'w') as f:
    f.write(f'subject_id_list: {subject_id_list}\n')
    f.write(f'clips: {clip_list}\n')
    f.write(f'classes: {class_list}\n')
    f.write(f'sliding_windows: {sliding_windows}\n')
  if create_video:
    video_saving_path = os.path.join(folder_path_tsne_results,'video')
    if not os.path.exists(video_saving_path):
      os.makedirs(video_saving_path)
    list_rgb_image_plot = []  
    start = time.time()
    print(f'X_tsne.shape {X_tsne.shape}')
    for i in range(1,X_tsne.shape[0]+1):
      list_rgb_image_plot.append(
                    tools.plot_tsne(X_tsne=X_tsne[:i],
                          labels=labels_to_plot[:i],
                          legend_label=legend_label,
                          title=f'{title_plot}_{i}',
                          # saving_path=video_saving_path,
                          axis_scale=axis_dict,
                          clip_length=dict_all_features['list_frames'][filter_idx].shape[1],
                          stride_windows=sliding_windows,
                          tot_labels=len(np.unique(labels_to_plot)),
                          plot_trajectory=True if plot_only_sample_id_list is not None else False,
                          last_point_bigger=True,
                          list_axis_name=list_axis_name))
      print(f'Elapsed time to get plot {i}: {time.time()-start} s')
    print(f'Elapsed time to get all plots: {time.time()-start} s')
    start = time.time()
    tools.generate_video_from_list_video_path(list_video_path=list_video_path,
                                              list_frames=list_frames,
                                              list_sample_id=list_sample_id,
                                              list_y_gt=list_y_gt,
                                              output_fps=10,
                                              list_subject_id=list_subject_id,
                                              idx_list_frames=list_idx_list_frames,
                                              saving_path=video_saving_path,
                                              list_rgb_image_plot=list_rgb_image_plot)
    print(f'Elapsed time to generate video: {time.time()-start} s')
    # for i in range(len(list_image_path)):
    #   remove_plot(list_image_path[i])

#ATTENTION: This function is old, use plot_and_generate_video instead  
def plot_tsne_per_subject(folder_path_features,folder_tsne_results):
  print('Loading features from SSD...')
  sliding_windows = 16
  # if sliding_windows == 16:
  dict_all_features = tools.load_dict_data(folder_path_features)
  # else:
  #   dict
  # TODO: select the right stride window for each video
  
  subject_id_list = [5]
  idx_subjects = np.any([dict_all_features['list_subject_id'] == id for id in subject_id_list],axis=0)
  # print(f'idx_all {dict_all_features["list_path"][idx_subjects]}')
  
  # for k,v in dict_all_features.items():
  #   print(f'key {k} - shape {v.shape}')
  for id in subject_id_list:
    idx_subject = np.where(dict_all_features['list_subject_id'] == id)
    subject_dict = {k:v[idx_subject] for k,v in dict_all_features.items()}
    folder_tsne_results = os.path.join(folder_tsne_results,f'personID_{id}')
    if not os.path.exists(folder_tsne_results):
      os.makedirs(folder_tsne_results)
      
    # plot per class given a subject
    for cls in np.unique(subject_dict['list_labels']):
      title = f"personID-{id}-sampleID-ALL_gt-{cls}"
      # if not os.path.exists(os.path.join(folder_tsne_results,title)):
      #   os.makedirs(os.path.join(folder_tsne_results,title))  
      idx_cls = np.where(subject_dict['list_labels'] == cls)
      
      X_tsne = tools.compute_tsne(X=subject_dict['features'][idx_cls],
                      labels=subject_dict['list_labels'][idx_cls],
                      apply_pca_before_tsne=False,
                      saving_path=os.path.join(folder_tsne_results,'dummy'), # saving_path log file considers the parent folder
                      plot=False)
      
      # tools.only_plot_tsne(X_tsne=X_tsne,
      #                      labels=subject_dict['list_labels'][idx_cls],
      #                      saving_path=folder_tsne_results,
      #                      title=title,
      #                      legend_label='gt ')
      unique_sample_id,nr_clips_per_video = np.unique(subject_dict['list_sample_id'][idx_cls],return_counts=True)
      concat_nr_clips_per_video = np.concatenate([np.arange(nr_clips_per_video[i]) for i in range(nr_clips_per_video.shape[0])])
      tools.plot_tsne(X_tsne=X_tsne,
                           labels=concat_nr_clips_per_video,
                           saving_path=folder_tsne_results,
                           title=title,
                           legend_label='clip ')
    
    # plot per one video given a subject
    for i in range(subject_dict['features'].shape[0]):
      title = f"personID-{id}_sampleID-{subject_dict['list_sample_id'][i]}_gt-{subject_dict['list_labels'][i]}"
      if not os.path.exists(os.path.join(folder_tsne_results,f'{title}')):
        os.makedirs(os.path.join(folder_tsne_results,f'{title}'))
      nr_clips = subject_dict['features'][i].shape[0]
      X_tsne = tools.compute_tsne(X=subject_dict['features'][i],
                      labels=subject_dict['list_labels'][i],
                      apply_pca_before_tsne=False,
                      saving_path=os.path.join(folder_tsne_results,'dummy'), # saving_path log file considers the parent folder
                      legend_label='gt ',
                      title=title,
                      plot=False)
      
      tools.plot_tsne(X_tsne=X_tsne,
                           labels=range(nr_clips),
                           saving_path=folder_tsne_results,
                           title=title,
                           legend_label='clip ')
      break
    break
  
def run_train_test(model_type, pooling_embedding_reduction, pooling_clips_reduction, sample_frame_strategy, 
                   path_csv_dataset, path_video_dataset, head, stride_window_in_video, features_folder_saving_path,head_params,
                   k_fold=1, is_save_features_extracted=False,
                   global_foder_name=os.path.join(GLOBAL_PATH.NAS_PATH,'history_run'),
                   train_size = 0.8, val_size=0.1, test_size=0.1, is_download_if_unavailable=False, 
                   batch_size_training = 1, batch_size_feat_extraction = 3,epochs = 10, 
                   criterion = nn.L1Loss(), optimizer_fn = optim.Adam, lr = 0.001,random_state_split_dataset=42,
                   is_plot_dataset_distribution=True,is_plot_loss=True,is_plot_tsne_backbone_feats=True,is_plot_tsne_head_pred=True,
                   is_plot_tsne_gru_feats=True,is_create_video_prediction=True,is_create_video_prediction_per_video=True,is_validation=False,
                   is_round_output_loss=False, is_shuffle_video_chunks=True,is_shuffle_training_batch=True,only_train=True,
                   init_network='default',
                   regularization_lambda=0.1,regularization_loss='L2'):

  def plot_dataset_distribuition(csv_path,run_folder_path,per_class=True,per_partecipant=True,total_classes=None):
    #  Create folder to save the dataset distribution 
    dataset_folder_path = os.path.join(run_folder_path,'dataset') #  history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_{timestamp}/dataset
    if not os.path.exists(dataset_folder_path):
      os.makedirs(os.path.join(run_folder_path,'dataset'))

    # Plot all the dataset distribution 
    tools.plot_dataset_distribution(csv_path=csv_path,
                                    per_class=per_class, 
                                    per_partecipant=per_partecipant,
                                    saving_path=dataset_folder_path,
                                    total_classes=total_classes) # 1  plot
    # TODO: Remove comments to plot the mean and std of the dataset
    # tools.plot_dataset_distribution_mean_std_duration(csv_path=csv_path,
    #                                                   video_path=path_video_dataset,
    #                                                   per_class=per_class, 
    #                                                   per_partecipant=per_partecipant, 
    #                                                   saving_path=dataset_folder_path) # 2 plots

  def train_model(dict_csv_path, train_folder_saving_path, is_validation=False,round_output_loss=False,shuffle_video_chunks=True,
                  shuffle_training_batch=True,init_network='default',regularization_lambda=0.0,regularization_loss='L2'):
    """
    Trains a model using the specified dataset and saves the results.\n
    Args:
      dict_csv_path (dict): Path to the train, test, validation csv.
      train_folder_saving_path (str): Path to the folder where training results will be saved.
    Returns:
      dict: A dictionary containing:
          'dict_results': {
                          - 'train_losses': List of training losses.
                          - 'train_loss_per_class': Training loss per class, reshaped to (1, -1).
                          - 'train_loss_per_subject': Training loss per subject, reshaped to (1, -1).
                          - 'test_losses': List of test losses.
                          - 'test_loss_per_class': Test loss per class, reshaped to (1, -1).
                          - 'test_loss_per_subject': Test loss per subject, reshaped to (1, -1).
                          - 'subject_ids_unique': Unique subject IDs in the combined training and test subject IDs.
                          - 'y_unique': Unique classes in the combined training and test labels.
                          - 'best_model_idx': best_model_epoch
                          }
          - 'count_y_train': Count of unique classes in the training set.
          - 'count_y_test': Count of unique classes in the testing set.
          - 'count_subject_ids_train': Count of unique subject IDs in the training set.
          - 'count_subject_ids_test': Count of unique subject IDs in the testing set.
    """
    if is_validation:
      print("training using validation set")
      test_csv_path = dict_csv_path['val']
    else:
      print("training using test set")
      test_csv_path = dict_csv_path['test']
    # Train the model  
    dict_train = model_advanced.train(train_csv_path=dict_csv_path['train'],
                                      test_csv_path=test_csv_path,
                                      num_epochs=epochs, 
                                      criterion=criterion,
                                      optimizer_fn=optimizer_fn,
                                      lr=lr,
                                      saving_path=train_folder_saving_path,
                                      round_output_loss=round_output_loss,
                                      shuffle_video_chunks=shuffle_video_chunks,
                                      shuffle_training_batch=shuffle_training_batch,
                                      init_network=init_network,
                                      regularization_lambda=regularization_lambda,
                                      regularization_loss=regularization_loss
                                      )
    return dict_train  
  
  def extract_and_save_all_features(csv_path,features_folder_saving_path):
    # features_folder_saving_path = os.path.join(features_folder_saving_path, path_csv_dataset[:-4])
    if not os.path.exists(features_folder_saving_path):
      os.makedirs(features_folder_saving_path)
    # get the name of the csv file
    if not os.path.exists(features_folder_saving_path):
      os.makedirs(features_folder_saving_path)
    dict_data = model_advanced.extract_features(csv_path=csv_path)
    # print(f' dict_data {dict_data}')
    tools.save_dict_data(dict_data=dict_data, saving_folder_path=features_folder_saving_path)
  
  def k_fold_cross_validation(path_to_extracted_features,is_validation=False,round_output_loss=False,shuffle_video_chunks=True,shuffle_training_batch=True,
                            init_network='default',regularization_lambda=0.01,regularization_loss='L2'):
    """
    Perform k-fold cross-validation on the dataset and train the model.
    This function performs k-fold cross-validation by splitting the dataset into k folds,
    training the model on each fold, and evaluating the performance. It saves the results
    and plots for each fold, and calculates the average performance metrics across all folds.
    Returns:
      dict: A dictionary containing the following keys:
        - 'fold_results': A list of dictionaries with training results for each fold.
        - 'list_saving_paths': A list of paths where results for each fold are saved.
        - 'list_path_csv_kth_fold': A list of dictionaries with paths to CSV files for each fold.
        - 'best_results_idx': A list of indices indicating the best model for each fold.
    """
    # list_saving_paths = []
    # list_path_csv_kth_fold = []
    csv_array,cols = tools.get_array_from_csv(path_csv_dataset)
    sgkf = StratifiedGroupKFold(n_splits=k_fold, random_state=random_state_split_dataset,shuffle=True)
    y_labels = csv_array[:,2].astype(int)
    subject_ids = csv_array[:,0].astype(int)
    sample_ids = csv_array[:,4].astype(int)
    # print(f' y_labels {y_labels}')
    # print(f' subject_ids {subject_ids}')
    # check is csv array and subject_ids are aligned
    list_splits_idxs = [] # contains indices for all k splits
    check_intersection = []
    for _, test_index in sgkf.split(X=torch.zeros(y_labels.shape), y=y_labels, groups=subject_ids): 
      list_splits_idxs.append(test_index)
      check_intersection.append(subject_ids[test_index])
    # Split check
    for i in range(len(check_intersection)):
      for j in range(i+1,len(check_intersection)):
        print(f'check intersection {i} and {j}')
        if i != j:
          if len(np.intersect1d(check_intersection[i],check_intersection[j])) > 0:
            raise ValueError(f'Error in splitting dataset: intersection between group {i} and group {j}')
    del check_intersection
    list_test_results = []
    list_best_model_idx = []
    dict_results_model_weights = {}
    fold_results_total = []
    dict_k_fold_logs = {}
    for i in range(k_fold):
      fold_results_kth = []
      test_idx_split = i % k_fold
      val_idx_split = (i + 1) % k_fold
      train_idxs_split = [j for j in range(k_fold) if j != test_idx_split and j != val_idx_split]
      test_sample_ids = sample_ids[list_splits_idxs[test_idx_split]] # video sample_id list for test
      val_sample_ids = sample_ids[list_splits_idxs[val_idx_split]] # video sample_id list for validation
      train_sample_ids = []
      for idx in train_idxs_split:
        train_sample_ids.extend(sample_ids[list_splits_idxs[idx]]) # video sample_id list for training
      
      split_indices = {
        'test': np.array([test_sample_ids,list_splits_idxs[test_idx_split]]),# [sample_ids,indices] for test
        'val': np.array([val_sample_ids,list_splits_idxs[val_idx_split]]), # [sample_ids,indices] for validation
        'train': np.array([train_sample_ids,np.concatenate([list_splits_idxs[idx] for idx in train_idxs_split])])# [sample_ids,indices] for training
      }
      saving_path_kth_fold = os.path.join(train_folder_path,f'k{i}_cross_val')
      if not os.path.exists(saving_path_kth_fold):
        os.makedirs(saving_path_kth_fold)
      # Generate csv for train,test and validation
      path_csv_kth_fold = {}
      for key,v in split_indices.items(): # [train,val,test]
        csv_data = csv_array[v[1]] # get the data from .csv for the split in the kth fold
        tools.generate_csv(cols=cols, data=csv_data, saving_path=os.path.join(saving_path_kth_fold,f'{key}.csv'))
        path_csv_kth_fold[key] = os.path.join(saving_path_kth_fold,f'{key}.csv')
      tools.save_split_indices(split_indices,saving_path_kth_fold)
      
      for _,csv_path in path_csv_kth_fold.items():
        plot_dataset_distribuition(csv_path=csv_path, 
                                   run_folder_path=saving_path_kth_fold,
                                   total_classes=model_advanced.dataset.total_classes)
      sub_k_fold_list = [list_splits_idxs[idx] for idx in train_idxs_split] # Get the split for the subfold considering train in kth fold...
      sub_k_fold_list.append(list_splits_idxs[val_idx_split])               # ... and validation in kth fold
      sub_path_csv_kth_fold = {}
      
      for sub_idx in range(k_fold-1): # generate the train-val split for the subfold
        saving_path_kth_sub_fold = os.path.join(saving_path_kth_fold,f'k{i}_cross_val_sub_{sub_idx}')
        train_sub_idx = [j for j in range(k_fold-1) if j != sub_idx] # get the train indices for the subfold excluding the validation
        split_indices = {
          'val': np.array([sample_ids[sub_k_fold_list[sub_idx]],sub_k_fold_list[sub_idx]]), # [sample_ids,indices] for validation
          'train': np.array([sample_ids[np.concatenate([sub_k_fold_list[j] for j in train_sub_idx])], # [sample_ids,indices] for training
                             np.concatenate([sub_k_fold_list[j] for j in train_sub_idx])])
        }
        for k,v in split_indices.items(): # generaye csv for train and validation in the subfold
          csv_data = csv_array[v[1]]
          if not os.path.exists(saving_path_kth_sub_fold):
            os.makedirs(saving_path_kth_sub_fold)
          tools.generate_csv(cols=cols, data=csv_data, saving_path=os.path.join(saving_path_kth_sub_fold,f'{k}.csv'))
          sub_path_csv_kth_fold[f'{k}'] = os.path.join(saving_path_kth_sub_fold,f'{k}.csv')
        tools.save_split_indices(split_indices,saving_path_kth_sub_fold)
        
        dict_train = train_model(dict_csv_path= sub_path_csv_kth_fold,
                                train_folder_saving_path=saving_path_kth_sub_fold,
                                is_validation=True,
                                round_output_loss=round_output_loss,
                                shuffle_video_chunks=shuffle_video_chunks,
                                shuffle_training_batch=shuffle_training_batch,
                                init_network=init_network,
                                regularization_loss=regularization_loss,
                                regularization_lambda=regularization_lambda)
        
        best_model_epoch = dict_train['dict_results']['best_model_idx']
        dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val-loss'] = dict_train['dict_results']['val_losses'][best_model_epoch]
        dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train-loss'] = dict_train['dict_results']['train_losses'][best_model_epoch]
        dict_k_fold_logs[f'l-{i}_s-{sub_idx}_train_macro_accuracy'] = dict_train['dict_results']['list_train_macro_accuracy'][best_model_epoch]
        dict_k_fold_logs[f'l-{i}_s-{sub_idx}_val_macro_accuracy'] = dict_train['dict_results']['list_val_macro_accuracy'][best_model_epoch]
        
        dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val-loss-class-avg'] = np.mean(dict_train['dict_results']['val_loss_per_class'][best_model_epoch])
        dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val-loss-subject-avg'] = np.mean(dict_train['dict_results']['val_loss_per_subject'][best_model_epoch])
        
        dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train-loss-class-avg'] = np.mean(dict_train['dict_results']['train_loss_per_class'][best_model_epoch])
        dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train-loss-subject-avg'] = np.mean(dict_train['dict_results']['train_loss_per_subject'][best_model_epoch])
        
        plot_loss_details(dict_train=dict_train,train_folder_path=saving_path_kth_sub_fold,total_epochs=epochs)
        # ADD ACCURACY IN CSV FOR TRAIN VAL and TEST
        # tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
        #                               count_subject_ids_train=dict_train['count_subject_ids_train'],
        #                               count_subject_ids_test=dict_train['count_subject_ids_test'],
        #                               count_y_test=dict_train['count_y_test'], 
        #                               count_y_train=dict_train['count_y_train'],
        #                               saving_path=saving_path_kth_sub_fold)
        # tools.plot_confusion_matrix
        fold_results_kth.append(dict_train)

      best_results_idx = [fold_results_kth[i]['dict_results']['best_model_idx'] for i in range(k_fold-1)]
      best_results_state_dict = [fold_results_kth[i]['dict_results']['best_model_state'] for i in range(k_fold-1)]
      # print(f'best_results_idx {best_results_idx}')
      dict_all_results = { 
                          'avg_train_loss_best_models': np.mean([fold_results_kth[i]['dict_results']['train_losses'][best_results_idx[i]] for i in range(k_fold-1)]),
                          'avg_val_loss_best_models': np.mean([fold_results_kth[i]['dict_results']['val_losses'][best_results_idx[i]] for i in range(k_fold-1)]),
                          'avg_train_loss_per_class_best_models': np.mean([fold_results_kth[i]['dict_results']['train_loss_per_class'][best_results_idx[i]] for i in range(k_fold-1)],axis=1),
                          'avg_val_loss_per_class_best_models': np.mean([fold_results_kth[i]['dict_results']['val_loss_per_class'][best_results_idx[i]] for i in range(k_fold-1)],axis=1),
                          # 'avg_train_loss_per_subject_best_models': np.mean([fold_results[i]['dict_results']['train_loss_per_subject'][best_results_idx[i]] for i in range(k_fold)]),
                          # 'avg_test_loss_per_subject_best_models': np.mean([fold_results[i]['dict_results']['test_loss_per_subject'][best_results_idx[i]] for i in range(k_fold)])
                          }
      
      # use best result model with the test set
      list_valid_losses = [dict_train['dict_results']['val_losses'] for dict_train in fold_results_kth]
      # get the best model according to the validation loss
      best_model_subfolder_idx = np.argmin([losses[best_results_idx[i]] for i,losses in enumerate(list_valid_losses)])
      # best_model_state_dict = fold_results[best_model_idx]['dict_results']['best_model_state']
      best_model_epoch = fold_results_kth[best_model_subfolder_idx]['dict_results']['best_model_idx']
      list_best_model_idx.append(best_model_epoch)
      path_model_weights = os.path.join(saving_path_kth_fold,f'k{i}_cross_val_sub_{best_model_subfolder_idx}',f'best_model_ep_{best_model_epoch}.pth')
      # list_path_model_weights.append(path_model_weights)
      dict_results_model_weights[f'{i}'] = {'sub':best_model_subfolder_idx,'epoch':best_model_epoch}
      # keep only the best model removing the others
      for k,dict_best_result in dict_results_model_weights.items():
        saving_path_kth_fold = os.path.join(train_folder_path,f'k{k}_cross_val')
        k=int(k)
        for j in range(k_fold-1):
          # delete models except the best one
          if j != dict_best_result['sub']:
            path_folder_model_weights = os.path.join(saving_path_kth_fold,f'k{k}_cross_val_sub_{j}',)
            # Get .pth file
            for file in os.listdir(path_folder_model_weights):
              if file.endswith(".pth"):
                os.remove(os.path.join(path_folder_model_weights, file))
        
        with open(os.path.join(saving_path_kth_fold,'results_k_fold_train_test_mean.txt'),'w') as f:
          f.write(str(dict_all_results))
      
      dict_test = model_advanced.evaluate_from_model(path_model_weights=path_model_weights,
                                         csv_path=sub_path_csv_kth_fold['val'], # sub_path_csv_kth_fold['val'],
                                         log_file_path=os.path.join(saving_path_kth_fold,'test_results.txt'),
                                         is_test=True)
      tools.plot_confusion_matrix(confusion_matrix=dict_test['test_confusion_matrix'],
                                  title=f'Confusion matrix Test folder k-{i} considering best model',
                                  saving_path=os.path.join(saving_path_kth_fold,f'confusion_matrix_test_submodel_{best_model_subfolder_idx}.png'))
      
      dict_k_fold_logs[f'k-{i}_test-loss'] = dict_test['test_loss']
      dict_k_fold_logs[f'k-{i}_test-loss-class-avg'] = np.mean(dict_test['test_loss_per_class'])
      dict_k_fold_logs[f'k-{i}_test-loss-subject-avg'] = np.mean(dict_test['test_loss_per_subject'])
      dict_k_fold_logs[f'k-{i}_test-accurracy'] = dict_test['test_macro_precision']
      
      dict_k_fold_logs[f'k-{i}_train-loss'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['train_losses'][best_model_epoch]
      dict_k_fold_logs[f'k-{i}_train-loss-class-avg'] = np.mean(fold_results_kth[best_model_subfolder_idx]['dict_results']['train_loss_per_class'][best_model_epoch])
      dict_k_fold_logs[f'k-{i}_train-loss-subject-avg'] = np.mean(fold_results_kth[best_model_subfolder_idx]['dict_results']['train_loss_per_subject'][best_model_epoch])
      dict_k_fold_logs[f'k-{i}_train_accuracy'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['list_train_macro_accuracy'][best_model_epoch]
      
      dict_k_fold_logs[f'k-{i}_val-loss'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['val_losses'][best_model_epoch]
      dict_k_fold_logs[f'k-{i}_val-loss-class-avg'] = np.mean(fold_results_kth[best_model_subfolder_idx]['dict_results']['val_loss_per_class'][best_model_epoch])
      dict_k_fold_logs[f'k-{i}_val-loss-subject-avg'] = np.mean(fold_results_kth[best_model_subfolder_idx]['dict_results']['val_loss_per_subject'][best_model_epoch]) 
      dict_k_fold_logs[f'k-{i}_val_accuracy'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['list_val_macro_accuracy'][best_model_epoch]
      
      list_test_results.append(dict_test)
      fold_results_total.append(fold_results_kth[best_model_subfolder_idx])
      
    with open(os.path.join(train_folder_path,'total_results_k_fold.txt'),'w') as f: 
      f.write("Evaluation results for each k_fold\n")
      for i,dict_test in enumerate(list_test_results):
        f.write(f'k_{i}\n')
        for k,v in dict_test.items():
          f.write(f' {k} : {v}\n')
    # use eval because we are using the validation set
    dict_final_val_results = {
      'avg_test_loss_best_models': np.mean([dict_test['test_loss'] for dict_test in list_test_results]),
      'avg_test_loss_per_class_best_models': np.mean([dict_test['test_loss_per_class'] for dict_test in list_test_results],axis=(0)),
      'avg_test_loss_per_subject_best_models': np.mean([dict_test['test_loss_per_subject'] for dict_test in list_test_results],axis=(0)),
      'avg_accuracy_best_models': np.mean([dict_test['test_macro_precision'] for dict_test in list_test_results]),
    }
    # for i,dict_test in enumerate(list_test_results):
    #   dict_k_fold_logs[f'k-{i}_test']
    dict_k_fold_logs['tot_test_loss-avg'] = dict_final_val_results['avg_test_loss_best_models']
    dict_k_fold_logs['tot_test_loss-class-avg'] = np.mean(dict_final_val_results['avg_test_loss_per_class_best_models'])
    dict_k_fold_logs['tot_test_loss-subject-avg'] = np.mean(dict_final_val_results['avg_test_loss_per_subject_best_models'])
    dict_k_fold_logs['tot_test_accuracy'] = dict_final_val_results['avg_accuracy_best_models']
    
    final_best_model_folder_idx = np.argmin([dict_val['test_loss'] for dict_val in list_test_results])
    with open(os.path.join(train_folder_path,'final_results.txt'),'w') as f:
      f.write(f'Final results {k_fold} cross-validation\n')  
      dict_train_total_results ={
        'avg_train_loss_best_models': np.mean([dict_train['dict_results']['train_losses'][dict_train['dict_results']['best_model_idx']] for dict_train in fold_results_total]),
        'avg_train_loss_per_class_best_models': np.mean([dict_train['dict_results']['train_loss_per_class'][dict_train['dict_results']['best_model_idx']] for dict_train in fold_results_total],axis=0),
        'avg_val_loss_best_models': np.mean([dict_train['dict_results']['val_losses'][dict_train['dict_results']['best_model_idx']] for dict_train in fold_results_total]),
        'avg_val_loss_per_class_best_models': np.mean([dict_train['dict_results']['val_loss_per_class'][dict_train['dict_results']['best_model_idx']] for dict_train in fold_results_total],axis=0),
      }
      f.write("AVG results considering the best model for each k_fold according to the min loss considerign eval set:\n")
      for k,v in dict_train_total_results.items():
        f.write(f' {k} : {v}\n')
      f.write("Final results considering the best models using test set:\n")
      for k,v in dict_final_val_results.items():
        f.write(f' {k} : {v}\n')
      f.write(f'\nBest model for eval loss in the evaluation set is in folder k{final_best_model_folder_idx}_cross_val/k{final_best_model_folder_idx}_cross_val_sub_{dict_results_model_weights[f"{final_best_model_folder_idx}"]["sub"]}')
       
    plot_models_loss = {}
    plot_subject_loss = {}
    plot_class_loss = {}

    for i,dict_test in enumerate(list_test_results):
      plot_models_loss[f'k_{i}'] = dict_test['test_loss']
      plot_subject_loss[f'k_{i}']=dict_test['test_loss_per_subject']
      plot_class_loss[f'k_{i}'] = dict_test['test_loss_per_class']
      
    tools.plot_bar(data=plot_models_loss,
                   x_label='k_fold nr.',
                   y_label='loss',
                   title='Loss per k_fold Evaluation',
                   saving_path=os.path.join(train_folder_path,'loss_per_k_fold.png'))
    # TODO: FIX y_axis to be consistent in all plots
    # TODO: PUT the subject ID and not the array position of subject_id
    tools.subplot_loss(dict_losses=plot_subject_loss,
                       list_title=[f'Eval Subject loss {k_fold} model {list_best_model_idx[k_fold]}' for k_fold in range(k_fold)],
                       saving_path=os.path.join(train_folder_path,'subject_loss_per_k_fold.png'),
                       x_label='subject_id',
                       y_label='loss')
    tools.subplot_loss(dict_losses=plot_class_loss,
                        list_title=[f'Eval Class loss {k_fold} model {list_best_model_idx[k_fold]}' for k_fold in range(k_fold)],
                        saving_path=os.path.join(train_folder_path,'class_loss_per_k_fold.png'),
                        x_label='class_id',
                        y_label='loss')
    
    
    dict_k_fold_logs['folder_path'] = train_folder_path
    
    return {
            'list_test_results':list_test_results,
            # 'list_saving_paths':list_saving_paths,
            # 'list_path_csv_kth_fold':list_path_csv_kth_fold,
            'best_results_idx':best_results_idx,
            'best_results_state_dict':best_results_state_dict,
            'dict_k_fold_logs':dict_k_fold_logs,
            }
      
  ###############################
  # START of the main function  #
  ###############################
  inputs_dict = {
    'k_fold': k_fold,
    'model_type': model_type.name,
    'epochs': epochs,
    # 'criterion': criterion.__class__.__name__,
    'optimizer_fn': optimizer_fn.__name__,
    'lr': lr,
    'criterion': type(criterion).__name__,
    'init_network': init_network,
    'regularization_lambda': regularization_lambda,
    'regularization_loss': regularization_loss,
    'batch_size_training': batch_size_training,
    'pooling_embedding_reduction': pooling_embedding_reduction.name,
    'pooling_clips_reduction': pooling_clips_reduction.name,
    'sample_frame_strategy': sample_frame_strategy.name,
    'path_csv_dataset': path_csv_dataset,
    'path_video_dataset': path_video_dataset,
    'head': head.name,
    'stride_window_in_video': stride_window_in_video,
    # 'head_params': head_params,
    'train_size': train_size,
    'val_size': val_size,
    'test_size': test_size,
    # 'download_if_unavailable': is_download_if_unavailable,
    'batch_size_feat_extraction': batch_size_feat_extraction,
    'random_state': random_state_split_dataset,
    'plot_dataset_distribution': is_plot_dataset_distribution,
    'plot_loss': is_plot_loss,
    'plot_tsne_backbone_feats': is_plot_tsne_backbone_feats,
    'plot_tsne_head_pred': is_plot_tsne_head_pred,
    'plot_tsne_gru_feats': is_plot_tsne_gru_feats,
    'create_video_prediction': is_create_video_prediction,
    'create_video_prediction_per_video': is_create_video_prediction_per_video,
    'round_output_loss': is_round_output_loss,
    'shuffle_video_chunks': is_shuffle_video_chunks,
    'shuffle_training_batch':is_shuffle_training_batch,
    'features_folder_saving_path': features_folder_saving_path
    }
  key_to_remove_inputs_dict = ['batch_size_feat_extraction','plot_loss','plot_dataset_distribution',
                               'plot_tsne_backbone_feats','plot_tsne_head_pred',
                               'plot_tsne_gru_feats','create_video_prediction',
                               'create_video_prediction_per_video','features_folder_saving_path',
                               'download_if_unavailable']
  for k,v in head_params.items():
    inputs_dict[f'GRU_{k}'] = v
  # Create the model
  # wandb.init(project="PainRegressionBiovid",config=inputs_dict)
  
  model_advanced = Model_Advanced(model_type=model_type,
                                  path_dataset=path_video_dataset,
                                  embedding_reduction=pooling_embedding_reduction,
                                  clips_reduction=pooling_clips_reduction,
                                  sample_frame_strategy=sample_frame_strategy,
                                  stride_window=stride_window_in_video,
                                  path_labels=path_csv_dataset,
                                  batch_size_training=batch_size_training,
                                  batch_size_feat_extraction=batch_size_feat_extraction,
                                  head=head.value,
                                  head_params=head_params,
                                  download_if_unavailable=is_download_if_unavailable,
                                  features_folder_saving_path= features_folder_saving_path
                                  )
  
  # Check if the global folder exists 
  # global_foder_name = os.path.join(GLOBAL_PATH.NAS_PATH,'history_run')
  print(f'Global folder name {global_foder_name}')
  if not os.path.exists(global_foder_name):
    os.makedirs(global_foder_name)
  
  # Create folder to save the run
  # print(f"Creating run folder at {global_foder_name}")
  timestamp = int(time.time())
  run_folder_name = (f'{timestamp}'+
                     f'{model_type.name}_'+
                     f'{pooling_embedding_reduction.name}_'+
                     f'{pooling_clips_reduction.name}_'+
                     f'{sample_frame_strategy.name}_{head.name}')
  run_folder_path = os.path.join(global_foder_name,run_folder_name) # history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_timestamp
  if not os.path.exists(run_folder_path):
    os.makedirs(os.path.join(global_foder_name,run_folder_name))
  # print(f"Run folder created at {run_folder_path}")
  
  # Save model configuration
  # print(f"Saving model configuration at {run_folder_path}")
  model_advanced.save_configuration(os.path.join(run_folder_path,'advanced_model_config.json'))

  with open(os.path.join(run_folder_path,'global_config.json'), 'w') as config_file:
      json.dump(inputs_dict, config_file, indent=4,cls=NpEncoder)
      
  # Plot dataset distribution of whole dataset
  # print(f"Plotting dataset distribution at {run_folder_path}")
  if is_plot_dataset_distribution:
    plot_dataset_distribuition(csv_path=path_csv_dataset, run_folder_path=run_folder_path, total_classes=model_advanced.dataset.total_classes)

  # Train the model
  print(f"Start training phase the model at {run_folder_path}")
  train_folder_path = os.path.join(run_folder_path,f'train_{head.value}') # history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_timestamp/train_{head.value}
  list_dict_csv_path = []
  list_saving_paths_k_fold = []

  if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
  
  # Save feature if required
  if is_save_features_extracted:
    extract_and_save_all_features(csv_path=path_csv_dataset,features_folder_saving_path=features_folder_saving_path)
  
  # Train the model
  if k_fold == 1:
    print("Generating train, test and validation csv files")
    dict_cvs_path = tools._generate_train_test_validation(csv_path=path_csv_dataset,
                                                          saving_path=train_folder_path,
                                                          train_size=train_size,
                                                          val_size=val_size,
                                                          test_size=test_size,
                                                          random_state=random_state_split_dataset
                                                          )
    print("Plotting train,test,val distribution")
    for _,csv_path in dict_cvs_path.items():
      if is_plot_dataset_distribution:
        plot_dataset_distribuition(csv_path=csv_path,run_folder_path=run_folder_path, total_classes=model_advanced.dataset.total_classes)
    print("Training model k=1")
    dict_train = train_model(is_validation=is_validation,
                             dict_csv_path=dict_cvs_path,
                             train_folder_saving_path=train_folder_path,
                             round_output_loss=is_round_output_loss,
                             shuffle_video_chunks=is_shuffle_video_chunks,
                             init_network=init_network,
                             regularization_loss=regularization_loss,
                             regularization_lambda=regularization_lambda)
    
    # plot loss details
    plot_loss_details(dict_train, train_folder_path, epochs)
    list_dict_csv_path.append(dict_cvs_path)
    list_saving_paths_k_fold.append(train_folder_path)
    # return dict_train, list_dict_csv_path, list_saving_paths_k_fold
  else: 
    fold_results = k_fold_cross_validation(path_to_extracted_features=model_advanced.path_to_extracted_features,
                                           is_validation=is_validation,
                                           round_output_loss=is_round_output_loss,
                                           shuffle_video_chunks=is_shuffle_video_chunks,
                                           shuffle_training_batch=is_shuffle_training_batch,
                                           init_network=init_network,
                                           regularization_loss=regularization_loss,
                                           regularization_lambda=regularization_lambda)
    # list_dict_csv_path = fold_results['list_path_csv_kth_fold'] # list of dictionaries with paths to CSV files for each fold
    # list_saving_paths_k_fold = fold_results['list_saving_paths'] # list of saving paths for each fold
    # list_best_model_state = fold_results['best_results_state_dict']
    # for i,dict_train in enumerate(fold_results['fold_results']):
    #   plot_loss_details(dict_train=dict_train, 
    #                     train_folder_path=list_saving_paths_k_fold[i],
    #                     epochs=epochs)
    log_dict = {}
    for k,v in inputs_dict.items():
      if k not in key_to_remove_inputs_dict:
        log_dict[k] = v
    log_dict.update(fold_results['dict_k_fold_logs'])
    return log_dict
  
  
  key_test_dataset = 'test' if not is_validation else 'val'
  # dict_cvs_path = dict_cvs_path[key_test_dataset]
  # For each dataset used plot tsne considering all features from backbone per subject and per class
  
  # TO FIX or REMOVE
  if not only_train:
    for dict_cvs_path,train_folder_path,best_model_state_dict in zip(list_dict_csv_path,list_saving_paths_k_fold,list_best_model_state):
      # create folder to save tsne plots
      model_advanced.head.model.load_state_dict(best_model_state_dict)
      print(f"Creating folder to save tsne plots at {train_folder_path}")
      saving_path_tsne = os.path.join(train_folder_path,'tsne')
      if not os.path.exists(saving_path_tsne):
        os.mkdir(saving_path_tsne)
    
    # plot tsne considering all features from backbone per subject and per class
      
      for split_dataset_name, split_dataset_csv_path in dict_cvs_path.items(): #{k = {"train", "test", "val"}; v = path_to_csv}
        if split_dataset_name == key_test_dataset:
          print(f'Split {split_dataset_name}')
          with torch.no_grad():
            dict_backbone_feats = model_advanced.extract_features(csv_path = split_dataset_csv_path)
          backbone_feats = dict_backbone_feats['features'] # [nr_video * windows, T=8, patch_w=1, patch_h=1, emb=384] 
          # print(f'backbone_feats {backbone_feats.shape}')
          y_gt = dict_backbone_feats['list_labels']    
          list_subject_id = dict_backbone_feats['list_subject_id']
          tools.compute_tsne(X=backbone_feats,
                        labels=y_gt,
                        apply_pca_before_tsne=True,
                        saving_path=os.path.join(saving_path_tsne,f'tsne_backbone_gt_{split_dataset_name}'),
                        legend_label='class ',
                        title=f'dataset {split_dataset_name} (from backbone per chunks)')
        
          tools.compute_tsne(X=backbone_feats,
                          labels=list_subject_id,
                          apply_pca_before_tsne=True,
                          saving_path=os.path.join(saving_path_tsne,f'tsne_backbone_subject_{split_dataset_name}'),
                          legend_label='subject',
                          title=f'dataset {split_dataset_name} (from backbone per chunks)')
      
          # plot tsne considering the prediction per subject and per class  
          # y_pred shape [nr_video, nr_windows]
          if head.value == HEAD.SVR.value:
            y_pred = model_advanced.head.predict(backbone_feats)                                                  
            tools.compute_tsne(X=backbone_feats,
                            labels=y_pred,
                            saving_path=os.path.join(saving_path_tsne,f'tsne_SVR_{split_dataset_name}_pred'),
                            legend_label='class ',
                            title=f'dataset {split_dataset_name} (from backbone)')
          
          elif head.value == HEAD.GRU.value:
            # SHAPE X_gru_feats -> tuple(output[batch_size, seq_length, hidden_size], h_n: [num_layers,hidde_size])
            X_gru_feats_padded, length_feature, subject_ids_per_sample_id = model_advanced.head.get_embeddings(X=backbone_feats,
                                                                                      sample_id=dict_backbone_feats['list_sample_id'],
                                                                                      subject_id=dict_backbone_feats['list_subject_id']) 
            # print(f'length_feature {length_feature}')
            # print(f'X_gru_feats_padded {X_gru_feats_padded.shape}')
            # print(f'subject_ids_per_sample_id {subject_ids_per_sample_id.shape}')
            X_gru_out_feats_padded_cpu = X_gru_feats_padded.detach().cpu()
            # print(f'X_gru_out_feats_padded_cpu {X_gru_out_feats_padded_cpu.shape}')
            X_gru_hn_feats_cpu = X_gru_feats_padded[torch.arange(X_gru_out_feats_padded_cpu.shape[0]),length_feature-1].detach().cpu().squeeze(dim=1) # final hidden state of the last GRU layer
            
            X_gru_out_feats_padded_cpu = torch.cat([X_gru_out_feats_padded_cpu[i,:length_feature[i]] for i in range(X_gru_out_feats_padded_cpu.shape[0])],dim=0)
            # print(f'CAT_X_gru_out_feats_padded_cpu {X_gru_out_feats_padded_cpu.shape}')
            # print(f'X_gru_hn_feats_cpu {X_gru_hn_feats_cpu.shape}')
            del X_gru_feats_padded
            torch.cuda.empty_cache()
            # considerin 1 embedding per chunk
            y_pred = model_advanced.head.predict(X=backbone_feats,
                                              sample_id=dict_backbone_feats['list_sample_id'],
                                              subject_id=dict_backbone_feats['list_subject_id'],
                                              pred_only_last_time_step=False)
            y_pred = y_pred.detach().cpu()
            # print(f'y_pred {y_pred.shape}') # [nr_video,nr_windows,pred=1]
            y_pred_unpadded = torch.cat([y_pred[i,:length_feature[i]] for i in range(y_pred.shape[0])])
            tools.compute_tsne(X = X_gru_out_feats_padded_cpu,
                          labels = torch.round(y_pred_unpadded),
                          # title=f'{k}_{v} (from backbone)')
                          title = f'dataset_{split_dataset_name} pred per chunks (using GRU features)',
                          legend_label='class',
                          saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_{split_dataset_name}_pred'))
            # considering 1 embedding per video
            y_last_hidden = torch.stack([torch.round(y_pred[i,length_feature[i]-1]) for i in range(y_pred.shape[0])])
            print(f'y_pred {y_pred}')
            print(f'y_last_hidden {y_last_hidden}')
            
            tools.compute_tsne(X = X_gru_hn_feats_cpu,
                            labels = y_last_hidden,
                            legend_label = 'class ',
                            title = f'dataset_{split_dataset_name} pred per video (using GRU last_hidden feature)',
                            saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_{split_dataset_name}_gt'))
            
            # subject_ids_extended = torch.cat([subject_ids_per_sample_id[i].repeat(length_feature[i]) for i in range(subject_ids_per_sample_id.shape[0])])
            tools.compute_tsne(X = X_gru_hn_feats_cpu,
                            labels = subject_ids_per_sample_id,
                            title = f'dataset_{split_dataset_name} subjects per video (using GRU last_hidden feature)',
                            legend_label='subject',
                            saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_{split_dataset_name}_subject'))
          
          # Create video with predictions
          # print(f'y_pred shape {y_pred.shape}')
          # print(f'y_gt shape {y_gt.shape}')
          # print(f'list_frames shape {dict_backbone_feats["list_frames"].shape}')
          create_unique_video_per_prediction(train_folder_path=train_folder_path,
                                            dict_cvs_path=dict_cvs_path,
                                            dataset_name=split_dataset_name,
                                            list_frames=dict_backbone_feats['list_frames'],
                                            y_pred=y_pred_unpadded,
                                            y=y_gt,
                                            sample_ids=dict_backbone_feats['list_sample_id'])
          
          # plot graph chunks prediciton for 2 samplesID
          rnd_perm_idx = torch.randperm(len(dict_backbone_feats['list_sample_id']))
          for i,sample_id in enumerate(dict_backbone_feats['list_sample_id'][rnd_perm_idx][:4]):
            folder_subject = os.path.join(train_folder_path,f'plot_chunks_prediction_{split_dataset_name}')
            if not os.path.exists(folder_subject):
              os.makedirs(folder_subject)
            idx = np.where(dict_backbone_feats['list_sample_id'] == sample_id)[0]
            print(f'idx {idx}')
            print(f'len(idx) {len(idx)}')
            tools.plot_prediction_chunks_per_subject(predictions=y_pred_unpadded[idx],
                                                    n_chunks = len(idx),
                                                    # [nr_video,num_clips,1] -> [nr_video,num_clips],
                                                    #  sample_id=sample_id,
                                                    #  gt=list_ground_truth[idx],
                                                    title=f'sampleID: {sample_id} - {split_dataset_name}',
                                                    saving_path=os.path.join(folder_subject,f'subject_{sample_id}_{split_dataset_name}'))
            # if i == 10:
            #   break
  return {'model_advanced':model_advanced, 
          'dict_train':dict_train}
 

def plot_loss_details(dict_train, train_folder_path, total_epochs):
  """
  Generate and save plots of the training and test results, and confusion matrices for each epoch.
  Parameters:
  dict_train (dict): Dictionary containing training results and other relevant data.
  train_folder_path (str): Path to the folder where the plots and confusion matrices will be saved.
  epochs (int): Number of epochs for which the confusion matrices will be plotted.
  """
    # Generate and save plots of the training and test results
  # tools.plot_losses(train_losses=dict_train['dict_results']['train_losses'], 
  #                   test_losses=dict_train['dict_results']['test_losses'], 
  #                   saving_path=os.path.join(train_folder_path,'train_test_losses'))
  

  tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
                                count_subject_ids_train=dict_train['count_subject_ids_train'],
                                count_subject_ids_test=dict_train['count_subject_ids_test'],
                                count_y_test=dict_train['count_y_test'], 
                                count_y_train=dict_train['count_y_train'],
                                saving_path=train_folder_path,
                                best_model_idx=dict_train['dict_results']['best_model_idx'])
  
  # Plot and save confusion matrices for each epoch
  confusion_matrix_path = os.path.join(train_folder_path,'confusion_matricies')
  
  if not os.path.exists(confusion_matrix_path):
    os.makedirs(confusion_matrix_path)
  _plot_confusion_matricies(total_epochs, dict_train, confusion_matrix_path,dict_train['dict_results']['best_model_idx'])
  
  tools.plot_macro_accuracy(list_train_accuracy=dict_train['dict_results']['list_train_macro_accuracy'],
                            list_val_accurcay=dict_train['dict_results']['list_val_macro_accuracy'],
                            title='Macro accuracy per epoch',
                            x_label='epochs',
                            y_label='accuracy',
                            saving_path=os.path.join(train_folder_path,'losses','macro_accuracy_train_val.png'))


def create_unique_video_per_prediction(train_folder_path, dict_cvs_path, sample_ids, list_frames, y_pred, y, dataset_name):
  # Create video with predictions
  print(f"Creating video with predictions for {dataset_name}")
  video_folder_path = os.path.join(train_folder_path,f'video')
  if not os.path.exists(video_folder_path):
    os.makedirs(video_folder_path)
  
  list_input_video_path = tools.get_list_video_path_from_csv(dict_cvs_path[dataset_name])
  output_video_path = os.path.join(video_folder_path,f'video_all_{dataset_name}.mp4')

  tools.save_frames_as_video(list_input_video_path=list_input_video_path, # [n_video=33]
                                              list_frame_indices=list_frames, # [33,2,16]
                                              output_video_path=output_video_path, # string
                                              all_predictions=y_pred,#  [33,2]
                                              list_ground_truth=y,
                                              sample_ids=sample_ids, # -> [33,2] 
                                              output_fps=4)


def _plot_confusion_matricies(epochs, dict_train, confusion_matrix_path,best_model_idx):
  for epoch in range(0,epochs,50): 
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][epoch],
                                title=f'Train_{epoch} confusion matrix',
                                saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_train_{epoch}.png'))
    
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['val_confusion_matricies'][epoch],
                              title=f'Val_{epoch} confusion matrix',
                              saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_val_{epoch}.png'))
    
  # Plot best model results
  tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][best_model_idx],
                                title=f'Train_{best_model_idx} confusion matrix',
                                saving_path=os.path.join(confusion_matrix_path,f'best_confusion_matrix_train_{best_model_idx}.png'))
  tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['val_confusion_matricies'][best_model_idx],
                              title=f'Val_{best_model_idx} confusion matrix',
                              saving_path=os.path.join(confusion_matrix_path,f'best_confusion_matrix_val_{best_model_idx}.png'))
  
  saving_path_precision_recall = os.path.join(confusion_matrix_path,'plot_over_epochs')
  if not os.path.exists(saving_path_precision_recall):
    os.makedirs(saving_path_precision_recall)
  # TODO: REMOVE comments if want to plot precision and recall
  # tools.plot_accuracy_confusion_matrix(confusion_matricies=dict_train['dict_results']['train_confusion_matricies'],
  #                                      type_conf='train',
  #                                      saving_path=saving_path_precision_recall)
  # tools.plot_accuracy_confusion_matrix(confusion_matricies=dict_train['dict_results']['val_confusion_matricies'],
  #                                      type_conf='test',
  #                                      saving_path=saving_path_precision_recall)
    

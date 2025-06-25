from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY,GLOBAL_PATH
from custom.model import Model_Advanced
from transformers import AutoImageProcessor
# from custom.head import HeadSVR, HeadGRU, HeadAttentive, HeadLinear
import time
from custom.tools import NpEncoder
import custom.tools as tools
import custom.helper as helper
from custom.helper import HEAD, MODEL_TYPE, EMBEDDING_REDUCTION, SAMPLE_FRAME_STRATEGY
import torch.nn as nn
import torch.optim as optim
import dataframe_image as dfi
import pandas as pd
import torch
import numpy as np
import time
import json
import pickle
from sklearn.model_selection import StratifiedGroupKFold
import os
import random
# import wandb
def check_intersection_splits(k_fold,subject_ids,list_splits_idxs):
  for i in range(k_fold):
    for j in range(i+1,k_fold):
      subject_ids_i = set(subject_ids[list_splits_idxs[i]])
      subject_ids_j = set(subject_ids[list_splits_idxs[j]])
      if len(subject_ids_i.intersection(subject_ids_j)) > 0:
        raise ValueError('The splits must be disjoint')
def set_seed(seed):
  random.seed(seed)  # Python random module
  np.random.seed(seed)  # NumPy
  torch.manual_seed(seed)  # PyTorch (CPU)
  torch.cuda.manual_seed(seed)  # PyTorch (GPU)
  torch.cuda.manual_seed_all(seed)  # Multi-GPU
  torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
  torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility

def k_fold_cross_validation(path_csv_dataset, train_folder_path, model_advanced, k_fold, seed_random_state,
                          lr, epochs, optimizer_fn, round_output_loss, shuffle_training_batch, criterion,
                          early_stopping, enable_scheduler, concatenate_temp_dim, init_network,
                          regularization_lambda_L1, key_for_early_stopping, target_metric_best_model,stop_after_kth_fold,
                          clip_grad_norm,regularization_lambda_L2,trial):
  """
  Perform k-fold cross-validation on the dataset and train the model.
  
  This function performs k-fold cross-validation by splitting the dataset into k folds,
  training the model on each fold, and evaluating the performance. It saves the results
  and plots for each fold, and calculates the average performance metrics across all folds.
  
  Returns:
      dict: A dictionary containing the following keys:
          - 'list_test_results': A list of dictionaries with test results for each fold
          - 'best_results_idx': A list of indices indicating the best model for each fold
          - 'best_results_state_dict': A list of model state dictionaries for the best models
          - 'dict_k_fold_results': A dictionary with detailed results for each fold
  """
  # Validate inputs
  if not isinstance(model_advanced, Model_Advanced):
    raise ValueError('model_advanced must be an instance of Model_Advanced')
  
  # Load dataset
  csv_array, cols = tools.get_array_from_csv(path_csv_dataset)
  y_labels = csv_array[:, 2].astype(int)
  subject_ids = csv_array[:, 0].astype(int)
  sample_ids = csv_array[:, 4].astype(int)
  
  # Create stratified group k-fold splits
  list_splits_idxs = create_stratified_splits(k_fold, seed_random_state, y_labels, subject_ids)
  # Uncomment to verify splits are disjoint
  # check_intersection_splits(k_fold, subject_ids, list_splits_idxs)
  
  # Initialize result containers
  list_test_results = []
  list_best_model_idx = []
  best_results_state_dict = []
  dict_results_model_weights = {}
  dict_k_fold_results = {}
  
  # Perform k-fold cross-validation
  for i in range(k_fold):
    fold_results = run_single_fold(
      i, k_fold, list_splits_idxs, csv_array, cols, sample_ids, subject_ids,
      train_folder_path, model_advanced, lr, epochs, optimizer_fn, 
      concatenate_temp_dim, criterion, round_output_loss, shuffle_training_batch,
      init_network, regularization_lambda_L1,
      key_for_early_stopping, early_stopping, enable_scheduler,
      target_metric_best_model, seed_random_state, clip_grad_norm,stop_after_kth_fold,
      regularization_lambda_L2,trial
    )
    
    # Store results
    # list_test_results.append(fold_results['dict_test'])
    # list_best_model_idx.append(fold_results['best_model_epoch'])
    # best_results_state_dict.append(fold_results['best_model_state'])
    dict_k_fold_results.update(fold_results['fold_results'])
    # dict_results_model_weights[f'{i}'] = {
    #   'sub': fold_results['best_model_subfolder_idx'],
    #   'epoch': fold_results['best_model_epoch']
    # }
    
    # stop to make the tests faster
    if stop_after_kth_fold is not None and i == stop_after_kth_fold[0] - 1:
      break
    # Clean up extra model weights to save space
    # cleanup_extra_models(train_folder_path, i, k_fold, dict_results_model_weights)
    
  # Create result summary plots
  # compile_test_loss_plots(list_test_results)
  
  return {
    # 'list_test_results': list_test_results,
    'best_results_idx': list_best_model_idx,
    'best_results_state_dict': best_results_state_dict,
    'dict_k_fold_results': dict_k_fold_results
  }

def create_stratified_splits(k_fold, seed_random_state, y_labels, subject_ids):
  """Create stratified group k-fold splits"""
  sgkf = StratifiedGroupKFold(n_splits=k_fold, random_state=seed_random_state, shuffle=True)
  list_splits_idxs = []  # contains indices for all k splits
  for _, test_index in sgkf.split(X=torch.zeros(y_labels.shape), y=y_labels, groups=subject_ids):
    list_splits_idxs.append(test_index)
  return list_splits_idxs

def run_single_fold(fold_idx, k_fold, list_splits_idxs, csv_array, cols, sample_ids, subject_ids,
                   train_folder_path, model_advanced, lr, epochs, optimizer_fn, 
                   concatenate_temp_dim, criterion, round_output_loss, shuffle_training_batch,
                   init_network, regularization_lambda_L1,
                   key_for_early_stopping, early_stopping, enable_scheduler,
                   target_metric_best_model, seed_random_state,clip_grad_norm,
                   stop_after_kth_fold,regularization_lambda_L2,trial):
  """Run a single fold of the cross-validation"""
  # Setup folder structure for this fold
  saving_path_kth_fold = os.path.join(train_folder_path, f'k{fold_idx}_cross_val')
  os.makedirs(saving_path_kth_fold, exist_ok=True)
  
  # Create train/val/test splits for this fold
  test_idx_split = fold_idx % k_fold
  val_idx_split = (fold_idx + 1) % k_fold
  train_idxs_split = [j for j in range(k_fold) if j != test_idx_split and j != val_idx_split]
  
  # Generate datasets
  split_indices = create_split_indices(test_idx_split, val_idx_split, train_idxs_split,
                                     list_splits_idxs, sample_ids)
  
  # Generate CSV files for each split
  path_csv_kth_fold = generate_fold_csv_files(split_indices, csv_array, cols, saving_path_kth_fold)
  
  # Generate dataset distribution plots
  # for _, csv_path in path_csv_kth_fold.items():
  #   tools.plot_dataset_distribuition(
  #     csv_path=csv_path,
  #     run_folder_path=saving_path_kth_fold,
  #     total_classes=model_advanced.dataset.total_classes
  #   )
  
  # Prepare sub-folds for training
  sub_k_fold_list = [list_splits_idxs[idx] for idx in train_idxs_split]
  sub_k_fold_list.append(list_splits_idxs[val_idx_split])
  
  # Train sub-fold models
  # AUGMENTAION: val and test csv will be filtered in function get_dataset_and_loader  
  fold_results_kth = train_subfold_models(
    fold_idx, k_fold, sub_k_fold_list, csv_array, cols, sample_ids,
    saving_path_kth_fold, model_advanced, lr, epochs, optimizer_fn,
    concatenate_temp_dim, criterion, round_output_loss, shuffle_training_batch,
    init_network, regularization_lambda_L1,
    key_for_early_stopping, early_stopping, enable_scheduler, seed_random_state,
    clip_grad_norm,stop_after_kth_fold,path_csv_kth_fold['test'],regularization_lambda_L2,
    trial
  )
    
  fold_results ={'fold_results':{}}
  # Add sub-fold results
  for sub_idx in range(k_fold - 1):
    if sub_idx < len(fold_results_kth):
      reduced_dict = reduce_logs_for_subfold(fold_results_kth[f'k{fold_idx}_cross_val_sub_{sub_idx}']['train'])
      if f'k{fold_idx}_cross_val_sub_{sub_idx}' not in fold_results['fold_results']:
        fold_results['fold_results'][f'k{fold_idx}_cross_val_sub_{sub_idx}'] = {}
      fold_results['fold_results'][f'k{fold_idx}_cross_val_sub_{sub_idx}']['train_val'] = reduced_dict
      fold_results['fold_results'][f'k{fold_idx}_cross_val_sub_{sub_idx}']['test'] = fold_results_kth[f'k{fold_idx}_cross_val_sub_{sub_idx}']['test']
    else:
      break
    
  return fold_results

def create_split_indices(test_idx_split, val_idx_split, train_idxs_split, list_splits_idxs, sample_ids):
  """Create train/val/test split indices"""
  test_sample_ids = sample_ids[list_splits_idxs[test_idx_split]]
  val_sample_ids = sample_ids[list_splits_idxs[val_idx_split]]
  train_sample_ids = []
  
  for idx in train_idxs_split:
    train_sample_ids.extend(sample_ids[list_splits_idxs[idx]])
  
  return {
    'test': np.array([test_sample_ids, list_splits_idxs[test_idx_split]]),
    'val': np.array([val_sample_ids, list_splits_idxs[val_idx_split]]),
    'train': np.array([
      train_sample_ids,
      np.concatenate([list_splits_idxs[idx] for idx in train_idxs_split])
    ])
  }

def generate_fold_csv_files(split_indices, csv_array, cols, saving_path):
  """Generate CSV files for each split and return their paths"""
  path_csv_kth_fold = {}
  
  for key, v in split_indices.items():
    csv_data = csv_array[v[1]]
    path = os.path.join(saving_path, f'{key}.csv')
    tools.generate_csv(cols=cols, data=csv_data, saving_path=path)
    path_csv_kth_fold[key] = path
  
  return path_csv_kth_fold

def train_subfold_models(fold_idx, k_fold, sub_k_fold_list, csv_array, cols, sample_ids,
                      saving_path_kth_fold, model_advanced, lr, epochs, optimizer_fn,
                      concatenate_temp_dim, criterion, round_output_loss, shuffle_training_batch,
                      init_network, regularization_lambda_L1,
                      key_for_early_stopping, early_stopping, enable_scheduler, seed_random_state,clip_grad_norm,
                      stop_after_kth_fold,test_csv_path,regularization_lambda_L2,trial):
  """Train models on sub-folds"""
  if not isinstance(model_advanced, Model_Advanced):
    raise ValueError('model_advanced must be an instance of Model_Advanced')
  fold_results_kth = {}
  
  for sub_idx in range(k_fold - 1):
    # Create subfold directory
    saving_path_kth_sub_fold = os.path.join(saving_path_kth_fold, f'k{fold_idx}_cross_val_sub_{sub_idx}')
    os.makedirs(saving_path_kth_sub_fold, exist_ok=True)
    
    # Generate train/val split for this subfold
    sub_path_csv_kth_fold = generate_subfold_csv_files(
      sub_idx, k_fold, sub_k_fold_list, csv_array, cols, 
      sample_ids, saving_path_kth_sub_fold
    )
    
    # Train model
    set_seed(seed=seed_random_state)
    dict_train = model_advanced.train(
      lr=lr,
      num_epochs=epochs,
      optimizer_fn=optimizer_fn,
      concatenate_temporal=concatenate_temp_dim,
      criterion=criterion,
      saving_path=saving_path_kth_sub_fold,
      train_csv_path=sub_path_csv_kth_fold['train'],
      val_csv_path=sub_path_csv_kth_fold['val'],
      round_output_loss=round_output_loss,
      shuffle_training_batch=shuffle_training_batch,
      init_network=init_network,
      regularization_lambda_L1=regularization_lambda_L1,
      regularization_lambda_L2=regularization_lambda_L2,
      key_for_early_stopping=key_for_early_stopping,
      early_stopping=early_stopping,
      enable_scheduler=enable_scheduler,
      clip_grad_norm=clip_grad_norm,
      trial=trial,
      enable_optuna_pruning = True if fold_idx == 0 and sub_idx == 0 else False
    )
    dict_test = test_model(
      model_advanced=model_advanced,
      path_model_weights=None, 
      test_csv_path=test_csv_path,
      state_dict=dict_train['dict_results']['best_model_state'],
      criterion=criterion, concatenate_temporal=concatenate_temp_dim)
    # Remove unnecessary data to save space
    dict_train['dict_results']['best_model_state'] = None
    
    fold_results_kth[f'k{fold_idx}_cross_val_sub_{sub_idx}']={'train':dict_train,'test':dict_test}
    
    # Stop to make the tests faster
    if stop_after_kth_fold is not None and sub_idx == stop_after_kth_fold[1] - 1:
      break
  
  return fold_results_kth

def generate_subfold_csv_files(sub_idx, k_fold, sub_k_fold_list, csv_array, cols, 
                            sample_ids, saving_path_kth_sub_fold):
  """Generate CSV files for a sub-fold"""
  sub_path_csv_kth_fold = {}
  
  # Get validation indices
  val_indices = sub_k_fold_list[sub_idx]
  val_sample_ids = sample_ids[val_indices]
  
  # Get training indices (all except validation)
  train_sub_idx = [j for j in range(k_fold - 1) if j != sub_idx]
  train_indices = np.concatenate([sub_k_fold_list[j] for j in train_sub_idx])
  train_sample_ids = sample_ids[train_indices]
  
  # Create split indices
  split_indices = {
    'val': np.array([val_sample_ids, val_indices]),
    'train': np.array([train_sample_ids, train_indices])
  }
  
  # Generate CSV files
  for key, v in split_indices.items():
    csv_data = csv_array[v[1]]
    path = os.path.join(saving_path_kth_sub_fold, f'{key}.csv')
    tools.generate_csv(cols=cols, data=csv_data, saving_path=path)
    sub_path_csv_kth_fold[key] = path
  
  return sub_path_csv_kth_fold

def select_best_model(fold_results_kth, target_metric_best_model, saving_path_kth_fold, fold_idx):
  """Select the best model from sub-folds based on validation metrics"""
  # Get best model indices for each sub-fold
  best_results_idx = [fold_results_kth[i]['dict_results']['best_model_idx'] for i in range(len(fold_results_kth))]
  best_results_state_dict = [fold_results_kth[i]['dict_results']['best_model_state'] for i in range(len(fold_results_kth))]
  
  # Choose target metric for model selection
  target_metric_key = 'val_losses' if target_metric_best_model == 'val_loss' else 'list_val_macro_accuracy'
  list_validation_best_result = [dict_train['dict_results'][target_metric_key] for dict_train in fold_results_kth]
  
  # Find the best sub-fold model
  if target_metric_key == 'val_losses':
    best_model_subfolder_idx = np.argmin([metric[best_results_idx[i]] for i, metric in enumerate(list_validation_best_result)])
  elif target_metric_key == 'list_val_macro_accuracy':
    best_model_subfolder_idx = np.argmax([metric[best_results_idx[i]] for i, metric in enumerate(list_validation_best_result)])
  else:
    raise ValueError('target_metric_best_model must be val_loss or list_val_macro_accuracy')
  
  # Get best model epoch
  best_model_epoch = fold_results_kth[best_model_subfolder_idx]['dict_results']['best_model_idx']
  
  # Get best model path
  path_model_weights = os.path.join(
    saving_path_kth_fold,
    f'k{fold_idx}_cross_val_sub_{best_model_subfolder_idx}',
    f'best_model_ep_{best_model_epoch}.pth'
  )
  
  return {
    'subfolder_idx': best_model_subfolder_idx,
    'epoch': best_model_epoch,
    'state_dict': best_results_state_dict[best_model_subfolder_idx],
    'path': path_model_weights
  }

def test_model(model_advanced, path_model_weights, test_csv_path,
                   state_dict, criterion, concatenate_temporal,
                   subfolder_idx=None,saving_path_kth_fold=None,fold_idx=None):
  """Test the best model on the test set"""
  if not isinstance(model_advanced, Model_Advanced):
    raise ValueError('model_advanced must be an instance of Model_Advanced')
  
  dict_test = model_advanced.test_pretrained_model(
    path_model_weights=path_model_weights,
    state_dict=state_dict,
    csv_path=test_csv_path,
    is_test=True,
    criterion=criterion,
    concatenate_temporal=concatenate_temporal
  )
  
  # # Plot confusion matrix
  # tools.plot_confusion_matrix(
  #   confusion_matrix=dict_test['test_confusion_matrix'],
  #   title=f'Confusion matrix Test folder k-{fold_idx} considering best model',
  #   saving_path=os.path.join(
  #     saving_path_kth_fold,
  #     f'confusion_matrix_test_submodel_{subfolder_idx}.png'
  #   )
  # )
  
  return dict_test

def reduce_logs_for_subfold(dict_train):
  """Reduce logs to save space, keeping only results every count_epochs//10 and the best model epoch"""
  list_to_reduce_for_logs = [
    'train_confusion_matricies', 'val_confusion_matricies'
  ]
  
  count_epochs = dict_train['dict_results']['epochs']
  best_model_idx = dict_train['dict_results']['best_model_idx']
  reduced_dict_train = {}
  
  # At least 12 logs are saved for each epoch
  target_nr_matricies = count_epochs//10 if count_epochs > 10 else 1
  for key, v in dict_train['dict_results'].items():
    if key != 'best_model_state':
      if key in list_to_reduce_for_logs:
        # Keep results every target_nr_matricies epochs
        reduced_dict_train[key] = {f'{epoch}': v[epoch] for epoch in range(0, count_epochs, target_nr_matricies)}
        if count_epochs - 1 not in reduced_dict_train[key]:
          reduced_dict_train[key].update({f'{count_epochs - 1}': v[count_epochs - 1]})
        # Also keep the best model epoch if not already included
        if best_model_idx % target_nr_matricies != 0:
          reduced_dict_train[key].update({f'{best_model_idx}': v[best_model_idx]})
      else:
        reduced_dict_train[key] = v
  # save only the last matrix
  return reduced_dict_train

def cleanup_extra_models(train_folder_path, fold_idx, k_fold, dict_results_model_weights):
  """Clean up extra model weights to save space"""
  saving_path_kth_fold = os.path.join(train_folder_path, f'k{fold_idx}_cross_val')
  best_sub_idx = dict_results_model_weights[f'{fold_idx}']['sub']
  
  for j in range(k_fold - 1):
    # Delete models except the best one
    if j != best_sub_idx:
      path_folder_model_weights = os.path.join(saving_path_kth_fold, f'k{fold_idx}_cross_val_sub_{j}')
      
      # Get .pth files
      for file in os.listdir(path_folder_model_weights):
        if file.endswith(".pth"):
          os.remove(os.path.join(path_folder_model_weights, file))

def compile_test_loss_plots(list_test_results):
  """Compile test loss plots for visualization"""
  plot_models_test_loss = {}
  plot_subject_test_loss = {}
  plot_class_test_loss = {}
  
  for i, dict_test in enumerate(list_test_results):
    plot_models_test_loss[f'k_{i}'] = dict_test['test_loss']
    plot_subject_test_loss[f'k_{i}'] = {
      'loss': dict_test['test_loss_per_subject'],
      'elements': dict_test['test_unique_subject_ids']
    }
    plot_class_test_loss[f'k_{i}'] = {
      'loss': dict_test['test_loss_per_class'],
      'elements': dict_test['test_unique_y']
    }
  
  return {
    'models': plot_models_test_loss,
    'subjects': plot_subject_test_loss,
    'classes': plot_class_test_loss
  }


def run_train_test(model_type, pooling_embedding_reduction, pooling_clips_reduction, sample_frame_strategy, 
                   path_csv_dataset, path_video_dataset, head, stride_window_in_video, features_folder_saving_path,head_params,
                   k_fold,
                   global_foder_name, 
                   batch_size_training, epochs, 
                   criterion, optimizer_fn, lr,seed_random_state,
                   is_plot_dataset_distribution,
                   is_round_output_loss, is_shuffle_video_chunks,is_shuffle_training_batch,
                   init_network,key_for_early_stopping,
                   regularization_lambda_L1,
                   regularization_lambda_L2,
                   clip_length,
                   target_metric_best_model,
                   enable_scheduler,
                   early_stopping,
                   concatenate_temp_dim,
                   stop_after_kth_fold,
                   n_workers,
                   clip_grad_norm,
                   label_smooth,
                   dict_augmented,
                   prefetch_factor,
                   soft_labels,
                   trial=None,
                  ):
 


  def get_obj_config():
    return{
      'k_fold': k_fold,
      'model_type': model_type,
      'epochs': epochs,
      'optimizer_fn': optimizer_fn.__name__,
      'lr': lr,
      'criterion': criterion,
      'criterion_dict': criterion.get_params() if hasattr(criterion,'get_params') else None,
      'init_network': init_network,
      'regularization_lambda_L1': regularization_lambda_L1,
      'regularization_lambda_L2': regularization_lambda_L2,
      'batch_size_training': batch_size_training,
      'pooling_embedding_reduction': pooling_embedding_reduction,
      'pooling_clips_reduction': pooling_clips_reduction,
      'sample_frame_strategy': sample_frame_strategy,
      'path_csv_dataset': path_csv_dataset.split('/')[-3:], # get the last 3 folders
      'path_video_dataset': path_video_dataset.split('/')[-3:], # get the last 3 folders
      'features_folder_saving_path': features_folder_saving_path.split('/')[-3:], # get the last 3 folders
      'head': head,
      'key_for_early_stopping': key_for_early_stopping,
      # 'stride_window_in_video': stride_window_in_video,
      'head_params': head_params,
      'random_state': seed_random_state,
      'plot_dataset_distribution': is_plot_dataset_distribution,
      'round_output_loss': is_round_output_loss,
      'shuffle_video_chunks': is_shuffle_video_chunks,
      'shuffle_training_batch':is_shuffle_training_batch,
      'clip_length': clip_length,
      'target_metric_best_model': target_metric_best_model,
      'early_stopping': early_stopping,
      'enable_scheduler': enable_scheduler,
      'concatenate_temp_dim': concatenate_temp_dim,
      'stop_after_kth_fold': stop_after_kth_fold,
      'n_workers': n_workers,
      'clip_grad_norm': clip_grad_norm,
      'label_smooth': label_smooth,
      'soft_labels': soft_labels,
      **dict_augmented,
      'type_regul': 'elastic' if regularization_lambda_L1 > 0 and regularization_lambda_L2 > 0 else 'L1' if regularization_lambda_L1 > 0 else 'L2' if regularization_lambda_L2 > 0 else 'none'
      }
  def get_json_config():
    return {
    'k_fold': k_fold,
    'model_type': model_type.name,
    'epochs': epochs,
    'optimizer_fn': optimizer_fn.__name__,
    'lr': lr,
    'criterion': type(criterion).__name__,
    'init_network': init_network,
    'regularization_lambda_L1': regularization_lambda_L1,
    'batch_size_training': batch_size_training,
    'pooling_embedding_reduction': pooling_embedding_reduction.name,
    'pooling_clips_reduction': pooling_clips_reduction.name,
    'sample_frame_strategy': sample_frame_strategy.name,
    'path_csv_dataset': path_csv_dataset.split('/')[-3:], # get the last 3 folders
    'path_video_dataset': path_video_dataset.split('/')[-3:], # get the last 3 folders
    'features_folder_saving_path': features_folder_saving_path.split('/')[-3:], # get the last 3 folders
    'head': head.name,
    'key_for_early_stopping': key_for_early_stopping,
    # 'stride_window_in_video': stride_window_in_video,
    'head_params': head_params,
    'random_state': seed_random_state,
    'plot_dataset_distribution': is_plot_dataset_distribution,
    'round_output_loss': is_round_output_loss,
    'shuffle_video_chunks': is_shuffle_video_chunks,
    'shuffle_training_batch':is_shuffle_training_batch,
    'clip_length': clip_length,
    'target_metric_best_model': target_metric_best_model,
    'early_stopping': str(early_stopping),
    'soft_labels': soft_labels,
    'regularization_lambda_L2': regularization_lambda_L2,
    }
  ###############################
  # START of the main function  #
  ###############################
  # Check if the global folder exists 
  print(f'Global folder name {global_foder_name}')
  if not os.path.exists(global_foder_name):
    os.makedirs(global_foder_name)
  
  # Create folder to save the run
  # print(f"Creating run folder at {global_foder_name}")
  timestamp = int(time.time()*1000) # get the current time in milliseconds
  run_folder_name = (f'{timestamp}_'+
                     f'{model_type.name}_'+
                     f'{pooling_embedding_reduction.name}_'+
                     f'{pooling_clips_reduction.name}_'+
                     f'{sample_frame_strategy.name}_{head.name}')
  run_folder_path = os.path.join(global_foder_name,run_folder_name) # history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_timestamp
  
  if not os.path.exists(run_folder_path):
    os.makedirs(os.path.join(global_foder_name,run_folder_name))
  set_seed(seed_random_state)  

  # Create the model
  new_csv_path = os.path.join(run_folder_path,'augmented_'+os.path.split(path_csv_dataset)[-1])
  # if stop_after_kth_fold and stop_after_kth_fold[0] != 1 and stop_after_kth_fold[1] != 1:
  #   helper.LOG_CROSS_ATTENTION['enable'] = False
  # else:
  #   helper.LOG_CROSS_ATTENTION['enable'] = True
  model_advanced = Model_Advanced(model_type=model_type,
                                  path_dataset=path_video_dataset,
                                  embedding_reduction=pooling_embedding_reduction,
                                  clips_reduction=pooling_clips_reduction,
                                  sample_frame_strategy=sample_frame_strategy,
                                  stride_window=stride_window_in_video,
                                  path_labels=path_csv_dataset,
                                  batch_size_training=batch_size_training,
                                  head=head.value,
                                  soft_labels=soft_labels,
                                  prefetch_factor=prefetch_factor,
                                  head_params=head_params,
                                  features_folder_saving_path= features_folder_saving_path,
                                  clip_length=clip_length,
                                  concatenate_temporal=concatenate_temp_dim,
                                  n_workers=n_workers,
                                  label_smooth=label_smooth,
                                  dict_augmented=dict_augmented,
                                  new_csv_path=new_csv_path if dict_augmented is not None else None,
                                  )
  
  # print(f"Run folder created at {run_folder_path}")
  
  # Save model configuration
  # print(f"Saving model configuration at {run_folder_path}")
  inputs_dict = get_json_config()
  with open(os.path.join(run_folder_path,'global_config.json'), 'w') as config_file:
    json.dump(inputs_dict, config_file, indent=4,cls=NpEncoder)
      
  # Plot dataset distribution of whole dataset
  # print(f"Plotting dataset distribution at {run_folder_path}")
  if is_plot_dataset_distribution:
    tools.plot_dataset_distribuition(csv_path=path_csv_dataset, run_folder_path=run_folder_path, total_classes=model_advanced.dataset.total_classes)

  # Train the model
  print(f"Start training phase the model at {run_folder_path}")
  train_folder_path = os.path.join(run_folder_path,f'train_{head.value}') # history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_timestamp/train_{head.value}
  
  if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
  
  # Train the model
  start = time.time()
  fold_results = k_fold_cross_validation(path_csv_dataset=new_csv_path if dict_augmented is not None else new_csv_path,
                                          train_folder_path=train_folder_path,
                                          model_advanced=model_advanced,
                                          k_fold=k_fold,
                                          seed_random_state=seed_random_state,
                                          lr=lr,
                                          epochs=epochs,
                                          optimizer_fn=optimizer_fn,
                                          round_output_loss=is_round_output_loss,
                                          shuffle_training_batch=is_shuffle_training_batch,
                                          criterion=criterion,
                                          early_stopping=early_stopping,
                                          concatenate_temp_dim=concatenate_temp_dim,
                                          init_network=init_network,
                                          regularization_lambda_L1=regularization_lambda_L1,
                                          regularization_lambda_L2=regularization_lambda_L2,
                                          key_for_early_stopping=key_for_early_stopping,
                                          enable_scheduler=enable_scheduler,
                                          clip_grad_norm=clip_grad_norm,
                                          target_metric_best_model=target_metric_best_model,
                                          stop_after_kth_fold=stop_after_kth_fold,
                                          trial=trial)
  summary_res = {}
  summary_res['results'] = fold_results['dict_k_fold_results']
  summary_res['time'] = time.time() - start  
  summary_res['config'] = get_obj_config()
  summary_res['config']['head_params']['T_S_S_shape'] = model_advanced.T_S_S_shape                                   
  if helper.LOG_CROSS_ATTENTION['enable']:
    summary_res['cross_attention_debug'] = helper.LOG_CROSS_ATTENTION
    helper.LOG_CROSS_ATTENTION = {'enable':helper.LOG_CROSS_ATTENTION['enable'],
                                  'state':'train'}
    
  tools.save_dict_k_fold_results(dict_k_fold_results=summary_res,
                                  folder_path=run_folder_path)
  model_advanced.free_gpu_memory()
  del model_advanced
  torch.cuda.empty_cache()
  return run_folder_path,summary_res 
 


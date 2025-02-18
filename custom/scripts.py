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
import random
# import wandb

def k_fold_cross_validation(path_csv_dataset, train_folder_path, model_advanced, k_fold, seed_random_state,lr,epochs,optimizer_fn,
                            round_output_loss, shuffle_video_chunks, shuffle_training_batch,  criterion,early_stopping,
                            # scheduler,
                            init_network,regularization_lambda,regularization_loss,key_for_early_stopping,target_metric_best_model):
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
  if not isinstance(model_advanced,Model_Advanced):
    raise ValueError('model_advanced must be an instance of Model_Advanced')
  csv_array,cols = tools.get_array_from_csv(path_csv_dataset)
  sgkf = StratifiedGroupKFold(n_splits=k_fold, random_state=seed_random_state,shuffle=True)
  y_labels = csv_array[:,2].astype(int)
  subject_ids = csv_array[:,0].astype(int)
  sample_ids = csv_array[:,4].astype(int)
  list_splits_idxs = [] # contains indices for all k splits
  for _, test_index in sgkf.split(X=torch.zeros(y_labels.shape), y=y_labels, groups=subject_ids): 
    list_splits_idxs.append(test_index)

  list_test_results = []
  list_best_model_idx = []
  dict_results_model_weights = {}
  fold_results_total = []
  dict_k_fold_results = {}
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
      'test': np.array([test_sample_ids,list_splits_idxs[test_idx_split]]),# [list_sample_id,indices] for test
      'val': np.array([val_sample_ids,list_splits_idxs[val_idx_split]]), # [list_sample_id,indices] for validation
      'train': np.array([train_sample_ids,np.concatenate([list_splits_idxs[idx] for idx in train_idxs_split])])# [list_sample_id,indices] for training
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
      tools.plot_dataset_distribuition(csv_path=csv_path, 
                                  run_folder_path=saving_path_kth_fold,
                                  total_classes=model_advanced.dataset.total_classes)
    sub_k_fold_list = [list_splits_idxs[idx] for idx in train_idxs_split] # Get the split for the subfold considering train in kth fold...
    sub_k_fold_list.append(list_splits_idxs[val_idx_split])               # ... and validation in kth fold
    sub_path_csv_kth_fold = {}
    # dict_k_fold_results = {}
    for sub_idx in range(k_fold-1): # generate the train-val split for the subfold
      saving_path_kth_sub_fold = os.path.join(saving_path_kth_fold,f'k{i}_cross_val_sub_{sub_idx}')
      train_sub_idx = [j for j in range(k_fold-1) if j != sub_idx] # get the train indices for the subfold excluding the validation
      split_indices = {
        'val': np.array([sample_ids[sub_k_fold_list[sub_idx]],sub_k_fold_list[sub_idx]]), # [sample_ids,indices] for validation
        'train': np.array([sample_ids[np.concatenate([sub_k_fold_list[j] for j in train_sub_idx])], # [sample_ids,indices] for training
                            np.concatenate([sub_k_fold_list[j] for j in train_sub_idx])])
      }
      for key,v in split_indices.items(): # generaye csv for train and validation in the subfold
        csv_data = csv_array[v[1]]
        if not os.path.exists(saving_path_kth_sub_fold):
          os.makedirs(saving_path_kth_sub_fold)
        tools.generate_csv(cols=cols, data=csv_data, saving_path=os.path.join(saving_path_kth_sub_fold,f'{key}.csv'))
        sub_path_csv_kth_fold[f'{key}'] = os.path.join(saving_path_kth_sub_fold,f'{key}.csv')
      tools.save_split_indices(split_indices,saving_path_kth_sub_fold)
      # os.environ['WANDB_MODE']='offline'
      # wandb.init(project=f"video_mae_{time.time()}", 
      #             config = config,
      #             name=f'k-{i}_s-{sub_idx}',
      #             reinit=True)
      dict_train = model_advanced.train(lr=lr,
                                        num_epochs=epochs,
                                        optimizer_fn=optimizer_fn,
                                        criterion=criterion,
                                        init_weights=init_network,
                                        saving_path=saving_path_kth_sub_fold,
                                        train_csv_path=sub_path_csv_kth_fold['train'],
                                        val_csv_path=sub_path_csv_kth_fold['val'],
                                        round_output_loss=round_output_loss,
                                        shuffle_video_chunks=shuffle_video_chunks,
                                        shuffle_training_batch=shuffle_training_batch,
                                        init_network=init_network,
                                        regularization_loss=regularization_loss,
                                        regularization_lambda=regularization_lambda,
                                        key_for_early_stopping=key_for_early_stopping,
                                        early_stopping=early_stopping,
                                        # scheduler=scheduler
                                        )
      
      count_epochs = dict_train['dict_results']['epochs'] # get the number of epochs that the model was trained, since it can be stopped before the total number of epochs
      best_model_epoch = dict_train['dict_results']['best_model_idx']
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val-loss'] = dict_train['dict_results']['val_losses'][best_model_epoch]
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train-loss'] = dict_train['dict_results']['train_losses'][best_model_epoch]
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train_macro_accuracy'] = dict_train['dict_results']['list_train_macro_accuracy'][best_model_epoch]
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val_macro_accuracy'] = dict_train['dict_results']['list_val_macro_accuracy'][best_model_epoch]
      
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val-loss-class-avg'] = np.mean(dict_train['dict_results']['val_loss_per_class'][best_model_epoch])
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_val-loss-subject-avg'] = np.mean(dict_train['dict_results']['val_loss_per_subject'][best_model_epoch])
      
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train-loss-class-avg'] = np.mean(dict_train['dict_results']['train_loss_per_class'][best_model_epoch])
      # dict_k_fold_logs[f'k-{i}_s-{sub_idx}_train-loss-subject-avg'] = np.mean(dict_train['dict_results']['train_loss_per_subject'][best_model_epoch])
      
      tools.plot_loss_and_precision_details(dict_train=dict_train,train_folder_path=saving_path_kth_sub_fold,total_epochs=count_epochs)
      # ADD ACCURACY IN CSV FOR TRAIN VAL and TEST
      # tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
      #                               count_subject_ids_train=dict_train['count_subject_ids_train'],
      #                               count_subject_ids_test=dict_train['count_subject_ids_test'],
      #                               count_y_test=dict_train['count_y_test'], 
      #                               count_y_train=dict_train['count_y_train'],
      #                               saving_path=saving_path_kth_sub_fold)
      # tools.plot_confusion_matrix
      #####################################################################
      ######################dict_train['dict_results']#####################
        # 'train_losses': train_losses,
        # 'train_loss_per_class': train_loss_per_class,
        # 'train_loss_per_subject': train_loss_per_subject,
        # 'val_losses': val_losses,
        # 'val_loss_per_class': val_loss_per_class,
        # 'val_loss_per_subject': val_loss_per_subject,
        # 'y_unique': unique_classes,
        # 'subject_ids_unique': unique_subjects,
        # 'train_confusion_matricies': train_confusion_matricies,
        # 'val_confusion_matricies': val_confusion_matricies,
        # 'best_model_idx': best_model_epoch,
        # 'best_model_state': best_model_state,
        # 'list_train_macro_accuracy': list_train_macro_accuracy,
        # 'list_val_macro_accuracy': list_val_macro_accuracy,
        # 'epochs': count_epoch
      ###################################################################
      fold_results_kth.append(dict_train)
      list_to_reduce_for_logs = [ 'train_loss_per_class','train_loss_per_subject',          #Keep entire data for:  
                                  'val_loss_per_class','val_loss_per_subject',
                                  'train_confusion_matricies','val_confusion_matricies']
      
      # reduce the logs to save space, keep only the results every 50 epochs and the best model epoch
      reduced_dict_train = {}
      for key,v in dict_train['dict_results'].items():
        if key not in 'best_model_state':
          if key in list_to_reduce_for_logs:
            reduced_dict_train[key] = {f'{epoch}':v[epoch] for epoch in range(0,count_epochs,50)} # to get the results every 50 epochs
            # reduced_dict_train[key] = v[::50]
            if dict_train['dict_results']['best_model_idx'] % 50 != 0:
              if isinstance(v,list):
                # reduced_dict_train[key].append(v[dict_train['dict_results']['best_model_idx']])
                reduced_dict_train[key].update({f'{dict_train["dict_results"]["best_model_idx"]}':v[dict_train["dict_results"]["best_model_idx"]]})
              elif isinstance(v,np.ndarray):
                reduced_dict_train[key].update({f'{dict_train["dict_results"]["best_model_idx"]}':v[dict_train["dict_results"]["best_model_idx"]]})
                # reduced_dict_train[key] = np.append(reduced_dict_train[key],v[dict_train['dict_results']['best_model_idx']])
              else:
                raise ValueError('Error in reducing the list')            
              # np.append(reduced_dict_train[k],v[dict_train['dict_results']['best_model_idx']])
              # reduced_dict_train[k].append(v[dict_train['dict_results']['best_model_idx']])
          else:
            reduced_dict_train[key] = v
      dict_k_fold_results[f'k{i}_cross_val_sub_{sub_idx}_train_val'] = reduced_dict_train
      # dict_k_fold_results.append({f'k{i}_cross_val_sub_{sub_idx}_train_val': reduced_dict_train})

    best_results_idx = [fold_results_kth[i]['dict_results']['best_model_idx'] for i in range(k_fold-1)]
    best_results_state_dict = [fold_results_kth[i]['dict_results']['best_model_state'] for i in range(k_fold-1)]
    
    # Use best result model with the test set according to the validation loss or validation accuracy
    target_metric_best_model_kth = 'val_losses' if target_metric_best_model == 'val_loss' else 'list_val_macro_accuracy'
    list_validation_best_result = [dict_train['dict_results'][target_metric_best_model_kth] for dict_train in fold_results_kth]
    if target_metric_best_model_kth == 'val_losses':
      best_model_subfolder_idx = np.argmin([metric[best_results_idx[i]] for i,metric in enumerate(list_validation_best_result)])
    elif target_metric_best_model_kth == 'val_macro_accuracy':
      best_model_subfolder_idx = np.argmax([metric[best_results_idx[i]] for i,metric in enumerate(list_validation_best_result)])
    else:
      raise ValueError('target_metric_best_model must be val_losses or val_macro_accuracy')
    
    best_model_epoch = fold_results_kth[best_model_subfolder_idx]['dict_results']['best_model_idx']
    list_best_model_idx.append(best_model_epoch)
    path_model_weights_best_model = os.path.join(saving_path_kth_fold,f'k{i}_cross_val_sub_{best_model_subfolder_idx}',f'best_model_ep_{best_model_epoch}.pth')
    # list_path_model_weights.append(path_model_weights)
    dict_results_model_weights[f'{i}'] = {'sub':best_model_subfolder_idx,'epoch':best_model_epoch}
    
    # keep only the best model removing the others
    for key,dict_best_result in dict_results_model_weights.items():
      saving_path_kth_fold = os.path.join(train_folder_path,f'k{key}_cross_val')
      key=int(key)
      for j in range(k_fold-1):
        # delete models except the best one
        if j != dict_best_result['sub']:
          path_folder_model_weights = os.path.join(saving_path_kth_fold,f'k{key}_cross_val_sub_{j}',)
          # Get .pth file
          for file in os.listdir(path_folder_model_weights):
            if file.endswith(".pth"):
              os.remove(os.path.join(path_folder_model_weights, file))
      
    dict_test = model_advanced.test_pretrained_model(path_model_weights=path_model_weights_best_model,
                                                      csv_path=path_csv_kth_fold['test'], 
                                                      log_file_path=os.path.join(saving_path_kth_fold,'test_results.txt'),
                                                      is_test=True,
                                                      criterion=criterion,
                                                      round_output_loss=round_output_loss)
    
    dict_k_fold_results[f'k{i}_test'] = { 'dict_test':dict_test,
                                          'best_model_subfolder_idx':best_model_subfolder_idx
                                        }
    # dict_k_fold_results.append({f'k{i}_cross_val_test':{'dict_test':dict_test,
    #                                                     'best_model_subfolder_idx':best_model_subfolder_idx}})
    
    tools.plot_confusion_matrix(confusion_matrix=dict_test['test_confusion_matrix'],
                                title=f'Confusion matrix Test folder k-{i} considering best model',
                                saving_path=os.path.join(saving_path_kth_fold,f'confusion_matrix_test_submodel_{best_model_subfolder_idx}.png'))
    
    # Log test results conisdering the best model selected according to the target_metric_best_model
    # dict_k_fold_logs[f'k-{i}_test-loss'] = dict_test['test_loss']
    # dict_k_fold_logs[f'k-{i}_test-loss-class'] = dict_test['test_loss_per_class']
    # dict_k_fold_logs[f'k-{i}_test-loss-subject'] = dict_test['test_loss_per_subject']
    # dict_k_fold_logs[f'k-{i}_test-accurracy'] = dict_test['test_macro_precision']
    
    # # Log train for the best model in the kth fold
    # dict_k_fold_logs[f'k-{i}_train-loss'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['train_losses'][best_model_epoch]
    # dict_k_fold_logs[f'k-{i}_train-loss-class'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['train_loss_per_class'][best_model_epoch]
    # dict_k_fold_logs[f'k-{i}_train-loss-subject'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['train_loss_per_subject'][best_model_epoch]
    # dict_k_fold_logs[f'k-{i}_train_accuracy'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['list_train_macro_accuracy'][best_model_epoch]
    
    # # Log val for the best model in the kth fold
    # dict_k_fold_logs[f'k-{i}_val-loss'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['val_losses'][best_model_epoch]
    # dict_k_fold_logs[f'k-{i}_val-loss-class'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['val_loss_per_class'][best_model_epoch]
    # dict_k_fold_logs[f'k-{i}_val-loss-subject'] =fold_results_kth[best_model_subfolder_idx]['dict_results']['val_loss_per_subject'][best_model_epoch] 
    # dict_k_fold_logs[f'k-{i}_val_accuracy'] = fold_results_kth[best_model_subfolder_idx]['dict_results']['list_val_macro_accuracy'][best_model_epoch]
    
    list_test_results.append(dict_test) # contains the results for each kth fold
    fold_results_total.append(fold_results_kth[best_model_subfolder_idx])
    
  
  # dict_final_test_results = {
  #   'avg_test_loss_best_models': np.mean([dict_test['test_loss'] for dict_test in list_test_results]),
  #   'avg_test_loss_per_class_best_models': np.mean([dict_test['test_loss_per_class'] for dict_test in list_test_results],axis=(0)),
  #   'avg_test_loss_per_subject_best_models': np.mean([dict_test['test_loss_per_subject'] for dict_test in list_test_results],axis=(0)),
  #   'avg_accuracy_best_models': np.mean([dict_test['test_macro_precision'] for dict_test in list_test_results]),
  # }

  # dict_k_fold_logs['avg_test_loss-avg'] = dict_final_test_results['avg_test_loss_best_models']
  # dict_k_fold_logs['avg_test_loss-class-avg'] = dict_final_test_results['avg_test_loss_per_class_best_models']
  # dict_k_fold_logs['avg_test_loss-subject-avg'] = dict_final_test_results['avg_test_loss_per_subject_best_models']
  # dict_k_fold_logs['avg_test_accuracy'] = dict_final_test_results['avg_accuracy_best_models']
  
  if target_metric_best_model == 'val_loss':
    final_best_model_folder_idx = np.argmin([dict_val['test_loss' ] for dict_val in list_test_results])
  elif target_metric_best_model == 'val_macro_accuracy':
    final_best_model_folder_idx = np.argmax([dict_val['test_macro_precision'] for dict_val in list_test_results])
  
  # dict_k_fold_logs['final_best_model_folder_idx'] = {'folder':final_best_model_folder_idx,
  #                                                     'sub':dict_results_model_weights[f"{final_best_model_folder_idx}"]["sub"]}

  plot_models_test_loss = {}
  plot_subject_test_loss = {}
  plot_class_test_loss = {}

  for i,dict_test in enumerate(list_test_results):
    plot_models_test_loss[f'k_{i}'] = dict_test['test_loss']
    plot_subject_test_loss[f'k_{i}']=dict_test['test_loss_per_subject']
    plot_class_test_loss[f'k_{i}'] = dict_test['test_loss_per_class']
    
  tools.plot_bar(data=plot_models_test_loss,
                  x_label='k_fold nr.',
                  y_label='loss',
                  title='Test Loss per k_fold',
                  saving_path=os.path.join(train_folder_path,'test_loss_per_k_fold.png'))
  # TODO: FIX y_axis to be consistent in all plots
  # TODO: PUT the subject ID and not the array position of subject_id
  tools.subplot_loss(dict_losses=plot_subject_test_loss,
                      list_title=[f'Test Subject loss {k_fold} model {list_best_model_idx[k_fold]}' for k_fold in range(k_fold)],
                      saving_path=os.path.join(train_folder_path,'subject_test_loss_per_k_fold.png'),
                      x_label='subject_id',
                      y_label='loss')
  tools.subplot_loss(dict_losses=plot_class_test_loss,
                      list_title=[f'Test Class loss {k_fold} model {list_best_model_idx[k_fold]}' for k_fold in range(k_fold)],
                      saving_path=os.path.join(train_folder_path,'class_test_loss_per_k_fold.png'),
                      x_label='class_id',
                      y_label='loss')
  
  # dict_k_fold_logs['folder_path'] = train_folder_path
  # wandb.finish()
  return {
          'list_test_results':list_test_results,
          # 'list_saving_paths':list_saving_paths,
          # 'list_path_csv_kth_fold':list_path_csv_kth_fold,
          'best_results_idx':best_results_idx,
          'best_results_state_dict':best_results_state_dict,
          'dict_k_fold_results':dict_k_fold_results
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
                   regularization_lambda,regularization_loss,
                   clip_length,
                   target_metric_best_model,
                   early_stopping,
                  #  scheduler
                   ):
 

  def set_seed(seed):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # PyTorch (GPU)
    torch.cuda.manual_seed_all(seed)  # Multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # May slow down training but ensures reproducibility
  def get_obj_config():
    return{
    'k_fold': k_fold,
    'model_type': model_type,
    'epochs': epochs,
    'optimizer_fn': optimizer_fn.__name__,
    'lr': lr,
    'criterion': criterion,
    'init_network': init_network,
    'regularization_lambda': regularization_lambda,
    'regularization_loss': regularization_loss,
    'batch_size_training': batch_size_training,
    'pooling_embedding_reduction': pooling_embedding_reduction,
    'pooling_clips_reduction': pooling_clips_reduction,
    'sample_frame_strategy': sample_frame_strategy,
    'path_csv_dataset': path_csv_dataset,
    'path_video_dataset': path_video_dataset,
    'head': head,
    'key_for_early_stopping': key_for_early_stopping,
    'stride_window_in_video': stride_window_in_video,
    'head_params': head_params,
    'random_state': seed_random_state,
    'plot_dataset_distribution': is_plot_dataset_distribution,
    'round_output_loss': is_round_output_loss,
    'shuffle_video_chunks': is_shuffle_video_chunks,
    'shuffle_training_batch':is_shuffle_training_batch,
    'features_folder_saving_path': features_folder_saving_path,
    'clip_length': clip_length,
    'target_metric_best_model': target_metric_best_model,
    'early_stopping': early_stopping,
      
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
    'regularization_lambda': regularization_lambda,
    'regularization_loss': regularization_loss,
    'batch_size_training': batch_size_training,
    'pooling_embedding_reduction': pooling_embedding_reduction.name,
    'pooling_clips_reduction': pooling_clips_reduction.name,
    'sample_frame_strategy': sample_frame_strategy.name,
    'path_csv_dataset': path_csv_dataset,
    'path_video_dataset': path_video_dataset,
    'head': head.name,
    'key_for_early_stopping': key_for_early_stopping,
    'stride_window_in_video': stride_window_in_video,
    'head_params': head_params,
    'random_state': seed_random_state,
    'plot_dataset_distribution': is_plot_dataset_distribution,
    'round_output_loss': is_round_output_loss,
    'shuffle_video_chunks': is_shuffle_video_chunks,
    'shuffle_training_batch':is_shuffle_training_batch,
    'features_folder_saving_path': features_folder_saving_path,
    'clip_length': clip_length,
    'target_metric_best_model': target_metric_best_model,
    'early_stopping': str(early_stopping),
    }
  ###############################
  # START of the main function  #
  ###############################
  set_seed(seed_random_state)

  # Create the model
  model_advanced = Model_Advanced(model_type=model_type,
                                  path_dataset=path_video_dataset,
                                  embedding_reduction=pooling_embedding_reduction,
                                  clips_reduction=pooling_clips_reduction,
                                  sample_frame_strategy=sample_frame_strategy,
                                  stride_window=stride_window_in_video,
                                  path_labels=path_csv_dataset,
                                  batch_size_training=batch_size_training,
                                  head=head.value,
                                  head_params=head_params,
                                  features_folder_saving_path= features_folder_saving_path,
                                  clip_length=clip_length,
                                  )
  
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
  if k_fold == 1:
    raise ValueError('K must be greater than 1')
  else: 
    fold_results = k_fold_cross_validation(path_csv_dataset=path_csv_dataset,
                                           train_folder_path=train_folder_path,
                                           model_advanced=model_advanced,
                                           k_fold=k_fold,
                                           seed_random_state=seed_random_state,
                                           lr=lr,
                                           epochs=epochs,
                                           optimizer_fn=optimizer_fn,
                                           round_output_loss=is_round_output_loss,
                                           shuffle_video_chunks=is_shuffle_video_chunks,
                                           shuffle_training_batch=is_shuffle_training_batch,
                                          #  config=get_json_config(),
                                           criterion=criterion,
                                           early_stopping=early_stopping,
                                          #  scheduler=scheduler,
                                           init_network=init_network,
                                           regularization_lambda=regularization_lambda,
                                           regularization_loss=regularization_loss,
                                           key_for_early_stopping=key_for_early_stopping,
                                           target_metric_best_model=target_metric_best_model)
    fold_results['dict_k_fold_results']['config'] = get_obj_config()                                     
    tools.save_dict_k_fold_results(dict_k_fold_results=fold_results['dict_k_fold_results'],
                                   folder_path=run_folder_path)
    # return fold_results
    # log_dict = {}
    # for k,v in inputs_dict.items():
    #   if k not in key_to_remove_inputs_dict:
    #     log_dict[k] = v
    # log_dict.update(fold_results['dict_k_fold_logs'])
    # return log_dict
 


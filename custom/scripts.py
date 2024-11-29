from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY
import os
from custom.model import Model_Advanced
from transformers import AutoImageProcessor
from custom.head import HeadSVR, HeadGRU
import time
import custom.tools as tools
from custom.helper import HEAD, MODEL_TYPE, EMBEDDING_REDUCTION, SAMPLE_FRAME_STRATEGY
import torch.nn as nn
import torch.optim as optim
import dataframe_image as dfi
import pandas as pd
import torch
import numpy as np

def run_train_test(model_type, pooling_embedding_reduction, pooling_clips_reduction, sample_frame_strategy, 
                   path_csv_dataset, path_video_dataset, head, stride_window_in_video, 
                   head_params, preprocess,
                   train_size = 0.8, val_size=0.1, test_size=0.1, download_if_unavailable=False, batch_size=1,
                   k_fold = 1, epochs = 10, criterion = nn.L1Loss(), optimizer_fn = optim.Adam, lr = 0.001):
  
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
    
    tools.plot_dataset_distribution_mean_std_duration(csv_path=csv_path,
                                                      video_path=path_video_dataset,
                                                      per_class=per_class, 
                                                      per_partecipant=per_partecipant, 
                                                      saving_path=dataset_folder_path) # 2 plots

  def train_and_plot_model_results(dict_csv_path, train_folder_saving_path, batch_size=1):
    """
    Trains a model using the specified dataset and saves the results.\n
    The function performs the following steps:
      1. Trains the model using the training dataset.
      2. Generates and saves plots of the training and test results.
      3. Plots and saves confusion matrices for each epoch.
    Args:
      dict_csv_path (dict): Path to the train, test, validation csv.
      train_folder_saving_path (str): Path to the folder where training results will be saved.
    Returns:
      dict: A dictionary containing training results, including metrics and other relevant information.
    """

    # Train the model  
    dict_train = model_advanced.train(train_csv_path=dict_csv_path['train'],
                                      test_csv_path=dict_csv_path['test'],
                                      num_epochs=epochs, 
                                      batch_size=batch_size, 
                                      criterion=criterion,
                                      optimizer_fn=optimizer_fn,
                                      lr=lr,
                                      saving_path=train_folder_saving_path)
    
    # Generate and save plots of the training and test results
    tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
                                  count_subject_ids_train=dict_train['count_subject_ids_train'],
                                  count_subject_ids_test=dict_train['count_subject_ids_test'],
                                  count_y_test=dict_train['count_y_test'], 
                                  count_y_train=dict_train['count_y_train'],
                                  saving_path=train_folder_saving_path)
    
    # Plot and save confusion matrices for each epoch
    confusion_matrix_path = os.path.join(train_folder_saving_path,'confusion_matricies')
    if not os.path.exists(confusion_matrix_path):
      os.makedirs(confusion_matrix_path)
    
    _plot_confusion_matricies(epochs, dict_train, confusion_matrix_path)

    return dict_train  

  def k_fold_cross_validation():
    """
    Perform k-fold cross-validation, train models, and save results.
    This function performs k-fold cross-validation on a dataset, trains models for each fold,
    and saves the results and plots. It also calculates and saves the average losses for the best models.
    Returns:
      fold_results (list): A list containing the results for each fold.
      list_saving_paths (list): A list of paths where the results for each fold are saved.
      best_results_idx (list): A list of indices indicating the best model for each fold.
    """
    fold_results = []
    list_saving_paths = []

    for i in range(k_fold):
      saving_path_kth_fold = os.path.join(train_folder_path,f'results_k{i}_cross_val')
      list_saving_paths.append(saving_path_kth_fold)
      if not os.path.exists(saving_path_kth_fold):
        os.makedirs(saving_path_kth_fold)
      path_csv_kth_fold = tools._generate_train_test_validation(csv_path=model_advanced.dataset.path_labels,
                                                              saving_path=saving_path_kth_fold,
                                                              train_size=train_size,
                                                              val_size=val_size,
                                                              test_size=test_size,
                                                              random_state=i)
       
      for _,csv_path in path_csv_kth_fold.items():
        plot_dataset_distribuition(csv_path=csv_path, 
                                   run_folder_path=path_csv_kth_fold,
                                   total_classes=model_advanced.dataset.total_classes)

      result = train_and_plot_model_results(dict_csv_path= path_csv_kth_fold,
                                            train_folder_saving_path=saving_path_kth_fold,
                                            batch_size=batch_size)
      fold_results.append(result)

    best_results_idx = [fold_results[i]['best_model_idx'] for i in range(k_fold)]
    dict_all_results = { 
                        'avg_train_loss_best_models': np.mean([fold_results[best_results_idx[i]]['dict_results']['train_losses'][-1] for i in range(k_fold)]),
                        'avg_test_loss_best_models': np.mean([fold_results[best_results_idx[i]]['dict_results']['test_losses'][-1] for i in range(k_fold)]),
                        'avg_train_loss_per_class_best_models': np.mean([fold_results[best_results_idx[i]]['dict_results']['train_loss_per_class'][-1] for i in range(k_fold)]),
                        'avg_test_loss_per_class_best_models': np.mean([fold_results[best_results_idx[i]]['dict_results']['test_loss_per_class'][-1] for i in range(k_fold)]),
                        'avg_train_loss_per_subject_best_models': np.mean([fold_results[best_results_idx[i]]['dict_results']['train_loss_per_subject'][-1] for i in range(k_fold)]),
                        'avg_test_loss_per_subject_best_models': np.mean([fold_results[best_results_idx[i]]['dict_results']['test_loss_per_subject'][-1] for i in range(k_fold)])
                        }
    with open(os.path.join(train_folder_path,'results_k_fold.txt'),'w') as f:
      f.write(str(dict_all_results))
    return fold_results, list_saving_paths, best_results_idx
    
  # Create the model
  model_advanced = Model_Advanced(model_type=model_type,
                                  path_dataset=path_video_dataset,
                                  embedding_reduction=pooling_embedding_reduction,
                                  clips_reduction=pooling_clips_reduction,
                                  sample_frame_strategy=sample_frame_strategy,
                                  stride_window=stride_window_in_video,
                                  path_labels=path_csv_dataset,
                                  preprocess=preprocess,
                                  batch_size=batch_size,
                                  head=head.value,
                                  head_params=head_params,
                                  download_if_unavailable=download_if_unavailable
                                  )
  
  # Check if the global folder exists 
  global_foder_name = 'history_run'
  if not os.path.exists(global_foder_name):
    os.makedirs(global_foder_name)
  
  # Create folder to save the run
  print(f"Creating run folder at {global_foder_name}")
  run_folder_name = (f'{model_type.name}_'+
                     f'{pooling_embedding_reduction.name}_'+
                     f'{pooling_clips_reduction.name}_'+
                     f'{sample_frame_strategy.name}_{head.name}_{int(time.time())}')
  run_folder_path = os.path.join(global_foder_name,run_folder_name) # history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_timestamp
  if not os.path.exists(run_folder_path):
    os.makedirs(os.path.join(global_foder_name,run_folder_name))
  print(f"Run folder created at {run_folder_path}")
  
  # Save model configuration
  print(f"Saving model configuration at {run_folder_path}")
  model_advanced.save_configuration(os.path.join(run_folder_path,'advancedmodel_config.json'))
  
  # Plot dataset distribution of whole dataset
  print(f"Plotting dataset distribution at {run_folder_path}")
  plot_dataset_distribuition(csv_path=path_csv_dataset, run_folder_path=run_folder_path, total_classes=model_advanced.dataset.total_classes)

  # Train the model
  print(f"Start training phase the model at {run_folder_path}")
  train_folder_path = os.path.join(run_folder_path,f'train_{head.value}')
  if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
  if k_fold == 1:
    print("Generating train, test and validation csv files")
    dict_cvs_path = tools._generate_train_test_validation(csv_path=path_csv_dataset,
                                                          saving_path=train_folder_path,
                                                          train_size=train_size,
                                                          val_size=val_size,
                                                          test_size=test_size)
    print("Plotting train,test,val distribution")
    for _,csv_path in dict_cvs_path.items():
      plot_dataset_distribuition(csv_path=csv_path,run_folder_path=train_folder_path, total_classes=model_advanced.dataset.total_classes)
    print("Training model")
    dict_train = train_and_plot_model_results(dict_csv_path=dict_cvs_path, train_folder_saving_path=train_folder_path)

  else: 
    fold_results, list_saving_paths_k_val, best_results_idx = k_fold_cross_validation(batch_size=batch_size)

  # create folder to save tsne plots
  print(f"Creating folder to save tsne plots at {train_folder_path}")
  saving_path_tsne = os.path.join(train_folder_path,'tsne')
  if not os.path.exists(saving_path_tsne):
    os.mkdir(saving_path_tsne)

  # plot tsne considering all features from backbone per subject and per class
  for k,v in dict_cvs_path.items():
    print(f"Plotting tsne for {k}")
    dict_feats = model_advanced._extract_features(path_csv_dataset = v)
    y = dict_feats['list_labels']
    tools.plot_tsne(X=dict_feats['features'],
                    labels=dict_feats['list_labels'],
                    saving_path=os.path.join(saving_path_tsne,'tsne_backbone_gt'),
                    legend_label='class ',
                    from_dataset=f'{k} (from backbone)')
    
    tools.plot_tsne(X=dict_feats['features'],
                    labels=dict_feats['list_subject_id'],
                    saving_path=os.path.join(saving_path_tsne,'tsne_backbone_subject'),
                    legend_label='subject',
                    from_dataset=f'{k} (from backbone)')
    
    if head.value == HEAD.GRU.value: # test split as above
      X_GRU = model_advanced.head.model.gru(dict_feats['features'].reshape(dict_feats['features'].shape[0],dict_feats['features'].shape[1],-1))
      y_pred = model_advanced.head.model.fc(X_GRU[1]) # TODO: separate in train and test
      tools.plot_tsne(X = X_GRU[1].detach().cpu(),
                    labels = torch.round(y_pred.detach().cpu()),
                    from_dataset = f' {k} prediction (using GRU features)',
                    legend_label='class',
                    saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_pred_{k}'))
    
      tools.plot_tsne(X = X_GRU[1].detach().cpu(),
                      labels = y.detach().cpu(),
                      legend_label = 'class ',
                      from_dataset = f' {k} groundtruth (using GRU features)',
                      saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_gt_{k}'),)
      
      tools.plot_tsne(X = X_GRU[1].detach().cpu(),
                      labels = dict_feats['list_subject_id'],
                      from_dataset = f' {k} subjects (using GRU)',
                      legend_label='subject',
                    saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_subject_{k}'))

    elif head.value == HEAD.SVR.value: 
      y_pred = model_advanced.head.predict(dict_feats['features'])
      tools.plot_tsne(dict_feats['features'],
                      y_pred,'pred ',
                      from_dataset=f'{k} (using SVR)',
                      saving_path=os.path.join(saving_path_tsne,f'tsne_SVR_pred_{k}'))
    
    # Create video with predictions
    print(f"Creating video with predictions for {k}")
    video_folder_path = os.path.join(train_folder_path,f'video')
    if not os.path.exists(video_folder_path):
      os.makedirs(video_folder_path)
    
    list_input_video_path = tools.get_list_video_path_from_csv(dict_cvs_path[k])
    output_video_path = os.path.join(video_folder_path,f'video_{k}')
    print(f'list_frame shape: {dict_feats["list_frames"].shape}')
    # Separare che ho la prediction per ogni video dato che uso la sliding windows in input alla GRU
    # Create metodo che dato un dataset per ogni sliding in list_frame fa la predicition e poi salvare il video
    model_advanced.dataset.save_frames_as_video(list_input_video_path=list_input_video_path,
                                                list_frame_indices=dict_feats['list_frames'],
                                                output_video_path=output_video_path,
                                                all_predictions=y_pred.detach().cpu().numpy(),
                                                list_ground_truth=y.detach().cpu().numpy(),
                                                output_fps=1)

def _plot_confusion_matricies(epochs, dict_train, confusion_matrix_path):
  for epoch in range(epochs): 
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][epoch],
                                title='Train confusion matrix',
                                saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_train_{epoch}.png'))
    
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['test_confusion_matricies'][epoch],
                              title='Test confusion matrix',
                              saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_test_{epoch}.png'))
    

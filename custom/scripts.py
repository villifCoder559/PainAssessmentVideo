from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY
import os
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



def run_train_test(model_type, pooling_embedding_reduction, pooling_clips_reduction, sample_frame_strategy, 
                   path_csv_dataset, path_video_dataset, head, stride_window_in_video, 
                   head_params, preprocess,k_fold=1, is_save_features_extracted=False,
                   train_size = 0.8, val_size=0.1, test_size=0.1, is_download_if_unavailable=False, 
                   batch_size_training = 1, batch_size_feat_extraction = 3,epochs = 10, 
                   criterion = nn.L1Loss(), optimizer_fn = optim.Adam, lr = 0.001,random_state_split_dataset=42,
                   is_plot_dataset_distribution=True,is_plot_loss=True,is_plot_tsne_backbone_feats=True,is_plot_tsne_head_pred=True,
                   is_plot_tsne_gru_feats=True,is_create_video_prediction=True,is_create_video_prediction_per_video=True,is_validation=False,
                   is_round_output_loss=False, is_shuffle_video_chunks=True,is_shuffle_training_batch=True):

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

  def train_model(dict_csv_path, train_folder_saving_path, is_validation=False,round_output_loss=False,shuffle_video_chunks=True,
                  shuffle_training_batch=True):
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
                                      shuffle_training_batch=shuffle_training_batch
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
    
  def k_fold_cross_validation(is_validation=False,round_output_loss=False,shuffle_video_chunks=True,shuffle_training_batch=True):
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
    fold_results = []
    list_saving_paths = []
    list_path_csv_kth_fold = []
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
      list_path_csv_kth_fold.append(path_csv_kth_fold) 
      for _,csv_path in path_csv_kth_fold.items():
        plot_dataset_distribuition(csv_path=csv_path, 
                                   run_folder_path=saving_path_kth_fold,
                                   total_classes=model_advanced.dataset.total_classes)

      dict_train = train_model(dict_csv_path= path_csv_kth_fold,
                               train_folder_saving_path=saving_path_kth_fold,
                               is_validation=is_validation,
                               round_output_loss=round_output_loss,
                               shuffle_video_chunks=shuffle_video_chunks,
                               shuffle_training_batch=shuffle_training_batch)
            # Generate and save plots of the training and test results
      # tools.plot_losses(train_losses=dict_train['dict_results']['train_losses'], 
      #                   test_losses=dict_train['dict_results']['test_losses'], 
      #                   saving_path=os.path.join(saving_path_kth_fold,'train_test_losses'))
      
      tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
                                    count_subject_ids_train=dict_train['count_subject_ids_train'],
                                    count_subject_ids_test=dict_train['count_subject_ids_test'],
                                    count_y_test=dict_train['count_y_test'], 
                                    count_y_train=dict_train['count_y_train'],
                                    saving_path=saving_path_kth_fold)
      fold_results.append(dict_train)

    best_results_idx = [fold_results[i]['dict_results']['best_model_idx'] for i in range(k_fold)]
    print(f'best_results_idx {best_results_idx}')
    dict_all_results = { 
                        'avg_train_loss_best_models': np.mean([fold_results[i]['dict_results']['train_losses'][best_results_idx[i]] for i in range(k_fold)]),
                        'avg_test_loss_best_models': np.mean([fold_results[i]['dict_results']['test_losses'][best_results_idx[i]] for i in range(k_fold)]),
                        'avg_train_loss_per_class_best_models': np.mean([fold_results[i]['dict_results']['train_loss_per_class'][best_results_idx[i]] for i in range(k_fold)]),
                        'avg_test_loss_per_class_best_models': np.mean([fold_results[i]['dict_results']['test_loss_per_class'][best_results_idx[i]] for i in range(k_fold)]),
                        # 'avg_train_loss_per_subject_best_models': np.mean([fold_results[i]['dict_results']['train_loss_per_subject'][best_results_idx[i]] for i in range(k_fold)]),
                        # 'avg_test_loss_per_subject_best_models': np.mean([fold_results[i]['dict_results']['test_loss_per_subject'][best_results_idx[i]] for i in range(k_fold)])
                        }
    with open(os.path.join(train_folder_path,'results_k_fold.txt'),'w') as f:
      f.write(str(dict_all_results))
    return {'fold_results':fold_results,
            'list_saving_paths':list_saving_paths,
            'list_path_csv_kth_fold':list_path_csv_kth_fold,
            'best_results_idx':best_results_idx}
    
  ###############################
  # START of the main function  #
  ###############################
  inputs_dict = {
    'model_type': model_type.name,
    'pooling_embedding_reduction': pooling_embedding_reduction.name,
    'pooling_clips_reduction': pooling_clips_reduction.name,
    'sample_frame_strategy': sample_frame_strategy.name,
    'path_csv_dataset': path_csv_dataset,
    'path_video_dataset': path_video_dataset,
    'head': head.name,
    'stride_window_in_video': stride_window_in_video,
    'head_params': head_params,
    'preprocess': preprocess.__class__.__name__,
    'k_fold': k_fold,
    'train_size': train_size,
    'val_size': val_size,
    'test_size': test_size,
    'download_if_unavailable': is_download_if_unavailable,
    'batch_size_feat_extraction': batch_size_feat_extraction,
    'batch_size_training': batch_size_training,
    'epochs': epochs,
    'criterion': criterion.__class__.__name__,
    'optimizer_fn': optimizer_fn.__name__,
    'lr': lr,
    'criterion': type(criterion).__name__,
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
    'shuffle_training_batch':is_shuffle_training_batch
  }
  features_folder_saving_path = os.path.join('partA','video','features',f'{os.path.split(path_csv_dataset)[-1][:-4]}_{stride_window_in_video}') # get the name of the csv file
  # Create the model
  model_advanced = Model_Advanced(model_type=model_type,
                                  path_dataset=path_video_dataset,
                                  embedding_reduction=pooling_embedding_reduction,
                                  clips_reduction=pooling_clips_reduction,
                                  sample_frame_strategy=sample_frame_strategy,
                                  stride_window=stride_window_in_video,
                                  path_labels=path_csv_dataset,
                                  preprocess=preprocess,
                                  batch_size_training=batch_size_training,
                                  batch_size_feat_extraction=batch_size_feat_extraction,
                                  head=head.value,
                                  head_params=head_params,
                                  download_if_unavailable=is_download_if_unavailable,
                                  features_folder_saving_path= features_folder_saving_path
                                  )
  
  # Check if the global folder exists 
  global_foder_name = 'history_run'
  if not os.path.exists(global_foder_name):
    os.makedirs(global_foder_name)
  
  # Create folder to save the run
  # print(f"Creating run folder at {global_foder_name}")
  run_folder_name = (f'{model_type.name}_'+
                     f'{pooling_embedding_reduction.name}_'+
                     f'{pooling_clips_reduction.name}_'+
                     f'{sample_frame_strategy.name}_{head.name}_{int(time.time())}')
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
    print("Training model")
    dict_train = train_model(is_validation=is_validation,
                             dict_csv_path=dict_cvs_path,
                             train_folder_saving_path=train_folder_path,
                             round_output_loss=is_round_output_loss,
                             shuffle_video_chunks=is_shuffle_video_chunks)
    
    # plot loss details
    plot_loss_details(dict_train, train_folder_path, epochs)
    list_dict_csv_path.append(dict_cvs_path)
    list_saving_paths_k_fold.append(train_folder_path)
    
  else: 
    fold_results = k_fold_cross_validation(is_validation=is_validation,
                                           round_output_loss=is_round_output_loss,
                                           shuffle_video_chunks=is_shuffle_video_chunks,
                                           shuffle_training_batch=is_shuffle_training_batch)
    list_dict_csv_path = fold_results['list_path_csv_kth_fold'] # list of dictionaries with paths to CSV files for each fold
    list_saving_paths_k_fold = fold_results['list_saving_paths'] # list of saving paths for each fold
    for i,dict_train in enumerate(fold_results['fold_results']):
      plot_loss_details(dict_train=dict_train, 
                        train_folder_path=list_saving_paths_k_fold[i],
                        epochs=epochs)
  

  key_test_dataset = 'test' if not is_validation else 'val'
  # dict_cvs_path = dict_cvs_path[key_test_dataset]
  # For each dataset used plot tsne considering all features from backbone per subject and per class
  for dict_cvs_path,train_folder_path in zip(list_dict_csv_path,list_saving_paths_k_fold):
    # create folder to save tsne plots
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
        tools.plot_tsne(X=backbone_feats,
                      labels=y_gt,
                      saving_path=os.path.join(saving_path_tsne,f'tsne_backbone_gt_{split_dataset_name}'),
                      legend_label='class ',
                      title=f'dataset {split_dataset_name} (from backbone per chunks)')
      
        tools.plot_tsne(X=backbone_feats,
                        labels=list_subject_id,
                        saving_path=os.path.join(saving_path_tsne,f'tsne_backbone_subject_{split_dataset_name}'),
                        legend_label='subject',
                        title=f'dataset {split_dataset_name} (from backbone per chunks)')
    
        # plot tsne considering the prediction per subject and per class  
        # y_pred shape [nr_video, nr_windows]
        if head.value == HEAD.SVR.value:
          y_pred = model_advanced.head.predict(backbone_feats)                                                  
          tools.plot_tsne(X=backbone_feats,
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
          tools.plot_tsne(X = X_gru_out_feats_padded_cpu,
                        labels = torch.round(y_pred_unpadded),
                        # title=f'{k}_{v} (from backbone)')
                        title = f'dataset_{split_dataset_name} pred per chunks (using GRU features)',
                        legend_label='class',
                        saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_{split_dataset_name}_pred'))
          # considering 1 embedding per video
          y_last_hidden = torch.stack([torch.round(y_pred[i,length_feature[i]-1]) for i in range(y_pred.shape[0])])
          print(f'y_pred {y_pred}')
          print(f'y_last_hidden {y_last_hidden}')
          tools.plot_tsne(X = X_gru_hn_feats_cpu,
                          labels = y_last_hidden,
                          legend_label = 'class ',
                          title = f'dataset_{split_dataset_name} pred per video (using GRU last_hidden feature)',
                          saving_path = os.path.join(saving_path_tsne,f'tsne_GRU_{split_dataset_name}_gt'))
          
          # subject_ids_extended = torch.cat([subject_ids_per_sample_id[i].repeat(length_feature[i]) for i in range(subject_ids_per_sample_id.shape[0])])
    
          tools.plot_tsne(X = X_gru_hn_feats_cpu,
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
 

def plot_loss_details(dict_train, train_folder_path, epochs):
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
                                saving_path=train_folder_path)
  
  # Plot and save confusion matrices for each epoch
  confusion_matrix_path = os.path.join(train_folder_path,'confusion_matricies')
  if not os.path.exists(confusion_matrix_path):
    os.makedirs(confusion_matrix_path)
  _plot_confusion_matricies(epochs, dict_train, confusion_matrix_path)


def create_unique_video_per_prediction(train_folder_path, dict_cvs_path, sample_ids, list_frames, y_pred, y, dataset_name):
  # Create video with predictions
  print(f"Creating video with predictions for {dataset_name}")
  video_folder_path = os.path.join(train_folder_path,f'video')
  if not os.path.exists(video_folder_path):
    os.makedirs(video_folder_path)
  
  list_input_video_path = tools.get_list_video_path_from_csv(dict_cvs_path[dataset_name])
  output_video_path = os.path.join(video_folder_path,f'video_all_{dataset_name}.mp4')
  
  # print(f'y_pred shape {y_pred.shape}') # [36]
  # print(f'y shape {y.shape}') # [36]
  # print(f'{list_frames.shape}') # [36, 16]
  # all_predictions = y_pred.reshape(y_pred.shape[0],-1) # [33,2,1] -> [33,2]
  # # list_ground_truth = y # [33,1]
  # if all_predictions.shape[1] != y.shape[1]:
  #   y = y.repeat(1,all_predictions.shape[1])
  # print(f'all_predictions shape {all_predictions.shape}') # [33,2]
  # print(f'list_ground_truth shape {y.shape}') # [33,1]
  tools.save_frames_as_video(list_input_video_path=list_input_video_path, # [n_video=33]
                                              list_frame_indices=list_frames, # [33,2,16]
                                              output_video_path=output_video_path, # string
                                              all_predictions=y_pred,#  [33,2]
                                              list_ground_truth=y,
                                              sample_ids=sample_ids, # -> [33,2] 
                                              output_fps=4)


def _plot_confusion_matricies(epochs, dict_train, confusion_matrix_path):
  for epoch in range(epochs): 
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][epoch],
                                title=f'Train_{epoch} confusion matrix',
                                saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_train_{epoch+1}.png'))
    
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['test_confusion_matricies'][epoch],
                              title=f'Test_{epoch} confusion matrix',
                              saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_test_{epoch+1}.png'))
  saving_path_precision_recall = os.path.join(confusion_matrix_path,'plot_over_epochs')
  if not os.path.exists(saving_path_precision_recall):
    os.makedirs(saving_path_precision_recall)
  tools.plot_accuracy_confusion_matrix(train_confusion_matricies=dict_train['dict_results']['train_confusion_matricies'],
                                       test_confusion_matricies=dict_train['dict_results']['test_confusion_matricies'],
                                       saving_path=saving_path_precision_recall)
    

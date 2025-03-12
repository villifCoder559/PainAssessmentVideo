from matplotlib.ticker import MaxNLocator
import torch
from custom.backbone import video_backbone, vit_image_backbone
from custom.neck import neck
from custom.dataset import customDataset, customDatasetCSV
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error as mea
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import custom.tools as tools
import torch.nn as nn
import torch.optim as optim
from custom.head import HeadSVR, HeadGRU, AttentiveHead
import os
import json
from sklearn.manifold import TSNE
from torchmetrics.classification import ConfusionMatrix
import time
from custom.helper import MODEL_TYPE
# import wandb
# from tsnecuda import TSNE as cudaTSNE # available only on Linux

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,
              path_labels, sample_frame_strategy, head, head_params,
              batch_size_training,stride_window,clip_length,
              features_folder_saving_path):
    """
    Initialize the custom model. 
    Parameters:
    model_type (str): Type of the model to be used. 
    embedding_reduction (int): Dimension reduction for embeddings.
    clips_reduction (int): Dimension reduction for clips.
    path_dataset (str): Path to the dataset.
    path_labels (str): Path to the labels.
    sample_frame_strategy (str): Strategy for sampling frames.
    download_if_unavailable (bool, optional): Flag to download the model if unavailable. Defaults to False.
    batch_size (int, optional): Batch size for data loading. Defaults to 1.
    stride_window (int, optional): Stride window for sampling frames. Defaults to 2.
    clip_length (int, optional): Length of each video clip. Defaults to 16.
    svr_params (dict, optional): Parameters for the Support Vector Regressor (SVR). Defaults to {'kernel': 'rbf', 'C': 1, 'epsilon': 0.1}.

    """
    if model_type != MODEL_TYPE.ViT_image:
      self.backbone = video_backbone(model_type)
    else:
      self.backbone = vit_image_backbone()
    self.neck = neck(embedding_reduction, clips_reduction)
    self.dataset = customDataset(path_dataset=path_dataset, 
                                 path_labels=path_labels, 
                                 sample_frame_strategy=sample_frame_strategy, 
                                 stride_window=stride_window, 
                                 clip_length=clip_length)
    self.batch_size_training = batch_size_training
    # self.dataloader = DataLoader(self.dataset, 
    #                              batch_size=batch_size, 
    #                              shuffle=False,
    #                              collate_fn=self.dataset._custom_collate_fn) # TODO: put inside customDataset and return a dataset and dataLoader
    
    if head == 'SVR':
      self.head = HeadSVR(svr_params=head_params)
    elif head == 'GRU':
      if model_type != MODEL_TYPE.ViT_image:
        assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
      self.head = HeadGRU(**head_params)
    elif head == 'ATTENTIVE':
      self.head = AttentiveHead(**head_params)
    self.path_to_extracted_features = features_folder_saving_path
    
  def test_pretrained_model(self,path_model_weights, csv_path, log_file_path,criterion=nn.L1Loss(), round_output_loss=False,is_test=True,pooled_features=True):
    """
    Evaluate the model using the specified dataset.
    Parameters:
      csv_path (str): Path to the CSV file containing the dataset.
      criterion (torch.nn.Module, optional): Loss function to be used. Default is nn.L1Loss().
      round_output_loss (bool, optional): Flag to round the output loss. Default is False.
    Returns:
      dict: A dictionary containing the results of the evaluation process, including:
      - 'losses': List of losses.
      - 'loss_per_class': Loss per class, reshaped to (1, -1).
      - 'loss_per_subject': Loss per subject, reshaped to (1, -1).
      - 'subject_ids_unique': Unique subject IDs.
      - 'y_unique': Unique classes.
    """
    if not is_test:
      raise Exception('Set is_test to True. Currently this function is only for testing.')
    if isinstance(self.head, HeadGRU):
      dict_feature_extraction = self.extract_features_from_SSD(csv_path)
      X = dict_feature_extraction['features']
      y = dict_feature_extraction['list_labels']
      subject_ids = dict_feature_extraction['list_subject_id']
      sample_ids = dict_feature_extraction['list_sample_id']
      unique_classes = np.unique(y)
      nr_uniq_classes =list(range((max(unique_classes)+2)))[-1] 
      test_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=unique_classes.shape[0] + 1)
      # load weights
      self.head.load_state_weights(path=path_model_weights)
      test_loader = self.head.get_data_loader(X=X, 
                                              y=y, 
                                              subject_ids=subject_ids,
                                              sample_ids=sample_ids,
                                              batch_size=self.batch_size_training)
      
      dict_test = self.head.evaluate(val_loader=test_loader,
                                    val_confusion_matricies=test_confusion_matricies,
                                    is_test=is_test,
                                    nr_uniq_classes=nr_uniq_classes,
                                    criterion=criterion, 
                                    device='cuda',
                                    round_output_loss=round_output_loss,
                                    log_file_path=log_file_path,
                                    unique_train_val_classes=unique_classes,
                                    unique_train_val_subjects=np.unique(subject_ids),
                                    )
      test_unique_subject_ids,test_count_subject_ids = np.unique(subject_ids,return_counts=True)
      test_unique_classes,test_count_classes = np.unique(y,return_counts=True)
      dict_test['test_unique_subject_ids'] = test_unique_subject_ids
      dict_test['test_count_subject_ids'] = test_count_subject_ids
      dict_test['test_unique_y'] = test_unique_classes
      dict_test['test_count_y'] = test_count_classes
    elif isinstance(self.head, AttentiveHead):
      root_folder_features="/media/villi/TOSHIBA EXT/samples_16_whole"
      test_dataset = customDatasetCSV(csv_path,root_folder_features=root_folder_features)
      test_loader = DataLoader(test_dataset,collate_fn=test_dataset._custom_collate,batch_size=self.batch_size_training)
      unique_test_subjects = test_dataset.get_unique_subjects()
      unique_classes = np.array(list(range(self.head.model.num_classes)))
      dict_test = self.head.evaluate(val_loader=test_loader, criterion=criterion, unique_val_subjects=unique_test_subjects,
                                      unique_val_classes=unique_classes, is_test=is_test)
      dict_test['test_unique_subject_ids'] = unique_test_subjects
      dict_test['test_count_subject_ids'] = test_dataset.get_count_subjects()
      dict_test['test_unique_y'] = unique_classes
      dict_test['test_count_y'] = test_dataset.get_count_classes()
    else:
      raise Exception('Head not supported.')
    return dict_test
  
  
  def free_gpu_memory(self):
    self.head.model.to('cpu')
    torch.cuda.empty_cache()
    
  def train(self, train_csv_path, val_csv_path, num_epochs, criterion,
            optimizer_fn, lr,saving_path,init_weights,round_output_loss,
            shuffle_video_chunks,shuffle_training_batch,init_network,
            regularization_loss,regularization_lambda,key_for_early_stopping,early_stopping,
            enable_scheduler,pooled_features=True
            ):
    """
    Train the model using the specified training and testing datasets.
    Parameters:
      train_csv_path (str): Path to the CSV file containing the training data.
      test_csv_path (str): Path to the CSV file containing the testing data.
      num_epochs (int, optional): Number of epochs for training. Default is 10.
      batch_size (int, optional): Batch size for training. Default is 1.
      criterion (torch.nn.Module, optional): Loss function to be used. Default is nn.L1Loss().
      optimizer_fn (torch.optim.Optimizer, optional): Optimizer function to be used. Default is optim.Adam.
      lr (float, optional): Learning rate for the optimizer. Default is 0.0001.
      saving_path (str, optional): Path to save the trained model. Default is None.
      init_weights (bool, optional): Flag to initialize the weights. Default is True.
    Returns:
      dict: A dictionary containing the results of the training process, including:
      - 'dict_results': {
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
    # print(self.path_to_extracted_features)
        # else:
        #   dict_feature_extraction_train[k] = v[dict_sample_indices['train'][1]]
        #   dict_feature_extraction_test[k] = v[dict_sample_indices[key][1]]
        #   print(f'v {v}')
        #   print(f"idx: {dict_sample_indices['train'][1]}")
        #   print(f'dict_feature_extraction_train[k] {dict_feature_extraction_train[k]}')
        
        
    # else:
    count_subject_ids_train, count_y_train = tools.get_unique_subjects_and_classes(train_csv_path)
    count_subject_ids_val, count_y_val = tools.get_unique_subjects_and_classes(val_csv_path) 
    if pooled_features:
      print('Extracting pooled features...')
      dict_feature_extraction_train = self.extract_features_from_SSD(train_csv_path, pooled_features)
      dict_feature_extraction_test = self.extract_features_from_SSD(val_csv_path, pooled_features)
      
      X_train = dict_feature_extraction_train['features']
      y_train = dict_feature_extraction_train['list_labels']
      subject_ids_train = dict_feature_extraction_train['list_subject_id']
      sample_ids_train = dict_feature_extraction_train['list_sample_id'] 
      
      X_test = dict_feature_extraction_test['features']
      y_test = dict_feature_extraction_test['list_labels']
      subject_ids_test = dict_feature_extraction_test['list_subject_id']
      sample_ids_test = dict_feature_extraction_test['list_sample_id']  

      memory_allocated_before = torch.cuda.memory_allocated()/1024/1024/1024
      memory_reserved_before = torch.cuda.memory_reserved()/1024/1024/1024
      print(f"Memory allocated before training: {memory_allocated_before} GB")
      print(f"Memory reserved before training: {memory_reserved_before} GB")
      if isinstance(self.head, HeadGRU):
        print('Training using GRU.....')
        print(train_csv_path)
        dict_results = self.head.start_train(X_train=X_train, y_train=y_train, subject_ids_train=subject_ids_train,
                                                  sample_ids_val=sample_ids_test, 
                                                  sample_ids_train=sample_ids_train,
                                                  X_val=X_test, 
                                                  y_val=y_test, 
                                                  subject_ids_val=subject_ids_test, 
                                                  num_epochs=num_epochs, 
                                                  batch_size=self.batch_size_training,
                                                  criterion=criterion, 
                                                  optimizer_fn=optimizer_fn, 
                                                  lr=lr,
                                                  saving_path=saving_path,
                                                  init_weights=init_weights,
                                                  round_output_loss=round_output_loss,
                                                  shuffle_video_chunks=shuffle_video_chunks,
                                                  shuffle_training_batch=shuffle_training_batch,
                                                  train_csv_path=train_csv_path,
                                                  init_network=init_network,
                                                  regularization_loss=regularization_loss,
                                                  regularization_lambda=regularization_lambda,
                                                  key_for_early_stopping=key_for_early_stopping,
                                                  early_stopping=early_stopping,
                                                  enable_scheduler=enable_scheduler
                                                  )
    else:
      if isinstance(self.head, AttentiveHead):
        print('Training using Attentive head..')
        dict_results = self.head.start_train(num_epochs=num_epochs,criterion=criterion,optimizer=optimizer_fn,lr=lr,
                                            saving_path=saving_path,train_csv_path=train_csv_path,val_csv_path=val_csv_path,
                                            batch_size=self.batch_size_training,regularization_loss=regularization_loss,
                                            regularization_lambda=regularization_lambda,early_stopping=early_stopping,
                                            key_for_early_stopping=key_for_early_stopping,enable_scheduler=enable_scheduler)
    return {'dict_results':dict_results, 
              'count_y_train':count_y_train, 
              'count_y_test':count_y_val,
              'count_subject_ids_train':count_subject_ids_train,
              'count_subject_ids_test':count_subject_ids_val
              }
      
        
    
  def extract_features_from_SSD(self,csv_path):
    dict_feature_extraction = {}
    print(f'csv_path:{csv_path}')
    if os.path.exists(self.path_to_extracted_features) and os.listdir(self.path_to_extracted_features):
      print('Loading features from SSD...')
      key = os.path.split(csv_path)[-1][:-4]
      # print('folder_indxs_path',os.path.split(train_csv_path)[:-1][0])
      split_indices_folder = os.path.join(os.path.split(csv_path)[:-1][0])
      dict_sample_indices = tools.read_split_indices(split_indices_folder)
      dict_all_features = tools.load_dict_data(self.path_to_extracted_features)
      list_idx = []
      count = 0
      for sample_id in dict_sample_indices[key][0]: # dict_sample_indices['train'][0] -> list of sample_ids,
                                                    # dict_sample_indices['train'][1] -> list of indices
        idxs = sample_id == dict_all_features['list_sample_id'] # get all video chunks of the same sample_id
        list_idx.append(idxs) # list of boolean tensors => torch.stack(list_idx) -> [n_video, n_clips]
        count+=1
      all_idxs_train = torch.any(torch.stack(list_idx), dim=0)
      
      for k,v in dict_all_features.items():
        dict_feature_extraction[k] = v[all_idxs_train]
      return dict_feature_extraction     

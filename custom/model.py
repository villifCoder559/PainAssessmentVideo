from matplotlib.ticker import MaxNLocator
import torch
from custom.backbone import backbone
from custom.neck import neck
from custom.dataset import customDataset
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
from custom.head import HeadSVR, HeadGRU, CrossValidationGRU
import os
import json
from sklearn.manifold import TSNE
from torchmetrics.classification import ConfusionMatrix
import time
# import wandb
# from tsnecuda import TSNE as cudaTSNE # available only on Linux

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,
              path_labels, sample_frame_strategy, head, head_params, download_if_unavailable=False,
              batch_size_feat_extraction=1,batch_size_training=1,stride_window=2,clip_length=16,
              features_folder_saving_path=''):
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
    self.backbone = backbone(model_type, download_if_unavailable)
    self.neck = neck(embedding_reduction, clips_reduction)
    self.dataset = customDataset(path_dataset=path_dataset, 
                                 path_labels=path_labels, 
                                 sample_frame_strategy=sample_frame_strategy, 
                                 stride_window=stride_window, 
                                 clip_length=clip_length)
    self.batch_size_training = batch_size_training
    self.batch_size_feat_extraction = batch_size_feat_extraction
    # self.dataloader = DataLoader(self.dataset, 
    #                              batch_size=batch_size, 
    #                              shuffle=False,
    #                              collate_fn=self.dataset._custom_collate_fn) # TODO: put inside customDataset and return a dataset and dataLoader
    
    if head == 'SVR':
      self.head = HeadSVR(svr_params=head_params)
    elif head == 'GRU':
      assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
      self.head = HeadGRU(**head_params)
    self.path_to_extracted_features = features_folder_saving_path

  def test_pretrained__model(self,path_model_weights, csv_path, log_file_path,criterion=nn.L1Loss(), round_output_loss=False,is_test=True):
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
    dict_feature_extraction = self.extract_features(csv_path)
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
                                   unique_classes=unique_classes,
                                   unique_subjects=np.unique(subject_ids),
                                   )
                                      
    return dict_test
  
  def train(self, train_csv_path, test_csv_path, num_epochs=10, criterion=nn.L1Loss(),
            optimizer_fn=optim.Adam, lr=0.0001,saving_path=None,init_weights=True,round_output_loss=False,
            shuffle_video_chunks=True,shuffle_training_batch=True,init_network='default',
            regularization_loss='L1',regularization_lambda=0.01):
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
    print('Extracting features...')
    dict_feature_extraction_train = self.extract_features(train_csv_path)
    dict_feature_extraction_test = self.extract_features(test_csv_path)
    
    count_subject_ids_train, count_y_train = tools.get_unique_subjects_and_classes(train_csv_path)
    X_train = dict_feature_extraction_train['features']
    y_train = dict_feature_extraction_train['list_labels']
    subject_ids_train = dict_feature_extraction_train['list_subject_id']
    sample_ids_train = dict_feature_extraction_train['list_sample_id'] 
    
    count_subject_ids_test, count_y_test = tools.get_unique_subjects_and_classes(test_csv_path) 
    X_test = dict_feature_extraction_test['features']
    y_test = dict_feature_extraction_test['list_labels']
    subject_ids_test = dict_feature_extraction_test['list_subject_id']
    sample_ids_test = dict_feature_extraction_test['list_sample_id']  
        
    if isinstance(self.head, HeadSVR):
      print('Use SVR...')
      dict_results = self.head.fit(X_train=X_train,y_train=y_train, subject_ids_train=subject_ids_train,
                                     X_test=X_test, y_test=y_test, subject_ids_test=subject_ids_test,)

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
                                                optimizer_fn=optimizer_fn, lr=lr,
                                                saving_path=saving_path,
                                                init_weights=init_weights,
                                                round_output_loss=round_output_loss,
                                                shuffle_video_chunks=shuffle_video_chunks,
                                                shuffle_training_batch=shuffle_training_batch,
                                                train_csv_path=train_csv_path,
                                                init_network=init_network,
                                                regularization_loss=regularization_loss,
                                                regularization_lambda=regularization_lambda,
                                                )
    
    return {'dict_results':dict_results, 
            'count_y_train':count_y_train, 
            'count_y_test':count_y_test,
            'count_subject_ids_train':count_subject_ids_train,
            'count_subject_ids_test':count_subject_ids_test}
    
  def extract_features(self,csv_path,read_from_memory=True):
    dict_feature_extraction = {}
    print(f'csv_path:{csv_path}')
    if read_from_memory and os.path.exists(self.path_to_extracted_features) and os.listdir(self.path_to_extracted_features):
      print('Loading features from SSD...')
      key = os.path.split(csv_path)[-1][:-4]
      # print('folder_indxs_path',os.path.split(train_csv_path)[:-1][0])
      split_indices_folder = os.path.join(os.path.split(csv_path)[:-1][0])
      dict_sample_indices = tools.read_split_indices(split_indices_folder)
      dict_all_features = tools.load_dict_data(self.path_to_extracted_features)
      list_idx = []
      for sample_id in dict_sample_indices[key][0]: # dict_sample_indices['train'][0] -> list of sample_ids, dict_sample_indices['train'][1] -> list of indices
        idxs = sample_id == dict_all_features['list_sample_id'] # get all video chunks of the same sample_id
        list_idx.append(idxs) # list of boolean tensors => torch.stack(list_idx) -> [n_video, n_clips]
      
      all_idxs_train = torch.any(torch.stack(list_idx), dim=0)
      
      for k,v in dict_all_features.items():
        dict_feature_extraction[k] = v[all_idxs_train]
      return dict_feature_extraction
    else:
      print('Computing features...')
      return self._extract_features(path_csv_dataset=csv_path)
      
  def _extract_features(self,path_csv_dataset):
    """
    Extract features from the dataset specified by the CSV file path.

    Args:
      path_csv_dataset (str): Path to the CSV file containing dataset information.
      batch_size (int, optional): Number of samples per batch to load. Default is 2.

    Returns:
      dict: A dictionary containing the following keys:
        - 'features' (torch.Tensor): shape [n_video * n_clips, temporal_dim=8, patch_h, patch_w, emb_dim].
        - 'list_labels' (torch.Tensor): shape [n_video * n_clips].
        - 'list_subject_id' (torch.Tensor): shape (n_video * n_clips).
        - 'list_sample_id' (torch.Tensor): shape (n_video * n_clips).
        - 'list_path' (np.ndarray): shape (n_video * n_clips,).
        - 'list_frames' (torch.Tensor): shape [n_video * n_clips, n_frames].

    """
    
    device = 'cuda' 
    print(f"extracting features using.... {device}")
    list_features = []
    list_labels = []
    list_subject_id = []
    list_sample_id = []
    list_path = []
    list_frames = []
    count = 0
    self.dataset.set_path_labels(path_csv_dataset)
    dataloader = DataLoader(self.dataset, 
                            batch_size=self.batch_size_feat_extraction,
                            num_workers=1,
                            shuffle=False,
                            collate_fn=self.dataset._custom_collate_fn)
    # move the model to the device
    self.backbone.model.to(device)
    self.backbone.model.eval()
    with torch.no_grad():
      # start_total_time = time.time()
      # start = time.time()
      for data, labels, subject_id,sample_id, path, list_sampled_frames in dataloader:
        #############################################################################################################
        # data shape -> [nr_clips, clip_length=16, channels=3, H=224, W=224]
        # 
        # nr_clips  = floor((total_frames-clip_length=16)/stride_window) + 1
        #           BIOVID -> floor((138-16)/4)) + 1 = 31
        # 
        # self.backbone.model ->   85 MB (small_model), 
        #                         400 MB (base_model), 
        #                           4 GB (giant_model)
        # 
        # video_feat_size [nr_video,8,768] => 8700 * 8 * 768 * 4 = 204 MB
        #############################################################################################################
        # print(f'Elapsed time for {batch_size} samples: {time.time() - start}')
        data = data.to(device)
        feature = self._compute_features(data)
        # feature -> [2, 8, 1, 1, 384]
        list_frames.append(list_sampled_frames)
        list_features.append(feature)
        list_labels.append(labels)
        list_sample_id.append(sample_id)
        list_subject_id.append(subject_id)
        list_path.append(path)
        count += 1
        # if count % 10 == 0:
        print(f'Batch {count}/{len(dataloader)}')
        print(f'GPU:\n Free : {torch.cuda.mem_get_info()[0]/1024/1024/1024:.2f} GB \n total: {torch.cuda.mem_get_info()[1]/1024/1024/1024:.2f} GB')
        # start = time.time()
    # print(f'Elapsed time for total feature extraction: {time.time() - start_total_time}')
    # print('Feature extraceton done')
    self.backbone.model.to('cpu')
    # print('backbone moved to cpu')
    # print(f'torch.cat features {torch.cat(list_features,dim=0).shape}')
    dict_data = {
      'features': torch.cat(list_features,dim=0),  # [n_video * n_clips, temporal_dim=8, patch_h, patch_w, emb_dim] 630GB
      'list_labels': torch.cat(list_labels,dim=0),  # [n_video * n_clips] 8700 * 10 * 4 = 340 KB
      'list_subject_id': torch.cat(list_subject_id).squeeze(),  # (n_video * n_clips) 8700 * 10 * 4 = 340 KB
      'list_sample_id': torch.cat(list_sample_id),  # (n_video * n_clips) 8700 * 10 * 4 = 340 KB
      'list_path': np.concatenate(list_path),  # (n_video * n_clips,) 8700 * 10 * 4 = 340 KB
      'list_frames': torch.cat(list_frames,dim=0)  # [n_video * n_clips, n_frames] 8700 * 10 * 4 = 340 KB
    }

    return dict_data 

  
  def _compute_features(self, data, remove_clip_reduction=False):
    """
    Compute features from the given data using the model's backbone and neck components.
    Assumption: The model and data already in the same device
    Args:
      data (torch.Tensor): Input data tensor with shape [nr_clips, channels=3, clip_length=16, H=224, W=224]
      labels (torch.Tensor): Labels corresponding to the input data.
      subject_id (torch.Tensor): Subject IDs corresponding to the input data.
      sample_id (torch.Tensor): Sample IDs corresponding to the input data.
      path (np.ndarray): Paths corresponding to the input data.
      remove_clip_reduction (bool, optional): Flag to remove clip reduction. Defaults to False.

    Returns:
      tuple: A tuple containing:
        - feature (torch.Tensor) : shape [batch_size, tubelet_size, patch_h, patch_w, self.embed_dim]
        - labels (torch.Tensor): shape [nr_video]
        - subject_id (torch.Tensor): shape (nr_video,)
        - sample_id (torch.Tensor): shape (nr_video,)
        - unique_path (np.ndarray): shape (nr_video,)
    """
    with torch.no_grad():
    # Extract features from clips -> return [B, clips/tubelets, W/patch_w, H/patch_h, emb_dim] 
      feature = self.backbone.forward_features(x=data) # output shape [batch,temporal_dim,patch_h,patch_w,emb_dim]
    # Apply dimensionality reduction [B,C,T,H,W] -> [B, reduction(C,T,H,W)]
    if self.neck.embedding_reduction is not None:
      feature = self.neck.embedding_reduction(feature)
    # Apply clip reduction [B, reduction(C,T,H,W)] -> [1, reduction(C,T,H,W)] => extract one feature per video using the mean
    if not remove_clip_reduction and self.neck.clips_reduction is not None:
      feature = self.neck.clips_reduction(feature)
    print(f'feature shape: {feature.shape}')
    # unique_path = np.unique(path, return_counts=False)
    
    return feature
    
  # def run_grid_search(self, param_grid,k_cross_validation=5): #
  #   if isinstance(self.head, HeadSVR):
  #     print('GridSearch using SVR...')
  #     self.dataset.set_path_labels('val')
  #     X, y, subjects_id,_ , _, _ = self.extract_features()
  #     X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
  #     y = y.squeeze().detach().cpu().numpy()
  #     # print(subjects_id.shape)
  #     grid_search, list_split_indices =self.head.run_grid_search(param_grid=param_grid, X=X, y=y, groups=subjects_id, k_cross_validation=k_cross_validation)
      
  #     return grid_search, list_split_indices, subjects_id, y
  #   else:
  #     return None

  def save_configuration(self, saving_path):
    """
    Save the configuration of the model to a file.
    
    Parameters:
    path (str): Path to the file where the configuration will be saved.
    """
    config = {
      'config_dataset': self.dataset.get_params_configuration(),
      'config_head': self.head.get_params_configuration(),
      'model_type': self.backbone.model_type.name,
      'embedding_reduction': self.neck.type_embedding_redcution.name,
      'clips_reduction': self.neck.type_embedding_redcution.name,
      'path_dataset': self.dataset.path_dataset,
      'path_labels': self.dataset.path_labels,
      'sample_frame_strategy': self.dataset.type_sample_frame_strategy.name,
      'stride_window': self.dataset.stride_window,
      'clip_length': self.dataset.clip_length,
      'head': self.head.params,
      'features_folder_saving_path': self.path_to_extracted_features
      # 'head_params': self.head.get_params() if hasattr(self.head, 'get_params') else {}
    }
    with open(saving_path, 'w') as config_file:
      json.dump(config, config_file, indent=4)
    
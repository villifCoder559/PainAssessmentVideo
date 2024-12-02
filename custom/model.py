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
from tsnecuda import TSNE as cudaTSNE # available only on Linux

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,
              path_labels, preprocess, sample_frame_strategy, head, head_params, download_if_unavailable=False,
              batch_size=1,stride_window=2,clip_length=16
              ):
    """
    Initialize the custom model. 
    Parameters:
    model_type (str): Type of the model to be used. 
    embedding_reduction (int): Dimension reduction for embeddings.
    clips_reduction (int): Dimension reduction for clips.
    path_dataset (str): Path to the dataset.
    path_labels (str): Path to the labels.
    preprocess (callable): Preprocessing function for the data.
    sample_frame_strategy (str): Strategy for sampling frames.
    download_if_unavailable (bool, optional): Flag to download the model if unavailable. Defaults to False.
    batch_size (int, optional): Batch size for data loading. Defaults to 1.
    stride_window (int, optional): Stride window for sampling frames. Defaults to 2.
    clip_length (int, optional): Length of each video clip. Defaults to 16.
    svr_params (dict, optional): Parameters for the Support Vector Regressor (SVR). Defaults to {'kernel': 'rbf', 'C': 1, 'epsilon': 0.1}.

    Raises:
    AssertionError: If batch_size is not 1.
    """
    self.backbone = backbone(model_type, download_if_unavailable)
    self.neck = neck(embedding_reduction, clips_reduction)
    self.dataset = customDataset(path_dataset, path_labels, preprocess, sample_frame_strategy, stride_window=stride_window, clip_length=clip_length,
                                 batch_size=batch_size)
    # self.dataloader = DataLoader(self.dataset, 
    #                              batch_size=batch_size, 
    #                              shuffle=False,
    #                              collate_fn=self.dataset._custom_collate_fn) # TODO: put inside customDataset and return a dataset and dataLoader
    
    if head == 'SVR':
      self.head = HeadSVR(svr_params=head_params)
    elif head == 'GRU':
      assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
      # Calculate the backbone output tensor size to use as input to the GRU head
      output_tensor = [1, int(self.backbone.frame_size/self.backbone.tubelet_size), self.backbone.out_spatial_size, self.backbone.out_spatial_size, self.backbone.embed_dim]
      if embedding_reduction:
        for dim in self.neck.dim_embed_reduction:
          output_tensor[dim] = 1
      if clips_reduction.value:
        print(self.neck.dim_clips_reduction)
        output_tensor[self.neck.dim_clips_reduction + 1] = 1
      head_params['input_size'] = np.prod(output_tensor).astype(int)
      print(f'head_params : {head_params}')
      print(f'output_tensor : {output_tensor}')
      self.head = HeadGRU(dropout=head_params['dropout'], input_size=head_params['input_size'], 
                          hidden_size=head_params['hidden_size'], num_layers=head_params['num_layers'])

  # def compute_output_tensor_for_gru(self):
  #   assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
  #     # Calculate the backbone output tensor size to use as input to the GRU head
  #     output_tensor = [1, int(self.backbone.frame_size/self.backbone.tubelet_size), self.backbone.out_spatial_size, self.backbone.out_spatial_size, self.backbone.embed_dim]
  #     if embedding_reduction:
  #       for dim in self.neck.dim_embed_reduction:
  #         output_tensor[dim] = 1
  #     if clips_reduction.value:
  #       print(self.neck.dim_clips_reduction)
  #       output_tensor[self.neck.dim_clips_reduction + 1] = 1
  #     head_params['input_size'] = np.prod(output_tensor).astype(int)
  #     print(f'head_params : {head_params}')
  #     print(f'output_tensor : {output_tensor}')
      
  def run_k_fold_cross_validation(self, k_fold, train_folder_path, batch_size=8, criterion=nn.L1Loss(), 
                                  optimizer_fn=optim.Adam, lr=0.0001,
                                  epochs=10,train_size=0.8,val_size=0.1,test_size=0.1):
    # def k_cross_valuation():
    fold_results = []
    list_saving_paths = []

    for i in range(k_fold):
      path = os.path.join(train_folder_path,f'results_k{i}_cross_val')
      list_saving_paths.append(path)
      if not os.path.exists(path):
        os.makedirs(path)
      path_csv_k_fold = tools._generate_train_test_validation(csv_path=self.dataset.path_labels,
                                                              saving_path=path,
                                                              train_size=train_size,
                                                              val_size=val_size,
                                                              test_size=test_size,
                                                              random_state=i)
      
      result = self.train(train_csv_path=path_csv_k_fold['train'],
                          test_csv_path=path_csv_k_fold['test'],
                          num_epochs=epochs, batch_size=batch_size, 
                          criterion=criterion,
                          optimizer_fn=optimizer_fn,
                          lr=lr, 
                          saving_path=path)
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
    
    # def k_cross_valuation_SVR():
    #   print('SVR with k-fold cross-validation...')
    #   self.dataset.set_path_labels('all')
    #   dict_feature_extraction_train = self._extract_features() # feats,labels,subject_id,sample_id,path
    #   X = dict_feature_extraction_train['features']
    #   y = dict_feature_extraction_train['list_labels']
    #   subject_ids = dict_feature_extraction_train['list_subject_id']
    #   X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
    #   y = y.squeeze().detach().cpu().numpy()
    #   list_split_indices,results = self.head.k_fold_cross_validation(X=X, y=y, k=k_fold, groups=subject_ids)

    # if isinstance(self.head, HeadGRU):
    #   results = k_cross_valuation()

    # elif isinstance(self.head, HeadSVR):
    #   results = k_cross_valuation_SVR()      

    # return results

  
  def train(self, train_csv_path, test_csv_path, num_epochs=10, batch_size=1, criterion=nn.L1Loss(),optimizer_fn=optim.Adam, lr=0.0001,saving_path=None,init_weights=True,round_output=True):
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
    if isinstance(self.head, HeadSVR):
      print('Training using SVR...')
      # Extract feature from training set
      # self.dataset.set_path_labels(train_csv_path) 
      count_subject_ids_train, count_y_train = tools.get_unique_subjects_and_classes(train_csv_path) 
      dict_feature_extraction_train = self._extract_features(train_csv_path)
      print('feature shape: ',dict_feature_extraction_train['features'].shape)
      X_train = dict_feature_extraction_train['features']
      y_train = dict_feature_extraction_train['list_labels']
      
      X_train = X_train.reshape(X_train.shape[0],-1).detach().cpu().numpy()
      y_train = y_train.squeeze().detach().cpu().numpy()
      subject_ids_train = dict_feature_extraction_train['list_subject_id']
      print('subject_ids_train',subject_ids_train)
      # Extract feature from test set
      # self.dataset.set_path_labels(test_csv_path) 
      count_subject_ids_test, count_y_test = tools.get_unique_subjects_and_classes(test_csv_path) 
      dict_feature_extraction_train = self._extract_features(test_csv_path)
      X_test = dict_feature_extraction_train['features']
      y_test = dict_feature_extraction_train['list_labels']
      
      X_test = X_test.reshape(X_test.shape[0],-1).detach().cpu().numpy()
      y_test = y_test.squeeze().detach().cpu().numpy()
      subject_ids_test = dict_feature_extraction_train['list_subject_id']
      print('subject_ids_test',subject_ids_test)

      dict_results = self.head.fit(X_train=X_train,y_train=y_train, subject_ids_train=subject_ids_train,
                                     X_test=X_test, y_test=y_test, subject_ids_test=subject_ids_test,)

    if isinstance(self.head, HeadGRU):
      print('Training using GRU.....')
      # self.dataset.set_path_labels(train_csv_path)
      count_subject_ids_train, count_y_train = tools.get_unique_subjects_and_classes(train_csv_path) 
      print(f'GRU extract features from {train_csv_path}')
      dict_feature_extraction_train = self._extract_features(train_csv_path) 
      X_train = dict_feature_extraction_train['features'] 
      y_train = dict_feature_extraction_train['list_labels']
      subject_ids_train = dict_feature_extraction_train['list_subject_id']
      # sample_ids_train = dict_feature_extraction_train['list_sample_id'] 
      
      # self.dataset.set_path_labels(test_csv_path)
      count_subject_ids_test, count_y_test = tools.get_unique_subjects_and_classes(test_csv_path) 
      dict_feature_extraction_test = self._extract_features(test_csv_path)
      X_test = dict_feature_extraction_test['features'] 
      y_test = dict_feature_extraction_test['list_labels'] 
      subjects_id_test = dict_feature_extraction_test['list_subject_id']
      device = 'cuda' if torch.cuda.is_available() else 'cpu'
      # self.head.model.to(device)
      dict_results = self.head.start_train_test(X_train=X_train, y_train=y_train, subject_ids_train=subject_ids_train,
                                                X_test=X_test, y_test=y_test, subject_ids_test=subjects_id_test, 
                                                num_epochs=num_epochs,batch_size=batch_size,criterion=criterion,
                                                optimizer_fn=optimizer_fn,lr=lr,
                                                saving_path=saving_path,
                                                init_weights=init_weights,
                                                round_output=round_output,
                                                )
    
    return {'dict_results':dict_results, 
            'count_y_train':count_y_train, 
            'count_y_test':count_y_test,
            'count_subject_ids_train':count_subject_ids_train,
            'count_subject_ids_test':count_subject_ids_test}
  
  def _extract_features(self,path_csv_dataset, batch_size=1): # TODO: Fix using batch_size > 1 (torch.stack wants all tensors to have the same size)
    """
    Extracts features from the dataset using the model's backbone.
    Data and model will be moved to the device ('cuda' if available, 'cpu' otherwise).
    Args:
      stop_after (int, optional): Number of iterations after which to stop the feature extraction. Defaults to 3.
    Returns:
      dict: A dictionary containing the following elements:
          - 'features' (torch.Tensor): Stacked features extracted from the dataset.
          - 'list_labels' (torch.Tensor): Stacked labels corresponding to the features.
          - 'list_subject_id' (numpy.ndarray): Array of subject IDs.
          - 'list_sample_id' (numpy.ndarray): Array of sample IDs.
          - 'list_path' (numpy.ndarray): Array of paths corresponding to the samples.
          - 'list_frames' (torch.Tensor): Stacked frames sampled from the dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
                            batch_size=batch_size,

                            shuffle=False,
                            collate_fn=self.dataset._custom_collate_fn)
    # move the model to the device
    self.backbone.model.to(device)
    self.backbone.model.eval()

    with torch.no_grad():
      for data, labels, subject_id,sample_id, path, list_sampled_frames in dataloader:
        # data shape -> [nr_clips, clip_length, channels=3, H=224, W=224]
        data = data.to(device)
        #compute features
        feature, unique_labels, unique_subject_id, unique_sample_id, unique_path = self._compute_features(data, labels, subject_id, sample_id, path, device)
        # print(f'unique_id: {unique_sample_id}')
        list_frames.append(list_sampled_frames)
        list_features.append(feature)
        list_labels.append(unique_labels)
        list_sample_id.append(unique_sample_id)
        list_subject_id.append(unique_subject_id)
        list_path.append(unique_path)
        count += 1
        # if count % stop_after == 0:
        #   break
    print('Feature extraceton done')
    self.backbone.model.to('cpu')
    print('backbone moved to cpu')
    dict_data={
            'features':torch.stack([feature for feature in list_features]), # [n_video, n_clips, tubelet_size, patch_h, patch_w, emb_dim]
           'list_labels':torch.stack([label for label in list_labels]), # [n_video, 1]
           'list_subject_id':np.stack([subject_id for subject_id in list_subject_id]).squeeze(), # (n_video,)
           'list_sample_id':np.stack([sample_id for sample_id in list_sample_id]), # (n_video, 1)
           'list_path':np.stack([path for path in list_path]), # (n_video, 1)
           'list_frames':torch.stack(list_frames) # [n_video, n_clips, n_frames]
           } 
    
    return dict_data 
    # dict_data features torch.Size([33, 2, 8, 1, 1, 768])
    # dict_data list_labels torch.Size([33, 1])
    # dict_data list_subject_id (33,)
    # dict_data list_sample_id (33, 1)
    # dict_data list_path (33, 1)
    # dict_data list_frames torch.Size([33, 2, 16])

    # using batch>1  doesn't work because of different size, try to transform the list in tensor
  
  def _compute_features(self, data, labels, subject_id, sample_id, path, remove_clip_reduction=False):
    """
    Compute features from the given data using the model's backbone and neck components.
    Assumption: The model and data already in the same device
    Args:
      data (torch.Tensor): Input data tensor with shape [nr_clips, channels=3, clip_length=16, H=224, W=224]
      labels (torch.Tensor): Labels corresponding to the input data.
      subject_id (torch.Tensor): Subject IDs corresponding to the input data.
      sample_id (torch.Tensor or np.ndarray): Sample IDs corresponding to the input data.
      path (torch.Tensor or np.ndarray): Paths corresponding to the input data.
      device (torch.device): Device to perform computations on.
      remove_clip_reduction (bool, optional): Flag to remove clip reduction. Defaults to False.

    Returns:
      tuple: A tuple containing:
        - feature (torch.Tensor) : Extracted features with shape [batch_size, tubelet_size, patch_h, patch_w, self.embed_dim]
        - unique_labels (torch.Tensor): Unique labels from the input data with shape [nr_video]
        - unique_subject_id (np.ndarray): Unique subject IDs from the input data with shape (nr_video,)
        - unique_sample_id (np.ndarray): Unique sample IDs from the input data with shape (nr_video,)
        - unique_path (np.ndarray): Unique paths from the input data with shape (nr_video,)
    """
    with torch.no_grad():
    # Extract features from clips -> return [B, clips/tubelets, W/patch_w, H/patch_h, emb_dim] 
      feature = self.backbone.forward_features(data) # output shape [batch_size, tubelet_size, patch_h, patch_w, self.embed_dim]
    unique_labels, unique_subject_id, unique_sample_id, unique_path = [], [], [], []
    # Apply dimensionality reduction [B,C,T,H,W] -> [B, reduction(C,T,H,W)]
    if self.neck.embedding_reduction is not None:
      feature = self.neck.embedding_reduction(feature)
    # Apply clip reduction [B, reduction(C,T,H,W)] -> [1, reduction(C,T,H,W)]
    if not remove_clip_reduction and self.neck.clips_reduction is not None:
      feature = self.neck.clips_reduction(feature)
    unique_labels = torch.unique(labels, return_counts=False)
    unique_sample_id = np.unique(sample_id, return_counts=False)
    unique_subject_id = np.unique(subject_id, return_counts=False)
    unique_path = np.unique(path, return_counts=False)

    return feature, \
           unique_labels, \
           unique_subject_id,\
           unique_sample_id, \
           unique_path
    
  def run_grid_search(self, param_grid,k_cross_validation=5): #
    if isinstance(self.head, HeadSVR):
      print('GridSearch using SVR...')
      self.dataset.set_path_labels('val')
      X, y, subjects_id,_ , _, _ = self._extract_features()
      X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
      y = y.squeeze().detach().cpu().numpy()
      # print(subjects_id.shape)
      grid_search, list_split_indices =self.head.run_grid_search(param_grid=param_grid, X=X, y=y, groups=subjects_id, k_cross_validation=k_cross_validation)
      
      return grid_search, list_split_indices, subjects_id, y
    else:
      return None

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
      # 'head_params': self.head.get_params() if hasattr(self.head, 'get_params') else {}
    }
    with open(saving_path, 'w') as config_file:
      json.dump(config, config_file, indent=4)
    
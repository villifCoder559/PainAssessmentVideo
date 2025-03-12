import pickle
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, GroupKFold,GroupShuffleSplit
from sklearn.metrics import mean_absolute_error
# from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import custom.tools as tools
from torchmetrics.classification import ConfusionMatrix
import math
import os
from datetime import datetime
import time
import torch.nn.init as init
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from custom.dataset import customSampler
from custom.dataset import customDatasetCSV
import copy
# import wandb

class HeadSVR:
  def __init__(self, svr_params):
    self.params = svr_params
    self.svr = SVR(**svr_params)

  def fit(self, X_train, y_train, subject_ids_train, X_test, y_test, subject_ids_test, saving_path=None):
    """
    Evaluation training of SVR model.
    Parameters:
    X_train (array-like): Shape (n_samples, n_features).
    y_train (array-like): Shape (n_samples,).
    subject_ids_train (array-like): Subject IDs for the training data.
    X_test (array-like): Test data features.
    y_test (array-like): Test data labels.
    subject_ids_test (array-like): Subject IDs for the test data.
    Returns:
    dict: A dictionary containing:
      - 'train_losses': List of training losses.
      - 'train_loss_per_class': Training loss per class, reshaped to (1, -1).
      - 'train_loss_per_subject': Training loss per subject, reshaped to (1, -1).
      - 'test_losses': List of test losses.
      - 'test_loss_per_class': Test loss per class, reshaped to (1, -1).
      - 'test_loss_per_subject': Test loss per subject, reshaped to (1, -1).
      - 'y_unique': Unique classes in the combined training and test labels.
      - 'subject_ids_unique': Unique subject IDs in the combined training and test subject IDs.
      - 'best_model_idx': best_model_epoch

    """

    regressor = self.svr.fit(X_train, y_train)
    # Save the regressor
    if saving_path:
      with open(saving_path, 'wb') as f:
        pickle.dump(regressor, f)
      print(f"Regressor saved to {saving_path}")

    pred_train = regressor.predict(X_train)
    pred_test = regressor.predict(X_test)

    train_loss = mean_absolute_error(y_train, pred_train)
    test_loss = mean_absolute_error(y_test, pred_test)

    # Compute loss per class
    # print(f'y_train-shape {y_train.shape}')
    # print(f'y_train-shape {y_test.shape}')
    y_unique = np.unique(np.concatenate((np.unique(y_train), np.unique(y_test))))
    train_loss_per_class = np.zeros(y_unique.shape[0])
    test_loss_per_class = np.zeros(y_unique.shape[0])
    for cls in y_unique:
      idx_train = np.where(y_train == cls)
      idx_test = np.where(y_test == cls)
      idx_y_gloabl = np.where(y_unique == cls)
      if idx_train[0].shape[0]:
        class_loss = mean_absolute_error(y_train[idx_train], pred_train[idx_train])
        train_loss_per_class[idx_y_gloabl] = class_loss
      if idx_test[0].shape[0]:
        test_los = mean_absolute_error(y_test[idx_test], pred_test[idx_test])
        test_loss_per_class[idx_y_gloabl] = test_los

    # Compute loss per subject
    subject_ids_unique = np.unique(np.concatenate((np.unique(subject_ids_train), np.unique(subject_ids_test))))
    # print(f'subject_ids_unique shape {subject_ids_unique.shape}')
    # print(f'subject_ids_train_unique shape {np.unique(subject_ids_train).shape}')
    # print(f'subject_ids_test_unique shape {np.unique(subject_ids_test).shape}')
    train_loss_per_subject = np.zeros(subject_ids_unique.shape[0])
    test_loss_per_subject = np.zeros(subject_ids_unique.shape[0])

    for id in subject_ids_unique:
      idx_train = np.where(subject_ids_train == id)
      idx_test = np.where(subject_ids_test == id)
      idx_subject_ids_unique_global = np.where(subject_ids_unique == id)
      if idx_train[0].shape[0]:
        subject_loss = mean_absolute_error(y_train[idx_train], pred_train[idx_train])
        train_loss_per_subject[idx_subject_ids_unique_global] = subject_loss
      if idx_test[0].shape[0]:
        test_loss = mean_absolute_error(y_test[idx_test], pred_test[idx_test])
        test_loss_per_subject[idx_subject_ids_unique_global] = test_loss

    train_confusion_matricies = []
    test_confusion_matricies = []
    train_confusion_matricies.append(self.compute_confusion_matrix(X_train, y_train, regressor))
    test_confusion_matricies.append(self.compute_confusion_matrix(X_test, y_test, regressor))

    return {'train_losses': [train_loss], 'train_loss_per_class': train_loss_per_class.reshape(1, -1),
            'train_loss_per_subject': train_loss_per_subject.reshape(1, -1),
            'test_losses': [test_loss], 'test_loss_per_class': test_loss_per_class.reshape(1, -1),
            'test_loss_per_subject': test_loss_per_subject.reshape(1, -1),
            'y_unique': y_unique, 'subject_ids_unique': subject_ids_unique,
            'test_confusion_matricies': test_confusion_matricies,
            'train_confusion_matricies': train_confusion_matricies,
            'best_model_idx': 0}



  # def predict(self, X):
  #   """
  #   Predicts the target values for the given input data using the trained SVR model.

  #   Parameters:
  #   X (array-like): The input data with shape (n_samples, n_features).

  #   Returns:
  #   array: The predicted target values.
  #   """
  #   assert len(X.shape) == 2, f"Input shape should be (n_samples, n_features), got {X.shape}"
  #   predictions = self.svr.predict(X)
  #   return predictions

  def k_fold_cross_validation(self, X, y, groups, k=3, list_saving_paths_k_val=None):
    """ k-fold cross-validation training of SVR model. """
    # Use dictionary so you cann add w/o changing code
    # print('X.shape', X.shape)
    # print('y.shapey', y.shape)
    gss = GroupShuffleSplit(n_splits = k)
    results = cross_validate(self.svr, X, y, cv=gss,scoring='neg_mean_absolute_error', groups=groups, return_train_score=True, return_estimator=True)
    # scores = - scores
    # Print the scores for each fold and the mean score
    # print("Keys:", results.keys())
    print("Train accuracy:", results['train_score'])
    print("Test accuracy:", results['test_score'])
    # print("Mean test accuracy:", results['test_accuracy'].mean())
    list_split_indices=[]
      # Save each model fitted during cross-validation
    for fold_idx, estimator in enumerate(results['estimator']):
      model_path = os.path.join(list_saving_paths_k_val[fold_idx],f'SVR_{fold_idx}.pkl')
      with open(model_path, 'wb') as f:
        pickle.dump(estimator, f)
      print(f"Model for fold {fold_idx + 1} saved to {model_path}")

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=groups), 1):
      list_split_indices.append((train_idx,test_idx))

    # Save the model for each fold
    model_path = os.path.join(list_saving_paths_k_val[fold_idx], f'SVR_{fold_idx}.pkl')
    with open(model_path, 'wb') as f:
      pickle.dump(estimator, f)
    print(f"Model for fold {fold_idx + 1} saved to {model_path}")

    # Initialize confusion matrices for each fold
    confusion_matrix_test = []
    confusion_matrix_train = []

    for fold_idx, estimator in enumerate(results['estimator']):
      X_train, y_train = X[results['train_score'][fold_idx]], y[results['train_score'][fold_idx]]
      X_test, y_test = X[results['test_score'][fold_idx]], y[results['test_score'][fold_idx]]
      cm_train = self.compute_confusion_matrix(X_train, y_train, estimator)
      cm_test = self.compute_confusion_matrix(X_test, y_test, estimator)
      confusion_matrix_train.append(cm_train)
      confusion_matrix_test.append(cm_test)

    dict_result = {'df_results': results,
                   'test_confusion_matricies': confusion_matrix_test,
                   'train_confusion_matricies': confusion_matrix_train}

    return list_split_indices, dict_result

  def compute_confusion_matrix(self,X, y, estimator):
    y_pred_train = estimator.predict(X)
    # y_pred_test = estimator.predict(X_test)
    cm = ConfusionMatrix(task="multiclass",num_classes=len(np.unique(y)))
    cm.update(torch.tensor(y_pred_train), torch.tensor(y))
    cm.compute()
    return cm


  def run_grid_search(self,param_grid, X, y, groups ,k_cross_validation):

    def _plot_cv_indices(cv, X, y, group, ax, n_splits, lw=20):
      """Create a sample plot for indices of a cross-validation object."""
      use_groups = "Group" in type(cv).__name__
      groups = group if use_groups else None
      cmap_data = plt.cm.Paired
      cmap_cv = plt.cm.coolwarm
      # Generate the training/testing visualizations for each CV split
      for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
          # Fill in indices with the training/test groups
          indices = np.array([np.nan] * len(X))
          indices[tt] = 1
          indices[tr] = 0

          # Visualize the results
          ax.scatter(
              range(len(indices)),
              [ii + 0.5] * len(indices),
              c=indices,
              marker="_",
              lw=lw,
              cmap=cmap_cv,
              vmin=-0.2,
              vmax=1.2,
          )

      # Plot the data classes and groups at the end
      ax.scatter(
          range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
      )

      ax.scatter(
          range(len(X)), [ii + 2.5] * len(X), c=group, marker="_", lw=lw, cmap=cmap_data
      )

      # Formatting
      yticklabels = list(range(n_splits)) + ["class", "group"]
      # bar_spacing = 2.5
      ax.set(
          yticks=np.arange(n_splits + 2) + 0.5,
          yticklabels=yticklabels,
          xlabel="Sample index",
          ylabel="CV iteration",
          ylim=[(n_splits + 2.2), -0.2],
          xlim=[0, 30],
      )
      ax.set_title("{}".format(type(cv).__name__), fontsize=15)
      return ax
    # Initialize GridSearchCV
    gss = GroupShuffleSplit(n_splits=k_cross_validation)
    fig, ax = plt.subplots()
    _plot_cv_indices(gss, X, y, groups, ax, k_cross_validation)
    plt.tight_layout()
    plt.show()
    grid_search = GridSearchCV(estimator=self.svr, param_grid=param_grid, cv=gss,scoring='neg_mean_absolute_error',return_train_score=True)

    # Fit the grid search to your data
    grid_search.fit(X, y, groups=groups)

    # Best parameters and score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    list_split_indices=[]

    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=groups), 1):
      list_split_indices.append((train_idx,test_idx))
    return grid_search, list_split_indices

class GRUModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, output_size,layer_norm):
    super(GRUModel, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.output_size = output_size
    self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    self.norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
    self.fc = nn.Linear(hidden_size, output_size)
    
  def get_configuration(self):
    dict_config = {
      'input_size': self.input_size,
      'hidden_size': self.hidden_size,
      'num_layers': self.num_layers,
      'dropout': self.dropout,
      'output_size': self.output_size
    }
    return dict_config

  def forward(self, x, pred_only_last_time_step=False):
    # assert len(x.shape) == 3, f"Input shape should be (batch_size, sequence_length, input_size), got {x.shape}"
    packed_out, _ = self.gru(x)
    if torch.is_tensor(x):
      out_padded = packed_out
      list_length = torch.tensor([x.shape[1]]*x.shape[0])
    else:
      out_padded,list_length = pad_packed_sequence(packed_out, batch_first=True)

    last_hidden_layer = out_padded[torch.arange(out_padded.shape[0]), list_length-1] # [1, hidden_size]
    if pred_only_last_time_step:
      last_hidden_layer = self.norm(last_hidden_layer)
      out = self.fc(last_hidden_layer) # [batch, output_size=1]
      out = out.squeeze(dim=1) 
    else:
      out_padded = self.norm(out_padded)
      out = self.fc(out_padded).squeeze(dim=2) # [batch, seq_len, output_size=1]
    return out

  def _initialize_weights(self,init_type='default'):
    # Initialize GRU weights
    print(f'  GRU Network initialized: {init_type}')
    if init_type == 'xavier':
      for name, param in self.gru.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            torch.nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0)
      # Initialize Linear weights
      init.xavier_uniform_(self.fc.weight)  # Xavier uniform initialization
      if self.fc.bias is not None:
          init.zeros_(self.fc.bias)
    elif init_type == 'uniform':
      for name, param in self.gru.named_parameters():
        if 'weight_ih' in name:
            torch.nn.init.uniform_(param.data, a=-math.sqrt(1/self.hidden_size), b=math.sqrt(1/self.hidden_size))
        elif 'weight_hh' in name:
            torch.nn.init.uniform_(param.data, a=-math.sqrt(1/self.hidden_size), b=math.sqrt(1/self.hidden_size))
        elif 'bias' in name:
            torch.nn.init.uniform_(param.data, a=-math.sqrt(1/self.hidden_size), b=math.sqrt(1/self.hidden_size))
      # Initialize Linear weights
      init.uniform_(self.fc.weight, a=-0.1, b=0.1)
      if self.fc.bias is not None:
          init.uniform_(self.fc.bias, a=-0.1, b=0.1)
    elif init_type == 'default':
      self.gru.reset_parameters()
      self.fc.reset_parameters()
    else:
      raise ValueError(f"Unknown initialization type: {init_type}")
    print('  GRU Network initialized')

class HeadGRU:
  def __init__(self, input_size, hidden_size, num_layers, dropout, output_size,layer_norm):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.reconstruction_factor = None
    self.output_size = output_size
    self.params = {
      'input_size': int(input_size),
      'hidden_size': int(hidden_size),
      'num_layers': int(num_layers),
      'dropout': float(dropout),
      'output_size': int(output_size)
    }
    print(f'\n\nself.params: {self.params}\n\n')
    self.model = GRUModel(input_size, hidden_size, num_layers, dropout, output_size,layer_norm)

  def get_params_configuration(self):
    return self.params
  
  def load_state_weights(self, path):
    self.model.load_state_dict(torch.load(path))
    print(f'Model weights loaded from {path}')

  def _group_features_by_sample_id(self, X, sample_ids, subject_ids, y=None):
    """
    Groups features by sample ID and pads sequences to the same length.
    Args:
      X (torch.Tensor): shape [nr_videos * nr_windows, temporal_dim, patch_h, patch_w, emb_dim]
      y (torch.Tensor): shape [nr_videos * nr_windows]
      sample_ids (torch.Tensor): shape [nr_videos * nr_windows]
      subject_ids (torch.Tensor): shape [nr_videos * nr_windows]
    Returns:
      tuple: A tuple containing:
        - padded_sequence (torch.nn.utils.rnn.pad_sequence): The padded sequence of X
        - y_train_per_sample_id (torch.Tensor): The mean target values per sample ID
        - subject_ids_per_sample_id (torch.Tensor): The subject IDs per sample ID
        - length_features (torch.Tensor): The lengths of features per sample ID
    """
    X_per_sample_id = []
    length_features = []
    y_train_per_sample_id = []
    subject_ids_per_sample_id = []
    max_len = 0
    if y == None:
      y = torch.zeros(X.shape[0])
    for id in torch.unique(sample_ids): 
      mask = torch.nonzero(sample_ids == id).squeeze()
      # print(f'GROUP.mask.shape: {mask.shape}')
      tmp_X_train = X[mask]
      # print(f'GROUP.tmp_X_train.shape: {tmp_X_train.shape}')
      length_features.append(tmp_X_train.shape[0]) # nr_windows
      if tmp_X_train.shape[0] > max_len:
        max_len = tmp_X_train.shape[0]
      #  video reshaped to [1, n_windows * temp_size, path_w,patch_h, emb_dim]
      X_per_sample_id.append(tmp_X_train)
      y_train_per_sample_id.append(torch.mean(y[mask].float())) # in Biovid dataset, target is the same for all windows in video, not in UNBC
      subject_ids_per_sample_id.append(subject_ids[mask][0])
    padded_sequence = torch.nn.utils.rnn.pad_sequence(X_per_sample_id, batch_first=True)
    
    length_features = torch.tensor(length_features)
    # print(f'GROUP-length_features.shape: {length_features}')
    subject_ids_per_sample_id = torch.tensor(subject_ids_per_sample_id)
    y_train_per_sample_id = torch.tensor(y_train_per_sample_id)
    padded_sequence = padded_sequence.reshape(padded_sequence.shape[0], # (batch_size)||| nr_videos  
                                                              -1,    # (seq_len)   ||| nr_windows * temp_size OR nr_windows  
                                                   self.input_size)
    
    return padded_sequence, y_train_per_sample_id, subject_ids_per_sample_id, length_features
  
  def get_data_loader(self, X, y, subject_ids, sample_ids, batch_size=2, shuffle=True,reconstruction_factor=None):
    
    padded_X, packed_y, packed_subject_ids, length_seq = self._group_features_by_sample_id(X=X,
                                                                                           y=y,
                                                                                           sample_ids=sample_ids,
                                                                                           subject_ids=subject_ids)
    X_embed_temp_size = X.shape[1]
    X_emb_dim = X.shape[2] * X.shape[3] * X.shape[4]
    extension_for_length_seq = X_embed_temp_size/(self.input_size/X_emb_dim) # is 1 if temp_size=8 is considered in input_dim, otherwise I have to considere the extension in the padding for computing the real length of the video
    self.reconstruction_factor = extension_for_length_seq * length_seq
    if self.reconstruction_factor is None:
      if reconstruction_factor is not None:
        self.reconstruction_factor = reconstruction_factor
      else:
        raise ValueError("Reconstruction factor is not set.")
       
    dataset = TensorDataset(padded_X, 
                            packed_y, 
                            packed_subject_ids, 
                            self.reconstruction_factor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  
  def start_train(self, X_train, y_train, subject_ids_train,train_csv_path,
                       X_val, y_val, subject_ids_val,sample_ids_train, sample_ids_val, 
                       shuffle_video_chunks,shuffle_training_batch,
                       num_epochs, batch_size, criterion, 
                       optimizer_fn, lr, saving_path, init_weights, 
                       round_output_loss,key_for_early_stopping,
                      #  scheduler,
                       early_stopping,enable_scheduler,
                       init_network,regularization_loss,regularization_lambda,write_logs=True):                  

    device = 'cuda'
    self.model.to(device)
    print('Model to', device)
    # Init model weights
    if init_weights:
      self.model._initialize_weights(init_type=init_network)

    # Reshape inputs
    # print(f'X_train.shape: {X_train.shape}') # [36, 8, 1, 1, 384] # 
    X_embed_temp_size = X_train.shape[1]
    X_emb_dim = X_train.shape[2] * X_train.shape[3] * X_train.shape[4]

    print('Loading data...')
    padded_X_train, packed_y_train, packed_subject_ids_train, length_seq_train = self._group_features_by_sample_id(X=X_train,
                                                                                                                   y=y_train,
                                                                                                                   sample_ids=sample_ids_train,
                                                                                                                   subject_ids=subject_ids_train)
    padded_X_val, packed_y_val, packed_subject_ids_val, len_seq_val = self._group_features_by_sample_id(X=X_val,
                                                                                                            y=y_val,
                                                                                                            sample_ids=sample_ids_val,
                                                                                                            subject_ids=subject_ids_val)

    extension_for_length_seq = X_embed_temp_size/(self.input_size/X_emb_dim) # is 1 if temp_size=8 is considered in input_dim, otherwise I have to considere the extension in the padding for computing the real length of the video
    self.reconstruction_factor = extension_for_length_seq * len_seq_val
    print('Creating datasets...')
    train_dataset = TensorDataset(padded_X_train, 
                                  packed_y_train,           # [nr_videos]
                                  packed_subject_ids_train, # [nr_videos]
                                  length_seq_train * extension_for_length_seq) # [nr_videos] To reconstruct the original length of the video (multiplied by temp_size)
    
    test_dataset = TensorDataset(padded_X_val,   
                                 packed_y_val,
                                 packed_subject_ids_val,
                                 len_seq_val * extension_for_length_seq)

    train_loader = None
    try:
      print('Try to use custom DataLoader...')
      customSampler_train = customSampler(path_cvs_dataset=train_csv_path, batch_size=batch_size, shuffle=shuffle_training_batch)
      customSampler_train.initialize()
      train_loader = DataLoader(train_dataset, sampler=customSampler_train)
      print('Custom DataLoader instantiated')
    except Exception as e:
      print(f'Err: {e}')
      print(f'Use standard DataLoader')
      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_training_batch)
      
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optimizer_fn(self.model.parameters(), 
                             lr=lr,
                             weight_decay=regularization_lambda if regularization_loss == 'L2' else 0)
    
    print(f'Optimizer settings:\n {optimizer}')
    if enable_scheduler:
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer=optimizer, 
                                                              mode='min' if key_for_early_stopping == 'val_loss' else 'max',
                                                              cooldown=5,
                                                              patience=10,
                                                              factor=0.1,
                                                              verbose=True,
                                                              threshold=1e-4,
                                                              threshold_mode='abs',
                                                              min_lr=1e-7)
    # Unique classes and subjects
    unique_train_val_classes = np.unique(np.concatenate((y_train, y_val)))
    unique_train_val_subjects = np.unique(np.concatenate((subject_ids_train, subject_ids_val)))
    train_losses = []
    val_losses = []
    nr_uniq_classes =list(range((max(unique_train_val_classes)+2)))[-1] # to get all classes in the confusion matrix, sometimes there are missing classes
    train_loss_per_class = np.zeros((num_epochs, nr_uniq_classes )) 
    val_loss_per_class = np.zeros((num_epochs, nr_uniq_classes ))
    train_loss_per_subject = np.zeros((num_epochs, unique_train_val_subjects.shape[0]))
    val_loss_per_subject = np.zeros((num_epochs, unique_train_val_subjects.shape[0]))
    train_confusion_matricies = []
    val_confusion_matricies = []
    list_train_macro_accuracy = []
    list_val_macro_accuracy = []
    saving_path_logs = os.path.join(saving_path,'logs')
    if not os.path.exists(saving_path_logs):
      os.makedirs(saving_path_logs)
    log_file_path = os.path.join(saving_path_logs, 'training_log.txt') 
    log_batch_path = os.path.join(saving_path_logs, 'batch_log.txt')
    # nr_samples_per_class = np.zeros(nr_uniq_classes)
    # nr_samples_per_subject = np.zeros(unique_subjects.shape[0])
    ##########################
    ##### START TRAINING #####
    # key_for_early_stopping = 'val_loss' if self.model.output_size == 1 else 'val_macro_precision'
    early_stopping.reset() 
    count_epoch = 0
    for epoch in range(num_epochs):
      count_epoch += 1
      self.model.train()
      epoch_loss = 0.0
      class_loss = np.zeros(nr_uniq_classes)
      subject_loss = np.zeros(unique_train_val_subjects.shape[0])
      # class_counts = np.zeros(unique_classes.shape[0])
      # subject_counts = np.zeros(unique_subjects.shape[0])
      # total_batches = len(train_loader)
      count_batch = 0
      train_confusion_matricies.append(ConfusionMatrix(task="multiclass",num_classes=(nr_uniq_classes+1)))
      val_confusion_matricies.append(ConfusionMatrix(task="multiclass",num_classes=(nr_uniq_classes+1)))
      for batch_X, batch_y, batch_subjects, batch_real_length_padded_feat in train_loader:
        optimizer.zero_grad()
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        if isinstance(criterion, nn.CrossEntropyLoss):
          batch_y = batch_y.long()
        if len(batch_X.shape) == 4:
          batch_X = batch_X.squeeze(0)
          batch_y = batch_y.squeeze(0)
          batch_subjects = batch_subjects.squeeze(0)
          batch_real_length_padded_feat = batch_real_length_padded_feat.squeeze(0)

        # Forward pass
        if shuffle_video_chunks:
          batch_X = torch.stack([batch_X[i, torch.randperm(batch_X.size(1))] for i in range(batch_X.size(0))])

        packed_input = pack_padded_sequence(batch_X, batch_real_length_padded_feat, batch_first=True, enforce_sorted=False)
        outputs = self.model.forward(x=packed_input, pred_only_last_time_step=True)
        
        if round_output_loss:
          outputs = torch.round(outputs)

        loss = criterion(outputs, batch_y)
        if regularization_loss == 'L1':
          l1_norm = sum(p.abs().sum() for p in self.model.parameters())
          loss += regularization_lambda * l1_norm
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        # Compute per-class and per-subject losses
        tools.compute_loss_per_class(class_loss=class_loss, 
                                     unique_train_val_classes=unique_train_val_classes,
                                     batch_y=batch_y, 
                                     outputs=outputs, criterion=criterion)


        tools.compute_loss_per_subject(subject_loss=subject_loss,
                                       unique_train_val_subjects=unique_train_val_subjects,
                                       batch_subjects=batch_subjects,
                                       outputs=outputs,
                                       criterion=criterion)
        if self.model.output_size == 1:
          outputs = torch.round(outputs).detach().cpu() # round to the nearest even number if 0.5
          # output_postprocessed = torch.where((outputs >= 0) & (outputs < unique_classes.shape[0]), outputs, torch.tensor(unique_classes.shape[0], device=device)) #from 0 to unique_classes.shape[0] - 1
          mask = torch.isin(outputs, torch.tensor(unique_train_val_classes))
          output_postprocessed = outputs*mask
        else:
          output_postprocessed = torch.argmax(outputs, dim=1)
        
        train_confusion_matricies[-1].update(output_postprocessed.detach().cpu(),
                                                batch_y.detach().cpu())
        
        # unique_batch_subject, count_subject = torch.unique(batch_subjects, return_counts=True)
        count_batch += 1

      train_confusion_matricies[epoch].compute()
      train_accuracy_dict = tools.get_accuracy_from_confusion_matrix(confusion_matrix=train_confusion_matricies[epoch],
                                                               list_real_classes=unique_train_val_classes)
      list_train_macro_accuracy.append(train_accuracy_dict['macro_precision'])
      # Class and subject losses
      
      
      train_loss_per_class[epoch] = class_loss / len(train_loader) # mean per nr samples in a class
      train_loss_per_subject[epoch] = subject_loss / len(train_loader) # mean per subject in a batch

      # Track training loss
      train_loss = epoch_loss / len(train_loader)
      train_losses.append(train_loss)
      
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      print(f'Epoch [{epoch}/{num_epochs}] | {current_time}')
      # test_accuracy_per_class[epoch] = dict_eval['test_accuracy_per_class']
      print(f' Train')
      print(f'  Loss             : {train_loss:.4f} ')
      print(f'  Accuracy         : {train_accuracy_dict["macro_precision"]:.4f}')

      # Save training log to an external file
      if write_logs:
        with open(log_file_path, 'a') as log_file:
          log_file.write(f'Epoch [{epoch}/{num_epochs}] | {current_time}\n')
          log_file.write(f' Train\n')
          log_file.write(f'  Loss             : {train_loss:.4f}\n')
          log_file.write(f'  Accuracy         : {train_accuracy_dict["macro_precision"]:.4f}\n')
          log_file.write(f'  Loss per_class   : {train_loss_per_class[epoch]}\n')
          log_file.write(f'  Prec. per_class  : {train_accuracy_dict["precision_per_class"]}\n')

      # Evaluate
      # del batch_X, batch_y, batch_subjects, batch_real_length_padded_feat
      # del packed_input, outputs, loss, output_postprocessed
      dict_eval = self.evaluate(
        val_loader=val_loader, criterion=criterion, device=device, unique_train_val_classes=unique_train_val_classes,nr_uniq_classes=nr_uniq_classes,
        unique_train_val_subjects= unique_train_val_subjects, val_confusion_matricies= val_confusion_matricies[-1],is_test=False,
        round_output_loss=round_output_loss, log_file_path=log_file_path if write_logs else None)

      # Save test loss
      val_losses.append(dict_eval['val_loss'])
      val_loss_per_class[epoch] = dict_eval['val_loss_per_class']
      val_loss_per_subject[epoch] = dict_eval['val_loss_per_subject']
      list_val_macro_accuracy.append(dict_eval['val_macro_precision'])

      free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
      total_gpu_mem = total_gpu_mem / 1024 ** 3
      free_gpu_mem = free_gpu_mem / 1024 ** 3
      print(f'GPU free/total (GB) : {free_gpu_mem:.2f}/{total_gpu_mem:.2f}')
      print('\n')  
      
      
      if epoch == 0 or (dict_eval[key_for_early_stopping] < best_test_loss if key_for_early_stopping == 'val_loss' else dict_eval[key_for_early_stopping] > best_test_loss):
        best_test_loss = dict_eval[key_for_early_stopping]
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_model_state = {key: value.cpu() for key, value in best_model_state.items()}
        best_model_epoch = epoch
      
      # Update learning rate
      if enable_scheduler:
        scheduler.step(dict_eval[key_for_early_stopping])
      
      if early_stopping(dict_eval[key_for_early_stopping]):
        break
      print(early_stopping)
    del batch_X, batch_y, batch_subjects, batch_real_length_padded_feat
    del packed_input, outputs, loss, output_postprocessed
    torch.cuda.empty_cache() # can help to reduce fragmentation
      
    # Save model weights
    if saving_path:
      print('Load and save best model for next steps...')
      torch.save(best_model_state, os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth'))
      print(f"Best model weights saved to {os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth')}")

    train_unique_subject_ids,train_count_subject_ids = np.unique(subject_ids_train,return_counts=True)
    val_unique_subject_ids,val_count_subject_ids = np.unique(subject_ids_val,return_counts=True)
    return {
      'train_losses': train_losses,
      'train_loss_per_class': train_loss_per_class,
      'train_loss_per_subject': train_loss_per_subject,
      'val_losses': val_losses,
      'val_loss_per_class': val_loss_per_class,
      'val_loss_per_subject': val_loss_per_subject,
      'y_unique': unique_train_val_classes,
      'train_unique_subject_ids': train_unique_subject_ids,
      'train_count_subject_ids': train_count_subject_ids,
      'val_unique_subject_ids': val_unique_subject_ids,
      'val_count_subject_ids': val_count_subject_ids,
      'train_unique_y': np.unique(y_train),
      'val_unique_y': np.unique(y_val),
      'subject_ids_unique': unique_train_val_subjects,
      # 'train_accuracy_per_class': train_accuracy_per_class,
      # 'test_accuracy_per_class': test_accuracy_per_class,
      'train_confusion_matricies': train_confusion_matricies,
      'val_confusion_matricies': val_confusion_matricies,
      'best_model_idx': best_model_epoch,
      'best_model_state': best_model_state,
      'list_train_macro_accuracy': list_train_macro_accuracy,
      'list_val_macro_accuracy': list_val_macro_accuracy,
      'epochs': count_epoch
    }

  def evaluate(self, val_loader, criterion, device,nr_uniq_classes, unique_train_val_classes, unique_train_val_subjects, val_confusion_matricies, round_output_loss,log_file_path,is_test):
    # self.model.to(device)
    val_loss = 0.0
    
    val_loss_per_class = np.zeros(nr_uniq_classes)
    subject_loss = np.zeros(unique_train_val_subjects.shape[0])
    subject_count = np.zeros(unique_train_val_subjects.shape[0])
    self.model.to('cuda')
    self.model.eval()
    with torch.no_grad():
      for batch_X, batch_y, batch_subjects, batch_real_length_padded_feat in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).to(int)

        # Forward pass
        packed_input = pack_padded_sequence(batch_X, batch_real_length_padded_feat, batch_first=True, enforce_sorted=False)
        outputs = self.model.forward(x=packed_input, pred_only_last_time_step=True)
        if round_output_loss: # split
          outputs = torch.round(outputs)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        if self.model.output_size == 1:
          outputs_cpu = torch.round(outputs.detach().cpu()) # round to the nearest even number if .5
          mask = torch.isin(outputs_cpu, torch.tensor(unique_train_val_classes))
          output_postprocessed_confusion_matrix = outputs_cpu * mask
        else:
          output_postprocessed_confusion_matrix = torch.argmax(outputs, dim=1)
        for cls in unique_train_val_classes:
          mask = (batch_y == cls).reshape(-1)
          if mask.any():
            class_idx = np.where(unique_train_val_classes == cls)[0][0]
            val_loss_per_class[class_idx] += criterion(outputs[mask], batch_y[mask]).item()
        for subj in unique_train_val_subjects:
          mask = (batch_subjects == subj).reshape(-1)
          if mask.any():
            subject_count[np.where(unique_train_val_subjects == subj)[0][0]] += mask.sum().item()
            subj_idx = np.where(unique_train_val_subjects == subj)[0][0]
            if 'reduction' in criterion.__dict__:
              old_reduction = criterion.reduction
              criterion.reduction = 'sum'
              subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).item()
              criterion.reduction = old_reduction
            else:
              subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).item()
        
        val_confusion_matricies.update(output_postprocessed_confusion_matrix.detach().cpu(),
                                       batch_y.detach().cpu())
        # del batch_X, batch_y, batch_subjects, batch_real_length_padded_feat, packed_input, outputs, loss, output_postprocessed
        # torch.cuda.empty_cache()
    # del batch_X, batch_y, packed_input, outputs
    torch.cuda.empty_cache()
    # Class and subject losses
  
    mask_non_zero_idx = np.where(subject_count > 0)[0]
    subject_loss_non_zero = subject_loss[mask_non_zero_idx] / subject_count[mask_non_zero_idx]
    subject_loss[mask_non_zero_idx] = subject_loss_non_zero
    # subject_loss = subject_loss / subject_count
    # len_val_loader = len(val_loader)
    val_loss = val_loss / len(val_loader)
    # subject_loss = subject_loss / len(val_loader)
    val_loss_per_class = val_loss_per_class / len(val_loader)
    # test_confusion_matricies.compute()
    dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                     list_real_classes=unique_train_val_classes)
    print(' Val')
    print(f'  Loss             : {val_loss:.4f} ')
    print(f'  Accuracy         : {dict_precision_recall["macro_precision"]:.4f}')

    if log_file_path:
      with open(log_file_path, 'a') as log_file:
        log_file.write(f' Val\n')
        log_file.write(f'  Loss             : {val_loss:.4f}\n')
        log_file.write(f'  Accuracy         : {dict_precision_recall["macro_precision"]:.4f}\n')
        log_file.write(f'  Loss per_class   : {val_loss_per_class}\n')
        log_file.write('\n')
    if not is_test:
      return {
        'val_loss':val_loss,
        'val_loss_per_class':  val_loss_per_class,
        'val_loss_per_subject': subject_loss,
        'val_macro_precision': dict_precision_recall["macro_precision"],
        # 'dict_precision_recall': dict_precision_recall
        }
    else:
      # After the return I add the unique count for classes and subjects
      return{
        'test_loss':val_loss,
        'test_loss_per_class':  val_loss_per_class,
        'test_loss_per_subject': subject_loss,
        'test_macro_precision': dict_precision_recall["macro_precision"],
        'test_confusion_matrix': val_confusion_matricies,
        'dict_precision_recall': dict_precision_recall
        }
      
      
  def get_embeddings(self, X,sample_id,subject_id):
    """
    Generate embeddings for the input tensor using the model's GRU layer.
    Args:
      X (torch.Tensor): The input tensor for which embeddings are to be generated.
      device (str, optional): The device to run the model on. If None, it will default to 'cuda' if available, .

    Returns:
      torch.Tensor: The generated embeddings from the model's GRU layer.
    """
    device = 'cuda'
    X_padded,_,subject_ids_per_sample_id,length_features = self._group_features_by_sample_id(X=X, sample_ids=sample_id, subject_ids=subject_id)
    X_padded = X_padded.to(device)
    packed_input = pack_padded_sequence(X_padded, length_features, batch_first=True, enforce_sorted=False)
    self.model.to(device)
    self.model.eval()
    with torch.no_grad():
      packed_out, _ = self.model.gru(packed_input)
    
    out_padded,list_length = pad_packed_sequence(packed_out, batch_first=True)
    # emb = out_padded[torch.arange(out_padded.shape[0]), :list_length-1] # [1, hidden_size]
    # self.model.to('cpu')
    return out_padded, list_length, subject_ids_per_sample_id

  def predict(self, X,sample_id,subject_id,device=None,pred_only_last_time_step=True):
    # CHECK: pad the sequence input, technically it should start from backbone to extract the features from new videos
    X_padded,_,_,_ = self._group_features_by_sample_id(X=X, sample_ids=sample_id, subject_ids=subject_id)
    # packed_input = pack_padded_sequence(batch_X, batch_real_length_padded_feat, batch_first=True, enforce_sorted=False)
    if device is None:
      device = 'cuda' 
    # self.model.to(device)
    self.model.eval()
    with torch.no_grad():
      predictions = self.model.forward(X_padded,pred_only_last_time_step=pred_only_last_time_step)
    # self.model.to('cpu')
    # print('predictions.shape', predictions.shape)
    return predictions

class BaseHead(nn.Module):
  def __init__(self, model):
    super(BaseHead, self).__init__()
    self.model = model

  def start_train(self,num_epochs,criterion,optimizer,lr,saving_path,train_csv_path,val_csv_path,batch_size,
                  regularization_loss,regularization_lambda,early_stopping,key_for_early_stopping,enable_scheduler):
    device = 'cuda'
    self.model.to(device)
    optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=regularization_lambda if regularization_loss == 'L2' else 0)
    if enable_scheduler:
      scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer=optimizer, 
                                                              mode='min' if key_for_early_stopping == 'val_loss' else 'max',
                                                              cooldown=5,
                                                              patience=10,
                                                              factor=0.1,
                                                              verbose=True,
                                                              threshold=1e-4,
                                                              threshold_mode='abs',
                                                              min_lr=1e-7)
    root_folder_features="/media/villi/TOSHIBA EXT/samples_16_whole"
    val_dataset = customDatasetCSV(val_csv_path,root_folder_features=root_folder_features)
    val_loader = DataLoader(val_dataset,collate_fn=val_dataset._custom_collate,batch_size=batch_size)
    train_dataset = customDatasetCSV(train_csv_path,root_folder_features=root_folder_features)
    train_loader = DataLoader(train_dataset,collate_fn=train_dataset._custom_collate,batch_size=batch_size)
    # train_loader = DataLoader(train_dataset, collate_fn=train_dataset._custom_collate, batch_size=batch_size, num_workers=4, pin_memory=True)
    list_train_losses = []
    list_train_losses_per_class = []
    list_train_losses_per_subject = []
    list_train_confusion_matricies = []
    train_unique_classes = np.array(list(range(self.model.num_classes)))
    train_unique_subjects = train_dataset.get_unique_subjects()
    val_unique_classes = np.array(list(range(self.model.num_classes)))
    val_unique_subjects = val_dataset.get_unique_subjects()
    list_val_losses = []
    list_val_losses_per_class = []
    list_val_losses_per_subject = []
    list_val_confusion_matricies = []
    list_train_macro_accuracy = []
    list_val_macro_accuracy = []
    for epoch in range(num_epochs):
      self.model.train()
      class_loss = np.zeros(train_unique_classes.shape[0])
      subject_loss = np.zeros(train_unique_subjects.shape[0])
      train_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=train_unique_classes.shape[0])
      train_loss = 0.0
      subject_count_batch = np.zeros(train_unique_subjects.shape[0])
      count = 0
      for batch_X, batch_y, batch_subjects, key_padding_mask in train_loader:
        tmp = np.isin(train_unique_subjects,batch_subjects)
        subject_count_batch[tmp] += 1
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).long()
        key_padding_mask = key_padding_mask.to(device)
        optimizer.zero_grad()
        # batch_X = batch_X
        outputs = self.model(x=batch_X,key_padding_mask=key_padding_mask) # input [batch, seq_len, emb_dim]
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        # outputs = torch.argmax(outputs, dim=1)
        train_loss += loss.item()
        count+=1
        free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
        total_gpu_mem = total_gpu_mem / 1024 ** 3
        free_gpu_mem = free_gpu_mem / 1024 ** 3
        print(f'Loss : {train_loss}')
        print(f'Batch : {count}/{len(train_loader)}')
        print(f'GPU free/total (GB) : {free_gpu_mem:.2f}/{total_gpu_mem:.2f}\n')
        
        # Compute loss per class and subject
        tools.compute_loss_per_class(batch_y=batch_y,
                                     class_loss=class_loss,
                                     unique_train_val_classes=train_unique_classes,
                                     outputs=outputs,
                                     criterion=criterion)
        tools.compute_loss_per_subject(batch_subjects=batch_subjects,
                                       criterion=criterion,
                                       batch_y=batch_y,
                                       outputs=outputs,
                                       subject_loss=subject_loss,
                                       unique_train_val_subjects=train_unique_subjects)
        batch_y = batch_y.detach().cpu().reshape(-1)
        predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        train_confusion_matrix.update(predictions, batch_y)
      
      train_confusion_matrix.compute()
      list_train_losses.append(train_loss / len(train_loader))
      list_train_losses_per_class.append(class_loss / len(train_loader))
      list_train_losses_per_subject.append(subject_loss / subject_count_batch)
      list_train_confusion_matricies.append(train_confusion_matrix)
      dict_eval = self.evaluate(criterion=criterion,is_test=False,
                                unique_val_classes=val_unique_classes,
                                unique_val_subjects=val_unique_subjects,
                                val_loader=val_loader)
      list_val_losses.append(dict_eval['val_loss'])
      list_val_losses_per_class.append(dict_eval['val_loss_per_class'])
      list_val_losses_per_subject.append(dict_eval['val_loss_per_subject'])
      list_val_confusion_matricies.append(dict_eval['val_confusion_matrix'])
      train_dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=train_confusion_matrix,list_real_classes=train_unique_classes)
      list_train_macro_accuracy.append(train_dict_precision_recall['macro_precision'])
      list_val_macro_accuracy.append(dict_eval['val_macro_precision'])
      print(f'Epoch [{epoch}/{num_epochs}]')
      print(f' Train')
      print(f'  Loss             : {list_train_losses[-1]:.4f}')
      print(f'  Accuracy         : {train_dict_precision_recall["macro_precision"]:.4f}')
      print(f' Val')
      print(f'  Loss             : {dict_eval["val_loss"]:.4f}')
      print(f'  Accuracy         : {dict_eval["val_macro_precision"]:.4f}')
      free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
      total_gpu_mem = total_gpu_mem / 1024 ** 3
      free_gpu_mem = free_gpu_mem / 1024 ** 3
      print(f'GPU free/total (GB) : {free_gpu_mem:.2f}/{total_gpu_mem:.2f}\n')
      if epoch == 0 or (dict_eval[key_for_early_stopping] < best_test_loss if key_for_early_stopping == 'val_loss' else dict_eval[key_for_early_stopping] > best_test_loss):
        best_test_loss = dict_eval[key_for_early_stopping]
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_model_state = {key: value.cpu() for key, value in best_model_state.items()}
        best_model_epoch = epoch
      
      if enable_scheduler:
        scheduler.step(dict_eval[key_for_early_stopping])
        
      if early_stopping(dict_eval[key_for_early_stopping]):
        break
    if saving_path:
        print('Load and save best model for next steps...')
        torch.save(best_model_state, os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth'))
        print(f"Best model weights saved to {os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth')}")
    return {
      'train_losses': list_train_losses,
      'train_loss_per_class': np.array(list_train_losses_per_class),
      'train_loss_per_subject': np.array(list_train_losses_per_subject),
      'val_losses': list_val_losses,
      'val_loss_per_class': np.array(list_val_losses_per_class),
      'val_loss_per_subject': np.array(list_val_losses_per_subject),
      'y_unique': np.unique(np.concatenate((train_unique_classes,val_unique_classes),axis=0)),
      'train_unique_subject_ids': train_unique_subjects,
      'train_count_subject_ids': train_dataset.get_count_subjects(),
      'val_unique_subject_ids': val_unique_subjects,
      'val_count_subject_ids': val_dataset.get_count_subjects(),
      'train_unique_y': train_unique_classes,
      'val_unique_y': val_unique_classes,
      'subject_ids_unique': np.unique(np.concatenate((train_unique_subjects,val_unique_subjects),axis=0)),
      # 'train_accuracy_per_class': train_accuracy_per_class,
      # 'test_accuracy_per_class': test_accuracy_per_class,
      'train_confusion_matricies': list_train_confusion_matricies,
      'val_confusion_matricies': list_val_confusion_matricies,
      'best_model_idx': best_model_epoch,
      'best_model_state': best_model_state,
      'list_train_macro_accuracy': list_train_macro_accuracy,
      'list_val_macro_accuracy': list_val_macro_accuracy,
      'epochs': epoch
    }

  def evaluate(self, val_loader, criterion, unique_val_subjects, unique_val_classes, is_test):
    # unique_train_val_classes is only for eval but kept the name for compatibility
    device = 'cuda'
    self.model.to(device)
    self.model.eval()
    count = 0
    with torch.no_grad():
      val_loss = 0.0
      val_loss_per_class = np.zeros(self.model.num_classes)
      subject_loss = np.zeros(unique_val_subjects.shape[0])
      val_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=self.model.num_classes)
      subject_batch_count = np.zeros(unique_val_subjects.shape[0])
      for batch_X, batch_y, batch_subjects, key_padding_mask in val_loader:
        tmp = np.isin(unique_val_subjects,batch_subjects)
        subject_batch_count[tmp] += 1
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).long()
        key_padding_mask = key_padding_mask.to(device)
        outputs = self.model(x=batch_X,key_padding_mask=key_padding_mask)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        tools.compute_loss_per_class(batch_y=batch_y, class_loss=val_loss_per_class, unique_train_val_classes=unique_val_classes,
                                     outputs=outputs, criterion=criterion)
        tools.compute_loss_per_subject(batch_subjects=batch_subjects, criterion=criterion, batch_y=batch_y, outputs=outputs,
                                       subject_loss=subject_loss, unique_train_val_subjects=unique_val_subjects)
        
        batch_y = batch_y.detach().cpu().reshape(-1)
        predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        val_confusion_matricies.update(predictions, batch_y)
        count += 1
        print(f'Batch val: {count}/{len(val_loader)}')
      
      val_confusion_matricies.compute()
      val_loss = val_loss / len(val_loader)
      val_loss_per_class = val_loss_per_class / len(val_loader)
      subject_loss = subject_loss / subject_batch_count
      dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                       list_real_classes=unique_val_classes)
      
      if is_test:
        return {
          'test_loss': val_loss,
          'test_loss_per_class': val_loss_per_class,
          'test_loss_per_subject': subject_loss,
          'test_macro_precision': dict_precision_recall["macro_precision"],
          'test_confusion_matrix': val_confusion_matricies,
          'dict_precision_recall': dict_precision_recall
        }
      else:
        return {
          'val_loss': val_loss,
          'val_loss_per_class': val_loss_per_class,
          'val_loss_per_subject': subject_loss,
          'val_macro_precision': dict_precision_recall["macro_precision"],
          'val_confusion_matrix': val_confusion_matricies,
          'dict_precision_recall': dict_precision_recall
        }      
    
class LinearHead(BaseHead):
  def __init__(self, input_dim, num_classes, dim_reduction):
    model = LinearProbe(input_dim=input_dim, num_classes=num_classes,dim_reduction=dim_reduction)
    super().__init__(model)
    
class AttentiveHead(BaseHead):
  def __init__(self,input_dim,num_classes,num_heads):
    model = AttentiveProbe(input_dim=input_dim,num_classes=num_classes,num_heads=num_heads)
    super().__init__(model)
    

class AttentiveProbe(nn.Module):
  def __init__(self,input_dim,num_classes,num_heads):
    super().__init__()
    self.query = nn.Parameter(torch.randn(1, input_dim)) # [1, emb_dim]
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.attn = nn.MultiheadAttention(embed_dim=input_dim,
                                      num_heads=num_heads,
                                      dropout=0.0,
                                      batch_first=True # [batch_size, seq_len, emb_dim]
                                      )
    self.linear = nn.Linear(input_dim, num_classes)
    
  def forward(self, x, key_padding_mask=None):
    # x: [batch_size, seq_len, emb_dim]
    # key_padding_mask: [batch_size, seq_len]
    q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1) # [batch_size, 1, emb_dim]
    # sum_key_padding = torch.sum(key_padding_mask, dim=1) # [batch_size]
    attn_output,_ = self.attn(q, x, x, key_padding_mask=key_padding_mask) # [batch_size, 1, emb_dim]
    pooled = attn_output.squeeze(1) # [batch_size, emb_dim]
    logits = self.linear(pooled)
    return logits
  
  def _initialize_weights():
    pass
  
class LinearProbe(nn.Module):
  def __init__(self,dim_reduction,input_dim, num_classes):
    super().__init__()
    self.linear = nn.Linear(input_dim, num_classes)
    self.dim_reduction = dim_reduction
    self.num_classes = num_classes
    
  def forward(self, x):
    x = x.mean(dim=self.dim_reduction)
    logits = self.linear(x)
    return logits
  def _initialize_weights():
    pass
  
class EarlyStopping:
  def __init__(self, best, patience=5, min_delta=0,threshold_mode='rel'):
    """
    Args:
        patience (int): Number of epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change in the monitored metric to be considered as an improvement.
    """
    self.patience = patience
    self.min_delta = min_delta
    self.best = best
    if threshold_mode not in ['rel', 'abs']:
      raise ValueError(f"threshold_mode must be 'rel' or 'abs'. Got {threshold_mode}.")
    self.threshold_mode = threshold_mode
    self.counter = 0

  def __call__(self, val_loss):
    pass

  def __str__(self):
    pass

class earlyStoppingAccuracy(EarlyStopping):
  def __init__(self, patience=5, min_delta=0, best=0,threshold_mode='rel'):
    super().__init__(best, patience, min_delta,threshold_mode)
    
  def __call__(self, val_accuracy):
    if self.threshold_mode == 'rel':
      if self.best * (1 + self.min_delta) < val_accuracy:
        self.best = val_accuracy
        self.counter = 0  # Reset patience counter if loss improves
      else:
        self.counter += 1  # Increment counter if no improvement
        if self.counter >= self.patience:
          return True  # Stop training
      return False  # Continue training
    else: # abs
      if self.best + self.min_delta < val_accuracy:
        self.best = val_accuracy
        self.counter = 0  # Reset patience counter if loss improves
      else:
        self.counter += 1  # Increment counter if no improvement
        if self.counter >= self.patience:
          return True  # Stop training
      return False  # Continue training
  def get_config(self):
    return {
      'patience': self.patience,
      'min_delta': self.min_delta,
      'best': self.best,
      'threshold_mode': self.threshold
    }
  def reset(self):
    self.best = 0
    self.counter = 0
  def __str__(self):
    return f'{self.__class__.__name__}: {self.counter}|{self.patience} (counter|patience) '
    
class earlyStoppingLoss(EarlyStopping):
  def __init__(self, patience=5, min_delta=0,best=float('inf'),threshold_mode='rel'):
    super().__init__(best, patience, min_delta,threshold_mode)
  def reset(self):
    self.best = float('inf')
    self.counter = 0
    
  def __call__(self, val_loss):
    if self.threshold_mode == 'rel':
      if self.best * (1 - self.min_delta) > val_loss:
        self.best = val_loss
        self.counter = 0  # Reset patience counter if loss improves
      else:
        self.counter += 1  # Increment counter if no improvement
        if self.counter >= self.patience:
          return True  # Stop training
      return False  # Continue training
    
    elif self.threshold_mode == 'abs': # abs
      if self.best - self.min_delta > val_loss:
        self.best = val_loss
        self.counter = 0
      else:
        self.counter += 1
        if self.counter >= self.patience:
          return True
      return False
  def get_config(self):
    return {
      'patience': self.patience,
      'min_delta': self.min_delta,
      'best': self.best,
      'threshold_mode': self.threshold_mode
    }
  def get_class_name(self):
    return self.__class__.__name__
  
  def __str__(self):
    return f'{self.__class__.__name__}: {self.counter}|{self.patience} (counter|patience) '
  
  

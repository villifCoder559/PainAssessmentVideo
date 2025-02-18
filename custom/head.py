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
import copy
# import wandb

# class head:
#   def __init__(self,head):
#     if head == 'SVR':
#       self.head = SVR_head
class head:
  def __init__(self, head_type,head_params):
    if head_type == 'SVR':
      self.head = HeadSVR(svr_params=head_params)
    elif head_type == 'GRU':
      self.head = HeadGRU(dropout=head_params['dropout'], input_size=head_params['input_size'],
                          hidden_size=head_params['hidden_size'], num_layers=head_params['num_layers'])
    self.head_type = head_type

  def train(self, X_train, y_train, X_test, y_test, subject_ids_train, subject_ids_test, sample_ids_train, sample_ids_test=None, num_epochs=100,criterion=nn.L1Loss(),optimizer=optim.Adam,
            init_network='default', regularization_loss='L1', regularization_lambda=0.01):
    """
    Train the model using the provided training and testing data.

    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
    X_test (array-like): Testing data features.
    y_test (array-like): Testing data labels.
    subject_ids_train (array-like): Subject IDs for the training data.
    subject_ids_test (array-like): Subject IDs for the testing data.
    sample_ids_train (array-like): Sample IDs for the training data.
    sample_ids_test (array-like, optional): Sample IDs for the testing data. Defaults to None.
    num_epochs (int, optional): Number of epochs for training. Defaults to 100.
    criterion (torch.nn.Module, optional): Loss function. Defaults to nn.L1Loss().
    optimizer (torch.optim.Optimizer, optional): Optimizer. Defaults to optim.Adam.

    Returns:
    dict: Dictionary containing training and testing results.
    """
    if isinstance(self.head, HeadSVR):
      dict_results = self.head.fit(X_train, y_train, sample_ids_train, subject_ids_train,
                             X_test, y_test, subject_ids_test)
      # print(f'train_loss: {dict_results["train_losses"][-1]} \t test_loss: {dict_results["test_losses"][-1]}')

    elif isinstance(self.head, HeadGRU):
      dict_results = self.head.start_train(X_train, y_train, sample_ids_train, subject_ids_train,
                             X_test, y_test, subject_ids_test,key_for_early_stopping,
                             num_epochs, init_network=init_network, criterion=criterion, optimizer=optimizer,
                             regularization_loss=regularization_loss, regularization_lambda=regularization_lambda)
      # self.plot_loss(dict_results['train_losses'], dict_results['test_losses'])
    return dict_results

  def predict(self, X, predict_per_video = True): # X = [n_video, n_windows, temp_size, path_w, patch_h, emb_dim]
    """
    Predicts the output based on the input data X.
    Parameters:
      X (numpy.ndarray): Input data with shape [n_video, n_windows, temp_size, path_w, patch_h, emb_dim].
      predict_per_video (bool, optional): If True, predictions are made per video. If False, predictions are made per window. Default is True.
    Returns:
      numpy.ndarray: Predicted output.
    """
    # print(f'X.shape: {X.shape}')
    if isinstance(self.head, HeadGRU):
      if predict_per_video:
        X = X.reshape(X.shape[0], X.shape[1], -1) # [n_video, n_windows, temp_size*path_w*patch_h*emb_dim]
      else:
        X = X.reshape(X.shape[0]*X.shape[1], 1, -1) # [n_video*n_windows, 1, temp_size*path_w*patch_h*emb_dim]

    elif isinstance(self.head, HeadSVR):
      if predict_per_video:
        X = X.reshape(X.shape[0], -1) # [n_video, n_windows*temp_size*path_w*patch_h*emb_dim]
      else:
        X = X.reshape(X.shape[0]*X.shape[1], -1) # [n_video*n_windows, temp_size*path_w*patch_h*emb_dim]
    # print(f'X prediction.shape: {X.shape}')
    return self.head.predict(X)


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
  def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, output_size=1):
    super(GRUModel, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.output_size = output_size
    self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
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
      # print('AAAAAAAAAAAAAAAAAAApacked_out shape',packed_out.data.shape)
      out_padded,list_length = pad_packed_sequence(packed_out, batch_first=True)
    # print(f'out_padded.shape: {out_padded.shape}')
    # if pred_only_last_time_step:
    #   out = self.fc(out[:, -1, :])
    # else:
    last_hidden_layer = out_padded[torch.arange(out_padded.shape[0]), list_length-1] # [1, hidden_size]
    # print('last_hidden_layet shape',last_hidden_layer.shape)
    # print('beforFC shape',().shape)

    if pred_only_last_time_step:
      # print('last_hidden_layer shape',last_hidden_layer.shape)
      out = self.fc(last_hidden_layer) # [batch, output_size=1]
      # print('FORWARD out_shape',out.shape)
      out = out.squeeze(dim=1) 
    else:
      out = self.fc(out_padded).squeeze(dim=2) # [batch, seq_len, output_size=1]
    # if self.classification:
    #   out = self.softmax(out)
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
      

class HeadGRU:
  def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, output_size=1):
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
    self.model = GRUModel(input_size, hidden_size, num_layers, dropout, output_size)

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
                       num_epochs, batch_size, criterion, # fix batch_size = 1
                       optimizer_fn, lr, saving_path, init_weights, 
                       round_output_loss,key_for_early_stopping,
                      #  scheduler,
                       early_stopping,
                       init_network,regularization_loss,regularization_lambda):

    # Init model weights
    if init_weights:
      self.model._initialize_weights(init_type=init_network)

    # Reshape inputs
    # print(f'X_train.shape: {X_train.shape}') # [36, 8, 1, 1, 384] # 
    X_embed_temp_size = X_train.shape[1]
    X_emb_dim = X_train.shape[2] * X_train.shape[3] * X_train.shape[4]
    # print(f'X_test.shape: {X_test.shape}')
    # print(f'subject_ids_train.shape: {subject_ids_train.shape}') # (36,)
    # print(f'sample_ids_train.shape: {sample_ids_train.shape}') # (36,)
    # print(f'sample_ids_test.shape: {sample_ids_test.shape}')
    # print(f'subject_ids_test.shape: {subject_ids_test.shape}')
    # print(f'y_train.shape: {y_train.shape}')

    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
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

    # catch
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Device and optimizer setup
    device = 'cuda'
    print('Moving model to', device)
    self.model.to(device)
    print('Device model start_train_test:', device)

    # print('Device model start_train_test:', device)
    # tmp = next(self.model.gru.parameters()).device
    # print(f"The model is on: {tmp}")
    
    optimizer = optimizer_fn(self.model.parameters(), 
                             lr=lr,
                             weight_decay=regularization_lambda if regularization_loss == 'L2' else 0)
    
    print(f'Optimizer settings:\n {optimizer}')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                                                              mode='min' if key_for_early_stopping == 'val_loss' else 'max',
                                                              cooldown=5,
                                                              patience=10,
                                                              factor=0.1,
                                                              verbose=True,
                                                              threshold=0.0001,
                                                              threshold_mode='abs',
                                                              min_lr=0)
    # Unique classes and subjects
    unique_classes = np.unique(np.concatenate((y_train, y_val)))
    unique_subjects = np.unique(np.concatenate((subject_ids_train, subject_ids_val)))

    train_losses = []
    val_losses = []
    nr_uniq_classes =list(range((max(unique_classes)+2)))[-1] # to get all classes in the confusion matrix, sometimes there are missing classes
    # TOFIX: unique_classes.shape[0] is not the number of classes if there are missing classes
    train_loss_per_class = np.zeros((num_epochs, nr_uniq_classes )) 
    val_loss_per_class = np.zeros((num_epochs, nr_uniq_classes ))
    train_loss_per_subject = np.zeros((num_epochs, unique_subjects.shape[0]))
    val_loss_per_subject = np.zeros((num_epochs, unique_subjects.shape[0]))
    # train_accuracy_per_class = np.zeros((num_epochs, unique_classes.shape[0] + 1)) # +1 for the null class
    # test_accuracy_per_class = np.zeros((num_epochs, unique_classes.shape[0] + 1)) # +1 for the null class
    # train_confusion_matricies = [ConfusionMatrix(task="multiclass",num_classes=(nr_uniq_classes+1)) for _ in range(num_epochs)]
    # val_confusion_matricies = [ConfusionMatrix(task="multiclass",num_classes=(nr_uniq_classes+1)) for _ in range(num_epochs)]
    train_confusion_matricies = []
    val_confusion_matricies = []
    list_train_macro_accuracy = []
    list_val_macro_accuracy = []
    best_test_loss = float('inf') if key_for_early_stopping == 'val_loss' else -float('inf')
    best_model_state = None
    best_model_epoch = -1
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
      subject_loss = np.zeros(unique_subjects.shape[0])
      # class_counts = np.zeros(unique_classes.shape[0])
      # subject_counts = np.zeros(unique_subjects.shape[0])
      total_batches = len(train_loader)
      count_batch = 0
      train_confusion_matricies.append(ConfusionMatrix(task="multiclass",num_classes=(nr_uniq_classes+1)))
      val_confusion_matricies.append(ConfusionMatrix(task="multiclass",num_classes=(nr_uniq_classes+1)))
      with open(log_batch_path, 'a') as log_file:
        log_file.write(f'EPOCH {epoch}/{num_epochs}\n')
      for batch_X, batch_y, batch_subjects, batch_real_length_padded_feat in train_loader:
        # # print(f'device: {device}')
        if len(batch_X.shape) == 4:
          batch_X = batch_X.squeeze(0)
          batch_y = batch_y.squeeze(0)
          batch_subjects = batch_subjects.squeeze(0)
          batch_real_length_padded_feat = batch_real_length_padded_feat.squeeze(0)
        # print(f'batch_subjects: {batch_subjects}')
        # print(f'batch_X shape: {batch_X.shape}')
        # print(f'batch_y shape: {batch_y.shape}')
        # print(f'batch_subjects shape: {batch_subjects.shape}')
        # print(f'batch_real_length_padded_feat shape: {batch_real_length_padded_feat.shape}') 
        batch_X, batch_y = batch_X.to(device), batch_y.to(int).to(device)
        optimizer.zero_grad()
        
        # print(f'batch_X.shape: {batch_X.shape}') # [2,4,384*8=3072]
        # print(f'batch_y.shape: {batch_y.shape}')
        # print(f'batch_real_length_padded_feat: {batch_real_length_padded_feat}')
        # Forward pass
        if shuffle_video_chunks:
          batch_X = torch.stack([batch_X[i, torch.randperm(batch_X.size(1))] for i in range(batch_X.size(0))])

        packed_input = pack_padded_sequence(batch_X, batch_real_length_padded_feat, batch_first=True, enforce_sorted=False)
        # print(f'X_tmp.shape: {X_tmp.shape}')
        # print packed input shape
        # print(f'packed_input.data.shape: {packed_input.data.shape}')
        outputs = self.model.forward(x=packed_input, pred_only_last_time_step=True)
        # print(f'outputs.shape: {outputs.shape}')
        if round_output_loss:
          outputs = torch.round(outputs)
        
        loss = criterion(outputs, batch_y)
        if regularization_loss == 'L1':
          l1_norm = sum(p.abs().sum() for p in self.model.parameters())
          loss += regularization_lambda * l1_norm
        # if regularization_loss == 'L2':
        #   l2_norm = sum(p.pow(2).sum() for p in self.model.parameters())
        #   loss += regularization_lambda * l2_norm
        epoch_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Compute per-class and per-subject losses
        for cls in unique_classes:
          mask = (batch_y == cls).reshape(-1)
          # if epoch == 0:
          #   nr_samples_per_class[cls] += mask.sum().item()
          # print(f'mask.shape: {mask.shape}')
          # print(f'outputs: {outputs}')
          # print(f'batch_y: {batch_y}')
          if mask.any():
            class_idx = np.where(unique_classes == cls)[0][0]
            class_loss[class_idx] += criterion(outputs[mask], batch_y[mask]).item()
            # class_counts[class_idx] += mask.sum().item()

        for subj in unique_subjects:
          mask = (batch_subjects == subj).reshape(-1)
          # if epoch == 0:
          #   nr_samples_per_subject[subj] += mask.sum().item()
          if mask.any():
            subj_idx = np.where(unique_subjects == subj)[0][0]
            subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).item()
            # subject_counts[subj_idx] += mask.sum().item()
        if self.model.output_size == 1:
          outputs = torch.round(outputs).detach().cpu() # round to the nearest even number if 0.5
          # output_postprocessed = torch.where((outputs >= 0) & (outputs < unique_classes.shape[0]), outputs, torch.tensor(unique_classes.shape[0], device=device)) #from 0 to unique_classes.shape[0] - 1
          mask = torch.isin(outputs, torch.tensor(unique_classes))
          output_postprocessed = outputs*mask
        else:
          output_postprocessed = torch.argmax(outputs, dim=1)
        
        train_confusion_matricies[-1].update(output_postprocessed.detach().cpu(),
                                                batch_y.detach().cpu())
        
        unique_batch_class, count_class = torch.unique(batch_y.detach().cpu(), return_counts=True)
        unique_batch_subject, count_subject = torch.unique(batch_subjects, return_counts=True)
        count_array = torch.zeros(nr_uniq_classes).to(int) 
        count_array[unique_batch_class.to(int)] = count_class
        free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
        total_gpu_mem = total_gpu_mem / 1024 ** 3
        free_gpu_mem = free_gpu_mem / 1024 ** 3
        count_batch += 1
        with open(log_batch_path, 'a') as log_file:
          log_file.write(f' Batch {count_batch}/{total_batches} \n')
          log_file.write(f'  nr_sample_per_class : {count_array.tolist()}\n')
          log_file.write(f'  unique_subject      : {unique_batch_subject.tolist()}\n')
          log_file.write(f'  count_subject       : {count_subject.tolist()}\n')
          log_file.write(f'  GPU free/total (GB) : {free_gpu_mem:.2f}/{total_gpu_mem:.2f}\n')
          log_file.write("\n")
        print(f'\nBatch: {count_batch}/{len(train_loader)}  \nGPU free/total (GB) : {free_gpu_mem:.2f}/{total_gpu_mem:.2f}\n')
        del batch_X, batch_y
        torch.cuda.empty_cache()
        # sk_conf_matrix = confusion_matrix(batch_y.detach().cpu(), output_postprocessed.detach().cpu())
        # train_confusion_matricies[epoch].compute()
        # print(train_confusion_matricies[epoch].compute())
        
      train_confusion_matricies[epoch].compute()
      train_accuracy_dict = tools.get_accuracy_from_confusion_matrix(confusion_matrix=train_confusion_matricies[epoch],
                                                               list_real_classes=unique_classes)
      list_train_macro_accuracy.append(train_accuracy_dict['macro_precision'])
      # Class and subject losses
      
      
      train_loss_per_class[epoch] = class_loss / len(train_loader) # mean per nr samples in a class
      train_loss_per_subject[epoch] = subject_loss / len(train_loader) # mean per subject in a batch

      # Track training loss
      train_loss = epoch_loss / len(train_loader)
      train_losses.append(train_loss)
      # print(f'Compute conf matrix calc {epoch}')
      # dict_precision_recall = tools.get_accuracy_from_confusion_matrix(test_confusion_matricies[epoch])
      current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      print(f'Epoch [{epoch}/{num_epochs}] | {current_time}')
      # test_accuracy_per_class[epoch] = dict_eval['test_accuracy_per_class']
      print(f' Train')
      print(f'  Loss             : {train_loss:.4f} ')
      print(f'  Accuracy         : {train_accuracy_dict["macro_precision"]:.4f}')
      # print(f'  Loss per_class   : {train_loss_per_class[epoch]}')
      # print(f'  Prec. per_class  : {accuracy_dict["precision_per_class"]}')
      # print(f'  Prec. macro      : {dict_precision_recall["macro_precision"]}')
      # print(f'  Prec. micro      : {dict_precision_recall["micro_precision"]}')
      # print(f'  Prec. weigh.     : {dict_precision_recall["weighted_precision"]}')
      # Save training log to an external file
      with open(log_file_path, 'a') as log_file:
        log_file.write(f'Epoch [{epoch}/{num_epochs}] | {current_time}\n')
        log_file.write(f' Train\n')
        log_file.write(f'  Loss             : {train_loss:.4f}\n')
        log_file.write(f'  Accuracy         : {train_accuracy_dict["macro_precision"]:.4f}\n')
        log_file.write(f'  Loss per_class   : {train_loss_per_class[epoch]}\n')
        log_file.write(f'  Prec. per_class  : {train_accuracy_dict["precision_per_class"]}\n')
        # log_file.write(f'  Prec. macro      : {dict_precision_recall["macro_precision"]}\n')
        # log_file.write(f'  Prec. micro      : {dict_precision_recall["micro_precision"]}\n')
        # log_file.write(f'  Prec. weigh.     : {dict_precision_recall["weighted_precision"]}\n')
      # print(f'\tTrain Loss Per Subject: {train_loss_per_subject[epoch]}')
      # print(f'\tTest Loss Per Subject: {test_loss_per_subject[epoch]}')

      # Evaluate
      dict_eval = self.evaluate(
        val_loader=val_loader, criterion=criterion, device=device, unique_classes=unique_classes,nr_uniq_classes=nr_uniq_classes,
        unique_subjects= unique_subjects, val_confusion_matricies= val_confusion_matricies[-1],
        round_output_loss=round_output_loss, log_file_path=log_file_path)
      # test_accuracy_per_class[epoch] = dict_eval['test_accuracy_per_class']
      # Save test loss
      val_losses.append(dict_eval['val_loss'])
      val_loss_per_class[epoch] = dict_eval['val_loss_per_class']
      val_loss_per_subject[epoch] = dict_eval['val_loss_per_subject']
      list_val_macro_accuracy.append(dict_eval['val_macro_precision'])

      # wandb.log({
      #   'train_loss': train_loss,
      #   'val_loss': dict_eval['val_loss'],
      #   'train_macro_accuracy': train_accuracy_dict["macro_precision"],
      #   'val_macro_accuracy': dict_eval['val_macro_precision'],
      #   'train_loss_per_class': train_loss_per_class[epoch],
      #   'val_loss_per_class': val_loss_per_class[epoch],
      #   'train_loss_per_subject': train_loss_per_subject[epoch],
      #   'val_loss_per_subject': val_loss_per_subject[epoch],
      # })
      # Save the best model

      if dict_eval[key_for_early_stopping] < best_test_loss:
        best_test_loss = dict_eval[key_for_early_stopping]
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_model_epoch = epoch
      
      # Update learning rate
      if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(dict_eval[key_for_early_stopping])
        
      if early_stopping(dict_eval[key_for_early_stopping]):
        break

    # Save model weights
    if saving_path:
      print('Load and save best model for next steps...')
      torch.save(best_model_state, os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth'))
      # load the best model
      # self.model.load_state_dict(best_model_state)
      print(f"Best model weights saved to {os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth')}")
    # # Plot losses
    # tools.plot_losses(train_losses=train_losses, test_losses=test_losses)
    # self.model.to('cpu')
    return {
      'train_losses': train_losses,
      'train_loss_per_class': train_loss_per_class,
      'train_loss_per_subject': train_loss_per_subject,
      'val_losses': val_losses,
      'val_loss_per_class': val_loss_per_class,
      'val_loss_per_subject': val_loss_per_subject,
      'y_unique': unique_classes,
      'subject_ids_unique': unique_subjects,
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

  def evaluate(self, val_loader, criterion, device,nr_uniq_classes, unique_classes, unique_subjects, val_confusion_matricies, round_output_loss=True,log_file_path=None,is_test=False):
    self.model.eval()
    # self.model.to(device)
    val_loss = 0.0
    val_loss_per_class = np.zeros(nr_uniq_classes)
    subject_loss = np.zeros(unique_subjects.shape[0])
    self.model.to('cuda')
    with torch.no_grad():
      for batch_X, batch_y, batch_subjects, batch_real_length_padded_feat in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(int).to(device)

        # Forward pass
        packed_input = pack_padded_sequence(batch_X, batch_real_length_padded_feat, batch_first=True, enforce_sorted=False)
        outputs = self.model.forward(x=packed_input, pred_only_last_time_step=True)
        if round_output_loss: # split
          outputs = torch.round(outputs)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        if self.model.output_size == 1:
          outputs_cpu = torch.round(outputs.detach().cpu()) # round to the nearest even number if .5
          mask = torch.isin(outputs_cpu, torch.tensor(unique_classes))
          output_postprocessed = outputs_cpu*mask
        else:
          output_postprocessed = torch.argmax(outputs, dim=1)
        # output_postprocessed = torch.where((outputs >= 0) & (outputs < unique_classes.shape[0]), outputs, torch.tensor(unique_classes.shape[0], device=device))
        # Compute per-class and per-subject losses
        # for cls in unique_classes:
        #   mask = (batch_y == cls).reshape(-1)
        #   if mask.any():
        #     class_idx = np.where(unique_classes == cls)[0][0]
        #     batch_y_class = batch_y[mask].reshape(-1, 1)
        #     outputs_class = outputs[mask].reshape(-1, 1)
        #     val_loss_per_class[class_idx] += criterion(outputs_class, batch_y_class).item()
        for cls in unique_classes:
          mask = (batch_y == cls).reshape(-1)
          # if epoch == 0:
          #   nr_samples_per_class[cls] += mask.sum().item()
          # print(f'mask.shape: {mask.shape}')
          # print(f'outputs: {outputs}')
          # print(f'batch_y: {batch_y}')
          if mask.any():
            class_idx = np.where(unique_classes == cls)[0][0]
            val_loss_per_class[class_idx] += criterion(outputs[mask], batch_y[mask]).item() / len(val_loader)
            
            
            # class_counts[class_idx] += mask.sum().item()
        for subj in unique_subjects:
          mask = (batch_subjects == subj).reshape(-1)
          if mask.any():
            subj_idx = np.where(unique_subjects == subj)[0][0]
            subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).item() / len(val_loader)
        val_confusion_matricies.update(output_postprocessed.detach().cpu(),
                                       batch_y.detach().cpu())
        del batch_y, batch_X
        torch.cuda.empty_cache()
    # Class and subject losses
    val_loss = val_loss / len(val_loader)
    # test_confusion_matricies.compute()
    dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                     list_real_classes=unique_classes)
    print(' Val')
    print(f'  Loss             : {val_loss:.4f} ')
    print(f'  Accuracy         : {dict_precision_recall["macro_precision"]:.4f}')
    # print(f'  Loss per_class   : {val_loss_per_class}')
    # print(f'  Prec. per_class  : {dict_precision_recall["precision_per_class"]}')
    # print(f'  Prec. macro      : {dict_precision_recall["macro_precision"]}')
    # print(f'  Prec. micro      : {dict_precision_recall["micro_precision"]}')
    # print(f'  Prec. weigh.     : {dict_precision_recall["weighted_precision"]}')
    if log_file_path:
      with open(log_file_path, 'a') as log_file:
        log_file.write(f' Val\n')
        log_file.write(f'  Loss             : {val_loss:.4f}\n')
        log_file.write(f'  Accuracy         : {dict_precision_recall["macro_precision"]:.4f}\n')
        log_file.write(f'  Loss per_class   : {val_loss_per_class}\n')
        # log_file.write(f'  Prec. per_class  : {dict_precision_recall["precision_per_class"]}\n')
        # log_file.write(f'  Prec. macro      : {dict_precision_recall["macro_precision"]}\n')
        # log_file.write(f'  Prec. micro      : {dict_precision_recall["micro_precision"]}\n')
        # log_file.write(f'  Prec. weigh.     : {dict_precision_recall["weighted_precision"]}\n')
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
      return{
        'test_loss':val_loss,
        'test_loss_per_class':  val_loss_per_class,
        'test_loss_per_subject': subject_loss,
        'test_macro_precision': dict_precision_recall["macro_precision"],
        'test_confusion_matrix': val_confusion_matricies,
        'dict_precision_recall': dict_precision_recall
        }
      
  def get_embeddings(self, X,sample_id,subject_id, device=None):
    """
    Generate embeddings for the input tensor using the model's GRU layer.
    Args:
      X (torch.Tensor): The input tensor for which embeddings are to be generated.
      device (str, optional): The device to run the model on. If None, it will default to 'cuda' if available, .

    Returns:
      torch.Tensor: The generated embeddings from the model's GRU layer.
    """
    if device is None:
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

class CrossValidationGRU:
  def __init__(self, head):
    self.head = head
    self.input_size = head.input_size
    self.hidden_size = head.hidden_size
    self.num_layers = head.num_layers
    self.dropout = head.dropout
    self.output_size = head.output_size

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
    print(f'count: {self.counter} | best: {self.best}')
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
      elif self.best - self.min_delta > val_accuracy:
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
    return f'{self.__class__.__name__}\n (patience={self.patience},\n min_delta={self.min_delta},\n best={self.best},\n threshold_mode={self.threshold_mode})'  

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
    return f'{self.__class__.__name__}\n (patience={self.patience},\n min_delta={self.min_delta},\n best={self.best},\n threshold_mode={self.threshold_mode})'
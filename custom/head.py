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
from custom.helper import CUSTOM_DATASET_TYPE
import torch.nn.init as init
from sklearn.metrics import confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from custom.dataset import customSampler
from custom.dataset import customDatasetWhole,customDatasetAggregated
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

class BaseHead(nn.Module):
  def __init__(self, model,is_classification):
    super(BaseHead, self).__init__()
    self.model = model
    self.is_classification = is_classification
  def log_performance(self,epoch, stage,num_epochs, loss, precision):
      if epoch>-1:
        print(f'Epoch [{epoch}/{num_epochs}]')
      print(f' {stage}')
      print(f'  Loss             : {loss:.4f}')
      print(f'  Accuracy         : {precision:.4f}')
      free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
      total_gpu_mem = total_gpu_mem / 1024 ** 3
      free_gpu_mem = free_gpu_mem / 1024 ** 3
      print(f'GPU free/total (GB) : {free_gpu_mem:.2f}/{total_gpu_mem:.2f}\n')
      
  def start_train(self, num_epochs, criterion, optimizer,lr, saving_path, train_csv_path, val_csv_path ,batch_size,dataset_type,
                  round_output_loss, shuffle_training_batch, regularization_loss,regularization_lambda,concatenate_temp_dim,
                  early_stopping, key_for_early_stopping,enable_scheduler,root_folder_features,init_network,):
    device = 'cuda'
    self.model.to(device)
    if init_network:
      self.model._initialize_weights(init_type=init_network)
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
    
    list_train_losses = []
    list_train_losses_per_class = []
    list_train_losses_per_subject = []
    list_train_confusion_matricies = []
    train_dataset, train_loader = self.get_dataset_and_loader(batch_size=batch_size,
                                                              csv_path=train_csv_path,
                                                              root_folder_features=root_folder_features,
                                                              shuffle_training_batch=shuffle_training_batch,
                                                              is_training=True,
                                                              concatenate_temporal=concatenate_temp_dim,
                                                              dataset_type=dataset_type)
    val_dataset, val_loader = self.get_dataset_and_loader(batch_size=batch_size,
                                                          csv_path=val_csv_path,
                                                          root_folder_features=root_folder_features,
                                                          shuffle_training_batch=False,
                                                          is_training=False,
                                                          concatenate_temporal=concatenate_temp_dim,
                                                          dataset_type=dataset_type)
    
    train_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    train_unique_subjects = train_dataset.get_unique_subjects()
    val_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
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
      train_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=train_unique_classes.shape[0]+1)
      train_loss = 0.0
      subject_count_batch = np.zeros(train_unique_subjects.shape[0])
      count = 0
      for dict_batch_X, batch_y, batch_subjects in train_loader:
        tmp = np.isin(train_unique_subjects,batch_subjects)
        subject_count_batch[tmp] += 1
        batch_y = batch_y.to(device)
        dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
        optimizer.zero_grad()
        outputs = self.model(**dict_batch_X) # input [batch, seq_len, emb_dim]
        if round_output_loss:
          outputs = torch.round(outputs)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        # outputs = torch.argmax(outputs, dim=1)
        train_loss += loss.item()
        count+=1
        free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
        total_gpu_mem = total_gpu_mem / 1024 ** 3
        free_gpu_mem = free_gpu_mem / 1024 ** 3
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
        if self.is_classification:
          predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        else:
          predictions = torch.round(outputs).detach().cpu() # round to the nearest even number if 0.5
          mask = torch.isin(predictions, torch.tensor(train_unique_classes))
          predictions[~mask] = self.model.num_classes # put prediction in the last class (bad_classified)
          
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
      # log performance
      self.log_performance(stage='Train', num_epochs=num_epochs, loss=list_train_losses[-1], precision=train_dict_precision_recall['macro_precision'],epoch=epoch)
      self.log_performance(stage='Val', num_epochs=num_epochs, loss=dict_eval['val_loss'], precision=dict_eval['val_macro_precision'],epoch=-1,)
      
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
      val_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=self.model.num_classes+1) # last class is for bad_classified in regression
      subject_batch_count = np.zeros(unique_val_subjects.shape[0])
      for dict_batch_X, batch_y, batch_subjects in val_loader:
        tmp = np.isin(unique_val_subjects,batch_subjects)
        dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
        batch_y = batch_y.to(device).long()
        subject_batch_count[tmp] += 1
        outputs = self.model(**dict_batch_X)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        tools.compute_loss_per_class(batch_y=batch_y, class_loss=val_loss_per_class, unique_train_val_classes=unique_val_classes,
                                     outputs=outputs, criterion=criterion)
        tools.compute_loss_per_subject(batch_subjects=batch_subjects, criterion=criterion, batch_y=batch_y, outputs=outputs,
                                       subject_loss=subject_loss, unique_train_val_subjects=unique_val_subjects)
        batch_y = batch_y.detach().cpu().reshape(-1)
        if self.is_classification:
          predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        else:
          predictions = torch.round(outputs).detach().cpu() # round to the nearest even number if 0.5
          mask = torch.isin(predictions, torch.tensor(unique_val_classes))
          predictions[~mask] = self.model.num_classes # put prediction in the last class (bad_classified)
          
        val_confusion_matricies.update(predictions, batch_y)
        count += 1
      
      val_confusion_matricies.compute()
      val_loss = val_loss / len(val_loader)
      val_loss_per_class = val_loss_per_class / len(val_loader)
      subject_loss = subject_loss / subject_batch_count
      dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                       list_real_classes=unique_val_classes)
      if is_test:
        self.log_performance(stage='Test', num_epochs=-1, loss=val_loss, precision=dict_precision_recall['macro_precision'],epoch=-1)
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
  
  def get_dataset_and_loader(self,csv_path,root_folder_features,batch_size,shuffle_training_batch,is_training,dataset_type,concatenate_temporal,output_type):
    if dataset_type.value == CUSTOM_DATASET_TYPE.WHOLE.value:
      dataset_ = customDatasetWhole(csv_path,root_folder_features=root_folder_features,concatenate_temporal=concatenate_temporal,
                                    model=self.model)
    elif dataset_type.value == CUSTOM_DATASET_TYPE.AGGREGATED.value:
      dataset_ = customDatasetAggregated(csv_path=csv_path,
                                         root_folder_features=root_folder_features,
                                         concatenate_temporal=concatenate_temporal,
                                         model=self.model)
    else:
      raise ValueError(f'Unknown dataset type: {dataset_type}. Can be either "original" or "unique"')
    if is_training:
      try:
        print('Try to use custom DataLoader...')
        customSampler_train = customSampler(path_cvs_dataset=csv_path, 
                                            batch_size=batch_size,
                                            shuffle=shuffle_training_batch)
        customSampler_train.initialize()
        loader_ = DataLoader(dataset=dataset_, sampler=customSampler_train,collate_fn=dataset_.fake_collate,batch_size=1)
        print('Custom DataLoader instantiated')
      except Exception as e:
        print(f'Err: {e}')
        print(f'Use standard DataLoader')
        loader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle_training_batch,collate_fn=dataset_._custom_collate)
    else:
      loader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False,collate_fn=dataset_._custom_collate)
    return dataset_,loader_
  
  def load_state_weights(self, path):
    self.model.load_state_dict(torch.load(path))
    print(f'Model weights loaded from {path}')
    
class LinearHead(BaseHead):
  def __init__(self, input_dim, num_classes, dim_reduction):
    model = LinearProbe(input_dim=input_dim, num_classes=num_classes,dim_reduction=dim_reduction)
    is_classification = True if num_classes > 1 else False
    super().__init__(model,is_classification)
  def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
    return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, False)
    
class AttentiveHead(BaseHead):
  def __init__(self,input_dim,num_classes,num_heads):
    model = AttentiveProbe(input_dim=input_dim,num_classes=num_classes,num_heads=num_heads)
    is_classification = True if num_classes > 1 else False
    super().__init__(model,is_classification)
  def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
    return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, True)

class GRUHead(BaseHead):
  def __init__(self, input_size, hidden_size, num_layers, dropout, output_size,layer_norm):
    model = GRUProbe(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=output_size,layer_norm=layer_norm)
    is_classification = True if output_size > 1 else False
    super().__init__(model,is_classification)
  def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
    return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, False)
    
class GRUProbe(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout, output_size,layer_norm,pred_only_last_time_step=True):
    super(GRUProbe, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.num_classes = 5
    self.output_size = output_size
    self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    self.norm = nn.LayerNorm(hidden_size) if layer_norm else nn.Identity()
    self.fc = nn.Linear(hidden_size, output_size)
    self.pred_only_last_time_step=pred_only_last_time_step
    
  def get_configuration(self):
    dict_config = {
      'input_size': self.input_size,
      'hidden_size': self.hidden_size,
      'num_layers': self.num_layers,
      'dropout': self.dropout,
      'output_size': self.output_size
    }
    return dict_config

  def forward(self, x):
    # assert len(x.shape) == 3, f"Input shape should be (batch_size, sequence_length, input_size), got {x.shape}"
    packed_out, _ = self.gru(x)
    if torch.is_tensor(x):
      out_padded = packed_out
      list_length = torch.tensor([x.shape[1]]*x.shape[0])
    else:
      out_padded,list_length = pad_packed_sequence(packed_out, batch_first=True)

    last_hidden_layer = out_padded[torch.arange(out_padded.shape[0]), list_length-1] # [1, hidden_size]
    if self.pred_only_last_time_step:
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
    x = x.data # [batch, ....]
    if self.dim_reduction is not None:
      x = x.mean(dim=self.dim_reduction)
    x=x.reshape(x.shape[0], -1)
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
  
  

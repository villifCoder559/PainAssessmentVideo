import pickle
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, GroupKFold,GroupShuffleSplit
from sklearn.metrics import mean_absolute_error 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import custom.tools as tools
from torchmetrics.classification import ConfusionMatrix
import os
import torch.nn.init as init


# class head:
#   def __init__(self,head):
#     if head == 'SVR':
#       self.head = SVR_head
class head:
  def __init__(self, head):
    self.head = head  

  def train(self, X_train, y_train, X_test, y_test, subject_ids_train, subject_ids_test, sample_ids_train, sample_ids_test=None, num_epochs=100,criterion=nn.L1Loss(),optimizer=optim.Adam):
    if isinstance(self.head, HeadSVR):
      dict_results = self.head.fit(X_train, y_train, sample_ids_train, subject_ids_train, 
                             X_test, y_test, subject_ids_test)
      # print(f'train_loss: {dict_results["train_losses"][-1]} \t test_loss: {dict_results["test_losses"][-1]}')

    elif isinstance(self.head, HeadGRU):
      dict_results = self.head.start_train(X_train, y_train, sample_ids_train, subject_ids_train, 
                             X_test, y_test, subject_ids_test,
                             num_epochs, criterion=criterion, optimizer=optimizer)
      # self.plot_loss(dict_results['train_losses'], dict_results['test_losses'])
    return dict_results
  
  def predict(self, X):
    return self.head.predict(X)
class HeadSVR:
  def __init__(self, svr_params):
    self.params = svr_params
    self.svr = SVR(**svr_params)

  def fit(self, X_train, y_train, subject_ids_train, X_test, y_test, subject_ids_test, saving_path=None):
    """
    Evaluation training of SVR model.
    Parameters:
    X_train (array-like): Training data features.
    y_train (array-like): Training data labels.
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
    print(f'y_train-shape {y_train.shape}')
    print(f'y_train-shape {y_test.shape}')
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
    print(f'subject_ids_unique shape {subject_ids_unique.shape}')
    print(f'subject_ids_train_unique shape {np.unique(subject_ids_train).shape}')
    print(f'subject_ids_test_unique shape {np.unique(subject_ids_test).shape}')
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
      
    # print(f'train_loss_per_class: {train_loss_per_class} \t test_loss_per_class: {test_loss_per_class}')
    # print(f'train_loss_per_subject: {train_loss_per_subject} \t test_loss_per_subject: {test_loss_per_subject}')

    # print("diffenece:", np.sum(np.abs(y_train - pred_train) >= 1))
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



  def predict(self, X):
    predictions = self.svr.predict(X)
    return predictions

  def k_fold_cross_validation(self, X, y, groups, k=3, list_saving_paths_k_val=None):
    """ k-fold cross-validation training of SVR model. """
    # Use dictionary so you cann add w/o changing code
    print('X.shape', X.shape)
    print('y.shapey', y.shape)
    gss = GroupShuffleSplit(n_splits = k)
    results = cross_validate(self.svr, X, y, cv=gss,scoring='neg_mean_absolute_error', groups=groups, return_train_score=True, return_estimator=True)
    # scores = - scores
    # Print the scores for each fold and the mean score
    print("Keys:", results.keys())
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
    self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.gru(x)
    out = self.fc(out[:, -1, :])  # Get the last time step output
    return out
  
  def _initialize_weights(self):
    # Initialize GRU weights
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

class HeadGRU:
  def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, output_size=1):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.output_size = output_size
    self.params = {
      'input_size': int(input_size),
      'hidden_size': int(hidden_size),
      'num_layers': int(num_layers),
      'dropout': float(dropout),
      'output_size': int(output_size)
    }
    self.model = GRUModel(input_size, hidden_size, num_layers, dropout, output_size)

  def start_train_test(self, X_train, y_train, subject_ids_train,
                       X_test, y_test, subject_ids_test,
                       num_epochs=10, batch_size=32, criterion=nn.L1Loss(), # fix batch_size = 1
                       optimizer_fn=optim.Adam, lr=0.0001, saving_path=None):
    # Init model weights 
    # self.model._initialize_weights()

    # Reshape inputs
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    # DataLoaders
    train_dataset = TensorDataset(X_train, 
                                  y_train, 
                                  torch.tensor(subject_ids_train,dtype=torch.int32))
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    test_dataset = TensorDataset(X_test,
                                 y_test,
                                 torch.tensor(subject_ids_test, dtype=torch.int32))
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # Device and optimizer setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)
    self.model.to(device)
    optimizer = optimizer_fn(self.model.parameters(), lr=lr)

    # Unique classes and subjects
    unique_classes = np.unique(np.concatenate((y_train, y_test)))
    unique_subjects = np.unique(np.concatenate((subject_ids_train, subject_ids_test)))

    train_losses = []
    test_losses = []
    train_loss_per_class = np.zeros((num_epochs, unique_classes.shape[0]))
    test_loss_per_class = np.zeros((num_epochs, unique_classes.shape[0]))
    train_loss_per_subject = np.zeros((num_epochs, unique_subjects.shape[0]))
    test_loss_per_subject = np.zeros((num_epochs, unique_subjects.shape[0]))
    train_confusion_matricies = [ConfusionMatrix(task="multiclass",num_classes=unique_classes.shape[0]) for _ in range(num_epochs)]
    test_confusion_matricies = [ConfusionMatrix(task="multiclass",num_classes=unique_classes.shape[0]) for _ in range(num_epochs)]

    best_test_loss = float('inf')
    best_model_state = None
    best_model_epoch = -1

    for epoch in range(num_epochs):
      self.model.train()
      epoch_loss = 0.0
      class_loss = np.zeros(unique_classes.shape[0])
      subject_loss = np.zeros(unique_subjects.shape[0])
      class_counts = np.zeros(unique_classes.shape[0])
      subject_counts = np.zeros(unique_subjects.shape[0])

      for batch_X, batch_y, batch_subjects in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = self.model(batch_X)
        # outputs = torch.round(outputs)
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        # Compute per-class and per-subject losses
        for cls in unique_classes:
          mask = (batch_y == cls).reshape(-1)
          if mask.any():
            class_idx = np.where(unique_classes == cls)[0][0]
            class_loss[class_idx] += criterion(outputs[mask], batch_y[mask]).item()
            # class_counts[class_idx] += mask.sum().item()

        for subj in unique_subjects:
          mask = (batch_subjects == subj).reshape(-1)
          if mask.any():
            subj_idx = np.where(unique_subjects == subj)[0][0]
            subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).item()
            # subject_counts[subj_idx] += mask.sum().item()
        print(f' output: {torch.round(outputs.detach().cpu())}')
        print(f' batch_y: {batch_y.detach().cpu()}')
        # train_confusion_matricies[epoch].update(torch.round(outputs.detach().cpu()), batch_y.detach().cpu()) # FIX:regression go over 4 or under 0, save that as Null(?)

      # train_confusion_matricies[epoch].compute()
      # Class and subject losses
      train_loss_per_class[epoch] = class_loss 
      train_loss_per_subject[epoch] = subject_loss 

      # Track training loss
      avg_train_loss = epoch_loss / len(train_loader)
      train_losses.append(avg_train_loss)

      # Evaluate
      avg_test_loss, class_test_loss, subject_test_loss = self.evaluate(
        test_loader, criterion, device, unique_classes, unique_subjects, test_confusion_matricies= test_confusion_matricies[epoch]
      )

      # Save test loss 
      test_losses.append(avg_test_loss)
      test_loss_per_class[epoch] = class_test_loss
      test_loss_per_subject[epoch] = subject_test_loss
      
      print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
      print(f'\tTrain Loss Per Class: {train_loss_per_class[epoch]}')
      print(f'\tTrain Loss Per Subject: {train_loss_per_subject[epoch]}')
      print(f'\tTest Loss Per Class: {test_loss_per_class[epoch]}')
      print(f'\tTest Loss Per Subject: {test_loss_per_subject[epoch]}')

      # Save the best model
      if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        best_model_state = self.model.state_dict()
        best_model_epoch = epoch

    # Save model weights
    if saving_path:
      torch.save(best_model_state, os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth'))
      print(f"Best model weights saved to {saving_path}")

    # Plot losses
    tools.plot_losses(train_losses=train_losses, test_losses=test_losses)

    return {
      'train_losses': train_losses,
      'train_loss_per_class': train_loss_per_class,
      'train_loss_per_subject': train_loss_per_subject,
      'test_losses': test_losses,
      'test_loss_per_class': test_loss_per_class,
      'test_loss_per_subject': test_loss_per_subject,
      'y_unique': unique_classes,
      'subject_ids_unique': unique_subjects,
      'train_confusion_matricies': train_confusion_matricies,
      'test_confusion_matricies': test_confusion_matricies,
      'best_model_idx': best_model_epoch
    }

  def evaluate(self, test_loader, criterion, device, unique_classes, unique_subjects, test_confusion_matricies):
    self.model.eval()
    total_loss = 0.0
    class_loss = np.zeros(unique_classes.shape[0])
    subject_loss = np.zeros(unique_subjects.shape[0])

    with torch.no_grad():
      for batch_X, batch_y, batch_subjects in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = self.model(batch_X)
        loss = criterion(outputs, batch_y)
        total_loss += loss.item()

        # Compute per-class and per-subject losses
        for cls in unique_classes:
          mask = (batch_y == cls).reshape(-1)
          if mask.any():
            class_idx = np.where(unique_classes == cls)[0][0]
            batch_y_class = batch_y[mask].reshape(-1, 1)
            outputs_class = outputs[mask].reshape(-1, 1)
            class_loss[class_idx] += criterion(outputs_class, batch_y_class).item()

        for subj in unique_subjects:
          mask = (batch_subjects == subj).reshape(-1)
          if mask.any():
            subj_idx = np.where(unique_subjects == subj)[0][0]
            subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).item()
        
        test_confusion_matricies.update(outputs.detach().cpu(), batch_y.detach().cpu())

    # Class and subject losses
    avg_loss = total_loss / len(test_loader)
    test_confusion_matricies.compute()
    return avg_loss, class_loss, subject_loss

  def predict(self, X):
    predictions = self.model(X)
    return predictions

class CrossValidationGRU:
  def __init__(self, head):
    self.head = head
    self.input_size = head.input_size
    self.hidden_size = head.hidden_size
    self.num_layers = head.num_layers
    self.dropout = head.dropout
    self.output_size = head.output_size

  def k_fold_cross_validation(self, X, y, group_ids, k=5, num_epochs=10, batch_size=32,
                               criterion=nn.L1Loss(), optimizer_fn=optim.Adam, lr=0.001, 
                               list_saving_paths_k_val=None):
    """
    Perform k-fold cross-validation using GroupShuffleSplit.
    
    Args:
      X (numpy.ndarray): The input data.
      y (numpy.ndarray): The target labels.
      group_ids (numpy.ndarray): Group IDs for each sample.
      k (int): Number of folds.
      num_epochs (int): Number of epochs for each fold.
      batch_size (int): Batch size for training.
      criterion: Loss function.
      optimizer_fn: Optimizer function.
      lr (float): Learning rate.
      save_model_path (str): Path to save the model weights.

    Returns:
      dict: Cross-validation results including per-fold losses.
    """
    gss = GroupShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
    fold_results = []
    list_split_indices = []
    for fold_idx, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=group_ids)):
      print(f"Starting Fold {fold_idx + 1}/{k}")
      list_split_indices.append((train_idx,test_idx))

      # Split the data into training and testing sets
      X_train, X_test = X[train_idx], X[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]
      groups_train, groups_test = group_ids[train_idx], group_ids[test_idx]

      # Initialize the model
      model = HeadGRU(self.input_size, self.hidden_size, self.num_layers, self.dropout, self.output_size)
      print('X_train shape', X_train.shape)
      print('y_train shape', y_train.shape)
      print('groups_train shape', groups_train.shape)
      # Train and test the model
      fold_result = model.start_train_test(
        X_train, y_train, groups_train, X_test, y_test, groups_test,
        num_epochs=num_epochs, batch_size=batch_size, criterion=criterion,
        optimizer_fn=optimizer_fn, lr=lr
      )
      
      fold_results.append(fold_result)

      # Save model weights
      if list_saving_paths_k_val:
        torch.save(model.model.state_dict(), os.path.join(list_saving_paths_k_val[fold_idx], 'model_weights.pth'))
        print(f"Model weights for fold {fold_idx + 1} saved to {list_saving_paths_k_val[fold_idx]}")
      
      # Print fold results
      print(f"Fold {fold_idx + 1} Results:")
      print(f"  Train Losses: {fold_result['train_losses'][-1]:.4f}")
      print(f"  Test Losses: {fold_result['test_losses'][-1]:.4f}")

    # Aggregate results across folds
    avg_train_loss = np.mean([result['train_losses'][-1] for result in fold_results])
    avg_test_loss = np.mean([result['test_losses'][-1] for result in fold_results])

    print("Cross-Validation Results:")
    print(f"  Average Train Loss: {avg_train_loss:.4f}")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")

    return fold_results, list_split_indices

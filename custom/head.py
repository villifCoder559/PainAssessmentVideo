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
class HeadSVR:
  def __init__(self, svr_params):
    self.svr = SVR(**svr_params)


  def fit(self, X_train, y_train, subject_ids_train, X_test, y_test, subject_ids_test):
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
    """
    regressor = self.svr.fit(X_train, y_train)
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
    
    return {'train_losses': [train_loss], 'train_loss_per_class': train_loss_per_class.reshape(1, -1), 'train_loss_per_subject': train_loss_per_subject.reshape(1, -1), 
            'test_losses': [test_loss], 'test_loss_per_class': test_loss_per_class.reshape(1, -1), 'test_loss_per_subject': test_loss_per_subject.reshape(1, -1),
            'y_unique': y_unique, 'subject_ids_unique': subject_ids_unique}



  def predict(self, X):
    predictions = self.svr.predict(X)
    return predictions

  def k_fold_cross_validation(self, X, y, groups, k=3):
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
    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=groups), 1):
      list_split_indices.append((train_idx,test_idx))
    return list_split_indices,results
  
  
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
  
  
# class GRUModel(nn.Module):
#   def __init__(self, input_size, hidden_size, num_layers, dropout = 0.0, output_size = 1):
#     super(GRUModel, self).__init__()
#     print('input_size', input_size)
#     self.input_size = input_size
#     self.hidden_size = hidden_size
#     self.num_layers = num_layers
#     self.gru = nn.GRU(input_size, hidden_size, num_layers,dropout=dropout, batch_first=True)
#     self.fc = nn.Linear(hidden_size, output_size)
  
#   def forward(self, x):
#     # x.shape = (batch_size, seq_length, input_size)
#     out, _ = self.gru(x)
#     out = self.fc(out[:, -1, :]) # get the last time step output
#     return out
  
# class HeadGRU():
#   def __init__(self, input_size, hidden_size, num_layers, dropout = 0.0, output_size = 1):
#     self.model = GRUModel(input_size, hidden_size, num_layers, dropout, output_size)
  
#   def start_train_test(self, X_train, y_train, sample_ids_train, subject_ids_train, 
#                              X_test, y_test, subject_ids_test,
#                              num_epochs=1, criterion=nn.L1Loss(), optimizer=optim.Adam,lr = 0.001):
#     # Reshape X_train and X_test in -> (num_samples, seq_length, input_size)
#     # [nr_videos, nr_windows, T=16, patch_w, patch_h, emb_dim] -> [nr_videos, nr_windows, T=16*patch_w*patch_h*emb_dim]
    
#     X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
#     X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
#     print(f'sample_ids_train {sample_ids_train}')
#     uniqe_sample_ids = np.unique(sample_ids_train) # CHECK: sample_ids should be uniqe for each video, no need np.unique
#     batch_sequence=[]
#     for id in uniqe_sample_ids:
#       batch_sequence.append(np.sum(sample_ids_train == id)) # should be a list of 1s
#     print(f'batch_sequence: {batch_sequence}')
    
#     # Get the unique classes
#     y_train_unique = np.unique(y_train)
#     y_test_unique = np.unique(y_test)
#     y_unique = np.unique(np.concatenate((y_train_unique, y_test_unique)))

#     # Get the unique subjects
#     subject_ids_train_unique = np.unique(subject_ids_train)
#     subject_ids_test_unique = np.unique(subject_ids_test)
#     subject_ids_unique = np.unique(np.concatenate((subject_ids_train_unique, subject_ids_test_unique)))
    
#     del y_train_unique, y_test_unique, subject_ids_train_unique, subject_ids_test_unique

#     train_losses = []
#     train_loss_per_class = np.zeros((num_epochs, y_unique.shape[0])) 
#     train_loss_per_subject = np.zeros((num_epochs, subject_ids_unique.shape[0])) 
#     test_losses = []
#     test_loss_per_class = np.zeros((num_epochs, y_unique.shape[0])) 
#     test_loss_per_subject = np.zeros((num_epochs, subject_ids_unique.shape[0]))
#     optimizer = optimizer(self.model.parameters(),lr=lr)
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print('device GRU:',device)
#     self.model.to(device)
#     y_train=y_train.to(device)     # move to gpu to compute the loss
#     print('y_unique ',y_unique)
#     y_unique = torch.from_numpy(y_unique).to(device)

#     for epoch in range(num_epochs):
#       self.model.train()
#       epoch_loss = 0.0
#       i = 0
#       for batch in batch_sequence: # 1 
#         # Get the batch data
#         batch_X = X_train[i:i+batch] # shape (batch_size, embed_size)
#         batch_y = y_train[i:i+batch] # shape (batch_size)  not granted same class for all the batch (UNBC can have differnet classes in the same video)
#         batch_subject_ids = subject_ids_train[i:i+batch] # shape (batch_size) # same id for all the batch
#         i += batch

#         # Forward pass
#         optimizer.zero_grad()
#         # print(f'batch_X shape {batch_X.shape}')
#         outputs = self.model(batch_X)
#         # print(f'outputs_device: {outputs.device} \t batch_y_device: {batch_y.device} \t')
#         # Compute loss
#         loss = criterion(outputs, batch_y)
#         epoch_loss += loss.item()

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()

#         # Compute loss per class
#         batch_y_unique = torch.unique(batch_y)

#         for cls in batch_y_unique:
#           idx = torch.nonzero(batch_y == cls,as_tuple=True)[0]
#           class_loss = criterion(outputs[idx], batch_y[idx])
#           idx_y_unique_global = torch.nonzero(cls == y_unique,as_tuple=True)[0]
#           train_loss_per_class[epoch, idx_y_unique_global.cpu()] = class_loss.detach().cpu().item()
        
#         # Compute loss per subject
#         for id in np.unique(batch_subject_ids):
#           idx = np.where(batch_subject_ids == id)[0]
#           subject_loss = criterion(outputs[idx], batch_y[idx])

#           idx_subject_ids_unique_global = np.where(subject_ids_unique == id)
#           train_loss_per_subject[epoch, idx_subject_ids_unique_global] = subject_loss.detach().cpu().item()

#       # Track the training loss
#       avg_epoch_loss = epoch_loss / (X_train.size(0) // batch)
#       train_losses.append(avg_epoch_loss)
#       self.evaluate(epoch=epoch, X_test=X_test, y_test=y_test, subject_ids_test=subject_ids_test, 
#                     test_losses=test_losses, test_loss_per_class=test_loss_per_class, 
#                     test_loss_per_subject=test_loss_per_subject, subject_ids_unique=subject_ids_unique, y_unique=y_unique)

#       print(f'Epoch [{epoch+1}/{num_epochs}]')
#       print(f'\ttrain_loss: {avg_epoch_loss:.4f} \t test_loss: {test_losses[-1]:.4f}')
#       print(f'\ttrain_loss_per_class: {train_loss_per_class[epoch]} \t test_loss_per_class: {test_loss_per_class[epoch]}')
#       print(f'\ttrain_loss_per_subject: {train_loss_per_subject[epoch]} \t test_loss_per_subject: {test_loss_per_subject[epoch]}')

#     return {'train_losses': train_losses, 'train_loss_per_class': train_loss_per_class, 'train_loss_per_subject': train_loss_per_subject, 
#             'test_losses': test_losses, 'test_loss_per_class': test_loss_per_class, 'test_loss_per_subject': test_loss_per_subject,
#             'y_unique': y_unique.detach().cpu(), 'subject_ids_unique': subject_ids_unique}
#     # # 5. Plot the Training Loss
#     # plt.plot(train_losses)
#     # plt.xlabel('Epochs')
#     # plt.ylabel('Training Loss')
#     # plt.title('Training Loss over Epochs')
#     # plt.show()  
      
#   def evaluate(self, X_test, y_test, subject_ids_test ,test_losses, test_loss_per_class, test_loss_per_subject, epoch, subject_ids_unique, y_unique):
#       # 6. Evaluate the Model
#     self.model.eval()
#     with torch.no_grad():
#       predictions = self.model(X_test).squeeze().detach().cpu().numpy()
#       mae = mean_absolute_error(y_test, predictions)
#       test_losses.append(mae)
#       y_test = y_test.cpu().numpy()
#       y_unique = y_unique.cpu().numpy()
#       # Compute loss per class
#       for cls in np.unique(y_test):
#         idx_local = np.where(y_test == cls)[0]
#         idx_y_unique_global = np.where(y_unique == cls)
#         print('test_loss_per_class shape ',test_loss_per_class.shape)
#         print('idx_y_unique_global shape ',idx_y_unique_global)
#         print('epoch ',epoch)
#         test_loss_per_class[epoch, idx_y_unique_global] = mean_absolute_error(y_test[idx_local], predictions[idx_local])
      
#       # Compute loss per subject
#       for id in np.unique(subject_ids_test):
#         idx_local = np.where(subject_ids_test == id)[0]
#         idx_subject_ids_unique_global = np.where(subject_ids_unique == id)
#         test_loss_per_subject[epoch, idx_subject_ids_unique_global] = mean_absolute_error(y_test[idx_local], predictions[idx_local])  
#     # chunk_nr = 0
#     # for pred, y, id in zip(predictions, y, samples_id):
#     #   print(f"Video_{id}_{chunk_nr}: Pred. = {pred:.2f}, gt = {y}")


class GRUModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, output_size=1):
    super(GRUModel, self).__init__()
    self.gru = nn.GRU(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    out, _ = self.gru(x)
    out = self.fc(out[:, -1, :])  # Get the last time step output
    return out


class HeadGRU:
  def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, output_size=1):
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.dropout = dropout
    self.output_size = output_size
    self.model = GRUModel(input_size, hidden_size, num_layers, dropout, output_size)

  def start_train_test(self, X_train, y_train, subject_ids_train,
                       X_test, y_test, subject_ids_test,
                       num_epochs=10, batch_size=32, criterion=nn.L1Loss(),
                       optimizer_fn=optim.Adam, lr=0.001):
    # Reshape inputs
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)

    # DataLoaders
    train_dataset = TensorDataset(X_train, 
                                  y_train, 
                                  torch.tensor(subject_ids_train,dtype=torch.int32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(X_test,
                                 y_test,
                                 torch.tensor(subject_ids_test, dtype=torch.int32))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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


      # Normalize class and subject losses
      train_loss_per_class[epoch] = class_loss 
      train_loss_per_subject[epoch] = subject_loss 

      # Track training loss
      avg_train_loss = epoch_loss / len(train_loader)
      train_losses.append(avg_train_loss)

      # Evaluate
      avg_test_loss, class_test_loss, subject_test_loss = self.evaluate(
        test_loader, criterion, device, unique_classes, unique_subjects
      )
      test_losses.append(avg_test_loss)
      test_loss_per_class[epoch] = class_test_loss
      test_loss_per_subject[epoch] = subject_test_loss

      print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}')
      print(f'\tTrain Loss Per Class: {train_loss_per_class[epoch]}')
      print(f'\tTest Loss Per Class: {test_loss_per_class[epoch]}')
      print(f'\tTrain Loss Per Subject: {train_loss_per_subject[epoch]}')
      print(f'\tTest Loss Per Subject: {test_loss_per_subject[epoch]}')
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
      'subject_ids_unique': unique_subjects
    }

  def evaluate(self, test_loader, criterion, device, unique_classes, unique_subjects):
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

    # Normalize class and subject losses
    avg_loss = total_loss / len(test_loader)
    
    return avg_loss, class_loss, subject_loss

class CrossValidationGRU:
  def __init__(self, head):
    self.head = head
    self.input_size = head.input_size
    self.hidden_size = head.hidden_size
    self.num_layers = head.num_layers
    self.dropout = head.dropout
    self.output_size = head.output_size

  def k_fold_cross_validation(self, X, y, group_ids, k=5, num_epochs=10, batch_size=32,
                               criterion=nn.L1Loss(), optimizer_fn=optim.Adam, lr=0.001):
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

    Returns:
      dict: Cross-validation results including per-fold losses.
    """
    gss = GroupShuffleSplit(n_splits=k, test_size=0.2, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(gss.split(X, y, groups=group_ids)):
      print(f"Starting Fold {fold_idx + 1}/{k}")

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

      # Print fold results
      print(f"Fold {fold_idx + 1} Results:")
      print(f"  Train Losses: {fold_result['train_losses'][-1]:.4f}")
      print(f"  Test Losses: {fold_result['test_losses'][-1]:.4f}")

    # Aggregate results across folds
    avg_train_loss = np.mean([result['train_losses'][-1] for result in fold_results])
    avg_test_loss = np.mean([result['test_losses'][-1] for result in fold_results])

    print("\nCross-Validation Results:")
    print(f"  Average Train Loss: {avg_train_loss:.4f}")
    print(f"  Average Test Loss: {avg_test_loss:.4f}")

    return fold_results

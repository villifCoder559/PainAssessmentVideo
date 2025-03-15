import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import custom.tools as tools
from torchmetrics.classification import ConfusionMatrix
import math
import os
from custom.helper import CUSTOM_DATASET_TYPE
import torch.nn.init as init
from custom.dataset import customSampler
from torch.nn.utils.rnn import pack_padded_sequence,pack_padded_sequence
from custom.dataset import get_dataset_and_loader
import tqdm
import copy

# import wandb

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
      print(f'  GPU memory free  : {free_gpu_mem:.2f} GB')
      
  def start_train(self, num_epochs, criterion, optimizer,lr, saving_path, train_csv_path, val_csv_path ,batch_size,dataset_type,
                  round_output_loss, shuffle_training_batch, regularization_loss,regularization_lambda,concatenate_temp_dim,
                  early_stopping, key_for_early_stopping,enable_scheduler,root_folder_features,init_network,backbone_dict):
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
    train_dataset, train_loader = get_dataset_and_loader(batch_size=batch_size,
                                                              csv_path=train_csv_path,
                                                              root_folder_features=root_folder_features,
                                                              shuffle_training_batch=shuffle_training_batch,
                                                              is_training=True,
                                                              concatenate_temporal=concatenate_temp_dim,
                                                              dataset_type=dataset_type,
                                                              backbone_dict=backbone_dict,
                                                              model=self.model)
    val_dataset, val_loader = get_dataset_and_loader(batch_size=batch_size,
                                                          csv_path=val_csv_path,
                                                          root_folder_features=root_folder_features,
                                                          shuffle_training_batch=False,
                                                          is_training=False,
                                                          concatenate_temporal=concatenate_temp_dim,
                                                          dataset_type=dataset_type,
                                                          backbone_dict=backbone_dict,
                                                          model=self.model)
    
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
      
      for dict_batch_X, batch_y, batch_subjects in tqdm.tqdm(train_loader,total=len(train_loader),desc=f'Train {epoch}/{num_epochs}'):
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
      for dict_batch_X, batch_y, batch_subjects in tqdm.tqdm(val_loader,total=len(val_loader),desc='Validation' if not is_test else 'Test'):
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

  
  def load_state_weights(self, path):
    self.model.load_state_dict(torch.load(path))
    print(f'Model weights loaded from {path}')
    
class LinearHead(BaseHead):
  def __init__(self, input_dim, num_classes, dim_reduction):
    model = LinearProbe(input_dim=input_dim, num_classes=num_classes,dim_reduction=dim_reduction)
    is_classification = True if num_classes > 1 else False
    super().__init__(model,is_classification)
  # def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
  #   return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, False)
    
class AttentiveHead(BaseHead):
  def __init__(self,input_dim,num_classes,num_heads,dropout):
    model = AttentiveProbe(input_dim=input_dim,num_classes=num_classes,num_heads=num_heads,dropout=dropout)
    is_classification = True if num_classes > 1 else False
    super().__init__(model,is_classification)
  # def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
  #   return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, True)

class GRUHead(BaseHead):
  def __init__(self, input_size, hidden_size, num_layers, dropout, output_size,layer_norm):
    model = GRUProbe(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=output_size,layer_norm=layer_norm)
    is_classification = True if output_size > 1 else False
    super().__init__(model,is_classification)
  # def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
  #   return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, False)
    
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
      out_padded,list_length = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

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
  def __init__(self,input_dim,num_classes,num_heads,dropout):
    super().__init__()
    self.query = nn.Parameter(torch.randn(1, input_dim)) # [1, emb_dim]
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.attn = nn.MultiheadAttention(embed_dim=input_dim,
                                      num_heads=num_heads,
                                      dropout=dropout,
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
    # self.dropout = nn.Dropout(dropout)
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
  
  

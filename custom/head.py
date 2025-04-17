import time
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
import custom.helper as helper
from custom.dataset import customSampler
from torch.nn.utils.rnn import pack_padded_sequence,pack_padded_sequence
from custom.dataset import get_dataset_and_loader
import tqdm
import copy
from jepa.src.models.attentive_pooler import AttentiveClassifier as AttentiveClassifierJEPA
from jepa.evals.video_classification_frozen import eval as jepa_eval 
from jepa.src.models.utils import pos_embs

import pickle
# import wandb

class BaseHead(nn.Module):
  def __init__(self, model,is_classification):
    super(BaseHead, self).__init__()
    self.model = model
    self.is_classification = is_classification
  
  def log_performance(self, stage, loss, precision,epoch=-1,num_epochs=-1,list_grad_norm=None,wds=None,lrs=None):
      if epoch>-1:
        print(f'Epoch [{epoch}/{num_epochs}]')
      print(f' {stage}')
      print(f'  Loss             : {loss:.4f}')
      print(f'  Precision        : {precision:.4f}')
      if stage == 'Train' and list_grad_norm:
        list_grad_norm = np.array(list_grad_norm)
        print(f'  Grad norm        ')
        print(f'    Mean           : {list_grad_norm.mean():.4f}+-{list_grad_norm.std():.4f}')
        # print(f'    Max            : {list_grad_norm.max():.4f}')
        # print(f'    Min            : {list_grad_norm.min():.4f}')
      if wds:
        if isinstance(wds,list):
          print(f'  Weight decay     : {[round(wd,8) for wd in wds]}')
        else:
          print(f'  Weight decay     : {wds:.8f}')
      if lrs:
        if isinstance(lrs,list):
          print(f'  Learning rate    : {[round(lr,8) for lr in lrs]}')
        else:
          print(f'  Learning rate    : {lrs:.8f}')  
      free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
      total_gpu_mem = total_gpu_mem / 1024 ** 3
      free_gpu_mem = free_gpu_mem / 1024 ** 3
      print(f'  GPU memory free  : {free_gpu_mem:.2f} GB')
      
  def start_train(self, num_epochs, criterion, optimizer,lr, saving_path, train_csv_path, val_csv_path ,batch_size,dataset_type,
                  round_output_loss, shuffle_training_batch, regularization_lambda_L1,concatenate_temp_dim,clip_grad_norm,
                  early_stopping, key_for_early_stopping,enable_scheduler,root_folder_features,init_network,backbone_dict,n_workers,label_smooth,
                  regularization_lambda_L2):
    device = 'cuda'
    self.model.to(device)
    if init_network:
      self.model._initialize_weights(init_type=init_network)
    
    train_dataset, train_loader = get_dataset_and_loader(batch_size=batch_size,
                                                              csv_path=train_csv_path,
                                                              root_folder_features=root_folder_features,
                                                              shuffle_training_batch=shuffle_training_batch,
                                                              is_training=True,
                                                              concatenate_temporal=concatenate_temp_dim,
                                                              dataset_type=dataset_type,
                                                              backbone_dict=backbone_dict,
                                                              model=self.model,
                                                              label_smooth=label_smooth,
                                                              n_workers=n_workers)
    val_dataset, val_loader = get_dataset_and_loader(batch_size=batch_size,
                                                          csv_path=val_csv_path,
                                                          root_folder_features=root_folder_features,
                                                          shuffle_training_batch=False,
                                                          is_training=False,
                                                          concatenate_temporal=concatenate_temp_dim,
                                                          dataset_type=dataset_type,
                                                          backbone_dict=backbone_dict,
                                                          model=self.model,
                                                          label_smooth=label_smooth,
                                                          n_workers=n_workers)
    
    if isinstance(self.model,AttentiveClassifierJEPA) and enable_scheduler:
      optimizer, _, scheduler, wd_scheduler = jepa_eval.init_opt(
          classifier=self.model,
          start_lr=lr,
          ref_lr=lr,
          iterations_per_epoch=len(train_loader),
          warmup=0.0,
          wd=regularization_lambda_L2,
          final_wd=regularization_lambda_L2,
          num_epochs=num_epochs,
          use_bfloat16=False)
    else:
      if enable_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=num_epochs,
                                                               eta_min=1e-7,
                                                               last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer=optimizer, 
        #                                                         mode='min' if key_for_early_stopping == 'val_loss' else 'max',
        #                                                         cooldown=5,
        #                                                         patience=10,
        #                                                         factor=0.1,
        #                                                         verbose=True,
        #                                                         threshold=1e-4,
        #                                                         threshold_mode='abs',
        #                                                         min_lr=1e-7)
        wd_scheduler = None
      else:
        scheduler = None
        wd_scheduler = None
      optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=regularization_lambda_L2)
    
    # train_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    train_unique_classes = train_dataset.get_unique_classes()
    train_unique_subjects = train_dataset.get_unique_subjects()
    # val_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    val_unique_classes = val_dataset.get_unique_classes()
    val_unique_subjects = val_dataset.get_unique_subjects()
    list_train_losses = []
    list_train_losses_per_class = []
    list_train_accuracy_per_class = []
    list_train_accuracy_per_subject = []
    list_train_losses_per_subject = []
    list_train_confusion_matricies = []
    list_val_losses = []
    list_val_losses_per_class = []
    list_val_accuracy_per_class = []
    list_val_losses_per_subject = []
    list_val_accuracy_per_subject = []
    list_val_confusion_matricies = []
    list_train_macro_accuracy = []
    list_val_macro_accuracy = []
    total_norm_epoch = []
    list_lrs = []
    list_wds = []
    # list_list_samples = []
    # list_list_y = []
    # best_epoch = False
    list_train_confidence_prediction_right_mean = []
    list_train_confidence_prediction_wrong_mean = []
    list_val_confidence_prediction_right_mean = []
    list_val_confidence_prediction_wrong_mean = []
    list_train_confidence_prediction_right_std = []
    list_train_confidence_prediction_wrong_std = []
    list_val_confidence_prediction_right_std = []
    list_val_confidence_prediction_wrong_std = []
    epochs_gradient_per_module = {}
    early_stopping.reset()
    for epoch in range(num_epochs):
      self.model.train()
      
      if scheduler:
        scheduler.step()
      if wd_scheduler:
        wd_scheduler.step()
      lrs, wds = tools.get_lr_and_weight_decay(optimizer)
      list_lrs.append(lrs)
      list_wds.append(wds)
      class_loss = np.zeros(train_unique_classes.shape[0])
      class_accuracy = np.zeros(train_unique_classes.shape[0])
      subject_loss = np.zeros(train_unique_subjects.shape[0])
      subject_accuracy = np.zeros(train_unique_subjects.shape[0])
      train_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=train_unique_classes.shape[0]+1)
      train_loss = 0.0
      subject_count_batch = np.zeros(train_unique_subjects.shape[0])
      count_batch=0
      total_norm_epoch.append([])
      batch_train_confidence_prediction_right_mean = []
      batch_train_confidence_prediction_right_std = []
      batch_train_confidence_prediction_wrong_mean = []
      batch_train_confidence_prediction_wrong_std = []
      batch_dict_gradient_per_module = {}

      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(train_loader,total=len(train_loader),desc=f'Train {epoch}/{num_epochs}'):
        tmp = np.isin(train_unique_subjects,batch_subjects)
        subject_count_batch[tmp] += 1
        # list_samples.append(sample_id)
        # list_y.append(batch_y)
        # [list_samples.append(sample.item()) for sample in sample_id]
        # [list_y.append(y.item()) for y in batch_y]
        batch_y = batch_y.to(device)
        dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
        optimizer.zero_grad()
        outputs = self.model(**dict_batch_X) # input [batch, seq_len, emb_dim]
        if round_output_loss:
          outputs = torch.round(outputs)
        loss = criterion(outputs, batch_y)
        if regularization_lambda_L1 > 0:
          # Sum absolute values of all trainable parameters except biases
          l1_norm = sum(param.abs().sum() for name,param in self.model.named_parameters() if param.requires_grad and 'bias' not in name)
          l1_penalty = regularization_lambda_L1 * l1_norm
          loss = loss + l1_penalty 
        # if regularization_lambda_L2 > 0:
        #   l2_norm = sum(param.pow(2).sum() for name,param in self.model.named_parameters() if param.requires_grad and 'bias' not in name)
        #   l2_penalty= regularization_lambda_L2 * l2_norm
        #   loss = loss + l2_penalty
        loss.backward()        
        if clip_grad_norm:
          total_norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm).detach().cpu() #  torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
          # total_norm = torch.tensor(total_norm)
        else:
          total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100).detach().cpu()
        
        total_norm_epoch[epoch].append(total_norm.item())
        # remove dict_batch_X from GPU
        optimizer.step()
        # outputs = torch.argmax(outputs, dim=1)
        train_loss += loss.item()
        # Compute loss per class and subject
        # if epoch % helper.saving_rate_training_logs == 0:
        tools.compute_loss_per_class_(batch_y=batch_y,
                                    class_loss=class_loss,
                                    unique_train_val_classes=train_unique_classes,
                                    outputs=outputs,
                                    class_accuracy=class_accuracy,
                                    criterion=criterion)
        tools.compute_loss_per_subject_(batch_subjects=batch_subjects,
                                      criterion=criterion,
                                      batch_y=batch_y,
                                      outputs=outputs,
                                      subject_loss=subject_loss,
                                      subject_accuracy=subject_accuracy,
                                      unique_train_val_subjects=train_unique_subjects)

        tools.compute_confidence_predictions_(list_prediction_right_mean=batch_train_confidence_prediction_right_mean,
                                              list_prediction_right_std=batch_train_confidence_prediction_right_std,
                                              list_prediction_wrong_mean=batch_train_confidence_prediction_wrong_mean,
                                              list_prediction_wrong_std=batch_train_confidence_prediction_wrong_std,
                                              gt=batch_y,
                                              outputs=outputs)
        self.log_gradient_per_module(batch_dict_gradient_per_module)
        
        count_batch+=1
        if self.is_classification:
          predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        else:
          predictions = torch.round(outputs).detach().cpu() # round to the nearest even number if 0.5
          mask = torch.isin(predictions, torch.tensor(train_unique_classes))
          predictions[~mask] = self.model.num_classes # put prediction in the last class (bad_classified)
          
        batch_y = batch_y.detach().cpu()
        if batch_y.dim() > 1:
          batch_y = torch.argmax(batch_y, dim=1).reshape(-1)
        train_confusion_matrix.update(predictions, batch_y)
        
      # tools.check_sample_id_y_from_csv(list_samples,list_y,train_csv_path)
      # list_list_samples.append(list_samples)
      # list_list_y.append(list_y)
      # frobenius_norm = self.calculate_frobenius_norm()

      dict_eval = self.evaluate(criterion=criterion,is_test=False,
                                unique_val_classes=val_unique_classes,
                                unique_val_subjects=val_unique_subjects,
                                val_loader=val_loader)
            
      if epoch == 0 or (dict_eval[key_for_early_stopping] < best_test_loss if key_for_early_stopping == 'val_loss' else dict_eval[key_for_early_stopping] > best_test_loss):
        best_test_loss = dict_eval[key_for_early_stopping]
        best_model_state = copy.deepcopy(self.model.state_dict())
        best_model_state = {key: value.cpu() for key, value in best_model_state.items()}
        best_model_epoch = epoch
        best_epoch = True
      
      list_train_losses.append(train_loss / len(train_loader))
      list_val_losses.append(dict_eval['val_loss'])
      list_train_confidence_prediction_right_mean.append(np.mean(batch_train_confidence_prediction_right_mean))
      list_train_confidence_prediction_wrong_mean.append(np.mean(batch_train_confidence_prediction_wrong_mean))
      list_train_confidence_prediction_right_std.append(np.std(batch_train_confidence_prediction_right_mean))
      list_train_confidence_prediction_wrong_std.append(np.std(batch_train_confidence_prediction_wrong_mean))
      if helper.LOG_GRADIENT_PER_MODULE:
        print('Logging gradients...')
        for k,v in batch_dict_gradient_per_module.items():
          if k not in epochs_gradient_per_module:
            epochs_gradient_per_module[k] = []
          epochs_gradient_per_module[k].append({'mean':np.mean(v),
                                                'std':np.std(v),})
      
      if epoch % helper.saving_rate_training_logs == 0 or best_epoch:
        list_train_confusion_matricies.append(train_confusion_matrix)
        list_train_losses_per_class.append(class_loss / len(train_loader))
        list_train_losses_per_subject.append(subject_loss / subject_count_batch)
        list_train_accuracy_per_class.append(class_accuracy / len(train_loader))
        list_train_accuracy_per_subject.append(subject_accuracy / subject_count_batch)
        list_val_losses_per_class.append(dict_eval['val_loss_per_class'])
        list_val_losses_per_subject.append(dict_eval['val_loss_per_subject'])
        list_val_accuracy_per_class.append(dict_eval['val_accuracy_per_class'])
        list_val_accuracy_per_subject.append(dict_eval['val_accuracy_per_subject'])
        
      list_val_confidence_prediction_right_mean.append(dict_eval['val_prediction_confidence_right_mean'])
      list_val_confidence_prediction_wrong_mean.append(dict_eval['val_prediction_confidence_wrong_mean'])
      list_val_confidence_prediction_right_std.append(dict_eval['val_prediction_confidence_right_std'])
      list_val_confidence_prediction_wrong_std.append(dict_eval['val_prediction_confidence_wrong_std'])
      list_val_confusion_matricies.append(dict_eval['val_confusion_matrix'])
      
      train_confusion_matrix.compute()
      train_dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=train_confusion_matrix,list_real_classes=train_unique_classes)
      list_train_macro_accuracy.append(train_dict_precision_recall['macro_precision'])
      list_val_macro_accuracy.append(dict_eval['val_macro_precision'])
      # log performance
      self.log_performance(stage='Train', num_epochs=num_epochs, loss=list_train_losses[-1], precision=train_dict_precision_recall['macro_precision'],epoch=epoch,list_grad_norm=total_norm_epoch[epoch],
                           lrs=lrs,wds=wds)
      self.log_performance(stage='Val', num_epochs=num_epochs, loss=dict_eval['val_loss'], precision=dict_eval['val_macro_precision'])
        
      if early_stopping(dict_eval[key_for_early_stopping]):
        break
      
    # if saving_path:
    #     print('Load and save best model for next steps...')
    #     torch.save(best_model_state, os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth'))
    #     print(f"Best model weights saved to {os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pth')}")
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
      'list_val_accuracy_per_subject': list_val_accuracy_per_subject,
      'list_train_accuracy_per_subject': list_train_accuracy_per_subject,
      'list_val_accuracy_per_class': list_val_accuracy_per_class,
      'list_train_accuracy_per_class': list_train_accuracy_per_class,
      'list_train_confidence_prediction_right_mean': list_train_confidence_prediction_right_mean,
      'list_train_confidence_prediction_wrong_mean': list_train_confidence_prediction_wrong_mean,
      'list_train_confidence_prediction_right_std': list_train_confidence_prediction_right_std,
      'list_train_confidence_prediction_wrong_std': list_train_confidence_prediction_wrong_std,
      'list_val_confidence_prediction_right_mean': list_val_confidence_prediction_right_mean,
      'list_val_confidence_prediction_wrong_mean': list_val_confidence_prediction_wrong_mean,
      'list_val_confidence_prediction_right_std': list_val_confidence_prediction_right_std,
      'list_val_confidence_prediction_wrong_std': list_val_confidence_prediction_wrong_std,
      'epochs_gradient_per_module': epochs_gradient_per_module,
      # 'train_accuracy_per_class': train_accuracy_per_class,
      # 'test_accuracy_per_class': test_accuracy_per_class,
      'train_confusion_matricies': list_train_confusion_matricies,
      'val_confusion_matricies': list_val_confusion_matricies,
      'best_model_idx': best_model_epoch,
      'best_model_state': best_model_state,
      'list_train_macro_accuracy': list_train_macro_accuracy,
      'list_val_macro_accuracy': list_val_macro_accuracy,
      'epochs': epoch,
      'list_mean_total_norm_epoch': np.array(total_norm_epoch).mean(axis=1),
      'list_std_total_norm_epoch': np.array(total_norm_epoch).std(axis=1),
      'list_max_total_norm_epoch': np.array(total_norm_epoch).max(axis=1),
      'list_min_total_norm_epoch': np.array(total_norm_epoch).min(axis=1),
      'list_lrs': list_lrs,
      'list_wds': list_wds,
      'optimizer': optimizer.state_dict()['param_groups'],
      'wd_scheduler': wd_scheduler.get_config() if wd_scheduler else None,
      'scheduler': scheduler.state_dict() if scheduler else None,
      # 'list_samples': list_list_samples,
      # 'list_y': list_list_y
    }

  def evaluate(self, val_loader, criterion, unique_val_subjects, unique_val_classes, is_test,save_log=True):
    # unique_train_val_classes is only for eval but kept the name for compatibility
    device = 'cuda'
    count = 0
    self.model.to(device)
    self.model.eval()
    with torch.no_grad():
      val_loss = 0.0
      loss_per_class = np.zeros(self.model.num_classes)
      accuracy_per_class = np.zeros(self.model.num_classes)
      subject_loss = np.zeros(unique_val_subjects.shape[0])
      accuracy_per_subject = np.zeros(unique_val_subjects.shape[0])
      val_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=self.model.num_classes+1) # last class is for bad_classified in regression
      subject_batch_count = np.zeros(unique_val_subjects.shape[0])
      batch_confidence_prediction_right_mean = []
      batch_confidence_prediction_wrong_mean = []
      batch_confidence_prediction_right_std = []
      batch_confidence_prediction_wrong_std = []
      for dict_batch_X, batch_y, batch_subjects,_ in tqdm.tqdm(val_loader,total=len(val_loader),desc='Validation' if not is_test else 'Test'):
        tmp = np.isin(unique_val_subjects,batch_subjects)
        dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
        batch_y = batch_y.to(device)
        subject_batch_count[tmp] += 1
        outputs = self.model(**dict_batch_X)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        if save_log:
          tools.compute_loss_per_class_(batch_y=batch_y, class_loss=loss_per_class, unique_train_val_classes=unique_val_classes,
                                      outputs=outputs, criterion=criterion,class_accuracy=accuracy_per_class)
          tools.compute_loss_per_subject_(batch_subjects=batch_subjects, criterion=criterion, batch_y=batch_y, outputs=outputs,
                                        subject_loss=subject_loss, unique_train_val_subjects=unique_val_subjects,subject_accuracy=accuracy_per_subject)
          tools.compute_confidence_predictions_(list_prediction_right_mean=batch_confidence_prediction_right_mean,
                                                list_prediction_wrong_mean=batch_confidence_prediction_wrong_mean,
                                                list_prediction_right_std=batch_confidence_prediction_right_std,
                                                list_prediction_wrong_std=batch_confidence_prediction_wrong_std,
                                                gt=batch_y, outputs=outputs)
        if self.is_classification:
          predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        else:
          predictions = torch.round(outputs).detach().cpu() # round to the nearest even number if 0.5
          mask = torch.isin(predictions, torch.tensor(unique_val_classes))
          predictions[~mask] = self.model.num_classes # put prediction in the last class (bad_classified)
          
        batch_y = batch_y.detach().cpu()
        if batch_y.dim() > 1:
          batch_y = torch.argmax(batch_y, dim=1).reshape(-1)
        val_confusion_matricies.update(predictions, batch_y)
        count += 1
      
      val_confusion_matricies.compute()
      val_loss = val_loss / len(val_loader)
      if save_log:
        loss_per_class = loss_per_class / len(val_loader)
        accuracy_per_class = accuracy_per_class / len(val_loader)
        subject_loss = subject_loss / subject_batch_count
        accuracy_per_subject = accuracy_per_subject / subject_batch_count
      dict_precision_recall = tools.get_accuracy_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                       list_real_classes=unique_val_classes)
      if is_test:
        self.log_performance(stage='Test', loss=val_loss, precision=dict_precision_recall['macro_precision'])
        return {
          'test_loss': val_loss,
          'test_loss_per_class': loss_per_class,
          'test_loss_per_subject': subject_loss,
          'test_accuracy_per_class': accuracy_per_class,
          'test_accuracy_per_subject': accuracy_per_subject,
          'test_macro_precision': dict_precision_recall["macro_precision"],
          'test_confusion_matrix': val_confusion_matricies,
          'test_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean),
          'test_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean),
          'test_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean),
          'test_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean),
          'dict_precision_recall': dict_precision_recall
        }
      else:
        return {
          'val_loss': val_loss,
          'val_loss_per_class': loss_per_class,
          'val_loss_per_subject': subject_loss,
          'val_accuracy_per_class': accuracy_per_class,
          'val_accuracy_per_subject': accuracy_per_subject,
          'val_macro_precision': dict_precision_recall["macro_precision"],
          'val_confusion_matrix': val_confusion_matricies,
          'val_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean),
          'val_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean),
          'val_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean),
          'val_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean),
          'dict_precision_recall': dict_precision_recall
        }      

  def log_and_save_gradients(model, epoch, batch_idx, saving_path):
    """
    Logs and saves the gradients of the model parameters to a .pkl file.

    Args:
        model (torch.nn.Module): The model whose gradients are to be logged.
        saving_path (str): The directory where the gradient file will be saved.
        epoch (int): The current epoch number.
        batch_idx (int): The current batch index.
    """
    gradients = {}
    for name, param in model.named_parameters():
      if param.grad is not None:
        # Detach the gradient and move it to CPU for saving
        gradients[name] = param.grad.detach().cpu().clone()

        # Optionally, print some stats
        print(f"{name} - grad mean: {gradients[name].mean().item():.4f}, std: {gradients[name].std().item():.4f}")
    
    # Create the file name with epoch and batch index
    grad_filename = os.path.join(saving_path, f'gradients_epoch{epoch}_batch{batch_idx}.pkl')
    with open(grad_filename, 'wb') as f:
        pickle.dump(gradients, f)

  def load_state_weights(self, path):
    self.model.load_state_dict(torch.load(path,weights_only=True))
    print(f'Model weights loaded from {path}')
    
  def log_gradient_per_module(self,dict_gradient_per_module):
    for name, param in self.model.named_parameters():
      if param.grad is not None:
        if name not in dict_gradient_per_module:
          dict_gradient_per_module[name] = []
        dict_gradient_per_module[name].append(param.grad.detach().cpu().clone())
    return dict_gradient_per_module
   
  def calculate_frobenius_norm(self):
    norm = 0.0
    for _, param in self.model.named_parameters():
      if param.grad is not None:
        norm += torch.linalg.vector_norm(param.grad, ord=2)
        # norm += torch.norm(param.grad, p='fro').item()
    return norm
  
  
class LinearHead(BaseHead):
  def __init__(self, input_dim, num_classes, dim_reduction):
    model = LinearProbe(input_dim=input_dim, num_classes=num_classes,dim_reduction=dim_reduction)
    is_classification = True if num_classes > 1 else False
    super().__init__(model,is_classification)
  # def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
  #   return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, False)
 
class AttentiveHeadJEPA(BaseHead):
  def __init__(self,
      embed_dim=768,
      num_heads=12,
      num_cross_heads=12,
      mlp_ratio=4.0,
      depth=1,
      norm_layer=nn.LayerNorm,
      init_std=0.02,
      qkv_bias=True,
      dropout=0.0,
      residual_dropout=0.0,
      attn_dropout=0.0,
      num_classes=5,
      pos_enc=False,
      use_sdpa=False,
      grid_size_pos=None,
      cross_block_after_transformers=True,
      complete_block=True):
    model = AttentiveClassifierJEPA(embed_dim=embed_dim,
                                              num_heads=num_heads,
                                              num_cross_heads=num_cross_heads,
                                              mlp_ratio=mlp_ratio,
                                              depth=depth,
                                              norm_layer=norm_layer,
                                              init_std=init_std,
                                              qkv_bias=qkv_bias,
                                              num_classes=num_classes,
                                              dropout_mlp=dropout,
                                              attn_dropout=attn_dropout,
                                              residual_dropout=residual_dropout,
                                              pos_enc=pos_enc,
                                              use_sdpa=use_sdpa,
                                              grid_size_pos=grid_size_pos,
                                              cross_block_after_transformers=cross_block_after_transformers,
                                              complete_block=complete_block)
    super().__init__(model,is_classification=True)
  

      
class AttentiveHead(BaseHead):
  def __init__(self,input_dim,num_classes,num_heads,dropout,pos_enc):
    model = AttentiveProbe(input_dim=input_dim,num_classes=num_classes,num_heads=num_heads,dropout=dropout,pos_enc=pos_enc)
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
    self.batchNorm = nn.BatchNorm1d(input_size)
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
  def __init__(self,input_dim,num_classes,num_heads,dropout,pos_enc=False):
    super().__init__()
    self.query = nn.Parameter(torch.ones(1, input_dim)) # [1, emb_dim]
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.num_heads = num_heads
    self.pos_enc = pos_enc
    self.attn = nn.MultiheadAttention(embed_dim=input_dim,
                                      num_heads=num_heads,
                                      dropout=dropout,
                                      batch_first=True # [batch_size, seq_len, emb_dim]
                                      )
    self.linear = nn.Linear(input_dim, num_classes)
    self.pos_enc_tensor = None
    self._initialize_weights()
    
  def forward(self, x, key_padding_mask=None):
    # x: [batch_size, seq_len, emb_dim]
    # key_padding_mask: [batch_size, seq_len]
    q = self.query.unsqueeze(0).expand(x.shape[0], -1, -1) # [batch_size, 1, emb_dim]
    # sum_key_padding = torch.sum(key_padding_mask, dim=1) # [batch_size]
    if self.pos_enc:
      if self.pos_enc_tensor is None or self.pos_enc_tensor.shape[0] != x.size(1):
        self.pos_enc_tensor = pos_embs.get_1d_sincos_pos_embed(grid_size=x.size(1), embed_dim=x.size(2), device=x.device.type)
      x = x + self.pos_enc_tensor
    attn_output,_ = self.attn(q, x, x, key_padding_mask=key_padding_mask) # [batch_size, 1, emb_dim]
    pooled = attn_output.squeeze(1) # [batch_size, emb_dim]
    logits = self.linear(pooled)
    return logits
  
  def _initialize_weights(self,init_type='default'):
    if init_type == 'default':
      nn.init.trunc_normal_(self.linear.weight, std=0.02)
      self.linear.reset_parameters()
      self.attn._reset_parameters() # Xavier uniform initialization
    else:
      raise NotImplementedError(f"Unknown initialization type: {init_type}. Can be 'default'")
    
class LinearProbe(nn.Module):
  def __init__(self,dim_reduction,input_dim, num_classes):
    super().__init__()
    self.linear = nn.Linear(input_dim, num_classes)
    # self.dropout = nn.Dropout(dropout)
    self.dim_reduction = dim_reduction
    self.num_classes = num_classes
    
  def forward(self, x):
    # The mean over the sequence is applied in dataset class
    x = x.data # [batch,T,S,S,emb_dim]
    if self.dim_reduction is not None:
      x = x.mean(dim=self.dim_reduction)
    x=x.reshape(x.shape[0], -1)
    logits = self.linear(x)
    return logits
  
  def _initialize_weights(self,init_type='default'):
    if init_type == 'default':
      self.linear.reset_parameters()
    else:
      raise ValueError(f"Unknown initialization type: {init_type}. Can be 'default'")
  
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
  
  

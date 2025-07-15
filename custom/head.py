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
from custom.dataset import customBatchSampler
from torch.nn.utils.rnn import pack_padded_sequence,pack_padded_sequence
from custom.dataset import get_dataset_and_loader
import tqdm
import copy
from jepa.src.models.attentive_pooler import AttentiveClassifier as AttentiveClassifierJEPA
from jepa.evals.video_classification_frozen import eval as jepa_eval 
from jepa.src.models.utils import pos_embs
from optuna.exceptions import TrialPruned
import pickle
import matplotlib.pyplot as plt
# import tracemalloc
# import wandb
from torch.amp import GradScaler, autocast
import threading
import jepa.src.models.attentive_pooler as jepa_attentive_pooler
import jepa.src.models.utils.modules as jepa_modules
from jepa.src.utils.tensors import trunc_normal_
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import proba_to_label
import custom.backbone as custom_backbone

class BaseHead(nn.Module):
  def __init__(self, model,is_classification):
    super(BaseHead, self).__init__()
    # self.model = model
    self.is_classification = is_classification
      
  def log_performance(self, stage, loss, accuracy,epoch=-1,dict_kwarg=None,num_epochs=-1,list_grad_norm=None,wds=None,lrs=None):
      if epoch>-1:
        print(f'Epoch [{epoch}/{num_epochs}]')
      print(f' {stage}')
      print(f'  Loss             : {loss:.4f}')
      print(f'  Accuracy         : {accuracy:.4f}')
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
      if dict_kwarg:
        for k,v in dict_kwarg.items():
          print(f'  {k}              : {v:.4f}' if isinstance(v,float) else f'  {k}              : {v}') 
      free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
      total_gpu_mem = total_gpu_mem / 1024 ** 3
      free_gpu_mem = free_gpu_mem / 1024 ** 3
      print(f'  GPU memory free  : {free_gpu_mem:.2f} GB')
      
      
  def check_and_apply_latent_augmentation_(self,features,list_sample_id):
    # polarity inversion + gaussian noise
    for i in range(features.shape[0]):
      if helper.is_latent_basic_augmentation(list_sample_id[i]): 
        gaussian_noise = torch.randn_like(features[i]) * 0.1
        features[i] = (-1 * features[i]) + gaussian_noise 
      
      # random masking patches
      elif helper.is_latent_masking_augmentation(list_sample_id[i]):
        B,T,S,S,C = features[i].shape
        mask_grid = torch.rand(S,S) < 0.1 # 10% of the grid
        mask_grid = mask_grid.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        mask_grid = mask_grid.expand(B,T,S,S,C) # [B,T,S,S,C]
        features[i] = features[i].masked_fill(mask_grid, 0.0) # set to zero the masked values
    
  def start_train(self, num_epochs, criterion, optimizer,lr, saving_path, train_csv_path, val_csv_path ,batch_size,dataset_type,
                  round_output_loss, shuffle_training_batch, regularization_lambda_L1,concatenate_temp_dim,clip_grad_norm,
                  early_stopping, key_for_early_stopping,enable_scheduler,root_folder_features,init_network,backbone_dict,n_workers,label_smooth,
                  regularization_lambda_L2,trial,enable_optuna_pruning,prefetch_factor,soft_labels,is_coral_loss):
    
    # Generate Thread for smooth stopping 
    stop_event = threading.Event()
    def _wait_for_s():
      cmd = input(">>> Type 's' + Enter to stop after this epoch: ")
      if cmd.strip().lower() == 's':
        stop_event.set()
        print("Stopping training after this epoch...")
    threading.Thread(target=_wait_for_s, daemon=True).start()
    
    # Start training 
    device = 'cuda'
    self.to(device)
    # if init_network:
    self._initialize_weights(init_type=init_network)
    
    train_dataset, train_loader = get_dataset_and_loader(batch_size=batch_size,
                                                              csv_path=train_csv_path,
                                                              root_folder_features=root_folder_features,
                                                              shuffle_training_batch=shuffle_training_batch,
                                                              is_training=True,
                                                              concatenate_temporal=concatenate_temp_dim,
                                                              dataset_type=dataset_type,
                                                              prefetch_factor=prefetch_factor,
                                                              backbone_dict=backbone_dict,
                                                              model=self,
                                                              is_coral_loss=is_coral_loss,
                                                              soft_labels=soft_labels,
                                                              label_smooth=label_smooth,
                                                              n_workers=n_workers)
    if val_csv_path is not None:
      val_dataset, val_loader = get_dataset_and_loader(batch_size=batch_size,
                                                            csv_path=val_csv_path,
                                                            root_folder_features=root_folder_features,
                                                            shuffle_training_batch=False,
                                                            is_training=False,
                                                            concatenate_temporal=concatenate_temp_dim,
                                                            dataset_type=dataset_type,
                                                            prefetch_factor=prefetch_factor,
                                                            backbone_dict=backbone_dict,
                                                            model=self, 
                                                            is_coral_loss=is_coral_loss,
                                                            soft_labels=soft_labels,
                                                            label_smooth=label_smooth,
                                                            n_workers=n_workers)
    
    if isinstance(self,AttentiveClassifierJEPA) and enable_scheduler: 
      optimizer, _, scheduler, wd_scheduler = jepa_eval.init_opt( 
          classifier=self,
          start_lr=lr,
          ref_lr=lr,
          iterations_per_epoch=1, # 1 because I update the optimizer every epoch
          warmup=0.0,
          wd=regularization_lambda_L2,
          final_wd=regularization_lambda_L2,
          num_epochs=num_epochs, 
          use_bfloat16=False)
    else:
      optimizer = optimizer(self.parameters(), lr=lr, weight_decay=regularization_lambda_L2) 
      if enable_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=num_epochs,
                                                               eta_min=1e-7,
                                                               last_epoch=-1)
        wd_scheduler = None
      else:
        scheduler = None
        wd_scheduler = None
    
    # train_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    train_unique_classes = torch.tensor(train_dataset.get_unique_classes())
    train_unique_subjects = torch.tensor(train_dataset.get_unique_subjects())
    # val_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    # if val_csv_path is not None:
    val_unique_classes = torch.tensor(val_dataset.get_unique_classes()) if val_csv_path is not None else torch.tensor([])
    val_unique_subjects = torch.tensor(val_dataset.get_unique_subjects()) if val_csv_path is not None else torch.tensor([])
    if helper.LOG_HISTORY_SAMPLE and torch.min(train_unique_classes)>=0 and torch.max(train_unique_classes)<=255 and torch.min(val_unique_classes)>=0 and torch.max(val_unique_classes)<=255:
      list_train_sample = train_dataset.get_all_sample_ids()
      history_train_sample_predictions = {id: torch.zeros(num_epochs, dtype=torch.uint8) for id in list_train_sample}
      # if val_csv_path is not None:
      list_val_sample = val_dataset.get_all_sample_ids() if val_csv_path is not None else None
      history_val_sample_predictions = {id: torch.zeros(num_epochs, dtype=torch.uint8) for id in list_val_sample} if val_csv_path is not None else None
      print('Size history_train_sample_predictions:')
      tools.print_dict_size(history_train_sample_predictions)
    else:
      if helper.LOG_HISTORY_SAMPLE:
        print(f'\nWarning: The model will not save the predictions of the samples in the dataset because the classes are not in the range [0,255]\n')
      else:
        print(f'\nWarning: The model will not save the predictions of the samples in the dataset because LOG_HISTORY_SAMPLE is set to False\n')
      history_train_sample_predictions = None
      history_val_sample_predictions = None
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
    list_train_performance_metric = []
    list_val_performance_metric = []
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
    # list_memory_snap= []
    # tracemalloc.start()
    # list_memory_snap.append(tracemalloc.take_snapshot())
    # accuracy_metric = torchmetrics.Accuracy(task='multiclass',num_classes=self.model.num_classes,average='none').to(device)
    # precision_metric = torchmetrics.Precision(task='multiclass',num_classes=self.model.num_classes,average='none').to(device)
    print(f'\nNumber of parameters for this head:')
    print(f'  Trainable: {((sum(p.numel() for p in self.parameters() if p.requires_grad))/1e6):.4f} M')
    print(f'  Total    : {sum(p.numel() for p in self.parameters())/1e6:.4f} M')
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(f'  {name}: {param.numel()}')
      else:
        print(name)

    metric_for_stopping = "_".join(key_for_early_stopping.split('_')[1:]) # ex: val_accuracy -> accuracy. the second part must be metric from tools.evaluate_classification_from_confusion_matrix
    amp_dtype = torch.bfloat16 if helper.AMP_DTYPE == 'bfloat16' else torch.float16
    enable_autocast = helper.AMP_ENABLED and amp_dtype == torch.bfloat16 # Use autocast for mixed precision training
    enable_scaler = helper.AMP_ENABLED and amp_dtype in [torch.float16] # Use GradScaler for mixed precision training
    scaler = GradScaler(device=device, enabled=enable_scaler)
    
    if enable_scaler:
      print(f'Using GradScaler for mixed precision training. Dtype {amp_dtype}')
    else:
      print(f'Using standard precision training without GradScaler')
    if enable_autocast:
      print(f'Using autocast for mixed precision training with {amp_dtype} dtype')
    else:
      print(f'Using standard precision training without autocast')
      
    for epoch in range(num_epochs):
      start_epoch = time.time()
      self.train() 
      lrs, wds = tools.get_lr_and_weight_decay(optimizer)
      list_lrs.append(lrs)
      list_wds.append(wds)
      class_loss = torch.zeros(2,train_unique_classes.shape[0])
      class_accuracy = torch.zeros(2,train_unique_classes.shape[0])
      # subject_loss = np.zeros(train_unique_subjects.shape[0])
      # subject_accuracy = np.zeros(train_unique_subjects.shape[0])
      subject_loss = torch.zeros(train_unique_subjects.shape[0])
      subject_accuracy = torch.zeros(train_unique_subjects.shape[0])
      sample_per_subject_count = torch.zeros(train_unique_subjects.shape[0])
      train_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=train_unique_classes.shape[0]+1)
      train_loss = 0.0
      # subject_count_batch = torch.zeros(train_unique_subjects.shape[0])
      count_batch=0
      total_norm_epoch.append([])
      batch_train_confidence_prediction_right_mean = []
      batch_train_confidence_prediction_right_std = []
      batch_train_confidence_prediction_wrong_mean = []
      batch_train_confidence_prediction_wrong_std = []
      batch_dict_gradient_per_module = {}
      dict_log_time = {}
      dict_log_time['pre_batch'] = dict_log_time.get('pre_batch',0) + time.time() - start_epoch
      start_load_batch = time.time()
      
      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(train_loader,total=len(train_loader),desc=f'Train {epoch}/{num_epochs}'):
        end_load_batch = time.time()
        dict_log_time['load_batch'] = dict_log_time.get('load_batch',0) + end_load_batch - start_load_batch
        time_to_count_subjects = time.time()
        tmp = torch.isin(train_unique_subjects,batch_subjects)
        _,count_sample_per_subject = torch.unique(batch_subjects, return_counts=True)
        sample_per_subject_count[tmp] += count_sample_per_subject
        dict_log_time['count_subjects'] = dict_log_time.get('count_subjects',0) + time.time() - time_to_count_subjects
        
        # Latent augmentation
        # time_for_latent_augmentation = time.time()
        # self.check_and_apply_latent_augmentation_(dict_batch_X['x'], sample_id)
        # dict_log_time['latent_augmentation'] = dict_log_time.get('latent_augmentation',0) + time.time() - time_for_latent_augmentation
        
        transfer_to_device = time.time() 
        # HuberLoss requires float32 inputs
        if batch_y.dtype == torch.int32: # if ce not has label smoothing
          if isinstance(criterion, torch.nn.HuberLoss):
            batch_y = batch_y.float()
          elif isinstance(criterion, torch.nn.CrossEntropyLoss):
            batch_y = batch_y.long()
        batch_y = batch_y.to(device)
        dict_batch_X = {key: value.to(device) if value is not None else None for key, value in dict_batch_X.items()}
        dict_log_time['transfer_to_device'] = dict_log_time.get('transfer_to_device',0) + time.time() - transfer_to_device
        optimizer.zero_grad()
        
        helper.LOG_CROSS_ATTENTION['state'] = 'train'
        dict_batch_X['list_sample_id'] = sample_id
        
        start_forward = time.time()
        with autocast(device_type=device, dtype=amp_dtype, enabled=enable_autocast): # Use autocast for mixed precision training
          outputs = self(**dict_batch_X) # input [batch, seq_len, emb_dim]
          if outputs.shape[1] == 1: # if regression I don't need to keep dim 1 
            outputs = outputs.squeeze(1)
          loss = criterion(outputs, batch_y)
          if regularization_lambda_L1 > 0:
            # Sum absolute values of all trainable parameters except biases
            l1_norm = sum(param.abs().sum() for name,param in self.named_parameters() if param.requires_grad and 'bias' not in name) 
            loss = loss + regularization_lambda_L1 * l1_norm 
        dict_log_time['forward'] = dict_log_time.get('forward',0) + time.time()-start_forward
        # dict_log_time['loss'] = dict_log_time.get('loss',0) + time.time()-start_loss
        # print(f'  Loss time: {dict_log_time['loss']-start_loss:.4f}')
        start_back = time.time()
        scaler.scale(loss).backward() # Use scaler for mixed precision training
        # loss.backward()     
        dict_log_time['backward'] = dict_log_time.get('backward',0) + time.time()-start_back
        log_gradient_start = time.time()
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        if clip_grad_norm:
          total_norm=torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm).detach().cpu() #  torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        else:
          total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 500).detach().cpu() 
        dict_log_time['log_gradient'] = dict_log_time.get('log_gradient',0) + time.time() - log_gradient_start
        total_norm_epoch[epoch].append(total_norm.item())
        start_optimizer = time.time()
        scaler.step(optimizer)  # Use scaler to step the optimizer
        scaler.update()
        dict_log_time['optimizer'] = dict_log_time.get('optimizer',0) + time.time()-start_optimizer
        train_loss += loss.item() 
        start_logs = time.time()
        if helper.LOG_PER_CLASS_AND_SUBJECT:
          if is_coral_loss:
            outputs = proba_to_label(torch.sigmoid(outputs)) # convert probabilities to labels for coral loss
            batch_y = torch.sum(batch_y, dim=1) # convert levels to labels for coral loss
          tools.compute_loss_per_class_(batch_y=batch_y,
                                      class_loss=class_loss if not is_coral_loss else None,
                                      unique_train_val_classes=train_unique_classes,
                                      outputs=outputs,
                                      class_accuracy=class_accuracy,
                                      criterion=criterion)
          tools.compute_loss_per_subject_v2_(batch_subjects=batch_subjects,
                                        criterion=criterion,
                                        batch_y=batch_y,
                                        outputs=outputs,
                                        subject_loss=subject_loss if not is_coral_loss else None,
                                        subject_accuracy=subject_accuracy,
                                        unique_train_val_subjects=train_unique_subjects)
          if not isinstance(criterion,torch.nn.L1Loss) and not isinstance(criterion,torch.nn.MSELoss) and not isinstance(criterion,torch.nn.HuberLoss) and not is_coral_loss:
            tools.compute_confidence_predictions_(list_prediction_right_mean=batch_train_confidence_prediction_right_mean,
                                                  list_prediction_right_std=batch_train_confidence_prediction_right_std,
                                                  list_prediction_wrong_mean=batch_train_confidence_prediction_wrong_mean,
                                                  list_prediction_wrong_std=batch_train_confidence_prediction_wrong_std,
                                                  gt=batch_y,
                                                  outputs=outputs)

        if helper.LOG_GRADIENT_PER_MODULE:
          self.log_gradient_per_module(batch_dict_gradient_per_module)
        
        count_batch+=1
        if self.is_classification:
          if is_coral_loss:
            predictions = outputs.detach().cpu() # outputs are already labels for coral loss
          else:
            predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        else:
          predictions = torch.copysign(torch.floor(torch.abs(outputs) + 0.5), outputs).detach().cpu().float() # to avoid the banker's rounding implemented as IEEE standard
          mask = torch.isin(predictions, train_unique_classes)
          predictions[~mask] = train_unique_classes.shape[0] # put prediction in the last class (bad_classified)
          
        batch_y = batch_y.detach().cpu()
        if batch_y.dim() > 1:
          batch_y = torch.argmax(batch_y, dim=1).reshape(-1)
        train_confusion_matrix.update(predictions, batch_y)
        
        if history_train_sample_predictions is not None: # add possibility to avoid this log
          tools.log_predictions_per_sample_(dict_log_sample=history_train_sample_predictions,
                                            tensor_sample_id=sample_id,
                                            tensor_predictions=predictions,
                                            epoch=epoch)
        # list_memory_snap.append(tracemalloc.take_snapshot())
        dict_log_time['batch_logs'] = dict_log_time.get('batch_logs',0) + time.time()-start_logs 
        # dict_log_time['batch'] = dict_log_time.get('batch',0) + time.time()-end_load_batch
        start_load_batch = time.time()

      time_eval = time.time()
      dict_eval = None
      if val_csv_path is not None:
        dict_eval = self.evaluate(criterion=criterion,is_test=False,
                                  unique_val_classes=val_unique_classes,
                                  unique_val_subjects=val_unique_subjects,
                                  val_loader=val_loader,
                                  is_coral_loss=is_coral_loss,
                                  epoch=epoch,
                                  history_val_sample_predictions=history_val_sample_predictions)
      dict_log_time['eval'] = dict_log_time.get('eval',0) + time.time()-time_eval
      # print(f'  Evaluation time: {dict_log_time["eval"]:.4f}')
      epoch_log_time = time.time()
      if epoch == 0 or helper.SAVE_LAST_EPOCH_MODEL or (dict_eval is not None and dict_eval[key_for_early_stopping] < best_eval_loss if key_for_early_stopping == 'val_loss' else dict_eval[key_for_early_stopping] > best_eval_loss):
        best_eval_loss = dict_eval[key_for_early_stopping] if dict_eval is not None else 0.0
        best_model_state = copy.deepcopy(self.state_dict())
        best_model_state = {key: value.cpu() for key, value in best_model_state.items()}
        best_model_epoch = epoch
        best_epoch = True
      
      list_train_losses.append(train_loss / len(train_loader))
      list_val_losses.append(dict_eval['val_loss'] if dict_eval is not None else 0.0)
      
      
      if helper.LOG_PER_CLASS_AND_SUBJECT:
        list_train_confidence_prediction_right_mean.append(np.mean(batch_train_confidence_prediction_right_mean) if len(batch_train_confidence_prediction_right_mean) > 0 else 0)
        list_train_confidence_prediction_wrong_mean.append(np.mean(batch_train_confidence_prediction_wrong_mean) if len(batch_train_confidence_prediction_wrong_mean) > 0 else 0)
        list_train_confidence_prediction_right_std.append(np.std(batch_train_confidence_prediction_right_mean) if len(batch_train_confidence_prediction_right_mean) > 0 else 0)
        list_train_confidence_prediction_wrong_std.append(np.std(batch_train_confidence_prediction_wrong_mean) if len(batch_train_confidence_prediction_wrong_mean) > 0 else 0)
      
      if helper.LOG_GRADIENT_PER_MODULE:
        print('Logging gradients...')
        for k,v in batch_dict_gradient_per_module.items():
          if k not in epochs_gradient_per_module:
            epochs_gradient_per_module[k] = []
          epochs_gradient_per_module[k].append({'mean':np.mean(v),
                                                'std':np.std(v),})
      
      if epoch % helper.saving_rate_training_logs == 0 or best_epoch:
        list_train_confusion_matricies.append(train_confusion_matrix)
        if helper.LOG_PER_CLASS_AND_SUBJECT:
          list_train_losses_per_class.append(class_loss[0] / class_loss[1]) # class_loss[0] is loss per class, class_loss[1] is total number of samples
          # print(f'Loss mean per class: {torch.mean(list_train_losses_per_class[-1])}')
          list_train_accuracy_per_class.append(class_accuracy[0] / class_accuracy[1]) # class_accuracy[0] is correct predictions, class_accuracy[1] is total)
          list_train_losses_per_subject.append((subject_loss / sample_per_subject_count))
          list_train_accuracy_per_subject.append(subject_accuracy / sample_per_subject_count)
          
          list_val_losses_per_class.append(dict_eval['val_loss_per_class'] if dict_eval is not None else None)
          list_val_losses_per_subject.append(dict_eval['val_loss_per_subject'] if dict_eval is not None else None)
          list_val_accuracy_per_class.append(dict_eval['val_accuracy_per_class'] if dict_eval is not None else None)
          list_val_accuracy_per_subject.append(dict_eval['val_accuracy_per_subject'] if dict_eval is not None else None)
        
      if helper.LOG_PER_CLASS_AND_SUBJECT:
        list_val_confidence_prediction_right_mean.append(dict_eval['val_prediction_confidence_right_mean'] if dict_eval is not None else 0.0)
        list_val_confidence_prediction_wrong_mean.append(dict_eval['val_prediction_confidence_wrong_mean'] if dict_eval is not None else 0.0)
        list_val_confidence_prediction_right_std.append(dict_eval['val_prediction_confidence_right_std'] if dict_eval is not None else 0.0)
        list_val_confidence_prediction_wrong_std.append(dict_eval['val_prediction_confidence_wrong_std'] if dict_eval is not None else 0.0)
        
      list_val_confusion_matricies.append(dict_eval['val_confusion_matrix'] if dict_eval is not None else None)
      
      train_confusion_matrix.compute()
      train_dict_precision_recall = tools.evaluate_classification_from_confusion_matrix(confusion_matrix=train_confusion_matrix,list_real_classes=train_unique_classes)
      list_train_performance_metric.append(train_dict_precision_recall[metric_for_stopping])
      list_val_performance_metric.append(dict_eval[key_for_early_stopping] if dict_eval is not None else 0.0)
      
      if scheduler:
        scheduler.step()
      if wd_scheduler:
        wd_scheduler.step()
      
      # log performance
      self.log_performance(stage='Train',
                           num_epochs=num_epochs,
                           loss=list_train_losses[-1], 
                           accuracy=train_dict_precision_recall[metric_for_stopping],
                           epoch=epoch,
                           list_grad_norm=total_norm_epoch[epoch],
                           lrs=lrs,
                           wds=wds)
      if dict_eval is not None:
        self.log_performance(stage='Val',
                            num_epochs=num_epochs,
                            loss=dict_eval['val_loss'],
                            #  dict_kwarg={'acc_per_class':dict_eval['val_accuracy_per_class'],},
                            accuracy=dict_eval[key_for_early_stopping])
      
      if helper.LOG_LOSS_ACCURACY and epoch > 0 and epoch % (helper.saving_rate_training_logs*2) == 0:
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        input_dict_loss_acc= {
          'list_1':list_train_losses,
          'list_2':list_train_performance_metric,
          'output_path':None,
          'title':f'Train loss, validation {metric_for_stopping}, test {key_for_early_stopping}',
          'point':None,
          'ax':ax,
          'x_label':'Epochs',
          'y_label_1':'Train loss',
          'y_label_2':f'Validation {metric_for_stopping}',
          'y_label_3':None,
          'y_lim_1':[0, 5],
          'y_lim_2':[0, 1],
          'y_lim_3':None,
          'step_ylim_1':0.25,
          'step_ylim_2':0.1,
          'step_ylim_3':None,
          'dict_to_string':None,
          'color_1':'tab:red',
          'color_2':'tab:blue',
          'color_3':None,
        }
        tools.plot_losses_and_test_new(**input_dict_loss_acc)
        fig.savefig(os.path.join(saving_path, f'loss_acc_epoch_{epoch}.png'), dpi=300)
        plt.close(fig)
        
      if dict_eval is not None and early_stopping(dict_eval[key_for_early_stopping]):
        break
      
      if enable_optuna_pruning and dict_eval is not None and trial is not None:
        trial.report(dict_eval[key_for_early_stopping], epoch)
        # trial.report(list_train_losses[-1], epoch)
        
        if trial is not None and trial.should_prune():
          raise TrialPruned()
      dict_log_time['log_epoch'] = dict_log_time.get('log_epoch',0) + time.time()-epoch_log_time
      dict_log_time['epoch'] = time.time()-start_epoch

      print(f'TIME LOGS:')
      for k,v in dict_log_time.items():
        print(f'  {k} time: {v:.4f} s')

      if helper.LOG_GRADIENT_PER_MODULE and np.mean(total_norm_epoch[epoch]) < 1e-2:
        print(f'Gradient norm is small (< 1e-1). Stopping training...')
        break
      
      if stop_event.is_set():
        print(f'Stop event is set. Stopping training...')
        break
      
    if saving_path and helper.SAVE_PTH_MODEL:
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
      'train_unique_subject_ids': train_unique_subjects.numpy(),
      'train_count_subject_ids': train_dataset.get_count_subjects(),
      'val_unique_subject_ids': val_unique_subjects.numpy(),
      'val_count_subject_ids': val_dataset.get_count_subjects() if val_csv_path is not None else None,
      'train_unique_y': train_unique_classes,
      'val_unique_y': val_unique_classes,
      'subject_ids_unique': np.unique(np.concatenate((train_unique_subjects.numpy(),val_unique_subjects.numpy()),axis=0)),
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
      'history_train_sample_predictions': history_train_sample_predictions,
      'history_val_sample_predictions': history_val_sample_predictions,
      # 'train_accuracy_per_class': train_accuracy_per_class,
      # 'test_accuracy_per_class': test_accuracy_per_class,
      'train_confusion_matricies': list_train_confusion_matricies,
      'val_confusion_matricies': list_val_confusion_matricies,
      'best_model_idx': best_model_epoch,
      'best_model_state': best_model_state,
      'metric_for_stopping': metric_for_stopping,
      # 'list_train_macro_accuracy': list_train_performance_metric,
      'list_train_performance_metric': list_train_performance_metric,
      'list_val_performance_metric': list_val_performance_metric,
      # 'list_val_macro_accuracy': list_val_performance_metric,
      'epochs': epoch,
      'list_mean_total_norm_epoch': np.array(total_norm_epoch).mean(axis=1) ,
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

  def evaluate(self, val_loader, criterion, unique_val_subjects, unique_val_classes, is_test,is_coral_loss,epoch,history_val_sample_predictions=None,save_log=True):
    # unique_train_val_classes is only for eval but kept the name for compatibility
    device = 'cuda'
    count = 0
    self.to(device) 
    self.eval() 
    with torch.no_grad():
      val_loss = 0.0
      loss_per_class = torch.zeros(2,len(val_loader.dataset.get_unique_classes()))
      accuracy_per_class = torch.zeros(2,len(val_loader.dataset.get_unique_classes()))
      subject_loss = torch.zeros(unique_val_subjects.shape[0])
      accuracy_per_subject = torch.zeros(unique_val_subjects.shape[0])
      # subject_batch_count = torch.zeros(unique_val_subjects.shape[0])
      sample_per_subject_count = torch.zeros(unique_val_subjects.shape[0])
      val_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=len(val_loader.dataset.get_unique_classes())+1) # last class is for bad_classified in regression
      batch_confidence_prediction_right_mean = []
      batch_confidence_prediction_wrong_mean = []
      batch_confidence_prediction_right_std = []
      batch_confidence_prediction_wrong_std = []
      nr_samples = 0 
      list_batch_size = []
      
      amp_dtype = torch.bfloat16 if helper.AMP_DTYPE == 'bfloat16' else torch.float16
      enable_autocast = helper.AMP_ENABLED and amp_dtype == torch.bfloat16
      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(val_loader,total=len(val_loader),desc=f'{("Validation" if not is_test else "Test")}'):
        list_batch_size.append(len(batch_y))
        nr_samples += len(batch_y)
        tmp = torch.isin(unique_val_subjects,batch_subjects)
        _,count_sample_per_subject = torch.unique(batch_subjects, return_counts=True)
        sample_per_subject_count[tmp] += count_sample_per_subject
        if isinstance(criterion, torch.nn.HuberLoss):
          batch_y = batch_y.float()
        elif isinstance(criterion, torch.nn.CrossEntropyLoss):
          batch_y = batch_y.long()
        
        dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
        batch_y = batch_y.to(device)
        
        # subject_batch_count[tmp] += 1
        helper.LOG_CROSS_ATTENTION['state'] = 'test' if is_test else 'val'
        dict_batch_X['list_sample_id'] = sample_id
        with autocast(device_type=device, dtype=amp_dtype, enabled=enable_autocast): # Use autocast for mixed precision training
          outputs = self(**dict_batch_X)
        
        if outputs.shape[1] == 1: # if regression I don't need to keep dim 1 
          outputs = outputs.squeeze(1)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        
        if is_coral_loss:
          outputs = proba_to_label(torch.sigmoid(outputs)) # convert probabilities to labels for coral loss
          batch_y = torch.sum(batch_y, dim=1) # convert levels to labels for coral loss
        
        if save_log and helper.LOG_PER_CLASS_AND_SUBJECT:
          tools.compute_loss_per_class_(batch_y=batch_y, 
                                        class_loss=loss_per_class if not is_coral_loss else None,
                                        unique_train_val_classes=unique_val_classes,
                                        outputs=outputs,
                                        criterion=criterion,
                                        class_accuracy=accuracy_per_class)
          tools.compute_loss_per_subject_v2_(batch_subjects=batch_subjects, 
                                             criterion=criterion,
                                             batch_y=batch_y,
                                             outputs=outputs,
                                             subject_loss=subject_loss if not is_coral_loss else None,
                                             unique_train_val_subjects=unique_val_subjects,
                                             subject_accuracy=accuracy_per_subject)
        if not isinstance(criterion,torch.nn.L1Loss) and not isinstance(criterion,torch.nn.MSELoss) and not isinstance(criterion,torch.nn.HuberLoss) and not is_coral_loss:  
            tools.compute_confidence_predictions_(list_prediction_right_mean=batch_confidence_prediction_right_mean,
                                                  list_prediction_wrong_mean=batch_confidence_prediction_wrong_mean,
                                                  list_prediction_right_std=batch_confidence_prediction_right_std,
                                                  list_prediction_wrong_std=batch_confidence_prediction_wrong_std,
                                                  gt=batch_y, outputs=outputs)
        if self.is_classification:
          if is_coral_loss:
            predictions = outputs.detach().cpu() # outputs are already labels for coral loss
          else:
            predictions = torch.argmax(outputs, dim=1).detach().cpu().reshape(-1)
        else:
          predictions = torch.copysign(torch.floor(torch.abs(outputs) + 0.5), outputs).detach().cpu().float() # to avoid the banker's rounding implemented as IEEE standard
          mask = torch.isin(predictions, unique_val_classes)
          predictions[~mask] = loss_per_class.shape[1] # put prediction in the last class (bad_classified)
          
        batch_y = batch_y.detach().cpu()
        if batch_y.dim() > 1:
          batch_y = torch.argmax(batch_y, dim=1).reshape(-1)
        val_confusion_matricies.update(predictions, batch_y)
        
        if history_val_sample_predictions is not None:
          tools.log_predictions_per_sample_(dict_log_sample=history_val_sample_predictions,
                                            tensor_sample_id=sample_id,
                                            tensor_predictions=predictions,
                                            epoch=epoch)
        count += 1
      
      val_confusion_matricies.compute()
      val_loss = val_loss / len(val_loader)
      if save_log and helper.LOG_PER_CLASS_AND_SUBJECT:
        loss_per_class = loss_per_class[0] / loss_per_class[1] # loss_per_class[0] is loss per class, loss_per_class[1] is total number of samples
        # print(f'VAL Loss mean per class: {torch.mean(loss_per_class)}') 
        accuracy_per_class = accuracy_per_class[0] / accuracy_per_class[1] # accuracy_per_class[0] is correct predictions, accuracy_per_class[1] is total
        subject_loss = subject_loss / sample_per_subject_count
        accuracy_per_subject = accuracy_per_subject / sample_per_subject_count
        
      dict_precision_recall = tools.evaluate_classification_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                       list_real_classes=unique_val_classes)
      if is_test:
        self.log_performance(stage='Test', loss=val_loss, accuracy=dict_precision_recall['accuracy'])
        return {
          'test_loss': val_loss,
          'test_loss_per_class': loss_per_class,
          'test_loss_per_subject': subject_loss,
          'test_accuracy_per_class': accuracy_per_class,
          'test_accuracy_per_subject': accuracy_per_subject,
          'test_macro_precision': dict_precision_recall["macro_precision"],
          'test_accuracy': dict_precision_recall["accuracy"],
          'test_confusion_matrix': val_confusion_matricies,
          'test_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
          'test_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
          'test_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
          'test_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
          'dict_precision_recall': dict_precision_recall,
        }
      else:
        return {
          'val_loss': val_loss,
          'val_loss_per_class': loss_per_class,
          'val_loss_per_subject': subject_loss,
          'val_accuracy_per_class': accuracy_per_class,
          'val_accuracy_per_subject': accuracy_per_subject,
          'val_macro_precision': dict_precision_recall["macro_precision"],
          'val_accuracy': dict_precision_recall["accuracy"],
          'val_confusion_matrix': val_confusion_matricies,
          'val_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
          'val_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
          'val_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
          'val_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0, 
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
    self.load_state_dict(torch.load(path,weights_only=True))
    print(f'Model weights loaded from {path}')
    
  def log_gradient_per_module(self,dict_gradient_per_module):
    for name, param in self.named_parameters():
      if param.grad is not None:
        if name not in dict_gradient_per_module:
          dict_gradient_per_module[name] = []
        dict_gradient_per_module[name].append(param.grad.detach().cpu().clone())
    return dict_gradient_per_module
   
  def calculate_frobenius_norm(self):
    norm = 0.0
    for _, param in self.named_parameters():
      if param.grad is not None:
        norm += torch.linalg.vector_norm(param.grad, ord=2)
        # norm += torch.norm(param.grad, p='fro').item()
    return norm
  
  
class LinearHead(BaseHead):
  def __init__(self, input_dim, num_classes, dim_reduction):
    super().__init__(self,is_classification)
    self._model = LinearProbe(input_dim=input_dim, num_classes=num_classes,dim_reduction=dim_reduction)
    is_classification = True if num_classes > 1 else False
    self.is_classification = is_classification
    self.num_classes = num_classes
  # def get_dataset_and_loader(self, csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal):
  #   return super().get_dataset_and_loader(csv_path, root_folder_features, batch_size, shuffle_training_batch, is_training, dataset_type, concatenate_temporal, False)


# class CrossAttentionBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         num_heads,
#         mlp_ratio=4.,
#         drop=0.0,
#         attn_drop=0.0,
#         residual_drop=0.0,
#         qkv_bias=False,
#         act_layer=nn.GELU,
#         norm_layer=nn.LayerNorm,
#         use_sdpa=False
#     ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.xattn = CrossAttention(dim, 
#                                     num_heads=num_heads,
#                                     qkv_bias=qkv_bias,
#                                     use_sdpa=use_sdpa,
#                                     attn_drop=attn_drop,
#                                     proj_drop=drop)
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         self.residual_drop = nn.Dropout(residual_drop)
  
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
      num_queries=1,
      agg_method='mean',
      pos_enc=False,
      use_sdpa=True,
      grid_size_pos=None,
      cross_block_after_transformers=True,
      coral_loss=False,
      custom_mlp=False,
      q_k_v_dim=None,
      head_init_path=None,
      backbone: custom_backbone.BackboneBase = None,
      embedding_reduction: helper.EMBEDDING_REDUCTION = None,
      complete_block=True):
    super().__init__(self,is_classification=True if num_classes > 1 else False)
    self.pooler = jepa_attentive_pooler.AttentivePooler(
            num_queries=num_queries,
            embed_dim=embed_dim,
            q_k_v_dim=q_k_v_dim if q_k_v_dim is not None else embed_dim,
            num_heads=num_heads,
            num_cross_heads=num_cross_heads,
            mlp_ratio=mlp_ratio,
            depth=depth,
            mlp_dropout=dropout,
            attn_dropout=attn_dropout,
            residual_dropout=residual_dropout,
            norm_layer=norm_layer,
            init_std=init_std,
            qkv_bias=qkv_bias,
            custom_mlp=custom_mlp,
            complete_block=complete_block,
            use_sdpa=use_sdpa,
            cross_block_after_transformers=cross_block_after_transformers
        )
    self.backbone = backbone
    self.embedding_reduction = embedding_reduction
    self.custom_mlp = custom_mlp
    self.num_classes = num_classes
    self.num_queries = num_queries
    self.pos_enc = pos_enc
    self.pos_enc_tensor = None
    self.total_grid_area = grid_size_pos[1]*grid_size_pos[2]
    self.grid_size_pos = grid_size_pos
    self.head_init_path = head_init_path
    self.coral_loss = coral_loss
    
    # Coral loss setup
    if coral_loss:
      self.linear = CoralLayer(embed_dim, num_classes)
    else:
      if self.custom_mlp:
        self.linear = nn.Linear(int(embed_dim * (mlp_ratio**2)), num_classes, bias=True) 
      else:
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)
    
    # Aggregator setup if num_queries > 1  
    if num_queries == 1:
      self.aggregator = nn.Identity() 
    else:
      if agg_method == 'max' or agg_method == 'mean':
        self.aggregator = MeanMaxAggregator(method=agg_method,num_query_tokens=num_queries)
      else:
        raise NotImplementedError(f'Unknown aggregation method {agg_method}')
    
    # Load pooler weights if there is an init. (head_init_path) 
    if head_init_path is not None:
      state_dict = torch.load(head_init_path, weights_only=True)
      # Remove final linear layer for classification/regression 
      layers = list(state_dict.keys())
      for layer_name in layers:
        if layer_name.startswith('linear.'):
          del state_dict[layer_name]
      state_dict = {k.replace('pooler.',''): v for k, v in state_dict.items() if k.startswith('pooler.')}  
      self.pooler.load_state_dict(state_dict)
      for param in self.pooler.parameters():
        param.requires_grad = False
      print(f'FROZEN Head weights loaded from {head_init_path}')
    
    # Init weights
    self._initialize_weights()  
    
  def apply_pos_enc(self, x):
    if self.pos_enc_tensor is None or self.pos_enc_tensor.shape[0] != x.size(1):
        if x.size(1) % self.total_grid_area != 0:
          raise ValueError(f'Input length {x.size(1)} is not divisible by batch size {x.size(0)}')
        self.pos_enc_tensor = pos_embs.get_1d_sincos_pos_embed_torch(embed_dim=x.size(2),
                                                                    grid_size=x.size(1)//self.total_grid_area).unsqueeze(0).to(x.device) # [1, nr_clips, emb_dim]
        if x.shape[1] % self.pos_enc_tensor.shape[1] != 0:
          raise ValueError(f'x.shape not divisible for pos_enc.shape. x size of {x.size(1)} is not divisible by pos_enc size {self.pos_enc.size(1)}')
        self.pos_enc_tensor = torch.repeat_interleave(self.pos_enc_tensor, x.shape[1]//self.pos_enc_tensor.shape[1], dim=1)
        # self.pos_enc_tensor = pos_embs.get_3d_sincos_pos_embed_torch(embed_dim=x.size(2),
        #                                                               grid_depth=x.size(1)//(self.total_grid_area), # Considering same length for all dimensions!
        #                                                               grid_size=self.grid_size_pos[1]).to(x.device).unsqueeze(0)
      # print(self.pos_enc_tensor.shape)
    return x + self.pos_enc_tensor

  def get_feats_from_backbone_v2(self, x, **kwargs):
    """
    Alternative implementation using torch.split for better memory patterns
    """
    # Get backbone features
    x = self.backbone.forward_features(x)
    
    lengths = kwargs['lengths']
    T, S = x.shape[1], x.shape[2]
    tokens_per_chunk = T * S * S
    
    # Split tensor in one operation instead of manual slicing
    samples = torch.split(x, lengths.tolist(), dim=0)
    
    # Calculate max length upfront
    max_len = lengths.max().item() * tokens_per_chunk
    batch_size = lengths.shape[0]
    
    # Pre-allocate output
    padded_x = torch.zeros(batch_size, max_len, x.shape[-1], 
                          dtype=x.dtype, device=x.device)
    
    # Process each sample with minimal memory overhead
    for i, sample in enumerate(samples):
      flat_sample = sample.reshape(-1, sample.shape[-1])
      seq_len = flat_sample.shape[0]
      padded_x[i, :seq_len] = flat_sample
    
    # Efficient mask creation
    actual_lengths = lengths * tokens_per_chunk
    key_padding_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < actual_lengths.unsqueeze(1)
    
    return padded_x, key_padding_mask
  
  def get_feats_from_backbone(self, x, **kwargs):
    x = self.backbone.forward_features(x) # backbone can be to fine-tune, [B * chunks, channels, frames=16, height, width] -> [B*chunks, T, S, S, embed_dim]
    list_x = []
    start = 0
    T,S = x.shape[1], x.shape[2]
    
    # reconstruct original shape
    for real_len in kwargs['lengths']:
      list_x.append(x[start:start+real_len]) # [B*chunks, T, S, S, embed_dim] -> list([chunks, T, S, S, embed_dim]) where len(list_x) = B
      start += real_len
    x = list_x # list([chunks, T, S, S, embed_dim]), len(x) = B  
    x = [sample.reshape(-1,sample.shape[-1]) for sample in x] # [chunks*T*S*S, emb_dim] , concat_temporal = False
    
    # pad sequences to the max length in the batch
    x = torch.nn.utils.rnn.pad_sequence(x,batch_first=True)
    lengths = kwargs['lengths'] # lengths of each sample in the batch, used for padding mask
    max_len = max(lengths) * T * S * S # max length of the sequence in the batch, used for padding mask
    
    # create key_padding_mask for attention
    key_padding_mask = torch.arange(max_len,device=max_len.device).expand(x.size(0), max_len) >= lengths.unsqueeze(1) # [B,seq_len]
    key_padding_mask = ~key_padding_mask # set True for attention, if True means use the token to compute the attention, otherwise don't use it 
    return x, key_padding_mask
  
  def forward(self, x, key_padding_mask=None, return_video_emb=False,**kwargs):
    if self.backbone is not None:
      x, key_padding_mask = self.get_feats_from_backbone_v2(x, **kwargs) 
    
    if self.embedding_reduction !=  helper.EMBEDDING_REDUCTION.NONE:
      x = torch.mean(x,dim=self.embedding_reduction.value,keepdim=True)
      
    if self.pos_enc:
      x = self.apply_pos_enc(x) # add positional encoding if enabled
    
    # FORWARD PASS
    x,xattn = self.pooler(x,key_padding_mask,helper.LOG_CROSS_ATTENTION['enable']) 
    x = x.squeeze(1) # # [B, num_queries=1, C] -> [B, C]
    x = self.aggregator(x)
    if return_video_emb:
      return x, self.linear(x)
    else:
      x = self.linear(x)
    
    # LOG CROSS ATTENTION if enabled
    if helper.LOG_CROSS_ATTENTION['enable'] and xattn is not None:
      self.log_xattn(xattn, **kwargs) 
    
    return x
  
  def _initialize_weights(self,init_type='default'):
    if self.backbone is not None:
      # self.backbone.load_pretrained_weights() 
      for adapter in self.backbone.model.adapters:
        adapter.init_weights()
    if init_type == 'default':
      if self.head_init_path is None:
        trunc_normal_(self.pooler.query_tokens, std=self.pooler.init_std)
        self.pooler.apply(self.pooler._init_weights)
        self.pooler._rescale_blocks()
      if self.coral_loss:
        self.linear.coral_weights.reset_parameters()
      else:
        # self.linear.reset_parameters()
        torch.nn.init.xavier_uniform_(self.linear.weight,gain=0.1)
        torch.nn.init.zeros_(self.linear.bias)
    else:
      raise NotImplementedError(f'Initialization method {init_type} not implemented')
  
  def log_xattn(self,xattn, **kwargs):
    xattn = xattn.detach().cpu().numpy()
    if f"debug_xattn_{helper.LOG_CROSS_ATTENTION['state']}" not in helper.LOG_CROSS_ATTENTION:
        helper.LOG_CROSS_ATTENTION[f"debug_xattn_{helper.LOG_CROSS_ATTENTION['state']}"] = []
    list_sample_id = kwargs['list_sample_id'].numpy().astype(np.uint16) # problem if use torch.uint16 -> no problem when save results.pkl using np.uint16
    
    helper.LOG_CROSS_ATTENTION[f"debug_xattn_{helper.LOG_CROSS_ATTENTION['state']}"].append((list_sample_id,xattn))
    
# class AttentiveHead(BaseHead):
#   def __init__(self,input_dim,num_classes,num_heads,dropout,pos_enc):
#     model = AttentiveProbe(input_dim=input_dim,num_classes=num_classes,num_heads=num_heads,dropout=dropout,pos_enc=pos_enc)
#     is_classification = True if num_classes > 1 else False
#     super().__init__(model,is_classification)
  
class GRUHead(BaseHead):
  def __init__(self, input_dim, hidden_size, num_layers, dropout, output_size,layer_norm):
    super().__init__(self,True if output_size > 1 else False)
    self._model = GRUProbe(input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, output_size=output_size,layer_norm=layer_norm)
    self._is_classification = True if output_size > 1 else False
  
  def forward(self, x, **kwargs): # x is already packed
    return self._model(x)
  
  def _initialize_weights(self,init_type):
    self._model._initialize_weights(init_type=init_type)
    
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
  
  

class MeanMaxAggregator(nn.Module):
  """
  Aggregator that either mean-pools or max-pools over Q CLS tokens.
  - mean: averages and divides by sqrt(Q) to stabilize magnitude.
  - max: takes element-wise maximum.
  """
  def __init__(self, num_query_tokens: int = 1, method: str = 'mean'):
    super().__init__()
    assert method in ['mean', 'max'], "method must be 'mean' or 'max'"
    self.method = method
    self.sqrt_Q = math.sqrt(num_query_tokens)

  def forward(self, pooled_feats: torch.Tensor) -> torch.Tensor:
    """
    pooled_feats: [B, Q, C]
    returns:    [B, C]
    """
    if self.method == 'mean':
      # compute average then normalize by sqrt(Q)
      mean = pooled_feats.mean(dim=1)          # [B, C]
      return mean  / self.sqrt_Q  # [B, C]
    else:  # 'max'
      # elementwise max
      return pooled_feats.max(dim=1)[0]  # [B, C]

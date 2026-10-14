from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import custom.tools as tools
from torchmetrics.classification import ConfusionMatrix
import os
from custom.optimizers import OptimizerFactory

import custom.helper as helper
from custom.dataset import get_dataset_and_loader
import tqdm
import copy

from jepa.src.models.utils import pos_embs
from optuna.exceptions import TrialPruned
import matplotlib.pyplot as plt
import threading
import jepa.src.models.attentive_pooler as jepa_attentive_pooler
from jepa.src.utils.tensors import trunc_normal_
from custom.model_selection import should_select_epoch
from custom.targets import (
  prepare_batch_targets,
  resolve_target_spec,
)


@dataclass
class _SplitTracker:
  """Holds per-epoch metric lists for one evaluation split (val or test)."""
  losses: list = field(default_factory=list)
  losses_per_class: list = field(default_factory=list)
  accuracy_per_class: list = field(default_factory=list)
  losses_per_subject: list = field(default_factory=list)
  accuracy_per_subject: list = field(default_factory=list)
  confusion_matrices: list = field(default_factory=list)
  accuracy: list = field(default_factory=list)
  performance_metric: list = field(default_factory=list)
  icc: list = field(default_factory=list)
  ccc: list = field(default_factory=list)
  pearson: list = field(default_factory=list)
  l1_error: list = field(default_factory=list)
  rmse: list = field(default_factory=list)
  confidence_right_mean: list = field(default_factory=list)
  confidence_wrong_mean: list = field(default_factory=list)
  confidence_right_std: list = field(default_factory=list)
  confidence_wrong_std: list = field(default_factory=list)


def save_selected_checkpoints(
  saving_path,
  *,
  best_model_state,
  best_model_epoch,
  last_model_state,
  last_model_epoch,
  save_best,
  save_last,
):
  saved_paths = {}
  if save_best:
    best_path = os.path.join(
      saving_path, f'best_model_ep_{best_model_epoch}.pt'
    )
    torch.save(best_model_state, best_path)
    saved_paths['best'] = best_path
  if save_last:
    last_path = os.path.join(
      saving_path, f'last_model_ep_{last_model_epoch}.pt'
    )
    torch.save(last_model_state, last_path)
    saved_paths['last'] = last_path
  return saved_paths


import numpy as np

class BaseHead(nn.Module):
  def __init__(self, is_classification):
    super(BaseHead, self).__init__()
    self.is_classification = is_classification

  def _initialize_weights(self, init_type='default'):
    """Keep PyTorch's native initialization unless a head overrides this hook."""
    pass
      
  def log_performance(self, stage, loss, accuracy,epoch=-1,dict_kwarg=None,num_epochs=-1,wds=None,lrs=None):
      if epoch>-1:
        print(f'Epoch [{epoch}/{num_epochs}]')
      print(f' {stage}')
      print(f'  Loss             : {loss:.4f}')
      print(f'  Accuracy         : {accuracy:.4f}')
      if wds:
        if isinstance(wds,list):
          print(f'  Weight decay     : {[round(wd,8) for wd in wds]}')
        elif isinstance(wds,dict):
          for k,v in wds.items():
            print(f'  Weight decay {k} : {v:.8f}')
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
      if torch.cuda.is_available():
        free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
        peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        total_gpu_mem = total_gpu_mem / 1024 ** 3
        free_gpu_mem = free_gpu_mem / 1024 ** 3
        print(f'  GPU memory free  : {free_gpu_mem:.2f} GB')
        print(f'  GPU memory peak  : {peak_gpu_mem:.2f} GB')
      
  def _append_eval_epoch(
    self,
    tracker: _SplitTracker,
    eval_dict: dict,
    perf_key: str,
    log_per_class: bool,
    log_per_subject: bool,
    log_confidence: bool,
    save_this_epoch: bool,
  ) -> None:
    """
    Appends all metrics from one epoch's evaluate() result into a _SplitTracker.

    Args:
      tracker:          The _SplitTracker instance to update (val or test split).
      eval_dict:        Dict returned by evaluate(); keys are always val_-prefixed.
      perf_key:         key_for_early_stopping (e.g. 'val_accuracy').
      log_per_class:    Whether to log per-class metrics (helper.LOG_PER_CLASS).
      log_per_subject:  Whether to log per-subject metrics (helper.LOG_PER_SUBJECT).
      log_confidence:   Whether to log confidence metrics (helper.LOG_CONFIDENCE_PREDICTION).
      save_this_epoch:  Whether this epoch is a checkpoint epoch (saving_rate or best_epoch).
    """
    tracker.losses.append(eval_dict['val_loss'])
    tracker.icc.append(eval_dict['val_ICC'])
    tracker.ccc.append(eval_dict['val_CCC'])
    tracker.pearson.append(eval_dict['val_pearson_correlation'])
    tracker.l1_error.append(eval_dict['val_l1_error'])
    tracker.rmse.append(eval_dict['val_rmse'])
    tracker.accuracy.append(eval_dict['val_accuracy'])
    tracker.performance_metric.append(eval_dict[perf_key])
    tracker.confusion_matrices.append(eval_dict['val_confusion_matrix'])
    if log_confidence:
      tracker.confidence_right_mean.append(eval_dict['val_prediction_confidence_right_mean'])
      tracker.confidence_wrong_mean.append(eval_dict['val_prediction_confidence_wrong_mean'])
      tracker.confidence_right_std.append(eval_dict['val_prediction_confidence_right_std'])
      tracker.confidence_wrong_std.append(eval_dict['val_prediction_confidence_wrong_std'])
    if log_per_class and save_this_epoch:
      tracker.losses_per_class.append(eval_dict['val_loss_per_class'])
      tracker.accuracy_per_class.append(eval_dict['val_accuracy_per_class'])
    if log_per_subject and save_this_epoch:
      tracker.losses_per_subject.append(eval_dict['val_loss_per_subject'])
      tracker.accuracy_per_subject.append(eval_dict['val_accuracy_per_subject'])

  def start_train(self, num_epochs, criterion, lr, saving_path, train_csv_path, val_csv_path ,batch_size,dataset_type,
                  shuffle_training_batch, regularization_lambda_L1,
                  early_stopping, key_for_early_stopping,root_folder_features,init_network,n_workers,label_smooth,
                  regularization_lambda_L2,trial,enable_optuna_pruning,prefetch_factor,**kwargs):
    
    target_values = pd.concat([
      pd.read_csv(path, sep='\t')['class_id']
      for path in (train_csv_path, val_csv_path)
      if path
    ])
    target_spec = resolve_target_spec(
      metadata=kwargs.get('target_spec'),
      values=target_values,
      normalize=bool(kwargs.get('normalize_labels', 0)),
      legacy_max_label=kwargs.get('max_label'),
    )

    # Generate Thread for smooth stopping 
    stop_event = threading.Event()
    def _wait_for_s():
      try:
        cmd = input(">>> Type 's' + Enter to stop after this epoch: ")
      except EOFError:  # non-interactive stdin (e.g. < /dev/null, nohup, CI)
        return
      if cmd.strip().lower() == 's':
        stop_event.set()
        print("Stopping training after this epoch...")
    threading.Thread(target=_wait_for_s, daemon=True).start()
    
    # Start training 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.to(device)
    self._initialize_weights(init_type=init_network)
    
    loader_args = dict(batch_size=batch_size,
                       root_folder_features=root_folder_features,
                       dataset_type=dataset_type,
                       prefetch_factor=prefetch_factor,
                       model=self,
                       label_smooth=label_smooth,
                       n_workers=n_workers,
                       **kwargs)
    train_dataset, train_loader = get_dataset_and_loader(csv_path=train_csv_path,
                                                         shuffle_training_batch=shuffle_training_batch,
                                                         is_training=True,
                                                         **loader_args)
    if val_csv_path is not None:
      val_dataset, val_loader = get_dataset_and_loader(csv_path=val_csv_path,
                                                       shuffle_training_batch=False,
                                                       is_training=False,
                                                       **loader_args)
    if kwargs['use_test_as_val']:
      test_dataset, test_loader = get_dataset_and_loader(csv_path=kwargs['test_csv_path_as_eval'],
                                                         shuffle_training_batch=False,
                                                         is_training=False,
                                                         **loader_args)
      
    kwargs['scheduler_config_dict']['steps_per_epoch'] = len(train_loader)
    optimizer_factory = OptimizerFactory(kwargs['scheduler_config_dict'])
    optimizer, lr_scheduler = optimizer_factory.create(model=self)
    print("\n-------- Created Optimizer and Schedulers -------")
    print("   Optimizer Configuration:", optimizer)
    print("   LR Scheduler Configuration:", lr_scheduler)
    print("   Warm-up ", f"{optimizer_factory.config['warm_up_epochs'] * optimizer_factory.config['steps_per_epoch']} steps" if optimizer_factory.config["warm_up_epochs"] else "disabled")    
    print("---------------------------------------------------")
    train_unique_classes = torch.tensor(train_dataset.get_unique_classes())
    train_unique_subjects = torch.tensor(train_dataset.get_unique_subjects())
    val_unique_classes = torch.tensor(val_dataset.get_unique_classes()) if val_csv_path is not None else torch.tensor([])
    val_unique_subjects = torch.tensor(val_dataset.get_unique_subjects()) if val_csv_path is not None else torch.tensor([])
    if kwargs.get('use_test_as_val', False):
      test_unique_classes = torch.tensor(test_dataset.get_unique_classes()) if kwargs.get('test_csv_path_as_eval', None) is not None else torch.tensor([])
      test_unique_subjects = torch.tensor(test_dataset.get_unique_subjects()) if kwargs.get('test_csv_path_as_eval', None) is not None else torch.tensor([])
    
    if helper.LOG_HISTORY_SAMPLE and torch.min(train_unique_classes)>=0 and torch.max(train_unique_classes)<=255 and torch.min(val_unique_classes)>=0 and torch.max(val_unique_classes)<=255:
      list_train_sample = train_dataset.get_all_sample_ids()
      history_train_sample_predictions = {id: torch.zeros(num_epochs, dtype=torch.float32) for id in list_train_sample}
      list_val_sample = val_dataset.get_all_sample_ids() if val_csv_path is not None else None
      history_val_sample_predictions = {id: torch.zeros(num_epochs, dtype=torch.float32) for id in list_val_sample} if val_csv_path is not None else None
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
    list_train_accuracy = []
    list_train_performance_metric = []
    list_train_confidence_prediction_right_mean = []
    list_train_confidence_prediction_wrong_mean = []
    list_train_confidence_prediction_right_std = []
    list_train_confidence_prediction_wrong_std = []
    list_train_ICC = []
    list_train_CCC = []
    list_train_pearson_correlation = []
    val_t = _SplitTracker()
    test_t = _SplitTracker() if kwargs.get('use_test_as_val', False) else None
    list_lrs = []
    list_wds = []
    early_stopping.reset()
    print(f'\nNumber of parameters for this head:')
    print(f'  Trainable: {((sum(p.numel() for p in self.parameters() if p.requires_grad))/1e6):.4f} M')
    print(f'  Total    : {sum(p.numel() for p in self.parameters())/1e6:.4f} M')
    for name, param in self.named_parameters():
      if param.requires_grad:
        print(f'  {name}: {param.numel()}')
      else:
        print(name)

    metric_for_stopping = "_".join(key_for_early_stopping.split('_')[1:]) # ex: val_accuracy -> accuracy. the second part must be metric from tools.evaluate_classification_from_confusion_matrix
    use_exact_targets = not self.is_classification
    # save .pt model before the traingn for the future logs of the untrained model
    torch.save(self.state_dict(), os.path.join(saving_path, f'model_epoch_-1.pt'))
    max_train_class = target_spec.bin_count
    best_eval_metric = None
    for epoch in range(num_epochs):
      start_epoch = time.perf_counter()
      self.train() 
      list_train_epoch_predictions = []
      list_train_ground_truths = []
      class_loss = torch.zeros(2,train_unique_classes.shape[0])
      class_accuracy = torch.zeros(2,train_unique_classes.shape[0])
      subject_loss = torch.zeros(train_unique_subjects.shape[0])
      subject_accuracy = torch.zeros(train_unique_subjects.shape[0])
      sample_per_subject_count = torch.zeros(train_unique_subjects.shape[0])
      train_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=max_train_class)
      train_loss = 0.0
      batch_train_confidence_prediction_right_mean = []
      batch_train_confidence_prediction_right_std = []
      batch_train_confidence_prediction_wrong_mean = []
      batch_train_confidence_prediction_wrong_std = []
      dict_log_time = {}
      dict_log_time['pre_batch'] = dict_log_time.get('pre_batch',0) + time.perf_counter() - start_epoch
      start_load_batch = time.perf_counter()
      if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
      list_batch_lr = []
      list_batch_wd = []
      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(train_loader,total=len(train_loader),desc=f'Train {epoch}/{num_epochs}'):
        end_load_batch = time.perf_counter()
        dict_log_time['load_batch'] = dict_log_time.get('load_batch',0) + end_load_batch - start_load_batch
        time_to_count_subjects = time.perf_counter()
        tmp = torch.isin(train_unique_subjects,batch_subjects)
        _,count_sample_per_subject = torch.unique(batch_subjects, return_counts=True)
        sample_per_subject_count[tmp] += count_sample_per_subject
        dict_log_time['count_subjects'] = dict_log_time.get('count_subjects',0) + time.perf_counter() - time_to_count_subjects
        transfer_to_device = time.perf_counter()
        batch_exact_targets = batch_y.detach().clone() if batch_y.ndim == 1 else None
        batch_y, batch_class_targets = prepare_batch_targets(
          dict_batch_X, batch_y, target_spec, use_exact_targets
        )
        batch_y = batch_y.to(device)
        batch_class_targets = batch_class_targets.to(device)
        batch_y_original = (
          batch_exact_targets.to(device)
          if use_exact_targets and batch_exact_targets is not None
          else batch_class_targets
        )
        dict_batch_X = {key: value.to(device) if value is not None else None for key, value in dict_batch_X.items()}
        dict_log_time['transfer_to_device'] = dict_log_time.get('transfer_to_device',0) + time.perf_counter() - transfer_to_device
        optimizer.zero_grad()
        
        dict_batch_X['list_sample_id'] = sample_id
        dict_batch_X['return_video_emb'] = False
        start_forward = time.perf_counter()
        start_model_forward = time.perf_counter()
        dict_out = self(**dict_batch_X) # input [batch, seq_len, emb_dim]
        if dict_out['logits'].shape[1] == 1: # if regression I don't need to keep dim 1
          dict_out['logits'] = dict_out['logits'].squeeze(1)
        dict_log_time['model_forward'] = dict_log_time.get('model_forward',0) + time.perf_counter() - start_model_forward
        start_loss_computation = time.perf_counter()
        loss = criterion(dict_out['logits'], batch_y)
        outputs = dict_out['logits']
        if regularization_lambda_L1 > 0:
          # Sum absolute values of all trainable parameters except biases
          l1_norm = sum(param.abs().sum() for name,param in self.named_parameters() if param.requires_grad and 'bias' not in name) 
          loss = loss + regularization_lambda_L1 * l1_norm 

        dict_log_time['loss_computation'] = dict_log_time.get('loss_computation',0) + time.perf_counter() - start_loss_computation
        dict_log_time['forward'] = dict_log_time.get('forward',0) + time.perf_counter()-start_forward
        start_back = time.perf_counter()
        loss.backward()
        dict_log_time['backward'] = dict_log_time.get('backward',0) + time.perf_counter()-start_back
        start_optimizer = time.perf_counter()
        optimizer.step()
        dict_log_time['optimizer'] = dict_log_time.get('optimizer',0) + time.perf_counter()-start_optimizer
        train_loss += loss.item()
        
        # Log learning rate and weight decay
        monitoring_values = OptimizerFactory.get_monitoring_values(optimizer)
        list_batch_lr.append(monitoring_values['lr'])
        list_batch_wd.append({'wd_no_bias': monitoring_values['wd_no_bias'], 'wd_bias': monitoring_values['wd_bias']})
        
        if lr_scheduler:
          lr_scheduler.step()
          
        start_logs = time.perf_counter()
        # De-normalize outputs for all metric logging when label normalization is active
        _outputs_for_log = target_spec.inverse(outputs)
        _batch_y_for_log = batch_y_original
        if helper.LOG_PER_CLASS:
          tools.compute_loss_per_class_(batch_y=_batch_y_for_log,
                                      class_loss=class_loss,
                                      unique_train_val_classes=train_unique_classes,
                                      outputs=_outputs_for_log,
                                      class_accuracy=class_accuracy,
                                      criterion=criterion,
                                      class_targets=batch_class_targets,
                                      target_spec=target_spec)
        if helper.LOG_PER_SUBJECT:
          tools.compute_loss_per_subject_v2_(batch_subjects=batch_subjects,
                                        criterion=criterion,
                                        batch_y=_batch_y_for_log,
                                        outputs=_outputs_for_log,
                                        subject_loss=subject_loss,
                                        subject_accuracy=subject_accuracy,
                                        unique_train_val_subjects=train_unique_subjects,
                                        unique_train_val_classes=train_unique_classes,
                                        class_targets=batch_class_targets,
                                        target_spec=target_spec)
        if helper.LOG_CONFIDENCE_PREDICTION and self.is_classification:
          tools.compute_confidence_predictions_(list_prediction_right_mean=batch_train_confidence_prediction_right_mean,
                                                list_prediction_right_std=batch_train_confidence_prediction_right_std,
                                                list_prediction_wrong_mean=batch_train_confidence_prediction_wrong_mean,
                                                list_prediction_wrong_std=batch_train_confidence_prediction_wrong_std,
                                                gt=_batch_y_for_log,
                                                outputs=_outputs_for_log)

        if self.is_classification:
          predictions = torch.argmax(_outputs_for_log, dim=1).detach().cpu().reshape(-1)
        else:
          # RAW continuous prediction (de-normalized, un-rounded) for the train regression metrics
          predictions = _outputs_for_log.detach().cpu().float().reshape(-1)

        metric_targets = _batch_y_for_log.detach().cpu().reshape(-1)
        diagnostic_targets = batch_class_targets.detach().cpu()

        list_train_epoch_predictions.append(predictions.numpy())
        list_train_ground_truths.append(metric_targets.numpy())

        # Round + clamp ONLY for the integer-class confusion matrix (regression branch only)
        if not self.is_classification:
          predictions = target_spec.predictions_to_bins(predictions)
        train_confusion_matrix.update(predictions, diagnostic_targets)
        dict_log_time['batch_logs'] = dict_log_time.get('batch_logs',0) + time.perf_counter()-start_logs 
        start_load_batch = time.perf_counter()

      # End of train loader
      list_lrs.append(list_batch_lr)
      list_wds.append(list_batch_wd)
      time_eval = time.perf_counter()
      dict_eval = None
      if val_csv_path is not None:
        dict_eval = self.evaluate(criterion=criterion,
                                  is_test=False,
                                  unique_val_classes=val_unique_classes,
                                  unique_val_subjects=val_unique_subjects,
                                  val_loader=val_loader,
                                  epoch=epoch,
                                  history_val_sample_predictions=history_val_sample_predictions,
                                  **kwargs)
      dict_test_as_eval = None
      if kwargs.get('test_csv_path_as_eval') is not None:
        dict_test_as_eval = self.evaluate(criterion=criterion,
                                          is_test=False,
                                          unique_val_classes=test_unique_classes,
                                          unique_val_subjects=test_unique_subjects,
                                          val_loader=test_loader,
                                          epoch=epoch,
                                          history_val_sample_predictions=None,
                                          **kwargs)
      dict_log_time['eval'] = dict_log_time.get('eval',0) + time.perf_counter()-time_eval
      
      epoch_log_time = time.perf_counter()
      
      candidate_eval_metric = (
        dict_eval[key_for_early_stopping] if dict_eval is not None else None
      )
      if should_select_epoch(
        best_eval_metric,
        candidate_eval_metric,
        key_for_early_stopping,
        has_validation=dict_eval is not None,
      ):
        best_eval_metric = candidate_eval_metric
        best_model_state = copy.deepcopy(self.state_dict())
        best_model_state = {key: value.cpu() for key, value in best_model_state.items()}
        best_model_epoch = epoch
        best_epoch = True
      
      list_train_losses.append(train_loss / len(train_loader))
      _eval_flags = dict(
        perf_key=key_for_early_stopping,
        log_per_class=helper.LOG_PER_CLASS,
        log_per_subject=helper.LOG_PER_SUBJECT,
        log_confidence=helper.LOG_CONFIDENCE_PREDICTION,
        save_this_epoch=epoch % helper.saving_rate_training_logs == 0 or best_epoch,
      )
      if dict_eval is not None:
        self._append_eval_epoch(val_t, dict_eval, **_eval_flags)
      else:
        val_t.losses.append(0.0)
        val_t.performance_metric.append(0.0)
        val_t.accuracy.append(0.0)
      if dict_test_as_eval is not None:
        self._append_eval_epoch(test_t, dict_test_as_eval, **_eval_flags)

      # Save model at certain epochs
      if helper.SAVE_MODEL_EVERY_N_EPOCHS > 0 and epoch % helper.SAVE_MODEL_EVERY_N_EPOCHS == 0:
        model_path_epoch = os.path.join(saving_path, f'model_epoch_{epoch}.pt')
        torch.save(self.state_dict(), model_path_epoch)
        fig,ax = plt.subplots(2,1,figsize=(15,10))
        ax = ax.flatten()
        input_dict_loss_acc= {
          'list_1':list_train_losses,
          'list_2':val_t.losses,
          'output_path':None,
          'title':f'Train loss, validation loss',
          'point':None,
          'ax':ax[0],
          'x_label':'Epochs',
          'y_label_1':'Train loss',
          'y_label_2':'Validation loss',
          'y_label_3':None,
          'y_lim_1':[0, 2],
          'y_lim_2':[0, 2],
          'y_lim_3':None,
          'step_ylim_1':0.2,
          'step_ylim_2':0.2,
          'step_ylim_3':None,
          'dict_to_string':None,
          'color_1':'tab:red',
          'color_2':'tab:blue',
          'color_3':None,
        }
        tools.plot_losses_and_test_new(**input_dict_loss_acc)
        fig.savefig(os.path.join(saving_path, f'loss_logs.png'), dpi=300)
        plt.close(fig)
        print(f'Checkpoint saved at epoch {epoch} to {model_path_epoch} and plotted the loss.')
      
      np_list_train_epoch_predictions = np.concatenate(list_train_epoch_predictions, axis=0)
      np_list_train_ground_truths = np.concatenate(list_train_ground_truths, axis=0)
      list_train_ICC.append(tools.intraclass_icc(np.column_stack((np_list_train_ground_truths, np_list_train_epoch_predictions))))
      list_train_CCC.append(tools.concordance_ccc(y_true=np_list_train_ground_truths, y_pred=np_list_train_epoch_predictions))
      list_train_pearson_correlation.append(tools.pearson_r(y_true=np_list_train_ground_truths, y_pred=np_list_train_epoch_predictions))
      train_confusion_matrix.compute()
      train_dict_precision_recall = tools.evaluate_classification_from_confusion_matrix(confusion_matrix=train_confusion_matrix, list_real_classes=train_unique_classes)
      train_dict_precision_recall['loss'] = list_train_losses[-1]
      list_train_accuracy.append(train_dict_precision_recall['accuracy'])
      list_train_performance_metric.append(train_dict_precision_recall[metric_for_stopping])

      if helper.LOG_CONFIDENCE_PREDICTION:
        list_train_confidence_prediction_right_mean.append(np.mean(batch_train_confidence_prediction_right_mean) if len(batch_train_confidence_prediction_right_mean) > 0 else 0)
        list_train_confidence_prediction_wrong_mean.append(np.mean(batch_train_confidence_prediction_wrong_mean) if len(batch_train_confidence_prediction_wrong_mean) > 0 else 0)
        list_train_confidence_prediction_right_std.append(np.std(batch_train_confidence_prediction_right_mean) if len(batch_train_confidence_prediction_right_mean) > 0 else 0)
        list_train_confidence_prediction_wrong_std.append(np.std(batch_train_confidence_prediction_wrong_mean) if len(batch_train_confidence_prediction_wrong_mean) > 0 else 0)

      if epoch % helper.saving_rate_training_logs == 0 or best_epoch:
        list_train_confusion_matricies.append(train_confusion_matrix)
        if helper.LOG_PER_CLASS:
          list_train_losses_per_class.append(class_loss[0] / class_loss[1])
          list_train_accuracy_per_class.append(class_accuracy[0] / class_accuracy[1])
        if helper.LOG_PER_SUBJECT:
          list_train_losses_per_subject.append(subject_loss / sample_per_subject_count)
          list_train_accuracy_per_subject.append(subject_accuracy / sample_per_subject_count)
      
      # log performance
      self.log_performance(stage='Train',
                           num_epochs=num_epochs,
                           loss=list_train_losses[-1], 
                           accuracy=train_dict_precision_recall.get('accuracy', 0.0),
                           epoch=epoch,
                           lrs=np.mean(list_batch_lr),
                           wds={k: np.mean([d[k] for d in list_batch_wd]) for k in list_batch_wd[0].keys()})
      if dict_eval is not None:
        self.log_performance(stage='Val',
                            num_epochs=num_epochs,
                            loss=dict_eval['val_loss'],
                            accuracy=dict_eval['val_accuracy'],)
      if dict_test_as_eval is not None:
        self.log_performance(stage='Test',
                            num_epochs=num_epochs,
                            loss=dict_test_as_eval['val_loss'],
                            accuracy=dict_test_as_eval['val_accuracy'],)
        
      if dict_eval is not None and early_stopping(dict_eval[key_for_early_stopping]):
        print(f'Early stopping triggered at epoch {epoch} with {key_for_early_stopping}: {dict_eval[key_for_early_stopping]:.4f}')
        break
      
      if enable_optuna_pruning and dict_eval is not None and trial is not None:
        trial.report(dict_eval[key_for_early_stopping], epoch)
        if trial is not None and trial.should_prune():
          raise TrialPruned()
      dict_log_time['log_epoch'] = dict_log_time.get('log_epoch',0) + time.perf_counter()-epoch_log_time
      dict_log_time['epoch'] = time.perf_counter()-start_epoch

      print(f'---------TRAIN LOOP LOGS---------')
      for k,v in dict_log_time.items():
        print(f'  {k} time: {v:.4f} s')

      print('-----------PROFILING LOGS-----------')
      summary_aggregated = {}
      for key,val in helper.time_profile_dict.items():
        split_key = key.split('_')
        key_name = '_'.join(split_key[1:])
        summary_aggregated[key_name] = summary_aggregated.get(key_name,0) + val
      for k,v in summary_aggregated.items():
        print(f'  {k}: {v:.4f} {"s" if "time" in k else "count"}')

      print('---------------------------------------')

      helper.time_profile_dict.clear() # reset logs for next epoch
      
      if stop_event.is_set():
        print(f'Stop event is set. Stopping training...')
        break
      
    if saving_path and helper.SAVE_PTH_MODEL:
      saved_checkpoint_paths = save_selected_checkpoints(
        saving_path,
        best_model_state=best_model_state,
        best_model_epoch=best_model_epoch,
        last_model_state=None,
        last_model_epoch=epoch,
        save_best=True,
        save_last=False,
      )
      print(f"Best model weights saved to {saved_checkpoint_paths['best']}")
    
    logs = {
      'train_losses': list_train_losses,
      'train_loss_per_class': np.array(list_train_losses_per_class),
      'train_loss_per_subject': np.array(list_train_losses_per_subject),
      'y_unique': np.unique(np.concatenate((train_unique_classes,val_unique_classes),axis=0)),
      'train_unique_subject_ids': train_unique_subjects.numpy(),
      'train_count_subject_ids': train_dataset.get_count_subjects(),
      'train_unique_y': train_unique_classes,
      'subject_ids_unique': np.unique(np.concatenate((train_unique_subjects.numpy(),val_unique_subjects.numpy()),axis=0)),
      'list_train_accuracy_per_subject': list_train_accuracy_per_subject,
      'list_train_accuracy_per_class': list_train_accuracy_per_class,
      'list_train_confidence_prediction_right_mean': list_train_confidence_prediction_right_mean,
      'list_train_confidence_prediction_wrong_mean': list_train_confidence_prediction_wrong_mean,
      'list_train_confidence_prediction_right_std': list_train_confidence_prediction_right_std,
      'list_train_confidence_prediction_wrong_std': list_train_confidence_prediction_wrong_std,
      'history_train_sample_predictions': history_train_sample_predictions,
      'history_val_sample_predictions': history_val_sample_predictions,
      'list_train_accuracy': list_train_accuracy,
      'train_confusion_matricies': list_train_confusion_matricies,
      'best_model_idx': best_model_epoch,
      'best_model_state': best_model_state,
      'metric_for_stopping': metric_for_stopping,
      'list_train_performance_metric': list_train_performance_metric,
      'epochs': epoch,
      'list_lrs': list_lrs,
      'list_wds': list_wds,
      'optimizer': optimizer.state_dict()['param_groups'],
      'optimizer_config': optimizer_factory.get_config(),
      'list_train_ICC': list_train_ICC,
      'list_train_CCC': list_train_CCC,
      'list_train_pearson_correlation': list_train_pearson_correlation,
      'val_losses': val_t.losses,
      'val_loss_per_class': np.array(val_t.losses_per_class),
      'val_loss_per_subject': np.array(val_t.losses_per_subject),
      'val_unique_subject_ids': val_unique_subjects.numpy(),
      'val_count_subject_ids': val_dataset.get_count_subjects() if val_csv_path is not None else None,
      'val_unique_y': val_unique_classes,
      'list_val_accuracy_per_subject': val_t.accuracy_per_subject,
      'list_val_accuracy_per_class': val_t.accuracy_per_class,
      'list_val_confidence_prediction_right_mean': val_t.confidence_right_mean,
      'list_val_confidence_prediction_wrong_mean': val_t.confidence_wrong_mean,
      'list_val_confidence_prediction_right_std': val_t.confidence_right_std,
      'list_val_confidence_prediction_wrong_std': val_t.confidence_wrong_std,
      'list_val_accuracy': val_t.accuracy,
      'val_confusion_matricies': val_t.confusion_matrices,
      'list_val_performance_metric': val_t.performance_metric,
      'list_val_ICC': val_t.icc,
      'list_val_CCC': val_t.ccc,
      'list_val_l1_error': val_t.l1_error,
      'list_val_rmse': val_t.rmse,
      'list_val_pearson_correlation': val_t.pearson,
      'train_batch_sampler_name': (train_loader.batch_sampler.__class__.__module__ + "." + train_loader.batch_sampler.__class__.__name__),
    }
    if kwargs.get('use_test_as_val', False):
      logs['test_as_eval'] = {
        'test_losses':                         test_t.losses,
        'test_list_l1_error':                  test_t.l1_error,
        'test_list_rmse':                      test_t.rmse,
        'test_loss_per_class':                 np.array(test_t.losses_per_class),
        'test_loss_per_subject':               np.array(test_t.losses_per_subject),
        'test_unique_subjects_ids':            test_unique_subjects.numpy(),
        'test_count_subjects_ids':             test_dataset.get_count_subjects() if kwargs['test_csv_path_as_eval'] is not None else None,
        'test_unique_y':                       test_unique_classes,
        'test_unique_classes_ids':             test_unique_classes.numpy(),
        'list_test_accuracy_per_subject':      test_t.accuracy_per_subject,
        'list_test_accuracy_per_class':        test_t.accuracy_per_class,
        'list_test_confidence_right_mean':     test_t.confidence_right_mean,
        'list_test_confidence_wrong_mean':     test_t.confidence_wrong_mean,
        'list_test_confidence_right_std':      test_t.confidence_right_std,
        'list_test_confidence_wrong_std':      test_t.confidence_wrong_std,
        'test_confusion_matricies':            test_t.confusion_matrices,
        'list_test_accuracy':                  test_t.accuracy,
        'list_test_performance_metric':        test_t.performance_metric,
        'list_test_ICC':                       test_t.icc,
        'list_test_CCC':                       test_t.ccc,
        'list_test_pearson_correlation':       test_t.pearson,
      }
    return logs

  def evaluate(self, val_loader, criterion, unique_val_subjects, unique_val_classes, is_test,epoch,history_val_sample_predictions=None,save_log=True,**kwargs):
    # unique_train_val_classes is only for eval but kept the name for compatibility
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.to(device) 
    self.eval() 
    use_exact_targets = not self.is_classification
    dataset_spec = getattr(val_loader.dataset, 'target_spec', None)
    target_values = (
      val_loader.dataset.df['class_id']
      if hasattr(val_loader.dataset, 'df') and 'class_id' in val_loader.dataset.df
      else unique_val_classes
    )
    target_spec = resolve_target_spec(
      metadata=kwargs.get('target_spec'),
      values=target_values,
      normalize=bool(kwargs.get('normalize_labels', 0)),
      legacy_max_label=kwargs.get('max_label'),
      fallback=dataset_spec,
    )
    list_val_epoch_predictions = []
    list_val_ground_truths = []
    with torch.no_grad():
      val_loss = 0.0
      loss_per_class = torch.zeros(2,len(val_loader.dataset.get_unique_classes()))
      accuracy_per_class = torch.zeros(2,len(val_loader.dataset.get_unique_classes()))
      subject_loss = torch.zeros(unique_val_subjects.shape[0])
      accuracy_per_subject = torch.zeros(unique_val_subjects.shape[0])
      sample_per_subject_count = torch.zeros(unique_val_subjects.shape[0])
      val_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=target_spec.bin_count)
      batch_confidence_prediction_right_mean = []
      batch_confidence_prediction_wrong_mean = []
      batch_confidence_prediction_right_std = []
      batch_confidence_prediction_wrong_std = []
      return_embeddings = helper.LOG_VIDEO_EMBEDDINGS['enable']

      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(val_loader,total=len(val_loader),desc=f'{("Validation" if not is_test else "Test")}'):
        tmp = torch.isin(unique_val_subjects,batch_subjects)
        _,count_sample_per_subject = torch.unique(batch_subjects, return_counts=True)
        sample_per_subject_count[tmp] += count_sample_per_subject
        batch_exact_targets = batch_y.detach().clone() if batch_y.ndim == 1 else None
        batch_y, batch_class_targets = prepare_batch_targets(
          dict_batch_X, batch_y, target_spec, use_exact_targets
        )
        dict_batch_X = {key: value.to(device) if value is not None else None for key, value in dict_batch_X.items()}
        batch_y = batch_y.to(device)
        batch_class_targets = batch_class_targets.to(device)
        batch_y_original = (
          batch_exact_targets.to(device)
          if use_exact_targets and batch_exact_targets is not None
          else batch_class_targets
        )
        dict_batch_X['list_sample_id'] = sample_id
        dict_batch_X['return_video_emb'] = return_embeddings
        dict_out = self(**dict_batch_X)

        if dict_out['logits'].shape[1] == 1: # if regression I don't need to keep dim 1
          dict_out['logits'] = dict_out['logits'].squeeze(1)

        if return_embeddings:
          helper.LOG_VIDEO_EMBEDDINGS['embeddings'].append(dict_out['embeddings'].detach().cpu())
          helper.LOG_VIDEO_EMBEDDINGS['labels'].extend(batch_y_original.detach().cpu().numpy().tolist())
          helper.LOG_VIDEO_EMBEDDINGS['sample_ids'].extend(sample_id.cpu().numpy().tolist())
          helper.LOG_VIDEO_EMBEDDINGS['predictions'].extend(target_spec.inverse(dict_out['logits']).detach().cpu().numpy().tolist())
        outputs = dict_out['logits']
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        
        # De-normalize outputs for all metric logging when label normalization is active
        _outputs_for_log = target_spec.inverse(outputs)
        _batch_y_for_log = batch_y_original

        if save_log and helper.LOG_PER_CLASS:
            tools.compute_loss_per_class_(batch_y=_batch_y_for_log,
                                          class_loss=loss_per_class,
                                          unique_train_val_classes=unique_val_classes,
                                          outputs=_outputs_for_log,
                                          criterion=criterion,
                                          class_accuracy=accuracy_per_class,
                                          class_targets=batch_class_targets,
                                          target_spec=target_spec)
        if save_log and helper.LOG_PER_SUBJECT:
          tools.compute_loss_per_subject_v2_(batch_subjects=batch_subjects,
                                            criterion=criterion,
                                            batch_y=_batch_y_for_log,
                                            outputs=_outputs_for_log,
                                            subject_loss=subject_loss,
                                            unique_train_val_subjects=unique_val_subjects,
                                            subject_accuracy=accuracy_per_subject,
                                            unique_train_val_classes=unique_val_classes,
                                            class_targets=batch_class_targets,
                                            target_spec=target_spec)
        if save_log and helper.LOG_CONFIDENCE_PREDICTION and self.is_classification:
          tools.compute_confidence_predictions_(list_prediction_right_mean=batch_confidence_prediction_right_mean,
                                                list_prediction_wrong_mean=batch_confidence_prediction_wrong_mean,
                                                list_prediction_right_std=batch_confidence_prediction_right_std,
                                                list_prediction_wrong_std=batch_confidence_prediction_wrong_std,
                                                gt=_batch_y_for_log, outputs=_outputs_for_log)
        if self.is_classification:
          predictions = torch.argmax(_outputs_for_log, dim=1).detach().cpu().reshape(-1)
        else:
          # RAW continuous prediction (de-normalized, un-rounded) — used for the history
          # dict and the regression metrics (Pearson/CCC/ICC/L1/RMSE).
          predictions = _outputs_for_log.detach().cpu().float().reshape(-1)

        if history_val_sample_predictions is not None:
          tools.log_predictions_per_sample_(dict_log_sample=history_val_sample_predictions,
                                            tensor_sample_id=sample_id,
                                            tensor_predictions=predictions,
                                            epoch=epoch)
        metric_targets = _batch_y_for_log.detach().cpu().reshape(-1)
        diagnostic_targets = batch_class_targets.detach().cpu()

        list_val_epoch_predictions.append(predictions.numpy())
        list_val_ground_truths.append(metric_targets.numpy())

        # Round + clamp ONLY for the integer-class confusion matrix (regression branch only)
        if not self.is_classification:
          predictions = target_spec.predictions_to_bins(predictions)

        val_confusion_matricies.update(predictions, diagnostic_targets)
      
      val_loss = val_loss / len(val_loader)
      val_confusion_matricies.compute()
      if save_log and helper.LOG_PER_CLASS:
        loss_per_class = loss_per_class[0] / loss_per_class[1] # loss_per_class[0] is loss per class, loss_per_class[1] is total number of samples
        accuracy_per_class = accuracy_per_class[0] / accuracy_per_class[1] # accuracy_per_class[0] is correct predictions, accuracy_per_class[1] is total
      if save_log and helper.LOG_PER_SUBJECT:  
        subject_loss = subject_loss / sample_per_subject_count
        accuracy_per_subject = accuracy_per_subject / sample_per_subject_count
        
      dict_precision_recall = tools.evaluate_classification_from_confusion_matrix(confusion_matrix=val_confusion_matricies,
                                                                     list_real_classes=unique_val_classes)
      list_val_epoch_predictions = np.concatenate(list_val_epoch_predictions, axis=0)
      list_val_ground_truths = np.concatenate(list_val_ground_truths, axis=0)
      pearson_corr = tools.pearson_r(y_pred=list_val_epoch_predictions,
                                    y_true=list_val_ground_truths)
      CCC = tools.concordance_ccc(y_pred=list_val_epoch_predictions,
                                  y_true=list_val_ground_truths)
      ICC = tools.intraclass_icc(data=np.column_stack((list_val_ground_truths, list_val_epoch_predictions)),
                                  icc_type='ICC2')
      l1_error = np.mean(np.abs(list_val_epoch_predictions - list_val_ground_truths))
      rmse = np.sqrt(np.mean((list_val_epoch_predictions - list_val_ground_truths)**2))

    prefix = 'test' if is_test else 'val'
    if is_test:
      self.log_performance(stage='Test', loss=val_loss, accuracy=dict_precision_recall.get('accuracy', 0.0))
    return {
      f'{prefix}_loss': val_loss,
      f'{prefix}_loss_per_class': loss_per_class,
      f'{prefix}_loss_per_subject': subject_loss,
      f'{prefix}_accuracy_per_class': accuracy_per_class,
      f'{prefix}_accuracy_per_subject': accuracy_per_subject,
      f'{prefix}_macro_precision': dict_precision_recall.get("macro_precision",0.0),
      f'{prefix}_accuracy': dict_precision_recall.get("accuracy",0.0),
      f'{prefix}_confusion_matrix': val_confusion_matricies,
      f'{prefix}_pearson_correlation': pearson_corr,
      f'{prefix}_CCC': CCC,
      f'{prefix}_ICC': ICC,
      f'{prefix}_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
      f'{prefix}_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
      f'{prefix}_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
      f'{prefix}_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
      'dict_precision_recall': dict_precision_recall,
      f'{prefix}_l1_error': l1_error,
      f'{prefix}_rmse': rmse,
    }

  def load_state_weights(self, path):
    self.load_state_dict(torch.load(path,weights_only=True))
    print(f'Model weights loaded from {path}')


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
      pos_enc=0,
      use_sdpa=True,
      grid_size_pos=None,
      cross_block_after_transformers=True,
      custom_mlp=False,
      q_k_v_dim=None,
      embedding_reduction: helper.EMBEDDING_REDUCTION = None,
      complete_block=1,
      mlp_num_hidden_layers=2):
    super().__init__(is_classification=True if num_classes > 1 else False)
    self.pooler = jepa_attentive_pooler.AttentivePooler(
            num_queries=1,
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
            mlp_num_hidden_layers=mlp_num_hidden_layers,
            complete_block=complete_block,
            drop_path_mode='row',
            use_sdpa=use_sdpa,
            cross_block_after_transformers=cross_block_after_transformers
        )
    if pos_enc not in (0, 3):
      raise NotImplementedError(f'pos_enc={pos_enc} is not supported (use 0 or 3)')
    self.embedding_reduction = embedding_reduction
    self.custom_mlp = custom_mlp
    self.num_classes = num_classes
    self.pos_enc = pos_enc
    self.pos_enc_tensor = None
    self.total_grid_area = grid_size_pos[1]*grid_size_pos[2] # spatial_area * spatial_area
    self.grid_size_pos = grid_size_pos
    self.linear = nn.Linear(embed_dim, num_classes, bias=True)

  def apply_pos_enc(self, x):
    """Add the 3D sin-cos positional encoding (time x height x width) to the token sequence."""
    if self.pos_enc_tensor is None or self.pos_enc_tensor.shape[0] != x.size(1):
      if x.size(1) % self.total_grid_area != 0:
        raise ValueError(f'Input length {x.size(1)} is not divisible by grid size {self.total_grid_area}.\nx.size // self.total_grid_area = {x.size(1)} // {self.total_grid_area} = {x.size(1)//self.total_grid_area}')
      grid_depth = x.size(1) // self.total_grid_area # depth => time_dimension
      grid_size = self.grid_size_pos[1] # height * width => spatial_dimension
      self.pos_enc_tensor = pos_embs.get_3d_sincos_pos_embed_torch(embed_dim=x.size(2), # [B, seq_len ,C]
                                                                  grid_depth=grid_depth,
                                                                  grid_size=grid_size
                                                                  ).unsqueeze(0).to(x.device)
      if x.shape[1] % self.pos_enc_tensor.shape[1] != 0:
        raise ValueError(f'x.shape not divisible for pos_enc.shape. x size of {x.size(1)} is not divisible by pos_enc size {self.pos_enc_tensor.size(1)}')
    return x + self.pos_enc_tensor

  def forward(self, x, key_padding_mask=None, return_video_emb=True,**kwargs):
    pid = os.getpid()
    # Apply positional encoding if enabled
    time_pos_enc = time.perf_counter()
    if self.pos_enc:
      x = self.apply_pos_enc(x) # add positional encoding if enabled
    helper.time_profile_dict[f'{pid}_pos_enc_time'] = helper.time_profile_dict.get(f'{pid}_pos_enc_time',0) + (time.perf_counter() - time_pos_enc)

    # FORWARD PASS
    time_head = time.perf_counter()
    x,_ = self.pooler(x,key_padding_mask,False)
    x = x.squeeze(1) # # [B, num_queries=1, C] -> [B, C]
    helper.time_profile_dict[f'{pid}_head_forward_time'] = helper.time_profile_dict.get(f'{pid}_head_forward_time',0) + (time.perf_counter() - time_head)
    
    logits = self.linear(x)
    return {'logits': logits, 'embeddings': x if return_video_emb else None}

  def _init_weights(self):
    trunc_normal_(self.pooler.query_tokens, std=self.pooler.init_std)
    self.pooler.apply(self.pooler._init_weights)
    self.pooler._rescale_blocks()

  def _initialize_weights(self,init_type='default'):
    if init_type != 'default':
      raise NotImplementedError(f'Initialization method {init_type} not implemented')
    self._init_weights()
    print('\nHead weights initialized from scratch\n')
    torch.nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
    torch.nn.init.zeros_(self.linear.bias)
    print(f'All trainable params sum: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')


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

class earlyStoppingDummy(EarlyStopping):
  """
  No-op early stopping that never halts training.

  Satisfies the EarlyStopping interface so key_early_stopping
  still drives best-model selection without imposing any stop.

  Args:
    None

  Returns:
    Always False from __call__.
  """
  def __init__(self):
    pass

  def __call__(self, metric):
    return False

  def reset(self):
    pass

  def __str__(self):
    return f'{self.__class__.__name__}: disabled (never stops)'



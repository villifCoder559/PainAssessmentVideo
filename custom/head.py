from collections import defaultdict
from dataclasses import dataclass, field
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import custom.tools as tools
from torchmetrics.classification import ConfusionMatrix
import math
import os
from custom.optimizers import OptimizerFactory

from custom.helper import CUSTOM_DATASET_TYPE
import torch.nn.init as init
import custom.helper as helper
from torch.nn.utils.rnn import pack_padded_sequence,pack_padded_sequence
from custom.dataset import get_dataset_and_loader
import custom.dataset as dataset_utils
import tqdm
import copy
from jepa.src.models.attentive_pooler import AttentiveClassifier as AttentiveClassifierJEPA

from jepa.src.models.utils import pos_embs
from jepa.src.models.utils import modules 
from optuna.exceptions import TrialPruned
import pickle
import matplotlib.pyplot as plt
# import tracemalloc
# import wandb
from torch.amp import GradScaler, autocast
import threading
import jepa.src.models.attentive_pooler as jepa_attentive_pooler
# import jepa.src.models.utils.modules as jepa_modules
from jepa.src.utils.tensors import trunc_normal_
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import proba_to_label
import custom.backbone as custom_backbone
import multiprocessing as mp
import torch.nn.functional as F
import custom.loss as losses


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


class GradientReversalFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, lambda_):
    ctx.lambda_ = lambda_
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_):
  return GradientReversalFunction.apply(x, lambda_)

import numpy as np

class AdvAlpha:
  def __init__(self, total_steps, method, gamma):
    """
    Initializes the class with precomputed alpha values for all steps.
    
    Args:
        total_steps (int): Total expected iterations or epochs.
        method (str): 'sigmoid', 'linear', or 'fixed' (default 'sigmoid').
        gamma (float): Steepness for sigmoid (default 10 from paper).
    """
    self.total_steps = total_steps
    self.method = method
    self.gamma = gamma
    self.alpha_values = self._precompute_alphas()

  def _precompute_alphas(self):
    """
    Precomputes the alpha values for all steps from 0 to total_steps.
    
    Returns:
        List[float]: Precomputed alpha values.
    """
    alphas = []
    for current_step in range(self.total_steps):
      p = float(current_step) / (self.total_steps - 1)  # Normalize to [0, 1]
      if self.method == 'sigmoid':
        alpha = 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0
      elif self.method == 'linear':
        alpha = p
      elif self.method == 'fixed':
        alpha = 1.0
      else:
        raise ValueError(f"Unknown method '{self.method}' for calculating adversarial alpha.")
      alphas.append(alpha)
    return alphas

  def get_adv_alpha(self, current_step):
    """
    Retrieves the precomputed adversarial alpha value for the given step.
    
    Args:
        current_step (int): Current iteration or epoch.
    
    Returns:
        float: The precomputed alpha value.
    """
    if current_step < 0 or current_step >= self.total_steps:
      raise ValueError(f"Current step {current_step} is out of range (0 to {self.total_steps - 1}).")
    return self.alpha_values[current_step]

class BaseHead(nn.Module):
  def __init__(self, is_classification):
    super(BaseHead, self).__init__()
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
      free_gpu_mem,total_gpu_mem = torch.cuda.mem_get_info()
      peak_gpu_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
      total_gpu_mem = total_gpu_mem / 1024 ** 3
      free_gpu_mem = free_gpu_mem / 1024 ** 3
      print(f'  GPU memory free  : {free_gpu_mem:.2f} GB')
      print(f'  GPU memory peak  : {peak_gpu_mem:.2f} GB')
      
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

  def _get_grad_category(self, name):
    if name.startswith('backbone.'):
      return 'backbone'
    if name.startswith('pooler.'):
      return 'pooler'
    if name.startswith('linear.'):
      return 'linear'
    if name.startswith('adversarial_head.'):
      return 'adversarial_head'
    return 'head_other'

  def _collect_gradient_flow_stats(self, epoch, batch_idx):
    categories = {
      'backbone': {'trainable_params': 0, 'with_grad': 0, 'without_grad': 0, 'zero_grad': 0, 'norms': []},
      'pooler': {'trainable_params': 0, 'with_grad': 0, 'without_grad': 0, 'zero_grad': 0, 'norms': []},
      'linear': {'trainable_params': 0, 'with_grad': 0, 'without_grad': 0, 'zero_grad': 0, 'norms': []},
      'adversarial_head': {'trainable_params': 0, 'with_grad': 0, 'without_grad': 0, 'zero_grad': 0, 'norms': []},
      'head_other': {'trainable_params': 0, 'with_grad': 0, 'without_grad': 0, 'zero_grad': 0, 'norms': []},
    }
    
    for name, param in self.named_parameters():
      if not param.requires_grad:
        continue
      category = self._get_grad_category(name)
      categories[category]['trainable_params'] += 1
      if param.grad is None:
        categories[category]['without_grad'] += 1
        continue

      grad_norm = torch.linalg.vector_norm(param.grad.detach()).item()
      categories[category]['with_grad'] += 1
      categories[category]['norms'].append(grad_norm)
      if grad_norm == 0.0:
        categories[category]['zero_grad'] += 1

    def _finalize(stats):
      norms = stats.pop('norms')
      stats['mean_grad_norm'] = float(np.mean(norms)) if len(norms) > 0 else 0.0
      stats['max_grad_norm'] = float(np.max(norms)) if len(norms) > 0 else 0.0
      return stats

    categories = {k: _finalize(v) for k, v in categories.items()}
    return {
      'epoch': int(epoch),
      'batch_idx': int(batch_idx),
      'categories': categories,
      'backbone_has_trainable': categories['backbone']['trainable_params'] > 0,
      'backbone_has_grads': categories['backbone']['with_grad'] > 0,
    }

  def _append_eval_epoch(
    self,
    tracker: _SplitTracker,
    eval_dict: dict,
    perf_key: str,
    log_per_class: bool,
    log_per_subject: bool,
    log_confidence: bool,
    log_correlation: bool,
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
      log_confidence:   Whether to log confidence metrics (helper.LOG_CONFIDENCE_PREDICTION and not special loss).
      log_correlation:  Whether ICC/CCC/Pearson/confusion are available (False for resupcon/disentangled/rnc).
      save_this_epoch:  Whether this epoch is a checkpoint epoch (saving_rate or best_epoch).
    """
    tracker.losses.append(eval_dict['val_loss'])
    if log_correlation:
      tracker.icc.append(eval_dict['val_ICC'])
      tracker.ccc.append(eval_dict['val_CCC'])
      tracker.pearson.append(eval_dict['val_pearson_correlation'])
      tracker.l1_error.append(eval_dict['val_l1_error'])
      tracker.rmse.append(eval_dict['val_rmse'])
      tracker.accuracy.append(eval_dict['val_accuracy'])
      tracker.performance_metric.append(eval_dict[perf_key])
      tracker.confusion_matrices.append(eval_dict['val_confusion_matrix'])
    else:
      tracker.performance_metric.append(eval_dict['val_loss'])
      tracker.accuracy.append(eval_dict.get('val_accuracy', 0.0))
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

  def _print_gradient_flow_stats(self, grad_flow_stats):
    bb = grad_flow_stats['categories']['backbone']
    print(
      f"[GradFlow][E{grad_flow_stats['epoch']} B{grad_flow_stats['batch_idx']}] "
      f"backbone trainable={bb['trainable_params']} with_grad={bb['with_grad']} "
      f"without_grad={bb['without_grad']} zero_grad={bb['zero_grad']} "
      f"mean_norm={bb['mean_grad_norm']:.3e}"
    )
    
  def start_train(self, num_epochs, criterion, lr, saving_path, train_csv_path, val_csv_path ,batch_size,dataset_type,
                  round_output_loss, shuffle_training_batch, regularization_lambda_L1,concatenate_temp_dim,clip_grad_norm,
                  early_stopping, key_for_early_stopping,root_folder_features,init_network,backbone_dict,n_workers,label_smooth,
                  regularization_lambda_L2,trial,enable_optuna_pruning,prefetch_factor,soft_labels,is_coral_loss,sample_frame_strategy,stride_inside_window,
                  num_clips_per_video, **kwargs):
    
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
                                                          stride_inside_window=stride_inside_window,
                                                          concatenate_temporal=concatenate_temp_dim,
                                                          num_clips_per_video=num_clips_per_video,
                                                          sample_frame_strategy=sample_frame_strategy,
                                                          dataset_type=dataset_type,
                                                          prefetch_factor=prefetch_factor,
                                                          backbone_dict=backbone_dict,
                                                          model=self,
                                                          is_coral_loss=is_coral_loss,
                                                          soft_labels=soft_labels,
                                                          label_smooth=label_smooth,
                                                          n_workers=n_workers,
                                                              **kwargs)
        
    if val_csv_path is not None:
      val_dataset, val_loader = get_dataset_and_loader(batch_size=batch_size,
                                                            csv_path=val_csv_path,
                                                            root_folder_features=root_folder_features,
                                                            shuffle_training_batch=False,
                                                            is_training=False,
                                                            sample_frame_strategy=sample_frame_strategy,
                                                            num_clips_per_video=num_clips_per_video,
                                                            stride_inside_window=stride_inside_window,
                                                            concatenate_temporal=concatenate_temp_dim,
                                                            dataset_type=dataset_type,
                                                            prefetch_factor=prefetch_factor,
                                                            backbone_dict=backbone_dict,
                                                            model=self, 
                                                            is_coral_loss=is_coral_loss,
                                                            soft_labels=soft_labels,
                                                            label_smooth=label_smooth,
                                                            n_workers=n_workers,
                                                            **kwargs)
    if kwargs['use_test_as_val']:
      test_dataset, test_loader = get_dataset_and_loader(batch_size=batch_size,
                                                         csv_path=kwargs['test_csv_path_as_eval'],
                                                         root_folder_features=root_folder_features,
                                                         shuffle_training_batch=False,
                                                         is_training=False,
                                                         sample_frame_strategy=sample_frame_strategy,
                                                         num_clips_per_video=num_clips_per_video,
                                                         stride_inside_window=stride_inside_window,
                                                         concatenate_temporal=concatenate_temp_dim,
                                                         dataset_type=dataset_type,
                                                         prefetch_factor=prefetch_factor,
                                                         backbone_dict=backbone_dict,
                                                         model=self, 
                                                         is_coral_loss=is_coral_loss,
                                                         soft_labels=soft_labels,
                                                         label_smooth=label_smooth,
                                                         n_workers=n_workers,
                                                         **kwargs)
      
    kwargs['scheduler_config_dict']['steps_per_epoch'] = len(train_loader)
    optimizer_factory = OptimizerFactory(kwargs['scheduler_config_dict'])

    # --- Center Loss: resolve feat_dim via a no-grad peek, then build param group ---
    center_loss_fn = None
    lambda_center_loss = kwargs.get('lambda_center_loss', 0.0)
    _center_extra_groups = None
    if lambda_center_loss > 0:
      _num_cl = kwargs.get('num_classes_center_loss', kwargs['num_classes'])
      center_loss_fn = losses.CenterLoss(num_classes=_num_cl,
                                         feat_dim=kwargs['head_embed_dim'],
                                         norm=1).to(device)
      _ratio_cl = max(float(kwargs.get('ratio_lr_center_loss', 1.0)), 1e-8)
      _initial_lr = kwargs['scheduler_config_dict']['lr']
      _center_extra_groups = [{
        'params': list(center_loss_fn.parameters()),
        'lr': _initial_lr / _ratio_cl,
        'weight_decay': 0.0,
        'is_center_loss': True,
      }]
      print(f"\n[CenterLoss] feat_dim={kwargs['head_embed_dim']}, num_classes={_num_cl}, "
            f"lambda={lambda_center_loss}, center_lr={_initial_lr / _ratio_cl:.2e}")
      kwargs['center_loss_fn'] = center_loss_fn

    optimizer, lr_scheduler, wd_scheduler = optimizer_factory.create(
      model=self,
      extra_param_groups=_center_extra_groups,
    )
    print("\n-------- Created Optimizer and Schedulers -------")
    print("   Optimizer Configuration:", optimizer)
    print("   LR Scheduler Configuration:", lr_scheduler)
    print("   WD Scheduler Configuration:", wd_scheduler)
    print("   Warm-up ", f"{optimizer_factory.config['warm_up_epochs'] * optimizer_factory.config['steps_per_epoch']} steps" if optimizer_factory.config["warm_up_epochs"] else "disabled")    
    print("---------------------------------------------------")
    # train_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    train_unique_classes = torch.tensor(train_dataset.get_unique_classes())
    train_unique_subjects = torch.tensor(train_dataset.get_unique_subjects())
    total_train_unique_subjects = len(train_unique_subjects)
    # val_unique_classes = np.array(list(range(self.model.num_classes))) # last class is for bad_classified in regression
    # if val_csv_path is not None:
    val_unique_classes = torch.tensor(val_dataset.get_unique_classes()) if val_csv_path is not None else torch.tensor([])
    val_unique_subjects = torch.tensor(val_dataset.get_unique_subjects()) if val_csv_path is not None else torch.tensor([])
    if kwargs.get('use_test_as_val', False):
      test_unique_classes = torch.tensor(test_dataset.get_unique_classes()) if kwargs.get('test_csv_path_as_eval', None) is not None else torch.tensor([])
      test_unique_subjects = torch.tensor(test_dataset.get_unique_subjects()) if kwargs.get('test_csv_path_as_eval', None) is not None else torch.tensor([])
    
    if helper.LOG_HISTORY_SAMPLE and torch.min(train_unique_classes)>=0 and torch.max(train_unique_classes)<=255 and torch.min(val_unique_classes)>=0 and torch.max(val_unique_classes)<=255:
      list_train_sample = train_dataset.get_all_sample_ids()
      history_train_sample_predictions = {id: torch.zeros(num_epochs, dtype=torch.float32) for id in list_train_sample}
      # if val_csv_path is not None:
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
    list_train_adv_losses = []
    list_train_mt_losses = []
    list_train_adv_accuracies = []
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
    total_norm_epoch = []
    list_lrs = []
    list_wds = []
    epochs_gradient_per_module = {}
    grad_flow_debug_logs = []
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
    enable_autocast = helper.AMP_ENABLED # Use autocast for mixed precision training (fp16/bf16)
    enable_scaler = helper.AMP_ENABLED and amp_dtype in [torch.float16] # Use GradScaler for mixed precision training
    scaler = GradScaler(device=device, enabled=enable_scaler)
    # check if adversarial_head exists and is not None
    if hasattr(self, 'adversarial_head') and self.adversarial_head is not None:
      print(f'\n\nAdversarial head detected with output dimension {self.adversarial_head.out_features}')
      adv_criterion = torch.nn.CrossEntropyLoss()
      adv_alpha = AdvAlpha(total_steps=num_epochs * len(train_loader),
                         method=kwargs['adv_alpha_strategy'],
                         gamma=kwargs['adv_alpha_gamma'])
    
      assert kwargs.get('adversarial_loss_lambda',0.0) > 0.0, "adversarial_loss_lambda must be greater than 0.0 when using adversarial head"
    else:
      adv_criterion = None
      adv_alpha = None
    # check if multitask_head exists and is not None
    if hasattr(self, 'multitask_head') and self.multitask_head is not None:
      print(f'\n\nMultitask head detected with output dimension {self.multitask_head.out_features}')
      mt_criterion = torch.nn.CrossEntropyLoss()
      lambda_multitask = kwargs['lambda_multitask']
      assert lambda_multitask > 0.0, "lambda_multitask must be > 0.0 when using multitask head"
    else:
      mt_criterion = None
      lambda_multitask = 0.0
    if enable_scaler:
      print(f'Using GradScaler for mixed precision training. Dtype {amp_dtype}')
    else:
      print(f'Using standard precision training without GradScaler')
    if enable_autocast:
      print(f'Using autocast for mixed precision training with {amp_dtype} dtype')
    else:
      print(f'Using standard precision training without autocast')
    
    return_embeddings = getattr(criterion, 'return_embeddings', False) or \
                        kwargs['HSIC_subject_lambda'] > 0.0 or \
                        kwargs['HSIC_pain_lambda'] > 0.0 or \
                        kwargs.get('lambda_center_loss', 0.0) > 0.0
                        
    if return_embeddings:
      print('The criterion requires to return the video embeddings from the model during training\n')
    is_composite_loss = isinstance(criterion, losses.CompositeLoss)
    is_resupcon_loss = isinstance(criterion, losses.RESupConLoss)
    is_disentangled_loss = isinstance(criterion, losses.DisentangledLoss)
    is_rnc_loss = isinstance(criterion, losses.RnCLossV2) or isinstance(criterion, losses.RnCLoss)
    debug_grad_flow = bool(kwargs.get('debug_grad_flow', 0))
    debug_grad_flow_batches = int(kwargs.get('debug_grad_flow_batches', 1))
    normalize_labels = bool(kwargs.get('normalize_labels', 0))
    max_label = kwargs.get('max_label', None)
    _label_denorm = float(max_label) if normalize_labels and max_label else 1.0
    
    # save .pt model before the traingn for the future logs of the untrained model
    torch.save(self.state_dict(), os.path.join(saving_path, f'model_epoch_-1.pt'))
    max_train_class = train_unique_classes.max().item() + 1 # start from 0
    train_loader_len = len(train_loader)
    grad_adv_norm_epoch = []
    grad_task_norm_epoch = []
    train_dict_log_loss_steps = []
    val_dict_log_loss_steps = []
    # dict_log_loader = {}
    # dict_view_logger = {}
    # epoch_mean_std_features = {}
    train_epoch_mean_hsic_sbj = []
    train_epoch_mean_hsic_pain = []
    val_epoch_mean_hsic_sbj = []
    val_epoch_mean_hsic_pain = []
    train_epoch_mean_center_loss = []
    val_epoch_mean_center_loss = []
    for epoch in range(num_epochs):
      start_epoch = time.perf_counter()
      self.train() 
      list_train_epoch_predictions = []
      list_train_ground_truths = []
      class_loss = torch.zeros(2,train_unique_classes.shape[0])
      class_accuracy = torch.zeros(2,train_unique_classes.shape[0])
      # subject_loss = np.zeros(train_unique_subjects.shape[0])
      # subject_accuracy = np.zeros(train_unique_subjects.shape[0])
      subject_loss = torch.zeros(train_unique_subjects.shape[0])
      subject_accuracy = torch.zeros(train_unique_subjects.shape[0])
      sample_per_subject_count = torch.zeros(train_unique_subjects.shape[0])
      train_confusion_matrix = ConfusionMatrix(task='multiclass',num_classes=max_train_class)
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
      dict_log_time['pre_batch'] = dict_log_time.get('pre_batch',0) + time.perf_counter() - start_epoch
      start_load_batch = time.perf_counter()
      torch.cuda.reset_peak_memory_stats(device)
      list_batch_lr = []
      total_steps = num_epochs * len(train_loader)
      list_batch_wd = []
      batch_adv_loss = 0.0
      batch_mt_loss = 0.0
      batch_adv_predictions = []
      batch_adv_gt = []
      batch_original_loss = 0.0
      batch_adv_grad_norm = []
      batch_task_grad_norm = []
      batch_rncloss_logs_list = []
      # dict_batch_loader = {}
      # dict_batch_view_loader = {}
      # batch_mean_std_feats = {}
      batch_mean_hsic_sbj = []
      batch_mean_hsic_pain = []
      batch_mean_center_loss = []
      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(train_loader,total=len(train_loader),desc=f'Train {epoch}/{num_epochs}'):
        end_load_batch = time.perf_counter()
        dict_log_time['load_batch'] = dict_log_time.get('load_batch',0) + end_load_batch - start_load_batch
        time_to_count_subjects = time.perf_counter()
        tmp = torch.isin(train_unique_subjects,batch_subjects)
        _,count_sample_per_subject = torch.unique(batch_subjects, return_counts=True)
        sample_per_subject_count[tmp] += count_sample_per_subject
        dict_log_time['count_subjects'] = dict_log_time.get('count_subjects',0) + time.perf_counter() - time_to_count_subjects
        transfer_to_device = time.perf_counter()
        # HuberLoss requires float32 inputs
        if batch_y.dtype == torch.int32: # if ce not has label smoothing
          if isinstance(criterion, (torch.nn.HuberLoss, torch.nn.MSELoss, torch.nn.L1Loss)):
            batch_y = batch_y.float()
          elif isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, losses.PHuberCrossEntropy):
            batch_y = batch_y.long()
        batch_y = batch_y.to(device)
        
        if adv_criterion is not None:
          batch_subjects = batch_subjects.to(torch.long).to(device)
        batch_y_original = batch_y.detach().clone()  # original integer labels for metric logging
        if normalize_labels and max_label:
          batch_y = batch_y.float() / max_label
        dict_batch_X = {key: value.to(device) if value is not None else None for key, value in dict_batch_X.items()}
        dict_log_time['transfer_to_device'] = dict_log_time.get('transfer_to_device',0) + time.perf_counter() - transfer_to_device
        optimizer.zero_grad()
        
        helper.LOG_CROSS_ATTENTION['state'] = 'train'
        dict_batch_X['list_sample_id'] = sample_id
        dict_batch_X['return_video_emb'] = return_embeddings
        if adv_criterion is not None:
          dict_batch_X['adv_alpha'] = adv_alpha.get_adv_alpha(current_step = epoch * train_loader_len + count_batch)
        start_forward = time.perf_counter()
        with autocast(device_type=device, dtype=amp_dtype, enabled=enable_autocast): # Use autocast for mixed precision training
          start_model_forward = time.perf_counter()
          dict_out = self(**dict_batch_X) # input [batch, seq_len, emb_dim]
          if dict_out['logits'].shape[1] == 1: # if regression I don't need to keep dim 1
            dict_out['logits'] = dict_out['logits'].squeeze(1)
          if helper.PROFILING_GPU_SYNC and torch.cuda.is_available():
            torch.cuda.synchronize()
          dict_log_time['model_forward'] = dict_log_time.get('model_forward',0) + time.perf_counter() - start_model_forward
          start_loss_computation = time.perf_counter()
          dict_out['targets'] = batch_y
          if is_composite_loss:
            loss = criterion(**dict_out)
            loss = loss['total_loss'] 
          elif is_disentangled_loss:
            batch_subjects = batch_subjects.to(device)
            loss, dict_log_loss = criterion(features=dict_out['logits'],
                                            target_pain=batch_y,
                                            target_subj=batch_subjects)
            train_dict_log_loss_steps.append(dict_log_loss)
          elif is_rnc_loss:
            loss = criterion(features=dict_out['logits'],
                             labels=batch_y)
            batch_rncloss_logs_list.append(criterion.last_stats)
            
          else:
            loss = criterion(dict_out['logits'], batch_y)
            
          if adv_criterion is not None:
            adv_logits = dict_out['adv_logits']
            adv_loss = adv_criterion(adv_logits, batch_subjects)
            
            batch_adv_loss += adv_loss.item()
            batch_original_loss += loss.item()
            batch_adv_predictions.append(torch.argmax(adv_logits, dim=1).detach().cpu())
            batch_adv_gt.append(batch_subjects.detach().cpu())
            if helper.LOG_GRADIENT_NORM:
              start_log_grad = time.perf_counter()
              task_norm = torch.autograd.grad(loss,
                                              [p for p in self.parameters() if p.requires_grad], 
                                              allow_unused=True,
                                              retain_graph=True)
              task_norm = [g for g in task_norm if g is not None]
              task_norm = torch.cat([g.flatten() for g in task_norm]).norm(2).detach().cpu().item() 
              adv_norm = torch.autograd.grad(adv_loss,
                                              [p for p in self.parameters() if p.requires_grad], 
                                              allow_unused=True,
                                              retain_graph=True)
              adv_norm = [g for g in adv_norm if g is not None]
              adv_norm = torch.cat([g.flatten() for g in adv_norm]).norm(2).detach().cpu().item()
              batch_task_grad_norm.append(task_norm)
              batch_adv_grad_norm.append(adv_norm*dict_batch_X['adv_alpha'])
              dict_log_time['log_adv_grad'] = dict_log_time.get('log_adv_grad',0) + time.perf_counter() - start_log_grad
            loss = loss + adv_loss * kwargs['adversarial_loss_lambda']

          if mt_criterion is not None:
            mt_logits = dict_out['mt_logits']
            mt_loss = mt_criterion(mt_logits, batch_y.long())
            batch_mt_loss += mt_loss.item()
            loss = loss + mt_loss * lambda_multitask

          outputs = dict_out['logits']
          if regularization_lambda_L1 > 0:
            # Sum absolute values of all trainable parameters except biases
            l1_norm = sum(param.abs().sum() for name,param in self.named_parameters() if param.requires_grad and 'bias' not in name) 
            loss = loss + regularization_lambda_L1 * l1_norm 
          if kwargs['HSIC_subject_lambda'] > 0:
            batch_subjects = batch_subjects.to(device)
            hsic_subject = losses.marginal_hsic_loss(
                    z=dict_out['embeddings'],
                    id_labels=batch_subjects,
                    normalize_features=bool(kwargs['HSIC_feat_normalization']),
                    )
            loss = loss + kwargs['HSIC_subject_lambda'] * hsic_subject
            batch_mean_hsic_sbj.append(hsic_subject.item())
          if kwargs['HSIC_pain_lambda'] > 0:
            hsic_pain = losses.marginal_hsic_loss(
                    z=dict_out['embeddings'],
                    id_labels=batch_y,
                    normalize_features=bool(kwargs['HSIC_feat_normalization']),
                    )
            # enforce similarity between samples of the same pain class by maximizing hsic_pain, which is equivalent to minimizing -hsic_pain
            loss = loss - kwargs['HSIC_pain_lambda'] * hsic_pain
            batch_mean_hsic_pain.append(hsic_pain.item())
          if center_loss_fn is not None and lambda_center_loss > 0:
            center_loss_val = center_loss_fn(dict_out['embeddings'], batch_y.long())
            loss = loss + lambda_center_loss * center_loss_val
            batch_mean_center_loss.append(center_loss_val.item())

        dict_log_time['loss_computation'] = dict_log_time.get('loss_computation',0) + time.perf_counter() - start_loss_computation
        if helper.PROFILING_GPU_SYNC and torch.cuda.is_available():
          torch.cuda.synchronize()
        dict_log_time['forward'] = dict_log_time.get('forward',0) + time.perf_counter()-start_forward
        start_back = time.perf_counter()
        scaler.scale(loss).backward() # Use scaler for mixed precision training
        if helper.PROFILING_GPU_SYNC and torch.cuda.is_available():
          torch.cuda.synchronize()
        dict_log_time['backward'] = dict_log_time.get('backward',0) + time.perf_counter()-start_back
        log_gradient_start = time.perf_counter()
        scaler.unscale_(optimizer)  # Unscale gradients before clipping
        if clip_grad_norm:
          total_norm=torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm).detach().cpu() #  torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        elif helper.LOG_GRADIENT_NORM:
          total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 500).detach().cpu() 
          
        else:
          total_norm = torch.tensor(0.0)
        dict_log_time['log_gradient'] = dict_log_time.get('log_gradient',0) + time.perf_counter() - log_gradient_start
        total_norm_epoch[epoch].append(total_norm.item())
        start_optimizer = time.perf_counter()
        prev_scale = scaler.get_scale()
        scaler.step(optimizer)  # Use scaler to step the optimizer
        scaler.update()
        optimizer_step_skipped = enable_scaler and (scaler.get_scale() < prev_scale)
        if helper.PROFILING_GPU_SYNC and torch.cuda.is_available():
          torch.cuda.synchronize()
        dict_log_time['optimizer'] = dict_log_time.get('optimizer',0) + time.perf_counter()-start_optimizer
        train_loss += loss.item() * _label_denorm
        
        # Log learning rate and weight decay
        monitoring_values = OptimizerFactory.get_monitoring_values(optimizer)
        list_batch_lr.append(monitoring_values['lr'])
        list_batch_wd.append({'wd_no_bias': monitoring_values['wd_no_bias'], 'wd_bias': monitoring_values['wd_bias']})
        
        if lr_scheduler and not optimizer_step_skipped:
          lr_scheduler.step()
          if center_loss_fn is not None:
            for _pg in optimizer.param_groups:
              if _pg.get('is_center_loss', False):
                _pg['lr'] = max(_pg['lr'], 1e-8)
        if wd_scheduler and not optimizer_step_skipped:
          wd_scheduler.step()
          
        start_logs = time.perf_counter()
        # De-normalize outputs for all metric logging when label normalization is active
        _outputs_for_log = (outputs * max_label) if normalize_labels and max_label else outputs
        _batch_y_for_log = batch_y_original  # always original integer scale
        if not is_resupcon_loss and not is_rnc_loss and (helper.LOG_PER_CLASS or helper.LOG_PER_SUBJECT or helper.LOG_CONFIDENCE_PREDICTION):
          if is_coral_loss:
            _outputs_for_log = proba_to_label(torch.sigmoid(outputs)) # convert probabilities to labels for coral loss
            _batch_y_for_log = torch.sum(batch_y, dim=1) # convert levels to labels for coral loss
          if helper.LOG_PER_CLASS:
            tools.compute_loss_per_class_(batch_y=_batch_y_for_log,
                                        class_loss=class_loss if not is_coral_loss else None,
                                        unique_train_val_classes=train_unique_classes,
                                        outputs=_outputs_for_log,
                                        class_accuracy=class_accuracy,
                                        criterion=criterion)
          if helper.LOG_PER_SUBJECT:
            tools.compute_loss_per_subject_v2_(batch_subjects=batch_subjects,
                                          criterion=criterion,
                                          batch_y=_batch_y_for_log,
                                          outputs=_outputs_for_log,
                                          subject_loss=subject_loss if not is_coral_loss else None,
                                          subject_accuracy=subject_accuracy,
                                          unique_train_val_subjects=train_unique_subjects,
                                          unique_train_val_classes=train_unique_classes)
          if helper.LOG_CONFIDENCE_PREDICTION:
            if self.is_classification and not is_coral_loss:
              tools.compute_confidence_predictions_(list_prediction_right_mean=batch_train_confidence_prediction_right_mean,
                                                    list_prediction_right_std=batch_train_confidence_prediction_right_std,
                                                    list_prediction_wrong_mean=batch_train_confidence_prediction_wrong_mean,
                                                    list_prediction_wrong_std=batch_train_confidence_prediction_wrong_std,
                                                    gt=_batch_y_for_log,
                                                    outputs=_outputs_for_log)

        if helper.LOG_GRADIENT_PER_MODULE:
          self.log_gradient_per_module(batch_dict_gradient_per_module)

        if debug_grad_flow and count_batch < debug_grad_flow_batches:
          grad_flow_stats = self._collect_gradient_flow_stats(epoch=epoch, batch_idx=count_batch)
          grad_flow_debug_logs.append(grad_flow_stats)
          self._print_gradient_flow_stats(grad_flow_stats)

        count_batch+=1

        if not is_resupcon_loss and not is_disentangled_loss and not is_rnc_loss:
          if self.is_classification:
            if is_coral_loss:
              predictions = _outputs_for_log.detach().cpu() # outputs are already labels for coral loss
            else:
              predictions = torch.argmax(_outputs_for_log, dim=1).detach().cpu().reshape(-1)
          else:
            # RAW continuous prediction (de-normalized, un-rounded) for the train regression metrics
            predictions = _outputs_for_log.detach().cpu().float().reshape(-1)

          batch_y = _batch_y_for_log.detach().cpu()
          if batch_y.dim() > 1:
            batch_y = torch.argmax(batch_y, dim=1).reshape(-1)

          list_train_epoch_predictions.append(predictions.numpy())
          list_train_ground_truths.append(batch_y.numpy())

          try:
            # Round + clamp ONLY for the integer-class confusion matrix (regression branch only)
            if not self.is_classification:
              predictions = torch.copysign(torch.floor(torch.abs(predictions) + 0.5), predictions).clamp(0, train_unique_classes.max().item()) # to avoid the banker's rounding implemented as IEEE standard
            train_confusion_matrix.update(predictions, batch_y)
          except Exception as e:
            print(f"Error updating confusion matrix: {e}")
            print(f"  predictions: {predictions}")
            print(f"  batch_y: {batch_y}")
            print(f' train_confusion_matrix shape: {train_confusion_matrix.confusion_matrix.shape}')
          # if history_train_sample_predictions is not None: 
          #   tools.log_predictions_per_sample_(dict_log_sample=history_train_sample_predictions,
          #                                     tensor_sample_id=sample_id,
          #                                     tensor_predictions=outputs.detach().cpu() if criterion in [torch.nn.L1Loss, torch.nn.MSELoss, torch.nn.HuberLoss] else predictions,
          #                                     epoch=epoch)
          # list_memory_snap.append(tracemalloc.take_snapshot())
          dict_log_time['batch_logs'] = dict_log_time.get('batch_logs',0) + time.perf_counter()-start_logs 
          # dict_log_time['batch'] = dict_log_time.get('batch',0) + time.time()-end_load_batch
        start_load_batch = time.perf_counter()

      # End of train loader
      list_lrs.append(list_batch_lr)
      list_wds.append(list_batch_wd)
      train_epoch_mean_hsic_sbj.append(np.mean(batch_mean_hsic_sbj) if len(batch_mean_hsic_sbj)>0 else 0.0)
      train_epoch_mean_hsic_pain.append(np.mean(batch_mean_hsic_pain) if len(batch_mean_hsic_pain)>0 else 0.0)
      train_epoch_mean_center_loss.append(np.mean(batch_mean_center_loss) if batch_mean_center_loss else 0.0)
      # if lambda_center_loss > 0:
      #   print(f'Center loss for epoch {epoch}: {train_epoch_mean_center_loss[-1]:.4f}')
      # val_epoch_mean_hsic_sbj.append(np.mean(val_epoch_mean_hsic_sbj) if len(val_epoch_mean_hsic_sbj)>0 else 0.0)
      # val_epoch_mean_hsic_pain.append(np.mean(val_epoch_mean_hsic_pain) if len(val_epoch_mean_hsic_pain)>0 else 0.0)
      if adv_criterion is not None:
        list_train_adv_losses.append([batch_adv_loss / len(train_loader), batch_original_loss / len(train_loader)])
        batch_adv_predictions = torch.cat(batch_adv_predictions)
        batch_adv_gt = torch.cat(batch_adv_gt)
        list_train_adv_accuracies.append(sum((batch_adv_predictions == batch_adv_gt).float()).item() / len(batch_adv_predictions))
        grad_adv_norm_epoch.append(torch.tensor(batch_adv_grad_norm).mean().item() if len(batch_adv_grad_norm)>0 else 0.0)
        grad_task_norm_epoch.append(torch.tensor(batch_task_grad_norm).mean().item() if len(batch_task_grad_norm)>0 else 0.0)
      if mt_criterion is not None:
        list_train_mt_losses.append(batch_mt_loss / len(train_loader))
      time_eval = time.perf_counter()
      dict_eval = None
      if val_csv_path is not None:
        dict_eval = self.evaluate(criterion=criterion,
                                  is_test=False,
                                  unique_val_classes=val_unique_classes,
                                  unique_val_subjects=val_unique_subjects,
                                  val_loader=val_loader,
                                  adv_criterion=adv_criterion,
                                  is_coral_loss=is_coral_loss,
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
                                          adv_criterion=adv_criterion,
                                          is_coral_loss=is_coral_loss,
                                          epoch=epoch,
                                          history_val_sample_predictions=None,
                                          **kwargs)
      val_epoch_mean_hsic_sbj.append(dict_eval['val_hsic_subject'] if dict_eval is not None and 'val_hsic_subject' in dict_eval else 0.0)
      val_epoch_mean_hsic_pain.append(dict_eval['val_hsic_pain'] if dict_eval is not None and 'val_hsic_pain' in dict_eval else 0.0)
      val_epoch_mean_center_loss.append(dict_eval.get('val_center_loss', 0.0) if dict_eval is not None else 0.0)
      dict_log_time['eval'] = dict_log_time.get('eval',0) + time.perf_counter()-time_eval
      # print(f'  Evaluation time: {dict_log_time["eval"]:.4f}')
      
      epoch_log_time = time.perf_counter()
      
      if epoch == 0 or \
         helper.SAVE_LAST_EPOCH_MODEL or \
         (  epoch >= min(1, num_epochs//2) 
            and dict_eval is not None 
            and (dict_eval[key_for_early_stopping] < best_eval_loss 
                  if key_for_early_stopping == 'val_loss' 
                  else dict_eval[key_for_early_stopping] > best_eval_loss)):
        best_eval_loss = dict_eval[key_for_early_stopping] if dict_eval is not None else 0.0
        best_model_state = copy.deepcopy(self.state_dict())
        best_model_state = {key: value.cpu() for key, value in best_model_state.items()}
        best_model_epoch = epoch
        best_epoch = True
      
      list_train_losses.append(train_loss / len(train_loader))
      _eval_flags = dict(
        perf_key=key_for_early_stopping,
        log_per_class=helper.LOG_PER_CLASS,
        log_per_subject=helper.LOG_PER_SUBJECT,
        log_confidence=helper.LOG_CONFIDENCE_PREDICTION and not is_resupcon_loss and not is_disentangled_loss,
        log_correlation=not is_resupcon_loss and not is_disentangled_loss and not is_rnc_loss,
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
      if dict_eval is not None and 'val_dict_log_loss_steps' in dict_eval:
        val_dict_log_loss_steps.append(dict_eval['val_dict_log_loss_steps'])
      if adv_criterion is not None:
        list_train_adv_losses.append([batch_adv_loss / len(train_loader), batch_original_loss / len(train_loader)])
        list_train_adv_accuracies.append(sum((batch_adv_predictions == batch_adv_gt).float()).item() / len(batch_adv_predictions))
      if mt_criterion is not None:
        list_train_mt_losses.append(batch_mt_loss / len(train_loader))

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
        # input_dict_std_mean_feats = {
        #   'list_1':epoch_mean_features_log_norm,
        #   'list_2':epoch_std_features_log_norm,
        #   'output_path':None,
        #   'title':f'Mean and std of features',
        #   'point':None,
        #   'ax':ax[1],
        #   'x_label':'Epochs',
        #   'y_label_1':'Mean features',
        #   'y_label_2':'Std features',
        #   'y_label_3':None,
        #   'y_lim_1':[0, max(35, np.max(epoch_mean_features_log_norm))],
        #   'y_lim_2':[0, 0.1],
        #   'y_lim_3':None,
        #   'step_ylim_1':None,
        #   'step_ylim_2':0.02,
        #   'step_ylim_3':None,
        #   'dict_to_string':None,
        #   'color_1':'tab:red',
        #   'color_2':'tab:blue',
        #   'color_3':None,
        # }
        
        tools.plot_losses_and_test_new(**input_dict_loss_acc)
        # tools.plot_losses_and_test_new(**input_dict_std_mean_feats)
        fig.savefig(os.path.join(saving_path, f'loss_logs.png'), dpi=300)
        plt.close(fig)
        print(f'Checkpoint saved at epoch {epoch} to {model_path_epoch} and plotted the loss.')
      
      if not is_resupcon_loss and not is_disentangled_loss and not is_rnc_loss:
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
      else:
        train_dict_precision_recall = {}
        list_train_performance_metric.append(list_train_losses[-1])

      if helper.LOG_CONFIDENCE_PREDICTION and not is_resupcon_loss and not is_disentangled_loss:
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
                           list_grad_norm=total_norm_epoch[epoch],
                           lrs=np.mean(list_batch_lr),
                           wds={k: np.mean([d[k] for d in list_batch_wd]) for k in list_batch_wd[0].keys()})
      if dict_eval is not None:
        self.log_performance(stage='Val',
                            num_epochs=num_epochs,
                            loss=dict_eval['val_loss'],
                            #  dict_kwarg={'acc_per_class':dict_eval['val_accuracy_per_class'],},
                            accuracy=dict_eval['val_accuracy'],)
      if dict_test_as_eval is not None:
        self.log_performance(stage='Test',
                            num_epochs=num_epochs,
                            loss=dict_test_as_eval['val_loss'],
                            accuracy=dict_test_as_eval['val_accuracy'],)
        
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
        print(f'Early stopping triggered at epoch {epoch} with {key_for_early_stopping}: {dict_eval[key_for_early_stopping]:.4f}')
        break
      
      if enable_optuna_pruning and dict_eval is not None and trial is not None:
        trial.report(dict_eval[key_for_early_stopping], epoch)
        # trial.report(list_train_losses[-1], epoch)
        
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
        pid = int(split_key[0])
        key_name = '_'.join(split_key[1:])
        summary_aggregated[key_name] = summary_aggregated.get(key_name,0) + val
      for k,v in summary_aggregated.items():
        print(f'  {k}: {v:.4f} {"s" if "time" in k else "count"}')

      print('---------------------------------------')

      helper.time_profile_dict.clear() # reset logs for next epoch
      if helper.LOG_GRADIENT_PER_MODULE and np.mean(total_norm_epoch[epoch]) < 1e-2:
        print(f'Gradient norm is small (< 1e-1). Stopping training...')
        break
      
      if stop_event.is_set():
        print(f'Stop event is set. Stopping training...')
        break
      
    if saving_path and helper.SAVE_PTH_MODEL:
      print('Load and save best model for next steps...')
      torch.save(best_model_state, os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pt'))
      print(f"Best model weights saved to {os.path.join(saving_path, f'best_model_ep_{best_model_epoch}.pt')}")
    
    train_dict_log_loss_epochs = {}
    if train_dict_log_loss_steps != []:
      train_dict_log_loss_epochs = self.get_dict_loss_per_epoch(train_dict_log_loss_steps, len(train_loader))
      
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
      'epochs_gradient_per_module': epochs_gradient_per_module,
      'history_train_sample_predictions': history_train_sample_predictions,
      'history_val_sample_predictions': history_val_sample_predictions,
      'list_train_accuracy': list_train_accuracy,
      # 'train_accuracy_per_class': train_accuracy_per_class,
      # 'test_accuracy_per_class': test_accuracy_per_class,
      'train_confusion_matricies': list_train_confusion_matricies,
      'best_model_idx': best_model_epoch,
      'best_model_state': best_model_state,
      'metric_for_stopping': metric_for_stopping,
      # 'list_train_macro_accuracy': list_train_performance_metric,
      'list_train_performance_metric': list_train_performance_metric,
      # 'list_val_macro_accuracy': list_val_performance_metric,
      'epochs': epoch,
      'list_mean_total_norm_epoch': np.array([np.mean(ep) if ep else 0.0 for ep in total_norm_epoch]),
      'list_std_total_norm_epoch': np.array([np.std(ep) if ep else 0.0 for ep in total_norm_epoch]),
      'list_max_total_norm_epoch': np.array([np.max(ep) if ep else 0.0 for ep in total_norm_epoch]),
      'list_min_total_norm_epoch': np.array([np.min(ep) if ep else 0.0 for ep in total_norm_epoch]),
      'list_lrs': list_lrs,
      'list_wds': list_wds,
      'train_epoch_mean_hsic_subject': train_epoch_mean_hsic_sbj,
      'train_epoch_mean_hsic_pain': train_epoch_mean_hsic_pain,
      'val_epoch_mean_hsic_subject': val_epoch_mean_hsic_sbj,
      'val_epoch_mean_hsic_pain': val_epoch_mean_hsic_pain,
      'train_epoch_mean_center_loss': train_epoch_mean_center_loss,
      'val_epoch_mean_center_loss': val_epoch_mean_center_loss,
      'optimizer': optimizer.state_dict()['param_groups'],
      'optimizer_config': optimizer_factory.get_config(),
      # 'wd_scheduler': wd_scheduler.get_config() if wd_scheduler else None,
      # 'scheduler': scheduler.state_dict() if scheduler else None,
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
      'list_train_adv_losses': list_train_adv_losses,
      'list_train_mt_losses': list_train_mt_losses,
      'list_train_adv_accuracies': list_train_adv_accuracies,
      'grad_adv_norm_epoch': grad_adv_norm_epoch,
      'grad_task_norm_epoch': grad_task_norm_epoch,
      'train_dict_log_loss_epochs': train_dict_log_loss_epochs,
      'val_dict_log_loss_steps': val_dict_log_loss_steps,
      'train_batch_sampler_name': (train_loader.batch_sampler.__class__.__module__ + "." + train_loader.batch_sampler.__class__.__name__),
      'grad_flow_debug': grad_flow_debug_logs,
      # 'list_samples': list_list_samples,
      # 'list_y': list_list_y
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

  def evaluate(self, val_loader, criterion, unique_val_subjects, unique_val_classes, is_test,is_coral_loss,epoch,history_val_sample_predictions=None,save_log=True,**kwargs):
    # unique_train_val_classes is only for eval but kept the name for compatibility
    device = 'cuda'
    count = 0
    self.to(device) 
    self.eval() 
    is_composite_loss = isinstance(criterion, losses.CompositeLoss)
    is_resupcon_loss = isinstance(criterion, losses.RESupConLoss)
    is_disentangled_loss = isinstance(criterion, losses.DisentangledLoss)
    is_rnc_loss = isinstance(criterion, losses.RnCLossV2) or isinstance(criterion, losses.RnCLoss)
    list_val_epoch_predictions = []
    list_val_ground_truths = []
    l1_error = 0.0
    rmse = 0.0
    # debug_dict = {}
    val_dict_log_loss_steps = []
    epoch_mean_hsic_sbj = []
    epoch_mean_hsic_pain = []
    epoch_center_loss = []
    with torch.no_grad():
      val_loss = 0.0
      loss_per_class = torch.zeros(2,len(val_loader.dataset.get_unique_classes()))
      accuracy_per_class = torch.zeros(2,len(val_loader.dataset.get_unique_classes()))
      subject_loss = torch.zeros(unique_val_subjects.shape[0])
      accuracy_per_subject = torch.zeros(unique_val_subjects.shape[0])
      # subject_batch_count = torch.zeros(unique_val_subjects.shape[0])
      sample_per_subject_count = torch.zeros(unique_val_subjects.shape[0])
      val_confusion_matricies = ConfusionMatrix(task="multiclass",num_classes=val_loader.dataset.get_unique_classes().max().item()+1) # start from 0, so +1
      batch_confidence_prediction_right_mean = []
      batch_confidence_prediction_wrong_mean = []
      batch_confidence_prediction_right_std = []
      batch_confidence_prediction_wrong_std = []
      nr_samples = 0 
      list_batch_size = []
      
      amp_dtype = torch.bfloat16 if helper.AMP_DTYPE == 'bfloat16' else torch.float16
      enable_autocast = helper.AMP_ENABLED
      normalize_labels = bool(kwargs.get('normalize_labels', 0))
      max_label = kwargs.get('max_label', None)
      _label_denorm = float(max_label) if normalize_labels and max_label else 1.0
      return_embeddings = getattr(criterion, 'return_embeddings', False) or \
                          helper.LOG_VIDEO_EMBEDDINGS['enable']
      hsic_enabled = kwargs['HSIC_subject_lambda'] > 0 or kwargs['HSIC_pain_lambda'] > 0
      center_loss_enabled = (kwargs.get('center_loss_fn', None) is not None
                             and kwargs.get('lambda_center_loss', 0.0) > 0)

      for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(val_loader,total=len(val_loader),desc=f'{("Validation" if not is_test else "Test")}'):
        list_batch_size.append(len(batch_y))
        nr_samples += len(batch_y)
        tmp = torch.isin(unique_val_subjects,batch_subjects)
        _,count_sample_per_subject = torch.unique(batch_subjects, return_counts=True)
        sample_per_subject_count[tmp] += count_sample_per_subject
        if isinstance(criterion, torch.nn.HuberLoss):
          batch_y = batch_y.float()
        elif isinstance(criterion, torch.nn.CrossEntropyLoss) or isinstance(criterion, losses.PHuberCrossEntropy):
          batch_y = batch_y.long()
        
        dict_batch_X = {key: value.to(device) if value is not None else None for key, value in dict_batch_X.items()}
        batch_y = batch_y.to(device)
        batch_y_original = batch_y.detach().clone()  # original integer labels for metric logging
        if normalize_labels and max_label:
          batch_y = batch_y.float() / max_label
        if kwargs.get('adv_criterion') is not None:
          batch_subjects = batch_subjects.to(torch.long).to(device)
        # subject_batch_count[tmp] += 1
        helper.LOG_CROSS_ATTENTION['state'] = 'test' if is_test else 'val'
        
        dict_batch_X['list_sample_id'] = sample_id
        dict_batch_X['return_video_emb'] = return_embeddings or hsic_enabled or center_loss_enabled
        if kwargs.get('adv_criterion') is not None:
          dict_batch_X['adv_alpha'] = 0
        with autocast(device_type=device, dtype=amp_dtype, enabled=enable_autocast): # Use autocast for mixed precision training
          dict_out = self(**dict_batch_X)

        if dict_out['logits'].shape[1] == 1 and not (is_resupcon_loss or is_disentangled_loss): # if regression I don't need to keep dim 1
          dict_out['logits'] = dict_out['logits'].squeeze(1)

        if return_embeddings:
          helper.LOG_VIDEO_EMBEDDINGS['embeddings'].append(dict_out['embeddings'].detach().cpu())
          helper.LOG_VIDEO_EMBEDDINGS['labels'].extend(batch_y_original.detach().cpu().numpy().tolist())
          helper.LOG_VIDEO_EMBEDDINGS['sample_ids'].extend(sample_id.cpu().numpy().tolist())
          helper.LOG_VIDEO_EMBEDDINGS['predictions'].extend((dict_out['logits'] * _label_denorm).detach().cpu().numpy().tolist())
        # if kwargs['CCC_loss']:
        #   loss = criterion(outputs, batch_y) + (kwargs['add_CCC_loss'] * kwargs['CCC_loss'](outputs, batch_y) if kwargs['CCC_loss'] else 0.0)
        # else:
        #   loss = criterion(outputs, batch_y)
        dict_out['targets'] = batch_y
        if is_composite_loss:
          loss = criterion(**dict_out)
          loss = loss['total_loss'] if isinstance(loss, dict) else loss
          outputs = dict_out['logits']
        elif is_disentangled_loss:
          batch_subjects = batch_subjects.to(device)
          loss, dict_log_loss = criterion(features=dict_out['logits'],
                                          target_pain=batch_y,
                                          target_subj=batch_subjects)
          val_dict_log_loss_steps.append(dict_log_loss)
        elif is_rnc_loss:
          loss = criterion(features=dict_out['logits'],
                            labels=batch_y)
        else:
          outputs = dict_out['logits']
          loss = criterion(outputs, batch_y)
        if kwargs['HSIC_subject_lambda'] > 0:
          batch_subjects = batch_subjects.to(device)
          hsic_subject = losses.marginal_hsic_loss(
                  z=dict_out['embeddings'],
                  id_labels=batch_subjects,
                  normalize_features=bool(kwargs['HSIC_feat_normalization']),
                  )
          # loss = loss + kwargs['HSIC_subject_lambda'] * hsic_subject
          epoch_mean_hsic_sbj.append(hsic_subject.item())
        if kwargs['HSIC_pain_lambda'] > 0:
          hsic_pain = losses.marginal_hsic_loss(
                  z=dict_out['embeddings'],
                  id_labels=batch_y,
                  normalize_features=bool(kwargs['HSIC_feat_normalization']),
                  )
          # loss = loss - kwargs['HSIC_pain_lambda'] * hsic_pain
          epoch_mean_hsic_pain.append(hsic_pain.item())
        if center_loss_enabled:
          _cl_eval = kwargs['center_loss_fn'](dict_out['embeddings'], batch_y.long())
          epoch_center_loss.append(_cl_eval.item())
        #   adv_logits = dict_out['adv_logits']
        #   adv_loss = kwargs['adv_criterion'](adv_logits, batch_subjects)
        #   loss = loss + adv_loss * kwargs['adversarial_loss_lambda']
        val_loss += loss.item() * _label_denorm
        
        if not is_resupcon_loss and not is_disentangled_loss and not is_rnc_loss:
          # De-normalize outputs for all metric logging when label normalization is active
          _outputs_for_log = (outputs * max_label) if normalize_labels and max_label else outputs
          _batch_y_for_log = batch_y_original  # always original integer scale
          if is_coral_loss:
            _outputs_for_log = proba_to_label(torch.sigmoid(outputs)) # convert probabilities to labels for coral loss
            _batch_y_for_log = torch.sum(batch_y, dim=1) # convert levels to labels for coral loss

          if save_log and helper.LOG_PER_CLASS:
              tools.compute_loss_per_class_(batch_y=_batch_y_for_log,
                                            class_loss=loss_per_class if not is_coral_loss else None,
                                            unique_train_val_classes=unique_val_classes,
                                            outputs=_outputs_for_log,
                                            criterion=criterion,
                                            class_accuracy=accuracy_per_class)
          if save_log and helper.LOG_PER_SUBJECT:
            tools.compute_loss_per_subject_v2_(batch_subjects=batch_subjects,
                                              criterion=criterion,
                                              batch_y=_batch_y_for_log,
                                              outputs=_outputs_for_log,
                                              subject_loss=subject_loss if not is_coral_loss else None,
                                              unique_train_val_subjects=unique_val_subjects,
                                              subject_accuracy=accuracy_per_subject,
                                              unique_train_val_classes=unique_val_classes)
          if save_log and helper.LOG_CONFIDENCE_PREDICTION:
            if self.is_classification and not is_coral_loss:
                tools.compute_confidence_predictions_(list_prediction_right_mean=batch_confidence_prediction_right_mean,
                                                      list_prediction_wrong_mean=batch_confidence_prediction_wrong_mean,
                                                      list_prediction_right_std=batch_confidence_prediction_right_std,
                                                      list_prediction_wrong_std=batch_confidence_prediction_wrong_std,
                                                      gt=_batch_y_for_log, outputs=_outputs_for_log)
          if self.is_classification:
            if is_coral_loss:
              predictions = _outputs_for_log.detach().cpu() # outputs are already labels for coral loss
            else:
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
          batch_y = _batch_y_for_log.detach().cpu()
          if batch_y.dim() > 1:
            batch_y = torch.argmax(batch_y, dim=1).reshape(-1)

          list_val_epoch_predictions.append(predictions.numpy())
          list_val_ground_truths.append(batch_y.numpy())

          # Round + clamp ONLY for the integer-class confusion matrix (regression branch only)
          if not self.is_classification:
            predictions = torch.copysign(torch.floor(torch.abs(predictions) + 0.5), predictions).clamp(0, unique_val_classes.max().item()) # to avoid the banker's rounding implemented as IEEE standard

          val_confusion_matricies.update(predictions, batch_y)
        count += 1
      
      val_loss = val_loss / len(val_loader)
      if not is_resupcon_loss and not is_disentangled_loss and not is_rnc_loss:
        val_confusion_matricies.compute()
        if save_log and helper.LOG_PER_CLASS:
          loss_per_class = loss_per_class[0] / loss_per_class[1] # loss_per_class[0] is loss per class, loss_per_class[1] is total number of samples
          # print(f'VAL Loss mean per class: {torch.mean(loss_per_class)}') 
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
      else:
        dict_precision_recall = {}
        pearson_corr = 0.0
        CCC = 0.0
        ICC = 0.0

    if is_test:
      self.log_performance(stage='Test', loss=val_loss, accuracy=dict_precision_recall.get('accuracy', 0.0))
      return {
        'test_loss': val_loss,
        'test_loss_per_class': loss_per_class,
        'test_loss_per_subject': subject_loss,
        'test_accuracy_per_class': accuracy_per_class,
        'test_accuracy_per_subject': accuracy_per_subject,
        'test_macro_precision': dict_precision_recall.get("macro_precision",0.0),
        'test_accuracy': dict_precision_recall.get("accuracy",0.0),
        'test_confusion_matrix': val_confusion_matricies,
        'test_pearson_correlation': pearson_corr,
        'test_CCC': CCC,
        'test_ICC': ICC,
        'test_hsic_subject': np.mean(epoch_mean_hsic_sbj) if len(epoch_mean_hsic_sbj) > 0 else 0.0,
        'test_hsic_pain': np.mean(epoch_mean_hsic_pain) if len(epoch_mean_hsic_pain) > 0 else 0.0,
        'test_center_loss': np.mean(epoch_center_loss) if epoch_center_loss else 0.0,
        'test_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
        'test_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
        'test_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
        'test_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
        'dict_precision_recall': dict_precision_recall,
        'test_l1_error': l1_error,
        'test_rmse': rmse,
        'test_loss_log_epochs': self.get_dict_loss_per_epoch(val_dict_log_loss_steps, len(val_loader)) if is_disentangled_loss else [],
      }
    else:
      return {
        'val_loss': val_loss,
        'val_loss_per_class': loss_per_class,
        'val_loss_per_subject': subject_loss,
        'val_accuracy_per_class': accuracy_per_class,
        'val_accuracy_per_subject': accuracy_per_subject,
        'val_macro_precision': dict_precision_recall.get("macro_precision",0.0),
        'val_accuracy': dict_precision_recall.get("accuracy",0.0),
        'val_confusion_matrix': val_confusion_matricies,
        'val_pearson_correlation': pearson_corr,
        'val_CCC': CCC,
        'val_ICC': ICC,
        'val_hsic_subject': np.mean(epoch_mean_hsic_sbj) if len(epoch_mean_hsic_sbj) > 0 else 0.0,
        'val_hsic_pain': np.mean(epoch_mean_hsic_pain) if len(epoch_mean_hsic_pain) > 0 else 0.0,
        'val_center_loss': np.mean(epoch_center_loss) if epoch_center_loss else 0.0,
        'val_prediction_confidence_right_mean': np.mean(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
        'val_prediction_confidence_wrong_mean': np.mean(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0,
        'val_prediction_confidence_right_std': np.std(batch_confidence_prediction_right_mean) if len(batch_confidence_prediction_right_mean) > 0 else 0,
        'val_prediction_confidence_wrong_std': np.std(batch_confidence_prediction_wrong_mean) if len(batch_confidence_prediction_wrong_mean) > 0 else 0, 
        'dict_precision_recall': dict_precision_recall,
        'val_l1_error': l1_error,
        'val_rmse': rmse,
        'val_loss_log_epochs': self.get_dict_loss_per_epoch(val_dict_log_loss_steps, len(val_loader)) if is_disentangled_loss else [],
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
  
  def get_dict_loss_per_epoch(self,dict_log_loss_steps, nr_batches):
    train_dict_log_loss_epochs = {}
    for dict_log in dict_log_loss_steps:
      for k,v in dict_log.items():
        if k not in train_dict_log_loss_epochs:
          train_dict_log_loss_epochs[k] = []
        train_dict_log_loss_epochs[k].append(v)
    for k,v in train_dict_log_loss_epochs.items():
      tensor_per_epoch = torch.tensor(v).reshape(-1,nr_batches)
      tensor_per_epoch = tensor_per_epoch.mean(dim=1)
      train_dict_log_loss_epochs[k] = tensor_per_epoch.numpy()
    return train_dict_log_loss_epochs
  
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
      pos_enc=0,
      use_sdpa=True,
      grid_size_pos=None,
      cross_block_after_transformers=True,
      coral_loss=False,
      custom_mlp=False,
      q_k_v_dim=None,
      remove_head=False,
      drop_path_mode='row',
      head_init_path=None,
      adversarial_head=False,
      adversarial_out_dim=None,
      multitask_head=False,
      multitask_num_classes=None,
      backbone: custom_backbone.BackboneBase = None,
      embedding_reduction: helper.EMBEDDING_REDUCTION = None,
      skip_init_weights=False, # used only for model debugging in log_cross_attention_from_model.py script
      complete_block=1,
      type_head=0,
      mlp_num_hidden_layers=2):
    super().__init__(is_classification=True if num_classes > 1 else False)
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
            mlp_num_hidden_layers=mlp_num_hidden_layers,
            complete_block=complete_block,
            drop_path_mode=drop_path_mode,
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
    self.total_grid_area = grid_size_pos[1]*grid_size_pos[2] # spatial_area * spatial_area
    self.grid_size_pos = grid_size_pos
    self.head_init_path = head_init_path
    self.coral_loss = coral_loss
    self.type_head = type_head
    
    if remove_head: # used for Supervised Contrastive Learning
      self.linear = nn.Identity()
      print('\n=== Removed classification head from AttentiveHeadJEPA ===\n')
    elif type_head == 1:
      # MLP followed by linear/coral layer
      mlp = modules.MLP(in_features=embed_dim, 
                         hidden_features=int(embed_dim * mlp_ratio), 
                         out_features=embed_dim,
                         act_layer=nn.GELU,
                         drop=0.0)
      if coral_loss:
        final_layer = CoralLayer(embed_dim, num_classes)
      else:
        final_layer = nn.Linear(embed_dim, num_classes, bias=True)
      self.linear = nn.Sequential(mlp, final_layer)
      print('\n=== Added MLP before the final layer in AttentiveHeadJEPA ===\n')
    elif type_head == 0:
      # type_head == 0 (actual behaviour)
      # Coral loss setup
      if coral_loss:
        self.linear = CoralLayer(embed_dim, num_classes)
      else:
        self.linear = nn.Linear(embed_dim, num_classes, bias=True)
    else: 
      raise NotImplementedError(f'Unknown type_head {type_head}')
    # Aggregator setup if num_queries > 1  
    if num_queries == 1:
      self.aggregator = nn.Identity() 
    else:
      if agg_method == 'max' or agg_method == 'mean':
        self.aggregator = MeanMaxAggregator(method=agg_method,num_query_tokens=num_queries)
      else:
        raise NotImplementedError(f'Unknown aggregation method {agg_method}')
    
      # for param in self.pooler.parameters():
      #   param.requires_grad = False
      # print(f'FROZEN Head weights loaded from {head_init_path}')
    if adversarial_head:
      assert adversarial_out_dim is not None, 'adversarial_out_dim must be specified if adversarial_head is not None'
      self.adversarial_head = nn.Linear(embed_dim, adversarial_out_dim)
    else:
      self.adversarial_head = None
    if multitask_head:
      assert multitask_num_classes is not None, 'multitask_num_classes must be specified when multitask_head is enabled'
      self.multitask_head = nn.Linear(embed_dim, multitask_num_classes)
    else:
      self.multitask_head = None
    # Init weights
    # if not skip_init_weights:
    #   self._initialize_weights()
    # else:
    #   print('===== Skipped head weights initialization =====\n')  
    
  def apply_pos_enc(self, x):
    if self.pos_enc_tensor is None or self.pos_enc_tensor.shape[0] != x.size(1):
      if x.size(1) % self.total_grid_area != 0:
        raise ValueError(f'Input length {x.size(1)} is not divisible by grid size {self.total_grid_area}.\nx.size // self.total_grid_area = {x.size(1)} // {self.total_grid_area} = {x.size(1)//self.total_grid_area}')
      
      if self.pos_enc == 1: # 1D temporal positional encoding  
        self.pos_enc_tensor = pos_embs.get_1d_sincos_pos_embed_torch(embed_dim=x.size(2), # [B, seq_len ,C]
                                                                    grid_size=x.size(1)//self.total_grid_area).unsqueeze(0).to(x.device) # -> [1, nr_clips, emb_dim]
        # Extracted the temporal dimension only, repeat for all spatial tokens and chunks
        self.pos_enc_tensor = torch.repeat_interleave(self.pos_enc_tensor, x.shape[1] // self.pos_enc_tensor.shape[1], dim=1)
      
      elif self.pos_enc == 3: # 3D positional encoding
        grid_depth = x.size(1) // self.total_grid_area # depth => time_dimension
        grid_size = self.grid_size_pos[1] # height * width => spatial_dimension
        self.pos_enc_tensor = pos_embs.get_3d_sincos_pos_embed_torch(embed_dim=x.size(2), # [B, seq_len ,C]
                                                                    grid_depth=grid_depth, # depth => time_dimension
                                                                    grid_size=grid_size # height * width => spatial_dimension
                                                                    ).unsqueeze(0).to(x.device)
      
      elif self.pos_enc == 41: # ONLY QUADRANTS 1D temporal position encoding
        nr_frames = x.size(1) // self.total_grid_area // 4 # 4 quadrants
        self.pos_enc_tensor = pos_embs.get_1d_sincos_pos_embed_torch(embed_dim=x.size(2), # [B, seq_len ,C]
                                                                    grid_size=nr_frames # depth => time_dimension
                                                                    ).unsqueeze(0).to(x.device)
        self.pos_enc_tensor = torch.repeat_interleave(self.pos_enc_tensor, x.shape[1] // self.pos_enc_tensor.shape[1], dim=1)

      elif self.pos_enc == 43: # ONLY QUADRANTS 3D positional encoding  
        grid_size = self.grid_size_pos[1] * 2 
        grid_depth = x.size(1) // self.total_grid_area // 4 
        self.pos_enc_tensor = pos_embs.get_3d_sincos_pos_embed_torch(embed_dim=x.size(2), # [B, seq_len ,C]
                                                                      grid_depth=grid_depth, # depth => time_dimension
                                                                      grid_size=grid_size # height * width => spatial_dimension
                                                                    ).unsqueeze(0).to(x.device)
        # print(f'Pos enc shape (3D): {self.pos_enc_tensor.shape}')
        # print(f'Input shape: {x.shape}')
      if x.shape[1] % self.pos_enc_tensor.shape[1] != 0:
        raise ValueError(f'x.shape not divisible for pos_enc.shape. x size of {x.size(1)} is not divisible by pos_enc size {self.pos_enc.size(1)}')
    # print(self.pos_enc_tensor.shape)
    return x + self.pos_enc_tensor

  def elaborate_feats_from_backbone_v2(self, x, **kwargs):
    """
    Alternative implementation using torch.split for better memory patterns
    """
    # Get backbone features
    
    lengths = kwargs['lengths']
    T, S = x.shape[1], x.shape[2]
    tokens_per_chunk = T * S * S
    
    # Split tensor 
    samples = torch.split(x, lengths.tolist(), dim=0)
    
    # Calculate max length upfront
    max_len = lengths.max().item() * tokens_per_chunk
    batch_size = lengths.shape[0]
    
    # Pre-allocate output
    padded_x = torch.zeros(batch_size, max_len, x.shape[-1], 
                          dtype=x.dtype, device=x.device)
    
    # Process each sample to apply padding
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
  
  def forward(self, x, key_padding_mask=None, return_video_emb=True,**kwargs):
    pid = os.getpid()
    # Extract features from backbone (if needed)
    if self.backbone is not None:
      time_backbone_forward = time.perf_counter()
      x = self.backbone.forward_features(x) # [B * num_chunks, channels, frames=16, height, width] -> [B*chunks, T, S, S, embed_dim]
      helper.time_profile_dict[f'{pid}_backbone_forward_time'] = helper.time_profile_dict.get(f'{pid}_backbone_forward_time',0) + (time.perf_counter() - time_backbone_forward)

      # Apply reduction if needed
      time_reduction = time.perf_counter()
      if self.embedding_reduction !=  helper.EMBEDDING_REDUCTION.NONE:
        x = torch.mean(x,dim=self.embedding_reduction.value,keepdim=True)
      helper.time_profile_dict[f'{pid}_embedding_reduction_time'] = helper.time_profile_dict.get(f'{pid}_embedding_reduction_time',0) + (time.perf_counter() - time_reduction)

      # Reshape x  : [B * num_chunks, T, S, S, embed_dim] -> [B, num_chunks*T*S*S, embed_dim]
      time_reshape = time.perf_counter()
      if torch.all(kwargs['lengths'] == kwargs['lengths'][0]):
        # x = x.reshape(x.shape[0]//kwargs['lengths'][0],-1, x.shape[-1])
        x = x.contiguous().view(x.shape[0]//kwargs['lengths'][0], -1, x.shape[-1]) # [B, num_chunks*T*S*S, embed_dim]
        key_padding_mask = None # no need to compute key_padding_mask if all sequences have the same length
      else:
        x, key_padding_mask = self.elaborate_feats_from_backbone_v2(x, **kwargs)
      helper.time_profile_dict[f'{pid}_embedding_reshape_time'] = helper.time_profile_dict.get(f'{pid}_embedding_reshape_time',0) + (time.perf_counter() - time_reshape)

    # Apply positional encoding if enabled
    time_pos_enc = time.perf_counter()
    if self.pos_enc:
      x = self.apply_pos_enc(x) # add positional encoding if enabled
    helper.time_profile_dict[f'{pid}_pos_enc_time'] = helper.time_profile_dict.get(f'{pid}_pos_enc_time',0) + (time.perf_counter() - time_pos_enc)

    # FORWARD PASS
    time_head = time.perf_counter()
    x,xattn = self.pooler(x,key_padding_mask,helper.LOG_CROSS_ATTENTION['enable'])
    x = x.squeeze(1) # # [B, num_queries=1, C] -> [B, C]
    x = self.aggregator(x)
    helper.time_profile_dict[f'{pid}_head_forward_time'] = helper.time_profile_dict.get(f'{pid}_head_forward_time',0) + (time.perf_counter() - time_head)
    
    # LOG CROSS ATTENTION if enabled
    if helper.LOG_CROSS_ATTENTION['enable'] and xattn is not None:
      self.log_xattn(xattn, **kwargs) 
    
    logits = self.linear(x)

    # Build return dict
    result = {'logits': logits}
    result['embeddings'] = x if return_video_emb else None

    # Adversarial head: only during training, uses gradient reversal
    if self.adversarial_head is not None and self.training:
      reversed_feats = grad_reverse(x, kwargs['adv_alpha'])
      result['adv_logits'] = self.adversarial_head(reversed_feats)
      result['embeddings'] = x

    # Multitask head: only during training, standard gradient flow (no reversal)
    if self.multitask_head is not None and self.training:
      result['mt_logits'] = self.multitask_head(x)
      result['embeddings'] = x

    return result
  
  def set_init_path(self, head_init_path):
    self.head_init_path = head_init_path
    
    
  def _init_weights(self):
    trunc_normal_(self.pooler.query_tokens, std=self.pooler.init_std)
    self.pooler.apply(self.pooler._init_weights)
    self.pooler._rescale_blocks()

  def _initialize_weights(self,init_type='default'):
    if self.backbone is not None:
      # self.backbone.load_pretrained_weights() 
      if hasattr(self.backbone.model, 'adapters'):
        for adapter in self.backbone.model.adapters:
          adapter.init_weights()
      else:
        print(f'No adapters found in the backbone model.')
        
    if init_type == 'default':
      ## Initialize pooler ##
      if self.head_init_path is None:
        self._init_weights()
        print('\nHead weights initialized from scratch\n')
      else: 
        assert self.head_init_path is not None, 'head_init_path must be specified to load head weights'
        state_dict = torch.load(self.head_init_path, weights_only=True)
        # Remove final linear layer for classification/regression 
        layers = list(state_dict.keys())
        for layer_name in layers:
          if layer_name.startswith('linear.'):
            del state_dict[layer_name]
        state_dict = {k.replace('pooler.',''): v for k, v in state_dict.items() if k.startswith('pooler.')}  
        self.pooler.load_state_dict(state_dict)
        print(f'LOADED Head weights from {self.head_init_path}\n  {state_dict.keys()}')
        print(f'\nHead weights loaded from {self.head_init_path}')
        # freeze pooler weights loaded,
        for param in self.pooler.parameters():
          param.requires_grad = False 
        
        
      # Initialize linear layer ##
      def init_layer(m):
        if isinstance(m, nn.Linear):
          torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
          if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
        elif isinstance(m, CoralLayer):
          m.coral_weights.reset_parameters()

      if not isinstance(self.linear, nn.Identity):
        self.linear.apply(init_layer)
        
      if self.adversarial_head is not None:
        torch.nn.init.xavier_uniform_(self.adversarial_head.weight,gain=0.1)
        torch.nn.init.zeros_(self.adversarial_head.bias)
      if self.multitask_head is not None:
        torch.nn.init.xavier_uniform_(self.multitask_head.weight, gain=0.1)
        torch.nn.init.zeros_(self.multitask_head.bias)
      print(f'All trainable params sum: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
      # print(f'======== FROZEN Head weights loaded from {self.head_init_path}========\n')
    else:
      raise NotImplementedError(f'Initialization method {init_type} not implemented')
  
  def log_xattn(self,xattn, **kwargs):
    xattn = xattn.detach().cpu().numpy()
    if f"debug_xattn_{helper.LOG_CROSS_ATTENTION['state']}" not in helper.LOG_CROSS_ATTENTION:
        helper.LOG_CROSS_ATTENTION[f"debug_xattn_{helper.LOG_CROSS_ATTENTION['state']}"] = []
    list_sample_id = kwargs['list_sample_id'].numpy() # problem if use torch.uint16 -> no problem when save results.pkl using np.uint16
    
    helper.LOG_CROSS_ATTENTION[f"debug_xattn_{helper.LOG_CROSS_ATTENTION['state']}"].append((list_sample_id,xattn))
  
  def freeze_pooler(self):
    for param in self.pooler.parameters():
      param.requires_grad = False
    print('FROZEN pooler parameters')
  
  def add_linear_head(self,num_classes, coral_loss=False):
    if coral_loss:
      self.linear = CoralLayer(self.pooler.embed_dim, num_classes)
    else: 
      self.linear = nn.Linear(self.pooler.embed_dim, num_classes, bias=True)
    print(f'Added linear head with {num_classes} classes to AttentiveHeadJEPA')


class GRUHead(BaseHead):
  def __init__(self,
               input_dim,
               hidden_size,
               num_layers,
               output_size=1,
               dropout=0.0,
               layer_norm=False,
               bidirectional=False,
               embedding_reduction=None,
               head_init_path=None,
               backbone: custom_backbone.BackboneBase = None,
               train_backbone=False,
               unfreeze_layers=0,
               skip_init_weights=False,
               num_classes=None,
               **kwargs):
    """
    GRU-based prediction head. Mirrors the public interface of AttentiveHeadJEPA
    so that BaseHead.start_train / BaseHead.evaluate work unchanged.

    Args:
      input_dim:           Embedding dimension D of per-frame features.
      hidden_size:         GRU hidden size.
      num_layers:          Number of stacked GRU layers.
      output_size:         Output dim. 1 = regression, >1 = classification logits.
      dropout:             Inter-layer dropout inside the GRU (ignored if num_layers==1).
      layer_norm:          If True, apply LayerNorm on the pooled GRU output.
      bidirectional:       If True, use a bidirectional GRU (output doubles).
      embedding_reduction: helper.EMBEDDING_REDUCTION enum; reduces spatial/temporal
                           axes of backbone features before the GRU.
      head_init_path:      Optional path to a .pt state_dict to warm-start from.
      backbone:            Backbone module (only when dataset_type == BASE).
      num_classes:         Alias of output_size forwarded by the pipeline; takes
                           precedence over output_size when explicitly > 1.
    """
    if num_classes is not None and num_classes > 1:
      output_size = num_classes
    super().__init__(is_classification=(output_size > 1))

    self.input_dim = input_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.output_size = output_size
    self.bidirectional = bidirectional
    self.embedding_reduction = embedding_reduction if embedding_reduction is not None else helper.EMBEDDING_REDUCTION.NONE
    self.head_init_path = head_init_path
    self.backbone = backbone

    self.gru = nn.GRU(
      input_size=input_dim,
      hidden_size=hidden_size,
      num_layers=num_layers,
      dropout=dropout if num_layers > 1 else 0.0,
      batch_first=True,
      bidirectional=bidirectional,
    )

    self.input_norm = nn.LayerNorm(input_dim) if layer_norm else nn.Identity()

    out_dim = hidden_size * (2 if bidirectional else 1)
    self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
    self.linear = nn.Linear(out_dim, output_size, bias=True)

    # if not skip_init_weights:
    #   print('\nInitializing GRUHead weights...\n')
    #   self._initialize_weights(init_type='default')
    #   print('GRUHead weights initialization complete.\n')

  def _reduce_to_sequence(self, x):
    """
    Collapse (B, T, S, S, D) -> (B, T, D) by honoring self.embedding_reduction
    for spatial/temporal axes and mean-pooling any leftover non-temporal dims.
    If x is already (B, T, D), return unchanged.

    Args:
      x: Tensor of shape (B, T, D) or (B, T, S, S, D).

    Returns:
      Tensor of shape (B, T, D).
    """
    if x.dim() == 3:
      return x
    if self.embedding_reduction != helper.EMBEDDING_REDUCTION.NONE:
      x = torch.mean(x, dim=self.embedding_reduction.value, keepdim=True)
    while x.dim() > 3:
      x = torch.mean(x, dim=2)
    return x

  def forward(self, x, key_padding_mask=None, return_video_emb=True, **kwargs):
    """
    Args:
      x:                Pre-extracted features (B, T, D) or (B, T, S, S, D),
                        or raw video clips when self.backbone is not None.
      key_padding_mask: Optional bool mask (B, T). True = valid token.
      return_video_emb: If True, include the pooled embedding in the output dict.

    Returns:
      dict with 'logits' (B, output_size) and optionally 'embeddings' (B, out_dim).
    """
    if self.backbone is not None:
      x = self.backbone.forward_features(x)

    x = self._reduce_to_sequence(x)
    x = self.input_norm(x)

    if (key_padding_mask is not None and key_padding_mask.dim() == 2
        and key_padding_mask.shape[0] == x.shape[0]
        and key_padding_mask.shape[1] == x.shape[1]):
      lengths = key_padding_mask.sum(dim=1).clamp(min=1).cpu()
      packed = torch.nn.utils.rnn.pack_padded_sequence(
        x, lengths, batch_first=True, enforce_sorted=False)
      _, h_n = self.gru(packed)
    else:
      _, h_n = self.gru(x)

    if self.bidirectional:
      pooled = torch.cat([h_n[-2], h_n[-1]], dim=-1)
    else:
      pooled = h_n[-1]

    pooled = self.norm(pooled)
    logits = self.linear(pooled)

    result = {'logits': logits}
    result['embeddings'] = pooled if return_video_emb else None
    return result

  def set_init_path(self, head_init_path):
    """Record a pretrained-weights path to load during _initialize_weights."""
    self.head_init_path = head_init_path

  def _initialize_weights(self, init_type='default'):
    """Orthogonal init for GRU recurrent weights, xavier for the final linear."""
    if init_type == 'default':
      for name, param in self.gru.named_parameters():
        if 'weight_ih' in name:
          nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
          nn.init.orthogonal_(param)
        elif 'bias' in name:
          nn.init.zeros_(param)
      nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
      if self.linear.bias is not None:
        nn.init.zeros_(self.linear.bias)

      if self.head_init_path is not None and os.path.isfile(self.head_init_path):
        state_dict = torch.load(self.head_init_path, weights_only=True)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('linear.')}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f'LOADED GRUHead weights from {self.head_init_path}')
        print(f'  missing: {missing}\n  unexpected: {unexpected}')
      print(f'All trainable params sum: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
    else:
      raise NotImplementedError(f'Initialization method {init_type} not implemented')


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


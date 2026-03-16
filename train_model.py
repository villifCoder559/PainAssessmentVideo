import os
import time
import math
import argparse
import sys
import platform
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import optunahub
import yaml
from coral_pytorch.losses import coral_loss

# Custom imports


import custom.scripts as scripts
import custom.helper as helper
import custom.loss as losses
from custom.helper import (
  CLIPS_REDUCTION,
  EMBEDDING_REDUCTION,
  MODEL_TYPE,
  SAMPLE_FRAME_STRATEGY,
  HEAD,
  GLOBAL_PATH,
)
from custom.head import earlyStoppingAccuracy, earlyStoppingLoss
from custom.tools import plot_masked_attention, get_pth_path_from_project_folder
from custom.optimizers import OptimizerFactory
from custom.logger import setup_logger

os.environ["OPTUNA_DISABLE_TELEMETRY"] = "1"

LOG_SCALE_OPTUNA_HYPERS = ['lr', 'regulariz_lambda_L1', 'regulariz_lambda_L2']


def get_optimizer(opt):
  """Get optimizer class from string name.

  Args:
    opt (str): Name of the optimizer (adam, sgd, adamw).

  Returns:
    class: The optimizer class.

  Raises:
    ValueError: If optimizer is not found.
  """
  optimizers = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'adamw': optim.AdamW
  }
  opt_lower = opt.lower()
  if opt_lower not in optimizers:
    raise ValueError(f'Optimizer not found: {opt}. Valid options: {list(optimizers.keys())}')
  return optimizers[opt_lower]


def get_sampling_frame_strategy(strategy):
  """Get sampling strategy from string name.

  Args:
    strategy (str): Name of the strategy.

  Returns:
    SAMPLE_FRAME_STRATEGY: The enum value for the strategy.

  Raises:
    ValueError: If strategy is not found.
  """
  strategy_lower = strategy.lower()
  if strategy_lower == SAMPLE_FRAME_STRATEGY.UNIFORM.value:
    return SAMPLE_FRAME_STRATEGY.UNIFORM
  elif strategy_lower == SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW.value:
    return SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  elif strategy_lower == SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING.value:
    return SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING
  elif strategy_lower == SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING.value:
    return SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING
  else:
    raise ValueError(
      f'Sampling strategy not found: {strategy}. Valid options: {list(SAMPLE_FRAME_STRATEGY)}'
    )


def get_loss(loss_name, dict_args=None):
  """Get loss function from string name.

  Args:
    loss_name (str): Name of the loss function.
    dict_args (dict, optional): Arguments required for specific losses.

  Returns:
    nn.Module or callable: The loss function.

  Raises:
    ValueError: If loss is not found or required args are missing.
  """
  loss_lower = loss_name.lower()
  if loss_lower == 'l1':
    return nn.L1Loss()
  elif loss_lower == 'ce_weight':
    if dict_args is None or 'class_weights' not in dict_args:
      raise ValueError("class_weights must be provided for ce_weight loss")
    return nn.CrossEntropyLoss(
      weight=torch.tensor(dict_args['class_weights'], dtype=torch.float32, device='cuda')
    )
  elif loss_lower == 'huber':
    return nn.HuberLoss(delta=dict_args['delta_huber'])
  elif loss_lower == 'l2':
    return nn.MSELoss()
  elif loss_lower == 'ce':
    return nn.CrossEntropyLoss()
  elif loss_lower == 'cdw_ce':
    if dict_args is None:
      raise ValueError("dict_args is required for 'cdw_ce' loss.")
    return losses.CDW_CELoss(
      num_classes=dict_args['num_classes'],
      alpha=dict_args['alpha'],
      delta=dict_args['cdw_ce_delta'],
      transform=dict_args['transform']
    )
  elif loss_lower == 'huber_ce':
    return losses.PHuberCrossEntropy(
      tau=dict_args['delta_huber'] 
    )
  elif loss_lower == 'sim_loss':
    if dict_args is None:
      raise ValueError("dict_args is required for 'sim_loss'.")
    return losses.SimLoss(
      number_of_classes=dict_args['num_classes'],
      reduction_factor=dict_args['sim_loss_reduction'],
      device='cuda'
    )
  elif loss_lower == 'contrastive_reg':
    # if ',' not in dict_args['contrastive_loss_temp']:
    dict_args['contrastive_loss_temp'] = float(dict_args['contrastive_loss_temp'])
    dict_args['contrastive_loss_theta'] = float(dict_args['contrastive_loss_theta'])
    dict_args['contrastive_lambda_weight'] = float(dict_args['contrastive_lambda_weight']) 
    dict_args['spearman_reg_strength'] = float(dict_args['spearman_reg_strength'])
    return losses.RESupConLoss(
      temperature=dict_args['contrastive_loss_temp'],
      theta=dict_args['contrastive_loss_theta'],
      lambda_weight=dict_args['contrastive_lambda_weight'],
      spearman_reg=dict_args['spearman_reg'],
      spearman_reg_strength=dict_args['spearman_reg_strength'],
    )
  elif loss_lower == 'only_contrastive':
    dict_args['only_contrastive_temp'] = float(dict_args['only_contrastive_temp'])
    dict_args['only_contrastive_theta'] = float(dict_args['only_contrastive_theta'])
    return losses.SupConLossModified(
      temperature=dict_args['only_contrastive_temp'],
      theta=dict_args['only_contrastive_theta']
    )
  elif loss_lower == 'coral':
    return coral_loss
  elif loss_lower == 'rncloss':
    dict_args['rncloss_temp'] = float(dict_args['rncloss_temp'])
    return losses.RnCLossV2(temperature=dict_args['rncloss_temp'])
  else:
    raise ValueError(
      f'Loss not found: {loss_name}. Valid options: l1, l2, ce, cdw_ce, sim_loss, coral, contrastive_reg, rncloss'
    )


def get_composite_loss_module(loss_types, losses_weights, **kwargs):
  """Factory for composite loss objects.

  Args:
    loss_types (list): List of loss type strings.
    losses_weights (list): List of weights for each loss.
    **kwargs: Additional arguments for specific losses (e.g., delta_huber).

  Returns:
    losses.CompositeLoss: The composite loss module.

  Raises:
    ValueError: If an unsupported loss type is provided.
  """
  list_losses = []
  for loss_type, weight in zip(loss_types, losses_weights):
    if loss_type == 'l1':
      list_losses.append({'type': 'l1', 'loss_weight': weight})
    elif loss_type == 'l2':
      list_losses.append({'type': 'l2', 'loss_weight': weight})
    elif loss_type == 'huber':
      list_losses.append({
        'type': 'huber',
        'delta_huber': kwargs['delta_huber'],
        'loss_weight': weight
      })
    elif loss_type == 'ce':
      list_losses.append({'type': 'ce', 'loss_weight': weight})
    elif loss_type == 'ce_weight':
      class_weights = kwargs.get('class_weights', None)
      if class_weights is None:
        raise ValueError("class_weights must be provided for ce_weight loss")
      list_losses.append({
        'type': 'ce_weight',
        'class_weights': class_weights,
        'loss_weight': weight
      })
    elif loss_type == 'contrastive_reg':
      raise ValueError("Contrastive regression loss not supported in composite loss.")
      list_losses.append({
        'type': 'contrastive_reg',
        'contrastive_loss_temp': kwargs['contrastive_loss_temp'],
        'contrastive_loss_theta': kwargs['contrastive_loss_theta'],
        'contrastive_lambda_weight': kwargs['contrastive_lambda_weight'],
        'loss_weight': weight
      })
    else:
      raise ValueError(f"Unsupported loss type in composite loss: {loss_type}")

  return losses.CompositeLoss(list_losses)

def get_disentanglement_loss(loss_type, split_idx, lambdas,ortho_lambda, dict_args=None):
  loss_fns = [get_loss(lt, dict_args) for lt in loss_type]
  disentanglement_loss = losses.DisentangledLoss(
    lambdas=lambdas,
    loss_fns=loss_fns,
    split_idx=split_idx,
    ortho_lambda=ortho_lambda)
  return disentanglement_loss


def get_class_weights(csv_path):
  """Calculates inverse class frequency weights from a CSV dataset.

  Args:
    csv_path (str): Path to the CSV file.

  Returns:
    numpy.ndarray: Array of weights.
  """
  df = pd.read_csv(csv_path, sep='\t')
  class_counts = df['class_id'].value_counts(normalize=True).sort_index()
  class_weights = 1.0 / class_counts.values  # Inverse of frequency
  return class_weights


def get_mean_val_accuracy(results, optuna_direction, use_median=False, min_epoch_threshold=0):
  """Calculates the mean (or median) validation metric across folds.

  Args:
    results (dict): Dictionary containing experiment results.
    optuna_direction (int): 1 for minimize, 2 for maximize (used for penalty return).
    use_median (bool, optional): Whether to use median instead of mean. Defaults to False.
    min_epoch_threshold (int, optional): Minimum epoch required. Defaults to 0.

  Returns:
    float: The aggregated metric value.
  """
  list_val_metric = []
  list_best_epochs = []
  for k_fold, dict_log_k_fold in results['results'].items():
    if 'final' not in k_fold:
      best_epoch = dict_log_k_fold['train_val']['best_model_idx']
      list_val_metric.append(
        dict_log_k_fold['train_val']['list_val_performance_metric'][best_epoch]
      )
      list_best_epochs.append(best_epoch)

  idx_median = np.argsort(list_val_metric)[len(list_val_metric) // 2]
  best_epoch_median = list_best_epochs[idx_median]

  if min_epoch_threshold > 0:
    if best_epoch_median < min_epoch_threshold:
      print(
        f"Trial rejected due to best epoch {best_epoch_median} < min_epoch_threshold {min_epoch_threshold}"
      )
      return float('inf') if optuna_direction == 1 else float('-inf')

  if use_median:
    return np.median(list_val_metric)
  else:
    return np.mean(list_val_metric)


def get_sampler_module(sampler_name, kwargs=None):
  """Returns the Optuna sampler module based on the name.

  Args:
    sampler_name (str): Name of the sampler (tpe, random, auto, grid).
    kwargs (dict, optional): Arguments for GridSampler.

  Returns:
    optuna.samplers.BaseSampler: The sampler instance.

  Raises:
    ValueError: If sampler name is invalid.
  """
  sampler_lower = sampler_name.lower()
  if sampler_lower == 'tpe':
    return optuna.samplers.TPESampler()
  elif sampler_lower == 'random':
    return optuna.samplers.RandomSampler()
  elif sampler_lower == 'auto':
    sampler_module = optunahub.load_module(package="samplers/auto_sampler")
    return sampler_module.AutoSampler()
  elif sampler_lower == 'grid':
    search_space = {k: v for k, v in kwargs.items() if isinstance(v, list) and k != 'stop'}
    sampler_module = optuna.samplers.GridSampler(search_space=search_space)
    print(f'\n Grid search trials: {len(sampler_module._all_grids)}\n')
    return sampler_module
  else:
    raise ValueError(
      f"Invalid sampler name: {sampler_name}. Must be one of ['tpe', 'random', 'auto', 'grid']"
    )


def _suggest(trial: optuna.trial.Trial, name, values, categorical):
  """Suggests a hyperparameter value based on available options.

  Args:
    trial (optuna.trial.Trial): The current trial.
    name (str): The name of the hyperparameter.
    values (list): The list of possible values or range.
    categorical (int/bool): Whether to treat as categorical.

  Returns:
    Any: The suggested value.
  """
  if categorical:
    return trial.suggest_categorical(name, values)

  if len(values) == 1:
    print(f"Only one value provided for {name}. Using {values[0]}.")
    return trial.suggest_categorical(name, values)

  low, high = values[0], values[-1]
  if low > high:
    low, high = high, low

  if isinstance(low, int) and isinstance(high, int):
    return trial.suggest_int(name, low=low, high=high)
  else:
    return trial.suggest_float(
      name, low=low, high=high, log=True if name in LOG_SCALE_OPTUNA_HYPERS else False
    )


def _suggest_scheduler_params(trial, kwargs, opt, regulariz_lambda_L2):
  """Helper to suggest scheduler and optimizer related parameters."""
  scheduler_name = trial.suggest_categorical('scheduler_name', kwargs['scheduler_name'])
  
  # Suggest scheduler-specific params
  config = {
    'optimizer_name': opt,
    'scheduler_name': scheduler_name,
    'wd_scheduler_name': trial.suggest_categorical('wd_scheduler_name', kwargs['wd_scheduler_name']),
    'epochs': kwargs['ep'],
    'exclude_bias_wd': trial.suggest_categorical('exclude_bias_wd', kwargs['exclude_bias_wd']),
    'warm_up_epochs': _suggest(trial, 'warm_up_epochs', kwargs['warm_up_epochs'], kwargs['optuna_categorical']),
    'warm_up_scheduler': trial.suggest_categorical('warm_up_scheduler', kwargs['warm_up_scheduler']),
    'warm_up_start_factor': _suggest(trial, 'warm_up_start_factor', kwargs['warm_up_start_factor'], kwargs['optuna_categorical']),
    'min_lr': _suggest(trial, 'min_lr', kwargs['min_lr'], kwargs['optuna_categorical']),
    'min_wd': _suggest(trial, 'min_wd', kwargs['min_wd'], kwargs['optuna_categorical']),
    'first_restart_epochs': _suggest(trial, 'first_restart_epochs', kwargs['first_restart_epochs'], kwargs['optuna_categorical']),
    'multiplier_restart': _suggest(trial, 'multiplier_restart', kwargs['multiplier_restart'], kwargs['optuna_categorical']),
    'step_size_epochs': _suggest(trial, 'step_size_epochs', kwargs['step_size_epochs'], kwargs['optuna_categorical']),
    'gamma': _suggest(trial, 'gamma', kwargs['gamma'], kwargs['optuna_categorical']),
    'onecycle_pct_start': _suggest(trial, 'onecycle_pct_start', kwargs['onecycle_pct_start'], kwargs['optuna_categorical']),
    'onecycle_max_lr': _suggest(trial, 'onecycle_max_lr', kwargs['onecycle_max_lr'], kwargs['optuna_categorical']),
    'onecycle_div_factor': _suggest(trial, 'onecycle_div_factor', kwargs['onecycle_div_factor'], kwargs['optuna_categorical']),
    'onecycle_final_div_factor': _suggest(trial, 'onecycle_final_div_factor', kwargs['onecycle_final_div_factor'], kwargs['optuna_categorical']),
    'onecycle_anneal_strategy': trial.suggest_categorical('onecycle_anneal_strategy', kwargs['onecycle_anneal_strategy']),
    'lr': _suggest(trial, 'lr', kwargs['lr'], kwargs['optuna_categorical']),
    'weight_decay': regulariz_lambda_L2,
  }
  return config


def objective(trial: optuna.trial.Trial, original_kwargs):
  """Objective function for Optuna hyperparameter optimization.

  Args:
    trial (optuna.trial.Trial): The Optuna trial object.
    original_kwargs (dict): The dictionary of parsed arguments.

  Returns:
    float: The result metric (validation accuracy).
  """
  kwargs = deepcopy(original_kwargs)
  model_type = MODEL_TYPE.get_model_type(kwargs['mt'])
  epochs = kwargs['ep']

  # --- Suggest Common Params ---
  lr = _suggest(trial, 'lr', kwargs['lr'], kwargs['optuna_categorical'])
  opt = trial.suggest_categorical('opt', kwargs['opt'])
  latent_coefficient = trial.suggest_categorical('latent_coefficient', kwargs['latent_coefficient'])
  type_group = trial.suggest_categorical('type_group', kwargs['type_group'])
  init_network = trial.suggest_categorical('init_network', kwargs['init_network'])
  batch_train = trial.suggest_categorical('batch_train', kwargs['batch_train'])
  regulariz_lambda_L1 = _suggest(trial, 'regulariz_lambda_L1', kwargs['regulariz_lambda_L1'], kwargs['optuna_categorical'])
  regulariz_lambda_L2 = _suggest(trial, 'regulariz_lambda_L2', kwargs['regulariz_lambda_L2'], kwargs['optuna_categorical'])
  label_smooth = trial.suggest_categorical('label_smooth', kwargs['label_smooth'])
  concatenate_temp_dim = trial.suggest_categorical('concatenate_temp_dim', kwargs['concatenate_temp_dim'])
  concatenate_quadrants = trial.suggest_categorical('concatenate_quadrants', kwargs['concatenate_quadrants'])
  stratified_training = trial.suggest_categorical('stratified_training', kwargs['stratified_training'])
  only_augments = trial.suggest_categorical('only_augments', kwargs['only_augments'])
  sampler_augmented_only_types = trial.suggest_categorical('sampler_augmented_only_types', kwargs['sampler_augmented_only_types'])
  sampler_loader_type = trial.suggest_categorical('sampler_loader_type', kwargs['sampler_loader_type'])
  min_subjects_per_level = trial.suggest_categorical('min_subjects_per_level', kwargs['min_subjects_per_level'])

  # --- Suggest Loss Params ---
  cdw_ce_alpha = _suggest(trial, 'cdw_ce_alpha', kwargs['cdw_ce_alpha'], kwargs['optuna_categorical'])
  cdw_ce_power_transform = trial.suggest_categorical('cdw_ce_power_transform', kwargs['cdw_ce_transform'])
  cdw_ce_delta = _suggest(trial, 'cdw_ce_delta', kwargs['cdw_ce_delta'], kwargs['optuna_categorical'])
  soft_labels = _suggest(trial, 'soft_labels', kwargs['soft_labels'], kwargs['optuna_categorical'])
  sim_loss_reduction = _suggest(trial, 'sim_loss_reduction', kwargs['sim_loss_reduction'], kwargs['optuna_categorical'])
  contrastive_loss_temp = _suggest(trial, 'contrastive_loss_temp', kwargs['contrastive_loss_temp'], kwargs['optuna_categorical'])
  contrastive_loss_theta = _suggest(trial, 'contrastive_loss_theta', kwargs['contrastive_loss_theta'], kwargs['optuna_categorical'])
  contrastive_lambda_weight = trial.suggest_categorical('contrastive_lambda_weight', kwargs['contrastive_lambda_weight'])
  contrastive_spearman_reg = trial.suggest_categorical('spearman_reg', kwargs['spearman_reg'])
  contrastive_spearman_reg_strength = _suggest(trial, 'spearman_reg_strength', kwargs['spearman_reg_strength'], kwargs['optuna_categorical'])
  delta_huber = _suggest(trial, 'delta_huber', kwargs['delta_huber'], kwargs['optuna_categorical'])
  rncloss_temp = _suggest(trial, 'rncloss_temp', kwargs['rncloss_temp'], kwargs['optuna_categorical'])
  only_contrastive_temp = _suggest(trial, 'only_contrastive_temp', kwargs['only_contrastive_temp'], kwargs['optuna_categorical'])
  only_contrastive_theta = _suggest(trial, 'only_contrastive_theta', kwargs['only_contrastive_theta'], kwargs['optuna_categorical'])
  HSIC_subject_lambda = _suggest(trial, 'HSIC_subject_lambda', kwargs['HSIC_subject_lambda'], kwargs['optuna_categorical'])
  HSIC_pain_lambda = _suggest(trial, 'HSIC_pain_lambda', kwargs['HSIC_pain_lambda'], kwargs['optuna_categorical'])
  HSIC_feat_normalization = trial.suggest_categorical('HSIC_feat_normalization', kwargs['HSIC_feat_normalization'])

  # --- Suggest Training Strategy Params ---
  perfect_bal_strategy = trial.suggest_categorical('perfect_bal_strategy', kwargs['perfect_bal_strategy'])
  balance_batches = trial.suggest_categorical('balance_batches', kwargs['balance_batches'])
  num_clips_per_video = trial.suggest_categorical('num_clips_per_video', kwargs['num_clips_per_video'])
  sample_frame_strategy = trial.suggest_categorical('sample_frame_strategy', kwargs['sample_frame_strategy'])
  stride_inside_window = trial.suggest_categorical('stride_inside_window', kwargs['stride_inside_window'])
  use_sdpa = trial.suggest_categorical('use_sdpa', kwargs['use_sdpa'])
  train_backbone = trial.suggest_categorical('train_backbone', kwargs['train_backbone'])
  unfreeze_layers = trial.suggest_categorical('unfreeze_layers', kwargs['unfreeze_layers'])
  if train_backbone == 1 and unfreeze_layers is None:
    raise ValueError("unfreeze_layers must be explicitly set when train_backbone=1.")
  if unfreeze_layers is not None and unfreeze_layers > 0 and train_backbone == 0:
    raise ValueError("unfreeze_layers > 0 requires train_backbone=1 for this trial.")
  debug_grad_flow = trial.suggest_categorical('debug_grad_flow', kwargs['debug_grad_flow'])
  debug_grad_flow_batches = trial.suggest_categorical('debug_grad_flow_batches', kwargs['debug_grad_flow_batches'])
  
  if use_sdpa and not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    raise ValueError("SDPA is not available in this PyTorch version. Set use_sdpa=0 or update PyTorch")
  
  add_CCC_loss = _suggest(trial, 'add_CCC_loss', kwargs['add_CCC_loss'], kwargs['optuna_categorical'])

  # --- Suggest Augmentation Params ---
  dict_augmented = {
    'hflip': _suggest(trial, 'hflip', kwargs['hflip'], kwargs['optuna_categorical']),
    'jitter': _suggest(trial, 'color_jitter', kwargs['jitter'], kwargs['optuna_categorical']),
    'rotation': _suggest(trial, 'rotation', kwargs['rotation'], kwargs['optuna_categorical']),
    'latent_basic': _suggest(trial, 'latent_basic', kwargs['latent_basic'], kwargs['optuna_categorical']),
    'latent_masking': _suggest(trial, 'latent_masking', kwargs['latent_masking'], kwargs['optuna_categorical']),
    'shift': _suggest(trial, 'shift_augm', kwargs['shift_augm'], kwargs['optuna_categorical']),
  }
  
  consider_only_lasts_n_chunks = trial.suggest_categorical('consider_only_lasts_n_chunks', kwargs['consider_only_lasts_n_chunks'])
  filtered_augm_n_keep = trial.suggest_categorical('filtered_augm_n_keep', kwargs['filtered_augm_n_keep'])
  filtered_augm_strategy = trial.suggest_categorical('filtered_augm_strategy', kwargs['filtered_augm_strategy'])

  # --- Scheduler Config ---
  # Construct config manually here to ensure we use the sampled 'lr' and 'regulariz_lambda_L2'
  scheduler_config_dict = _suggest_scheduler_params(trial, kwargs, opt, regulariz_lambda_L2)
  # Ensure correct lr is in config
  scheduler_config_dict['lr'] = lr
  OptimizerFactory.check_schedulers_parms(scheduler_name=scheduler_config_dict['scheduler_name'], config=scheduler_config_dict)

  # --- Setup Loss Object ---
  composite_loss = None
  loss = None
  loss_args = {
    'delta_huber': delta_huber,
    'class_weights': get_class_weights(kwargs['csv']),
    'contrastive_loss_temp': contrastive_loss_temp,
    'contrastive_loss_theta': contrastive_loss_theta,
    'contrastive_lambda_weight': contrastive_lambda_weight,
    'spearman_reg': contrastive_spearman_reg,
    'spearman_reg_strength': contrastive_spearman_reg_strength,
    'rncloss_temp': rncloss_temp,
    'only_contrastive_temp': only_contrastive_temp,
    'only_contrastive_theta': only_contrastive_theta,
  }
  remove_head = False
  if kwargs['loss']:
    loss = trial.suggest_categorical('loss', kwargs['loss'])
    remove_head = True if (loss == 'contrastive_reg' or loss == 'rncloss') else False
  elif kwargs['composite_loss'][0] is not None:
    composite_loss = trial.suggest_categorical('composite_loss', kwargs['composite_loss'])
    composite_loss = composite_loss.split(',')
    composite_loss_lambda = trial.suggest_categorical('composite_loss_lambda', kwargs['composite_loss_lambda'])
    composite_loss_lambda = [float(l) for l in composite_loss_lambda.split(',')]
    composite_loss = get_composite_loss_module(composite_loss, composite_loss_lambda, **loss_args)
  elif kwargs['disent_loss_p_s'][0] is not None:
    remove_head = True
  print(f'*********\nREMOVE HEAD: {remove_head}\n*********')
  if loss and loss == 'cdw_ce' and label_smooth > 0:
    raise ValueError('Label smoothing not supported for CDW loss. Use label_smooth=0.')

  # --- Setup Model Input Dims and Head ---
  emb_dim = MODEL_TYPE.get_embedding_size(kwargs['mt'])
  input_dim = emb_dim
  if concatenate_temp_dim:
    input_dim *= 8  # 8 temporal segments
  elif concatenate_quadrants:
    input_dim *= 4  # 4 quadrants
  
  # --- Determine Num Classes ---
  num_classes = 0
  if composite_loss is not None:
    num_classes = 1  # regression
    print(f"\nComposite loss detected. Setting num_classes to 1 for regression.\n")
    if 'ce' in [l['type'] for l in composite_loss.config_losses]:
      raise ValueError("Classification loss (ce) not supported in composite loss for regression.")
  elif loss is not None:
    if loss in ['l1', 'l2', 'huber']:
      num_classes = 1
      print(f"\nRegression task detected. Setting num_classes to 1 for {loss} loss.")
    else:
      num_classes = pd.read_csv(kwargs['csv'], sep='\t')['class_id'].nunique()
  
  # --- Head Specific Params ---
  head_name = kwargs['head']
  num_adapters = trial.suggest_categorical('num_adapters', kwargs['num_adapters'])
  if num_adapters > 0:
    adapter_dict = {
      'type': trial.suggest_categorical('adapter_type', kwargs['adapter_type']),
      'mlp_ratio': trial.suggest_categorical('adpater_mlp_ratio', kwargs['adpater_mlp_ratio']),
      'kernel_size': trial.suggest_categorical('adapter_kernel_size', kwargs['adapter_kernel_size']),
      'dilation': trial.suggest_categorical('adapter_dilation', kwargs['adapter_dilation']),
      'num_adapters': num_adapters,
    }
  else:
    adapter_dict = None

  head_params = {}
  head_enum = None
  head_dependent_add_kwargs = {}

  if head_name.upper() == 'GRU':
    GRU_hidden_size = _suggest(trial, 'GRU_hidden_size', kwargs['GRU_hidden_size'], kwargs['optuna_categorical'])
    GRU_num_layers = _suggest(trial, 'GRU_num_layers', kwargs['GRU_num_layers'], kwargs['optuna_categorical'])
    GRU_output_size = kwargs['GRU_output_size']
    layer_norm = trial.suggest_categorical('layer_norm', kwargs['layer_norm'])
    GRU_dropout = _suggest(trial, 'GRU_dropout', kwargs['GRU_dropout'], kwargs['optuna_categorical'])
    GRU_round_output_loss = trial.suggest_categorical('GRU_round_output_loss', kwargs['GRU_round_output_loss'])

    head_params = {
      'input_dim': input_dim,
      'hidden_size': GRU_hidden_size,
      'num_layers': GRU_num_layers,
      'output_size': GRU_output_size,
      'layer_norm': layer_norm,
      'dropout': GRU_dropout,
    }
    head_enum = HEAD.GRU
    num_classes = GRU_output_size  # Override for GRU
    head_dependent_add_kwargs = {'is_round_output_loss': GRU_round_output_loss}

  elif head_name.upper() == 'ATTENTIVE_JEPA':
    adversarial_head = trial.suggest_categorical('adversarial_head', kwargs['adversarial_head'])
    adv_alpha_strategy = trial.suggest_categorical('adv_alpha_strategy', kwargs['adv_alpha_strategy'])
    adv_alpha_gamma = _suggest(trial, 'adv_alpha_gamma', kwargs['adv_alpha_gamma'], kwargs['optuna_categorical'])
    adversarial_loss_lambda = _suggest(trial, 'adversarial_loss_lambda', kwargs['adversarial_loss_lambda'], kwargs['optuna_categorical'])
    nr_blocks = _suggest(trial, 'nr_blocks', kwargs['nr_blocks'], kwargs['optuna_categorical'])
    num_heads = trial.suggest_categorical('num_heads', kwargs['num_heads'])
    num_cross_head = trial.suggest_categorical('num_cross_head', kwargs['num_cross_head'])
    model_dropout = _suggest(trial, 'model_dropout', kwargs['model_dropout'], kwargs['optuna_categorical'])
    drop_attn = _suggest(trial, 'drop_attn', kwargs['drop_attn'], kwargs['optuna_categorical'])
    drop_residual = _suggest(trial, 'drop_residual', kwargs['drop_residual'], kwargs['optuna_categorical'])
    mlp_ratio = trial.suggest_categorical('mlp_ratio', kwargs['mlp_ratio'])
    pos_enc = trial.suggest_categorical('pos_enc', kwargs['pos_enc'])
    num_queries = _suggest(trial, 'num_queries', kwargs['num_queries'], kwargs['optuna_categorical'])
    queries_agg_method = trial.suggest_categorical('queries_agg_method', kwargs['queries_agg_method'])
    cross_block_after_transformers = trial.suggest_categorical('cross_block_after_transformers', kwargs['cross_block_after_transformers'])
    q_k_v_dim = trial.suggest_categorical('q_k_v_dim', kwargs['q_k_v_dim'])
    custom_mlp = trial.suggest_categorical('custom_mlp', kwargs['custom_mlp'])
    drop_path_mode = trial.suggest_categorical('drop_path_mode', kwargs['drop_path_mode'])
    apply_xattn_mask = trial.suggest_categorical('apply_xattn_mask', kwargs['apply_xattn_mask'])
    type_head = trial.suggest_categorical('type_head', kwargs['type_head'])
    xattn_mask = None
    
    if apply_xattn_mask > 0:
      grid_size = 16 if kwargs['mt'] == 'G' else 14
      pkl_path = os.path.join(kwargs['path_video_dataset'], f'video_analysis_grid_{grid_size}_{grid_size}.pkl')
      plot_masked_attention(pkl_path, mask_threshold=apply_xattn_mask, plot_all_subject=False, folder_plot_path=kwargs['global_folder_name'])
      with open(pkl_path, 'rb') as f:
        video_analysis_dict = pickle.load(f)
      xattn_mask = torch.tensor(video_analysis_dict['mean_value_matrix']).mean(axis=0)
      xattn_mask = ~(xattn_mask < apply_xattn_mask)
      print(f"\nATTENTION_MASK DETECTED! Applied with threshold {apply_xattn_mask}. {xattn_mask.sum().item()} cells kept.\n")
      del video_analysis_dict
    head_params = {
      'input_dim': input_dim,
      'q_k_v_dim': q_k_v_dim if q_k_v_dim is not None else emb_dim,
      'num_classes': num_classes,
      'num_cross_heads': num_cross_head,
      'num_heads': num_heads or num_cross_head,
      'dropout': model_dropout,
      'adversarial_head': adversarial_head,
      'attn_dropout': drop_attn,
      'head_init_path': kwargs['head_init_path'],
      'residual_dropout': drop_residual,
      'drop_path_mode': drop_path_mode,
      'mlp_ratio': mlp_ratio,
      'custom_mlp': custom_mlp,
      'pos_enc': pos_enc,
      'coral_loss': True if loss == 'coral' else False,
      'depth': nr_blocks,
      'remove_head': remove_head,
      'cross_block_after_transformers': cross_block_after_transformers,
      'num_queries': num_queries,
      'agg_method': queries_agg_method,
      'type_head': type_head,
      'complete_block': trial.suggest_categorical('complete_block', kwargs['complete_block']),
      'embedding_reduction': kwargs['embedding_reduction'],
    }
    head_enum = HEAD.ATTENTIVE_JEPA
    head_dependent_add_kwargs = {
      'xattn_mask': xattn_mask,
      'adv_alpha_strategy': adv_alpha_strategy,
      'adv_alpha_gamma': adv_alpha_gamma,
      'adversarial_loss_lambda': adversarial_loss_lambda,
    }

  elif head_name.upper() == 'POOL_MLP':
    mlp_ratio = trial.suggest_categorical('mlp_ratio', kwargs['mlp_ratio'])
    model_dropout = _suggest(trial, 'model_dropout', kwargs['model_dropout'], kwargs['optuna_categorical'])
    head_params = {
      'input_dim': input_dim,
      'num_classes': num_classes,
      'dropout': model_dropout,
      'mlp_ratio': mlp_ratio,
    }
    head_enum = HEAD.POOL_MLP
    head_dependent_add_kwargs = {
      'xattn_mask': None,
      'adv_alpha_strategy': None,
      'adv_alpha_gamma': None,
      'adversarial_loss_lambda': None,
    }

  head_params['train_backbone'] = train_backbone
  head_params['unfreeze_layers'] = unfreeze_layers

  # Avoid duplicate trials check
  for past in trial.study.trials:
    if past.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
      continue
    if past.params == trial.params:
      return past.value

  # --- Final Argument Assembly ---
  if num_classes == 1 and loss in ['cdw_ce', 'ce', 'ce_weight', 'sim_loss', 'huber_ce']:
    raise ValueError('Loss function not supported for regression. Use l1 or l2 loss.')
     
  dict_args_loss = {
    'num_classes': num_classes,
    'alpha': cdw_ce_alpha,
    'sim_loss_reduction': sim_loss_reduction,
    'cdw_ce_delta': cdw_ce_delta,
    **loss_args,
    'transform': cdw_ce_power_transform
  }

  if loss is not None:
    criterion = get_loss(loss, dict_args=dict_args_loss)
  elif composite_loss is not None:
    criterion = composite_loss
  elif kwargs['disent_loss_p_s'][0] is not None:
    loss_type = trial.suggest_categorical('disent_loss_p_s', kwargs['disent_loss_p_s'])
    loss_type = loss_type.lower().split(',')
    lambda_weight = trial.suggest_categorical('disent_loss_lambda', kwargs['disent_loss_lambda'])
    lambda_weight = [float(lw) for lw in lambda_weight.split(',')]
    split_idx = trial.suggest_categorical('disent_split_idx', kwargs['disent_split_idx'])
    ortho_lambda = _suggest(trial, 'disent_ortho_lambda', kwargs['disent_ortho_lambda'], kwargs['optuna_categorical'])
    criterion = get_disentanglement_loss(loss_type=loss_type,
                                         lambdas=lambda_weight,
                                         split_idx=split_idx,
                                         ortho_lambda=ortho_lambda,
                                         dict_args=dict_args_loss)

  # Pre-trained Check
  if (isinstance(criterion, losses.RESupConLoss) or isinstance(criterion, losses.DisentangledLoss) or isinstance(criterion, losses.RnCLossV2)) and kwargs['head_init_path']:
    list_pth_path = get_pth_path_from_project_folder(kwargs['head_init_path'])
    if len(list_pth_path) < np.prod(kwargs['stop']):
      raise ValueError(f"Not enough pretrained models in {kwargs['head_init_path']}.")
  
  add_kwargs = {
    'dict_args_loss': dict_args_loss,
    'add_CCC_loss': add_CCC_loss,
    'CCC_loss': losses.CCCLoss() if add_CCC_loss > 0 else None,
    'concatenate_quadrants': concatenate_quadrants,
    'stratified_training': stratified_training,
    'is_subject_independent': kwargs['is_subject_independent'],
    'skip_test': kwargs['skip_test'],
    'scheduler_config_dict': scheduler_config_dict,
    'use_test_as_val': kwargs['use_test_as_val'],
    'type_group': type_group,
    'perfect_bal_strategy': perfect_bal_strategy,
    'load_idxs_splits': kwargs['load_idxs_splits'],
    'balance_batches': balance_batches,
    'consider_only_lasts_n_chunks': consider_only_lasts_n_chunks,
    'latent_coefficient': latent_coefficient,
    'filtered_augm_n_keep': filtered_augm_n_keep,
    'filtered_augm_strategy': filtered_augm_strategy,
    'only_augments': only_augments,
    'sampler_loader_type': sampler_loader_type,
    'sampler_augmented_only_types': sampler_augmented_only_types,
    'min_subjects_per_level': min_subjects_per_level,
    'debug_grad_flow': debug_grad_flow,
    'debug_grad_flow_batches': debug_grad_flow_batches,
    'HSIC_subject_lambda': HSIC_subject_lambda,
    'HSIC_pain_lambda': HSIC_pain_lambda,
    'HSIC_feat_normalization': HSIC_feat_normalization,
  }
  add_kwargs.update(head_dependent_add_kwargs)
  

  print(f"\n[Trial {trial.number}] Params: {trial.params}")
  
  # --- Execute Training ---
  run_folder_path, results = scripts.run_train_test(
    load_dataset_in_memory=kwargs['load_dataset_in_memory'],
    model_type=model_type,
    soft_labels=soft_labels,
    criterion=criterion,
    concatenate_temp_dim=concatenate_temp_dim,
    pooling_embedding_reduction=kwargs['pooling_embedding_reduction'],
    pooling_clips_reduction=kwargs['pooling_clips_reduction'],
    sample_frame_strategy=get_sampling_frame_strategy(sample_frame_strategy),
    num_clips_per_video=num_clips_per_video,
    path_csv_dataset=kwargs['csv'],
    path_video_dataset=kwargs['path_video_dataset'],
    head=head_enum,
    use_sdpa=use_sdpa,
    adapter_dict=adapter_dict,
    stride_window_in_video=kwargs['stride_window_in_video'],
    stride_inside_window=stride_inside_window,
    features_folder_saving_path=kwargs['ffsp'],
    head_params=head_params,
    k_fold=kwargs['k_fold'],
    global_foder_name=kwargs['global_folder_name'],
    batch_size_training=batch_train,
    epochs=epochs,
    lr=lr,
    seed_random_state=kwargs['seed_random_state'],
    is_plot_dataset_distribution=False,
    is_round_output_loss=head_dependent_add_kwargs.get('is_round_output_loss', 0),
    is_shuffle_video_chunks=False,
    is_shuffle_training_batch=True,
    init_network=init_network,
    key_for_early_stopping=kwargs['key_early_stopping'],
    regularization_lambda_L1=regulariz_lambda_L1,
    regularization_lambda_L2=regulariz_lambda_L2,
    clip_length=kwargs['clip_length'],
    target_metric_best_model=kwargs['target_metric_best_model'],
    early_stopping=kwargs['early_stopping'],
    stop_after_kth_fold=kwargs['stop'],
    n_workers=kwargs['n_workers'],
    clip_grad_norm=kwargs['clip_grad_norm'],
    label_smooth=label_smooth,
    dict_augmented=dict_augmented,
    trial=trial,
    prefetch_factor=kwargs['prefetch_factor'],
    validate=kwargs['validation_enabled'],
    **add_kwargs
  )

  trial.set_user_attr('id_test', os.path.basename(run_folder_path).split('_')[0])
  trial.set_user_attr('head', [kwargs['head']])
  trial.set_user_attr('csv', [kwargs['csv']])
  trial.set_user_attr('ffsp', [kwargs['ffsp']])
  trial.set_user_attr('k_fold', [kwargs['k_fold']])
  trial.set_user_attr('stop', kwargs['stop'])
  trial.set_user_attr('n_workers', [kwargs['n_workers']])
  
  return get_mean_val_accuracy(
    results=results,
    optuna_direction=trial.study.direction,
    use_median=kwargs['optuna_use_median'],
    min_epoch_threshold=kwargs['optuna_min_epoch_thr']
  )


def hyper_search(kwargs):
  """Executes the hyperparameter search using Optuna.

  Args:
    kwargs (dict): Configuration dictionary.
  """
  study_name = f'{kwargs["head"]}_{os.path.split(kwargs["global_folder_name"])[-1].split("_")[-1]}'
  try:
    sampler_module = get_sampler_module(kwargs['optuna_sampler'], kwargs)
  except SystemError as e:
    print(f"Error loading sampler module: {e}. Using default TPE sampler.")
    sampler_module = optuna.samplers.TPESampler()

  study = optuna.create_study(
    direction='maximize' if kwargs['key_early_stopping'] == 'val_accuracy' else 'minimize',
    sampler=sampler_module,
    storage=f'sqlite:///{os.path.join(kwargs["global_folder_name"], f"{study_name}.db")}',
    study_name=study_name,
    pruner=optuna.pruners.ThresholdPruner(
      lower=kwargs['pruner_threshold_lower'],
      n_warmup_steps=kwargs['pruner_n_warmup_steps'],
      interval_steps=2
    )
  )

  study.optimize(lambda trial: objective(trial, kwargs),
           n_trials=kwargs['n_trials'],
           timeout=kwargs['timeout'])

  optuna_path = os.path.join(kwargs['global_folder_name'], f'{study_name}.pkl')
  with open(optuna_path, "wb") as f:
    pickle.dump(study, f)
    print(f'Study saved to {optuna_path}')

  print('Best hyperparameters:')
  print(study.best_params)
  print(f'Best {kwargs["key_early_stopping"]}:', study.best_value)

  if len(study.trials) > 1:
    try:
      fig_hyper_importance = optuna.visualization.plot_param_importances(study)
      fig_hyper_importance.update_layout(width=1300, height=900)
      fig_hyper_importance.write_image(
        os.path.join(kwargs['global_folder_name'], f'{study_name}_hyper_importance.png')
      )
      print(f"Hyperparameter importance plot saved.")
    except Exception as e:
      print(f"Error generating hyperparameter importance plot: {e}")


def load_yaml_as_flat_dict(yaml_path):
  """Loads a YAML file and flattens one level of nesting."""
  try:
    with open(yaml_path, 'r') as f:
      yaml_config = yaml.safe_load(f) or {}
  except Exception as e:
    raise RuntimeError(f"Error reading YAML config '{yaml_path}': {e}")

  flat = {}
  if isinstance(yaml_config, dict):
    for k, v in yaml_config.items():
      if isinstance(v, dict):
        for kk, vv in v.items():
          flat_key = str(kk).lstrip('-').replace('-', '_')
          flat[flat_key] = vv
      else:
        flat_key = str(k).lstrip('-').replace('-', '_')
        flat[flat_key] = v
  else:
    raise RuntimeError("YAML config root must be a mapping (dict).")
  return flat


def validate_arguments(dict_args):
  """Validates the parsed arguments and raises errors for invalid configurations."""
  list_augmentations_available = helper.get_augmentation_availables(dict_args['ffsp'])
  list_augmentations_available.extend(['latent_masking','latent_basic'])
  
  # Set only one of disent_loss_p_s or loss
  provided = [
    (dict_args['disent_loss_p_s'][0] is not None),
    (dict_args['loss'] is not None),
    (dict_args['composite_loss'][0] is not None)
  ]
  if sum(provided) != 1:
    print(dict_args['disent_loss_p_s'][0], dict_args['loss'], dict_args['composite_loss'][0])
    raise ValueError("Set either disent_loss_p_s or loss or composite_loss, not multiple.")
  
  
  if dict_args['only_augments'][0] is not None:
    if dict_args['stratified_training'][0] != 1:
      raise ValueError("only_augments can only be used with stratified_training set to 1.")
    # list_augmentations_available = helper.get_augmentation_availables(dict_args['ffsp'])
    for augm in dict_args['only_augments']:
      list_augm = augm.split(',')
      for a in list_augm:
        if a not in list_augmentations_available:
          raise ValueError(f"Augmentation '{a}' not available in features folder '{dict_args['ffsp']}'. Available: {list_augmentations_available}")

  if dict_args['loss'] == 'contrastive_reg':
    for augm in dict_args['sampler_augmented_only_types']:
      if augm is None:
        raise ValueError("sampler_augmented_only_types cannot contain None when using contrastive_reg loss.")
      split_array = augm.split(',')
      for a in split_array:
        if a not in list_augmentations_available:
          raise ValueError(f"Augmentation '{a}' not available in features folder '{dict_args['ffsp']}'. Available: {list_augmentations_available}")
  
  for el in dict_args['sampler_loader_type']:
    if el == 'subject_batch':
      if dict_args['min_subjects_per_level'] is None or dict_args['min_subjects_per_level'][0] < 1:
        raise ValueError("min_subjects_per_level must be >= 1 when using 'subject_batch' sampler.")
    if el == 'augmented_only':
      if dict_args['sampler_augmented_only_types'] is None:
        raise ValueError("sampler_augmented_only_types must be set when using 'augmented_only' sampler_loader_type.")
      else:
        for augm in dict_args['sampler_augmented_only_types']:
          if augm.startswith('strategy'):
            continue  # Skip strategy-based sampler types  
          if augm is None:
            raise ValueError("sampler_augmented_only_types cannot contain None when using 'augmented_only' sampler_loader_type.")
          split_array = augm.split(',')
          for a in split_array:
            if a is None or a not in list_augmentations_available:
              raise ValueError(f"Augmentation '{augm}' not available in features folder '{dict_args['ffsp']}'. Available: {list_augmentations_available}")
      
  if dict_args['warm_up_scheduler'][0] is not None and dict_args['scheduler_name'][0] == 'onecycle':
    raise ValueError("Onecycle scheduler already includes warm-up phase. Remove warm_up_scheduler argument.")

  if dict_args['loss'] and dict_args['composite_loss'][0] is not None:
    raise ValueError("Cannot use both primary loss and composite loss at the same time. Choose one.")

  if any(dict_args['filtered_augm_n_keep']) and any(dict_args['balance_batches']):
    raise ValueError("Cannot use both filtered augmentation sampler and balanced batches at the same time.")

  if any(dict_args['filtered_augm_n_keep']) ^ any(dict_args['filtered_augm_strategy']):
    raise ValueError("Both filtered_augm_n_keep and filtered_augm_strategy must be set together.")

  if 'caer' in dict_args['csv']:
    print("\n Detected CAER, FORCE SPLIT INDICES FOR TRAINING K-FOLD \n")
    helper.step_shift = 13176
    helper.FORCE_SPLIT_K_FOLD = True
    if dict_args['k_fold'] != 3:
      raise ValueError("CAER dataset only supports 3-fold cross-validation. Set --k_fold 3.")
    if dict_args['stop'] is None or (dict_args['stop'][0] != 1 and dict_args['stop'][1] != 1):
       pass 

  if dict_args['concatenate_quadrants'][0] and 'combined' not in dict_args['ffsp']:
    raise ValueError("To use quadrants concatenation, the feature folder must contain combined quadrants features.")

  if dict_args['mlp_ratio'][0] == 0:
    raise ValueError("mlp_ratio cannot be 0.")

  if dict_args['train_amp_enabled']:
    helper.AMP_ENABLED = True
    helper.AMP_DTYPE = dict_args['train_amp_dtype'].lower()

  if dict_args['add_CCC_loss'][0] and dict_args['loss'][0] not in ['l1', 'l2', 'huber']:
    raise ValueError("CCC loss can only be added to 'l1', 'l2', or 'huber' loss.")

  for method in dict_args['queries_agg_method']:
    if method not in helper.QUERIES_AGG_METHOD:
       raise ValueError(f"Invalid queries aggregation method: {method}")

  if not None in dict_args['delta_huber'] and 'huber' not in dict_args['loss'][0]:
     raise ValueError("delta_huber is set but loss is not 'huber'.")

  if dict_args['loss'] is not None and 'huber' in dict_args['loss'] and any(d <= 0 for d in dict_args['delta_huber'] if d is not None):
    raise ValueError("delta_huber must be greater than 0 for Huber loss.")

  if dict_args['train_amp_dtype'] is not None and dict_args['train_amp_enabled'] is False:
    raise ValueError("train_amp_dtype is set but train_amp_enabled is False.")

  if dict_args['label_smooth'] and dict_args['soft_labels']:
    if sum(dict_args['label_smooth']) > 0 and sum(dict_args['soft_labels']) > 0:
      raise ValueError("Label smoothing and soft labels cannot be used together.")

  if dict_args['loss'] is not None and dict_args['loss'][0] in ['l1', 'l2'] and (sum(dict_args['label_smooth']) > 0 or sum(dict_args['soft_labels']) > 0):
    raise ValueError("Label smoothing/soft labels not supported for l1/l2.")

  if sum(dict_args['sim_loss_reduction']) != 0. and dict_args['loss'][0] != 'sim_loss':
    raise ValueError("Sim loss reduction is only supported for 'sim_loss'.")

  if dict_args['loss'] is not None and ('ce' not in dict_args['loss']) and (sum(dict_args['label_smooth']) > 0 or sum(dict_args['soft_labels']) > 0):
     raise ValueError("Label smoothing/soft labels only supported for 'ce' loss.")

  if dict_args['loss'] is not None and dict_args['loss'][0] not in ['l1', 'l2', 'ce'] and (dict_args['train_amp_dtype'] == 'float16'):
    raise ValueError(f"AMP float16 only supported for l1, l2, ce.")

  if dict_args['key_early_stopping'] not in ['val_accuracy', 'val_loss']:
    raise ValueError(f"Invalid key for early stopping: {dict_args['key_early_stopping']}")

  if dict_args['apply_xattn_mask'][0] and dict_args['head'].upper() != 'ATTENTIVE_JEPA':
    raise ValueError("apply_xattn_mask can only be used with ATTENTIVE_JEPA head.")

  if dict_args['apply_xattn_mask'][0] and 'combined' in dict_args['ffsp']:
    raise ValueError("apply_xattn_mask cannot be used with combined quadrants features.")

  for coeff in dict_args['latent_coefficient']:
    if coeff < 0.0 or coeff > 1.0:
      raise ValueError("latent_coefficient values must be between 0 and 1.")

  for train_backbone_flag in dict_args['train_backbone']:
    if train_backbone_flag not in [0, 1]:
      raise ValueError("train_backbone must be 0 or 1.")

  for unfreeze_layers_val in dict_args.get('unfreeze_layers', [None]):
    if unfreeze_layers_val is not None and unfreeze_layers_val < 0:
      raise ValueError("unfreeze_layers must be >= 0.")
  if any(v == 1 for v in dict_args['train_backbone']) and all(v is None for v in dict_args.get('unfreeze_layers', [None])):
    raise ValueError("--unfreeze_layers must be explicitly set when --train_backbone 1 is used.")
  non_none_ul = [v for v in dict_args.get('unfreeze_layers', [None]) if v is not None]
  if non_none_ul and max(non_none_ul) > 0 and all(v == 0 for v in dict_args['train_backbone']):
    raise ValueError("unfreeze_layers > 0 requires train_backbone=1.")

  for debug_grad_flow_flag in dict_args['debug_grad_flow']:
    if debug_grad_flow_flag not in [0, 1]:
      raise ValueError("debug_grad_flow must be 0 or 1.")

  for n_debug_batches in dict_args['debug_grad_flow_batches']:
    if n_debug_batches < 1:
      raise ValueError("debug_grad_flow_batches must be >= 1.")


if __name__ == '__main__':
  # --- Quick Parse for Config File ---
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--config', type=str, help='Path to YAML configuration file.')
  pre_args, _ = pre_parser.parse_known_args()

  yaml_defaults = {}
  if pre_args.config:
    try:
      yaml_defaults = load_yaml_as_flat_dict(pre_args.config)
    except Exception as e:
      print(f"Failed to load config file {pre_args.config}: {e}", file=sys.stderr)
      sys.exit(2)

  # --- Full Argument Parser ---
  parser = argparse.ArgumentParser(
    description='Train video analysis model with various configurations'
  )
  if yaml_defaults:
    parser.set_defaults(**yaml_defaults)

  # Arguments definitions
  parser.add_argument('--use_test_as_val', type=int, choices=[0, 1], default=0, help='Use test set as validation set.')
  parser.add_argument('--mt', type=str, default='B', help="Model type.")
  parser.add_argument('--head', type=str, default='ATTENTIVE_JEPA', help="Head type.")
  parser.add_argument('--n_workers', type=int, default=1, help='Number of workers.')
  parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor.')
  parser.add_argument('--embedding_reduction', type=str, default='none', help="Embedding reduction method.")
  parser.add_argument('--gp', action='store_true', help='Use global path prefix.')
  parser.add_argument('--csv', type=str, default=os.path.join('partA', 'starting_point', 'samples_exc_no_detection.csv'), help='Path to CSV.')
  parser.add_argument('--ffsp', type=str, default="partA/video/features/samples_16_frontalized_new", help='Feature folder saving path.')
  parser.add_argument('--global_folder_name', type=str, default=f'history_run', help='Global folder name.')
  parser.add_argument('--path_video_dataset', type=str, default=os.path.join('partA', 'video', 'video_frontalized_interpolated_mirror'), help='Path to video dataset.')
  parser.add_argument('--load_dataset_in_memory', type=int, default=0, help='Load dataset in RAM.')
  parser.add_argument('--is_subject_independent', type=int, choices=[0, 1], default=1, help='Subject-independent split.')
  parser.add_argument('--skip_test', type=int, default=0, help='Skip test phase.')
  parser.add_argument('--consider_only_lasts_n_chunks', nargs='*', type=int, default=[0], help='Consider only last n chunks.')
  parser.add_argument('--sampler_loader_type', type=str,nargs='*', default=['balanced'], help='Sampler loader type availables: selective_augm(set also filtered_augm_n_keep and filtered_augm_strategy), augmented_only, balanced, standard, subject_batch')
  parser.add_argument('--min_subjects_per_level', nargs='*', type=int, default=[0], help='Min unique subjects per pain level per batch (used with subject_batch sampler).')
  parser.add_argument('--filtered_augm_n_keep', nargs='*', type=int, default=[0], help='Samples to keep from each augmentation type.')
  parser.add_argument('--sampler_augmented_only_types', type=str, nargs='*', default=[None], help="Types of augmentations to use when sampler_loader_type is 'augmented_only'. Options can be like: 'hflip', 'jitter', 'rotation,jitter', 'latent_basic', 'latent_masking', 'shift_augm'.")
  parser.add_argument('--filtered_augm_strategy', type=int, nargs='*', default=[0], help="Strategy for filtered augmentation sampler."+
                      " 1: Randomly keep n samples from what augmentation is available (all h_flip, all jitter, etc.)"+
                      " 2: Randomly choose n types from GLOBAL available types where"+
                      " n is --filtered_augm_n_keep")
  parser.add_argument('--save_model_every_n_epochs', type=int, default=0, help='Save model every n epochs.')
  parser.add_argument('--balance_batches', type=int, nargs='*', default=[0], help='Try to balance batches.')
  parser.add_argument('--type_group', type=int, nargs='*', default=[0], help='Type of group for stratified sampling.')
  parser.add_argument('--lr', type=float, nargs='*', default=[0.0001], help='Learning rate(s)')
  parser.add_argument('--opt', type=str, nargs='*', default=['adamw'], help="Optimizer(s).")
  parser.add_argument('--validation_enabled', type=int, choices=[0, 1], default=1, help='Enable validation.')
  parser.add_argument('--train_amp_enabled', action='store_true', help='Enable AMP training')
  parser.add_argument('--train_amp_dtype', type=str, default=None, help='AMP training data type.')
  parser.add_argument('--ep', type=int, default=50, help='Number of epochs')
  parser.add_argument('--k_fold', type=int, default=3, help='Number of k-fold splits')
  parser.add_argument('--batch_train', type=int, nargs='*', default=[64], help='Training batch size(s)')
  parser.add_argument('--stop', type=int, nargs='*', default=None, help='Stop after [kth fold, ith subfold]')
  parser.add_argument('--clip_grad_norm', type=float, default=None, help='Clip gradient norm.')
  parser.add_argument('--concatenate_temp_dim', type=int, nargs='*', default=[0], help='Concatenate temporal dimension.')
  parser.add_argument('--concatenate_quadrants', type=int, nargs='*', default=[0], help='Concatenate quadrants dimension.')
  parser.add_argument('--loss', type=str, nargs='*', default=None, help="Loss function.")
  parser.add_argument('--disent_loss_p_s', type=str, nargs='*', default=[None], help="Disentanglement loss type. First is for pain, second for subject id. Ex: l1,ce. ")
  parser.add_argument('--disent_loss_lambda', type=str, nargs='*', default=[None], help="Disentanglement loss lambda. First is for pain, second for subject id. Ex: 1.0,0.1 ")
  parser.add_argument('--disent_split_idx', type=int, nargs='*', default=[None], help="Index to split features for disentanglement.")
  parser.add_argument('--disent_ortho_lambda', type=float, nargs='*', default=[0.0], help="Orthogonality loss lambda for disentanglement. (0 to disable).")
  parser.add_argument('--HSIC_subject_lambda', type=float, nargs='*', default=[0.0], help="HSIC loss lambda for subject identity disentanglement. (0 to disable).")
  parser.add_argument('--HSIC_pain_lambda', type=float, nargs='*', default=[0.0], help="HSIC loss lambda for pain label disentanglement. (0 to disable).")
  parser.add_argument('--HSIC_feat_normalization', type=int, choices=[0, 1], nargs='*', default=[0], help="L2-normalize features before computing HSIC losses. (0 to disable).")
  parser.add_argument('--add_CCC_loss', type=float, nargs='*', default=[0.0], help='Add CCC loss.')
  parser.add_argument('--cdw_ce_alpha', type=float, nargs='*', default=[2], help='Alpha for CDW loss.')
  parser.add_argument('--cdw_ce_transform', type=str, nargs='*', default=['power'], help='Transform for CDW loss.')
  parser.add_argument('--cdw_ce_delta', type=float, nargs='*', default=[0.5], help='Delta for CDW loss.')
  parser.add_argument('--sim_loss_reduction', type=float, nargs='*', default=[0.0], help='Reduction for sim loss.')
  parser.add_argument('--delta_huber', type=float, nargs='*', default=[None], help='Delta for Huber loss.')
  parser.add_argument('--num_clips_per_video', type=int, nargs='*', default=[1], help='Num clips per video.')
  parser.add_argument('--sample_frame_strategy', type=str, nargs='*', default=['sliding_window'], help="Sampling strategy.")
  parser.add_argument('--stride_inside_window', type=int, nargs='*', default=[1], help='Stride inside window.')
  parser.add_argument('--composite_loss', nargs='*', type=str, default=[None], help='Composite loss type.')
  parser.add_argument('--composite_loss_lambda', nargs='*', type=str, default=[None], help='Lambda for composite loss.')
  parser.add_argument('--use_sdpa', type=int, nargs='*', default=[1], help='Use SDPA.')
  parser.add_argument('--train_backbone', type=int, nargs='*', default=[0], help='Train video backbone weights (1) or freeze them (0).')
  parser.add_argument('--unfreeze_layers', type=int, nargs='*', default=[None], help='Number of last backbone encoder blocks to unfreeze when train_backbone=1. Required when train_backbone=1. 0 = unfreeze all trainable params.')
  parser.add_argument('--debug_grad_flow', type=int, nargs='*', default=[0], help='Log gradient-flow summary during training (1 to enable).')
  parser.add_argument('--debug_grad_flow_batches', type=int, nargs='*', default=[1], help='Number of batches per epoch to log gradient-flow stats for.')
  parser.add_argument('--rncloss_temp', type=str, nargs='*', default=[None], help='Temperature for RnCLoss.')
  parser.add_argument('--only_contrastive_temp', type=str, nargs='*', default=[None], help='Only contrastive loss temperature.')
  parser.add_argument('--only_contrastive_theta', type=str, nargs='*', default=[None], help='Only contrastive loss theta.')
  parser.add_argument('--contrastive_loss_temp', type=str, nargs='*', default=[None], help='Temp for contrastive loss.')
  parser.add_argument('--contrastive_loss_theta', type=str, nargs='*', default=[None], help='Theta for contrastive loss.')
  parser.add_argument('--contrastive_lambda_weight', type=str, nargs='*', default=[None], help='Lambda weight for contrastive loss.')
  parser.add_argument('--spearman_reg', type=str, nargs='*', choices=['l2', 'kl'], default=['l2'], help='Spearman regression type.')
  parser.add_argument('--spearman_reg_strength', type=str, nargs='*', default=[None], help='Spearman regression strength.')
  parser.add_argument('--stratified_training', type=int, nargs='*', default=[0], help='Use stratified sampling. If Biovid it adds the following latent augm:'+
                                                                              ' 2: latent_basic; '+
                                                                              ' 3: latent_masking; '+
                                                                              ' 4: both latent_basic and latent_masking.')
  parser.add_argument('--only_augments', nargs='*', type=str, default=[None], help="Only use these augmentations during training. Options can bel like: 'hflip', 'jitter', 'rotation', 'latent_basic', 'latent_masking', 'shift_augm'."+
                      "The startified_training argument must be set to 1 when using this.")
  parser.add_argument('--perfect_bal_strategy', type=str, nargs='*', default=[None], help="Perfect balancing strategy.")
  parser.add_argument('--num_heads', type=int, nargs='*', default=[8], help='Number of heads.')
  parser.add_argument('--num_cross_head', type=int, nargs='*', default=[8], help='Number of cross heads.')
  parser.add_argument('--model_dropout', type=float, nargs='*', default=[0.0], help='Model dropout.')
  parser.add_argument('--drop_attn', type=float, nargs='*', default=[0.0], help='Attention dropout.')
  parser.add_argument('--drop_residual', type=float, nargs='*', default=[0.0], help='Residual dropout.')
  parser.add_argument('--drop_path_mode', type=str, nargs='*', default=['row'], help='Drop path mode.')
  parser.add_argument('--mlp_ratio', type=float, nargs='*', default=[2.0], help='MLP ratio.')
  parser.add_argument('--custom_mlp', type=int, nargs='*', default=[0], help='Use custom MLP.')
  parser.add_argument('--type_head', type=int, nargs='*', default=[0], help='Type of head. 0: only finale liner, 1: MLP + linear')
  parser.add_argument('--pos_enc', type=int, nargs='*', default=[0], help='Use positional encoding.')
  parser.add_argument('--nr_blocks', type=int, nargs='*', default=[1], help='Number of blocks.')
  parser.add_argument('--cross_block_after_transformers', type=int, nargs='*', default=[0], help='Cross block after transformers.')
  parser.add_argument('--num_queries', type=int, nargs='*', default=[1], help='Number of queries.')
  parser.add_argument('--queries_agg_method', type=str, nargs='*', default=['mean'], help="Aggregation method for queries.")
  parser.add_argument('--complete_block', type=int, nargs='*', default=[1], help='Use complete block.')
  parser.add_argument('--q_k_v_dim', type=int, nargs='*', default=[None], help='Dimension of query, key, value.')
  parser.add_argument('--apply_xattn_mask', type=int, nargs='*', default=[0], help='Apply mask to cross-attention.')
  parser.add_argument('--adpater_mlp_ratio', type=float, nargs='*', default=[0.25], help='MLP ratio for adapter.')
  parser.add_argument('--adapter_kernel_size', type=int, nargs='*', default=[3], help='Kernel size for adapter.')
  parser.add_argument('--adapter_dilation', type=int, nargs='*', default=[1], help='Dilation for adapter.')
  parser.add_argument('--adapter_type', type=str, nargs='*', default=['adapter'], help='Adapter type.')
  parser.add_argument('--num_adapters', type=int, nargs='*', default=[0], help='Number of adapters.')
  parser.add_argument('--linear_dim_reduction', type=str, default='spatial', help="Dim reduction for Linear head.")
  parser.add_argument('--GRU_hidden_size', type=int, nargs='*', default=[1024], help='GRU hidden size.')
  parser.add_argument('--GRU_num_layers', type=int, nargs='*', default=[2], help='GRU number of layers.')
  parser.add_argument('--GRU_dropout', type=float, nargs='*', default=[0.0], help='GRU dropout.')
  parser.add_argument('--GRU_output_size', type=int, default=1, help='Output size of GRU.')
  parser.add_argument('--layer_norm', type=int, nargs='*', default=[0], help='Add Layer norm.')
  parser.add_argument('--GRU_round_output_loss', type=int, default=[0], nargs='*', help='Round output loss.')
  parser.add_argument('--init_network', type=str, nargs='*', default=['default'], help='Network initialization.')
  parser.add_argument('--regulariz_lambda_L1', type=float, nargs='*', default=[0], help='L1 regularization.')
  parser.add_argument('--regulariz_lambda_L2', type=float, nargs='*', default=[0], help='L2 regularization.')
  parser.add_argument('--label_smooth', type=float, nargs='*', default=[0.0], help='Label smoothing.')
  parser.add_argument('--soft_labels', type=float, nargs='*', default=[0.0], help='Soft labels.')
  parser.add_argument('--loss_regression', type=str, default='L1', help='Regression loss.')
  parser.add_argument('--head_init_path', type=str, default=None, help='Path to pre-trained head.')
  parser.add_argument('--hflip', type=float, nargs='*', default=[0.0], help='Hflip prob.')
  parser.add_argument('--jitter', type=float, nargs='*', default=[0.0], help='Jitter prob.')
  parser.add_argument('--rotation', type=float, nargs='*', default=[0.0], help='Rotation prob.')
  parser.add_argument('--latent_basic', type=float, nargs='*', default=[0.0], help='Latent basic prob.')
  parser.add_argument('--latent_masking', type=float, nargs='*', default=[0.0], help='Latent masking prob.')
  parser.add_argument('--shift_augm', type=float, nargs='*', default=[0.0], help='Shift prob.')
  parser.add_argument('--latent_coefficient', type=float, nargs='*', default=[0.0], help='Latent coeff.')
  parser.add_argument('--key_early_stopping', type=str, default=None, help='Metric for early stopping.')
  parser.add_argument('--p_early_stop', type=int, default=2000, help='Patience.')
  parser.add_argument('--min_delta', type=float, default=0.001, help='Min delta.')
  parser.add_argument('--threshold_mode', type=str, default='abs', help='Threshold mode.')
  parser.add_argument('--log_grad_per_module', action='store_true', help='Log grad per module.')
  parser.add_argument('--log_history_sample', action='store_true', help='Log history sample.')
  parser.add_argument('--plot_live_loss', action='store_true', help='Plot live loss.')
  parser.add_argument('--log_xattn', action='store_true', help='Log xattn.')
  parser.add_argument('--log_grad_norm', action='store_true', help='Log grad norm.')
  parser.add_argument('--log_workers', action='store_true', help='Log workers.')
  parser.add_argument('--profile_gpu_sync', action='store_true', help='Enable torch.cuda.synchronize() in profiling timers for accurate GPU timing.')
  parser.add_argument('--save_best_model', action='store_true', help='Save best model.')
  parser.add_argument('--save_last_epoch_model', action='store_true', help='Save last epoch model.')
  parser.add_argument('--adversarial_head', type=int, nargs='*', default=[0], help='Add adversarial head.')
  parser.add_argument('--adv_alpha_strategy', type=str, nargs='*', default=[None], help='Adv alpha strategy.')
  parser.add_argument('--adv_alpha_gamma', type=float, nargs='*', default=[0.0], help='Gamma for adv.')
  parser.add_argument('--adversarial_loss_lambda', type=float, nargs='*', default=[0.0], help='Adv loss lambda.')
  parser.add_argument('--load_idxs_splits', type=int, default=0, help='Load idxs splits.')
  parser.add_argument('--scheduler_name', type=str, nargs='*', default=['cosine'], help='Scheduler name.')
  parser.add_argument('--wd_scheduler_name', type=str, nargs='*', default=[None], help='WD scheduler.')
  parser.add_argument('--exclude_bias_wd', type=int, choices=[0, 1], nargs='*', default=[1], help='Exclude bias wd.')
  parser.add_argument('--min_lr', type=float, nargs='*', default=[1e-7], help='Min lr.')
  parser.add_argument('--warm_up_epochs', type=float, nargs='*', default=[None], help='Warm up epochs.')
  parser.add_argument('--warm_up_scheduler', type=str, nargs='*', default=[None], help='Warm up scheduler.')
  parser.add_argument('--warm_up_start_factor', type=float, nargs='*', default=[None], help='Warm up start factor.')
  parser.add_argument('--min_wd', type=float, nargs='*', default=[None], help='Min wd.')
  parser.add_argument('--first_restart_epochs', type=int, nargs='*', default=[None], help='First restart.')
  parser.add_argument('--multiplier_restart', type=int, nargs='*', default=[None], help='Multiplier restart.')
  parser.add_argument('--step_size_epochs', type=int, nargs='*', default=[None], help='Step size.')
  parser.add_argument('--gamma', type=float, nargs='*', default=[None], help='Gamma.')
  parser.add_argument('--onecycle_pct_start', type=float, nargs='*', default=[None], help='Pct start.')
  parser.add_argument('--onecycle_max_lr', type=float, nargs='*', default=[None], help='Max lr.')
  parser.add_argument('--onecycle_div_factor', type=float, nargs='*', default=[None], help='Div factor.')
  parser.add_argument('--onecycle_final_div_factor', type=float, nargs='*', default=[None], help='Final div factor.')
  parser.add_argument('--onecycle_anneal_strategy', type=str, nargs='*', default=[None], help='Anneal strategy.')
  parser.add_argument('--optuna_use_median', type=int, default=0, help='Use median.')
  parser.add_argument('--optuna_min_epoch_thr', type=int, default=0, help='Min epoch thr.')
  parser.add_argument('--pruner_threshold_lower', type=float, default=0.20, help='Pruner threshold.')
  parser.add_argument('--pruner_n_warmup_steps', type=int, default=30, help='Pruner warmup.')
  parser.add_argument('--optuna_categorical', type=int, default=1, help='Use categorical optimization.')
  parser.add_argument('--optuna_sampler', type=str, default='auto', help='Optuna sampler.')
  parser.add_argument('--n_trials', type=int, default=100, help='Number of trials.')
  parser.add_argument('--timeout', type=float, default=14, help='Timeout in hours.')

  args = parser.parse_args() 
  args.timeout = int(args.timeout * 3600)  # Convert hours to seconds
  dict_args = vars(args)

  # --- Global Constants & Side Effects ---
  if dict_args['key_early_stopping'] is None:
    if dict_args['loss'] and 'ce' in dict_args['loss'][0]:
      dict_args['key_early_stopping'] = 'val_accuracy'
    else:
      dict_args['key_early_stopping'] = 'val_loss'
    print(f"\n*********key_early_stopping not set. Set: {dict_args['key_early_stopping']}*********\n")
  validate_arguments(dict_args)

  dict_args['pooling_embedding_reduction'] = helper.EMBEDDING_REDUCTION.get_embedding_reduction(
    dict_args['embedding_reduction']
  )
  dict_args['pooling_clips_reduction'] = CLIPS_REDUCTION.NONE
  dict_args['stride_window_in_video'] = 16
  dict_args['seed_random_state'] = 42
  dict_args['clip_length'] = 16

  if dict_args['save_model_every_n_epochs']:
    helper.SAVE_MODEL_EVERY_N_EPOCHS = dict_args['save_model_every_n_epochs']
    print(f"Models will be saved every {helper.SAVE_MODEL_EVERY_N_EPOCHS} epochs.\n")
  
  if dict_args['plot_live_loss']:
    helper.PLOT_LIVE_LOSS = True

  if dict_args['save_best_model']:
    helper.SAVE_PTH_MODEL = True

  if dict_args['log_xattn']:
    helper.LOG_CROSS_ATTENTION['enable'] = True
    helper.LOG_HISTORY_SAMPLE = True

  if dict_args['log_history_sample']:
    helper.LOG_HISTORY_SAMPLE = True

  if dict_args['log_workers']:
    helper.time_profiling_enabled = True

  if dict_args['profile_gpu_sync']:
    helper.PROFILING_GPU_SYNC = True

  if dict_args['save_last_epoch_model']:
    helper.SAVE_LAST_EPOCH_MODEL = True
    helper.SAVE_PTH_MODEL = True
  
  if not dict_args['validation_enabled']:
    helper.SAVE_LAST_EPOCH_MODEL = True
    print("\nValidation is disabled. Last epoch model will be used for testing \n")

  if dict_args['log_grad_norm']:
    helper.LOG_GRADIENT_NORM = True
    
  if dict_args['log_grad_per_module']:
    helper.LOG_GRADIENT_PER_MODULE = True

  # Generate unique folder name
  timestamp = int(time.time())
  server_name = platform.node()
  pid = os.getpid()
  args.global_folder_name = f'{args.global_folder_name}_{pid}_{args.head}_{server_name}_{timestamp}'

  if args.gp:
    args.csv = helper.GLOBAL_PATH.get_global_path(args.csv)
    args.ffsp = helper.GLOBAL_PATH.get_global_path(args.ffsp)
    args.path_video_dataset = helper.GLOBAL_PATH.get_global_path(args.path_video_dataset)
    args.global_folder_name = helper.GLOBAL_PATH.get_global_path(args.global_folder_name)

  print('\n\nTraining configuration:')
  print(args)

  dict_args['target_metric_best_model'] = args.key_early_stopping

  if args.key_early_stopping == 'val_loss':
    dict_args['early_stopping'] = earlyStoppingLoss(
      patience=args.p_early_stop,
      min_delta=args.min_delta,
      threshold_mode=args.threshold_mode
    )
  else:
    dict_args['early_stopping'] = earlyStoppingAccuracy(
      patience=args.p_early_stop,
      min_delta=args.min_delta,
      threshold_mode=args.threshold_mode
    )

  print(f'Early stopping configuration: {str(dict_args["early_stopping"])}\n')

  # Save Config Prompt
  if not os.path.exists(args.global_folder_name):
    os.makedirs(args.global_folder_name)

  setup_logger(args.global_folder_name)

  with open(os.path.join(args.global_folder_name, '_config_prompt.txt'), 'w') as f:
    f.write(f'launch_command: {" ".join(sys.argv)}\n')
    for key, value in dict_args.items():
      f.write(f'{key}: {value}\n')

  hyper_search(dict_args)
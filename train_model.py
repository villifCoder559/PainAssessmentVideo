import os
import time
import argparse
import sys
import platform
import pickle
from copy import deepcopy

import numpy as np
import torch.nn as nn
import optuna
import optunahub
import yaml

import custom.scripts as scripts
import custom.helper as helper
from custom.helper import (
  CLIPS_REDUCTION,
  MODEL_TYPE,
  HEAD,
)
from custom.head import earlyStoppingDummy
from custom.optimizers import OptimizerFactory
from custom.logger import setup_logger
from custom.model_selection import (
  optuna_direction_for_metric,
  resolve_selection_metric,
  threshold_pruner_kwargs,
)
from custom.predefined_splits import configure_csv_input
from custom.targets import (
  SCALAR_REGRESSION_LOSSES,
  TargetSpec,
  validate_primary_losses,
)

os.environ["OPTUNA_DISABLE_TELEMETRY"] = "1"

LOG_SCALE_OPTUNA_HYPERS = ['lr', 'regulariz_lambda_L1', 'regulariz_lambda_L2']


def configure_selection_policy(config):
  metric = resolve_selection_metric(
    config.get('loss'),
    config.get('key_early_stopping'),
  )
  config['key_early_stopping'] = metric
  config['target_metric_best_model'] = metric
  return metric


def configure_target_spec(config):
  split_paths = config.get('predefined_csv_splits')
  csv_paths = (
    [split_paths[role] for role in ('train', 'val', 'test')]
    if split_paths is not None
    else [config['training_csv']]
  )
  spec = TargetSpec.from_csv_paths(
    csv_paths, normalize=bool(config.get('normalize_labels', 0))
  )
  validate_primary_losses(list(config.get('loss') or []), spec)
  config['target_spec'] = spec.to_metadata()
  return spec


def get_loss(loss_name):
  """Get the loss function from its name: 'l1' (regression) or 'ce' (classification)."""
  loss_lower = loss_name.lower()
  if loss_lower == 'l1':
    return nn.L1Loss()
  elif loss_lower == 'ce':
    return nn.CrossEntropyLoss()
  raise ValueError(f'Loss not supported: {loss_name}. Valid options: l1, ce')


def get_mean_val_accuracy(results):
  """Mean over folds of the validation metric at each fold's best epoch."""
  list_val_metric = []
  for k_fold, dict_log_k_fold in results['results'].items():
    if 'final' not in k_fold:
      best_epoch = dict_log_k_fold['train_val']['best_model_idx']
      list_val_metric.append(
        dict_log_k_fold['train_val']['list_val_performance_metric'][best_epoch]
      )
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
    # check if there is a 0 in the search space lists, which would cause an error in GridSampler
    for k, v in search_space.items():
      if isinstance(v, list) and len(v) == 0:
        raise ValueError(f"GridSampler does not support empty lists in search space. Found empty list in {k}.")
    print(f'{sampler_module._all_grids}')
    print(f'\n Grid search trials: {len(sampler_module._all_grids)}\n')
    # print all grid combinations    
    # print(f'Grid combinations:')
    # for i, grid in enumerate(sampler_module._all_grids):
    #   print(f'  Trial {i + 1}: {grid}')
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
  return {
    'optimizer_name': opt,
    'scheduler_name': scheduler_name,
    'epochs': kwargs['ep'],
    'warm_up_epochs': _suggest(trial, 'warm_up_epochs', kwargs['warm_up_epochs'], kwargs['optuna_categorical']),
    'warm_up_scheduler': trial.suggest_categorical('warm_up_scheduler', kwargs['warm_up_scheduler']),
    'warm_up_start_factor': _suggest(trial, 'warm_up_start_factor', kwargs['warm_up_start_factor'], kwargs['optuna_categorical']),
    'min_lr': _suggest(trial, 'min_lr', kwargs['min_lr'], kwargs['optuna_categorical']),
    'first_restart_epochs': _suggest(trial, 'first_restart_epochs', kwargs['first_restart_epochs'], kwargs['optuna_categorical']),
    'multiplier_restart': _suggest(trial, 'multiplier_restart', kwargs['multiplier_restart'], kwargs['optuna_categorical']),
    'lr': _suggest(trial, 'lr', kwargs['lr'], kwargs['optuna_categorical']),
    'weight_decay': regulariz_lambda_L2,
  }


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
  init_network = trial.suggest_categorical('init_network', kwargs['init_network'])
  batch_train = trial.suggest_categorical('batch_train', kwargs['batch_train'])
  regulariz_lambda_L1 = _suggest(trial, 'regulariz_lambda_L1', kwargs['regulariz_lambda_L1'], kwargs['optuna_categorical'])
  regulariz_lambda_L2 = _suggest(trial, 'regulariz_lambda_L2', kwargs['regulariz_lambda_L2'], kwargs['optuna_categorical'])
  label_smooth = trial.suggest_categorical('label_smooth', kwargs['label_smooth'])
  concatenate_quadrants = trial.suggest_categorical('concatenate_quadrants', kwargs['concatenate_quadrants'])
  stratified_training = trial.suggest_categorical('stratified_training', kwargs['stratified_training'])
  target_samples_per_class_training = trial.suggest_categorical('target_samples_per_class_training', kwargs['target_samples_per_class_training'])
  sampler_loader_type = trial.suggest_categorical('sampler_loader_type', kwargs['sampler_loader_type'])
  filtered_augm_n_keep = trial.suggest_categorical('filtered_augm_n_keep', kwargs['filtered_augm_n_keep'])
  filtered_augm_strategy = trial.suggest_categorical('filtered_augm_strategy', kwargs['filtered_augm_strategy'])
  keep_original = trial.suggest_categorical('keep_original', kwargs['keep_original'])
  undersample_max_per_class = trial.suggest_categorical('undersample_max_per_class', kwargs['undersample_max_per_class'])

  # --- Scheduler Config ---
  # Construct config manually here to ensure we use the sampled 'lr' and 'regulariz_lambda_L2'
  scheduler_config_dict = _suggest_scheduler_params(trial, kwargs, opt, regulariz_lambda_L2)
  scheduler_config_dict['lr'] = lr
  OptimizerFactory.check_schedulers_parms(scheduler_name=scheduler_config_dict['scheduler_name'], config=scheduler_config_dict)

  # --- Loss and number of outputs ---
  target_spec = TargetSpec.from_metadata(kwargs['target_spec'])
  loss = trial.suggest_categorical('loss', kwargs['loss'])
  criterion = get_loss(loss)
  if loss in SCALAR_REGRESSION_LOSSES:
    num_classes = 1
    print(f"\nContinuous-target task detected. Setting num_classes to 1 for {loss} loss.")
  else:
    num_classes = target_spec.bin_count

  # --- Head Params ---
  emb_dim = MODEL_TYPE.get_embedding_size(kwargs['mt'])
  input_dim = emb_dim * 4 if concatenate_quadrants else emb_dim  # 4 quadrants
  num_heads = trial.suggest_categorical('num_heads', kwargs['num_heads'])
  num_cross_head = trial.suggest_categorical('num_cross_head', kwargs['num_cross_head'])
  head_params = {
    'input_dim': input_dim,
    'q_k_v_dim': emb_dim,
    'num_classes': num_classes,
    'num_cross_heads': num_cross_head,
    'num_heads': num_heads or num_cross_head,
    'dropout': _suggest(trial, 'model_dropout', kwargs['model_dropout'], kwargs['optuna_categorical']),
    'attn_dropout': _suggest(trial, 'drop_attn', kwargs['drop_attn'], kwargs['optuna_categorical']),
    'residual_dropout': _suggest(trial, 'drop_residual', kwargs['drop_residual'], kwargs['optuna_categorical']),
    'mlp_ratio': trial.suggest_categorical('mlp_ratio', kwargs['mlp_ratio']),
    'custom_mlp': trial.suggest_categorical('custom_mlp', kwargs['custom_mlp']),
    'mlp_num_hidden_layers': _suggest(trial, 'mlp_num_hidden_layers', kwargs['mlp_num_hidden_layers'], kwargs['optuna_categorical']),
    'pos_enc': trial.suggest_categorical('pos_enc', kwargs['pos_enc']),
    'depth': _suggest(trial, 'nr_blocks', kwargs['nr_blocks'], kwargs['optuna_categorical']),
    'cross_block_after_transformers': trial.suggest_categorical('cross_block_after_transformers', kwargs['cross_block_after_transformers']),
    'complete_block': trial.suggest_categorical('complete_block', kwargs['complete_block']),
    'embedding_reduction': kwargs['embedding_reduction'],
  }

  # Avoid duplicate trials check
  for past in trial.study.trials:
    if past.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
      continue
    if past.params == trial.params:
      return past.value

  add_kwargs = {
    'concatenate_quadrants': concatenate_quadrants,
    'stratified_training': stratified_training,
    'target_samples_per_class_training': target_samples_per_class_training,
    'is_subject_independent': kwargs['is_subject_independent'],
    'skip_test': kwargs['skip_test'],
    'scheduler_config_dict': scheduler_config_dict,
    'use_test_as_val': kwargs['use_test_as_val'],
    'filtered_augm_n_keep': filtered_augm_n_keep,
    'filtered_augm_strategy': filtered_augm_strategy,
    'keep_original': keep_original,
    'undersample_max_per_class': undersample_max_per_class,
    'sampler_loader_type': sampler_loader_type,
    'normalize_labels': kwargs['normalize_labels'],
    'target_spec': target_spec.to_metadata(),
    'predefined_csv_splits': kwargs['predefined_csv_splits'],
  }

  print(f"\n[Trial {trial.number}] Params: {trial.params}")

  # --- Execute Training ---
  run_folder_path, results = scripts.run_train_test(
    load_dataset_in_memory=kwargs['load_dataset_in_memory'],
    model_type=model_type,
    criterion=criterion,
    pooling_embedding_reduction=kwargs['pooling_embedding_reduction'],
    pooling_clips_reduction=kwargs['pooling_clips_reduction'],
    path_csv_dataset=kwargs['training_csv'],
    path_video_dataset=kwargs['path_video_dataset'],
    head=HEAD.ATTENTIVE_JEPA,
    stride_window_in_video=kwargs['stride_window_in_video'],
    features_folder_saving_path=kwargs['ffsp'],
    head_params=head_params,
    k_fold=kwargs['k_fold'],
    global_foder_name=kwargs['global_folder_name'],
    batch_size_training=batch_train,
    epochs=epochs,
    lr=lr,
    seed_random_state=kwargs['seed_random_state'],
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
    label_smooth=label_smooth,
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

  return get_mean_val_accuracy(results=results)


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
    direction=optuna_direction_for_metric(kwargs['key_early_stopping']),
    sampler=sampler_module,
    storage=f'sqlite:///{os.path.join(kwargs["global_folder_name"], f"{study_name}.db")}',
    study_name=study_name,
    pruner=optuna.pruners.ThresholdPruner(
      **threshold_pruner_kwargs(
        kwargs['key_early_stopping'],
        kwargs['pruner_threshold_lower'],
      ),
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
  if dict_args['head'].upper() != 'ATTENTIVE_JEPA':
    raise ValueError("Only the ATTENTIVE_JEPA head is supported.")
  if not dict_args['loss']:
    raise ValueError("Set --loss (l1 or ce).")
  if any(str(v).lower() not in ('l1', 'ce') for v in dict_args['loss']):
    raise ValueError(f"loss must be l1 or ce, got {dict_args['loss']}")
  if any(v not in (0, 3) for v in dict_args['pos_enc']):
    raise ValueError(f"pos_enc must be 0 or 3, got {dict_args['pos_enc']}")
  if any(v not in ('cosine', 'cosine_restart') for v in dict_args['scheduler_name']):
    raise ValueError(f"scheduler_name must be cosine or cosine_restart, got {dict_args['scheduler_name']}")

  if any(v not in (0, 1) for v in dict_args['stratified_training']):
    raise ValueError("stratified_training must be 0 or 1.")

  for el in dict_args['sampler_loader_type']:
    if el not in ('selective_augm', 'standard'):
      raise ValueError(f"Unknown sampler_loader_type: {el}. Choose 'selective_augm' or 'standard'.")
    if el == 'selective_augm' and dict_args['stratified_training'][0] == 0:
      raise ValueError("sampler_loader_type='selective_augm' requires stratified_training to be != 0.")

  if any(dict_args['filtered_augm_n_keep']) ^ any(dict_args['filtered_augm_strategy']):
    raise ValueError("Both filtered_augm_n_keep and filtered_augm_strategy must be set together.")

  for v in dict_args['undersample_max_per_class']:
    if v < -1:
      raise ValueError(f"undersample_max_per_class values must be >= -1 (0=disabled, -1=match minority class), got {v}")
  if any(v != 0 for v in dict_args['undersample_max_per_class']) and 'selective_augm' not in dict_args['sampler_loader_type']:
    raise ValueError("undersample_max_per_class is only used with sampler_loader_type='selective_augm'.")

  if dict_args['concatenate_quadrants'][0] and 'combined' not in dict_args['ffsp']:
    raise ValueError("To use quadrants concatenation, the feature folder must contain combined quadrants features.")

  if dict_args['mlp_ratio'][0] == 0:
    raise ValueError("mlp_ratio cannot be 0.")

  if dict_args.get('normalize_labels', 0):
    configured = set(dict_args.get('loss') or [])
    if not configured <= SCALAR_REGRESSION_LOSSES:
      raise ValueError(
        f"--normalize_labels=1 requires continuous-target losses. Got: {sorted(configured)}"
      )

  if set(dict_args['loss']) & SCALAR_REGRESSION_LOSSES and sum(dict_args.get('label_smooth') or [0]) > 0:
    raise ValueError("Label smoothing is not supported for continuous-target losses.")

  if dict_args['key_early_stopping'] not in ['val_accuracy', 'val_loss']:
    raise ValueError(f"Invalid key for early stopping: {dict_args['key_early_stopping']}")


if __name__ == '__main__':
  # --- Quick Parse for Config File ---
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument('--config', type=str, help='Path to YAML configuration file.')
  pre_args, _ = pre_parser.parse_known_args()

  yaml_defaults = {}
  if pre_args.config:
    try:
      yaml_defaults = load_yaml_as_flat_dict(pre_args.config)
      yaml_defaults.pop('add_CCC_loss', None)
      yaml_defaults.pop('CCC_loss', None)
    except Exception as e:
      print(f"Failed to load config file {pre_args.config}: {e}", file=sys.stderr)
      sys.exit(2)

  # --- Full Argument Parser ---
  parser = argparse.ArgumentParser(
    description='Train video analysis model with various configurations'
  )

  # Arguments definitions
  parser.add_argument('--config', type=str, help='Path to YAML configuration file (CLI flags override it).')
  parser.add_argument('--use_test_as_val', type=int, choices=[0, 1], default=0, help='Use test set as validation set.')
  parser.add_argument('--mt', type=str, default='B', help="Model type.")
  parser.add_argument('--head', type=str, default='ATTENTIVE_JEPA', help="Head type.")
  parser.add_argument('--n_workers', type=int, default=1, help='Number of workers.')
  parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor.')
  parser.add_argument('--embedding_reduction', type=str, default='none', help="Embedding reduction method.")
  parser.add_argument('--gp', action='store_true', help='Use global path prefix.')
  parser.add_argument(
    '--csv', type=str,
    default=os.path.join('partA', 'starting_point', 'samples_exc_no_detection.csv'),
    help='Path to a CSV file or to a directory containing predefined train/val/test CSVs.',
  )
  parser.add_argument('--ffsp', type=str, default="partA/video/features/samples_16_frontalized_new", help='Feature folder saving path.')
  parser.add_argument('--global_folder_name', type=str, default=f'history_run', help='Global folder name.')
  parser.add_argument('--path_video_dataset', type=str, default=os.path.join('partA', 'video', 'video_frontalized_interpolated_mirror'), help='Path to video dataset.')
  parser.add_argument('--load_dataset_in_memory', type=int, default=0, help='Load dataset in RAM.')
  parser.add_argument('--is_subject_independent', type=int, choices=[0, 1], default=1, help='Subject-independent split.')
  parser.add_argument('--skip_test', type=int, default=0, help='Skip test phase.')
  parser.add_argument('--sampler_loader_type', type=str,nargs='*', default=['standard'], help='Training batch sampler: standard (shuffled batches) or selective_augm (set also filtered_augm_n_keep and filtered_augm_strategy).')
  parser.add_argument('--filtered_augm_n_keep', nargs='*', type=int, default=[0], help='Samples to keep from each augmentation type.')
  parser.add_argument('--filtered_augm_strategy', type=int, nargs='*', default=[0], help="Strategy for filtered augmentation sampler."+
                      " 1: Randomly keep n samples from what augmentation is available (all h_flip, all jitter, etc.)"+
                      " 2: Randomly choose n types from GLOBAL available types where"+
                      " n is --filtered_augm_n_keep")
  parser.add_argument('--keep_original', type=float, nargs='*', default=[1.0],
                      help='Fraction of original samples to keep per epoch (0-1). When < 1, selected originals'
                      ' exclude their augmentations; non-selected originals are excluded but their augmentations'
                      ' remain available via the augmentation strategy.')
  parser.add_argument('--undersample_max_per_class', type=int, nargs='*', default=[0],
                      help='Max base-sample groups per class per epoch for the selective_augm sampler (undersampling).'
                      ' 0 = disabled, -1 = match the minority class group count, N > 0 = absolute cap.'
                      ' The subset kept for over-represented classes is redrawn randomly each epoch.')
  parser.add_argument('--save_model_every_n_epochs', type=int, default=0, help='Save model every n epochs.')
  parser.add_argument('--lr', type=float, nargs='*', default=[0.0001], help='Learning rate(s)')
  parser.add_argument('--opt', type=str, nargs='*', default=['adamw'], help="Optimizer(s).")
  parser.add_argument('--validation_enabled', type=int, choices=[0, 1], default=1, help='Enable validation.')
  parser.add_argument('--ep', type=int, default=50, help='Number of epochs')
  parser.add_argument('--k_fold', type=int, default=3, help='Number of k-fold splits')
  parser.add_argument('--batch_train', type=int, nargs='*', default=[64], help='Training batch size(s)')
  parser.add_argument('--stop', type=int, nargs='*', default=None, help='Stop after [kth fold, ith subfold]')
  parser.add_argument('--concatenate_quadrants', type=int, nargs='*', default=[0], help='Concatenate quadrants dimension.')
  parser.add_argument('--loss', type=str, nargs='*', default=None, help="Loss function.")
  parser.add_argument('--stratified_training', type=int, nargs='*', default=[0],
                      help='1: add every available augmentation of each training sample to the training CSV; 0: disabled.')
  parser.add_argument('--target_samples_per_class_training', type=int, nargs='*', default=[None],
                      help='Target number of samples per class during training. Uses generate_balanced_dataframe to '
                           'add augmented samples until each class reaches this count. Dataset-agnostic.')
  parser.add_argument('--num_heads', type=int, nargs='*', default=[8], help='Number of heads.')
  parser.add_argument('--num_cross_head', type=int, nargs='*', default=[8], help='Number of cross heads.')
  parser.add_argument('--model_dropout', type=float, nargs='*', default=[0.0], help='Model dropout.')
  parser.add_argument('--drop_attn', type=float, nargs='*', default=[0.0], help='Attention dropout.')
  parser.add_argument('--drop_residual', type=float, nargs='*', default=[0.0], help='Residual dropout.')
  parser.add_argument('--mlp_ratio', type=float, nargs='*', default=[2.0], help='MLP ratio.')
  parser.add_argument('--custom_mlp', type=int, nargs='*', default=[0], help='Use custom MLP.')
  parser.add_argument('--mlp_num_hidden_layers', type=int, nargs='*', default=[2], help='Number of hidden layers in MLP_custom (used when custom_mlp=1).')

  parser.add_argument('--pos_enc', type=int, nargs='*', default=[0], help='Use positional encoding.')
  parser.add_argument('--nr_blocks', type=int, nargs='*', default=[1], help='Number of blocks.')
  parser.add_argument('--cross_block_after_transformers', type=int, nargs='*', default=[0], help='Cross block after transformers.')
  parser.add_argument('--complete_block', type=int, nargs='*', default=[1], help='Use complete block.')
  parser.add_argument('--init_network', type=str, nargs='*', default=['default'], help='Network initialization.')
  parser.add_argument('--regulariz_lambda_L1', type=float, nargs='*', default=[0], help='L1 regularization.')
  parser.add_argument('--regulariz_lambda_L2', type=float, nargs='*', default=[0], help='L2 regularization.')
  parser.add_argument('--label_smooth', type=float, nargs='*', default=[0.0], help='Label smoothing.')
  parser.add_argument(
    '--key_early_stopping', type=str, default=None,
    help=(
      'Metric for early stopping and best-model selection. The loss family '
      'enforces val_accuracy for CE-family losses and val_loss otherwise.'
    ),
  )
  parser.add_argument('--save_best_model', action='store_true', help='Save best model.')
  parser.add_argument('--scheduler_name', type=str, nargs='*', default=['cosine'], help='Learning-rate scheduler. Choices: cosine, cosine_restart. Default: cosine.')
  parser.add_argument('--min_lr', type=float, nargs='*', default=[1e-7], help='Min lr.')
  parser.add_argument('--warm_up_epochs', type=float, nargs='*', default=[None], help='Warm up epochs.')
  parser.add_argument('--warm_up_scheduler', type=str, nargs='*', default=[None], help='Warm up scheduler.')
  parser.add_argument('--warm_up_start_factor', type=float, nargs='*', default=[None], help='Warm up start factor.')
  parser.add_argument('--first_restart_epochs', type=int, nargs='*', default=[None], help='First restart.')
  parser.add_argument('--multiplier_restart', type=int, nargs='*', default=[None], help='Multiplier restart.')
  parser.add_argument(
    '--pruner_threshold_lower', type=float, default=0.20,
    help=(
      'Directional pruning threshold: accuracy below this value or loss '
      'above this value is pruned.'
    ),
  )
  parser.add_argument('--pruner_n_warmup_steps', type=int, default=30, help='Pruner warmup.')
  parser.add_argument('--optuna_categorical', type=int, default=1, help='Use categorical optimization.')
  parser.add_argument('--optuna_sampler', type=str, default='auto', help='Optuna sampler.')
  parser.add_argument('--n_trials', type=int, default=100, help='Number of trials.')
  parser.add_argument('--timeout', type=float, default=14, help='Timeout in hours.')

  parser.add_argument('--normalize_labels', type=int, choices=[0, 1], default=0,
                      help='Min-max normalize continuous labels to [0,1] using observed bounds. '
                           'All logged metrics are reported in the original label scale.')

  # YAML values become defaults after all add_argument calls, so argparse's own defaults don't win.
  actions = {a.dest: a for a in parser._actions}
  unknown = set(yaml_defaults) - set(actions)
  if unknown:
    parser.error(f'Unknown keys in {pre_args.config}: {sorted(unknown)}')
  for k, v in yaml_defaults.items():  # nargs='*' args are iterated as grids downstream
    if actions[k].nargs == '*' and not isinstance(v, list):
      yaml_defaults[k] = [v]
  parser.set_defaults(**yaml_defaults)

  args = parser.parse_args()
  args.timeout = int(args.timeout * 3600)  # Convert hours to seconds
  if args.gp:
    args.csv = helper.GLOBAL_PATH.get_global_path(args.csv)
  dict_args = vars(args)
  configure_csv_input(dict_args)

  # --- Global Constants & Side Effects ---
  requested_selection_metric = dict_args['key_early_stopping']
  configure_selection_policy(dict_args)
  if requested_selection_metric is None:
    print(f"\n*********key_early_stopping not set. Set: {dict_args['key_early_stopping']}*********\n")
  target_spec = configure_target_spec(dict_args)
  validate_arguments(dict_args)

  if dict_args['normalize_labels']:
    print(
      f"\nLabel normalization enabled. observed range = "
      f"[{target_spec.target_min}, {target_spec.target_max}]\n"
    )
  else:
    print(f"\nLabel normalization disabled. observed range = [{target_spec.target_min}, {target_spec.target_max}]\n")

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
  
  if dict_args['save_best_model']:
    helper.SAVE_PTH_MODEL = True

  if not dict_args['validation_enabled']:
    print("\nValidation is disabled. Last epoch model will be used for testing \n")

  # Generate unique folder name
  timestamp = int(time.time())
  server_name = platform.node()
  pid = os.getpid()
  args.global_folder_name = f'{args.global_folder_name}_{pid}_{args.head}_{server_name}_{timestamp}'

  if args.gp:
    args.ffsp = helper.GLOBAL_PATH.get_global_path(args.ffsp)
    args.path_video_dataset = helper.GLOBAL_PATH.get_global_path(args.path_video_dataset)
    args.global_folder_name = helper.GLOBAL_PATH.get_global_path(args.global_folder_name)

  # print('\n\nTraining configuration:')
  # print(args)

  # Training always runs every epoch; the best epoch is selected on key_early_stopping.
  dict_args['early_stopping'] = earlyStoppingDummy()

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

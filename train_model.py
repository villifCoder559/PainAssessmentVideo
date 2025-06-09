import os
os.environ["OPTUNA_DISABLE_TELEMETRY"] = "1"
# os.environ["TMPDIR"] = "tmp"
import cdw_cross_entropy_loss.cdw_cross_entropy_loss
from custom.helper import CLIPS_REDUCTION, EMBEDDING_REDUCTION, MODEL_TYPE, SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import time
import math
import argparse
import torch.nn as nn
import torch.optim as optim
from custom.head import earlyStoppingAccuracy, earlyStoppingLoss
import custom.scripts as scripts
import custom.helper as helper
import platform
import pandas as pd
# import cProfile, pstats
import numpy as np
from pstats import SortKey
import optuna
from copy import deepcopy
import pickle
from cdw_cross_entropy_loss import cdw_cross_entropy_loss
import optunahub

# ------------ Helper Functions ------------


def get_optimizer(opt):
  """Get optimizer class from string name."""
  optimizers = {
    'adam': optim.Adam,
    'sgd': optim.SGD,
    'adamw': optim.AdamW
  }
  opt = opt.lower()
  if opt not in optimizers:
    raise ValueError(f'Optimizer not found: {opt}. Valid options: {list(optimizers.keys())}')
  return optimizers[opt]
  
def get_loss(loss,dict_args=None):
  """Get loss function from string name."""
  losses = {
    'l1': nn.L1Loss(),
    'l2': nn.MSELoss(),
    'ce': nn.CrossEntropyLoss(),
    'cdw_ce': cdw_cross_entropy_loss.CDW_CELoss(num_classes=dict_args['num_classes'],
                                                alpha=dict_args['alpha'],
                                                transform=dict_args['transform']),
  }
  loss = loss.lower()
  if loss not in losses:
    raise ValueError(f'Loss not found: {loss}. Valid options: {list(losses.keys())}')
  return losses[loss]



def get_head(head_name):
  """Get HEAD enum from string name."""
  heads = {
    'gru': HEAD.GRU,
    'attentive': HEAD.ATTENTIVE,
    'linear': HEAD.LINEAR,
    'attentive_jepa': HEAD.ATTENTIVE_JEPA
  }
  head_name = head_name.lower()
  if head_name not in heads:
    raise ValueError(f'Head not found: {head_name}. Valid options: {list(heads.keys())}')
  return heads[head_name]

def get_embedding_reduction(reduction):
  """Get EMBEDDING_REDUCTION enum from string name."""
  reductions = {
    'temporal': EMBEDDING_REDUCTION.MEAN_TEMPORAL,
    'spatial': EMBEDDING_REDUCTION.MEAN_SPATIAL,
    'temporal_spatial': EMBEDDING_REDUCTION.MEAN_TEMPORAL_SPATIAL,
    'none': EMBEDDING_REDUCTION.NONE
  }
  reduction = reduction.lower()
  if reduction not in reductions:
    raise ValueError(f'Embedding reduction not found: {reduction}. Valid options: {list(reductions.keys())}')
  return reductions[reduction]

# ------------ Main Entry Point ------------

log_scale_optuna_hypers = ['lr','regulariz_lambda_L1','regulariz_lambda_L2']

def _suggest(trial: optuna.trial.Trial, name, values, categorical):
  """Suggest a hyperparameter value based on the type of values provided."""
  if categorical:
    return trial.suggest_categorical(name, values)

  if len(values) == 1:
    print(f"Only one value provided for {name}. Using {values[0]}.")
    return trial.suggest_categorical(name, values)
    # raise ValueError(f"Only one value provided for {name}. Please provide a range of values. Otherwise use categorical=1.")
  low, high = values[0], values[-1]
  if low > high:
    low, high = high, low
  if isinstance(low, int) and isinstance(high, int):
    return trial.suggest_int(name, low=low, high=high)
  else:
    return trial.suggest_float(name, low=low, high=high, log=True if name in log_scale_optuna_hypers else False)

def objective(trial: optuna.trial.Trial, original_kwargs):
  """Objective function for Optuna hyperparameter optimization."""
  kwargs = deepcopy(original_kwargs)
  model_type = MODEL_TYPE.get_model_type(kwargs['mt'])
  epochs = kwargs['ep']

  # common training params
  lr = _suggest(trial, 'lr', kwargs['lr'], kwargs['optuna_categorical'])
  opt = trial.suggest_categorical('opt', kwargs['opt'])
  optimizer_fn = get_optimizer(opt)
  init_network = trial.suggest_categorical('init_network', kwargs['init_network'])
  batch_train = trial.suggest_categorical('batch_train', kwargs['batch_train'])
  regulariz_lambda_L1 = _suggest(trial, 'regulariz_lambda_L1', kwargs['regulariz_lambda_L1'], kwargs['optuna_categorical'])
  regulariz_lambda_L2 = _suggest(trial, 'regulariz_lambda_L2', kwargs['regulariz_lambda_L2'], kwargs['optuna_categorical'])
  label_smooth = trial.suggest_categorical('label_smooth', kwargs['label_smooth'])
  concatenate_temp_dim = trial.suggest_categorical('concatenate_temp_dim', kwargs['concatenate_temp_dim'])
  loss = trial.suggest_categorical('loss', kwargs['loss'])
  cdw_ce_alpha = _suggest(trial, 'cdw_ce_alpha', kwargs['cdw_ce_alpha'], kwargs['optuna_categorical'])
  cdw_ce_power_transform = trial.suggest_categorical('cdw_ce_power_transform', kwargs['cdw_ce_transform'])
  if loss == 'cdw_ce' and label_smooth > 0:
    raise ValueError('Label smoothing not supported for CDW loss. Use label_smooth=0.')
  
  # augmentation
  hflip = _suggest(trial, 'hflip', kwargs['hflip'], kwargs['optuna_categorical'])
  color_jitter = _suggest(trial, 'color_jitter', kwargs['jitter'], kwargs['optuna_categorical'])
  rotation = _suggest(trial, 'rotation', kwargs['rotation'], kwargs['optuna_categorical'])

  # choose head-specific hyperparameters
  head_name = kwargs['head']
  if head_name.upper() == 'GRU':
    # Sample GRU-specific params
    GRU_hidden_size = _suggest(trial, 'GRU_hidden_size', kwargs['GRU_hidden_size'], kwargs['optuna_categorical'])
    GRU_num_layers = _suggest(trial, 'GRU_num_layers', kwargs['GRU_num_layers'], kwargs['optuna_categorical'])
    GRU_output_size = kwargs['GRU_output_size']
    layer_norm = trial.suggest_categorical('layer_norm', kwargs['layer_norm'])
    GRU_dropout = _suggest(trial, 'GRU_dropout', kwargs['GRU_dropout'], kwargs['optuna_categorical'])
    GRU_round_output_loss = trial.suggest_categorical('GRU_round_output_loss', kwargs['GRU_round_output_loss'])
    # Build head parameters
    emb_dim = MODEL_TYPE.get_embedding_size(kwargs['mt'])
    params = {
      'input_dim': emb_dim * 8 if concatenate_temp_dim else emb_dim,
      'hidden_size': GRU_hidden_size,
      'num_layers': GRU_num_layers,
      'output_size': GRU_output_size,
      'layer_norm': layer_norm,
      'dropout': GRU_dropout,
    }
    head_enum = HEAD.GRU
    num_classes = GRU_output_size

  else:
    # JEPA-attentive head parameters
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
    cross_block_after_transformers = trial.suggest_categorical(
      'cross_block_after_transformers', kwargs['cross_block_after_transformers'])

    emb_dim = MODEL_TYPE.get_embedding_size(kwargs['mt'])
    num_classes = pd.read_csv(kwargs['csv'], sep='\t')['class_id'].nunique()
    params = {
      'input_dim': emb_dim * 8 if concatenate_temp_dim else emb_dim,
      'num_classes': num_classes,
      'num_cross_heads': num_cross_head,
      'num_heads': num_heads or num_cross_head,
      'dropout': model_dropout,
      'attn_dropout': drop_attn,
      'residual_dropout': drop_residual,
      'mlp_ratio': mlp_ratio,
      'pos_enc': pos_enc,
      'depth': nr_blocks,
      'cross_block_after_transformers': cross_block_after_transformers,
      'num_queries': num_queries,
      'agg_method': queries_agg_method,
    }
    head_enum = HEAD.ATTENTIVE_JEPA

  # avoid duplicate trials
  for past in trial.study.trials:
    if past.state not in (
      optuna.trial.TrialState.COMPLETE,
      optuna.trial.TrialState.PRUNED
    ):
      continue
    if past.params == trial.params:
      return past.value

  if num_classes == 1 and (loss == 'cdw_ce' or loss == 'ce'):
    raise ValueError('Loss function not supported for regression. Use l1 or l2 loss.')
  dict_args_loss = {'num_classes': num_classes, 
                    'alpha': cdw_ce_alpha, 
                    'transform': cdw_ce_power_transform}
  
  run_folder_path, results = scripts.run_train_test(
    model_type=model_type,
    criterion=get_loss(loss, dict_args=dict_args_loss),
    concatenate_temp_dim=concatenate_temp_dim,
    pooling_embedding_reduction=kwargs['pooling_clips_reduction'],
    pooling_clips_reduction=kwargs['pooling_clips_reduction'],
    sample_frame_strategy=kwargs['sample_frame_strategy'],
    path_csv_dataset=kwargs['csv'],
    path_video_dataset=kwargs['path_video_dataset'],
    head=head_enum,
    stride_window_in_video=kwargs['stride_window_in_video'],
    features_folder_saving_path=kwargs['ffsp'],
    head_params=params,
    k_fold=kwargs['k_fold'],
    global_foder_name=kwargs['global_folder_name'],
    batch_size_training=batch_train,
    epochs=epochs,
    optimizer_fn=optimizer_fn,
    lr=lr,
    seed_random_state=seed_random_state,
    is_plot_dataset_distribution=False,
    is_round_output_loss=GRU_round_output_loss if head_enum == HEAD.GRU else 0,
    is_shuffle_video_chunks=False,
    is_shuffle_training_batch=True,
    init_network=init_network,
    key_for_early_stopping=kwargs['key_early_stopping'],
    regularization_lambda_L1=regulariz_lambda_L1,
    regularization_lambda_L2=regulariz_lambda_L2,
    clip_length=clip_length,
    target_metric_best_model=target_metric_best_model,
    early_stopping=early_stopping,
    enable_scheduler=kwargs['enable_scheduler'],
    stop_after_kth_fold=kwargs['stop'],
    n_workers=kwargs['n_workers'],
    clip_grad_norm=kwargs['clip_grad_norm'],
    label_smooth=label_smooth,
    dict_augmented={
      'hflip': hflip,
      'jitter': color_jitter,
      'rotation': rotation
    },
    trial=trial,
    prefetch_factor=kwargs['prefetch_factor']
  )
    # save_stats(pr, os.path.join(run_folder_path, 'profiling_results.txt'))
    # pr.dump_stats(os.path.join(run_folder_path, 'profiling_results.prof'))

  trial.set_user_attr('id_test', os.path.basename(run_folder_path).split('_')[0])

  # record config
  trial.set_user_attr('head', [kwargs['head']])
  trial.set_user_attr('csv', [kwargs['csv']])
  trial.set_user_attr('ffsp', [kwargs['ffsp']])
  trial.set_user_attr('k_fold', [kwargs['k_fold']])
  trial.set_user_attr('stop', kwargs['stop'])
  trial.set_user_attr('n_workers', [kwargs['n_workers']])

  # return metric
  mean_val_accuracy = get_mean_val_accuracy(results)
  return mean_val_accuracy

# def hyper_search_gru(kwargs):
#   """Hyperparameter search for GRU head using Optuna."""
#   study = optuna.create_study(
#     direction='maximize',
#     storage=f"sqlite:///{os.path.join(kwargs['global_folder_name'], 'optuna_gru_study.db')}",
#     study_name=f"GRU_{os.path.basename(kwargs['global_folder_name'])}",
#     pruner=optuna.pruners.ThresholdPruner(
#       lower=kwargs['--pruner_n_warmup_steps'],
#       n_warmup_steps=30,
#       interval_steps=2
#     )
#   )
#   study.optimize(lambda trial: objective(trial, kwargs), n_trials=kwargs['n_trials'], timeout=kwargs['timeout'])

#   # save study
#   optuna_path = os.path.join(kwargs['global_folder_name'], 'optuna_gru_study.pkl')
#   with open(optuna_path, 'wb') as f:
#     pickle.dump(study, f)
#     print(f'Study saved to {optuna_path}')

#   print('Best hyperparameters:')
#   print(study.best_params)
#   print('Best accuracy:', study.best_value)

# def objective(trial:optuna.trial.Trial, original_kwargs):
#   """Objective function for Optuna hyperparameter optimization."""
#   kwargs = deepcopy(original_kwargs)
  
#   # Define the hyperparameters to be optimized
#   model_type = MODEL_TYPE.get_model_type(kwargs['mt'])
#   epochs = kwargs['ep']
  
#   # NOT SUPPORTED BY OPTUNA. TODO: find workaround
#   # if nr_blocks == 1: # if nr_blocks == 1, there is only the cross-attention block
#   #   kwargs['num_heads'] = [0]
#   #   kwargs['cross_block_after_transformers'] = [0]
  
#   # Training params  
  
#   nr_blocks = _suggest(trial, 'nr_blocks', kwargs['nr_blocks'], kwargs['optuna_categorical'])
#   lr = _suggest(trial, 'lr', kwargs['lr'], kwargs['optuna_categorical'])
#   opt = trial.suggest_categorical('opt', kwargs['opt'])
#   optimizer_fn = get_optimizer(opt)
#   init_network = trial.suggest_categorical('init_network', kwargs['init_network'])
#   batch_train = _suggest(trial, 'batch_train', kwargs['batch_train'], kwargs['optuna_categorical'])
#   regulariz_lambda_L1 = _suggest(trial, 'regulariz_lambda_L1', kwargs['regulariz_lambda_L1'], kwargs['optuna_categorical'])
#   regulariz_lambda_L2 = _suggest(trial, 'regulariz_lambda_L2', kwargs['regulariz_lambda_L2'], kwargs['optuna_categorical'])
#   label_smooth = _suggest(trial, 'label_smooth', kwargs['label_smooth'], kwargs['optuna_categorical'])
  
#   concatenate_temp_dim=trial.suggest_categorical('concatenate_temp_dim',kwargs['concatenate_temp_dim'])
  
#   # augmentation
#   hflip = _suggest(trial, 'hflip', kwargs['hflip'], kwargs['optuna_categorical'])
#   color_jitter = _suggest(trial, 'color_jitter', kwargs['jitter'], kwargs['optuna_categorical'])
#   rotation = _suggest(trial, 'rotation', kwargs['rotation'], kwargs['optuna_categorical'])
  
#   # jepa_attentive params
#   num_heads = _suggest(trial, 'num_heads', kwargs['num_heads'], kwargs['optuna_categorical'])
#   num_cross_head = _suggest(trial, 'num_cross_head', kwargs['num_cross_head'], kwargs['optuna_categorical'])
#   model_dropout = _suggest(trial, 'model_dropout', kwargs['model_dropout'], kwargs['optuna_categorical'])
#   drop_attn = _suggest(trial, 'drop_attn', kwargs['drop_attn'], kwargs['optuna_categorical'])
#   drop_residual = _suggest(trial, 'drop_residual', kwargs['drop_residual'], kwargs['optuna_categorical'])
#   mlp_ratio = _suggest(trial, 'mlp_ratio', kwargs['mlp_ratio'], kwargs['optuna_categorical'])
#   pos_enc = trial.suggest_categorical('pos_enc', kwargs['pos_enc'])
#   cross_block_after_transformers = trial.suggest_categorical('cross_block_after_transformers', kwargs['cross_block_after_transformers'])
  
#   emb_dim = MODEL_TYPE.get_embedding_size(kwargs['mt'])
#   num_classes = pd.read_csv(kwargs['csv'],sep='\t')['class_id'].unique().shape[0]  
#   params = {
#     'input_dim': emb_dim * 8 if concatenate_temp_dim else emb_dim,
#     'num_classes': num_classes,
#     'num_cross_heads': num_cross_head,
#     'num_heads': num_heads if num_heads is not None else num_cross_head,
#     'dropout': model_dropout,
#     'attn_dropout': drop_attn,
#     'residual_dropout': drop_residual,
#     'mlp_ratio': mlp_ratio,
#     'pos_enc': pos_enc,
#     'depth': nr_blocks,
#     'cross_block_after_transformers': cross_block_after_transformers}
  
#   # Check if the hyperparameters have been tried before
#   for past in trial.study.trials:
#     if past.state != optuna.trial.TrialState.COMPLETE and past.state != optuna.trial.TrialState.PRUNED:
#       continue
#     if past.params == trial.params:
#       return past.value
    
#   # Call the training function with the suggested hyperparameters
#   with cProfile.Profile() as pr:
#     run_folder_path,results = scripts.run_train_test(
#                                     model_type=model_type, 
#                                     criterion=get_loss('CE'), 
#                                     concatenate_temp_dim=concatenate_temp_dim,
#                                     pooling_embedding_reduction=kwargs['pooling_clips_reduction'],
#                                     pooling_clips_reduction=kwargs['pooling_clips_reduction'],
#                                     sample_frame_strategy=kwargs['sample_frame_strategy'], 
#                                     path_csv_dataset=kwargs['csv'], 
#                                     path_video_dataset=kwargs['path_video_dataset'],
#                                     head=HEAD.ATTENTIVE_JEPA,
#                                     stride_window_in_video=kwargs['stride_window_in_video'],
#                                     features_folder_saving_path=kwargs['ffsp'],
#                                     head_params=params,
#                                     k_fold=kwargs['k_fold'],
#                                     global_foder_name=kwargs['global_folder_name'], 
#                                     batch_size_training=batch_train, 
#                                     epochs=epochs, 
#                                     optimizer_fn=optimizer_fn,
#                                     lr=lr,
#                                     seed_random_state=seed_random_state,
#                                     is_plot_dataset_distribution=False,
#                                     is_round_output_loss=kwargs['is_round_output_loss'],
#                                     is_shuffle_video_chunks=False,
#                                     is_shuffle_training_batch=True,
#                                     init_network=init_network,
#                                     key_for_early_stopping=kwargs['key_early_stopping'],
#                                     regularization_lambda_L1=regulariz_lambda_L1,
#                                     regularization_lambda_L2=regulariz_lambda_L2,
#                                     clip_length=clip_length,
#                                     target_metric_best_model=target_metric_best_model,
#                                     early_stopping=early_stopping,
#                                     enable_scheduler=kwargs['enable_scheduler'],
#                                     stop_after_kth_fold=kwargs['stop'],
#                                     n_workers=kwargs['n_workers'],
#                                     clip_grad_norm=kwargs['clip_grad_norm'],
#                                     label_smooth=label_smooth,
#                                     dict_augmented={'hflip': hflip,
#                                                     'jitter': color_jitter,
#                                                     'rotation': rotation},
#                                     trial=trial
#                                   )
#     save_stats(pr, os.path.join(run_folder_path, 'profiling_results.txt'))
#     pr.dump_stats(os.path.join(run_folder_path, 'profiling_results.prof'))
    
#   id_test = os.path.split(run_folder_path)[-1].split('_')[0]
#   trial.set_user_attr('id_test', id_test)
#   # Just to record this data
#   trial.set_user_attr('head', [kwargs['head']])
#   trial.set_user_attr('csv', [kwargs['csv']])
#   trial.set_user_attr('ffsp', [kwargs['ffsp']])
#   trial.set_user_attr('k_fold', [kwargs['k_fold']])
#   trial.set_user_attr('stop', kwargs['stop'])
#   trial.set_user_attr('n_workers', [kwargs['n_workers']])
  
#   # Extract the validation accuracy from the results
#   mean_val_accuracy = get_mean_val_accuracy(results)
#   return mean_val_accuracy

def get_mean_val_accuracy(results):
  list_val_accuracy = []
  for k_fold,dict_log_k_fold in results['results'].items():
    best_epoch = dict_log_k_fold['train_val']['best_model_idx']
    list_val_accuracy.append(dict_log_k_fold['train_val']['list_val_performance_metric'][best_epoch])
  return np.mean(list_val_accuracy)

def get_sampler_module(sampler_name,kwargs=None):
  if sampler_name.lower() == 'tpe':
    return optuna.samplers.TPESampler()
  elif sampler_name.lower() == 'random':
    return optuna.samplers.RandomSampler()
  elif sampler_name.lower() == 'auto':
    sampler_module = optunahub.load_module(package="samplers/auto_sampler")
    sampler_module = sampler_module.AutoSampler()
    return sampler_module
  elif sampler_name.lower() == 'grid':
    search_space = {k:v for k,v in kwargs.items() if isinstance(v, list) and k != 'stop'}
    sampler_module = optuna.samplers.GridSampler(search_space=search_space)
    print(f'\n Grid search trials: {len(sampler_module._all_grids)}\n')
    return sampler_module
  else:
    raise ValueError(f"Invalid sampler name: {sampler_name}. Must be one of ['tpe', 'random', 'auto', 'grid']")
  
def hyper_search(kwargs):
  """Hyperparameter search for AttentiveJepa head using Optuna."""
  # Define the study
  study_name = f'{kwargs["head"]}_{os.path.split(kwargs["global_folder_name"])[-1].split("_")[-1]}'
  try:
    sampler_module = get_sampler_module(kwargs['optuna_sampler'], kwargs)
  except SystemError as e:
    print(f"Error loading sampler module: {e}. Using default TPE sampler.")
    sampler_module = optuna.samplers.TPESampler()
  study = optuna.create_study(direction='maximize',
                              sampler=sampler_module,
                              storage=f'sqlite:///{os.path.join(kwargs["global_folder_name"],f"{study_name}.db")}',
                              study_name=study_name,
                              pruner=optuna.pruners.ThresholdPruner(lower=dict_args['pruner_threshold_lower'],
                                                                    n_warmup_steps=dict_args['pruner_n_warmup_steps'],
                                                                    interval_steps=2))
  
  # Optimize the objective function
  study.optimize(lambda trial: objective(trial, kwargs), 
                 n_trials=kwargs['n_trials'],
                 timeout=kwargs['timeout'])
  optuna_path = os.path.join(kwargs['global_folder_name'],f'{study_name}.pkl')
  with open(optuna_path,"wb") as f:
    pickle.dump(study, f)
    print(f'Study saved to {optuna_path}')
  # Print the best hyperparameters and their corresponding accuracy
  print('Best hyperparameters:')
  print(study.best_params)
  print('Best accuracy:', study.best_value)
  # len of trials
  if len(study.trials) > 1:
    try:
      fig_hyper_importance = optuna.visualization.plot_param_importances(study)
      fig_hyper_importance.update_layout(width=1300, height=900)
      fig_hyper_importance.write_image(os.path.join(kwargs['global_folder_name'],f'{study_name}_hyper_importance.png'))
      print(f"Hyperparameter importance plot saved to {os.path.join(kwargs['global_folder_name'],f'{study_name}_hyper_importance.png')}")
    except Exception as e:
      print(f"Error generating hyperparameter importance plot: {e}")
  
  
if __name__ == '__main__':
  # mp.set_start_method('spawn', force=True)
  # Set up argument parser
  parser = argparse.ArgumentParser(description='Train video analysis model with various configurations')
  
  
  # Model configuration
  parser.add_argument('--mt', type=str, default='B', help='Model type: B (Base), S (Small), or I (Image)')
  parser.add_argument('--head', type=str, default='ATTENTIVE_JEPA', help='Head type: GRU, ATTENTIVE, LINEAR, ATTENTIVE_JEPA')
  parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for data loading. Default is 1')
  parser.add_argument('--prefetch_factor', type=int, default=2, help='Prefetch factor for data loading. Default is 2')
  
  
  # Path configuration
  parser.add_argument('--gp', action='store_true', help='Use global path prefix for file paths')
  parser.add_argument('--csv', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), 
                    help='Path to CSV dataset file')
  parser.add_argument('--ffsp', type=str, default="partA/video/features/samples_16_frontalized_new", 
                    help='Feature folder saving path')
  parser.add_argument('--global_folder_name', type=str, default=f'history_run', 
                    help='Global folder name for saving results')
  parser.add_argument('--path_video_dataset', type=str, default=os.path.join('partA','video','video_frontalized'), 
                    help='Path to video dataset')
  
  # Training parameters
  parser.add_argument('--lr', type=float, nargs='*', default=[0.0001], help='Learning rate(s)')
  parser.add_argument('--ep', type=int, default=50, help='Number of epochs')
  parser.add_argument('--k_fold', type=int, default=3, help='Number of k-fold cross validation splits')
  parser.add_argument('--opt', type=str, nargs='*', default=['adamw'], help='Optimizer(s): adam, sgd, adamw')
  parser.add_argument('--batch_train', type=int, nargs='*', default=[64], help='Training batch size(s)')
  parser.add_argument('--enable_scheduler', action='store_true', help='Enable learning rate scheduler')
  parser.add_argument('--stop', type=int,nargs='*' ,default=None, help='Stop after [kth fold, ith subfold]')
  parser.add_argument('--clip_grad_norm', type=float, default=None, help='Clip gradient norm. Default is None (not applied)')
  parser.add_argument('--concatenate_temp_dim', type=int, nargs='*', default=[0],
                    help='Concatenate temporal dimension in input to the model. (ex: the embeddind is [temporal*emb_dim]=6144 if model base)')
  parser.add_argument('--loss', type=str, nargs='*', default='ce', help='Loss function: l1, l2, ce, cdw_ce')
  parser.add_argument('--cdw_ce_alpha', type=float, nargs='*', default=[2], help='Alpha parameter for CDW loss.') 
  parser.add_argument('--cdw_ce_transform', type=str, nargs='*', default=['power'], help='Transform for CDW loss. Default is power, can Be also "huber" or "log"')
  
  # Attention parameters
  parser.add_argument('--num_heads', type=int, nargs='*',default=[8], help='Number of heads for attention in transformer (when nr_blocks >1). Default is 8')
  parser.add_argument('--num_cross_head',type=int, nargs='*',default=[8], help='Number of heads for cross-attention.')
  parser.add_argument('--model_dropout', type=float, nargs='*', default=[0.0], help='Model dropout rate(s). This is drop_mlp for AttentiveJepa')
  parser.add_argument('--drop_attn', type=float, nargs='*', default=[0.0], help='Attention dropout rate(s)')
  parser.add_argument('--drop_residual', type=float, nargs='*', default=[0.0], help='Residual dropout rate(s)')
  parser.add_argument('--mlp_ratio', type=float, nargs='*', default=[2.0], help='MLP ratio(s) for AttentiveJepa')
  parser.add_argument('--pos_enc', type=int,nargs='*',default=[0], help='Use positional encoding for Attentive head')
  parser.add_argument('--nr_blocks',type=int,nargs='*',default=[1], help='Number of blocks for Jepa Attentive head. Default is 1 (only cross-attention)')
  parser.add_argument('--cross_block_after_transformers', type=int,nargs='*', default=[0],
                    help='Use cross block after transformers for Jepa Attentive head')
  parser.add_argument('--num_queries', type=int, nargs='*', default=[1], help='Number of queries for Attentive head. Default is 1')
  parser.add_argument('--queries_agg_method',type=str, nargs='*',default=['mean'], help=f'Aggregation method for queries: {[print(agg_method) for agg_method in helper.QUERIES_AGG_METHOD]} . Default is mean')
  
  # Linear parameters
  parser.add_argument('--linear_dim_reduction', type=str, default='spatial', help=f'Dimension reduction for Linear head. Can be {[d.name.lower() for d in EMBEDDING_REDUCTION]}')
  
  # GRU parameters
  parser.add_argument('--GRU_hidden_size', type=int, nargs='*', default=[1024], help='GRU hidden layer size(s)')
  parser.add_argument('--GRU_num_layers', type=int, nargs='*', default=[2], help='GRU number of layers')
  parser.add_argument('--GRU_dropout', type=float, nargs='*', default=[0.0], help='GRU dropout rate(s)')
  parser.add_argument('--GRU_output_size', type=int, default=1, 
                    help='Output size of GRU: 1 for regression, >1 for classification')
  parser.add_argument('--layer_norm', type=int, nargs='*', default=[0],
                    help='Add Layer normalization before final linear layer') # Only in GRU
  parser.add_argument('--GRU_round_output_loss', type=int, default=[0],nargs='*',
                    help='Round output from regression before computing loss')
  
  # Network initialization
  parser.add_argument('--init_network', type=str, nargs='*', default=['default'], 
                    help='Network initialization: xavier, default, uniform. Default init. is "default"')
  parser.add_argument('--regulariz_lambda_L1', type=float, nargs='*', default=[0], help='Regularization strength(s) L1')
  parser.add_argument('--regulariz_lambda_L2', type=float, nargs='*', default=[0], help='Regularization strength(s) L2')
  parser.add_argument('--label_smooth', type=float, nargs='*',default=[0.0], help='Label smoothing factor. Default is 0.0 (no smoothing)')
  parser.add_argument('--loss_regression', type=str, default='L1', help='Regression loss function: L1 or L2. Default is L1')
  
  # Network augmentation
  parser.add_argument('--hflip', type=float,nargs='*', default=[0.0], help='Horizontal flip augmentation probability. Default is 0.0')
  parser.add_argument('--jitter', type=float,nargs='*',default=[0.0], help='Jitter augmentation probability. Default is 0.0')
  parser.add_argument('--rotation', type=float, nargs='*',default=[0.0], help='Rotation augmentation probability. Default is 0.0')
  
  # Early stopping parameters
  parser.add_argument('--key_early_stopping', type=str, default='val_accuracy', 
                    help='Metric for early stopping: val_loss or val_accuracy. Default is val_accuracy')
  parser.add_argument('--p_early_stop', type=int, default=2000, help='Patience for early stopping. Default is 2000')
  parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum delta for early stopping. Default is 0.001')
  parser.add_argument('--threshold_mode', type=str, default='abs', help='Early stopping threshold mode: abs or rel. Default is abs')
  
  # Logging parameters
  parser.add_argument('--log_grad_per_module', action='store_true',help='Log gradient per module')
  parser.add_argument('--log_history_sample', action='store_true', help='Log the train,val, test prediction of the model')
  parser.add_argument('--plot_live_loss', action='store_true', help='Plot live loss during training. Every 6 epochs')
  parser.add_argument('--log_xattn', action='store_true', help='Log cross attention weights')
  
  # Optuna parameters
  parser.add_argument('--pruner_threshold_lower', type=float, default=0.20, help='Threshold for Optuna pruner. Default is 0.2')
  parser.add_argument('--pruner_n_warmup_steps', type=int, default=30, help='Number of warmup steps for Optuna pruner. Default is 30')
  parser.add_argument('--optuna_categorical',type=int, default=1, help='Use categorical optimization for Optuna. Default is 1 (True). Otherwise, use continuous optimization in range [list[0],list[-1]]')
  parser.add_argument('--optuna_sampler', type=str, default='auto', help='Optuna sampler: tpe, random, auto, grid. Default is auto')
  parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna hyperparameter optimization. Default is 100')
  parser.add_argument('--timeout', type=float, default=14, help='Timeout for Optuna hyperparameter optimization in hours. Default is 14 hours')
  
  # Parse arguments
  args = parser.parse_args()
  args.timeout = int(args.timeout * 3600) # Convert hours to seconds

  dict_args = vars(args)
  if dict_args['plot_live_loss']:
    helper.PLOT_LIVE_LOSS = True
  
  if dict_args['log_xattn']:
    helper.LOG_CROSS_ATTENTION['enable']= True 
    helper.LOG_HISTORY_SAMPLE = True  # is True to get the prediction accoridng to the cross attention, so it's possible to check if the prediction makes sense
  
  if dict_args['log_history_sample']:
    helper.LOG_HISTORY_SAMPLE = True
    
  for method in dict_args['queries_agg_method']:
    if method not in helper.QUERIES_AGG_METHOD:
      raise ValueError(f"Invalid queries aggregation method: {method}. Must be one of {[m for m in helper.QUERIES_AGG_METHOD]}")
  
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  dict_args['pooling_clips_reduction'] = pooling_clips_reduction
  dict_args['sample_frame_strategy'] = sample_frame_strategy
  dict_args['stride_window_in_video'] = 16 # To avoid errors but not used
  # Generate timestamp for unique folder name
  timestamp = int(time.time())
  server_name = platform.node()
  pid = os.getpid()
  args.global_folder_name = f'{args.global_folder_name}_{pid}_{args.head}_{server_name}_{timestamp}'
  
    
  if args.log_grad_per_module:
    helper.LOG_GRADIENT_PER_MODULE = True
  # Apply global path prefixes if requested
  if args.gp:
    args.csv =helper.GLOBAL_PATH.get_global_path(args.csv)
    args.ffsp = helper.GLOBAL_PATH.get_global_path(args.ffsp)
    args.path_video_dataset = helper.GLOBAL_PATH.get_global_path(args.path_video_dataset)
    args.global_folder_name = helper.GLOBAL_PATH.get_global_path(args.global_folder_name)
  
  # Print all arguments
  print('\n\nTraining configuration:')
  print(args)
  
  # Set up fixed parameters
  clip_length = 16
  seed_random_state = 42
  
  # Determine target metric for model selection
  target_metric_best_model = args.key_early_stopping 
  
  # Configure early stopping
  if args.key_early_stopping == 'val_loss':
    early_stopping = earlyStoppingLoss(
      patience=args.p_early_stop, 
      min_delta=args.min_delta, 
      threshold_mode=args.threshold_mode
    )
  else:
    early_stopping = earlyStoppingAccuracy(
      patience=args.p_early_stop, 
      min_delta=args.min_delta, 
      threshold_mode=args.threshold_mode
    )
  
  print(f'Early stopping configuration: {str(early_stopping)}\n')
  dict_augmented = {
      'hflip': args.hflip,
      'jitter': args.jitter,
      'rotation': args.rotation
    }
  # Create config summary for later reference
  config_prompt = {
    'target_metric_best_model': target_metric_best_model,
    'seed_random_state': seed_random_state,
    'early_stopping': early_stopping,
    'clip_length': clip_length,
    'dict_augmented': dict_augmented,
    **dict_args
  }
  
  # Create output directory and save configuration
  if not os.path.exists(args.global_folder_name):
    os.makedirs(args.global_folder_name)
    
  with open(os.path.join(args.global_folder_name, '_config_prompt.txt'), 'w') as f:
    for key, value in config_prompt.items():
      f.write(f'{key}: {value}\n')
  
  # if args.head.upper() == 'ATTENTIVE_JEPA':
  hyper_search(dict_args)
  # elif args.head.upper() == 'GRU':
  #   hyper_search_gru(dict_args)
  # else:
  #   raise ValueError(f"Hyperparameter search not supported for head '{args.head}'")
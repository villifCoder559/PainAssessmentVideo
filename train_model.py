import os
os.environ["OPTUNA_DISABLE_TELEMETRY"] = "1"
# os.environ["TMPDIR"] = "tmp"
# import cdw_cross_entropy_loss.cdw_cross_entropy_loss
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
import torch
from pstats import SortKey
import optuna
from copy import deepcopy
import pickle
from cdw_cross_entropy_loss import cdw_cross_entropy_loss
import optunahub
from sim_loss.age_estimation.loss import SimLoss # type: ignore
from coral_pytorch.losses import coral_loss
import sys

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
 
def get_sampling_frame_startegy(strategy):
  """Get sampling strategy from string name."""
  strategy = strategy.lower()
  if strategy == SAMPLE_FRAME_STRATEGY.UNIFORM.value:
    return SAMPLE_FRAME_STRATEGY.UNIFORM
  elif strategy == SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW.value:
    return SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  elif strategy == SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING.value:
    return SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING
  elif strategy == SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING.value:
    return SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING 
  else:
    raise ValueError(f'Sampling strategy not found: {strategy}. Valid options: {list(SAMPLE_FRAME_STRATEGY)}')
  
def get_loss(loss, dict_args=None):
  """Get loss function from string name using if/elif."""
  loss = loss.lower()
  if loss == 'l1':
    return nn.L1Loss()
  elif loss == 'ce_weight':
    return nn.CrossEntropyLoss(weight=torch.tensor(dict_args['class_weights'], dtype=torch.float32, device='cuda'))
  elif loss == 'huber':
    return nn.HuberLoss(delta=dict_args['delta_huber'])
  elif loss == 'l2':
    return nn.MSELoss()
  elif loss == 'ce':
    return nn.CrossEntropyLoss()
  elif loss == 'cdw_ce':
    if dict_args is None:
      raise ValueError("dict_args is required for 'cdw_ce' loss.")
    return cdw_cross_entropy_loss.CDW_CELoss(
      num_classes=dict_args['num_classes'],
      alpha=dict_args['alpha'],
      transform=dict_args['transform']
    )
  elif loss == 'sim_loss':
    if dict_args is None:
      raise ValueError("dict_args is required for 'sim_loss'.")
    return SimLoss(
      number_of_classes=dict_args['num_classes'],
      reduction_factor=dict_args['sim_loss_reduction'],
      device='cuda'
    )
  elif loss == 'coral':
    return coral_loss
  else:
    raise ValueError(f'Loss not found: {loss}. Valid options: l1, l2, ce, cdw_ce, sim_loss, coral')

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
  soft_labels = _suggest(trial, 'soft_labels', kwargs['soft_labels'], kwargs['optuna_categorical'])
  sim_loss_reduction = _suggest(trial, 'sim_loss_reduction', kwargs['sim_loss_reduction'], kwargs['optuna_categorical'])
  delta_huber = _suggest(trial, 'delta_huber', kwargs['delta_huber'], kwargs['optuna_categorical'])
  if loss == 'cdw_ce' and label_smooth > 0:
    raise ValueError('Label smoothing not supported for CDW loss. Use label_smooth=0.')
  num_clips_per_video = trial.suggest_categorical('num_clips_per_video', kwargs['num_clips_per_video'])
  sample_frame_strategy = trial.suggest_categorical('sample_frame_strategy', kwargs['sample_frame_strategy'])
  stride_inside_window = trial.suggest_categorical('stride_inside_window', kwargs['stride_inside_window'])
  use_sdpa = trial.suggest_categorical('use_sdpa', kwargs['use_sdpa'])
  if use_sdpa and not hasattr(torch.nn.functional,"scaled_dot_product_attention"):
    raise ValueError("SDPA (Scaled Dot Product Attention) is not available in this PyTorch version. Set use_sdpa=0 or update PyTorch")
  
  # augmentation
  hflip = _suggest(trial, 'hflip', kwargs['hflip'], kwargs['optuna_categorical'])
  color_jitter = _suggest(trial, 'color_jitter', kwargs['jitter'], kwargs['optuna_categorical'])
  rotation = _suggest(trial, 'rotation', kwargs['rotation'], kwargs['optuna_categorical'])
  latent_basic = _suggest(trial, 'latent_basic', kwargs['latent_basic'], kwargs['optuna_categorical'])  
  latent_masking = _suggest(trial, 'latent_masking', kwargs['latent_masking'], kwargs['optuna_categorical'])
  
  
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
    head_params = {
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
    # Attentive head parameters
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
    emb_dim = MODEL_TYPE.get_embedding_size(kwargs['mt'])
    custom_mlp = trial.suggest_categorical('custom_mlp', kwargs['custom_mlp'])
    if loss in ['l1', 'l2', 'huber']:
      num_classes = 1
      print(f"\nRegression task detected. Setting num_classes to 1 for {loss} loss.")
    else:
      num_classes = pd.read_csv(kwargs['csv'], sep='\t')['class_id'].nunique()
    adapter_dict = {
      'type': trial.suggest_categorical('adapter_type', kwargs['adapter_type']),
      'mlp_ratio': trial.suggest_categorical('adpater_mlp_ratio', kwargs['adpater_mlp_ratio']),
      'kernel_size': trial.suggest_categorical('adapter_kernel_size', kwargs['adapter_kernel_size']),
      'dilation': trial.suggest_categorical('adapter_dilation', kwargs['adapter_dilation']),
      'num_adapters': trial.suggest_categorical('num_adapters', kwargs['num_adapters']),
    }
    head_params = {
      'input_dim': emb_dim * 8 if concatenate_temp_dim else emb_dim,
      'q_k_v_dim': q_k_v_dim if q_k_v_dim != None else emb_dim,
      'num_classes': num_classes,
      'num_cross_heads': num_cross_head,
      'num_heads': num_heads or num_cross_head,
      'dropout': model_dropout,
      'attn_dropout': drop_attn,
      'head_init_path': kwargs['head_init_path'],
      'residual_dropout': drop_residual,
      'mlp_ratio': mlp_ratio,
      'custom_mlp': custom_mlp,
      'pos_enc': pos_enc,
      'coral_loss': True if loss == 'coral' else False,
      'depth': nr_blocks,
      'cross_block_after_transformers': cross_block_after_transformers,
      'num_queries': num_queries,
      'agg_method': queries_agg_method,
      'complete_block': trial.suggest_categorical('complete_block', kwargs['complete_block']),
      'embedding_reduction': kwargs['embedding_reduction'],
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

  if num_classes == 1 and (loss == 'cdw_ce' or loss == 'ce' or loss == 'ce_weight' or loss == 'sim_loss'):
    raise ValueError('Loss function not supported for regression. Use l1 or l2 loss.')

  if loss == 'ce_weight':
    df = pd.read_csv(kwargs['csv'], sep='\t')
    class_counts = df['class_id'].value_counts(normalize=True).sort_index()
    class_weights = 1.0 / class_counts.values  # Inverse of frequency
    print(f"Class weights: {class_weights}")
  else:
    class_weights = None
    
  dict_args_loss = {'num_classes': num_classes, 
                    'alpha': cdw_ce_alpha, 
                    'sim_loss_reduction': sim_loss_reduction,
                    'delta_huber': delta_huber,
                    'class_weights': class_weights,
                    'transform': cdw_ce_power_transform}
  dict_augmented={
      'hflip': hflip,
      'jitter': color_jitter,
      'rotation': rotation,
      'latent_basic': latent_basic,
      'latent_masking': latent_masking
    }
  
  run_folder_path, results = scripts.run_train_test(
    load_dataset_in_memory=kwargs['load_dataset_in_memory'],
    model_type=model_type,
    soft_labels=soft_labels,
    criterion=get_loss(loss, dict_args=dict_args_loss),
    concatenate_temp_dim=concatenate_temp_dim,
    pooling_embedding_reduction=kwargs['pooling_embedding_reduction'],
    pooling_clips_reduction=kwargs['pooling_clips_reduction'],
    sample_frame_strategy=get_sampling_frame_startegy(sample_frame_strategy),
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
    dict_augmented=dict_augmented,
    trial=trial,
    prefetch_factor=kwargs['prefetch_factor'],
    validate = kwargs['validation_enabled']
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
  parser.add_argument('--embedding_reduction', type=str, default='none', help='Embedding reduction method: spatial, temporal, all, adaptive_pooling_3d, none. Default is spatial')
    
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

  # Data loading parameters
  parser.add_argument('--load_dataset_in_memory', type=int, default=0, help='Load the entire dataset into RAM memory. Default is 0 (False)')
  
  # Training parameters
  parser.add_argument('--validation_enabled', type=int, choices=[0,1], default=1, help='Enable validation set during training. Default is 1 (enabled)')
  parser.add_argument('--save_best_model', action='store_true', help='Save the best model')
  parser.add_argument('--save_last_epoch_model', action='store_true', help='Save the last epoch model')
  parser.add_argument('--train_amp_enabled', action='store_true', help='Enable AMP training')
  parser.add_argument('--train_amp_dtype', type=str, default=None, help='AMP training data type: bfloat16 or float16. Default is float16')
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
  parser.add_argument('--loss', type=str, nargs='*', default='ce', help='Loss function: l1, l2, ce, cdw_ce,sim_loss,huber, coral, ce_weight. Default is ce')
  parser.add_argument('--cdw_ce_alpha', type=float, nargs='*', default=[2], help='Alpha parameter for CDW loss.') 
  parser.add_argument('--cdw_ce_transform', type=str, nargs='*', default=['power'], help='Transform for CDW loss. Default is power, can Be also "huber" or "log"')
  parser.add_argument('--sim_loss_reduction', type=float, nargs='*', default=[0.0], help='Reduction factor for sim loss. Default is 0.0 (no reduction)')
  parser.add_argument('--delta_huber', type=float, nargs='*',default=[None], help='Delta parameter for Huber loss.')
  parser.add_argument('--num_clips_per_video',type=int, nargs='*', default=[1], help='Number of clips per video for random sampling strategy. Default is 1')
  parser.add_argument('--sample_frame_strategy', type=str, nargs='*', default=['sliding_window'],help=f'Sampling strategy for frames in a video when not use pre-computed feats: {list(SAMPLE_FRAME_STRATEGY)}. Default is sliding_window')
  parser.add_argument('--stride_inside_window', type=int, nargs='*', default=[1], help='Stride inside the sampling window. Default is 1')
  parser.add_argument('--use_sdpa', type=int, nargs='*', default=[1], help='Use SDPA (Scaled dot product attention) for backbone when feats are not precomputed. Default is 0 (not used)')
  
  # Attention parameters
  parser.add_argument('--num_heads', type=int, nargs='*',default=[8], help='Number of heads for attention in transformer (when nr_blocks >1). Default is 8')
  parser.add_argument('--num_cross_head',type=int, nargs='*',default=[8], help='Number of heads for cross-attention.')
  parser.add_argument('--model_dropout', type=float, nargs='*', default=[0.0], help='Model dropout rate(s). This is drop_mlp for AttentiveJepa')
  parser.add_argument('--drop_attn', type=float, nargs='*', default=[0.0], help='Attention dropout rate(s)')
  parser.add_argument('--drop_residual', type=float, nargs='*', default=[0.0], help='Residual dropout rate(s)')
  parser.add_argument('--mlp_ratio', type=float, nargs='*', default=[2.0], help='MLP ratio(s) for AttentiveJepa')
  parser.add_argument('--custom_mlp', type=int, nargs='*', default=[0],
                    help='Use custom MLP for AttentiveJepa head. Default is 0 (no custom MLP)')
  parser.add_argument('--pos_enc', type=int,nargs='*',default=[0], help='Use positional encoding for Attentive head')
  parser.add_argument('--nr_blocks',type=int,nargs='*',default=[1], help='Number of blocks for Jepa Attentive head. Default is 1 (only cross-attention)')
  parser.add_argument('--cross_block_after_transformers', type=int,nargs='*', default=[0],
                    help='Use cross block after transformers for Jepa Attentive head')
  parser.add_argument('--num_queries', type=int, nargs='*', default=[1], help='Number of queries for Attentive head. Default is 1')
  parser.add_argument('--queries_agg_method',type=str, nargs='*',default=['mean'], help=f'Aggregation method for queries: {[print(agg_method) for agg_method in helper.QUERIES_AGG_METHOD]} . Default is mean')
  parser.add_argument('--complete_block', type=int, nargs='*', default=[1],
                    help='Use complete block for Attentive head (after cross-attn there is MLP). Default is 1 (complete block), if 0 remove the MLP block after cross-attention')
  parser.add_argument('--q_k_v_dim', type=int, nargs='*', default=[None],
                    help='Dimension of query, key, value for Attentive head. Default is None (use input_dim)')
  
  # Adapter backbone finetuning parameters
  parser.add_argument('--adpater_mlp_ratio', type=float, nargs='*', default=[0.25], help='MLP ratio(s) for backbone adapter. Default is 0.25')
  parser.add_argument('--adapter_kernel_size', type=int, nargs='*', default=[3], help='Kernel size(s) for backbone adapter. Default is 3')
  parser.add_argument('--adapter_dilation', type=int, nargs='*', default=[1], help='Dilation(s) for backbone adapter. Default is 1')
  parser.add_argument('--adapter_type', type=str, nargs='*', default=['adapter'], help='Adapter type. Default is adapter (unique available)')
  parser.add_argument('--num_adapters', type=int, nargs='*', default=[0], help='Number of adapters to use. Default is 1')
  
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
  
  # Network regularization
  parser.add_argument('--init_network', type=str, nargs='*', default=['default'], 
                    help='Network initialization: xavier, default, uniform. Default init. is "default"')
  parser.add_argument('--regulariz_lambda_L1', type=float, nargs='*', default=[0], help='Regularization strength(s) L1')
  parser.add_argument('--regulariz_lambda_L2', type=float, nargs='*', default=[0], help='Regularization strength(s) L2')
  parser.add_argument('--label_smooth', type=float, nargs='*',default=[0.0], help='Label smoothing factor. Default is 0.0 (no smoothing)')
  parser.add_argument('--soft_labels', type=float, nargs='*', default=[0.0], help='Soft labels factor. Default is 0.0 (no soft labels), otherwise it will be use as hardness factor for the labels')
  parser.add_argument('--loss_regression', type=str, default='L1', help='Regression loss function: L1 or L2. Default is L1')
  parser.add_argument('--head_init_path', type=str, default=None, help='Path to pre-trained head weights. Default is None (no pre-trained weights)')
  
  # Network augmentation
  parser.add_argument('--hflip', type=float,nargs='*', default=[0.0], help='Horizontal flip augmentation probability. Default is 0.0')
  parser.add_argument('--jitter', type=float,nargs='*',default=[0.0], help='Jitter augmentation probability. Default is 0.0')
  parser.add_argument('--rotation', type=float, nargs='*',default=[0.0], help='Rotation augmentation probability. Default is 0.0')
  parser.add_argument('--latent_basic', type=float, nargs='*', default=[0.0], help='Latent basic augmentation probability. Default is 0.0')
  parser.add_argument('--latent_masking', type=float, nargs='*', default=[0.0], help='Latent masking augmentation (0.2). Default is 0.0')
  
  # Early stopping parameters
  parser.add_argument('--key_early_stopping', type=str, default='val_accuracy', 
                    help='Metric for early stopping: val_accuracy. Default is val_accuracy')
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
  
  # Set up fixed parameters
  clip_length = 16
  seed_random_state = 42
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  dict_args['pooling_embedding_reduction'] = helper.EMBEDDING_REDUCTION.get_embedding_reduction(dict_args['embedding_reduction'])
  dict_args['pooling_clips_reduction'] = pooling_clips_reduction
  dict_args['stride_window_in_video'] = 16 # To avoid errors but not used
  
  if dict_args['plot_live_loss']:
    helper.PLOT_LIVE_LOSS = True
  
  if dict_args['log_xattn']:
    helper.LOG_CROSS_ATTENTION['enable']= True 
    helper.LOG_HISTORY_SAMPLE = True  # is True to get the prediction accoridng to the cross attention, so it's possible to check if the prediction makes sense
  
  if dict_args['log_history_sample']:
    helper.LOG_HISTORY_SAMPLE = True
  
  if dict_args['save_last_epoch_model']:
    helper.SAVE_LAST_EPOCH_MODEL = True
  
  if dict_args['train_amp_enabled']:
    helper.AMP_ENABLED = True
    helper.AMP_DTYPE = dict_args['train_amp_dtype'].lower()
  
  for method in dict_args['queries_agg_method']:
    if method not in helper.QUERIES_AGG_METHOD:
      raise ValueError(f"Invalid queries aggregation method: {method}. Must be one of {[m for m in helper.QUERIES_AGG_METHOD]}")
  
  if not None in dict_args['delta_huber'] and 'huber' not in dict_args['loss']:
    raise ValueError("delta_huber is set but loss is not 'huber'. Set loss to 'huber' to use delta_huber.")
  
  if None in dict_args['delta_huber'] and 'huber' in dict_args['loss']:
    raise ValueError("delta_huber is not set but loss is 'huber'. Set delta_huber to use Huber loss.")
  
  if dict_args['train_amp_dtype'] is not None and dict_args['train_amp_enabled'] is False:
    raise ValueError("train_amp_dtype is set but train_amp_enabled is False. Set train_amp_enabled to use AMP training.")
  
  if dict_args['label_smooth'] and dict_args['soft_labels']:
    if sum(dict_args['label_smooth']) > 0 and sum(dict_args['soft_labels']) > 0:
      raise ValueError("Label smoothing and soft labels cannot be used together. Choose one.")
  
  if dict_args['loss'] in ['l1', 'l2'] and (sum(dict_args['label_smooth']) > 0 or sum(dict_args['soft_labels']) > 0):
    raise ValueError("Label smoothing and soft labels are not supported for 'l1' or 'l2' loss. Set them to 0.")
  
  if sum(dict_args['sim_loss_reduction']) != 0. and dict_args['loss'] != 'sim_loss':
    raise ValueError("Sim loss reduction is only supported for 'sim_loss'. Set it to 0 for other losses.")
  
  if ('ce' not in dict_args['loss']) and (sum(dict_args['label_smooth']) > 0 or sum(dict_args['soft_labels']) > 0):
    raise ValueError("Label smoothing and soft labels are only supported for 'ce' loss. Set them to 0 for other losses.")
  
  if dict_args['loss'] not in ['l1', 'l2', 'ce'] and (dict_args['train_amp_dtype'] == 'float16'):
    raise ValueError("AMP training with float16 is only supported for 'l1', 'l2', or 'ce' loss.")
  
  if dict_args['key_early_stopping'] not in ['val_accuracy']:
    raise ValueError(f"Invalid key for early stopping: {dict_args['key_early_stopping']}. Must be 'val_accuracy'.")
  
  if not dict_args['validation_enabled']:
    helper.SAVE_LAST_EPOCH_MODEL = True
    print("\nValidation is disabled. Last epoch model will be used for testing \n")
  
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
    **dict_args,
    'launch_command':" ".join(sys.argv), 
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
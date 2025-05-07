from custom.helper import CLIPS_REDUCTION, EMBEDDING_REDUCTION, MODEL_TYPE, SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import os
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
  
def get_loss(loss):
  """Get loss function from string name."""
  losses = {
    'l1': nn.L1Loss(),
    'l2': nn.MSELoss(),
    'ce': nn.CrossEntropyLoss()
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
# ------------ Training Functions ------------

def train_with_gru_head(
  model_type, emb_dim, pooling_clips_reduction, sample_frame_strategy,
  path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
  list_batch_train, list_init_network, list_GRU_hidden_size, list_GRU_num_layers,
  list_model_dropout, concatenate_temp_dim, GRU_output_size, list_regularization_lambda_L1,clip_grad_norm,
  lr_list, optim_list, loss_reg, k_fold, epochs, seed_random_state, is_plot_dataset_distribution,
  is_round_output_loss, is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,
  target_metric_best_model, early_stopping, enable_scheduler, clip_length, layer_norm, stop_after_kth_fold,n_workers,
  list_regularization_lambda_L2,dict_augmented
):
  """Run training with GRU head configuration."""
  stride_window_in_video = 16  # sliding window
  print(f'\nLearning rates: {lr_list}')
  print(f'Optimizers: {optim_list}')
  
  for batch_train in list_batch_train:
    for init_network in list_init_network:
      for GRU_hidden_size in list_GRU_hidden_size:
        for GRU_num_layers in list_GRU_num_layers:
          for GRU_dropout in list_model_dropout:
            for regularization_lambda_L1 in list_regularization_lambda_L1:
              for regularization_lambda_L2 in list_regularization_lambda_L2:
                # Configure GRU parameters
                input_size = emb_dim * 8 if concatenate_temp_dim else emb_dim
                params = {
                  'output_size': GRU_output_size,
                  'hidden_size': GRU_hidden_size,
                  'num_layers': GRU_num_layers,
                  'dropout': GRU_dropout,
                  'layer_norm': layer_norm,
                  'input_size': input_size
                }
                
                for lr in lr_list:
                  for optim_fn in optim_list:
                    # Set loss function based on output size
                    criterion = get_loss('CE') if GRU_output_size > 1 else get_loss(loss_reg)
                    start = time.time()
                    
                    # Run training
                    scripts.run_train_test(
                      model_type=model_type, 
                      pooling_embedding_reduction=pooling_clips_reduction,
                      pooling_clips_reduction=pooling_clips_reduction,
                      sample_frame_strategy=sample_frame_strategy, 
                      path_csv_dataset=path_csv_dataset, 
                      path_video_dataset=path_dataset,
                      head=HEAD.GRU,
                      stride_window_in_video=stride_window_in_video,
                      features_folder_saving_path=feature_folder_saving_path,
                      head_params=params,
                      k_fold=k_fold,
                      global_foder_name=global_foder_name, 
                      batch_size_training=batch_train, 
                      epochs=epochs, 
                      criterion=criterion, 
                      optimizer_fn=optim_fn,
                      lr=lr,
                      seed_random_state=seed_random_state,
                      is_plot_dataset_distribution=is_plot_dataset_distribution,
                      is_round_output_loss=is_round_output_loss,
                      is_shuffle_video_chunks=is_shuffle_video_chunks,
                      is_shuffle_training_batch=is_shuffle_training_batch,
                      init_network=init_network,
                      key_for_early_stopping=key_for_early_stopping,
                      regularization_lambda_L1=regularization_lambda_L1,
                      regularization_lambda_L2=regularization_lambda_L2,
                      clip_length=clip_length,
                      target_metric_best_model=target_metric_best_model,
                      early_stopping=early_stopping,
                      enable_scheduler=enable_scheduler,
                      concatenate_temp_dim=concatenate_temp_dim,
                      stop_after_kth_fold=stop_after_kth_fold,
                      n_workers=n_workers,
                      clip_grad_norm=clip_grad_norm,
                      dict_augmented=dict_augmented
                    )
                    print(f'Time taken for this run: {(time.time()-start)//60} min')

# def train_with_attentive_head(
#   model_type, pooling_clips_reduction, sample_frame_strategy, concatenate_temp_dim,
#   path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
#   list_batch_train, list_regularization_lambda_L1, lr_list, optim_list, k_fold, epochs,
#   seed_random_state, is_plot_dataset_distribution, is_round_output_loss, 
#   is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,
#   target_metric_best_model, early_stopping, enable_scheduler, clip_grad_norm,
#   clip_length, stop_after_kth_fold,emb_dim,list_num_heads,list_attn_dropout,n_workers,list_init_network,num_classes,pos_encoder,
#   list_label_smooth,list_regularization_lambda_L2,dict_augmented
# ):
#   """Run training with Attentive head configuration."""
#   stride_window_in_video = 16  # sliding window
#   print(f'\nLearning rates: {lr_list}')
#   print(f'Optimizers: {optim_list}')
#   for init_network in list_init_network:
#     for num_heads in list_num_heads:
#       for batch_train in list_batch_train:
#         for regularization_lambda_L1 in list_regularization_lambda_L1:
#           for regularization_lambda_L2 in list_regularization_lambda_L2:
#             for dropout in list_attn_dropout:
#               for label_smooth in list_label_smooth:
#                 # Configure Attentive head parameters
#                 params = {
#                   'input_dim': emb_dim*8 if concatenate_temp_dim else emb_dim,
#                   'num_classes': num_classes,
#                   'num_heads': num_heads,
#                   'dropout': dropout,
#                   'pos_enc': pos_encoder
#                 }
                
#                 for lr in lr_list:
#                   for optim_fn in optim_list:
#                     criterion = get_loss('CE')
#                     start = time.time()
                    
#                     # Run training
#                     scripts.run_train_test(
#                       model_type=model_type, 
#                       concatenate_temp_dim=concatenate_temp_dim,
#                       pooling_embedding_reduction=pooling_clips_reduction,
#                       pooling_clips_reduction=pooling_clips_reduction,
#                       sample_frame_strategy=sample_frame_strategy, 
#                       path_csv_dataset=path_csv_dataset, 
#                       path_video_dataset=path_dataset,
#                       head=HEAD.ATTENTIVE,
#                       stride_window_in_video=stride_window_in_video,
#                       features_folder_saving_path=feature_folder_saving_path,
#                       head_params=params,
#                       k_fold=k_fold,
#                       global_foder_name=global_foder_name, 
#                       batch_size_training=batch_train, 
#                       epochs=epochs, 
#                       criterion=criterion, 
#                       optimizer_fn=optim_fn,
#                       lr=lr,
#                       seed_random_state=seed_random_state,
#                       is_plot_dataset_distribution=is_plot_dataset_distribution,
#                       is_round_output_loss=is_round_output_loss,
#                       is_shuffle_video_chunks=is_shuffle_video_chunks,
#                       is_shuffle_training_batch=is_shuffle_training_batch,
#                       init_network=init_network,
#                       key_for_early_stopping=key_for_early_stopping,
#                       regularization_lambda_L1=regularization_lambda_L1,
#                       regularization_lambda_L2=regularization_lambda_L2,
#                       clip_length=clip_length,
#                       target_metric_best_model=target_metric_best_model,
#                       early_stopping=early_stopping,
#                       enable_scheduler=enable_scheduler,
#                       stop_after_kth_fold=stop_after_kth_fold,
#                       label_smooth=label_smooth,
#                       n_workers=n_workers,
#                       clip_grad_norm=clip_grad_norm,
#                       dict_augmented=dict_augmented
#                     )
#                     # scrit example: python3 train_model.py --mt B --head ATTENTIVE --lr 0.0001 --ep 500 --opt adamw --batch_train 8700  --stop 3 --num_heads 8 --csv partA/starting_point/samples_exc_no_detection.csv --ffsp partA/video/video_frontalized --global_folder_name history_run_att --path_video_dataset partA/video/video_frontalized  --k_fold 3 
#                     print(f'Time taken for this run: {(time.time()-start)//60} min')

def train_with_jepa_attentive_head(
  model_type, pooling_clips_reduction, sample_frame_strategy, concatenate_temp_dim,
  path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
  list_batch_train, list_regularization_lambda_L1, lr_list, optim_list, k_fold, epochs,
  seed_random_state, is_plot_dataset_distribution, is_round_output_loss, 
  is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,list_regularization_lambda_L2,
  target_metric_best_model, early_stopping, enable_scheduler, clip_grad_norm,
  clip_length, stop_after_kth_fold,emb_dim,list_num_heads,list_model_dropout,n_workers,head_type,list_init_network,
  list_drop_attn, list_drop_residual, list_mlp_ratio, pos_encoder, num_classes, list_label_smooth,dict_augmented,nr_blocks,
  cross_block_after_transformers,list_num_cross_heads
):
  """Run training with Attentive head configuration."""
  stride_window_in_video = 16  # sliding window
  print(f'\nLearning rates: {lr_list}')
  print(f'Optimizers: {optim_list}')
  
  for init_network in list_init_network:
    for num_heads in list_num_heads:
      for num_cross_heads in list_num_cross_heads: 
        for batch_train in list_batch_train:
          for regularization_lambda_L1 in list_regularization_lambda_L1:
            for regularization_lambda_L2 in list_regularization_lambda_L2:
              for dropout in list_model_dropout:
                for drop_attn in list_drop_attn:
                  for drop_residual in list_drop_residual:
                    for mlp_ratio in list_mlp_ratio:
                      for label_smooth in list_label_smooth:
                        for depth in nr_blocks: 
                      # Configure Attentive head parameters
                          params = {
                            'input_dim': emb_dim*8 if concatenate_temp_dim else emb_dim,
                            'num_classes': num_classes,
                            'num_cross_heads': num_cross_heads,
                            'num_heads': num_heads if num_heads is not None else num_cross_heads,
                            'dropout': dropout,
                            'attn_dropout': drop_attn,
                            'residual_dropout': drop_residual,
                            'mlp_ratio': mlp_ratio,
                            'pos_enc': pos_encoder,
                            'depth': depth,
                            'cross_block_after_transformers': cross_block_after_transformers
                          }
                          
                          for lr in lr_list:
                            for optim_fn in optim_list:
                              criterion = get_loss('CE')
                              start = time.time()
                              
                              # Run training
                              # with cProfile.Profile() as pr:
                              run_folder_path = scripts.run_train_test(
                                model_type=model_type, 
                                concatenate_temp_dim=concatenate_temp_dim,
                                pooling_embedding_reduction=pooling_clips_reduction,
                                pooling_clips_reduction=pooling_clips_reduction,
                                sample_frame_strategy=sample_frame_strategy, 
                                path_csv_dataset=path_csv_dataset, 
                                path_video_dataset=path_dataset,
                                head=HEAD.ATTENTIVE_JEPA,
                                stride_window_in_video=stride_window_in_video,
                                features_folder_saving_path=feature_folder_saving_path,
                                head_params=params,
                                k_fold=k_fold,
                                global_foder_name=global_foder_name, 
                                batch_size_training=batch_train, 
                                epochs=epochs, 
                                criterion=criterion, 
                                optimizer_fn=optim_fn,
                                lr=lr,
                                seed_random_state=seed_random_state,
                                is_plot_dataset_distribution=is_plot_dataset_distribution,
                                is_round_output_loss=is_round_output_loss,
                                is_shuffle_video_chunks=is_shuffle_video_chunks,
                                is_shuffle_training_batch=is_shuffle_training_batch,
                                init_network=init_network,
                                key_for_early_stopping=key_for_early_stopping,
                                regularization_lambda_L1=regularization_lambda_L1,
                                regularization_lambda_L2=regularization_lambda_L2,
                                clip_length=clip_length,
                                target_metric_best_model=target_metric_best_model,
                                early_stopping=early_stopping,
                                enable_scheduler=enable_scheduler,
                                stop_after_kth_fold=stop_after_kth_fold,
                                n_workers=n_workers,
                                clip_grad_norm=clip_grad_norm,
                                label_smooth=label_smooth,
                                dict_augmented=dict_augmented
                              )
                                # scrit example: python3 train_model.py --mt B --head ATTENTIVE --lr 0.0001 --ep 500 --opt adamw --batch_train 8700  --stop 3 --num_heads 8 --csv partA/starting_point/samples_exc_no_detection.csv --ffsp partA/video/video_frontalized --global_folder_name history_run_att --path_video_dataset partA/video/video_frontalized  --k_fold 3 
                              print(f'Time taken for this run: {(time.time()-start)//60} min')
                                # pr.print_stats(sort=SortKey.CUMULATIVE)
                                # Save the profiling results to a file
                                # save_stats(pr, os.path.join(run_folder_path, 'profiling_results.txt'))
                              
# def train_with_linear_head(
#   model_type, pooling_clips_reduction, sample_frame_strategy, concatenate_temp_dim,
#   path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
#   list_batch_train, list_regularization_lambda_L1, lr_list, optim_list, k_fold, epochs,
#   seed_random_state, is_plot_dataset_distribution, is_round_output_loss, 
#   is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,
#   target_metric_best_model, early_stopping, enable_scheduler, 
#   clip_length, stop_after_kth_fold,emb_dim,dim_reduction,n_workers,list_init_network,clip_grad_norm, num_classes
# ):
#   """Run training with Linear head configuration."""
#   stride_window_in_video = 16
#   print(f'\nLearning rates: {lr_list}')
#   print(f'Optimizers: {optim_list}')
#   dim_reduction = get_embedding_reduction(dim_reduction)
#   for init_network in list_init_network:
#     for batch_train in list_batch_train:
#       for regularization_lambda in list_regularization_lambda_L1:
#         # Configure feature shape and dimension reduction
#         feature_shape = [1, 8, 14, 14, emb_dim]  # 8 temporal dimension, 14x14 spatial dimension
#         if dim_reduction.value:
#           for dim in dim_reduction.value:
#             feature_shape[dim] = 1
                  
#         params = {
#           'dim_reduction': dim_reduction.value,
#           'input_dim': math.prod(feature_shape),
#           'num_classes': num_classes,
#         }
        
#         for lr in lr_list:
#           for optim_fn in optim_list:
#             criterion = get_loss('CE')
#             start = time.time()
            
#             # Run training
#             scripts.run_train_test(
#               model_type=model_type,
#               concatenate_temp_dim=concatenate_temp_dim,
#               pooling_embedding_reduction=pooling_clips_reduction,
#               pooling_clips_reduction=pooling_clips_reduction,
#               sample_frame_strategy=sample_frame_strategy,
#               path_csv_dataset=path_csv_dataset,
#               path_video_dataset=path_dataset,
#               head=HEAD.LINEAR,
#               stride_window_in_video=stride_window_in_video,
#               features_folder_saving_path=feature_folder_saving_path,
#               head_params=params,
#               k_fold=k_fold,
#               global_foder_name=global_foder_name,
#               batch_size_training=batch_train,
#               epochs=epochs,
#               criterion=criterion,
#               optimizer_fn=optim_fn,
#               lr=lr,
#               seed_random_state=seed_random_state,
#               is_plot_dataset_distribution=is_plot_dataset_distribution,
#               is_round_output_loss=is_round_output_loss,
#               is_shuffle_video_chunks=is_shuffle_video_chunks,
#               is_shuffle_training_batch=is_shuffle_training_batch,
#               init_network=init_network,
#               key_for_early_stopping=key_for_early_stopping,
#               regularization_lambda_L1=regularization_lambda,
#               clip_length=clip_length,
#               target_metric_best_model=target_metric_best_model,
#               early_stopping=early_stopping,
#               enable_scheduler=enable_scheduler,
#               stop_after_kth_fold=stop_after_kth_fold,
#               n_workers=n_workers,
#               clip_grad_norm=clip_grad_norm
#             )
#             print(f'Time taken for this run: {(time.time()-start)//60} min')

def save_stats(pr, file_path):
  with open(file_path, 'w') as f:
    ps = pstats.Stats(pr, stream=f)
    ps.get_stats_profile()
    ps.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
  data = []
  for func, stat in ps.stats.items():
    file_path, lineno, func_name = func
    primitive_calls, total_calls, total_time, cumulative_time, _ = stat
    data.append({
      'Filename': file_path,
      'Line No': lineno,
      'Function': func_name,
      'Primitive Calls': primitive_calls,
      'Total Calls': total_calls,
      'Total Time': total_time,
      'Cumulative Time': cumulative_time
    })
  # Create a pandas DataFrame and write it to a CSV file
  df = pd.DataFrame(data)
  df.to_csv(os.path.join(os.path.dirname(file_path),'profiling_results.csv'), index=False)
  
def train(
  model_type, epochs, lr, path_csv_dataset, feature_folder_saving_path, global_foder_name, path_dataset,
  k_fold, opt_list, list_batch_train, list_GRU_hidden_size, list_GRU_num_layers, list_model_dropout,
  concatenate_temp_dim, list_init_network, early_stopping, list_regularization_lambda_L1,
  is_round_output_loss, GRU_output_size, key_for_early_stopping, seed_random_state, is_shuffle_training_batch,
  is_shuffle_video_chunks, clip_length, target_metric_best_model, is_plot_dataset_distribution, layer_norm,
  enable_scheduler, loss_reg, head, list_stop_fold,list_num_heads,linear_dim_reduction,n_workers,clip_grad_norm,
  list_drop_attn, list_drop_residual, list_mlp_ratio,pos_encoder,list_label_smooth,list_regularization_lambda_L2,dict_augmented,
  nr_blocks,cross_block_after_transformers,nr_cross_heads
):
  """Main training function that dispatches to specific head training functions."""
  # Initialize common parameters
  emb_dim = MODEL_TYPE.get_embedding_size(model_type)
  model_type = MODEL_TYPE.get_model_type(model_type)
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  head_type = get_head(head)
  
  # Convert string parameters to appropriate objects
  lr_list = lr if isinstance(lr, list) else [lr]
  optim_list = [get_optimizer(opt) for opt in opt_list]
  num_classes = pd.read_csv(path_csv_dataset,sep='\t')['class_id'].unique().shape[0]
  # Dispatch to appropriate head training function
  if head_type.name == 'GRU':
    if list_label_smooth[0] > 0:
      raise NotImplementedError('Label smoothing is not implemented for GRU head')
    train_with_gru_head(
      model_type=model_type, emb_dim=emb_dim, 
      pooling_clips_reduction=pooling_clips_reduction,
      sample_frame_strategy=sample_frame_strategy,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path,
      global_foder_name=global_foder_name, list_batch_train=list_batch_train,
      list_init_network=list_init_network, list_GRU_hidden_size=list_GRU_hidden_size,
      list_GRU_num_layers=list_GRU_num_layers, list_model_dropout=list_model_dropout,
      concatenate_temp_dim=concatenate_temp_dim, GRU_output_size=GRU_output_size,
      list_regularization_lambda_L1=list_regularization_lambda_L1, lr_list=lr_list,
      list_regularization_lambda_L2=list_regularization_lambda_L2,
      optim_list=optim_list, loss_reg=loss_reg, k_fold=k_fold,
      epochs=epochs, seed_random_state=seed_random_state,
      is_plot_dataset_distribution=is_plot_dataset_distribution,
      is_round_output_loss=is_round_output_loss,
      is_shuffle_video_chunks=is_shuffle_video_chunks,
      is_shuffle_training_batch=is_shuffle_training_batch,
      key_for_early_stopping=key_for_early_stopping,
      target_metric_best_model=target_metric_best_model,
      early_stopping=early_stopping, enable_scheduler=enable_scheduler,
      clip_length=clip_length, layer_norm=layer_norm,
      stop_after_kth_fold=list_stop_fold,
      n_workers=n_workers,
      clip_grad_norm=clip_grad_norm,
      dict_augmented=dict_augmented
    )
  elif head_type.name == 'ATTENTIVE':
    train_with_attentive_head(
      model_type=model_type, pooling_clips_reduction=pooling_clips_reduction,emb_dim=emb_dim,
      sample_frame_strategy=sample_frame_strategy, concatenate_temp_dim=concatenate_temp_dim,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path, global_foder_name=global_foder_name,
      list_batch_train=list_batch_train, list_regularization_lambda_L1=list_regularization_lambda_L1,
      list_regularization_lambda_L2=list_regularization_lambda_L2,
      lr_list=lr_list, optim_list=optim_list, k_fold=k_fold, epochs=epochs,
      seed_random_state=seed_random_state, is_plot_dataset_distribution=is_plot_dataset_distribution,
      is_round_output_loss=is_round_output_loss, is_shuffle_video_chunks=is_shuffle_video_chunks,
      is_shuffle_training_batch=is_shuffle_training_batch, key_for_early_stopping=key_for_early_stopping,
      target_metric_best_model=target_metric_best_model,
      early_stopping=early_stopping, enable_scheduler=enable_scheduler, clip_length=clip_length,
      stop_after_kth_fold=list_stop_fold,list_num_heads=list_num_heads,list_attn_dropout=list_drop_attn,
      n_workers=n_workers, list_init_network=list_init_network, clip_grad_norm=clip_grad_norm,num_classes=num_classes,
      pos_encoder=pos_encoder,list_label_smooth=list_label_smooth,dict_augmented=dict_augmented
      
    )
  elif head_type.name == 'ATTENTIVE_JEPA':
    train_with_jepa_attentive_head(
      model_type=model_type, pooling_clips_reduction=pooling_clips_reduction,emb_dim=emb_dim,
      sample_frame_strategy=sample_frame_strategy, concatenate_temp_dim=concatenate_temp_dim,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path, global_foder_name=global_foder_name,
      list_batch_train=list_batch_train, list_regularization_lambda_L1=list_regularization_lambda_L1,
      list_regularization_lambda_L2=list_regularization_lambda_L2,
      lr_list=lr_list, optim_list=optim_list, k_fold=k_fold, epochs=epochs,
      seed_random_state=seed_random_state, is_plot_dataset_distribution=is_plot_dataset_distribution,
      is_round_output_loss=is_round_output_loss, is_shuffle_video_chunks=is_shuffle_video_chunks,
      is_shuffle_training_batch=is_shuffle_training_batch, key_for_early_stopping=key_for_early_stopping,
      target_metric_best_model=target_metric_best_model,
      early_stopping=early_stopping, enable_scheduler=enable_scheduler, clip_length=clip_length,
      stop_after_kth_fold=list_stop_fold,list_num_heads=list_num_heads,list_model_dropout=list_model_dropout,
      n_workers=n_workers,head_type=head_type, list_init_network=list_init_network, clip_grad_norm=clip_grad_norm,
      list_drop_attn=list_drop_attn, list_drop_residual=list_drop_residual, list_mlp_ratio=list_mlp_ratio,
      pos_encoder=pos_encoder,num_classes=num_classes, list_label_smooth=list_label_smooth,
      dict_augmented=dict_augmented,nr_blocks=nr_blocks, cross_block_after_transformers=cross_block_after_transformers,
      list_num_cross_heads=nr_cross_heads
    )
  elif head_type.name == 'LINEAR':
    # Label smoothing and augmentation not implemeted
    train_with_linear_head(
      model_type=model_type, pooling_clips_reduction=pooling_clips_reduction,emb_dim=emb_dim,
      sample_frame_strategy=sample_frame_strategy, concatenate_temp_dim=concatenate_temp_dim,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path, global_foder_name=global_foder_name,
      list_batch_train=list_batch_train, list_regularization_lambda_L1=list_regularization_lambda_L1,
      lr_list=lr_list, optim_list=optim_list, k_fold=k_fold, epochs=epochs,
      seed_random_state=seed_random_state, is_plot_dataset_distribution=is_plot_dataset_distribution,
      is_round_output_loss=is_round_output_loss, is_shuffle_video_chunks=is_shuffle_video_chunks,
      is_shuffle_training_batch=is_shuffle_training_batch, key_for_early_stopping=key_for_early_stopping,
      target_metric_best_model=target_metric_best_model,
      early_stopping=early_stopping, enable_scheduler=enable_scheduler, clip_length=clip_length,
      stop_after_kth_fold=list_stop_fold,dim_reduction = linear_dim_reduction, n_workers=n_workers,
      list_init_network = list_init_network, clip_grad_norm=clip_grad_norm,num_classes=num_classes 
    )
  
  print('Training completed')
  print(f'Check the path {os.path.join(global_foder_name,"summary_log.csv")} file for the results of the training')

# ------------ Main Entry Point ------------
training_params = ['lr', 'ep', 'opt','batch_train','concatenate_temp_dim','init_network','regulariz_lambda_L1',
                   'regulariz_lambda_L2','label_smooth']

gru_params = ['GRU_hidden_size', 'GRU_num_layers', 'GRU_output_size', 'layer_norm']

jepa_attentive_params = ['num_heads', 'num_cross_head', 'model_dropout', 'drop_attn', 'drop_residual', 'mlp_ratio', 'pos_enc', 
                         'nr_blocks','cross_block_after_transformers']


def _suggest(trial: optuna.trial.Trial, name, values, categorical):
  """Suggest a hyperparameter value based on the type of values provided."""
  if categorical:
    return trial.suggest_categorical(name, values)

  low, high = values[0], values[-1]
  if low > high:
    low, high = high, low
  if isinstance(low, int) and isinstance(high, int):
    return trial.suggest_int(name, low=low, high=high)
  else:
    return trial.suggest_float(name, low=low, high=high)

def objective(trial: optuna.trial.Trial, original_kwargs):
  """Objective function for Optuna hyperparameter optimization."""
  kwargs = deepcopy(original_kwargs)
  model_type = MODEL_TYPE.get_model_type(kwargs['mt'])
  epochs = kwargs['ep']

  # common training params
  nr_blocks = _suggest(trial, 'nr_blocks', kwargs['nr_blocks'], kwargs['optuna_categorical'])
  lr = _suggest(trial, 'lr', kwargs['lr'], kwargs['optuna_categorical'])
  opt = trial.suggest_categorical('opt', kwargs['opt'])
  optimizer_fn = get_optimizer(opt)
  init_network = trial.suggest_categorical('init_network', kwargs['init_network'])
  batch_train = _suggest(trial, 'batch_train', kwargs['batch_train'], kwargs['optuna_categorical'])
  regulariz_lambda_L1 = _suggest(trial, 'regulariz_lambda_L1', kwargs['regulariz_lambda_L1'], kwargs['optuna_categorical'])
  regulariz_lambda_L2 = _suggest(trial, 'regulariz_lambda_L2', kwargs['regulariz_lambda_L2'], kwargs['optuna_categorical'])
  label_smooth = _suggest(trial, 'label_smooth', kwargs['label_smooth'], kwargs['optuna_categorical'])
  concatenate_temp_dim = trial.suggest_categorical('concatenate_temp_dim', kwargs['concatenate_temp_dim'])

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
    num_heads = _suggest(trial, 'num_heads', kwargs['num_heads'], kwargs['optuna_categorical'])
    num_cross_head = _suggest(trial, 'num_cross_head', kwargs['num_cross_head'], kwargs['optuna_categorical'])
    model_dropout = _suggest(trial, 'model_dropout', kwargs['model_dropout'], kwargs['optuna_categorical'])
    drop_attn = _suggest(trial, 'drop_attn', kwargs['drop_attn'], kwargs['optuna_categorical'])
    drop_residual = _suggest(trial, 'drop_residual', kwargs['drop_residual'], kwargs['optuna_categorical'])
    mlp_ratio = _suggest(trial, 'mlp_ratio', kwargs['mlp_ratio'], kwargs['optuna_categorical'])
    pos_enc = trial.suggest_categorical('pos_enc', kwargs['pos_enc'])
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
      'cross_block_after_transformers': cross_block_after_transformers
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

  # run training
  with cProfile.Profile() as pr:
    run_folder_path, results = scripts.run_train_test(
      model_type=model_type,
      criterion=get_loss('ce') if num_classes > 1 else get_loss('l1'),
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
      trial=trial
    )
    save_stats(pr, os.path.join(run_folder_path, 'profiling_results.txt'))
    pr.dump_stats(os.path.join(run_folder_path, 'profiling_results.prof'))

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

def _suggest(trial:optuna.trial.Trial,name,values,categorical):
  """Suggest a hyperparameter value based on the type of values provided."""
  if categorical:
    return trial.suggest_categorical(name, values)

  low, high = values[0], values[-1]
  if low > high:
    low, high = high, low
  # if both endpoints are ints, do integer
  if isinstance(low, int) and isinstance(high, int):
    return trial.suggest_int(name, low=low, high=high)
  else:
    return trial.suggest_float(name, low=low, high=high)

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
    list_val_accuracy.append(dict_log_k_fold['train_val']['list_val_macro_accuracy'][best_epoch])
  return np.mean(list_val_accuracy)


def hyper_search(kwargs):
  """Hyperparameter search for AttentiveJepa head using Optuna."""
  # Define the study
  study = optuna.create_study(direction='maximize',
                              storage=f'sqlite:///{os.path.join(kwargs["global_folder_name"],"optuna_study.db")}',
                              study_name=f'{kwargs["head"]}_{os.path.split(kwargs["global_folder_name"])[-1].split("_")[-1]}',
                              pruner=optuna.pruners.ThresholdPruner(lower=dict_args['pruner_threshold_lower'],
                                                                    n_warmup_steps=dict_args['pruner_n_warmup_steps'],
                                                                    interval_steps=2))
  
  # Optimize the objective function
  study.optimize(lambda trial: objective(trial, kwargs), n_trials=kwargs['n_trials'], timeout=kwargs['timeout'])
  optuna_path = os.path.join(kwargs['global_folder_name'],'optuna_study.pkl')
  with open(optuna_path,"wb") as f:
    pickle.dump(study, f)
    print(f'Study saved to {optuna_path}')
  # Print the best hyperparameters and their corresponding accuracy
  print('Best hyperparameters:')
  print(study.best_params)
  print('Best accuracy:', study.best_value)
  
  
if __name__ == '__main__':
  # mp.set_start_method('spawn', force=True)
  # Set up argument parser
  parser = argparse.ArgumentParser(description='Train video analysis model with various configurations')
  
  parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for data loading. Default is 1')
  
  # Model configuration
  parser.add_argument('--mt', type=str, default='B', help='Model type: B (Base), S (Small), or I (Image)')
  parser.add_argument('--head', type=str, default='ATTENTIVE_JEPA', help='Head type: GRU, ATTENTIVE, LINEAR, ATTENTIVE_JEPA')
  
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
                    help='Concatenate temporal dimension in input to the model. So the embeddind is [temporal*emb_dim]=6144 if model base')
  
  
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
  parser.add_argument('--n_trials', type=int, default=100, help='Number of trials for Optuna hyperparameter optimization. Default is 100')
  parser.add_argument('--timeout', type=int, default=14, help='Timeout for Optuna hyperparameter optimization in hours. Default is 14 hours')
  
  # Optuna parameters
  parser.add_argument('--pruner_threshold_lower', type=float, default=0.15, help='Threshold for Optuna pruner. Default is 0.15')
  parser.add_argument('--pruner_n_warmup_steps', type=int, default=30, help='Number of warmup steps for Optuna pruner. Default is 30')
  parser.add_argument('--optuna_categorical',type=int, default=1, help='Use categorical optimization for Optuna. Default is 1 (True). Otherwise, use continuous optimization in range [list[0],list[-1]]')
  
  # Parse arguments
  args = parser.parse_args()
  args.timeout *= 3600 # Convert hours to seconds
  dict_args = vars(args)
  
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
  
  # Start training
  print(args)
  # if args.head.upper() == 'ATTENTIVE_JEPA':
  hyper_search(dict_args)
  # elif args.head.upper() == 'GRU':
  #   hyper_search_gru(dict_args)
  # else:
  #   raise ValueError(f"Hyperparameter search not supported for head '{args.head}'")
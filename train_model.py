from custom.helper import CLIPS_REDUCTION, EMBEDDING_REDUCTION, MODEL_TYPE, SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import os
import time
import math
import argparse
import torch.nn as nn
import torch.optim as optim
from custom.head import earlyStoppingAccuracy, earlyStoppingLoss
import custom.scripts as scripts

# ------------ Helper Functions ------------

def get_model_type(model_type):
  """Convert model type string to MODEL_TYPE enum."""
  model_types = {
    'B': MODEL_TYPE.VIDEOMAE_v2_B,
    'S': MODEL_TYPE.VIDEOMAE_v2_S,
    'I': MODEL_TYPE.ViT_image
  }
  if model_type not in model_types:
    raise ValueError(f'Model type not found: {model_type}. Valid options: {list(model_types.keys())}')
  return model_types[model_type]
  
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
    'linear': HEAD.LINEAR
  }
  head_name = head_name.lower()
  if head_name not in heads:
    raise ValueError(f'Head not found: {head_name}. Valid options: {list(heads.keys())}')
  return heads[head_name]

def generate_path(path):
  """Generate full path by joining with NAS_PATH."""
  return os.path.join(GLOBAL_PATH.NAS_PATH, path)

def get_embedding_reduction(reduction):
  """Get EMBEDDING_REDUCTION enum from string name."""
  reductions = {
    'temporal': EMBEDDING_REDUCTION.MEAN_TEMPORAL,
    'spatial': EMBEDDING_REDUCTION.MEAN_SPATIAL,
    'temporal_spatial': EMBEDDING_REDUCTION.MEAN_TEMPORAL_SPATIAL
  }
  reduction = reduction.lower()
  if reduction not in reductions:
    raise ValueError(f'Embedding reduction not found: {reduction}. Valid options: {list(reductions.keys())}')
  return reductions[reduction]

# ------------ Training Functions ------------

def train_with_gru_head(
  model_type, emb_dim, pooling_clips_reduction, sample_frame_strategy,
  path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
  list_batch_train, list_init_network, list_GRU_hidden_size, list_GRU_num_layers,regularization_loss,
  lsit_GRU_dropout, GRU_concatenate_temp_dim, GRU_output_size, list_regularization_lambda,
  lr_list, optim_list, loss_reg, k_fold, epochs, seed_random_state, is_plot_dataset_distribution,
  is_round_output_loss, is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,
  target_metric_best_model, early_stopping, enable_scheduler, clip_length, layer_norm, stop_after_kth_fold
):
  """Run training with GRU head configuration."""
  stride_window_in_video = 16  # sliding window
  print(f'\nLearning rates: {lr_list}')
  print(f'Optimizers: {optim_list}')
  
  for batch_train in list_batch_train:
    for init_network in list_init_network:
      for GRU_hidden_size in list_GRU_hidden_size:
        for GRU_num_layers in list_GRU_num_layers:
          for GRU_dropout in lsit_GRU_dropout:
            for regularization_lambda in list_regularization_lambda:
              # Configure GRU parameters
              input_size = emb_dim * 8 if GRU_concatenate_temp_dim else emb_dim
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
                    regularization_lambda=regularization_lambda,
                    regularization_loss=regularization_loss,
                    clip_length=clip_length,
                    target_metric_best_model=target_metric_best_model,
                    early_stopping=early_stopping,
                    enable_scheduler=enable_scheduler,
                    concatenate_temp_dim=GRU_concatenate_temp_dim,
                    stop_after_kth_fold=stop_after_kth_fold
                  )
                  print(f'Time taken for this run: {(time.time()-start)//60} min')

def train_with_attentive_head(
  model_type, pooling_clips_reduction, sample_frame_strategy, GRU_concatenate_temp_dim,
  path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
  list_batch_train, list_regularization_lambda, lr_list, optim_list, k_fold, epochs,
  seed_random_state, is_plot_dataset_distribution, is_round_output_loss, 
  is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,
  regularization_loss, target_metric_best_model, early_stopping, enable_scheduler, 
  clip_length, stop_after_kth_fold,emb_dim,list_num_heads
):
  """Run training with Attentive head configuration."""
  stride_window_in_video = 16  # sliding window
  print(f'\nLearning rates: {lr_list}')
  print(f'Optimizers: {optim_list}')
  for num_heads in list_num_heads:
    for batch_train in list_batch_train:
      for regularization_lambda in list_regularization_lambda:
        # Configure Attentive head parameters
        params = {
          'input_dim': emb_dim*8 if GRU_concatenate_temp_dim else emb_dim,
          'num_classes': 5,
          'num_heads': num_heads,
        }
        
        for lr in lr_list:
          for optim_fn in optim_list:
            criterion = get_loss('CE')
            start = time.time()
            
            # Run training
            scripts.run_train_test(
              model_type=model_type, 
              concatenate_temp_dim=GRU_concatenate_temp_dim,
              pooling_embedding_reduction=pooling_clips_reduction,
              pooling_clips_reduction=pooling_clips_reduction,
              sample_frame_strategy=sample_frame_strategy, 
              path_csv_dataset=path_csv_dataset, 
              path_video_dataset=path_dataset,
              head=HEAD.ATTENTIVE,
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
              init_network=None,
              key_for_early_stopping=key_for_early_stopping,
              regularization_lambda=regularization_lambda,
              regularization_loss=regularization_loss,
              clip_length=clip_length,
              target_metric_best_model=target_metric_best_model,
              early_stopping=early_stopping,
              enable_scheduler=enable_scheduler,
              stop_after_kth_fold=stop_after_kth_fold
            )
            # scrit example: python3 train_model.py --mt B --head ATTENTIVE --lr 0.0001 --ep 500 --opt adamw --batch_train 8700  --stop 3 --num_heads 8 --csv partA/starting_point/samples_exc_no_detection.csv --ffsp partA/video/video_frontalized --global_folder_name history_run_att --path_video_dataset partA/video/video_frontalized  --k_fold 3 
            print(f'Time taken for this run: {(time.time()-start)//60} min')

def train_with_linear_head(
  model_type, pooling_clips_reduction, sample_frame_strategy, GRU_concatenate_temp_dim,
  path_csv_dataset, path_dataset, feature_folder_saving_path, global_foder_name,
  list_batch_train, list_regularization_lambda, lr_list, optim_list, k_fold, epochs,
  seed_random_state, is_plot_dataset_distribution, is_round_output_loss, 
  is_shuffle_video_chunks, is_shuffle_training_batch, key_for_early_stopping,
  regularization_loss, target_metric_best_model, early_stopping, enable_scheduler, 
  clip_length, stop_after_kth_fold,emb_dim,dim_reduction
):
  """Run training with Linear head configuration."""
  stride_window_in_video = 16
  print(f'\nLearning rates: {lr_list}')
  print(f'Optimizers: {optim_list}')
  dim_reduction = get_embedding_reduction(dim_reduction)
  for batch_train in list_batch_train:
    for regularization_lambda in list_regularization_lambda:
      # Configure feature shape and dimension reduction
      feature_shape = [1, 8, 14, 14, emb_dim]  # 8 temporal dimension, 14x14 spatial dimension, 768 feature dimension
      if dim_reduction.value:
        for dim in dim_reduction.value:
          feature_shape[dim] = 1
                
      params = {
        'dim_reduction': dim_reduction.value,
        'input_dim': math.prod(feature_shape),
        'num_classes': 5,
      }
      
      for lr in lr_list:
        for optim_fn in optim_list:
          criterion = get_loss('CE')
          start = time.time()
          
          # Run training
          scripts.run_train_test(
            model_type=model_type,
            concatenate_temp_dim=GRU_concatenate_temp_dim,
            pooling_embedding_reduction=pooling_clips_reduction,
            pooling_clips_reduction=pooling_clips_reduction,
            sample_frame_strategy=sample_frame_strategy,
            path_csv_dataset=path_csv_dataset,
            path_video_dataset=path_dataset,
            head=HEAD.LINEAR,
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
            init_network=None,
            key_for_early_stopping=key_for_early_stopping,
            regularization_lambda=regularization_lambda,
            regularization_loss=regularization_loss,
            clip_length=clip_length,
            target_metric_best_model=target_metric_best_model,
            early_stopping=early_stopping,
            enable_scheduler=enable_scheduler,
            stop_after_kth_fold=stop_after_kth_fold
          )
          print(f'Time taken for this run: {(time.time()-start)//60} min')

def train(
  model_type, epochs, lr, path_csv_dataset, feature_folder_saving_path, global_foder_name, path_dataset,
  k_fold, opt_list, list_batch_train, list_GRU_hidden_size, list_GRU_num_layers, lsit_GRU_dropout,
  GRU_concatenate_temp_dim, list_init_network, early_stopping, regularization_loss, list_regularization_lambda,
  is_round_output_loss, GRU_output_size, key_for_early_stopping, seed_random_state, is_shuffle_training_batch,
  is_shuffle_video_chunks, clip_length, target_metric_best_model, is_plot_dataset_distribution, layer_norm,
  enable_scheduler, loss_reg, head, stop_after_kth_fold,list_num_heads,linear_dim_reduction
):
  """Main training function that dispatches to specific head training functions."""
  # Initialize common parameters
  emb_dim = 384 if 'S' in model_type else 768
  model_type = get_model_type(model_type)
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  head_type = get_head(head)
  
  # Convert string parameters to appropriate objects
  lr_list = lr if isinstance(lr, list) else [lr]
  optim_list = [get_optimizer(opt) for opt in opt_list]
  
  # Dispatch to appropriate head training function
  if head_type.name == 'GRU':
    train_with_gru_head(
      model_type=model_type, emb_dim=emb_dim, 
      pooling_clips_reduction=pooling_clips_reduction,
      sample_frame_strategy=sample_frame_strategy,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path,
      global_foder_name=global_foder_name, list_batch_train=list_batch_train,
      list_init_network=list_init_network, list_GRU_hidden_size=list_GRU_hidden_size,
      list_GRU_num_layers=list_GRU_num_layers, lsit_GRU_dropout=lsit_GRU_dropout,
      GRU_concatenate_temp_dim=GRU_concatenate_temp_dim, GRU_output_size=GRU_output_size,
      list_regularization_lambda=list_regularization_lambda, lr_list=lr_list,
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
      stop_after_kth_fold=stop_after_kth_fold,
      regularization_loss=regularization_loss
    )
  elif head_type.name == 'ATTENTIVE':
    train_with_attentive_head(
      model_type=model_type, pooling_clips_reduction=pooling_clips_reduction,emb_dim=emb_dim,
      sample_frame_strategy=sample_frame_strategy, GRU_concatenate_temp_dim=GRU_concatenate_temp_dim,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path, global_foder_name=global_foder_name,
      list_batch_train=list_batch_train, list_regularization_lambda=list_regularization_lambda,
      lr_list=lr_list, optim_list=optim_list, k_fold=k_fold, epochs=epochs,
      seed_random_state=seed_random_state, is_plot_dataset_distribution=is_plot_dataset_distribution,
      is_round_output_loss=is_round_output_loss, is_shuffle_video_chunks=is_shuffle_video_chunks,
      is_shuffle_training_batch=is_shuffle_training_batch, key_for_early_stopping=key_for_early_stopping,
      regularization_loss=regularization_loss, target_metric_best_model=target_metric_best_model,
      early_stopping=early_stopping, enable_scheduler=enable_scheduler, clip_length=clip_length,
      stop_after_kth_fold=stop_after_kth_fold,list_num_heads=list_num_heads
    )
  elif head_type.name == 'LINEAR':
    train_with_linear_head(
      model_type=model_type, pooling_clips_reduction=pooling_clips_reduction,emb_dim=emb_dim,
      sample_frame_strategy=sample_frame_strategy, GRU_concatenate_temp_dim=GRU_concatenate_temp_dim,
      path_csv_dataset=path_csv_dataset, path_dataset=path_dataset,
      feature_folder_saving_path=feature_folder_saving_path, global_foder_name=global_foder_name,
      list_batch_train=list_batch_train, list_regularization_lambda=list_regularization_lambda,
      lr_list=lr_list, optim_list=optim_list, k_fold=k_fold, epochs=epochs,
      seed_random_state=seed_random_state, is_plot_dataset_distribution=is_plot_dataset_distribution,
      is_round_output_loss=is_round_output_loss, is_shuffle_video_chunks=is_shuffle_video_chunks,
      is_shuffle_training_batch=is_shuffle_training_batch, key_for_early_stopping=key_for_early_stopping,
      regularization_loss=regularization_loss, target_metric_best_model=target_metric_best_model,
      early_stopping=early_stopping, enable_scheduler=enable_scheduler, clip_length=clip_length,
      stop_after_kth_fold=stop_after_kth_fold,dim_reduction = linear_dim_reduction 
    )
  
  print('Training completed')
  print(f'Check the path {os.path.join(global_foder_name,"summary_log.csv")} file for the results of the training')

# ------------ Main Entry Point ------------

if __name__ == '__main__':
  # Set up argument parser
  parser = argparse.ArgumentParser(description='Train video analysis model with various configurations')
  
  # Model configuration
  parser.add_argument('--mt', type=str, default='B', help='Model type: B (Base), S (Small), or I (Image)')
  parser.add_argument('--head', type=str, default='GRU', help='Head type: GRU, ATTENTIVE, or LINEAR')
  
  # Path configuration
  parser.add_argument('--gp', action='store_true', help='Use global path prefix for file paths')
  parser.add_argument('--csv', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), 
                    help='Path to CSV dataset file')
  parser.add_argument('--ffsp', type=str, default=os.path.join('partA','video','features','samples_16_frontalized'), 
                    help='Feature folder saving path')
  parser.add_argument('--global_folder_name', type=str, default=f'history_run', 
                    help='Global folder name for saving results')
  parser.add_argument('--path_video_dataset', type=str, default=os.path.join('partA','video','video_frontalized'), 
                    help='Path to video dataset')
  
  # Training parameters
  parser.add_argument('--lr', type=float, nargs='*', default=0.0001, help='Learning rate(s)')
  parser.add_argument('--ep', type=int, default=500, help='Number of epochs')
  parser.add_argument('--k_fold', type=int, default=3, help='Number of k-fold cross validation splits')
  parser.add_argument('--opt', type=str, nargs='*', default='adam', help='Optimizer(s): adam, sgd, adamw')
  parser.add_argument('--batch_train', type=int, nargs='*', default=64, help='Training batch size(s)')
  parser.add_argument('--is_round_output_loss', action='store_true', 
                    help='Round output from regression before computing loss')
  parser.add_argument('--enable_scheduler', action='store_true', help='Enable learning rate scheduler')
  parser.add_argument('--stop', type=int, default=None, help='Stop after kth fold')
  
  # Attention parameters
  parser.add_argument('--num_heads', type=int, nargs='*',default=8, help='Number of heads for Attentive head')
  
  # Linear parameters
  parser.add_argument('--linear_dim_reduction', type=str, default='spatial', help=f'Dimension reduction for Linear head. Can be {[d.name.lower() for d in EMBEDDING_REDUCTION]}')
  
  # GRU parameters
  parser.add_argument('--GRU_hidden_size', type=int, nargs='*', default=1024, help='GRU hidden layer size(s)')
  parser.add_argument('--GRU_num_layers', type=int, nargs='*', default=2, help='GRU number of layers')
  parser.add_argument('--GRU_dropout', type=float, nargs='*', default=0.0, help='GRU dropout rate(s)') # Only in GRU
  parser.add_argument('--GRU_concatenate_temp_dim', action='store_true', 
                    help='Concatenate temporal dimension in GRU input')
  parser.add_argument('--GRU_output_size', type=int, default=1, 
                    help='Output size of GRU: 1 for regression, >1 for classification')
  parser.add_argument('--layer_norm', action='store_true', 
                    help='Add Layer normalization before final linear layer') # Only in GRU
  
  # Network initialization and regularization
  parser.add_argument('--init_network', type=str, nargs='*', default='default', 
                    help='Network initialization: xavier, default, uniform')
  parser.add_argument('--reg_loss', type=str, default='', help='Regularization type: L1 or L2')
  parser.add_argument('--reg_lambda', type=float, nargs='*', default=[0], help='Regularization strength(s)')
  parser.add_argument('--loss_regression', type=str, default='L1', help='Regression loss function: L1 or L2')
  
  # Early stopping parameters
  parser.add_argument('--key_early_stopping', type=str, default='val_macro_precision', 
                    help='Metric for early stopping: val_loss or val_macro_precision')
  parser.add_argument('--p_early_stop', type=int, default=2000, help='Patience for early stopping')
  parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum delta for early stopping')
  parser.add_argument('--threshold_mode', type=str, default='abs', help='Early stopping threshold mode: abs or rel')
  
  # Parse arguments
  args = parser.parse_args()
  
  # Generate timestamp for unique folder name
  timestamp = int(time.time())
  args.global_folder_name = f'{args.global_folder_name}_{timestamp}'
  
  # Apply global path prefixes if requested
  if args.gp:
    if args.csv[0] != '/':
      args.csv = generate_path(args.csv)
    if args.ffsp[0] != '/':
      args.ffsp = generate_path(args.ffsp)
    if args.path_video_dataset[0] != '/':
      args.path_video_dataset = generate_path(args.path_video_dataset)
    if args.global_folder_name[0] != '/':
      args.global_folder_name = generate_path(args.global_folder_name)
  
  # Print all arguments
  print('\n\nTraining configuration:')
  print(args)
  
  # Set up fixed parameters
  clip_length = 16
  seed_random_state = 42
  
  # Determine target metric for model selection
  target_metric_best_model = 'val_loss' if args.key_early_stopping == 'val_loss' else 'val_macro_precision'
  
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
  
  # Create config summary for later reference
  config_prompt = {
    'model_type': args.mt,
    'epochs': args.ep,
    'lr': args.lr,
    'path_csv_dataset': args.csv,
    'feature_folder_saving_path': args.ffsp,
    'global_foder_name': args.global_folder_name,
    'path_dataset': args.path_video_dataset,
    'k_fold': args.k_fold,
    'opt_list': args.opt,
    'batch_train': args.batch_train,
    'list_init_network': args.init_network,
    'list_GRU_hidden_size': args.GRU_hidden_size,
    'list_GRU_num_layers': args.GRU_num_layers,
    'lsit_GRU_dropout': args.GRU_dropout,
    'GRU_concatenate_temp_dim': args.GRU_concatenate_temp_dim,
    'list_regularization_lambda': args.reg_lambda,
    'regularization_loss': args.reg_loss,
    'GRU_output_size': args.GRU_output_size,
    'is_round_output_loss': args.is_round_output_loss,
    'key_for_early_stopping': args.key_early_stopping,
    'clip_length': clip_length,
    'target_metric_best_model': target_metric_best_model,
    'seed_random_state': seed_random_state,
    'early_stopping': early_stopping,
    'layer_norm': args.layer_norm,
    'p_early_stop': args.p_early_stop,
    'min_delta': args.min_delta,
    'threshold_mode': args.threshold_mode,
    'loss_regression': args.loss_regression,
    'enable_scheduler': args.enable_scheduler,
    'head': args.head,
    'stop_after_kth_fold': args.stop,
    'num_heads': args.num_heads,
    'linear_dim_reduction': args.linear_dim_reduction
  }
  
  # Create output directory and save configuration
  if not os.path.exists(args.global_folder_name):
    os.makedirs(args.global_folder_name)
    
  with open(os.path.join(args.global_folder_name, '_config_prompt.txt'), 'w') as f:
    for key, value in config_prompt.items():
      f.write(f'{key}: {value}\n')
  
  # Start training
  train(
    model_type=args.mt,
    epochs=args.ep,
    lr=args.lr,
    path_csv_dataset=args.csv,
    feature_folder_saving_path=args.ffsp,
    global_foder_name=args.global_folder_name,
    path_dataset=args.path_video_dataset,
    k_fold=args.k_fold,
    opt_list=args.opt,
    list_batch_train=args.batch_train,
    list_init_network=args.init_network,
    list_GRU_hidden_size=args.GRU_hidden_size,
    list_GRU_num_layers=args.GRU_num_layers,
    lsit_GRU_dropout=args.GRU_dropout,
    GRU_concatenate_temp_dim=args.GRU_concatenate_temp_dim,
    list_regularization_lambda=args.reg_lambda,
    regularization_loss=args.reg_loss,
    GRU_output_size=args.GRU_output_size,
    is_round_output_loss=args.is_round_output_loss,
    key_for_early_stopping=args.key_early_stopping,
    clip_length=clip_length,
    target_metric_best_model=target_metric_best_model,
    seed_random_state=seed_random_state,
    early_stopping=early_stopping,
    layer_norm=args.layer_norm,
    loss_reg=args.loss_regression,
    enable_scheduler=args.enable_scheduler,
    head=args.head,
    stop_after_kth_fold=args.stop,
    list_num_heads=args.num_heads,
    linear_dim_reduction = args.linear_dim_reduction,
    is_shuffle_video_chunks=False,
    is_shuffle_training_batch=True,
    is_plot_dataset_distribution=False
  )
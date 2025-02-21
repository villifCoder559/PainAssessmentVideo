from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY, HEAD,GLOBAL_PATH
import os
from custom.model import Model_Advanced
from transformers import AutoImageProcessor
from custom.head import HeadSVR, HeadGRU
import time
from custom.head import earlyStoppingAccuracy, earlyStoppingLoss
# import torch.optim.lr_scheduler as torch_scheduler
import torch.nn as nn
import torch.optim as optim
import custom.scripts as scripts
import argparse
from custom.helper import GLOBAL_PATH,CSV
import pandas as pd
# import wandb

def get_model_type(model_type):
  if model_type == 'B':
    return MODEL_TYPE.VIDEOMAE_v2_B
  elif model_type == 'S':
    return MODEL_TYPE.VIDEOMAE_v2_S
  elif model_type == 'I':
    return MODEL_TYPE.ViT_image
  else:
    raise ValueError('Model type not found')
  
def get_optimizer(opt):
  if opt.lower() == 'adam':
    return optim.Adam
  elif opt.lower() == 'sgd':
    return optim.SGD
  elif opt.lower() == 'adamw':
    return optim.AdamW
  else:
    raise ValueError('Optimizer not found')  
def train(model_type,epochs,lr,path_csv_dataset,feature_folder_saving_path,global_foder_name,path_dataset,k_fold,opt_list,batch_train,
          list_GRU_hidden_size,list_GRU_num_layers,lsit_GRU_dropout,GRU_concatenate_temp_dim,list_init_network,early_stopping,
          regularization_loss,list_regularization_lambda,is_round_output_loss, GRU_output_size,key_for_early_stopping,seed_random_state,
          is_shuffle_training_batch,is_shuffle_video_chunks,clip_length,target_metric_best_model,is_plot_dataset_distribution,layer_norm):
  
  emb_dim = 384 if 'S' in model_type else 768
  
  model_type = get_model_type(model_type)
  
  pooling_embedding_reduction = EMBEDDING_REDUCTION.MEAN_SPATIAL
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  
  # path_dict ={
  #   'all' : os.path.join('partA','starting_point','samples.csv'),
    # 'train' : os.path.join('partA','starting_point','train_21.csv'),
    # 'val' : os.path.join('partA','starting_point','val_26.csv'),
    # 'test' : os.path.join('partA','starting_point','test_5.csv')
  # }
  # print(f'global_path: {global_path}')
  # if global_path:
  #   path_dataset = os.path.join(GLOBAL_PATH.NAS_PATH,'partA','video','video')  
  #   path_csv_dataset = os.path.join(GLOBAL_PATH.NAS_PATH,'partA','starting_point','samples_exc_no_detection.csv')
  #   feature_folder_saving_path = os.path.join(GLOBAL_PATH.NAS_PATH,'partA','video','features','samples_16_aligned_cropped')  
  #   global_foder_name=os.path.join(GLOBAL_PATH.NAS_PATH,'history_run')
  # else:
  #   path_dataset = os.path.join('partA','video','video')  
  #   path_csv_dataset = os.path.join('partA','starting_point','samples_exc_no_detection.csv')
  #   feature_folder_saving_path = os.path.join('partA','video','features','samples_16_aligned_cropped')
  #   global_foder_name= 'history_run'
  head = HEAD.GRU
  stride_window_in_video = 16 # sliding window
  lr_list = lr
  optim_list = [get_optimizer(opt) for opt in opt_list] 
  
                                                                        # can be 384*8(small), 768*8(base), 1408*8(large) [temporal_dim considered feature in GRU] 
  # features_folder_saving_path = os.path.join('partA','video','features',f'{os.path.split(path_csv_dataset)[-1][:-4]}_{stride_window_in_video}') # get the name of the csv file
  print(f'\nlr_list: {lr_list}')
  print(f'optim_list: {optim_list}')
  
  # config_test_dict = {
  #   'list_init_network': list_init_network,
  #   'list_GRU_hidden_size': list_GRU_hidden_size,
  #   'list_GRU_num_layers': list_GRU_num_layers,
  #   'lsit_GRU_dropout': lsit_GRU_dropout,
  #   'opt_list': opt_list,
  #   'lr_list': lr_list,
  # }
  for init_network in list_init_network:
    for GRU_hidden_size in list_GRU_hidden_size:
      for GRU_num_layers in list_GRU_num_layers:
        for GRU_dropout in lsit_GRU_dropout:
          for regularization_lambda in list_regularization_lambda:
            params = {
              'output_size': GRU_output_size,
              'hidden_size': GRU_hidden_size,
              'num_layers': GRU_num_layers,
              'dropout': GRU_dropout,
              'layer_norm': layer_norm,
              'input_size': emb_dim * 8 if GRU_concatenate_temp_dim  else emb_dim # can be 384  (small), 768  (base), 1408  (large) [temporal_dim considered as input sequence for GRU]
            }
            criterion = nn.L1Loss() if params['output_size'] == 1 else nn.CrossEntropyLoss()
            for lr in lr_list:
              for optim_fn in optim_list:
                
                start = time.time()
                scripts.run_train_test(
                  model_type=model_type, 
                  pooling_embedding_reduction=pooling_clips_reduction,
                  pooling_clips_reduction=pooling_clips_reduction,
                  sample_frame_strategy=sample_frame_strategy, 
                  path_csv_dataset=path_csv_dataset, 
                  path_video_dataset=path_dataset,
                  head=head,
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
                  # scheduler=scheduler,
                  )
                # TODO generate csv file to summarize the results of k-cross validation
                # log_dict['time_taken_min']=int((time.time()-start)/60)
                # df = pd.DataFrame([log_dict])
                # df = df[CSV.sort_cols + [col for col in df.columns if col not in CSV.sort_cols]]
                # print(f'cols df: {df.columns}')
                # if not os.path.exists(os.path.join(global_foder_name,'summary_log.csv')):
                #   df.to_csv(os.path.join(global_foder_name,'summary_log.csv'),index=False)
                # else:
                #   df.to_csv(os.path.join(global_foder_name,'summary_log.csv'),index=False,mode='a', header=False)
                print(f'Time taken for this run: {(time.time()-start)//60} min')
  
  print('Training completed')
  print(f'Check the path {os.path.join(global_foder_name,"summary_log.csv")} file for the results of the training')              
              
def generate_path(path):
  return os.path.join(GLOBAL_PATH.NAS_PATH,path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train model')
  parser.add_argument('--mt', type=str, default='B', help='Model type. Can be either B or S or I')
  parser.add_argument('--gp', action='store_true', help='Global path')
  parser.add_argument('--lr', type=float, nargs='*',default=0.0001, help='Learning rate')
  parser.add_argument('--ep', type=int, default=500, help='Number of epochs')
  parser.add_argument('--csv', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), help='Path to csv file')
  parser.add_argument('--ffsp', type=str, default=os.path.join('partA','video','features','samples_16_frontalized'), help='Path to feature folder saving path')
  parser.add_argument('--global_folder_name', type=str, default=f'history_run', help='Global folder name for logging')
  parser.add_argument('--path_video_dataset', type=str, default=os.path.join('partA','video','video_frontalized'), help='Path to dataset')
  parser.add_argument('--k_fold', type=int, default=3, help='Number of k fold cross validation')
  parser.add_argument('--opt', type=str, nargs='*',default='adam', help='Optimizer, can be adam, sgd')
  parser.add_argument('--batch_train', type=int, default=8700, help='Batch size for training')
  parser.add_argument('--GRU_hidden_size', type=int,nargs='*', default=1024, help='GRU hidden size')
  parser.add_argument('--GRU_num_layers', type=int, nargs='*',default=2, help='GRU number of layers')
  parser.add_argument('--GRU_dropout', type=float,nargs='*' ,default=0.0, help='GRU dropout')
  parser.add_argument('--GRU_concatenate_temp_dim', action='store_true', help='Concatenate temporal dimension')
  parser.add_argument('--GRU_output_size', type=int, default=1, help='Output size of GRU. If 1 is regression, if >1 is classification')
  parser.add_argument('--init_network', type=str,nargs='*',default='default', help='Initialize network, can be xavier,default,uniform')
  parser.add_argument('--reg_loss', type=str, default='', help='Regularization type, can be L1 or L2')
  parser.add_argument('--reg_lambda', type=float,nargs='*', default=[0], help='Regularization lambda')
  parser.add_argument('--is_round_output_loss', action='store_true', help='Round output from regression before compute the loss')
  parser.add_argument('--key_early_stopping', type=str, default='val_loss', help='Key for early stopping. Can be val_loss or val_macro_precision' )# must be in dict_eval keys
  parser.add_argument('--layer_norm', action='store_true', help='Put Layer normalization before linear layer')
  # python3 train_model.py --mt I --gp -- lr 0.00001 0.0001 --ep 500 --csv partA/starting_point/samples_exc_no_detection.csv --ffsp partA/video/features/samples_vit_front --global_folder_name history_run --path_video_dataset partA/video/video_frontalized --k_fold 3 --opt adam --batch_train 8700 --GRU_hidden_size 1024 --GRU_num_layers 2 --GRU_dropout 0.3 0.5 --init_network default --reg_loss L2 --reg_lambda 0.000001 0.000005
  args = parser.parse_args()
  ti = int(time.time())
  args.global_folder_name = f'{args.global_folder_name}_{ti}'
  if args.gp:
    args.csv = generate_path(args.csv)
    args.ffsp = generate_path(args.ffsp)
    args.path_video_dataset = generate_path(args.path_video_dataset)
    args.global_folder_name = generate_path(args.global_folder_name)
  # else:
  #   args.global_folder_name = f'{args.global_folder_name}_{int(time.time())}'
  # print all args
  print('\n \nFollowing are the arguments passed:')
  print(args)
  clip_length = 16
  target_metric_best_model = 'val_loss'
  seed_random_state = 42
  early_stopping = (earlyStoppingLoss(patience=50,min_delta=0.0001,threshold_mode='abs') if args.key_early_stopping == 'val_loss' 
                  else 
                  earlyStoppingAccuracy(patience=50,min_delta=0.001,threshold_mode='abs'))
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
  }
  with open(os.path.join(args.global_folder_name,'_config_prompt.txt'),'w') as f:
    for key, value in config_prompt.items():
      f.write(f'{key}: {value}\n')
  train(model_type=args.mt,
        epochs=args.ep,
        lr=args.lr,
        path_csv_dataset=args.csv,
        feature_folder_saving_path=args.ffsp,
        global_foder_name=args.global_folder_name,
        path_dataset=args.path_video_dataset,
        k_fold=args.k_fold,
        opt_list=args.opt,
        batch_train=args.batch_train,
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
        is_shuffle_video_chunks=False,
        is_shuffle_training_batch=True,
        is_plot_dataset_distribution=False
        )
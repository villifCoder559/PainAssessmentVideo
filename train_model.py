from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY, HEAD,GLOBAL_PATH
import os
from custom.model import Model_Advanced
from transformers import AutoImageProcessor
from custom.head import HeadSVR, HeadGRU
import time
import torch.nn as nn
import torch.optim as optim
import custom.scripts as scripts
import argparse
from custom.helper import GLOBAL_PATH,CSV
import pandas as pd

# import wandb

def train(model_type,epochs,lr,path_csv_dataset,feature_folder_saving_path,global_foder_name,path_dataset,k_fold,opt_list,batch_train,
          list_GRU_hidden_size,list_GRU_num_layers,lsit_GRU_dropout,GRU_concatenate_temp_dim,list_init_network):
  emb_dim = 768 if 'B' in model_type else 384
  model_type = MODEL_TYPE.VIDEOMAE_v2_B if model_type == 'B' else MODEL_TYPE.VIDEOMAE_v2_S
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
  stride_window_in_video = 16
  lr_list = lr
  optim_list = [optim.Adam if opt == 'adam' else optim.SGD for opt in opt_list] 
  
                                                                        # can be 384*8(small), 768*8(base), 1408*8(large) [temporal_dim considered feature in GRU] 
  # features_folder_saving_path = os.path.join('partA','video','features',f'{os.path.split(path_csv_dataset)[-1][:-4]}_{stride_window_in_video}') # get the name of the csv file
  print(f'\nlr_list: {lr_list}')
  print(f'optim_list: {optim_list}')
  
  config_test_dict = {
    'list_init_network': list_init_network,
    'list_GRU_hidden_size': list_GRU_hidden_size,
    'list_GRU_num_layers': list_GRU_num_layers,
    'lsit_GRU_dropout': lsit_GRU_dropout,
    'opt_list': opt_list,
    'lr_list': lr_list,
  }

  for init_network in list_init_network:
    for GRU_hidden_size in list_GRU_hidden_size:
      for GRU_num_layers in list_GRU_num_layers:
        for GRU_dropout in lsit_GRU_dropout:
          params = {
            'hidden_size': GRU_hidden_size,
            'num_layers': GRU_num_layers,
            'dropout': GRU_dropout,
            'input_size': emb_dim * 8 if GRU_concatenate_temp_dim  else emb_dim # can be 384  (small), 768  (base), 1408  (large) [temporal_dim considered as input sequence for GRU]
          }
          for lr in lr_list:
            for optim_fn in optim_list:
              start = time.time()
              log_dict = scripts.run_train_test(model_type=model_type, 
                                    pooling_embedding_reduction=pooling_embedding_reduction, 
                                    pooling_clips_reduction=pooling_clips_reduction, 
                                    sample_frame_strategy=sample_frame_strategy, 
                                    path_csv_dataset=path_csv_dataset, 
                                    path_video_dataset=path_dataset,
                                    head=head,
                                    stride_window_in_video=stride_window_in_video, 
                                    head_params=params,
                                    global_foder_name=global_foder_name,
                                    k_fold = k_fold,
                                    epochs = epochs,
                                    train_size=0.8,
                                    test_size=0.1,
                                    val_size=0.1,
                                    batch_size_training=batch_train,
                                    batch_size_feat_extraction=100,
                                    features_folder_saving_path=feature_folder_saving_path,
                                    criterion = nn.L1Loss(),
                                    optimizer_fn = optim_fn,
                                    lr = lr,
                                    regularization_loss='',
                                    regularization_lambda=0.0,
                                    init_network=init_network,
                                    random_state_split_dataset=42,
                                    only_train=True,
                                    is_save_features_extracted=False, 
                                    is_validation=True,
                                    is_plot_dataset_distribution=True,
                                    is_plot_loss=True,
                                    is_plot_tsne_backbone_feats=True,
                                    is_plot_tsne_head_pred=True,
                                    is_plot_tsne_gru_feats=True,
                                    is_create_video_prediction=True,
                                    is_create_video_prediction_per_video=True,
                                    is_round_output_loss=False,
                                    is_shuffle_training_batch=True,
                                    is_shuffle_video_chunks=False,
                                    is_download_if_unavailable=False
                                  )
              # TODO generate csv file to summarize the results of k-cross validation
              log_dict['time_taken_min']=int((time.time()-start)/60)
              df = pd.DataFrame([log_dict])
              # check if file exists
              # df.reindex(columns=cols_new)
              # df=df[cols_new]
              df = df[CSV.sort_cols + [col for col in df.columns if col not in CSV.sort_cols]]
              print(f'cols df: {df.columns}')
              if not os.path.exists(os.path.join(global_foder_name,'summary_log.csv')):
                df.to_csv(os.path.join(global_foder_name,'summary_log.csv'),index=False)
              else:
                df.to_csv(os.path.join(global_foder_name,'summary_log.csv'),index=False,mode='a', header=False)
              print(f'Time taken for this run: {(time.time()-start)//60} min')
  
  print('Training completed')
  print(f'Check the path {os.path.join(global_foder_name,"summary_log.csv")} file for the results of the training')              
              
def generate_path(path):
  return os.path.join(GLOBAL_PATH.NAS_PATH,path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train model')
  parser.add_argument('--mt', type=str, default='B', help='Model type. Can be either B or S')
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
  parser.add_argument('--init_network', type=str,nargs='*',default='default', help='Initialize network, can be xavier,default,uniform')
  
  
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
        GRU_concatenate_temp_dim=args.GRU_concatenate_temp_dim
        )
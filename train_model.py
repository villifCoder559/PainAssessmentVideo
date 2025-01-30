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

def train(model_type,global_path,epochs,lr):
  model_type = MODEL_TYPE.VIDEOMAE_v2_B
  pooling_embedding_reduction = EMBEDDING_REDUCTION.MEAN_SPATIAL
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  # path_dict ={
  #   'all' : os.path.join('partA','starting_point','samples.csv'),
    # 'train' : os.path.join('partA','starting_point','train_21.csv'),
    # 'val' : os.path.join('partA','starting_point','val_26.csv'),
    # 'test' : os.path.join('partA','starting_point','test_5.csv')
  # }
  print(f'global_path: {global_path}')
  if global_path:
    path_dataset = os.path.join(GLOBAL_PATH.NAS_PATH,'partA','video','video')  
    path_cvs_dataset = os.path.join(GLOBAL_PATH.NAS_PATH,'partA','starting_point','samples_exc_no_detection.csv')
    feature_folder_saving_path = os.path.join(GLOBAL_PATH.NAS_PATH,'partA','video','features','samples_16_aligned_cropped')  
    global_foder_name=os.path.join(GLOBAL_PATH.NAS_PATH,'history_run')
  else:
    path_dataset = os.path.join('partA','video','video')  
    path_cvs_dataset = os.path.join('partA','starting_point','samples_exc_no_detection.csv')
    feature_folder_saving_path = os.path.join('partA','video','features','samples_16_aligned_cropped')
    global_foder_name= 'history_run'
  head = HEAD.GRU
  stride_window_in_video = 16
  params = {
    'hidden_size': 1024,
    'num_layers': 2,
    'dropout': 0.0,
    'input_size': 768 * 8 # can be 384  (small), 768  (base), 1408  (large) [temporal_dim considered as input sequence for GRU]
                      # can be 384*8(small), 768*8(base), 1408*8(large) [temporal_dim considered feature in GRU] 
  }
  # features_folder_saving_path = os.path.join('partA','video','features',f'{os.path.split(path_csv_dataset)[-1][:-4]}_{stride_window_in_video}') # get the name of the csv file
  lr_list = [lr]
  optim_list = [optim.Adam]
  for lr in lr_list:
    for optim_fn in optim_list:
      model_advanced_dict = scripts.run_train_test(model_type=model_type, 
                            pooling_embedding_reduction=pooling_embedding_reduction, 
                            pooling_clips_reduction=pooling_clips_reduction, 
                            sample_frame_strategy=sample_frame_strategy, 
                            path_csv_dataset=path_cvs_dataset, 
                            path_video_dataset=path_dataset,
                            head=head,
                            stride_window_in_video=stride_window_in_video, 
                            head_params=params,
                            global_foder_name=global_foder_name,
                            k_fold = 5,
                            epochs = epochs,
                            train_size=0.8,
                            test_size=0.1,
                            val_size=0.1,
                            batch_size_training=100,
                            batch_size_feat_extraction=100,
                            features_folder_saving_path=feature_folder_saving_path,
                            criterion = nn.L1Loss(),
                            optimizer_fn = optim_fn,
                            lr = lr,
                            regularization_loss='',
                            regularization_lambda=0.0,
                            init_network='default',
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
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train model')
  parser.add_argument('--mt', type=str, default='B', help='Model type. Can be either B or S')
  parser.add_argument('--gp', type=int, default=0, help='Global path')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--ep', type=int, default=500, help='Number of epochs')
  args = parser.parse_args()
  train(model_type=args.mt,
        global_path=args.gp,
        epochs=args.ep,
        lr=args.lr)
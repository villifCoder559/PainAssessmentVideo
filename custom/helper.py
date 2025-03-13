from enum import Enum
import os
class SAMPLE_FRAME_STRATEGY(Enum):
  UNIFORM = 'uniform'
  SLIDING_WINDOW = 'sliding_window'
  CENTRAL_SAMPLING = 'central_sampling'
  RANDOM_SAMPLING = 'random_sampling' 
  
class MODEL_TYPE(Enum):
  VIDEOMAE_v2_S = os.path.join('VideoMAEv2','pretrained',"vit_s_k710_dl_from_giant.pth")
  VIDEOMAE_v2_B = os.path.join('VideoMAEv2','pretrained',"vit_b_k710_dl_from_giant.pth")
  VIDEOMAE_v2_G_pt_1200e = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e.pth")
  VIDEOMAE_v2_G_pt_1200e_K710_it_HMDB51_ft = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e_k710_it_hmdb51_ft.pth")
  ViT_image = 'ViT_image'
class EMBEDDING_REDUCTION(Enum):  
  # [B,t,p,p,emb] -> [B,1,p,p,emb] ex: [3,8,14,14,768] -> [3,1,14,14,768]
  MEAN_TEMPORAL = (1) 
  
  # [B,t,p,p,emb] -> [B,t,1,1,emb] ex: [3,8,14,14,768] -> [3,8,1,1,768]
  MEAN_SPATIAL = (2,3) 
  
  # [B,t,p,p,emb] -> [B,1,1,1,emb] ex: [3,8,14,14,768] -> [3,1,1,1,768]
  MEAN_TEMPORAL_SPATIAL = (1,2,3)
  NONE = None
  
class INSTANCE_MODEL_NAME(Enum): # model.__class__.__name__
  LINEARPROBE = 'LinearProbe'
  GRUPROBE = 'GRUProbe'
  ATTENTIVEPROBE = 'AttentiveProbe'
  
class CLIPS_REDUCTION(Enum): 
  # [B,t,p,p,emb] -> [1,t,p,p,emb] ex: [3,8,14,14,768] -> [1,8,14,14,768]
  MEAN = (0)
  NONE = None

class HEAD(Enum):
  SVR = 'SVR'
  GRU = 'GRU'
  ATTENTIVE = 'ATTENTIVE'
  LINEAR = 'LINEAR'

class GLOBAL_PATH:
  NAS_PATH = os.path.join('/equilibrium','fvilli','PainAssessmentVideo')
class CUSTOM_DATASET_TYPE(Enum):
  AGGREGATED = 'aggregated' # features reduced (spatial reduction) and saved in one folder 
  WHOLE = 'whole' # features not reduced and saved in more folders (like Biovid video)
  BASE = 'base' # video->frames->backbone->features
class CSV:
  sort_cols = [
                'k_fold', 
                'tot_test_accuracy',
                'k-0_test-accurracy',
                'k-1_test-accurracy',
                'k-2_test-accurracy',
                'k-0_val_accuracy',
                'k-1_val_accuracy',
                'k-2_val_accuracy',
                'k-0_train_accuracy',
                'k-1_train_accuracy',
                'k-2_train_accuracy',
                'epochs',
                'optimizer_fn',
                'lr',
                'criterion',
                'batch_size_training', 
                'GRU_hidden_size',
                'GRU_num_layers',
                'GRU_dropout',
                'GRU_input_size',
                'regularization_lambda',
                'regularization_loss',
                'init_network',
                'k-0_train-loss',
                'k-0_val-loss',
                'k-0_test-loss',
                'k-1_train-loss',
                'k-1_val-loss',
                'k-1_test-loss',
                'k-2_train-loss',
                'k-2_val-loss',
                'k-2_test-loss',
                # 'pooling_embedding_reduction',
                # 'pooling_clips_reduction',
                # 'sample_frame_strategy',
                # 'path_csv_dataset',
                # 'path_video_dataset',
                # 'head',
                # 'model_type',
                # 'stride_window_in_video',
                # 'train_size',
                # 'val_size', 
                # 'test_size',
                # 'random_state',
                # 'round_output_loss',
                # 'shuffle_video_chunks',
                # 'shuffle_training_batch',
                # 'k-0_s-0_val-loss',
                # 'k-0_s-0_train-loss',
                # 'k-0_s-0_val-loss-class-avg',
                # 'k-0_s-0_val-loss-subject-avg',
                # 'k-0_s-0_train-loss-class-avg',
                # 'k-0_s-0_train-loss-subject-avg',
                # 'k-0_s-1_val-loss',
                # 'k-0_s-1_train-loss',
                # 'k-0_s-1_val-loss-class-avg',
                # 'k-0_s-1_val-loss-subject-avg',
                # 'k-0_s-1_train-loss-class-avg',
                # 'k-0_s-1_train-loss-subject-avg',
                # 'k-0_test-loss-class-avg',
                # 'k-0_test-loss-subject-avg',
                # 'k-0_train-loss-class-avg',
                # 'k-0_train-loss-subject-avg',
                # 'k-0_val-loss-class-avg',
                # 'k-0_val-loss-subject-avg',
                # 'k-1_s-0_val-loss',
                # 'k-1_s-0_train-loss',
                # 'k-1_s-0_val-loss-class-avg',
                # 'k-1_s-0_val-loss-subject-avg',
                # 'k-1_s-0_train-loss-class-avg',
                # 'k-1_s-0_train-loss-subject-avg',
                # 'k-1_s-1_val-loss',
                # 'k-1_s-1_train-loss',
                # 'k-1_s-1_val-loss-class-avg',
                # 'k-1_s-1_val-loss-subject-avg',
                # 'k-1_s-1_train-loss-class-avg',
                # 'k-1_s-1_train-loss-subject-avg',
                # 'k-1_test-loss-class-avg',
                # 'k-1_test-loss-subject-avg',
                # 'k-1_train-loss-class-avg',
                # 'k-1_train-loss-subject-avg',
                # 'k-1_val-loss-class-avg',
                # 'k-1_val-loss-subject-avg',
                # 'k-2_s-0_val-loss',
                # 'k-2_s-0_train-loss',
                # 'k-2_s-0_val-loss-class-avg',
                # 'k-2_s-0_val-loss-subject-avg',
                # 'k-2_s-0_train-loss-class-avg',
                # 'k-2_s-0_train-loss-subject-avg',
                # 'k-2_s-1_val-loss', 
                # 'k-2_s-1_train-loss',
                # 'k-2_s-1_val-loss-class-avg',
                # 'k-2_s-1_val-loss-subject-avg',
                # 'k-2_s-1_train-loss-class-avg',
                # 'k-2_s-1_train-loss-subject-avg',
                # 'k-2_test-loss-class-avg',
                # 'k-2_test-loss-subject-avg',
                # 'k-2_train-loss-class-avg',
                # 'k-2_train-loss-subject-avg',
                # 'k-2_val-loss-class-avg',
                # 'k-2_val-loss-subject-avg',
                # 'tot_test_loss-avg',
                # 'tot_test_loss-class-avg',
                # 'tot_test_loss-subject-avg',
                # 'time_taken_min',
                # 'folder_path'
                ] 
              

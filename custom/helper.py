from enum import Enum
import os
from pathlib import Path

stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
saving_rate_training_logs = 3
step_shift = 8700
dict_data = None

LOG_GRADIENT_PER_MODULE = False

def get_shift_for_sample_id(folder_feature):
  if 'hflip' in folder_feature:
    return step_shift * 1
  elif 'jitter' in folder_feature:
    return step_shift * 2
  elif 'rotation' in folder_feature:
    return step_shift * 3
  return 0
  
class SAMPLE_FRAME_STRATEGY(Enum):
  UNIFORM = 'uniform'
  SLIDING_WINDOW = 'sliding_window'
  CENTRAL_SAMPLING = 'central_sampling'
  RANDOM_SAMPLING = 'random_sampling' 
  
class MODEL_TYPE(Enum):
  VIDEOMAE_v2_S = os.path.join('VideoMAEv2','pretrained',"vit_s_k710_dl_from_giant.pth")
  VIDEOMAE_v2_B = os.path.join('VideoMAEv2','pretrained',"vit_b_k710_dl_from_giant.pth")
  # VIDEOMAE_v2_G_pt_1200e = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e.pth")
  VIDEOMAE_v2_G = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e_k710_ft.pth")
  ViT_image = 'ViT_image'
  def get_model_type(type):
    """
    Retrieves the corresponding model type based on the provided type identifier.

    Args:
      type (str): A string representing the model type. 
            Accepted values are:
            - 'S': Corresponds to MODEL_TYPE.VIDEOMAE_v2_S
            - 'B': Corresponds to MODEL_TYPE.VIDEOMAE_v2_B
            - 'G': Corresponds to MODEL_TYPE.VIDEOMAE_v2_G
            - 'ViT_image': Corresponds to MODEL_TYPE.ViT_image

    Returns:
      MODEL_TYPE: The corresponding model type constant.

    Raises:
      ValueError: If the provided type is not one of the accepted values.
    """
    if type == 'S':
      return MODEL_TYPE.VIDEOMAE_v2_S
    elif type == 'B':
      return MODEL_TYPE.VIDEOMAE_v2_B
    elif type == 'G':
      return MODEL_TYPE.VIDEOMAE_v2_G
    elif type == 'ViT_image':
      return MODEL_TYPE.ViT_image
    else:
      raise ValueError(f"Model type {type} not found. Choose between 'S','B','G' or 'ViT_image'")
  def get_embedding_size(type):
    """
    Returns the embedding size for a given model type.

    Parameters:
    type (str): The type of the model. Accepted values are:
          - 'S': Small model, returns an embedding size of 384.
          - 'B': Base model, returns an embedding size of 768.
          - 'G': Large model, returns an embedding size of 1408.
          - 'ViT_image': Vision Transformer model, returns an embedding size of 768.

    Returns:
    int: The embedding size corresponding to the specified model type.

    Raises:
    ValueError: If the provided model type is not one of the accepted values.
    """
    if type == 'S':
      return 384
    elif type == 'B':
      return 768
    elif type == 'G':
      return 1408
    elif type == 'ViT_image':
      return 768
    else:
      raise ValueError(f"Model type {type} not found. Choose between 'S','B','G' or 'ViT_image'")
class EMBEDDING_REDUCTION(Enum):  
  # [B,t,p,p,emb] -> [B,1,p,p,emb] ex: [3,8,14,14,768] -> [3,1,14,14,768]
  MEAN_TEMPORAL = (1,) 
  
  # [B,t,p,p,emb] -> [B,t,1,1,emb] ex: [3,8,14,14,768] -> [3,8,1,1,768]
  MEAN_SPATIAL = (2,3) 
  
  # [B,t,p,p,emb] -> [B,1,1,1,emb] ex: [3,8,14,14,768] -> [3,1,1,1,768]
  MEAN_TEMPORAL_SPATIAL = (1,2,3)
  NONE = None
  
  def get_embedding_reduction(pooling_embedding_reduction):
    if pooling_embedding_reduction.lower() == 'spatial':
      return EMBEDDING_REDUCTION.MEAN_SPATIAL
    elif pooling_embedding_reduction.lower() == 'temporal':
      return EMBEDDING_REDUCTION.MEAN_TEMPORAL
    elif pooling_embedding_reduction.lower() == 'all':
      return EMBEDDING_REDUCTION.MEAN_TEMPORAL_SPATIAL
    elif pooling_embedding_reduction.lower() == 'none':
      return EMBEDDING_REDUCTION.NONE
    else:
      raise ValueError(f'Pooling embedding reduction not recognized: {pooling_embedding_reduction}. Can be spatial, temporal, all or none')
class INSTANCE_MODEL_NAME(Enum): # model.__class__.__name__
  LINEARPROBE = 'LinearProbe'
  GRUPROBE = 'GRUProbe'
  ATTENTIVEPROBE = 'AttentiveProbe'
  AttentiveClassifier = 'AttentiveClassifier' # JEPA implementation
  
class CLIPS_REDUCTION(Enum): 
  # [B,t,p,p,emb] -> [1,t,p,p,emb] ex: [3,8,14,14,768] -> [1,8,14,14,768]
  MEAN = (0)
  NONE = None

class HEAD(Enum):
  GRU = 'GRU'
  ATTENTIVE = 'ATTENTIVE'
  ATTENTIVE_JEPA = 'ATTENTIVE_JEPA'
  LINEAR = 'LINEAR'

class GLOBAL_PATH:
  NAS_PATH = os.path.join('/equilibrium','fvilli','PainAssessmentVideo')
  def get_global_path(path):
    if path is not  None:
      tmp_path = path
      if isinstance(path,Path):
        tmp_path = str(path)
      if tmp_path[0] == '/':
        return path
      else:
        if isinstance(path,Path):
          return Path(GLOBAL_PATH.NAS_PATH) / path
        else:
          return os.path.join(GLOBAL_PATH.NAS_PATH,path)
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
              

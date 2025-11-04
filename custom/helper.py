from enum import Enum
import os
from pathlib import Path
import multiprocessing as mp


stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
saving_rate_training_logs = 3
dict_data = None
desired_order_csv = ['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']
train_time_logs = {}

# Profile workers in dataloader
time_profile_dict = mp.Manager().dict()
time_profiling_enabled = False

AMP_ENABLED = False
AMP_DTYPE = None
LOG_GRADIENT_PER_MODULE = False
LOG_PER_CLASS = True
LOG_PER_SUBJECT = True
LOG_CONFIDENCE_PREDICTION = False
LOG_HISTORY_SAMPLE = False
QUERIES_AGG_METHOD = ['mean','max']
LOG_LOSS_ACCURACY = False
LOG_GRADIENT_NORM = False
SAVE_LAST_EPOCH_MODEL = False
FORCE_SPLIT_K_FOLD = False # Only for CAER dataset!, turns on in the train_model.py script

SAVE_PTH_MODEL = False

LOG_CROSS_ATTENTION = {
  'enable': False,
  'state':'train'
}
LOG_VIDEO_EMBEDDINGS = {
  'enable': False,
  'embeddings':[],
  'predictions':[],
  'labels':[],
  'sample_ids':[]
}

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
step_shift = 8700 # nr of samples in Biovid video dataset, set in train_model because depends on dataset

def get_augmentation_type(sample_id):
  if is_hflip_augmentation(sample_id):
    return 'hflip'
  elif is_color_jitter_augmentation(sample_id):
    return 'jitter'
  elif is_rotation_augmentation(sample_id):
    return 'rotation'
  elif is_latent_basic_augmentation(sample_id):
    return 'latent_basic'
  elif is_latent_masking_augmentation(sample_id):
    return 'latent_masking'
  elif is_spatial_shift_augmentation(sample_id):
    return 'shift'
  else:
    raise ValueError(f'Augmentation type not found for sample_id: {sample_id}')

def is_hflip_augmentation(sample_id):
  return sample_id > step_shift and sample_id <= step_shift * 2

def is_color_jitter_augmentation(sample_id):
  return sample_id > step_shift * 2 and sample_id <= step_shift * 3

def is_rotation_augmentation(sample_id):
  return sample_id > step_shift * 3 and sample_id <= step_shift * 4

def is_latent_basic_augmentation(sample_id):
  return sample_id > step_shift * 4 and sample_id <= step_shift * 5

def is_latent_masking_augmentation(sample_id):
  return sample_id > step_shift * 5 and sample_id <= step_shift * 6

def is_spatial_shift_augmentation(sample_id):
  return sample_id > step_shift * 10 and sample_id <= step_shift * 11

def get_shift_for_sample_id(folder_feature):
  # step_shift * 4 => gaussian_noise in feats space
  # step_shift * 5 => masking in feats space
  if 'hflip' in folder_feature:
    return step_shift * 1
  if 'jitter' in folder_feature:
    return step_shift * 2
  if 'rotation' in folder_feature:
    return step_shift * 3
  if 'latent_basic' in folder_feature:
    return step_shift * 4
  if 'latent_masking' in folder_feature:
    return step_shift * 5
  if 'bottom_left' in folder_feature:
    return step_shift * 6
  if 'bottom_right' in folder_feature:
    return step_shift * 7
  if 'upper_left' in folder_feature:
    return step_shift * 8
  if 'upper_right' in folder_feature:
    return step_shift * 9
  if 'shift' in folder_feature:
    return step_shift * 10
  return 0
  
class SAMPLE_FRAME_STRATEGY(Enum):
  UNIFORM = 'uniform'
  SLIDING_WINDOW = 'sliding_window'
  CENTRAL_SAMPLING = 'central'
  RANDOM_SAMPLING = 'random' 
  
# class MODEL_TYPE(Enum):
#   # VideoMAEv2
#   VIDEOMAE_v2_G_unl = os.path.join("VideoMAEv2","pretrained","vit_g_hybrid_pt_1200e.pth")
#   VIDEOMAE_v2_S = os.path.join('VideoMAEv2','pretrained',"vit_s_k710_dl_from_giant.pth")
#   VIDEOMAE_v2_B = os.path.join('VideoMAEv2','pretrained',"vit_b_k710_dl_from_giant.pth")
#   VIDEOMAE_v2_G = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e_k710_ft.pth")
  
#   # V-Jepa 2
#   VJEPA2_G_384 = os.path.join('vjepa2','pretrained','vitg-384.pt')
  
#   # standard ViT
#   ViT_image = 'ViT_image'
#   def set_custom_model_type(type,custom_model_path):
#     if type == 'S':
#       MODEL_TYPE.VIDEOMAE_v2_S = custom_model_path
#     elif type == 'B':
#       MODEL_TYPE.VIDEOMAE_v2_B = custom_model_path
#     elif type == 'G':
#       MODEL_TYPE.VIDEOMAE_v2_G = custom_model_path
#     elif type == 'G_unl':
#       MODEL_TYPE.VIDEOMAE_v2_G_unl = custom_model_path
#     elif type == 'VJEPA2_G_384':
#       MODEL_TYPE.VJEPA2_G_384 = custom_model_path
#     elif type == 'ViT_image':
#       MODEL_TYPE.ViT_image = custom_model_path
#     else:
#       raise ValueError(f"Model type {type} not found. Choose between 'S','B','G','G_unl', 'VJEPA2_G_384' or 'ViT_image'")
#   def get_model_type(type):
#     """
#     Retrieves the corresponding model type based on the provided type identifier.

#     Args:
#       type (str): A string representing the model type. 
#             Accepted values are:
#             - 'S': Corresponds to MODEL_TYPE.VIDEOMAE_v2_S
#             - 'B': Corresponds to MODEL_TYPE.VIDEOMAE_v2_B
#             - 'G': Corresponds to MODEL_TYPE.VIDEOMAE_v2_G
#             - 'ViT_image': Corresponds to MODEL_TYPE.ViT_image

#     Returns:
#       MODEL_TYPE: The corresponding model type constant.

#     Raises:
#       ValueError: If the provided type is not one of the accepted values.
#     """
#     if type == 'S':
#       return MODEL_TYPE.VIDEOMAE_v2_S
#     elif type == 'B':
#       return MODEL_TYPE.VIDEOMAE_v2_B
#     elif type == 'G':
#       return MODEL_TYPE.VIDEOMAE_v2_G
#     elif type == 'G_unl':
#       return MODEL_TYPE.VIDEOMAE_v2_G_unl
#     elif type == 'VJEPA2_G_384':
#       return MODEL_TYPE.VJEPA2_G_384
#     elif type == 'ViT_image':
#       return MODEL_TYPE.ViT_image
#     else:
#       raise ValueError(f"Model type {type} not found. Choose between 'S','B','G' or 'ViT_image'")
#   def get_embedding_size(type):
#     """
#     Returns the embedding size for a given model type.

#     Parameters:
#     type (str): The type of the model. Accepted values are:
#           - 'S': Small model, returns an embedding size of 384.
#           - 'B': Base model, returns an embedding size of 768.
#           - 'G': Large model, returns an embedding size of 1408.
#           - 'ViT_image': Vision Transformer model, returns an embedding size of 768.

#     Returns:
#     int: The embedding size corresponding to the specified model type.

#     Raises:
#     ValueError: If the provided model type is not one of the accepted values.
#     """
#     if type == 'S':
#       return 384
#     elif type == 'B':
#       return 768
#     elif type == 'G':
#       return 1408
#     elif type == 'ViT_image':
#       return 768
#     else:
#       raise ValueError(f"Model type {type} not found. Choose between 'S','B','G' or 'ViT_image'")


class ModelTypeEntry:
  def __init__(self, name, value):
    self.name = name
    self.value = value
    
  def __eq__(self, other):
    if isinstance(other, ModelTypeEntry):
      return self.name == other.name
    return False
  
  def __hash__(self):
    return hash(self.name)

class MODEL_TYPE:
  _entries = {
    'VIDEOMAE_v2_G_unl': ModelTypeEntry('VIDEOMAE_v2_G_unl', os.path.join("VideoMAEv2", "pretrained", "vit_g_hybrid_pt_1200e.pth")),
    'VIDEOMAE_v2_S': ModelTypeEntry('VIDEOMAE_v2_S', os.path.join('VideoMAEv2', 'pretrained', "vit_s_k710_dl_from_giant.pth")),
    'VIDEOMAE_v2_B': ModelTypeEntry('VIDEOMAE_v2_B', os.path.join('VideoMAEv2', 'pretrained', "vit_b_k710_dl_from_giant.pth")),
    'VIDEOMAE_v2_G': ModelTypeEntry('VIDEOMAE_v2_G', os.path.join('VideoMAEv2', 'pretrained', "vit_g_hybrid_pt_1200e_k710_ft.pth")),
    # 'VJEPA2_G_384': ModelTypeEntry('VJEPA2_G_384', os.path.join('vjepa2', 'pretrained', 'vitg-384.pt')),
    'vjepa2_L_fpc64_256': ModelTypeEntry('vjepa2_L_fpc64_256', 'facebook/vjepa2-vitl-fpc64-256'),
    'vjepa2_G_fpc64_384': ModelTypeEntry('vjepa2_G_fpc64_384', 'facebook/vjepa2-vitg-fpc64-384'),
    'ViT_image': ModelTypeEntry('ViT_image', 'ViT_image'),
  }

  VIDEOMAE_v2_G_unl = _entries['VIDEOMAE_v2_G_unl']
  VIDEOMAE_v2_S     = _entries['VIDEOMAE_v2_S']
  VIDEOMAE_v2_B     = _entries['VIDEOMAE_v2_B']
  VIDEOMAE_v2_G     = _entries['VIDEOMAE_v2_G']
  VJEPA_v2_L_fpc64_256 = _entries['vjepa2_L_fpc64_256'] 
  VJEPA_v2_G_fpc64_384 = _entries['vjepa2_G_fpc64_384']
  ViT_image         = _entries['ViT_image']
  
  @classmethod
  def set_custom_model_type(cls, type, custom_model_path):
    mapping = {
      'S': 'VIDEOMAE_v2_S',
      'B': 'VIDEOMAE_v2_B',
      'G': 'VIDEOMAE_v2_G',
      'G_unl': 'VIDEOMAE_v2_G_unl',
      # 'VJEPA2_G_384': 'VJEPA2_G_384',
      'vjepa2_L_fpc64_256': 'vjepa2_L_fpc64_256',
      'vjepa2_G_fpc64_384': 'vjepa2_G_fpc64_384',
      'ViT_image': 'ViT_image'
    }
    key = mapping.get(type)
    if key is None or key not in cls._entries:
      raise ValueError(f"Model type '{type}' not found. Choose from: {list(mapping.keys())}")
    cls._entries[key].value = custom_model_path

  @classmethod
  def get_model_type(cls, type):
    mapping = {
      'S': 'VIDEOMAE_v2_S',
      'B': 'VIDEOMAE_v2_B',
      'G': 'VIDEOMAE_v2_G',
      'G_unl': 'VIDEOMAE_v2_G_unl',
      # 'VJEPA2_G_384': 'VJEPA2_G_384',
      'vjepa2_L_fpc64_256': 'vjepa2_L_fpc64_256',
      'vjepa2_G_fpc64_384': 'vjepa2_G_fpc64_384',
      'ViT_image': 'ViT_image'
    }
    key = mapping.get(type)
    if key and key in cls._entries:
      return cls._entries[key]
    raise ValueError(f"Model type '{type}' not found. Choose from: {list(mapping.keys())}")

  @classmethod
  def get_embedding_size(cls, typ):
    if 'S' in typ:
      return 384
    elif 'B' in typ:
      return 768
    elif 'G' in typ:
      return 1408
    elif 'ViT_image' in typ:
      return 768
    elif 'L' in typ:
      return 1024
    elif 'H' in typ:
      return 1280
    else:
      raise ValueError(
        f"Model type '{typ}' not supported for embedding size.")
    # if type == 'S':
    #   return 384
    # elif type in ['B', 'ViT_image']:
    #   return 768
    # elif
    # elif type == 'G':
    #   return 1408
    # else:
    #   raise ValueError(
    #     f"Model type '{type}' not supported for embedding size. Choose between 'S','B','G','ViT_image'")

class EMBEDDING_REDUCTION(Enum):  
  # [B,t,p,p,emb] -> [B,1,p,p,emb] ex: [3,8,14,14,768] -> [3,1,14,14,768]
  MEAN_TEMPORAL = (1,) 
  
  # [B,t,p,p,emb] -> [B,t,1,1,emb] ex: [3,8,14,14,768] -> [3,8,1,1,768]
  MEAN_SPATIAL = (2,3) 
  
  # [B,t,p,p,emb] -> [B,1,1,1,emb] ex: [3,8,14,14,768] -> [3,1,1,1,768]
  MEAN_TEMPORAL_SPATIAL = (1,2,3)
  NONE = None
  
  ADAPTIVE_POOLING_3D = (0,0,0) # [B,emb,t,p,p] -> [B,emb,2,2,2] ex: [3,768,8,14,14] -> [3,768,2,2,2]
  
  def get_embedding_reduction(pooling_embedding_reduction):
    if pooling_embedding_reduction.lower() == 'spatial':
      return EMBEDDING_REDUCTION.MEAN_SPATIAL
    elif pooling_embedding_reduction.lower() == 'temporal':
      return EMBEDDING_REDUCTION.MEAN_TEMPORAL
    elif pooling_embedding_reduction.lower() == 'all':
      return EMBEDDING_REDUCTION.MEAN_TEMPORAL_SPATIAL
    elif pooling_embedding_reduction.lower() == 'none':
      return EMBEDDING_REDUCTION.NONE
    elif pooling_embedding_reduction.lower() == 'adaptive_pooling_3d':
      return EMBEDDING_REDUCTION.ADAPTIVE_POOLING_3D
    else:
      raise ValueError(f'Pooling embedding reduction not recognized: {pooling_embedding_reduction}. Can be spatial, temporal, all or none')

class INSTANCE_MODEL_NAME(Enum): # model.__class__.__name__
  LINEARPROBE = 'LinearProbe'
  GRUPROBE = 'GRUProbe'
  ATTENTIVEPROBE = 'AttentiveProbe'
  AttentiveClassifier = 'AttentiveHeadJEPA' # JEPA implementation
  
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

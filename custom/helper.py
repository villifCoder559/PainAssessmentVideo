from enum import Enum
import os
from pathlib import Path
import multiprocessing as mp
from copy import deepcopy
import pandas as pd
import numpy as np

SAVE_MODEL_EVERY_N_EPOCHS = 0  # 0 to disable

unbc_count_to_id = {
  0: 115,
  1: 42,
  2: 66,
  3: 120,
  4: 43,
  5: 64,
  6: 124,
  7: 47,
  8: 95,
  9: 109,
  10: 101,
  11: 121,
  12: 97,
  13: 96,
  14: 92,
  15: 48,
  16: 106,
  17: 80,
  18: 108,
  19: 59,
  20: 49,
  21: 103,
  22: 52,
  23: 107,
  24: 123
}

stoic_subjects = [27,28,32,33,34,35,36,39,40,41,42,44,51,53,55,56,61,64,74,87]
saving_rate_training_logs = 3
dict_data = None
dict_data_in_memory = False
desired_order_csv = ['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']

# Profile workers in dataloader
time_profile_dict = mp.Manager().dict()
time_profiling_enabled = True


LOG_PER_CLASS = True
LOG_PER_SUBJECT = True
LOG_CONFIDENCE_PREDICTION = True
LOG_HISTORY_SAMPLE = False
SAVE_PTH_MODEL = False

LOG_VIDEO_EMBEDDINGS = {
  'enable': False,
  'embeddings':[],
  'predictions':[],
  'labels':[],
  'sample_ids':[]
}

LOG_NORM_CROSS_ATTN_BLOCK = {
  'enable': True,
  'logs':[]
}
  
def init_log_video_embeddings():
  global LOG_VIDEO_EMBEDDINGS
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
  
step_shift = None  ## keep none to force setting it before use

def set_step_shift(folder_feature):
  global step_shift
  if 'unbc' in folder_feature.lower(): # UNBC-McMaster dataset, sample_id starts from 1, 
    step_shift = 200
  elif 'caer' in folder_feature.lower(): # CAER dataset
    step_shift = 13176
  elif 'parta' in folder_feature.lower() or 'biovid' in folder_feature.lower(): # Biovid Part A, sample_id starts from 1 
    step_shift = 8700
  elif 'mintpain' in folder_feature.lower(): # MINTpain dataset, sample_id starts from 1
    step_shift = 3122
  elif 'morph' in folder_feature.lower(): # MORPH dataset, sample_id starts from 1
    step_shift = 50015
  elif 'agedb' in folder_feature.lower(): # AgeDB dataset, sample_id starts from 1 
    step_shift = 16488
  elif 'xite' in folder_feature.lower(): # XITE challenge train + test IDs, sample_id starts from 1
    step_shift = 3597
  elif 'pemf' in folder_feature.lower(): # PEMF dataset, sample_id starts from 1
    step_shift = 277
  else:  
    raise ValueError(f'Dataset not recognized in folder_feature: {folder_feature}')

def transform_sample_id(original_sample_id, augmentation_type):
  return original_sample_id + get_shift_for_sample_id(augmentation_type)

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

def is_zoom_augmentation(sample_id):
  return sample_id > step_shift * 11 and sample_id <= step_shift * 12

def is_shift_hflip_augmentation(sample_id):
  return sample_id > step_shift * 12 and sample_id <= step_shift * 13

def is_jitter_shift_augmentation(sample_id):
  return sample_id > step_shift * 13 and sample_id <= step_shift * 14

def is_jitter_rotation_augmentation(sample_id):
  return sample_id > step_shift * 14 and sample_id <= step_shift * 15

def is_rotation_zoom_augmentation(sample_id):
  return sample_id > step_shift * 15 and sample_id <= step_shift * 16

def is_reverse_augmentation(sample_id):
  return sample_id > step_shift * 16 and sample_id <= step_shift * 17

def is_timelapse_augmentation(sample_id):
  return sample_id > step_shift * 17 and sample_id <= step_shift * 18

def is_slowmotion_augmentation(sample_id):
  return sample_id > step_shift * 18 and sample_id <= step_shift * 19

def is_gaussian_smooth_augmentation(sample_id):
  """
  Check if the sample_id corresponds to a Gaussian smooth augmentation.

  Args:
    sample_id (int): The sample identifier to check.

  Returns:
    bool: True if the sample_id falls in the Gaussian smooth augmentation range.
  """
  return sample_id > step_shift * 22 and sample_id <= step_shift * 23

def get_augmentation_type(sample_id): # folder based on cvs_path, so must contain dataset name to
  if sample_id <= step_shift:
    return False
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
  elif is_zoom_augmentation(sample_id):
    return 'zoom'
  elif is_shift_hflip_augmentation(sample_id):
    return 'shift_hflip'
  elif is_jitter_shift_augmentation(sample_id):
    return 'jitter_shift'
  elif is_jitter_rotation_augmentation(sample_id):
    return 'jitter_rotation'
  elif is_rotation_zoom_augmentation(sample_id):
    return 'rotation_zoom'
  elif is_reverse_augmentation(sample_id):
    return 'reverse'
  elif is_timelapse_augmentation(sample_id):
    return 'timelapse'
  elif is_slowmotion_augmentation(sample_id):
    return 'slowmotion'
  elif is_gaussian_smooth_augmentation(sample_id):
    return 'gaussian'
  else:
    raise ValueError(f"Sample id {sample_id} not recognized for augmentation type. step_shift: {step_shift}")

augmentations_list = ['hflip','jitter','rotation','latent_basic','latent_masking','shift','zoom','gaussian']

def get_augmentation_availables(fold_feature_path):
  folder_name = os.path.basename(fold_feature_path)
  parent_dir = os.path.dirname(fold_feature_path)
  # Only sibling directories named <base>_<aug> or <base>$N count: a stray file such as
  # <base>_unmatched_augmented.txt would otherwise become an augmentation type.
  # os.listdir order is kept on purpose, it sets the row order of the augmented CSV.
  list_folder = [f for f in os.listdir(parent_dir)
                 if (f.startswith(folder_name + '_') or f.startswith(folder_name + '$'))
                 and os.path.isdir(os.path.join(parent_dir, f))]
  augment_available = [f.replace(folder_name+'_',"") for f in list_folder]
  print(f'Total augmentations: {len(augment_available)}.\n')
  # Strip $N variant suffixes and deduplicate — $N variants are handled transparently
  # by customDatasetWhole._load_element via aug_variant_folders, not via sample_id shifts
  augment_available = list(dict.fromkeys(a.split('$')[0] for a in augment_available))
  return augment_available


def get_sample_augmented(pain, list_subject, orig_df, dict_augmentation_per_sample, new_df, class_bins=None):
  class_bins = pd.Series(class_bins, index=orig_df.index) if class_bins is not None else orig_df['class_id']
  for sbj in list_subject:
    mask = (orig_df['subject_id'] == sbj) & (class_bins == pain)
    if mask.any():
      # filter by subject and pain class
      all_candidate_samples = orig_df[mask]
      while all_candidate_samples.shape[0]>0:  
        # pick the lowest count existing as selection criteria
        count_existing = (all_candidate_samples['sample_name'].values == new_df['sample_name'].values[:, None]).sum(axis=0)
        min_count = count_existing.min()
        all_candidate_samples = all_candidate_samples[count_existing == min_count]
        candidate = all_candidate_samples.sample(n=1).iloc[0].copy()
        id_candidate = candidate['sample_id']
        
        # check if there are augmentations available for this sample, if yes pick one randomly
        if len(dict_augmentation_per_sample[id_candidate]) > 0:
          augm = np.random.choice(dict_augmentation_per_sample[id_candidate], size=1)[0]
          candidate['sample_id'] = id_candidate + get_shift_for_sample_id(augm)
          dict_augmentation_per_sample[id_candidate].remove(augm)
          return candidate.to_dict()
        else:
          # no augmentations available, remove from candidates
          all_candidate_samples = all_candidate_samples[all_candidate_samples['sample_id'] != id_candidate]
  return None

def generate_balanced_dataframe(df_original,list_augmentations_available,target_samples_per_class,class_bins=None):
  # Deepcopy to modify each key separately
  dict_augm_available = {id: deepcopy(list_augmentations_available) for id in df_original['sample_id'].values}
  grouping = pd.Series(class_bins if class_bins is not None else df_original['class_id'].to_numpy(), index=df_original.index)
  target_samples_distribution = {class_id: target_samples_per_class for class_id in grouping.unique()}
  dict_class_upsample = {}
  original_class_distribution = grouping.value_counts().sort_index()
  
  # Determine how many samples to upsample per class
  for class_id, count in original_class_distribution.items():
    if count < target_samples_distribution[class_id]:
      dict_class_upsample[class_id] = target_samples_distribution[class_id] - count    
  upsample = np.array(list(dict_class_upsample.keys()))
  upsample = np.repeat(upsample, list(dict_class_upsample.values()))

  # Generate new balanced dataframe
  new_df = df_original.copy(deep=True)
  for pain in upsample:
    list_subject = new_df['subject_id'].value_counts().sort_values(ascending=True).keys().to_list()
    sample = get_sample_augmented(pain,
                                list_subject,
                                df_original,
                                new_df=new_df,
                                dict_augmentation_per_sample=dict_augm_available,
                                class_bins=grouping)
    if sample is not None:
      new_df = pd.concat([new_df, pd.DataFrame([sample])], ignore_index=True)
  return new_df


def get_shift_for_sample_id(folder_feature):
  ### Mixed augmentations ###
  is_composed = len([aug for aug in augmentations_list if aug in folder_feature]) > 1
  if is_composed:
    if 'hflip' in folder_feature and 'shift' in folder_feature:
      return step_shift * 12
    elif 'jitter' in folder_feature and 'shift' in folder_feature:
      return step_shift * 13
    elif 'rotation' in folder_feature and 'jitter' in folder_feature:
      return step_shift * 14
    elif 'rotation' in folder_feature and 'zoom' in folder_feature:
      return step_shift * 15
    else:
      raise ValueError(f"Mixed augmentation not recognized in folder_feature: {folder_feature}")
  else:
    ### Single augmentations ###
    if 'hflip' in folder_feature or 'h_flip' in folder_feature:
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
    if 'zoom' in folder_feature:
      return step_shift * 11
    if 'reverse' in folder_feature:
      return step_shift * 16
    if 'timelapse' in folder_feature:
      return step_shift * 17
    if 'slowmotion' in folder_feature:
      return step_shift * 18
    if 'top' in folder_feature:
      return step_shift * 20
    if 'down' in folder_feature:
      return step_shift * 21
    if 'gaussian' in folder_feature:
      return step_shift * 22
    return 0
  
class SAMPLE_FRAME_STRATEGY(Enum):
  UNIFORM = 'uniform'
  SLIDING_WINDOW = 'sliding_window'
  CENTRAL_SAMPLING = 'central'
  RANDOM_SAMPLING = 'random' 
  
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

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

class MODEL_TYPE:
  _entries = {
    'VIDEOMAE_v2_G_unl': ModelTypeEntry('VIDEOMAE_v2_G_unl', os.path.join("VideoMAEv2", "pretrained", "vit_g_hybrid_pt_1200e.pth")),
    'VIDEOMAE_v2_S': ModelTypeEntry('VIDEOMAE_v2_S', os.path.join('VideoMAEv2', 'pretrained', "vit_s_k710_dl_from_giant.pth")),
    'VIDEOMAE_v2_B': ModelTypeEntry('VIDEOMAE_v2_B', os.path.join('VideoMAEv2', 'pretrained', "vit_b_k710_dl_from_giant.pth")),
    'VIDEOMAE_v2_G': ModelTypeEntry('VIDEOMAE_v2_G', os.path.join('VideoMAEv2', 'pretrained', "vit_g_hybrid_pt_1200e_k710_ft.pth")),
    # 'VJEPA2_G_384': ModelTypeEntry('VJEPA2_G_384', os.path.join('vjepa2', 'pretrained', 'vitg-384.pt')),
    'vjepa2_L_fpc64_256': ModelTypeEntry('vjepa2_L_fpc64_256', 'facebook/vjepa2-vitl-fpc64-256'),
    'vjepa2_G_fpc64_384': ModelTypeEntry('vjepa2_G_fpc64_384', 'facebook/vjepa2-vitg-fpc64-384'),
    'vjepa2_1_B': ModelTypeEntry('vjepa2_1_B', os.path.join('vjepa2', 'pretrained_pth', 'vjepa2_1_vitb_dist_vitG_384.pt')),
    'ViT_image': ModelTypeEntry('ViT_image', 'ViT_image'),
    'DFER': ModelTypeEntry('DFER', os.path.join("MAE_DFER", "saved", "model", "pretraining", "voxceleb2", "videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170", "checkpoint-49.pth")),
  }

  VIDEOMAE_v2_G_unl = _entries['VIDEOMAE_v2_G_unl']
  VIDEOMAE_v2_S     = _entries['VIDEOMAE_v2_S']
  VIDEOMAE_v2_B     = _entries['VIDEOMAE_v2_B']
  VIDEOMAE_v2_G     = _entries['VIDEOMAE_v2_G']
  VJEPA_v2_L_fpc64_256 = _entries['vjepa2_L_fpc64_256']
  VJEPA_v2_G_fpc64_384 = _entries['vjepa2_G_fpc64_384']
  VJEPA_v2_1_B         = _entries['vjepa2_1_B']
  ViT_image         = _entries['ViT_image']
  DFER             = _entries['DFER']
  
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
      'vjepa2_1_B': 'vjepa2_1_B',
      'ViT_image': 'ViT_image',
      'DFER': 'DFER'
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
      'vjepa2_1_B': 'vjepa2_1_B',
      'ViT_image': 'ViT_image',
      'DFER': 'DFER'
    }
    key = mapping.get(type)
    if key and key in cls._entries:
      return cls._entries[key]
    raise ValueError(f"Model type '{type}' not found. Choose from: {list(mapping.keys())}")

  @classmethod
  def get_embedding_size(cls, typ):
    if 'S' in typ:
      return 384
    elif 'DFER' in typ:
      return 512
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

class EMBEDDING_REDUCTION(Enum):  
  # [B,t,p,p,emb] -> [B,1,p,p,emb] ex: [3,8,14,14,768] -> [3,1,14,14,768]
  MEAN_TEMPORAL = (1,) 
  
  # [B,t,p,p,emb] -> [B,t,1,1,emb] ex: [3,8,14,14,768] -> [3,8,1,1,768]
  MEAN_SPATIAL = (2,3) 
  
  # [B,t,p,p,emb] -> [B,1,1,1,emb] ex: [3,8,14,14,768] -> [3,1,1,1,768]
  MEAN_TEMPORAL_SPATIAL = (1,2,3)
  NONE = None
  
  ADAPTIVE_POOLING_3D = (0,0,0) # [B,emb,t,p,p] -> [B,emb,2,2,2] ex: [3,768,8,14,14] -> [3,768,2,2,2]
  
  # [B,t,p,p,emb] -> [B,t,1,1,emb] with masked positions excluded (face boundaries)
  # Applies masked spatial mean reducing over S,S dimensions while excluding specified positions
  SPATIAL_MASKED = 'spatial_masked'
  
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
    elif pooling_embedding_reduction.lower() == 'spatial_masked':
      return EMBEDDING_REDUCTION.SPATIAL_MASKED
    else:
      raise ValueError(f'Pooling embedding reduction not recognized: {pooling_embedding_reduction}. Can be spatial, temporal, all, adaptive_pooling_3d, spatial_masked or none')

class INSTANCE_MODEL_NAME(Enum): # model.__class__.__name__
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
  # Prefix for relative paths when --gp is passed. Defaults to the repository root;
  # override with the PAIN_PROJECT_ROOT environment variable.
  NAS_PATH = os.environ.get('PAIN_PROJECT_ROOT',
                            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

def generate_csv_augmented(original_csv_path, dict_augmentation, out_csv_path, stratified_training=False):
  """
  Generate an augmented CSV by appending shifted sample IDs for each active augmentation type.

  Args:
    original_csv_path: Path to the original TSV label file.
    dict_augmentation: Dict mapping augmentation name to sampling fraction (0 < p <= 1).
    out_csv_path:      Destination path for the merged TSV file.
    stratified_training: Unused — kept for interface compatibility. Augmentation mode
                         is controlled via dict_augmentation / augment_csv in model.py.

  Returns:
    pd.DataFrame: Merged DataFrame (originals + augmented rows).
  """
  def _get_rnd_from_type(type_augm):
    if type_augm == 'hflip':
      return 42
    elif type_augm == 'jitter':
      return 53
    elif type_augm == 'rotation':
      return 63
    elif type_augm == 'latent_basic':
      return 73
    elif type_augm == 'latent_masking':
      return 83
    elif type_augm == 'shift':
      return 93
    elif type_augm == 'zoom':
      return 103
    elif type_augm == 'reverse':
      return 113
    elif type_augm == 'timelapse':
      return 123
    elif type_augm == 'slowmotion':
      return 133
    elif type_augm == 'gaussian':
      return 143
    else:
      return 21
  list_df = []
  df = pd.read_csv(original_csv_path,sep='\t', dtype={'sample_name':str,'subject_name':str})
  list_df.append(df)
  for type_augm, p in dict_augmentation.items():
    if p > 0 and p<= 1:
      df_sampled = df.sample(frac=p, random_state=_get_rnd_from_type(type_augm))
      df_sampled['sample_id'] = df_sampled['sample_id'].apply(lambda x: x + get_shift_for_sample_id(type_augm))
      list_df.append(df_sampled)
  df_merged = pd.concat(list_df, ignore_index=True)
  os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
  df_merged.to_csv(out_csv_path, index=False, sep='\t')
  print(f'CSV file with augmentations saved to {out_csv_path}')
  return df_merged


# Options of older releases, with the value they had in every paper run. Saved run configs
# (k_fold_results.pkl: model_advanced_params, head_params, config) still carry them.
REMOVED_OPTION_DEFAULTS = {
  # Model_Advanced
  'adapter_dict': None, 'concatenate_temporal': 0, 'num_clips_per_video': 1,
  'sample_frame_strategy': 'sliding_window', 'soft_labels': 0.0, 'stride_inside_window': 1,
  'use_sdpa': 1, 'gaussian_sigma_min': 0.5, 'gaussian_sigma_max': 2.0, 'gaussian_kernel_size': 5,
  # head_params
  'adversarial_head': 0, 'multitask_head': 0, 'multitask_num_classes': None, 'coral_loss': False,
  'drop_path_mode': 'row', 'head_init_path': None, 'num_queries': 1, 'agg_method': 'mean',
  'remove_head': False, 'train_backbone': 0, 'type_head': 0, 'unfreeze_layers': None,
  'use_batch_norm': 0,
  # run config
  'HSIC_subject_lambda': 0.0, 'HSIC_pain_lambda': 0.0, 'HSIC_feat_normalization': 0,
  'adversarial_loss_lambda': 0.0, 'adv_alpha_strategy': None, 'adv_alpha_gamma': 0.0,
  'lambda_center_loss': 0.0, 'ratio_lr_center_loss': 1.0, 'lambda_multitask': 0.0,
  'balance_batches': 0, 'clip_grad_norm': None, 'concatenate_temp_dim': 0,
  'consider_only_lasts_n_chunks': 0, 'debug_grad_flow': 0, 'debug_grad_flow_batches': 1,
  'feature_merge_type': None, 'feature_merge_lambda': None, 'feature_merge_orig': 1,
  'gaussian': 0.0, 'hflip': 0.0, 'jitter': 0.0, 'rotation': 0.0, 'shift': 0.0,
  'latent_basic': 0.0, 'latent_masking': 0.0, 'latent_coefficient': 0.0,
  'load_idxs_splits': 0, 'min_subjects_per_level': 0, 'only_augments': None,
  'perfect_bal_strategy': None, 'sampler_augmented_only_types': None, 'type_group': 0,
  'xattn_mask': None, 'round_output_loss': 0, 'add_CCC_loss': 0.0, 'CCC_loss': None,
}


def _is_default(value, default):
  if isinstance(value, list) and len(value) == 1:
    value = value[0]
  if default is None:
    return value is None
  value = getattr(value, 'value', value)  # enums are compared by value
  try:
    return bool(value == default)
  except (TypeError, ValueError, RuntimeError):
    return False


def drop_removed_options(params, where):
  """Return params without the removed options; raise if one of them holds a non-default value."""
  for key, value in params.items():
    if key in REMOVED_OPTION_DEFAULTS and not _is_default(value, REMOVED_OPTION_DEFAULTS[key]):
      raise ValueError(f"{where}: {key}={value!r} is not supported in this release "
                       f"(only the default {REMOVED_OPTION_DEFAULTS[key]!r}); use an older release of the code.")
  return {key: value for key, value in params.items() if key not in REMOVED_OPTION_DEFAULTS}

import warnings
import re
import torch
import pandas as pd
import os
import math
import numpy as np
import torch
import cv2
import time
import torch.nn as nn
import torch.utils
from custom.backbone import BackboneBase
from custom.helper import CUSTOM_DATASET_TYPE, SAMPLE_FRAME_STRATEGY, EMBEDDING_REDUCTION
import custom.helper as helper
import custom.tools as tools
from sklearn.model_selection import StratifiedKFold
# from torch.utils.data import  Sampler
from torchvision.transforms import v2
from concurrent.futures import ThreadPoolExecutor
# import torchvision.transforms.functional as F
# import torchvision.transforms as T
# import custom.faceExtractor as extractor
import pickle
from torch.utils.data import DataLoader
from custom.helper import INSTANCE_MODEL_NAME,get_shift_for_sample_id
import custom.helper as helper
from pathlib import Path 
from torchvision import tv_tensors
from torch.utils.data import BatchSampler
from making_better_mistakes.better_mistakes.model import labels as soft_labels_generator
from coral_pytorch.dataset import levels_from_labelbatch
import multiprocessing as mp


class customDataset(torch.utils.data.Dataset):
  """
  A dataset class for handling video data with various frame sampling strategies and preprocessing options.
  
  This class supports multiple sampling strategies (uniform, sliding window, central, random),
  and preprocessing operations like face alignment, frontalization, and detection-based cropping.
  """
  def __init__(
      self,
      path_dataset,
      sample_frame_strategy,
      model_type,
      path_labels=None,
      stride_window=16,
      clip_length=16,
      stride_inside_window=1,
      preprocess_align=False,
      preprocess_frontalize=False,
      preprocess_crop_detection=False,
      saving_folder_path_extracted_video=None,
      video_labels=None,
      h_flip=False,
      color_jitter=None,
      spatial_shift=False,
      rotation=None,
      backbone_dict=None,
      image_resize_w=224,
      image_resize_h=224,
      smooth_labels= 0.0,
      num_clips_per_video=1, # only for random sampling strategy
      coral_loss=None,
      soft_labels=None,
      video_extension = '.mp4',
      shift_frame_idx = 0,
      quadrant=None,
      consider_only_lasts_n_chunks=None,
      gaussian_sigma_min=None,
      gaussian_sigma_max=None,
      gaussian_kernel_size=None,
      zoom=False,
      gaussian_smooth=False,
      framepermute=False,
      **kwargs
  ):
    """
    Initialize the dataset with specified parameters.
    
    Args:
        path_dataset (str): Path to the directory containing video data
        path_labels (str, optional): Path to CSV file with video labels
        sample_frame_strategy (str): Strategy for sampling frames. Must be one of SAMPLE_FRAME_STRATEGY
        stride_window (int): Step size between consecutive clips
        clip_length (int): Number of frames in each clip
        stride_inside_window (int): Step size between frames within a clip
        preprocess_align (bool): Whether to align faces in frames
        preprocess_frontalize (bool): Whether to frontalize faces in frames
        preprocess_crop_detection (bool): Whether to crop faces using detection
        saving_folder_path_extracted_video (str, optional): Path to save extracted video frames
        video_labels (pd.DataFrame, optional): DataFrame containing video labels
    """
    # Validate inputs
    if backbone_dict != None and not isinstance(backbone_dict['backbone'], BackboneBase):
      instances = [BackboneBase,INSTANCE_MODEL_NAME,bool, nn.Module]
      for key in ['backbone', 'instance_model_name', 'concatenate_temporal', 'model']:
        if key not in backbone_dict:
          assert key in backbone_dict, f"backbone_dict must contain key: {key}"
        if not isinstance(backbone_dict[key], instances.pop(0)):
          raise ValueError(f"backbone_dict['{key}'] must be an instance of {instances[0]}")
        
    if video_labels is not None:
      assert isinstance(video_labels, pd.DataFrame), "video_labels must be a pandas DataFrame."
      
    assert os.path.exists(path_dataset), f"Dataset path {path_dataset} does not exist."
    assert clip_length > 0, "Clip length must be greater than 0."
    assert stride_window > 0, "Stride window must be greater than 0."
    assert sample_frame_strategy in SAMPLE_FRAME_STRATEGY, f"Sample frame strategy must be one of {SAMPLE_FRAME_STRATEGY}."
    
    # Initialize instance variables
    self.video_extension = video_extension
    self.path_dataset = path_dataset
    self.path_labels = path_labels
    self.type_sample_frame_strategy = sample_frame_strategy
    self.clip_length = clip_length
    self.stride_window = stride_window
    self.stride_inside_window = stride_inside_window
    self.saving_folder_path_extracted_video = saving_folder_path_extracted_video
    self.backbone_dict = backbone_dict
    self.model_type = model_type
    
    # Preprocessing options
    self.preprocess_align = preprocess_align
    self.preprocess_frontalize = preprocess_frontalize
    self.preprocess_crop_detection = preprocess_crop_detection
    self.h_flip = h_flip
    self.color_jitter = color_jitter
    self.rotation = rotation
    self.shift_frame_idx = shift_frame_idx 
    self.spatial_shift = spatial_shift
    self.smooth_labels = smooth_labels
    self.num_clips_per_video = num_clips_per_video
    self.quadrant = quadrant
    self.gaussian_sigma_min = gaussian_sigma_min
    self.gaussian_sigma_max = gaussian_sigma_max
    self.gaussian_kernel_size = gaussian_kernel_size
    self.zoom = zoom
    self.gaussian_smooth = gaussian_smooth
    self.framepermute = framepermute

    # if rotation is not None:
    #   warnings.warn('The rotation is not implemented yet')
    # Image dimensions and channels
    if backbone_dict is not None:
      self.image_resize_h = backbone_dict['backbone'].img_size
      self.image_resize_w = backbone_dict['backbone'].img_size
    else:
      self.image_resize_w = image_resize_w
      self.image_resize_h = image_resize_h
    self.image_channels = 3
    self.consider_only_lasts_n_chunks = consider_only_lasts_n_chunks
    # Set sampling strategy method based on parameter
    self._set_sampling_strategy(sample_frame_strategy)

    # if 'caer' in path_dataset.lower():
    #   print("\n Detected CAER, Change step shift \n")
    #   helper.step_shift = 13176
    # Set video labels
    if video_labels is None and path_labels != None:
      self.set_path_labels(path_labels)
    else:
      self.video_labels = video_labels[helper.desired_order_csv]
        
    # Get unique subjects and classes
    tmp = tools.get_unique_subjects_and_classes(self.video_labels)
    self.total_subjects, self.total_classes = len(tmp[0]), len(tmp[1])
    
    # Initialize face processing components
    self.face_extractor = None # TODO: None for testing more workers during training, REAL => extractor.FaceExtractor()
    self._load_reference_landmarks()
    
    if soft_labels:
      classes = list(range(self.num_classes))
      distances = {(i, j): abs(i - j) for i in classes for j in classes}
      self.soft_labels = soft_labels_generator.make_all_soft_labels(
        classes=classes,
        distances=distances,
        hardness=soft_labels
      )
    else:
      self.soft_labels = None
    helper.set_step_shift(path_dataset)
    print(f"\ncustomDataset: Using step shift = {helper.step_shift} for dataset {path_dataset}\n")
    self.coral_loss = coral_loss
    self.num_classes = len(self.get_unique_classes())
  
  def _set_sampling_strategy(self, strategy):
    """
    Set the frame sampling strategy based on the specified parameter.
    
    Args:
        strategy (str): The sampling strategy to use
    """
    if self.num_clips_per_video is not None and self.num_clips_per_video != 1 and strategy != SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING:
      raise ValueError(
        f"num_clips_per_video must be None for sampling strategy {strategy}. "
        f"Use {SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING} for multiple clips per video."
      )
    if strategy == SAMPLE_FRAME_STRATEGY.UNIFORM:
      self.sample_frame_strategy = self._single_uniform_sampling
      warnings.warn(f"The {SAMPLE_FRAME_STRATEGY.UNIFORM} sampling strategy does not take into account the stride window.")
        
    elif strategy == SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW:
      self.sample_frame_strategy = self._sliding_window_sampling
        
    elif strategy == SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING:
      self.sample_frame_strategy = self._central_sampling
        
    elif strategy == SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING:
      self.sample_frame_strategy = self._random_sampling
      # warnings.warn(f"The {SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING} sampling strategy does not take into account the stride window.")
  
  def _load_reference_landmarks(self):
    """Load reference facial landmarks for face frontalization"""
    landmarks_path = os.path.join('partA', 'video', 'mean_face_landmarks_per_subject', 'all_subjects_mean_landmarks.pkl')
    landmarks_data = pickle.load(open(landmarks_path, 'rb'))
    self.reference_landmarks = landmarks_data['mean_facial_landmarks']

  def set_path_labels(self, path):
    """
    Sets the path to the labels file and loads the video labels from a CSV file.

    Args:
        path (str): The file path to the CSV file containing the video labels.
    """
    self.path_labels = path
    self.video_labels = pd.read_csv(self.path_labels,sep='\t',dtype={'sample_name':str,'subject_name':str})
    self.df = self.video_labels[helper.desired_order_csv]
    print(f'Set path_labels: {self.path_labels}')
  
  def __len__(self):
    """Return the number of samples in the dataset"""
    return len(self.video_labels)

  @staticmethod
  def apply_frame_permutation(tensors, seed=None):
    """Randomly permute frame order for a tensor of shape [B, C, H, W]."""
    if tensors.shape[0] <= 1:
      return tensors, None
    rng = np.random.default_rng(seed)
    frame_permutation = rng.permutation(tensors.shape[0])
    return tensors[frame_permutation], frame_permutation
  
  @staticmethod
  def preprocess_images(tensors,crop_size=None,to_visualize=False,get_params=False,h_flip=False,color_jitter=False,rotation=False,spatial_shift=False,zoom=False,gaussian_smooth=False,gaussian_sigma_min=None,gaussian_sigma_max=None,gaussian_kernel_size=None,framepermute=False,framepermute_seed=None):
    """
    Preprocess a batch of image tensors.

    Args:
      tensors (torch.Tensor):          Input tensor. Shape: (B, C, H, W) where B=batch, C=channels, H=height, W=width.
      crop_size (tuple, optional):      Target (H, W) after resize. Defaults to (224, 224) when to_visualize is False.
      to_visualize (bool):              If True, skip normalization and rescaling.
      get_params (bool):                If True, return (transformed_tensors, params_dict).
      h_flip (bool):                    Apply horizontal flip.
      color_jitter (bool):              Apply color jitter.
      rotation (bool):                  Apply random rotation.
      spatial_shift (bool):             Apply random spatial shift.
      zoom (bool):                      Apply random zoom (resize crop).
      gaussian_smooth (bool):           Apply 2D Gaussian smoothing. Same sigma for all frames.
      gaussian_sigma_min (float):       Min sigma for uniform sampling. Required if gaussian_smooth=True.
      gaussian_sigma_max (float):       Max sigma for uniform sampling. Required if gaussian_smooth=True.
      gaussian_kernel_size (int):       Fixed kernel size (must be odd). Required if gaussian_smooth=True.
      framepermute (bool):              Randomly permute frame order.
      framepermute_seed (int, optional): Seed for frame permutation reproducibility.

    Returns:
      torch.Tensor: Preprocessed tensor. Shape: (B, C, crop_H, crop_W).
      dict (optional): Augmentation parameters, returned only when get_params=True.
    """
    if framepermute:
      tensors, frame_permutation = customDataset.apply_frame_permutation(tensors, seed=framepermute_seed)
    else:
      frame_permutation = None
    tensors = tv_tensors.Video(tensors)
    # Define preprocessing constants
    # crop_size = (224, 224)
    if not to_visualize and crop_size is None:
      crop_size = (224, 224)
    rescale_factor = 1/255  
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    transform = []
    params = {}
    
    if h_flip:
      transform.append(v2.RandomHorizontalFlip(p=1))
      params['h_flip'] = True
    
    if spatial_shift:
      transform.append(v2.RandomAffine(degrees=0, translate=(0.03, 0.03))) # 0.03 = 3% shift (256x256 -> 7.68 pixels)
      angle, translation,_,_ = v2.RandomAffine.get_params(degrees=transform[-1].degrees, 
                                                      translate=transform[-1].translate,
                                                      scale_ranges=None,
                                                      img_size=(tensors.shape[2], tensors.shape[3]),
                                                      shears=None)
      params['spatial_shift'] = {'translate': translation}

    if color_jitter:
      brightness = (0.8, 1.2)
      contrast = (0.8, 1.2)
      saturation = (0.8, 1.2)
      hue = (-0.05, 0.05)
      transform.append(v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)) 
      params['color_jitter'] = {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'hue': hue
      }
    if zoom:
      zoom_range_min = 0.85
      zoom_range_max = 1.15
      zoom_val = np.random.choice([np.random.uniform(zoom_range_min, 0.95),np.random.uniform(1.05, zoom_range_max)])
      zoom_range = (zoom_val, zoom_val)
      if crop_size is None:
        crop_size = (tensors.shape[2], tensors.shape[3])
      transform.append(v2.RandomResizedCrop(size=crop_size, scale=zoom_range))
      params['zoom'] = v2.RandomResizedCrop.get_params(
        img=tensors,
        scale=transform[-1].scale,
        ratio=transform[-1].ratio
      )
      
    if rotation:
      pos_degrees = (0.5, 4)
      neg_degrees = (-4, -0.5)
      # avoid 0 degree rotation
      angle = np.random.choice([np.random.uniform(pos_degrees[0], pos_degrees[1]),
                               np.random.uniform(neg_degrees[0], neg_degrees[1])])
      transform.append(v2.RandomRotation(degrees=(angle, angle)))
      params['rotation'] = v2.RandomRotation.get_params(degrees=transform[-1].degrees)

    if gaussian_smooth:
      if gaussian_sigma_min is None or gaussian_sigma_max is None or gaussian_kernel_size is None:
        raise ValueError("gaussian_sigma_min, gaussian_sigma_max, and gaussian_kernel_size must be provided when gaussian_smooth is True")
      if gaussian_kernel_size % 2 == 0:
        raise ValueError("gaussian_kernel_size must be odd")
      sampled_sigma = float(np.random.uniform(gaussian_sigma_min, gaussian_sigma_max))
      transform.append(v2.GaussianBlur(kernel_size=gaussian_kernel_size, sigma=sampled_sigma))
      params['gaussian'] = {
        'sigma': sampled_sigma,
        'kernel_size': gaussian_kernel_size
      }

    if framepermute:
      params['framepermute'] = {
        'seed': framepermute_seed,
        'indices': frame_permutation.tolist() if frame_permutation is not None else None
      }
      
    # Define transform pipeline
    if not to_visualize:
      transform += [
        v2.Resize(crop_size),
        v2.Lambda(lambda x: x * rescale_factor),  # Rescale (1/255)
        v2.Normalize(mean=image_mean, std=image_std),  # Normalize
      ]
    else:
      if crop_size is not None:
        transform += [v2.Resize(crop_size)]
        
    if len(transform) == 0:
      return tensors, params if get_params else tensors
    transform = v2.Compose(transform)
    
    if get_params:
      return transform(tensors),params
    else:
      return transform(tensors)
  

  def generate_video(self, sample_id,output_folder=None,fps_out=24):
    """
    Generate a video from the frames at the specified index.
    
    Args:
        idx (int): Index of the sample to generate a video for.
        
    Returns:
        None
    """
    idx = self.video_labels[self.video_labels['sample_id'] == sample_id].index[0]
    if idx is None:
      raise ValueError(f"Sample ID {sample_id} not found in video labels {self.path_labels}.")
    subject_name = self.video_labels.iloc[idx]['subject_name'] # csv_array.iloc[1]
    sample_name = self.video_labels.iloc[idx]['sample_name'] # csv_array.iloc[5]
    # csv_array = self.video_labels.iloc[idx]
    video_path = os.path.join(self.path_dataset, subject_name, sample_name + self.video_extension)
    container = cv2.VideoCapture(video_path)
    if not container.isOpened():
      raise ValueError(f"Video {video_path} cannot be opened. Please check the video file.")
    tot_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set frame dimensions based on preprocessing requirements
    width_frames = int(container.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_frames = int(container.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # if tot_frames == 0:
    list_indices = self.sample_frame_strategy(tot_frames)
    frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames) # [nr_clips, nr_frames, H, W, C]
    # original_shape = frames_list.shape
    frames_list = frames_list.reshape(-1,*frames_list.shape[2:]).permute(0,3,1,2) # [nr_clips * nr_frames, H, W, C] -> [B,C,H,W]
    frames_list,params = self.preprocess_images(frames_list,
                                                crop_size=(self.image_resize_h, self.image_resize_w),
                                                to_visualize=True,
                                                get_params=True,
                                                color_jitter=self.color_jitter,
                                                h_flip=self.h_flip,
                                                rotation=self.rotation,
                                                spatial_shift=self.spatial_shift) # [B,C,H,W]
    
    # Save the generated video
    frames_list = frames_list.permute(0,2,3,1) # [B,C,H,W] -> [B,H,W,C]
    if output_folder is None:
      return {'params': params, 'list_frames': frames_list, 'list_indices': list_indices}
    else:
      os.makedirs(output_folder, exist_ok=True)
      output_path = os.path.join(output_folder, f'{sample_name}_inside_{self.stride_inside_window}_stride_{self.stride_window}_fps_{fps_out}{self.video_extension}')
      tools.generate_video_from_list_frame(list_frame=frames_list,path_video_output=output_path,fps=fps_out)
      return params   
  
  def __standard_getitem__(self, idx, is_training):
    
    subject_name = self.video_labels.iloc[idx]['subject_name']
    sample_name = self.video_labels.iloc[idx]['sample_name']
    if self.quadrant is not None:
      video_path = os.path.join(self.path_dataset, subject_name, f'{sample_name}${self.quadrant}{self.video_extension}')
    else:
      video_path = os.path.join(self.path_dataset, subject_name, f"{sample_name}{self.video_extension}")

    # Open video and get properties
    pid = os.getpid()
    t_video_open = time.perf_counter()
    container = cv2.VideoCapture(video_path)
    tot_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set frame dimensions based on preprocessing requirements
    width_frames = int(container.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_frames = int(container.get(cv2.CAP_PROP_FRAME_HEIGHT))
    helper.time_profile_dict[f'{pid}_video_open_time'] = helper.time_profile_dict.get(f'{pid}_video_open_time', 0) + (time.perf_counter() - t_video_open)

    if tot_frames == 0:
      raise ValueError(f"Video {video_path} has no frames. Please check the video file.")
    # Get frame indices based on sampling strategy
    t_frame_sampling = time.perf_counter()
    try:
      list_indices = self.sample_frame_strategy(tot_frames)
    except ValueError as e:
      print(f"Error in sampling strategy for video {video_path}: {e}")
      raise e
    helper.time_profile_dict[f'{pid}_video_frame_sampling_time'] = helper.time_profile_dict.get(f'{pid}_video_frame_sampling_time', 0) + (time.perf_counter() - t_frame_sampling)
    # Read and process frames
    time_read_video = time.perf_counter()
    frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames)
    container.release()
    helper.time_profile_dict[f'{pid}_read_video_time'] = helper.time_profile_dict.get(f'{pid}_read_video_time', 0) + (time.perf_counter() - time_read_video)

    # Reshape frames for preprocessing
    nr_clips, nr_frames = frames_list.shape[:2]
    
    # Preprocess frames (considering only 1 video at time)
    # [nr_clips, nr_frames, H, W, C] -> [nr_clips * nr_frames, H, W, C] -> [B,C,H,W]
    frames_list = frames_list.reshape(-1, *frames_list.shape[2:]).permute(0, 3, 1, 2)
    time_preprocess_video = time.perf_counter()
    # Apply augmentation based on training status
    if not is_training:
      preprocessed_tensors = self.preprocess_images(frames_list,
                                                    crop_size=(self.image_resize_h, self.image_resize_w),
                                                    color_jitter=self.color_jitter,
                                                    h_flip=self.h_flip,
                                                    rotation=self.rotation,
                                                    spatial_shift=self.spatial_shift,
                                                    zoom=self.zoom,
                                                    gaussian_smooth=self.gaussian_smooth,
                                                    gaussian_sigma_min=self.gaussian_sigma_min,
                                                    gaussian_sigma_max=self.gaussian_sigma_max,
                                                    gaussian_kernel_size=self.gaussian_kernel_size,
                                                    framepermute=self.framepermute) # [B,C,H,W]
    else:
      # latent based augm. approaches are applied in _get_element function (at the end of file)
      t_augm_setup = time.perf_counter()
      sample_id = self.video_labels.iloc[idx]['sample_id'] # csv_array.iloc[4]
      augmentation_type = helper.get_augmentation_type(sample_id)
      h_flip = True if augmentation_type == 'h_flip' else False
      color_jitter = True if augmentation_type == 'jitter' else False
      rotation = True if augmentation_type == 'rotation' else False
      spatial_shift = True if augmentation_type == 'shift' else False
      zoom = True if augmentation_type == 'zoom' else False
      gaussian_smooth = True if augmentation_type == 'gaussian' else False
      helper.time_profile_dict[f'{pid}_preprocess_augm_setup_time'] = helper.time_profile_dict.get(f'{pid}_preprocess_augm_setup_time', 0) + (time.perf_counter() - t_augm_setup)
      preprocessed_tensors = self.preprocess_images(frames_list,
                                                    crop_size=(self.image_resize_h, self.image_resize_w),
                                                    color_jitter=color_jitter,
                                                    h_flip=h_flip,zoom=zoom,
                                                    spatial_shift=spatial_shift,
                                                    rotation=rotation,
                                                    gaussian_smooth=gaussian_smooth,
                                                    gaussian_sigma_min=self.gaussian_sigma_min,
                                                    gaussian_sigma_max=self.gaussian_sigma_max,
                                                    gaussian_kernel_size=self.gaussian_kernel_size)
    preprocessed_tensors = preprocessed_tensors.reshape(nr_clips, nr_frames, *preprocessed_tensors.shape[1:]) # [nr_clips, nr_frames, C, H, W]
    # if nr_clips == 1:
    #   preprocessed_tensors = preprocessed_tensors.unsqueeze(0) # in case of single frame
    preprocessed_tensors = preprocessed_tensors.permute(0,2,1,3,4) # [B=nr_clips, T=nr_frames, C, H, W] -> [B, C, T, H, W]
    helper.time_profile_dict[f'{pid}_preprocess_video_time'] = helper.time_profile_dict.get(f'{pid}_preprocess_video_time', 0) + (time.perf_counter() - time_preprocess_video)
    
    # Create metadata tensors
    time_metadata = time.perf_counter()
    sample_id = torch.full((nr_clips,), int(self.video_labels.iloc[idx]['sample_id']), dtype=torch.int32)
    labels = torch.full((nr_clips,), int(self.video_labels.iloc[idx]['class_id']), dtype=torch.int32)
    subject_id = torch.full((nr_clips,), int(self.video_labels.iloc[idx]['subject_id']), dtype=torch.int32)
    path = np.repeat(video_path, nr_clips)
    helper.time_profile_dict[f'{pid}_metadata_time'] = helper.time_profile_dict.get(f'{pid}_metadata_time', 0) + (time.perf_counter() - time_metadata)
    return {
        'preprocess': preprocessed_tensors, # [B,C,T,H,W]
        'labels': labels,
        'subject_id': subject_id,
        'sample_id': sample_id,
        'path': path,
        'frame_list': list_indices
      }
    
  def __getitem__(self, idx): # idx can be int or list
    """
    Retrieve a sample from the dataset at the given index.
    
    Args:
        idx (int): Index of the sample to retrieve.
        
    Returns:
        dict: A dictionary containing preprocessed video data and metadata.
    """
    if isinstance(idx, int):
      if self.backbone_dict is None:
        # extract_feature.py path
        return self.__standard_getitem__(idx, is_training=False)
      else:
        # Training path
        dict_data = self.__standard_getitem__(idx,is_training=True)
        # features = self.backbone_dict['backbone'].forward_features(dict_data['preprocess'])
        
        # # Apply pooling if specified in backbone_dict
        # if self.backbone_dict['embedding_reduction'] is not EMBEDDING_REDUCTION.NONE:
        #   features = torch.mean(features,dim=self.backbone_dict['embedding_reduction'].value,keepdim=True)
        # dict_data = {
        #   'features': features,
        #   'list_labels': dict_data['labels'],
        #   'list_sample_id': dict_data['sample_id'],
        #   'list_subject_id': dict_data['subject_id'],
        # }
        return {
              'features': dict_data['preprocess'],      # [B,C,frame_length,H,W]
              'labels': dict_data['labels'],            # [8]
              'subject_id': dict_data['subject_id'],    # [8]
              'sample_id': dict_data['sample_id'][0]    # int
          }
    else: # list case
      raise NotImplementedError("Batch processing is not implemented yet in this dataset class.")
      if self.backbone_dict is None:
        batch_preprocess = [self.__standard_getitem__(i) for i in idx]
        return self._custom_collate_fn_extraction(batch_preprocess)
      else:
        batch_preprocess = [self.__standard_getitem__(i) for i in idx]
        # batch_features = torch.cat([item['preprocess'] for item in batch_preprocess], dim=0) # For batch processing in backbone
        # batch_features = self.backbone_dict['backbone'].forward_features(batch_features) # as above
        batch_features = torch.cat([self.backbone_dict['backbone'].forward_features(item['preprocess']) for item in batch_preprocess], dim=0)
        batch = []
        for i, item in enumerate(batch_preprocess):
          batch.append({
            'features': batch_features[i],
            'list_labels': item['labels'],
            'list_sample_id': item['sample_id'],
            'list_subject_id': item['subject_id']
          })
        batch = [_get_element(dict_data=dict_data,df=self.video_labels,idx=i) for i,dict_data in zip(idx, batch)]
        return self._custom_collate(batch)
      #      return {
      #     'features': features,     # [8, 8, 1, 1, 768]-> [seq_len,Temporal,Space,Space,Emb]
      #     'labels': labels,         # [8]
      #     'subject_id': subject_id, # [8]
      #     'sample_id': sample_id    # int
      # }
  
  def _read_video_cv2_and_process(self, container, list_indices, width_frames, height_frames):
    """
    Read and process video frames based on provided indices.
    
    Args:
        container: OpenCV video capture object
        list_indices: Tensor of frame indices to extract
        width_frames: Width to resize frames to
        height_frames: Height to resize frames to
        
    Returns:
        torch.Tensor: Processed video frames
    """
    # Get start and end frames for each clip
    # start_frame_idx = list_indices[:, 0]
    # end_frame_idx = list_indices[:, -1]
    num_clips, clip_length = list_indices.shape
    
    # Initialize tensors to store extracted frames
    extracted_frames = torch.zeros(
      num_clips, clip_length, height_frames, width_frames, self.image_channels, 
      dtype=torch.uint8
    )
    pos = torch.zeros(num_clips, dtype=torch.int32)
    
    # Find max end frame to optimize loop
    max_end_frame = list_indices[:, -1].max().item()
    
    # Read frames from video
    frame_idx = 0
    # t_decode_total = 0.0
    # t_cvt_total = 0.0
    # t_copy_total = 0.0
    while container.isOpened():
      # t_decode = time.perf_counter()
      ret, frame = container.read()
      # t_decode_total += time.perf_counter() - t_decode
      if not ret or frame_idx > max_end_frame:
        break

      # Convert color format
      # t_cvt = time.perf_counter()
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # t_cvt_total += time.perf_counter() - t_cvt

      # Apply preprocessing if needed
      if self.preprocess_align or self.preprocess_frontalize or self.preprocess_crop_detection:
        raise NotImplementedError("Face preprocessing is not implemented yet.")

      # Check if this frame should be included in any clip
      # t_copy = time.perf_counter()
      mask = np.any(np.isin(list_indices, frame_idx),axis=1)
      if mask.any():
        frame_rgb = torch.tensor(frame, dtype=torch.uint8)
        extracted_frames[mask, pos[mask].long()] = frame_rgb
        pos[mask] += 1
      # t_copy_total += time.perf_counter() - t_copy

      frame_idx += 1

    pid = os.getpid()
    # helper.time_profile_dict[f'{pid}_video_frame_decode_time'] = helper.time_profile_dict.get(f'{pid}_video_frame_decode_time', 0) + t_decode_total
    # helper.time_profile_dict[f'{pid}_video_color_convert_time'] = helper.time_profile_dict.get(f'{pid}_video_color_convert_time', 0) + t_cvt_total
    # helper.time_profile_dict[f'{pid}_video_frame_select_copy_time'] = helper.time_profile_dict.get(f'{pid}_video_frame_select_copy_time', 0) + t_copy_total

    return extracted_frames # [nr_clips, nr_frames, H, W, C]

  def _custom_collate_fn_extraction(self, batch):
    """
    Custom collate function to process and stack batch data for a DataLoader.

    Args:
        batch (list): A list of dictionaries containing sample data
        
    Returns:
        tuple: A tuple containing processed batch data
    """
    # Combine preprocessed tensors from all samples
    data = torch.cat([item['preprocess'] for item in batch], dim=0) # [chunks, C, clip_length, H, W]
    if 'vjepa2' in self.model_type.value.lower():
      data = data.permute(0,2,1,3,4) # [chunks, clip_length, C, H, W]
    # Combine metadata from all samples
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    path = np.concatenate([item['path'] for item in batch])
    subject_id = torch.cat([item['subject_id'] for item in batch], dim=0)
    sample_id = torch.cat([item['sample_id'] for item in batch], dim=0)
    list_frames = torch.cat([item['frame_list'] for item in batch], dim=0).squeeze()

    return data, labels, subject_id, sample_id, path, list_frames

  
  def _custom_collate(self, batch):
    time_custom_collate = time.perf_counter()
    labels = torch.stack([item['labels'][0] for item in batch], dim=0)
    subject_id = torch.stack([item['subject_id'][0] for item in batch], dim=0)
    sample_id = torch.stack([item['sample_id'] for item in batch], dim=0)
    data = torch.cat([item['features'] for item in batch], dim=0)
    lengths = torch.tensor([item['features'].shape[0] for item in batch]) 
    # features = torch.nn.utils.rnn.pad_sequence(data,batch_first=True) # [batch_size,seq_len,emb_dim]
    # lengths_tensor = torch.tensor(lengths, dtype=torch.int32)
    # max_len = max(lengths)
    # key_padding_mask = torch.arange(max_len).expand(len(batch), max_len) >= lengths_tensor.unsqueeze(1)
    # key_padding_mask = ~key_padding_mask # set True for attention, if True means use the token to compute the attention, otherwise don't use it 
      
    if self.smooth_labels > 0.0: # and model.output_size > 1:
      labels = smooth_labels_batch(gt_classes=labels, num_classes=self.num_classes, smoothing=self.smooth_labels)
    elif self.soft_labels != None:
      labels = self.soft_labels[labels].to(torch.float32) # soft labels
    elif self.coral_loss:
      labels = levels_from_labelbatch(labels, self.num_classes, dtype=torch.float32)
    
    pid = os.getpid()
    helper.time_profile_dict[f'{pid}_custom_collate_time'] = helper.time_profile_dict.get(f'{pid}_custom_collate_time', 0) + (time.perf_counter() - time_custom_collate)
    # [B, C, T, H, W], padding applied in the model forward pass
    return {'x':data, 
           'key_padding_mask': None,
           'lengths': lengths, 
           }, \
            labels, \
            subject_id, \
            sample_id, \

  
  def _single_uniform_sampling(self, video_len):
    """
    Generate indices for uniform sampling of frames from a video.
    
    Args:
        video_len (int): Length of the video in frames
        
    Returns:
        torch.Tensor: Frame indices
    """
    indices = np.linspace(0, video_len-1, self.clip_length, dtype=int)
    return torch.from_numpy(indices)[None, :]
  
  def get_all_sample_ids(self):
    """
    Returns all sample_id values from path_labels.

    Returns:
        list: A list of all sample_id values.
    """
    
    # csv_array = self.video_labels.to_numpy()
    # list_samples = [entry[0].split("\t") for entry in csv_array]
    # list_samples = np.stack(list_samples)
    # sample_ids = list_samples[:, 4].astype(int)
    sample_ids = self.video_labels['sample_id'].tolist()
    return np.array(sample_ids).astype(int)
  
  def _central_sampling(self, video_len):
    """
    Generate indices for central sampling of frames from a video.
    
    Args:
        video_len (int): Length of the video in frames
        
    Returns:
        torch.Tensor: Frame indices
    """
    required_length = self.clip_length * self.stride_inside_window
    half_length = required_length // 2
    
    assert video_len // 2 >= half_length, (
      f"Video is too short for the given clip length and stride. "
      f"Video length: {video_len}, required half length: {half_length}"
    )
    
    start_idx = video_len // 2 - half_length
    indices = torch.arange(
      start_idx, 
      start_idx + required_length, 
      self.stride_inside_window
    )[None, :]
    
    return indices
  
  def _random_sampling(self, video_len):
    """
    Generate indices for random sampling of frames from a video.
    
    Args:
        video_len (int): Length of the video in frames
        
    Returns:
        torch.Tensor: Frame indices
    """
    valid_starts = np.arange(0, video_len - self.clip_length * self.stride_inside_window + 1)
    if valid_starts.size == 0:
      raise ValueError(
        f"Video length {video_len} is too short for the specified clip length {self.clip_length} "
        f"and stride inside window {self.stride_inside_window}. "
      )
    try:
      start_idxs = np.random.choice(valid_starts, size=self.num_clips_per_video, replace=False)
    except ValueError as e:
      start_idxs = np.random.choice(valid_starts, size=self.num_clips_per_video, replace=True) # If not enough unique starts, allow replacement
    start_idxs = np.sort(start_idxs.astype(int))
    indices = np.array([np.arange(start_idx, start_idx + self.clip_length * self.stride_inside_window, self.stride_inside_window) for start_idx in start_idxs])
    indices = torch.tensor(indices, dtype=torch.int32)
    # indices = torch.randperm(video_len, dtype=torch.int16)[:self.clip_length]
    return indices  # [num_clips, clip_length]
  
  def _sliding_window_sampling(self, video_len):
    """
    Generates a list of indices for sliding window sampling of a video.

    Args:
      video_len (int): The total length of the video in frames.
      stride_inside_window (int): The stride inside each window.

    Returns:
      torch.Tensor: A tensor containing the indices for each sliding window.
              Each row corresponds to a window and contains the indices
              of the frames within that window.
    """
    upper_bound = video_len - self.clip_length * self.stride_inside_window + 1
    if upper_bound <= 0:
      raise ValueError(
        f"Video length {video_len} is too short for the specified clip length {self.clip_length} "
        f"and stride inside window {self.stride_inside_window}. "
        f"Please adjust these parameters."
      )
    # indices = torch.arange(0, video_len - self.clip_length * self.stride_inside_window + 1, self.stride_window)
    indices = torch.arange(0, upper_bound - self.shift_frame_idx, self.stride_window)
    list_indices = torch.stack([torch.arange(start_idx, start_idx + self.clip_length * self.stride_inside_window, self.stride_inside_window) for start_idx in indices])
    
    # Give importance to the end of the video (Biovid has important info at the end)
    list_indices = video_len - list_indices - 1
    list_indices = list_indices.flip(dims=(0, 1))  # row-wise + clip-wise flip
    
    return list_indices # shape [num_windows, clip_length]
  
  def get_unique_subjects(self):
    return np.sort(self.video_labels['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.video_labels['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.video_labels['class_id'],return_counts=True)[1]
  def get_unique_classes(self):
    return np.sort(self.video_labels['class_id'].unique().tolist())

  def generate_csv_augmented(self,original_csv_path,out_csv_path,dict_augmentation,stratified_training=False):
    return helper.generate_csv_augmented(original_csv_path=original_csv_path,
                           out_csv_path=out_csv_path,
                           dict_augmentation=dict_augmentation,
                           path_to_extracted_features=self.path_dataset,
                           stratified_training=stratified_training)

class customDatasetAggregated(torch.utils.data.Dataset):
  def __init__(self,root_folder_features,concatenate_temporal,concatenate_quadrants,model,is_train,csv_path,smooth_labels,soft_labels,coral_loss,latent_coefficient,consider_only_lasts_n_chunks=None):
    self.root_folder_feature = root_folder_features
    self.csv_path = csv_path
    helper.set_step_shift(root_folder_features)
    print(f"Using step shift = {helper.step_shift} for dataset {root_folder_features}")
    self.concatenate_temporal = concatenate_temporal
    self.concatenate_quadrants = concatenate_quadrants
    self.latent_coefficient = latent_coefficient
    self.df = pd.read_csv(csv_path,sep='\t')
    self.num_classes = len(self.get_unique_classes())
    self.is_quadrant = True if 'combined' in root_folder_features else False
    if not is_train:
      # Keep only original samples if validation/test (sample_id<=8700)
      filter_mask = self.df['sample_id'] <= helper.step_shift
      
      # Save the filtered DataFrame to a new CSV file
      self.df = self.df[filter_mask]
      self.df.to_csv(csv_path,sep='\t',index=False)
      print(f"Filtered DataFrame saved to {csv_path}")
    # Save mask to use to filter helper.dict_data
    # self.mask = np.isin(helper.dict_data['list_sample_id'],self.df['sample_id'].to_list())
    self.model = model
    self.instance_model_name = tools.get_instace_model_name(model)
    self.smooth_labels = smooth_labels
    self.consider_only_lasts_n_chunks = consider_only_lasts_n_chunks
    if soft_labels:
      classes = list(range(self.num_classes))
      distances = {(i, j): abs(i - j) for i in classes for j in classes}
      self.soft_labels = soft_labels_generator.make_all_soft_labels(
        classes=classes,
        distances=distances,
        hardness=soft_labels
      )
    else:
      self.soft_labels = None
    self.coral_loss = coral_loss
  

      
  def _populate_feature_dict(self):
    sample_id_list = torch.tensor(self.df['sample_id'].tolist())
    dict_samples = {
      f'': sample_id_list[sample_id_list <= helper.step_shift], # from 1 to 8700 inclusive
      f'_hflip': sample_id_list[(sample_id_list > get_shift_for_sample_id('hflip'))   * (sample_id_list <= get_shift_for_sample_id('hflip')+helper.step_shift)],
      f'_jitter': sample_id_list[(sample_id_list > get_shift_for_sample_id('jitter')) * (sample_id_list <= get_shift_for_sample_id('jitter')+helper.step_shift)],
      f'_rotate': sample_id_list[(sample_id_list > get_shift_for_sample_id('rotation')) * (sample_id_list <= get_shift_for_sample_id('rotation')+helper.step_shift)],
    }
    list_dict_data = []
    for aug_type, aug_mask_sample_id in dict_samples.items():
      if aug_mask_sample_id.any():    
        split_folder_feature = Path(self.root_folder_feature).parts
        aug_folder_feats = split_folder_feature[-1].split('.')[0] + aug_type
        if aug_type == '' :
          if not helper.dict_data:
            dict_data = tools.load_dict_data(os.path.join(*split_folder_feature[:-1],aug_folder_feats))
            helper.dict_data = dict_data
          else:
            dict_data = helper.dict_data
        # FIlter the sample_id list
        is_in = torch.isin(dict_data['list_sample_id'],aug_mask_sample_id)
        for k,v in dict_data.items():
          dict_data[k] = v[is_in]
          # dict_data[k] = v[aug_mask_sample_id]
        tensor_sample_id = torch.tensor(self.df['sample_id'])
        mask_is_in = torch.isin(tensor_sample_id,dict_data['list_sample_id'])
        tensor_sample_id = tensor_sample_id[mask_is_in].tolist() 
        count = 0
        for sample_id in tensor_sample_id:
          count+=1
          # get the real label from csv
          real_csv_label = self.df[self.df['sample_id'] == sample_id]['class_id'].values[0] 
          mask = dict_data['list_sample_id'] == sample_id # torch tensor
          nr_values = mask.sum().item()
          if nr_values == 0:
            print(f"Sample ID {sample_id} not found in the dataset. populate_feature_dict will not work properly.")
          # create a tensor with the same label
          csv_labels = torch.full((nr_values,),real_csv_label,dtype=dict_data['list_labels'].dtype) 
          dict_data['list_labels'][mask] = csv_labels
        list_dict_data.append(dict_data)
    merged_dict_data = {}
    for k,v in list_dict_data[0].items():
      if isinstance(v,torch.Tensor):
        merged_dict_data[k] = torch.cat([d[k] for d in list_dict_data],dim=0)
      elif isinstance(v,np.ndarray):
        merged_dict_data[k] = np.concatenate([d[k] for d in list_dict_data],axis=0)
    return merged_dict_data 
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    if isinstance(idx,int):
      return _get_element(df=self.df,dict_data=helper.dict_data,idx=idx,
                          dataset_type=CUSTOM_DATASET_TYPE.AGGREGATED,is_quadrant=self.is_quadrant,
                          consider_only_lasts_n_chunks=self.consider_only_lasts_n_chunks,
                          latent_coefficient=self.latent_coefficient,
                          embedding_reduction=self.model.embedding_reduction)
    else:
      raise NotImplementedError("Batch loading is not implemented for customDatasetAggregated")
      batch = [_get_element(df=self.df,dict_data=helper.dict_data,idx=idx) for idx in idx]
      batch = self._custom_collate(batch)
      return batch
  
  def _custom_collate(self,batch):
    return _custom_collate(batch,
                           self.instance_model_name,
                           self.concatenate_temporal,
                           self.model,
                           self.num_classes,
                           self.smooth_labels,
                           self.soft_labels,
                           self.coral_loss,
                           self.concatenate_quadrants)
  
  def get_unique_subjects(self):
    return np.sort(self.df['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.df['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.df['class_id'],return_counts=True)[1]
  def get_unique_classes(self):
    return np.sort(self.df['class_id'].unique().tolist())
  def get_all_sample_ids(self):
    return self.df['sample_id'].tolist()
  
  def generate_csv_augmented(self,original_csv_path,out_csv_path,dict_augmentation,stratified_training=False):
    return helper.generate_csv_augmented(original_csv_path=original_csv_path,
                           out_csv_path=out_csv_path,
                           dict_augmentation=dict_augmentation,
                           path_to_extracted_features=self.root_folder_feature,
                           stratified_training=stratified_training)


class customDatasetWhole(torch.utils.data.Dataset):
  def __init__(self,csv_path,root_folder_features,concatenate_temporal,concatenate_quadrants,model,smooth_labels,soft_labels,coral_loss, xattn_mask,latent_coefficient,split_chunks=False, consider_only_lasts_n_chunks=None,
               feature_merge_type=None,feature_merge_lambda=None,feature_merge_orig=1):
    self.csv_path = csv_path
    helper.set_step_shift(root_folder_features)
    print(f"\n CustomDatasetWhole: Using step shift = {helper.step_shift} for dataset {root_folder_features} \n")
    self.xattn_mask = xattn_mask
    self.root_folder_features = root_folder_features
    self.df = pd.read_csv(csv_path,sep='\t',dtype={'sample_name':str,'subject_name':str}) # subject_id, subject_name, class_id, class_name, sample_id, sample_name
    # count subject_id uniqueness
    unique_subject_ids, counts = np.unique(self.df['subject_id'], return_counts=True)
    print(f"\nUnique subject_ids: {len(unique_subject_ids)}, Total entries: {len(self.df)}")
    # if 'CAER' in root_folder_features:
    #   # remove string in subject name
    #   print("\nRemoving string after / in subject_name for caer dataset\n")
    #   self.df['subject_name'] = self.df['subject_name'].apply(lambda x: x.split('/')[1])
    self.concatenate_temporal = concatenate_temporal
    self.concatenate_quadrants = concatenate_quadrants
    self.model = model
    self.latent_coefficient = latent_coefficient
    self.is_quadrant = True if 'combined' in root_folder_features else False
    self.instance_model_name = tools.get_instace_model_name(model)
    self.smooth_labels = smooth_labels
    self.num_classes = len(self.get_unique_classes())
    self.split_chunks = split_chunks
    self.consider_only_lasts_n_chunks = consider_only_lasts_n_chunks
    if soft_labels:
      classes = list(range(self.num_classes))
      distances = {(i, j): abs(i - j) for i in classes for j in classes}
      self.soft_labels = soft_labels_generator.make_all_soft_labels(
        classes=classes,
        distances=distances,
        hardness=soft_labels
      )
    else:
      self.soft_labels = None
    self.coral_loss = coral_loss
    self.feature_merge_type = feature_merge_type
    self.feature_merge_lambda = feature_merge_lambda
    self.feature_merge_orig = feature_merge_orig
    self._discover_variant_folders()

  def _discover_variant_folders(self):
    """
    Scan the filesystem to discover multi-variant augmentation folders.

    For each augmentation type that has a base folder ({root_folder_features}_{aug_type}),
    discovers additional variant folders with naming convention
    {root_folder_features}_{aug_type}$N where N is a non-negative integer.

    Populates self.aug_variant_folders: dict mapping aug_type (str) to a list of
    absolute folder paths. The base folder (without $N) is included when present.

    Returns:
      None. Sets self.aug_variant_folders as a side effect.
    """
    self.aug_variant_folders = {}
    parent_dir = os.path.dirname(self.root_folder_features)
    base_name = os.path.basename(self.root_folder_features)
    if not parent_dir or not os.path.isdir(parent_dir):
      return
    pattern = re.compile(r'^' + re.escape(base_name) + r'_([^$]+?)(?:\$(\d+))?$')
    for entry in os.listdir(parent_dir):
      m = pattern.match(entry)
      if not m:
        continue
      full_path = os.path.join(parent_dir, entry)
      if not os.path.isdir(full_path):
        continue
      aug_type = m.group(1)
      if aug_type not in self.aug_variant_folders:
        self.aug_variant_folders[aug_type] = []
      self.aug_variant_folders[aug_type].append(full_path)
    for aug_type in self.aug_variant_folders:
      self.aug_variant_folders[aug_type].sort()

  def __len__(self):
    return len(self.df)
    
  def _load_element(self,idx):
    pid = os.getpid()
    # if pid not in helper.time_profile_dict:
    #   helper.time_profile_dict[pid] = {}
    csv_row = self.df.iloc[idx]
    sample_id = csv_row['sample_id']
    if sample_id > helper.step_shift and not helper.is_latent_basic_augmentation(sample_id) and not helper.is_latent_masking_augmentation(sample_id):
      aug_type = helper.get_augmentation_type(sample_id)
      if aug_type in self.aug_variant_folders:
        variants = self.aug_variant_folders[aug_type]
        aug_base_folder = variants[np.random.randint(len(variants))]
      else:
        aug_base_folder = f"{self.root_folder_features}_{aug_type}"
      folder_path = os.path.join(aug_base_folder, csv_row['subject_name'], f"{csv_row['sample_name']}.safetensors")
    else:
      if 'bottom_left' in self.root_folder_features:
        folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],f"{csv_row['sample_name']}$bottom_left.safetensors")
      elif 'bottom_right' in self.root_folder_features:
        folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],f"{csv_row['sample_name']}$bottom_right.safetensors")
      elif 'upper_left' in self.root_folder_features:
        folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],f"{csv_row['sample_name']}$upper_left.safetensors")
      elif 'upper_right' in self.root_folder_features:
        folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],f"{csv_row['sample_name']}$upper_right.safetensors")
      else:
        folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],f"{csv_row['sample_name']}.safetensors")
    # load_start = time.perf_counter()
    with profile_workers(f'{pid}_loading_dict_time',helper.time_profiling_enabled,helper.time_profile_dict):
      if helper.dict_data_in_memory:
        # abs_path = os.path.abspath(folder_path)
        # key = os.path.splitext(os.path.basename(folder_path))[0] # get the filename without extension
        features = helper.dict_data[folder_path]
      else:
        features = tools.load_dict_data(folder_path)

      if self.feature_merge_type is not None:
        features = self._merge_top_down_features(features, csv_row, sample_id)

    return _get_element(dict_data=features,
                        df=self.df,idx=idx,
                        dataset_type=CUSTOM_DATASET_TYPE.WHOLE, 
                        is_quadrant=self.is_quadrant,
                        consider_only_lasts_n_chunks=self.consider_only_lasts_n_chunks,
                        embedding_reduction=self.model.embedding_reduction,
                        latent_coefficient=self.latent_coefficient,
                        xattn_mask=self.xattn_mask)

  def _merge_top_down_features(self, features, csv_row, sample_id):
    """Merge top/down feature variants with the original features.

    Args:
      features (dict): Loaded original feature dict with keys 'features', 'list_labels', etc.
      csv_row (pd.Series): CSV row for the current sample.
      sample_id (int): Sample ID used to determine augmentation routing.

    Returns:
      dict: The features dict with 'features' key replaced by the merged tensor.
    """
    if sample_id > helper.step_shift and not helper.is_latent_basic_augmentation(sample_id) and not helper.is_latent_masking_augmentation(sample_id):
      aug_type = helper.get_augmentation_type(sample_id)

    top_path = os.path.join(self.root_folder_features, csv_row['subject_name'], f"{csv_row['sample_name']}$top.safetensors")
    down_path = os.path.join(self.root_folder_features, csv_row['subject_name'], f"{csv_row['sample_name']}$down.safetensors")

    top_features = tools.load_dict_data(top_path)
    down_features = tools.load_dict_data(down_path)

    if self.feature_merge_type == 'sum':
      k = self.feature_merge_lambda
      if self.feature_merge_orig:
        features['features'] = features['features'] + k * (top_features['features'] + down_features['features'])
      else:
        features['features'] = k * (top_features['features'] + down_features['features'])
    elif self.feature_merge_type == 'concat':
      if self.feature_merge_orig:
        features['features'] = torch.cat([features['features'], top_features['features'], down_features['features']], dim=-1)
      else:
        features['features'] = torch.cat([top_features['features'], down_features['features']], dim=-1)

    return features

  def __getitem__(self,idx):
    if isinstance(idx,int):
      el = self._load_element(idx=idx)
      return el
    else:
      raise NotImplementedError("Batch loading is not implemented for customDatasetWhole")
      batch_time_start = time.perf_counter()
      # batch = [_load_element(idx=idx) for idx in idx]
      with ThreadPoolExecutor(max_workers=len(idx)) as executor:
        batch = list(executor.map(self._load_element, idx))
      batch_time_end = time.perf_counter()
      print(f"Batch {idx} loading time: {batch_time_end - batch_time_start:.2f} seconds")
      batch = self._custom_collate(batch)
      return batch

  def _custom_collate(self,batch):
    return _custom_collate(batch,self.instance_model_name,
                           self.concatenate_temporal,
                           self.model,
                           self.num_classes,
                           self.smooth_labels,
                           self.soft_labels,
                           self.coral_loss,
                           self.concatenate_quadrants,
                           self.xattn_mask,
                           self.split_chunks)  

  def get_unique_subjects(self):
    return np.sort(self.df['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.df['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.df['class_id'],return_counts=True)[1]
  def get_unique_classes(self,return_all=False):
    list_unique_classes = self.df['class_id'].unique().tolist()
    max_class = max(list_unique_classes)
    if not return_all:
      return np.sort(list_unique_classes)
    else:
      return np.sort(list(range(max_class+1))) # in case some class is missing
  def get_all_sample_ids(self):
    return self.df['sample_id'].tolist()

  def generate_csv_augmented(self,original_csv_path,out_csv_path,dict_augmentation,stratified_training=False):
    return helper.generate_csv_augmented(original_csv_path=original_csv_path,
                           out_csv_path=out_csv_path,
                           dict_augmentation=dict_augmentation,
                           path_to_extracted_features=self.root_folder_features,
                           stratified_training=stratified_training)
  
# Alternative version with even more optimizations for large batches
def highly_optimized_custom_collate(batch, pid, is_training,concatenate_temporal=False, smooth_labels=0.0,
  soft_labels_mat=None, coral_loss=False, num_classes=None,concatenate_quadrants=False, split_chunks=False,xattn_mask=None):
  """Highly optimized version using vectorized operations where possible"""
  
  batch_size = len(batch)
  # Try to stack tensors directly if they have the same shape (more efficient)
  if split_chunks:
    _, T, _, _, _ = batch[0]['features'].shape
    features = torch.cat([sample['features'] for sample in batch], dim=0) # [B*quadrants*nr_chunks,T,S,S,Emb]
    features = features.view(features.shape[0], -1, features.shape[-1]).to(torch.float32)
    lengths = None
    batch_size = features.shape[0]
  else:
    try:
      # Check if all features have the same shape
      # first_shape = batch[0]['features'].shape
      # Stack all features at once - much faster
      # shape_array = torch.tensor([sample['features'].shape for sample in batch])
      # print(f"Shapes in batch: {shape_array}")
      features = torch.stack([sample['features'] for sample in batch]) # [B,quadrants*nr_chunks,T,S,S,Emb]
      B, QnC, T, S1, S2, emb = features.shape
      if concatenate_quadrants:
        # features stacked as [upper_left,upper_left,upper_left,upper_left,
        #                      upper_right,upper_right,...] # [quadrants * nr_chunks]
        # So i have to reshape 1D vector [upp_left,upp_left,...,upp_right,...] into 2D matrix [[upp_left,upp_left,...],[upp_right,upp_right,...],[...],...] -> [quadrants, nr_chunks]
        # from 2D matrix I want to concatenate different quadrants from the same chunk ->So from [quadrants,nr_chunks] to [nr_chunks, quadrants]
        features = features.reshape(B, 4, QnC//4, T, S1, S2, emb).transpose(1,2) # [B,4=quadrants,QnC/4,T,S,S,Emb] -> [B,QnC/4,4,T,S,S,Emb]
        # concatenate quadrants in channel dimension
        features = features.permute(0,1,3,4,5,2,6).contiguous()
        features = features.view(batch_size, (QnC//4)*T*S1*S2, 4*emb).to(torch.float32)
      elif concatenate_temporal:
        raise NotImplementedError("concatenate_temporal=True not implemented yet in highly_optimized_custom_collate")
        # features = all_features.reshape()
        features = all_features.permute(0,2,3,4,1,5).contiguous().view(
            batch_size, -1, all_features.shape[-1] * all_features.shape[1]).to(torch.float32)
      else:
        features = features.view(batch_size, -1, features.shape[-1]).to(torch.float32)
        # lengths = torch.full((batch_size,), features.size(1), dtype=torch.int32)
      
      lengths = None
    # else:
    #     # Fall back to individual processing
    #     raise ValueError("Shapes don't match")
        
    except (ValueError, RuntimeError):
      # Fall back to the standard approach for variable shapes
      features = []
      lengths = []

      for sample in batch:
        feat = sample['features'] # [quadrants*nr_chunks,T,S,S,Emb]
        QnC, T, S1, S2, emb = feat.shape
        if concatenate_temporal:
          raise NotImplementedError("concatenate_temporal=True not implemented yet in highly_optimized_custom_collate")
          processed_feat = feat.permute(0,2,3,4,1).contiguous().view(
            -1, feat.shape[-1] * feat.shape[1]).to(torch.float32)
        elif concatenate_quadrants:
          processed_feat = feat.view(4, QnC//4, T, S1, S2, emb).transpose(0,1) # [4=quadrants,QnC/4,T,S,S,Emb] -> [QnC/4,4,T,S,S,Emb]
          processed_feat = processed_feat.permute(0,2,3,4,1,5).contiguous().view(
            QnC//4*T*S1*S2,4*emb).to(torch.float32)
        else:
          processed_feat = feat.view(-1, feat.shape[-1]).to(torch.float32)
        # if not concatenate_temporal:
        #   processed_feat = feat.view(-1, feat.shape[-1]).to(torch.float32)
        # else:
        #   processed_feat = feat.permute(0,2,3,1,4).contiguous().view(
        #     -1, feat.shape[-1] * feat.shape[-2]).to(torch.float32)
        
        features.append(processed_feat)
        lengths.append(processed_feat.size(0))
      
      with profile_workers(f'{pid}_pad_sequence_time',helper.time_profiling_enabled,helper.time_profile_dict):
        lengths = torch.tensor(lengths, dtype=torch.int32)
        features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
        
      helper.time_profile_dict[f'{pid}_pad_sequence_calls'] = helper.time_profile_dict.get(f'{pid}_pad_sequence_calls', 0) + 1
  
  # Build key_padding_mask for attention
  max_len = features.size(1)
  if lengths is None:
    # Case when shapes matched perfectly (no padding)
    key_padding_mask = None
  else:
    # Variable lengths -> pad mask
    # max_len = features.size(1)
    key_padding_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

  # Apply xattn_mask if provided (shape [S, S])
  with profile_workers(f'{pid}_xattn_mask_time',helper.time_profiling_enabled,helper.time_profile_dict):
    if xattn_mask is not None and not concatenate_quadrants:
      nr_chunks = max_len // (T*S1*S2) if not split_chunks else 1
      if key_padding_mask is not None:
        tmp = xattn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(key_padding_mask.shape[0],nr_chunks,T,1,1) # [B,nr_chunks,T,S,S]
        tmp = tmp.reshape(tmp.shape[0],-1).unsqueeze(1).unsqueeze(1) # [B,1,1,nr_chunks*T*S*S]
        key_padding_mask = key_padding_mask & tmp
      else:
        key_padding_mask = xattn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_size,nr_chunks,T,1,1) # [B,nr_chunks,T,S,S]
        if split_chunks:
          key_padding_mask = key_padding_mask.reshape(key_padding_mask.shape[0]*key_padding_mask.shape[1],-1) # [B*nr_chunks,T*S*S]
        else:
          key_padding_mask = key_padding_mask.reshape(key_padding_mask.shape[0],-1) # [B,nr_chunks*T*S*S]
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1) # [B,1,1,nr_chunks*T*S*S]

  # Vectorized extraction of labels, subject_ids, and sample_ids
  labels = torch.tensor([sample['labels'][0] for sample in batch], dtype=torch.int32)
  subject_id = torch.tensor([sample['subject_id'][0] for sample in batch], dtype=torch.int32)
  sample_id = torch.tensor([sample['sample_id'] for sample in batch], dtype=torch.int32)
  
  if split_chunks:
    repeats = torch.tensor([sample['features'].shape[0] for sample in batch], dtype=torch.int32)
    subject_id = torch.repeat_interleave(subject_id, repeats=repeats, dim=0)
    sample_id = torch.repeat_interleave(sample_id, repeats=repeats, dim=0)
    labels = torch.repeat_interleave(labels, repeats=repeats, dim=0)
  
  # Handle label smoothing when in training mode
  if is_training and smooth_labels > 0.0:
    labels = smooth_labels_batch(gt_classes=labels, num_classes=num_classes,
      smoothing=smooth_labels)
  
  # Handle label transformations (same as before)
  if soft_labels_mat is not None:
    labels = soft_labels_mat[labels].to(torch.float32)
  elif coral_loss:
    labels = levels_from_labelbatch(labels, num_classes, dtype=torch.float32)
    
  return {
    'features': features,
    'lengths': lengths,
    'labels': labels,
    'subject_id': subject_id,
    'sample_id': sample_id,
    'key_padding_mask': key_padding_mask,
    # 'preprocessing_time': time_preprocess_end - time_preprocess_start
  }


def _custom_collate(batch,instance_model_name,concatenate_temporal,model,num_classes,smooth_labels, soft_labels_mat,coral_loss,concatenate_quadrants,xattn_mask=None,split_chunks=False):
  # Pre-flatten features: reshape each sample to (sequence_length, emb_dim)
  pid = os.getpid()
  # if pid not in helper.time_profile_dict:
  #   helper.time_profile_dict[pid] = 
  
  if instance_model_name != helper.INSTANCE_MODEL_NAME.Pool_MLP:
    with profile_workers(f'{pid}_collate_preprocess_time',helper.time_profiling_enabled,helper.time_profile_dict):
      dict_res = highly_optimized_custom_collate(batch=batch,
                                                pid=pid,
                                                is_training = model.training,
                                                concatenate_temporal=concatenate_temporal,
                                                smooth_labels=smooth_labels,
                                                soft_labels_mat=soft_labels_mat,
                                                coral_loss=coral_loss,
                                                split_chunks=split_chunks,
                                                concatenate_quadrants=concatenate_quadrants,
                                                xattn_mask=xattn_mask,
                                                num_classes=num_classes)
        
    if instance_model_name == helper.INSTANCE_MODEL_NAME.AttentiveClassifier or instance_model_name == helper.INSTANCE_MODEL_NAME.ATTENTIVEPROBE:

      return {'x':dict_res['features'], 'key_padding_mask': dict_res['key_padding_mask']},\
              dict_res['labels'],\
              dict_res['subject_id'],\
              dict_res['sample_id']
    elif instance_model_name == helper.INSTANCE_MODEL_NAME.GRUPROBE:
      raise NotImplementedError("GRUPROBE not implemented in collate function.")
      packed_input = torch.nn.utils.rnn.pack_padded_sequence(features,lengths_tensor,batch_first=True,enforce_sorted=False)
      if model.output_size == 1:
        labels = labels.float()
      return {'x':packed_input},\
              labels,\
              subject_id,\
              sample_id
    else:
      raise NotImplementedError(f"Instance model {instance_model_name} not implemented for collate function.")
      
  else:
    features = torch.cat([torch.mean(sample['features'],dim=(0,1,2,3),keepdim=True,dtype=torch.float32) for sample in batch],dim=0) # mean over the sequence
    features = features.squeeze(1).squeeze(1).squeeze(1) # [B, Emb]
    labels = torch.tensor([sample['labels'][0] for sample in batch],dtype=torch.int32)
    subject_id = torch.tensor([sample['subject_id'][0] for sample in batch], dtype=torch.int32)
    sample_id = torch.tensor([sample['sample_id'] for sample in batch], dtype=torch.int32)
    return {'x':features},labels,subject_id,sample_id
 

class balancedBatchSampler(BatchSampler):
  def __init__(self, batch_size, shuffle,path_cvs_dataset=None, random_state=42,df=None):
    if df is None:
      csv_array,_ = tools.get_array_from_csv(path_cvs_dataset)
    else:
      csv_array = df.to_numpy()
    # self.y_labels = np.array(csv_array[:,2]).astype(int)
    self.y_labels = df['class_id'].to_numpy().astype(int) if df is not None else np.array(csv_array[:,2]).astype(int)
    self.n_batch_size = batch_size
    self.random_state = random_state
    self.shuffle = shuffle
    nr_samples = len(self.y_labels)
    self.calculated_n_splits = math.ceil(nr_samples/self.n_batch_size)
    _,count = np.unique(self.y_labels,return_counts=True)
    min_class_count = np.min(count)
    if self.calculated_n_splits > min_class_count:
      raise ValueError(f"Cannot create stratified batches: "+
                       f"the calculated number of splits {self.calculated_n_splits} is higher than the minimum class count {min_class_count}. ")
    self.initialize()
    
  def initialize(self):
    self.skf = StratifiedKFold(n_splits = self.calculated_n_splits,
                               shuffle=self.shuffle,
                               random_state=self.random_state)
    self.n_batches = self.skf.get_n_splits()
    
    # Informative but batches can vary slightly in size
    self.real_batch_size = math.ceil(len(self.y_labels)/self.n_batches) 
    
  def __iter__(self):
    for _,test in self.skf.split(np.zeros(self.y_labels.shape[0]), self.y_labels):
      yield test.astype(np.int32).tolist() 
      
    # Update random state for next epoch, so that shuffling is different
    self.random_state += 13
    self.initialize()
      

  def __len__(self):
    return self.n_batches

class SelectiveAugmentationBatchSampler(BatchSampler):
  """
  A batch sampler that controls how many and which augmented samples are included per training batch.

  Each original sample in the dataset may have multiple pre-computed augmented variants
  (e.g., hflip, rotation, jitter). This sampler groups samples by their base ID and, for
  each group, always includes the original sample plus a controlled subset of its augmentations.

  Two strategies are supported:
    - Strategy 1 (random samples): For each group, randomly pick `n_keep_augmentations`
      augmented variants from those available for that specific sample.
    - Strategy 2 (random types): Randomly pick `n_keep_augmentations` augmentation *types*
      from the globally available set and include only those types for each group (if present).

  This avoids inflating the dataset with all augmentations at once while still providing
  augmentation diversity across epochs (since the selection is re-randomized each epoch via `__iter__`).

  Args:
    df:                      DataFrame with at least 'sample_id' column; index used as dataset indices.
    batch_size:              Number of samples per batch.
    shuffle:                 Whether to shuffle the selected indices before batching.
    n_keep_augmentations:    Maximum number of augmented variants to keep per original sample.
    augmentation_strategy:   1 = pick random augmented samples; 2 = pick random augmentation types globally.
    root_folder_features:    Optional path to feature folder, used to discover available augmentation types.
    drop_last:               Whether to drop the last incomplete batch.
  """
  def __init__(self, df, batch_size, shuffle, n_keep_augmentations, augmentation_strategy, root_folder_features=None, drop_last=False, keep_original=1.0):
    """
    Args:
      df:                      DataFrame with at least 'sample_id' column; index used as dataset indices.
      batch_size:              Number of samples per batch.
      shuffle:                 Whether to shuffle the selected indices before batching.
      n_keep_augmentations:    Maximum number of augmented variants to keep per original sample.
      augmentation_strategy:   1 = pick random augmented samples; 2 = pick random augmentation types globally.
      root_folder_features:    Optional path to feature folder, used to discover available augmentation types.
      drop_last:               Whether to drop the last incomplete batch.
      keep_original:           Fraction of original samples to keep per epoch (0-1). When < 1, a random subset
                               of originals is selected each epoch; their augmentations are excluded. Non-selected
                               originals are excluded but their augmentations remain available via the strategy.
    """
    if not 0 <= keep_original <= 1:
      raise ValueError(f"keep_original must be between 0 and 1, got {keep_original}")
    self.keep_original = keep_original
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.n_keep_augmentations = n_keep_augmentations
    self.augmentation_strategy = augmentation_strategy # 1: random samples, 2: random types
    self.drop_last = drop_last
    
    self.log_augmented_usage = {idx: 0 for idx in df.index}
    self.base_id_groups = {} # base_id -> {'original': idx, 'augmented': {type: idx}}
    self.available_augmentation_types = set()
    
    # Get available augmentations from folder if possible
    if root_folder_features:
      try:
        self.available_augmentation_types = set(helper.get_augmentation_availables(root_folder_features))
        print(f"\n\nAvailable augmentation types retrieved from folder: {self.available_augmentation_types}")
      except Exception as e:
        print(f"Warning: Could not get available augmentations from folder: {e}")
    
    for idx, row in df.iterrows():
      sample_id = row['sample_id']
      if sample_id <= helper.step_shift:
        base_id = sample_id
        is_original = True
        aug_type = None
      else:
        base_id = (sample_id - 1) % helper.step_shift + 1
        is_original = False
        aug_type = helper.get_augmentation_type(sample_id)
        if aug_type:
          self.available_augmentation_types.add(aug_type)
      
      if base_id not in self.base_id_groups:
        self.base_id_groups[base_id] = {'original': None, 'augmented': {}}
        
      if is_original:
        self.base_id_groups[base_id]['original'] = idx
      elif aug_type:
        self.base_id_groups[base_id]['augmented'][aug_type] = idx

    self.available_augmentation_types = sorted(list(self.available_augmentation_types))
    self._calculate_length()

  def _calculate_length(self):
    """
    Estimates the total number of samples per epoch to compute the number of batches.

    When keep_original < 1, only a fraction of groups contribute their original (no augmentations),
    and the remaining groups contribute only augmentations (no original).

    Returns:
      None. Sets self.n_batches as a side effect.
    """
    total_samples = 0
    n_groups_with_original = sum(1 for g in self.base_id_groups.values() if g['original'] is not None)

    if self.keep_original >= 1.0:
      for group in self.base_id_groups.values():
        if group['original'] is not None:
          total_samples += 1
        n_available = len(group['augmented'])
        total_samples += min(self.n_keep_augmentations, n_available)
    else:
      n_original_kept = int(self.keep_original * n_groups_with_original)
      # Original-only groups contribute 1 sample each (no augmentations)
      total_samples += n_original_kept
      # Augmentation-only groups contribute augmentations only
      n_augment_groups = n_groups_with_original - n_original_kept
      avg_aug = np.mean([min(self.n_keep_augmentations, len(g['augmented']))
                         for g in self.base_id_groups.values() if g['augmented']]) if n_augment_groups > 0 else 0
      total_samples += int(n_augment_groups * avg_aug)

    if self.drop_last:
      self.n_batches = total_samples // self.batch_size
    else:
      self.n_batches = (total_samples + self.batch_size - 1) // self.batch_size

  def __len__(self):
    return self.n_batches

  def _select_augmentations_for_group(self, group):
    """
    Selects augmented indices for a group according to the current augmentation strategy.

    Args:
      group: Dict with 'original' (int or None) and 'augmented' (dict of aug_type -> idx).

    Returns:
      List of selected augmented dataset indices.
    """
    augmented_indices_to_keep = []
    if self.augmentation_strategy == 1:
      available_indices = list(group['augmented'].values())
      if available_indices:
        n_to_pick = min(self.n_keep_augmentations, len(available_indices))
        augmented_indices_to_keep = np.random.choice(available_indices, size=n_to_pick, replace=False).tolist()
    elif self.augmentation_strategy == 2:
      if self.available_augmentation_types:
        n_types_to_pick = min(self.n_keep_augmentations, len(self.available_augmentation_types))
        picked_types = np.random.choice(self.available_augmentation_types, size=n_types_to_pick, replace=False)
        augmented_indices_to_keep = [group['augmented'][t] for t in picked_types if t in group['augmented']]
    else:
      raise ValueError(f"Unknown augmentation strategy: {self.augmentation_strategy}")
    return augmented_indices_to_keep

  def __iter__(self):
    """
    Yields batches of dataset indices for one epoch.

    When keep_original < 1, a random subset of base_ids is selected as "original-only" (their
    augmentations are excluded). The remaining base_ids become "augmentation-only" (their originals
    are excluded, augmentations selected via the configured strategy). The subset changes each epoch.

    Returns:
      Iterator of lists of dataset indices (batches).
    """
    selected_indices = []
    all_base_ids = list(self.base_id_groups.keys())

    if self.keep_original < 1.0:
      n_groups_with_original = sum(1 for g in self.base_id_groups.values() if g['original'] is not None)
      n_to_keep = int(self.keep_original * n_groups_with_original)
      base_ids_with_original = [bid for bid in all_base_ids if self.base_id_groups[bid]['original'] is not None]
      kept_base_ids = set(np.random.choice(base_ids_with_original, size=n_to_keep, replace=False)) if n_to_keep > 0 else set()
    else:
      kept_base_ids = None  # sentinel: keep all originals

    for base_id in all_base_ids:
      group = self.base_id_groups[base_id]

      if kept_base_ids is None:
        # keep_original == 1.0: original behavior — include original + augmentations
        if group['original'] is not None:
          selected_indices.append(group['original'])
        else:
          raise ValueError("Original sample missing for a group, which should not happen.")
        augmented_indices_to_keep = self._select_augmentations_for_group(group)

      elif base_id in kept_base_ids:
        # Original-only group: include original, skip augmentations
        selected_indices.append(group['original'])
        augmented_indices_to_keep = []

      else:
        # Augmentation-only group: skip original, select augmentations via strategy
        augmented_indices_to_keep = self._select_augmentations_for_group(group)

      selected_indices.extend(augmented_indices_to_keep)
      for idx in augmented_indices_to_keep:
        self.log_augmented_usage[idx] += 1

    if self.shuffle:
      np.random.shuffle(selected_indices)

    batch = []
    for idx in selected_indices:
      batch.append(idx)
      if len(batch) == self.batch_size:
        yield batch
        batch = []

    if len(batch) > 0 and not self.drop_last:
      yield batch


class AugmentedOnlyBatchSampler(BatchSampler):
  def __init__(self, df, batch_size, augmentations, shuffle=True, random_state=42, balance_batch=True,root_folder_features=None):
    """
    BatchSampler that creates batches containing ONLY augmented versions of samples.
    
    Logic:
    1. Filters df to find 'original' samples (sample_id <= helper.step_shift).
    2. Uses StratifiedKFold on these original samples to determine which 'base' samples go into a batch.
       The number of base samples per batch is batch_size // n_augmentations.
    3. For each selected base sample, finds the requested augmented versions.
    4. The final batch consists ONLY of these augmented versions.
    
    Args:
        df (pd.DataFrame): The full dataset dataframe.
        batch_size (int): The desired output batch size.
        augmentations (str): Comma-separated string of augmentations (e.g. "hflip,jitter").
        shuffle (bool): Whether to shuffle the data (via StratifiedKFold).
        random_state (int): Random seed for reproducibility.
        balance_batch (bool): Whether to use StratifiedKFold (True) or simple random sampling (False).
        root_folder_features (str): Path to the root folder of features, used to get all available augmentations.
    """
    self.df = df
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.random_state = random_state
    self.balance_batch = balance_batch
    self.batches = []
    self.root_folder_features = root_folder_features
    self.available_augmentation_types = None
    self.strategy = None
    
    # Get available augmentations from folder if possible
    if root_folder_features:
      try:
        self.available_augmentation_types = sorted(list(set(helper.get_augmentation_availables(root_folder_features))))
        print(f"\n\nAvailable augmentation types retrieved from folder: {self.available_augmentation_types}")
      except Exception as e:
        print(f"Warning: Could not get available augmentations from folder: {e}")
    
    if augmentations.startswith('strategy'):
      if root_folder_features is None:
        raise ValueError("root_folder_features is required when using augmentation strategies.")
      
      if self.available_augmentation_types is None or len(self.available_augmentation_types) == 0:
         raise ValueError(f"Could not retrieve available augmentations from {root_folder_features}. Required for strategy.")

      parts = augmentations.split(',')
      strategy_name = parts[0]
      try:
        self.k = int(parts[1])
      except (IndexError, ValueError):
        raise ValueError(f"Invalid format for strategy. Expected 'strategyX,K', got '{augmentations}'")
      
      if strategy_name == 'strategy1': # Select augmentations for the whole epoch
        self.strategy = 1
      elif strategy_name == 'strategy2': # Select random augmentations for each sample independently
        self.strategy = 2
      else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
      
      self.n_augment = self.k
      self.augmentations_list = [] # Placeholder, determined at runtime
    else:
      self.augmentations_list = [aug.strip() for aug in augmentations.split(',')]
      self.n_augment = len(self.augmentations_list)
    
    if self.n_augment == 0:
      raise ValueError("At least one augmentation must be specified.")
      
    self.sub_batch_size = self.batch_size // self.n_augment
    
    if self.sub_batch_size < 1:
      raise ValueError(f"Batch size {self.batch_size} is too small for {self.n_augment} augmentations. "
                       f"Must be at least {self.n_augment}.")

    # 1. Filter for original samples
    self.original_df = self.df[self.df['sample_id'] <= helper.step_shift].copy()
    if len(self.original_df) == 0:
        raise ValueError("No original samples (sample_id <= step_shift) found in the dataframe.")
        
    self.y_labels = self.original_df['class_id'].to_numpy().astype(int)
    self.original_indices = self.original_df.index.to_numpy() # Indices in the global df
    
    # 2. Create lookup map: sample_id -> global dataframe index
    # We need to map sample_ids back to the indices expected by the DataLoader (which iterates over the full df)
    self.sample_id_to_idx = dict(zip(self.df['sample_id'], self.df.index))
    
    # Calculate number of batches
    nr_samples = len(self.y_labels)
    self.calculated_n_splits = math.ceil(nr_samples / self.sub_batch_size)
    
    if self.balance_batch:
      # Check if we can perform stratified sampling
      _, count = np.unique(self.y_labels, return_counts=True)
      min_class_count = np.min(count)
      if self.calculated_n_splits > min_class_count:
        raise ValueError(f"Cannot create stratified batches: "+
                         f"the calculated number of splits {self.calculated_n_splits} is higher than the minimum class count {min_class_count}. ")
        
    self.initialize()

  def initialize(self):
    self.batches = []
    
    # Get the indices of original samples (relative to self.original_df) that will form each batch
    batch_indices_list = []
    
    if self.balance_batch:
      try:
        rnd_state = self.random_state if self.shuffle else None
        skf = StratifiedKFold(n_splits=self.calculated_n_splits,
                                   shuffle=self.shuffle,
                                   random_state=rnd_state)
        # split returns indices relative to the input array (which matches self.original_df rows)
        for _, test_indices in skf.split(np.zeros(len(self.y_labels)), self.y_labels):
          batch_indices_list.append(test_indices)
            
      except ValueError as e:
        raise ValueError(f"Failed to initialize StratifiedKFold: {e}. Check if batch size is appropriate for class distribution.") from e
    else:
      # Simple random sampling
      indices = np.arange(len(self.y_labels))
      if self.shuffle:
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(indices)
        
      # Split into chunks of size sub_batch_size
      for i in range(0, len(indices), self.sub_batch_size):
        batch_indices_list.append(indices[i : i + self.sub_batch_size])

    # Strategy 1: Select augmentations for the whole epoch
    epoch_augmentations = None
    if self.strategy == 1:
      epoch_augmentations = np.random.choice(self.available_augmentation_types, size=self.k, replace=False)

    # Now construct the actual batches with augmented global indices
    for indices_in_orig in batch_indices_list:
      final_batch = []
      
      for idx_in_orig in indices_in_orig:
        original_row = self.original_df.iloc[idx_in_orig]
        original_sample_id = original_row['sample_id']
        
        # Determine augmentations to use
        if self.strategy == 1:
            augmentations_to_use = epoch_augmentations
        elif self.strategy == 2: # 
            augmentations_to_use = np.random.choice(self.available_augmentation_types, size=self.k, replace=False)
        else:
            augmentations_to_use = self.augmentations_list

        # Find the requested augmented versions
        for aug_type in augmentations_to_use:
          shift = helper.get_shift_for_sample_id(aug_type)
          augmented_sample_id = original_sample_id + shift
          
          if augmented_sample_id not in self.sample_id_to_idx:
            raise ValueError(f"Augmented sample {augmented_sample_id} (original: {original_sample_id}, type: {aug_type}) "
                             f"not found in the dataset.")
            
          global_idx = self.sample_id_to_idx[augmented_sample_id]
          final_batch.append(global_idx)
      
      if len(final_batch) > 0:
        self.batches.append(final_batch)

    self.n_batches = len(self.batches)

  def __iter__(self):
    for batch in self.batches:
      yield batch

    # Update random state and re-initialize for next epoch
    self.random_state += 13
    self.initialize()

  def __len__(self):
    return self.n_batches


class SubjectBatchSampler(BatchSampler):
  """
  Ensures each batch contains a minimum number of unique subjects per pain level,
  so HSIC can measure dependence reliably.

  Strategy: For each batch, sample subjects first, then sample instances from
  each subject across pain levels. This guarantees subject diversity within
  each pain-level group while maintaining class balance.
  """

  def __init__(self, df, batch_size, min_subjects_per_level,
               include_augmented=True, require_all_levels=True,
               shuffle=True, random_state=42):
    self.df = df
    self.batch_size = batch_size
    self.min_subjects_per_level = min_subjects_per_level
    self.include_augmented = include_augmented
    self.require_all_levels = require_all_levels
    self.shuffle = shuffle
    self.random_state = random_state

    # Filter out augmented samples if needed
    if not include_augmented:
      self.working_df = df[df['sample_id'] <= helper.step_shift].copy()
    else:
      self.working_df = df.copy()

    # Extract pain levels
    self.pain_levels = sorted(self.working_df['class_id'].unique())
    self.n_levels = len(self.pain_levels)

    # Build lookup: {class_id: {subject_id: [df_indices]}}
    self.level_subject_map = {}
    for level in self.pain_levels:
      level_df = self.working_df[self.working_df['class_id'] == level]
      subjects = level_df['subject_id'].unique()
      self.level_subject_map[level] = {}
      for subj in subjects:
        indices = level_df[level_df['subject_id'] == subj].index.tolist()
        self.level_subject_map[level][subj] = indices

    # Validate min_subjects_per_level constraint
    for level in self.pain_levels:
      n_subjects = len(self.level_subject_map[level])
      if n_subjects < min_subjects_per_level:
        raise ValueError(
          f"Pain level {level} has only {n_subjects} unique subjects, "
          f"but min_subjects_per_level={min_subjects_per_level} is required."
        )

    # Compute derived values
    self.samples_per_level = batch_size // self.n_levels

    if self.samples_per_level < self.min_subjects_per_level:
      raise ValueError(
        f"samples_per_level ({self.samples_per_level}) < min_subjects_per_level "
        f"({self.min_subjects_per_level}). Increase batch_size or decrease "
        f"min_subjects_per_level."
      )

    self.initialize()

  def initialize(self):
    self.batches = []
    rng = np.random.default_rng(self.random_state)

    # Build shuffled subject lists and sample pools per level
    level_data = {}
    for level in self.pain_levels:
      subjects = list(self.level_subject_map[level].keys())
      if self.shuffle:
        rng.shuffle(subjects)

      pools = {}
      for subj in subjects:
        pool = list(self.level_subject_map[level][subj])
        if self.shuffle:
          rng.shuffle(pool)
        pools[subj] = pool

      level_data[level] = {'subjects': subjects, 'pools': pools}

    # Number of batches limited by smallest level's total samples
    min_total_samples = min(
      sum(len(indices) for indices in self.level_subject_map[level].values())
      for level in self.pain_levels
    )
    n_batches = min_total_samples // self.samples_per_level

    # Generate each batch
    for batch_idx in range(n_batches):
      batch = []
      for level in self.pain_levels:
        ld = level_data[level]
        subjects = ld['subjects']
        pools = ld['pools']
        n_subjects = len(subjects)

        # Select min_subjects_per_level subjects via rotation
        start = (batch_idx * self.min_subjects_per_level) % n_subjects
        selected_subjects = []
        for i in range(self.min_subjects_per_level):
          idx = (start + i) % n_subjects
          selected_subjects.append(subjects[idx])

        # Distribute samples_per_level evenly across selected subjects
        base_per_subject = self.samples_per_level // self.min_subjects_per_level
        remainder = self.samples_per_level % self.min_subjects_per_level

        for i, subj in enumerate(selected_subjects):
          n_samples = base_per_subject + (1 if i < remainder else 0)
          pool = pools[subj]

          for _ in range(n_samples):
            if len(pool) == 0:
              # Refill and reshuffle
              pool = list(self.level_subject_map[level][subj])
              if self.shuffle:
                rng.shuffle(pool)
              pools[subj] = pool
            batch.append(pool.pop())

      if len(batch) > 0:
        self.batches.append(batch)

    self.n_batches = len(self.batches)

  def __iter__(self):
    for batch in self.batches:
      yield batch

    # Update random state for next epoch
    self.random_state += 13
    self.initialize()

  def __len__(self):
    return self.n_batches


def smooth_labels_batch(gt_classes: torch.Tensor, num_classes: int, smoothing: float = 0.2) -> torch.Tensor:
  """
  Highly optimized implementation of neighbor-only label smoothing.
  
  Logic:
    - GT: 1.0 - smoothing
    - Neighbors: smoothing / 2
    - Edge Correction: If a neighbor is missing (e.g., gt=0), 
      the existing neighbor gets the full smoothing value.
  """
  B = gt_classes.size(0)
  device = gt_classes.device
  
  # 1. Initialize output tensor (Faster than creating empty and filling)
  out = torch.zeros((B, num_classes), device=device, dtype=torch.float32)
  
  # 2. Set Ground Truth Confidence
  # Uses advanced indexing: out[row_indices, col_indices] = value
  row_indices = torch.arange(B, device=device)
  out[row_indices, gt_classes] = 1.0 - smoothing
  
  # Pre-calculate smoothing values
  half_smooth = smoothing * 0.5
  
  # 3. Handle LEFT Neighbors (Indices: gt - 1)
  # Mask: Where does a left neighbor exist? (gt > 0)
  mask_left = gt_classes > 0
  if mask_left.any():
    # Compute indices for valid left neighbors
    idx_left = gt_classes[mask_left] - 1
    
    # Create value array filled with half_smooth
    vals_left = torch.full((mask_left.sum(),), half_smooth, device=device)
    
    # EDGE CASE OPTIMIZATION: 
    # If gt was the Max Class (e.g. 4), it has no right neighbor.
    # So the Left neighbor gets the FULL smoothing value.
    # We check the original gt values using the same mask
    is_right_edge = (gt_classes[mask_left] == num_classes - 1)
    vals_left[is_right_edge] = smoothing
    
    # Assign values in one go
    out[mask_left, idx_left] = vals_left

  # 4. Handle RIGHT Neighbors (Indices: gt + 1)
  # Mask: Where does a right neighbor exist? (gt < max)
  mask_right = gt_classes < num_classes - 1
  if mask_right.any():
    # Compute indices
    idx_right = gt_classes[mask_right] + 1
    
    # Create value array filled with half_smooth
    vals_right = torch.full((mask_right.sum(),), half_smooth, device=device)
    
    # EDGE CASE OPTIMIZATION:
    # If gt was 0, it has no left neighbor.
    # So the Right neighbor gets the FULL smoothing value.
    is_left_edge = (gt_classes[mask_right] == 0)
    vals_right[is_left_edge] = smoothing
    
    # Assign values in one go
    out[mask_right, idx_right] = vals_right

  return out
 
def get_dataset_and_loader(csv_path,root_folder_features,batch_size,shuffle_training_batch,is_training,dataset_type,concatenate_temporal,model,
                           label_smooth,soft_labels,is_coral_loss,stride_inside_window=1,num_clips_per_video=1,sample_frame_strategy=None,n_workers=None,backbone_dict=None,split_chunks=False,prefetch_factor=None,**kwargs):
  if dataset_type.value == CUSTOM_DATASET_TYPE.WHOLE.value:
    dataset_ = customDatasetWhole(csv_path,root_folder_features=root_folder_features,
                                  concatenate_temporal=concatenate_temporal,
                                  model=model,smooth_labels=label_smooth,
                                  concatenate_quadrants=kwargs['concatenate_quadrants'],
                                  soft_labels=soft_labels,
                                  consider_only_lasts_n_chunks=kwargs['consider_only_lasts_n_chunks'],
                                  xattn_mask=kwargs.get('xattn_mask',None),
                                  split_chunks=split_chunks,
                                  latent_coefficient=kwargs.get('latent_coefficient',0.0),
                                  coral_loss=is_coral_loss,
                                  feature_merge_type=kwargs.get('feature_merge_type',None),
                                  feature_merge_lambda=kwargs.get('feature_merge_lambda',None),
                                  feature_merge_orig=kwargs.get('feature_merge_orig',1))
    if 'caer' in csv_path.lower() or 'combined' in root_folder_features.lower():
      pin_memory = False
    else:
      pin_memory = False
    persistent_workers = True
    prefetch_factor = 2 if prefetch_factor is None else prefetch_factor
  elif dataset_type.value == CUSTOM_DATASET_TYPE.AGGREGATED.value:
    dataset_ = customDatasetAggregated(csv_path=csv_path,
                                        root_folder_features=root_folder_features,
                                        concatenate_temporal=concatenate_temporal,
                                        is_train=is_training,
                                        model=model,
                                        latent_coefficient=kwargs.get('latent_coefficient',0.0),
                                        consider_only_lasts_n_chunks=kwargs['consider_only_lasts_n_chunks'],
                                        concatenate_quadrants=kwargs['concatenate_quadrants'],
                                        smooth_labels=label_smooth,
                                        coral_loss=is_coral_loss,
                                        soft_labels=soft_labels)
    pin_memory = True
    persistent_workers = False
    prefetch_factor = 2 if prefetch_factor is None else prefetch_factor
    
  elif dataset_type.value == CUSTOM_DATASET_TYPE.BASE.value:
    if backbone_dict is not None:
      dataset_ = customDataset(path_dataset=root_folder_features,
                               model_type=backbone_dict['backbone'].model_type,
                              sample_frame_strategy=sample_frame_strategy,
                              num_clips_per_video=num_clips_per_video,
                              path_labels=csv_path,
                              stride_window=16,
                              clip_length=16,
                              stride_inside_window=stride_inside_window,
                              preprocess_align=False,
                              preprocess_frontalize=False,
                              consider_only_lasts_n_chunks=kwargs['consider_only_lasts_n_chunks'],
                              preprocess_crop_detection=False,
                              saving_folder_path_extracted_video=None,
                              video_labels=None,
                              backbone_dict=backbone_dict,
                              coral_loss=is_coral_loss,
                              soft_labels=soft_labels,
                              smooth_labels=label_smooth,
                              **kwargs['dict_augmented'])
      pin_memory = False
      persistent_workers = True
      prefetch_factor = 5 if prefetch_factor is None else prefetch_factor
    else:
      raise ValueError(f'backbone_dict must be provided for dataset_type: {dataset_type}')
  else:
    raise ValueError(f'Unknown dataset type: {dataset_type}. Choose one of {CUSTOM_DATASET_TYPE}')
  
  # Trainloader
  if is_training:
    if kwargs['sampler_loader_type'] == 'selective_augm':
      sampler = SelectiveAugmentationBatchSampler(df=dataset_.df,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle_training_batch,
                                                  n_keep_augmentations=kwargs['filtered_augm_n_keep'],
                                                  augmentation_strategy=kwargs['filtered_augm_strategy'],
                                                  root_folder_features=root_folder_features,
                                                  drop_last=False,
                                                  keep_original=kwargs.get('keep_original', 1.0))
      print(f'Use FilteredAugmentationBatchSampler with {n_workers} workers!\nPersistent workers: {persistent_workers} \nPin memory: {pin_memory}\nPrefetch factor: {prefetch_factor}')
    elif kwargs['sampler_loader_type'] == 'augmented_only':
      sampler = AugmentedOnlyBatchSampler(df=dataset_.df,
                                          batch_size=batch_size,
                                          augmentations=kwargs['sampler_augmented_only_types'],
                                          shuffle=shuffle_training_batch,
                                          balance_batch=False,
                                          root_folder_features=root_folder_features)
      print(f'Use AugmentedOnlyBatchSampler with {n_workers} workers!\nPersistent workers: {persistent_workers} \nPin memory: {pin_memory}\nPrefetch factor: {prefetch_factor}')
    elif kwargs['sampler_loader_type'] == 'balanced':
      sampler = balancedBatchSampler(df=dataset_.df, 
                                      batch_size=batch_size,
                                      shuffle=shuffle_training_batch)
      print(f'Use balancedBatchSampler with {n_workers} workers!\nPersistent workers: {persistent_workers} \nPin memory: {pin_memory}\nPrefetch factor: {prefetch_factor}')
    elif kwargs['sampler_loader_type'] == 'subject_batch':
      sampler = SubjectBatchSampler(
        df=dataset_.df,
        batch_size=batch_size,
        min_subjects_per_level=kwargs.get('min_subjects_per_level', 3),
        include_augmented=True,
        require_all_levels=True,
        shuffle=shuffle_training_batch,
        random_state=42
      )
      print(f'Use SubjectBatchSampler with {n_workers} workers!\nPersistent workers: {persistent_workers}\nPin memory: {pin_memory}\nPrefetch factor: {prefetch_factor}')
    elif kwargs['sampler_loader_type'] == 'standard':
      sampler = None
      print(f'Use standard DataLoader with {n_workers} workers!\nPersistent workers: True')
    else:
      raise ValueError(f"Unknown sampler_loader_type: {kwargs['sampler_loader_type']}. Choose from 'selective_augm', 'augmented_only', 'balanced', 'subject_batch', 'standard'.")
    if sampler is not None:
      loader_ = DataLoader(dataset=dataset_,
                          batch_sampler=sampler,
                          collate_fn=dataset_._custom_collate,
                          # batch_size=1,
                          num_workers=n_workers,
                          persistent_workers=persistent_workers,
                          prefetch_factor=prefetch_factor,
                          pin_memory=pin_memory)
    else:
      loader_ = DataLoader(dataset=dataset_,
                           batch_size=batch_size,
                           shuffle=shuffle_training_batch,
                           collate_fn=dataset_._custom_collate,
                           num_workers=n_workers,
                           persistent_workers=persistent_workers,
                           prefetch_factor=prefetch_factor,
                           pin_memory=pin_memory)
  # Eval loader
  else:
    if n_workers > 1:
      nr_batches = len(dataset_.df) // batch_size + 1
      n_workers = min(n_workers, nr_batches)
      loader_ = DataLoader(dataset=dataset_,
                           batch_size=batch_size,
                           shuffle=False,
                           collate_fn=dataset_._custom_collate,
                           num_workers=n_workers,
                           prefetch_factor=prefetch_factor,
                           persistent_workers=persistent_workers,
                           pin_memory=pin_memory)
    else:
      loader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False,collate_fn=dataset_._custom_collate)
  return dataset_,loader_

  
def _get_element(dict_data,df,idx,dataset_type,embedding_reduction,consider_only_lasts_n_chunks,latent_coefficient,is_quadrant=False,concatenate_quadrants=False,xattn_mask=None):
  pid = os.getpid()
  # if pid not in helper.time_profile_dict:
  #   helper.time_profile_dict[pid] = mp.Manager().dict()
  with profile_workers(f'{pid}_get_element_time',helper.time_profiling_enabled,helper.time_profile_dict):
    csv_row = df.iloc[idx]
    sample_id = csv_row['sample_id']
    csv_label = torch.tensor(csv_row['class_id'])
    subject_id = torch.tensor([csv_row['subject_id']])
    # If AGGREGATED or WHOLE dataset, sample_id is the real sample_id
    # start_time_mask = time.perf_counter()

    mask = None
    if dataset_type in [CUSTOM_DATASET_TYPE.AGGREGATED]:
      if helper.is_latent_basic_augmentation(sample_id) or helper.is_latent_masking_augmentation(sample_id): # latent augm.
        mask = dict_data['list_sample_id'] == ((sample_id - 1) % helper.step_shift + 1) # to get the real sample_id
      elif not is_quadrant and dataset_type == CUSTOM_DATASET_TYPE.AGGREGATED: # base aggregated dataset
        mask = dict_data['list_sample_id'] == sample_id
    elif dataset_type == CUSTOM_DATASET_TYPE.BASE: # BASE dataset
      mask = dict_data['list_sample_id'] == sample_id

  if mask is not None and mask.sum() == 0:
    print(f"Sample ID {sample_id} not found in the dataset.")
    
  with profile_workers(f'{pid}_selection_time',helper.time_profiling_enabled,helper.time_profile_dict):
    if dataset_type != CUSTOM_DATASET_TYPE.AGGREGATED:
      features = dict_data['features'] # shape [nr_chunks,T,S,S,Emb] 
      if consider_only_lasts_n_chunks is not None:
        features = features[-consider_only_lasts_n_chunks:] # consider only the last n chunks
      labels = csv_label.repeat(features.shape[0])
      subject_id = subject_id.repeat(labels.shape[0])
      
    else:
      features = dict_data['features'][mask]
      if consider_only_lasts_n_chunks is not None:
        features = features[-consider_only_lasts_n_chunks:] # consider only the last n chunks
      labels = csv_label.repeat(features.shape[0])
      subject_id = subject_id.repeat(labels.shape[0])

  with profile_workers(f'{pid}_embedding_reduction_time',helper.time_profiling_enabled,helper.time_profile_dict):
    if embedding_reduction != EMBEDDING_REDUCTION.NONE:
      features = torch.mean(features,dim=embedding_reduction.value,keepdim=True) 
    
  ###### LATENT AUGMENTATIONS ######
  with profile_workers(f'{pid}_latent_augm_time',helper.time_profiling_enabled,helper.time_profile_dict):
    if helper.is_latent_basic_augmentation(sample_id): # latent_basic_augmentation: invert the features + add gaussian noise
      gaussian_noise = torch.randn_like(features) * 0.1 # mean 0, std 0.1
      features = (-1 * features) + gaussian_noise

    # random masking patches
    elif helper.is_latent_masking_augmentation(sample_id):
      B,T,S,S,C = features.shape # B->nr_frames, T->temporal, S->spatial, C->emb_dim
      if S == 1: # mask Temporal
        mask_grid = torch.rand(T,dtype=features.dtype) < latent_coefficient # mask % of the temporal, if 0 all unmasked
        mask_grid = mask_grid.view(1,T,1,1,1)
      else: # mask Spatial
        mask_grid = torch.rand(S,S,dtype=features.dtype) < latent_coefficient # mask % of the grid, if 0 all unmasked
        mask_grid = mask_grid.view(1,1,S,S,1)
      features = features.masked_fill(mask_grid, 0.0) # set to zero the masked values
  
  ##################################
  ## shuffle temporal dimension of the features, dim 0 and 1
  # def shuffle_dim(x, dim):
  #   idx = torch.randperm(x.size(dim))
  #   return x.index_select(dim, idx)
  # features = shuffle_dim(features, dim=1) # shuffle temporal dimension
  # features = shuffle_dim(features, dim=0) # shuffle chunk dimension
  ######
  return {
      'features': features,     # [8,8,1,1,768]-> [chunks,Temporal,Space,Space,Emb]
      'labels': labels,         # [8]
      'subject_id': subject_id, # [8]
      'sample_id': sample_id    # int
  }
    
from contextlib import contextmanager
@contextmanager
def profile_workers(name,enabled,profile_dict):
  if enabled:
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    profile_dict[name] = profile_dict.get(name,0) + (end_time - start_time)
  else:
    yield
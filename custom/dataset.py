import warnings
import re
import torch
import pandas as pd
import os
import numpy as np
import torch
import cv2
import time
import torch.nn as nn
import torch.utils
from custom.backbone import BackboneBase
from custom.helper import CUSTOM_DATASET_TYPE, SAMPLE_FRAME_STRATEGY, EMBEDDING_REDUCTION, MODEL_TYPE
import custom.helper as helper
import custom.tools as tools
# from torch.utils.data import  Sampler
from torchvision.transforms import v2
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
from custom.targets import TargetSpec, resolve_target_spec


def _target_spec(spec, values):
  if isinstance(spec, TargetSpec):
    return spec
  if isinstance(spec, dict):
    return TargetSpec.from_metadata(spec)
  return TargetSpec.from_values(values)


def _class_bin_series(df, class_bins=None):
  values = class_bins if class_bins is not None else TargetSpec.from_values(df['class_id']).to_bins(df['class_id'])
  if len(values) != len(df):
    raise ValueError("class_bins must contain one value per dataframe row")
  return pd.Series(np.asarray(values, dtype=np.int64), index=df.index)


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
      backbone_type='video',
      image_mean=None,
      image_std=None,
      target_spec=None,
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
      
    if not os.path.exists(path_dataset):
      # Raw videos are optional when training from precomputed features (--ffsp);
      # the path is still used to infer the dataset (see helper.set_step_shift).
      print(f"WARNING: dataset path {path_dataset} does not exist; only precomputed features can be used.")
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

    # Image backbones (e.g. ViT) read single images instead of videos and may require a
    # backbone-specific normalization (defaults below match the video ImageNet stats).
    self.is_image = backbone_type == 'image'
    self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
    self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]

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
    self.df = self.video_labels
    self.target_spec = _target_spec(target_spec, self.video_labels['class_id'])
    self.class_bins = self.target_spec.to_bins(self.video_labels['class_id'])
    self.num_classes = self.target_spec.bin_count
        
    # Get unique subjects and classes
    tmp = tools.get_unique_subjects_and_classes(self.video_labels, self.target_spec)
    self.total_subjects, self.total_classes = len(tmp[0]), self.target_spec.bin_count
    
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
    self.num_classes = self.target_spec.bin_count
  
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
    if not os.path.exists(landmarks_path):
      # Optional asset (BioVid-derived); not needed when training from precomputed features.
      self.reference_landmarks = None
      return
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
  def preprocess_images(tensors,crop_size=None,to_visualize=False,get_params=False,h_flip=False,color_jitter=False,rotation=False,spatial_shift=False,zoom=False,gaussian_smooth=False,gaussian_sigma_min=None,gaussian_sigma_max=None,gaussian_kernel_size=None,framepermute=False,framepermute_seed=None,image_mean=None,image_std=None):
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
      image_mean (list, optional):       Per-channel normalization mean. Defaults to ImageNet stats
                                         ([0.485, 0.456, 0.406]) used by the video backbones.
      image_std (list, optional):        Per-channel normalization std. Defaults to ImageNet stats
                                         ([0.229, 0.224, 0.225]).

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
    image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
    image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]
    
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

    pid = os.getpid()
    if self.is_image:
      # Image backbone: read a single image as a 1-clip, 1-frame "video".
      frames_list = self._read_image(video_path)  # [1, 1, H, W, C]
      list_indices = torch.tensor([[0]], dtype=torch.int32)
    else:
      # Open video and get properties
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
                                                    framepermute=self.framepermute,
                                                    image_mean=self.image_mean,
                                                    image_std=self.image_std) # [B,C,H,W]
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
                                                    gaussian_kernel_size=self.gaussian_kernel_size,
                                                    image_mean=self.image_mean,
                                                    image_std=self.image_std)
    preprocessed_tensors = preprocessed_tensors.reshape(nr_clips, nr_frames, *preprocessed_tensors.shape[1:]) # [nr_clips, nr_frames, C, H, W]
    # if nr_clips == 1:
    #   preprocessed_tensors = preprocessed_tensors.unsqueeze(0) # in case of single frame
    preprocessed_tensors = preprocessed_tensors.permute(0,2,1,3,4) # [B=nr_clips, T=nr_frames, C, H, W] -> [B, C, T, H, W]
    helper.time_profile_dict[f'{pid}_preprocess_video_time'] = helper.time_profile_dict.get(f'{pid}_preprocess_video_time', 0) + (time.perf_counter() - time_preprocess_video)
    
    # Create metadata tensors
    time_metadata = time.perf_counter()
    sample_id = torch.full((nr_clips,), int(self.video_labels.iloc[idx]['sample_id']), dtype=torch.int32)
    labels = torch.full((nr_clips,), float(self.video_labels.iloc[idx]['class_id']), dtype=torch.float32)
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
  
  def _read_image(self, image_path):
    """
    Read a single image as a one-clip, one-frame "video" tensor.

    Mirrors the `[nr_clips, nr_frames, H, W, C]` uint8 contract of
    `_read_video_cv2_and_process` so the rest of `__standard_getitem__` is shared with the
    video path.

    Args:
      image_path (str): Path to the image file on disk.

    Returns:
      torch.Tensor: RGB image. Shape: (1, 1, H, W, C), dtype uint8.

    Raises:
      ValueError: If the image cannot be read.
    """
    frame = cv2.imread(image_path)  # BGR, [H, W, C]
    if frame is None:
      raise ValueError(f"Image {image_path} cannot be read. Please check the file.")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame, dtype=torch.uint8)
    # [H, W, C] -> [nr_clips=1, nr_frames=1, H, W, C]
    return frame.unsqueeze(0).unsqueeze(0)

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
    if self.model_type in (MODEL_TYPE.VJEPA_v2_L_fpc64_256, MODEL_TYPE.VJEPA_v2_G_fpc64_384):
      # HF transformers VJEPA2 wrapper consumes [B, T, C, H, W]; local encoders (incl. vjepa2_1_B) take [B, C, T, H, W].
      data = data.permute(0,2,1,3,4) # → [chunks, clip_length, C, H, W]
    # Combine metadata from all samples
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    path = np.concatenate([item['path'] for item in batch])
    subject_id = torch.cat([item['subject_id'] for item in batch], dim=0)
    sample_id = torch.cat([item['sample_id'] for item in batch], dim=0)
    list_frames = torch.cat([item['frame_list'] for item in batch], dim=0).squeeze(0)

    return data, labels, subject_id, sample_id, path, list_frames

  
  def _custom_collate(self, batch):
    time_custom_collate = time.perf_counter()
    labels = torch.stack([item['labels'][0] for item in batch], dim=0)
    class_targets = self.target_spec.to_bin_tensor(labels)
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
      labels = smooth_labels_batch(gt_classes=class_targets, num_classes=self.num_classes, smoothing=self.smooth_labels)
    elif self.soft_labels != None:
      labels = self.soft_labels[class_targets].to(torch.float32) # soft labels
    elif self.coral_loss:
      labels = levels_from_labelbatch(class_targets, self.num_classes, dtype=torch.float32)
    
    pid = os.getpid()
    helper.time_profile_dict[f'{pid}_custom_collate_time'] = helper.time_profile_dict.get(f'{pid}_custom_collate_time', 0) + (time.perf_counter() - time_custom_collate)
    # [B, C, T, H, W], padding applied in the model forward pass
    return {'x':data, 
           'key_padding_mask': None,
           'lengths': lengths, 
           'class_targets': class_targets,
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
    return np.unique(self.class_bins, return_counts=True)[1]
  def get_unique_classes(self):
    return np.unique(self.class_bins)

  def generate_csv_augmented(self,original_csv_path,out_csv_path,dict_augmentation,stratified_training=False):
    return helper.generate_csv_augmented(original_csv_path=original_csv_path,
                           out_csv_path=out_csv_path,
                           dict_augmentation=dict_augmentation,
                           stratified_training=stratified_training)

class customDatasetAggregated(torch.utils.data.Dataset):
  def __init__(self,root_folder_features,concatenate_quadrants,model,is_train,csv_path,smooth_labels,target_spec=None):
    self.root_folder_feature = root_folder_features
    self.csv_path = csv_path
    helper.set_step_shift(root_folder_features)
    print(f"Using step shift = {helper.step_shift} for dataset {root_folder_features}")
    self.concatenate_quadrants = concatenate_quadrants
    self.df = pd.read_csv(csv_path,sep='\t')
    self.is_quadrant = True if 'combined' in root_folder_features else False
    if not is_train:
      # Keep only original samples if validation/test (sample_id<=8700)
      filter_mask = self.df['sample_id'] <= helper.step_shift
      
      # Save the filtered DataFrame to a new CSV file
      self.df = self.df[filter_mask]
      self.df.to_csv(csv_path,sep='\t',index=False)
      print(f"Filtered DataFrame saved to {csv_path}")
    self.target_spec = _target_spec(target_spec, self.df['class_id'])
    self.class_bins = self.target_spec.to_bins(self.df['class_id'])
    self.num_classes = self.target_spec.bin_count
    # Save mask to use to filter helper.dict_data
    # self.mask = np.isin(helper.dict_data['list_sample_id'],self.df['sample_id'].to_list())
    self.model = model
    self.instance_model_name = tools.get_instace_model_name(model)
    self.smooth_labels = smooth_labels

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
          if not dict_data['list_labels'].dtype.is_floating_point:
            dict_data['list_labels'] = dict_data['list_labels'].float()
          csv_labels = torch.full((nr_values,), float(real_csv_label), dtype=torch.float32)
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
                          embedding_reduction=self.model.embedding_reduction)
    raise NotImplementedError("Batch loading is not implemented for customDatasetAggregated")
  
  def _custom_collate(self,batch):
    return _custom_collate(batch,
                           self.instance_model_name,
                           self.model,
                           self.num_classes,
                           self.smooth_labels,
                           self.concatenate_quadrants,
                           target_spec=self.target_spec)
  
  def get_unique_subjects(self):
    return np.sort(self.df['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.df['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.class_bins, return_counts=True)[1]
  def get_unique_classes(self):
    return np.unique(self.class_bins)
  def get_all_sample_ids(self):
    return self.df['sample_id'].tolist()
  
  def generate_csv_augmented(self,original_csv_path,out_csv_path,dict_augmentation,stratified_training=False):
    return helper.generate_csv_augmented(original_csv_path=original_csv_path,
                           out_csv_path=out_csv_path,
                           dict_augmentation=dict_augmentation,
                           stratified_training=stratified_training)


class customDatasetWhole(torch.utils.data.Dataset):
  def __init__(self,csv_path,root_folder_features,concatenate_quadrants,model,smooth_labels,target_spec=None):
    self.csv_path = csv_path
    helper.set_step_shift(root_folder_features)
    print(f"\n CustomDatasetWhole: Using step shift = {helper.step_shift} for dataset {root_folder_features} \n")
    self.root_folder_features = root_folder_features
    self.df = pd.read_csv(csv_path,sep='\t',dtype={'sample_name':str,'subject_name':str}) # subject_id, subject_name, class_id, class_name, sample_id, sample_name
    self.target_spec = _target_spec(target_spec, self.df['class_id'])
    self.class_bins = self.target_spec.to_bins(self.df['class_id'])
    # count subject_id uniqueness
    unique_subject_ids, counts = np.unique(self.df['subject_id'], return_counts=True)
    print(f"\nUnique subject_ids: {len(unique_subject_ids)}, Total entries: {len(self.df)}")
    self.concatenate_quadrants = concatenate_quadrants
    self.model = model
    self.is_quadrant = True if 'combined' in root_folder_features else False
    self.instance_model_name = tools.get_instace_model_name(model)
    self.smooth_labels = smooth_labels
    self.num_classes = self.target_spec.bin_count
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
    # look for folders matching the pattern {base_name}_{aug_type} or {base_name}_{aug_type}$N
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
    if sample_id > helper.step_shift:
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

    return _get_element(dict_data=features,
                        df=self.df,idx=idx,
                        dataset_type=CUSTOM_DATASET_TYPE.WHOLE, 
                        is_quadrant=self.is_quadrant,
                        embedding_reduction=self.model.embedding_reduction)

  def __getitem__(self,idx):
    if isinstance(idx,int):
      return self._load_element(idx=idx)
    raise NotImplementedError("Batch loading is not implemented for customDatasetWhole")

  def _custom_collate(self,batch):
    return _custom_collate(batch,self.instance_model_name,
                           self.model,
                           self.num_classes,
                           self.smooth_labels,
                           self.concatenate_quadrants,
                           target_spec=self.target_spec)

  def get_unique_subjects(self):
    return np.sort(self.df['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.df['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.class_bins, return_counts=True)[1]
  def get_unique_classes(self,return_all=False):
    if not return_all:
      return np.unique(self.class_bins)
    return np.arange(self.target_spec.bin_count)
  def get_all_sample_ids(self):
    return self.df['sample_id'].tolist()

  def generate_csv_augmented(self,original_csv_path,out_csv_path,dict_augmentation,stratified_training=False):
    return helper.generate_csv_augmented(original_csv_path=original_csv_path,
                           out_csv_path=out_csv_path,
                           dict_augmentation=dict_augmentation,
                           stratified_training=stratified_training)
  
def highly_optimized_custom_collate(batch, pid, is_training, smooth_labels=0.0,
  num_classes=None,concatenate_quadrants=False,target_spec=None):
  """Stack per-sample chunk features into [B, tokens, emb] (padding variable lengths) and build the targets."""
  batch_size = len(batch)
  try:
    # Stack all features at once when they have the same shape
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
    else:
      features = features.view(batch_size, -1, features.shape[-1]).to(torch.float32)
    lengths = None
  except (ValueError, RuntimeError):
    # Fall back to the standard approach for variable shapes
    features = []
    lengths = []

    for sample in batch:
      feat = sample['features'] # [quadrants*nr_chunks,T,S,S,Emb]
      QnC, T, S1, S2, emb = feat.shape
      if concatenate_quadrants:
        processed_feat = feat.view(4, QnC//4, T, S1, S2, emb).transpose(0,1) # [4=quadrants,QnC/4,T,S,S,Emb] -> [QnC/4,4,T,S,S,Emb]
        processed_feat = processed_feat.permute(0,2,3,4,1,5).contiguous().view(
          QnC//4*T*S1*S2,4*emb).to(torch.float32)
      else:
        processed_feat = feat.view(-1, feat.shape[-1]).to(torch.float32)
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
    key_padding_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1)

  # Vectorized extraction of labels, subject_ids, and sample_ids
  labels = torch.tensor([sample['labels'][0] for sample in batch], dtype=torch.float32)
  spec = _target_spec(target_spec, labels.numpy())
  class_targets = spec.to_bin_tensor(labels)
  subject_id = torch.tensor([sample['subject_id'][0] for sample in batch], dtype=torch.int32)
  sample_id = torch.tensor([sample['sample_id'] for sample in batch], dtype=torch.int32)
  
  # Handle label smoothing when in training mode
  if is_training and smooth_labels > 0.0:
    labels = smooth_labels_batch(gt_classes=class_targets, num_classes=num_classes,
      smoothing=smooth_labels)
    
  return {
    'features': features,
    'lengths': lengths,
    'labels': labels,
    'class_targets': class_targets,
    'subject_id': subject_id,
    'sample_id': sample_id,
    'key_padding_mask': key_padding_mask,
  }


def _custom_collate(batch,instance_model_name,model,num_classes,smooth_labels,concatenate_quadrants,target_spec=None):
  pid = os.getpid()
  with profile_workers(f'{pid}_collate_preprocess_time',helper.time_profiling_enabled,helper.time_profile_dict):
    dict_res = highly_optimized_custom_collate(batch=batch,
                                              pid=pid,
                                              is_training = model.training,
                                              smooth_labels=smooth_labels,
                                              concatenate_quadrants=concatenate_quadrants,
                                              num_classes=num_classes,
                                              target_spec=target_spec)
  if instance_model_name != helper.INSTANCE_MODEL_NAME.AttentiveClassifier:
    raise NotImplementedError(f"Instance model {instance_model_name} not implemented for collate function.")
  return {'x':dict_res['features'], 'key_padding_mask': dict_res['key_padding_mask'], 'class_targets': dict_res['class_targets']},\
          dict_res['labels'],\
          dict_res['subject_id'],\
          dict_res['sample_id']


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
    undersample_max_per_class: Per-class cap on base-id groups per epoch (undersampling). 0 = disabled,
                             -1 = cap every class at the minority class group count, N > 0 = absolute cap.
                             The subset of groups kept for over-represented classes is redrawn each epoch.
  """
  def __init__(self, df, batch_size, shuffle, n_keep_augmentations, augmentation_strategy, root_folder_features=None, drop_last=False, keep_original=1.0, undersample_max_per_class=0,class_bins=None):
    """
    Args:
      df:                      DataFrame with at least 'sample_id' and 'class_id' columns; index used as dataset indices.
      batch_size:              Number of samples per batch.
      shuffle:                 Whether to shuffle the selected indices before batching.
      n_keep_augmentations:    Maximum number of augmented variants to keep per original sample.
      augmentation_strategy:   1 = pick random augmented samples; 2 = pick random augmentation types globally.
      root_folder_features:    Optional path to feature folder, used to discover available augmentation types.
      drop_last:               Whether to drop the last incomplete batch.
      keep_original:           Fraction of original samples to keep per epoch (0-1). When < 1, a random subset
                               of originals is selected each epoch; their augmentations are excluded. Non-selected
                               originals are excluded but their augmentations remain available via the strategy.
      undersample_max_per_class: Per-class cap on base-id groups per epoch. 0 = disabled, -1 = cap every class
                               at the minority class group count, N > 0 = absolute cap. Over-represented classes
                               get a fresh random subset of groups each epoch; classes at or below the cap keep
                               all their groups. Applied before the keep_original split.
    """
    if not 0 <= keep_original <= 1:
      raise ValueError(f"keep_original must be between 0 and 1, got {keep_original}")
    if undersample_max_per_class < -1:
      raise ValueError(f"undersample_max_per_class must be >= -1, got {undersample_max_per_class}")
    self.keep_original = keep_original
    self.undersample_max_per_class = undersample_max_per_class
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.n_keep_augmentations = n_keep_augmentations
    self.augmentation_strategy = augmentation_strategy # 1: random samples, 2: random types
    self.drop_last = drop_last
    
    self.log_augmented_usage = {idx: 0 for idx in df.index}
    self.base_id_groups = {} # base_id -> {'original': idx, 'augmented': {type: idx}}
    self.class_of_base_id = {} # base_id -> class_id (originals and augmentations share the class)
    self.available_augmentation_types = set()
    
    # Get available augmentations from folder if possible
    if root_folder_features:
      try:
        self.available_augmentation_types = set(helper.get_augmentation_availables(root_folder_features))
        print(f"\n\nAvailable augmentation types retrieved from folder: {self.available_augmentation_types}")
      except Exception as e:
        print(f"Warning: Could not get available augmentations from folder: {e}")
    
    bin_series = _class_bin_series(df, class_bins)
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
      self.class_of_base_id[base_id] = int(bin_series.loc[idx])

    self.available_augmentation_types = sorted(list(self.available_augmentation_types))

    self.base_ids_per_class = {} # class_id -> [base_id, ...]
    for base_id, class_id in self.class_of_base_id.items():
      self.base_ids_per_class.setdefault(class_id, []).append(base_id)

    if self.undersample_max_per_class == -1:
      self.effective_undersample_cap = min(len(b) for b in self.base_ids_per_class.values())
    elif self.undersample_max_per_class > 0:
      self.effective_undersample_cap = self.undersample_max_per_class
    else:
      self.effective_undersample_cap = None # undersampling disabled

    self._calculate_length()

  def _calculate_length(self):
    """
    Estimates the total number of samples per epoch to compute the number of batches.

    When undersampling is active, contributions are estimated per class: each class's expected
    total is scaled by the fraction of its groups surviving the cap (the actual subset is redrawn
    randomly each epoch, so this is exact in expectation but the true count can vary slightly when
    augmentation counts differ between groups of the same class).
    When keep_original < 1, only a fraction of groups contribute their original (no augmentations),
    and the remaining groups contribute only augmentations (no original).

    Returns:
      None. Sets self.n_batches as a side effect.
    """
    total_samples = 0
    n_groups_with_original = sum(1 for g in self.base_id_groups.values() if g['original'] is not None)

    # Per-class selection ratios (1.0 for every class when undersampling is disabled)
    if self.effective_undersample_cap is not None:
      class_ratios = {class_id: min(self.effective_undersample_cap, len(b)) / len(b)
                      for class_id, b in self.base_ids_per_class.items()}
    else:
      class_ratios = {class_id: 1.0 for class_id in self.base_ids_per_class}

    if self.keep_original >= 1.0:
      expected_total = 0.0
      for class_id, base_ids in self.base_ids_per_class.items():
        class_total = 0
        for base_id in base_ids:
          group = self.base_id_groups[base_id]
          if group['original'] is not None:
            class_total += 1
          class_total += min(self.n_keep_augmentations, len(group['augmented']))
        expected_total += class_total * class_ratios[class_id]
      total_samples = int(round(expected_total))
    else:
      n_selected_with_original = int(round(sum(
        class_ratios[class_id] * sum(1 for bid in base_ids if self.base_id_groups[bid]['original'] is not None)
        for class_id, base_ids in self.base_ids_per_class.items())))
      n_original_kept = int(self.keep_original * n_selected_with_original)
      # Original-only groups contribute 1 sample each (no augmentations)
      total_samples += n_original_kept
      # Augmentation-only groups contribute augmentations only
      n_augment_groups = n_selected_with_original - n_original_kept
      if n_augment_groups > 0 and not any(g['augmented'] for g in self.base_id_groups.values()):
        raise FileNotFoundError(
          f'keep_original={self.keep_original} < 1 needs augmented samples, but none were found. '
          'Augmented embeddings must sit next to the --ffsp folder as sibling folders named '
          '<ffsp>_<augmentation> (e.g. <ffsp>_jitter, <ffsp>_jitter$0); see README.md.')
      avg_aug = np.mean([min(self.n_keep_augmentations, len(g['augmented']))
                         for g in self.base_id_groups.values() if g['augmented']]) if n_augment_groups > 0 else 0
      total_samples += int(n_augment_groups * avg_aug)

    if self.drop_last:
      self.n_batches = total_samples // self.batch_size
    else:
      self.n_batches = (total_samples + self.batch_size - 1) // self.batch_size

  def __len__(self):
    return self.n_batches

  def _select_undersampled_base_ids(self):
    """
    Selects the base_ids to use for the current epoch according to the per-class undersampling cap.

    For each class, if the number of base-id groups exceeds the effective cap, a random subset of
    cap groups is drawn without replacement (fresh draw on every call, i.e. every epoch); otherwise
    all groups of that class are kept. When undersampling is disabled, all base_ids are returned.

    Returns:
      List of selected base_ids.
    """
    if self.effective_undersample_cap is None:
      return list(self.base_id_groups.keys())
    selected_base_ids = []
    for base_ids in self.base_ids_per_class.values():
      if len(base_ids) > self.effective_undersample_cap:
        selected_base_ids.extend(np.random.choice(base_ids, size=self.effective_undersample_cap, replace=False).tolist())
      else:
        selected_base_ids.extend(base_ids)
    return selected_base_ids

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

    When undersampling is active, a per-class random subset of base_ids (at most the effective cap
    per class) is drawn first; only the surviving groups contribute samples. The subset changes each epoch.
    When keep_original < 1, a random subset of the surviving base_ids is selected as "original-only"
    (their augmentations are excluded). The remaining base_ids become "augmentation-only" (their originals
    are excluded, augmentations selected via the configured strategy). The subset changes each epoch.

    Returns:
      Iterator of lists of dataset indices (batches).
    """
    selected_indices = []
    all_base_ids = self._select_undersampled_base_ids()

    if self.keep_original < 1.0:
      base_ids_with_original = [bid for bid in all_base_ids if self.base_id_groups[bid]['original'] is not None]
      n_to_keep = int(self.keep_original * len(base_ids_with_original))
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
 
def get_dataset_and_loader(csv_path,root_folder_features,batch_size,shuffle_training_batch,is_training,dataset_type,model,
                           label_smooth,n_workers=None,prefetch_factor=None,**kwargs):
  """Dataset and DataLoader over precomputed embeddings (WHOLE: one file per video, AGGREGATED: one file)."""
  target_values = pd.read_csv(csv_path, sep='\t')['class_id']
  runtime_target_spec = resolve_target_spec(
    metadata=kwargs.get('target_spec'),
    values=target_values,
    normalize=bool(kwargs.get('normalize_labels', 0)),
    legacy_max_label=kwargs.get('max_label'),
  )
  if dataset_type.value == CUSTOM_DATASET_TYPE.WHOLE.value:
    dataset_ = customDatasetWhole(csv_path,root_folder_features=root_folder_features,
                                  model=model,smooth_labels=label_smooth,
                                  concatenate_quadrants=kwargs['concatenate_quadrants'],
                                  target_spec=runtime_target_spec)
    pin_memory = False
    persistent_workers = True
    prefetch_factor = 2 if prefetch_factor is None else prefetch_factor
  elif dataset_type.value == CUSTOM_DATASET_TYPE.AGGREGATED.value:
    dataset_ = customDatasetAggregated(csv_path=csv_path,
                                        root_folder_features=root_folder_features,
                                        is_train=is_training,
                                        model=model,
                                        concatenate_quadrants=kwargs['concatenate_quadrants'],
                                        smooth_labels=label_smooth,
                                        target_spec=runtime_target_spec)
    pin_memory = True
    persistent_workers = False
    prefetch_factor = 2 if prefetch_factor is None else prefetch_factor
  else:
    raise ValueError(f'Training needs precomputed embeddings; unsupported dataset type: {dataset_type}')
  
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
                                                  keep_original=kwargs.get('keep_original', 1.0),
                                                  undersample_max_per_class=kwargs.get('undersample_max_per_class', 0),
                                                  class_bins=dataset_.class_bins)
      print(f'Use FilteredAugmentationBatchSampler with {n_workers} workers!\nPersistent workers: {persistent_workers} \nPin memory: {pin_memory}\nPrefetch factor: {prefetch_factor}')
      loader_ = DataLoader(dataset=dataset_,
                          batch_sampler=sampler,
                          collate_fn=dataset_._custom_collate,
                          num_workers=n_workers,
                          persistent_workers=persistent_workers,
                          prefetch_factor=prefetch_factor,
                          pin_memory=pin_memory)
    elif kwargs['sampler_loader_type'] == 'standard':
      print(f'Use standard DataLoader with {n_workers} workers!\nPersistent workers: True')
      loader_ = DataLoader(dataset=dataset_,
                           batch_size=batch_size,
                           shuffle=shuffle_training_batch,
                           collate_fn=dataset_._custom_collate,
                           num_workers=n_workers,
                           persistent_workers=persistent_workers,
                           prefetch_factor=prefetch_factor,
                           pin_memory=pin_memory)
    else:
      raise ValueError(f"Unknown sampler_loader_type: {kwargs['sampler_loader_type']}. Choose from 'selective_augm', 'standard'.")
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

  
def _get_element(dict_data,df,idx,dataset_type,embedding_reduction,is_quadrant=False):
  pid = os.getpid()
  with profile_workers(f'{pid}_get_element_time',helper.time_profiling_enabled,helper.time_profile_dict):
    csv_row = df.iloc[idx]
    sample_id = csv_row['sample_id']
    csv_label = torch.tensor(csv_row['class_id'], dtype=torch.float32)
    subject_id = torch.tensor([csv_row['subject_id']])
    # AGGREGATED: select this sample's rows from the shared tensors; WHOLE: dict_data is this sample
    mask = None
    if dataset_type == CUSTOM_DATASET_TYPE.AGGREGATED and not is_quadrant:
      mask = dict_data['list_sample_id'] == sample_id

  if mask is not None and mask.sum() == 0:
    print(f"Sample ID {sample_id} not found in the dataset.")
    
  with profile_workers(f'{pid}_selection_time',helper.time_profiling_enabled,helper.time_profile_dict):
    if dataset_type != CUSTOM_DATASET_TYPE.AGGREGATED:
      features = dict_data['features'] # shape [nr_chunks,T,S,S,Emb] 
    else:
      features = dict_data['features'][mask]
    labels = csv_label.repeat(features.shape[0])
    subject_id = subject_id.repeat(labels.shape[0])

  with profile_workers(f'{pid}_embedding_reduction_time',helper.time_profiling_enabled,helper.time_profile_dict):
    if embedding_reduction != EMBEDDING_REDUCTION.NONE:
      features = torch.mean(features,dim=embedding_reduction.value,keepdim=True) 
    
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

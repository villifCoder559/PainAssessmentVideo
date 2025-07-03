import warnings
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
from custom.helper import CUSTOM_DATASET_TYPE, SAMPLE_FRAME_STRATEGY
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
      path_labels=None,
      stride_window=16,
      clip_length=16,
      stride_inside_window=1,
      preprocess_align=False,
      preprocess_frontalize=False,
      preprocess_crop_detection=False,
      saving_folder_path_extracted_video=None,
      video_labels=None,
      flip_horizontal=False,
      color_jitter=None,
      rotation=None,
      backbone_dict=None,
      image_resize_w=224,
      image_resize_h=224,
      smooth_labels= 0.0,
      video_extension = '.mp4'
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
    if backbone_dict is not None and not isinstance(backbone_dict['backbone'], BackboneBase):
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
    
    # Preprocessing options
    self.preprocess_align = preprocess_align
    self.preprocess_frontalize = preprocess_frontalize
    self.preprocess_crop_detection = preprocess_crop_detection
    self.h_flip = flip_horizontal
    self.color_jitter = color_jitter
    self.rotation = rotation
    self.smooth_labels = smooth_labels
    
    # if rotation is not None:
    #   warnings.warn('The rotation is not implemented yet')
    # Image dimensions and channels
    self.image_resize_w = image_resize_w
    self.image_resize_h = image_resize_h
    self.image_channels = 3
    
    # Set sampling strategy method based on parameter
    self._set_sampling_strategy(sample_frame_strategy)
    
    # Set video labels
    if video_labels is None and path_labels is not None:
      self.set_path_labels(path_labels)
    else:
      self.video_labels = video_labels
        
    # Get unique subjects and classes
    tmp = tools.get_unique_subjects_and_classes(self.path_labels)
    self.total_subjects, self.total_classes = len(tmp[0]), len(tmp[1])
    
    # Initialize face processing components
    self.face_extractor = None # TODO: None for testing more workers during training, REAL => extractor.FaceExtractor()
    self._load_reference_landmarks()
  
  def _set_sampling_strategy(self, strategy):
    """
    Set the frame sampling strategy based on the specified parameter.
    
    Args:
        strategy (str): The sampling strategy to use
    """
    if strategy == SAMPLE_FRAME_STRATEGY.UNIFORM:
      self.sample_frame_strategy = self._single_uniform_sampling
      warnings.warn(f"The {SAMPLE_FRAME_STRATEGY.UNIFORM} sampling strategy does not take into account the stride window.")
        
    elif strategy == SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW:
      self.sample_frame_strategy = self._sliding_window_sampling
        
    elif strategy == SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING:
      self.sample_frame_strategy = self._central_sampling
        
    elif strategy == SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING:
      self.sample_frame_strategy = self._random_sampling
      warnings.warn(f"The {SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING} sampling strategy does not take into account the stride window.")
  
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
    self.video_labels = pd.read_csv(self.path_labels,sep='\t')
    self.df = self.video_labels
    print(f'Set path_labels: {self.path_labels}')
  
  def __len__(self):
    """Return the number of samples in the dataset"""
    return len(self.video_labels)
  
  @staticmethod
  def preprocess_images(tensors,crop_size=(224,224),to_visualize=False,get_params=False,h_flip=False,color_jitter=False,rotation=False):
    """
    Preprocess a batch of image tensors.
    
    Args:
        tensors (torch.Tensor): A tensor of shape (B, C, H, W)
                                B = batch size,
                                C = number of channels,
                                H = height,
                                W = width.
    
    Returns:
        torch.Tensor: Preprocessed tensor of shape (B, C, 224, 224).
    """
    tensors = tv_tensors.Video(tensors)
    # Define preprocessing constants
    # crop_size = (224, 224)
    rescale_factor = 1/255  
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    
    transform = []
    params = {}
    
    if h_flip:
      transform.append(v2.RandomHorizontalFlip(p=1))
      params['h_flip'] = True

    if color_jitter:
      brightness = (0.7, 1.3)
      contrast = (0.7, 1.3)
      saturation = (0.7, 1.3)
      hue = (-0.05, 0.05)
      transform.append(v2.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)) #
      params['color_jitter'] = {'brightness': brightness, 'contrast': contrast, 'saturation': saturation, 'hue': hue} 
    if rotation:
      degrees = (-20, 20)
      transform.append(v2.RandomRotation(degrees=degrees))
      params['rotation'] = degrees
      
    # Define transform pipeline
    if not to_visualize:
      transform += [
        v2.Resize(crop_size),
        v2.Lambda(lambda x: x * rescale_factor),  # Rescale (1/255)
        v2.Normalize(mean=image_mean, std=image_std),  # Normalize
      ]
    else:
      transform += [v2.Resize(crop_size)]
    transform = v2.Compose(transform)
    
    if get_params:
      return transform(tensors),params
    else:
      return transform(tensors)
  

  def generate_video(self, idx,output_folder,fps_out=24):
    """
    Generate a video from the frames at the specified index.
    
    Args:
        idx (int): Index of the sample to generate a video for.
        
    Returns:
        None
    """
    csv_array = self.video_labels.iloc[idx]
    video_path = os.path.join(self.path_dataset, csv_array.iloc[1], csv_array.iloc[5] + self.video_extension)
    container = cv2.VideoCapture(video_path)
    tot_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
    # Set frame dimensions based on preprocessing requirements
    width_frames = int(container.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_frames = int(container.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # if tot_frames == 0:
    list_indices = self.sample_frame_strategy(tot_frames)
    frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames) # [nr_clips, nr_frames, H, W, C]
    # original_shape = frames_list.shape
    frames_list,params = self.preprocess_images(frames_list.reshape(-1,*frames_list.shape[2:]).permute(0,3,1,2),
                                                to_visualize=True,
                                                get_params=True,
                                                color_jitter=self.color_jitter,
                                                h_flip=self.h_flip,
                                                rotation=self.rotation) # [nr_clips, nr_frames, H, W, C] -> [B, C, H, W]

    # frames_list = frames_list.reshape(*original_shape)
    # Save the generated video
    if output_folder is None:
      output_path = os.path.join(self.saving_folder_path_extracted_video, f'{csv_array.iloc[5]}_inside_{self.stride_inside_window}_stride_{self.stride_window}_fps_{fps_out}{self.video_extension}')
    else:
      os.makedirs(output_folder, exist_ok=True)
      output_path = os.path.join(output_folder, f'{csv_array.iloc[5]}_inside_{self.stride_inside_window}_stride_{self.stride_window}_fps_{fps_out}{self.video_extension}')
    frames_list = frames_list.permute(0,2,3,1) # [B,C,H,W] -> [B,H,W,C]
    tools.generate_video_from_list_frame(list_frame=frames_list,path_video_output=output_path,fps=fps_out)
    # print(f"Video saved at {output_path}")
    return params
  
  def __standard_getitem__(self, idx):
    
    csv_array = self.video_labels.iloc[idx]
    video_path = os.path.join(self.path_dataset, csv_array.iloc[1], csv_array.iloc[5] + self.video_extension)

    # Open video and get properties
    container = cv2.VideoCapture(video_path)
    tot_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set frame dimensions based on preprocessing requirements
    # if self.preprocess_align or self.preprocess_frontalize or self.preprocess_crop_detection:
    #   width_frames = height_frames = 256
    # else:
    width_frames = int(container.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_frames = int(container.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video {video_path} has {tot_frames} frames")
    if tot_frames == 0:
      raise ValueError(f"Video {video_path} has no frames. Please check the video file.")
    # Get frame indices based on sampling strategy
    list_indices = self.sample_frame_strategy(tot_frames)

    # Read and process frames
    frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames)
    container.release()

    # Reshape frames for preprocessing
    nr_clips, nr_frames = frames_list.shape[:2]
    # Preprocess frames (considering only 1 video at time)
    # [nr_clips, nr_frames, H, W, C] -> [nr_clips * nr_frames, H, W, C] -> [B,C,H,W]
    frames_list = frames_list.reshape(-1, *frames_list.shape[2:]).permute(0, 3, 1, 2)
    preprocessed_tensors = self.preprocess_images(frames_list, 
                                                  crop_size=(self.image_resize_h, self.image_resize_w),
                                                  color_jitter = self.color_jitter,
                                                  h_flip = self.h_flip,
                                                  rotation = self.rotation) # [B,C,H,W]
    
    preprocessed_tensors = preprocessed_tensors.reshape(nr_clips, nr_frames, *preprocessed_tensors.shape[1:]) # [nr_clips, nr_frames, C, H, W]
    preprocessed_tensors = preprocessed_tensors.permute(0,2,1,3,4) # [B=nr_clips, T=nr_frames, C, H, W] -> [B, C, T, H, W]
    
    # Create metadata tensors
    sample_id = torch.full((nr_clips,), int(csv_array.iloc[4]), dtype=torch.int32)
    labels = torch.full((nr_clips,), int(csv_array.iloc[2]), dtype=torch.int32)
    subject_id = torch.full((nr_clips,), int(csv_array.iloc[0]), dtype=torch.int32)
    path = np.repeat(video_path, nr_clips)
      
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
        return self.__standard_getitem__(idx)
      else:
        dict_data = self.__standard_getitem__(idx)
        features = self.backbone_dict['backbone'].forward_features(dict_data['preprocess'])
        # logging.info(f'Features shape: {features.shape}')
        dict_data = {
          'features': features,
          'list_labels': dict_data['labels'],
          'list_sample_id': dict_data['sample_id'],
          'list_subject_id': dict_data['subject_id'],
        }
        dict_data = _get_element(dict_data=dict_data,df=self.video_labels,idx=idx)
        return dict_data
    else: # list case
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
    while container.isOpened():
      ret, frame = container.read()
      if not ret or frame_idx > max_end_frame:
        break
            
      # Convert color format
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
      # Apply preprocessing if needed
      if self.preprocess_align:
        frame = self.face_extractor.align_face(frame)
            
      if self.preprocess_crop_detection:
        frame = self.face_extractor.crop_face_detection(frame)
            
      if self.preprocess_frontalize:
        frame, _ = self.face_extractor.frontalize_img(
          frame=frame,
          ref_landmarks=self.reference_landmarks,
          frontalization_mode='SVD',
          v2=True,
          align=True
        )
        
      # Check if this frame should be included in any clip
      mask = np.any(np.isin(list_indices, frame_idx),axis=1)
      # mask =  
      if mask.any():
        frame_rgb = torch.tensor(frame, dtype=torch.uint8)
        extracted_frames[mask, pos[mask].long()] = frame_rgb
        pos[mask] += 1
            
      frame_idx += 1
    
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
    data = torch.cat([item['preprocess'].squeeze() for item in batch], dim=0) # [chunks, C, clip_length, H, W]
    # data = data.permute()
    # data = data.reshape(
    #   -1, self.image_channels, self.clip_length, 
    #   self.image_resize_h, self.image_resize_w
    # )  # 
    # Combine metadata from all samples
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    path = np.concatenate([item['path'] for item in batch])
    subject_id = torch.cat([item['subject_id'] for item in batch], dim=0)
    sample_id = torch.cat([item['sample_id'] for item in batch], dim=0)
    list_frames = torch.cat([item['frame_list'] for item in batch], dim=0).squeeze()

    return data, labels, subject_id, sample_id, path, list_frames
  
  def _custom_collate(self, batch):
    return _custom_collate(batch=batch,
                           instance_model_name=self.backbone_dict['instance_model_name'],
                           concatenate_temporal=self.backbone_dict['concatenate_temporal'],
                           model=self.backbone_dict['model'],
                           smooth_labels=self.smooth_labels,
                           num_classes=self.total_classes)
  
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
    csv_array = self.video_labels.to_numpy()
    list_samples = [entry[0].split("\t") for entry in csv_array]
    list_samples = np.stack(list_samples)
    sample_ids = list_samples[:, 4].astype(int)
    return sample_ids.tolist()
  
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
    indices = torch.randperm(video_len, dtype=torch.int16)[:self.clip_length]
    return torch.sort(indices).values[None, :]
  
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
    indices = torch.arange(0, video_len - self.clip_length * self.stride_inside_window + 1, self.stride_window)
    list_indices = torch.stack([torch.arange(start_idx, start_idx + self.clip_length * self.stride_inside_window, self.stride_inside_window) for start_idx in indices])
    # print('Sliding shape', list_indices.shape)
    return list_indices
  
  def get_unique_subjects(self):
    return np.sort(self.video_labels['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.video_labels['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.video_labels['class_id'],return_counts=True)[1]
  def get_unique_classes(self):
    return np.sort(self.video_labels['class_id'].unique().tolist())



class customDatasetAggregated(torch.utils.data.Dataset):
  def __init__(self,root_folder_features,concatenate_temporal,model,is_train,csv_path,smooth_labels,soft_labels,coral_loss):
    self.root_folder_feature = root_folder_features
    self.csv_path = csv_path
    self.concatenate_temporal = concatenate_temporal
    self.df = pd.read_csv(csv_path,sep='\t')
    self.num_classes = len(self.get_unique_classes())
    
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
      f'_rotate': sample_id_list[(sample_id_list > get_shift_for_sample_id('rotate')) * (sample_id_list <= get_shift_for_sample_id('rotate')+helper.step_shift)],
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
      return _get_element(df=self.df,dict_data=helper.dict_data,idx=idx)
    else:
      batch = [_get_element(df=self.df,dict_data=helper.dict_data,idx=idx) for idx in idx]
      batch = self._custom_collate(batch)
      return batch
  
  def _custom_collate(self,batch):
    return _custom_collate(batch,self.instance_model_name,self.concatenate_temporal,self.model,self.num_classes,self.smooth_labels,self.soft_labels,self.coral_loss)
  
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

class customDatasetWhole(torch.utils.data.Dataset):
  def __init__(self,csv_path,root_folder_features,concatenate_temporal,model,smooth_labels,soft_labels,coral_loss):
    self.csv_path = csv_path
    self.root_folder_features = root_folder_features
    self.df = pd.read_csv(csv_path,sep='\t') # subject_id, subject_name, class_id, class_name, sample_id, sample_name
    self.concatenate_temporal = concatenate_temporal
    self.model = model
    self.instance_model_name = tools.get_instace_model_name(model)
    self.smooth_labels = smooth_labels
    self.num_classes = len(self.get_unique_classes())
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
    
  def __len__(self):
    return len(self.df)
    
  def _load_element(self,idx):
    csv_row = self.df.iloc[idx]
    folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],f"{csv_row['sample_name']}.safetensors")
    # load_start = time.time()
    features = tools.load_dict_data(folder_path)
    features['list_labels'].fill_(csv_row['class_id']) # fill the labels with the class_id from csv
    # if features['features'].shape[0] > 4:
    #   for k,v in features.items():
    #     features[k] = v[4:] # 3 chunks -> 1.92 secat the begin

    # load_end = time.time()
    # print(f"Element {idx} loading time: {load_end - load_start:.2f} seconds")
    return _get_element(dict_data=features,df=self.df,idx=idx)
  
  def __getitem__(self,idx):
    if isinstance(idx,int):
      el = self._load_element(idx=idx)
      return el
    else:
      raise NotImplementedError("Batch loading is not implemented for customDatasetWhole")
      batch_time_start = time.time()
      # batch = [_load_element(idx=idx) for idx in idx]
      with ThreadPoolExecutor(max_workers=len(idx)) as executor:
        batch = list(executor.map(self._load_element, idx))
      batch_time_end = time.time()
      print(f"Batch {idx} loading time: {batch_time_end - batch_time_start:.2f} seconds")
      batch = self._custom_collate(batch)
      return batch

  def _custom_collate(self,batch):
    return _custom_collate(batch,self.instance_model_name,self.concatenate_temporal,self.model,self.num_classes,self.smooth_labels,self.soft_labels,self.coral_loss)  

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

def _custom_collate(batch,instance_model_name,concatenate_temporal,model,num_classes,smooth_labels, soft_labels_mat,coral_loss):
  # Pre-flatten features: reshape each sample to (sequence_length, emb_dim)
  if instance_model_name != helper.INSTANCE_MODEL_NAME.LINEARPROBE:
    if not concatenate_temporal:
      features = [sample['features'].reshape(-1,sample['features'].shape[-1]).to(torch.float32) for sample in batch] # [chunks,T,S,S,C] -> [chunks*T*S*S,C]
    else:
      features = [sample['features'].permute(0,2,3,1,4).reshape(-1,sample['features'].shape[-1]*sample['features'].shape[-2]).to(torch.float32) for sample in batch] # [chunks,T,S,S,C] ->[chunks,S,S,T,C]-> [chunks*S*S,C*T]
    # features -> [seq_len,emb_dim]
    lengths = [feat.size(0) for feat in features]
    features = torch.nn.utils.rnn.pad_sequence(features,batch_first=True) # [batch_size,seq_len,emb_dim]
    lengths_tensor = torch.tensor(lengths)  
    labels = torch.tensor([sample['labels'][0] for sample in batch],dtype=torch.int32)
    if smooth_labels > 0.0: # and model.output_size > 1:
      labels = smooth_labels_batch(gt_classes=labels, num_classes=num_classes, smoothing=smooth_labels)
    elif soft_labels_mat is not None:
      labels = soft_labels_mat[labels].to(torch.float32) # soft labels
    elif coral_loss:
      labels = levels_from_labelbatch(labels, num_classes, dtype=torch.float32)
    subject_id = torch.tensor([sample['subject_id'][0] for sample in batch])
    sample_id = torch.tensor([sample['sample_id'] for sample in batch])
    if instance_model_name == helper.INSTANCE_MODEL_NAME.AttentiveClassifier or instance_model_name == helper.INSTANCE_MODEL_NAME.ATTENTIVEPROBE:
      max_len = max(lengths)
      key_padding_mask = torch.arange(max_len).expand(len(batch), max_len) >= lengths_tensor.unsqueeze(1)
      key_padding_mask = ~key_padding_mask # set True for attention, if True means use the token to compute the attention, otherwise don't use it 
      return {'x':features, 'key_padding_mask': key_padding_mask},\
              labels,\
              subject_id,\
              sample_id
    elif instance_model_name == helper.INSTANCE_MODEL_NAME.GRUPROBE:
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
    features = torch.cat([torch.mean(sample['features'],dim=0,keepdim=True) for sample in batch],dim=0) # mean over the sequence
    labels = torch.tensor([sample['labels'][0] for sample in batch],dtype=torch.long)
    subject_id = torch.tensor([sample['subject_id'][0] for sample in batch])
    return {'x':features},\
            labels,\
            subject_id, 

def fake_collate(batch): # to avoid strange error when use customSampler
  return batch[0]  

class customBatchSampler(BatchSampler):
  def __init__(self, batch_size, shuffle,path_cvs_dataset=None, random_state=42,df=None):
    if df is None:
      csv_array,_ = tools.get_array_from_csv(path_cvs_dataset)
    else:
      csv_array = df.to_numpy()
    self.y_labels = np.array(csv_array[:,2]).astype(int)
    self.n_batch_size = batch_size
    self.random_state = random_state
    self.shuffle = shuffle
    _, count = np.unique(self.y_labels, return_counts=True)
    min_member = np.min(count)
    # max_member = np.max(count)
    self.initialize()
    # if max_member < self.skf.get_n_splits():
    #   raise ValueError(f"Impossible to split the dataset in {self.skf.get_n_splits()} splits. The maximum number of samples per class is {max_member}")
    if min_member < self.skf.get_n_splits():
      raise ValueError(f"Impossible to split the dataset in {self.skf.get_n_splits()} splits. The minimum number of samples per class is {min_member} and the batch size is {self.n_batch_size}. ")
    # -(-a//b) is the same as math.ceil(a/b)

    
  def initialize(self):
    nr_samples = len(self.y_labels)
    self.skf = StratifiedKFold(n_splits = math.ceil(nr_samples/self.n_batch_size) , shuffle=self.shuffle, random_state=self.random_state)
    self.n_batches = self.skf.get_n_splits()
    
  def __iter__(self):
    for _,test in self.skf.split(np.zeros(self.y_labels.shape[0]), self.y_labels):
      yield test.astype(np.int32).tolist() 
    
    self.random_state += 13
    self.initialize()
      

  def __len__(self):
    return self.n_batches
  
def smooth_labels_batch(gt_classes: torch.Tensor, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
  """
  Create label-smoothed targets for a batch of ground truth classes.
  
  Args:
    gt_classes (torch.Tensor): 1D tensor of shape (batch_size,) with ground truth class indices.
    num_classes (int): The total number of classes.
    smoothing (float): Smoothing factor in [0, 1). Default is 0.1.
  
  Returns:
    torch.Tensor: A tensor of shape (batch_size, num_classes) with smoothed probabilities.
  """
  assert 0 <= smoothing < 1, "Smoothing value should be in the range [0, 1)"
  # Probability for the ground truth class
  confidence = 1.0 - smoothing
  # Value for all other classes
  smooth_value = smoothing / (num_classes - 1)
  
  batch_size = gt_classes.size(0)
  # Create a tensor filled with smooth_value for each sample and class
  smoothed_labels = torch.full((batch_size, num_classes), smooth_value)
  # Set the ground truth class indices to have the confidence value
  smoothed_labels[torch.arange(batch_size), gt_classes] = confidence
  
  return smoothed_labels

 
def get_dataset_and_loader(csv_path,root_folder_features,batch_size,shuffle_training_batch,is_training,dataset_type,concatenate_temporal,model,
                           label_smooth,soft_labels,is_coral_loss,n_workers=None,backbone_dict=None,prefetch_factor=None):
  if dataset_type.value == CUSTOM_DATASET_TYPE.WHOLE.value:
    dataset_ = customDatasetWhole(csv_path,root_folder_features=root_folder_features,
                                  concatenate_temporal=concatenate_temporal,
                                  model=model,smooth_labels=label_smooth,
                                  soft_labels=soft_labels,
                                  coral_loss=is_coral_loss)
    pin_memory = True
    persistent_workers = True
    prefetch_factor = 10 if prefetch_factor is None else prefetch_factor
  elif dataset_type.value == CUSTOM_DATASET_TYPE.AGGREGATED.value:
    dataset_ = customDatasetAggregated(csv_path=csv_path,
                                        root_folder_features=root_folder_features,
                                        concatenate_temporal=concatenate_temporal,
                                        is_train=is_training,
                                        model=model,
                                        smooth_labels=label_smooth,
                                        coral_loss=is_coral_loss,
                                        soft_labels=soft_labels)
    pin_memory = False
    persistent_workers = True
    prefetch_factor = 2 if prefetch_factor is None else prefetch_factor
    
  elif dataset_type.value == CUSTOM_DATASET_TYPE.BASE.value:
    if backbone_dict is not None:
      dataset_ = customDataset(path_dataset=root_folder_features,
                              sample_frame_strategy=SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW,
                              path_labels=csv_path,
                              stride_window=16,
                              clip_length=16,
                              stride_inside_window=1,
                              preprocess_align=False,
                              preprocess_frontalize=False,
                              preprocess_crop_detection=False,
                              saving_folder_path_extracted_video=None,
                              video_labels=None,
                              backbone_dict=backbone_dict,
                              smooth_labels=label_smooth)
      pin_memory = False
      persistent_workers = True
      prefetch_factor = 5 if prefetch_factor is None else prefetch_factor
    else:
      raise ValueError(f'backbone_dict must be provided for dataset_type: {dataset_type}')
  else:
    raise ValueError(f'Unknown dataset type: {dataset_type}. Choose one of {CUSTOM_DATASET_TYPE}')

  if is_training:
    # try:
    customBatchSampler_train = customBatchSampler(df=dataset_.df, 
                                        batch_size=batch_size,
                                        shuffle=shuffle_training_batch)
    if n_workers > 1:
      loader_ = DataLoader(
                          dataset=dataset_,
                          batch_sampler=customBatchSampler_train,
                          collate_fn=dataset_._custom_collate,
                          # batch_size=1,
                          num_workers=n_workers,
                          persistent_workers= persistent_workers,
                          prefetch_factor=prefetch_factor,
                          pin_memory=pin_memory)
      print(f'Use custom Dataloader with {n_workers} workers!\nPersistent workers: {persistent_workers} \nPin memory: {pin_memory}\nPrefetch factor: {prefetch_factor}')
    else:
      loader_ = DataLoader(
                          dataset=dataset_,
                          batch_sampler=customBatchSampler_train,
                          collate_fn=dataset_._custom_collate,
                          # batch_size=1,
                          pin_memory=True)
      print(f'Use custom Dataloader!')
    # except Exception as e:
    #   raise ValueError(f'Error in customBatchSampler: {e}') from e
    #   print(f'Err: {e}')
    #   print(f'Use standard DataLoader')
    #   loader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle_training_batch,collate_fn=dataset_._custom_collate,persistent_workers=True)
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
 
  
def _get_element(dict_data,df,idx):
  csv_row = df.iloc[idx]
  sample_id = csv_row['sample_id']
  # if 'hflip' in self.root_folder_feature:
  #   sample_id = sample_id + 8700
  if sample_id > helper.step_shift * 4 and sample_id <= helper.step_shift * 6: # latent augm.
    mask = dict_data['list_sample_id'] == ((sample_id - 1) % helper.step_shift + 1) # to get the real sample_id
  else:
    mask = dict_data['list_sample_id'] == sample_id
  if mask.sum() == 0:
    print(f"Sample ID {sample_id} not found in the dataset.")
  features = dict_data['features'][mask]
  labels = dict_data['list_labels'][mask]
  subject_id = dict_data['list_subject_id'][mask]
  
  # polarity inversion + gaussian noise
  if sample_id > helper.step_shift * 4 and sample_id <= helper.step_shift * 5: 
    gaussian_noise = torch.randn_like(features) * 0.1
    features = (-1 * features) + gaussian_noise 
  
  # random masking patches
  elif sample_id > helper.step_shift * 5 and sample_id <= helper.step_shift * 6:
    B,T,S,S,C = features.shape
    mask_grid = torch.rand(S,S) < 0.1 # 10% of the grid
    mask_grid = mask_grid.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    mask_grid = mask_grid.expand(B,T,S,S,C) # [B,T,S,S,C]
    features = features.masked_fill(mask_grid, 0.0) # set to zero the masked values
  
  return {
      'features': features,     # [8,8,1,1,768]-> [seq_len,Temporal,Space,Space,Emb]
      'labels': labels,         # [8]
      'subject_id': subject_id, # [8]
      'sample_id': sample_id    # int
  }    
    
        
      
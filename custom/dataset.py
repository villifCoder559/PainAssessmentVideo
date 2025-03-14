import warnings
import torch
import pandas as pd
import os
import math
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.utils
from custom.backbone import BackboneBase
from custom.helper import CUSTOM_DATASET_TYPE, SAMPLE_FRAME_STRATEGY
import custom.tools as tools
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import  Sampler

import torchvision.transforms as T
import custom.faceExtractor as extractor
import pickle
from torch.utils.data import DataLoader
from custom.helper import INSTANCE_MODEL_NAME


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
      backbone_dict=None
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
    
    # Image dimensions and channels
    self.image_resize_w = 224
    self.image_resize_h = 224
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
    self.face_extractor = extractor.FaceExtractor()
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
    print(f'Set path_labels: {self.path_labels}')
  
  def __len__(self):
    """Return the number of samples in the dataset"""
    return len(self.video_labels)
  
  def preprocess_images(self, tensors):
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
    
    # Define preprocessing constants
    crop_size = (224, 224)
    rescale_factor = 1/255  # More readable than 0.00392156862745098
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    shortest_edge = 224
    
    # Define transform pipeline
    transform = T.Compose([
      T.Resize(shortest_edge),  # Resize the shortest edge to 224, preserving aspect ratio
      T.CenterCrop(crop_size),  # Center crop
      T.Lambda(lambda x: x * rescale_factor),  # Rescale (1/255)
      T.Normalize(mean=image_mean, std=image_std)  # Normalize
    ])
    
    # Apply transforms to each tensor in batch
    return torch.stack([transform(t) for t in tensors])
  
  def __standard_getitem__(self, idx):
    csv_array = self.video_labels.iloc[idx]
    video_path = os.path.join(self.path_dataset, csv_array.iloc[1], csv_array.iloc[5] + '.mp4')

    # Open video and get properties
    container = cv2.VideoCapture(video_path)
    tot_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set frame dimensions based on preprocessing requirements
    if self.preprocess_align or self.preprocess_frontalize or self.preprocess_crop_detection:
      width_frames = height_frames = 256
    else:
      width_frames = int(container.get(cv2.CAP_PROP_FRAME_WIDTH))
      height_frames = int(container.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get frame indices based on sampling strategy
    list_indices = self.sample_frame_strategy(tot_frames)

    # Read and process frames
    frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames)
    container.release()

    # Reshape frames for preprocessing
    nr_clips, nr_frames = frames_list.shape[:2]
    frames_list = frames_list.reshape(-1, *frames_list.shape[2:]).permute(0, 3, 1, 2) # [nr_clips, nr_frames, H, W, C] -> [nr_clips * nr_frames, H, W, C] -> [B,C,H,W]

    # Preprocess frames
    preprocessed_tensors = self.preprocess_images(frames_list) # frame list to [B,C,H,W]
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
        return {
          'features': features,
          'labels': dict_data['labels'],
          'subject_id': dict_data['subject_id'],
        }
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
            'labels': item['labels'],
            'subject_id': item['subject_id']
          })
        return self._custom_collate(batch)
      
  
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
    start_frame_idx = list_indices[:, 0]
    end_frame_idx = list_indices[:, -1]
    num_clips, clip_length = list_indices.shape
    
    # Initialize tensors to store extracted frames
    extracted_frames = torch.zeros(
      num_clips, clip_length, height_frames, width_frames, self.image_channels, 
      dtype=torch.uint8
    )
    pos = torch.zeros(num_clips, dtype=torch.int32)
    
    # Find max end frame to optimize loop
    max_end_frame = end_frame_idx.max().item()
    
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
      mask = (start_frame_idx <= frame_idx) & (end_frame_idx >= frame_idx)
        
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
    data = torch.cat([item['preprocess'].squeeze().transpose(1, 2) for item in batch], dim=0) 
    data = data.reshape(
      -1, self.image_channels, self.clip_length, 
      self.image_resize_h, self.image_resize_w
    ) 
    # Combine metadata from all samples
    labels = torch.cat([item['labels'] for item in batch], dim=0)
    path = np.concatenate([item['path'] for item in batch])
    subject_id = torch.cat([item['subject_id'] for item in batch], dim=0)
    sample_id = torch.cat([item['sample_id'] for item in batch], dim=0)
    list_frames = torch.cat([item['frame_list'] for item in batch], dim=0).squeeze()

    return data, labels, subject_id, sample_id, path, list_frames
  
  def _custom_collate(self, batch):
    return _custom_collate(batch,self.backbone_dict['instance_model_name'],self.backbone_dict['concatenate_temporal'],self.backbone_dict['model'])
  
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
  
  def _sliding_window_sampling(self, video_len, stride_inside_window=1):
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
    indices = torch.arange(0, video_len - self.clip_length * stride_inside_window + 1, self.stride_window)
    list_indices = torch.stack([torch.arange(start_idx, start_idx + self.clip_length * stride_inside_window, stride_inside_window) for start_idx in indices])
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
  def __init__(self,root_folder_features,csv_path,concatenate_temporal,model):
    self.root_folder_feature = root_folder_features
    self.df = pd.read_csv(csv_path,sep='\t')
    self.concatenate_temporal = concatenate_temporal
    self.dict_data = self._feature_extraction()
    self.model = model
    self.instance_model_name = tools.get_instace_model_name(model)
  
  def _feature_extraction(self):
    dict_data = tools.load_dict_data(self.root_folder_feature)
    sample_id_list = torch.tensor(self.df['sample_id'].tolist())
    is_in = torch.isin(dict_data['list_sample_id'],sample_id_list)
    for k,v in dict_data.items():
      dict_data[k] = v[is_in]
    return dict_data
  
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    def _get_element(idx):
      csv_row = self.df.iloc[idx]
      sample_id = csv_row['sample_id']
      mask = self.dict_data['list_sample_id'] == sample_id
      features = self.dict_data['features'][mask]
      labels = self.dict_data['list_labels'][mask]
      subject_id = self.dict_data['list_subject_id'][mask]
      return {
          'features': features,     # [8,8,1,1,768]-> [seq_len,Temporal,Space,Space,Emb]
          'labels': labels,         # [8]
          'subject_id': subject_id  # [8]
      }
    if isinstance(idx,int):
      return _get_element(idx)
    else:
      batch = [_get_element(idx) for idx in idx]
      batch = self._custom_collate(batch)
      return batch
  
  def _custom_collate(self,batch):
    return _custom_collate(batch,self.instance_model_name,self.concatenate_temporal,self.model)
  
  def get_unique_subjects(self):
    return np.sort(self.df['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.df['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.df['class_id'],return_counts=True)[1]
  def get_unique_classes(self):
    return np.sort(self.df['class_id'].unique().tolist())

class customDatasetWhole(torch.utils.data.Dataset):
  def __init__(self,csv_path,root_folder_features,concatenate_temporal,model):
    self.csv_path = csv_path
    self.root_folder_features = root_folder_features
    self.df = pd.read_csv(csv_path,sep='\t') # subject_id, subject_name, class_id, class_name, sample_id, sample_name
    self.concatenate_temporal = concatenate_temporal
    self.model = model
    self.instance_model_name = tools.get_instace_model_name(model)
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self,idx):
    def _get_element(idx=idx):
      csv_row = self.df.iloc[idx]
      folder_path = os.path.join(self.root_folder_features,csv_row['subject_name'],csv_row['sample_name'])
      features = tools.load_dict_data(folder_path)
      return {
          'features': features['features'],
          'labels': features['list_labels'],
          'subject_id': features['list_subject_id'],
          # 'sample_id': torch.tensor(csv_row['sample_id'],dtype=torch.int16)
      }
    
    if isinstance(idx,int):
      return _get_element(idx=idx)
    else:
      batch = [_get_element(idx=idx) for idx in idx]
      batch = self._custom_collate(batch)
      return batch

  def _custom_collate(self,batch):
    return _custom_collate(batch,self.instance_model_name,self.concatenate_temporal,self.model)  
      
  
  def get_unique_subjects(self):
    return np.sort(self.df['subject_id'].unique().tolist())
  def get_count_subjects(self):
    return np.unique(self.df['subject_id'],return_counts=True)[1]
  def get_count_classes(self):
    return np.unique(self.df['class_id'],return_counts=True)[1]
  def get_unique_classes(self):
    return np.sort(self.df['class_id'].unique().tolist())

def _custom_collate(batch,instance_model_name,concatenate_temporal,model):
  # Pre-flatten features: reshape each sample to (sequence_length, emb_dim)
  if instance_model_name.value != 'LinearProbe':
    if not concatenate_temporal:
      features = [sample['features'].reshape(-1,sample['features'].shape[-1]) for sample in batch]
    else:
      features = [sample['features'].reshape(-1,sample['features'].shape[-1]*sample['features'].shape[-4]) for sample in batch]
    # features -> [seq_len,emb_dim]
    lengths = [feat.size(0) for feat in features]
    padded_features = torch.nn.utils.rnn.pad_sequence(features,batch_first=True) # [batch_size,seq_len,emb_dim]
    lengths_tensor = torch.tensor(lengths)  
    labels = torch.tensor([sample['labels'][0] for sample in batch],dtype=torch.long)
    subject_id = torch.tensor([sample['subject_id'][0] for sample in batch])
    
    if instance_model_name.value == 'AttentiveProbe':
      max_len = max(lengths)
      key_padding_mask = torch.arange(max_len, device=padded_features.device).expand(len(batch), max_len) >= lengths_tensor.unsqueeze(1)
      return {'x':padded_features,
              'key_padding_mask': key_padding_mask},\
              labels,\
              subject_id
    elif instance_model_name.value == 'GRUProbe':
      packed_input = torch.nn.utils.rnn.pack_padded_sequence(padded_features,lengths_tensor,batch_first=True,enforce_sorted=False)
      if model.output_size == 1:
        labels = labels.float()
      return {'x':packed_input},\
              labels,\
              subject_id
  else:
    features = torch.cat([torch.mean(sample['features'],dim=0,keepdim=True) for sample in batch],dim=0) # mean over the sequence
    labels = torch.tensor([sample['labels'][0] for sample in batch],dtype=torch.long)
    subject_id = torch.tensor([sample['subject_id'][0] for sample in batch])
    return {'x':features},\
            labels,\
            subject_id

def fake_collate(batch): # to avoid strange error when use customSampler
  return batch[0]  

class customSampler(Sampler):
  def __init__(self,path_cvs_dataset, batch_size, shuffle, random_state=0):
    csv_array,_ = tools.get_array_from_csv(path_cvs_dataset)
    self.y_labels = np.array(csv_array[:,2]).astype(int)
    nr_samples = len(self.y_labels)
    self.n_batch_size = batch_size
    # -(-a//b) is the same as math.ceil(a/b)
    self.skf = StratifiedKFold(n_splits = math.ceil(nr_samples/batch_size) , shuffle=shuffle, random_state=random_state)
    self.n_batches = self.skf.get_n_splits()
    
  def initialize(self):
    _, count = np.unique(self.y_labels, return_counts=True)
    min_member = np.min(count)
    max_member = np.max(count)
    idx_min = np.argmin(count)
    print(f'Min count class: {min_member} for class: {idx_min}')
    print(f'Min count member: {min_member}')
    print(f'Number of splits: {self.skf.get_n_splits()}')
    # n_splits cannot be greater than the number of members in each class.
    if max_member < self.skf.get_n_splits():
      raise ValueError(f"Impossible to split the dataset in {self.skf.get_n_splits()} splits. The maximum number of samples per class is {max_member}")
    if min_member < self.skf.get_n_splits():
      raise ValueError(f"Impossible to split the dataset in {self.skf.get_n_splits()} splits. The minimum number of samples per class is {min_member}")
  
  def __iter__(self):
    for _,test in self.skf.split(np.zeros(self.y_labels.shape[0]), self.y_labels):
      yield test
      
      
  def __len__(self):
    return self.n_batches
  
  
  
def get_dataset_and_loader(csv_path,root_folder_features,batch_size,shuffle_training_batch,is_training,dataset_type,concatenate_temporal,model,backbone_dict=None):
  if dataset_type.value == CUSTOM_DATASET_TYPE.WHOLE.value:
    dataset_ = customDatasetWhole(csv_path,root_folder_features=root_folder_features,concatenate_temporal=concatenate_temporal,
                                  model=model)
  elif dataset_type.value == CUSTOM_DATASET_TYPE.AGGREGATED.value:
    dataset_ = customDatasetAggregated(csv_path=csv_path,
                                        root_folder_features=root_folder_features,
                                        concatenate_temporal=concatenate_temporal,
                                        model=model)
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
                              backbone_dict=backbone_dict)
    else:
      raise ValueError(f'backbone_dict must be provided for dataset_type: {dataset_type}')
  else:
    raise ValueError(f'Unknown dataset type: {dataset_type}. Choose one of {CUSTOM_DATASET_TYPE}')
  if is_training:
    try:
      print('Try to use custom DataLoader...')
      customSampler_train = customSampler(path_cvs_dataset=csv_path, 
                                          batch_size=batch_size,
                                          shuffle=shuffle_training_batch)
      customSampler_train.initialize()
      loader_ = DataLoader(dataset=dataset_, sampler=customSampler_train,collate_fn=fake_collate,batch_size=1)
      print('Custom DataLoader instantiated')
    except Exception as e:
      print(f'Err: {e}')
      print(f'Use standard DataLoader')
      loader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=shuffle_training_batch,collate_fn=dataset_._custom_collate)
  else:
    loader_ = DataLoader(dataset=dataset_, batch_size=batch_size, shuffle=False,collate_fn=dataset_._custom_collate)
  return dataset_,loader_
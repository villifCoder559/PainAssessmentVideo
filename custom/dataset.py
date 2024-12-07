import torch
import pandas as pd
import av
import os
import numpy as np
import torch
import time
# import cv2
from custom.helper import SAMPLE_FRAME_STRATEGY
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import custom.tools as tools

class customDataset(torch.utils.data.Dataset):
  def __init__(self,path_dataset, path_labels, preprocess, sample_frame_strategy, batch_size=1, stride_window=2, clip_length=16,stride_inside_window=1):
    assert os.path.exists(path_dataset), f"Dataset path {path_dataset} does not exist."
    # assert os.path.exists(path_labels), f"Labels path {path_labels} does not exist."
    assert clip_length > 0, f"Clip length must be greater than 0."
    assert stride_window > 0, f"Stride window must be greater than 0."
    assert sample_frame_strategy in SAMPLE_FRAME_STRATEGY, f"Sample frame strategy must be one of {SAMPLE_FRAME_STRATEGY}."
    
    self.path_dataset = path_dataset
    self.path_labels = path_labels
    self.preprocess = preprocess
    self.type_sample_frame_strategy = sample_frame_strategy
    # self.video_labels = pd.read_csv(path_labels)
    if sample_frame_strategy == SAMPLE_FRAME_STRATEGY.UNIFORM:
      self.sample_frame_strategy = self._single_uniform_sampling
      Warning(f"The {SAMPLE_FRAME_STRATEGY.UNIFORM} sampling strategy does not take into account the stride window.")
      
    elif sample_frame_strategy == SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW:
      self.sample_frame_strategy = self._sliding_window_sampling
      self.stride_window = stride_window
      self.stride_inside_window = stride_inside_window
    elif sample_frame_strategy == SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING:
      self.sample_frame_strategy = self._central_sampling
      self.stride_inside_window = stride_inside_window
    elif sample_frame_strategy == SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING:
      self.sample_frame_strategy = self._random_sampling
      Warning(f"The {SAMPLE_FRAME_STRATEGY.RANDOM_SAMPLING} sampling strategy does not take into account the stride window.")
    
    self.image_resize_w = 224
    self.image_resize_h = 224
    self.image_channels = 3
    self.clip_length = clip_length
    # self.set_path_labels('all')
    # self.set_path_labels('train')
    tmp = tools.get_unique_subjects_and_classes(self.path_labels)
    self.total_subjects, self.total_classes = len(tmp[0]), len(tmp[1])

  def set_path_labels(self, path):
    """
    Sets the path to the labels file and loads the video labels from a CSV file.

    Args:
      path (str): The file path to the CSV file containing the video labels.

    Returns:
      None
    """
    self.path_labels = path
    self.video_labels = pd.read_csv(self.path_labels)
    print(f'Set path_labels: {self.path_labels}')
    
  def __len__(self):
    return len(self.video_labels)


  def __getitem__(self, idx):
    """
    Retrieve a sample from the dataset at the given index.
    Args:
      idx (int): Index of the sample to retrieve.
    Returns:
      dict: A dictionary containing the following keys:
        - 'preprocess' (torch.Tensor): Preprocessed video frames with shape 
          [nr_clips, batch_video=1, clip_length, channels=3, H=224, W=224].
        - 'labels' (torch.Tensor): Labels for the video frames with shape [nr_clips].
        - 'subject_id' (torch.Tensor): Subject IDs for the video frames with shape [nr_clips].
        - 'sample_id' (torch.Tensor): Sample IDs for the video frames with shape [nr_clips].
        - 'path' (numpy.ndarray): Paths to the video files with shape (nr_clips,).
        - 'frame_list' (torch.Tensor): Indices of the frames sampled from the video with shape [nr_clips, clip_length].
    """
    # split in csv_array ["subject_id, subject_name, class_id, class_name, sample_id, sample_name"]
    csv_array = self.video_labels.iloc[idx,0].split('\t') 
    video_path = os.path.join(self.path_dataset, csv_array[1], csv_array[5])
    video_path += '.mp4'
    container = av.open(video_path)
    tot_frames = container.streams.video[0].frames
    
    list_indices = self.sample_frame_strategy(tot_frames)
    start_time_load_video = time.time()
    frames_list=[self._read_video_pyav(container,indices) for indices in list_indices]
    # print('time to load video:', time.time()-start_time_load_video)
    start_time_preprocess = time.time()
    preprocessed_tensors = torch.stack([self.preprocess(list(frames), return_tensors="pt")['pixel_values'] for frames in frames_list])
    # print('Time to preprocess video', time.time()-start_time_preprocess)
    # preprocessed output shape [2, 1, 16, 3, 224, 224] -> [nr_clips, batch_video, clip_length, RGB_channels=3, H=224, W=224]
    # print(len(frames_list))
    path = np.repeat(video_path, preprocessed_tensors.shape[0])
    
    unit_vector = torch.ones(preprocessed_tensors.shape[0],dtype=torch.int16)    
    sample_id = unit_vector * int(csv_array[4])
    labels = unit_vector * int(csv_array[2])
    subject_id = unit_vector * int(csv_array[0])
    return  {'preprocess':preprocessed_tensors, # shape [nr_clips, batch_video=1, clip_length, channels=3, H=224, W=224]
             'labels': labels,  # torch shape [nr_clips]
             'subject_id': subject_id, # torch shape [nr_clips]
             'sample_id': sample_id, # torch shape [nr_clips]
             'path': path, # np shape (nr_clips,)
             'frame_list': list_indices} #torch shape [nr_clips, clip_length]

  def _read_video_pyav(self,container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

  def _custom_collate_fn(self, batch):
    """
    Custom collate function to process and stack batch data for a DataLoader.

    Args:
      batch (list): A list of dictionaries, where each dictionary contains the following keys:
        - 'preprocess': Preprocessed video tensor shape [nr_clips, batch_video=1, clip_length=16, channels=3, H=224, W=224]
        - 'labels': Corresponding labels tensor.
        - 'path': Path to the video file.
        - 'subject_id': Subject ID tensor.
        - 'sample_id': Sample ID tensor.
        - 'frame_list': List of frames tensor.

    Returns:
      tuple: A tuple containing:
        - 'data' (torch.Tensor): shape [nr_video* nr_windows, channels=3, clip_length=16, H=224, W=224]
        - 'labels' (torch.Tensor): shape [nr_video * nr_windows].
        - 'subject_id' (torch.Tensor): shape [nr_video * nr_windows].
        - 'sample_id' (torch.Tensor): shape [nr_video * nr_windows].
        - 'path' (numpy.ndarray): shape (nr_video * nr_windows).
        - 'list_frames' (torch.Tensor): shape [nr_video * nr_windows].
    """
    
    data = torch.cat([batch[index]['preprocess'].squeeze().transpose(1,2) for index in range(len(batch))], dim=0) 
    data = data.reshape(-1,self.image_channels, self.clip_length, self.image_resize_h, self.image_resize_w) 

    labels = torch.cat([batch[index]['labels'] for index in range(len(batch))],dim=0)
    path = np.concatenate([batch[index]['path'] for index in range(len(batch))])
    subject_id = torch.cat([batch[index]['subject_id'] for index in range(len(batch))],dim=0)
    sample_id = torch.cat([batch[index]['sample_id'] for index in range(len(batch))],dim=0)
    list_frames = torch.cat([batch[index]['frame_list'] for index in range(len(batch))],dim=0).squeeze()

    return data, \
           labels,\
           subject_id,\
           sample_id,\
           path,\
           list_frames
  
  def get_params_configuration(self):
    """
    Returns the configuration parameters of the dataset.

    Returns:
      dict: A dictionary containing the configuration parameters of the dataset.
    """
    return {
        'path_dataset': self.path_dataset,
        'path_labels': self.path_labels,
        'preprocess': self.preprocess.__class__.__name__,
        'sample_frame_strategy': self.type_sample_frame_strategy.name,
        'clip_length': self.clip_length,
        'stride_window': self.stride_window,
        'stride_inside_window': self.stride_inside_window
    }  
  
  def _single_uniform_sampling(self, video_len):
    indices = np.linspace(0, video_len-1, self.clip_length, dtype=int)
    indices = torch.from_numpy(indices)
    return indices[None, :]
  
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
    assert video_len // 2 - (self.clip_length // 2) * self.stride_inside_window >= 0, f"Video is too short for the given clip length. Reduce the stride (given {self.stride_window})"
    start_idx = video_len // 2 - (self.clip_length // 2) * self.stride_inside_window
    # print('Start index',start_idx)
    indices = torch.arange(start_idx, start_idx + self.clip_length * self.stride_inside_window, self.stride_inside_window)[None,:]
    # print('Indices central',indices)
    return indices
  
  def _random_sampling(self, video_len):
    indices = torch.randperm(video_len,dtype=torch.int16)[:self.clip_length]
    indices = torch.sort(indices).values[None, :]
    # print('Random non-repeated indices', indices)
    # print('Random shape', indices.shape)
    return indices
  
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
  
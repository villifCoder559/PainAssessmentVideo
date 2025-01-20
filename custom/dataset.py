import torch
import pandas as pd
import av
import os
import math
import numpy as np
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
from custom.helper import SAMPLE_FRAME_STRATEGY
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import custom.tools as tools
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import  Sampler
from sklearn.utils import shuffle
import torchvision.transforms as T
import custom.faceExtractor as extractor
import pickle

class customDataset(torch.utils.data.Dataset):
  def __init__(self,path_dataset, path_labels, sample_frame_strategy, stride_window=2, clip_length=16,stride_inside_window=1,
               preprocess_align=False,preprocess_frontalize=False,preprocess_crop_detection=False):
    assert os.path.exists(path_dataset), f"Dataset path {path_dataset} does not exist."
    # assert os.path.exists(path_labels), f"Labels path {path_labels} does not exist."
    assert clip_length > 0, f"Clip length must be greater than 0."
    assert stride_window > 0, f"Stride window must be greater than 0."
    assert sample_frame_strategy in SAMPLE_FRAME_STRATEGY, f"Sample frame strategy must be one of {SAMPLE_FRAME_STRATEGY}."
    
    self.path_dataset = path_dataset
    self.path_labels = path_labels
    # self.preprocess = preprocess
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
    self.preprocess_align = preprocess_align
    self.preprocess_frontalize = preprocess_frontalize
    self.preprocess_crop_detection = preprocess_crop_detection
    self.image_resize_w = 224
    self.image_resize_h = 224
    self.image_channels = 3
    self.clip_length = clip_length
    # self.set_path_labels('all')
    self.set_path_labels(path_labels)
    tmp = tools.get_unique_subjects_and_classes(self.path_labels)
    self.total_subjects, self.total_classes = len(tmp[0]), len(tmp[1])
    self.face_extractor = extractor.FaceExtractor()
    self.reference_landmarks = pickle.load(open(os.path.join('partA', 'video', 'mean_face_landmarks_per_subject', 'all_subjects_mean_landmarks.pkl'), 'rb'))
    self.reference_landmarks = self.reference_landmarks['mean_facial_landmarks']
    # self.preprocess_torchvision = transforms.Compose([
    #                               # transforms.ToTensor(),  # Convert PIL image to tensor and scale pixel values to [0, 1]
    #                               transforms.Resize(224),  # Resize so that shortest edge is 224 (bilinear interpolation by default)
    #                               transforms.CenterCrop((224, 224)),  # Center crop to 224x224
    #                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    #                               ])

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
  
  def preprocess_images(self, tensors):
    """
    Preprocess a batch of image tensors and plot the preprocessed images.
    
    Args:
        tensors (torch.Tensor): A tensor of shape (B, C, H, W) where:
                                B = batch size,
                                C = number of channels,
                                H = height,
                                W = width.
    
    Returns:
        torch.Tensor: Preprocessed tensor of shape (B, C, 224, 224).
    """
    crop_size = (224, 224)
    rescale_factor = 0.00392156862745098  # 1/255
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    shortest_edge = 224
    
    transform = T.Compose([
      T.Resize(shortest_edge),  # Resize the shortest edge to 224, preserving aspect ratio
      T.CenterCrop(crop_size),  # Center crop
      T.Lambda(lambda x: x * rescale_factor),  # Rescale (1/255)
      T.Normalize(mean=image_mean, std=image_std)  # Normalize
    ])
    
    preprocessed_tensors = torch.stack([transform(t) for t in tensors])
    
    # Plot the preprocessed images
    # self.plot_preprocessed_images(preprocessed_tensors[:4])
    
    return preprocessed_tensors

  def plot_preprocessed_images(self, tensors):
    """
    Plot a batch of preprocessed image tensors.
    
    Args:
        tensors (torch.Tensor): A tensor of shape (B, C, H, W) where:
                                B = batch size,
                                C = number of channels,
                                H = height,
                                W = width.
    """
    num_images = min(tensors.shape[0], 4)  # Plot up to 4 images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    for i in range(num_images):
      image = tensors[i].permute(1, 2, 0).numpy()
      image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
      image = np.clip(image, 0, 1)
      
      axes[i].imshow(image)
      axes[i].axis('off')
    
    plt.show()
  
  def __getitem__(self, idx):
    """
    Retrieve a sample from the dataset at the given index.
    Args:
      idx (int): Index of the sample to retrieve.
    Returns:
      dict: A dictionary containing preprocessed video data and metadata.
    """
    # start_time= time.time()
    # Extract metadata from CSV
    csv_array = self.video_labels.iloc[idx, 0].split('\t')
    video_path = os.path.join(self.path_dataset, csv_array[1], csv_array[5] + '.mp4')
    # print(f'video_path: {video_path}')
    # Open video container and get total frames
    # container = av.open(video_path,options={'hwaccel': 'cuda', 'hwaccel_device': '0'})
    # tot_frames = container.streams.video[0].frames
    # width_frames = container.streams.video[0].width
    # height_frames = container.streams.video[0].height
    # Sample frame indices using the provided strategy
    container = cv2.VideoCapture(video_path)
    tot_frames = int(container.get(cv2.CAP_PROP_FRAME_COUNT))
    if self.preprocess_align or self.preprocess_frontalize or self.preprocess_crop_detection:
      width_frames = 256
      height_frames = 256
    else:
      width_frames = container.get(cv2.CAP_PROP_FRAME_WIDTH)
      height_frames = container.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # width_frames = int(container.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(f'Width frames: {width_frames}')
    # height_frames = int(container.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(f'Height frames: {height_frames}')
    list_indices = self.sample_frame_strategy(tot_frames)

    # Load and preprocess frames
    # print(f'list_indices shape: {list_indices.shape}')
    # start_time_load_video = time.time()
    frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames)
    # Generate video from frames_list and save to partA/video
    # generate video with 1% probability
    if np.random.rand() < 0.01:
      saving_folder_path = os.path.join('partA','video','features','samples_16_cropped_aligned')
      tools.generate_video_from_list_frame(list_frame = frames_list.reshape(-1,height_frames,width_frames,3),
                                          path_video_output=os.path.join(saving_folder_path,'video',f'{csv_array[5]}.mp4'))
    # self.preprocess_crop_detection = True
    # self.preprocess_align = False
    # self.preprocess_frontalize = False
    # frames_list = self._read_video_cv2_and_process(container, list_indices, width_frames, height_frames)
    # print('frame_list_detect',len(frames_list))
    # tools.generate_video_from_list_frame(list_frame = frames_list.reshape(-1,height_frames,width_frames,3),
    #                                      path_video_output=os.path.join('partA','video','video_from_feat_extr_crop',f'{csv_array[5]}.mp4'))
    # frames_list = self._read_video_pyav(container,list_indices= list_indices,width_frames=width_frames,height_frames=height_frames)
    # frames_list = [self._read_video_pyav(container, indices) for indices in list_indices]
    # print(f'Frames list shape: {frames_list.shape}') # shape torch.Size([11, 16, 1038, 1388, 3])
    nr_clips = frames_list.shape[0]
    nr_frames = frames_list.shape[1]
    frames_list = frames_list.reshape(-1,*frames_list.shape[2:])
    # end_time_load_video = time.time()
    # print(f'Elapsed time load video: {end_time_load_video-start_time_load_video}')
    
    # Preprocess frames and stack into a tensor
    # start_time_preprocess = time.time()
    # print(f'Frames list shape: {frames_list.shape}') 
    preprocessed_tensors = self.preprocess_images(frames_list.permute(0,3,1,2))
    # end_time_preprocess = time.time()
    # print(f'Elapsed time preprocess: {end_time_preprocess-start_time_preprocess}')
    preprocessed_tensors = preprocessed_tensors.reshape(nr_clips, nr_frames, *preprocessed_tensors.shape[1:]) 
    # print(f'Preprocessed tensors shape: {preprocessed_tensors.shape}')
    # Create metadata tensors
    
    path = np.repeat(video_path, nr_clips)
    
    unit_vector = torch.ones(nr_clips, dtype=torch.int16)
    sample_id = unit_vector * int(csv_array[4])
    labels = unit_vector * int(csv_array[2])
    subject_id = unit_vector * int(csv_array[0])
    # end_time = time.time()
    # print(f'Elapsed total time: {end_time-start_time}')
    # print(f'preprocessed_tensors shape: {preprocessed_tensors.shape}')
    return {
        'preprocess': preprocessed_tensors,
        'labels': labels,
        'subject_id': subject_id,
        'sample_id': sample_id,
        'path': path,
        'frame_list': list_indices
    }
  def _read_video_cv2_and_process(self,container,list_indices,width_frames,height_frames):
    # Assume list_indices is sorted
    start_frame_idx = list_indices[:, 0]
    end_frame_idx = list_indices[:, -1]
    num_clips, clip_length = list_indices.shape
    # idx_frame_saved = torch.zeros(num_clips,clip_length, dtype=torch.int32)
    extracted_frames = torch.zeros(num_clips, clip_length, height_frames, width_frames, self.image_channels, dtype=torch.uint8)
    pos = torch.zeros(num_clips, dtype=torch.int32)
    i = 0
    max_end_frame = end_frame_idx.max().item()
    while container.isOpened():
      ret, frame = container.read()
      if not ret:
        break
      if i > max_end_frame:
        break
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      if self.preprocess_align:
        frame = self.face_extractor.align_face(frame)
      if self.preprocess_crop_detection:
        frame = self.face_extractor.crop_face_detection(frame)
      if self.preprocess_frontalize:
        start = time.time()
        frame,_ = self.face_extractor.frontalize_img(frame=frame,
                                                    ref_landmarks=self.reference_landmarks,
                                                    frontalization_mode='SVD',
                                                    align=self.preprocess_align)
        print(f'Time to frontalize frame {time.time()-start}')
      mask = (start_frame_idx <= i) & (end_frame_idx >= i)  
      
      if mask.any():
        frame_rgb = torch.tensor(frame, dtype=torch.uint8)
        extracted_frames[mask, pos[mask].long()] = frame_rgb
        # idx_frame_saved[mask, pos[mask].long()] = i
        pos[mask] += 1
      i += 1
    # container.seek(0)
    # max_end_frame = end_frame_idx.max().item()
    # for i, frame in enumerate(container.decode(video=0)):
    #   if i > max_end_frame:
    #     break  
      
    #   mask = (start_frame_idx <= i) & (end_frame_idx >= i)  
    #   if mask.any():
    #     frame_rgb = torch.tensor(frame.to_ndarray(format="rgb24"), dtype=torch.uint8)
    #     extracted_frames[mask, pos[mask].long()] = frame_rgb
    #     # idx_frame_saved[mask, pos[mask].long()] = i
    #     pos[mask] += 1
    
    return extracted_frames #,idx_frame_saved
  
  # too slow
  def _read_video_pyav(self, container, list_indices,width_frames,height_frames):
    # ATTENTION: Assume that list_indices is sorted, can be fix using max and min 
    start_frame_idx = list_indices[:, 0]
    end_frame_idx = list_indices[:, -1]
    num_clips, clip_length = list_indices.shape
    # idx_frame_saved = torch.zeros(num_clips,clip_length, dtype=torch.int32)
    extracted_frames = torch.zeros(num_clips, clip_length, height_frames, width_frames, self.image_channels, dtype=torch.uint8)
    pos = torch.zeros(num_clips, dtype=torch.int32)
    container.seek(0)
    max_end_frame = end_frame_idx.max().item()
    for i, frame in enumerate(container.decode(video=0)):
      if i > max_end_frame:
        break  
      
      mask = (start_frame_idx <= i) & (end_frame_idx >= i)  
      if mask.any():
        frame_rgb = torch.tensor(frame.to_ndarray(format="rgb24"), dtype=torch.uint8)
        extracted_frames[mask, pos[mask].long()] = frame_rgb
        # idx_frame_saved[mask, pos[mask].long()] = i
        pos[mask] += 1
    
    return extracted_frames #,idx_frame_saved

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
        # 'preprocess': self.preprocess.__class__.__name__,
        'sample_frame_strategy': self.type_sample_frame_strategy.name,
        'clip_length': self.clip_length,
        'stride_window': self.stride_window,
        'stride_inside_window': self.stride_inside_window,
        'preprocess_align': self.preprocess_align,
        'preprocess_frontalize': self.preprocess_frontalize
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

class customSampler(Sampler):
  def __init__(self,path_cvs_dataset, batch_size, shuffle, random_state=0):
    csv_array,_ = tools.get_array_from_csv(path_cvs_dataset)
    self.y_labels = np.array(csv_array[:,2]).astype(int)
    nr_samples = len(self.y_labels)
    self.n_batch_size = batch_size
    # -(-a//b) is the same as math.ceil(a/b)
    self.skf = StratifiedKFold(n_splits = int(-(-nr_samples//batch_size)) , shuffle=shuffle, random_state=random_state)
    self.n_batches = self.skf.get_n_splits()
    self.initialize()
    
  def initialize(self):
    _, count = np.unique(self.y_labels, return_counts=True)
    max_member = np.max(count)
    print(f'Max count member: {max_member}')
    print(f'Number of splits: {self.skf.get_n_splits()}')
    if max_member < self.skf.get_n_splits():
      raise ValueError(f"n_splits = {len(self.y_labels)//self.n_batch_size +1} cannot be greater than max_count_member ({max_member}) ")
  
  def __iter__(self):
    for _,test in self.skf.split(np.zeros(self.y_labels.shape[0]), self.y_labels):
      yield test
      
  def __len__(self):
    return self.n_batches
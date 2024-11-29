import torch
import pandas as pd
import av
import os
import numpy as np
import torch
import cv2
from custom.helper import SAMPLE_FRAME_STRATEGY
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
import custom.tools as tools

class customDataset(torch.utils.data.Dataset):
  def __init__(self,path_dataset, path_labels, preprocess, sample_frame_strategy, batch_size=1, stride_window=2, clip_length=16):
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
    elif sample_frame_strategy == SAMPLE_FRAME_STRATEGY.CENTRAL_SAMPLING:
      self.sample_frame_strategy = self._central_sampling
      self.stride_window = stride_window
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


  def _generate_csv_subsampled(self, nr_samples_per_class=2):
    csv_array=self.video_labels.to_numpy()
    video_labels_columns = self.video_labels.columns.to_numpy()[0].split('\t')
    # ['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']
    list_samples=[]
    for entry in (csv_array):
      tmp = entry[0].split("\t")
      list_samples.append(tmp)
    list_samples = np.stack(list_samples)
    nr_classes = np.max(list_samples[:,2].astype(int))
    print(f'number of classes: {nr_classes}, \ntotal number of samples: {nr_samples_per_class*nr_classes}')
    for cls in range(nr_classes):
      samples = list_samples[list_samples[:,2].astype(int) == cls]
      samples = samples[np.random.choice(samples.shape[0], nr_samples_per_class, replace=False), :]
      if cls == 0:
        samples_subsampled = samples
      else:
        samples_subsampled = np.concatenate((samples_subsampled,samples),axis=0)
    print(f'samples_subsampled: {samples_subsampled}')
    save_path = os.path.join('partA','starting_point','subsamples_'+str(nr_samples_per_class)+'_'+str(nr_samples_per_class*nr_classes)+'.csv')
    subsampled_df = pd.DataFrame(samples_subsampled, columns=video_labels_columns)
    subsampled_df.to_csv(save_path, index=False, sep='\t')
    print(f'Subsampled video labels saved to {save_path}')
  
  def __getitem__(self, idx):
    # split in # ["subject_id, subject_name, class_id, class_name, sample_id, sample_name"]
    # print(f'idx: {idx}')
    csv_array = self.video_labels.iloc[idx,0].split('\t') 
    video_path = os.path.join(self.path_dataset, csv_array[1], csv_array[5])
    video_path += '.mp4'
    container = av.open(video_path)
    tot_frames = container.streams.video[0].frames
    list_indices = self.sample_frame_strategy(tot_frames)
    frames_list=[self._read_video_pyav(container,indices) for indices in list_indices]
    preprocessed_tensors = torch.stack([self.preprocess(list(frames), return_tensors="pt")['pixel_values'] for frames in frames_list])
    path = np.repeat(video_path, preprocessed_tensors.shape[0])
    
    unit_vector = torch.ones(preprocessed_tensors.shape[0],dtype=torch.int16)    
    sample_id = unit_vector * int(csv_array[4])
    labels = unit_vector * int(csv_array[2])
    subject_id = unit_vector * int(csv_array[0])
    # print(list_indices.shape)
    return  {'preprocess':preprocessed_tensors, 
             'labels': labels, 
             'subject_id': subject_id, 
             'sample_id': sample_id, 
             'path': path, 
             'frame_list': list_indices}

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
    # if torch.cuda.is_available():
    #   device = torch.device('cuda')
    # else:
    device = torch.device('cpu')
    data = torch.stack([batch[index]['preprocess'].squeeze().transpose(1,2).to(device) for index in range(len(batch))]) #[B,T,C,H,W] ->[B,C,T,H,W]
    labels = torch.stack([batch[index]['labels'] for index in range(len(batch))])
    path = np.stack([batch[index]['path'] for index in range(len(batch))])
    subject_id = torch.stack([batch[index]['subject_id'] for index in range(len(batch))])
    sample_id = torch.stack([batch[index]['sample_id'] for index in range(len(batch))])
    list_frames = torch.stack([batch[index]['frame_list'] for index in range(len(batch))]).squeeze()
    # print(f'list_frame_collate: {list_frames.shape}')
    return data.reshape(-1,self.image_channels, self.clip_length, self.image_resize_h, self.image_resize_w), labels, subject_id,\
           sample_id, path, list_frames
  
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
    assert video_len // 2 - (self.clip_length // 2) * self.stride_window >= 0, f"Video is too short for the given clip length. Reduce the stride (given {self.stride_window})"
    start_idx = video_len // 2 - (self.clip_length // 2) * self.stride_window
    # print('Start index',start_idx)
    indices = torch.arange(start_idx, start_idx + self.clip_length * self.stride_window, self.stride_window)[None,:]
    # print('Indices central',indices)
    return indices
  
  def _random_sampling(self, video_len):
    indices = torch.randperm(video_len,dtype=torch.int16)[:self.clip_length]
    indices = torch.sort(indices).values[None, :]
    # print('Random non-repeated indices', indices)
    # print('Random shape', indices.shape)
    return indices
  
  def _sliding_window_sampling(self,video_len):
    """
    Generates a list of indices for sliding window sampling of a video.

    Args:
      video_len (int): The total length of the video in frames.

    Returns:
      torch.Tensor: A tensor containing the indices for each sliding window.
              Each row corresponds to a window and contains the indices
              of the frames within that window.
    """
    indices = torch.arange(0, video_len - self.clip_length, self.stride_window)
    list_indices = torch.stack([torch.arange(start_idx, start_idx + self.clip_length) for start_idx in indices])
    # print('Sliding shape',list_indices.shape)
    return list_indices
  
  # def plot_dataset_distribution_mean_std_duration(self,per_class=False,per_partecipant=False,saving_path=None):
  #   def plot_distribution(key,title):  
  #     key_dict = {} # key: subject_id -> value: [duration, N]
  #     for idx, sample in enumerate(list_samples):
  #       # csv_array = self.video_labels.iloc[idx,0].split('\t') 
  #       key_id = sample[key]
  #       video_path = os.path.join(self.path_dataset, sample[1], sample[5])
  #       video_path += '.mp4'
  #       container = av.open(video_path)
  #       duration_secs = container.streams.video[0].frames // container.streams.video[0].average_rate
  #       if key_id not in key_dict:
  #         key_dict[key_id] = []
  #         key_dict[key_id].append(duration_secs)
  #     key_id_vector = np.array(list(key_dict.keys())).astype(int)
  #     mean_duration = np.array([np.mean(key_dict[key]) for key in key_dict.keys()])
  #     std_duration = np.array([np.std(key_dict[key]) for key in key_dict.keys()])
  #     indices = key_id_vector.argsort() # TODO: print also elements not availables 
  #     dataset_name = f'{os.path.split(self.path_labels)[-1].split(".")[0]}'
  #     plt.figure(figsize=(30, 12))
  #     plt.bar(key_id_vector[indices].astype(str), mean_duration[indices], yerr=std_duration[indices], color="blue", alpha=0.8)
  #     plt.xlabel(title,fontsize=16)
  #     plt.ylabel("mean duration (s)",fontsize=16)
  #     plt.title(f"Mean Duration per {title} with std ({dataset_name})",fontsize=16)
  #     plt.xticks(rotation=45,fontsize=13)
  #     plt.yticks(fontsize=13)
  #     plt.grid(axis="y", linestyle="--", alpha=0.7)
  #     # Show the plot
  #     # plt.tight_layout()
  #     if saving_path is not None:
  #       plt.savefig(os.path.join(saving_path,f'{title}_{dataset_name}.png'))
  #     else:
  #       plt.show()
    
  #   csv_array=self.video_labels.to_numpy() # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  #   list_samples=[]
  #   for entry in (csv_array):
  #     tmp = entry[0].split("\t")
  #     list_samples.append(tmp)
  #   list_samples = np.stack(list_samples)
  #   if per_partecipant is True:
  #     key = 0
  #     plot_distribution(key,'participant')
  #   if per_class is True:
  #     key = 2
  #     plot_distribution(key,'class')
    
          
  def save_frames_as_video(self,list_input_video_path, list_frame_indices, output_video_path,all_predictions,list_ground_truth, output_fps=1):
    """
    Extract specific frames from a video and save them as a new video.

    :param input_video_path: Path to the original video file.
    :param frame_indices: List of frame indices to extract and save.
    :param output_video_path: Path to save the output video.
    :param output_fps: Frames per second for the output video (default is 30).
    """
    # Open the original video
    out = None
    # if len(all_predictions.shape)>2:
    #   all_predictions = all_predictions.reshape(all_predictions.shape[0],-1)
    # if len(list_ground_truth.shape) :
    print(f'all_predictions shape: {all_predictions.shape}')
    # print('all_labels',list_ground_truth)
    # print(f'output_video_path: {output_video_path}')
    # print('pred',all_predictions)
    # print('gt',list_ground_truth)
    # partA/video/video/112209_m_51_112209_m_51-BL1-086.mp4
    for i, input_video_path in enumerate(list_input_video_path): # i->[0...33]
      print(f'input_video_path: {input_video_path}')
      cap = cv2.VideoCapture(input_video_path)
      if not cap.isOpened():
        raise IOError(f"Err: Unable to open video file: {input_video_path}")

      # Get the width and height of the frames in the video
      frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      frame_size = (frame_width, frame_height)

      # Define the codec and create VideoWriter object
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
      if out is None:
        out = cv2.VideoWriter(output_video_path, fourcc, output_fps, frame_size)
      frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      # print(f'frame_count: {frame_count}')
      black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
      font = cv2.FONT_HERSHEY_SIMPLEX
      font_scale = 1
      font_color = (255, 255, 255)
      thickness = 2
      count = 0
      # print(f'frame_indices: {len(list_frame_indices)}')
      for j,frame_indices in enumerate(list_frame_indices[i]): #j->[0..1] [16]
        # print('frame_indices',frame_indices) # i->[0,...,32] framle_inidces->[2,16]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the video capture to the beginning
        for _ in range(output_fps):
          number_frame = black_frame.copy()
          text = os.path.split(input_video_path)[-1]
          text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
          text_x = (number_frame.shape[1] - text_size[0]) // 2
          text_y = (number_frame.shape[0] + text_size[1]) // 2
          cv2.putText(number_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
          out.write(number_frame)
        for frame_idx in range(frame_count):
          ret, frame = cap.read()
          if not ret:
            break
            # Check if the current frame index is in the list
          # print(f'frame_indices: {frame_indices}')
          if frame_idx in frame_indices: # [2,16]
            # print(f'GT:{list_ground_truth.shape}')
            # print(f'pred:{all_predictions.shape}')
            # print(i,j)
            # print(list_ground_truth[i][j])
            # print(all_predictions[i][j])
            pred_rounded = np.round(all_predictions[i][j],2)
            cv2.putText(frame, str(count)+'/'+str(frame_idx), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), thickness, cv2.LINE_AA)
            cv2.putText(frame, f'gt:{list_ground_truth[i][j]}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), thickness, cv2.LINE_AA)
            cv2.putText(frame, f'pred:{np.round(pred_rounded,2)}', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), thickness, cv2.LINE_AA)
            out.write(frame)
        count+=1
          # print(f'frame_idx: {frame_idx}')
    # Release resources
    cap.release()
    out.release()
    print(f"Saved extracted frames to {output_video_path}")
    # Add 4 black frames at the end
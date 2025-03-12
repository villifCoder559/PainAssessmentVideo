import torch
import torchvision.transforms as T
import cv2
from custom.backbone import video_backbone
from custom.helper import MODEL_TYPE
from custom.dataset import customDataset
import os 
import numpy as np
import torch.nn.functional as F
import tqdm
import argparse

def preprocess_images(tensors):
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
    T.Normalize(mean=image_mean, std=image_std)  # Normalize,
  ])
  
  preprocessed_tensors = torch.stack([transform(t) for t in tensors])
  return preprocessed_tensors

def read_video(video_path):
  """
  Read a video file and return a tensor.
  
  Args:
      video_path (str): Path to the video file.
  
  Returns:
      torch.Tensor: Tensor of shape (T, H, W, C).
  """
  video = cv2.VideoCapture(video_path)
  frames = []
  while True:
    ret, frame = video.read()
    if not ret:
      break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
  video.release()
  frames_np = np.array(frames)
  return torch.from_numpy(frames_np).permute(0, 3, 1, 2)

def uniform_sampling(video_tensor, num_frames):
  """
  Uniformly sample `num_frames` frames from the video tensor.
  
  Args:
      video_tensor (torch.Tensor): Tensor of shape (T, C, H, W).
      num_frames (int): Number of frames to sample.
  
  Returns:
      torch.Tensor: Tensor of shape (num_frames, C, H, W).
  """
  T = video_tensor.size(0)
  indices = torch.linspace(0, T - 1, num_frames).round().long()
  return video_tensor[indices]

def stride_sampling(video_tensor,stride, num_frames):
  """
  Stride sample `num_frames` frames from the video tensor.
  
  Args:
      video_tensor (torch.Tensor): Tensor of shape (T, C, H, W).
      num_frames (int): Number of frames to sample.
  
  Returns:
      torch.Tensor: Tensor of shape (num_frames, C, H, W).
  """
  T = video_tensor.size(0)
  start_idx = torch.randint(0, T - stride*num_frames, (1,)).item()
  indices = torch.arange(start_idx, start_idx + stride*num_frames, stride)
  return video_tensor[indices]

def get_clips(video_tensor, num_frames_per_clip, num_clips,stride):
  """
  Split the video tensor into clips.
  
  Args:
      video_tensor (torch.Tensor): Tensor of shape (T, C, H, W).
      num_frames_per_clip (int): Number of frames in each clip.
      num_clips (int): Number of clips to sample.
  
  Returns:
      torch.Tensor: Tensor of shape (num_clips, num_frames_per_clip, C, H, W).
  """
  clips = []
  for _ in range(num_clips):
    clip = stride_sampling(video_tensor, stride, num_frames_per_clip)
    clips.append(clip)
  return torch.stack(clips)

def fix_seeds(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.enabled = False

if __name__ == "__main__":
  arg_parser =  argparse.ArgumentParser()
  arg_parser.add_argument("--model_type", type=str, default="B", help="Model type to use")
  arg_parser.add_argument("--num_clips", type=int, default=4, help="Number of clips to sample")
  arg_parser.add_argument("--stride", type=int, default=2, help="Stride to use when sampling clips")
  args = arg_parser.parse_args()
  model_type = MODEL_TYPE.VIDEOMAE_v2_S if args.model_type == "S" else MODEL_TYPE.VIDEOMAE_v2_B
  model = video_backbone(model_type = model_type,remove_head=False)
  dataset_path = os.path.join("toy_dataset", "Weizmann_dataset")
  classes = os.listdir(dataset_path)
  # print(classes)
  list_video_path = []
  list_labels = []
  for c in classes:
    class_path = os.path.join(dataset_path, c)
    for video in os.listdir(class_path):
      list_video_path.append(os.path.join(class_path, video))
      list_labels.append(c)
  list_results = [] 
  range_len = range(len(list_video_path))
  fix_seeds()
  for i in tqdm.tqdm(range_len):
    video_frames = read_video(list_video_path[i])
    video_tensor = preprocess_images(video_frames)
    # video_tensor = stride_sampling(video_tensor,2 ,16).unsqueeze(0)
    video_tensor = get_clips(video_tensor=video_tensor,
                            num_frames_per_clip=16,
                            num_clips=args.num_clips,
                            stride=args.stride)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)
    model.model.eval()
    with torch.no_grad():
      video_embedding = model.model.forward(video_tensor) # nr_clips, channels, nr_frames=16, H, W
    mean_softmax = torch.mean(F.softmax(video_embedding, dim=1), dim=0)
    prediction = torch.argmax(mean_softmax,dim=0).detach().cpu().item()
    list_results.append(prediction)
    del video_tensor
    
  with open("k710_labels/k710_label_map.txt",'r') as f:
    label_map = f.readlines()
    label_map = [x.strip() for x in label_map]
  
  label_map = np.array(label_map)
  label_prediction_dict = {}
  np_labels = np.array(list_labels)
  class_results = np.array(list_results)
  for label in np.unique(np_labels):
    # print(f"Label: {label}, Prediction: {prediction}")
    mask = (np_labels == label)
    masked_results  = class_results[mask]
    label_prediction_dict[label] = masked_results
  prediction_to_label = {k:(list(zip(label_map[np.unique(v)],np.unique(v,return_counts=True)[1]))) for k,v in label_prediction_dict.items()}
  for k,v in prediction_to_label.items():
    print(f'{k}: {v}')
  accuracy_dict = {}
  for label in label_prediction_dict:
    count_prediction = np.bincount(label_prediction_dict[label])
    accuracy_dict[label] = count_prediction.max().item() / count_prediction.sum().item()
  print('\nAccuracy dict:')
  for k,v in accuracy_dict.items():
    print(f'  {k}: {v}')
  print('Mean accuracy:', np.mean(list(accuracy_dict.values())))

import pickle
import matplotlib.pyplot as plt
import numpy as np  
import os

import tqdm
from custom.dataset import customDataset
import cv2
import custom.tools as tools
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import argparse


def plot_space_importance(xattn):
  xattn = xattn / xattn.sum()  # Normalize to sum to 1
  spatial_total = xattn.sum(axis=0)  # shape (H, W)
  plt.imshow(spatial_total, interpolation='bilinear')
  plt.title("Spatial attention summed over time (normalized)")
  plt.colorbar()
  plt.axis('off')
  plt.show()

def plot_frame_importance(xattn,list_indices=None):
  if list_indices is not None:
    temporal_indices = np.array(list_indices)
  else:
    temporal_indices = np.arange(xattn.shape[0])
  xattn = xattn / xattn.sum()  # Normalize to sum to 1
  frame_importance = xattn.sum(axis=(1,2))  # shape (T,)
  plt.figure(figsize=(12,8))
  plt.bar(temporal_indices, frame_importance, width=1.0, align='center')
  # show all indices
  plt.xticks(temporal_indices)
  plt.xlabel("Frame index")
  plt.ylabel("Total attention (normalized)")
  plt.title("Frame importance (sum of spatial attention)")
  plt.show()
  
  
def get_pred_gt(sample_id, df, data):
  dict_test_predictions = data['results']['history_test_sample_predictions']
  gt = df[df['sample_id'] == sample_id]['class_id'].values[0]
  if sample_id not in dict_test_predictions:
    raise ValueError(f"Sample id {sample_id} not found in test predictions")
  pred = dict_test_predictions[sample_id].item()
  return gt, pred

def get_xattn_from_sample(data, sample_id):
  list_batches = [x[0] for x in data['cross_attention']['debug_xattn_test']]
  batch_idx = -1
  batch_pos = -1
  for idx, batch in enumerate(list_batches):
    if sample_id in batch:
      batch_idx = idx
      batch_pos = np.where(batch == sample_id)[0][0]
      print(f"Found sample id {sample_id} in batch {idx} at position {batch_pos}")  
      break
  if batch_idx == -1 or batch_pos == -1:
    raise ValueError(f"Sample id {sample_id} not found in any batch")  
  xattn = data['cross_attention']['debug_xattn_test'][batch_idx][1][batch_pos]
  return xattn  # shape (1, 1, T * S * S)


def get_list_frames_from_sample(data, sample_id, create_video=False):
  # Get config
  feats_path = data['config_model']['model_advanced_params']['features_folder_saving_path']

  config_dict = os.path.join(feats_path, "config_dict.pkl")
  with open(config_dict, 'rb') as f:
    config = pickle.load(f)

  # Retrieve list frames for the sample id
  custom_ds = customDataset(**config)
  dict_video = custom_ds.generate_video(sample_id=sample_id) # list_frames -> [B,H,W,C]

  # Generate video from frames
  if create_video:
    tools.generate_video_from_list_frame(list_frame=dict_video['list_frames'],
                                        fps=4,
                                        path_video_output=f"z_debug_frontalization/test_video_{sample_id}.mp4")  
  return dict_video['list_frames'], dict_video['list_indices'].reshape(-1)

def get_resized_xattn(data, xattn, video_frames):
  # Check if quadrants are used
  quadrants = 1
  if 'combined' in data['config_model']['model_advanced_params']['features_folder_saving_path'] and not data['config_model']['config']['concatenate_quadrants']:
    quadrants = 4
    print("Using quadrants for attention visualization")
  
  # Reshape xattn to (quadrants, nr_chunks * T, S, S)  
  T,S,S = data['config_model']['model_advanced_params']['head_params']['T_S_S_shape']
  nr_frames, Hf, Wf, _ = video_frames.shape
  xattn = xattn.reshape(quadrants, -1, T, S, S)  
  
  # Due to padding for batching, the number of chunks in xattn might be higher than expected
  expected_nr_chunks = nr_frames // (T * 2) # 16 chunk input size for VideoMAE_v2 
  if xattn.shape[1] != expected_nr_chunks:
    xattn = xattn[:, :expected_nr_chunks, :, :, :]
    print(f"Trimmed xattn to expected number of chunks: {expected_nr_chunks}")
  xattn = xattn.reshape(quadrants, -1, S, S)  # shape (quadrants, nr_chunks * T, S, S)
  
  # Resize attention to frame size, repeating if needed
  repeated_xattn = np.repeat(xattn,repeats=nr_frames//xattn.shape[1], axis=1)  # shape (nr_frames, S, S)
  if quadrants == 4:
    Hf = Hf // 2
    Wf = Wf // 2
    print(f"Resized frame to {Hf}x{Wf} for quadrants")
  repeated_xattn = np.stack([[cv2.resize(repeated_xattn[q,t], (Wf, Hf), interpolation=cv2.INTER_LINEAR) # cv2.INTER_NEAREST, INTER_LINEAR
                          for t in range(nr_frames)] for q in range(quadrants)])  # shape (T, Hf, Wf)
  repeated_xattn = np.round(repeated_xattn, decimals=6)
  
  # Compose quadrants if needed, otherwise squeeze
  if quadrants == 4:
    composed_xattn = repeated_xattn.transpose(1,0,2,3).reshape(nr_frames, quadrants//2,quadrants//2, Hf, Wf)  # The new shape => (nr_frames, upper_l+r, bottom_l+r , Hf, Wf)
    composed_xattn = composed_xattn.transpose(0,1,3,2,4)
    composed_xattn = composed_xattn.reshape(nr_frames, 
                                            composed_xattn.shape[1]*composed_xattn.shape[2], # merge upper quadrants
                                            composed_xattn.shape[3]*composed_xattn.shape[4]) # merge bottom quadrants
    xattn = composed_xattn
  else:
    xattn = repeated_xattn.squeeze(0) # remove quadrants dim
  eps = 1e-6
  
  return xattn  # shape (nr_frames, Hf, Wf)

def create_frame_with_overlaid_attention(list_indices, video_frames, xattn, sample_id, gt, pred, run_id, head,path_video_output,
                                         return_overlaid_frames=False, create_video=True):
  image_list = []
  frames_plot = zip(list_indices, video_frames, xattn)
  for idx , frame, attn_map in frames_plot:
    # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(frame)
    im = ax.imshow(attn_map, cmap='jet', alpha=0.5)
    fig.colorbar(im, fraction=0.046, pad = 0.04)
    ax.set_title(f"Head {head} - Sample {sample_id} - Frame {idx} - gt: {gt}, pred: {pred}")
    ax.axis('off')
    fig.tight_layout(pad=0)
    # plt.show()
    
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.2)
    buf.seek(0)
    img_arr = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    image_list.append(img_arr)
    plt.close(fig)
    
  if create_video:
    tools.generate_video_from_list_frame(list_frame=image_list,
                                        fps=2,
                                        path_video_output=os.path.join(path_video_output, f"test_attention_{sample_id}.mp4"))
  if return_overlaid_frames:
    return image_list

def get_data_from_pkl(xattn_path):
  with open(xattn_path, 'rb') as f:
    data = pickle.load(f)
  if data['config_model']['config']['concatenate_quadrants']:
    raise NotImplementedError("Concatenate quadrants not implemented")
  run_id = xattn_path.split('/')[-4].split('_')[0]

  return data, run_id

def multihead_video_creation(xattn,list_indices, video_frames, sample_id, gt, pred, run_id, path_video_output):
  nr_heads = xattn.shape[0]
  if nr_heads == 8:
    grid_size = (2,4)
  elif nr_heads == 4:
    grid_size = (2,2)
  else:
    raise NotImplementedError(f"Multihead video creation not implemented for {nr_heads} heads")
  list_videos = []
  for h in tqdm.tqdm(range(nr_heads), desc="Creating head videos"):
    xattn_h = xattn[h]
    xattn_h = get_resized_xattn(data, xattn_h, video_frames)  # shape (nr_frames, Hf, Wf)
    list_videos.append(create_frame_with_overlaid_attention(list_indices, video_frames, xattn_h, sample_id, gt, pred, run_id, head=h,
                                         return_overlaid_frames=True, create_video=False))
  
  # Create video
  try:
    list_videos = np.array(list_videos)  # shape (nr_heads, nr_frames, H, W, C)
  except Exception as e:
    # Resize frames to the minimum size
    min_H = min([min([frame.shape[0] for frame in video]) for video in list_videos])
    min_W = min([min([frame.shape[1] for frame in video]) for video in list_videos])
    print(f"Resizing frames to minimum size {min_H}x{min_W}")
    list_videos = np.array([[cv2.resize(frame, (min_W, min_H), interpolation=cv2.INTER_NEAREST) for frame in video] for video in list_videos]) # cv2.INTER_NEAREST, INTER_LINEAR
  
  # Create grid video  
  nr_frames, H, W, C = list_videos.shape[1:]
  grid_H = grid_size[0] * H
  grid_W = grid_size[1] * W
  grid_video = []
  for f in range(nr_frames):
    grid_frame = np.zeros((grid_H, grid_W, C), dtype=list_videos.dtype)
    for h in range(nr_heads):
      row = h // grid_size[1]
      col = h % grid_size[1]
      grid_frame[row*H:(row+1)*H, col*W:(col+1)*W, :] = list_videos[h, f]
    grid_video.append(grid_frame)
  # Save grid video
  tools.generate_video_from_list_frame(list_frame=grid_video,
                                      fps=2,
                                      path_video_output=os.path.join(path_video_output, f"test_attention_{sample_id}_multihead_grid.mp4"))
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--xattn_path', type=str, help='Path to the pkl file with cross attention')
  parser.add_argument('--sample_id', default = None, type=int, help='Sample ID to visualize. If none all samples will be processed')
  dict_args = vars(parser.parse_args())
  # # NO REGUL 
  # xattn_path = "TRAIN_tests_CAER/history_run_filtered_CROPPED_CAER_reg_441775_ATTENTIVE_JEPA_lechuck_1758744831/1758744833192_VIDEOMAE_v2_S_NONE_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/train_ATTENTIVE_JEPA/k0_cross_val/test_with_cross_attention_1758815333.pkl"

  # sample_id = dict_args['sample_id']
  # xattn_path = dict_args['xattn_path']
  # xattn_path = "TRAIN_tests_model/history_run_small_ATTN_high_drop_670666_ATTENTIVE_JEPA_targaryen_1758703044/1758703059114_VIDEOMAE_v2_S_NONE_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/train_ATTENTIVE_JEPA/k0_cross_val/test_with_cross_attention_1758715798.pkl"
  path_video_output = os.path.join(*dict_args['xattn_path'].split('/')[:-1], 'video_attention') 
  
  if dict_args['sample_id']:
    list_sample_id = [dict_args['sample_id']]
  else:
    csv_path = os.path.join(*dict_args['xattn_path'].split('/')[:-1], 'csv_subset.csv')
    df = pd.read_csv(csv_path,sep='\t')
    list_sample_id = df['sample_id'].to_list()
    print(f"Processing all {len(list_sample_id)} samples from csv {csv_path}")
    
  # Get data from pkl
  data, run_id = get_data_from_pkl(dict_args['xattn_path'])
  
  # Get the df from csv
  csv_path = data['csv_path']
  df = pd.read_csv(csv_path,sep='\t')
  
  for sample_id in tqdm.tqdm(list_sample_id, desc="Processing samples"):
    if sample_id not in df['sample_id'].values:
      raise ValueError(f"Sample id {sample_id} not found in csv {csv_path}")
    gt, pred = get_pred_gt(sample_id, df, data)
    video_frames,list_indices = get_list_frames_from_sample(data, sample_id, create_video=False)
    xattn = get_xattn_from_sample(data, sample_id)
    if xattn.shape[0]>1:
      multihead_video_creation(xattn, list_indices, video_frames, sample_id, gt, pred, run_id)
      # Resize attention to frame size
    else:
      xattn = get_resized_xattn(data, xattn, video_frames)  # shape (nr_frames, Hf, Wf)
      
      # Plot attention on frames
      create_frame_with_overlaid_attention(list_indices,
                                           video_frames,
                                           xattn,
                                           sample_id,
                                           gt,
                                           pred,
                                           run_id,
                                           head=0,
                                           path_video_output=path_video_output)

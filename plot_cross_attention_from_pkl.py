import pickle
import matplotlib.pyplot as plt
import numpy as np  
import os
from scipy.ndimage import gaussian_filter1d
import tqdm
from custom.dataset import customDataset
import cv2
import custom.tools as tools
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import argparse
import shutil
from pathlib import Path

def plot_space_importance(xattn,plot_folder=None,title='',png_name=None): # xattn shape (T, S, S) or (S,S)
  if xattn.ndim != 3 and xattn.ndim != 2:
    raise ValueError(f"xattn should be 2D or 3D array (S, S) or (T, S, S), got {xattn.shape}")
  xattn = xattn / xattn.sum()  # Normalize to sum to 1
  
  if xattn.ndim == 3:
    spatial_total = xattn.sum(axis=0)  # shape (S, S)
  else:
    spatial_total = xattn
    
  plt.title(f"{title}.\n Spatial attention summed over time (normalized). ", wrap=True)
  plt.imshow(spatial_total, interpolation='bilinear')
  plt.colorbar()
  plt.axis('off')
  plt.tight_layout()
  if plot_folder is not None:
    os.makedirs(plot_folder, exist_ok=True)
    png_name = png_name if png_name is not None else title
    plt.savefig(os.path.join(plot_folder, f"{png_name}.png"))
    plt.close()
  else:
    plt.show()

def plot_frame_importance(xattn,list_indices=None): # xattn shape (T, S, S) 
  if xattn.ndim != 3:
    raise ValueError(f"xattn should be 3D array (T, S, S), got {xattn.shape}")
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
  
  
def get_gt_pred(sample_id, df, data):
  epoch = 0 # always 0 for test
  dict_test_predictions = data['results']['history_test_sample_predictions']
  gt = df[df['sample_id'] == sample_id]['class_id'].values[0]
  if sample_id not in dict_test_predictions:
    raise ValueError(f"Sample id {sample_id} not found in test predictions")
  pred = dict_test_predictions[sample_id][epoch]
  return gt, pred

def get_xattn_from_sample(data, sample_id, multiple_ids=False):
  list_batches = [x[0] for x in data['cross_attention']['debug_xattn_test']]
  if multiple_ids:
    batch_idx = []
    batch_pos = []
    xattn = []
    for idx, batch in enumerate(list_batches):
      if sample_id in batch:
        batch_idx.append(idx)
        batch_pos.append(np.where(batch == sample_id)[0])
        batch_pos = np.concatenate(batch_pos).tolist()
        xattn.extend([data['cross_attention']['debug_xattn_test'][idx][1][b_pos] for b_pos in batch_pos])
        # print(f"Found sample id {sample_id} in batch {idx} at position {batch_pos}")  
    if len(batch_idx) == 0:
      raise ValueError(f"Sample id {sample_id} not found in any batch")
    # xattn = [data['cross_attention']['debug_xattn_test'][b_idx][1][b_pos] for b_idx, b_pos in zip(batch_idx, batch_pos)]
    return np.concatenate(xattn)  # shape (N, 1, T * S * S)
  else:
    batch_idx = -1
    batch_pos = -1
    for idx, batch in enumerate(list_batches):
      if sample_id in batch:
        batch_idx = idx
        batch_pos = np.where(batch == sample_id)[0][0]
        # print(f"Found sample id {sample_id} in batch {idx} at position {batch_pos}")  
        if not multiple_ids:
          break
    if batch_idx == -1 or batch_pos == -1:
      raise ValueError(f"Sample id {sample_id} not found in any batch")  
    xattn = data['cross_attention']['debug_xattn_test'][batch_idx][1][batch_pos]
    return xattn  # shape (1, 1, T * S * S)


def get_list_frames_from_sample(data, sample_id, create_video=False):
  # Get config
  feats_path = data['config_model']['model_advanced_params']['features_folder_saving_path']

  config_dict = os.path.join(feats_path, "config_dict.pkl")
  if 'raid' in config_dict:
    config_dict = config_dict.replace('raid','dune')
  
  with open(config_dict, 'rb') as f:
    config = pickle.load(f)
  # Retrieve list frames for the sample id
  if 'caer' in feats_path.lower():
    config['video_extension'] = '.avi'
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
  xattn = xattn.reshape(quadrants, -1, T, S, S)  
  
  if video_frames is None:
    return xattn
  
  nr_frames, Hf, Wf, _ = video_frames.shape
  
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
  # repeated_xattn = np.round(repeated_xattn, decimals=6)
  
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

def create_frame_with_overlaid_attention(list_indices, video_frames, xattn, sample_id, gt, pred, run_id, head,path_video_output
                                         ,FPS=4,max_range_plot=None,min_range_plot=None):
  image_list = []
  frames_plot = zip(list_indices, video_frames, xattn)
  for idx , frame, attn_map in frames_plot:
    # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(frame)
    # set color range if specified
    if max_range_plot is not None and min_range_plot is not None:
      im = ax.imshow(attn_map, cmap='jet', alpha=0.5, vmin=min_range_plot, vmax=max_range_plot)
    else:
      im = ax.imshow(attn_map, cmap='jet', alpha=0.5)
    fig.colorbar(im, fraction=0.046, pad = 0.04, )
    ax.set_title(f"Head {head} - Sample {sample_id} - gt: {gt}, pred: {pred:.2f}")
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
    
  if path_video_output is not None:
    tools.generate_video_from_list_frame(list_frame=image_list,
                                        fps=FPS,
                                        path_video_output=os.path.join(path_video_output, f"test_attention_{sample_id}_PA{gt}_{pred}.mp4"))
  else:
    return image_list

def get_data_from_pkl(xattn_path):
  with open(xattn_path, 'rb') as f:
    data = pickle.load(f)
  if data['config_model']['config']['concatenate_quadrants']:
    raise NotImplementedError("Concatenate quadrants not implemented")
  
  run_id = Path(xattn_path).parts[-4].split('_')[0]

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
    
def plot_average_attention_over_samples(data, list_sample_id,plot_folder,title,png_name=None,multiple_ids=False):
  list_xattn = []
  for sample_id in list_sample_id:
    xattn = get_xattn_from_sample(data, sample_id, multiple_ids=multiple_ids)
    xattn = get_resized_xattn(data, xattn, video_frames=None)  # shape (quadrants, nr_chunks, T, S, S)
    
    # Mean per sample over quadrants, chunks, T
    list_xattn.append(xattn.mean(axis=(0,1,2))) # shape (S, S)
  
  list_xattn = np.array(list_xattn)
  plot_space_importance(list_xattn,plot_folder=plot_folder,title=title,png_name=png_name)
  
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import io

def make_base_plot_image(pain_predictions, smoothing_sigma=1, fig_size=(4, 3), dpi=100,**kwargs):
  """
  Create the base plot (smoothed curve + scatter + axes) WITHOUT the vertical current-frame line.
  Returns:
    base_img_bgr: numpy array (H,W,3) BGR uint8 - ready for cv2 operations
    ax: matplotlib Axes object (so we can map data coords to pixel coordinates)
    fig: the Matplotlib figure (must be kept alive until done)
  """
  smoothed = gaussian_filter1d(pain_predictions, sigma=smoothing_sigma)
  time = np.arange(len(pain_predictions))

  fig = plt.figure(figsize=fig_size, dpi=dpi)
  ax = fig.add_subplot(111)
  ax.plot(time, smoothed, color='crimson', linewidth=2, label='Smoothed Pain')
  ax.scatter(time, pain_predictions, color='black', s=20, zorder=3, label='Predictions')
  ax.set_xlim(0, len(pain_predictions))
  ax.set_xticks(np.arange(0, len(pain_predictions)+1, step=16))
  ax.set_ylim(0, 5)  # Assuming pain levels are between 0 and 5
  ax.set_xlabel("Time (frames)")
  ax.set_ylabel("Pain Level")
  ax.legend()
  ax.grid(alpha=0.3)
  ax.set_title(f"Pain predictions over time - video gt: {kwargs.get('gt','N/A')}\nSample name: {kwargs.get('sample_name','N/A')} - csv: {kwargs.get('csv_name','N/A')}", fontsize=10)
  plt.tight_layout()

  # Render figure to RGB buffer
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
  rgb = buf.reshape((h, w, 4))  # 4 channels for RGBA

  # Convert RGB -> BGR for OpenCV
  bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)

  return bgr, ax, fig

def draw_vertical_line_on_plot(base_img_bgr, ax, fig, x_data, line_color=(255, 144, 30), line_thickness=2):
  """
  Draw a vertical line at data x coordinate x_data onto a copy of base_img_bgr.
  line_color is in BGR (default: a dodgerblue-ish in BGR).
  Returns the image with the line.
  """
  disp_x, _ = ax.transData.transform((x_data, 0))  # data->pixel coords
  x_px = int(round(disp_x))
  img = base_img_bgr.copy()
  h, w, _ = img.shape
  x_px = np.clip(x_px, 0, w - 1)
  cv2.line(img, (x_px, 0), (x_px, h - 1), color=line_color, thickness=line_thickness, lineType=cv2.LINE_AA)
  return img

def create_video_with_plot_and_predictions(video_path, pain_predictions, gt=None,
                                smoothing_sigma=1, fig_size=(8, 4), dpi=100, **kwargs):
  """
  Efficient version: pre-renders the plot base image once and draws the vertical line per-frame using OpenCV.
  If len(pain_predictions) != video frame count, it will interpolate predictions to match frames.
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {video_path}")

  # fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if total_frames <= 0:
    raise RuntimeError("Video has zero frames or cannot read frame count.")

  # Interpolate pain predictions to match video frames if necessary
  preds = np.asarray(pain_predictions, dtype=float)
  if len(preds) != total_frames:
    repeats = total_frames // len(preds)
    preds_interp = np.repeat(preds, repeats)
    if len(preds_interp) < total_frames:
      # add 0 padding at the end if needed
      preds_interp = np.pad(preds_interp, (0, total_frames - len(preds_interp)), mode='edge')
    elif len(preds_interp) > total_frames:
      preds_interp = preds_interp[:total_frames]
    # Alternative interpolation method (commented out)
    # src_idx = np.linspace(0, total_frames - 1, num=len(preds))
    # tgt_idx = np.arange(total_frames)
    # preds_interp = np.interp(tgt_idx, src_idx, preds)
  else:
    preds_interp = preds

  # Build base plot image
  base_img_bgr, ax, fig = make_base_plot_image(preds_interp, smoothing_sigma=smoothing_sigma,
                                               fig_size=fig_size, dpi=dpi, gt=gt, **kwargs)

  # Read one frame to get size
  cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  ret, frame = cap.read()
  if not ret:
    cap.release()
    plt.close(fig)
    raise RuntimeError("Error reading first frame of video.")

  frame_h, frame_w = frame.shape[:2]
  plot_h, plot_w = base_img_bgr.shape[:2]

  # Resize plot to match video frame height
  plot_resized = cv2.resize(base_img_bgr, (int(plot_w * frame_h / plot_h), frame_h))
  plot_w_resized = plot_resized.shape[1]
  # out_w = frame_w + plot_w_resized
  # fourcc = cv2.VideoWriter_fourcc(*'avc1')
  # out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, frame_h))

  scale_x = plot_w_resized / plot_w

  # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
  list_frames = []
  for t in range(total_frames):
    ret, frame = cap.read()
    if not ret:
      break

    disp_x_orig, _ = ax.transData.transform((t, 0))
    disp_x_scaled = int(round(disp_x_orig * scale_x))
    disp_x_scaled = np.clip(disp_x_scaled, 0, plot_w_resized - 1)

    plot_with_line = plot_resized.copy()
    cv2.line(plot_with_line, (disp_x_scaled, 0), (disp_x_scaled, frame_h - 1),
             color=(255, 144, 30), thickness=2, lineType=cv2.LINE_AA)

    combined = np.hstack((frame, plot_with_line))
    list_frames.append(combined)
    # out.write(combined)

  # cap.release()
  # out.release()
  plt.close(fig)
  return list_frames
  
def plot_histo_split_video_predictions(predictions, title, path_images_output):
  predictions = np.array(predictions)
  plt.figure(figsize=(10,5))
  plt.bar(np.arange(0, len(predictions)), predictions, width=0.8, align='center', color='skyblue', edgecolor='black', alpha=0.7)
  plt.xticks(np.arange(0, len(predictions), step=1))
  plt.xlabel("Chunk number")
  plt.ylabel("Mean pain prediction")
  plt.title(title)
  plt.ylim(0, 5)  # Assuming pain levels are between 0 and 5
  plt.grid(axis='y', alpha=0.75)
  plt.tight_layout()
  plt.savefig(path_images_output)
  plt.close()

def generate_video_from_folder(folder_path, output_video_path, fps=2):
  images_path = [img for img in os.listdir(folder_path) if img.endswith(".png")]
  images_path.sort()
  frame = cv2.imread(os.path.join(folder_path, images_path[0]))
  height, width, _ = frame.shape
  video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
  for image_path in tqdm.tqdm(images_path,desc="Generating video from images"):
    frame = cv2.imread(os.path.join(folder_path, image_path))
    if frame.shape[0] != height or frame.shape[1] != width:
      frame = cv2.resize(frame, (width, height))
    video.write(frame)
  video.release()
  
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--xattn_path',required=True, type=str, help='Path to the pkl file with cross attention')
  parser.add_argument('--plot_space_correctly_pred', type=int,default=1, help='Plot average attention for correctly predicted samples')
  parser.add_argument('--plot_space_wrongly_pred', type=int,default=1, help='Plot average attention for wrongly predicted samples')
  parser.add_argument('--plot_space_per_subject', type=int,default=1, help='Plot average attention per subject')
  parser.add_argument('--plot_space_per_class', type=int,default=1, help='Plot average attention per class')
  parser.add_argument('--plot_image_attention_per_sample', type=int,default=1, help='Plot attention map over space per single sample')
  parser.add_argument('--plot_image_attention_pain_level_and_subject', type=int,default=1, help='Plot attention map over space grouping per pain level and subject')
  parser.add_argument('--create_attention_video', type=int,default=50, help='Visualize attention in video per sample')
  parser.add_argument('--merge_videos', type=int,default=1, help='Merge all videos in the output folder into a single video.')
  parser.add_argument('--video_csv_path', type=str,default=None, help='Path to csv to create attention videos for samples in the csv only.')
  parser.add_argument('--fps', type=int,default=4, help='FPS for the video')
  dict_args = vars(parser.parse_args())
  
  # if dict_args['merge_videos'] and dict_args['create_attention_video'] == 0:
  #   raise ValueError("If merge_videos is set, create_attention_video must be > 0")
  data, run_id = get_data_from_pkl(dict_args['xattn_path'])
  if 'config_logging' not in data:
    data['config_logging'] = {}
  log_path_folder = os.path.join(*Path(dict_args['xattn_path']).parts[:-1], 'log_attention' if not data['config_logging'].get('split_chunks', False) else 'log_attention_split_chunks')
  if data['config_logging'].get('split_chunks', False):
    path_images_output = os.path.join(log_path_folder, 'images_split_chunks')
    path_video_output = os.path.join(log_path_folder, 'video_split_chunks')
  else:
    path_images_output = os.path.join(log_path_folder, 'images')
    path_video_output = os.path.join(log_path_folder, 'video')
    
  os.makedirs(log_path_folder, exist_ok=True)
  os.makedirs(path_images_output, exist_ok=True)
  os.makedirs(path_video_output, exist_ok=True)
  
  # Get data from pkl
  
  # Get the df from csv
  csv_path = data['csv_path']
  df = pd.read_csv(csv_path,sep='\t')
  list_sample_id = df['sample_id'].to_list()
  print(f"Processing all {len(list_sample_id)} samples from csv {csv_path}")
  
  
  ## SPLIT CHUNKS PART ##
  if 'config_logging' in data and data['config_logging'].get('split_chunks', False):
    reduced_list_sample_id = np.random.choice(list_sample_id, size=min(50, len(list_sample_id)), replace=False).tolist()
    all_frames = []
    for sample_id in tqdm.tqdm(reduced_list_sample_id, desc="Creating pain prediction videos for random samples"):
      gt,pred = get_gt_pred(sample_id, df, data)
      subject_name = df[df['sample_id'] == sample_id]['subject_name'].values[0]
      sample_name = df[df['sample_id'] == sample_id]['sample_name'].values[0]
      wargs = {'sample_name': sample_name, 'gt': gt, 'csv_name': Path(csv_path).stem}
      all_frames.extend(create_video_with_plot_and_predictions(video_path=os.path.join(data['config_model']['model_advanced_params']['path_dataset'],subject_name, f"{sample_name}.mp4"),
                                  pain_predictions=pred,
                                  output_path=os.path.join(path_video_output, f"{wargs['csv_name']}_{sample_id}_PA_{gt}.mp4"),
                                  smoothing_sigma=1,
                                  fig_size=(6, 2.5),
                                  dpi=200, 
                                  **wargs))
    tools.generate_video_from_list_frame(list_frame=all_frames,
                                         fps=dict_args['fps'],
                                         progress_bar=True,
                                         already_bgr=True,
                                        path_video_output=os.path.join(path_video_output, f"{Path(csv_path).stem}_all_samples_predictions.mp4"))
     
    all_classes = data['results']['test_unique_y']
    samples = np.array(list(data['results']['history_test_sample_predictions'].keys()))
    predictions = [data['results']['history_test_sample_predictions'][s][0] for s in samples]
    leghts = [len(p) for p in predictions]
    if len(set(leghts)) != 1:
      unique_len, count = np.unique(leghts, return_counts=True)
      # keep the most common length
      most_common_len = unique_len[np.argmax(count)]
      print(f"Warning: Not all predictions have the same length. Found lengths: {unique_len} with counts {count}. Keeping only the most common length: {most_common_len}")
      mask = np.array(leghts) == most_common_len
      samples = samples[mask]
      predictions = np.array([predictions[i] for i in range(len(predictions)) if mask[i]])
    else:
      predictions = np.array(predictions)
      
    for class_id in all_classes:
      class_samples = df[df['class_id'] == class_id.item()]['sample_id'].to_list()
      mask = np.isin(samples, class_samples)
      if not mask.any():
        raise ValueError(f"No samples found for class {class_id} in csv {csv_path}")
      class_preds = predictions[mask]
      class_samples = samples[mask]
      class_preds = np.mean(class_preds, axis=0)  # mean over epochs
      plot_histo_split_video_predictions(class_preds,
                                        title=f"Histogram of pain predictions for class {class_id}",
                                        path_images_output=os.path.join(path_images_output,f'histogram_predictions_class_{class_id}.png'))

  # Plot attention importance over space for wrongly and correctly predicted samples
  if not data['config_logging'].get('split_chunks', False):
    list_sample_id_wrong_pred = []
    list_sample_id_correct_pred = []
    folder_path = os.path.join
    for sample_id in list_sample_id:
      gt, pred = get_gt_pred(sample_id, df, data)
      if gt == int(pred + 0.5):  # rounding to nearest int
        list_sample_id_correct_pred.append(sample_id)
      else:
        list_sample_id_wrong_pred.append(sample_id)
    correctly_pred_ratio = len(list_sample_id_correct_pred) / len(list_sample_id)
    print(f"Found {len(list_sample_id_correct_pred)} correctly predicted samples and {len(list_sample_id_wrong_pred)} wrongly predicted samples. \nCorrect prediction ratio: {correctly_pred_ratio:.2f}")
    if dict_args['plot_space_correctly_pred']:
      plot_average_attention_over_samples(data,
                                          list_sample_id_correct_pred, 
                                          plot_folder=os.path.join(path_images_output, 'predicted_samples'),
                                          title=rf'Average attention over $\mathbf{{correctly\ predicted}}$ samples (ratio {correctly_pred_ratio:.2f})',
                                          png_name='predicted_samples_correctly'
                                          )
    if dict_args['plot_space_wrongly_pred']:
      plot_average_attention_over_samples(data,
                                          list_sample_id_wrong_pred,
                                          plot_folder=os.path.join(path_images_output, 'predicted_samples'),
                                          title=rf'Average attention over $\mathbf{{wrongly\ predicted}}$ samples (ratio {1-correctly_pred_ratio:.2f})',
                                          png_name='predicted_samples_wrongly'
                                          )

  # Plot attention importance over space per sample
  if dict_args['plot_image_attention_per_sample']:
    for sample_id in tqdm.tqdm(list_sample_id, desc="Plotting attention per sample"):
      xattn = get_xattn_from_sample(data, sample_id)
      xattn = get_resized_xattn(data, xattn, video_frames=None)  # shape (quadrants, nr_chunks, T, S, S)
      xattn = xattn.mean(axis=(0,1,2)) # shape (S, S)
      sample_name = df[df['sample_id'] == sample_id]['sample_name'].values[0]
      plot_space_importance(xattn,
                            plot_folder=os.path.join(path_images_output, 'attention_per_sample'),
                            title=rf'Attention map for sample $\mathbf{{{sample_id}}}$ - {sample_name}',
                            png_name=f'{sample_name}_{sample_id}_space_attention_map'
                            )
  
  # # Plot attention importance per subject with the same pain level
  if dict_args['plot_image_attention_pain_level_and_subject']:
    list_pain_levels = df['class_id'].unique().tolist()
    for pain_level in tqdm.tqdm(list_pain_levels, desc="Processing pain levels"):
      df_pain_level = df[df['class_id'] == pain_level]
      list_subject_id = df_pain_level['subject_id'].unique().tolist()
      for subject_id in list_subject_id:
        list_sample_id_subject_pain = df_pain_level[df_pain_level['subject_id'] == subject_id]['sample_id'].to_list()
        subject_name = df_pain_level[df_pain_level['subject_id'] == subject_id]['subject_name'].values[0]
        plot_average_attention_over_samples(data,
                                            list_sample_id_subject_pain,
                                            plot_folder=os.path.join(path_images_output, f'attention_per_subject_pain_level_{pain_level}'),
                                            title=rf'Average attention for $\mathbf{{subject\ {subject_id}}}$ with pain level {pain_level} over {len(list_sample_id_subject_pain)} samples',
                                            png_name=f'{subject_name}_id_{subject_id}_pain_level_{pain_level}',
                                            multiple_ids=data['config_logging'].get('split_chunks', False)
                                            )
  
  # Plot attention importance over space filtered by subjects
  if dict_args['plot_space_per_subject']:
    list_subject_id = df['subject_id'].unique().tolist()
    for subject_id in tqdm.tqdm(list_subject_id, desc="Processing subjects"):
      list_sample_id_subject = df[df['subject_id'] == subject_id]['sample_id'].to_list()
      plot_average_attention_over_samples(data,
                                          list_sample_id_subject,
                                          plot_folder=os.path.join(path_images_output, 'attention_per_subject'),
                                          title=rf'Average attention for $\mathbf{{subject\ {subject_id}}}$ over {len(list_sample_id_subject)} samples',
                                          png_name=f'subject_{subject_id}',
                                          multiple_ids=data['config_logging'].get('split_chunks', False)
                                          )

  # Plot attention importance over space filtered by classes
  if dict_args['plot_space_per_class']:
    list_class_id = df['class_id'].unique().tolist()
    for class_id in tqdm.tqdm(list_class_id, desc="Processing classes"):
      list_sample_id_class = df[df['class_id'] == class_id]['sample_id'].to_list()
      plot_average_attention_over_samples(data, 
                                          list_sample_id_class,
                                          plot_folder=os.path.join(path_images_output, 'attention_per_class'),
                                          title=rf'Average attention for $\mathbf{{class\ {class_id}}}$ over {len(list_sample_id_class)} samples',
                                          png_name=f'class_{class_id}'
                                          )
    
    
  # Visualize attention in video per sample, If > 1, randomly select n samples to visualize
  if dict_args['create_attention_video'] and not data['config_logging'].get('split_chunks', False):
    if dict_args['merge_videos']:
      list_frames = []
      timestamp_df_records = []
      count_frames = 0
      
    if dict_args['create_attention_video']:
      if dict_args['video_csv_path'] is None:
        n_samples = dict_args['create_attention_video']
        if n_samples > len(list_sample_id):
          n_samples = len(list_sample_id)
        list_sample_id = np.random.choice(list_sample_id_correct_pred, n_samples//2, replace=False).tolist() + \
                        np.random.choice(list_sample_id_wrong_pred, n_samples//2, replace=False).tolist()
        df_selected = df[df['sample_id'].isin(list_sample_id)]
        # save selected sample ids to csv
        df_selected.to_csv(os.path.join(path_video_output, f'selected_sample_ids_for_attention_videos_{n_samples}_samples.csv'), index=False, sep='\t')
      else:
        csv_path = dict_args['video_csv_path']
        df_csv = pd.read_csv(csv_path,sep='\t')
        list_sample_id = df_csv['sample_id'].to_list()
        # check if are different paths
        if os.path.abspath(csv_path) != os.path.abspath(os.path.join(path_video_output, os.path.basename(csv_path))):
          shutil.copy(csv_path, path_video_output)
          print(f"Creating attention videos for {len(list_sample_id)} samples from csv {csv_path}")
        
    # Iterate over samples and create video with attention overlaid
    for sample_id in tqdm.tqdm(list_sample_id, desc="Processing samples videos"):
      if sample_id not in df['sample_id'].values:
        raise ValueError(f"Sample id {sample_id} not found in csv {csv_path}")
      gt, pred = get_gt_pred(sample_id, df, data)
      video_frames,list_indices = get_list_frames_from_sample(data, sample_id, create_video=False)
      xattn = get_xattn_from_sample(data, sample_id)
      if xattn.shape[0] > 1:
        if dict_args['merge_videos']:
          raise ValueError("Merging videos not supported for multihead attention")
        multihead_video_creation(xattn, list_indices, video_frames, sample_id, gt, pred, run_id)
      else:
        # Resize attention to frame size
        xattn = get_resized_xattn(data, xattn, video_frames)  # shape (nr_frames, Hf, Wf)
        
        # Plot attention on frames
        vmin = xattn.min()
        vmax = xattn.max()
        if dict_args['merge_videos']:
          start_minute = count_frames / dict_args['fps'] // 60
          start_second = (count_frames / dict_args['fps']) % 60
          count_frames += len(video_frames)
          list_frames.append(create_frame_with_overlaid_attention(list_indices,
                                              video_frames,
                                              xattn,
                                              sample_id,
                                              gt,
                                              pred,
                                              run_id,
                                              head=0,
                                              FPS=dict_args['fps'],
                                              min_range_plot=vmin,
                                              max_range_plot=vmax,
                                              path_video_output=None))
          # Add FPS black frame between samples
          list_frames.append([np.zeros_like(list_frames[-1][0]) for _ in range(dict_args['fps'])])
          total_duration = len(list_frames[-2]) / dict_args['fps']
          timestamp_df_records.append({'sample_id': sample_id,
                                       'sample_name': df[df['sample_id'] == sample_id]['sample_name'].values[0],
                                        'gt': gt,
                                        'pred': pred,
                                        'start': f"{int(start_minute):02d}:{int(start_second):02d}",
                                        'duration_seconds': total_duration
                                       })
          
        else:
          create_frame_with_overlaid_attention(list_indices,
                                              video_frames,
                                              xattn,
                                              sample_id,
                                              gt,
                                              pred,
                                              run_id,
                                              min_range_plot=vmin,
                                              max_range_plot=vmax,
                                              FPS=dict_args['fps'],
                                              head=0,
                                              path_video_output=path_video_output)
    if dict_args['merge_videos']:
      list_frames = [frame for video in list_frames for frame in video]  # flatten list
      timestamp_df_records = pd.DataFrame(timestamp_df_records)
      timestamp_df_records.to_csv(os.path.join(path_video_output, 'merged_video_timestamps.csv'), index=False, sep='\t')
      tools.generate_video_from_list_frame(list_frame=list_frames,
                                          fps=dict_args['fps'],
                                          progress_bar=True,
                                          path_video_output=os.path.join(path_video_output, f"test_attention_{len(list_sample_id)}_samples_merged.mp4"))

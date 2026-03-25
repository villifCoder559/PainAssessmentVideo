import sys
import custom.dataset as ds
import cv2
import torch
import os
import numpy as np
import tqdm

import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--video_root', type=str, required=True)
argument_parser.add_argument('--video_extension', type=str, default='.mp4')
argument_parser.add_argument('--shift', action='store_true')
argument_parser.add_argument('--color_jitter', action='store_true')
argument_parser.add_argument('--h_flip', action='store_true')
argument_parser.add_argument('--rotation', action='store_true')
argument_parser.add_argument('--zoom', action='store_true')
argument_parser.add_argument('--framepermute', action='store_true')
argument_parser.add_argument('--framepermute_seed', type=int, default=22)
argument_parser.add_argument('--gaussian_smooth', action='store_true')
argument_parser.add_argument('--gaussian_sigma_min', type=float, default=0.5)
argument_parser.add_argument('--gaussian_sigma_max', type=float, default=2.0)
argument_parser.add_argument('--gaussian_kernel_size', type=int, default=3)
args = argument_parser.parse_args()

video_root = args.video_root
video_output_root = args.video_root 
if not os.path.exists(video_root):
  raise ValueError(f"Video root does not exist: {video_root}")

augmentation_dict = {
  'shift': args.shift,
  'color_jitter': args.color_jitter,
  'h_flip': args.h_flip,
  'rotation': args.rotation,
  'zoom': args.zoom,
  'framepermute': args.framepermute,
  'gaussian': args.gaussian_smooth
}

if np.all([not v for v in augmentation_dict.values()]):
  raise ValueError("At least one augmentation must be enabled")

for k,v in augmentation_dict.items():
  if v:
    video_output_root += f"_{k}"

if video_root == video_output_root:
  raise ValueError("Video root and output video root must be different")

print(f'Video root: {video_root}')
print(f'Output video root: {video_output_root}')
# count = 0
os.makedirs(video_output_root, exist_ok=True)
# Get all video paths
list_video_path = []
for root, dirs, files in os.walk(video_root):
  for file in files:
    if file.endswith(args.video_extension) and '$' not in file:
      list_video_path.append(os.path.join(root,file))

# Process each video
params_all = {}
for video_path in tqdm.tqdm(list_video_path, desc="Processing videos..."):
  video = cv2.VideoCapture(video_path)
  frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = video.get(cv2.CAP_PROP_FPS)
  original_frames = []
  # Read all frames
  for i in range(frame_count):
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frames.append(frame)
  video.release()
  
  # Preprocess frames with spatial shift augmentation
  original_frames = np.array(original_frames)  # Shape: (num_frames, height, width, channels)
  prep_frames,params = ds.customDataset.preprocess_images(tensors=torch.tensor(original_frames).permute(0,3,1,2),
                                    to_visualize=True,
                                    get_params=True,
                                    color_jitter=augmentation_dict['color_jitter'],
                                    h_flip=augmentation_dict['h_flip'],
                                    rotation=augmentation_dict['rotation'],
                                    zoom=augmentation_dict['zoom'],
                                    spatial_shift=augmentation_dict['shift'],
                                    gaussian_smooth=augmentation_dict['gaussian'],
                                    gaussian_sigma_min=args.gaussian_sigma_min,
                                    gaussian_sigma_max=args.gaussian_sigma_max,
                                    gaussian_kernel_size=args.gaussian_kernel_size,
                                    framepermute=augmentation_dict['framepermute'],
                                    framepermute_seed=args.framepermute_seed)
  video_id = os.path.splitext(os.path.basename(video_path))[0]
  params_all[video_id] = params
  
  # Save the preprocessed frames as a new video
  output_video = os.path.join(video_output_root, os.path.relpath(video_path, video_root))
  os.makedirs(os.path.dirname(output_video), exist_ok=True)
  out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'avc1'), fps, (prep_frames.shape[3], prep_frames.shape[2]))
  for i in range(prep_frames.shape[0]):
    frame = prep_frames[i].permute(1,2,0).numpy().astype('uint8')
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
    
  out.release()
  # count += 1
  # if count % 10 == 0:
  #   break
  # break # For testing only one video

# Save all_params to a file
import json
with open(os.path.join(video_output_root, "augmentation_params.json"), 'w') as f:
  json.dump(params_all, f, indent=2)
  print(f"Saved augmentation parameters to {f.name}")

# save augmentation info
with open(os.path.join(video_output_root, "augmentation_info.txt"), 'w') as f:
  f.write("Augmentation settings used:\n")
  for k,v in augmentation_dict.items():
    f.write(f"{k}: {v}\n")
  print(f"Saved augmentation info to {f.name}")
  
import sys
command = {'launch_command':" ".join(sys.argv),
           'video_root':video_root,
           'video_output_root':video_output_root,
           'augmentation_dict':augmentation_dict,
           'framepermute_seed': args.framepermute_seed,
           'gaussian_sigma_min': args.gaussian_sigma_min,
           'gaussian_sigma_max': args.gaussian_sigma_max,
           'gaussian_kernel_size': args.gaussian_kernel_size}
with open(os.path.join(video_output_root, "augmentation_command.txt"), 'w') as f:
  for key, value in command.items():
    f.write(f"{key}: {value}\n")
  print(f"Saved augmentation command to {f.name}")


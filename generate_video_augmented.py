import custom.dataset as ds
import cv2
import torch
import os
import numpy as np
import tqdm

video_root = "/equilibrium/fvilli/PainAssessmentVideo/partA/video/video_frontalized_interpolated_resolution_original"
video_output_root = "/equilibrium/fvilli/PainAssessmentVideo/partA/video/video_frontalized_interpolated_resolution_original"
augmentation_dict = {
  'shift': False,
  'color_jitter': False,
  'h_flip': False,
  'rotation': False,
  }

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
    if file.endswith(".mp4") and '$' not in file:
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
                                    spatial_shift=augmentation_dict['shift'])
  video_id = os.path.splitext(os.path.basename(video_path))[0]
  params_all[video_id] = params
  
  # Save the preprocessed frames as a new video
  output_video = os.path.join(video_output_root, os.path.relpath(video_path, video_root))
  os.makedirs(os.path.dirname(output_video), exist_ok=True)
  out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (prep_frames.shape[3], prep_frames.shape[2]))
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
  



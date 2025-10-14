import pickle
import os 
import cv2
import numpy as np
import tqdm
import argparse

parser = argparse.ArgumentParser(description="Process video files")
parser.add_argument("--video_folder", type=str, required=True, help="Path to the folder containing video files")
parser.add_argument("--grid_size", type=int, nargs=2, default=(14, 14), help="Grid size for video analysis")
args = parser.parse_args()
video_path_folder = args.video_folder
grid_size = tuple(args.grid_size)
video_path_list = []

# Collect all .mp4 video files in the specified folder and its subfolders
for root, dirs, files in os.walk(video_path_folder):
  for file in files:
    if file.endswith(".mp4") and '$' not in file:
      video_path_list.append(os.path.join(root, file))

# Initialize matrices to store results
tot_frames = 0
full_black_count = np.zeros((len(video_path_list), grid_size[0], grid_size[1]), dtype=int)
mean_value_matrix = np.zeros((len(video_path_list), grid_size[0], grid_size[1]), dtype=float)
threshold = 240  # Threshold to consider a cell as "full black"
threshold_count = np.zeros((len(video_path_list), grid_size[0], grid_size[1]), dtype=int)

# Process each video file
for video_idx, video_path in enumerate(tqdm.tqdm(video_path_list)):
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error opening video file:", video_path)
    continue
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  grid_h = frame_height // grid_size[0]
  grid_w = frame_width // grid_size[1]
  tot_frames += frame_count
  
  # Process each frame in the video
  for frame_idx in range(frame_count):
    ret, frame = cap.read()
    if not ret:
      break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Analyze each cell in the grid, updating counts and mean values
    for i in range(grid_size[0]):
      for j in range(grid_size[1]):
        cell = gray_frame[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
        mean_value_matrix[video_idx, i, j] += np.mean(cell)
        if np.all(cell == 0):
          full_black_count[video_idx, i, j] += 1
        if np.mean(cell) < threshold:
          threshold_count[video_idx, i, j] += 1
  cap.release()

# Save results to a pickle file
pkl_file = {
  "video_path_list": video_path_list,
  'video_path_folder': video_path_folder,
  "full_black_count": full_black_count,
  'mean_value_matrix': mean_value_matrix,
  "threshold_count": threshold_count,
  "threshold": threshold,
  "grid_size": grid_size,
  "total_frames": tot_frames
}
with open(os.path.join(video_path_folder, f"video_analysis_grid_{grid_size[0]}_{grid_size[1]}.pkl"), "wb") as f:
  pickle.dump(pkl_file, f)
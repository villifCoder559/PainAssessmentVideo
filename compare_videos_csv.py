import cv2
import numpy as np
import pandas as pd
import custom.tools as tools
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import time

def get_video_properties(video_path):
  """Get the total frame count and FPS of the video."""
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError(f"Cannot open video: {video_path}")
  
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  cap.release()
  
  return frame_count, fps

def resize_to_common(frame1, frame2):
  """Resize both frames to a common size based on the minimum width and height."""
  h1, w1 = frame1.shape[:2]
  h2, w2 = frame2.shape[:2]
  
  target_width = min(w1, w2)
  target_height = min(h1, h2)
  
  # Only resize if needed to avoid unnecessary computation
  if h1 != target_height or w1 != target_width:
    frame1_resized = cv2.resize(frame1, (target_width, target_height))
  else:
    frame1_resized = frame1
    
  if h2 != target_height or w2 != target_width:
    frame2_resized = cv2.resize(frame2, (target_width, target_height))
  else:
    frame2_resized = frame2
  
  return frame1_resized, frame2_resized

def mse(imageA, imageB):
  """
  Compute the Mean Squared Error between two images.
  A lower value indicates a closer match.
  """
  # Using numpy's built-in MSE calculation for speed
  return np.mean(np.square(imageA.astype(np.float32) - imageB.astype(np.float32)))

def compare_videos(video_pair,convert_to_gray=False):
  """
  Compare every frame of two videos.
  
  Args:
      video_pair: Tuple containing (video1_path, video2_path)
  """
  video1_path, video2_path = video_pair
  
  try:
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
      raise IOError(f"Cannot open one or both videos: {video1_path}, {video2_path}")
    
    frame_count1, _ = get_video_properties(video1_path)
    frame_count2, _ = get_video_properties(video2_path)
    total_frames = min(frame_count1, frame_count2)
    
    # Pre-allocate arrays for better performance
    mse_values = np.zeros(total_frames)
    ssim_values = np.zeros(total_frames)
    
    # Use a smaller batch size to process frames
    batch_size = 10
    for frame_idx in range(0, total_frames, batch_size):
      end_idx = min(frame_idx + batch_size, total_frames)
      
      for i in range(frame_idx, end_idx):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
          total_frames = i  # Update total frames processed
          break
        
        frame1_resized, frame2_resized = resize_to_common(frame1, frame2)
        
        # Convert to grayscale for SSIM to save computation if needed
        if convert_to_gray:
          frame1_gray = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY) if frame1_resized.ndim > 2 else frame1_resized
          frame2_gray = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY) if frame2_resized.ndim > 2 else frame2_resized
        else:
          frame1_gray = frame1_resized
          frame2_gray = frame2_resized
        mse_values[i] = mse(frame1_resized, frame2_resized)
        ssim_values[i], _ = compare_ssim(frame1_gray, 
                                        frame2_gray,
                                        win_size=3,
                                        full=True, 
                                        multichannel=False)  # Updated parameter for newer versions
    
    cap1.release()
    cap2.release()
    
    # Resize arrays to actual frames processed
    mse_values = mse_values[:total_frames]
    ssim_values = ssim_values[:total_frames]
    
    avg_mse = np.mean(mse_values)
    avg_ssim = np.mean(ssim_values)
    
    return {
      'video_1': os.path.basename(video1_path),
      'video_2': os.path.basename(video2_path),
      'avg_mse': avg_mse,
      'avg_ssim': avg_ssim
    }
  
  except Exception as e:
    print(f"Error processing {video1_path} and {video2_path}: {str(e)}")
    return {
      'video_1': os.path.basename(video1_path),
      'video_2': os.path.basename(video2_path),
      'avg_mse': np.nan,
      'avg_ssim': np.nan
    }

def main():
  folder_video_path_1 = 'partA/video/video_frontalized_new'
  folder_video_path_2 = 'partA/video/video_frontalized'
  csv_path = 'partA/starting_point/samples_exc_no_detection.csv'
  # Get video paths
  list_video_path_1 = tools.get_list_video_path_from_csv(csv_path=csv_path, 
                                                       video_folder_path=folder_video_path_1)
  list_video_path_2 = tools.get_list_video_path_from_csv(csv_path=csv_path,
                                                       video_folder_path=folder_video_path_2)
  
  
  
  
  # Determine optimal number of workers based on CPU cores
  num_workers = min(os.cpu_count() or 4, 8)  # Limit to 8 workers to avoid excessive memory usage
  results = []
  
  video_pairs = list(zip(list_video_path_1, list_video_path_2))
  total_pairs = len(video_pairs)
  # Process videos in parallel using ThreadPoolExecutor
  with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Process video pairs with progress bar
    for result in tqdm(executor.map(compare_videos, video_pairs), total=total_pairs, 
                     desc="Comparing videos", unit="pair"):
      results.append(result)
  
  # Create DataFrame and save results
  df = pd.DataFrame(results)
  output_file = f'comparison_results_{os.path.basename(folder_video_path_2)}.csv'
  df.to_csv(output_file, index=False)
  print(f'All results saved in {output_file}')
  return output_file

def merge_comparison_results_with_pitch(comparison_path, pitch_path):
  df_comparison = pd.read_csv(comparison_path)
  df_pitch_per_video = pd.read_csv(pitch_path)

  df_comparison['sample'] = df_comparison['video_1'].apply(lambda x: x.split('.')[0])  
  merged_df = df_comparison.merge(right=df_pitch_per_video, how='left', on='sample').drop(columns=['video_2']).drop(columns=['video_1'])  
  csv_path = f'comparison_results_with_pitch{time.time()}.csv'
  merged_df.to_csv(csv_path, index=False)
  print(f'All merged csv results saved in {csv_path}')
  
if __name__ == "__main__":
  comparison_path = main()
  pitch_path = 'partA/video/roll_pitch_yaw_per_subject/roll_pitch_yaw_per_subject_all.csv'
  merge_comparison_results_with_pitch(comparison_path, pitch_path)
  
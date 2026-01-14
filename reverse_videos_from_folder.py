import cv2
import os
from pathlib import Path
import tqdm
import argparse
import subprocess

def get_video_fps(video_path):
  cmd = [
      "ffprobe", "-v", "error",
      "-select_streams", "v:0",
      "-show_entries", "stream=r_frame_rate",
      "-of", "default=noprint_wrappers=1:nokey=1",
      str(video_path)
  ]
  fps_str = subprocess.check_output(cmd).decode().strip()
  # Handle both "30000/1001" and "30" formats
  if '/' in fps_str:
      num, den = fps_str.split("/")
      return float(num) / float(den)
  return float(fps_str)

def slow_motion_video(video_path, output_path, slow_factor=2):
  """
  High-quality slow motion using FFmpeg motion interpolation.
  It generates intermediate frames and stretches time.
  """
  fps = get_video_fps(video_path)
  
  # 1. Calculate target interpolation FPS (e.g., 30fps * 4 = 120fps)
  interpolated_fps = float(fps) * slow_factor
  
  # 2. Build Filter Graph
  # minterpolate: Creates the extra frames (smoothness)
  # setpts: Stretches the time (actual slow motion)
  filter_graph = f"minterpolate=fps={interpolated_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1,setpts={slow_factor}*PTS"

  ffmpeg_cmd = [
      "ffmpeg", "-y",
      "-i", str(video_path),
      "-vf", filter_graph,
      "-r", str(fps),     # Force output file to report the ORIGINAL FPS
      "-c:v", "libx264",
      "-crf", "18",       # High quality
      "-pix_fmt", "yuv420p",
      "-an",              # Remove audio (slow audio usually sounds bad)
      str(output_path)
  ]

  # Using subprocess.run is safer than check_output for commands that don't return text
  subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def timelapse_video(video_path, output_path=None, speed_factor=2):
  cap = cv2.VideoCapture(str(video_path))
  
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  
  if output_path is None:
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_timelapse{ext}"
  
  out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
  
  frame_idx = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    if frame_idx % speed_factor == 0:
      out.write(frame)
    frame_idx += 1
    
  cap.release()
  out.release()

def reverse_video(video_path, output_path=None):
  cap = cv2.VideoCapture(str(video_path))
  
  fps = cap.get(cv2.CAP_PROP_FPS)
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  
  if output_path is None:
    base, ext = os.path.splitext(video_path)
    output_path = f"{base}_reversed{ext}"
  
  out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
  
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  
  for frame in reversed(frames):
    out.write(frame)
  
  cap.release()
  out.release()

if __name__ == "__main__":
  list_video_path = []
  parser = argparse.ArgumentParser(description="Process videos in a folder")
  parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing videos")
  parser.add_argument("--video_operation", type=str, default="reverse", help="Options: reverse, slowmotion, timelapse")
  parser.add_argument("--factor", type=int, default=2, help="Speed factor for slowmotion or timelapse")
  
  args = parser.parse_args()
  
  root_video_path = Path(args.input_folder)
  
  # Recursively find video files
  for root, dirs, files in os.walk(root_video_path):
    for file in files:
      if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")) and '$' not in file:
        list_video_path.append(Path(root) / file)

  # Create base output directory
  output_root = Path(f"{str(root_video_path)}_{args.video_operation}")
  output_root.mkdir(parents=True, exist_ok=True)

  for video_path in tqdm.tqdm(list_video_path, desc=f"Performing {args.video_operation}"):
    try:
      # Determine output structure to mirror input structure
      rel_path = video_path.relative_to(root_video_path)
      out_path = output_root / rel_path
      
      # Create sub-directories if they exist in source
      out_path.parent.mkdir(parents=True, exist_ok=True)

      if args.video_operation == 'reverse':
        reverse_video(video_path, out_path)
      elif args.video_operation == 'slowmotion':
        slow_motion_video(video_path, out_path, slow_factor=args.factor)
      elif args.video_operation == 'timelapse':
        timelapse_video(video_path, out_path, speed_factor=args.factor)
      else:
        raise ValueError("Unsupported operation")
          
    except Exception as e:
      print(f"\nError processing {video_path.name}: {e}")
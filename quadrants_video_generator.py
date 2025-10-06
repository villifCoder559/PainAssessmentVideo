import cv2
import os
import argparse
import tqdm

def split_and_save_video(input_path, part_to_save, output_resolution):
  assert part_to_save in ['upper_left', 'upper_right', 'bottom_left', 'bottom_right'], \
    "part_to_save must be one of: 'upper_left', 'upper_right', 'bottom_left', 'bottom_right'"

  cap = cv2.VideoCapture(input_path)
  if not cap.isOpened():
    print(f"Error opening video file {input_path}")
    return

  # Get video properties
  fps = cap.get(cv2.CAP_PROP_FPS)
  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  codec = int(cap.get(cv2.CAP_PROP_FOURCC))

  # Determine split dimensions
  half_width = width // 2
  half_height = height // 2

  # Get output filename
  base_name, ext = os.path.splitext(os.path.basename(input_path))
  base_folder = os.path.dirname(input_path)
  output_path = os.path.join(base_folder, f"{base_name}${part_to_save}.mp4")

  # Video writer
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(output_path, fourcc, fps, (output_resolution[0], output_resolution[1]))

  while True:
    ret, frame = cap.read()
    if not ret:
      break  
    # Crop based on part_to_save
    if part_to_save == 'upper_left':
      cropped = frame[0:half_height, 0:half_width]
    elif part_to_save == 'upper_right':
      cropped = frame[0:half_height, half_width:width]
    elif part_to_save == 'bottom_left':
      cropped = frame[half_height:height, 0:half_width]
    elif part_to_save == 'bottom_right':
      cropped = frame[half_height:height, half_width:width]
    
    # Resize to output resolution
    cropped = cv2.resize(cropped, (output_resolution[0], output_resolution[1]))
    out.write(cropped)

  cap.release()
  out.release()
  # print(f"Saved cropped video to {output_path}")

def get_list_videos(video_folder):
  """Get a list of video files in the specified folder."""
  list_videos = []
  for root, dirs, files in os.walk(video_folder):
    for file in files:
      if file.endswith('.mp4') and '$' not in file:
        list_videos.append(os.path.join(root, file))
  return list_videos


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Split video into quadrants and save specified part.")
  parser.add_argument("--video_folder", type=str, help="Path to the input video folder")
  parser.add_argument("--part", type=str, choices=['upper_left', 'upper_right', 'bottom_left', 'bottom_right'], help="Part of the video to save")
  parser.add_argument("--output_resolution", nargs='*', type=int, default=[224,224], help="Output resolution in format [WIDTH,HEIGHT]. Default is [224,224].")
  args = vars(parser.parse_args())
  
  list_videos = get_list_videos(args['video_folder'])
  for video_path in tqdm.tqdm(list_videos, desc="Processing videos"):
    split_and_save_video(video_path, args['part'], args['output_resolution'])
  print(f'{len(list_videos)} videos saved in {args["video_folder"]} with part {args["part"]} and resolution {args["output_resolution"]}')

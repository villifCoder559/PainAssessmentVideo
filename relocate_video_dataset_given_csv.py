import tqdm
import os
import shutil
import pandas as pd
import argparse


def relocate_video_given_csv(csv_path, video_folder, output_folder):
  df = pd.read_csv(csv_path, sep='\t')
  for _, row in tqdm.tqdm(df.iterrows(),total=len(df),desc="Relocating videos"):
    video_name = row['sample_name'] + '.mp4'
    source_path = os.path.join(video_folder, video_name)
    if os.path.exists(source_path):
      dest_path = os.path.join(output_folder,row['subject_name'], video_name)
      os.makedirs(os.path.dirname(dest_path), exist_ok=True)
      shutil.move(source_path, dest_path)
    else:
      print(f"Video {video_name} not found in {video_folder}")

parser = argparse.ArgumentParser(description="Relocate videos based on CSV file")
parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
parser.add_argument('--video_folder', type=str, required=True, help='Path to the video folder')
parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder')
args = parser.parse_args()
relocate_video_given_csv(**vars(args))
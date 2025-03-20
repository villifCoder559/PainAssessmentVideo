from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
import tqdm
import argparse
from custom.helper import GLOBAL_PATH

def main(csv_path,root_folder_path,log_folder_path=None):
  list_files = ["features.pt","list_frames.pt","list_labels.pt","list_path.npy","list_sample_id.pt","list_subject_id.pt"]
  df = pd.read_csv(csv_path,sep='\t')
  list_subject = np.unique(df['subject_name'],return_counts=False)
  list_missing_files = []
  for subject in tqdm.tqdm(list_subject):
    list_samples = df[df['subject_name'] == subject]['sample_name']
    for sample in list_samples:
      sample_path = Path(root_folder_path / subject / sample)
      if not sample_path.exists():
        list_missing_files.append(sample_path)
        continue
      intersection = np.intersect1d(list_files, os.listdir(sample_path))
      if not intersection.size == len(list_files):
        list_missing_files.append(sample_path)
        
  if len(list_missing_files)>0:
    print(f'Number of missing files: {len(list_missing_files)}')      
    print(f'List missing files in {root_folder_path}')
    for missing_file in list_missing_files:
      print(missing_file)
    if log_folder_path:
      os.makedirs(log_folder_path,exist_ok=True)
      log_folder_path = Path(log_folder_path) / f'log_video_{csv_path.stem}.txt'
      with open(log_folder_path,'w') as f:
        for missing_file in list_missing_files:
          f.write(f'{missing_file}\n')
  else:
    print(f'All files are present in {root_folder_path} according to {csv_path}')
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.description = 'Check if the whole extracted features are in the csv'
  parser.add_argument('--csv_path', type=str, required=True,help='Path to csv file')
  parser.add_argument('--f_video_path', type=str, required=True,help='Path to features extracted')
  parser.add_argument('--gp',action='store_true', help='Add /equilibrium/fvilli/PainAssessmentVideo to all paths')
  parser.add_argument('--log_folder_path', type=str, default=None, help='Path to log folder')
  args = parser.parse_args()
  root_folder_path = Path(args.f_video_path)
  csv_path = Path(args.csv_path)
  log_folder_path = args.log_folder_path
  if args.gp:
    root_folder_path = GLOBAL_PATH.get_global_path(root_folder_path)
    csv_path = GLOBAL_PATH.get_global_path(csv_path)
    log_folder_path = GLOBAL_PATH.get_global_path(log_folder_path)
  print(f'csv path: {csv_path}\n')
  print(f'video path: {root_folder_path}')
  main(csv_path,root_folder_path,log_folder_path)

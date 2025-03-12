from pathlib import Path
import numpy as np
import pandas as pd
import torch
import os
import tqdm
if __name__ == '__main__':
  root_folder_path = Path('/media/villi/TOSHIBA EXT/samples_16_whole')
  csv_path = Path('partA/starting_point/samples_exc_no_detection.csv')
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
  else:
    print(f'All files are present in {root_folder_path} according to {csv_path}')
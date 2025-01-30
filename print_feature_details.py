import os
import numpy as np
import custom.tools as tools
import torch
import argparse

# /equilibrium/fvilli/partA/video/features/samples_16_cropped_aligned
def print_feature_details(folder_feat_path):
  if not os.path.exists(folder_feat_path):
    print(f'Folder {folder_feat_path} does not exist')
    return
  dict_data = tools.load_dict_data(saving_folder_path=folder_feat_path)
  for k,v in dict_data.items():
    print(f'key:{k}')
    print(f'  v.shape: {v.shape}')
    print(f'  last element {v[-1]}')
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Print feature details')
  parser.add_argument('--fold_p',required=True, type=str, help='Path to the folder containing the features')
  args = parser.parse_args()
  print_feature_details(args.fold_p)
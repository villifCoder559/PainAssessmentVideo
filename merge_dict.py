import custom.tools as tools
import os
import numpy as np
import torch
import argparse


def merge_dict(list_folder_path,save_result_path=None):
  merged_dict = {}
  for dict_path in list_folder_path:
    dict_data = tools.load_dict_data(saving_folder_path=dict_path)
    print(f'{dict_path}:')
    for k,v in dict_data.items():
      print(f'  {k}: {v.shape}')
      if k not in merged_dict:
        merged_dict.update({k:v})
      else:
        if isinstance(v,np.ndarray):
          merged_dict[k] = np.concatenate([merged_dict[k],v],axis=0)
        elif isinstance(v,torch.Tensor):
          merged_dict[k] = torch.concat([merged_dict[k],v], dim=0)
        else:
          print(f'Error: {k} is not a numpy array or torch tensor')
  print('-'*200)
  print('Shape new elements:')
  for k,v in merged_dict.items():
    print(f'  {k}: {v.shape}')
  print('-'*200)
  tools.save_dict_data(dict_data=merged_dict,
                       saving_folder_path=os.path.join(save_result_path,'merged_dict'))
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Merge dict given a list of folder paths using --p. \nNew dict will be saved in the first folder path')
  parser.add_argument('--p',type=str, nargs='+', required=True, help='List of folder paths to merge')
  parser.add_argument('--s',type=str, help='Path to save the merged dict')
  # python3 merge_dict.py --p /equilibrium/fvilli/PainAssessmentVideo/partA/video/features/sample_16_temporal_mean --s /equilibrium/fvilli/PainAssessmentVideo/video/features/samples_16_temporal_mean
  pars_arg = parser.parse_args()
  list_root_path = pars_arg.p
  list_dict_path = []
  for el in list_root_path:
    print(f'root folder path: {el}\n')
    dir_list = os.listdir(el)
    for dir in dir_list:
      if os.path.isdir(os.path.join(el,dir)):# contains dict
        list_dict_path.append(os.path.join(el,dir))
  for el in list_dict_path:
    print(f'List dict {el}\n')
  merge_dict(list_folder_path=list_dict_path,
             save_result_path=pars_arg.s)
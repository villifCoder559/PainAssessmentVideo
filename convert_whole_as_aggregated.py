import os
import custom.tools as tools
import torch
import numpy as np
import tqdm
import custom.helper as helper
import safetensors.torch

root_dataset_folder = "/equilibrium/fvilli/PainAssessmentVideo/partA/video/features/VideoMaeV2_S/whole_features_BIOVID_S_stride_32_inside_2_shifted_last143"
print(f"Root dataset folder: {root_dataset_folder}")
list_path_safetensor = []
for root, dirs, files in os.walk(root_dataset_folder):
  for file in files:
    if file.endswith(".safetensors"):
      list_path_safetensor.append(os.path.join(root,file))
print(f"Number of safetensors found: {len(list_path_safetensor)}")

new_path = "/equilibrium/fvilli/PainAssessmentVideo/partA/video/features/VideoMaeV2_S/whole_features_BIOVID_S_stride_32_inside_2_shifted_last143_aggregated"
new_dict_data = {}
count = 0
for path_safetensor in tqdm.tqdm(list_path_safetensor,desc="Converting to float16..."):
  # Load data
  data = tools.load_dict_data(path_safetensor)
  original_sample_id = data['list_sample_id'][0]
  if original_sample_id <= 0 or original_sample_id > 8700:
    raise ValueError(f"Original sample id {original_sample_id} is out of range [1, 8700]")
  
  # Convert feats to float16
  data['features'] = data['features'].half()
  for k,v in data.items():
    if k not in new_dict_data:
      new_dict_data[k] = []
    new_dict_data[k].append(v)
  count += 1

print(f"Number of aggregated safetensors: {count}")
safetensors.torch.save_file({f'{k}':torch.concat(v, dim=0) for k,v in new_dict_data.items()}, new_path+'.safetensors')
print(f"Saved aggregated safetensors to {new_path+'.safetensors'}")
  # # Save partial to avoid system kill due to large memory usage
  # dict_size_GB = sum(
  #     vv.element_size() * vv.nelement()
  #     for values in new_dict_data.values()
  #     for vv in values
  # ) / (1024**3)
  
  # if dict_size_GB >= 20:  # threshold
  #   for k,v in tqdm.tqdm(new_dict_data.items(), desc="Concatenating tensors..."):
  #     cat_tensor = torch.concat(v, dim=0)
  #     new_dict_data[k] = cat_tensor
  #   partial_path = os.path.join(new_path, f"aggregated_part_{count}.safetensors")
  #   tools.save_dict_data(new_dict_data, partial_path, True)
  #   new_dict_data = {}
    
# Save remaining data
# if len(new_dict_data) > 0:
#   for k,v in tqdm.tqdm(new_dict_data.items(), desc="Concatenating tensors..."):
#     partial_path = os.path.join(new_path, f"aggregated_{k}.safetensors")
#     safetensors.torch.save_file({f'{k}':torch.concat(v, dim=0)}, partial_path)
#     # tools.save_dict_data(torch.concat(v, dim=0), partial_path, True)
#     del new_dict_data[k]
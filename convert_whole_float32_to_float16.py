import os
import custom.tools as tools
import torch
import tqdm
import custom.helper as helper

root_dataset_folder = "/equilibrium/fvilli/features_BIOVID_S_stride_32_inside_2_combined_last126"
print(f"Root dataset folder: {root_dataset_folder}")
list_path_safetensor = []
for root, dirs, files in os.walk(root_dataset_folder):
  for file in files:
    if file.endswith(".safetensors"):
      list_path_safetensor.append(os.path.join(root,file))
print(f"Number of safetensors found: {len(list_path_safetensor)}")

new_path = "/equilibrium/fvilli/PainAssessmentVideo/partA/video/features/VideoMaeV2_S/features_BIOVID_S_stride_32_inside_2_combined_last126_float16"
for path_safetensor in tqdm.tqdm(list_path_safetensor,desc="Converting to float16..."):
  data = tools.load_dict_data(path_safetensor)
  keys = list(data.keys())
  original_sample_id = min(data['list_sample_id']-helper.get_shift_for_sample_id('bottom_left'))
  if original_sample_id <= 0 or original_sample_id > 8700:
    raise ValueError(f"Original sample id {original_sample_id} is out of range [1, 8700]")
  data['features'] = data['features'].half()
  # count += 1
  old_path = os.path.split(path_safetensor)
  new_path = os.path.join(new_path,old_path[-2],old_path[-1])
  tools.save_dict_data(data,new_path,True)
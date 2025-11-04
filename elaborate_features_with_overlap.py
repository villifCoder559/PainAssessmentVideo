import os
import torch
import custom.tools as tools
import tqdm 
import math


def get_mask(arr,pad=2): # shape [chunks,T,S,S,emb] or [chunks,T,frame_pair=2] or [chunks,1,1]
  # tmp_list_frame =feat['list_frames'].reshape(feat['list_frames'].shape[0], feat['features'].shape[1], -1)
  mask = torch.zeros_like(arr, dtype=bool)
    
  mask[:, pad:-pad, :] = True
  mask[0, 0:pad, :] = True
  mask[ -1, -pad:, :] = True
  
  return mask

def elaborate_features(feat, pad=2):
  new_feat = {}
  for k,v in feat.items():
    orig_shape = v.shape
    # try:
    if k == 'features':  
      mask = get_mask(v, pad=pad)
      new_feat[k] = v[mask].reshape(-1, *orig_shape[1:])
    elif k == 'list_frames':
      v = v.reshape(v.shape[0], -1, 2)
      mask = get_mask(v.reshape(v.shape[0], -1, 2), pad=pad)
      new_feat[k] = v[mask].reshape(-1, orig_shape[1])
    else:
      new_feat[k] = v[:new_feat['features'].shape[0]]
    # except:
      # add -inf as padding
      # print(f"Exception for key {k} with shape {v.shape}, skipping elaboration.")
      # print("Add padding with -inf values...")
      # sum_mask = mask.sum().item()
      # fixed_shape = torch.prod(torch.tensor(orig_shape[1:])).item()
      # target_chunks = math.ceil(sum_mask / fixed_shape)
      # target_mask = target_chunks * fixed_shape
      # padding_needed = target_mask - sum_mask
      # inf_tensor = torch.zeros(padding_needed) - float('inf')
      # v_padded = torch.cat([v[mask], inf_tensor], dim=0)
      # new_feat[k] = v_padded.reshape(-1, *orig_shape[1:])
      
  
  return new_feat
    

whole_feats_folder = "partA/video/features/VideoMaeV2_S/whole_full_features_BIOVID_S_last143_stride8"
new_feats_folder = "partA/video/features/VideoMaeV2_S/whole_full_features_BIOVID_S_last143_stride8_elaborated"
list_feats_path = []
for root,dirs,files in os.walk(whole_feats_folder):
  for file in files:
    if file.endswith(".safetensors") and '$' not in file:
      list_feats_path.append(os.path.join(root,file))
      
print(f"Found {len(list_feats_path)} feature files.")

for feat_path in tqdm.tqdm(list_feats_path, desc="Elaborating features..."):
  feat = tools.load_dict_data(feat_path)
  new_feat = elaborate_features(feat, pad=2)
  relative_path = os.path.relpath(feat_path, whole_feats_folder)
  new_feat_path = os.path.join(new_feats_folder, relative_path)
  os.makedirs(os.path.dirname(new_feat_path), exist_ok=True)
  tools.save_dict_data(new_feat, new_feat_path)

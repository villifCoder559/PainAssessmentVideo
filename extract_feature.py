from custom.dataset import customDataset
from custom.backbone import VideoBackbone,VitImageBackbone
from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import custom.helper as helper
import torch
from torch.utils.data import DataLoader
import os
import custom.tools as tools
import time
import pickle
import gc
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import tqdm
import sys


def masked_spatial_mean(features, excluded_positions_s14=None):
    """
    Apply masked spatial mean reduction to features, excluding face boundary positions.
    
    Args:
        features (torch.Tensor): Tensor of shape [B, T, S, S, C] where:
            - B: batch size
            - T: temporal dimension
            - S: spatial dimension (height and width, must be 10, 14, or 16)
            - C: number of channels
        excluded_positions_s14 (list, optional): List of (row, col) tuples to exclude for S=14.
            If None, uses default face boundary exclusions.
            Negative indices follow Python convention (e.g., -1 is last position).
    
    Returns:
        torch.Tensor: Tensor of shape [B, T, 1, 1, C] with mean computed only over 
                     non-excluded positions.
    
    Raises:
        ValueError: If S is not 10, 14, or 16, or if input shape is invalid.
    
    Examples:
        >>> features = torch.randn(2, 8, 14, 14, 256)  # batch_size=2, temporal=8, spatial=14, channels=256
        >>> output = masked_spatial_mean(features)
        >>> print(output.shape)  # torch.Size([2, 8, 1, 1, 256])
    """
    if len(features.shape) != 5:
        raise ValueError(f"Expected 5D tensor [B,T,S,S,C], got shape {features.shape}")
    
    B, T, S_h, S_w, C = features.shape
    
    if S_h != S_w:
        raise ValueError(f"Expected square spatial dimensions, got [{S_h}, {S_w}]")
    
    S = S_h
    
    if S not in [10, 14, 16]:
        raise ValueError(f"S must be 10, 14, or 16, got {S}")
    
    # Default excluded positions for S=14 (face boundary exclusions)
    if excluded_positions_s14 is None:
        excluded_positions_s14 = [
            (0, 0), (0, 1), (0, 2), (0, -3), (0, -2), (0, -1),
            (1, 0), (1, -1),
            (2, 0), (2, -1),
            (-5, 0), (-5, -1),
            (-4, 0), (-4, -1),
            (-3, 0), (-3, 1), (-3, -2), (-3, -1),
            (-2, 0), (-2, 1), (-2, 2), (-2, -3), (-2, -2), (-2, -1),
            (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4), (-1, -5), (-1, -4), (-1, -3), (-1, -2), (-1, -1)
        ]
    
    # Get excluded positions for current S
    if S == 14:
        excluded_pos = excluded_positions_s14
    elif S == 16:  # S == 16
        # Scale positions from S=14 to S=16 proportionally
        scale_factor = S / 14
        excluded_pos = []
        for row, col in excluded_positions_s14:
            # Convert negative indices to positive for S=14
            r = row if row >= 0 else 14 + row
            c = col if col >= 0 else 14 + col
            
            # Scale positions
            r_scaled = int(round(r * scale_factor))
            c_scaled = int(round(c * scale_factor))
            
            # Handle boundary cases
            r_scaled = min(r_scaled, S - 1)
            c_scaled = min(c_scaled, S - 1)
            
            excluded_pos.append((r_scaled, c_scaled))
        
        excluded_pos = list(set(excluded_pos))  # Remove duplicates
    elif S == 10:
      excluded_pos = [
            # Top Section (Row 0 and 1)
            (0, 0), (0, 1), (0, -2), (0, -1),
            (1, 0), (1, -1),
            
            # Bottom Section (Rows 7, 8, and 9)
            (-3, 0), (-3, -1),
            (-2, 0), (-2, 1), (-2, -2), (-2, -1),
            (-1, 0), (-1, 1), (-1, 2), (-1, -3), (-1, -2), (-1, -1)]
    # Convert to positive indices and create exclusion set
    excluded_set = set()
    for row, col in excluded_pos:
        r = row if row >= 0 else S + row
        c = col if col >= 0 else S + col
        if 0 <= r < S and 0 <= c < S:
            excluded_set.add((r, c))
    
    # Create mask: True where we want to include, False where we exclude
    mask = torch.ones(S, S, dtype=torch.bool, device=features.device)
    for r, c in excluded_set:
        mask[r, c] = False
    
    # Apply mask to features
    # Reshape for broadcasting: [S, S] → [1, 1, S, S, 1]
    mask_expanded = mask.view(1, 1, S, S, 1)
    # Preserve dtype of input
    mask_float = mask_expanded.float().to(dtype=features.dtype)
    masked_features = features * mask_float
    
    # Count valid positions
    valid_count = mask.sum().float()
    
    # Sum over spatial dimensions [B, T, S, S, C] → [B, T, C]
    sum_over_spatial = masked_features.sum(dim=(2, 3))
    
    # Divide by count to get mean
    mean = sum_over_spatial / valid_count
    
    # Reshape to [B, T, 1, 1, C]
    result = mean.unsqueeze(2).unsqueeze(2)
    
    return result


def _apply_pooling(feature, backbone, pooling_embedding_reduction, adaptive_avg_pool3d_out_shape, float_16):
  """
  Apply embedding reduction and optional float16 conversion to extracted features.

  Args:
    feature:                       Raw feature tensor from backbone. Shape: [B, T, S, S, C].
    backbone:                      The backbone model instance.
    pooling_embedding_reduction:   EMBEDDING_REDUCTION enum value.
    adaptive_avg_pool3d_out_shape: Output shape for 3D adaptive pooling, or None.
    float_16:                      If True, convert output to float16.

  Returns:
    Reduced (and optionally half-precision) feature tensor.
  """
  if isinstance(backbone, VideoBackbone) and pooling_embedding_reduction != EMBEDDING_REDUCTION.NONE:
    if pooling_embedding_reduction == EMBEDDING_REDUCTION.ADAPTIVE_POOLING_3D:
      feature = feature.permute(0, 4, 1, 2, 3)
      feature = torch.nn.functional.adaptive_avg_pool3d(feature, output_size=adaptive_avg_pool3d_out_shape)
      feature = feature.permute(0, 2, 3, 4, 1)
    elif pooling_embedding_reduction == EMBEDDING_REDUCTION.SPATIAL_MASKED:
      feature = masked_spatial_mean(feature)
    else:
      feature = torch.mean(feature, dim=pooling_embedding_reduction.value, keepdim=True)
  if float_16:
    feature = feature.half()
  return feature


def _get_available_folder(path):
  """
  Return path unchanged if it doesn't exist, otherwise append $0, $1, … until a free slot is found.

  Args:
    path (str): Desired saving folder path.

  Returns:
    str: A folder path that does not yet exist on disk.
  """
  if not os.path.exists(path):
    return path
  i = 0
  while os.path.exists(f"{path}${i}"):
    i += 1
  return f"{path}${i}"


def main(model_type,pooling_embedding_reduction,adaptive_avg_pool3d_out_shape,enable_batch_extraction,batch_size_feat_extraction,n_workers,saving_chunk_size=100,  preprocess_align = False,
         preprocess_crop_detection = False,preprocess_frontalize = True,path_dataset=None,path_labels=None,stride_window=16,clip_length=16,
         log_file_path=None,root_saving_folder_path=None,backbone_type='video',from_=None,to_=None,save_big_feature=False,h_flip=False,num_clips_per_video=None,
         stride_inside_window=1,float_16=False,color_jitter=False,rotation=False,save_as_safetensors=True,video_extension='.mp4',sample_frame_strategy=SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW,
          backbone_model_path=None,dict_augmentation=None,quadrant=None,shift_frame_idx=0,
         zoom=False,spatial_shift=False,gaussian_smooth=False,gaussian_sigma_min=None,gaussian_sigma_max=None,gaussian_kernel_size=None,framepermute=False,
         **kwargs
         ):
  if backbone_model_path is not None:
    MODEL_TYPE.set_custom_model_type(type=model_type,custom_model_path=backbone_model_path)
    print(f'Using custom backbone model from {backbone_model_path}')
  model_type = MODEL_TYPE.get_model_type(model_type)    
  
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  
  def _write_log_file(log_message):
    if not os.path.exists(os.path.dirname(log_file_path)):
      os.makedirs(os.path.dirname(log_file_path))
    with open(log_file_path,'a') as f:
      f.write(log_message+'\n')

   
  def batch_extraction(data,device,backbone):
    list_batch_data = []
    for batch_data in tqdm.tqdm(data,desc='Extracting features'):
      batch_data = batch_data.unsqueeze(0)
      batch_data = batch_data.to(device)
      with torch.no_grad():
        feature = backbone.forward_features(x=batch_data) # [1,8,14,14,768]
      feature = _apply_pooling(feature, backbone, pooling_embedding_reduction, adaptive_avg_pool3d_out_shape, float_16)
      list_batch_data.append(feature.detach().cpu())
    feature = torch.cat(list_batch_data,dim=0)
    return feature
      
  def _extract_features(dataset,batch_size_feat_extraction,n_workers,backbone,df):
    device = 'cuda'
    print(f"extracting features using.... {device}")
    list_features = []
    list_labels = []
    list_subject_id = []
    list_sample_id = []
    list_path = []
    list_frames = []
    count = 0
    dataloader = DataLoader(dataset, 
                batch_size=batch_size_feat_extraction,
                shuffle=False,
                num_workers=n_workers,
                persistent_workers=True if n_workers > 0 else False,
                collate_fn=dataset._custom_collate_fn_extraction)
    
    
    backbone.model.to(device)
    backbone.model.eval()
    start = time.time()
    str_augmentation = '_'.join([f'{k}' for k,v in dict_augmentation.items() if v])
    for data, labels, subject_id, sample_id, path, list_sampled_frames in dataloader:
      if enable_batch_extraction:
        feature = batch_extraction(data=data,device=device,backbone=backbone)
      else:
        data = data.to(device)
        with torch.no_grad():
          feature = backbone.forward_features(x=data) # [1,T,S,S,C]
        feature = _apply_pooling(feature, backbone, pooling_embedding_reduction, adaptive_avg_pool3d_out_shape, float_16)
      list_frames.append(list_sampled_frames)
      list_features.append(feature.detach().cpu())
      list_labels.append(labels)
      split_path = path[-1].split('$')
      base_shift = 0
      
      # Check if is the case where the video is split into multiple quadrants
      if len(split_path) > 1:
        video_part = split_path[-1].split('.')[0]
        base_shift = helper.get_shift_for_sample_id(video_part)
      sample_id = sample_id + base_shift
      
      # update sample_id based on augmentation
      list_sample_id.append(sample_id + (helper.get_shift_for_sample_id(str_augmentation) if str_augmentation != '' else 0))
      list_subject_id.append(subject_id)
      list_path.append(path)
      
      count += 1
      end = time.time()
      elapsed = end - start
      expected_end = elapsed * (len(dataloader) / count)
      print(f'Batch {count}/{len(dataloader)} | feature {feature.shape} dtype {feature.dtype} | sample_id {list_sample_id[-1]}')
      print(f'Elapsed: {elapsed//3600:.0f}h {elapsed//60%60:.0f}m {elapsed%60:.0f}s | ETA: {expected_end//3600:.0f}h {expected_end//60%60:.0f}m {expected_end%60:.0f}s')
      if count % 10 == 0:
        free_gb, total_gb = (v / 1024**3 for v in torch.cuda.mem_get_info())
        print(f'GPU: {free_gb:.2f}/{total_gb:.2f} GB free | feat size: {feature.element_size()*feature.nelement()/1024**3:.3f} GB')
        torch.cuda.empty_cache()
      if count % saving_chunk_size == 0:
        if not save_big_feature:
          dict_data = {
            'features': torch.cat(list_features,dim=0),
            'list_labels': torch.cat(list_labels,dim=0).to(torch.int32),
            'list_subject_id': torch.cat(list_subject_id).squeeze().to(torch.int32),
            'list_sample_id': torch.cat(list_sample_id).to(torch.int32),
            'list_path': np.concatenate(list_path), # not saved in .safetensors
            'list_frames': torch.cat(list_frames,dim=0).to(torch.int32)
          }
          tools.save_dict_data(dict_data=dict_data,
                               save_as_safetensors=save_as_safetensors,
                               saving_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)))
          _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} \n time elapsed: {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s\n')
        else:
          
          dict_data = {
            'features': torch.cat(list_features,dim=0),
            'list_labels': torch.cat(list_labels,dim=0).to(torch.int32),
            'list_subject_id': torch.cat(list_subject_id).squeeze().to(torch.int32),
            'list_sample_id': torch.cat(list_sample_id).to(torch.int32),
            'list_path': np.concatenate(list_path),
            'list_frames': torch.cat(list_frames,dim=0).to(torch.int32)
          }
          path = Path(list_path[0][0])
          person_id = path.parts[-2] if 'caer' not in str(path).lower() else os.path.join(*path.parts[-3:-1])
          sample_id = path.parts[-1][:-4]
          tools.save_dict_data(dict_data=dict_data,
                               save_as_safetensors=save_as_safetensors,
                               saving_folder_path=os.path.join(root_saving_folder_path,person_id,sample_id))
          _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} \n')
        list_features = []
        list_labels = []
        list_subject_id = []
        list_sample_id = []
        list_path = []
        list_frames = []
        del dict_data
    if len(list_features) > 0:
      dict_data = {
            'features': torch.cat(list_features,dim=0),
            'list_labels': torch.cat(list_labels,dim=0).to(torch.int32),
            'list_subject_id': torch.cat(list_subject_id).squeeze().to(torch.int32),
            'list_sample_id': torch.cat(list_sample_id).to(torch.int32),
            'list_path': np.concatenate(list_path),
            'list_frames': torch.cat(list_frames,dim=0).to(torch.int32)
          }
      tools.save_dict_data(dict_data=dict_data,
                           save_as_safetensors=save_as_safetensors,
                           saving_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)) if not save_big_feature else root_saving_folder_path)
      _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} \n')


  print(f'Model type: {model_type.name}, {model_type.value}')
  if backbone_type == 'video':
    backbone_model = VideoBackbone(model_type=model_type,
                                   custom_model_path=True if backbone_model_path is not None else False,
                                   use_sdpa=True)
  elif backbone_type == 'image':
    backbone_model = VitImageBackbone()
    stride_window = 1
    clip_length = 1
    # Images use a single still frame; default the extension to .jpg unless explicitly overridden.
    if video_extension == '.mp4':
      video_extension = '.jpg'
  else:
    raise ValueError('Backbone type must be video or image')
  # Image backbones expose a backbone-specific normalization (ViT mean/std); video backbones do not.
  image_mean = getattr(backbone_model, 'image_mean', None)
  image_std = getattr(backbone_model, 'image_std', None)
  video_labels = None
  if from_ is not None or to_ is not None:
    video_labels = pd.read_csv(path_labels,sep='\t')
    video_labels = video_labels.iloc[from_:(to_)]
    video_labels = pd.DataFrame(video_labels)
    print(f'video_labels: {video_labels}')
    print(f"First element of video_labels: {video_labels.iloc[0,0]}")
    print(f"Last element of video_labels : {video_labels.iloc[-1,0]}")
  custom_ds = customDataset(path_dataset=path_dataset,
                path_labels=path_labels,
                num_clips_per_video=num_clips_per_video,
                sample_frame_strategy=sample_frame_strategy,
                image_resize_w=backbone_model.img_size,
                image_resize_h=backbone_model.img_size,
                stride_window=stride_window,
                clip_length=clip_length,
                model_type=model_type,
                video_labels=video_labels,
                h_flip=h_flip,
                color_jitter=color_jitter,
                rotation=rotation,
                spatial_shift=spatial_shift,
                zoom=zoom,
                gaussian_smooth=gaussian_smooth,
                gaussian_sigma_min=gaussian_sigma_min,
                gaussian_sigma_max=gaussian_sigma_max,
                gaussian_kernel_size=gaussian_kernel_size,
                framepermute=framepermute,
                shift_frame_idx=shift_frame_idx,
                video_extension=video_extension,
                preprocess_align=preprocess_align,
                preprocess_frontalize=preprocess_frontalize,
                stride_inside_window=stride_inside_window,
                quadrant=quadrant,
                preprocess_crop_detection=preprocess_crop_detection,
                backbone_type=backbone_type,
                image_mean=image_mean,
                image_std=image_std,
                saving_folder_path_extracted_video=None)
  
  config_dict = {
    'path_dataset': path_dataset,
    'path_labels': path_labels,
    'from_': from_,
    'to_': to_,
    'log_file_path': log_file_path,
    'saving_folder_path': root_saving_folder_path,
    'saving_chunk_size': saving_chunk_size,
    'model_type': model_type.value,
    'pooling_embedding_reduction': pooling_embedding_reduction,
    'pooling_clips_reduction': pooling_clips_reduction,
    'sample_frame_strategy': sample_frame_strategy,
    'stride_window': stride_window,
    'clip_length': clip_length,
    'preprocess_align': preprocess_align,
    'preprocess_frontalize': preprocess_frontalize,
    'preprocess_crop_detection': preprocess_crop_detection,
    'batch_size_feat_extraction': 1,
    'backbone_type': backbone_type,
    'n_workers': n_workers,
    'save_big_feature': save_big_feature,
    **dict_augmentation,
    'stride_inside_window': stride_inside_window,
    'float_16': float_16,
    'video_extension': video_extension,
    'shift_frame_idx': shift_frame_idx,
    'command_prompt': ' '.join(sys.argv),
    # 'backbone_model': backbone_model,
  }
  if not os.path.exists(root_saving_folder_path):
    os.makedirs(root_saving_folder_path)
  with open(os.path.join(root_saving_folder_path,'config_dict.pkl'),'wb') as f:
    pickle.dump(config_dict,f)
    print(f'config_dict saved in {os.path.join(root_saving_folder_path,"config_dict.pkl")}')
  # prompt_used = ' '.join(os.sys.argv)
  with open(os.path.join(root_saving_folder_path,'config_dict.txt'),'w') as f:
    for k,v in config_dict.items():
      f.write(f'{k}: {v}\n')
    print(f'config txt saved in {os.path.join(root_saving_folder_path,"config_dict.txt")}')
  _extract_features(dataset=custom_ds,
                  batch_size_feat_extraction=batch_size_feat_extraction,
                  n_workers=n_workers,
                  backbone=backbone_model,
                  df=video_labels)
  gc.collect()
  torch.cuda.empty_cache()
  
import tempfile
import multiprocessing as mp
tempfile.tempdir = '/tmp'
mp.util._exit_function = lambda: None


if __name__ == "__main__":
  
  timestamp = int(time.time())
  parser = argparse.ArgumentParser(description='Extract features from video dataset.')
  parser.add_argument('--gp', action='store_true', help='Global path')
  parser.add_argument('--model_type', type=str, required=False, default="B")
  parser.add_argument('--backbone_model_path', type=str, required=False, default=None, help='Path to custom model for feats_extraction')
  parser.add_argument('--saving_after', type=int, required=False, default=8800,help='Number of batch to save in one file')
  parser.add_argument('--emb_red', type=str, default='spatial', help='Embedding reduction. Can be spatial, temporal, all, none, adaptive_pooling_3d, spatial_masked')
  # parser.add_argument('--prep_al', action='store_true', help='Preprocess align') # deprecated not use
  # parser.add_argument('--prep_crop', action='store_true', help='Preprocess crop') # deprecated not use
  # parser.add_argument('--prep_front', action='store_true', help='Preprocess frontalize') # deprecated not use
  parser.add_argument('--from_', type=int, default=None, help='START idx (included) extracting features from (--path_labels) row. Start from 0')
  parser.add_argument('--to_', type=int, default=None, help='STOP idx (excluded) extracting features from (--path_labels) row')
  parser.add_argument('--path_dataset', type=str, default=os.path.join('partA','video','video'), help='Path to dataset')
  parser.add_argument('--path_labels', type=str, default=os.path.join('partA','starting_point','samples.csv'), help='Path to csv file')
  parser.add_argument('--saving_folder_path',required=True, type=str, default=None, help='Path to saving folder')
  parser.add_argument('--log_file_path', type=str, default=None, help='Path to log file')
  parser.add_argument('--backbone_type', type=str, default='video', help='Type of backbone. Can be video or image')
  parser.add_argument('--batch_size_feat_extraction', type=int, default=1, help='Batch size for feature extraction')
  parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for dataloader')
  parser.add_argument('--save_big_feature', action='store_true', help='Save one feature per sample')
  parser.add_argument('--stride_window', type=int, default=16, help='Stride window')
  parser.add_argument('--stride_inside_window', type=int, default=1, help='Stride inside window')
  parser.add_argument('--clip_length', type=int, default=16, help='Clip length')
  parser.add_argument('--shift_frame_idx', type=int, default=0, help='Shift frame index to change the last frame sampled')
  
  parser.add_argument('--h_flip', action='store_true', help='Apply Horizontal flip')
  parser.add_argument('--color_jitter', action='store_true', help='Apply color jitter')
  parser.add_argument('--spatial_shift', action='store_true', help='Apply spatial shift to video')
  parser.add_argument('--rotation', action='store_true', help='Apply rotation')
  parser.add_argument('--zoom', action='store_true', help='Apply zoom')
  parser.add_argument('--gaussian_smooth', action='store_true', help='Apply Gaussian smoothing')
  parser.add_argument('--gaussian_sigma_min', type=float, default=None, help='Min sigma for Gaussian smoothing')
  parser.add_argument('--gaussian_sigma_max', type=float, default=None, help='Max sigma for Gaussian smoothing')
  parser.add_argument('--gaussian_kernel_size', type=int, default=None, help='Kernel size for Gaussian smoothing (must be odd)')
  parser.add_argument('--framepermute', action='store_true', help='Randomly permute frame order within each clip')
  
  parser.add_argument('--float_16', action='store_true', help='Use float 16')
  parser.add_argument('--save_as_safetensors', action='store_true', help='Save as safetensors')
  parser.add_argument('--adaptive_avg_pool3d_out_shape', type=int, nargs='*', default=[2,2,2], help='3d pooling kernel size')
  parser.add_argument('--enable_batch_extraction', action='store_true', help='Enable batch extraction')
  parser.add_argument('--video_extension', type=str, default='.mp4', help='Video extension to use for dataset')
  parser.add_argument('--sample_frame_strategy', type=str, default='sliding_window', help=f'Strategy to sample frames from video. Can be {list(SAMPLE_FRAME_STRATEGY)}')
  parser.add_argument('--num_clips_per_video', type=int, default=None, help='Number of clips per video. If None, all clips will be used. Can be used only with random sampling strategy')
  parser.add_argument('--quadrant', type=str, default=None, help="Filter quadrants part (upper_left,upper_right,bottom_left,bottom_right,top,down). If None it will be ignored")
  args = parser.parse_args()
  
  if len(args.adaptive_avg_pool3d_out_shape) != 3:
    raise ValueError('adaptive_avg_pool3d_out_shape must have 3 integers')
  adaptive_avg_pool3d_out_shape = args.adaptive_avg_pool3d_out_shape
  args.emb_red = EMBEDDING_REDUCTION.get_embedding_reduction(args.emb_red)
  print(f'\nEmbedding reduction:\n name:{args.emb_red.name} \n value: {args.emb_red.value}')
  # if args.from_ is not None or args.to_ is not None:
  #   args.saving_folder_path = f'{args.saving_folder_path}_{args.from_}_{args.to_}'
  #   print(f'Saving folder path: {args.saving_folder_path}')
  if args.gp:
    args.path_dataset = GLOBAL_PATH.get_global_path(args.path_dataset)
    args.path_labels = GLOBAL_PATH.get_global_path(args.path_labels)
    args.log_file_path = GLOBAL_PATH.get_global_path(args.log_file_path)
    args.saving_folder_path = GLOBAL_PATH.get_global_path(args.saving_folder_path)
    print(f'path_dataset: {args.path_dataset}\n')
    print(f'path_labels: {args.path_labels}')
    print(f'log_file_path: {args.log_file_path}')
    print(f'saving_folder_path: {args.saving_folder_path}')
  print(args)
  if args.save_big_feature:
    args.saving_after = 1
  dict_augmentation = {
    'hflip': False,
    'jitter': False,
    'rotation': False,
    'shift': False,
    'zoom': False,
    'reverse':False,
    'timelapse':False,
    'slowmotion':False,
    'gaussian': False,
    'framepermute': False,
  }
  for k,v in dict_augmentation.items():
    if k in args.path_dataset:
      dict_augmentation[k] = True
  # Also set from CLI flags
  dict_augmentation['hflip'] = dict_augmentation['hflip'] or args.h_flip
  dict_augmentation['jitter'] = dict_augmentation['jitter'] or args.color_jitter
  dict_augmentation['rotation'] = dict_augmentation['rotation'] or args.rotation
  dict_augmentation['shift'] = dict_augmentation['shift'] or args.spatial_shift
  dict_augmentation['zoom'] = dict_augmentation['zoom'] or args.zoom
  dict_augmentation['gaussian'] = dict_augmentation['gaussian'] or args.gaussian_smooth
  dict_augmentation['framepermute'] = dict_augmentation['framepermute'] or args.framepermute
  print(f'\nData augmentation used for feature extraction:\n')
  for k,v in dict_augmentation.items():
    print(f'  {k}: {v}')
  args.saving_folder_path = _get_available_folder(args.saving_folder_path)
  print(f'Saving folder path (resolved): {args.saving_folder_path}')
  # sleep 15 seconds to let user see the args
  time.sleep(15)
  if args.log_file_path is None:
    args.log_file_path = os.path.join(args.saving_folder_path,'log_file.txt')
  dict_args = vars(args)
  main(model_type=args.model_type,
       saving_chunk_size=args.saving_after,
       sample_frame_strategy=helper.get_sampling_frame_startegy(args.sample_frame_strategy),
       num_clips_per_video=args.num_clips_per_video,
       preprocess_align=None,
       preprocess_crop_detection=None,
       preprocess_frontalize=None,
       path_dataset=args.path_dataset,
       path_labels=args.path_labels,
       log_file_path=args.log_file_path,
       pooling_embedding_reduction=args.emb_red,
       root_saving_folder_path=args.saving_folder_path,
       backbone_type=args.backbone_type,
       from_=args.from_,
       to_=args.to_,
       batch_size_feat_extraction=args.batch_size_feat_extraction,
       n_workers=args.n_workers,
       save_big_feature=args.save_big_feature,
       stride_window=args.stride_window,
       clip_length=args.clip_length,
       dict_augmentation=dict_augmentation,
       stride_inside_window=args.stride_inside_window,
       float_16=args.float_16,
       save_as_safetensors=args.save_as_safetensors,
       adaptive_avg_pool3d_out_shape=adaptive_avg_pool3d_out_shape,
       enable_batch_extraction=args.enable_batch_extraction,
       video_extension=args.video_extension,
       backbone_model_path=args.backbone_model_path,
       quadrant=args.quadrant,
       shift_frame_idx=args.shift_frame_idx,
       h_flip=args.h_flip,
       color_jitter=args.color_jitter,
       rotation=args.rotation,
       spatial_shift=args.spatial_shift,
       zoom=args.zoom,
       gaussian_smooth=args.gaussian_smooth,
       gaussian_sigma_min=args.gaussian_sigma_min,
       gaussian_sigma_max=args.gaussian_sigma_max,
       gaussian_kernel_size=args.gaussian_kernel_size,
       framepermute=args.framepermute,
      #  image_resize=args.image_resize,
      #  **dict_args
       )

from custom.dataset import customDataset
from custom.backbone import VideoBackbone,VitImageBackbone
from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import custom.helper as helper
import torch
from torch.utils.data import DataLoader
import os
# from transformers import AutoImageProcessor
import custom.tools as tools
import time
import pickle
import gc
import argparse
import pandas as pd
# import torch.multiprocessing
from pathlib import Path
import numpy as np
import tqdm
import pandas as pd
import sys
# import torch.nn as nn

def main(model_type,pooling_embedding_reduction,adaptive_avg_pool3d_out_shape,enable_batch_extraction,batch_size_feat_extraction,n_workers,saving_chunk_size=100,  preprocess_align = False,
         preprocess_crop_detection = False,preprocess_frontalize = True,path_dataset=None,path_labels=None,stride_window=16,clip_length=16,
         log_file_path=None,root_saving_folder_path=None,backbone_type='video',from_=None,to_=None,save_big_feature=False,h_flip=False,num_clips_per_video=None,
         stride_inside_window=1,float_16=False,color_jitter=False,rotation=False,save_as_safetensors=True,video_extension='.mp4',sample_frame_strategy=SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW,
          backbone_model_path=None,dict_augmentation=None,quadrant=None,shift_frame_idx=0,**kwargs
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
      if isinstance(backbone,VideoBackbone) and pooling_embedding_reduction != EMBEDDING_REDUCTION.NONE:
        if adaptive_avg_pool3d_out_shape is not None and pooling_embedding_reduction == EMBEDDING_REDUCTION.ADAPTIVE_POOLING_3D:
          feature = feature.permute(0,4,1,2,3) # [1,768,8,14,14]
          feature = torch.nn.functional.adaptive_avg_pool3d(feature,output_size=adaptive_avg_pool3d_out_shape) # [1,768,2,2,2]
          feature = feature.permute(0,2,3,4,1) # [1,2,2,2,768]
        else:
          feature = torch.mean(feature,dim=pooling_embedding_reduction.value,keepdim=True)
      if float_16:
        feature = feature.half()
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
                collate_fn=dataset._custom_collate_fn_extraction)
    backbone.model.to(device)
    backbone.model.eval()

    start = time.time()
    for data, labels, subject_id, sample_id, path, list_sampled_frames in dataloader:
      if enable_batch_extraction:
        feature = batch_extraction(data=data,device=device,backbone=backbone)
      else:
        print(f'extracting features from {path[0]}')
        data = data.to(device)
        with torch.no_grad():
          feature = backbone.forward_features(x=data) # [1,8,14,14,768]
        if isinstance(backbone,VideoBackbone) and pooling_embedding_reduction != EMBEDDING_REDUCTION.NONE:
          if adaptive_avg_pool3d_out_shape is not None and pooling_embedding_reduction == EMBEDDING_REDUCTION.ADAPTIVE_POOLING_3D:
            feature = feature.permute(0,4,1,2,3) # [1,768,8,14,14]
            feature = torch.nn.functional.adaptive_avg_pool3d(feature,output_size=adaptive_avg_pool3d_out_shape) # [1,768,2,2,2]
            feature = feature.permute(0,2,3,4,1) # [1,2,2,2,768]
          else:
            feature = torch.mean(feature,dim=pooling_embedding_reduction.value,keepdim=True)
        if float_16:
          feature = feature.half()
      print(f'batch feature shape {feature.shape}')
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
      if dict_augmentation['h_flip']:
        list_sample_id.append(sample_id+helper.get_shift_for_sample_id('hflip')) # if h_flip, add 8700 to sample_id
      elif dict_augmentation['color_jitter']:
        list_sample_id.append(sample_id+helper.get_shift_for_sample_id('jitter'))
      elif dict_augmentation['rotation']:
        list_sample_id.append(sample_id+helper.get_shift_for_sample_id('rotate'))
      elif dict_augmentation['spatial_shift']:
        list_sample_id.append(sample_id+helper.get_shift_for_sample_id('spatial_shift'))
      else:
        list_sample_id.append(sample_id)
      print(f'sample_id: {list_sample_id[-1]}')
      print(f'list_frames: {list_sampled_frames[-1] if list_sampled_frames.ndim > 1 else list_sampled_frames}')
      list_subject_id.append(subject_id)
      list_path.append(path)
      
      # add list for random cropped part
      count += 1
      print(f'Batch {count}/{len(dataloader)}')
      print(f'GPU:\n Free : {torch.cuda.mem_get_info()[0]/1024/1024/1024:.2f} GB \n total: {torch.cuda.mem_get_info()[1]/1024/1024/1024:.2f} GB')
      torch.cuda.empty_cache()
      end = time.time()
      print(f'Elapsed time : {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s')
      expected_end = (end - start) * (len(dataloader) / count)
      if count % 20 == 0:
        print(f'Dict size in GB: {feature.element_size()*feature.nelement()/1024/1024/1024:.2f} GB')
      print(f'Expected time: {expected_end//60//60:.0f} h {expected_end//60%60:.0f} m {expected_end%60:.0f} s\n')
      # if count % 10 == 0:
      #   _write_log_file(f'Batch {count}/{len(dataloader)}')
      # save_big_feature = True 
      if count % saving_chunk_size == 0:
        if not save_big_feature:
          dict_data = {
            'features': torch.cat(list_features,dim=0).half() if float_16 else torch.cat(list_features,dim=0),
            'list_labels': torch.cat(list_labels,dim=0).to(torch.int32),
            'list_subject_id': torch.cat(list_subject_id).squeeze().to(torch.int32),
            'list_sample_id': torch.cat(list_sample_id).to(torch.int32),
            'list_path': np.concatenate(list_path), # not saved in .safetensors
            'list_frames': torch.cat(list_frames,dim=0).to(torch.int32)
          }
          # dict_data_size = dict_data["features"].element_size()*dict_data["features"].nelement()/1024/1024
          tools.save_dict_data(dict_data=dict_data,
                               save_as_safetensors=save_as_safetensors,
                               saving_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)))
          _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} \n time elapsed: {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s\n')
        else:
          
          dict_data = {
            'features': torch.cat(list_features,dim=0).half() if float_16 else torch.cat(list_features,dim=0),
            'list_labels': torch.cat(list_labels,dim=0).to(torch.int32),
            'list_subject_id': torch.cat(list_subject_id).squeeze().to(torch.int32),
            'list_sample_id': torch.cat(list_sample_id).to(torch.int32),
            'list_path': np.concatenate(list_path),
            'list_frames': torch.cat(list_frames,dim=0).to(torch.int32)
          }
          # dict_data_size = dict_data["features"].element_size()*dict_data["features"].nelement()/1024/1024
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
    # backbone.model.to('cpu')
    # save last batch
    if len(list_features)>0:
      dict_data = {
            'features': torch.cat(list_features,dim=0).half() if float_16 else torch.cat(list_features,dim=0),
            'list_labels': torch.cat(list_labels,dim=0).to(torch.int32),
            'list_subject_id': torch.cat(list_subject_id).squeeze().to(torch.int32),
            'list_sample_id': torch.cat(list_sample_id).to(torch.int32),
            'list_path': np.concatenate(list_path),
            'list_frames': torch.cat(list_frames,dim=0).to(torch.int32)
          }
      tools.save_dict_data(dict_data=dict_data,
                           save_as_safetensors=save_as_safetensors,
                           saving_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)) if not save_as_safetensors else root_saving_folder_path)
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
  else:
    raise ValueError('Backbone type must be video or image')
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
                video_labels=video_labels,
                h_flip=dict_augmentation['h_flip'],
                color_jitter=dict_augmentation['color_jitter'],
                rotation=dict_augmentation['rotation'],
                spatial_shift=dict_augmentation['spatial_shift'],
                shift_frame_idx=shift_frame_idx,
                video_extension=video_extension,
                preprocess_align=preprocess_align,
                preprocess_frontalize=preprocess_frontalize,
                stride_inside_window=stride_inside_window,
                quadrant=quadrant,
                preprocess_crop_detection=preprocess_crop_detection,
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
  
  

if __name__ == "__main__":
  # print('Setting sharing strategy')
  # torch.multiprocessing.set_sharing_strategy('file_system')
  
  timestamp = int(time.time())
  parser = argparse.ArgumentParser(description='Extract features from video dataset.')
  parser.add_argument('--gp', action='store_true', help='Global path')
  parser.add_argument('--model_type', type=str, required=False, default="B")
  parser.add_argument('--backbone_model_path', type=str, required=False, default=None, help='Path to custom model for feats_extraction')
  parser.add_argument('--saving_after', type=int, required=False, default=8800,help='Number of batch to save in one file')
  parser.add_argument('--emb_red', type=str, default='spatial', help='Embedding reduction. Can be spatial, temporal, all, none, adaptive_pooling_3d')
  # parser.add_argument('--prep_al', action='store_true', help='Preprocess align') # deprecated not use
  # parser.add_argument('--prep_crop', action='store_true', help='Preprocess crop') # deprecated not use
  # parser.add_argument('--prep_front', action='store_true', help='Preprocess frontalize') # deprecated not use
  parser.add_argument('--from_', type=int, default=None, help='START idx (included) extracting features from (--path_labels) row. Start from 0')
  parser.add_argument('--to_', type=int, default=None, help='STOP idx (excluded) extracting features from (--path_labels) row')
  parser.add_argument('--path_dataset', type=str, default=os.path.join('partA','video','video'), help='Path to dataset')
  parser.add_argument('--path_labels', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), help='Path to csv file')
  parser.add_argument('--saving_folder_path', type=str, default=os.path.join('partA','video','features',f'samples_16_{timestamp}'), help='Path to saving folder')
  parser.add_argument('--log_file_path', type=str, default=None, help='Path to log file')
  parser.add_argument('--backbone_type', type=str, default='video', help='Type of backbone. Can be video or image')
  parser.add_argument('--batch_size_feat_extraction', type=int, default=1, help='Batch size for feature extraction')
  parser.add_argument('--n_workers', type=int, default=0, help='Number of workers for dataloader')
  parser.add_argument('--save_big_feature', action='store_true', help='Save one feature per sample')
  parser.add_argument('--stride_window', type=int, default=16, help='Stride window')
  parser.add_argument('--stride_inside_window', type=int, default=1, help='Stride inside window')
  parser.add_argument('--clip_length', type=int, default=16, help='Clip length')
  parser.add_argument('--h_flip', action='store_true', help='Apply Horizontal flip')
  parser.add_argument('--shift_frame_idx', type=int, default=0, help='Shift frame index to change the last frame sampled')
  parser.add_argument('--color_jitter', action='store_true', help='Apply color jitter')
  parser.add_argument('--spatial_shift', action='store_true', help='Apply spatial shift to video')
  parser.add_argument('--rotation', action='store_true', help='Apply rotation')
  parser.add_argument('--float_16', action='store_true', help='Use float 16')
  parser.add_argument('--save_as_safetensors', action='store_true', help='Save as safetensors')
  parser.add_argument('--adaptive_avg_pool3d_out_shape', type=int, nargs='*', default=[2,2,2], help='3d pooling kernel size')
  parser.add_argument('--enable_batch_extraction', action='store_true', help='Enable batch extraction')
  parser.add_argument('--video_extension', type=str, default='.mp4', help='Video extension to use for dataset')
  parser.add_argument('--sample_frame_strategy', type=str, default='sliding_window', help=f'Strategy to sample frames from video. Can be {list(SAMPLE_FRAME_STRATEGY)}')
  parser.add_argument('--num_clips_per_video', type=int, default=None, help='Number of clips per video. If None, all clips will be used. Can be used only with random sampling strategy')
  parser.add_argument('--quadrant', type=str, default=None, help="Filter quadrants part (ipper_left,upper_right,bottom_left,bottom_right). If None it will be ignored")
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
    'h_flip': args.h_flip,
    'color_jitter': args.color_jitter,
    'rotation': args.rotation,
    'spatial_shift': args.spatial_shift
  }
  
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
      #  **dict_args
       )
  
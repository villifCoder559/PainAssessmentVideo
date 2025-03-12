from custom.dataset import customDataset
from custom.backbone import video_backbone,vit_image_backbone
from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import torch
from torch.utils.data import DataLoader
import os
from transformers import AutoImageProcessor
import custom.tools as tools
import time
import pickle
import gc
import argparse
import pandas as pd
import torch.multiprocessing
from pathlib import Path
import numpy as np

def main(model_type,pooling_embedding_reduction,batch_size_feat_extraction,n_workers,saving_chunk_size=100,  preprocess_align = False,
         preprocess_crop_detection = False,preprocess_frontalize = True,path_dataset=None,path_labels=None,
         log_file_path=None,root_saving_folder_path=None,backbone_type='video',from_=None,to_=None,save_big_feature=False
         ):
  if model_type == 'B':
    model_type = MODEL_TYPE.VIDEOMAE_v2_B
  elif model_type == 'S':
    model_type = MODEL_TYPE.VIDEOMAE_v2_S
  else:
    raise ValueError('Model type not recognized. Please use "B" or "S"')
    
  # pooling_embedding_reduction = EMBEDDING_REDUCTION.MEAN_SPATIAL
  # print(f'Pooling_clips_reduction: {pooling_clips_reduction}')  
  print(f'Pooling_embedding_reduction: {pooling_embedding_reduction.name}')
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  
  def _write_log_file(log_message):
    if not os.path.exists(os.path.dirname(log_file_path)):
      os.makedirs(os.path.dirname(log_file_path))
    with open(log_file_path,'a') as f:
      f.write(log_message+'\n')

  def _extract_features(dataset,batch_size_feat_extraction,n_workers,backbone):
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
                collate_fn=dataset._custom_collate_fn)
    backbone.model.to(device)
    backbone.model.eval()
    count = 0
    start = time.time()
    with torch.no_grad():
      for data, labels, subject_id,sample_id, path, list_sampled_frames in dataloader:
        data = data.to(device)
        print(f'extracting features from {path[0]}')
        with torch.no_grad():
          feature = backbone.forward_features(x=data) # [1,8,14,14,768]
        if isinstance(backbone,video_backbone) and pooling_embedding_reduction != EMBEDDING_REDUCTION.NONE:
          feature = torch.mean(feature,dim=pooling_embedding_reduction.value,keepdim=True)
        print(f'feature shape {feature.shape}')
        list_frames.append(list_sampled_frames)
        list_features.append(feature.detach().cpu())
        list_labels.append(labels)
        list_sample_id.append(sample_id)
        list_subject_id.append(subject_id)
        list_path.append(path)
        # add list for random cropped part
        count += 1
        print(f'\nBatch {count}/{len(dataloader)}')
        print(f'GPU:\n Free : {torch.cuda.mem_get_info()[0]/1024/1024/1024:.2f} GB \n total: {torch.cuda.mem_get_info()[1]/1024/1024/1024:.2f} GB\n')
        del data, feature
        torch.cuda.empty_cache()
        end = time.time()
        print(f'Elapsed time : {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s')
        expected_end = (end - start) * (len(dataloader) / count)
        print(f'Expected time: {expected_end//60//60:.0f} h {expected_end//60%60:.0f} m {expected_end%60:.0f} s')
        # if count % 10 == 0:
        #   _write_log_file(f'Batch {count}/{len(dataloader)}')
        # save_big_feature = True 
        if count % saving_chunk_size == 0:
          if not save_big_feature:
            dict_data = {
              'features': torch.cat(list_features,dim=0),
              'list_labels': torch.cat(list_labels,dim=0),
              'list_subject_id': torch.cat(list_subject_id).squeeze(),
              'list_sample_id': torch.cat(list_sample_id),
              'list_path': np.concatenate(list_path),
              'list_frames': torch.cat(list_frames,dim=0)
            }
            dict_data_size = dict_data["features"].element_size()*dict_data["features"].nelement()/1024/1024
            tools.save_dict_data(dict_data=dict_data,
                      savinsaving_chunk_sizeg_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)))
            _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} with size {dict_data_size:.2f} MB \n time elapsed: {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s\n')
          else:
            
            dict_data = {
              'features': torch.cat(list_features,dim=0),
              'list_labels': torch.cat(list_labels,dim=0),
              'list_subject_id': torch.cat(list_subject_id).squeeze(),
              'list_sample_id': torch.cat(list_sample_id),
              'list_path': np.concatenate(list_path),
              'list_frames': torch.cat(list_frames,dim=0)
            }
            dict_data_size = dict_data["features"].element_size()*dict_data["features"].nelement()/1024/1024
            path = Path(list_path[0][0])
            person_id = path.parts[-2]
            sample_id = path.parts[-1][:-4]
            tools.save_dict_data(dict_data=dict_data,
                      saving_folder_path=os.path.join(root_saving_folder_path,person_id,sample_id))
            _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} \n')
          list_features = []
          list_labels = []
          list_subject_id = []
          list_sample_id = []
          list_path = []
          list_frames = []
          del dict_data
    backbone.model.to('cpu')
    # save last batch
    if len(list_features)>0:
      dict_data = {
        'features': torch.cat(list_features,dim=0),
        'list_labels': torch.cat(list_labels,dim=0),
        'list_subject_id': torch.cat(list_subject_id).squeeze(),
        'list_sample_id': torch.cat(list_sample_id),
        'list_path': np.concatenate(list_path),
        'list_frames': torch.cat(list_frames,dim=0)
      }
      tools.save_dict_data(dict_data=dict_data,
                  saving_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)))
      _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} \n')
  
  print('Model type:',model_type)
  if backbone_type == 'video':
    backbone_model = video_backbone(model_type=model_type)
    stride_window = 16
    clip_length = 16
  elif backbone_type == 'image':
    backbone_model = vit_image_backbone()
    stride_window = 1
    clip_length = 1
  else:
    raise ValueError('Backbone type not recognized. Please use "video" or "image"')
  video_labels = None
  if from_ is not None or to_ is not None:
    video_labels = pd.read_csv(path_labels)
    video_labels = video_labels.iloc[from_:(to_)]
    video_labels = pd.DataFrame(video_labels)
    print(f'video_labels: {video_labels}')
    print(f"First element of video_labels: {video_labels.iloc[0,0]}")
    print(f"Last element of video_labels : {video_labels.iloc[-1,0]}")
  custom_ds = customDataset(path_dataset=path_dataset,
                path_labels=path_labels,
                sample_frame_strategy=sample_frame_strategy,
                stride_window=stride_window,
                clip_length=clip_length,
                video_labels=video_labels,
                preprocess_align=preprocess_align,
                preprocess_frontalize=preprocess_frontalize,
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
    'model_type': model_type,
    'pooling_embedding_reduction': pooling_embedding_reduction,
    'pooling_clips_reduction': pooling_clips_reduction,
    'sample_frame_strategy': sample_frame_strategy,
    'stride_window': stride_window,
    'clip_length': clip_length,
    'preprocess_align': preprocess_align,
    'preprocess_frontalize': preprocess_frontalize,
    'preprocess_crop_detection': preprocess_crop_detection,
    'batch_size_feat_extraction': 1,
    'backbone_type': backbone_type
  }
  if not os.path.exists(root_saving_folder_path):
    os.makedirs(root_saving_folder_path)
  with open(os.path.join(root_saving_folder_path,'config_dict.pkl'),'wb') as f:
    pickle.dump(config_dict,f)
    print(f'config_dict saved in {os.path.join(root_saving_folder_path,"config_dict.pkl")}')
  with open(os.path.join(root_saving_folder_path,'config_dict.txt'),'w') as f:
    for k,v in config_dict.items():
      f.write(f'{k}: {v}\n')
    print(f'config txt saved in {os.path.join(root_saving_folder_path,"config_dict.txt")}')
  _extract_features(dataset=custom_ds,
                  batch_size_feat_extraction=batch_size_feat_extraction,
                  n_workers=n_workers,
                  backbone=backbone_model)
  gc.collect()
  torch.cuda.empty_cache()
  
def set_embedding_reduction_from_string(pooling_embedding_reduction):
  if pooling_embedding_reduction.lower() == 'spatial':
    return EMBEDDING_REDUCTION.MEAN_SPATIAL
  elif pooling_embedding_reduction.lower() == 'temporal':
    return EMBEDDING_REDUCTION.MEAN_TEMPORAL
  elif pooling_embedding_reduction.lower() == 'all':
    return EMBEDDING_REDUCTION.MEAN_TEMPORAL_SPATIAL
  elif pooling_embedding_reduction.lower() == 'none':
    return EMBEDDING_REDUCTION.NONE
  else:
    raise ValueError(f'Pooling embedding reduction not recognized: {pooling_embedding_reduction}. Can be spatial, temporal, all or none')
  
def generate_path(path):
  return os.path.join(GLOBAL_PATH.NAS_PATH,path)

if __name__ == "__main__":
  print('Setting sharing strategy')
  torch.multiprocessing.set_sharing_strategy('file_system')
  
  timestamp = int(time.time())
  parser = argparse.ArgumentParser(description='Extract features from video dataset.')
  parser.add_argument('--gp', action='store_true', help='Global path')
  parser.add_argument('--model_type', type=str, required=False, default="B")
  parser.add_argument('--saving_after', type=int, required=False, default=150,help='Number of batch to save in one file')
  parser.add_argument('--emb_red', type=str, default='spatial', help='Embedding reduction. Can be spatial, temporal, all,none')
  parser.add_argument('--prep_al', action='store_true', help='Preprocess align') # not use
  parser.add_argument('--prep_crop', action='store_true', help='Preprocess crop') # not use
  parser.add_argument('--prep_front', action='store_true', help='Preprocess frontalize') #not use
  parser.add_argument('--from_', type=int, default=None, help='START idx (included) extracting features from (--path_labels) row. Start from 0')
  parser.add_argument('--to_', type=int, default=None, help='STOP idx (excluded) extracting features from (--path_labels) row')
  parser.add_argument('--path_dataset', type=str, default=os.path.join('partA','video','video'), help='Path to dataset')
  parser.add_argument('--path_labels', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), help='Path to csv file')
  parser.add_argument('--saving_folder_path', type=str, default=os.path.join('partA','video','features',f'samples_16_{timestamp}'), help='Path to saving folder')
  parser.add_argument('--log_file_path', type=str, default=os.path.join('partA','video','features','log_file_feat_extr',f'log_file_{timestamp}.txt'), help='Path to log file')
  parser.add_argument('--backbone_type', type=str, default='video', help='Type of backbone. Can be video or image')
  parser.add_argument('--batch_size_feat_extraction', type=int, default=1, help='Batch size for feature extraction')
  parser.add_argument('--n_workers', type=int, default=1, help='Number of workers for dataloader')
  parser.add_argument('--save_big_feature', action='store_true', help='Save big feature')
  # CUDA_VISIBLE_DEVICES=0 python3 extract_feature.py --gp --model_type B --saving_after 150 --emb_red spatial --path_dataset partA/video/video_frontalized_new --path_labels partA/starting_point/samples_exc_no_detection.csv --saving_folder_path partA/video/features/samples_16_frontalized_new --backbone_type video --from_ 0 --to_ 1500 --batch_size_feat_extraction 5 --n_workers 5
  # prompt example: python3 extract_feature.py --gp --model_type B --saving_after 5000  --emb_red temporal  --path_dataset partA/video/video_frontalized --path_labels partA/starting_point/samples_exc_no_detection.csv --saving_folder_path partA/video/features/samples_vit_img --log_file_path partA/video/features/samples_vit_img/log_file.txt --backbone_type image 
  args = parser.parse_args()
  args.emb_red = set_embedding_reduction_from_string(args.emb_red)
  if args.from_ is not None or args.to_ is not None:
    args.saving_folder_path = f'{args.saving_folder_path}_{args.from_}_{args.to_}'
    print(f'Saving folder path: {args.saving_folder_path}')
  if args.gp:
    args.path_dataset = generate_path(args.path_dataset)
    args.path_labels = generate_path(args.path_labels)
    args.log_file_path = generate_path(args.log_file_path)
    args.saving_folder_path = generate_path(args.saving_folder_path)
    print(f'\n\npath_dataset: {args.path_dataset}\n')
    print(f'path_labels: {args.path_labels}')
    print(f'log_file_path: {args.log_file_path}')
    print(f'saving_folder_path: {args.saving_folder_path}')
  print(args)
  
  main(model_type=args.model_type,
       saving_chunk_size=args.saving_after,
       preprocess_align=args.prep_al,
       preprocess_crop_detection=args.prep_crop,
       preprocess_frontalize=args.prep_front,
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
       save_big_feature=args.save_big_feature
       )
  
  
  
  
  
  
  
  
  
  

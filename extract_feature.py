from custom.dataset import customDataset
from custom.backbone import backbone
from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY, HEAD, GLOBAL_PATH
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from transformers import AutoImageProcessor
import custom.tools as tools
import time
import pickle
import gc
import argparse

def main(model_type,saving_chunk_size=100,  preprocess_align = False,
         preprocess_crop_detection = False,preprocess_frontalize = True,path_dataset=None,path_labels=None,
         log_file_path=None,root_saving_folder_path=None
         ):
  if model_type == 'B':
    model_type = MODEL_TYPE.VIDEOMAE_v2_B
  elif model_type == 'S':
    model_type = MODEL_TYPE.VIDEOMAE_v2_S
  else:
    raise ValueError('Model type not recognized. Please use "B" or "S"')
    
  pooling_embedding_reduction = EMBEDDING_REDUCTION.MEAN_SPATIAL
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  
  def _write_log_file(log_message):
    if not os.path.exists(os.path.dirname(log_file_path)):
      os.makedirs(os.path.dirname(log_file_path))
    with open(log_file_path,'a') as f:
      f.write(log_message+'\n')

  def _extract_features(dataset,path_csv_dataset,batch_size_feat_extraction,backbone):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"extracting features using.... {device}")
    list_features = []
    list_labels = []
    list_subject_id = []
    list_sample_id = []
    list_path = []
    list_frames = []
    count = 0
    dataset.set_path_labels(path_csv_dataset)
    dataloader = DataLoader(dataset, 
                batch_size=batch_size_feat_extraction,
                shuffle=False,
                collate_fn=dataset._custom_collate_fn)
    backbone.model.to(device)
    backbone.model.eval()
    count = 0
    start = time.time()
    with torch.no_grad():
      for data, labels, subject_id,sample_id, path, list_sampled_frames in dataloader:
        data = data.to(device)
        with torch.no_grad():
          feature = backbone.forward_features(x=data) # [1,8,14,14,768]
        feature = torch.mean(feature,dim=pooling_embedding_reduction.value,keepdim=True)
        print(f'feature shape {feature.shape}')
        list_frames.append(list_sampled_frames)
        list_features.append(feature.detach().cpu())
        list_labels.append(labels)
        list_sample_id.append(sample_id)
        list_subject_id.append(subject_id)
        list_path.append(path)
        count += 1
        print(f'\nBatch {count}/{len(dataloader)}')
        print(f'GPU:\n Free : {torch.cuda.mem_get_info()[0]/1024/1024/1024:.2f} GB \n total: {torch.cuda.mem_get_info()[1]/1024/1024/1024:.2f} GB\n')
        del data, feature
        torch.cuda.empty_cache()
        end = time.time()
        print(f'Elapsed time: {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s')
        expected_end = (end - start) * (len(dataloader) / count)
        print(f'Expected time: {expected_end//60//60:.0f} h {expected_end//60%60:.0f} m {expected_end%60:.0f} s')
        if count % 10 == 0:
          _write_log_file(f'Batch {count}/{len(dataloader)}')
        if count % saving_chunk_size == 0:
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
                    saving_folder_path=os.path.join(root_saving_folder_path,'batch_'+str(count-saving_chunk_size)+'_'+str(count)))
          _write_log_file(f'Batch {count-saving_chunk_size}_{count} saved in {os.path.join(root_saving_folder_path,"batch_"+str(count-saving_chunk_size)+"_"+str(count))} with size {dict_data_size:.2f} MB \n time elapsed: {((end - start)//60//60):.0f} h {(((end - start)//60%60)):.0f} m {(((end - start)%60)):.0f} s\n')
          list_features = []
          list_labels = []
          list_subject_id = []
          list_sample_id = []
          list_path = []
          list_frames = []
          del dict_data
    backbone.model.to('cpu')
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
  
  custom_ds = customDataset(path_dataset=path_dataset,
                path_labels=path_labels,
                sample_frame_strategy=sample_frame_strategy,
                stride_window=16,
                clip_length=16,
                preprocess_align=preprocess_align,
                preprocess_frontalize=preprocess_frontalize,
                preprocess_crop_detection=preprocess_crop_detection,
                saving_folder_path_extracted_video=None)
  
  backbone_model = backbone(model_type=model_type)
  config_dict = {
    'path_dataset': path_dataset,
    'path_labels': path_labels,
    'model_type': model_type,
    'pooling_embedding_reduction': pooling_embedding_reduction,
    'pooling_clips_reduction': pooling_clips_reduction,
    'sample_frame_strategy': sample_frame_strategy,
    'stride_window': 16,
    'clip_length': 16,
    'preprocess_align': preprocess_align,
    'preprocess_frontalize': preprocess_frontalize,
    'preprocess_crop_detection': preprocess_crop_detection,
    'batch_size_feat_extraction': 1,
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
                  path_csv_dataset=path_labels,
                  batch_size_feat_extraction=1,
                  backbone=backbone_model)
  gc.collect()
  torch.cuda.empty_cache()
  
def generate_path(path):
  return os.path.join(GLOBAL_PATH.NAS_PATH,path)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Extract features from video dataset.')
  parser.add_argument('--gp', action='store_true', help='Global path')
  parser.add_argument('--model_type', type=str, required=False, default="B")
  parser.add_argument('--saving_after', type=int, required=False, default=100,help='Number of batch to save in one file')
  parser.add_argument('--prep_al', action='store_true', help='Preprocess align')
  parser.add_argument('--prep_crop', action='store_true', help='Preprocess crop')
  parser.add_argument('--prep_front', action='store_true', help='Preprocess frontalize')
  parser.add_argument('--path_dataset', type=str, default=os.path.join('partA','video','video'), help='Path to dataset')
  parser.add_argument('--path_labels', type=str, default=os.path.join('partA','starting_point','samples_exc_no_detection.csv'), help='Path to csv file')
  parser.add_argument('--saving_folder_path', type=str, default=os.path.join('partA','video','features','samples_16_cropped_aligned'), help='Path to saving folder')
  parser.add_argument('--log_file_path', type=str, default=os.path.join('partA','video','features','samples_16_cropped_aligned','log_file.txt'), help='Path to log file')
  # prompt example complete: python3 extract_feature.py --csv_idx 1 --gp --model_type B --saving_after 100 --prep_al --prep_crop --prep_front --path_dataset partA/video/video --path_labels partA/starting_point/samples_exc_no_detection.csv --log_file_path partA/video/features/samples_16_cropped_aligned/log_file.txt --saving_folder_path partA/video/features/samples_16_cropped_aligned
  args = parser.parse_args()
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
       root_saving_folder_path=args.saving_folder_path,
       )
  
  
  
  
  
  
  
  
  
  

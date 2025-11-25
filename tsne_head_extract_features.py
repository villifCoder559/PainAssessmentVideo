import custom.head as head
import custom.dataset as dataset
import os
import pickle
import custom.tools as tools
import torch
import tqdm
import custom.helper as helper
import argparse
import time
from pathlib import Path
import pandas as pd
import shutil

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the model checkpoint')
  parser.add_argument('--csv_path', type=str, default=None,
                      help='Path to the CSV file for logging cross-attention / embeddings')
  parser.add_argument('--root_folder_features', type=str, required=True,
                      help='Path to the root folder of the features')
  parser.add_argument('--nr_samples', type=int, default=None,
                      help='Number of samples to use from the CSV file (for quick testing)')
  parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for the dataloader')
  parser.add_argument('--n_workers', type=int, default=4,
                      help='Number of workers for the dataloader')
  dict_args = vars(parser.parse_args())

  model_path = dict_args['model_path']
  root_folder_features = dict_args['root_folder_features']

  # Load config (k_fold_results.pkl sits 4 levels above checkpoint)
  config_path = "" if model_path[0] != '/' else "/"
  config_path += os.path.join(*(model_path.split(os.sep)[:-4]),"k_fold_results.pkl")

  with open(config_path, 'rb') as f:
    data = pickle.load(f)

  uid = int(time.time())

  # Output folder
  out_folder_name = f'tsne_features_{uid}'
  folder_out = os.path.join(
    *Path(model_path).parts[:-1],
    out_folder_name
  )
  os.makedirs(folder_out, exist_ok=True)

  # Model instantiation
  head_params = data['config']['head_params']
  att_head = head.AttentiveHeadJEPA(embed_dim=head_params['input_dim'],
                                num_classes=head_params['num_classes'],
                                num_heads=head_params['num_heads'],
                                num_cross_heads=head_params['num_cross_heads'],
                                dropout=head_params['dropout'],
                                attn_dropout=head_params['attn_dropout'],
                                residual_dropout=head_params['residual_dropout'],
                                mlp_ratio=head_params['mlp_ratio'],
                                pos_enc=head_params['pos_enc'],
                                grid_size_pos=head_params['T_S_S_shape'], # [T, S, S]
                                depth=head_params['depth'],
                                num_queries=head_params['num_queries'],
                                agg_method=head_params['agg_method'],
                                use_sdpa=True,
                                coral_loss=head_params['coral_loss'],
                                complete_block=head_params['complete_block'],
                                cross_block_after_transformers=head_params['cross_block_after_transformers'])

  att_head.load_state_dict(torch.load(model_path))

  # Decide test CSV
  if dict_args['csv_path'] is not None:
    test_csv_path = dict_args['csv_path']
  else:
    test_csv_path = os.path.join(*Path(model_path).parts[:-2], 'test.csv')

  # Optionally create subset CSV
  if dict_args['nr_samples'] is not None:
    df = pd.read_csv(test_csv_path, sep='\t', dtype={'sample_name': str})
    df = df.iloc[:dict_args['nr_samples']]
    subset_csv_path = os.path.join(folder_out, f'csv_subset_{dict_args["nr_samples"]}.csv')
    df.to_csv(subset_csv_path, index=False, sep='\t')
    test_csv_path = subset_csv_path
    print(f'Created subset CSV at {test_csv_path}')

  csv_name = Path(test_csv_path).stem
  print(f"CSV path: {test_csv_path}")
  
  dataset_, loader_ = dataset.get_dataset_and_loader(batch_size=dict_args['batch_size'],
                                              csv_path=test_csv_path,
                                              root_folder_features=root_folder_features,
                                              shuffle_training_batch=False,
                                              is_training=False,
                                              xattn_mask = None,
                                              concatenate_temporal=False,
                                              dataset_type=tools.get_dataset_type(root_folder_features),
                                              prefetch_factor=2,
                                              backbone_dict=None,
                                              model=att_head, 
                                              is_coral_loss=False,
                                              
                                              soft_labels=0,
                                              label_smooth=0,
                                              n_workers=dict_args['n_workers'])

  helper.dict_data = tools.load_dict_data(root_folder_features)
  feature_list = []
  y_list = []
  subject_list = []
  sample_id_list = []
  device = 'cuda'
  att_head.to(device)
  att_head.eval()

  with torch.no_grad():
    for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(loader_,total=len(loader_),desc=f'Feature extraction'):
      dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
      batch_y = batch_y.to(device)
      dict_batch_X['list_sample_id'] = sample_id
      outputs,preds = att_head(**dict_batch_X,return_video_emb=True)
      if preds.shape[1] == 1: # if regression I don't need to keep dim 1 
        preds = preds.squeeze(1)
      feature_list.append(outputs.detach().cpu())
      y_list.append(batch_y.detach().cpu())
      subject_list.append(batch_subjects.detach().cpu())
      sample_id_list.append(sample_id.detach().cpu())

  dict_data = {
    'features': torch.cat(feature_list, dim=0),
    'labels': torch.cat(y_list, dim=0),
    'subjects': torch.cat(subject_list, dim=0),
    'sample_ids': torch.cat(sample_id_list, dim=0)
  }

  # Copy the CSV file to the output folder (avoid overwriting)
  if os.path.exists(os.path.join(folder_out, f'{csv_name}.csv')):
    csv_name = f'{csv_name}_{int(time.time())}'
  csv_copy_path = os.path.join(folder_out, f'{csv_name}.csv')
  shutil.copy(test_csv_path, csv_copy_path)

  out_path = os.path.join(folder_out, f'{Path(test_csv_path).stem}_tsne_features_{uid}.pkl')
  config_logging = {
    'model_path': model_path,
    'out_path': out_path,
    'config_path': config_path,
    'config_model': data,
    'csv_original_path': test_csv_path,
    'csv_path': csv_copy_path,
    'uid': uid,
    'nr_samples': dict_args['nr_samples']
  }

  dict_res = {
    'features': dict_data,
    'config_logging': config_logging
  }

  with open(out_path, 'wb') as f:
    pickle.dump(dict_res, f)
    print(f'Saved results to {out_path}')
  
  config_txt_path = os.path.join(folder_out, 'config_logging.txt')
  with open(config_txt_path, 'w') as f:
    for k, v in config_logging.items():
      f.write(f'{k}: {v}\n')
    print(f'Saved config logging to {config_txt_path}')

if __name__ == '__main__':
  main()
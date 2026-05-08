#!/usr/bin/env python3
import argparse
import os
import pickle
import re
import time
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import tqdm
import custom.helper as helper
from custom.model import Model_Advanced


def _head_supports_cross_attention(config_model):
  """
  Return True if the model head produces cross-attention weights (AttentiveHeadJEPA),
  False for heads that do not (e.g. GRU).

  Args:
    config_model: dict loaded from k_fold_results.pkl.

  Returns:
    bool — True when cross-attention logging is meaningful.
  """
  head = str(config_model.get('model_advanced_params', {}).get('head', '')).upper()
  return 'GRU' not in head


def clean_csv_from_augmentations(csv_path):
  df = pd.read_csv(csv_path, sep='\t', dtype={'sample_name': str})
  print(f'Remove augmentations with threshold step shift: {helper.step_shift}')
  mask_to_keep = df['sample_id'] <= helper.step_shift
  df_clean = df[mask_to_keep]
  cleaned_csv_path = csv_path.replace('.csv', '_cleaned.csv')
  df_clean.to_csv(cleaned_csv_path, index=False, sep='\t')
  return cleaned_csv_path
  
def log_cross_attention_from_model(model_pth_path, split_chunks=0, csv_path=None, nr_samples=None, free_space=False,
                                   disable_video_embeddings=False, disable_cross_attention=False, slim=True,
                                   ffsp_override=None):
  # Initialize logging flags
  helper.init_log_cross_attention()
  helper.init_log_video_embeddings()

  # Load config (k_fold_results.pkl sits 4 levels above checkpoint)
  config_model_path = os.path.join(*Path(model_pth_path).parts[:-4], 'k_fold_results.pkl')
  with open(config_model_path, 'rb') as f:
    config_model = pickle.load(f)

  # Auto-disable cross-attention for heads that don't produce it (e.g. GRU)
  if not _head_supports_cross_attention(config_model):
    print(f'[auto-detect] Head "{config_model["model_advanced_params"].get("head")}" does not produce cross-attention — disabling cross-attention logging.')
    disable_cross_attention = True

  uid = int(time.time())

  # Output folder
  out_folder_name = f'{"cross_attention" if not disable_cross_attention else ""}{"_" if not disable_cross_attention and not disable_video_embeddings else ""}{"video_embeddings" if not disable_video_embeddings else ""}_{uid}_{"split_chunks" if split_chunks>0 else ""}'
  folder_out = os.path.join(
    *Path(model_pth_path).parts[:-1],
    out_folder_name
  )
  os.makedirs(folder_out, exist_ok=True)

  # Model instantiation
  model_advanced_params = config_model['model_advanced_params']
  model_advanced_params['head_params']['skip_init_weights'] = True  # Skip head weight initialization to avoid loading issues
  if ffsp_override is not None:
    model_advanced_params['features_folder_saving_path'] = ffsp_override
    print(f'***Overriding features_folder_saving_path to {ffsp_override}***')
  model = Model_Advanced(**model_advanced_params)

  # Decide test CSV
  if csv_path is not None:
    test_csv_path = csv_path
  else:
    test_csv_path = os.path.join(*Path(model_pth_path).parts[:-2], 'test.csv')

  # Remove augmentations from CSV if present
  test_csv_path = clean_csv_from_augmentations(test_csv_path) 
  
  # Optionally create subset CSV
  if nr_samples is not None:
    df = pd.read_csv(test_csv_path, sep='\t', dtype={'sample_name': str})
    df = df.iloc[:nr_samples]
    subset_csv_path = os.path.join(folder_out, f'csv_subset_{nr_samples}.csv')
    df.to_csv(subset_csv_path, index=False, sep='\t')
    test_csv_path = subset_csv_path
    print(f'Created subset CSV at {test_csv_path}')

  csv_name = Path(test_csv_path).stem

  # Configure helper logging flags
  helper.LOG_HISTORY_SAMPLE = True
  helper.LOG_VIDEO_EMBEDDINGS['enable'] = not disable_video_embeddings
  helper.LOG_CROSS_ATTENTION['enable'] = not disable_cross_attention

  # Prepare arguments for model test
  kwargs = config_model['config']
  test_pretarined_args = {
    'path_model_weights': model_pth_path,
    'state_dict': None,
    'csv_path': test_csv_path ,
    'criterion': config_model['config']['criterion'],
    'is_test': True,
    'concatenate_temporal': config_model['config']['concatenate_temp_dim'],
    'concatenate_quadrants': config_model['config']['concatenate_quadrants'],
    'CCC_loss': config_model['config']['CCC_loss']
  }
  kwargs = {k: v for k, v in kwargs.items() if k not in test_pretarined_args.keys()}
  kwargs['split_chunks'] = split_chunks

  # Run test (single run covers both logs)
  results = model.test_pretrained_model(**test_pretarined_args, **kwargs)

  # Collect logs if enabled
  video_embeddings = helper.LOG_VIDEO_EMBEDDINGS if helper.LOG_VIDEO_EMBEDDINGS.get('enable', False) else None
  cross_attention = helper.LOG_CROSS_ATTENTION if helper.LOG_CROSS_ATTENTION.get('enable', False) else None

  # Copy the CSV file to the output folder (avoid overwriting)
  if os.path.exists(os.path.join(folder_out, f'{csv_name}.csv')):
    csv_name = f'{csv_name}_{int(time.time())}'
  csv_copy_path = os.path.join(folder_out, f'{csv_name}.csv')
  shutil.copy(test_csv_path, csv_copy_path)

  # Prepare output path and logging info
  out_path = os.path.join(folder_out, f'{Path(test_csv_path).stem}_xattn_embeds_{uid}.pkl')
  config_logging = {
    'model_pth_path': model_pth_path,
    'out_path': out_path,
    'config_model_path': config_model_path,
    'config_model': config_model,
    'csv_original_path': test_csv_path,
    'csv_path': csv_copy_path,
    'uid': uid,
    'split_chunks': split_chunks,
    'video_embeddings_enabled': helper.LOG_VIDEO_EMBEDDINGS.get('enable', False),
    'cross_attention_enabled': helper.LOG_CROSS_ATTENTION.get('enable', False),
    'nr_samples': nr_samples
  }

  # Audit log: strip config_model entirely — config_model_path already points to the full
  # k_fold_results.pkl, so consumers can reload it if needed. Always applied, independent of --slim.
  config_logging_slim = {k: v for k, v in config_logging.items() if k != 'config_model'}

  # Aggregate results
  dict_res = {
    'results': results,
    # 'config_model': config_model,
    'csv_original_path': test_csv_path,
    'csv_path': csv_copy_path,
    'config_logging': config_logging_slim,  # nested key so consumers can access it reliably
    **config_logging
  }
  if slim:
    # Strip k-fold training history (~250 MB) — no consumer script reads config_model['results'].
    dict_res['config_model'] = {k: v for k, v in dict_res['config_model'].items() if k != 'results'}
  if free_space:
    dict_res['config_model'] = None

  if video_embeddings is not None:
    if slim:
      # Flatten list-of-batch-tensors into a single float32 numpy array for compact storage.
      # Consumers handle both formats via the _load_embeddings() compat shim.
      flat_embeddings = [
        desc.cpu().numpy()
        for batch_list in video_embeddings['embeddings']
        for desc in batch_list
      ]
      video_embeddings = {
        **video_embeddings,
        'embeddings': np.array(flat_embeddings, dtype=np.float32),
        'pkl_format_version': 2,
      }
    dict_res['video_embeddings'] = video_embeddings
  if cross_attention is not None:
    dict_res['cross_attention'] = cross_attention

  # Save results
  pkl_bytes = pickle.dumps(dict_res)
  print(f'Saving pkl ({len(pkl_bytes) / 1e6:.1f} MB) to {out_path}')
  with open(out_path, 'wb') as f:
    f.write(pkl_bytes)
    print(f'Saved results to {out_path}')

  # Save config logging (pickle + human readable) — always uses slim version (no config_model)
  config_pickle_path = os.path.join(folder_out, 'config_logging.pkl')
  with open(config_pickle_path, 'wb') as f:
    pickle.dump(config_logging_slim, f)
    print(f'Saved config logging to {config_pickle_path}')

  config_txt_path = os.path.join(folder_out, 'config_logging.txt')
  with open(config_txt_path, 'w') as f:
    for k, v in config_logging_slim.items():
      f.write(f'{k}: {v}\n')
    print(f'Saved config logging to {config_txt_path}')
  return dict_res


def get_all_pth_files_in_folders(dict_args):
  """
  Collect all .pth/.pt files under model_pth_path restricted to k{i}_cross_val_sub_{j} folders.

  Args:
    dict_args: CLI argument dict. When dict_args['idx_pth_folders'] == ['all'],
               every k{i}_cross_val_sub_{j} folder in the tree is searched;
               otherwise only the explicitly listed i,j pairs are searched.

  Returns:
    list of str — absolute paths to found checkpoint files.
  """
  list_pth_files = []
  if dict_args['idx_pth_folders'] == ['all']:
    pattern = re.compile(r'k\d+_cross_val_sub_\d+')
    for root, dirs, files in os.walk(dict_args['model_pth_path']):
      if pattern.search(root):
        for file in files:
          if file.endswith(".pth") or file.endswith(".pt"):
            list_pth_files.append(os.path.join(root, file))
  else:
    folders_to_search = [f'k{i}_cross_val_sub_{j}' for idxs in dict_args['idx_pth_folders'] for i,j in [map(int, idxs.split(','))]]
    for root, dirs, files in os.walk(dict_args['model_pth_path']):
      if any(folder in root for folder in folders_to_search):
        for file in files:
          if file.endswith(".pth") or file.endswith(".pt"):
            list_pth_files.append(os.path.join(root, file))
  print(f"Found {len(list_pth_files)} .pth files.")
  return list_pth_files


def get_csv_paths_for_pth(model_pth_path):
  train_path = os.path.join(*Path(model_pth_path).parts[:-1], 'train.csv')
  test_path = os.path.join(*Path(model_pth_path).parts[:-2], 'test.csv')
  if not os.path.exists(test_path):
    test_path = os.path.join(*Path(model_pth_path).parts[:-1], 'val.csv')
  assert os.path.isfile(test_path), f"Test CSV file {test_path} does not exist."
  assert os.path.isfile(train_path), f"Train CSV file {train_path} does not exist."
  # assert os.path.isfile(val_path), f"Val CSV file {val_path} does not exist."
  return [train_path,  test_path]

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_pth_path', type=str, required=True,
                      help='Path to the model checkpoint')
  parser.add_argument('--split_chunks', type=int, default=0,
                      help='Number of chunks to split the input video (0 means no split)')
  parser.add_argument('--csv_path', type=str, default=None,
                      help='Path to the CSV file for logging cross-attention / embeddings')
  parser.add_argument('--nr_samples', type=int, default=None,
                      help='Number of samples to use from the CSV file (for quick testing)')
  parser.add_argument('--disable_video_embeddings', action='store_true',
                      help='Disable logging video embeddings (enabled by default)')
  parser.add_argument('--disable_cross_attention', action='store_true',
                      help='Disable logging cross-attention (enabled by default)')
  parser.add_argument('--idx_pth_folders', nargs='*', type=str, default=None,
                      help='List of folder paths containing model .pth files to process.'+
                      ' Ex: 0,0 2,3. They means k0_cross_val_sub_0 and k2_cross_val_sub_3')
  parser.add_argument('--no_slim', action='store_true',
                      help='Disable pkl slimming (by default, config_model[results] and raw '
                           'tensor embeddings are stripped to reduce file size ~250 MB).')
  parser.add_argument('--features_folder_saving_path', type=str, default=None,
                      help='Override features_folder_saving_path from the stored config. '
                           'Use when the stored path is wrong (e.g. moved features, different machine, wrong backbone path).')
  dict_args = vars(parser.parse_args())
  
  # MULTIPLE .pth files in specified folders
  if dict_args['idx_pth_folders'] is not None:
    assert os.path.isdir(dict_args['model_pth_path']), "When using --idx_pth_folders, --model_pth_path must be a directory."
    list_pth_files = get_all_pth_files_in_folders(dict_args)
    
    # Process each .pth file found to generate logs
    for pth_path in tqdm.tqdm(list_pth_files, desc="Processing .pth files..."):
      dict_args['model_pth_path'] = pth_path      
      if dict_args['csv_path'] is None:
        csv_paths = get_csv_paths_for_pth(pth_path)
      else:
        csv_paths = [dict_args['csv_path']]
      for csv_path in csv_paths:
        assert os.path.isfile(csv_path), f"CSV file {csv_path} does not exist."
        log_cross_attention_from_model(csv_path=csv_path,
                                    disable_video_embeddings=dict_args['disable_video_embeddings'],
                                    disable_cross_attention=dict_args['disable_cross_attention'],
                                    model_pth_path=dict_args['model_pth_path'],
                                    split_chunks=dict_args['split_chunks'],
                                    nr_samples=dict_args['nr_samples'],
                                    slim=not dict_args['no_slim'],
                                    ffsp_override=dict_args.get('features_folder_saving_path'))
  else:
  # SINGLE .pth file
    if dict_args['csv_path'] is None:
      csv_paths = get_csv_paths_for_pth(dict_args['model_pth_path'])
    else:
      csv_paths = [dict_args['csv_path']]

    for csv_path in csv_paths:
      log_cross_attention_from_model(csv_path=csv_path,
                                    disable_video_embeddings=dict_args['disable_video_embeddings'],
                                    disable_cross_attention=dict_args['disable_cross_attention'],
                                    model_pth_path=dict_args['model_pth_path'],
                                    split_chunks=dict_args['split_chunks'],
                                    nr_samples=dict_args['nr_samples'],
                                    slim=not dict_args['no_slim'],
                                    ffsp_override=dict_args.get('features_folder_saving_path'))
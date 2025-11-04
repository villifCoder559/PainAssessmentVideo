#!/usr/bin/env python3
import argparse
import os
import pickle
import time
from pathlib import Path
import shutil
import pandas as pd

import custom.helper as helper
from custom.model import Model_Advanced

def main():
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
  dict_args = vars(parser.parse_args())

  model_pth_path = dict_args['model_pth_path']

  # Load config (k_fold_results.pkl sits 4 levels above checkpoint)
  config_model_path = os.path.join(*Path(model_pth_path).parts[:-4], 'k_fold_results.pkl')
  with open(config_model_path, 'rb') as f:
    config_model = pickle.load(f)

  uid = int(time.time())

  # Output folder
  out_folder_name = f'{"cross_attention" if not dict_args["disable_cross_attention"] else ""}{"_" if not dict_args["disable_cross_attention"] and not dict_args["disable_video_embeddings"] else ""}{"video_embeddings" if not dict_args["disable_video_embeddings"] else ""}_{uid}_{"split_chunks" if dict_args["split_chunks"]>0 else ""}'
  folder_out = os.path.join(
    *Path(model_pth_path).parts[:-1],
    out_folder_name
  )
  os.makedirs(folder_out, exist_ok=True)

  # Model instantiation
  model_advanced_params = config_model['model_advanced_params']
  model = Model_Advanced(**model_advanced_params)

  # Decide test CSV
  if dict_args['csv_path'] is not None:
    test_csv_path = dict_args['csv_path']
  else:
    test_csv_path = os.path.join(*Path(model_pth_path).parts[:-2], 'test.csv')

  # Optionally create subset CSV
  if dict_args['nr_samples'] is not None:
    df = pd.read_csv(test_csv_path, sep='\t', dtype={'sample_name': str})
    df = df.iloc[:dict_args['nr_samples']]
    subset_csv_path = os.path.join(folder_out, f'csv_subset_{dict_args["nr_samples"]}.csv')
    df.to_csv(subset_csv_path, index=False, sep='\t')
    test_csv_path = subset_csv_path
    print(f'Created subset CSV at {test_csv_path}')

  csv_name = Path(test_csv_path).stem

  # Configure helper logging flags
  helper.LOG_HISTORY_SAMPLE = True
  helper.LOG_VIDEO_EMBEDDINGS['enable'] = not dict_args['disable_video_embeddings']
  helper.LOG_CROSS_ATTENTION['enable'] = not dict_args['disable_cross_attention']

  # Prepare arguments for model test
  kwargs = config_model['config']
  test_pretarined_args = {
    'path_model_weights': model_pth_path,
    'state_dict': None,
    'csv_path': test_csv_path,
    'criterion': config_model['config']['criterion'],
    'is_test': True,
    'concatenate_temporal': config_model['config']['concatenate_temp_dim'],
    'concatenate_quadrants': config_model['config']['concatenate_quadrants'],
    'CCC_loss': config_model['config']['CCC_loss']
  }
  kwargs = {k: v for k, v in kwargs.items() if k not in test_pretarined_args.keys()}
  kwargs['split_chunks'] = dict_args['split_chunks']

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
    'split_chunks': dict_args['split_chunks'],
    'video_embeddings_enabled': helper.LOG_VIDEO_EMBEDDINGS.get('enable', False),
    'cross_attention_enabled': helper.LOG_CROSS_ATTENTION.get('enable', False),
    'nr_samples': dict_args['nr_samples']
  }

  # Aggregate results
  dict_res = {
    'results': results,
    'config_model': config_model,
    'csv_original_path': test_csv_path,
    'csv_path': csv_copy_path,
    'config_logging': config_logging
  }
  if video_embeddings is not None:
    dict_res['video_embeddings'] = video_embeddings
  if cross_attention is not None:
    dict_res['cross_attention'] = cross_attention

  # Save results
  with open(out_path, 'wb') as f:
    pickle.dump(dict_res, f)
    print(f'Saved results to {out_path}')

  # Save config logging (pickle + human readable)
  config_pickle_path = os.path.join(folder_out, 'config_logging.pkl')
  with open(config_pickle_path, 'wb') as f:
    pickle.dump(config_logging, f)
    print(f'Saved config logging to {config_pickle_path}')

  config_txt_path = os.path.join(folder_out, 'config_logging.txt')
  with open(config_txt_path, 'w') as f:
    for k, v in config_logging.items():
      f.write(f'{k}: {v}\n')
    print(f'Saved config logging to {config_txt_path}')

if __name__ == '__main__':
  main()

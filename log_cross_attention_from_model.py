import custom.helper as helper
from custom.model import Model_Advanced
import argparse
import os
import pickle
import time
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--model_pth_path', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('--csv_path', type=str, default=None, help='Path to the CSV file for logging cross-attention')
parser.add_argument('--nr_samples', type=int, default=None, help='Number of samples to use from the CSV file (for quick testing)')
dict_args = vars(parser.parse_args())

# dict_args = {
#     'model_pth_path': 'TRAIN_tests_CAER/history_run_filtered_origCAER_424784_ATTENTIVE_JEPA_ultron_1758736909/1758736914349_VIDEOMAE_v2_S_NONE_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/train_ATTENTIVE_JEPA/k0_cross_val/k0_cross_val_sub_0/best_model_ep_28.pth'
# }

model_pth_path = dict_args['model_pth_path']

config_model = os.path.join(*Path(model_pth_path).parts[:-4], 'k_fold_results.pkl')
with open(config_model, 'rb') as f:
  config_model = pickle.load(f)

uid = int(time.time())
folder_out = os.path.join(*Path(model_pth_path).parts[:-1],f'cross_attention_{uid}')
os.makedirs(folder_out, exist_ok=True)
config_model['model_advanced_params']['n_workers'] = 4  
model_advanced_params = config_model['model_advanced_params']
# model_advanced_params['new_csv_path'] = "TRAIN_tests_model/history_run_small_ATTN_high_drop_670666_ATTENTIVE_JEPA_targaryen_1758703044/1758703059114_VIDEOMAE_v2_S_NONE_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/augmented_samples.csv"
# model_advanced_params['batch_size_training'] = 1
model = Model_Advanced(**model_advanced_params)
if dict_args['csv_path'] is not None:
  test_csv_path = dict_args['csv_path']
else:
  test_csv_path = os.path.join(*Path(model_pth_path).parts[:-2], 'test.csv')

if dict_args['nr_samples'] is not None: 
  df = pd.read_csv(test_csv_path,sep='\t',dtype={'sample_name':str})
  df = df.iloc[:dict_args['nr_samples']]
  test_csv_path = os.path.join(folder_out, f'csv_subset.csv')
  df.to_csv(test_csv_path, index=False,sep='\t')
  print(f'Created subset CSV at {test_csv_path}')


csv_name = Path(test_csv_path).stem


helper.LOG_HISTORY_SAMPLE = True
helper.LOG_CROSS_ATTENTION['enable'] = True

results = model.test_pretrained_model(path_model_weights=model_pth_path,
                            state_dict=None,
                            csv_path=test_csv_path,
                            criterion=config_model['config']['criterion'],
                            is_test=True,
                            concatenate_temporal=config_model['config']['concatenate_temp_dim'],
                            concatenate_quadrants=config_model['config']['concatenate_quadrants'],
                            CCC_loss=config_model['config']['CCC_loss']
                            # **config_model['config']
                            )
xattn = helper.LOG_CROSS_ATTENTION
dict_res = {
  'results': results,
  'cross_attention': xattn,
  'config_model': config_model,
  'csv_path': test_csv_path,
}
os.makedirs(folder_out, exist_ok=True)
out_path = os.path.join(folder_out, f'{Path(test_csv_path).stem}_xattn_{uid}.pkl')
with open(out_path, 'wb') as f:
  pickle.dump(dict_res, f)
  print(f'Saved results to {out_path}')
  
config_logging = {
  'model_pth_path': model_pth_path,
  'out_path': out_path,
  'config_model': config_model,
  'test_csv_path': test_csv_path,
  'uid': uid,
}
with open(os.path.join(folder_out, f'config_logging.pkl'), 'wb') as f:
  pickle.dump(config_logging, f)
  print(f'Saved config logging to {os.path.join(folder_out, f"config_logging.pkl")}')

with open(os.path.join(folder_out, f'config_logging.txt'), 'w') as f:
  for k,v in config_logging.items():
    f.write(f'{k}: {v}\n')
  print(f'Saved config logging to {os.path.join(folder_out, f"config_logging.txt")}')

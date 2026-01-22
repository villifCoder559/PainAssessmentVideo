import argparse
import os
from pathlib import Path
import new_plot_tsne_post_head as tsne_plotter
import log_cross_attention_from_model as logger 
import pickle
import tqdm
import numpy as np

def get_pth_path_best_models(proj_dir):
  # Retrive pkl project path
  pkl_data_path = ""
  for root, dirs, files in os.walk(proj_dir):
    for file in files:
      if file == "k_fold_results.pkl":
        pkl_data_path = os.path.join(root, file)
        break
  if pkl_data_path == "":
    raise FileNotFoundError("k_fold_results.pkl not found in the project directory.")
  
  # Load pkl and get best model paths
  with open(pkl_data_path, 'rb') as f:
    data = pickle.load(f)
  final_folders = [key for key in data['results'].keys() if 'final' in key.lower()]
  keys_pth_models = [data['results'][final_key]['best_model']['fold_sub_fold_idx'] for final_key in final_folders]
  final_pth_paths = [f'k{i}_cross_val_sub_{j}' for i,j in keys_pth_models]
  
  # Retrive only best model .pth paths
  model_paths = []
  for root, dirs, files in os.walk(proj_dir):
    for file in files:
      if file.endswith(".pth") or file.endswith(".pt"):
        full_path = os.path.join(root, file)
        if any(final_pth in full_path for final_pth in final_pth_paths):
          model_paths.append(full_path)
  if not model_paths:
    raise FileNotFoundError("No .pth model files found in the project directory.")
  
  return model_paths
  
def get_list_csv_path(proj_dir,target):
  list_csv_path = []
  list_files = os.listdir(proj_dir)
  for file in list_files:
    if target in file and file.endswith('.csv'):
      list_csv_path.append(os.path.join(proj_dir, file))
  return list_csv_path


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--proj_dir', type=str, required=True, help='Path to the project directory')
  parser.add_argument('--type_logs', type=str, default='tsne', help='Type of logs to generate. Can be "tsne" or "xattn"')
  
  args = parser.parse_args()
  list_pth_paths= get_pth_path_best_models(args.proj_dir)
  
  # Iterate over each .pth model path and [test,train] csv files to plot t-SNE
  dict_logs = {}
  for pth_path in tqdm.tqdm(list_pth_paths, desc="Processing..."):
    list_csv_path = get_list_csv_path(Path(pth_path).parent, 'train') + get_list_csv_path(Path(pth_path).parent.parent, 'test')
    pth_folder = Path(pth_path).parts[-2]
    log_path_folder = os.path.join(args.proj_dir, 'logs_tsne', pth_folder)
    dict_logs[pth_folder] = {}
    os.makedirs(log_path_folder, exist_ok=True)
    for csv_path in list_csv_path:
      csv_basename = os.path.basename(csv_path).replace('.csv','')
      if args.type_logs == 'tsne':
        log_dict = logger.log_cross_attention_from_model(model_pth_path=pth_path,
                                                          csv_path=csv_path,
                                                          free_space=True,
                                                          disable_cross_attention=True)
        reduced_embeddings, valid_indices, valid_sample_ids = tsne_plotter.compute_valid_tsne_embeddings(log_dict, return_valid_indices=True)
        tsne_plotter.run_tsne_and_plot(pkl_file=log_dict,
                                        group_by='subjects',
                                        cmap='tab20',
                                        png_output_name=os.path.join(log_path_folder, f'{csv_basename}_tsne_plot_subjects.png'),
                                        log_path_folder=log_path_folder,
                                        reduced_embeddings=(reduced_embeddings, valid_indices))
        tsne_plotter.run_tsne_and_plot(pkl_file=log_dict,
                                        group_by='labels',
                                        cmap='jet',
                                        png_output_name=os.path.join(log_path_folder, f'{csv_basename}_tsne_plot_labels.png'),
                                        log_path_folder=log_path_folder,
                                        reduced_embeddings=(reduced_embeddings, valid_indices))
        dict_logs[pth_folder][csv_basename] = {
          'log_dict': log_dict,
          'tsne_embeddings': (reduced_embeddings, valid_sample_ids),
          'csv_path': csv_path,
          'pth_path': pth_path
        }
      elif args.type_logs == 'xattn':
        log_dict = logger.log_cross_attention_from_model(model_pth_path=pth_path,
                                                          csv_path=csv_path,
                                                          free_space=True,
                                                          disable_cross_attention=False,
                                                          disable_video_embeddings=True)
        dict_logs[pth_folder][csv_basename] = {
          'log_dict': log_dict,
          'csv_path': csv_path,
          'pth_path': pth_path
        }
  
  # Save overall log
  overall_log_path = os.path.join(args.proj_dir, f'logs_{args.type_logs}', f'{args.type_logs}_overall_logs.pkl')
  with open(overall_log_path, 'wb') as f:
    pickle.dump(dict_logs, f)
    print(f'Saved overall logs to {overall_log_path}')
      
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
import custom.tools as tools
from torchmetrics.classification import MulticlassConfusionMatrix
import tqdm
def find_results_files(parent_folder):
  results_files = []
  list_history_folder = os.listdir(parent_folder)
  for folder in list_history_folder:
    list_runs = os.listdir(os.path.join(parent_folder,folder))
    for run in list_runs:
      # check if os.path.join(parent_folder,folder,run) is a dir
      if os.path.isdir(os.path.join(parent_folder,folder,run)):
        pkl_files = [f for f in os.listdir(os.path.join(parent_folder,folder,run)) if f.endswith('.pkl')]
        if len(pkl_files) == 1:
          results_files.append(os.path.join(parent_folder,folder,run,pkl_files[0]))
      
  return results_files

def load_results(file_path):
  with open(file_path, 'rb') as f:
    return pickle.load(f)

def plot_losses(data, run_output_folder,test_id):
  os.makedirs(run_output_folder, exist_ok=True)

  for key in data.keys():
    if 'cross_val' in key and 'train_val' in key:
      train_losses = data[key].get('train_losses', [])
      val_losses = data[key].get('val_losses', [])

      if train_losses and val_losses:
        plt.figure()
        plt.ylim([0,5])
        plt.yticks(np.arange(0, 5, 0.25))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        data['config']['test_id'] = test_id
        dict_to_string = convert_dict_to_string(filter_dict(data['config']))
        # substitute head.params with GRU
        dict_to_string = dict_to_string.replace('head_params','GRU')
        plt.figtext(0.95,0.5,dict_to_string,ha='left', va='center',fontsize=12, color='black')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Losses - {key}')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(run_output_folder, f'losses_{key}.png'),bbox_inches='tight')
        plt.close()

def plot_accuracies(data, run_output_folder):
  os.makedirs(run_output_folder, exist_ok=True)

  for key in data.keys():
    if 'cross_val' in key and 'train_val' in key:
      train_acc = data[key].get('list_train_macro_accuracy', [])
      val_acc = data[key].get('list_val_macro_accuracy', [])

      if train_acc and val_acc:
        plt.figure()
        plt.ylim([0,1])
        plt.yticks(np.arange(0, 1, 0.1))
        plt.plot(train_acc, label='Train Macro Accuracy')
        plt.plot(val_acc, label='Val Macro Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Macro Accuracy')
        plt.title(f'Accuracy - {key}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_output_folder, f'accuracy_{key}.png'))
        plt.close()

def plot_test_metrics(data, run_output_folder):
    # test_folder = os.path.basename(os.path.dirname(file))
  os.makedirs(run_output_folder, exist_ok=True)

  test_losses = []
  macro_precisions = []
  labels = []

  for key in data.keys():
    if key.endswith('_test'):
      test_loss = data[key]['dict_test'].get('test_loss', None)
      macro_precision = data[key]['dict_test']['dict_precision_recall'].get('macro_precision', None)

      if test_loss is not None and macro_precision is not None:
        test_losses.append(test_loss)
        macro_precisions.append(macro_precision)
        labels.append(key)

  if test_losses and macro_precisions:
    x = np.arange(len(labels))
    width = 0.4

    plt.figure()
    plt.bar(x - width/2, test_losses, width, label='Test Loss')
    plt.bar(x + width/2, macro_precisions, width, label='Macro Precision')
    plt.ylim([0,2])
    plt.yticks(np.arange(0, 2, 0.2))
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Metric Value')
    plt.title('Test Loss and Macro Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_output_folder, 'test_metrics_histogram.png'))
    plt.close()

def convert_dict_to_string(d):
  new_d = flatten_dict(d)
  return '\n'.join([f'{k}: {v}' for k, v in new_d.items()])

def filter_dict(d):
  keys_to_exclude = ['model_type','epochs','pooling_embedding_reduction','pooling_clips_reduction','shuffle_video_chunks',
                     'sample_frame_strategy','head','stride_window_in_video','plot_dataset_distribution','clip_length','early_stopping']
  new_d = {k: v for k, v in d.items() if k not in keys_to_exclude}
  return new_d

# def get_value_from_dict(d, root_name=''):
#   result = []
#   for k, v in d.items():
#     if isinstance(v, dict):
#       # If the value is a dictionary, recurse with the updated root_name
#       result.append(get_value_from_dict(v, f'{root_name}.{k}' if root_name else k))
#     else:
#       # If it's not a dictionary, just append the key-value pair
#       result.append(f'{root_name}.{k}: {v}' if root_name else f'{k}: {v}')
#   return '\n'.join(result)

def flatten_dict(d, root_name=''):
  result = {}
  for k, v in d.items():
    if isinstance(v, dict):
      # If the value is a dictionary, recurse with the updated root_name
      result.update(flatten_dict(v, f'{root_name}.{k}' if root_name else k))
    else:
      # If it's not a dictionary, just append the key-value pair
      if root_name:
        result[f'{root_name}.{k}'] = v
      else:
        result[k] = v
  return result

def generate_csv_row(data, test_id):
    """
    Generates a dictionary containing test and training statistics from cross-validation experiments.
    """
    
    # Extract cross-validation settings
    k_fold = data['config']['k_fold']
    
    # Retrieve test statistics
    test_losses = {f'test_loss_k{i}': data[f'k{i}_test']['dict_test']['test_loss'] for i in range(k_fold)}
    test_accuracies = {f'test_accuracy_k{i}': data[f'k{i}_test']['dict_test']['test_macro_precision'].item() for i in range(k_fold)}
    val_accuracies = {}
    train_accuracies = {}
    for i in range(k_fold):
      best_model_subfolder_idx = data[f'k{i}_test']['best_model_subfolder_idx']
      val_accuracies[f'val_accuracy_k{i}'] = data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['list_val_macro_accuracy'][data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['best_model_idx']]
      train_accuracies[f'train_accuracy_k{i}'] = data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['list_train_macro_accuracy'][data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['best_model_idx']]
    best_sub_folders = [data[f'k{i}_test']['best_model_subfolder_idx'] for i in range(k_fold)]
    
    # Retrieve best epoch indices for each fold
    best_epochs = {
        f'best_epoch_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['best_model_idx']
        for i, sub in zip(range(k_fold), best_sub_folders)
    }
    
    # Retrieve training and validation losses at the best epoch
    train_losses = {
        f'train_loss_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['train_losses'][epoch]
        for i, sub, epoch in zip(range(k_fold), best_sub_folders, best_epochs.values())
    }
    
    val_losses = {
        f'val_loss_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['val_losses'][epoch]
        for i, sub, epoch in zip(range(k_fold), best_sub_folders, best_epochs.values())
    }
    
    # Extract model configuration parameters
    config = data['config']
    head_params = flatten_dict({'GRU':config['head_params']})
    row_dict = {
        'test_id': test_id,
        'model': config['model_type'].name,
        'optimizer': config['optimizer_fn'],
        'learning_rate': config['lr'],
        'criterion': type(config['criterion']).__name__,
        'init_network': config['init_network'],
        'reg_lambda': config['regularization_lambda'],
        'reg_loss': config['regularization_loss'],
        'feature_type': config['features_folder_saving_path'][-1], # is a path saved as a list
        'early_stopping_key': config['key_for_early_stopping']+f'(pat={config["early_stopping"].patience},eps={config["early_stopping"].min_delta},t_mod={config["early_stopping"].threshold_mode})',
        'target_metric': config['target_metric_best_model'],
        'round_output_loss': config['round_output_loss'],
        'batch_size_training': config['batch_size_training'],
        **head_params,
        **test_losses,
        **test_accuracies,
        'max_epochs': config['epochs'],
        **best_epochs,
        **train_losses,
        **val_losses,
        'mean_train_loss':np.mean(list(train_losses.values())),
        'mean_val_loss':np.mean(list(val_losses.values())),
        'mean_test_loss':np.mean(list(test_losses.values())),
        'mean_train_accuracy':np.mean(list(train_accuracies.values())),
        'mean_val_accuracy':np.mean(list(val_accuracies.values())),
        'mean_test_accuracy':np.mean(list(test_accuracies.values())),
        **train_accuracies,
        **val_accuracies,
    }
    
    return row_dict

def plot_confusion_matrices(data, root_output_folder):
  def plot_matrices(matrices,output_folder,stage):
    os.makedirs(output_folder, exist_ok=True)
    if isinstance(matrices, dict):
      for epoch,cnf in matrices.items():
        tools.plot_confusion_matrix(confusion_matrix=cnf,
                                    title=f'{stage}_epoch_{epoch}',
                                    saving_path=os.path.join(output_folder,f'{stage}_epoch_{epoch}.png'))
    elif isinstance(matrices,MulticlassConfusionMatrix):
      tools.plot_confusion_matrix(confusion_matrix=matrices,
                                  title='Confusion Matrix',
                                  saving_path=os.path.join(output_folder,f'{stage}.png'))
  
  for k,v in data.items():
    if 'cross_val' in k:
      dict_conf_matricies = v.get('train_confusion_matricies')
      plot_matrices(matrices=dict_conf_matricies,output_folder=os.path.join(root_output_folder,k),stage='train')
      dict_conf_matricies = v.get('val_confusion_matricies')
      plot_matrices(matrices=dict_conf_matricies,output_folder=os.path.join(root_output_folder,k),stage='val')
    elif 'test' in k:
      dict_conf_matricies = v['dict_test']['test_confusion_matrix']
      plot_matrices(matrices=dict_conf_matricies,output_folder=os.path.join(root_output_folder,k),stage=f'test_using_subfolder_{v["best_model_subfolder_idx"]}')
      
def plot_run_details(parent_folder, output_root):
  results_files = find_results_files(parent_folder)
  results_data = {file: load_results(file) for file in results_files}
  print(f'Loaded {len(results_data)} results files')

  list_row_csv = []
  for file,data in tqdm.tqdm(results_data.items()):
    test_folder = os.path.basename(os.path.dirname(file))
    test_id = test_folder.split('_')[0]       
    run_output_folder = os.path.join(output_root, test_folder)
    list_row_csv.append(generate_csv_row(data,test_id))
    # plot_losses(data, os.path.join(run_output_folder, 'loss_plots'),test_id)
    # plot_accuracies(data, os.path.join(run_output_folder, 'accuracy_plots'))
    # plot_test_metrics(data, os.path.join(run_output_folder, 'test_plots'))
    # plot_confusion_matrices(data, os.path.join(run_output_folder, 'confusion_matrices'))
    # print(f'Plots saved to {output_root}/{test_folder}')
  df = pd.DataFrame(list_row_csv)
  df = df.fillna('ND')
  df.to_csv(os.path.join(output_root,'summary.csv'),index=False,)

def collect_loss_plots(summary_folder, output_folder):
  """
  Collects all .png loss plots from various subdirectories under summary_folder
  and creates symbolic links in output_folder.

  :param summary_folder: Path to the root summary folder.
  :param output_folder: Path to the folder where symbolic links will be stored.
  """
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  # for root, folders, files in os.walk(summary_folder):
  folders = [f for f in os.listdir(summary_folder) if os.path.isdir(os.path.join(summary_folder, f))]
  for folder in folders:
    losses_png_path = os.path.join(summary_folder, folder, 'loss_plots')
    for _,_,files in os.walk(losses_png_path):
      for file in files:
        if file.endswith(".png"):
          src_path = os.path.join(losses_png_path, file)
          new_name = f"{folder.split('_')[0]}_{file}"  # Rename as foldername_lossname.png
          dst_path = os.path.join(output_folder, new_name)
          # Create a symbolic link (if supported) or copy the file
          try:
            if not os.path.exists(dst_path):  # Avoid overwriting existing links
              os.symlink(src_path, dst_path)
          except OSError:
            shutil.copy(src_path, dst_path)

  print(f"All loss plots collected in {output_folder}")

if __name__ == '__main__':
  parent_folder = '/media/villi/TOSHIBA EXT/test_24_02_18'  # Change this to the actual path
  output_root = '/media/villi/TOSHIBA EXT/test_24_02_18/new_summary'  # Change this to where you want the plots
  if not os.path.exists(output_root):
    os.makedirs(output_root)
  plot_run_details(parent_folder, os.path.join(output_root,'plot_per_run'))
  losses_folder = os.path.join(output_root,'all_loss_plots')
  summary_folder = os.path.join(output_root,'plot_per_run')
  collect_loss_plots(summary_folder, losses_folder)

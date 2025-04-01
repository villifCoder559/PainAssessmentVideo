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
from pathlib import Path
import argparse
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

def find_results_files(parent_folder):
  results_files = []
  list_history_folder = os.listdir(parent_folder)
  for folder in list_history_folder:
    if os.path.isdir(os.path.join(parent_folder, folder)):
      list_runs = os.listdir(os.path.join(parent_folder, folder))
      for run in list_runs:
        if os.path.isdir(os.path.join(parent_folder, folder, run)):
          pkl_files = [f for f in os.listdir(os.path.join(parent_folder, folder, run)) if f.endswith('.pkl')]
          if len(pkl_files) == 1:
            results_files.append(os.path.join(parent_folder, folder, run, pkl_files[0]))
  return results_files

def load_results(file_path):
  with open(file_path, 'rb') as f:
    return pickle.load(f)

def retrieve_subject_ids(data, key, best_epoch):
  # For retrocompatibility
  if data[key]['train_loss_per_subject'][best_epoch].shape != data[key]['train_unique_subject_ids'].shape:
    uniqie_subject_ids_train = data[key]['subject_ids_unique']
    uniqie_subject_ids_val = data[key]['subject_ids_unique']
  else:
    uniqie_subject_ids_train = data[key]['train_unique_subject_ids']
    uniqie_subject_ids_val = data[key]['val_unique_subject_ids']
  return uniqie_subject_ids_train, uniqie_subject_ids_val
def plot_train_val_loss(train_losses, val_losses, dict_to_string, output_path,key):
  plt.figure()
  plt.ylim([0, 5])
  plt.yticks(np.arange(0, 5, 0.25))
  plt.plot(train_losses, label='Train Loss',color='tab:red')
  plt.plot(val_losses, label='Val Loss',color='tab:orange')
  plt.figtext(0.95, 0.5, dict_to_string, ha='left', va='center', fontsize=12, color='black')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title(f'Losses - {key}')
  plt.grid(True)
  plt.legend()
  plt.savefig(output_path, bbox_inches='tight')
  plt.close()

import matplotlib.pyplot as plt
import numpy as np

def plot_train_loss_val_accuracy(train_losses, val_accuracies, dict_to_string, output_path, key):
  """
  Plots training loss and validation accuracy on the same graph using two y-axes.

  Parameters:
  - train_losses (list): List of training loss values.
  - val_accuracies (list): List of validation accuracy values.
  - dict_to_string (str): String representation of some dictionary metadata to display.
  - output_path (str): Path to save the output plot.
  - key (str): Key identifier for the title.
  """
  # plt.figure()
  fig, ax1 = plt.subplots()

  # Plot training loss (primary y-axis)
  ax1.set_xlabel('Epochs')
  ax1.set_ylabel('Training Loss', color='tab:red')
  ax1.plot(train_losses, label='Train Loss', color='tab:red')
  ax1.set_ylim([0, 5])  # Scale y-axis
  ax1.set_yticks(np.arange(0, 5, 0.25))
  ax1.tick_params(axis='y', labelcolor='tab:red')

  # Create second y-axis for validation accuracy
  ax2 = ax1.twinx()
  ax2.set_ylabel('Validation Accuracy', color='tab:blue')
  ax2.plot(val_accuracies, label='Val Accuracy', color='tab:blue')
  ax2.set_ylim([0, 1])  # Accuracy in percentage
  ax2.set_yticks(np.arange(0, 1, 0.2))
  ax2.tick_params(axis='y', labelcolor='tab:blue')

  # Add additional text on the figure
  plt.figtext(1.1, 0.5, dict_to_string, ha='left', va='center', fontsize=12, color='black')

  # Titles, grid, and legend
  plt.title(f'Train Loss & Val Accuracy - {key}')
  ax1.grid(True)
  fig.legend(loc='upper right')

  # Save plot and close
  plt.savefig(output_path, bbox_inches='tight')
  plt.close(fig)


def plot_losses(data, run_output_folder, test_id, additional_info='', plot_mae_per_subject=False, plot_mae_per_class=False,plot_train_loss_val_acc=True):
  # Adjust run_output_folder to store plots
  run_output_folder = Path(run_output_folder).parts[:-3]
  run_output_folder = os.path.join(*run_output_folder, 'plot_loss_per_k')
  os.makedirs(run_output_folder, exist_ok=True)
  y_lim = 3
  for key in data.keys():
    only_losses_folder = os.path.join(run_output_folder, f'only_losses_{key.split("_")[0]}')
    os.makedirs(only_losses_folder, exist_ok=True)
    class_subject_loss_folder = os.path.join(run_output_folder, f'class_subject_loss_{key.split("_")[0]}')
    os.makedirs(class_subject_loss_folder, exist_ok=True)
    if 'cross_val' in key and 'train_val' in key:
      if plot_train_loss_val_acc:
        train_losses = data[key].get('train_losses', [])
        val_accuracy = data[key].get('list_val_macro_accuracy', [])
        if train_losses and val_accuracy:
          dict_to_string = convert_dict_to_string(filter_dict(data['config']))
          dict_to_string = dict_to_string.replace('head_params', data['config']['head'].value)
          output_path = os.path.join(only_losses_folder, f'{test_id}{additional_info}_losses_{key}.png')
          plot_train_loss_val_accuracy(train_losses, val_accuracy, dict_to_string, output_path, key)
      else:
        train_losses = data[key].get('train_losses', [])
        val_losses = data[key].get('val_losses', [])
        if train_losses and val_losses:
          data['config']['test_id'] = test_id
          dict_to_string = convert_dict_to_string(filter_dict(data['config']))
          dict_to_string = dict_to_string.replace('head_params', data['config']['head'].value)
          output_path = os.path.join(only_losses_folder, f'{test_id}{additional_info}_losses_{key}.png')
          plot_train_loss_val_acc(train_losses, val_losses, dict_to_string, output_path, key)
      if plot_mae_per_subject:
        best_epoch = f"{data[key]['best_model_idx']}"
        uniqie_subject_ids_train, uniqie_subject_ids_val = retrieve_subject_ids(data, key, best_epoch)
        tools.plot_error_per_subject(loss_per_subject=data[key]['train_loss_per_subject'][best_epoch],
                                     unique_subject_ids=uniqie_subject_ids_train,
                                     saving_path=os.path.join(class_subject_loss_folder, f'{test_id}{additional_info}_train_mae_per_subject_{key}.png'),
                                     title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                     criterion=data['config']['criterion'],
                                     y_lim=y_lim)
        tools.plot_error_per_subject(loss_per_subject=data[key]['val_loss_per_subject'][best_epoch],
                                     unique_subject_ids=uniqie_subject_ids_val,
                                     saving_path=os.path.join(class_subject_loss_folder, f'{test_id}{additional_info}_val_mae_per_subject_{key}.png'),
                                     title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                     criterion=data['config']['criterion'],
                                     y_lim=y_lim)
      if plot_mae_per_class:
        best_epoch = f"{data[key]['best_model_idx']}"
        tools.plot_error_per_class(unique_classes=data[key]['y_unique'],
                                   mae_per_class=data[key]['train_loss_per_class'][best_epoch],
                                   saving_path=os.path.join(class_subject_loss_folder, f'{test_id}{additional_info}_train_mae_per_class_{key}.png'),
                                   title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                   criterion=data['config']['criterion'],
                                   y_lim=y_lim)
        tools.plot_error_per_class(unique_classes=data[key]['y_unique'],
                                   mae_per_class=data[key]['val_loss_per_class'][best_epoch],
                                   saving_path=os.path.join(class_subject_loss_folder, f'{test_id}{additional_info}_val_mae_per_class_{key}.png'),
                                   title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                   criterion=data['config']['criterion'],
                                   y_lim=y_lim)
    if (plot_mae_per_class or plot_mae_per_subject) and 'test' in key:
      tools.plot_error_per_class(mae_per_class=data[key]['dict_test']['test_loss_per_class'],
                                 unique_classes=data[key]['dict_test']['test_unique_y'],
                                 saving_path=os.path.join(class_subject_loss_folder, f'{test_id}{additional_info}_test_mae_per_class_{key}.png'),
                                 title=f'TEST {key} - {test_id}',
                                 criterion=data['config']['criterion'],
                                 y_lim=y_lim)
      tools.plot_error_per_subject(loss_per_subject=data[key]['dict_test']['test_loss_per_subject'],
                                   unique_subject_ids=data[key]['dict_test']['test_unique_subject_ids'],
                                   saving_path=os.path.join(class_subject_loss_folder, f'{test_id}{additional_info}_test_mae_per_subject_{key}.png'),
                                   title=f'TEST {key} - {test_id}',
                                   criterion=data['config']['criterion'],
                                   y_lim=y_lim)

def plot_accuracies(data, run_output_folder, test_id, additional_info=''):
  os.makedirs(run_output_folder, exist_ok=True)
  for key in data.keys():
    if 'cross_val' in key and 'train_val' in key:
      train_acc = data[key].get('list_train_macro_accuracy', [])
      val_acc = data[key].get('list_val_macro_accuracy', [])
      if train_acc and val_acc:
        plt.figure()
        plt.ylim([0, 1])
        plt.yticks(np.arange(0, 1, 0.1))
        plt.plot(train_acc, label='Train Macro Accuracy')
        plt.plot(val_acc, label='Val Macro Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Macro Accuracy')
        plt.title(f'Accuracy id:{test_id} key:{key}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(run_output_folder, f'{test_id}{additional_info}_accuracy_{key}.png'))
        plt.close()

def plot_test_metrics(data, run_output_folder, test_id, additional_info=''):
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
    plt.bar(x + width/2, macro_precisions, width, label='Test Macro Precision')
    plt.ylim([0, 2])
    plt.yticks(np.arange(0, 2, 0.2))
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel('Metric Value')
    plt.title('Test Loss and Macro Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_output_folder, f'{test_id}{additional_info}_test_metrics_histogram.png'))
    plt.close()

def convert_dict_to_string(d):
  new_d = flatten_dict(d)
  return '\n'.join([f'{k}: {v}' for k, v in new_d.items()])

def filter_dict(d):
  keys_to_exclude = ['model_type', 'epochs', 'pooling_embedding_reduction', 'pooling_clips_reduction',
                     'shuffle_video_chunks', 'sample_frame_strategy', 'head', 'stride_window_in_video',
                     'plot_dataset_distribution', 'clip_length', 'early_stopping']
  new_d = {k: v for k, v in d.items() if k not in keys_to_exclude}
  return new_d

def flatten_dict(d, root_name=''):
  result = {}
  for k, v in d.items():
    if isinstance(v, dict):
      result.update(flatten_dict(v, f'{root_name}.{k}' if root_name else k))
    else:
      if root_name:
        result[f'{root_name}.{k}'] = v
      else:
        result[k] = v
  return result

def get_range_k_fold(data):
  count = 0
  for k in data.keys():
    if '_test' in k:
      count += 1
  return count

def get_range_subfold(data):
  count = 0
  for k in data.keys():
    if 'cross_val' in k:
      count += 1
  return count

def generate_csv_row(data, test_id):
  real_k_fold = data['config']['real_k_fold']
  real_sub_fold = data['config']['real_sub_fold']
  k_fold = data['config']['k_fold']
  test_losses = {f'test_loss_k{i}': data[f'k{i}_test']['dict_test']['test_loss'] for i in range(real_k_fold)}
  test_accuracies = {f'test_accuracy_k{i}': data[f'k{i}_test']['dict_test']['test_macro_precision'].item() for i in range(real_k_fold)}
  val_accuracies = {}
  train_accuracies = {}
  for i in range(real_k_fold):
    best_model_subfolder_idx = data[f'k{i}_test']['best_model_subfolder_idx']
    val_accuracies[f'val_accuracy_k{i}'] = data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['list_val_macro_accuracy'][data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['best_model_idx']]
    train_accuracies[f'train_accuracy_k{i}'] = data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['list_train_macro_accuracy'][data[f'k{i}_cross_val_sub_{best_model_subfolder_idx}_train_val']['best_model_idx']]
  
  mean_val_accuracies_last_epoch = {}
  mean_train_losses_last_epoch = {}
  mean_train_accuracies_last_epoch = {}
  mean_train_losses_last_epoch = {f'mean_train_loss_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}_train_val']['train_losses'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_accuracies_last_epoch = {f'mean_val_accuracy_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}_train_val']['list_val_macro_accuracy'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_train_accuracies_last_epoch = {f'mean_train_accuracy_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}_train_val']['list_train_macro_accuracy'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  # try:
  #   mean_val_accuracies_200_epoch = {}
  #   mean_train_losses_200_epoch = {}
  #   mean_train_accuracies_200_epoch = {}
  #   mean_train_losses_200_epoch = {f'mean_train_loss_200_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}_train_val']['train_losses'][199] for j in range(k_fold-1)]) for i in range(real_k_fold)}
  #   mean_val_accuracies_200_epoch = {f'mean_val_accuracy_200_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}_train_val']['list_val_macro_accuracy'][199] for j in range(k_fold-1)]) for i in range(real_k_fold)}
  #   mean_train_accuracies_200_epoch = {f'mean_train_accuracy_200_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}_train_val']['list_train_macro_accuracy'][199] for j in range(k_fold-1)]) for i in range(real_k_fold)}
  # except Exception as e:
  #   print(f'Error in {test_id} - {e}')
  
  best_sub_folders = [data[f'k{i}_test']['best_model_subfolder_idx'] for i in range(real_k_fold)]
  best_epochs = {f'best_epoch_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['best_model_idx'] for i, sub in zip(range(real_k_fold), best_sub_folders)}
  train_losses = {f'train_loss_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['train_losses'][epoch] for i, sub, epoch in zip(range(real_k_fold), best_sub_folders, best_epochs.values())}
  val_losses = {f'val_loss_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['val_losses'][epoch] for i, sub, epoch in zip(range(real_k_fold), best_sub_folders, best_epochs.values())}
  train_losses_last = {f'train_loss_last_ep_k{i}': data[f'k{i}_cross_val_sub_{sub}_train_val']['train_losses'][-1] for i, sub in zip(range(real_k_fold), best_sub_folders)}
  
  config = data['config']
  time_ = (int(data['time']/60)) if 'time' in data else 'ND'
  head_type = config['head'].name
  clip_grad_norm = config['clip_grad_norm'] if 'clip_grad_norm' in config else 'ND'
  head_params = flatten_dict({f'{head_type}': config['head_params']})
  row_dict = {
    'test_id': test_id,
    'model': config['model_type'].name,
    'head': head_type,
    'optimizer': config['optimizer_fn'],
    'enable_scheduler': config['enable_scheduler'] if 'enable_scheduler' in config else 'ND',
    'learning_rate': config['lr'],
    'criterion': type(config['criterion']).__name__,
    'init_network': config['init_network'],
    'reg_lambda': config['regularization_lambda'],
    'reg_loss': config['regularization_loss'],
    'feature_type': config['features_folder_saving_path'][-1] if config['features_folder_saving_path'][-1] != '' else config['features_folder_saving_path'][-2],
    'early_stopping_key': config['key_for_early_stopping'] + f'(pat={config["early_stopping"].patience},eps={config["early_stopping"].min_delta},t_mod={config["early_stopping"].threshold_mode})',
    'target_metric': config['target_metric_best_model'],
    'round_output_loss': config['round_output_loss'],
    'batch_size_training': config['batch_size_training'],
    'max_epochs': config['epochs'],
    **head_params,
    **mean_val_accuracies_last_epoch,
    **mean_train_accuracies_last_epoch,
    # **mean_val_accuracies_200_epoch,
    # **mean_train_accuracies_200_epoch,
    **mean_train_losses_last_epoch,
    # **mean_train_losses_200_epoch,
    **train_losses_last,
    **test_losses,
    **test_accuracies,
    **best_epochs,
    **train_losses,
    **val_losses,
    'mean_train_loss': np.mean(list(train_losses.values())),
    'mean_val_loss': np.mean(list(val_losses.values())),
    'mean_test_loss': np.mean(list(test_losses.values())),
    'mean_train_accuracy': np.mean(list(train_accuracies.values())),
    'mean_val_accuracy': np.mean(list(val_accuracies.values())),
    'mean_test_accuracy': np.mean(list(test_accuracies.values())),
    **train_accuracies,
    **val_accuracies,
    'clip_grad_norm': clip_grad_norm,
    'time_min': time_,
  }
  return row_dict

def plot_confusion_matrices(data, root_output_folder, test_id, additional_info=''):
  def plot_matrices(matrices, output_folder, stage, folder_name, test_id):
    os.makedirs(output_folder, exist_ok=True)
    if isinstance(matrices, dict):
      for epoch, cnf in matrices.items():
        tools.plot_confusion_matrix(confusion_matrix=cnf,
                                    title=f'{stage}_epoch_{epoch}',
                                    saving_path=os.path.join(output_folder, f'{test_id}{additional_info}_{folder_name}_{stage}_epoch_{epoch}.png'))
    elif isinstance(matrices, MulticlassConfusionMatrix):
      tools.plot_confusion_matrix(confusion_matrix=matrices,
                                  title=f'Confusion Matrix {stage} - id:{test_id} - {additional_info}',
                                  saving_path=os.path.join(output_folder, f'{test_id}{additional_info}_{folder_name}_{stage}.png'))
  for k, v in data.items():
    if 'cross_val' in k:
      dict_conf_matricies = v.get('train_confusion_matricies')
      plot_matrices(matrices=dict_conf_matricies, output_folder=root_output_folder, folder_name=k, test_id=test_id, stage='train')
      dict_conf_matricies = v.get('val_confusion_matricies')
      plot_matrices(matrices=dict_conf_matricies, output_folder=root_output_folder, folder_name=k, test_id=test_id, stage='val')
    elif 'test' in k:
      dict_conf_matricies = v['dict_test']['test_confusion_matrix']
      plot_matrices(matrices=dict_conf_matricies, output_folder=root_output_folder, folder_name=k, test_id=test_id, stage=f'test_using_subfolder_{v["best_model_subfolder_idx"]}')

def get_best_result(data):
  config = data['config']
  target_metric_best_model = config['target_metric_best_model']
  k_fold = config['real_k_fold']
  best_fold = 0
  for k in range(1, k_fold):
    if target_metric_best_model == 'val_loss':
      if data[f'k{k}_test']['dict_test']['test_loss'] < data[f'k{best_fold}_test']['dict_test']['test_loss']:
        best_fold = k
    elif target_metric_best_model == 'val_macro_precision':
      if data[f'k{k}_test']['dict_test']['test_macro_precision'] > data[f'k{best_fold}_test']['dict_test']['test_macro_precision']:
        best_fold = k
    else:
      raise ValueError(f'Not implemented target metric {target_metric_best_model}')
  best_sub_folder = data[f'k{best_fold}_test']['best_model_subfolder_idx']
  dict_best_result = {f'k{best_fold}_cross_val_sub_{best_sub_folder}_train_val': data[f'k{best_fold}_cross_val_sub_{best_sub_folder}_train_val'],
                      f'k{best_fold}_test': data[f'k{best_fold}_test'],
                      'best_sub_folder': best_sub_folder,
                      'best_fold': best_fold,
                      'config': config}
  return dict_best_result

def collect_loss_plots(summary_folder, output_folder):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  folders = [f for f in os.listdir(summary_folder) if os.path.isdir(os.path.join(summary_folder, f))]
  for folder in folders:
    losses_png_path = os.path.join(summary_folder, folder, 'loss_plots')
    for _, _, files in os.walk(losses_png_path):
      for file in files:
        if file.endswith(".png"):
          src_path = os.path.join(losses_png_path, file)
          new_name = f"{folder.split('_')[0]}_{file}"
          dst_path = os.path.join(output_folder, new_name)
          try:
            if not os.path.exists(dst_path):
              os.symlink(src_path, dst_path)
          except OSError:
            shutil.copy(src_path, dst_path)
  print(f"All loss plots collected in {output_folder}")

def plot_run_details(parent_folder, output_root,only_csv):
  results_files = find_results_files(parent_folder)
  results_data = {file: load_results(file) for file in results_files}
  print(f'Loaded {len(results_data)} results files')
  list_row_csv = []
  for file, data in tqdm.tqdm(results_data.items()):
    test_folder = os.path.basename(os.path.dirname(file))
    data['config']['real_k_fold'] = get_range_k_fold(data)
    data['config']['real_sub_fold'] = get_range_subfold(data)
    if data['config']['real_k_fold'] == 0:
      print(f'No TEST file found in {file}')
      continue
    grid_search_folder = Path(file).parts[-3]
    test_id = test_folder.split('_')[0]
    run_output_folder = os.path.join(output_root)
    list_row_csv.append(generate_csv_row(data, test_id))
    if not only_csv:
      data_best = get_best_result(data)
      data_wo_best = {k: v for k, v in data.items() if k != f'k{data_best["best_fold"]}_cross_val_sub_{data_best["best_sub_folder"]}_train_val' and k != f'k{data_best["best_fold"]}_test'}
      try:
        plot_losses(data_wo_best, os.path.join(run_output_folder, grid_search_folder, 'loss_plots'), test_id)
        plot_accuracies(data_wo_best, os.path.join(run_output_folder, grid_search_folder, 'accuracy_plots'), test_id)
        plot_confusion_matrices(data_wo_best, os.path.join(run_output_folder, grid_search_folder, 'confusion_matrices'), test_id)
        plot_test_metrics(data, os.path.join(run_output_folder, grid_search_folder, 'test_plots'), test_id)
        plot_losses(data_best, os.path.join(run_output_folder, grid_search_folder, 'loss_plots'), test_id, '_best', plot_mae_per_subject=True, plot_mae_per_class=True)
        plot_accuracies(data_best, os.path.join(run_output_folder, grid_search_folder, 'accuracy_plots'), test_id, '_best')
        plot_confusion_matrices(data_best, os.path.join(run_output_folder, grid_search_folder, 'confusion_matrices'), test_id, '_best')
      except Exception as e:
        print(f'Error in {file} - {e}')
  df = pd.DataFrame(list_row_csv)
  df = df.fillna('ND')
  if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)
  df.to_csv(os.path.join(output_root, 'summary.csv'), index=False)
  print(f'Summary CSV saved to {output_root}/summary.csv')

def plot_filtered_run_details(parent_folder, output_root, filter_dict,only_csv):
  """
  Processes only results whose config matches the filter_dict criteria.
  Filtered plots are saved under 'filtered_plots' along with a 'filters.txt'
  file listing the applied filters.
  """
  results_files = find_results_files(parent_folder)
  results_data = {file: load_results(file) for file in results_files}
  print(f'Loaded {len(results_data)} results files')
  filtered_data = {}
  
  # Filter data
  for file, data in results_data.items():
    config = data.get('config', {})
    match = True
    for key, value in filter_dict.items():
      if key not in config or str(config[key]).strip() != str(value).strip():
        match = False
        break
    if match:
      filtered_data[file] = data
  print(f'After filtering, {len(filtered_data)} results files remain.')
  
  # Generate folder structure
  str_filter_dict = '_'.join([f'{k}={v}' for k, v in filter_dict.items()])
  filtered_output_folder = os.path.join(output_root, f'filtered_plots_{str_filter_dict}')
  os.makedirs(filtered_output_folder, exist_ok=True)
  filters_txt_path = os.path.join(filtered_output_folder, 'filters.txt')
  with open(filters_txt_path, 'w') as f:
    f.write('\n'.join([f'{k}: {v}' for k, v in filter_dict.items()]))
  print(f'Filter details saved in {filters_txt_path}')
  list_row_csv = []
  
  # Process filtered data
  for file, data in tqdm.tqdm(filtered_data.items()):
    test_folder = os.path.basename(os.path.dirname(file))
    data['config']['real_k_fold'] = get_range_k_fold(data)
    data['config']['real_sub_fold'] = get_range_subfold(data)
    if data['config']['real_k_fold'] == 0:
      print(f'No TEST file found in {file}')
      continue
    grid_search_folder = Path(file).parts[-3]
    test_id = test_folder.split('_')[0]
    run_output_folder = os.path.join(filtered_output_folder, grid_search_folder, 'plot_per_run')
    os.makedirs(run_output_folder, exist_ok=True)
    list_row_csv.append(generate_csv_row(data, test_id))
    if not only_csv:
      data_best = get_best_result(data)
      data_wo_best = {k: v for k, v in data.items() if k != f'k{data_best["best_fold"]}_cross_val_sub_{data_best["best_sub_folder"]}_train_val' and k != f'k{data_best["best_fold"]}_test'}
      try:
        plot_losses(data_wo_best, os.path.join(run_output_folder, 'loss_plots'), test_id)
        plot_accuracies(data_wo_best, os.path.join(run_output_folder, 'accuracy_plots'), test_id)
        plot_confusion_matrices(data_wo_best, os.path.join(run_output_folder, 'confusion_matrices'), test_id)
        plot_test_metrics(data, os.path.join(run_output_folder, 'test_plots'), test_id)
        plot_losses(data_best, os.path.join(run_output_folder, 'loss_plots'), test_id, '_best', plot_mae_per_subject=True, plot_mae_per_class=True)
        plot_accuracies(data_best, os.path.join(run_output_folder, 'accuracy_plots'), test_id, '_best')
        plot_confusion_matrices(data_best, os.path.join(run_output_folder, 'confusion_matrices'), test_id, '_best')
      except Exception as e:
        print(f'Error in {file} - {e}')
  df = pd.DataFrame(list_row_csv).fillna('ND')
  csv_path = os.path.join(filtered_output_folder, 'summary.csv')
  df.to_csv(csv_path, index=False)
  print(f'Summary CSV saved to {csv_path}')

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot results from a folder')
  parser.add_argument('--parent_folder', type=str, required=True,
                      help='Path to folder containing all the results')
  parser.add_argument('--filter', type=str, default='',
                      help='Optional filter criteria in format key1=val1,key2=val2')
  parser.add_argument('--only_csv', action='store_true',
                      help='Generate only the summary CSV file without generating any plots')
  args = parser.parse_args()
  parent_folder = args.parent_folder
  only_csv = args.only_csv
  print(f'Parent folder: {parent_folder}')
  output_root = os.path.join(parent_folder, '_summary')
  os.makedirs(output_root, exist_ok=True)
  # If filter criteria is provided, parse and process filtered results
  if args.filter:
    filter_dict_arg = {}
    for pair in args.filter.split(','):
      if '=' in pair:
        key, value = pair.split('=')
        filter_dict_arg[key.strip()] = value.strip()
    print(f'Applying filter: {filter_dict_arg}')
    plot_filtered_run_details(parent_folder, output_root, filter_dict_arg,only_csv)
  else:
  # Generate the unfiltered plots
    plot_run_details(parent_folder, os.path.join(output_root, 'plot_per_run'),only_csv)

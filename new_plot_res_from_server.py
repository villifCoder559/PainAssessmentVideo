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
from custom import helper


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


def plot_losses(data, run_output_folder, test_id, additional_info='', plot_mae_per_subject=True, plot_mae_per_class=True,plot_train_loss_val_acc=True):
  # Adjust run_output_folder to store plots
  # run_output_folder = Path(run_output_folder).parts[:-3]
  test_output_folder = os.path.join(run_output_folder, test_id)
  loss_plots_output_folder = os.path.join(run_output_folder, 'loss_plots')
  os.makedirs(loss_plots_output_folder, exist_ok=True)
  os.makedirs(test_output_folder, exist_ok=True)
  for key in data['results'].keys():
    only_losses_folder_per_k = os.path.join(loss_plots_output_folder, f'only_losses_{key.split("_")[0]}_{key.split("_")[-1]}')
    os.makedirs(only_losses_folder_per_k, exist_ok=True)
    # class_subject_loss_folder = os.path.join(run_output_folder, f'class_subject_loss_{key.split("_")[0]}_{key.split("_")[1]}')
    # os.makedirs(class_subject_loss_folder, exist_ok=True)
    if plot_train_loss_val_acc:
      train_losses = data['results'][key]['train_val'].get('train_losses', [])
      train_accuracy = data['results'][key]['train_val'].get('list_train_macro_accuracy', [])
      val_loss = data['results'][key]['train_val'].get('val_losses', [])
      val_accuracy = data['results'][key]['train_val'].get('list_val_macro_accuracy', [])
      test_accuracy = data['results'][key]['test'].get('test_macro_precision', None)
      test_loss = data['results'][key]['test'].get('test_loss', None)
      grad_norm_mean = data['results'][key]['train_val']['list_mean_total_norm_epoch']
      grad_norm_std = data['results'][key]['train_val']['list_std_total_norm_epoch']
      if 'list_train_confidence_prediction_right_mean' in data['results'][key]['train_val']:
        val_prediction_confidence_right_mean = data['results'][key]['train_val']['list_val_confidence_prediction_right_mean']
        val_prediction_confidence_right_std = data['results'][key]['train_val']['list_val_confidence_prediction_right_std']
        val_prediction_confidence_wrong_mean = data['results'][key]['train_val']['list_val_confidence_prediction_wrong_mean']
        val_prediction_confidence_wrong_std = data['results'][key]['train_val']['list_val_confidence_prediction_wrong_std']
        train_prediction_confidence_right_mean = data['results'][key]['train_val']['list_train_confidence_prediction_right_mean']
        train_prediction_confidence_right_std = data['results'][key]['train_val']['list_train_confidence_prediction_right_std']
        train_prediction_confidence_wrong_mean = data['results'][key]['train_val']['list_train_confidence_prediction_wrong_mean']
        train_prediction_confidence_wrong_std = data['results'][key]['train_val']['list_train_confidence_prediction_wrong_std']
        fig, axs = plt.subplots(3,2,figsize=(20,15))
      else:
        fig, axs = plt.subplots(2,2,figsize=(20,15))
      
      # Plot train and val loss
      if train_losses and val_accuracy:
        dict_to_string = convert_dict_to_string(filter_dict(data['config']))
        dict_to_string = dict_to_string.replace('head_params', data['config']['head'].value)
        # add test_id in dict_to_string
        dict_to_string += f'\nTest ID: {test_id}'
        dict_to_string += f'\nfold_subfold: {key.split("_")[0]}_{key.split("_")[-1]}'
        input_dict_loss_acc= {
          'list_1':train_losses,
          'list_2':val_accuracy,
          'output_path':None,
          'title':'Train loss, validation accuracy, test accuracy',
          'point':{
            'value':test_accuracy,
            'epoch':data['results'][key]['train_val']['best_model_idx']
            },
          'ax':axs[0][0],
          'x_label':'Epochs',
          'y_label_1':'Train loss',
          'y_label_2':'Validation accuracy',
          'y_label_3':'Test accuracy',
          'y_lim_1':[0, 5],
          'y_lim_2':[0, 1],
          'y_lim_3':[0, 1],
          'step_ylim_1':0.25,
          'step_ylim_2':0.1,
          'step_ylim_3':0.1,
          'dict_to_string':None,
          'color_1':'tab:red',
          'color_2':'tab:blue',
          'color_3':'tab:green',
        }
        input_dict_loss={
          'list_1':train_losses,
          'list_2':val_loss,
          'output_path':None,
          'title':'Train loss + validation and test loss',
          'point':{
            'value':test_loss,
            'epoch':data['results'][key]['train_val']['best_model_idx']
            },
          'ax':axs[1][0],
          'x_label':'Epochs',
          'y_label_1':'Train loss',
          'y_label_2':'Validation loss',
          'y_label_3':'Test loss',
          'y_lim_1':[0, 5],
          'y_lim_2':[0, 5],
          'y_lim_3':[0, 5],
          'step_ylim_1':0.25,
          'step_ylim_2':0.25,
          'step_ylim_3':0.25,
          'dict_to_string':None,
          'color_1':'tab:red',
          'color_2':'tab:purple',
          'color_3':(0,0,0), # (r,g,b)
        }
        input_dict_accuracy_gap={
          'list_1':np.array(train_accuracy)-np.array(val_accuracy),
          'title':'Accuracy Gap in Train-Validation',
          'ax':axs[0][1],
          'x_label':'Epochs',
          'y_label_1':'Accuracy Gap',
          'y_lim_1':[-1, 1],
          'color_1':'tab:orange',
        }
        tools.plot_losses_and_test_new(**input_dict_loss_acc)
        tools.plot_losses_and_test_new(**input_dict_accuracy_gap)
        tools.plot_losses_and_test_new(**input_dict_loss)
        
        axs[0][1].text(1.04, 0.0, dict_to_string, fontsize=12, color='black',transform=axs[0][1].transAxes,ha='left', va='center')
        
        # Plot GRADIENT
        tools.plot_with_std(ax=axs[1][1],x=list(range(len(grad_norm_mean))),mean=grad_norm_mean,std=grad_norm_std,
                            title='Mean and std of GRADIENT norm',x_label='Epochs',y_label='Gradient norm',y_lim=[0, 15])
        
        if 'list_train_confidence_prediction_right_mean' in data['results'][key]['train_val']:
          tools.plot_with_std(ax=axs[2][0],x=list(range(len(train_prediction_confidence_right_mean))),
                              mean=train_prediction_confidence_right_mean,
                              std=train_prediction_confidence_right_std,
                              title='TRAIN prediction confidence',
                              x_label='Epochs',
                              color='green',
                              y_label='Prediction confidence',
                              legend_label_mean='Correct pred. mean',
                              legend_label_std='Correct pred. std',
                              y_lim=[0, 1])
          tools.plot_with_std(ax=axs[2][0],x=list(range(len(train_prediction_confidence_wrong_mean))),
                              mean=train_prediction_confidence_wrong_mean,
                              std=train_prediction_confidence_wrong_std,
                              title='TRAIN Mean and std of prediction confidence',
                              x_label='Epochs',
                              color='red',
                              y_label='Prediction confidence',
                              legend_label_mean='Wrong pred. mean',
                              legend_label_std='Wrong pred. std',
                              y_lim=[0, 1])
          
          
          tools.plot_with_std(ax=axs[2][1],
                              x=list(range(len(val_prediction_confidence_right_mean))),
                              mean=val_prediction_confidence_right_mean,
                              std=val_prediction_confidence_right_std,
                              color='green',
                              title='VALIDATION prediction confidence',
                              x_label='Epochs',
                              y_label='Right prediction confidence',
                              legend_label_mean='Correct pred. mean',
                              legend_label_std='Correct pred. std',
                              y_lim=[0, 1])
          
          tools.plot_with_std(ax=axs[2][1],
                              x=list(range(len(val_prediction_confidence_wrong_mean))),
                              mean=val_prediction_confidence_wrong_mean,
                              std=val_prediction_confidence_wrong_std,
                              color='red',
                              title='VALIDATION Mean and std of prediction confidence',
                              x_label='Epochs',
                              y_label='Wrong prediction confidence',
                              legend_label_mean='Wrong mean',
                              legend_label_std='Wrong std',
                              y_lim=[0, 1])
        
        plot_path = os.path.join(test_output_folder, f'{test_id}{additional_info}_losses_{key}.png')
        
        fig.savefig(plot_path, bbox_inches='tight')
        plt.close(fig)
        symlink_path = os.path.join(only_losses_folder_per_k, f'{test_id}{additional_info}_losses_{key}.png')
        # Remove the link if it already exists to avoid errors
        try:
          os.remove(symlink_path)
        except FileNotFoundError:
          pass
        # Create a new symlink
        os.symlink(plot_path, symlink_path)
      
      if plot_mae_per_subject:
        y_lim = 3
        fig, axs = plt.subplots(4,1,figsize=(20,20))
        best_epoch = data['results'][key]['train_val']['best_model_idx']
        # Train accuracy
        # Train
        tools.plot_error_per_subject(loss_per_subject=data['results'][key]['train_val']['list_train_accuracy_per_subject'][best_epoch],
                                     unique_subject_ids=data['results'][key]['train_val']['train_unique_subject_ids'],
                                     y_label='Train Accuracy',
                                     title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                     criterion=data['config']['criterion'],
                                     list_stoic_subject=helper.stoic_subjects,
                                     bar_color='blue',
                                    #  accuracy_per_subject=data['results'][key]['train_val']['list_train_accuracy_per_subject'][best_epoch],
                                    #  count_subjects=data['results'][key]['train_val']['train_count_subject_ids'],
                                     y_lim=1,
                                     ax=axs[0])
        # Train Loss
        tools.plot_error_per_subject(loss_per_subject=data['results'][key]['train_val']['train_loss_per_subject'][best_epoch],
                                     unique_subject_ids=data['results'][key]['train_val']['train_unique_subject_ids'],
                                     title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                     criterion=data['config']['criterion'],
                                     list_stoic_subject=helper.stoic_subjects,
                                    #  accuracy_per_subject=data['results'][key]['train_val']['list_train_accuracy_per_subject'][best_epoch],
                                    #  count_subjects=data['results'][key]['train_val']['train_count_subject_ids'],
                                     y_lim=y_lim,
                                     ax=axs[1])
        # Val
        tools.plot_error_per_subject(loss_per_subject=data['results'][key]['train_val']['val_loss_per_subject'][best_epoch],
                                     unique_subject_ids=data['results'][key]['train_val']['val_unique_subject_ids'],
                                     title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                     criterion=data['config']['criterion'],
                                     list_stoic_subject=helper.stoic_subjects,
                                    #  accuracy_per_subject=data['results'][key]['train_val']['list_val_accuracy_per_subject'][best_epoch],
                                    #  count_subjects=data['results'][key]['train_val']['val_count_subject_ids'],
                                     y_lim=y_lim,
                                     ax=axs[2])
        # Test
        tools.plot_error_per_subject(loss_per_subject=data['results'][key]['test']['test_loss_per_subject'],
                                   unique_subject_ids=data['results'][key]['test']['test_unique_subject_ids'],
                                   title=f'TEST {key} - {test_id}',
                                   criterion=data['config']['criterion'],
                                  #  accuracy_per_subject=data['results'][key]['test']['test_accuracy_per_subject'],
                                  #  count_subjects=data['results'][key]['test']['test_count_subject_ids'],
                                   list_stoic_subject=helper.stoic_subjects,
                                   y_lim=y_lim,
                                   ax=axs[3])
        fig.tight_layout()
        fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_mae_per_subject_{key}.png'))
        plt.close(fig)
      if plot_mae_per_class:
        y_lim = 3
        fig, axs = plt.subplots(3,1,figsize=(10,8))
        best_epoch = data['results'][key]['train_val']['best_model_idx']
        tools.plot_error_per_class(unique_classes=data['results'][key]['train_val']['y_unique'],
                                   mae_per_class=data['results'][key]['train_val']['train_loss_per_class'][best_epoch],
                                   title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                   criterion=data['config']['criterion'],
                                   accuracy_per_class=data['results'][key]['train_val']['list_train_accuracy_per_class'][best_epoch],
                                   y_lim=y_lim,
                                   ax=axs[0])
        tools.plot_error_per_class(unique_classes=data['results'][key]['train_val']['y_unique'],
                                   mae_per_class=data['results'][key]['train_val']['val_loss_per_class'][best_epoch],
                                   title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                   criterion=data['config']['criterion'],
                                   accuracy_per_class=data['results'][key]['train_val']['list_val_accuracy_per_class'][best_epoch],
                                   y_lim=y_lim,
                                   ax=axs[1])
        tools.plot_error_per_class(mae_per_class=data['results'][key]['test']['test_loss_per_class'],
                                 unique_classes=data['results'][key]['test']['test_unique_y'],
                                 title=f'TEST {key} - {test_id}',
                                 criterion=data['config']['criterion'],
                                 accuracy_per_class=data['results'][key]['test']['test_accuracy_per_class'],
                                 y_lim=y_lim,
                                 ax=axs[2])
        fig.tight_layout()
        fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_mae_per_class_{key}.png'))
        plt.close(fig)



def convert_dict_to_string(d):
  new_d = flatten_dict(d)
  return '\n'.join([f'{k}: {v}' for k, v in new_d.items()])

def filter_dict(d):
  keys_to_exclude = ['model_type', 'epochs', 'pooling_embedding_reduction', 'pooling_clips_reduction',
                     'shuffle_video_chunks', 'sample_frame_strategy', 'head', 'stride_window_in_video',
                     'plot_dataset_distribution', 'clip_length', 'early_stopping']
  new_d = {k: v for k, v in d.items() if k not in keys_to_exclude}
  for k,v in new_d.items():
    if 'path' in k:
      new_d[k] = v[-1]
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


def generate_csv_row(data,config,time_, test_id):

  list_fold = [int(k.split('_')[0][1:]) for k in data.keys()]
  list_sub_fold = [int(k.split('_')[-1]) for k in data.keys()]
  real_k_fold = max(list_fold) + 1
  real_sub_fold = max(list_sub_fold) + 1
  
  mean_train_losses_last_epoch = {f'mean_train_loss_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_train_accuracies_last_epoch = {f'mean_train_accuracy_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_train_macro_accuracy'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_accuracies_last_epoch = {f'mean_val_accuracy_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_macro_accuracy'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_losses_last_epoch = {f'mean_val_loss_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    
  mean_train_losses_best_epoch = {f'mean_train_loss_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_train_accuracies_best_epoch = {f'mean_train_accuracy_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_train_macro_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_accuracies_best_epoch = {f'mean_val_accuracy_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_macro_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_losses_best_epoch = {f'mean_val_loss_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  
  mean_test_accuracies = {f'mean_test_accuracy_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['test']['test_macro_precision'] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_test_losses = {f'mean_test_loss_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['test']['test_loss'] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  
  total_mean_train_losses_best_epoch = {f'total_mean_train_loss_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_mean_train_accuracy_best_epoch = {f'total_mean_train_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_train_macro_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_mean_val_accuracy_best_epoch = {f'total_mean_val_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_macro_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_mean_val_losses_best_epoch = {f'total_mean_val_loss_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_mean_test_accuracy_best_epoch = {f'total_mean_test_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['test']['test_macro_precision'] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_mean_test_losses_best_epoch = {f'total_mean_test_loss_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['test']['test_loss'] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  
  total_mean_train_losses_last_epoch = {f'total_mean_train_loss_last_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][-1] for i in range(real_k_fold) for j in range(real_sub_fold)])}
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
    'reg_lambda_L1': config['regularization_lambda_L1'] if 'regularization_lambda_L1' in config else config['regularization_lambda'] if config['regularization_loss'] == 'L1' else 0,
    'reg_lambda_L2': config['regularization_lambda_L2'] if 'regularization_lambda_L2' in config else config['regularization_lambda'] if config['regularization_loss'] == 'L2' else 0,
    # 'reg_lambda_L2': config['regularization_lambda_L2'],
    # 'reg_lambda': config['regularization_lambda'],
    # 'reg_loss': config['regularization_loss'],
    'feature_type': config['features_folder_saving_path'][-1] if config['features_folder_saving_path'][-1] != '' else config['features_folder_saving_path'][-2],
    'early_stopping_key': config['key_for_early_stopping'] + f'(pat={config["early_stopping"].patience},eps={config["early_stopping"].min_delta},t_mod={config["early_stopping"].threshold_mode})',
    'target_metric': config['target_metric_best_model'],
    'round_output_loss': config['round_output_loss'],
    'batch_size_training': config['batch_size_training'],
    'max_epochs': config['epochs'],
    **head_params,
    **total_mean_train_losses_best_epoch,
    **total_mean_val_accuracy_best_epoch,
    **total_mean_test_accuracy_best_epoch,
    **total_mean_train_accuracy_best_epoch,
    **total_mean_test_losses_best_epoch,
    **total_mean_val_losses_best_epoch,
    **total_mean_train_losses_last_epoch,
    **mean_train_losses_last_epoch,
    **mean_val_accuracies_last_epoch,
    **mean_test_accuracies,
    **mean_test_losses,
    **mean_train_accuracies_last_epoch,
    **mean_val_losses_last_epoch,
    **mean_train_losses_best_epoch,
    **mean_train_accuracies_best_epoch,
    **mean_val_accuracies_best_epoch,
    **mean_val_losses_best_epoch,
    'clip_grad_norm': clip_grad_norm,
    'time_min': int(time_//60) if time_ is not None else 'ND',
  }
  return row_dict

def plot_confusion_matrices(data, root_output_folder, test_id, additional_info=''):
  test_output_folder = os.path.join(root_output_folder, test_id)
  os.makedirs(test_output_folder, exist_ok=True)
  for key,dict_sub_fold in data['results'].items():
    best_epoch_idx =  dict_sub_fold['train_val']['best_model_idx']
    dict_train_conf_matrix = dict_sub_fold['train_val']['train_confusion_matricies']
    dict_val_conf_matrix = dict_sub_fold['train_val']['val_confusion_matricies']
    dict_test_conf_matrix = {f'{best_epoch_idx}':dict_sub_fold['test']['test_confusion_matrix']}
    for epoch in dict_train_conf_matrix.keys():
      # if int(epoch) == best_epoch_idx:
      #   fig, axs = plt.subplots(3, 1, figsize=(10, 7))
      # else:
      fig, axs = plt.subplots(2, 1, figsize=(5, 10))
      tools.plot_confusion_matrix(dict_train_conf_matrix[epoch], ax=axs[0], title=f'TRAIN - Epoch {epoch}   - {test_id}')
      tools.plot_confusion_matrix(dict_val_conf_matrix[epoch], ax=axs[1], title=f'VAL - Epoch {epoch}   - {test_id}')
      # if epoch == best_epoch_idx:
      #   tools.plot_confusion_matrix(dict_test_conf_matrix[epoch], ax=axs[2], title=f'TEST {test_id} - {key} - Epoch {epoch}')
      fig.tight_layout()
      fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_confusion_matrix_{key}_epoch_{epoch}.png'))
      plt.close(fig)

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


def plot_run_details(results_data, output_root,only_csv):
  list_row_csv = []
  for file, data in tqdm.tqdm(results_data.items()):
    test_folder = os.path.basename(os.path.dirname(file))
    data['config']['real_k_fold'] = len(data['results'])
    if data['config']['real_k_fold'] == 0:
      print(f'No TEST file found in {file}')
      continue
    grid_search_folder = Path(file).parts[-3]
    test_id = test_folder.split('_')[0]
    # run_output_folder = os.path.join(output_root)
    list_row_csv.append(generate_csv_row(data['results'],data['config'],data['time'], test_id))
    if not only_csv:
      # try:
      plot_losses(data, os.path.join(output_root), test_id)
      plot_confusion_matrices(data, os.path.join(output_root), test_id)
      # except Exception as e:
      #   print(f'Error in {file} - {e}')
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
  
  # Process filtered data
  plot_run_details(filtered_data, filtered_output_folder,only_csv)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot results from a folder') 
  parser.add_argument('--parent_folder', type=str, required=True,
                      help='Path to folder containing all the results')
  parser.add_argument('--filter', type=str, default='',
                      help='Optional filter criteria in format key1=val1,key2=val2')
  parser.add_argument('--only_csv', action='store_true',
                      help='Generate only the summary CSV file without generating any plots')
  parser.add_argument('--print_filter_from', type=str, default=None,
                      help='Print list of avilable filter from the given .pkl file')
  args = parser.parse_args()
  
  parent_folder = args.parent_folder
  only_csv = args.only_csv
  print(f'Parent folder: {parent_folder}')
  output_root = os.path.join(parent_folder, '_summary')
  os.makedirs(output_root, exist_ok=True)
  if args.print_filter_from is not None:
    print(f'Printing filter from: {args.print_filter_from}')
    results_files = find_results_files(args.print_filter_from)
    if len(results_files) == 0:
      raise ValueError(f'No results files found in {args.print_filter_from}')
    data = load_results(results_files[0])
    config = data.get('config', {})
    print(f'Available filter keys:\n{[f" {k}" for k in config.keys()]}')  

  else:
    if args.filter:
      # raise ValueError('Filtering to fix')
      filter_dict_arg = {}
      for pair in args.filter.split(','):
        if '=' in pair:
          key, value = pair.split('=')
          filter_dict_arg[key.strip()] = value.strip()
      print(f'Applying filter: {filter_dict_arg}')
      
      plot_filtered_run_details(parent_folder, output_root, filter_dict_arg,only_csv)
    else:
    # Generate the unfiltered plots
      results_files = find_results_files(parent_folder) # get .pkl files
      results_data = {file: load_results(file) for file in results_files} # load .pkl files with path as a key
      print(f'Loaded {len(results_data)} results files')
      plot_run_details(results_data, os.path.join(output_root, 'plot_run_details'), only_csv)

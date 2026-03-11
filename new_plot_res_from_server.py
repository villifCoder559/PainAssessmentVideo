import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import custom.tools as tools
from torchmetrics.classification import MulticlassConfusionMatrix
import tqdm
from pathlib import Path
import argparse
import logging
from custom import helper
import custom.loss as losses  # For pickle loading
import seaborn as sns
import cv2
import torch



logging.getLogger('matplotlib').setLevel(logging.WARNING)

def find_results_files(parent_folder):
  results_files = []
  list_history_folder = os.listdir(parent_folder)
  for folder in list_history_folder:
    if os.path.isdir(os.path.join(parent_folder, folder)):
      list_runs = os.listdir(os.path.join(parent_folder, folder))
      for run in list_runs:
        if os.path.isdir(os.path.join(parent_folder, folder, run)):
          pkl_files = [f for f in os.listdir(os.path.join(parent_folder, folder, run)) if 'k_fold_results.pkl' in f]
          if len(pkl_files) == 1:
            results_files.append(os.path.join(parent_folder, folder, run, pkl_files[0]))
          # elif len(pkl_files) == 0:
          #   # delete the folder if no .pkl files are found
          #   shutil.rmtree(os.path.join(parent_folder, folder))
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

def get_grouped_losses(data, config):
  
  def update_dict_grouped_losses(k_fold, res, best_epoch, key, value, key_target_dict, upper_dict = 'train_val'):
    # convert to numpy if tensor
    train_class_labels = np.array(res[upper_dict][key])
    if 'test' in upper_dict:
      train_class_loss = np.array(res[upper_dict][value])
    else:
      train_class_loss = np.array(res[upper_dict][value][best_epoch])
    if len(train_class_labels) == 0:
      print(f'No {value} found in results for {k_fold} - skipping')
    for k,loss in zip(train_class_labels, train_class_loss):
      if k not in dict_grouped_losses[k_fold][key_target_dict]:
        dict_grouped_losses[k_fold][key_target_dict][k] = []
      dict_grouped_losses[k_fold][key_target_dict][k].append(loss)

  
  # Group results by K-Fold, separating final results if present
  dict_grouped_k_fold = {}
  real_k_fold = set([int(key.split('_')[0][1]) for key in data['results'].keys()])
  final_key = [key for key in data['results'].keys() if '_final' in key]
  for k in real_k_fold:
    keys = [key for key in data['results'].keys() if key.startswith(f'k{k}_') and '_final' not in key]
    dict_grouped_k_fold[f'k{k}'] = {} 
    for key in keys:
      dict_grouped_k_fold[f'k{k}'][key] = data['results'][key]
  if final_key != []:
    dict_grouped_k_fold['final'] = {key:data['results'][key] for key in final_key}
  dict_grouped_losses = {}
  
  # Compute grouped losses, per class and subject
  for k_fold,sub_k_dict in dict_grouped_k_fold.items():
    if 'final' in k_fold:
      final_flag = True
      dict_grouped_losses[k_fold] = {
          'subject_train_loss':{},
          'train_loss':[],
          'class_train_loss':{},
          'subject_test_loss':{},
          'class_test_loss':{},
          'test_loss':[],
      }
    else:
      final_flag = False
      dict_grouped_losses[k_fold] = {
          'subject_train_loss':{},
          'train_loss':[],
          'class_train_loss':{},
          'subject_val_loss':{},
          'class_val_loss':{},
          'val_loss':[],
      }
     
    for sub_k, res in sub_k_dict.items():
      if 'train_val' in res:
        best_epoch = res['train_val']['best_model_idx']
        # Get overall train and val loss
        dict_grouped_losses[k_fold]['train_loss'].append(res['train_val']['train_losses'][best_epoch])
      if not final_flag:
        dict_grouped_losses[k_fold]['val_loss'].append(res['train_val']['val_losses'][best_epoch])
      else:
        dict_grouped_losses[k_fold]['test_loss'].append(res['test']['test_loss'])
      if not isinstance(config['criterion'],losses.RESupConLoss):
        # Get train loss per class
        if 'train_val' in res:
          update_dict_grouped_losses(key='train_unique_y',value='train_loss_per_class',key_target_dict='class_train_loss',
                                    k_fold=k_fold, res=res, best_epoch=best_epoch)
        
        # Get val/test loss per class, only if available
        if ('train_val' in res and res['train_val']['count_y_val'] is not None) or final_flag:
          update_dict_grouped_losses(key='val_unique_y' if not final_flag else 'test_unique_y',
                                    value='val_loss_per_class' if not final_flag else 'test_loss_per_class',
                                    key_target_dict='class_val_loss' if not final_flag else 'class_test_loss',
                                    upper_dict= 'train_val' if not final_flag else 'test',
                                    k_fold=k_fold, res=res, best_epoch=best_epoch)      
        
        # Get train loss per subject
        if 'train_val' in res:
          update_dict_grouped_losses(k_fold=k_fold, res=res, best_epoch=best_epoch,
                                    key='train_unique_subject_ids', value='train_loss_per_subject', key_target_dict='subject_train_loss')
        
        # Get val/test loss per subject, only if available
        if ('train_val' in res and res['train_val']['count_y_val'] is not None) or final_flag:
          update_dict_grouped_losses(k_fold=k_fold, res=res, best_epoch=best_epoch,
                                    key='val_unique_subject_ids' if not final_flag else 'test_unique_subject_ids',
                                    value='val_loss_per_subject' if not final_flag else 'test_loss_per_subject',
                                    upper_dict='train_val' if not final_flag else 'test',
                                    key_target_dict='subject_val_loss' if not final_flag else 'subject_test_loss')
      
    # compute the mean loss per class and subject
    for k,v in dict_grouped_losses[k_fold].items():
      if 'loss' in k:
        if len(v) != 0:
          if isinstance(v, list):
            dict_grouped_losses[k_fold][k] = np.mean(v)
          elif isinstance(v, dict):
            for key_inner, list_loss in v.items():
              dict_grouped_losses[k_fold][k][key_inner] = np.mean(list_loss)
  return dict_grouped_losses
        
    
def plot_grouped_k_fold(data, run_output_folder, test_id, additional_info='', plot_type='loss', group_folds=True):
  # Create output folder for grouped K-Fold plots
  grouped_output_folder = os.path.join(run_output_folder, test_id)
  os.makedirs(grouped_output_folder, exist_ok=True)
  # real_k_fold = set([int(key.split('_')[0][1]) for key in data['results'].keys()])

  dict_grouped_losses = get_grouped_losses(data, data['config'])
  is_adversarial_training = True if int(data['config']['head_params'].get('adversarial_head', 0)) == 1 else False
  # Plot grouped losses per subject and class
  is_unbc = 'unbc' in "".join(data['config']['path_csv_dataset']).lower()
  for k_fold, grouped_losses in dict_grouped_losses.items():
    
    # Plot error per subject
    fig, ax = plt.subplots(2,1,figsize=(18,10))
    ax = ax.flatten()
    train_subject_loss = grouped_losses['subject_train_loss']
    val_subject_loss = grouped_losses['subject_val_loss'] if 'subject_val_loss' in grouped_losses else grouped_losses['subject_test_loss']
    
    # Convert UNBC counts to IDs if adversarial training
    if is_adversarial_training and is_unbc:
      train_subject_loss = {helper.unbc_count_to_id[k]: v for k,v in train_subject_loss.items()}
      val_subject_loss = {helper.unbc_count_to_id[k]: v for k,v in val_subject_loss.items()}
       
    unique_subject_ids_train = list(train_subject_loss.keys())
    unique_subject_ids_val = list(val_subject_loss.keys())
    y_lim = 10
    if 'unbc' in "".join(data['config']['path_csv_dataset']).lower():
      y_lim = 3
    elif 'agedb' in "".join(data['config']['path_csv_dataset']).lower():
      y_lim = 20
    tools.plot_error_per_subject(loss_per_subject=[train_subject_loss[k] for k in sorted(train_subject_loss.keys())],
                                  unique_subject_ids=sorted(unique_subject_ids_train),
                                  criterion=data['config']['criterion'],
                                  title=f'Grouped mean TRAIN Loss per Subject - {k_fold} - {test_id}',
                                  ax=ax[0],
                                  y_lim=y_lim)
    title_plt = 'VAL' if 'subject_val_loss' in grouped_losses else 'TEST'
    tools.plot_error_per_subject(loss_per_subject=[val_subject_loss[k] for k in sorted(val_subject_loss.keys())],
                                  unique_subject_ids=sorted(unique_subject_ids_val),
                                  criterion=data['config']['criterion'],
                                  title=f'Grouped mean {title_plt} Loss per Subject - {k_fold} - {test_id}',
                                  ax=ax[1],
                                  y_lim=y_lim)
    fig.tight_layout()
    fig.savefig(os.path.join(grouped_output_folder, f'{test_id}{additional_info}_grouped_loss_per_subject_{k_fold}.png'))
    plt.close(fig)
    
    # Plot error per class
    fig, ax = plt.subplots(2,1,figsize=(18,9))
    train_class_loss = grouped_losses['class_train_loss']
    val_class_loss = grouped_losses['class_val_loss'] if 'class_val_loss' in grouped_losses else grouped_losses['class_test_loss']
    
    tools.plot_error_per_class(mae_per_class=[train_class_loss[k] for k in sorted(train_class_loss.keys())],
                               unique_classes=sorted(train_class_loss.keys()),
                               criterion=data['config']['criterion'],
                               ax=ax[0],
                               title=f'Grouped mean TRAIN Loss per Class - {k_fold} - {test_id}',
                                y_lim=y_lim)
    tools.plot_error_per_class(mae_per_class=[val_class_loss[k] for k in sorted(val_class_loss.keys())],
                                unique_classes=sorted(val_class_loss.keys()),
                                criterion=data['config']['criterion'],
                                ax=ax[1],
                                title=f'Grouped mean {title_plt} Loss per Class - {k_fold} - {test_id}',
                                y_lim=y_lim)
    fig.tight_layout()
    fig.savefig(os.path.join(grouped_output_folder, f'{test_id}{additional_info}_grouped_loss_per_class_{k_fold}.png'))
    plt.close(fig)
    
def plot_CCC_ICC_pearson(data, run_output_folder, test_id, additional_info=''):
  def plot_metric(dict_value_to_plot, metric_name, ax: plt.Axes, best_epoch_idx, title_suffix='', y_lim_bottom = -1, y_lim_top = 1):
    
    for phase in dict_value_to_plot[metric_name]:
      values = dict_value_to_plot[metric_name][phase]
      if isinstance(values, float) or isinstance(values, int):
        ax.plot(best_epoch_idx, values, marker='o', markersize=8, label=f'{phase} (best @ epoch {best_epoch_idx})')
      else:
        ax.plot(values, label=phase)
    ax.set_title(f'{metric_name} over Epochs {title_suffix}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric_name)
    ax.set_ylim([y_lim_bottom, y_lim_top])
    ax.legend()
    return ax
  
  for k in data['results'].keys():
    train_val_results = data['results'][k].get('train_val', {})
    if not train_val_results or 'list_train_ICC' not in train_val_results:
      continue
      
    dict_value_to_plot = {}
    dict_value_to_plot['ICC'] = {'train': train_val_results.get('list_train_ICC', []),
                                 'val': train_val_results.get('list_val_ICC', [])}

    dict_value_to_plot['CCC'] = {'train': train_val_results.get('list_train_CCC', []),
                                 'val': train_val_results.get('list_val_CCC', [])}

    dict_value_to_plot['pearson_correlation'] = {'train': [val for val, pval in train_val_results.get('list_train_pearson_correlation', [])],
                                                 'val': [val for val, pval in train_val_results.get('list_val_pearson_correlation', [])]}
    
    if 'final' in k or 'test' in data['results'][k]:
      test_results = data['results'][k].get('test', {})
      if test_results:
        if 'test_ICC' in test_results and test_results['test_ICC'] is not None:
          dict_value_to_plot['ICC']['test'] = test_results['test_ICC']
        if 'test_CCC' in test_results and test_results['test_CCC'] is not None:
          dict_value_to_plot['CCC']['test'] = test_results['test_CCC']
        pearson_corr = test_results.get('test_pearson_correlation')
        if pearson_corr and len(pearson_corr) > 0:
            dict_value_to_plot['pearson_correlation']['test'] = pearson_corr[0]

    nr_plots = 3
    fig, axs = plt.subplots(nr_plots, figsize=(10, 15))
    axs = axs.flatten()
    nr_epochs = len(train_val_results.get('list_train_ICC', []))
    best_epoch_idx = train_val_results.get('best_model_idx', nr_epochs - 1 if nr_epochs > 0 else 0)

    for i, metric_name in enumerate(['ICC', 'CCC', 'pearson_correlation']):
      if metric_name in dict_value_to_plot and dict_value_to_plot[metric_name]:
        plot_metric(dict_value_to_plot, metric_name, axs[i], best_epoch_idx=best_epoch_idx, title_suffix=f'for {k}')
        
    out_path = os.path.join(run_output_folder, test_id, f'{test_id}{additional_info}_ICC_CCC_Pearson_over_epochs_{k}.png')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_losses(data, run_output_folder, test_id, loss_plot_type,additional_info='', plot_loss_per_subject=True,plot_acc_per_subject=True, plot_loss_per_class=True,plot_train_loss_val_acc=True,**kwargs):
  # Adjust run_output_folder to store plots
  # run_output_folder = Path(run_output_folder).parts[:-3]
  test_output_folder = os.path.join(run_output_folder, test_id)
  loss_plots_output_folder = os.path.join(run_output_folder, 'loss_plots')
  os.makedirs(loss_plots_output_folder, exist_ok=True)
  os.makedirs(test_output_folder, exist_ok=True)
  metric_for_training = "_".join(data['config']['key_for_early_stopping'].split('_')[1:]) # key for early stopping === target_metric_best_model
  
  is_adversarial_training = True if int(data['config']['head_params'].get('adversarial_head', 0)) == 1 else False
  for key in data['results'].keys():
    if 'final' in key:
      only_losses_folder_per_k = os.path.join(loss_plots_output_folder, f'only_losses_final')
    else:
      only_losses_folder_per_k = os.path.join(loss_plots_output_folder, f'only_losses_{key.split("_")[0]}_{key.split("_")[-1]}')
    
    os.makedirs(only_losses_folder_per_k, exist_ok=True)
    # class_subject_loss_folder = os.path.join(run_output_folder, f'class_subject_loss_{key.split("_")[0]}_{key.split("_")[1]}')
    # os.makedirs(class_subject_loss_folder, exist_ok=True)
    if plot_train_loss_val_acc and 'train_val' in data['results'][key]:
      # test_key = 'test_accuracy' if 'test_accuracy' in data['results'][key]['test'] else 'test_macro_precision'
      # test_key = data['results'][key]['test']['dict_precision_recall'][metric_for_training] 
      train_losses = data['results'][key]['train_val'].get('train_losses', [])
      val_loss = data['results'][key]['train_val'].get('val_losses', [])
      if 'list_train_accuracy' in data['results'][key]['train_val'] and 'list_val_accuracy' in data['results'][key]['train_val']:
        train_accuracy = data['results'][key]['train_val'].get('list_train_accuracy', [])
        val_accuracy = data['results'][key]['train_val'].get('list_val_accuracy', [])
      
      ###### TODO: To remove in future
      # elif 'list_train_performance_metric' in data['results'][key]['train_val']:
      #   train_accuracy = data['results'][key]['train_val'].get('list_train_performance_metric', [])
      #   val_accuracy = data['results'][key]['train_val'].get('list_val_performance_metric', [])
      elif 'list_train_macro_accuracy' in data['results'][key]['train_val']:
        train_accuracy = data['results'][key]['train_val'].get('list_train_macro_accuracy', []) # list_train_performance_metric
        val_accuracy = data['results'][key]['train_val'].get('list_val_macro_accuracy', [])
      ######
      
      else:
        raise ValueError('No train accuracy or val accuracy found in the data')
      if ('test' in data['results'][key] and data['results'][key]['test'] != {}) or 'final' in key:
        test_accuracy = data['results'][key]['test']['dict_precision_recall'].get('accuracy', None)
        test_loss = data['results'][key]['test']['test_loss']
        if test_accuracy is not None:
          point_accuracy = {
              'value': test_accuracy,
              'epoch': data['results'][key]['train_val']['best_model_idx']
          }
        else:
          point_accuracy = None
        point_loss = {
            'value':test_loss,
            'epoch':data['results'][key]['train_val']['best_model_idx']
            }
      else:
        point_accuracy = None
        point_loss = None
      grad_norm_mean = data['results'][key]['train_val']['list_mean_total_norm_epoch']
      grad_norm_std = data['results'][key]['train_val']['list_std_total_norm_epoch']
      is_unbc = 'unbc' in "".join(data['config']['path_csv_dataset']).lower()
      if 'list_train_confidence_prediction_right_mean' in data['results'][key]['train_val'] and data['results'][key]['train_val']['list_val_confidence_prediction_right_mean']:
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
        if not is_unbc:
          fig, axs = plt.subplots(2,2,figsize=(20,15))
        else:
          fig, axs = plt.subplots(3,2,figsize=(21,18))
      # Plot train and val loss
      # set global title
      fig.suptitle(f'Losses for {key} - id: {test_id}', fontsize=24,y=0.93,fontstyle='italic',fontweight='bold',color='darkred')
      if train_losses and val_loss:
        xattn_mask = data['config'].get('xattn_mask', [False])
        if xattn_mask is not None and any([xattn_mask]):
          data['config']['xattn_mask'] = torch.any(xattn_mask) # compact view
        else:
          data['config']['xattn_mask'] = False
        
        
        dict_to_string = convert_dict_to_string(filter_dict(data['config']))
        dict_to_string = dict_to_string.replace('head_params', data['config']['head'].value)
            
        dict_to_string = dict_to_string.replace('criterion_dict',f"{type(data['config']['criterion']).__name__}")
        # add test_id in dict_to_string
        dict_to_string += f'\nTest ID: {test_id}'
        dict_to_string += f'\nfold_subfold: {key.split("_")[0]}_{key.split("_")[-1]}'
        y_lim_loss = 20.0
        step_lim = 2.0
        if isinstance(data['config']['criterion'],torch.nn.MSELoss):
          y_lim_loss = 15.1
          step_lim = 3
        elif isinstance(data['config']['criterion'],losses.CDW_CELoss):
          y_lim_loss = 35.1
          step_lim = 6
        elif isinstance(data['config']['criterion'],losses.RESupConLoss):
          y_lim_loss = 20.1
          step_lim = 3
        elif isinstance(data['config']['criterion'],losses.RnCLossV2):
          y_lim_loss = 10.1
          step_lim = 1
        elif isinstance(data['config']['criterion'],losses.DisentangledLoss):
          y_lim_loss = 15.1
          step_lim = 2
        # y_lim_loss = 15.1 if isinstance(data['config']['criterion'],torch.nn.MSELoss) else 5.1
        x_lim_loss = -1.1 if isinstance(data['config']['criterion'],losses.RESupConLoss) else 0
        
        # Plot accuracy gap and dataset distribution if not UNBC dataset
        if not is_unbc: # Biovid
          # input_dict_accuracy_gap={
          #   'list_1': np.array(train_accuracy) - np.array(val_accuracy),
          #   'title':f'Accuracy Gap in Train-Validation',
          #   'ax':axs[1][0],
          #   'x_label':'Epochs',
          #   'y_label_1':f'Accuracy Gap',
          #   'y_lim_1':[-1, 1],
          #   'color_1':'tab:orange',
          # }
          
          y_lim_error = 20 if isinstance(data['config']['criterion'],torch.nn.MSELoss) else 10
          create_lr_wd_plot(data['results'][key],
                            test_id,
                            key,
                            title=f'Learning Rate and Weight Decay Across Epochs',
                            ax1=axs[1][0])
          input_dict_loss_acc= {
          'list_1':train_losses,
          'list_2':val_accuracy,
          'output_path':None,
          'title':f'Train loss, validation accuracy, test accuracy',
          'point':point_accuracy,
          'ax':axs[0][1],
          'x_label':'Epochs',
          'y_label_1':'Train loss',
          'y_label_2':'Validation accuracy',
          'y_label_3':'Test accuracy',
          'y_lim_1':[0, y_lim_loss],
          'y_lim_2':[0, 0.5],
          'y_lim_3':[0, 0.5],
          'step_ylim_1':step_lim,
          'step_ylim_2':0.05,
          'step_ylim_3':0.05,
          'dict_to_string':None,
          'color_1':'tab:red',
          'color_2':'tab:blue',
          'color_3':'tab:green',
        }
        
          # tools.plot_losses_and_test_new(**input_dict_accuracy_gap)
          tools.plot_losses_and_test_new(**input_dict_loss_acc)
        else:
          if loss_plot_type == 'dist':
            dict_count_labels_val = data['results'][key]['train_val']['count_y_val'] if not 'final' in key else dict(zip(data['results'][key]['test']['test_unique_y'].numpy(), data['results'][key]['test']['test_count_y']))
            dict_count_sbjs_val = data['results'][key]['train_val']['count_subject_ids_val'] if not 'final' in key else dict(zip(data['results'][key]['test']['test_unique_subject_ids'], data['results'][key]['test']['test_count_subject_ids']))
            y_lim_train = 40
            y_limit_val = 20
            tools.plot_distribution(
              data_dict=data['results'][key]['train_val']['count_y_train'],
              x_label='Pain intensity labels',
              title='Dataset distribution on TRAIN set - Pain intensity labels',
              color='blue',
              y_lim=y_lim_train,
              show_missing=(0,10),
              ax=axs[1][0],
            )
            tools.plot_distribution(
              data_dict=data['results'][key]['train_val']['count_subject_ids_train'],
              x_label='Subject IDs',
              title='Dataset distribution on TRAIN set - Subject IDs',
              color='green',
              y_lim=y_lim_train,
              ax=axs[2][0],
            )
            tools.plot_distribution(
              data_dict=dict_count_labels_val,
              x_label='Pain intensity labels',
              title=f'Dataset distribution on {"TEST" if "final" in key else "VAL"} set - Pain intensity labels',
              color='blue',
              y_lim=y_limit_val,
              show_missing=(0,10),
              ax=axs[1][1],
            )
            tools.plot_distribution(
            data_dict=dict_count_sbjs_val,
            x_label='Subject IDs',
            title=f'Dataset distribution on {"TEST" if "final" in key else "VAL"} set - Subject IDs',
            color='green',
            y_lim=y_limit_val,
            ax=axs[2][1],
          )
          elif loss_plot_type == 'loss': # grouped loss per subject,class in train, val (or test) + lr schedule
            # axs[0][1]: lr schedule
            y_lim_error = 20 if isinstance(data['config']['criterion'],torch.nn.MSELoss) else 10
            create_lr_wd_plot(data['results'][key],
                              test_id,
                              key,
                              title=f'Learning Rate and Weight Decay Across Epochs',
                              ax1=axs[0][1])
            
            # axs[1][0]: loss per subject train
            best_epoch = data['results'][key]['train_val']['best_model_idx']
            loss_per_subject_train = data['results'][key]['train_val'].get('train_loss_per_subject', None)
            unique_subject_ids_train=data['results'][key]['train_val']['train_unique_subject_ids']
            
            if is_adversarial_training and is_unbc:
              unique_subject_ids_train = helper.convert_unbc_count_to_id(unique_subject_ids_train)
                
            tools.plot_error_per_subject(loss_per_subject=loss_per_subject_train[best_epoch],
                                        unique_subject_ids=unique_subject_ids_train,
                                        title=f'TRAIN error per subject - Epoch_{best_epoch} ',
                                        criterion=data['config']['criterion'],
                                        list_stoic_subject=None,
                                        y_lim=y_lim_error,
                                        ax=axs[1][0])
            
            # axs[1][1]: loss per class train
            loss_per_class_train = data['results'][key]['train_val'].get('train_loss_per_class', None)
            tools.plot_error_per_class(mae_per_class=loss_per_class_train[best_epoch],
                                       unique_classes=data['results'][key]['train_val']['train_unique_y'],
                                       criterion=data['config']['criterion'],
                                       ax=axs[1][1],
                                       y_lim=y_lim_error,
                                       title=f'TRAIN error per class - Epoch_{best_epoch}')
            
            # axs[2][0]: loss per subject val (or test)
            dict_per_subject_test = data['results'][key].get('test', None)
            if dict_per_subject_test is None:
              loss_per_subject_val = data['results'][key]['train_val'].get('val_loss_per_subject', None)
              unique_subject_ids_val=data['results'][key]['train_val']['val_unique_subject_ids']
              
              if is_adversarial_training and is_unbc:
                unique_subject_ids_val = helper.convert_unbc_count_to_id(unique_subject_ids_val)
              
              tools.plot_error_per_subject(loss_per_subject=loss_per_subject_val[best_epoch],
                                          unique_subject_ids=unique_subject_ids_val,
                                          title=f'VAL error per subject - Epoch_{best_epoch} ',
                                          criterion=data['config']['criterion'],
                                          list_stoic_subject=None,
                                          y_lim=y_lim_error,
                                          ax=axs[2][0])
            else:
              unique_subject_ids_test=dict_per_subject_test['test_unique_subject_ids']
              if is_adversarial_training and is_unbc:
                unique_subject_ids_test = helper.convert_unbc_count_to_id(unique_subject_ids_test)
            
              tools.plot_error_per_subject(loss_per_subject=dict_per_subject_test['test_loss_per_subject'],
                                          unique_subject_ids=unique_subject_ids_test,
                                          title=f'TEST error per subject - Epoch_{best_epoch} ',
                                          criterion=data['config']['criterion'],
                                          list_stoic_subject=None,
                                          y_lim=y_lim_error,
                                          ax=axs[2][0])
        
              
            # axs[2][1]: loss per class val (or test)
            dict_per_subject_test = data['results'][key].get('test', None)
            if dict_per_subject_test is None:
              loss_per_class_val = data['results'][key]['train_val'].get('val_loss_per_class', None)
              tools.plot_error_per_class(mae_per_class=loss_per_class_val[best_epoch],
                                         unique_classes=data['results'][key]['train_val']['val_unique_y'],
                                         criterion=data['config']['criterion'],
                                         ax=axs[2][1],
                                         y_lim=y_lim_error,
                                         title=f'VAL error per class - Epoch_{best_epoch}')
            else:
              tools.plot_error_per_class(mae_per_class=dict_per_subject_test['test_loss_per_class'],
                                        unique_classes=dict_per_subject_test['test_unique_y'],
                                        criterion=data['config']['criterion'],
                                        ax=axs[2][1],
                                        y_lim=y_lim_error,
                                        title=f'TEST error per class - Epoch_{best_epoch}')
              
            
        # data['results']['k0_cross_val_sub_0']['train_val']['count_y_train']
        # data['results']['k0_cross_val_sub_0']['train_val']['count_y_val']
        # data['results']['k0_cross_val_sub_0']['train_val']['count_subject_ids_train']
        # data['results']['k0_cross_val_sub_0']['train_val']['count_subject_ids_val']
        # Plot dataset distribution on train and val set per class and subject
        
        # Plot GRADIENT
        if (grad_norm_mean > 0).any():
          tools.plot_with_std(ax=axs[1][1],
                              x=list(range(len(grad_norm_mean))),
                              mean=grad_norm_mean,
                              std=grad_norm_std,
                              title='Mean and std of GRADIENT norm',
                              x_label='Epochs',
                              y_label='Gradient norm',
                              y_lim=[0, 100],
                              cap_line=data['config']['clip_grad_norm'] if data['config']['clip_grad_norm'] else None,
                              )
          l2_norm_grad_task = data['results'][key]['train_val']['grad_task_norm_epoch']
          l2_norm_grad_adv = data['results'][key]['train_val']['grad_adv_norm_epoch']
          if l2_norm_grad_adv != [] and l2_norm_grad_task != []:
            fig_adv, axs_adv = plt.subplots(1,1,figsize=(20,10))
            # axs_adv = axs_adv.flatten()
            axs_adv.plot(l2_norm_grad_task,  label=f"Gradient of {data['config']['criterion']}")
            axs_adv.plot(l2_norm_grad_adv, label="Gradient of CE Identity Aware Network")

            axs_adv.set_xlabel("Epochs")
            axs_adv.set_ylabel("Value")
            axs_adv.legend()

            fig_adv.suptitle("Separated gradient Norm of the losses during training")
            fig_adv.tight_layout()
            fig_adv.savefig(os.path.join(test_output_folder, f'{test_id}_separated_gradient_losses_{key}.png'))
            plt.close(fig_adv)
            
        # Plot prediction confidence if available
        if 'list_train_confidence_prediction_right_mean' in data['results'][key]['train_val'] and data['results'][key]['train_val']['list_val_confidence_prediction_right_mean']:
          tools.plot_with_std(ax=axs[2][0],x=list(range(len(train_prediction_confidence_right_mean))),
                              mean=train_prediction_confidence_right_mean,
                              std=train_prediction_confidence_right_std,
                              title='TRAIN prediction confidence',
                              x_label='Epochs',
                              color='green',
                              y_label='Prediction confidence',
                              legend_label_mean='Correct pred. mean',
                              legend_label_std='Correct pred. std',
                              y_step=0.1,
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
                              y_lim=[0, 1],
                              y_step=0.1,)
          
          
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
                              y_lim=[0, 1],
                              y_step=0.1,)
          
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
                              y_lim=[0, 1],
                              y_step=0.1,)
        
        
        input_dict_loss={
          'list_1':train_losses,
          'list_2':val_loss,
          'output_path':None,
          'title':'Train loss + validation and test loss',
          'point':point_loss,
          'ax':axs[0][0],
          'x_label':'Epochs',
          'y_label_1':'Train loss',
          'y_label_2':'Validation loss',
          'y_label_3':'Test loss',
          'y_lim_1':[x_lim_loss, y_lim_loss],
          'y_lim_2':[x_lim_loss, y_lim_loss],
          'y_lim_3':[x_lim_loss, y_lim_loss],
          'step_ylim_1':step_lim,
          'step_ylim_2':step_lim,
          'step_ylim_3':step_lim,
          'dict_to_string':None,
          'color_1':'tab:red',
          'color_2':'tab:purple',
          'color_3':(0,0,0), # (r,g,b)
        }
        tools.plot_losses_and_test_new(**input_dict_loss) # ax[0][0]
        axs[0][1].text(1.14, -0.5, dict_to_string, fontsize=12, color='black',transform=axs[0][1].transAxes,ha='left', va='center')
        plot_path = os.path.join(test_output_folder, f'{test_id}{additional_info}_losses_{key}.png')
        fig.savefig(plot_path, bbox_inches='tight')
        symlink_path = os.path.join(only_losses_folder_per_k, f'{test_id}{additional_info}_losses_{key}.png')
        # Remove the link if it already exists to avoid errors
        try:
          os.remove(symlink_path)
        except FileNotFoundError:
          pass
        # Create a new symlink
        os.symlink(plot_path, symlink_path)
        
        # Create and save plot for test_as_validation if applicable
        if kwargs.get('test_as_validation', False):
          tmp = data['results'][key]['train_val'].get('test_as_eval', None)
          if tmp is not None:
            input_dict_loss['point'] = tmp['test_losses'] # list of test losses per epoch
            tools.plot_losses_and_test_new(**input_dict_loss) # ax[0][0]
            plot_path = os.path.join(test_output_folder, f'{test_id}{additional_info}_losses_test_as_validation_{key}.png')
            fig.savefig(plot_path, bbox_inches='tight')
            symlink_path = os.path.join(only_losses_folder_per_k, f'{test_id}{additional_info}_losses_test_as_validation_{key}.png')
            # Remove the link if it already exists to avoid errors
            try:
              os.remove(symlink_path)
            except FileNotFoundError:
              pass
            # Create a new symlink
            os.symlink(plot_path, symlink_path)
        ###########
        plt.close(fig)

      if not isinstance(data['config']['criterion'],losses.RESupConLoss) and plot_loss_per_subject and len(data['results'][key]['train_val']['list_val_accuracy_per_subject']) > 0:
        y_lim = 10 if 'unbc' in "".join(data['config']['path_csv_dataset']).lower() else 3
        
        loss_per_subject_train = data['results'][key]['train_val'].get('train_loss_per_subject', None)
        loss_per_subject_val = data['results'][key]['train_val'].get('val_loss_per_subject', None)
        loss_per_subject_val = loss_per_subject_val if (loss_per_subject_val != None).all() else None
        dict_per_subject_test = data['results'][key].get('test', None)
        # dict_per_subject_test = data['results'][key.split('_')[0]].get('test', None)
        loss_list = [loss_per_subject_train, loss_per_subject_val, dict_per_subject_test]
        total_plots = sum([1 for loss in loss_list if loss is not None])
        fig, axs = plt.subplots(total_plots, 1, figsize=(20, 20))
        count_axs = 0
        best_epoch = data['results'][key]['train_val']['best_model_idx']
        
        # Train Loss
        if loss_per_subject_train is not None:
          unique_subject_ids_train=data['results'][key]['train_val']['train_unique_subject_ids']
          if is_adversarial_training and is_unbc:
            unique_subject_ids_train = helper.convert_unbc_count_to_id(unique_subject_ids_train)
          tools.plot_error_per_subject(loss_per_subject=loss_per_subject_train[best_epoch],
                                        unique_subject_ids=unique_subject_ids_train,
                                        title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                        criterion=data['config']['criterion'],
                                        list_stoic_subject=helper.stoic_subjects,
                                        y_lim=y_lim,
                                        ax=axs[count_axs] if total_plots > 1 else axs)
          count_axs += 1
        # Val Loss
        if loss_per_subject_val is not None:
          unique_subject_ids_val=data['results'][key]['train_val']['val_unique_subject_ids']
          if is_adversarial_training and is_unbc:
            unique_subject_ids_val = helper.convert_unbc_count_to_id(unique_subject_ids_val)
          tools.plot_error_per_subject(loss_per_subject=loss_per_subject_val[best_epoch],
                                      unique_subject_ids=unique_subject_ids_val,
                                      title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                      criterion=data['config']['criterion'],
                                      list_stoic_subject=helper.stoic_subjects,
                                      y_lim=y_lim,
                                      ax=axs[count_axs])
          count_axs += 1
        # Test Loss
        if dict_per_subject_test is not None and dict_per_subject_test != {} :
          unique_subject_ids_test=dict_per_subject_test['test_unique_subject_ids']
          if is_adversarial_training and is_unbc:
            unique_subject_ids_test = helper.convert_unbc_count_to_id(unique_subject_ids_test)
          tools.plot_error_per_subject(loss_per_subject=dict_per_subject_test['test_loss_per_subject'],
                                    unique_subject_ids=dict_per_subject_test['test_unique_subject_ids'],
                                    title=f'TEST {key} - {test_id}',
                                    criterion=data['config']['criterion'],
                                    list_stoic_subject=helper.stoic_subjects,
                                    y_lim=y_lim,
                                    ax=axs[count_axs])

        fig.tight_layout()
        fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_loss_per_subject_{key}.png'))
        plt.close(fig)
      if not is_unbc and not isinstance(data['config']['criterion'],losses.RESupConLoss) and plot_acc_per_subject and len(data['results'][key]['train_val']['train_loss_per_subject']) > 0:
        y_lim = 1
        accuracy_per_subject_train = data['results'][key]['train_val'].get('list_train_accuracy_per_subject', None)
        accuracy_per_subject_val = data['results'][key]['train_val'].get('list_val_accuracy_per_subject', None)
        accuracy_per_subject_val = accuracy_per_subject_val if accuracy_per_subject_val != None else None
        dict_per_subject_test = data['results'][key].get('test', None)
        acc_list = [accuracy_per_subject_train, accuracy_per_subject_val, dict_per_subject_test]
        total_plots = sum([1 for acc in acc_list if acc is not None])
        count_axs = 0
        fig, axs = plt.subplots(total_plots,1,figsize=(20,20))
        best_epoch = data['results'][key]['train_val']['best_model_idx']
        # Train accuracy
        if accuracy_per_subject_train is not None:
          tools.plot_error_per_subject(loss_per_subject=accuracy_per_subject_train[best_epoch],
                                       unique_subject_ids=data['results'][key]['train_val']['train_unique_subject_ids'],
                                       y_label='Train Accuracy',
                                       title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                       criterion=data['config']['criterion'],
                                       list_stoic_subject=helper.stoic_subjects,
                                       bar_color='blue',
                                       step_y_axis=0.1, 
                                       y_lim=y_lim,
                                       ax=axs[count_axs])
          count_axs += 1
        # Val accuracy
        if accuracy_per_subject_val is not None:
          tools.plot_error_per_subject(loss_per_subject=accuracy_per_subject_val[best_epoch],
                                      unique_subject_ids=data['results'][key]['train_val']['val_unique_subject_ids'],
                                      y_label='Val Accuracy',
                                      title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                      criterion=data['config']['criterion'],
                                      step_y_axis=0.1,
                                      list_stoic_subject=helper.stoic_subjects,
                                      bar_color='blue',
                                      y_lim=y_lim,
                                      ax=axs[count_axs])
          count_axs += 1
        # Test accuracy
        if dict_per_subject_test is not None and dict_per_subject_test != {}:
          tools.plot_error_per_subject(loss_per_subject=dict_per_subject_test['test_accuracy_per_subject'],
                                unique_subject_ids=data['results'][key]['test']['test_unique_subject_ids'],
                                y_label='Test Accuracy',
                                title=f'TEST Epoch_{best_epoch} {key} - {test_id}',
                                criterion=data['config']['criterion'],
                                step_y_axis=0.1,
                                list_stoic_subject=helper.stoic_subjects,
                                bar_color='blue',
                                y_lim=y_lim,
                                ax=axs[count_axs])
        fig.tight_layout()
        fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_acc_per_subject_{key}.png'))
        plt.close(fig)
        
      if plot_loss_per_class and not isinstance(data['config']['criterion'],losses.RESupConLoss):
        y_lim = 10 if 'unbc' in "".join(data['config']['path_csv_dataset']).lower() else 3
        train_loss_per_class = data['results'][key]['train_val'].get('train_loss_per_class', None)
        val_loss_per_class = data['results'][key]['train_val'].get('val_loss_per_class', None)
        val_loss_per_class = val_loss_per_class if (val_loss_per_class != None).all() else None
        accuracy_per_class_test = data['results'][key].get('test', None)
        class_loss_list = [train_loss_per_class, val_loss_per_class, accuracy_per_class_test]
        total_plots = sum([1 for class_loss in class_loss_list if class_loss is not None])
        fig, axs = plt.subplots(total_plots,1,figsize=(10,8))
        count_axs = 0
        best_epoch = data['results'][key]['train_val']['best_model_idx']
        # Train Loss
        if train_loss_per_class is not None:
          df = pd.DataFrame({'class': data['results'][key]['train_val']['train_unique_y'], 'mae': train_loss_per_class[best_epoch]})
          df = df.sort_values('class')
          df.to_csv(os.path.join(test_output_folder, f'{test_id}{additional_info}_train_loss_per_class_{key}.csv'), index=False)
          tools.plot_error_per_class(unique_classes=data['results'][key]['train_val']['train_unique_y'],
                                      mae_per_class=train_loss_per_class[best_epoch],
                                      title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                      criterion=data['config']['criterion'],
                                      # accuracy_per_class=data['results'][key]['train_val']['list_train_accuracy_per_class'][best_epoch],
                                      y_lim=y_lim,
                                      ax=axs[count_axs])
          count_axs += 1
        # Val Loss
        if val_loss_per_class is not None and val_loss_per_class[0] is not None and np.sum(val_loss_per_class[0]) != 0:
          if data['results'][key]['train_val']['val_unique_y'].shape[0] != val_loss_per_class[best_epoch].shape[0]:
            raise ValueError('Number of unique classes does not match number of classes in val_loss_per_class')
          df = pd.DataFrame({'class': data['results'][key]['train_val']['val_unique_y'], 'mae': val_loss_per_class[best_epoch]})
          df = df.sort_values('class')
          df.to_csv(os.path.join(test_output_folder, f'{test_id}{additional_info}_val_loss_per_class_{key}.csv'), index=False)
          tools.plot_error_per_class(unique_classes=data['results'][key]['train_val']['val_unique_y'],
                                    mae_per_class=val_loss_per_class[best_epoch],
                                    title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                    criterion=data['config']['criterion'],
                                    # accuracy_per_class=data['results'][key]['train_val']['list_val_accuracy_per_class'][best_epoch],
                                    y_lim=y_lim,
                                    ax=axs[count_axs])
          count_axs += 1
        # Test Loss
        if accuracy_per_class_test is not None and accuracy_per_class_test != {} and isinstance:
          try:
            df = pd.DataFrame({'class': data['results'][key]['test']['test_unique_y'], 'mae': accuracy_per_class_test['test_loss_per_class']})
            df = df.sort_values('class')
            df.to_csv(os.path.join(test_output_folder, f'{test_id}{additional_info}_test_loss_per_class_{key}.csv'), index=False)
            tools.plot_error_per_class(mae_per_class=accuracy_per_class_test['test_loss_per_class'],
                                    unique_classes=data['results'][key]['test']['test_unique_y'],
                                    title=f'TEST {key} - {test_id}',
                                    criterion=data['config']['criterion'],
                                    # accuracy_per_class=data['results'][key]['test']['test_accuracy_per_class'],
                                    y_lim=y_lim,
                                    ax=axs[count_axs])
          except Exception as e:
            print(f"Error plotting test loss per class for {key} - {test_id}: {e}")
        fig.tight_layout()
        fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_mae_per_class_{key}.png'))
        plt.close(fig)
        
    if not is_unbc and plot_loss_per_class and not isinstance(data['config']['criterion'],losses.RESupConLoss):
        y_lim = 10 if 'unbc' in "".join(data['config']['path_csv_dataset']).lower() else 3
        train_accuracy_per_class = data['results'][key]['train_val']['list_train_accuracy_per_class'][best_epoch]
        val_accuracy_per_class = data['results'][key]['train_val']['list_val_accuracy_per_class'][best_epoch]
        dict_test = data['results'][key].get('test', None)
        class_loss_list = [train_accuracy_per_class, val_accuracy_per_class, dict_test]  
        total_plots = sum([1 for class_loss in class_loss_list if class_loss is not None])
        fig, axs = plt.subplots(total_plots,1,figsize=(10,8))
        count_axs = 0
        best_epoch = data['results'][key]['train_val']['best_model_idx']
        # Train Loss
        if train_accuracy_per_class is not None:
          tools.plot_error_per_class(unique_classes=data['results'][key]['train_val']['train_unique_y'],
                                      # mae_per_class=train_loss_per_class[best_epoch],
                                      title=f'TRAIN Epoch_{best_epoch} {key} - {test_id}',
                                      criterion=data['config']['criterion'],
                                      accuracy_per_class=train_accuracy_per_class,
                                      y_lim=y_lim,
                                      ax=axs[count_axs])
          count_axs += 1
        # Val Loss
        if val_accuracy_per_class is not None and val_accuracy_per_class[0] is not None and val_accuracy_per_class[0].reshape(-1).sum() != 0:
          if data['results'][key]['train_val']['val_unique_y'].shape[0] != val_loss_per_class[best_epoch].shape[0]:
            
            raise ValueError('Number of unique classes does not match number of classes in val_loss_per_class')
          tools.plot_error_per_class(unique_classes=data['results'][key]['train_val']['val_unique_y'],
                                    # mae_per_class=val_loss_per_class[best_epoch],
                                    title=f'VAL Epoch_{best_epoch} {key} - {test_id}',
                                    criterion=data['config']['criterion'],
                                    accuracy_per_class=val_accuracy_per_class,
                                    y_lim=y_lim,
                                    ax=axs[count_axs])
          count_axs += 1
        # Test Loss
        if dict_test is not None and dict_test != {}:
          try:
            tools.plot_error_per_class(unique_classes=data['results'][key]['test']['test_unique_y'],
                                    # mae_per_class=dict_per_class_test['test_loss_per_class'],
                                    title=f'TEST {key} - {test_id}',
                                    criterion=data['config']['criterion'],
                                    accuracy_per_class=dict_test['test_accuracy_per_class'],
                                    y_lim=y_lim,
                                    ax=axs[count_axs])
          except Exception as e:
            print(f"Error plotting test accuracy per class for {key} - {test_id}: {e}")
        fig.tight_layout()
        fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_accuracy_per_class_{key}.png'))
        plt.close(fig)

def plot_gradient_per_module(data, run_output_folder, test_id, additional_info='',):
  
  def get_total_mean_per_epoch(filtered_dict):
    total_grad_mean_per_epoch = []
    for k,_ in filtered_dict.items():
      total_grad_mean_per_epoch.append([v['mean'] for v in filtered_dict[k]])
    total_grad_mean_per_epoch = np.sum(total_grad_mean_per_epoch, axis=0)
    return total_grad_mean_per_epoch
  
  def plot_gradient_distrib(ax,percentage_grad_per_module,title,xlabel,ylabel):
    for k,v in percentage_grad_per_module.items():
      list_x = list(range(len(v)))
      label = k
      ax.plot(list_x, v, label=label)
      ax.legend()
      ax.set_ylim(0, 1)
      ax.set_title(title)
      ax.set_xlabel(xlabel)
      ax.set_ylabel(ylabel)  
    
      
  for key,_ in data['results'].items():
    if 'epochs_gradient_per_module' in data['results'][key]['train_val'] and data['results'][key]['train_val']['epochs_gradient_per_module']:
      epochs_gradient_per_module = data['results'][key]['train_val']['epochs_gradient_per_module']
      group_elements = ['linear','pooler.cross_attention_block','pooler.blocks','query_tokens']
      test_output_folder = os.path.join(run_output_folder, test_id)
      # loss_plots_output_folder = os.path.join(run_output_folder, 'loss_plots')
      total_grad_per_epoch = get_total_mean_per_epoch(epochs_gradient_per_module)
      total_grad_module_per_epoch = []
      for query in group_elements:
        if query == 'linear' or query == 'query_tokens':
          fig,axs = plt.subplots(1,1,figsize=(10,5))
          filtered_dict_mlp = {k:v for k,v in epochs_gradient_per_module.items() if query in k}
          total_grad_mean_per_epoch = get_total_mean_per_epoch(filtered_dict=filtered_dict_mlp)
          total_grad_module_per_epoch.append({query:total_grad_mean_per_epoch})
          tools.plot_with_std(ax=axs,
                              x_label='Epochs',
                              y_label='Gradient norm',
                              x=list(range(len(total_grad_mean_per_epoch))),
                              mean=total_grad_mean_per_epoch,
                              std=np.zeros(len(total_grad_mean_per_epoch)),title=f'Gradient norm {query}')
          fig.tight_layout()
          fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_gradient_per_module_{key}_{query}.png'))
        elif query == 'pooler.cross_attention_block':
          fig,axs = plt.subplots(1,2,figsize=(20,5))
          filtered_dict_mlp = {k:v for k,v in epochs_gradient_per_module.items() if query in k and 'mlp' in k}
          filtered_dict_not_mlp = {k:v for k,v in epochs_gradient_per_module.items() if query in k and 'mlp' not in k}
          total_grad_mean_per_epoch_mlp = get_total_mean_per_epoch(filtered_dict=filtered_dict_mlp)
          total_grad_mean_per_epoch_not_mlp = get_total_mean_per_epoch(filtered_dict=filtered_dict_not_mlp)
          total_grad_module_per_epoch.append({query+'.mlp':total_grad_mean_per_epoch_mlp})
          total_grad_module_per_epoch.append({query+'.NOT_mlp':total_grad_mean_per_epoch_not_mlp})
          tools.plot_with_std(ax=axs[0],
                              x_label='Epochs',
                              y_label='Gradient norm',
                              x=list(range(len(total_grad_mean_per_epoch_mlp))),
                              mean=total_grad_mean_per_epoch_mlp,
                              std=np.zeros(len(total_grad_mean_per_epoch_mlp)),title=f'Gradient norm {query}.mlp')
          tools.plot_with_std(ax=axs[1],
                              x_label='Epochs',
                              y_label='Gradient norm',
                              x=list(range(len(total_grad_mean_per_epoch_not_mlp))),
                              mean=total_grad_mean_per_epoch_not_mlp,
                              std=np.zeros(len(total_grad_mean_per_epoch_not_mlp)),title=f'Gradient norm {query}.NOT_mlp')
          fig.tight_layout()
          fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_gradient_per_module_{key}_{query}.png'))
        elif query == 'pooler.blocks':
          keys = [k for k in epochs_gradient_per_module.keys() if query in k]
          if len(keys) == 0:
            continue
          nr_blocks = int(keys[-1].split('.')[2])
          for idx_block in range(nr_blocks+1):
            fig,axs = plt.subplots(1,2,figsize=(10,5))
            filtered_dict_mlp = {k:v for k,v in epochs_gradient_per_module.items() if query in k and f'.{idx_block}.' in k and 'mlp' in k}
            filtered_dict_not_mlp = {k:v for k,v in epochs_gradient_per_module.items() if query in k and f'.{idx_block}.' in k and 'mlp' not in k}
            total_grad_mean_per_epoch_mlp = get_total_mean_per_epoch(filtered_dict=filtered_dict_mlp)
            total_grad_mean_per_epoch_not_mlp = get_total_mean_per_epoch(filtered_dict=filtered_dict_not_mlp)
            total_grad_module_per_epoch.append({query+f'.{idx_block}.mlp':total_grad_mean_per_epoch_mlp})
            total_grad_module_per_epoch.append({query+f'.{idx_block}.NOT_mlp':total_grad_mean_per_epoch_not_mlp})
            tools.plot_with_std(ax=axs[0],
                                x_label='Epochs',
                                y_label='Gradient norm',
                                x=list(range(len(total_grad_mean_per_epoch_mlp))),
                                mean=total_grad_mean_per_epoch_mlp,
                                std=np.zeros(len(total_grad_mean_per_epoch_mlp)),title=f'Gradient norm {query}.{idx_block}.MLP')
            tools.plot_with_std(ax=axs[1],
                                x_label='Epochs',
                                y_label='Gradient norm',
                                x=list(range(len(total_grad_mean_per_epoch_not_mlp))),
                                mean=total_grad_mean_per_epoch_not_mlp,
                                std=np.zeros(len(total_grad_mean_per_epoch_not_mlp)),title=f'Gradient norm {query}.{idx_block}.NOT_mlp')
            fig.tight_layout()
            fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_gradient_per_module_{key}_{query}.{idx_block}.png'))
      
      percentage_grad_per_module = {}
      sum_grad_per_module = []
      for dict_data in total_grad_module_per_epoch:
        for k,v in dict_data.items():
          values = np.array(v)
          sum_grad_per_module.append(values)
          percentage_grad_per_module[k] = np.array(v) / np.array(total_grad_per_epoch)
      sum_grad_per_module = np.sum(sum_grad_per_module, axis=0)
      diff = np.abs(sum_grad_per_module - total_grad_per_epoch)
      if np.any(diff > 1e-8):
        raise ValueError('Sum of gradients per module is not equal to total gradient per epoch')
      
      fig, ax = plt.subplots(figsize=(10, 5))
      plot_gradient_distrib(ax=ax,
                            percentage_grad_per_module=percentage_grad_per_module,
                            title='Gradient distribution per module in percentage',
                            xlabel='Epochs',
                            ylabel='Gradient norm in percentage')
      fig.tight_layout()
      fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_gradient_distrib_{key}.png'))
      plt.close()


def plot_history_model_prediction(data, run_output_folder, test_id, root_csv_path):
  test_output_folder = os.path.join(run_output_folder, test_id)
  os.makedirs(test_output_folder, exist_ok=True)
  list_file_root_csv = os.listdir(root_csv_path)
  csv_file = [f for f in list_file_root_csv if f.endswith('.csv')][0]
  df = pd.read_csv(os.path.join(root_csv_path,csv_file),sep='\t')
  top_k = 20
  for key in data['results'].keys():
    if (not 'train_val' in data['results'][key]) or data['results'][key]['train_val']['history_train_sample_predictions'] is None:
      continue
    best_epoch = data['results'][key]['train_val']['best_model_idx']
    num_epochs = len(data['results'][key]['train_val']['train_losses'])
    train_history_pred = data['results'][key]['train_val']['history_train_sample_predictions']
    val_history_pred = data['results'][key]['train_val']['history_val_sample_predictions']
    miss_predictions_train_label,miss_predictions_train_sbj = tools.count_mispredictions(train_history_pred,df,top_k=top_k,return_miss_per_subject=True)
    miss_predictions_val_label,miss_predictions_val_sbj = tools.count_mispredictions(val_history_pred,df,top_k=top_k,return_miss_per_subject=True)
    
    # TOP-K TRAIN and VAL missprediction over epochs
    fig, ax = plt.subplots(2,1,figsize=(10, 10))
    tools.plot_bar(data=miss_predictions_train_label,
                   ax_=ax[0],
                   title=f'TRAIN Sample misclassification over {num_epochs} epochs {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                   x_label='Sample ID',
                   y_label='Number of mispredictions over epochs',
                   color='red')
    tools.plot_bar(data=miss_predictions_val_label,
                   ax_=ax[1],
                   title=f'VAL Sample misclassification over {num_epochs} epochs {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                   x_label='Sample ID',
                   y_label='Number of mispredictions over epochs',
                   color='red')
    fig.savefig(os.path.join(test_output_folder, f'{test_id}_missclassification_train_val_epochs_{key}.png'), bbox_inches='tight')
    plt.close(fig)
    
    # TOP-K SUBJECTS MISSPREDICTIONS over epochs TRAIN and VAL
    fig, ax = plt.subplots(2,1,figsize=(10, 10))
    tools.plot_bar(data=miss_predictions_train_sbj,
                   ax_=ax[0],
                   title=f'TRAIN Subject misclassification over {num_epochs} epochs {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                   x_label='Subject ID',
                   y_label='Number of mispredictions over epochs',
                   list_stoic_subject=helper.stoic_subjects,
                   color='orange')
    tools.plot_bar(data=miss_predictions_val_sbj,
                    ax_=ax[1],
                    title=f'VAL Subject misclassification over {num_epochs} epochs - {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                    x_label='Subject ID',
                    list_stoic_subject=helper.stoic_subjects,
                    y_label='Number of mispredictions over epochs',
                    color='orange')
    fig.savefig(os.path.join(test_output_folder, f'{test_id}_missclassification_train_val_subjects_{key}.png'), bbox_inches='tight')
    plt.close(fig)
    
    # TOP-K TRAIN, VAL and TEST subject missprediction over epochs
    miss_predictions_train_sbj_best_epoch = {k:v[best_epoch] for k,v in train_history_pred.items()}
    miss_predictions_val_sbj_best_epoch = {k:v[best_epoch] for k,v in val_history_pred.items()}
    test_history_pred = data['results'][key]['test']['history_test_sample_predictions']
    _,miss_predictions_train_sbj = tools.count_mispredictions(miss_predictions_train_sbj_best_epoch,df,top_k=top_k,return_miss_per_subject=True)
    _,miss_predictions_val_sbj = tools.count_mispredictions(miss_predictions_val_sbj_best_epoch,df,top_k=top_k,return_miss_per_subject=True)
    _,miss_predictions_test_sbj = tools.count_mispredictions(test_history_pred,df,top_k=top_k,return_miss_per_subject=True)
    fig, ax = plt.subplots(3,1,figsize=(15, 10))
    tools.plot_bar(data=miss_predictions_train_sbj,
                   ax_=ax[0],
                   title=f'TRAIN Subject misclassification BEST epochs {best_epoch} {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                   x_label='Subject ID',
                   y_label='Number of mispredictions',
                   list_stoic_subject=helper.stoic_subjects,
                   color='gray')
    tools.plot_bar(data=miss_predictions_val_sbj,
                   ax_=ax[1],
                   title=f'VAL Subject misclassification BEST epochs {best_epoch} {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                   x_label='Subject ID',
                   y_label='Number of mispredictions',
                   list_stoic_subject=helper.stoic_subjects,
                   color='gray')
    tools.plot_bar(data=miss_predictions_test_sbj,
                   ax_=ax[2],
                   title=f'TEST Subject misclassification BEST epochs {best_epoch} {"TOP-"+str(top_k) if top_k else ""} - {key} - {test_id} ',
                   x_label='Subject ID',
                   y_label='Number of mispredictions',
                   list_stoic_subject=helper.stoic_subjects,
                   color='gray')
    fig.savefig(os.path.join(test_output_folder, f'{test_id}_missclassification_train_val_test_subjects_{key}.png'), bbox_inches='tight')
    plt.close(fig)


  
def convert_dict_to_string(d):
  new_d = flatten_dict(d)
  return '\n'.join([f'{k}: {v}' for k, v in new_d.items()])

def filter_dict(d):
  keys_to_exclude = ['epochs', 'pooling_embedding_reduction', 'pooling_clips_reduction',
                     'shuffle_video_chunks', 'sample_frame_strategy', 'head', 'stride_window_in_video',
                     'plot_dataset_distribution', 'clip_length', 'early_stopping']
  # 'criterion_dict' if not d['criterion_dict'] else ''
  
  new_d = {k: v for k, v in d.items() if k not in keys_to_exclude}
  if new_d['head_params'].get('head_init_path',None) is not None:
    id_test = new_d['head_params']['head_init_path'].split('/')[-5].split('_')[0]
    new_d['head_params']['head_init_path'] = "/".join([id_test,*new_d['head_params']['head_init_path'].split('/')[-4:]])
  for k,v in new_d.items():
    if 'path' in k:
      if isinstance(v,str):
        new_d[k] = v.split('/')
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

def generate_subject_class_loss_csv(results_data, output_root_folder):
  # list_result_files_path = find_results_files(project_folder) # .pkl files
  dict_loss_per_class = {}
  dict_loss_per_subject = {}
  os.makedirs(output_root_folder, exist_ok=True)
  for file,data in results_data.items():
    # with open(pkl_path, 'rb') as f:
    #   data = pickle.load(f)
    
    test_folder = os.path.basename(os.path.dirname(file))
    test_id = test_folder.split('_')[0]
    dataset = 'unbc' if 'unbc' in os.path.join(*data['config']['path_csv_dataset']).lower() else 'biovid'
    if dataset not in dict_loss_per_class:
      dict_loss_per_class[dataset] = {}
    if dataset not in dict_loss_per_subject:
      dict_loss_per_subject[dataset] = {}
      
    dict_grouped_final_loss = get_grouped_losses(data, data['config'])
    if 'final' not in dict_grouped_final_loss:
      continue
    class_test_loss = dict_grouped_final_loss['final']['class_test_loss']
    subject_test_loss = dict_grouped_final_loss['final']['subject_test_loss']
    dict_loss_per_class[dataset][test_id] = class_test_loss
    dict_loss_per_subject[dataset][test_id] = subject_test_loss
  if dict_loss_per_class == {} or dict_loss_per_subject == {}:
    return
  for dataset in dict_loss_per_class:
    df_class = pd.DataFrame.from_dict(dict_loss_per_class[dataset], orient='index')
    df_subject = pd.DataFrame.from_dict(dict_loss_per_subject[dataset], orient='index')
    # bold the best results in the cols
    df_class.to_csv(os.path.join(output_root_folder, f'{dataset}_class_loss.csv'))
    df_subject.to_csv(os.path.join(output_root_folder, f'{dataset}_subject_loss.csv'))
    # df_class_style = df_class.style.apply(lambda x: ['font-weight: bold' if v == x.min() else '' for v in x], axis=0)
    # df_subject_style = df_subject.style.apply(lambda x: ['font-weight: bold' if v == x.min() else '' for v in x], axis=0)
    
    # # Save styled dataframes to Excel files
    # class_excel_path = os.path.join(output_root_folder, f'{dataset}_class_loss.xlsx')
    # subject_excel_path = os.path.join(output_root_folder, f'{dataset}_subject_loss.xlsx')

    # df_class_style.to_excel(class_excel_path, engine='openpyxl')
    # df_subject_style.to_excel(subject_excel_path, engine='openpyxl')


def generate_csv_row(data,config,time_, test_id): 

  list_fold = [int(k.split('_')[0][1:]) for k in data.keys() if 'final' not in k]
  list_sub_fold = [int(k.split('_')[-1]) for k in data.keys() if 'final' not in k]
  list_final_test = [int(k.split('_')[0][1:]) for k in data.keys() if 'final' in k]
  real_k_fold = max(list_fold) + 1
  real_sub_fold = max(list_sub_fold) + 1
  
  # test_key = 'test_accuracy' if 'test_accuracy' in data[f'k{list_fold[0]}_cross_val_sub_{list_sub_fold[0]}']['test'] else 'test_macro_precision'
  metric = data['k0_cross_val_sub_0']['train_val']['metric_for_stopping']
  key_metric_val = 'list_val_performance_metric'
  
  mean_train_losses_last_epoch = {f'mean_train_loss_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_losses_last_epoch = {f'mean_val_loss_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  if data['k0_cross_val_sub_0']['train_val']['list_val_accuracy'] == []:
    mean_val_accuracies_last_epoch = {}
    mean_train_accuracies_last_epoch = {}
    mean_train_accuracies_best_epoch = {}
    mean_val_accuracies_best_epoch = {}
    total_mean_train_accuracy_best_epoch = {}
    total_mean_val_accuracy_best_epoch = {}
  else:
    mean_train_accuracies_last_epoch = {f'mean_train_accuracy_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_train_accuracy'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    mean_val_accuracies_last_epoch = {f'mean_val_accuracy_last_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_accuracy'][-1] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    mean_train_accuracies_best_epoch = {f'mean_train_accuracy_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_train_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    mean_val_accuracies_best_epoch = {f'mean_val_accuracy_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    total_mean_train_accuracy_best_epoch = {f'total_mean_train_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_train_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
    total_mean_val_accuracy_best_epoch = {f'total_mean_val_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_accuracy'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
    
  mean_train_losses_best_epoch = {f'mean_train_loss_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  mean_val_losses_best_epoch = {f'mean_val_loss_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  
  
  # if list_final_test != []: 
  #   mean_test_accuracies = {f'mean_test_{metric}_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['test'][test_key] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  #   mean_test_losses = {f'mean_test_loss_best_ep_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['test']['test_loss'] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  #   total_mean_test_accuracy_best_epoch = {f'total_mean_test_{metric}_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['test'][test_key] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  #   total_mean_test_losses_best_epoch = {f'total_mean_test_loss_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['test']['test_loss'] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  # else:
  if list_final_test != []:
    mean_test_accuracies = {f'mean_all_test_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_final']['test']['test_accuracy'] for i in list_final_test])}
    mean_test_losses = {f'mean_all_test_loss_best_ep': np.mean([data[f'k{i}_cross_val_final']['test']['test_loss'] for i in list_final_test])}
    total_mean_test_accuracy_best_epoch = {f'total_mean_all_test_accuracy_best_ep': np.mean([data[f'k{i}_cross_val_final']['test']['test_accuracy'] for i in list_final_test])}
    total_mean_test_losses_best_epoch = {f'total_mean_all_test_loss_best_ep': np.mean([data[f'k{i}_cross_val_final']['test']['test_loss'] for i in list_final_test])}
    total_median_test_losses_best_epoch = {f'total_median_all_test_loss_best_ep': np.median([data[f'k{i}_cross_val_final']['test']['test_loss'] for i in list_final_test])}
    final_best_epoch_values = [data[f'k{i}_cross_val_final']['train_val']['best_model_idx'] for i in list_final_test]
    all_test_losses_best_epoch = [data[f'k{i}_cross_val_final']['test']['test_loss'] for i in list_final_test]
    all_test_l1_error = {'mean_test_l1_error': np.mean([data[f'k{i}_cross_val_final']['test'].get('test_l1_error', None) for i in list_final_test])}
    list_l1_error = [data[f'k{i}_cross_val_final']['test'].get('test_l1_error', None) for i in list_final_test]
    test_l1_errors = {'all_test_l1_error': ",".join([f"{error:.3f}" for error in list_l1_error])}
    # 3 decimal places
    all_test_losses_best_epoch = ",".join([f"{loss:.3f}" for loss in all_test_losses_best_epoch]) 
    all_test_losses_best_epoch = {'all_test_losses_best_epoch': all_test_losses_best_epoch}
    fold_used_for_final_test = {'fold_subfold_final_test': [data[f'k{i}_cross_val_final']['best_model']['fold_sub_fold_idx'] for i in list_final_test]}
    total_mean_val_loss_best_epoch = {f'total_mean_all_val_loss_best_ep': np.mean([data[f'k{i}_cross_val_final']['train_val'][key_metric_val][data[f'k{i}_cross_val_final']['train_val']['best_model_idx']] for i in list_final_test])}

  else:
    mean_test_accuracies = {}
    mean_test_losses = {}
    total_mean_test_accuracy_best_epoch = {}
    total_mean_test_losses_best_epoch = {}
    total_median_test_losses_best_epoch = {}
    final_best_epoch_values = []
    total_mean_val_loss_best_epoch = {}
    all_test_losses_best_epoch = {}
    fold_used_for_final_test = {}
    all_test_l1_error = {}
    test_l1_errors = {}
  
  if 'list_val_l1_error' not in data[f'k0_cross_val_sub_0']['train_val'] or len(data[f'k0_cross_val_sub_0']['train_val']['list_val_l1_error']) == 0:
    val_l1_error = {}
    val_l2_error = {}
  else:
    val_l1_error = {f'mean_val_l1_error_best_epochs_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_l1_error'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    val_l2_error = {f'mean_val_l2_error_best_epochs_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_l2_error'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  
  if 'list_val_icc' not in data[f'k0_cross_val_sub_0']['train_val'] or len(data[f'k0_cross_val_sub_0']['train_val']['list_val_ICC']) == 0:
    val_icc_error_mean = {}
    val_ccc_error_mean = {}
    val_pearson_corr_mean = {}
  else:
    val_icc_error_mean = {f'mean_val_icc_error_best_epochs_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_ICC'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    val_ccc_error_mean = {f'mean_val_ccc_error_best_epochs_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_CCC'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
    val_pearson_corr_mean = {f'mean_val_pearson_corr_best_epochs_k{i}': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['list_val_pearson_correlation'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx'][0]] for j in range(real_sub_fold)]) for i in range(real_k_fold)}
  
  total_mean_train_losses_best_epoch = {f'total_mean_train_loss_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_mean_val_losses_best_epoch = {f'total_mean_val_loss_best_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  total_median_val_losses_best_epoch = {f'total_median_val_loss_best_ep': np.median([data[f'k{i}_cross_val_sub_{j}']['train_val']['val_losses'][data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx']] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  best_epoch_values = [data[f'k{i}_cross_val_sub_{j}']['train_val']['best_model_idx'] for i in range(real_k_fold) for j in range(real_sub_fold)]
  best_epoch_values = {'best_epoch_values': best_epoch_values,
                       'best_epoch_mean': np.mean(best_epoch_values),
                       'best_epoch_median': np.median(best_epoch_values)}
  total_mean_train_losses_last_epoch = {f'total_mean_train_loss_last_ep': np.mean([data[f'k{i}_cross_val_sub_{j}']['train_val']['train_losses'][-1] for i in range(real_k_fold) for j in range(real_sub_fold)])}
  head_type = config['head'].name
  clip_grad_norm = config['clip_grad_norm'] if 'clip_grad_norm' in config else 'ND'
  head_params = flatten_dict({f'{head_type}': config['head_params']})
  criterion_params = flatten_dict({f'{type(config["criterion"]).__name__}': config["criterion_dict"]}) if "criterion_dict" in config else {}
  train_batch_sampler = [data[f'k{i}_cross_val_sub_{j}']['train_val'].get('train_batch_sampler_name',None) for i in range(real_k_fold) for j in range(real_sub_fold)]
  if 'adv_alpha_strategy' in config:
    adversarial_config = {
      'adv_alpha_strategy': config['adv_alpha_strategy'],
      'adv_alpha_gamma': config['adv_alpha_gamma'],
      'adversarial_loss_lambda': config['adversarial_loss_lambda'],
    }
  else:
    adversarial_config = {}
  if np.all([value is not None for value in train_batch_sampler]): 
    train_batch_sampler = [value.split('.')[-1] for value in train_batch_sampler]
  else:
    train_batch_sampler = ['ND' for _ in train_batch_sampler]
  scheduler_config = flatten_dict({'scheduler_config': config['scheduler_config_dict']}) if 'scheduler_config_dict' in config else {}
  row_dict = {
    'test_id': test_id,
    'k_fold_(real_k_fold)': f'{config["k_fold"]}_({config["real_k_fold"]})',
    'model': config['model_type'].name,
    'head': head_type,
    'optimizer': config['scheduler_config_dict']['optimizer_name'] if 'scheduler_config_dict' in config else config['optimizer_fn'],
    **scheduler_config,
    'target_metric': config['target_metric_best_model'],
    'criterion': type(config['criterion']).__name__,
    'train_batch_sampler': ','.join(train_batch_sampler),
    'round_output_loss': config['round_output_loss'],
    'early_stopping_key': config['key_for_early_stopping'] + f'(pat={config["early_stopping"].patience},eps={config["early_stopping"].min_delta},t_mod={config["early_stopping"].threshold_mode})',
    'feature_type': config['features_folder_saving_path'][-1] if config['features_folder_saving_path'][-1] != '' else config['features_folder_saving_path'][-2],
    'max_epochs': config['epochs'],
    'init_network': config['init_network'],
    'learning_rate': config['lr'],
    'batch_size_training': config['batch_size_training'],
    'reg_lambda_L1': config['regularization_lambda_L1'] if 'regularization_lambda_L1' in config else config['regularization_lambda'] if config['regularization_loss'] == 'L1' else 0,
    'reg_lambda_L2': config['regularization_lambda_L2'] if 'regularization_lambda_L2' in config else config['regularization_lambda'] if config['regularization_loss'] == 'L2' else 0,
    **adversarial_config,
    'label_smooth': config['label_smooth'] if 'label_smooth' in config else 0,
    'hflip': config['hflip'] if 'hflip' in config else 0,
    'jitter': config['jitter'] if 'jitter' in config else 0,
    'rotation': config['rotation'] if 'rotation' in config else 0,
    # 'reg_lambda_L2': config['regularization_lambda_L2'],
    # 'reg_lambda': config['regularization_lambda'],
    # 'reg_loss': config['regularization_loss'],
    **criterion_params,
    **head_params,
    **val_icc_error_mean,
    **val_ccc_error_mean,
    **val_pearson_corr_mean,
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
    **total_mean_train_losses_best_epoch,
    **total_mean_val_accuracy_best_epoch,
    **total_mean_test_accuracy_best_epoch,
    **total_mean_train_accuracy_best_epoch,
    **total_mean_train_losses_last_epoch,
    **total_mean_val_losses_best_epoch,
    **val_l1_error,
    **val_l2_error,
    **total_median_test_losses_best_epoch,
    **total_median_val_losses_best_epoch,
    **all_test_l1_error,
    **total_mean_test_losses_best_epoch,
    **total_mean_val_loss_best_epoch,
    **all_test_losses_best_epoch,
    **test_l1_errors,
    **best_epoch_values,
    'final_best_epoch_values': final_best_epoch_values,
    **fold_used_for_final_test,
    'clip_grad_norm': clip_grad_norm,
    'time_min': int(time_//60) if time_ is not None else 'ND',
  }
  return row_dict

def plot_confusion_matrices(data, root_output_folder, test_id, additional_info=''):
  test_output_folder = os.path.join(root_output_folder, test_id)
  os.makedirs(test_output_folder, exist_ok=True)
  if isinstance(data['config']['criterion'],losses.RESupConLoss) or isinstance(data['config']['criterion'],losses.RnCLossV2) or isinstance(data['config']['criterion'],losses.DisentangledLoss):
    return  # No confusion matrix for RESupConLoss
  is_cross_entropy = isinstance(data['config']['criterion'],torch.nn.CrossEntropyLoss) or isinstance(data['config']['criterion'],losses.PHuberCrossEntropy)
  #### Plot confusion matrix for each epoch ####
  for key,dict_sub_fold in data['results'].items():
    
    # if 'final' not in key:
    #   continue  # For cross-validation folds, only plot for final model
    
    best_epoch_idx =  dict_sub_fold['train_val']['best_model_idx']
    dict_train_conf_matrix = dict_sub_fold['train_val']['train_confusion_matricies']
    dict_val_conf_matrix = dict_sub_fold['train_val']['val_confusion_matricies']
    dict_test_conf_matrix = None
      
    if 'test' in data['results'][key] and data['results'][key]['test'] != {}:
      dict_test_conf_matrix = {f'{best_epoch_idx}':dict_sub_fold['test']['test_confusion_matrix']}
    
    # Plot confusion matrix for each epoch
    for epoch in dict_train_conf_matrix.keys():
      if int(epoch) == best_epoch_idx and dict_test_conf_matrix is not None:
        fig, axs = plt.subplots(3, 1, figsize=(5, 15))
        if is_cross_entropy:
          fig_l1_error, axs_l1_error = plt.subplots(3, 1, figsize=(5, 15))
      else:
        fig, axs = plt.subplots(2, 1, figsize=(5, 10))
        if is_cross_entropy:
          fig_l1_error, axs_l1_error = plt.subplots(2, 1, figsize=(5, 10))
      axs_count = 0
      kk = key.split('_')
      if len(kk) > 2:
        kk = f'{kk[0]}_{kk[-1]}'
      else:
        kk = key
      tools.plot_confusion_matrix(dict_train_conf_matrix[epoch], ax=axs[axs_count], title=f'TRAIN - Epoch {epoch} - {test_id} - {kk}')
      if is_cross_entropy:
        dict_mae_per_class = compute_mae_per_class(conf_matrix=dict_train_conf_matrix[epoch])
        tools.plot_error_per_class(mae_per_class=[dict_mae_per_class[i] for i in sorted(dict_mae_per_class.keys())],
                                  unique_classes=sorted(dict_mae_per_class.keys()),
                                  criterion='L1',
                                  ax=axs_l1_error[0],
                                  title=f'TRAIN -Loss per class - Epoch {epoch} - {kk} - {test_id}',
                                  y_lim=10 if 'unbc' in "".join(data['config']['path_csv_dataset']).lower() else 3)
      axs_count += 1
      if data['config'].get('validate', True) and dict_val_conf_matrix is not None:
        tools.plot_confusion_matrix(dict_val_conf_matrix[epoch], ax=axs[axs_count], title=f'VAL - Epoch {epoch} - {test_id} - {kk}')
        if is_cross_entropy:
          dict_mae_per_class = compute_mae_per_class(conf_matrix=dict_val_conf_matrix[epoch])
          tools.plot_error_per_class(mae_per_class=[dict_mae_per_class[i] for i in sorted(dict_mae_per_class.keys())],
                                    unique_classes=sorted(dict_mae_per_class.keys()),
                                    criterion='L1',
                                    ax=axs_l1_error[1],
                                    title=f'VAL -Loss per class - Epoch {epoch} - {kk} - {test_id}',
                                    y_lim=10 if 'unbc' in "".join(data['config']['path_csv_dataset']).lower() else 3)
        axs_count += 1
      if int(epoch) == best_epoch_idx and dict_test_conf_matrix is not None:
        tools.plot_confusion_matrix(dict_test_conf_matrix[epoch], ax=axs[axs_count], title=f'TEST - Epoch {epoch} - {test_id} - {kk}')
        if is_cross_entropy:
          dict_mae_per_class = compute_mae_per_class(conf_matrix=dict_test_conf_matrix[epoch])
          tools.plot_error_per_class(mae_per_class=[dict_mae_per_class[i] for i in sorted(dict_mae_per_class.keys())],
                                    unique_classes=sorted(dict_mae_per_class.keys()),
                                    criterion='L1',
                                    ax=axs_l1_error[2],
                                    title=f'TEST -Loss per class - Epoch {epoch} - {kk} - {test_id}',
                                    y_lim=10 if 'unbc' in "".join(data['config']['path_csv_dataset']).lower() else 3)
      fig.tight_layout()
      fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_confusion_matrix_{key}_epoch_{epoch}.png'))
      plt.close(fig)
      if is_cross_entropy:
        fig_l1_error.tight_layout()
        fig_l1_error.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_loss_per_class_L1_conf_matrix_{key}_epoch_{epoch}.png'))
        plt.close(fig_l1_error)
      
  return # Skipping for now to save time
  #### Plot only best epoch confusion matrix in percentage ####
  fig, axs = plt.subplots(3, 1, figsize=(5, 15))

  if str(best_epoch_idx) not in dict_train_conf_matrix.keys():
    print(f"Test id {test_id}:   Best epoch {best_epoch_idx} not in training confusion matrix keys {list(dict_train_conf_matrix.keys())}, skipping best epoch confusion matrix plot")
    return

  cms = []
  titles = []
  conf_matrix_train_percent = convert_conf_matrix_to_percent(dict_train_conf_matrix[str(best_epoch_idx)])
  cms.append(conf_matrix_train_percent)
  titles.append(f'TRAIN - Epoch {best_epoch_idx}   - {test_id}')
  if 'final' not in key and data['config'].get('validate', True):
    conf_matrix_val_percent = convert_conf_matrix_to_percent(dict_val_conf_matrix[str(best_epoch_idx)])
    cms.append(conf_matrix_val_percent)
    titles.append(f'VAL - Epoch {best_epoch_idx}   - {test_id}')
  if 'final' in key or 'test' in data['results'][key]:
    conf_matrix_test_percent = convert_conf_matrix_to_percent(dict_test_conf_matrix[str(best_epoch_idx)])
    cms.append(conf_matrix_test_percent)
    titles.append(f'TEST {test_id} - {key} - Epoch {best_epoch_idx}')
  
  for ax, cm, title in zip(axs, cms, titles):
    sns.heatmap(cm.cpu().numpy(), annot=True, fmt=".2f", cmap='Blues', cbar_kws={'label': 'Percentage (%)'}, ax=ax,
                xticklabels=[str(i) for i in range(cm.shape[0])],
                yticklabels=[str(i) for i in range(cm.shape[0])])
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
  fig.tight_layout()
  fig.savefig(os.path.join(test_output_folder, f'{test_id}{additional_info}_confusion_matrix_percentage_{key}_best_epoch_{best_epoch_idx}.png'))
  plt.close(fig)
    # ##########################################################


def convert_conf_matrix_to_percent(conf_matrix: MulticlassConfusionMatrix):
    matrix = conf_matrix.confmat if hasattr(conf_matrix, "confmat") else conf_matrix
    matrix = matrix.float()
    # Calculate row sums
    row_sums = matrix.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1

    # Convert to percentages
    percent_matrix = matrix / row_sums * 100    
    return percent_matrix


def compute_mae_per_class(conf_matrix, compute_column_wise=False):
    """
    Computes the Mean Absolute Error (MAE) for each class from a confusion matrix.
    Ordinal labels are assumed to correspond to indices.
    
    Args:
        conf_matrix: Confusion matrix (raw counts) as Tensor, ndarray or MulticlassConfusionMatrix.
        compute_column_wise: If True, also computes MAE based on predicted labels.
        
    Returns:
        A dictionary with class indices as keys and MAE values as values (row-wise by default).
        If compute_column_wise is True, returns a dictionary with 'row_mae' and 'col_mae'.
    """
    matrix = conf_matrix.confmat if hasattr(conf_matrix, "confmat") else conf_matrix
    if isinstance(matrix, np.ndarray):
      matrix = torch.from_numpy(matrix)
    matrix = matrix.float()
    
    n_classes = matrix.shape[0]
    device = matrix.device
    
    # Row-wise MAE: "Given the true class is i, what is the average error of predictions?"
    row_mae = {}
    for i in range(n_classes):
      row_sum = matrix[i, :].sum()
      if row_sum == 0:
        continue
      else:
        # Absolute distance between true class i and all possible predictions j
        distances = torch.abs(torch.arange(n_classes, device=device) - i)
        mae = (matrix[i, :] * distances).sum() / row_sum
        row_mae[i] = mae.item()
            
    if compute_column_wise:
      # Column-wise MAE: "Given the prediction is j, what is the average error from the true label?"
      col_mae = {}
      for j in range(n_classes):
        col_sum = matrix[:, j].sum()
        if col_sum == 0:
          continue
        else:
          # Absolute distance between predicted class j and all possible true labels i
          distances = torch.abs(torch.arange(n_classes, device=device) - j)
          mae = (matrix[:, j] * distances).sum() / col_sum
          col_mae[j] = mae.item()
      return {"row_mae": row_mae, "col_mae": col_mae}
        
    return row_mae


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
    elif target_metric_best_model == 'val_accuracy':
      if data[f'k{k}_test']['dict_test']['test_accuracy'] > data[f'k{best_fold}_test']['dict_test']['test_accuracy']:
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

def plot_accuray_per_class_across_epochs(data, run_output_folder, test_id, additional_info=''):
  # Assuming val_accuracy and val_loss are already stacked arrays of shape (n_epochs, n_classes)
  if isinstance(data['config']['criterion'],losses.RESupConLoss) or isinstance(data['config']['criterion'],losses.RnCLossV2) or isinstance(data['config']['criterion'],losses.DisentangledLoss):
    return  # No accuracy per class for RESupConLoss
  for key in data['results'].keys():
    if 'final' in key:
      continue
    val_accuracy = np.stack(data['results'][key]['train_val']['list_val_accuracy_per_class'])
    val_loss = np.stack(data['results'][key]['train_val']['val_loss_per_class'])

    n_epochs, n_classes = val_accuracy.shape

    # Set up the subplot grid
    n_cols = 3
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 5))
    axes = axes.flatten()

    for class_idx in range(n_classes):
      ax1 = axes[class_idx]  # Main axis for loss
      ax2 = ax1.twinx()      # Twin axis for accuracy

      # Plot loss
      line1, = ax1.plot(val_loss[:, class_idx], color='tab:purple', label='Loss')
      ax1.set_ylim(0, 3)
      ax1.set_ylabel('Loss', color='tab:purple')
      ax1.tick_params(axis='y', labelcolor='tab:purple')

      # Plot accuracy
      line2, = ax2.plot(val_accuracy[:, class_idx], color='tab:blue', label='Accuracy')
      ax2.set_ylim(0, 1.1)
      ax2.set_ylabel('Accuracy', color='tab:blue')
      ax2.tick_params(axis='y', labelcolor='tab:blue')

      # Combine legends (top right position)
      lines = [line1, line2]
      labels = [line.get_label() for line in lines]
      ax1.legend(lines, labels, loc='upper right', fontsize='small')

      ax1.set_title(f'Class {class_idx}')
      ax1.set_xlabel('Epoch')
      ax1.grid(True)

    # Hide any unused subplots
    for i in range(n_classes, len(axes)):
      fig.delaxes(axes[i])

    fig.suptitle('Validation Loss and Accuracy per Class Across Epochs', fontsize=16)
    fig.savefig(os.path.join(run_output_folder,test_id, f'{test_id}{additional_info}_val_accuracy_loss_per_class_across_epochs.png'), bbox_inches='tight')
    plt.close(fig)
  # fig.tight_layout(rect=[0, 0, 1, 0.97])
  
  # plt.show()

def create_lr_wd_plot(dict_sub_fold, test_id, key, ax1=None, title=None):
  lr_values = dict_sub_fold['train_val'].get('list_lrs', None)
  lr_values = np.array([np.mean(sublist) for sublist in lr_values]) if lr_values is not None else None
  # lr_values = np.concatenate(lr_values) if lr_values is not None else None
  wd_values = dict_sub_fold['train_val'].get('list_wds', None)
  if wd_values is not None and isinstance(wd_values[0], dict):
    wd_no_biases = [[el['wd_no_bias'] for el in batch] for batch in wd_values] if wd_values is not None else None
    wd_no_biases = np.array([np.mean(sublist) for sublist in wd_no_biases]) if (wd_values is not None) else None
  else:
    wd_no_biases = None
  # wd_values = [item for sublist in wd_values for item in sublist] if wd_values is not None else None
  # wd_biases = np.array([wd['wd_bias'] for wd in wd_values]) if (wd_values is not None and 'wd_bias' in wd_values[0]) else None
  if lr_values is None:
    return  # Nothing to plot
  
  steps = list(range(0, len(lr_values)))
  if ax1 is None:
    fig, ax1 = plt.subplots(figsize=(10, 6))
  else:
    fig = None
  if lr_values is not None:
    ax1.plot(steps, lr_values, 'b-', label='Learning Rate')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Learning Rate', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
  if wd_no_biases is not None:
    ax2 = ax1.twinx()
    ax2.plot(steps, wd_no_biases, 'r-', label='Weight Decay No Bias')
    ax2.set_ylabel('Weight Decay No Bias', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
  # if wd_no_biases is not None:
  #   ax2.plot(steps, wd_no_biases, 'g-', label='Weight Decay No Bias')
  #   ax2.set_ylabel('Weight Decay', color='g')
    # ax3.tick_params(axis='y', labelcolor='g')
  
  if title is None:
    title = f'Learning Rate and Weight Decay Across Epochs - {test_id} - {key}'
  
  if fig is not None:
    plt.title(title)
  else:
    ax1.set_title(title)
  # add legends for both axes
  lines = []
  labels = []
  if lr_values is not None:
    line1, = ax1.plot([], [], 'b-')
    lines.append(line1)
    labels.append('Learning Rate')
  if wd_no_biases is not None:
    line2, = ax2.plot([], [], 'r-')
    lines.append(line2)
    labels.append('Weight Decay No Bias')
  # if wd_biases is not None:
  #   line3, = ax2.plot([], [], 'g-')
  #   lines.append(line3)
  #   labels.append('Weight Decay Bias')
  ax1.legend(lines, labels)

def plot_lr_wd_across_epochs(data, run_output_folder, test_id, save_fig=True,title=None):
  test_output_folder = os.path.join(run_output_folder, test_id)
  os.makedirs(test_output_folder, exist_ok=True)
  for key,dict_sub_fold in data['results'].items():
    if 'final' in key:
      continue
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    create_lr_wd_plot(dict_sub_fold, test_id, key, ax1=ax1, title=title)
    fig.tight_layout()
    if save_fig:
      fig.savefig(os.path.join(test_output_folder, f'{test_id}_lr_wd_across_epochs_{key}.png'), bbox_inches='tight')
    plt.close(fig)
    
# def link_cross_attention_logs(data, run_output_folder, test_id):


def generate_video_from_loss_plots(run_output_folder, test_id):
  test_output_folder = os.path.join(run_output_folder, test_id)
  png_files_val = sorted([f for f in os.listdir(test_output_folder) if f.endswith('.png') and '_losses_' in f and 'final' not in f])
  png_files_final = sorted([f for f in os.listdir(test_output_folder) if f.endswith('.png') and '_losses_' in f and 'final' in f])
  png_files = png_files_val + png_files_final
  if len(png_files) == 0:
    print(f'No loss plot PNG files found for {test_id}, skipping video generation.')
    return
  video_path = os.path.join(test_output_folder, f'{test_id}_losses_over_epochs.mp4')
  list_images = []
  for png_file in png_files:
    image_path = os.path.join(test_output_folder, png_file)
    image = cv2.imread(image_path)
    list_images.append(image)
  height, width, layers = list_images[0].shape
  video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'avc1'), 1, (width, height))
  for image in list_images:
    if image.shape != (height, width, layers):
      print(f'Image has different shape, resizing to {(height, width)}')
      image = cv2.resize(image, (width, height))
    video.write(image)
  video.release()
  # print(f'Generated loss video at {video_path}')

def plot_separated_losses_adversarial(data, run_output_folder, test_id):
  test_output_folder = os.path.join(run_output_folder, test_id)
  os.makedirs(test_output_folder, exist_ok=True)
  if not data['config']['head_params']['adversarial_head']:
    return  # No adversarial loss to plot 
  for key,dict_sub_fold in data['results'].items():
    adv_loss = [v[0] for v in dict_sub_fold['train_val']['list_train_adv_losses']]
    main_loss = [v[1] for v in dict_sub_fold['train_val']['list_train_adv_losses']]
    epochs = list(range(len(adv_loss)))
    loss = str(data['config']['criterion']).lower()
    plt.figure(figsize=(18, 9))
    plt.plot(epochs, adv_loss, label='Adversarial Loss')
    plt.plot(epochs, main_loss, label=f'{loss.capitalize()} Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training - {test_id} - Adversarial and {loss.capitalize()} Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(test_output_folder, f'{test_id}_separated_losses_adversarial_{key}.png'))
    plt.close()

def clean_data(result, config):
# remove all non-final folds if use_test_as_val is True and k_fold is 87 (special case)
  if not config['skip_test'] and not config['validate'] and config['k_fold'] == 87:
    for key in list(result.keys()):
      if 'final' not in key:
        del result[key]
  
  
def plot_run_details(results_data, output_root,only_csv, dict_args,plot_only_loss=False):
  list_row_csv = []
  generate_subject_class_loss_csv(results_data,output_root)
  for file, data in tqdm.tqdm(results_data.items()):
    test_folder = os.path.basename(os.path.dirname(file))
    test_id = test_folder.split('_')[0]
    data['config']['real_k_fold'] = len(data['results'])
    if data['config']['real_k_fold'] == 0:
      print(f'No TEST file found in {file}')
      continue
    grid_search_folder = Path(file).parts[-3]
    is_unbc = 'unbc' in "".join(data['config']['path_csv_dataset']).lower()
    list_row_csv.append(generate_csv_row(data['results'],data['config'],data['time'], test_id))
    if not only_csv:
      clean_data(data['results'], data['config'])
      data['config']['model_type'] = data['config']['model_type'].name
      
      plot_grouped_k_fold(data, os.path.join(output_root), test_id)
      plot_losses(data, os.path.join(output_root), test_id, loss_plot_type=dict_args['loss_plot_type'], test_as_validation=dict_args['test_as_validation'])
      plot_separated_losses_adversarial(data, os.path.join(output_root), test_id)
      # if 
      if not plot_only_loss:
        plot_confusion_matrices(data, os.path.join(output_root), test_id)
        plot_lr_wd_across_epochs(data, os.path.join(output_root), test_id)
        if is_unbc:
          generate_video_from_loss_plots(os.path.join(output_root), test_id)
        if not is_unbc:
          plot_gradient_per_module(data, os.path.join(output_root), test_id)
          
        plot_CCC_ICC_pearson(data, os.path.join(output_root), test_id)
        plot_history_model_prediction(data, os.path.join(output_root), test_id,root_csv_path=os.path.dirname(file))
        if data['config'].get('validate', True):
          plot_accuray_per_class_across_epochs(data, os.path.join(output_root), test_id)
      link_attention_logs(os.path.dirname(file), output_root, test_id)        
      # except Exception as e:
      #   print(f'Error in {file} - {e}')
  df = pd.DataFrame(list_row_csv)
  df = df.fillna('ND')
  if not os.path.exists(output_root):
    os.makedirs(output_root, exist_ok=True)
  df.to_csv(os.path.join(output_root, 'summary.csv'), index=False)
  print(f'Summary CSV saved to {output_root}/summary.csv')

def link_attention_logs(file_folder, output_root, test_id):
  # folder name is cross_attention_*
  cross_attention_folders = []
  for root, dirs, files in os.walk(file_folder):
    attention_log_folder = [d for d in dirs if ('cross_attention_' or 'video_embeddings') in d]
    if len(attention_log_folder) > 0:
      for d in attention_log_folder:
        attention_log_folder_path = os.path.join(root, d)
        cross_attention_folders.append(attention_log_folder_path)

  # Create symlink in output_root/test_id/attention_logs
  for attention_log_folder in cross_attention_folders:
    output_attention_folder = os.path.join(output_root, test_id, Path(attention_log_folder).name)
    if not os.path.exists(output_attention_folder):
      os.makedirs(output_attention_folder, exist_ok=True)
    for file in os.listdir(attention_log_folder):
      src = os.path.join(attention_log_folder, file)
      dst = os.path.join(output_attention_folder, file)
      if not os.path.exists(dst):
        os.symlink(src, dst)

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
  parser.add_argument('--loss_plot_type', type=str, default='loss', help='Type of loss plot: dist or loss')
  parser.add_argument('--filter', type=str, default='',
                      help='Optional filter criteria in format key1=val1,key2=val2')
  parser.add_argument('--only_csv', action='store_true',
                      help='Generate only the summary CSV file without generating any plots')
  parser.add_argument('--print_filter', action='store_true',
                      help='Print list of avilable filter from the first .pkl file in parent_folder')
  parser.add_argument('--test_as_validation', default=0, type=int,
                      help='If set to 1, use plot test set as validation set (for UNBC) if possible')
  parser.add_argument('--test_ids', type=str, default='',
                      help='Comma separated list of test IDs to process (e.g., test1,test2). If empty, process all tests.')
  parser.add_argument('--plot_only_loss', action='store_true',
                      help='If set, only plot the loss curves without other plots.')
  args = parser.parse_args()
  dict_args = vars(args)
  parent_folder = args.parent_folder
  only_csv = args.only_csv
  
  print(f'Parent folder: {parent_folder}')
  output_root = os.path.join(parent_folder, '_summary')
  os.makedirs(output_root, exist_ok=True)
  if args.print_filter == True:
    print(f'Printing filter from: {args.print_filter}')
    results_files = find_results_files(args.parent_folder)
    if len(results_files) == 0:
      raise ValueError(f'No results files found in {args.parent_folder}')
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
          try:
            filter_dict_arg[key.strip()] = float(value.strip())
          except ValueError:
            filter_dict_arg[key.strip()] = value.strip() 
      print(f'Applying filter: {filter_dict_arg}')
      plot_filtered_run_details(parent_folder, output_root, filter_dict_arg,only_csv, dict_args)
      
    else:
    # Generate the unfiltered plots
      if args.test_ids:
        test_ids_list = [test_id.strip() for test_id in args.test_ids.split(',')]
        print(f'Filtering to only process test IDs: {test_ids_list}')
        results_files = find_results_files(parent_folder)
        results_data = {}
        for file in results_files:
          test_folder = os.path.basename(os.path.dirname(file))
          test_id = test_folder.split('_')[0]
          if test_id in test_ids_list:
            results_data[file] = load_results(file)
        print(f'Loaded {len(results_data)} results files after filtering by test IDs')
        plot_run_details(results_data, os.path.join(output_root, 'plot_run_details'), only_csv, dict_args, plot_only_loss=args.plot_only_loss)
      else:
        results_files = find_results_files(parent_folder) # get .pkl files
        results_data = {file: load_results(file) for file in results_files} # load .pkl files with path as a key
        # print(f'Loaded {len(results_data)} results files')
        plot_run_details(results_data, os.path.join(output_root, 'plot_run_details'), only_csv, dict_args, plot_only_loss=args.plot_only_loss)


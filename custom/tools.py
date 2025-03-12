import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import pandas as pd
from torchmetrics.classification import ConfusionMatrix, MulticlassConfusionMatrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator
import cv2
import av
import torch
import json
from openTSNE import TSNE as openTSNE
import time
from custom.faceExtractor import FaceExtractor
from tqdm import tqdm
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
# if os.name == 'posix':
  # from tsnecuda import TSNE as cudaTSNE # available only on Linux
# else:
  # print('tsnecuda available only on Linux')

class NpEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if isinstance(obj, torch.Tensor):
      return obj.cpu().detach().numpy().tolist()
    if isinstance(obj,MulticlassConfusionMatrix):
      return obj.compute().cpu().detach().numpy().tolist()
    return super(NpEncoder, self).default(obj)


def save_csv_file(cols,csv_array,saving_path,sliding_windows):
  """
  Save the CSV file with the given columns and array of data.

  Args:
    cols (list): List of column names for the CSV file.
    csv_array (np.ndarray): Array of data to be saved in the CSV file.
    saving_path (str): Path to save the CSV file.
    sliding_windows (bool): Whether to save the data with sliding windows or not.
  """
  df = pd.DataFrame(csv_array, columns=cols)
  csv_path = os.path.join(saving_path, f'video_labels_{sliding_windows}.csv')
  if not os.path.exists(saving_path):
    os.makedirs(saving_path)
  if sliding_windows:
    df.to_csv(csv_path, index=False, sep='\t')
  else:
    df.to_csv(csv_path, index=False)
  print(f'CSV file saved to {csv_path}')
  return csv_path


def save_dict_data(dict_data, saving_folder_path):
  """
  Save the dictionary containing numpy and torch elements to the specified path.

  Args:
    dict_data (dict): Dictionary containing the data to be saved.
    saving_path (str): Path to save the dictionary data.
  """
  if not os.path.exists(saving_folder_path):
    os.makedirs(saving_folder_path)
  print(f'Saving dictionary data to {saving_folder_path}...')
  for key, value in tqdm(dict_data.items(), desc="Saving files"):
    if isinstance(value, torch.Tensor):
      torch.save(value, os.path.join(saving_folder_path, f"{key}.pt"))
    elif isinstance(value, np.ndarray):
      np.save(os.path.join(saving_folder_path, f"{key}.npy"), value)
    else:
      print(f"Unsupported data type for key {key}: {type(value)}")
      # raise ValueError(f"Unsupported data type for key {key}: {type(value)}")
  print(f'Dictionary data saved to {saving_folder_path}')
def is_dict_data(folder_path):
  """
  Check if the specified folder contains dictionary data.

  Args:
    folder_path (str): Path to check for dictionary data.

  Returns:
    bool: True if the folder contains dictionary data, False otherwise.
  """
  list_target_files = ['features.pt','list_frames.pt','list_labels.pt','list_path.npy','list_subject_id.pt','list_sample_id.pt']
  list_files = os.listdir(folder_path)
  if len(list_files) != len(list_target_files):
    return False
  for file in list_files:
    if file not in list_target_files:
      return False
  return True

def load_dict_data(saving_folder_path):
  """
  Load the dictionary containing numpy and torch elements from the specified path.

  Args:
    saving_path (str): Path to load the dictionary data.

  Returns:
    dict: Dictionary containing the loaded data.
  """
  dict_data = {}
  for file in os.listdir(saving_folder_path):
    if file.endswith(".pt"):
      dict_data[file[:-3]] = torch.load(os.path.join(saving_folder_path, file))
    elif file.endswith(".npy"):
      dict_data[file[:-4]] = np.load(os.path.join(saving_folder_path, file))
    else:
      print(f"Unsupported file format: {file}")
  return dict_data

def plot_error_per_class(unique_classes, mae_per_class, criterion,title='', count_classes=None, saving_path=None,y_lim=None):
  """ Plot Mean Absolute Error per class. """
  plt.figure(figsize=(10, 5))
  if y_lim:
    plt.ylim(0,y_lim)
  plt.bar(unique_classes, mae_per_class, color='blue', width=0.8, label='MAE per Class',edgecolor='black')
  plt.xlabel('Class')
  plt.ylabel(f'{criterion}')
  plt.xticks(unique_classes)  # Show each element in x-axis
  plt.title(f'{criterion} per Class {title}')
  
  if count_classes is not None:
    for cls,count in count_classes.items():
      plt.text(unique_classes[cls], mae_per_class[cls], str(count), ha='center', va='bottom')
  plt.legend()
  if saving_path is not None:
    plt.savefig(saving_path)
  else:
    plt.show()
  plt.close()

def get_accuracy_from_confusion_matrix(confusion_matrix,list_real_classes=None):
  """
  Calculate various accuracy metrics from a given confusion matrix in torch.tensor format.
  Args:
    confusion_matrix (torch.Tensor): A square tensor representing the confusion matrix.

  Returns:
    dict: A dictionary containing the following keys:
      - 'accuracy_per_class' (torch.Tensor): The accuracy for each class.
      - 'mean_balanced_accuracy_per_class' (torch.Tensor): The balanced accuracy for each class.
      - 'mean_accuracy' (torch.Tensor): The mean accuracy across all classes except the last one (considered the Null class).
      - 'mean_balanced_accuracy' (torch.Tensor): The mean balanced accuracy across all classes except the last one (considered the Null class).
  """
  if isinstance(confusion_matrix, MulticlassConfusionMatrix):
    # print('COmpute conf matrix')
    confusion_matrix = confusion_matrix.compute()
  # Drop the last class (missclassification class)
  if list_real_classes is not None:
    sum_all = torch.sum(confusion_matrix)
    sum_per_rows = torch.sum(confusion_matrix,1)[list_real_classes] # fn considering also the missclassification
    confusion_matrix = confusion_matrix[list_real_classes,:][:,list_real_classes]
  else:
    sum_all = torch.sum(confusion_matrix)
    sum_per_rows = torch.sum(confusion_matrix,1)
  tp = confusion_matrix.diag()
  fn = sum_per_rows - tp 
  fp = torch.sum(confusion_matrix,0) - tp
  precision_per_class = torch.stack([tp[i] / (tp[i]+fp[i]) if tp[i]+fp[i]!=0 else torch.tensor(0) for i in range(len(tp))]).float()
  recall_per_class = torch.stack([tp[i] / (tp[i] + fn[i]) if tp[i]+fn[i]!=0 else torch.tensor(0) for i in range(len(tp))]).float() 
  
  tp_sum = torch.sum(tp)
  fp_sum = torch.sum(fp)
  fn_sum = torch.sum(fn)
  
  # Treats all instances equally (larger classes have more weight)-> sensitive to imbalance
  micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) != 0 else torch.tensor(0.0)
  micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) != 0 else torch.tensor(0.0)
  
  # Weighted by the size of each class
  weighted_precision = torch.sum(precision_per_class * sum_per_rows) / sum_all
  weighted_recall = torch.sum(recall_per_class * sum_per_rows) / sum_all
  
  # Treats all classes equally
  macro_precision = torch.mean(precision_per_class)
  macro_recall = torch.mean(recall_per_class)
  
  return {
    'precision_per_class': precision_per_class.detach().numpy(),
    'recall_per_class': recall_per_class.detach().numpy(),
    'macro_precision': macro_precision.detach().numpy(), 
    'macro_recall': macro_recall.detach().numpy(), 
    'micro_precision': micro_precision.detach().numpy(), # same as accuracy
    'micro_recall': micro_recall.detach().numpy(), 
    'weighted_precision': weighted_precision.detach().numpy(), 
    'weighted_recall': weighted_recall.detach().numpy(),
  }

def plot_accuracy_confusion_matrix(confusion_matricies, type_conf,title='', saving_path=None, list_real_classes=None):
  if isinstance(confusion_matricies[0], MulticlassConfusionMatrix):
    confusion_matricies = torch.stack([confusion_matricies[i].compute() for i in range(len(confusion_matricies))])
  # if isinstance(test_confusion_matricies[0], MulticlassConfusionMatrix):
  #   test_confusion_matricies=torch.stack([test_confusion_matricies[i].compute() for i in range(len(test_confusion_matricies))])
  
  list_acc_confusion_matrix = []
  # list_test_acc_confusion_matrix = []
  for confusion_matrix in confusion_matricies:
    list_acc_confusion_matrix.append(get_accuracy_from_confusion_matrix(confusion_matrix=confusion_matrix,
                                                                        list_real_classes=list_real_classes))
  # for test_confusion_matrix in test_confusion_matricies:
  #   list_test_acc_confusion_matrix.append(get_accuracy_from_confusion_matrix(test_confusion_matrix))
  keys = list_acc_confusion_matrix[0].keys()
  for key in keys:
    # print(f'key: {key}')
    list_key_values = [list_acc_confusion_matrix[i][key] for i in range(len(list_acc_confusion_matrix))]
    # test_list_key_values = [list_test_acc_confusion_matrix[i][key] for i in range(len(list_test_acc_confusion_matrix))]
    labels = []
    # labels_test = []
    if len(list_key_values[0].shape) == 0:
      labels.append(f'{type_conf}')
      # labels_test.append(f'Test')
    else:
      for i in range(list_key_values[0].shape[0]):
        labels.append(f'{type_conf} class {i}')
        # labels_test.append(f'Test class {i}')
    plt.figure(figsize=(10, 5))
    
    # Plot train results
    plt.plot(list_key_values, label=labels)
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.title(f'{type_conf} {key} over Epochs {title}')
    plt.legend()
    if saving_path is not None:
      path=os.path.join(saving_path,f'{type_conf}_{key}.png')
      plt.savefig(path)
      print(f'Plot {key} over Epochs {title} saved to {path}.png')
    else:
      plt.show()
    plt.close()
    #Plot test results  
    # plt.figure(figsize=(10, 5))
    # plt.plot(test_list_key_values, label=labels_test)
    # plt.xlabel('Epochs')
    # plt.ylabel(key)
    # plt.title(f'test_{key} over Epochs {title}')
    # plt.legend()
    # if saving_path is not None:
    #   path=os.path.join(saving_path,f'test_{key}.png')
    #   plt.savefig(path)
    #   print(f'Plot {key} over Epochs {title} saved to {path}.png')
    # else:
    #   plt.show()

def plot_error_per_subject(uniqie_subject_ids, criterion,mae_per_subject,title='', count_subjects=None, saving_path=None,y_lim=None):
  """ Plot Mean Absolute Error per participant. """
  plt.figure(figsize=(15, 5))
  if y_lim:
    plt.ylim(0,y_lim)
  str_uniqie_subject_ids = [str(id) for id in uniqie_subject_ids]
  plt.bar(str_uniqie_subject_ids, mae_per_subject,width=0.8, color='green', edgecolor='black')
  plt.xlabel('Participant')
  plt.ylabel(f'{criterion}')
  plt.title(f'{criterion} per Participant -- {title}')
  plt.xticks(fontsize=11,rotation=45)
  # filter_elements = np.array([True if mae >= 0.0 else False for mae in mae_per_subject])
  # plt.xticks(uniqie_subject_ids[filter_elements])  # Show each element in x-axis
  if count_subjects is not None:
    for id,count in count_subjects.items():
      idx = np.where(id == uniqie_subject_ids)[0]
      plt.text(str(uniqie_subject_ids[idx]), mae_per_subject[idx], str(count), ha='center', va='bottom')
  if saving_path is not None:
    plt.savefig(saving_path)
  else:
    plt.show()
  plt.close()

def plot_losses(train_losses, val_losses, saving_path=None):
  plt.figure(figsize=(10, 5))
  plt.yticks(fontsize=12)
  plt.plot(train_losses, label='Training Loss')
  plt.plot(val_losses, label='Val Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Val Losses over Epochs')
  plt.legend()
  plt.grid(True)
  if saving_path is not None:
    plt.savefig(saving_path)
    print(f'Plot losses saved to {saving_path}.png')
  else:
    plt.show()
  plt.close()

def _generate_train_test_validation(csv_path, saving_path,train_size=0.8,val_size=0.1,test_size=0.1, random_state=42):
  """
  Generate train, validation, and test splits from a CSV file and save them to specified paths.
  Parameters:
    csv_path (str): Path to the input CSV file containing video labels.
    saving_path (str): Path where the split CSV files will be saved.
    train_size (float, optional): Proportion of the dataset to include in the train split. Default is 0.8.
    val_size (float, optional): Proportion of the dataset to include in the validation split. Default is 0.1.
    test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.1.
    random_state (int, optional): Random seed for reproducibility. Default is 42.
  Returns:
    dict: A dictionary {'train','test','val'} containing the paths to the saved train, validation, and test split CSV files.
  Raises:
    AssertionError: If the sum of train_size, val_size, and test_size is not equal to 1.
    Exception: If train, validation, and test splits have common elements.
  Notes:
  - The function attempts to generate splits up to 50 times to ensure each split has at least one sample per class.
  - The input CSV file is expected to have columns separated by tabs ('\t'). 
  """
  def _check_class_distribution(split_dict,y_unique):
    """ Check if each split has at least one sample per class. """
    for split_name, split_data in split_dict.items():
      classes_in_split = np.unique(split_data[:, 2])
      if len(classes_in_split) != len(y_unique):
        # print(f"Error: Not all classes are represented in the {split_name} split. Try another split...")
        return False
    print("All splits have at least one sample per class.")
    return True
  def _save_split(split_dict, video_labels_columns, saving_path, split_dict_indices):
    # Sanity check
    set_train = set(split_dict['train'][:, 0].astype(int))
    set_test = set(split_dict['test'][:, 0].astype(int)) if test_size != 0.0 else set()
    set_val = set(split_dict['val'][:, 0].astype(int)) if val_size != 0.0 else set()

    if set_train & set_test:
        raise ValueError('Error: train and test split have common elements')
    if set_train & set_val:
        raise ValueError('Error: train and validation split have common elements')
    if set_test & set_val:
        raise ValueError('Error: validation and test split have common elements') 
    # Save the splits
    save_path_dict = {}
    for k,v in split_dict.items():
      pth = os.path.join(saving_path, f'{k}.csv')
      save_path_dict[k] = pth
      split_df = pd.DataFrame(v, columns=video_labels_columns)
      split_df.to_csv(pth, index=False, sep='\t')
      print(f'{k} csv split saved to {pth}')
    # Save all the indices of the split
    pth = os.path.join(saving_path, 'split_indices.json')
    with open(pth, 'w') as f:
      json.dump(split_dict_indices, f, cls=NpEncoder)
      print(f'Split indices saved to {pth}')
    return save_path_dict
  
  def _generate_splits():
    video_labels = pd.read_csv(csv_path)
    csv_array = video_labels.to_numpy()
    video_labels_columns = video_labels.columns.to_numpy()[0].split('\t')
    # ['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']
    list_samples = []
    for entry in csv_array:
      tmp = entry[0].split("\t")
      list_samples.append(tmp)
    list_samples = np.stack(list_samples)
    split_dict = {}
    X = list_samples
    y = list_samples[:, 2]  # class ID
    groups = list_samples[:, 0]  # subject ID

    gss = GroupShuffleSplit(n_splits=1,
                            train_size=train_size,
                            test_size=test_size + val_size,
                            random_state=random_state)

    split = list(gss.split(X, y, groups=groups))  # tmp to split in validation and test
    train_split_idx = split[0][0]
    split_dict['train'] = list_samples[train_split_idx]
    # Further split temp into validation and test
    tmp_split = split[0][1]
    X_tmp = X[tmp_split]
    y_tmp = y[tmp_split]
    groups_tmp = groups[tmp_split]
    # print(f'split generation seed: {random_state}')
    gss_tmp = GroupShuffleSplit(n_splits=1,
                                 train_size=(1 / (val_size + test_size)) * val_size,  # 1/(val_size+test_size) = 1/0.2 = 5 => 5 * val_size = 0.5
                                 test_size=(1 / (val_size + test_size)) * test_size,
                                 random_state=random_state)

    if val_size == 0:
      split_dict['test'] = list_samples[tmp_split]
      split_dict_sample_ids = {'train': list_samples[train_split_idx][:, 4], 'test': list_samples[tmp_split][:, 4]}
      return split_dict, video_labels_columns, split_dict_sample_ids

    if test_size == 0:
      split_dict['val'] = list_samples[tmp_split]
      split_dict_sample_ids = {'train': list_samples[train_split_idx][:, 4], 'val': list_samples[tmp_split][:, 4]}
      return split_dict, video_labels, split_dict_sample_ids

    val_test_split = list(gss_tmp.split(X_tmp, y_tmp, groups=groups_tmp))
    val_split_idx = val_test_split[0][0]
    test_split_idx = val_test_split[0][1]
    split_dict['val'] = list_samples[tmp_split[val_split_idx]]
    split_dict['test'] = list_samples[tmp_split[test_split_idx]]
    
    split_dict_sample_ids = {'train': (list_samples[train_split_idx][:, 4].astype(int),train_split_idx),
                'val': (list_samples[tmp_split[val_split_idx]][:, 4].astype(int),tmp_split[val_split_idx]),
                'test': (list_samples[tmp_split[test_split_idx]][:, 4].astype(int),tmp_split[test_split_idx])}

    return split_dict, video_labels_columns, split_dict_sample_ids
  
  ################################################################################################################
  assert train_size + val_size + test_size == 1, "train_size + validation_size + test_size must be equal to 1"
  _,classes=get_unique_subjects_and_classes(csv_path)
  for _ in range(50): # attemps to generate a split
    split_dict, video_labels_columns, split_dict_indices = _generate_splits()
    if _check_class_distribution(split_dict=split_dict, 
                                 y_unique=classes):
      break
  print(f'nr_train_samples: {len(split_dict["train"])}')
  print(f'nr_test_samples: {len(split_dict["test"])}')
  print(f'nr_val_samples: {len(split_dict["val"])}')
  save_path_dict = _save_split(split_dict=split_dict,video_labels_columns=video_labels_columns,
                               saving_path=saving_path,split_dict_indices=split_dict_indices) # in saving_path

  return save_path_dict

def save_split_indices(split_indices, folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
  split_indices_file = os.path.join(folder_path, 'split_indices.json')
  with open(split_indices_file, 'w') as f:
    json.dump(split_indices, f, cls=NpEncoder)

def read_split_indices(folder_path):
  split_indices_file = os.path.join(folder_path, 'split_indices.json')
  with open(split_indices_file, 'r') as f:
    split_indices = json.load(f)
  return split_indices

def generate_plot_train_val_results(dict_results,best_model_idx, count_y_train, count_y_test, count_subject_ids_train, count_subject_ids_test, saving_path,criterion):  
  saving_path_losses = os.path.join(saving_path, 'losses')
  if not os.path.exists(saving_path_losses):
    os.makedirs(saving_path_losses)
  print(f'saving_path_losses: {saving_path_losses}')
  plot_error_per_class(title='training', 
                     mae_per_class=dict_results['train_loss_per_class'][best_model_idx], 
                     unique_classes=dict_results['y_unique'], count_classes=count_y_train,
                     criterion=criterion,
                     saving_path=os.path.join(saving_path_losses,f'train_mae_per_class_{best_model_idx}.png'))
  plot_error_per_class(title='val',
                     mae_per_class=dict_results['val_loss_per_class'][best_model_idx], 
                     unique_classes=dict_results['y_unique'], count_classes=count_y_test,
                     criterion=criterion,
                     saving_path=os.path.join(saving_path_losses,f'val_mae_per_class_{best_model_idx}.png'))
  if dict_results['train_loss_per_subject'][best_model_idx].shape != dict_results['train_unique_subject_ids'].shape:
    uniqie_subject_ids_train = dict_results['subject_ids_unique']
    uniqie_subject_ids_val = dict_results['subject_ids_unique']
  else:
    uniqie_subject_ids_train = dict_results['train_unique_subject_ids']
    uniqie_subject_ids_val = dict_results['val_unique_subject_ids']    
  plot_error_per_subject(title='training', 
                       mae_per_subject=dict_results['train_loss_per_subject'][best_model_idx], 
                       uniqie_subject_ids=uniqie_subject_ids_train,
                       count_subjects=count_subject_ids_train,
                       criterion=criterion,
                       saving_path=os.path.join(saving_path_losses,f'train_mae_per_subject_{best_model_idx}.png'))
  plot_error_per_subject(title='val',
                       mae_per_subject=dict_results['val_loss_per_subject'][best_model_idx], 
                       uniqie_subject_ids=uniqie_subject_ids_val,
                       count_subjects=count_subject_ids_test,
                       criterion=criterion,
                       saving_path=os.path.join(saving_path_losses,f'val_mae_per_subject_{best_model_idx}.png'))
  
  plot_losses(dict_results['train_losses'], dict_results['val_losses'], saving_path=os.path.join(saving_path_losses,'train_val_loss.png'))
      
def plot_confusion_matrix(confusion_matrix, title, saving_path):
  # confusion_matrix must be from torchmetrics
  # assert not isinstance(confusion_matrix, ConfusionMatrix), 'confusion_matrix must be from torchmetrics.classification'
  # if not os.path.exists(os.path):
  fig, _ = confusion_matrix.plot() 
  fig.suptitle(title)
  fig.savefig(saving_path)
  plt.close()

def get_unique_subjects_and_classes(csv_path):
  """
  Get the number of times each unique subject ID and class ID appears in video_labels.

  Returns:
    dict: A dictionary with keys 'subject_counts' and 'class_counts', each containing a dictionary
      where the keys are the unique IDs and the values are the counts.
  """
  csv_array = pd.read_csv(csv_path).to_numpy()
  list_samples = [entry[0].split("\t") for entry in csv_array]
  list_samples = np.stack(list_samples)
  
  subject_ids = list_samples[:, 0].astype(int)
  class_ids = list_samples[:, 2].astype(int)
  
  subject_counts = {subject_id: np.sum(subject_ids == subject_id) for subject_id in np.unique(subject_ids)}
  class_counts = {class_id: np.sum(class_ids == class_id) for class_id in np.unique(class_ids)}
  
  return subject_counts, class_counts

def _plot_dataset_distribution(csv_path, total_classes=None,per_class=False, per_partecipant=False, saving_path=None): 
    def plot_distribution(unique,count,title):  
      plt.figure(figsize=(10, 5))
      plt.bar(unique.astype(str), count, color='blue')
      plt.xlabel('User ID', fontsize=16)
      plt.ylabel('Samples', fontsize=16)
      plt.xticks(fontsize=16, rotation=0)
      plt.yticks(fontsize=16)
      plt.grid(axis="y", linestyle="--", alpha=0.7)
      # dataset_name = f'{os.path.split(self.path_labels)[-1]}'
      plt.title('Dataset Distribution ' + title +f' ({os.path.split(csv_path)[-1]})',fontsize=16)
      if saving_path is not None:
        plt.savefig(os.path.join(saving_path,f'{title}.png'))
      else:
        plt.show()
      plt.xticks(fontsize=14,rotation=0)
      plt.close()
    def plot_distribution_stacked(unique, title, class_counts,total_classes): # suppose that there is at least one sample per class
      # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
      fig, ax = plt.subplots(figsize=(20, 10))
      cmap = plt.get_cmap('tab10')
      cmap = [cmap(i) for i in range(total_classes)]
      bottom = np.zeros(len(unique))
      for i, (class_id, class_count) in enumerate(class_counts.items()):
        unique_people = np.sum(class_count > 0)
        
        ax.bar(unique.astype(str), class_count, bottom=bottom, label=f'{class_id} ({unique_people}/{len(class_count)})',color=cmap[i])
        bottom += class_count
      ax.set_xlabel('User ID', fontsize=14)
      ax.set_ylabel('# Samples', fontsize=14)
      dataset_name = f'{os.path.split(csv_path)[-1].split(".")[0]}'
      ax.set_title('Dataset Distribution ' + title + f' ({dataset_name})', fontsize=16)
      ax.legend(title='Pain level (unique_people/tot_people)')
      plt.xticks(fontsize=13,rotation=45)
      plt.yticks(fontsize=13)
      plt.grid(axis="y", linestyle="--", alpha=0.7)
      if saving_path is not None:
        plt.savefig(os.path.join(saving_path,f'{title}_{dataset_name}.png'))
      else:
        plt.show()
      plt.xticks(fontsize=14,rotation=0)
      plt.close()
    #Extract csv and postprocess
    csv_array = pd.read_csv(csv_path).to_numpy()  # subject_id, subject_name, class_id, class_name, sample_id, sample_name
    list_samples = []
    for entry in csv_array:
      tmp = entry[0].split("\t")
      list_samples.append(tmp)
    list_samples = np.stack(list_samples)
    # total_classes = len(np.unique(list_samples[:, 2].astype(int)))
    if per_class and per_partecipant:
      assert total_classes is not None, 'total_classes must be provided having per_class=True and per_partecipant=True'
      unique_subject_id = np.unique(list_samples[:, 0].astype(int)) # subject_id
      # print(f'unique_subject_id: {unique_subject_id}')
      class_ids =np.unique(list_samples[:, 2].astype(int)) # class_id TODO: use a number of predefinited class to see if there are missing classes
      # class_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # class_id
      class_counts = {class_id: np.zeros(len(unique_subject_id)) for class_id in class_ids} # for each class, create a list of zeros for each participant
      for i, subject_id in enumerate(unique_subject_id):
        for class_id in class_ids:
          # Count the number of samples for each class and participant
          class_counts[class_id][i] = np.sum((list_samples[:, 0].astype(int) == subject_id) & (list_samples[:, 2].astype(int) == class_id))
      # print(class_counts)
      plot_distribution_stacked(unique_subject_id, 'per participant and class', class_counts, total_classes)
    
    elif per_class:
      unique_subject_id,count = np.unique(list_samples[:,2],return_counts=True) 
      unique_subject_id = np.sort(unique_subject_id.astype(int))
      plot_distribution(unique_subject_id,count,'per class')
    
    elif per_partecipant: 
      unique_subject_id,count = np.unique(list_samples[:,0],return_counts=True)
      unique_subject_id = np.sort(unique_subject_id.astype(int))
      plot_distribution(unique_subject_id,count,'per participant')

def plot_dataset_distribution_mean_std_duration(csv_path, video_path=None, per_class=False,per_partecipant=False,saving_path=None):
  def plot_distribution(key,title,video_folder_path):  
    key_dict = {} # key: subject_id -> value: [duration, N]
    for idx, sample in enumerate(list_samples):
      # csv_array = self.video_labels.iloc[idx,0].split('\t') 
      key_id = sample[key]
      video_path = os.path.join(video_folder_path, sample[1], sample[5])
      video_path += '.mp4'
      container = av.open(video_path)
      duration_secs = container.streams.video[0].frames // container.streams.video[0].average_rate
      if key_id not in key_dict:
        key_dict[key_id] = []
        key_dict[key_id].append(duration_secs)
    key_id_vector = np.array(list(key_dict.keys())).astype(int)
    mean_duration = np.array([np.mean(key_dict[key]) for key in key_dict.keys()])
    std_duration = np.array([np.std(key_dict[key]) for key in key_dict.keys()])
    indices = key_id_vector.argsort() # TODO: print also elements not availables 
    dataset_name = f'{os.path.split(csv_path)[-1].split(".")[0]}'

    plt.figure(figsize=(30, 12))
    plt.bar(key_id_vector[indices].astype(str), mean_duration[indices], yerr=std_duration[indices], color="blue", alpha=0.8)
    plt.xlabel(title,fontsize=16)
    plt.ylabel("mean duration (s)",fontsize=16)
    plt.title(f"Mean Duration per {title} with std ({dataset_name})",fontsize=16)
    plt.xticks(rotation=0,fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # Show the plot
    # plt.tight_layout()
    if saving_path is not None:
      plt.savefig(os.path.join(saving_path,f'{title}_{dataset_name}.png'))
    else:
      plt.show()
    plt.close()
  
  csv_array=pd.read_csv(csv_path).to_numpy() # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  list_samples=[]
  for entry in (csv_array):
    tmp = entry[0].split("\t")
    list_samples.append(tmp)
  list_samples = np.stack(list_samples)
  if video_path is None:
    video_path = os.path.join('partA','video','video')
  if per_partecipant is True:
    key = 0
    plot_distribution(key,'participant',video_path)
  if per_class is True:
    key = 2
    plot_distribution(key,'class',video_path)

def plot_prediction_chunks_per_subject(predictions, n_chunks,title,saving_path=None):
  print('Plotting...')
  print('  predictions shape', predictions.shape) # (n_samples, n_chunks)
  print('  n_chunks', n_chunks)
  indices = np.arange(n_chunks).astype(int)
  print('indices', indices)
  plt.figure(figsize=(6, 4))
  plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.plot(indices, predictions)
  plt.xlabel('Frame chunk index', fontsize=11)
  plt.ylabel('Prediction', fontsize=11)
  plt.title(title, fontsize=14)
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  plt.tight_layout()
  if saving_path is not None:
    plt.savefig(saving_path+'.png')
  else:
    plt.show()
  plt.close()

 
def compute_tsne(X, labels=None, tsne_n_component = 2,apply_pca_before_tsne=False,legend_label='', title = '', perplexity=1, saving_path=None,plot=True,cmap='copper'):
  """
  Plots the t-SNE reduction of the features in 2D with colors based on subject, gt, or predicted class.
  Args:
    color_by (str, optional): Criterion for coloring the points ('subject', 'label', 'prediction'). Defaults to 'subject'.
    use_cuda (bool, optional): Whether to use CUDA for t-SNE computation. Defaults to False.
    perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 20.
  """
  print(f'TSNE_X.shape: {X.shape}')
  start = time.time()
  if labels is not None and len(labels.shape) != 1:
    print(f'labels shape: {labels.shape}')
    labels = labels.reshape(-1)
    # unique_labels = np.unique(labels)
  perplexity = min(20, X.shape[0]-1)
  print('Using CPU')
  tsne = openTSNE(n_components=tsne_n_component, perplexity=perplexity, random_state=42, n_jobs = 1)
  if isinstance(X, torch.Tensor):
    X_cpu = X.detach().cpu().squeeze()
  X_cpu = X_cpu.reshape(X_cpu.shape[0], -1)
  # apply PCA from scikit-learn to reduce the dimensionality of the data
  if apply_pca_before_tsne:
    n_components_pca = min(50, X_cpu.shape[0])
    print(f'PCA using {n_components_pca} components...')
    pca = PCA(n_components=n_components_pca)
    X_cpu = pca.fit_transform(X_cpu)
    # X_cpu = X_cpu[:, 1:]  # Exclude the first principal component
  # print(f' X_cpu.shape: {X_cpu.shape}')
  print("Start t-SNE computation...")
  X_tsne = tsne.fit(X_cpu) # OpenTSNE
  # X_tsne = X_cpu
  print(f'X_tsne shape: {X_tsne.shape}')
  # get the folder of saving_path
  # print(f'path {os.path.split(saving_path)[:-1]}')
  if saving_path:
    path_log_tsne = os.path.join(os.path.split(saving_path)[:-1][0],'log_time_tsne.txt')
    if not os.path.exists(os.path.split(saving_path)[:-1][0]):
      os.makedirs(os.path.split(saving_path)[:-1][0])
    with open(path_log_tsne, 'a') as f:
      f.write(f'{title} \n')
      f.write(f'  time: {time.time()-start} secs\n')
      # f.write(f'  perplexity: {X_tsne.affinities.perplexities}\n')
      f.write(f'  X_tsne.shape: {X_tsne.shape}\n')
      f.write(f'  apply_pca_before_tsne: {apply_pca_before_tsne}\n')
      if apply_pca_before_tsne:
        f.write(f'  n_components_pca: {n_components_pca}\n')
        f.write(f'  PCA explained variance ratio: {pca.explained_variance_ratio_.cumsum()}\n')
      f.write('\n')
  # X_tsne = tsne.fit_transform(X_cpu) # in: X=(n_samples, n_features)
                                     # out: (n_samples, n_components=2)
  # print(" t-SNE computation done.")
  # print(f' X_tsne.shape: {X_tsne.shape}')
  if plot:
    plot_tsne(X_tsne=X_tsne,
              labels=labels,
              cmap=cmap,
              legend_label=legend_label,
              title=title,
              saving_path=saving_path)
    return np.array(X_tsne)
  else:
    # print(f'X_tsne type: {np.array(X_tsne)}')
    return np.array(X_tsne)
  # print(f' labels shape: {labels.shape}')


def plot_tsne(X_tsne, labels, cmap='copper',tot_labels = None,legend_label='', title='', cluster_measure='', 
              saving_path=None, axis_scale=None, last_point_bigger=False, plot_trajectory=False, 
              stride_windows=None,clip_length=None,list_axis_name=None,ax=False,return_ax=False):
  unique_labels = np.unique(labels)
  if tot_labels is None:
    color_map = plt.cm.get_cmap(cmap, len(unique_labels))
  else:
    color_map = plt.cm.get_cmap(cmap, tot_labels)
  color_dict = {val: color_map(i) for i, val in enumerate(unique_labels)}
  sizes = None
  if not return_ax:
    fig = plt.figure(figsize=(10, 8))

  
  if not return_ax:
    ax = fig.add_subplot(111)
  if axis_scale is not None:
    ax.set_xlim(axis_scale['min_x'], axis_scale['max_x'])
    ax.set_ylim(axis_scale['min_y'], axis_scale['max_y'])
  if last_point_bigger:
    sizes = [50] * (X_tsne.shape[0] - 1) + [200]
    sizes = np.array(sizes)
  for val in unique_labels:
    idx = np.array(labels == val)
    if clip_length is not None and legend_label == 'clip':
      
      idx_color = 0 if stride_windows*val < 48 else max(color_dict.keys())
      label = f'{legend_label} [{stride_windows * val}, {clip_length + stride_windows * (val) - 1}] ({cluster_measure[idx_color]:.2f})' if cluster_measure != '' and idx_color < len(cluster_measure) else f'{legend_label} [{stride_windows * val}, {clip_length + stride_windows * (val) - 1}]'
      ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=color_dict[idx_color], label=label, alpha=0.7, s=sizes[idx] if sizes is not None else 50)
    else:
      label = f'{legend_label} {val} ({cluster_measure[val]:.2f})' if cluster_measure != '' else f'{legend_label} {val}'
      ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=color_dict[val], label=label, alpha=0.7, s=sizes[idx] if sizes is not None else 50)
  if plot_trajectory:
    ax.plot(X_tsne[:, 0], X_tsne[:, 1], linestyle='--', color=color_dict[0], label='Trajectory', alpha=0.7)
  if list_axis_name is not None:
    ax.set_xlabel(list_axis_name[0])
    ax.set_ylabel(list_axis_name[1])
  else:
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
  
  ax.legend()
  # add another legend
  
  ax.set_title(f'{title} (Colored by {legend_label})')
  
  if not return_ax:
    plt.close(fig)
  
  if saving_path is None:
    # plt.show()
    if return_ax:
      return None
    fig.canvas.draw()
    rgb_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    width, height = fig.canvas.get_width_height()
    rgb_array = rgb_array.reshape(height, width, 3)
    return rgb_array
  else:
    print(f'Saving plot to {saving_path}')
    if saving_path[-4:] != '.png':
      os.makedirs(saving_path, exist_ok=True)
      pth = os.path.join(saving_path, f'{title}_{legend_label}.png')
    else:
      pth = saving_path
    fig.savefig(pth)
    print(f'Plot saved to {pth}')
    return pth

def get_list_video_path_from_csv(csv_path, cols_csv_idx=[1,5], video_folder_path=None):
  list_samples,_ = get_array_from_csv(csv_path) # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  if video_folder_path is None:
    video_folder_path = os.path.join('partA','video','video')
  list_video_path = []
  for sample in list_samples:
    # sample_video = union_segment.join(sample[cols_csv_idx])
    sample_video = os.path.join(sample[cols_csv_idx[0]], sample[cols_csv_idx[1]])
    video_path = os.path.join(video_folder_path, sample_video)
    video_path += '.mp4'
    list_video_path.append(video_path)
  return list_video_path

def get_array_from_csv(csv_path):
  """
  Reads a CSV file and converts it into a NumPy array.

  Args:
    csv_path (str): The file path to the CSV file.

  Returns:
    tuple: A tuple containing:
      - np.ndarray: Array where each row represents a sample from the CSV file.
      - np.ndarray: Array containing the column names from the CSV file.
  """
  csv_array = pd.read_csv(csv_path)  # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  cols_array = csv_array.columns.to_numpy()[0].split('\t')
  csv_array = csv_array.to_numpy()
  list_samples = []
  for entry in csv_array:
    tmp = entry[0].split("\t")
    list_samples.append(tmp)
  return np.stack(list_samples),cols_array

def save_frames_as_video(list_input_video_path, list_frame_indices,sample_ids, output_video_path, all_predictions, list_ground_truth, output_fps=1):
  """
  Extract specific frames from a video and save them as a new video.

  :param input_video_path: Path to the original video file.
  :param frame_indices: List of frame indices to extract and save.
  :param output_video_path: Path to save the output video.
  :param output_fps: Frames per second for the output video (default is 30).
  """
  # Open the original video
  out = None
  # print(f' list_input_video_path: {list_input_video_path}')
  # print(f' list_frame_indices: {list_frame_indices.shape}')
  # print(f' sample_ids: {sample_ids.shape}')
  # print(f' all_predictions: {all_predictions.shape}')
  # print(f' list_ground_truth: {list_ground_truth.shape}')
  
  unique_sample_ids = np.unique(sample_ids,return_counts=False)
  for input_video_path, sample_id in (zip(list_input_video_path,unique_sample_ids)): # i->[0...33]
    print(f'input_video_path: {input_video_path}')
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
      raise IOError(f"Err: Unable to open video file: {input_video_path}")

    # Get the width ansample_idsd height of the frames in the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')   
    if out is None:
      out = cv2.VideoWriter(output_video_path, fourcc, output_fps, frame_size)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f'frame_count: {frame_count}')
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 2
    count = 0
    idxs = torch.nonzero(sample_ids == sample_id)
    # print(f'frame_indices: {len(list_frame_indices)}') # [36,16]
    for j, frame_indices in (zip(idxs, list_frame_indices[idxs])): #j->[0..1] [16]
      # print('frame_indices',frame_indices) # i->[0,...,32] framle_inidces->[2,16]
      cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the video capture to the beginning
      for _ in range(output_fps):
        number_frame = black_frame.copy()
        text = str(count)
        # print(f'   text: {text}')
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (number_frame.shape[1] - text_size[0]) // 2
        text_y = (number_frame.shape[0] + text_size[1]) // 2
        cv2.putText(number_frame, text, (text_x, text_y), font, font_scale, font_color, thickness)
        out.write(number_frame)
      for frame_idx in range(frame_count):
        if frame_idx in frame_indices: # [2,16]
          if frame_idx >= frame_count:
            print(f"WARNING: Frame index {frame_idx} out of range for video {input_video_path}")
            continue
          cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
          ret, frame = cap.read()
          if not ret:
            print(f"WARNING: Failed to read frame {frame_idx} from video {input_video_path}")
            continue
          # print(f'j: {j}')
          # print(f'list_ground_truth[j]: {list_ground_truth[j]}')
          # print(f'all_predictions[j]: {all_predictions[j]}')
          cv2.putText(frame, str(count)+'/'+str(frame_idx), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0), thickness, cv2.LINE_AA)
          cv2.putText(frame, f'gt:{list_ground_truth[j].item()}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), thickness, cv2.LINE_AA)
          cv2.putText(frame, f'pred:{(all_predictions[j].item()):.2f}', (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), thickness, cv2.LINE_AA)
          out.write(frame)
      count+=1
        # print(f'frame_idx: {frame_idx}')
  # Release resources
  cap.release()
  out.release()
  print(f"Saved extracted frames to {output_video_path}")
  # Add 4 black frames at the end

def _generate_csv_subsampled(csv_dataset_path, nr_samples_per_class=2):
  csv_array, video_labels_columns =get_array_from_csv(csv_dataset_path)
  # ['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']
  list_samples=[]
  for entry in (csv_array):
    tmp = entry[0].split("\t")
    list_samples.append(tmp)
  list_samples = np.stack(list_samples)
  nr_classes = np.max(list_samples[:,2].astype(int))
  print(f'number of classes: {nr_classes}, \ntotal number of samples: {nr_samples_per_class*nr_classes}')
  for cls in range(nr_classes):
    samples = list_samples[list_samples[:,2].astype(int) == cls]
    samples = samples[np.random.choice(samples.shape[0], nr_samples_per_class, replace=False), :]
    if cls == 0:
      samples_subsampled = samples
    else:
      samples_subsampled = np.concatenate((samples_subsampled,samples),axis=0)
  # print(f'samples_subsampled: {samples_subsampled}')
  save_path = os.path.join('partA','starting_point','subsamples_'+str(nr_samples_per_class)+'_'+str(nr_samples_per_class*nr_classes)+'.csv')
  subsampled_df = pd.DataFrame(samples_subsampled, columns=video_labels_columns)
  subsampled_df.to_csv(save_path, index=False, sep='\t')
  print(f'Subsampled video labels saved to {save_path}')

def generate_csv(cols, data, saving_path):
  df = pd.DataFrame(data, columns=cols)
  df.to_csv(saving_path, index=False, sep='\t')
  print(f'CSV saved to {saving_path}')

def generate_video_from_list_video_path(list_video_path, list_frames, list_subject_id, idx_list_frames, list_sample_id, list_y_gt, saving_path, output_fps=4,list_rgb_image_plot=None):
  """
  Generate a video by extracting specific frames from a list of videos.

  Args:
    list_video_path (list): List of paths to the input video files.
    list_frames (list): List of lists, where each sublist contains the frame indices to extract from the corresponding video.
    idx_list_frames (list): List of indices corresponding to the frames to be extracted.
    list_sample_id (list): List of sample IDs corresponding to the videos.
    list_y_gt (list): List of ground truth labels corresponding to the videos.
    saving_path (str): Path to save the generated video.
    list_image_path (list, optional): List of paths to images to be merged with the video frames. Defaults to None.
  """
  out = None
  # output_fps = 4
  count = 0
  print('Generating video...')
  current_video_path = None
  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  if list_video_path[0].split('/') != 'partA':
    list_video_path = [video_path.split('PainAssessmentVideo')[1][1:] for video_path in list_video_path]
  for video_path, frames, sample_id, clip, y_gt, image, subject_id in (zip(list_video_path, list_frames, list_sample_id, idx_list_frames,
                                                                    list_y_gt, list_rgb_image_plot or [None]*len(list_video_path),
                                                                    list_subject_id)):
    if current_video_path is None or current_video_path != video_path:
      print(f'video_path: {video_path}')
      if current_video_path is not None:
        cap.release()
      cap = cv2.VideoCapture(video_path)
      if not cap.isOpened():
        raise IOError(f"Error: Unable to open video file: {video_path}")

      frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      frame_size = (frame_width, frame_height)
      frames_to_process = []
      while True:
        ret, frame = cap.read()
        if not ret:
          break
        frames_to_process.append(frame)    
      current_video_path = video_path
    # else:
    #   cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
      
    if image is not None:
      # image = cv2.imread(image_path)
      image_height, image_width = image.shape[:2]
      frame_size = (frame_width + image_width, max(frame_height, image_height))
    if out is None:
      out = cv2.VideoWriter(os.path.join(saving_path, 'video.mp4'), fourcc, output_fps, frame_size)
    
    overlay_text = [
        f'Sample ID  : {sample_id}',
        f'Subject ID : {subject_id}',
        f'Pain class : {y_gt}',
        f'Clip num.  : {clip}',
        f'Frame range: [{frames[0]},{frames[-1]}]'
    ]
    
    for frame in frames_to_process[frames[0]:frames[-1]+1]:
      # print(f'type frame_idx: {type(frame_idx)}')
      # cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
      # ret, frame = cap.read()
      # # fps = cap.get(cv2.CAP_PROP_FPS)
      # if not ret:
      #   print(f"Warning: Failed to read frame {frame_idx} from video {video_path}")
      #   continue

      if image is not None:
        # image = cv2.imread(image_path)
        combined_frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        combined_frame[:frame_height, :frame_width] = frame
        combined_frame[:image.shape[0], frame_width:(frame_width + image.shape[1])] = image
        frame = combined_frame
      for i, text in enumerate(overlay_text):
        cv2.putText(frame, text, (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

      # cv2.putText(frame, f'Sample ID  : {sample_id}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
      # cv2.putText(frame, f'Subject ID : {subject_id}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
      # cv2.putText(frame, f'Pain class : {y_gt}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
      # cv2.putText(frame, f'Clip num.  : {clip} ', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
      # cv2.putText(frame, f'Frame range: [{frames[0]},{frames[-1]}] ', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
      out.write(frame)
    count+=1
    if count % 10 == 0:
      print(f'Processed {count}/{list_video_path.shape[0]} videos')
    # black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    # for _ in range(output_fps // 2):
    #   number_frame = black_frame.copy()
    #   out.write(number_frame)

  out.release()
  print(f"Generated video saved to folder {saving_path}")
    
def generate_video_from_list_frame(list_frame,path_video_output,fps=25):
  if not os.path.exists(os.path.split(path_video_output)[0]):
    os.makedirs(os.path.split(path_video_output)[0])
  out = cv2.VideoWriter(path_video_output, cv2.VideoWriter_fourcc(*'avc1'), fps, (list_frame[0].shape[1], list_frame[0].shape[0]))
  for frame in list_frame:
    if not isinstance(frame,np.ndarray):
      frame = np.array(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
  out.release()
  print(f'Video saved to {path_video_output}')
# def save_tsne_incrementsl_plots_(X_tsne, labels, saving_path):
  
def get_list_frame_from_video_path(video_path):
  cap = cv2.VideoCapture(video_path)
  frame_list = []
  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_list.append(frame)
  return frame_list

def plot_macro_accuracy(list_train_accuracy,list_val_accurcay, title, x_label, y_label, saving_path=None):
  if not os.path.exists(os.path.split(saving_path)[0]):
    os.makedirs(os.path.split(saving_path)[0])
  fig, ax = plt.subplots()
  ax.plot(list(range(len(list_train_accuracy))), list_train_accuracy)
  ax.plot(list(range(len(list_val_accurcay))), list_val_accurcay)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  # plt.xticks(rotation=45)
  plt.tight_layout()
  plt.legend(['train_accuracy','val_accuracy'])
  if saving_path is not None:
    plt.savefig(saving_path)
  else:
    return fig
  plt.close()
  
def plot_bar(data, title, x_label, y_label, saving_path=None):
  fig, ax = plt.subplots()
  ax.bar(data.keys(), data.values(), color='blue', width=0.8, label='Error per Class',edgecolor='black',align='center')
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  # plt.xticks(rotation=45)
  plt.tight_layout()
  if saving_path is not None:
    plt.savefig(saving_path)
  else:
    return fig
  plt.close()

def subplot_loss(dict_losses,x_label,y_label,list_title, saving_path=None):
  fig, ax = plt.subplots(len(dict_losses),1, figsize=(20, 20))
  i = 0
  for k,v in dict_losses.items():
    # if v['loss'].shape != v['elements'].shape:
    #   v['loss'] = v['loss'][:v['elements'].shape[0]]
    elements = [str(i) for i in v['elements']]
    ax[i].bar(elements,v['loss'], color='blue', width=0.8, label='Error per Class',edgecolor='black',align='center')
    ax[i].set_xlabel(x_label)
    ax[i].set_ylabel(y_label)
    ax[i].set_title(list_title[i])
    i+=1
  plt.tight_layout()
  if saving_path is not None:
    plt.savefig(saving_path)
  else:
    return fig
  plt.close()
  

def save_dict_k_fold_results(dict_k_fold_results, folder_path):
  """
  Save the dict of k-fold results to the specified folder.
  Args:
      list_k_fold_results (list): List of k-fold results to be saved.
      folder_path (str): Path to the folder where the results will be saved.
  """
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)
  
  results_path = os.path.join(folder_path, 'k_fold_results.pkl')
  start = time.time()
  print(f'Saving k-fold results to {results_path}...')
  with open(results_path, 'wb') as f:
    pickle.dump(dict_k_fold_results, f)
  print(f'k-fold results saved to {results_path} in {int(time.time()-start)} secs')
  
def convert_split_indices_to_video_path(path_split_indices_folder,
                                       path_to_extracted_feature,
                                       split_key='val'):
  dict_sample_indices = read_split_indices(path_split_indices_folder) # [[list_sample_id],[indices]]
  dict_all_features = load_dict_data(path_to_extracted_feature)
  unique_sample_ids = dict_sample_indices[split_key][0]
  list_path = []
  for sample_id in unique_sample_ids:
    mask = (sample_id == dict_all_features['list_sample_id'])
    unique_path = np.unique(dict_all_features['list_path'][mask])
    list_path.append(unique_path[0])
  return list_path
def calculate_wcss(embeddings, labels):
  """Within-Cluster Sum of Squares implementation"""
  clusters = np.unique(labels)
  wcss = 0
  for cluster in clusters:
    cluster_points = embeddings[labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    wcss += np.sum((cluster_points - centroid)**2)
  return wcss

def pairwise_distance(embeddings, labels):
  """Mean intra-cluster distance calculation"""
  clusters = np.unique(labels)
  avg_distances = {}
  for cluster in clusters:
    cluster_points = embeddings[labels == cluster]
    if len(cluster_points) > 1:
        distances = pdist(cluster_points)
        avg_distances[cluster] = np.mean(distances)
  return avg_distances

def cluster_radius(embeddings, labels):
  """Maximum distance from centroid implementation"""
  clusters = np.unique(labels)
  radii = {}
  for cluster in clusters:
    cluster_points = embeddings[labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    distances = np.linalg.norm(cluster_points - centroid, axis=1)
    radii[cluster] = np.max(distances)
    
  return radii

# Built-in metrics from scikit-learn
def get_silhouette_score(embeddings, labels):
    return silhouette_score(embeddings, labels)

def get_davies_bouldin_index(embeddings, labels):
    return davies_bouldin_score(embeddings, labels)
    
# TODO: select the right stride window for each video when read data from SSD
def plot_and_generate_video(folder_path_features,folder_path_tsne_results,subject_id_list,clip_list,class_list,sliding_windows,legend_label,create_video=True,
                            plot_only_sample_id_list=None,tsne_n_component=2,plot_third_dim_time=False,apply_pca_before_tsne=False,cmap='copper',
                            sort_elements=True,axis_dict=None,feat_mean=False,csv_path=None):

  dict_all_features = load_dict_data(folder_path_features)
  # print(dict_all_features.keys())
  print(f'dict_all_features["list_subject_id"] shape {dict_all_features["list_subject_id"].shape}')
  time_start = time.time()
  idx_subjects = np.any([dict_all_features['list_subject_id'] == id for id in subject_id_list],axis=0)
  idx_class = np.any([dict_all_features['list_labels'] == id for id in class_list],axis=0)
  filter_idx = np.logical_and(idx_subjects,idx_class)
  if plot_only_sample_id_list is not None:
    print(f'Warning: Using sample id will ignore subject_id_list and class_list')
    filter_idx = np.any([dict_all_features['list_sample_id'] == id for id in plot_only_sample_id_list],axis=0)
  # Filter for clip_list
  _, list_count_clips = np.unique(dict_all_features['list_sample_id'],return_counts=True)
  arange_clip = range(max(list_count_clips)) # suppose clip_list is ordered
  clip_list_array = np.array([True if i in clip_list else False for i in arange_clip]) 
  filter_clip = np.concatenate([clip_list_array[:end] for end in list_count_clips])
  filter_idx = np.logical_and(filter_idx, filter_clip)
  
  list_frames = []
  list_sample_id = []
  list_subject_id = []
  list_video_path = []
  list_feature = []
  list_idx_list_frames = []
  list_y_gt = []
  csv_array,_ = get_array_from_csv(csv_path)
  if csv_path:
    sample_id_csv = csv_array[:,4].astype(int)
    filter_sample_csv = np.any([dict_all_features['list_sample_id'] == id for id in sample_id_csv],axis=0)
    print(f'kept {np.sum(filter_sample_csv)} over {len(filter_sample_csv)} samples')
    print(f'Missing video: {np.unique(dict_all_features["list_sample_id"][~filter_sample_csv]).shape[0]}')
    filter_idx = np.logical_and(filter_idx, filter_sample_csv)
    
  list_frames=dict_all_features['list_frames'][filter_idx]
  list_sample_id=dict_all_features['list_sample_id'][filter_idx]
  list_video_path=dict_all_features['list_path'][filter_idx]
  if feat_mean:
    list_feature=torch.mean(dict_all_features['features'][filter_idx],dim=1,keepdim=True)
  else:
    list_feature=dict_all_features['features'][filter_idx]
  # print(f'length list_feature {len(list_feature)}')
  list_y_gt=dict_all_features['list_labels'][filter_idx]
  list_subject_id=dict_all_features['list_subject_id'][filter_idx]
  # print(f'list_sample_id {list_sample_id}')
  list_idx_list_frames=np.concatenate([np.arange(end) for end in list_count_clips])[filter_idx]
  
  if sort_elements:
    class_bool_idxs = [list_y_gt == i for i in class_list]
    list_frames = torch.cat([list_frames[bool_idx] for bool_idx in class_bool_idxs])
    list_sample_id = torch.cat([list_sample_id[bool_idx] for bool_idx in class_bool_idxs])
    list_video_path = np.concatenate([list_video_path[bool_idx] for bool_idx in class_bool_idxs])
    list_feature = torch.cat([list_feature[bool_idx] for bool_idx in class_bool_idxs])
    list_y_gt = torch.cat([list_y_gt[bool_idx] for bool_idx in class_bool_idxs])
    list_subject_id = torch.cat([list_subject_id[bool_idx] for bool_idx in class_bool_idxs])
    list_idx_list_frames = np.concatenate([list_idx_list_frames[bool_idx] for bool_idx in class_bool_idxs])
  
  print('Elasped time to get all features: ',time.time()-time_start)
  print(f'list_frames {list_frames.shape}')
  print(f'list_sample_id {list_sample_id.shape}')
  print(f'list_video_path {list_video_path.shape}')
  print(f'list_feature {list_feature.shape}')
  print(f'list_idx_list_frames {list_idx_list_frames.shape}')
  print(f'list_y_gt {list_y_gt.shape}')
  
  tsne_plot_path = os.path.join(folder_path_tsne_results,f'tsne_plot_{sliding_windows}_{legend_label}')
  
  X_tsne = compute_tsne(X=list_feature,
                           plot=False,
                           saving_path=os.path.join(folder_path_tsne_results,'dummy'),
                           tsne_n_component=tsne_n_component,
                           apply_pca_before_tsne=apply_pca_before_tsne)
  # add 3th dimension to X_tsne
  list_axis_name = None
  if tsne_n_component == 2 and plot_third_dim_time:
    X_tsne = np.concatenate([X_tsne,np.expand_dims(list_idx_list_frames,axis=1)],axis=1)
    X_tsne = X_tsne[:,[2,0,1]] 
    list_axis_name = ['nr_clip','t-SNE_x','t-SNE_y']
  if axis_dict is None:
    if X_tsne.shape[1] == 2:
      min_x,min_y = X_tsne.min(axis=0)
      max_x,max_y = X_tsne.max(axis=0)
      axis_dict = {'min_x':min_x-3,'min_y':min_y-3,'max_x':max_x+3,'max_y':max_y+3}
    else:
      min_x,min_y,min_z = X_tsne.min(axis=0)
      max_x,max_y,max_z = X_tsne.max(axis=0)
      axis_dict = {'min_x':min_x-3,'min_y':min_y-3,'min_z':min_z-3,'max_x':max_x+3,'max_y':max_y+3,'max_z':max_z+3}
  
  print(f'axis_dict {axis_dict}')
  if legend_label == 'clip':
    labels_to_plot = list_idx_list_frames
  elif legend_label == 'subject':
    labels_to_plot = list_subject_id
  elif legend_label == 'class':
    labels_to_plot = list_y_gt
  else:
    raise ValueError('legend_label must be one of the following: "clip", "subject", "class"') 
  
  # labels_to_plot = list_idx_list_frames
  # print(f'clip_length {dict_all_features["list_frames"][filter_idx].shape[1]}')
  if not os.path.exists(tsne_plot_path):
    os.makedirs(tsne_plot_path)
  if plot_only_sample_id_list is None:
    if len(subject_id_list) > 1:
      title_plot = f'{os.path.split(folder_path_features)[-1]}_sliding_{sliding_windows}_tot-subjects_{len(subject_id_list)}__clips_{clip_list}__classes_{(class_list)}'
    else:
      title_plot = f'{os.path.split(folder_path_features)[-1]}_sliding_{sliding_windows}__clips_{clip_list}__classes_{(np.unique(list_y_gt))}__subjectID_{np.unique(list_subject_id)}'
  else:
    title_plot = f'{os.path.split(folder_path_features)[-1]}_sliding_{sliding_windows}_sample_id_{plot_only_sample_id_list}__clips_{len(clip_list)}__classes_{(np.unique(list_y_gt))}__subjectID_{np.unique(list_subject_id)}'
  # print('START PLOT TSNE')
  return {
    'X_tsne':X_tsne,
    'labels':labels_to_plot,
    'saving_path':tsne_plot_path,
    'title':title_plot,
    'legend_label':legend_label,
    'plot_trajectory':True if plot_only_sample_id_list is not None else False,
    'clip_length':dict_all_features['list_frames'][filter_idx].shape[1],
    'stride_windows':sliding_windows,
    'axis_scale':axis_dict,
    'list_axis_name':list_axis_name,
    'cmap':cmap
  }
  plot_tsne(X_tsne=X_tsne,
            labels=labels_to_plot,
            saving_path=tsne_plot_path,
            title=title_plot,
            legend_label=legend_label,
            plot_trajectory = True if plot_only_sample_id_list is not None else False,
            clip_length=dict_all_features['list_frames'][filter_idx].shape[1],
            stride_windows=sliding_windows,
            axis_scale=axis_dict,
            list_axis_name=list_axis_name,
            cmap=cmap)  
  # print('END ONLY PLOT TSNE')
  with open(os.path.join(folder_path_tsne_results,'config.txt'),'w') as f:
    f.write(f'subject_id_list: {subject_id_list}\n')
    f.write(f'clips: {clip_list}\n')
    f.write(f'classes: {class_list}\n')
    f.write(f'sliding_windows: {sliding_windows}\n')
    
  if create_video:
    video_saving_path = os.path.join(folder_path_tsne_results,'video')
    if not os.path.exists(video_saving_path):
      os.makedirs(video_saving_path)
    list_rgb_image_plot = []  
    start = time.time()
    print(f'X_tsne.shape {X_tsne.shape}')
    for i in range(1,X_tsne.shape[0]+1):
      list_rgb_image_plot.append(
                    plot_tsne(X_tsne=X_tsne[:i],
                          labels=labels_to_plot[:i],
                          legend_label=legend_label,
                          title=f'{title_plot}_{i}',
                          # saving_path=video_saving_path,
                          axis_scale=axis_dict,
                          clip_length=dict_all_features['list_frames'][filter_idx].shape[1],
                          stride_windows=sliding_windows,
                          tot_labels=len(np.unique(labels_to_plot)),
                          plot_trajectory=True if plot_only_sample_id_list is not None else False,
                          last_point_bigger=True,
                          list_axis_name=list_axis_name))
      print(f'Elapsed time to get plot {i}: {time.time()-start} s')
    print(f'Elapsed time to get all plots: {time.time()-start} s')
    start = time.time()
    generate_video_from_list_video_path(list_video_path=list_video_path,
                                              list_frames=list_frames,
                                              list_sample_id=list_sample_id,
                                              list_y_gt=list_y_gt,
                                              output_fps=10,
                                              list_subject_id=list_subject_id,
                                              idx_list_frames=list_idx_list_frames,
                                              saving_path=video_saving_path,
                                              list_rgb_image_plot=list_rgb_image_plot)
    print(f'Elapsed time to generate video: {time.time()-start} s')
    # for i in range(len(list_image_path)):
    #   remove_plot(list_image_path[i])
    
def plot_loss_and_precision_details(dict_train, train_folder_path, total_epochs,criterion):
  """
  Generate and save plots of the training and test results, and confusion matrices for each epoch.
  Parameters:
  dict_train (dict): Dictionary containing training results and other relevant data.
  train_folder_path (str): Path to the folder where the plots and confusion matrices will be saved.
  epochs (int): Number of epochs for which the confusion matrices will be plotted.
  """
  # Generate and save plots of the training and test results
  # tools.plot_losses(train_losses=dict_train['dict_results']['train_losses'], 
  #                   test_losses=dict_train['dict_results']['test_losses'], 
  #                   saving_path=os.path.join(train_folder_path,'train_test_losses'))
  

  generate_plot_train_val_results(dict_results=dict_train['dict_results'], 
                                count_subject_ids_train=dict_train['count_subject_ids_train'],
                                count_subject_ids_test=dict_train['count_subject_ids_test'],
                                count_y_test=dict_train['count_y_test'], 
                                count_y_train=dict_train['count_y_train'],
                                saving_path=train_folder_path,
                                criterion=criterion,
                                best_model_idx=dict_train['dict_results']['best_model_idx'])
  
  # Plot and save confusion matrices for each epoch
  confusion_matrix_path = os.path.join(train_folder_path,'confusion_matricies')
  
  if not os.path.exists(confusion_matrix_path):
    os.makedirs(confusion_matrix_path)
  _plot_confusion_matricies(total_epochs, dict_train, confusion_matrix_path,dict_train['dict_results']['best_model_idx'])
  
  plot_macro_accuracy(list_train_accuracy=dict_train['dict_results']['list_train_macro_accuracy'],
                            list_val_accurcay=dict_train['dict_results']['list_val_macro_accuracy'],
                            title='Macro accuracy per epoch',
                            x_label='epochs',
                            y_label='accuracy',
                            saving_path=os.path.join(train_folder_path,'losses','macro_accuracy_train_val.png'))


def create_unique_video_per_prediction(train_folder_path, dict_cvs_path, sample_ids, list_frames, y_pred, y, dataset_name):
  # Create video with predictions
  print(f"Creating video with predictions for {dataset_name}")
  video_folder_path = os.path.join(train_folder_path,f'video')
  if not os.path.exists(video_folder_path):
    os.makedirs(video_folder_path)
  
  list_input_video_path = get_list_video_path_from_csv(dict_cvs_path[dataset_name])
  output_video_path = os.path.join(video_folder_path,f'video_all_{dataset_name}.mp4')

  save_frames_as_video(list_input_video_path=list_input_video_path, # [n_video=33]
                                              list_frame_indices=list_frames, # [33,2,16]
                                              output_video_path=output_video_path, # string
                                              all_predictions=y_pred,#  [33,2]
                                              list_ground_truth=y,
                                              sample_ids=sample_ids, # -> [33,2] 
                                              output_fps=4)


def _plot_confusion_matricies(epochs, dict_train, confusion_matrix_path,best_model_idx):
  for epoch in range(0,epochs,50): 
    plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][epoch],
                                title=f'Train_{epoch} confusion matrix',
                                saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_train_{epoch}.png'))
    
    plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['val_confusion_matricies'][epoch],
                              title=f'Val_{epoch} confusion matrix',
                              saving_path=os.path.join(confusion_matrix_path,f'confusion_matrix_val_{epoch}.png'))
    
  # Plot best model results
  plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][best_model_idx],
                                title=f'Train_{best_model_idx} confusion matrix',
                                saving_path=os.path.join(confusion_matrix_path,f'best_confusion_matrix_train_{best_model_idx}.png'))
  plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['val_confusion_matricies'][best_model_idx],
                              title=f'Val_{best_model_idx} confusion matrix',
                              saving_path=os.path.join(confusion_matrix_path,f'best_confusion_matrix_val_{best_model_idx}.png'))
  
  saving_path_precision_recall = os.path.join(confusion_matrix_path,'plot_over_epochs')
  if not os.path.exists(saving_path_precision_recall):
    os.makedirs(saving_path_precision_recall)
  # TODO: REMOVE comments if want to plot precision and recall
  # unique_classes_train = np.unique(dict_train['dict_results']['list_y_train'])
  # tools.plot_accuracy_confusion_matrix(confusion_matricies=dict_train['dict_results']['train_confusion_matricies'],
  #                                      type_conf='train',
  #                                      saving_path=saving_path_precision_recall,
  #                                      list_real_classes=unique_classes_train)
  
  # tools.plot_accuracy_confusion_matrix(confusion_matricies=dict_train['dict_results']['val_confusion_matricies'],
  #                                      type_conf='test',
  #                                      saving_path=saving_path_precision_recall,
  #                                      list_real_classes=unique_classes_train)
    
def plot_dataset_distribuition(csv_path,run_folder_path,per_class=True,per_partecipant=True,total_classes=None):
  #  Create folder to save the dataset distribution 
  dataset_folder_path = os.path.join(run_folder_path,'dataset') #  history_run/VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU_{timestamp}/dataset
  if not os.path.exists(dataset_folder_path):
    os.makedirs(os.path.join(run_folder_path,'dataset'))

  # Plot all the dataset distribution 
  _plot_dataset_distribution(csv_path=csv_path,
                                  per_class=per_class, 
                                  per_partecipant=per_partecipant,
                                  saving_path=dataset_folder_path,
                                  total_classes=total_classes) # 1  plot
  # TODO: Remove comments to plot the mean and std of the dataset
  # tools.plot_dataset_distribution_mean_std_duration(csv_path=csv_path,
  #                                                   video_path=path_video_dataset,
  #                                                   per_class=per_class, 
  #                                                   per_partecipant=per_partecipant, 
  #                                                   saving_path=dataset_folder_path) # 2 plots


def compute_loss_per_class(criterion,unique_train_val_classes,batch_y,class_loss,outputs):
  for cls in unique_train_val_classes:
    mask = (batch_y == cls).reshape(-1)
    if mask.any():
      class_idx = np.where(unique_train_val_classes == cls)[0][0]
      loss = criterion(outputs[mask], batch_y[mask]).detach().cpu().item()
      class_loss[class_idx] += loss
      
def compute_loss_per_subject(criterion,unique_train_val_subjects,batch_subjects,batch_y,subject_loss,outputs):
  for subj in unique_train_val_subjects:
    mask = (batch_subjects == subj).reshape(-1)
    # if epoch == 0:
    #   nr_samples_per_subject[subj] += mask.sum().item()
    if mask.any():
      subj_idx = np.where(unique_train_val_subjects == subj)[0][0]
      subject_loss[subj_idx] += criterion(outputs[mask], batch_y[mask]).detach().cpu().item()


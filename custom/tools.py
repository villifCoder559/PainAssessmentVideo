import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator, MultipleLocator
import cv2
import json
from openTSNE import TSNE as openTSNE
import time
from tqdm import tqdm
import pickle
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist
from custom.helper import CUSTOM_DATASET_TYPE,INSTANCE_MODEL_NAME
import av
import safetensors.torch
import torch
from torchmetrics.classification import  MulticlassConfusionMatrix
import torch.nn.functional as F
import cdw_cross_entropy_loss.cdw_cross_entropy_loss as cdw

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


def save_dict_data(dict_data, saving_folder_path,save_as_safetensors=False):
  """
  Save the dictionary containing numpy and torch elements to the specified path.

  Args:
    dict_data (dict): Dictionary containing the data to be saved.
    saving_path (str): Path to save the dictionary data.
  """
  if not os.path.exists(saving_folder_path):
    os.makedirs(saving_folder_path)
  print(f'Saving dictionary data to {saving_folder_path}...')
  if save_as_safetensors:
    # Save the dictionary as a .safetensors file
    dict_data = {k:v for k,v in dict_data.items() if isinstance(v,torch.Tensor)} 
    safetensors.torch.save_file(dict_data, saving_folder_path+'.safetensors')
    print(f'Safetensors data saved to {saving_folder_path}')
  else:
    for key, value in tqdm(dict_data.items(), desc="Saving files"):
      if isinstance(value, torch.Tensor):
        torch.save(value, os.path.join(saving_folder_path, f"{key}.pt"))
      elif isinstance(value, np.ndarray):
        np.save(os.path.join(saving_folder_path, f"{key}.npy"), value)
      else:
        print(f"Unsupported data type for key {key}: {type(value)}")
        # raise ValueError(f"Unsupported data type for key {key}: {type(value)}")
    print(f'Dictionary data saved to {saving_folder_path}')

def get_dataset_type(folder_path):
  if is_dict_data(folder_path=folder_path):
    return CUSTOM_DATASET_TYPE.AGGREGATED
  elif is_whole_data(folder_path=folder_path):
    return CUSTOM_DATASET_TYPE.WHOLE
  else:
    return CUSTOM_DATASET_TYPE.BASE

def is_whole_data(folder_path):
  list_subjects = os.listdir(folder_path)
  for subject in list_subjects:
    if os.path.isdir(os.path.join(folder_path, subject)):
      list_video_features = os.listdir(os.path.join(folder_path, subject))
      for folder_feature in list_video_features:
        folder_path = os.path.join(folder_path, subject, folder_feature)
        if not os.path.isdir(folder_path) and folder_feature.endswith('.mp4'):
          return False
        folder_files = os.listdir(folder_path)
        for video_feature in folder_files:
          if video_feature.endswith('.pt') or video_feature.endswith('.npy'):
            return True
          else:
            raise ValueError(f"Unsupported file format: {video_feature}")
          
def is_dict_data(folder_path):
  """
  Check if the specified folder contains dictionary data.

  Args:
    folder_path (str): Path to check for dictionary data.

  Returns:
    bool: True if the folder contains dictionary data, False otherwise.
  """
  if ".safetensors" in folder_path:
    return True
  list_target_files = ['features.pt','list_frames.pt','list_labels.pt','list_path.npy','list_subject_id.pt','list_sample_id.pt']
  list_files = os.listdir(folder_path)
  for file in list_files:
    if file in list_target_files:
      list_target_files.remove(file)
  return len(list_target_files) == 0

def get_instace_model_name(model):
  model_name = model.__class__.__name__
  for key in INSTANCE_MODEL_NAME:
    if model_name == key.value:
      return key
  raise ValueError(f'Instance model name not found for {model_name}')

def load_dict_data(saving_folder_path):
  """
  Load the dictionary containing numpy and torch elements from the specified path.

  Args:
    saving_path (str): Path to load the dictionary data.

  Returns:
    dict: Dictionary containing the loaded data.
  """
  if not ".safetensors" in saving_folder_path:
    dict_data = {}
    list_dir = os.listdir(saving_folder_path)
    for file in tqdm(list_dir,desc="Loading files"):
      if file.endswith(".pt"):
        dict_data[file[:-3]] = torch.load(os.path.join(saving_folder_path, file),weights_only=True)
      elif file.endswith(".npy"):
        dict_data[file[:-4]] = np.load(os.path.join(saving_folder_path, file))
      else:
        print(f"Unsupported file format: {file}")
  else:
    dict_data = safetensors.torch.load_file(saving_folder_path, device='cpu').copy()
    # copy data in RAM
    # if psutil.virtual_memory().available > os.path.getsize(saving_folder_path):
    #   for k,v in tqdm(dict_data.items(),desc="Copying data"):
    #     dict_data[k] = v
    # else:
    #   print(f"Not enough RAM available to copy data from {saving_folder_path}.")
  return dict_data

def plot_error_per_class(unique_classes, mae_per_class, criterion, title='', accuracy_per_class=None,y_label=None,
                         count_classes=None, saving_path=None, y_lim=None, ax=None):
  """
  If `ax` is provided, the plot is drawn on that axis; otherwise, a new figure and axis are created.
  If `accuracy_per_class` is provided, it also plots accuracy for the same class.
  """
  # Create a new figure and axis if none is provided
  
  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 5))
  
  # Set y-axis limit if provided
  if y_lim:
    ax.set_ylim(0, y_lim)
  
  # Define bar width
  bar_width = 0.4  
  indices = np.arange(len(unique_classes))  # X positions for bars

  # Plot the MAE (loss) per class
  ax.bar(indices - bar_width/2, mae_per_class, color='blue', width=bar_width,
         label='Loss per Class', edgecolor='black')
  ax.tick_params(axis='y', labelcolor='blue')
  
  # Plot accuracy per class if provided
  if accuracy_per_class is not None:
    ax2 = ax.twinx()  # Create a second y-axis for accuracy
    ax2.bar(indices + bar_width/2, accuracy_per_class, color='orange', width=bar_width,
           label='Accuracy per Class', edgecolor='black')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)  # Set y-axis limit for accuracy
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))  # Set y-ticks to be integers
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.set_yticks(np.arange(0, 1.1, 0.1))  # Set y-ticks for accuracy
    
  # Set axis labels and title
  ax.set_xlabel('Class')
  ax.set_ylabel(criterion)
  ax.set_title(f'{criterion} per Class {title}')
  
  # Set x-ticks with class names
  ax.set_xticks(indices)
  ax.set_xticklabels(unique_classes)

  # Add count text above each bar if count_classes is provided
  if count_classes is not None:
    for i, cls in enumerate(unique_classes):
      if cls in count_classes:
        ax.text(indices[i] - bar_width/2, mae_per_class[i], str(count_classes[cls]),
                ha='center', va='bottom', fontsize=10)
  
  # Add legend
  handles, labels = ax.get_legend_handles_labels()
  if accuracy_per_class is not None:
      handles2, labels2 = ax2.get_legend_handles_labels()
      handles += handles2
      labels += labels2
  ax.legend(handles, labels, loc='upper right')
  
  # Save or show the plot
  if saving_path is not None:
    plt.savefig(saving_path)
  
  plt.close()


def evaluate_classification_from_confusion_matrix(confusion_matrix,list_real_classes=None):
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
  all_predictions = torch.sum(confusion_matrix,None)
  accuracy = tp_sum / all_predictions 
  
  return {
    'precision_per_class': precision_per_class.detach().numpy(),
    'recall_per_class': recall_per_class.detach().numpy(),
    'macro_precision': macro_precision.detach().numpy(), 
    'macro_recall': macro_recall.detach().numpy(), 
    'micro_precision': micro_precision.detach().numpy(), # same as accuracy
    'micro_recall': micro_recall.detach().numpy(), 
    'weighted_precision': weighted_precision.detach().numpy(), 
    'weighted_recall': weighted_recall.detach().numpy(),
    'accuracy': accuracy.detach().numpy(),
  }

def plot_accuracy_confusion_matrix(confusion_matricies, type_conf,title='', saving_path=None, list_real_classes=None):
  if isinstance(confusion_matricies[0], MulticlassConfusionMatrix):
    confusion_matricies = torch.stack([confusion_matricies[i].compute() for i in range(len(confusion_matricies))])
  # if isinstance(test_confusion_matricies[0], MulticlassConfusionMatrix):
  #   test_confusion_matricies=torch.stack([test_confusion_matricies[i].compute() for i in range(len(test_confusion_matricies))])
  
  list_acc_confusion_matrix = []
  # list_test_acc_confusion_matrix = []
  for confusion_matrix in confusion_matricies:
    list_acc_confusion_matrix.append(evaluate_classification_from_confusion_matrix(confusion_matrix=confusion_matrix,
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
      
    plt.close()


import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy_per_subject(
  unique_subject_ids,
  criterion,
  accuracy_per_subject,
  title='',
  count_subjects=None,
  saving_path=None,
  bar_color='blue',
  y_label=None,
  list_stoic_subject=None,
  y_lim=(0, 1),
  ax=None
):
  """Plot Accuracy per participant.

  If `ax` is provided, the plot is drawn on that axes; otherwise, a new figure and axes are created.
  If `list_stoic_subject` is given, the x-axis labels corresponding to those subject IDs are colored differently.
  """
  if y_label is None:
    y_label = criterion if criterion.lower().startswith('acc') else f'Accuracy ({criterion})'

  # Create a new figure and axis if none is provided
  if ax is None:
    fig, ax = plt.subplots(figsize=(15, 5))

  # Apply y-axis limit (default 0–1 for accuracy)
  if y_lim is not None:
    ax.set_ylim(*y_lim)

  # Convert subject IDs to strings for the x-axis
  str_ids = [str(sid) for sid in unique_subject_ids]

  # Plot the bar chart
  ax.bar(str_ids, accuracy_per_subject, width=0.8,
    color=bar_color, edgecolor='black')

  # Labels and title
  ax.set_xlabel('Participant')
  ax.set_ylabel(y_label)
  ax.set_title(f'{y_label} per Participant — {title}')

  # Rotate and size the x-ticks
  ax.tick_params(axis='x', labelsize=11, rotation=45)

  # Highlight stoic subjects if provided
  if list_stoic_subject is not None:
    stoic_ids_str = {str(s) for s in list_stoic_subject}
    for tick in ax.get_xticklabels():
      tick.set_color('red' if tick.get_text() in stoic_ids_str else 'black')

  # Annotate with counts above bars
  if count_subjects is not None:
    for sid, count in count_subjects.items():
      # find index of this subject
      idxs = np.where(np.array(unique_subject_ids) == sid)[0]
      if idxs.size > 0:
        idx = idxs[0]
        ax.text(
          idx,
          accuracy_per_subject[idx] + 0.01,
          str(count),
          ha='center',
          va='bottom',
          fontsize=10
        )

  # Save or show
  if saving_path is not None:
    plt.savefig(saving_path)
  plt.close()



def plot_error_per_subject(unique_subject_ids, criterion, loss_per_subject,
  title='', count_subjects=None, saving_path=None,bar_color='green',y_label=None,step_y_axis=None,
  list_stoic_subject=None, y_lim=None, ax=None):
  """Plot Mean Absolute Error per participant.
  
  If `ax` is provided, the plot is drawn on that axes; otherwise, a new figure and axes are created.
  If `list_stoic_subject` is given, the x-axis labels corresponding to those subject IDs are colored differently.
  """
  if y_label is None:
    y_label = str(criterion)
    if y_label.lower() == 'cdw_celoss()':
      y_label = 'L1Loss'
  # Create a new figure and axis if none is provided
  if ax is None:
    fig, ax = plt.subplots(figsize=(15, 5))
  
  # Set y-axis limit if provided
  if y_lim:
    ax.set_ylim(0, y_lim)
    # set step to 0.1
  if step_y_axis:
    ax.yaxis.set_major_locator(MultipleLocator(step_y_axis))
    
  
  # Convert subject IDs to strings for the x-axis
  str_unique_subject_ids = [str(id) for id in unique_subject_ids]
  
  # Plot the bar chart (all bars are green)
  ax.bar(str_unique_subject_ids, loss_per_subject, width=0.8,
    color=bar_color, edgecolor='black')
  
  # Label the axes and add title
  ax.set_xlabel('Participant')
  ax.set_ylabel(y_label)
  ax.set_title(f'{y_label} per Participant -- {title}')
  
  # Rotate the x-axis tick labels
  ax.tick_params(axis='x', labelsize=11, rotation=45)
  
  # Change the color of x-axis tick labels if they are in list_stoic_subject
  if list_stoic_subject is not None:
    # Convert list_stoic_subject to string form to compare with tick labels
    stoic_ids_str = {str(x) for x in list_stoic_subject}
    for tick in ax.get_xticklabels():
      if tick.get_text() in stoic_ids_str:
        tick.set_color('red')
      else:
        tick.set_color('black')
  
  # Add count text above each bar if count_subjects is provided
  if count_subjects is not None:
    for id, count in count_subjects.items():
      idx = np.where(id == unique_subject_ids)[0]
      # Use the first index if there are multiple
      if idx.size > 0:
        ax.text(str(unique_subject_ids[idx[0]]),
          loss_per_subject[idx[0]], str(count),
          ha='center', va='bottom')
  
  # Save or show the plot
  if saving_path is not None:
    plt.savefig(saving_path)

  plt.close()


def plot_losses_and_test(train_losses, val_losses, saving_path=None):
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
                       loss_per_subject=dict_results['train_loss_per_subject'][best_model_idx], 
                       unique_subject_ids=uniqie_subject_ids_train,
                       count_subjects=count_subject_ids_train,
                       criterion=criterion,
                       saving_path=os.path.join(saving_path_losses,f'train_mae_per_subject_{best_model_idx}.png'))
  plot_error_per_subject(title='val',
                       loss_per_subject=dict_results['val_loss_per_subject'][best_model_idx], 
                       unique_subject_ids=uniqie_subject_ids_val,
                       count_subjects=count_subject_ids_test,
                       criterion=criterion,
                       saving_path=os.path.join(saving_path_losses,f'val_mae_per_subject_{best_model_idx}.png'))
  
  plot_losses_and_test(dict_results['train_losses'], dict_results['val_losses'], saving_path=os.path.join(saving_path_losses,'train_val_loss.png'))
      
def plot_confusion_matrix(confusion_matrix, title, ax=None, saving_path=None):
  # confusion_matrix must be from torchmetrics
  if not isinstance(confusion_matrix, MulticlassConfusionMatrix):
    raise ValueError('confusion_matrix must be an instance of torchmetrics.classification.MulticlassConfusionMatrix')
  fig, ax = confusion_matrix.plot(ax=ax)
  ax.set_title(title) 
  if saving_path is not None:
    fig.savefig(saving_path)


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

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_tsne(X_tsne,
  labels,
  cmap='copper',
  tot_labels=None,
  legend_label='',
  title='',
  cluster_measure='',
  saving_path=None,
  chunk_interval=None,
  axis_scale=None,
  last_point_bigger=False,
  plot_trajectory=False,
  stride_windows=None,
  clip_length=None,
  list_axis_name=None,
  ax=None,
  return_ax=False):
  """
  Plot a 2D t-SNE embedding, return either:
    - an RGB array of the rendered plot, or
    - the Axes object for further in-code modifications, or
    - save to disk and return the path.
  """

  unique_labels = np.unique(labels)
  n_colors = tot_labels if tot_labels is not None else len(unique_labels)
  color_map = plt.cm.get_cmap(cmap, n_colors)
  color_dict = {val: color_map(i) for i, val in enumerate(unique_labels)}

  # decide whether to create a new figure/ax or use the one passed in
  if ax is None:
    fig, ax = plt.subplots(figsize=(10, 8))
  else:
    fig = ax.figure

  # apply axis bounds if given
  if axis_scale is not None:
    ax.set_xlim(axis_scale['min_x'], axis_scale['max_x'])
    ax.set_ylim(axis_scale['min_y'], axis_scale['max_y'])

  # optionally enlarge just the last point
  sizes = None
  if last_point_bigger:
    sizes = np.full(X_tsne.shape[0], 50)
    sizes[-1] = 200

  # plot each label
  i=0
  for val in unique_labels:
    idx = (labels == val)
    # special “clip” labeling logic
    if clip_length is not None and legend_label == 'clip':
      idx_color = 0 if stride_windows * val < 48 else max(color_dict.keys())
      label = (
        f'{legend_label} [{stride_windows * val}, '
        f'{clip_length + stride_windows * val - 1}]'
        + (f' ({cluster_measure[idx_color]:.2f})'
           if cluster_measure and idx_color < len(cluster_measure) else '')
      )
      c = color_dict[idx_color]
    else:
      label = (
        f'{legend_label} {val}'
        + (f'{chunk_interval[i]}' if chunk_interval is not None else '')
        + (f' ({cluster_measure[val]:.2f})' if cluster_measure else '')
      )
      i+=1
      c = color_dict[val]

    ax.scatter(
      X_tsne[idx, 0],
      X_tsne[idx, 1],
      color=c,
      label=label,
      alpha=0.7,
      s=sizes[idx] if sizes is not None else 50
    )

  # optional trajectory line
  if plot_trajectory:
    ax.plot(
      X_tsne[:, 0],
      X_tsne[:, 1],
      linestyle='--',
      color=color_dict[0],
      label='Trajectory',
      alpha=0.7
    )

  # axis labels & legend/title
  if list_axis_name:
    ax.set_xlabel(list_axis_name[0])
    ax.set_ylabel(list_axis_name[1])
  else:
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')

  ax.legend()
  ax.set_title(f'{title} (Colored by {legend_label})')

  # --- OUTPUT BRANCHES ---

  # 1) SAVE TO DISK
  if saving_path is not None:
    # ensure .png extension
    if not saving_path.lower().endswith('.png'):
      os.makedirs(saving_path, exist_ok=True)
      saving_path = os.path.join(saving_path, f'{title}_{legend_label}.png')
    fig.savefig(saving_path)
    if not return_ax:
      plt.close(fig)
    return saving_path

  # 2) RETURN AXES FOR FURTHER IN-CODE USE
  if return_ax:
    # leave the figure open so the caller can continue to modify or close it
    return ax

  # 3) RENDER TO RGB ARRAY
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  img = buf.reshape(h, w, 3)
  plt.close(fig)
  return img


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
  list_entry = []
  for entry in csv_array:
    tmp = entry[0].split("\t")
    list_entry.append(tmp)
  return np.stack(list_entry),cols_array

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
  """
  Generates a video file from a list of frames.

  Args:
    list_frame (list): Input shape (B, H, W, C) representing a video sequence in RGB.
    path_video_output (str): The file path where the output video will be saved.
    fps (int, optional): Frames per second for the output video. Defaults to 25.

  Raises:
    OSError: If the output directory cannot be created.
  """

  if not os.path.exists(os.path.split(path_video_output)[0]):
    os.makedirs(os.path.split(path_video_output)[0])
  out = cv2.VideoWriter(path_video_output, cv2.VideoWriter_fourcc(*'avc1'), fps, (list_frame[0].shape[0], list_frame[0].shape[1]))
  for frame in list_frame:
    if not isinstance(frame,np.ndarray):
      frame = np.array(frame,dtype=np.uint8)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
  out.release()
  print(f'Video saved to {path_video_output}')

  
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
  
def plot_bar(data, title, x_label, y_label,color='red',ax_=None, saving_path=None,list_stoic_subject=None):
  if ax_ is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_
  keys = [str(i) for i in data.keys()]
  values = [int(i) for i in data.values()]
  ax.bar(keys, values, color=color, width=0.8, edgecolor='black',align='center')
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  # plt.xticks(rotation=45)
  if list_stoic_subject is not None:
    # Convert list_stoic_subject to string form to compare with tick labels
    stoic_ids_str = {str(x) for x in list_stoic_subject}
    for tick in ax.get_xticklabels():
      if tick.get_text() in stoic_ids_str:
        tick.set_color('red')
      else:
        tick.set_color('black')
  plt.tight_layout()
  if ax_ is not None:
    return None
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
                                count_subject_ids_test=dict_train['count_subject_ids_val'],
                                count_y_test=dict_train['count_y_val'], 
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
import torch

def compute_loss_per_class_(criterion,
                            unique_train_val_classes,
                            batch_y,
                            outputs,
                            class_loss=None,
                            class_accuracy=None):
  if batch_y.dim() != 1:
    batch_y = torch.argmax(batch_y,1)
  for cls in unique_train_val_classes:
    mask = (batch_y == cls).reshape(-1)
    if mask.any():
      class_idx = np.where(unique_train_val_classes == cls)[0][0]
      if class_accuracy is not None:
        predicted = torch.argmax(outputs[mask], 1 if outputs.dim() == 2 else 0)
        correct = (predicted == batch_y[mask]).sum().item() if batch_y.dim()== 1 else (predicted == batch_y[mask].argmax(1)).sum().item()
        total = mask.sum().item()
        class_accuracy[class_idx] += correct / total
      if class_loss is not None:
        loss = criterion(outputs[mask], batch_y[mask]).detach().cpu().item()
        class_loss[class_idx] += loss

import sys
def print_dict_size(d):
  # Measure table alone
  base = sys.getsizeof(d)  # ~240 bytes

  # Measure every key + value
  per_items = sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in d.items())

  # print("Approx bytes:", base + per_items)
  print("Approx MB:", (base + per_items) / 1024**2)

def log_predictions_per_sample_(dict_log_sample,tensor_predictions,tensor_sample_id,epoch):
  # Consider both tensors on cpu where:
  #   tensor_predictions: shape (N,)
  #   tensor_sample_id: shape (N,)
  #   dict_log_sample: dict with keys as sample_id and values a tensor 
  
  for id,prediction in zip(tensor_sample_id,tensor_predictions):
    dict_log_sample[id.item()][epoch] = prediction.item() # .type(torch.uint8)

def compute_loss_per_subject_v2_(
    criterion,
    unique_train_val_subjects: torch.Tensor,
    batch_subjects: torch.Tensor,
    batch_y: torch.Tensor,
    outputs: torch.Tensor,
    subject_loss: torch.Tensor = None,
    subject_accuracy: torch.Tensor = None,
):
    """
    Updates subject_loss and subject_accuracy in place by accumulating
    loss and accuracy for each subject in unique_train_val_subjects.

    Args:
      criterion: loss function. If its reduction != 'none', we'll fallback to F.cross_entropy(..., reduction='none').
      unique_train_val_subjects: 1D tensor of shape (S,) with all subject IDs.
      batch_subjects: 1D tensor of shape (N,) giving the subject ID for each sample.
      batch_y:      labels, shape (N,) or one-hot (N, C).
      outputs:      model logits, shape (N, C).
      subject_loss:     1D float tensor of shape (S,). Will be incremented by per-sample losses.
      subject_accuracy: 1D float tensor of shape (S,). Will be incremented by (correct/total) for each subject.
    """
    device = outputs.device
    S = unique_train_val_subjects.shape[0]

    # Build a mapping from subject ID to index [0..S-1]
    subj_to_idx = {int(s.item()): i for i, s in enumerate(unique_train_val_subjects)}
    # Map each sample's subject to its index
    idx = batch_subjects.detach().cpu().apply_(lambda s: subj_to_idx[int(s)]).to(device)  # shape (N,)

    # --- LOSS AGGREGATION ---
    if subject_loss is not None:
        # compute per-sample losses
        # if getattr(criterion, 'reduction', None) == 'none': # check if the criterion has a reduction attribute
        #     per_sample_losses = criterion(outputs, batch_y)          # (N,)
        if (isinstance(criterion, torch.nn.CrossEntropyLoss)):
            per_sample_losses = F.cross_entropy(outputs, batch_y, reduction='none')
        elif (isinstance(criterion, torch.nn.MSELoss)):
            per_sample_losses = F.mse_loss(outputs, batch_y, reduction='none')
        elif (isinstance(criterion, torch.nn.L1Loss)):
            per_sample_losses = F.l1_loss(outputs, batch_y, reduction='none')
        elif (isinstance(criterion, cdw.CDW_CELoss)): # if chage remeber to change also in new_plot_res_from_server
            per_sample_losses = F.l1_loss(torch.argmax(outputs,dim=1), batch_y, reduction='none')
        else:
          raise ValueError(f"Unsupported criterion in loss per subject computation: {criterion}")
        # sum losses by subject index
        loss_sum = torch.bincount(idx, weights=per_sample_losses, minlength=S)
        subject_loss += loss_sum.detach().cpu()

    # --- ACCURACY AGGREGATION ---
    if subject_accuracy is not None:
        # predictions
        _, preds = torch.max(outputs, dim=1 if outputs.dim() == 2 else 0)
        if batch_y.dim() == 1:
            correct = (preds == batch_y)
        else:
            correct = (preds == batch_y.argmax(dim=1))
        correct = correct.to(torch.float32)

        correct_sum   = torch.bincount(idx, weights=correct, minlength=S)
        total_counts  = torch.bincount(idx, minlength=S).to(correct.dtype).clamp(min=1)
        subject_accuracy += (correct_sum / total_counts).detach().cpu()

def compute_confidence_predictions_(list_prediction_right_mean,list_prediction_wrong_mean,list_prediction_right_std,list_prediction_wrong_std,outputs,gt,pred_before_softmax=True):
  if pred_before_softmax:
    outputs = torch.softmax(outputs, dim=1)
  if not isinstance(gt,torch.Tensor):
    gt = torch.tensor(gt)
  if gt.dim() > 1: 
    gt = torch.argmax(gt, dim=1) # if gt is not one hot encoded
     
  prediction_class = torch.argmax(outputs, dim=1)
  mask_right = (prediction_class == gt).reshape(-1)
  
  outputs_right,_ = torch.max(outputs[mask_right],dim=1)
  outputs_wrong,_ = torch.max(outputs[~mask_right],dim=1)
  if len(outputs_right) != 0:
    list_prediction_right_mean.append(torch.mean(outputs_right, dim=0).detach().cpu().numpy())
    list_prediction_right_std.append(torch.std(outputs_right, dim=0).detach().cpu().numpy() if len(outputs_right) > 1 else 0)
  if len(outputs_wrong) != 0:
    list_prediction_wrong_mean.append(torch.mean(outputs_wrong, dim=0).detach().cpu().numpy())
    list_prediction_wrong_std.append(torch.std(outputs_wrong, dim=0).detach().cpu().numpy() if len(outputs_wrong) > 1 else 0)


def generate_new_csv(csv_path,filter):
  df = pd.read_csv(csv_path, sep='\t')
  for key,value in filter.items():
    df = df[df[key].isin(value)]
    
def check_sample_id_y_from_csv(list_samples, list_y,csv_path):
  df = pd.read_csv(csv_path,sep='\t')
  for sample,y in zip(list_samples,list_y):
    csv_label = df[df['sample_id'] == sample]['class_id'].values[0]
    if y != csv_label:
      print(f'Error: {sample} y: {y} csv: {csv_label}')
      raise ValueError('Error: sample_id and y do not match with csv')
    
def get_lr_and_weight_decay(optimizer):
  """Retrieve the current learning rate and weight decay from the optimizer."""
  lrs = []
  wds = []

  for param_group in optimizer.param_groups:
    lrs.append(param_group['lr'])
    wds.append(param_group.get('weight_decay', 0))  # Defaults to 0 if not set

  return lrs, wds

def plot_losses_and_test_new(list_1,title, list_2=None,output_path=None,point=None,ax=None,x_label='Epochs',
                         y_label_1='Training Loss',y_label_2='Val loss',y_label_3='Test loss', 
                         y_lim_1=[0,5],y_lim_2=[0,1], y_lim_3 = [0,1],step_ylim_1=0.2,step_ylim_2=0.2,step_ylim_3=0.2,
                         dict_to_string=None,color_1='tab:red',color_2='tab:blue', color_3='tab:green'):
  # plt.figure()
  if ax is None:
    fig, ax1 = plt.subplots()
  else:
    ax1 = ax
  # Plot training loss (primary y-axis)
  ax1.set_xlabel(x_label)
  ax1.set_ylabel(y_label_1, color=color_1)
  ax1.plot(list_1, label=y_label_1, color=color_1)
  ax1.set_ylim(y_lim_1)  # Scale y-axis
  ax1.set_yticks(np.arange(y_lim_1[0], y_lim_1[1], step_ylim_1))
  ax1.tick_params(axis='y', labelcolor=color_1)
  # ax1.legend()
  
  if point is not None:
    ax3 = ax1.twinx()
    # ax3.spines["right"].set_position(("axes", 1.05))  # Offset the third axis to the right
    ax3.spines["right"].set_visible(False)
    ax3.set_ylabel(y_label_3, color=color_3, labelpad=15)
    ax3.plot(point['epoch'],point['value'],'o', label=y_label_3, color=color_3)
    ax3.set_ylim(y_lim_3)
    ax3.set_yticks(np.arange(y_lim_3[0], y_lim_3[1], step=step_ylim_3))
    ax3.tick_params(axis='y', labelcolor=color_3)
    # ax3.legend()
  
  # Create second y-axis for validation accuracy
  if list_2 is not None:
    ax2 = ax1.twinx()
    ax2.set_ylabel(y_label_2, color=color_2)
    ax2.plot(list_2, label=y_label_2, color=color_2)
    ax2.set_ylim(y_lim_2)  # Accuracy in percentage
    ax2.set_yticks(np.arange(y_lim_2[0], y_lim_2[1], step=step_ylim_2))
    ax2.tick_params(axis='y', labelcolor=color_2)
  # ax2.legend()
  lines, labels = ax1.get_legend_handles_labels()
  if list_2 is not None:
      lines2, labels2 = ax2.get_legend_handles_labels()
      lines += lines2
      labels += labels2
  if point is not None:
      lines3, labels3 = ax3.get_legend_handles_labels()
      lines += lines3
      labels += labels3
  ax1.legend(lines, labels, loc='best')
  # Add additional text on the 
  if dict_to_string:
    plt.figtext(1.1, 0.5, dict_to_string, ha='left', va='center', fontsize=12, color='black')

  # Titles, grid, and legend
  # plt.title(f'{title}')
  ax1.grid(True)
  ax1.title.set_text(title)
  if ax is None and output_path is not None:
  # Save plot and close
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
def plot_with_std(ax,x, mean, std,x_label,y_label,title,y_lim=None,legend_label_mean='Mean',legend_label_std='std',color='blue',y_step=None, cap_line=None):
  if ax is None:
    fig, ax = plt.subplots()
  if not isinstance(mean, np.ndarray):
    mean = np.array(mean)
  if not isinstance(std, np.ndarray):
    std = np.array(std)
  ax.plot(x, mean, label=legend_label_mean, color=color)
  if y_lim is not None:
    ax.set_ylim(y_lim)
    ax.grid(True)
    if y_step:
      ax.set_yticks(np.arange(y_lim[0], y_lim[1], step=y_step))
  if cap_line:
    ax.axhline(y=cap_line, color='r', linestyle='--', label='Cap line for gradient')
  ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.2, label=legend_label_std)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_title(title)
  ax.legend()
  if ax is None:
    plt.show()
    plt.close(fig)
    

def add_grid_to_video(video_path: str, grid_size: tuple,output_folder):
  """
  Adds a grid overlay to a video and saves the result as a new video file.
  
  Args:
    video_path (str): Path to the input video.
    grid_size (tuple): Grid size in format (num_rows, num_cols).
                       Determines how many grid cells the frame is divided into.
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error: Could not open video.")
    return

  frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps = cap.get(cv2.CAP_PROP_FPS)
  output_video_path = os.path.join(output_folder,os.path.split(video_path)[-1][:-4]+'_grid.mp4')
  out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'avc1'),
    fps,
    (frame_width, frame_height)
  )

  num_rows, num_cols = grid_size
  cell_width = frame_width / num_cols
  cell_height = frame_height / num_rows

  frame_idx = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    for i in range(1, num_cols):
      x = int(i * cell_width)
      cv2.line(frame, (x, 0), (x, frame_height), color=(0, 255, 0), thickness=1)

    for j in range(1, num_rows):
      y = int(j * cell_height)
      cv2.line(frame, (0, y), (frame_width, y), color=(0, 255, 0), thickness=1)

    # cv2.putText(frame, f"Frame: {frame_idx}", (10, frame_height - 10),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    out.write(frame)
    frame_idx += 1

  cap.release()
  out.release()
  print(f"The video with grid overlay has been saved at {output_video_path}")
  

def split_video_with_chunks(video_path: str, chunk_size: int,output_folder, separator_duration: int = None, new_fps: float = None):
  """
  Splits a video into chunks by inserting black frames labeled with the chunk number.

  Args:
    video_path (str): Path to the input video.
    chunk_size (int): Number of frames in each chunk before inserting a separator.
    separator_duration (int, optional): Number of black frames to insert as a separator. 
                                        Defaults to original FPS.
    new_fps (float, optional): FPS of the output video. Defaults to original FPS.
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    print("Error: Could not open video.")
    return

  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  original_fps = cap.get(cv2.CAP_PROP_FPS)
  fps = new_fps if new_fps else original_fps
  separator_frames = int(separator_duration if separator_duration else original_fps)
  output_video_path = os.path.join(output_folder,os.path.split(video_path)[-1][:-4]+'_chunks.mp4')
  out = cv2.VideoWriter(
    output_video_path,
    cv2.VideoWriter_fourcc(*'avc1'),
    fps,
    (width, height)
  )

  frame_count = 0
  chunk_index = 0

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    if frame_count % chunk_size == 0:
      for i in range(separator_frames):
        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
        text = f"CHUNK {chunk_index}"
        font_scale = 1
        thickness = 3
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        cv2.putText(black_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), thickness)
        out.write(black_frame)
      chunk_index += 1
    out.write(frame)
    frame_count += 1


  cap.release()
  out.release()
  print(f"Video saved at {output_video_path}")

def convert_dict_to_safetensors(dict_folder_path,output_folder_path=None):
  dict_data = load_dict_data(dict_folder_path)
  new_dict = {}
  for k,v in dict_data.items():
    if isinstance(v,torch.Tensor):
      new_dict[k] = v
    else:
      try:
        new_dict[k] = torch.tensor(v)
      except:
        print(f'Error converting {k} to tensor. Not saved in the new dict')
  if output_folder_path is None:
    safetensors_path = os.path.splitext(dict_folder_path)[0] + '.safetensors'
  else:
    safetensors_path = f'{output_folder_path}.safetensors'
  safetensors.torch.save_file(new_dict, safetensors_path,metadata={'format': 'torch'})
  print(f"Converted dict saved to {safetensors_path}")
  
def test_speed_safetensors_vs_standard(path_1, path_2):
  start_1 = time.time()
  a = load_dict_data(path_1)
  end_1 = time.time()
  print(f"Path1 load time: {end_1 - start_1} seconds")
  for k,v in a.items():
    print(f'  {k}: {v.shape}')
  del a
  # del dict_1
  start_2 = time.time()
  b = load_dict_data(path_2)
  end_2 = time.time()
  print(f"Path2 load time: {end_2 - start_2} seconds")
  for k,v in b.items():
    print(f'  {k}: {v.shape}')
  del b
  print(f'Speedup (path1/path2): {(end_1 - start_1) / (end_2 - start_2)}x')    


def convert_safetensors_dict_to_int32(path):
  dict_data = safetensors.torch.load_file(path)
  for k,v in dict_data.items():
    if k != 'features' and v.dtype != torch.int32:
      print(f'Converting {k} from {v.dtype} to int32')
      dict_data[k] = dict_data[k].to(torch.int32)
  safetensors.torch.save_file(dict_data, path)
  print(f"Converted dict saved to {path}")
  
def check_feats(path):
  dict_data = load_dict_data(path)
  for k,v in dict_data.items():
    print(f'{k}: {v.shape}')
    if k == 'list_sample_id':
      print(f'  Maximum sample id: {max(v)}')
      print(f'  Minimum sample id: {min(v)}')
      

def count_mispredictions(history_pred,df,return_miss_per_subject=None,top_k=None):
  # Count the number of mispredictions and if top_k is not None, return the top_k mispredictions
  misspredictions_per_label = {}
  misspredictions_per_sbj = {}
  
  for sample_id,pred_history in history_pred.items():
    # Get the ground truth class ID for the sample ID and count the number of mispredictions
    gt = df[df['sample_id'] == sample_id]['class_id'].values[0]
    count_miss = np.sum(np.array(pred_history) != gt)
    misspredictions_per_label[sample_id] = count_miss
    
    # Get the unique subject ID for the sample ID and get the number of mispredictions
    if return_miss_per_subject:
      sbj = df[df['sample_id'] == sample_id]['subject_id'].values[0]
      if sbj not in misspredictions_per_sbj:
        misspredictions_per_sbj[sbj] = 0
      misspredictions_per_sbj[sbj] += count_miss
    
  # Sort the dictionary by the number of mispredictions
  if top_k is not None:
    misspredictions_per_label = dict(sorted(misspredictions_per_label.items(), key=lambda item: item[1],reverse=True)[:top_k])
    if return_miss_per_subject:
      misspredictions_per_sbj = dict(sorted(misspredictions_per_sbj.items(), key=lambda item: item[1],reverse=True)[:top_k])
  
  if return_miss_per_subject:
    return misspredictions_per_label, misspredictions_per_sbj
  else:
    return misspredictions_per_label

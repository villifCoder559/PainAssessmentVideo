import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2
import json
import time
from tqdm import tqdm
import pickle
from custom.helper import CUSTOM_DATASET_TYPE,INSTANCE_MODEL_NAME, desired_order_csv
from custom.targets import TargetSpec
import safetensors.torch
import torch
from torchmetrics.classification import  MulticlassConfusionMatrix
import torch.nn.functional as F
import sys
from pathlib import Path
from scipy import stats
from custom.helper import unbc_count_to_id

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
    return str(obj)


def save_dict_data(dict_data, saving_folder_path,save_as_safetensors=True,save_as_npy=False):
  """
  Save the dictionary containing numpy and torch elements to the specified path.

  Args:
    dict_data (dict): Dictionary containing the data to be saved.
    saving_path (str): Path to save the dictionary data.
  """
  print(f'Saving dictionary data to {saving_folder_path}...')
    
  if save_as_safetensors:
    if not os.path.exists(os.path.dirname(saving_folder_path)):
      os.makedirs(os.path.dirname(saving_folder_path))
    # Save the dictionary as a .safetensors file 
    dict_data = {k:v for k,v in dict_data.items() if isinstance(v,torch.Tensor)} 
    safetensors.torch.save_file(dict_data, saving_folder_path+('.safetensors' if not saving_folder_path.endswith('.safetensors') else ''))
    print(f'Safetensors data saved to {saving_folder_path}')
  else:
    if not os.path.exists(saving_folder_path):
      os.makedirs(saving_folder_path)
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
        if not '.safetensors' in folder_path:
          folder_files = os.listdir(folder_path)
          for video_feature in folder_files:
            if video_feature.endswith('.pt') or video_feature.endswith('.npy') or 'safetensors' in video_feature:
              return True
            else:
              raise ValueError(f"Unsupported file format: {video_feature}")
        else:
          # if the folder contains a safetensors file, it is considered whole data
          return True
          
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
    assert os.path.exists(saving_folder_path), f"File missing: {saving_folder_path}"
    assert os.path.isfile(saving_folder_path), f"Not a file: {saving_folder_path}"
    dict_data = safetensors.torch.load_file(saving_folder_path)

  return dict_data

import numpy as np
import matplotlib.pyplot as plt

try:
  import torch
except Exception:
  torch = None


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

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def plot_distribution(data_dict, x_label,title,show_missing=False, color=None,ax=None,y_lim=None):
  """
  Plots a bar chart of label/subject distribution.

  Parameters:
  -----------
  data_dict : dict
      Dictionary mapping labels/subjects to counts, e.g. {0: 19, 1: 18, 2: 17}.
  title : str
      Title for the plot (e.g., 'Train Label Distribution').
  ax : matplotlib.axes.Axes, optional
      Axis object to draw the plot on. If None, a new figure and axis are created.
  """

  if ax is None:
    fig, ax = plt.subplots(figsize=(6, 4))

  # Extract keys and values
  if not show_missing:
    filtered_data = {k: v for k, v in data_dict.items() if v != 0}
    keys = list(filtered_data.keys())
    values = [filtered_data[k] for k in keys]
  else:
    # add dummy keys for lower (show_missing[0]) and upper (show_missing[1]) bounds if they don't exist
    for k in range(show_missing[0], show_missing[1]+1):
      if k not in data_dict:
        data_dict[k] = 0
    keys = list(data_dict.keys())
    values = list(data_dict.values())
  # Plot bars
  ax.bar([str(k) for k in keys] if not show_missing else keys, values, color=color if color is not None else 'steelblue', edgecolor='black', alpha=0.8)
  # Add labels and title
  ax.set_title(title, fontsize=12, fontweight='bold')
  ax.set_xlabel(x_label)
  ax.set_ylabel('Count')

  # Optional styling
  y_lim = (0,y_lim) if y_lim is not None else (0, max(values) * 1.1)
  ax.set_ylim(y_lim)
  ax.set_xticks([str(k) for k in keys] if not show_missing else keys)
  ax.grid(axis='y', linestyle='--', alpha=0.6)
  ax.set_axisbelow(True)
  ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

  return ax


def plot_confusion_matrix(confusion_matrix, title, ax=None, saving_path=None):
  # confusion_matrix must be from torchmetrics
  if not isinstance(confusion_matrix, MulticlassConfusionMatrix):
    raise ValueError('confusion_matrix must be an instance of torchmetrics.classification.MulticlassConfusionMatrix')
  fig, ax = confusion_matrix.plot(ax=ax)
  ax.set_title(title) 
  if saving_path is not None:
    fig.savefig(saving_path)


def get_unique_subjects_and_classes(df, target_spec=None):
  """
  Get the number of times each unique subject ID and class ID appears in video_labels.

  Returns:
    dict: A dictionary with keys 'subject_counts' and 'class_counts', each containing a dictionary
      where the keys are the unique IDs and the values are the counts.
  """
  if not isinstance(df, pd.DataFrame):
    df = pd.read_csv(df,sep='\t')  # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  
  list_samples = df.to_numpy()  # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  subject_ids = list_samples[:, 0].astype(int)
  spec = target_spec or TargetSpec.from_values(list_samples[:, 2])
  class_ids = spec.to_bins(list_samples[:, 2])
  
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
      class_ids = np.unique(target_bins)
      # class_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # class_id
      class_counts = {class_id: np.zeros(len(unique_subject_id)) for class_id in class_ids} # for each class, create a list of zeros for each participant
      for i, subject_id in enumerate(unique_subject_id):
        for class_id in class_ids:
          # Count the number of samples for each class and participant
          class_counts[class_id][i] = np.sum((list_samples[:, 0].astype(int) == subject_id) & (target_bins == class_id))
      # print(class_counts)
      plot_distribution_stacked(unique_subject_id, 'per participant and class', class_counts, total_classes)
    
    elif per_class:
      unique_subject_id,count = np.unique(target_bins,return_counts=True)
      plot_distribution(unique_subject_id,count,'per class')
    
    elif per_partecipant: 
      unique_subject_id,count = np.unique(list_samples[:,0],return_counts=True)
      unique_subject_id = np.sort(unique_subject_id.astype(int))
      plot_distribution(unique_subject_id,count,'per participant')

  # print(f' labels shape: {labels.shape}')

import os
import numpy as np
import matplotlib.pyplot as plt

def get_list_video_path_from_csv(csv_path, cols_csv_idx=[1,5], video_folder_path=None, extension='.mp4'):
  list_samples,_ = get_array_from_csv(csv_path) # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  if video_folder_path is None:
    video_folder_path = os.path.join('partA','video','video')
  list_video_path = []
  for sample in list_samples:
    # sample_video = union_segment.join(sample[cols_csv_idx])
    sample_video = os.path.join(sample[cols_csv_idx[0]], sample[cols_csv_idx[1]])
    video_path = os.path.join(video_folder_path, sample_video)
    video_path += extension
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
  print(f'Reading CSV file from: {csv_path}')
  csv_array = pd.read_csv(csv_path, sep='\t', dtype={'subject_name':str,'sample_name':str,'class_id':float,'sample_id':int,'subject_id':int,'class_name':str})  # subject_id, subject_name, class_id, class_name, sample_id, sample_name
  csv_array = csv_array[desired_order_csv]

  cols_array = csv_array.columns.to_numpy()
  csv_array = csv_array.to_numpy()
  list_entry = []
  for entry in csv_array:
    list_entry.append(entry)
  return np.stack(list_entry),cols_array

  # Add 4 black frames at the end

def generate_csv(cols, data, saving_path):
  df = pd.DataFrame(data, columns=cols)
  df.to_csv(saving_path, index=False, sep='\t')
  print(f'CSV saved to {saving_path}')

def generate_video_from_list_frame(list_frame,path_video_output,fps=25,resize=None,progress_bar=False,already_bgr=False):
  """
  Generates a video file from a list of frames.

  Args:
    list_frame (list): Input shape (B, H, W, C) representing a video sequence in RGB.
    path_video_output (str): The file path where the output video will be saved.
    fps (int, optional): Frames per second for the output video. Defaults to 25.

  Raises:
    OSError: If the output directory cannot be created.
  """

  if not os.path.exists(os.path.dirname(path_video_output)) and os.path.dirname(path_video_output) != '':
    os.makedirs(os.path.dirname(path_video_output))
  # force the video to be a .mp4 file
  ext = os.path.splitext(path_video_output)[-1]
  if len(ext) == 0 or ext.lower() != '.mp4':
    path_video_output = os.path.splitext(path_video_output)[0] + '.mp4'
    print(f'Warning: Output video path does not have .mp4 extension, changing to {path_video_output}')  
  heights = [frame.shape[0] for frame in list_frame]
  widths  = [frame.shape[1] for frame in list_frame]

  dict_frame_check = {
    'max_width': max(widths),
    'min_width': min(widths),
    'max_height': max(heights),
    'min_height': min(heights),
  }
  if dict_frame_check['max_width'] != dict_frame_check['min_width'] or dict_frame_check['max_height'] != dict_frame_check['min_height']:
    resize = (dict_frame_check['max_width'], dict_frame_check['max_height'])
    print(f'Warning: Frames have different sizes, resizing to (H={resize[1]}, W={resize[0]})')

  out_size = (list_frame[0].shape[1], list_frame[0].shape[0]) if resize is None else resize
  out = cv2.VideoWriter(path_video_output, cv2.VideoWriter_fourcc(*'avc1'), fps, out_size)
  count = 0
  for frame in tqdm(list_frame, desc="Processing frames", disable=not progress_bar):
    if not isinstance(frame,np.ndarray):
      frame = np.array(frame,dtype=np.uint8)
    if not already_bgr:
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if resize is not None:
      frame = cv2.resize(frame, resize)
    # print(f'Frame {count} size: {frame.shape}')
    out.write(frame)
    count += 1
  out.release()
  print(f'Video saved to {path_video_output}')

  
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
  
# Built-in metrics from scikit-learn
# TODO: select the right stride window for each video when read data from SSD
    
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
  
  for sample_id, prediction in zip(tensor_sample_id, tensor_predictions):
    sid = sample_id.item() if hasattr(sample_id, "item") else int(sample_id)
    pred = prediction.item() if hasattr(prediction, "item") else float(prediction)

    # Initialize nested dict if needed
    if sid not in dict_log_sample:
      dict_log_sample[sid] = {}

    # if epoch not in dict_log_sample[sid]:
      # First value, store directly
    dict_log_sample[sid][epoch] = pred
    # else:
    #   # Already have a value, switch to list if needed
    #   existing = dict_log_sample[sid][epoch]
    #   if isinstance(existing, list):
    #     existing.append(pred)
    #   else:
    #     dict_log_sample[sid][epoch] = [existing, pred]


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def _pick_effective_loss(criterion):
  """
  Returns (loss_module, mode) where mode ∈ {'ce','reg','argmax'}.
  Accepts torch.nn loss or CompositeLoss-like (with `list_losses`).
  """
  if hasattr(criterion, 'list_losses'):
    for lm in criterion.list_losses:
      if isinstance(lm, nn.CrossEntropyLoss):
        return lm, 'ce'
    for lm in criterion.list_losses:
      if isinstance(lm, (nn.MSELoss, nn.L1Loss, nn.HuberLoss, nn.SmoothL1Loss)):
        return lm, 'reg'
    return None, 'argmax'
  else:
    if isinstance(criterion, nn.CrossEntropyLoss):
      return criterion, 'ce'
    if isinstance(criterion, (nn.MSELoss, nn.L1Loss, nn.HuberLoss, nn.SmoothL1Loss)):
      return criterion, 'reg'
    return None, 'argmax'


def compute_loss_per_class_(
  criterion,
  unique_train_val_classes: torch.Tensor,
  batch_y: torch.Tensor,
  outputs: torch.Tensor,
  class_loss: Optional[torch.Tensor] = None,
  class_accuracy: Optional[torch.Tensor] = None,
  class_targets: Optional[torch.Tensor] = None,
  target_spec=None,
):
  """
  Aggregates loss and accuracy per class.
  Works with plain torch losses or CompositeLoss-like wrapper.
  """
  device = outputs.device
  unique_train_val_classes = unique_train_val_classes.to(device)

  if class_targets is not None:
    batch_y_labels = class_targets.to(device)
  elif batch_y.dim() != 1:
    batch_y_labels = torch.argmax(batch_y, dim=1).to(device)
  else:
    batch_y_labels = batch_y.to(device)

  subcrit, mode = _pick_effective_loss(criterion)

  for cls in unique_train_val_classes:
    mask = (batch_y_labels == cls).view(-1)
    if not mask.any():
      continue
    class_idx = (unique_train_val_classes == cls).nonzero(as_tuple=True)[0].item()

    # ----- Accuracy -----
    if class_accuracy is not None:
      if mode == 'reg':
        out_masked = outputs[mask]
        if out_masked.dim() > 1 and out_masked.size(1) == 1:
          out_masked = out_masked.view(-1)
        if target_spec is not None:
          predicted = target_spec.predictions_to_bins(out_masked)
        else:
          predicted = torch.copysign(torch.floor(torch.abs(out_masked) + 0.5), out_masked)
          predicted = predicted.clamp(0, unique_train_val_classes.max().item())
      else:
        if outputs.dim() == 2:
          predicted = torch.argmax(outputs[mask], dim=1)
        else:
          predicted = outputs[mask].view(-1).long()
      if batch_y.dim() == 1:
        correct = (predicted == batch_y_labels[mask]).sum().item()
      else:
        correct = (predicted == batch_y_labels[mask]).sum().item()
      total = int(mask.sum().item())
      class_accuracy[0, class_idx] += correct
      class_accuracy[1, class_idx] += total

    # ----- Loss -----
    if class_loss is not None:
      out_masked = outputs[mask].to(device)
      y_masked = batch_y_labels[mask].to(device)
      if mode == 'ce':
        loss_sum = F.cross_entropy(out_masked, y_masked, reduction='sum').detach().cpu().item()
      elif mode == 'reg':
        if isinstance(subcrit, nn.MSELoss):
          loss_sum = F.mse_loss(out_masked, batch_y[mask].to(device).float(), reduction='sum').detach().cpu().item()
        elif isinstance(subcrit, (nn.L1Loss, nn.SmoothL1Loss)):
          loss_sum = F.l1_loss(out_masked, batch_y[mask].to(device).float(), reduction='sum').detach().cpu().item()
        elif isinstance(subcrit, nn.HuberLoss):
          loss_sum = F.l1_loss(out_masked, batch_y[mask].to(device).float(), reduction='sum').detach().cpu().item()
        else:
          loss_sum = F.l1_loss(out_masked, batch_y[mask].to(device).float(), reduction='sum').detach().cpu().item()
      else:
        preds = torch.argmax(out_masked, dim=1) if out_masked.dim() == 2 else out_masked.view(-1).long()
        loss_sum = F.l1_loss(preds.to(device).float(), y_masked.to(device).float(), reduction='sum').detach().cpu().item()
      class_loss[0, class_idx] += loss_sum
      class_loss[1, class_idx] += int(mask.sum().item())


def compute_loss_per_subject_v2_(
  criterion,
  unique_train_val_subjects: torch.Tensor,
  batch_subjects: torch.Tensor,
  batch_y: torch.Tensor,
  outputs: torch.Tensor,
  subject_loss: Optional[torch.Tensor] = None,
  subject_accuracy: Optional[torch.Tensor] = None,
  unique_train_val_classes: Optional[torch.Tensor] = None,
  class_targets: Optional[torch.Tensor] = None,
  target_spec=None,
):
  """
  Updated to support CompositeLoss-like criterion.
  """
  device = outputs.device
  S = unique_train_val_subjects.shape[0]

  subj_to_idx = {int(s.item()): i for i, s in enumerate(unique_train_val_subjects)}
  batch_subjects_cpu = batch_subjects.detach().cpu().long()
  idx_list = [subj_to_idx[int(s.item())] for s in batch_subjects_cpu]
  idx = torch.tensor(idx_list, dtype=torch.long, device=device)

  if class_targets is not None:
    batch_y_labels = class_targets.to(device)
  elif batch_y.dim() != 1:
    batch_y_labels = torch.argmax(batch_y, dim=1).to(device)
  else:
    batch_y_labels = batch_y.to(device)

  subcrit, mode = _pick_effective_loss(criterion)

  # --- LOSS ---
  if subject_loss is not None:
    if mode == 'ce':
      per_sample_losses = F.cross_entropy(outputs.to(device), batch_y_labels.to(device), reduction='none')
    elif mode == 'reg':
      regression_targets = batch_y.to(device).float()
      if isinstance(subcrit, nn.MSELoss):
        per_sample_losses = F.mse_loss(outputs, regression_targets, reduction='none')
      elif isinstance(subcrit, (nn.L1Loss, nn.SmoothL1Loss)):
        per_sample_losses = F.l1_loss(outputs, regression_targets, reduction='none')
      elif isinstance(subcrit, nn.HuberLoss):
        per_sample_losses = F.l1_loss(outputs, regression_targets, reduction='none')
      else:
        per_sample_losses = F.l1_loss(outputs.float(), regression_targets, reduction='none')
    else:
      if outputs.dim() == 2:
        preds = torch.argmax(outputs, dim=1)
      else:
        preds = outputs.view(-1).long()
      per_sample_losses = (preds != batch_y_labels).to(torch.float32)

    loss_sum = torch.bincount(idx.cpu(), weights=per_sample_losses.detach().cpu(), minlength=S)
    subject_loss += loss_sum.detach().cpu()

  # --- ACCURACY ---
  if subject_accuracy is not None:
    if mode == 'reg':
      if target_spec is not None:
        preds = target_spec.predictions_to_bins(outputs)
      else:
        preds = torch.copysign(torch.floor(torch.abs(outputs) + 0.5), outputs)
        if preds.dim() > 1 and preds.size(1) == 1:
          preds = preds.view(-1)
        if unique_train_val_classes is not None:
          preds = preds.clamp(0, unique_train_val_classes.max().item())
    else:
      if outputs.dim() == 2:
        preds = torch.argmax(outputs, dim=1)
      else:
        preds = outputs.view(-1).long()
    if batch_y.dim() == 1:
      correct = (preds == batch_y_labels).to(torch.float32)
    else:
      correct = (preds == batch_y_labels).to(torch.float32)
    correct_sum = torch.bincount(idx.cpu(), weights=correct.detach().cpu(), minlength=S)
    subject_accuracy += correct_sum.detach().cpu()

def compute_confidence_predictions_(list_prediction_right_mean,list_prediction_wrong_mean,list_prediction_right_std,list_prediction_wrong_std,outputs,gt,pred_before_softmax=True):
  if pred_before_softmax and outputs.dim() == 2:
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
    list_prediction_right_mean.append(torch.mean(outputs_right, dim=0).detach().cpu().float().numpy())
    list_prediction_right_std.append(torch.std(outputs_right, dim=0).detach().cpu().float().numpy() if len(outputs_right) > 1 else 0)
  if len(outputs_wrong) != 0:
    list_prediction_wrong_mean.append(torch.mean(outputs_wrong, dim=0).detach().cpu().float().numpy())
    list_prediction_wrong_std.append(torch.std(outputs_wrong, dim=0).detach().cpu().float().numpy() if len(outputs_wrong) > 1 else 0)


def plot_losses_and_test_new(list_1,title, list_2=None,output_path=None,point=None,ax=None,x_label='Epochs',
                         y_label_1='Training Loss',y_label_2='Val loss',y_label_3='Test loss', 
                         y_lim_1=[0,2],y_lim_2=[0,1], y_lim_3 = [0,1],step_ylim_1=0.2,step_ylim_2=0.2,step_ylim_3=0.2,
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
    if isinstance(point,list):
      ax3.plot(point, label=y_label_3, color=color_3)
    elif isinstance(point,dict):
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
  ax1.title.set_fontweight('bold')
  if ax is None and output_path is not None:
  # Save plot and close
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
def plot_masked_attention(pkl_path, mask_threshold=7680, plot_all_subject=False,folder_plot_path=None):
  def get_one_video_per_subject(video_path_folder):
    folder_list = [f for f in os.listdir(video_path_folder) if os.path.isdir(os.path.join(video_path_folder, f))]
    lst = []
    for folder in tqdm(folder_list, desc="Getting one video per subject"):
      folder_path = os.path.join(video_path_folder, folder)
      for file in os.listdir(folder_path):
        if file.endswith(".mp4") and '$' not in file:
          lst.append(os.path.join(folder_path, file))
          break    
    return lst

                                                       
  list_video_per_subject = []
  # --- Load pickle ---
  if not Path(pkl_path).exists():
      raise FileNotFoundError(f"Pickle not found: {pkl_path}")
  with open(pkl_path, "rb") as f:
    pkl_file = pickle.load(f)

  # --- Compute boolean mask ---
  mask_small = np.mean(pkl_file['mean_value_matrix'], axis=0) < mask_threshold  # (14,14)

  folder_plot_path = Path(folder_plot_path) / f"plots_grid_{mask_small.shape[0]}x{mask_small.shape[1]}_thr_{mask_threshold}"
  os.makedirs(folder_plot_path, exist_ok=True)
  
  if plot_all_subject:
    list_video_per_subject = get_one_video_per_subject(Path(pkl_path).parent)
    print(f"Found {len(list_video_per_subject)} videos, one per subject.")
    print(f"Plots will be saved to: {folder_plot_path}")
  else:
    list_video_per_subject = [f"{Path(pkl_path).parent}/071309_w_21/071309_w_21-BL1-081.mp4"]
    
  # --- Load one video frame per subject ---
  for ref_video_path in tqdm(list_video_per_subject, desc="Processing videos"):
    if not Path(ref_video_path).exists():
      raise FileNotFoundError(f"Video not found: {ref_video_path}")
    
    cap = cv2.VideoCapture(ref_video_path)
    if not cap.isOpened():
      raise RuntimeError("Error opening video file")

    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
      raise RuntimeError("Couldn't read first frame from video")

    # Convert to RGBA
    frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    h, w = frame_rgba.shape[:2]

    # --- Compute mask ---
    mask_resized = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    # --- Overlay coloring ---
    overlay_alpha_value = 70
    overlay = np.zeros_like(frame_rgba, dtype=np.uint8)
    overlay[mask_resized] = [255, 0, 0, overlay_alpha_value]   # red semi-transparent
    overlay[~mask_resized] = [0, 255, 0, overlay_alpha_value]  # green semi-transparent

    alpha = overlay[..., 3:4].astype(np.float32) / 255.0
    blended_rgb = (alpha * overlay[..., :3] + (1 - alpha) * frame_rgba[..., :3]).astype(np.uint8)
    final_img = np.concatenate([blended_rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)

    # --- Draw grid borders (14×14) ---
    num_rows, num_cols = mask_small.shape
    step_y = h // num_rows
    step_x = w // num_cols

    grid_img = final_img.copy()
    line_color = (0, 0, 0, 128)  # black lines
    thickness = 1

    # Vertical lines
    for x in range(0, w, step_x):
      cv2.line(grid_img, (x, 0), (x, h), line_color, thickness)

    # Horizontal lines
    for y in range(0, h, step_y):
      cv2.line(grid_img, (0, y), (w, y), line_color, thickness)

    # --- Display ---
    plt.figure(figsize=(10, 6))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.title(f"Red = Masked, Green = Unmasked \nThreshold = {mask_threshold} Grid Size = {num_rows}x{num_cols}\n\{Path(ref_video_path).stem}\n{pkl_path}")
    png_path = folder_plot_path / f"{Path(ref_video_path).stem}_grid_{num_rows}x{num_cols}_thr{mask_threshold}.png"
    plt.savefig(png_path)
    plt.close()

def get_pth_path_from_project_folder(project_folder):
  pth_path_list = []
  for root, dirs, files in os.walk(project_folder):
    for file in files:
      if file.endswith('.pth'):
        pth_path_list.append(os.path.join(root, file))
  return pth_path_list

def concordance_ccc(y_true, y_pred):
  """Compute the Concordance Correlation Coefficient (CCC) between two 1-D arrays.

  CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

  Parameters
  ----------
  y_true, y_pred : array-like, shape (n,)

  Returns
  -------
  ccc : float
  """
  y_true = np.asarray(y_true).astype(np.float64)
  y_pred = np.asarray(y_pred).astype(np.float64)
  if y_true.shape != y_pred.shape:
    raise ValueError("y_true and y_pred must have the same shape")
  x_mean = y_true.mean()
  y_mean = y_pred.mean()
  x_var = y_true.var(ddof=0)
  y_var = y_pred.var(ddof=0)
  cov = np.mean((y_true - x_mean) * (y_pred - y_mean))
  numerator = 2.0 * cov
  denominator = x_var + y_var + (x_mean - y_mean) ** 2
  if denominator == 0:
    return 1.0 if np.allclose(y_true, y_pred) else np.nan
  return numerator / denominator


def pearson_r(y_true, y_pred):
  """Compute Pearson correlation coefficient between two 1-D arrays.

  Returns Pearson's r (float) and p-value (float).
  """
  y_true = np.asarray(y_true).astype(np.float64)
  y_pred = np.asarray(y_pred).astype(np.float64)
  if y_true.shape != y_pred.shape:
    raise ValueError("y_true and y_pred must have the same shape")
  if y_true.size < 2:
    return np.nan, np.nan
  rval, pval = stats.pearsonr(y_true, y_pred)
  return (rval, pval)


def intraclass_icc(data, icc_type='ICC2'):
  """Compute Intraclass Correlation Coefficient (ICC) for data shaped (n_subjects, n_raters).

  Supports ICC(2,1) (two-way random effects, single measures).
  """
  arr = np.asarray(data, dtype=np.float64)
  if arr.ndim != 2:
    raise ValueError('data must be 2-D: shape (n_subjects, n_raters)')
  n, k = arr.shape
  if n < 2:
    return np.nan
    # raise ValueError('Need at least 2 subjects to compute ICC')

  mean_per_subject = arr.mean(axis=1)
  mean_per_rater = arr.mean(axis=0)
  grand_mean = arr.mean()

  ss_total = np.sum((arr - grand_mean) ** 2)
  ss_between_subjects = k * np.sum((mean_per_subject - grand_mean) ** 2)
  ss_between_raters = n * np.sum((mean_per_rater - grand_mean) ** 2)
  ss_residual = ss_total - ss_between_subjects - ss_between_raters

  msr = ss_between_subjects / (n - 1)
  msc = ss_between_raters / (k - 1) if k > 1 else 0.0
  mse = ss_residual / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0.0

  if icc_type.upper() in ('ICC2', 'ICC(2,1)'):
    denom = msr + (k - 1) * mse + (k * (msc - mse) / n)
    if denom == 0:
      return np.nan
    return (msr - mse) / denom
  else:
    raise NotImplementedError(f"icc_type '{icc_type}' not implemented. Only 'ICC2' is available.")

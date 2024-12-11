import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import pandas as pd
from torchmetrics.classification import ConfusionMatrix,MulticlassConfusionMatrix
from sklearn.manifold import TSNE
from matplotlib.ticker import MaxNLocator
import cv2
import av
import torch
import json

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
    return super(NpEncoder, self).default(obj)

def save_dict_data(dict_data, saving_folder_path):
  """
  Save the dictionary containing numpy and torch elements to the specified path.

  Args:
    dict_data (dict): Dictionary containing the data to be saved.
    saving_path (str): Path to save the dictionary data.
  """
  if not os.path.exists(saving_folder_path):
    os.makedirs(saving_folder_path)
  
  for key, value in dict_data.items():
    if isinstance(value, torch.Tensor):
      torch.save(value, os.path.join(saving_folder_path, f"{key}.pt"))
    elif isinstance(value, np.ndarray):
      np.save(os.path.join(saving_folder_path, f"{key}.npy"), value)
    else:
      raise ValueError(f"Unsupported data type for key {key}: {type(value)}")
  print(f'Dictionary data saved to {saving_folder_path}')

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
  return dict_data

def plot_mae_per_class(unique_classes, mae_per_class, title='', count_classes=None, saving_path=None):
  """ Plot Mean Absolute Error per class. """
  print(f'MAE_PER_CLASS {title}: {mae_per_class}')
  plt.figure(figsize=(10, 5))
  plt.bar(unique_classes, mae_per_class, color='blue', width=0.4, label='MAE per Class')
  plt.xlabel('Class')
  plt.ylabel('Mean Absolute Error')
  plt.xticks(unique_classes)  # Show each element in x-axis
  plt.title(f'Mean Absolute Error per Class {title}')
  
  if count_classes is not None:
    for cls,count in count_classes.items():
      plt.text(unique_classes[cls], mae_per_class[cls], str(count), ha='center', va='bottom')
  plt.legend()
  if saving_path is not None:
    plt.savefig(saving_path)
  else:
    plt.show()

def get_accuracy_from_confusion_matrix(confusion_matrix):
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
  tp = confusion_matrix.diag()
  fn = torch.sum(confusion_matrix,1) - tp
  fp = torch.sum(confusion_matrix,0) - tp

  precision_per_class = torch.stack([tp[i] / (tp[i]+fp[i]) if tp[i]+fp[i]!=0 else torch.tensor(0) for i in range(len(tp))]).float()
  recall_per_class = torch.stack([tp[i] / (tp[i] + fn[i]) if tp[i]+fn[i]!=0 else torch.tensor(0) for i in range(len(tp))]).float() 
  
  # Treats all instances equally (larger classes have more weight)-> sensitive to imbalance
  micro_precision = torch.sum(tp) / (torch.sum(tp) + torch.sum(fp)) if torch.sum(tp + fp) != 0 else torch.tensor(0.0)
  micro_recall = torch.sum(tp) / (torch.sum(tp) + torch.sum(fn)) if torch.sum(tp + fn) != 0 else torch.tensor(0.0)
  
  # Weighted by the size of each class
  # print('factor 1',torch.sum(precision_per_class * torch.sum(confusion_matrix,1)))
  # print('factor 2',torch.sum(confusion_matrix))
  weighted_precision = torch.sum(precision_per_class * torch.sum(confusion_matrix,1)) / torch.sum(confusion_matrix)
  weighted_recall = torch.sum(recall_per_class * torch.sum(confusion_matrix,1)) / torch.sum(confusion_matrix)
  
  # Treats all classes equally
  macro_precision = torch.mean(precision_per_class)
  # print('np macro',np.mean(precision_per_class.numpy()))
  macro_recall = torch.mean(recall_per_class)
  
  return {
    'precision_per_class': precision_per_class,
    'recall_per_class': recall_per_class,
    'macro_precision': macro_precision, 
    'macro_recall': macro_recall, 
    'micro_precision': micro_precision, 
    'micro_recall': micro_recall, 
    'weighted_precision': weighted_precision, 
    'weighted_recall': weighted_recall,
  }

def plot_accuracy_confusion_matrix(train_confusion_matricies,test_confusion_matricies, title='', saving_path=None):
  if isinstance(train_confusion_matricies[0], MulticlassConfusionMatrix):
    train_confusion_matricies = torch.stack([train_confusion_matricies[i].compute() for i in range(len(train_confusion_matricies))])
  if isinstance(test_confusion_matricies[0], MulticlassConfusionMatrix):
    test_confusion_matricies=torch.stack([test_confusion_matricies[i].compute() for i in range(len(test_confusion_matricies))])
  
  list_train_acc_confusion_matrix = []
  list_test_acc_confusion_matrix = []
  for train_confusion_matrix in train_confusion_matricies:
    list_train_acc_confusion_matrix.append(get_accuracy_from_confusion_matrix(train_confusion_matrix))
  for test_confusion_matrix in test_confusion_matricies:
    list_test_acc_confusion_matrix.append(get_accuracy_from_confusion_matrix(test_confusion_matrix))
  keys = list_train_acc_confusion_matrix[0].keys()
  for key in keys:
    # print(f'key: {key}')
    train_list_key_values = [list_train_acc_confusion_matrix[i][key] for i in range(len(list_train_acc_confusion_matrix))]
    test_list_key_values = [list_test_acc_confusion_matrix[i][key] for i in range(len(list_test_acc_confusion_matrix))]
    labels_train = []
    labels_test = []
    if len(train_list_key_values[0].shape) == 0:
      labels_train.append(f'Training')
      labels_test.append(f'Test')
    else:
      for i in range(train_list_key_values[0].shape[0]):
        labels_train.append(f'Train class {i}')
        labels_test.append(f'Test class {i}')
    plt.figure(figsize=(10, 5))
    plt.plot(train_list_key_values, label=labels_train)
    plt.plot(test_list_key_values, label=labels_test)
    plt.xlabel('Epochs')
    plt.ylabel(key)
    plt.title(f'{key} over Epochs {title}')
    plt.legend()
    if saving_path is not None:
      path=os.path.join(saving_path,f'{key}.png')
      plt.savefig(path)
      print(f'Plot {key} over Epochs {title} saved to {path}.png')
    else:
      plt.show()
    # else:
      # for i in range(train_list_key_values[0].shape[0]):
      #   plt.figure(figsize=(10, 5))
      #   plt.plot([train_list_key_values[j][i] for j in range(len(train_list_key_values))], label='Training '+key+' class '+str(i))
      #   plt.plot([test_list_key_values[j][i] for j in range(len(test_list_key_values))], label='Test '+key+' class '+str(i))
      #   plt.xlabel('Epochs')
      #   plt.ylabel(key)
      #   plt.title(f'{key} over Epochs {title} class {i}')
      #   plt.legend()
      #   if saving_path is not None:
      #     plt.savefig(saving_path+f'_class_{i}.png')
      #     print(f'Plot {key} over Epochs {title} class {i} saved to {saving_path}_class_{i}.png')
      #   else:
      #     plt.show()

def plot_mae_per_subject(uniqie_subject_ids, mae_per_subject,title='', count_subjects=None, saving_path=None):
  """ Plot Mean Absolute Error per participant. """
  plt.figure(figsize=(10, 5))
  plt.bar(uniqie_subject_ids, mae_per_subject, color='green')
  plt.xlabel('Participant')
  plt.ylabel('Mean Absolute Error')
  plt.title(f'Mean Absolute Error per Participant {title}')
  plt.xticks(uniqie_subject_ids)  # Show each element in x-axis
  if count_subjects is not None:
    for id,count in count_subjects.items():
      idx = np.where(id == uniqie_subject_ids)[0]
      plt.text(uniqie_subject_ids[idx], mae_per_subject[idx], str(count), ha='center', va='bottom')
  if saving_path is not None:
    plt.savefig(saving_path)
    print(f'Plot MAE per subject saved to {saving_path}.png')
  else:
    plt.show()

def plot_losses(train_losses, test_losses, saving_path=None):
  plt.figure(figsize=(10, 5))
  plt.plot(train_losses, label='Training Loss')
  plt.plot(test_losses, label='Test Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Test Losses over Epochs')
  plt.legend()
  plt.grid(True)
  if saving_path is not None:
    plt.savefig(saving_path)
    print(f'Plot losses saved to {saving_path}.png')
  else:
    plt.show()

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

def read_split_indices(folder_path):
  split_indices_file = os.path.join(folder_path, 'split_indices.json')
  with open(split_indices_file, 'r') as f:
    split_indices = json.load(f)
  return split_indices

def generate_plot_train_test_results(dict_results, count_y_train, count_y_test, count_subject_ids_train, count_subject_ids_test, saving_path):  
  saving_path_losses = os.path.join(saving_path, 'losses')
  if not os.path.exists(saving_path_losses):
    os.makedirs(saving_path_losses)
  print(f'saving_path_losses: {saving_path_losses}')
  plot_mae_per_class(title='training', 
                     mae_per_class=dict_results['train_loss_per_class'][-1], 
                     unique_classes=dict_results['y_unique'], count_classes=count_y_train,
                     saving_path=os.path.join(saving_path_losses,'train_mae_per_class.png'))
  plot_mae_per_class(title='test',
                     mae_per_class=dict_results['test_loss_per_class'][-1], 
                     unique_classes=dict_results['y_unique'], count_classes=count_y_test,
                     saving_path=os.path.join(saving_path_losses,'test_mae_per_class.png'))
  
  plot_mae_per_subject(title='training', 
                       mae_per_subject=dict_results['train_loss_per_subject'][-1], 
                       uniqie_subject_ids=dict_results['subject_ids_unique'],
                       count_subjects=count_subject_ids_train,
                       saving_path=os.path.join(saving_path_losses,'train_mae_per_subject.png'))
  plot_mae_per_subject(title='test',
                       mae_per_subject=dict_results['test_loss_per_subject'][-1], 
                       uniqie_subject_ids=dict_results['subject_ids_unique'],
                       count_subjects=count_subject_ids_test,
                       saving_path=os.path.join(saving_path_losses,'test_mae_per_subject.png'))
  
  plot_losses(dict_results['train_losses'], dict_results['test_losses'], saving_path=os.path.join(saving_path_losses,'train_test_loss.png'))
      
def plot_confusion_matrix(confusion_matrix, title, saving_path):
  # confusion_matrix must be from torchmetrics
  # assert not isinstance(confusion_matrix, ConfusionMatrix), 'confusion_matrix must be from torchmetrics.classification'
  fig, _ = confusion_matrix.plot() 
  fig.suptitle(title)
  fig.savefig(saving_path+'.png')
  # matplotlib.pyplot.close()

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

def plot_dataset_distribution(csv_path, total_classes=None,per_class=False, per_partecipant=False, saving_path=None): 
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
      
    def plot_distribution_stacked(unique, title, class_counts,total_classes):
      # colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
      fig, ax = plt.subplots(figsize=(20, 10))
      cmap = plt.get_cmap('tab10')
      cmap = [cmap(i) for i in range(total_classes)]
      bottom = np.zeros(len(unique))
      for i, (class_id, class_count) in enumerate(class_counts.items()):
        unique_people = np.sum(class_count > 0)
        ax.bar(unique.astype(str), class_count, bottom=bottom, label=f'{class_id} ({unique_people}/{len(class_count)})',color=cmap[class_id])
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

def plot_tsne(X, labels, legend_label='', title = '', use_cuda=False, perplexity=1, saving_path=None):
  """
  Plots the t-SNE reduction of the features in 2D with colors based on subject, gt, or predicted class.
  Args:
    color_by (str, optional): Criterion for coloring the points ('subject', 'label', 'prediction'). Defaults to 'subject'.
    use_cuda (bool, optional): Whether to use CUDA for t-SNE computation. Defaults to False.
    perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 20.
  """
  print(f'TSNE_X.shape: {X.shape}')
  print(f'TSNE_labels.shape: {labels.shape}')
  if len(labels.shape) != 1:
    labels = labels.reshape(-1)
  unique_labels = np.unique(labels)
  color_map = plt.cm.get_cmap('tab20', len(unique_labels)) # FIX: if uniqelabels > 20 create maore a differnt plot
  color_dict = {val: color_map(i) for i, val in enumerate(unique_labels)}
  perplexity = min(30, X.size(0) - 1)
  if use_cuda and X.shape[0] > 194:
    print('Using CUDA')
    # tsne = cudaTSNE(n_components=2, perplexity=perplexity)
  else:
    print('Using CPU')
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
  X_cpu = X.detach().cpu().squeeze()
  X_cpu = X_cpu.reshape(X_cpu.shape[0], -1)
  # print(f' X_cpu.shape: {X_cpu.shape}')
  print(" Start t-SNE computation...")
  X_tsne = tsne.fit_transform(X_cpu) # in: X=(n_samples, n_features)
                                     # out: (n_samples, n_components=2)
  print(" t-SNE computation done.")
  # print(f' X_tsne.shape: {X_tsne.shape}')
  # print(f' labels shape: {labels.shape}')
  plt.figure(figsize=(10, 8))
  
  for val in unique_labels:
    idx = (labels == val).squeeze()
    plt.scatter(X_tsne[idx,0], X_tsne[idx,1], color=color_dict[val], label=f'{legend_label} {val}', alpha=0.7)
  
  if legend_label != 'subject':
    plt.legend()
  plt.title(f'{title} (Colored by {legend_label})')
  plt.xlabel('t-SNE Component 1')
  plt.ylabel('t-SNE Component 2')
  if saving_path is None:
    plt.show()
  else:
    plt.savefig(saving_path+'.png')

def get_list_video_path_from_csv(csv_path, cols_csv_idx=[1,5], union_segment='_'):
  list_samples,_ = get_array_from_csv(csv_path) # subject_id, subject_name, class_id, class_name, sample_id, sample_name
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
  Reads a CSV file, converts it to a NumPy array, and processes each entry by splitting
  the first column using a tab delimiter. The processed entries are then stacked into
  a single NumPy array.

  Args:
    csv_path (str): The file path to the CSV file.

  Returns:
    np.ndarray: A NumPy array containing the processed entries from the CSV file.\n
                (BIOVID cols-> subject_id, subject_name, class_id, class_name, sample_id, sample_name)
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
  print(f' list_input_video_path: {list_input_video_path}')
  print(f' list_frame_indices: {list_frame_indices.shape}')
  print(f' sample_ids: {sample_ids.shape}')
  print(f' all_predictions: {all_predictions.shape}')
  print(f' list_ground_truth: {list_ground_truth.shape}')
  
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

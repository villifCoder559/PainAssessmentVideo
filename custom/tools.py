import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os
import pandas as pd
from torchmetrics.classification import ConfusionMatrix
from sklearn.manifold import TSNE
import platform
if os.name == 'posix':
  from tsnecuda import TSNE as cudaTSNE # available only on Linux
else:
  print('tsnecuda availbale only on Linux')


def plot_mea_per_class(unique_classes, mae_per_class, title='', count_classes=None, saving_path=None):
  """ Plot Mean Absolute Error per class. """
  plt.figure(figsize=(10, 5))
  print(f'unique_classes shape {unique_classes.shape}')
  print(f'mae_per_class shape {mae_per_class.shape}')
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
    plt.savefig(saving_path+'.png')
  else:
    plt.show()

def plot_mea_per_subject(uniqie_subject_ids, mae_per_subject,title='', count_subjects=None, saving_path=None):
  """ Plot Mean Absolute Error per participant. """
  plt.figure(figsize=(10, 5))
  print(f'uniqie_subject_ids shape {uniqie_subject_ids.shape}')
  print(f'mae_per_subject shape {mae_per_subject.shape}')
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
    plt.savefig(saving_path+'.png')
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
    plt.savefig(saving_path+'.png')
  else:
    plt.show()

def _generate_train_test_validation(csv_path,random_state=42):
  def _check_class_distribution(split_dict):
    """ Check if each split has at least one sample per class. """
    for split_name, split_data in split_dict.items():
      classes_in_split = np.unique(split_data[:, 2])
      if len(classes_in_split) != len(np.unique(y)):
        print(f"Error: Not all classes are represented in the {split_name} split. Try another split...")
        return False
    print("All splits have at least one sample per class.")
    return True
  def _save_split(split_dict,video_labels_columns):
    # Sanity check
    if set(split_dict['train'][:,0].astype(int)).intersection(split_dict['val'][:,0].astype(int)) or \
        set(split_dict['train'][:,0].astype(int)).intersection(split_dict['test'][:,0].astype(int)) or \
        set(split_dict['val'][:,0].astype(int)).intersection(split_dict['test'][:,0].astype(int)):
      raise ('Error: train, validation and test split have common elements')

    # Save the splits
    save_path_dict = {}
    for element in split_dict:
      save_path = os.path.join('partA','starting_point',element+f'_{len(split_dict[element])}.csv')
      save_path_dict[element] = save_path
      split_df = pd.DataFrame(split_dict[element], columns=video_labels_columns)
      split_df.to_csv(save_path, index=False, sep='\t')
      print(f'{element} split saved to {save_path}')
    return save_path_dict
  def _generate_splits():
    video_labels = pd.read_csv(csv_path)
    csv_array=video_labels.to_numpy()
    video_labels_columns = video_labels.columns.to_numpy()[0].split('\t')
    # ['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name']
    list_samples=[]
    for entry in (csv_array):
      tmp = entry[0].split("\t")
      list_samples.append(tmp)
    list_samples = np.stack(list_samples)
    split_dict={}
    X = list_samples
    y = list_samples[:,2]
    groups = list_samples[:,0]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    split = list(gss.split(X, y, groups=groups)) # tmp to split in validation and test
    train_split_idx = split[0][0]
    split_dict['train'] = list_samples[train_split_idx]
    
    # Further split temp into validation and test
    tmp_split = split[0][1]
    X_temp = X[tmp_split]
    y_temp = y[tmp_split]
    groups_temp = groups[tmp_split]

    gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    test_val_split  = list(gss_temp.split(X_temp, y_temp, groups=groups_temp))
    test_split_idx = test_val_split[0][0]
    val_split_idx = test_val_split[0][1]
    split_dict['test'] = list_samples[tmp_split[test_split_idx]]
    split_dict['val'] = list_samples[tmp_split[val_split_idx]]
    return split_dict,video_labels_columns
  
  for _ in range(50): # attemps to generate a split
    split_dict,video_labels_columns = _generate_splits()
    if _check_class_distribution(split_dict):
      break
  
  save_path_dict = _save_split(split_dict,video_labels_columns)
  save_path_dict['all'] = csv_path
  return save_path_dict

def generate_plot_train_test_results(dict_results, count_y_train, count_y_test, count_subject_ids_train, count_subject_ids_test, saving_path):  
  plot_mea_per_class(title='training', mae_per_class=dict_results['train_loss_per_class'][-1], 
                            unique_classes=dict_results['y_unique'], count_classes=count_y_train,
                            saving_path=saving_path)
  plot_mea_per_class(title='test', mae_per_class=dict_results['test_loss_per_class'][-1], 
                            unique_classes=dict_results['y_unique'], count_classes=count_y_test,
                            saving_path=saving_path)
  
  plot_mea_per_subject(title='training', mae_per_subject=dict_results['train_loss_per_subject'][-1], 
                              uniqie_subject_ids=dict_results['subject_ids_unique'],count_subjects=count_subject_ids_train,
                              saving_path=saving_path)
  plot_mea_per_subject(title='test', mae_per_subject=dict_results['test_loss_per_subject'][-1], 
                              uniqie_subject_ids=dict_results['subject_ids_unique'],count_subjects=count_subject_ids_test,
                              saving_path=saving_path)
  
  plot_losses(dict_results['train_loss'], dict_results['test_loss'], saving_path=saving_path)
      
def plot_confusion_matrix(confusion_matrix, title, saving_path):
  # confusion_matrix must be from torchmetrics
  assert isinstance(confusion_matrix, ConfusionMatrix), 'confusion_matrix must be from torchmetrics.classification'
  fig, _ = confusion_matrix.plot() 
  fig.suptitle(title)
  fig.savefig(saving_path+'.png')

def plot_tsne(X, labels, legend_label, use_cuda=False, perplexity=20, saving_path=None):
  """
  Plots the t-SNE reduction of the features in 2D with colors based on subject, gt, or predicted class.
  Args:
    color_by (str, optional): Criterion for coloring the points ('subject', 'label', 'prediction'). Defaults to 'subject'.
    use_cuda (bool, optional): Whether to use CUDA for t-SNE computation. Defaults to False.
    perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 20.
  """
  # assert not (color_by == 'subject' and subjects_id is None), "If color_by is 'subject', subjects_id must be provided"
  print(X.shape)
  # if color_by == 'subject':
  #   colors = subjects_id
  #   color_label = 'Subject ID'
  # elif color_by == 'gt':
  #   colors = y
  #   color_label = 'Groundtruth Label'
  # elif color_by == 'prediction':
  #   predictions = self.head.predict(X)
  #   colors = predictions
  #   color_label = 'Predicted Class'
  # else:
  #   raise ValueError("color_by must be 'subject', 'gt', or 'prediction'")
  
  unique_labels = np.unique(labels)
  color_map = plt.cm.get_cmap('tab20', len(unique_labels))
  color_dict = {val: color_map(i) for i, val in enumerate(unique_labels)}
  
  if use_cuda and X.shape[0] > 194:
    tsne = cudaTSNE(n_components=2, perplexity=perplexity)
  else:
    tsne = TSNE(n_components=2, perplexity=perplexity)
  X_tsne = tsne.fit_transform(X)
  
  plt.figure(figsize=(10, 8))
  for val in unique_labels:
    idx = labels == val
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=color_dict[val], label=f'{legend_label} {val}', alpha=0.7)
  if legend_label != 'subject':
    plt.legend()
  plt.title(f't-SNE Reduction to 2D (Colored by {legend_label})')
  plt.xlabel('t-SNE Component 1')
  plt.ylabel('t-SNE Component 2')
  if saving_path is None:
    plt.show()
  else:
    plt.savefig(saving_path+'.png')
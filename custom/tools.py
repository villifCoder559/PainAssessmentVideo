import matplotlib.pyplot as plt
import numpy as np

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
    plt.savefig(saving_path)
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
    plt.savefig(saving_path)
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
  else:
    plt.show()
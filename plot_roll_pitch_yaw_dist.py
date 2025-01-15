import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_roll_pitch_yaw_dist(dict_subject,png_folder):
  fig,ax = plt.subplots(3,1,figsize=(10,10))
  # set bar width
  for i,(k,v) in enumerate(dict_subject.items()):
    ax[i].set_xticks(np.arange(min(v.keys()),max(v.keys())+1))
    ax[i].bar(v.keys(),v.values(),width=0.5)
    ax[i].set_title(k)
    ax[i].set_xlabel('Angle (degrees)')
    ax[i].set_ylabel('Frequency')
    
    
  fig.suptitle('Roll, pitch, and yaw distribution')
  plt.tight_layout()
  # png_folder = os.path.split(pkl_path)[0]
  plt.savefig(png_folder)
  print(f'Roll, pitch, and yaw distribution saved in {png_folder}')
  plt.close()
  
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

if __name__ == '__main__':
  csv_array,cols = get_array_from_csv(os.path.join('partA','starting_point','samples.csv'))
  # partA/video/roll_pitch_yaw_per_subject
  folder_path = os.path.join('partA','video','roll_pitch_yaw_per_subject') 
  list_subjects = np.unique(csv_array[:,1])
  for sbj_name in list_subjects:
    pkl_path = os.path.join(folder_path,sbj_name,f'dict_{sbj_name}.pkl')
    dict_subject = pickle.load(open(pkl_path,'rb'))
    png_folder = os.path.join(folder_path,sbj_name,'roll_pitch_yaw_dist.png')
    plot_roll_pitch_yaw_dist(dict_subject=dict_subject,
                             png_folder=png_folder) 
    
  all_pkl_path = os.path.join(folder_path,'dict_results_all_subjects.pkl')
  dict_all_subjects = pickle.load(open(all_pkl_path,'rb'))
  merge_dict = {}
  for sbj,dict_sbj in dict_all_subjects.items():
    for type_angle,dict_angle in dict_sbj.items():
      for angle,freq in dict_angle.items():
        merge_dict[type_angle] = merge_dict.get(type_angle,{})
        merge_dict[type_angle][angle] = merge_dict[type_angle].get(angle,0) + freq
  print(f'merge_dict: {merge_dict}')
  plot_roll_pitch_yaw_dist(dict_subject=merge_dict,
                           png_folder=os.path.join(folder_path,'roll_pitch_yaw_dist_all_subjects.png'))
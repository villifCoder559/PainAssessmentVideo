import custom.tools as tools
import os
import numpy as np
import time
import argparse
import torch
from pathlib import Path

def main(saving_folder_dict,saving_path_tsne):
  dict_data = tools.load_dict_data(saving_folder_path=saving_folder_dict)
  for k,v in dict_data.items():
    print(f'{k}: {v.shape}')
  X = dict_data['features']
  # list_subject_id = dict_data['list_subject_id']
  list_subject_id = dict_data['list_path']
  # list_subject_path = dict_data['list_path']
  list_subject_id = np.array([Path(path).parts[-2] for path in list_subject_id])
  max_subject_per_plot = 20
  unique_subject_id = np.unique(list_subject_id)
  print(f'unique_subject_id: {unique_subject_id}')
  for i in range(0,unique_subject_id.shape[0]//max_subject_per_plot+1):
    list_subject_id_tsne = []
    X_tsne = []
    for id in unique_subject_id[i*max_subject_per_plot:(i+1)*max_subject_per_plot]:
      mask = list_subject_id == id
      list_subject_id_tsne.append(list_subject_id[mask])
      X_tsne.append(X[mask])
    X_tsne = torch.concat(X_tsne,axis=0)
    list_subject_id_tsne = np.concatenate(list_subject_id_tsne,axis=0)
    # print(f'\nlist_subject_id_tsne: {list_subject_id_tsne}')
    print(f'list_subject_id_tsne shape: {list_subject_id_tsne.shape}')
    print(f'X_tsne: {X_tsne.shape}')
    tools.compute_tsne(X=X_tsne,
                    labels=list_subject_id_tsne,
                    cmap='tab20',
                    apply_pca_before_tsne=True,
                    saving_path=os.path.join(saving_path_tsne,f'tsne_plot_{i}.png'),
                    legend_label=' ',
                    title=f'from data {os.path.split(saving_folder_dict)[1]}',)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Compute tsne plot')
  parser.add_argument('--sfd', type=str, help='Path to the folder containing the data extracted')
  parser.add_argument('--sptsne', type=str, help='Path to save the tsne plot')
  
  args = parser.parse_args()
  main(saving_folder_dict=args.sfd,
       saving_path_tsne=args.sptsne)
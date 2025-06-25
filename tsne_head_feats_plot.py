import custom.tools as tools
import custom.tsne_cuda_tools as tsne_cuda_tools
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle


feat_path = "history_run_samples_0_4_selftrained_ce_1_1_1_stride4_L1_366069_ATTENTIVE_JEPA_villi-Inspiron-16-Plus-7620_1750673405/1750673406842_VIDEOMAE_v2_G_NONE_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/train_ATTENTIVE_JEPA/k0_cross_val/k0_cross_val_sub_0/head_features_train.safetensors"

dict_data = tools.load_dict_data(saving_folder_path=feat_path)
tsne_feats, config, _, _ = tsne_cuda_tools.compute_tsne(X=dict_data['features'])

group_key = 'subjects' # 'labels' or 'subjects'
group_by = dict_data[group_key]
unique_groups = np.unique(group_by)
plot_path = os.path.join(os.path.dirname(feat_path), f'{os.path.basename(feat_path)}_{group_key}.png')
if group_key == 'labels':
  colors = plt.get_cmap('tab10', len(unique_groups))
  plt.figure(figsize=(10, 10))
  for i, group in enumerate(unique_groups):
    indices = np.where(group_by == group)[0]
    plt.scatter(tsne_feats[indices, 0], tsne_feats[indices, 1], label=f"Pain intensity {group}", color=colors(i), alpha=0.5)
  plt.title('t-SNE of Head Features')
  plt.xlabel('t-SNE Component 1')
  plt.ylabel('t-SNE Component 2')
  plt.legend()
  plt.grid()
  plt.savefig(plot_path)
  plt.close()

  plot_name = os.path.basename(plot_path).replace('.png','').replace('.safetensors','')
  # Save readable plot path and config
  with open(os.path.join(f'{os.path.dirname(feat_path)}',f'{plot_name}_config.txt'), 'w') as f:
    f.write(f"t-SNE plot saved at: {plot_path}\n")
    f.write(f"Config: {config}\n")
    f.write(f"Unique groups: {unique_groups}\n")

  # save config obj 
  with open(os.path.join(f'{os.path.dirname(feat_path)}',f'{plot_name}_config.pkl'), 'wb') as f:
    pickle.dump(config, f)

elif group_key == 'subjects':
  unique_groups = np.unique(group_by)
  batch_groups = np.array_split(unique_groups, len(unique_groups) // 20 + 1) # batch size of 20
  colors = plt.get_cmap('tab20', len(unique_groups))
  
  # Calculate proper grid dimensions
  n_batches = len(batch_groups)
  n_cols = min(4, n_batches)  # Max 4 columns
  n_rows = (n_batches + n_cols - 1) // n_cols  # Ceiling division
  fig_size = (max(n_cols * 5, 30), max(n_rows * 4, 30))  # Each subplot is 5x5 inches
  
  fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))
  
  # Ensure axes is always 2D array
  if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
  elif n_rows == 1:
    axes = axes.reshape(1, -1)
  elif n_cols == 1:
    axes = axes.reshape(-1, 1)
  
  # Plot each batch
  for batch_idx, batch in enumerate(batch_groups):
    row = batch_idx // n_cols
    col = batch_idx % n_cols
    
    for group in batch:
      indices = np.where(group_by == group)[0]
      axes[row, col].scatter(tsne_feats[indices, 0], tsne_feats[indices, 1], 
                            label=f"Subject {group}", 
                            color=colors(batch_idx), alpha=0.5)
    
    axes[row, col].set_title(f'Subjects Batch {batch_idx + 1}')
    axes[row, col].set_xlabel('t-SNE Component 1')
    axes[row, col].set_ylabel('t-SNE Component 2')
    axes[row, col].legend()
    axes[row, col].grid()
  
  # Hide unused subplots
  for batch_idx in range(n_batches, n_rows * n_cols):
    row = batch_idx // n_cols
    col = batch_idx % n_cols
    axes[row, col].set_visible(False)
  
  plt.tight_layout()
  plt.savefig(plot_path)
  plt.close(fig)
    
    
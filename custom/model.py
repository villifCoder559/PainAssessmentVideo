from matplotlib.ticker import MaxNLocator
import torch
from custom.backbone import backbone
from custom.neck import neck
from custom.dataset import customDataset
from sklearn.svm import SVR
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error as mea
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from custom.head import SVR_head
from joblib import dump
import os
from sklearn.manifold import TSNE

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,
              path_labels, preprocess, sample_frame_strategy, head, download_if_unavailable=False,
              batch_size=1,stride_window=2,clip_length=16
              ):
    """
    Initialize the custom model. 
    Parameters:
    model_type (str): Type of the model to be used. 
    embedding_reduction (int): Dimension reduction for embeddings.
    clips_reduction (int): Dimension reduction for clips.
    path_dataset (str): Path to the dataset.
    path_labels (str): Path to the labels.
    preprocess (callable): Preprocessing function for the data.
    sample_frame_strategy (str): Strategy for sampling frames.
    download_if_unavailable (bool, optional): Flag to download the model if unavailable. Defaults to False.
    batch_size (int, optional): Batch size for data loading. Defaults to 1.
    stride_window (int, optional): Stride window for sampling frames. Defaults to 2.
    clip_length (int, optional): Length of each video clip. Defaults to 16.
    svr_params (dict, optional): Parameters for the Support Vector Regressor (SVR). Defaults to {'kernel': 'rbf', 'C': 1, 'epsilon': 0.1}.

    Raises:
    AssertionError: If batch_size is not 1.
    """
    self.backbone = backbone(model_type, download_if_unavailable)
    self.neck = neck(embedding_reduction, clips_reduction)
    self.dataset = customDataset(path_dataset, path_labels, preprocess, sample_frame_strategy, stride_window=stride_window, clip_length=clip_length)
    self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.dataset._custom_collate_fn) # TODO: put inside customDataset and return a dataset and dataLoader
    self.head = head
    
  # def fit(self, stop_after=5,plot_dataset_split_distribution=False):
  #   """ Evaluation training of SVR model. """
  #   X, y, subjects_id, _, paths, list_frames = self._extract_features_from_dataset(stop_after=stop_after) # feats,labels,subject_id,sample_id,path
  #   X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
  #   y = y.squeeze().detach().cpu().numpy()
  #   print(f'list_frames_FIT.shape {list_frames.shape}') # [video, nr_frame, indices] for each video I have many chunks sampled using following the indices in the list   
  #   if plot_dataset_split_distribution:
  #     merged_array = np.concatenate((y[:,None],subjects_id[:,None]),axis=1)
  #     print(f'merged_array.shape: {merged_array.shape}')

  #   X_train, X_test, y_train, y_test =  train_test_split(X, merged_array, test_size=0.2, random_state=42)
  #   print(f'y_train shape: {y_train.shape}')
  #   print(f'y_test shape: {y_test.shape}')
  #   y_train, subjects_id_train = y_train[:,0], y_train[:,1]
  #   y_test, subjects_id_test = y_test[:,0], y_test[:,1]
  #   # print(f"subjects_id shape: {subjects_id.shape}, y_train shape: {y_train.shape}")
  #   if plot_dataset_split_distribution:
  #     self.plot_dataset_split_distribution(subjects_id_train, y_train, title='Train Dist.')
  #     self.plot_dataset_split_distribution(subjects_id_test, y_test, title='Test Dist.')
  #   # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
  #   # print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
  #   # # return None
  #   regressor = self.svr.fit(X_train, y_train)
  #   y_pred = regressor.predict(X_test)
  #   # Model evaluation
  #   print("Mean absolute Error:", mea(y_test, y_pred))
  #   frames_path_dict = [] 
  #   for path,list_f in zip(paths,list_frames):
  #     frames_path_dict.append({'path':path,'frame_list':list_f})
  #   return frames_path_dict
  #   # print("R2 Score:", r2_score(y_test, y_pred))
      
  # def k_fold_cross_validation(self, k=3, stop_after=5,plot_dataset_split_distribution=False):
  #   """ k-fold cross-validation training of SVR model. """
  #   # Use dictionary so you cann add w/o changing code
  #   X, y, subjects_id,_ , _ = self._extract_features_from_dataset(stop_after=stop_after) # feats,labels,subject_id,sample_id,path
  #   X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
  #   y = y.squeeze().detach().cpu().numpy()
  #   if plot_dataset_split_distribution:
  #     self.plot_dataset_split_distribution(subjects_id, y)
    
  #   print('X.shape', X.shape)
  #   print('y.shapey', y.shape)
  #   kf = KFold(n_splits = k, shuffle = True,random_state = 42)
  #   scores = cross_val_score(self.svr, X, y, cv=kf)
    
  #   # Print the scores for each fold and the mean score
  #   print(f"Cross-validation scores (MSE): {scores}")
  #   print(f"Mean cross-validation score (MSE): {np.mean(scores)}")
  #   print(f'Std cross-validation score (MSE): {np.std(scores)}')  
  def train(self, stop_after=5,k_cross_validation=0):
    if isinstance(self.head, SVR_head):
      if k_cross_validation:
        print('Training using SVR with k-fold cross-validation...')
        X, y, subjects_id,_ , _, _ = self._extract_features_from_dataset(stop_after=stop_after) # feats,labels,subject_id,sample_id,path
        X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
        y = y.squeeze().detach().cpu().numpy()
        
        list_split_indices,results = self.head.k_fold_cross_validation(k=k_cross_validation,X=X, y=y, groups=subjects_id)
        return list_split_indices, results
      else:
        print('Training using SVR...')
        self.dataset.set_path_labels('val') # TODO: change to train
        X_train, y_train, subjects_id_train, _, paths_train, list_frames_train = self._extract_features_from_dataset(stop_after=stop_after)
        X_train = X_train.reshape(X_train.shape[0],-1).detach().cpu().numpy()
        y_train = y_train.squeeze().detach().cpu().numpy()
        
        # self.dataset.set_path_labels('test')
        # X_test, y_test, subjects_id_test, _, paths_test, list_frames_test = self._extract_features_from_dataset(stop_after=stop_after)
        # X_test = X.reshape(X.shape[0],-1).detach().cpu().numpy()
        # y_test = y.squeeze().detach().cpu().numpy()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        # print(subjects_id)
        # print(X_train.shape)
        self.head.fit(X_train=X_train, y_train=y_train, subject_ids=subjects_id_train)
        # dump(self.head.svr, os.path.join('PartA','starting_point',f'svr_model_{self.dataset.path_labels}.joblib'))
    # elif isinstance(self.head, GRU_head):      
  # def fit(self):
      
  def _extract_features_from_dataset(self,stop_after=3):
    """
    Extracts features from the dataset using the model's backbone.
    Args:
      stop_after (int, optional): Number of iterations after which to stop the feature extraction. Defaults to 3.
    Returns:
      tuple: A tuple containing the following elements:
        - torch.Tensor: Stacked features extracted from the dataset.
        - torch.Tensor: Stacked labels corresponding to the features.
        - numpy.ndarray: Array of subject IDs.
        - numpy.ndarray: Array of sample IDs.
        - numpy.ndarray: Array of paths corresponding to the samples.
        - torch.Tensor: Stacked frames sampled from the dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"extracting features using... {device}")
    list_features = []
    list_labels = []
    list_subject_id = []
    list_sample_id = []
    list_path = []
    list_frames = []
    count = 0
    
    self.backbone.model.to(device)
    self.backbone.model.eval()
    with torch.no_grad():
      for data, labels, subject_id,sample_id, path, list_sampled_frames in self.dataloader:
        feature, unique_labels, unique_subject_id, unique_sample_id, unique_path = self._compute_features(data, labels, subject_id, sample_id, path, device)
        list_frames.append(list_sampled_frames)
        list_features.append(feature)
        list_labels.append(unique_labels)
        list_sample_id.append(unique_sample_id)
        list_subject_id.append(unique_subject_id)
        list_path.append(unique_path)
        count += 1
        # if count % stop_after == 0:
        #   break
    print('Feature extracetion done')
    return torch.stack([feature for feature in list_features]),\
           torch.stack([label for label in list_labels]),\
           np.stack([subject_id for subject_id in list_subject_id]).squeeze(),\
           np.stack([sample_id for sample_id in list_sample_id]),\
           np.stack([path for path in list_path]),\
           torch.stack(list_frames)  
  
  # def plot_dataset_split_distribution(self, subjects_id, labels, title=''):
  #   """
  #   Plots histograms to show the distribution of labels and subjects_id.

  #   Parameters:
  #   subjects_id (tensor): Tensor containing subject IDs.
  #   labels (tensor): Tensor containing labels.
  #   title (str): Title for the entire plot.
  #   """
  #   plt.figure(figsize=(12, 6))

  #   # Plot distribution of labels
  #   plt.subplot(1, 2, 1)
  #   bins = np.arange(min(labels), max(labels) + 2) - 0.5
  #   plt.hist(labels, bins=bins, edgecolor='k', alpha=0.7)
  #   plt.title('Distribution of Labels')
  #   plt.xlabel('Labels')
  #   plt.ylabel('Frequency')
  #   plt.xticks(np.arange(min(labels), max(labels) + 1))
    
  #   # Plot distribution of subjects_id
  #   plt.subplot(1, 2, 2)
  #   bins = np.arange(min(subjects_id), max(subjects_id) + 2) - 0.5
  #   plt.hist(subjects_id, bins=bins, edgecolor='k', alpha=0.7)
  #   plt.title('Distribution of Subject IDs')
  #   plt.xlabel('Subject IDs')
  #   plt.ylabel('Frequency')
  #   plt.xticks(np.arange(min(subjects_id), max(subjects_id) + 1))

  #   plt.suptitle(title)
  #   plt.tight_layout()
  #   plt.show()
  
  def _compute_features(self, data, labels, subject_id, sample_id, path, device, remove_clip_reduction=False):
  
    # Extract features from clips -> return [B, clips/tubelets, W/patch_w, H/patch_h, emb_dim] 
    feature = self.backbone.forward_features(data.to(device)) # ex. [B,8*14*14,emb_dim]
    unique_labels, unique_subject_id, unique_sample_id, unique_path = [], [], [], []  
    # Apply dimensionality reduction [B,C,T,H,W] -> [B, reduction(C,T,H,W)]
    if self.neck.embedding_reduction is not None:
      feature = self.neck.embedding_reduction(feature)
    # Apply clip reduction [B, reduction(C,T,H,W)] -> [1, reduction(C,T,H,W)]
    if not remove_clip_reduction and self.neck.clips_reduction is not None:
      feature = self.neck.clips_reduction(feature)
      unique_labels = torch.unique(labels, return_counts=False)
      unique_sample_id = np.unique(sample_id, return_counts=False)
      unique_subject_id = np.unique(subject_id, return_counts=False)
      unique_path = np.unique(path, return_counts=False)
    return feature, unique_labels, unique_subject_id, unique_sample_id, unique_path
    
  def run_grid_search(self, param_grid,k_cross_validation=5): #
    if isinstance(self.head, SVR_head):
      print('GridSearch using SVR.-..')
      self.dataset.set_path_labels('val')
      X, y, subjects_id,_ , _, _ = self._extract_features_from_dataset()
      X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
      y = y.squeeze().detach().cpu().numpy()
      # print(subjects_id.shape)
      grid_search, list_split_indices =self.head.run_grid_search(param_grid=param_grid, X=X, y=y, groups=subjects_id, k_cross_validation=k_cross_validation)
      
      return grid_search, list_split_indices, subjects_id, y
    else:
      return None
  
  def plot_prediction_graph(self, sample_id, stride_window=0):
    with torch.no_grad():
      csv_array = self.dataset.video_labels.to_numpy()
      print(csv_array)
      list_samples=[]
      for entry in (csv_array):
        tmp = entry[0].split("\t")
        list_samples.append(tmp)
      csv_array = np.stack(list_samples)
      print(f'csv_array: {csv_array[:,4].astype(int)}')
      print(f'sample_id: {sample_id}')
      idx = np.where(csv_array[:,4].astype(int) == sample_id) # get the index of the sample_id in the cvs
      print(f'idx len: {len(idx[0])}')
      assert len(idx[0]) > 0, "Sample_id not found in the dataset."
      idx = idx[0][0]
      print(f'idx: {idx}')
      if stride_window>0:
        old_value_stride = self.dataset.stride_window
        self.dataset.stride_window = stride_window 
      features = self.dataset._custom_collate_fn([self.dataset.__getitem__(idx)])
      if stride_window>0:
        self.dataset.stride_window = old_value_stride
      print(features[0].shape)
      device = 'cuda' if torch.cuda.is_available() else 'cpu' #TODO: move somewhere else 
      self.backbone.model.to(device)
      self.backbone.model.eval()
      print('Computing features...')
      feature, unique_labels, unique_subject_id, unique_sample_id, unique_path = self._compute_features(features[0], features[1], features[2], features[3], features[4], device,True)
      # del features
      print('Predicting...')
      feature = feature.detach().cpu().squeeze()
      print(feature.shape)
      
      predictions = self.head.svr.predict(feature) # TODO: Change to use self.head.predict
      print('predictions',predictions)
      print('Plotting...')
      indices = np.arange(len(predictions)).astype(int)
      print('indices', indices)
      plt.figure(figsize=(6, 4))
      plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
      plt.plot(indices, predictions)
      plt.xlabel('Frame chunk index', fontsize=11)
      plt.ylabel('Prediction', fontsize=11)
      plt.title(f'Prediction Graph for sample ID: {sample_id}, ground truth: {csv_array[idx,2]}', fontsize=14)
      plt.xticks(fontsize=12)
      plt.yticks(fontsize=12)
      plt.tight_layout()
      plt.show()
      print('csv', csv_array[idx])
      input_video_path = os.path.join(self.dataset.path_dataset, csv_array[idx,1], csv_array[idx,5]+'.mp4')
      print('input_video_path', input_video_path)
      # for frames in features[5]:
      #   print(f'frames: {frames}')
      output_video_path = os.path.join('PartA','video','custom_video',csv_array[idx,5]+f'_{self.dataset.stride_window}'+'.mp4')
      self.dataset.save_frames_as_video([input_video_path],[features[5].numpy()], output_video_path ,[predictions], [features[1][0]])
    
  def plot_prediction_graph_all(self, sample_ids, stride_window=0):
    all_predictions = []
    with torch.no_grad():
      csv_array = self.dataset.video_labels.to_numpy()
      list_samples = []
      for entry in csv_array:
        tmp = entry[0].split("\t")
        list_samples.append(tmp)
      csv_array = np.stack(list_samples)
      
      input_video_paths = []
      all_frames = []
      all_labels = []
      for sample_id in sample_ids:
        idx = np.where(csv_array[:, 4].astype(int) == sample_id)
        assert len(idx[0]) > 0, f"Sample_id {sample_id} not found in the dataset."
        idx = idx[0][0]
        
        if stride_window > 0:
          old_value_stride = self.dataset.stride_window
          self.dataset.stride_window = stride_window
        
        features = self.dataset._custom_collate_fn([self.dataset.__getitem__(idx)])
        
        if stride_window > 0:
          self.dataset.stride_window = old_value_stride
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.backbone.model.to(device)
        self.backbone.model.eval()
        
        feature, _, _, _, _ = self._compute_features(features[0], features[1], features[2], features[3], features[4], device, True)
        print(f'features[1]->labels', features[1])
        feature = feature.detach().cpu().squeeze()
        predictions = self.head.svr.predict(feature)
        all_predictions.append(predictions)
        
        input_video_path = os.path.join(self.dataset.path_dataset, csv_array[idx, 1], csv_array[idx, 5] + '.mp4')
        output_video_path = os.path.join('PartA', 'video', 'custom_video', 'all' + f'_{len(sample_ids)}' + '.mp4')
        
        input_video_paths.append(input_video_path)
        all_frames.append(features[5].numpy())
        all_labels.append(features[1][0])
      self.dataset.save_frames_as_video(input_video_paths, all_frames, output_video_path,all_predictions, all_labels)
    
    return all_predictions
  
  
  
  
  def plot_comparison_prediction_gt(self, list_samples=None):
    # Supposition -> I have 1 prediction for one video  
    if list_samples is None: # all samples in the self.dataset.path_labels are used
      with torch.no_grad():
        features_list, gt_list, subject_ids, sample_ids, _, _ = self._extract_features_from_dataset()
      print('Computing features...')
      features_list = features_list.detach().cpu().numpy().squeeze()
      gt_list = gt_list.detach().cpu().numpy().squeeze()
      prediction_list = self.head.predict(features_list)
      print(prediction_list)
      tensor_subjects_prediction_gt = np.stack((subject_ids, prediction_list, gt_list), axis=1)
      unique_subject = np.unique(subject_ids)
      # print(f'tensor_subjects_prediction_gt.shape {tensor_subjects_prediction_gt.shape}')
      unique_subject = np.unique(subject_ids)
      # unique_gt = np.unique(gt_list)
      unique_gt = np.arange(0,4) # TODO: add class_list in dataset when created
      # print(f'unique_subject {unique_subject},  unique_gt {unique_gt}')
      class_count_gt = {class_id: np.zeros(len(unique_subject)) for class_id in (gt_list)}
      class_count_pred = {class_id: np.zeros(len(unique_subject)) for class_id in (gt_list)}
      # print(class_count_gt)
      for i, subject in enumerate(unique_subject):
        for gt in unique_gt:
          idx_gt = np.where((tensor_subjects_prediction_gt[:,0] == subject) & (tensor_subjects_prediction_gt[:,2] == gt))
          idx_pred = np.where((tensor_subjects_prediction_gt[:,0] == subject) & (tensor_subjects_prediction_gt[:,1] == gt))
          # print(f'idx {idx_gt[0].shape}')
          class_count_gt[gt][i] = idx_gt[0].shape[0]
          class_count_pred[gt][i] = idx_pred[0].shape[0]

    # Plotting the comparison
    # class_count_gt = {
    #     0: [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    #     1: [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
    #     2: [0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    #     3: [1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    # }

    # class_count_pred = {
    #     0: [3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0],
    #     1: [2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
    #     2: [1.0, 3.0, 7.0, 2.0, 4.0, 2.0, 1.0, 3.0],
    #     3: [1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 4.0, 0.0]
    # } 
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(unique_subject))
    cmap = cm.get_cmap('tab20', len(unique_gt))  # Use a categorical colormap like 'tab10'
    colors = [cmap(i) for i in range(len(unique_gt))]
    for gt,i  in enumerate(unique_gt):
      ax.bar(index, class_count_gt[gt], bar_width, label=f'Class {gt}', color=colors[gt],edgecolor='k')
      ax.bar(index + bar_width, class_count_pred[gt], bar_width, color=colors[gt],edgecolor='k')
      
    ax.set_xlabel('Subjects')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of Ground Truth and Predictions by Subject')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels([('grt '+str(sbj)+ ' prd') for sbj in unique_subject])
    ax.legend()
    plt.tight_layout()
    plt.show()
    

  def plot_tsne_colored(self, color_by='subject'):
    """
    Plots the t-SNE reduction of the features in 2D with colors based on subject, gt, or predicted class.
    Args:
      stop_after (int, optional): Number of iterations after which to stop the feature extraction. Defaults to 5.
      color_by (str, optional): Criterion for coloring the points ('subject', 'label', 'prediction'). Defaults to 'subject'.
    """
    X, y, subjects_id, _, _, _ = self._extract_features_from_dataset()
    X = X.reshape(X.shape[0], -1).detach().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy()
    
    if color_by == 'subject':
      colors = subjects_id
      color_label = 'Subject ID'
    elif color_by == 'gt':
      colors = y
      color_label = 'Groundtruth Label'
    elif color_by == 'prediction':
      predictions = self.head.predict(X)
      colors = predictions
      color_label = 'Predicted Class'
    else:
      raise ValueError("color_by must be 'subject', 'label', or 'prediction'")
    
    unique_colors = np.unique(colors)
    color_map = plt.cm.get_cmap('tab20', len(unique_colors))
    color_dict = {val: color_map(i) for i, val in enumerate(unique_colors)}
    
    unique_subjects = np.unique(subjects_id)
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_', '+', '1', '2', '3', '4']
    marker_dict = {val: markers[i % len(markers)] for i, val in enumerate(unique_subjects)}
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for val in unique_colors:
      idx = colors == val
      for subject in unique_subjects:
        subject_idx = subjects_id == subject
        combined_idx = idx & subject_idx
        plt.scatter(X_tsne[combined_idx, 0], X_tsne[combined_idx, 1], color=color_dict[val],  alpha=0.7, marker=marker_dict[subject])
    
    plt.legend()
    plt.title(f't-SNE Reduction to 2D (Colored by {color_label})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    
    
    
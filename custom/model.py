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
import custom.tools as tools

from custom.head import HeadSVR, HeadGRU, CrossValidationGRU
import os
from sklearn.manifold import TSNE

from tsnecuda import TSNE as cudaTSNE # available only on Linux

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,
              path_labels, preprocess, sample_frame_strategy, head, head_params, download_if_unavailable=False,
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
    self.dataset = customDataset(path_dataset, path_labels, preprocess, sample_frame_strategy, stride_window=stride_window, clip_length=clip_length,
                                 batch_size=batch_size)
    self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False, collate_fn=self.dataset._custom_collate_fn) # TODO: put inside customDataset and return a dataset and dataLoader
    if head == 'SVR':
      self.head = HeadSVR(svr_params=head_params)
    elif head == 'GRU':
      assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
      output_tensor = [1, int(self.backbone.frame_size/self.backbone.tubelet_size), self.backbone.out_spatial_size, self.backbone.out_spatial_size, self.backbone.embed_dim]
      if embedding_reduction:
        for dim in self.neck.dim_embed_reduction:
          output_tensor[dim] = 1

      if clips_reduction:
        print(self.neck.dim_clips_reduction)
        output_tensor[self.neck.dim_clips_reduction+1] = 1
      head_params['input_size'] = np.prod(output_tensor).astype(int)
      print(f'head_params : {head_params}')
      print(f'output_tensor : {output_tensor}')
      self.head = HeadGRU(dropout=head_params['dropout'], input_size=head_params['input_size'], 
                          hidden_size=head_params['hidden_size'], num_layers=head_params['num_layers'])

  def cross_validation(self, k=5, batch_size=16):
    if isinstance(self.head, HeadGRU):
      cv_gru = CrossValidationGRU(head=self.head)
      # Perform k-fold cros-validation
      print('GRU with k-fold cross-validation...')
      self.dataset.set_path_labels('train')
      dict_feature_extraction_train = self._extract_features() # feats,labels,subject_id,sample_id,path
      X = dict_feature_extraction_train['features']
      y = dict_feature_extraction_train['list_labels']
      subject_ids = dict_feature_extraction_train['list_subject_id']
      results = cv_gru.k_fold_cross_validation(X, y, subject_ids, k=k, num_epochs=5, batch_size=batch_size, lr=0.001)
      return results
    
    elif isinstance(self.head, HeadSVR):
      print('SVR with k-fold cross-validation...')
      dict_feature_extraction_train = self._extract_features() # feats,labels,subject_id,sample_id,path
      X = dict_feature_extraction_train['features']
      y = dict_feature_extraction_train['list_labels']
      subject_ids = dict_feature_extraction_train['list_subject_id']
      X = X.reshape(X.shape[0],-1).detach().cpu().numpy()
      y = y.squeeze().detach().cpu().numpy()
      list_split_indices,results = self.head.k_fold_cross_validation(k=k,X=X, y=y, groups=subject_ids)
      return list_split_indices, results

  
  def train(self):
    if isinstance(self.head, HeadSVR):
      print('Training using SVR...')
      # Extract feature from training set
      self.dataset.set_path_labels('train') 
      dict_feature_extraction_train = self._extract_features() # TODO: Optimize keeping only dict key usefull
      print('feature shape: ',dict_feature_extraction_train['features'].shape)
      X_train = dict_feature_extraction_train['features']
      y_train = dict_feature_extraction_train['list_labels']
      
      X_train = X_train.reshape(X_train.shape[0],-1).detach().cpu().numpy()
      y_train = y_train.squeeze().detach().cpu().numpy()
      subject_ids_train = dict_feature_extraction_train['list_subject_id']
      print('subject_ids_train',subject_ids_train)
      # Extract feature from test set
      self.dataset.set_path_labels('test') 
      dict_feature_extraction_train = self._extract_features() # TODO: Optimize keeping only dict key usefull
      X_test = dict_feature_extraction_train['features']
      y_test = dict_feature_extraction_train['list_labels']
      
      X_test = X_test.reshape(X_test.shape[0],-1).detach().cpu().numpy()
      y_test = y_test.squeeze().detach().cpu().numpy()
      subject_ids_test = dict_feature_extraction_train['list_subject_id']
      print('subject_ids_test',subject_ids_test)

      dict_results = self.head.fit(X_train=X_train,y_train=y_train, subject_ids_train=subject_ids_train,
                                     X_test=X_test, y_test=y_test, subject_ids_test=subject_ids_test,)

    if isinstance(self.head, HeadGRU):
      print('Training using GRU...')
      self.dataset.set_path_labels('train')
      count_subject_ids_train, count_y_train = self.dataset.get_unique_subjects_and_classes() 
      dict_feature_extraction_train = self._extract_features() # TODO: Optimize keeping only dict key usefull
      X_train = dict_feature_extraction_train['features'] 
      y_train = dict_feature_extraction_train['list_labels']
      subject_ids_train = dict_feature_extraction_train['list_subject_id']
      # sample_ids_train = dict_feature_extraction_train['list_sample_id'] 
      
      self.dataset.set_path_labels('test')
      count_subject_ids_test, count_y_test = self.dataset.get_unique_subjects_and_classes() 
      dict_feature_extraction_test = self._extract_features()
      X_test = dict_feature_extraction_test['features'] 
      y_test = dict_feature_extraction_test['list_labels'] 
      subjects_id_test = dict_feature_extraction_test['list_subject_id']
      dict_results = self.head.start_train_test(X_train=X_train, y_train=y_train, subject_ids_train=subject_ids_train,
                                                X_test=X_test, y_test=y_test, subject_ids_test=subjects_id_test, 
                                                num_epochs=2,batch_size=1)
    
    # Plot the results
    tools.plot_mea_per_class(title='training', mae_per_class=dict_results['train_loss_per_class'][-1], unique_classes=dict_results['y_unique'], count_classes=count_y_train)
    tools.plot_mea_per_class(title='test', mae_per_class=dict_results['test_loss_per_class'][-1], unique_classes=dict_results['y_unique'], count_classes=count_y_test)
    
    tools.plot_mea_per_subject(title='training', mae_per_subject=dict_results['train_loss_per_subject'][-1], uniqie_subject_ids=dict_results['subject_ids_unique'],count_subjects=count_subject_ids_train)
    tools.plot_mea_per_subject(title='test', mae_per_subject=dict_results['test_loss_per_subject'][-1], uniqie_subject_ids=dict_results['subject_ids_unique'],count_subjects=count_subject_ids_test )
    
    return dict_results

  
  def _extract_features(self,stop_after=3):
    """
    Extracts features from the dataset using the model's backbone.
    Args:
      stop_after (int, optional): Number of iterations after which to stop the feature extraction. Defaults to 3.
    Returns:
      dict: A dictionary containing the following elements:
          - 'features' (torch.Tensor): Stacked features extracted from the dataset.
          - 'list_labels' (torch.Tensor): Stacked labels corresponding to the features.
          - 'list_subject_id' (numpy.ndarray): Array of subject IDs.
          - 'list_sample_id' (numpy.ndarray): Array of sample IDs.
          - 'list_path' (numpy.ndarray): Array of paths corresponding to the samples.
          - 'list_frames' (torch.Tensor): Stacked frames sampled from the dataset.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"extracting features using.... {device}")
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
        print(f'unique_id: {unique_sample_id}')
        list_frames.append(list_sampled_frames)
        list_features.append(feature)
        list_labels.append(unique_labels)
        list_sample_id.append(unique_sample_id)
        list_subject_id.append(unique_subject_id)
        list_path.append(unique_path)
        count += 1
        # if count % stop_after == 0:
        #   break
    print('Feature extraceton done')
    return {
            'features':torch.stack([feature for feature in list_features]),
           'list_labels':torch.stack([label for label in list_labels]),
           'list_subject_id':np.stack([subject_id for subject_id in list_subject_id]).squeeze(),
           'list_sample_id':np.stack([sample_id for sample_id in list_sample_id]),
           'list_path':np.stack([path for path in list_path]),
           'list_frames':torch.stack(list_frames)
           }  
  
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
    if isinstance(self.head, HeadSVR):
      print('GridSearch using SVR...')
      self.dataset.set_path_labels('val')
      X, y, subjects_id,_ , _, _ = self._extract_features()
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
      output_video_path = os.path.join('partA','video','custom_video',csv_array[idx,5]+f'_{self.dataset.stride_window}'+'.mp4')
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
        output_video_path = os.path.join('partA', 'video', 'custom_video', 'all' + f'_{len(sample_ids)}' + '.mp4')
        
        input_video_paths.append(input_video_path)
        all_frames.append(features[5].numpy())
        all_labels.append(features[1][0])
      self.dataset.save_frames_as_video(input_video_paths, all_frames, output_video_path,all_predictions, all_labels)
    
    return all_predictions
  
  
  
  
  def plot_comparison_prediction_gt(self, list_samples=None):
    # Supposition -> I have 1 prediction for one video  

    if list_samples is None: # all samples in the self.dataset.path_labels are used
      with torch.no_grad():
        features_list, gt_list, subject_ids, sample_ids, _, _ = self._extract_features()
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
    

  def plot_tsne_colored_fixed_subject(self, color_by='subject',use_cuda=False,perplexity=20):
    """
    Plots the t-SNE reduction of the features in 2D with colors based on either gt or predicted class and shapes represent the subject.
    Args:
      stop_after (int, optional): Number of iterations after which to stop the feature extraction. Defaults to 5.
      color_by (str, optional): Criterion for coloring the points ('subject', 'label', 'prediction'). Defaults to 'subject'.
    """
    X, y, subjects_id, _, _, _ = self._extract_features()
    X = X.reshape(X.shape[0], -1).detach().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy()
    
    if color_by == 'gt':
      colors = y
      color_label = 'Groundtruth Label'
    elif color_by == 'prediction':
      predictions = self.head.predict(X)
      colors = predictions
      color_label = 'Predicted Class'
    else:
      raise ValueError("color_by must be 'subject', 'gt', or 'prediction'")
    
    unique_colors = np.unique(colors)
    color_map = plt.cm.get_cmap('tab20', len(unique_colors))
    color_dict = {val: color_map(i) for i, val in enumerate(unique_colors)}
    
    unique_subjects = np.unique(subjects_id)
    
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'x', 'd', '|', '_', '+', '1', '2', '3', '4']
    marker_dict = {val: markers[i % len(markers)] for i, val in enumerate(unique_subjects)}
    
    if use_cuda and X.shape[0] > 194:
      tsne = cudaTSNE(n_components=2, perplexity=perplexity, random_state=42)
    else:
      tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(10, 8))
    for val in unique_colors:
      idx = colors == val
      for subject in unique_subjects:
        subject_idx = (subjects_id == subject)
        combined_idx = idx & subject_idx
        plt.scatter(X_tsne[combined_idx, 0], X_tsne[combined_idx, 1], color=color_dict[val],  alpha=0.7, marker=marker_dict[subject])
      handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dict[val], markersize=10, label=f'Class {val}') for val in unique_colors]
      plt.legend(handles=handles)

    # plt.legend()
    plt.title(f't-SNE Reduction to 2D (Colored by {color_label})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

  def plot_tsne(self, color_by='subject', use_cuda=False, perplexity=20):
    """
    Plots the t-SNE reduction of the features in 2D with colors based on subject, gt, or predicted class.
    Args:
      color_by (str, optional): Criterion for coloring the points ('subject', 'label', 'prediction'). Defaults to 'subject'.
      use_cuda (bool, optional): Whether to use CUDA for t-SNE computation. Defaults to False.
      perplexity (int, optional): Perplexity parameter for t-SNE. Defaults to 20.
    """
    X, y, subjects_id, _, _, _ = self._extract_features()
    X = X.reshape(X.shape[0], -1).detach().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy()
    print(X.shape)
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
      raise ValueError("color_by must be 'subject', 'gt', or 'prediction'")
    
    unique_colors = np.unique(colors)
    color_map = plt.cm.get_cmap('tab20', len(unique_colors))
    color_dict = {val: color_map(i) for i, val in enumerate(unique_colors)}
    
    if use_cuda and X.shape[0] > 194:
      tsne = cudaTSNE(n_components=2, perplexity=perplexity, learning_rate=10)
    else:
      tsne = TSNE(n_components=2, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)
    
    plt.figure(figsize=(10, 8))
    for val in unique_colors:
      idx = colors == val
      plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=color_dict[val], label=f'{color_label} {val}', alpha=0.7)
    
    plt.legend()
    plt.title(f't-SNE Reduction to 2D (Colored by {color_label})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    
    
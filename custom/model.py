from sklearn.utils import resample
import re
import torch
from custom.backbone import VideoBackbone, VitImageBackbone
from custom.dataset import customDataset
import numpy as np
import os
import custom.tools as tools
from custom.dataset import get_dataset_and_loader
from custom.head import GRUHead, AttentiveHeadJEPA
from custom.helper import CUSTOM_DATASET_TYPE, MODEL_TYPE, get_shift_for_sample_id
import pandas as pd
import custom.helper as helper
# import wandb
import tqdm

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,stride_inside_window,
              path_labels, sample_frame_strategy, head, head_params, adapter_dict,num_clips_per_video, use_sdpa,
              batch_size_training,stride_window,clip_length,dict_augmented,prefetch_factor,soft_labels,
              features_folder_saving_path,concatenate_temporal,label_smooth,n_workers,new_csv_path,
              load_dataset_in_memory,stratified_training=False,
              gaussian_sigma_min=None,gaussian_sigma_max=None,gaussian_kernel_size=None):
    """
    Initialize the custom model. 
    Parameters:
    model_type (str): Type of the model to be used. 
    embedding_reduction (int): Dimension reduction for embeddings.
    clips_reduction (int): Dimension reduction for clips.
    path_dataset (str): Path to the dataset.
    path_labels (str): Path to the labels.
    sample_frame_strategy (str): Strategy for sampling frames.
    download_if_unavailable (bool, optional): Flag to download the model if unavailable. Defaults to False.
    batch_size (int, optional): Batch size for data loading. Defaults to 1.
    stride_window (int, optional): Stride window for sampling frames. Defaults to 2.
    clip_length (int, optional): Length of each video clip. Defaults to 16.
    svr_params (dict, optional): Parameters for the Support Vector Regressor (SVR). Defaults to {'kernel': 'rbf', 'C': 1, 'epsilon': 0.1}.
    """
    
    if model_type != MODEL_TYPE.ViT_image:
      freeze_backbone = not head_params.get('train_backbone', 0)
      unfreeze_layers = head_params.get('unfreeze_layers', 0)
      self.backbone = VideoBackbone(
        model_type,
        adapter_dict=adapter_dict,
        use_sdpa=use_sdpa,
        freeze_backbone=freeze_backbone,
        unfreeze_layers=unfreeze_layers,
      )
    else:
      self.backbone = VitImageBackbone()
      if head_params.get('adapter_dict', None) is not None:
        raise ValueError("Adapter dictionary is not supported for ViT image model type.")
    
    self.dataset = customDataset(path_dataset=path_dataset,
                                 path_labels=path_labels,
                                 model_type=model_type,
                                 num_clips_per_video=num_clips_per_video,
                                 sample_frame_strategy=sample_frame_strategy,
                                 stride_window=stride_window,
                                 stride_inside_window=stride_inside_window,
                                 clip_length=clip_length,
                                 gaussian_sigma_min=gaussian_sigma_min,
                                 gaussian_sigma_max=gaussian_sigma_max,
                                 gaussian_kernel_size=gaussian_kernel_size)
    self.batch_size_training = batch_size_training
    
    self.n_workers = n_workers
    self.dict_augmented = dict_augmented
    if os.path.exists(new_csv_path):
      print(f'CSV file with augmentations already exists at {new_csv_path}. Using the existing file.')
      complete_df = pd.read_csv(new_csv_path,sep='\t', dtype={'sample_name':str,'subject_name':str})
    else:
      complete_df = self.dataset.generate_csv_augmented(original_csv_path=path_labels,
                                  dict_augmentation=dict_augmented,
                                  stratified_training=stratified_training,
                                  out_csv_path=new_csv_path)
            
    self.concatenate_temporal = concatenate_temporal
    self.path_to_extracted_features = features_folder_saving_path
    self.dataset_type = tools.get_dataset_type(self.path_to_extracted_features)
    self.load_dataset_in_memory = load_dataset_in_memory
    
    self.label_smooth = label_smooth
    self.soft_labels = soft_labels 
    self.prefetch_factor = prefetch_factor
    self.T_S_S_shape = None
    self.embedding_reduction = embedding_reduction
    self.clips_reduction = clips_reduction
    
    # Get dataset type
    # if self.dataset_type == CUSTOM_DATASET_TYPE.WHOLE and self.load_dataset_in_memory:
    #   if not helper.dict_data_in_memory:
    #     if stratified_training:
          
    #     self.load_whole_dict_data_in_memory(complete_df=complete_df)
    
    if self.dataset_type == CUSTOM_DATASET_TYPE.AGGREGATED and helper.dict_data is None:
      self.set_global_dict_data_from_hdd_(complete_df=complete_df)
      # Set label according to the csv (ex: binary classification instead of multiclass)
      self._set_real_label_from_csv_(complete_df=complete_df)
    
    # Instantiate the head model
    if head == 'GRU':
      if model_type != MODEL_TYPE.ViT_image:
        assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
      self.head = GRUHead(**head_params)
    elif head == 'ATTENTIVE_JEPA':
      if hasattr(torch.nn.functional,"scaled_dot_product_attention"):
        use_sdpa = True
        print('Use SDPA for computing')
      else:
        use_sdpa = False
        print('SDPA not available. Using standard attention.')
        
      if self.dataset_type == CUSTOM_DATASET_TYPE.AGGREGATED:
        T_S_S_shape = np.array(helper.dict_data['features'][0].shape[:3])
        if embedding_reduction.value is not None:
          shape_feats = helper.dict_data['features'][0].numpy()
          T_S_S_shape = np.mean(shape_feats, axis=embedding_reduction.value, keepdims=True).shape[:3]
          print(f'\nApplied embedding reduction, T_S_S shape: {T_S_S_shape} ')
          # T_S_S_shape = T_S_S_shape.squeeze()
      elif self.dataset_type == CUSTOM_DATASET_TYPE.WHOLE:
        # T_S_S_shape = [8,16,16] if head_params['input_dim'] == 1408 else [8,14,14] # 1408 giant model
        T_S_S_shape = [self.backbone.frame_size // self.backbone.tubelet_size, self.backbone.out_spatial_size, self.backbone.out_spatial_size]
        if embedding_reduction.value is not None:
          for el in embedding_reduction.value:
            T_S_S_shape[el - 1] = 1
          print(f'\nApplied embedding reduction, T_S_S shape: {T_S_S_shape} ')
      elif self.dataset_type == CUSTOM_DATASET_TYPE.BASE:
        T_S_S_shape = [self.backbone.frame_size // self.backbone.tubelet_size, self.backbone.out_spatial_size, self.backbone.out_spatial_size]
        if embedding_reduction.value is not None:
          dummy_arr = np.zeros((1,*T_S_S_shape))
          T_S_S_shape = np.mean(dummy_arr,axis=embedding_reduction.value,keepdims=True).shape[1:]
      print(f"Shape of T_S_S_shape: {T_S_S_shape}\n")
      print(f'\n num_heads: {head_params["num_heads"]},\n num_cross_heads: {head_params["num_cross_heads"]}\n')
      self.T_S_S_shape = T_S_S_shape
      # check if head_params['head_init_path'] is a folder or .pt/.safetensors file
      
      self.head_init_path = head_params['head_init_path'] if 'head_init_path' in head_params else None
      if head_params['adversarial_head']:
        adversarial_out_dim = complete_df['subject_id'].nunique()
      else:
        adversarial_out_dim = None
      if head_params.get('multitask_head', False):
        multitask_num_classes = head_params['multitask_num_classes']
      else:
        multitask_num_classes = None
      self.head_params = head_params
      self.head = AttentiveHeadJEPA(embed_dim=head_params['input_dim'],
                                          num_classes=head_params['num_classes'],
                                          num_heads=head_params['num_heads'],
                                          q_k_v_dim=head_params['q_k_v_dim'] if 'q_k_v_dim' in head_params else None,
                                          num_cross_heads=head_params['num_cross_heads'],
                                          dropout=head_params['dropout'],
                                          attn_dropout=head_params['attn_dropout'],
                                          residual_dropout=head_params['residual_dropout'],
                                          mlp_ratio=head_params['mlp_ratio'],
                                          remove_head=head_params['remove_head'] if 'remove_head' in head_params else False,
                                          pos_enc=head_params['pos_enc'],
                                          custom_mlp=head_params['custom_mlp'],
                                          grid_size_pos=T_S_S_shape, # [T, S, S]
                                          depth=head_params['depth'],
                                          adversarial_head=head_params['adversarial_head'],
                                          adversarial_out_dim=adversarial_out_dim,
                                          multitask_head=head_params.get('multitask_head', False),
                                          multitask_num_classes=multitask_num_classes,
                                          head_init_path=head_params['head_init_path'],
                                          num_queries=head_params['num_queries'],
                                          agg_method=head_params['agg_method'],
                                          drop_path_mode=head_params['drop_path_mode'] if 'drop_path_mode' in head_params else 'row',
                                          use_sdpa=use_sdpa,
                                          norm_layer=torch.nn.BatchNorm1d if head_params.get('use_batch_norm', 0) else torch.nn.LayerNorm,
                                          embedding_reduction=helper.EMBEDDING_REDUCTION.get_embedding_reduction(head_params['embedding_reduction']),
                                          backbone=self.backbone if self.dataset_type == CUSTOM_DATASET_TYPE.BASE else None,
                                          coral_loss=head_params['coral_loss'],
                                          complete_block=head_params['complete_block'],
                                          cross_block_after_transformers=head_params['cross_block_after_transformers'],
                                          skip_init_weights=head_params.get('skip_init_weights', False),
                                          type_head=head_params.get('type_head', 0),
                                          mlp_num_hidden_layers=head_params.get('mlp_num_hidden_layers', 1)
                                          )
    else:
      raise NotImplementedError(f'{head} not implemented')
    # If features are not precomputed, instntiate the backbone model
    if self.dataset_type == CUSTOM_DATASET_TYPE.BASE:
      self.backbone_dict = {
        'backbone': self.backbone,
        'instance_model_name': tools.get_instace_model_name(self.head),
        'model': self.head,
        'concatenate_temporal': self.concatenate_temporal,
        'embedding_reduction':self.embedding_reduction,
      }
    else:
      self.backbone_dict = None
  
  def load_whole_dict_data_in_memory(self,complete_df):
    # list_sample_name = set(complete_df['sample_name'].tolist())

    # Collect folders to scan: base + augmentation variant folders (including $N variants)
    folders_to_scan = [self.path_to_extracted_features]
    base_name = os.path.basename(self.path_to_extracted_features)
    parent_dir = os.path.dirname(self.path_to_extracted_features)
    active_aug_types = set()
    for type_augm, p in self.dict_augmented.items():
      if p > 0 and p <= 1 and 'latent' not in type_augm:
        active_aug_types.add(type_augm)
        aug_folder = f"{self.path_to_extracted_features}_{type_augm}"
        if os.path.isdir(aug_folder):
          folders_to_scan.append(aug_folder)
    if active_aug_types and os.path.isdir(parent_dir):
      variant_pattern = re.compile(
        r'^' + re.escape(base_name) + r'_([^$]+?)\$(\d+)$'
      )
      for entry in os.listdir(parent_dir):
        m = variant_pattern.match(entry)
        if m and m.group(1) in active_aug_types:
          full_path = os.path.join(parent_dir, entry)
          if os.path.isdir(full_path):
            folders_to_scan.append(full_path)
    formatted_folders = "\n- ".join(folders_to_scan)
    print(f'Folders to LOAD in memory:\n {formatted_folders}')
    
    def _scan_safetensors(folder):
      """
      Recursively yield (path, size_bytes) for .safetensors files under folder,
      excluding any name containing '$'. Uses os.scandir for lower overhead than os.walk.

      Args:
        folder: Root directory to scan.

      Yields:
        tuple[str, int]: (file path, file size in bytes).
      """
      stack = [folder]
      while stack:
        current = stack.pop()
        try:
          with os.scandir(current) as it:
            for entry in it:
              if entry.is_dir(follow_symlinks=False):
                stack.append(entry.path)
              elif entry.is_file(follow_symlinks=False):
                name = entry.name
                if name.endswith('.safetensors') and '$' not in name:
                  yield entry.path, entry.stat(follow_symlinks=False).st_size
        except (PermissionError, FileNotFoundError):
          continue

    # Find all matching safetensors files and accumulate total size in a single pass
    all_paths = []
    total_bytes = 0
    pbar = tqdm.tqdm(folders_to_scan, desc="Scanning folders for .safetensors", unit="folder")
    for folder in pbar:
      for path, size in _scan_safetensors(folder):
        all_paths.append(path)
        total_bytes += size
        pbar.set_postfix(found=len(all_paths), gb=f"{total_bytes / (1024**3):.3f}")

    print(f"\nEstimated memory needed: {total_bytes / (1024**3):.2f} GB for {len(all_paths)} files")

    # Load all files and move tensors to shared memory
    cache = {}
    for path in tqdm.tqdm(all_paths, desc="Loading WHOLE features into memory"):
      # abs_path = os.path.abspath(path)
      dict_data = tools.load_dict_data(saving_folder_path=path)
      # key = os.path.splitext(os.path.basename(path))[0]
      cache[path] = dict_data

    helper.dict_data = cache
    helper.dict_data_in_memory = True
    print(f"Loaded {len(cache)} safetensors files into shared memory.")
    
    
  def set_global_dict_data_from_hdd_(self,complete_df):
    list_sample_id = complete_df['sample_id'].tolist()
    dict_data_original = tools.load_dict_data(saving_folder_path=self.path_to_extracted_features)
    
    # Mask to keep only the data for the sample IDs in the CSV file
    mask_keep_data = np.isin(dict_data_original['list_sample_id'],list_sample_id)

    # Filter the data in the dictionary using the mask to save memory
    if dict_data_original['features'].shape[0] != np.sum(mask_keep_data):
      for k,v in dict_data_original.items():
        dict_data_original[k] = v[mask_keep_data]
      
    for type_augm, p in self.dict_augmented.items():
      if p > 0 and p<= 1 and not 'latent' in type_augm:
        dict_data_augm = tools.load_dict_data(saving_folder_path=os.path.splitext(self.path_to_extracted_features)[0]+f'_{type_augm}'+
                                              ('.safetensors' if "safetensors" in self.path_to_extracted_features else ''))
        mask_keep_data = np.isin(dict_data_augm['list_sample_id'],list_sample_id)
        
        # Filter the data in the dictionary using the mask to save memory
        for k,v in dict_data_augm.items():
          # dict_data_augm[k] = v[mask_keep_data]
          # Concatenate the data from the original and augmented dictionaries
          if isinstance(v, np.ndarray):
            dict_data_original[k] = np.concatenate((dict_data_original[k], v[mask_keep_data]), axis=0)
          if isinstance(v, torch.Tensor):
            dict_data_original[k] = torch.cat((dict_data_original[k], v[mask_keep_data]), dim=0)
    
    helper.dict_data = dict_data_original
    
  def _set_real_label_from_csv_(self,complete_df):
    tensor_sample_id = torch.tensor(complete_df['sample_id'])
    for sample_id in tensor_sample_id:
      # get the real label from csv
      real_csv_label = complete_df[complete_df['sample_id'] == sample_id.item()]['class_id'].values[0] 
      if helper.is_latent_basic_augmentation(sample_id) or helper.is_latent_masking_augmentation(sample_id):
        mask = helper.dict_data['list_sample_id'] == ((sample_id - 1) % helper.step_shift + 1) # to get the real sample_id
      else:
        mask = helper.dict_data['list_sample_id'] == sample_id
      nr_values = mask.sum().item()
      
      if nr_values == 0:
        print(f"Sample ID {sample_id} not found in the dataset. set_real_label_from_csv will not work properly.")
      # create a tensor with the same label
      csv_labels = torch.full((nr_values,),real_csv_label,dtype=helper.dict_data['list_labels'].dtype) 
      helper.dict_data['list_labels'][mask] = csv_labels

  def test_pretrained_model(self,path_model_weights,state_dict, csv_path, criterion, concatenate_temporal,is_test=True,**kwargs):
    """
    Evaluate the model using the specified dataset.
    Parameters:
      csv_path (str): Path to the CSV file containing the dataset.
      criterion (torch.nn.Module, optional): Loss function to be used. Default is nn.L1Loss().
      round_output_loss (bool, optional): Flag to round the output loss. Default is False.
    Returns:
      dict: A dictionary containing the results of the evaluation process, including:
      - 'losses': List of losses.
      - 'loss_per_class': Loss per class, reshaped to (1, -1).
      - 'loss_per_subject': Loss per subject, reshaped to (1, -1).
      - 'subject_ids_unique': Unique subject IDs.
      - 'y_unique': Unique classes.
    """
    if not is_test:
      raise Exception('Set is_test to True. Currently this function is only for testing.')
    
    if path_model_weights is not None:
      self.head.load_state_weights(path_model_weights)
    else:
      self.head.load_state_dict(state_dict) # TODO: change to self.head.load_state_dict()
    
    # CHeck if the criterion is coral_loss
    try:
      is_coral_loss = True if criterion.__name__ == 'coral_loss' else False
    except AttributeError:
      is_coral_loss = False
      print('No coral loss. Using standard loss function.')
    args_get_dataset = {
      'csv_path': csv_path,
      'batch_size': self.batch_size_training,
      'dataset_type': self.dataset_type,
      'is_training': False,
      'num_clips_per_video': self.dataset.num_clips_per_video,
      'sample_frame_strategy': self.dataset.type_sample_frame_strategy,
      'root_folder_features': self.path_to_extracted_features,
      'concatenate_temporal': concatenate_temporal,
      'shuffle_training_batch': False,
      'backbone_dict': self.backbone_dict,
      'stride_inside_window': self.dataset.stride_inside_window,
      'model': self.head,
      'is_coral_loss': is_coral_loss,
      'soft_labels': self.soft_labels,
      'prefetch_factor': self.prefetch_factor,
      'label_smooth': self.label_smooth,
      'n_workers': self.n_workers,
    }
    wargs = {k:v for k,v in kwargs.items() if k not in args_get_dataset.keys()}
    
    test_dataset, test_loader = get_dataset_and_loader(**args_get_dataset,
                                                        **wargs
                                                        )
    unique_test_subjects = torch.tensor(test_dataset.get_unique_subjects())
    unique_classes = torch.tensor(test_dataset.get_unique_classes())
    
    # LOG HISTORY SAMPLE -> check is is possible to use npint

    if helper.LOG_HISTORY_SAMPLE and torch.min(unique_classes)>=0 and torch.max(unique_classes)<=255:
      # list_test_sample = test_dataset.get_all_sample_ids()
      # history_test_sample_predictions = {id: torch.zeros(1) for id in list_test_sample}
      history_test_sample_predictions = {}
    else:
      history_test_sample_predictions = None
    
    # Check if is coral_loss
    try:
      is_coral_loss = True if criterion.__name__ == 'coral_loss' else False
    except AttributeError:
      is_coral_loss = False
      print('No coral loss. Using standard loss function.')
    evaluate_args = {
      'val_loader': test_loader,
      'criterion': criterion,
      'unique_val_subjects': unique_test_subjects,
      'unique_val_classes': unique_classes,
      'is_test': is_test,
      'is_coral_loss': is_coral_loss,
      'epoch': 0, # to get the right position for history_test_sample_predictions
      'history_val_sample_predictions': history_test_sample_predictions,
      'adversarial_loss_lambda': kwargs.get('adversarial_loss_lambda', 0)
    }
    wargs = {k:v for k,v in kwargs.items() if k not in evaluate_args.keys()}
    dict_test = self.head.evaluate(val_loader=test_loader, 
                                   criterion=criterion,
                                   unique_val_subjects=unique_test_subjects,
                                   unique_val_classes=unique_classes,
                                   is_test=is_test,
                                   adv_criterion= torch.nn.CrossEntropyLoss() if 'adversarial_loss_lambda' in kwargs and kwargs['adversarial_loss_lambda'] > 0 else None,
                                   adversarial_loss_lambda=kwargs['adversarial_loss_lambda'] if 'adversarial_loss_lambda' in kwargs else None,
                                   is_coral_loss=is_coral_loss,
                                   epoch=0, # to get the right position for history_test_sample_predictions
                                   history_val_sample_predictions=history_test_sample_predictions,
                                   **wargs)
    
    dict_test['history_test_sample_predictions'] = history_test_sample_predictions
    dict_test['test_unique_subject_ids'] = unique_test_subjects.numpy()
    dict_test['test_count_subject_ids'] = test_dataset.get_count_subjects()
    dict_test['test_unique_y'] = unique_classes
    dict_test['test_count_y'] = test_dataset.get_count_classes()
    return dict_test
  
  
  def free_gpu_memory(self):
    del self.head
    # self.head.model.to('cpu')
    self.backbone.model.to('cpu')
    torch.cuda.empty_cache()
  
  def augment_csv(self, train_csv_path,target_list_string=None,stratified_training=False):
    
    list_augmentations_available = helper.get_augmentation_availables(self.path_to_extracted_features)
    if target_list_string is not None and not target_list_string.startswith('strategy'):
      list_augmentations_available = target_list_string.split(',')
      # list_augmentations_available = [augm for augm in list_only_augments]
    else:
      if stratified_training == 2:
        list_augmentations_available.append('latent_basic')
      elif stratified_training == 3:
        list_augmentations_available.append('latent_masking')
      elif stratified_training == 4:
        list_augmentations_available.append('latent_basic')
        list_augmentations_available.append('latent_masking')
      
    self.dict_augmented = {augm: 1 for augm in list_augmentations_available}
    print(f'\nAugmenting BIOVID CSV with the following augmentations: {list_augmentations_available}')
    augm_df = helper.generate_csv_augmented(original_csv_path=train_csv_path,
                                  dict_augmentation=self.dict_augmented,
                                  out_csv_path=train_csv_path,
                                  stratified_training=stratified_training)
    print(f'\n\nStratified training BIOVID CSV generated at {train_csv_path}.') 
    return augm_df
      
  def train(self, train_csv_path, val_csv_path, num_epochs, criterion,
            lr,saving_path,round_output_loss,
            shuffle_training_batch,init_network,
            regularization_lambda_L1,key_for_early_stopping,early_stopping,
            concatenate_temporal,clip_grad_norm,regularization_lambda_L2,
            enable_optuna_pruning=False,trial=None,**kwargs
            ):

        # CHeck if the criterion is coral_loss
    try:
      is_coral_loss = True if criterion.__name__ == 'coral_loss' else False
    except AttributeError:
      is_coral_loss = False
      print('No coral loss. Using standard loss function.')
    batch_size = self.batch_size_training
    final_df = pd.read_csv(train_csv_path,sep='\t', dtype={'sample_name':str,'subject_name':str})
    _original_df = final_df.copy()  # save before augment_csv overwrites train_csv_path
    # Stratified augmentation: add latent augmentation types based on stratified_training value
    if kwargs.get('stratified_training', False):
      final_df = self.augment_csv(train_csv_path=train_csv_path,
                        target_list_string=kwargs['only_augments'],
                        stratified_training=kwargs['stratified_training'])
    # Target K samples per class via balanced augmentation
    if kwargs.get('target_samples_per_class_training') is not None:
      # Use the pre-augmentation df to avoid double-shifting already-augmented sample_ids
      df_original = _original_df
      target_samples_per_class = kwargs['target_samples_per_class_training']
      df_final = helper.generate_balanced_dataframe(df_original=df_original,
                                         list_augmentations_available=helper.get_augmentation_availables(self.path_to_extracted_features),
                                         target_samples_per_class=target_samples_per_class)
      if kwargs['perfect_bal_strategy'] is not None:
        class_counts = df_final['class_id'].value_counts()
        n_classes = len(class_counts)
        batch_size = batch_size * n_classes
        print(f'Adjusted batch size for perfect balancing: {batch_size}')
        assert batch_size % n_classes == 0, "Batch size must be divisible by number of classes for perfect balancing."
        print(f'################ Using perfectly balanced stratification with {target_samples_per_class} samples per class ################')
        df_final = helper.refine_perfectly_balanced_target_samples_per_class(df_final, target_samples_per_class, kwargs['perfect_bal_strategy'])
      df_original.to_csv(os.path.basename(train_csv_path).replace('.csv', '_original.csv'), index=False, sep='\t')
      df_final.to_csv(train_csv_path, index=False, sep='\t')
      print(f"Original class distribution:\n{df_original['class_id'].value_counts().sort_index()}")
      print(f"Stratification completed. New class distribution:\n{df_final['class_id'].value_counts().sort_index()}")
      final_df = df_final
        
    if kwargs.get('sampler_augmented_only_types', False) and not kwargs.get('stratified_training', False):
      final_df = self.augment_csv(train_csv_path=train_csv_path,
                        target_list_string=kwargs['sampler_augmented_only_types'],
                        stratified_training=0)
                         
    # print feature dimension for debugging
    if self.dataset_type == CUSTOM_DATASET_TYPE.WHOLE and self.load_dataset_in_memory:
      if not helper.dict_data_in_memory:
        self.load_whole_dict_data_in_memory(complete_df=final_df)
    # Stage 1
    dict_results = self.head.start_train(batch_size=batch_size,
                                          criterion=criterion,
                                          lr=lr,
                                          soft_labels=self.soft_labels,
                                          concatenate_temp_dim=concatenate_temporal,
                                          early_stopping=early_stopping,
                                          key_for_early_stopping=key_for_early_stopping,
                                          saving_path=saving_path,
                                          init_network=init_network,
                                          dataset_type=self.dataset_type,
                                          num_clips_per_video=self.dataset.num_clips_per_video,
                                          sample_frame_strategy=self.dataset.type_sample_frame_strategy,
                                          stride_inside_window=self.dataset.stride_inside_window,
                                          num_epochs=num_epochs,
                                          is_coral_loss=is_coral_loss,
                                          train_csv_path=train_csv_path,
                                          regularization_lambda_L1=regularization_lambda_L1,
                                          regularization_lambda_L2=regularization_lambda_L2,
                                          root_folder_features=self.path_to_extracted_features,
                                          round_output_loss=round_output_loss,
                                          shuffle_training_batch=shuffle_training_batch,
                                          val_csv_path=val_csv_path,
                                          prefetch_factor=self.prefetch_factor,
                                          n_workers=self.n_workers,
                                          clip_grad_norm=clip_grad_norm,
                                          label_smooth=self.label_smooth,
                                          backbone_dict=self.backbone_dict,
                                          enable_optuna_pruning=enable_optuna_pruning,
                                          trial=trial,
                                          # head_embed_dim=self.head_params['input_dim'],
                                          # num_classes=self.head_params['num_classes'],
                                          **kwargs
                                          )
    
    count_subject_ids_train, count_y_train = tools.get_unique_subjects_and_classes(train_csv_path)
    if val_csv_path is not None:
      count_subject_ids_val, count_y_val = tools.get_unique_subjects_and_classes(val_csv_path) 
    else:
      count_subject_ids_val = None
      count_y_val = None
    return {'dict_results':dict_results, 
              'count_y_train':count_y_train, 
              'count_y_val':count_y_val,
              'count_subject_ids_train':count_subject_ids_train,
              'count_subject_ids_val':count_subject_ids_val
              }    



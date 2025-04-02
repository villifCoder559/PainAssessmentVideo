import torch
from custom.backbone import VideoBackbone, VitImageBackbone
from custom.dataset import customDataset
import numpy as np
import sys
import custom.tools as tools
from custom.dataset import get_dataset_and_loader
from custom.head import LinearHead, GRUHead, AttentiveHead, AttentiveHeadJEPA
from custom.helper import CUSTOM_DATASET_TYPE, MODEL_TYPE, get_shift_for_sample_id
import pandas as pd
# import wandb

class Model_Advanced: # Scenario_Advanced
  def __init__(self, model_type, embedding_reduction, clips_reduction, path_dataset,
              path_labels, sample_frame_strategy, head, head_params,
              batch_size_training,stride_window,clip_length,dict_augmented,
              features_folder_saving_path,concatenate_temporal,label_smooth=0.0,n_workers=1,new_csv_path=None):
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
      self.backbone = VideoBackbone(model_type)
    else:
      self.backbone = VitImageBackbone()
    
    self.dataset = customDataset(path_dataset=path_dataset, 
                                 path_labels=path_labels, 
                                 sample_frame_strategy=sample_frame_strategy, 
                                 stride_window=stride_window, 
                                 clip_length=clip_length)
    self.batch_size_training = batch_size_training
    
    if head == 'GRU':
      if model_type != MODEL_TYPE.ViT_image:
        assert self.backbone.frame_size % self.backbone.tubelet_size == 0, "Frame size must be divisible by tubelet size."
      self.head = GRUHead(**head_params)
    elif head == 'ATTENTIVE':
      self.head = AttentiveHead(**head_params)
    elif head == 'ATTENTIVE_JEPA':
      self.head = AttentiveHeadJEPA(embed_dim=head_params['input_dim'],
                                          num_classes=head_params['num_classes'],
                                          num_heads=head_params['num_heads'],
                                          dropout=head_params['dropout'],
                                          attn_dropout=head_params['attn_dropout'],
                                          residual_dropout=head_params['residual_dropout'],
                                          mlp_ratio=head_params['mlp_ratio'],
                                          pos_enc=head_params['pos_enc'],
                                          )
      print(f'sys path: {sys.path}')
      print('Warning: Dropout is not implemented for the AttentiveClassifierJEPA model.')
    elif head == 'LINEAR':
      self.head = LinearHead(**head_params)
      
    self.concatenate_temporal = concatenate_temporal
    self.path_to_extracted_features = features_folder_saving_path
    self.dataset_type = tools.get_dataset_type(self.path_to_extracted_features)
    self.label_smooth = label_smooth
    if self.dataset_type == CUSTOM_DATASET_TYPE.BASE:
      self.backbone_dict = {
        'backbone': self.backbone,
        'instance_model_name': tools.get_instace_model_name(self.head.model),
        'model': self.head.model,
        'concatenate_temporal': self.concatenate_temporal
      }
    else:
      self.backbone_dict = None
    self.n_workers = n_workers

    if dict_augmented is not None:
      self.generate_csv_augmented(original_csv_path=path_labels,
                                  dict_augmentation=dict_augmented,
                                  out_csv_path=new_csv_path)
    
  def generate_csv_augmented(self, original_csv_path, dict_augmentation, out_csv_path):
    def _get_rnd_from_type(type_augm):
      if type_augm == 'hflip':
        return 42
      elif type_augm == 'jitter':
        return 53
      elif type_augm == 'rotate':
        return 63
      else:
        raise ValueError(f'Unknown augmentation type: {type_augm}')
    list_df = []
    df = pd.read_csv(original_csv_path,sep='\t')
    list_df.append(df)
    for type_augm, p in dict_augmentation.items():
      if p > 0 and p<= 1:
        df_sampled = df.sample(frac=p, random_state=_get_rnd_from_type(type_augm))
        df_sampled['sample_id'] = df_sampled['sample_id'].apply(lambda x: x + get_shift_for_sample_id(type_augm))
        list_df.append(df_sampled)
    df_merged = pd.concat(list_df, ignore_index=True)
    df_merged.to_csv(out_csv_path, index=False, sep='\t')
    print(f'CSV file with augmentations saved to {out_csv_path}')
    
  def test_pretrained_model(self,path_model_weights,state_dict, csv_path, criterion, concatenate_temporal,is_test):
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
      self.head.load_state_weights()
    else:
      self.head.model.load_state_dict(state_dict)
    test_dataset, test_loader = get_dataset_and_loader(csv_path=csv_path,
                                                        batch_size=self.batch_size_training,
                                                        concatenate_temporal=concatenate_temporal,
                                                        dataset_type=self.dataset_type,
                                                        is_training=False,
                                                        root_folder_features=self.path_to_extracted_features,
                                                        shuffle_training_batch=False,
                                                        backbone_dict=self.backbone_dict,
                                                        model=self.head.model,
                                                        label_smooth=self.label_smooth,
                                                        n_workers=self.n_workers
                                                                 )
    unique_test_subjects = test_dataset.get_unique_subjects()
    unique_classes = np.array(list(range(self.head.model.num_classes)))
    dict_test = self.head.evaluate(val_loader=test_loader, criterion=criterion, unique_val_subjects=unique_test_subjects,
                                    unique_val_classes=unique_classes, is_test=is_test)
    dict_test['test_unique_subject_ids'] = unique_test_subjects
    dict_test['test_count_subject_ids'] = test_dataset.get_count_subjects()
    dict_test['test_unique_y'] = unique_classes
    dict_test['test_count_y'] = test_dataset.get_count_classes()
    return dict_test
  
  
  def free_gpu_memory(self):
    self.head.model.to('cpu')
    self.backbone.model.to('cpu')
    torch.cuda.empty_cache()
    
  def train(self, train_csv_path, val_csv_path, num_epochs, criterion,
            optimizer_fn, lr,saving_path,round_output_loss,
            shuffle_training_batch,init_network,
            regularization_lambda_L1,key_for_early_stopping,early_stopping,
            enable_scheduler,concatenate_temporal,clip_grad_norm,regularization_lambda_L2
            ):
    """
    Train the model using the specified training and testing datasets.
    Parameters:
      train_csv_path (str): Path to the CSV file containing the training data.
      test_csv_path (str): Path to the CSV file containing the testing data.
      num_epochs (int, optional): Number of epochs for training. Default is 10.
      batch_size (int, optional): Batch size for training. Default is 1.
      criterion (torch.nn.Module, optional): Loss function to be used. Default is nn.L1Loss().
      optimizer_fn (torch.optim.Optimizer, optional): Optimizer function to be used. Default is optim.Adam.
      lr (float, optional): Learning rate for the optimizer. Default is 0.0001.
      saving_path (str, optional): Path to save the trained model. Default is None.
      init_weights (bool, optional): Flag to initialize the weights. Default is True.
    Returns:
      dict: A dictionary containing the results of the training process, including:
      - 'dict_results': {
                      - 'train_losses': List of training losses.
                      - 'train_loss_per_class': Training loss per class, reshaped to (1, -1).
                      - 'train_loss_per_subject': Training loss per subject, reshaped to (1, -1).
                      - 'test_losses': List of test losses.
                      - 'test_loss_per_class': Test loss per class, reshaped to (1, -1).
                      - 'test_loss_per_subject': Test loss per subject, reshaped to (1, -1).
                      - 'subject_ids_unique': Unique subject IDs in the combined training and test subject IDs.
                      - 'y_unique': Unique classes in the combined training and test labels.
                      - 'best_model_idx': best_model_epoch
                      }
      - 'count_y_train': Count of unique classes in the training set.
      - 'count_y_test': Count of unique classes in the testing set.
      - 'count_subject_ids_train': Count of unique subject IDs in the training set.
      - 'count_subject_ids_test': Count of unique subject IDs in the testing set.
    """
    count_subject_ids_train, count_y_train = tools.get_unique_subjects_and_classes(train_csv_path)
    count_subject_ids_val, count_y_val = tools.get_unique_subjects_and_classes(val_csv_path) 
    dict_results = self.head.start_train(batch_size=self.batch_size_training,
                                          criterion=criterion,
                                          optimizer=optimizer_fn,
                                          lr=lr,
                                          concatenate_temp_dim=concatenate_temporal,
                                          early_stopping=early_stopping,
                                          key_for_early_stopping=key_for_early_stopping,
                                          enable_scheduler=enable_scheduler,
                                          saving_path=saving_path,
                                          init_network=init_network,
                                          dataset_type=self.dataset_type,
                                          num_epochs=num_epochs,
                                          train_csv_path=train_csv_path,
                                          regularization_lambda_L1=regularization_lambda_L1,
                                          regularization_lambda_L2=regularization_lambda_L2,
                                          root_folder_features=self.path_to_extracted_features,
                                          round_output_loss=round_output_loss,
                                          shuffle_training_batch=shuffle_training_batch,
                                          val_csv_path=val_csv_path,
                                          n_workers=self.n_workers,
                                          clip_grad_norm=clip_grad_norm,
                                          label_smooth=self.label_smooth,
                                          backbone_dict=self.backbone_dict)
    return {'dict_results':dict_results, 
              'count_y_train':count_y_train, 
              'count_y_val':count_y_val,
              'count_subject_ids_train':count_subject_ids_train,
              'count_subject_ids_val':count_subject_ids_val
              }    

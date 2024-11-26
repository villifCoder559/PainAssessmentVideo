from custom.helper import CLIPS_REDUCTION,EMBEDDING_REDUCTION,MODEL_TYPE,SAMPLE_FRAME_STRATEGY
import os
from custom.model import Model_Advanced
from transformers import AutoImageProcessor
from custom.head import HeadSVR, HeadGRU
import time
import custom.tools as tools
from custom.helper import HEAD, MODEL_TYPE, EMBEDDING_REDUCTION, SAMPLE_FRAME_STRATEGY
import torch.nn as nn
import torch.optim as optim
import dataframe_image as dfi
import pandas as pd

def run_train_test(model_type, pooling_embedding_reduction, pooling_clips_reduction, sample_frame_strategy, 
                   path_csv_dataset, path_video_dataset, head, stride_window_in_video, 
                   head_params, preprocess, download_if_unavailable=False,
                   k_fold = 1, epochs = 10, criterion = nn.L1Loss(), optimizer_fn = optim.Adam, lr = 0.001):
  
  model_type = MODEL_TYPE.VIDEOMAE_v2_B
  pooling_embedding_reduction = EMBEDDING_REDUCTION.MEAN_SPATIAL
  pooling_clips_reduction = CLIPS_REDUCTION.NONE
  sample_frame_strategy = SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  # stride_window_in_video = 70 # not applid for UNIFORM and CENTRAL_SAMPLING

  path_dict = tools._generate_train_test_validation(path_csv_dataset)
  # path_dict ={
  #   'train' : os.path.join('partA','starting_point','train_21.csv'),
  #   'val' : os.path.join('partA','starting_point',''),
  #   'test' : os.path.join('partA','starting_point','test_5.csv')
  # }

  # path_video_dataset = os.path.join('partA','video','video')  
  # head = HeadSVR(svr_params={'kernel':'rbf','C':1,'epsilon':10})
  # head = head

  preprocess = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

  model_advanced = Model_Advanced(model_type=model_type,
                                  path_dataset=path_video_dataset,
                                  embedding_reduction=pooling_embedding_reduction,
                                  clips_reduction=pooling_clips_reduction,
                                  sample_frame_strategy=sample_frame_strategy,
                                  stride_window=stride_window_in_video,
                                  path_labels=path_dict,
                                  preprocess=preprocess,
                                  batch_size=1,
                                  head=head.value,
                                  head_params=head_params,
                                  download_if_unavailable=download_if_unavailable
                                  )
  # Check if the global folder exists 
  global_foder_name = 'history_run'
  if not os.path.exists(global_foder_name):
    os.makedirs(global_foder_name)
  run_folder_name = f'{os.path.split(model_type.value)[-1].split('.')[0]}_
                      {pooling_embedding_reduction.name if pooling_clips_reduction else None}_
                      {pooling_clips_reduction.name if pooling_clips_reduction else None}_
                      {sample_frame_strategy.name}_{head.name}' 
  
  # Create folder to save the run
  run_folder_path = os.path.join(global_foder_name,run_folder_name)
  if not os.path.exists(run_folder_path):
    os.makedirs(os.path.join(global_foder_name,run_folder_name))
  print(f"Run folder created at {run_folder_path}")
  
  # Save model configuration
  model_advanced.save_configuration(os.path.join(run_folder_path,'advancedmodel_config.json'))
  
  # Plot dataset distribution
  dataset_folder_path = os.path.join(run_folder_path,'dataset')
  if not os.path.exists(dataset_folder_path):
    os.makedirs(os.path.join(run_folder_path,'dataset'))

  for key in path_dict.keys():
    model_advanced.dataset.set_path_labels(key)
    model_advanced.dataset.plot_dataset_distribution(per_class=True, per_partecipant=True,
                                                     save_path=dataset_folder_path) # 1 plot
    model_advanced.dataset.plot_distribution_mean_std_duration(per_class=True, 
                                                               per_partecipant=True, 
                                                               save_path=dataset_folder_path) # 2 plots
  
  # Train the model
  train_folder_path = os.path.join(run_folder_path,f'train_{head}')
  if not os.path.exists(train_folder_path):
    os.makedirs(train_folder_path)
  if k_fold == 1:
    dict_train = model_advanced.train(num_epochs=epochs, batch_size=4, criterion=criterion,optimizer_fn=optimizer_fn, lr=lr)
    tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
                                  count_subject_ids_test=dict_train['count_subject_ids_train'],
                                  count_subject_ids_train=dict_train['count_subject_ids_test'],
                                  count_y_test=dict_train['count_y_test'], 
                                  count_y_train=dict_train['count_y_train'],
                                  saving_path=train_folder_path)
    
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][-1],
                                title='Train confusion matrix',
                                saving_path=os.path.join(train_folder_path,f'confusion_matrix_train.png'))
    
    tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['test_confusion_matricies'][-1],
                                title='Test confusion matrix',
                                saving_path=os.path.join(train_folder_path,f'confusion_matrix_test.png'))

  else: # k_fold > 1
    # Create folder to save results
    list_saving_paths_k_val = []
    path = os.path.join(train_folder_path,f'results_k{i}_cross_val')

    for i in range(k_fold):
      list_saving_paths_k_val.append(path)
      if not os.path.exists(path):
        os.makedirs(path)

    results,list_split_indices = model_advanced.run_k_fold_cross_validation(k=k_fold,
                                                                            batch_size=2,
                                                                            list_saving_paths_k_val=list_saving_paths_k_val)
    # SVR
    if isinstance(head,HEAD.SVR):
      df_results = results['df_results']
      df_results = pd.DataFrame(results)
      df_results['test_score'] = -df_results['test_score']
      df_results['train_score'] = -df_results['train_score']
      df_results.insert(df_results.columns.get_loc('estimator') + 1, 'kernel', df_results['estimator'].apply(lambda x: x.kernel))
      dfi.export(df_results,os.path.join(train_folder_path,f'results_k{k_fold}.png'))
      for idx in range(k_fold):
        tools.plot_confusion_matrix(confusion_matrix=results['train_confusion_matricies'][idx],
                                title=f'Train confusion matrix k_{idx}',
                                saving_path=os.path.join(list_saving_paths_k_val[idx],f'confusion_matrix_train.png'))                        
        tools.plot_confusion_matrix(confusion_matrix=results['test_confusion_matricies'][idx],
                                title=f'Test confusion matrix k_{idx}',
                                saving_path=os.path.join(list_saving_paths_k_val[idx],f'confusion_matrix_test.png'))
    #GRU
    elif isinstance(head,HEAD.GRU):
      for idx,dict_train in enumerate(results):
        tools.generate_plot_train_test_results(dict_results=dict_train['dict_results'], 
                                  count_subject_ids_test=dict_train['count_subject_ids_train'],
                                  count_subject_ids_train=dict_train['count_subject_ids_test'],
                                  count_y_test=dict_train['count_y_train'], 
                                  count_y_train=dict_train['count_y_test'],
                                  saving_path=list_saving_paths_k_val[idx])
         
        tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['train_confusion_matricies'][-1],
                                title='Train confusion matrix',
                                saving_path=os.path.join(list_saving_paths_k_val[idx],f'confusion_matrix_train.png'))
        
        tools.plot_confusion_matrix(confusion_matrix=dict_train['dict_results']['test_confusion_matricies'][-1],
                                title='Test confusion matrix',
                                saving_path=os.path.join(list_saving_paths_k_val[idx],f'confusion_matrix_test.png'))

  # plot tsne pre and post processing (if used GRU)
  # plot tsne considering all features from backbone
  saving_path_tsne = os.path.join(train_folder_path,'tsne')
  if not os.path.exists(saving_path_tsne):
    os.mkdir(saving_path_tsne)
  model_advanced.dataset.set_path_labels('test')
  dict_feats = model_advanced._extract_features()
  tools.plot_tsne(X=dict_feats['features'], labels=dict_feats['list_labels'],
                  saving_path=os.path.join(saving_path_tsne,'tsne_backbone'),
                  legend_label='gt')
  
  if isinstance(head, HEAD.GRU): # test split as above
    X_GRU = model_advanced.head.model.gru(dict_feats['features'])
    y_pred = model_advanced.head.model.fc(X_GRU) # TODO: separate in train and test
    y = dict_feats['list_labels']
    tools.plot_tsne(X_GRU,y_pred,'pred ')
    tools.plot_tsne(X_GRU,y,'gt ')

  elif isinstance(head, HEAD.SVR): #TODO: check if it is correct
    y_pred = model_advanced.head.predict(dict_feats['features'])
    y = dict_feats['list_labels']
    tools.plot_tsne(dict_feats['features'],y_pred,'pred ')
    tools.plot_tsne(dict_feats['features'],y  ,'gt ')

  # Create video with predictions
  video_folder_path = os.path.join(train_folder_path,'video')
  if not os.path.exists(video_folder_path):
    os.makedirs(video_folder_path)
  samples_ids = model_advanced.dataset.get_samples_ids('test')
  model_advanced.dataset.set_path_labels('test')
  model_advanced.plot_prediction_graph_all(sample_ids=samples_ids,predictions=y_pred, save_path=video_folder_path)

from custom.helper import *
from custom.model import *
import torch.nn as nn
import pickle

def get_model_type(model_type):
  if model_type == "VIDEOMAE_v2_S":
    return MODEL_TYPE.VIDEOMAE_v2_S
  elif model_type == "VIDEOMAE_v2_B":
    return MODEL_TYPE.VIDEOMAE_v2_B
  else:
    raise ValueError("Model type not found")
  
def get_pooling_embedding_reduction(pooling_embedding_reduction):
  if pooling_embedding_reduction == "MEAN_SPATIAL":
    return EMBEDDING_REDUCTION.MEAN_SPATIAL
  elif pooling_embedding_reduction == "MEAN_TEMPORAL":
    return EMBEDDING_REDUCTION.MEAN_TEMPORAL
  elif pooling_embedding_reduction == "MEAN_TEMPORAL_SPATIAL":
    return EMBEDDING_REDUCTION.MEAN_TEMPORAL_SPATIAL
  else:
    return EMBEDDING_REDUCTION.NONE
  
def get_pooling_clips_reduction(pooling_clips_reduction):
  if pooling_clips_reduction == "MEAN":
    return CLIPS_REDUCTION.MEAN
  elif pooling_clips_reduction == "NONE":
    return CLIPS_REDUCTION.NONE
  else:
    return None
  
def get_head(head):
  if head == "GRU":
    return HEAD.GRU
  else:
    raise ValueError("Head not found")
      
def get_sample_frame_strategy(sample_frame_strategy):
  if sample_frame_strategy == "SLIDING_WINDOW":
    return SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW
  else:
    raise ValueError("Sample frame strategy not found")  
  
def load_model_from_configuration(config_path):
  with open(config_path, 'r') as f: 
    config = json.load(f)  
    print(config)
  
  model_type = get_model_type(config['model_type'])
  pooling_embedding_reduction = get_pooling_embedding_reduction(config['pooling_embedding_reduction'])
  pooling_clips_reduction = get_pooling_clips_reduction(config['pooling_clips_reduction'])
  sample_frame_strategy = get_sample_frame_strategy(config['sample_frame_strategy'])
  path_dataset = "/".join((config['path_video_dataset'].split('/')[-3:]))
  path_csv_dataset = "/".join(config['path_csv_dataset'].split('/')[-3:])
  stride_window_in_video = config['stride_window_in_video']
  head = get_head(config['head'])
  params = config['head_params']
  features_folder_saving_path = "/".join(config['features_folder_saving_path'].split('/')[-4:])

  model_advanced = Model_Advanced(model_type=model_type,
                          path_dataset=path_dataset,
                          embedding_reduction=pooling_embedding_reduction,
                          clips_reduction=pooling_clips_reduction,
                          sample_frame_strategy=sample_frame_strategy,
                          stride_window=stride_window_in_video,
                          path_labels=path_csv_dataset,
                          batch_size_training=1,
                          batch_size_feat_extraction=1,
                          head=head.value,
                          head_params=params,
                          download_if_unavailable=False,
                          clip_length=16,
                          features_folder_saving_path=features_folder_saving_path
                          )
  return model_advanced

config_path = "Tests/regression/history_run_no_preprocess_1738971411/1738983367VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU/global_config.json"
model_advanced = load_model_from_configuration(config_path)
path_model_weights =  "Tests/regression/history_run_no_preprocess_1738971411/1738983367VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU/train_GRU/k0_cross_val/k0_cross_val_sub_0/best_model_ep_24.pth"
csv_path =            "Tests/regression/history_run_no_preprocess_1738971411/1738983367VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU/train_GRU/k0_cross_val/k0_cross_val_sub_0/val.csv"
dict_results = model_advanced.test_pretrained_model( path_model_weights=path_model_weights,
                                                      csv_path=csv_path,
                                                      log_file_path='test_model_regr.txt',
                                                      criterion=nn.L1Loss(),
                                                      round_output_loss=False,
                                                      is_test=True)
saving_path = os.path.join('Test_models', path_model_weights.split('/')[3], path_model_weights.split('/')[-2])
if not os.path.exists(saving_path):
  os.makedirs(saving_path)
tools.plot_confusion_matrix(confusion_matrix=dict_results['test_confusion_matrix'],
                            title=f'{csv_path}',
                            saving_path=os.path.join(saving_path,csv_path.split('/')[-1][:-4]+'_conf_matrix.png'),
                            )
# read file.pkl
with open('Tests/regression/history_run_no_preprocess_1738971411/1738983367VIDEOMAE_v2_B_MEAN_SPATIAL_NONE_SLIDING_WINDOW_GRU/k_fold_results.pkl', 'rb') as f:
  k_fold_results = pickle.load(f)
count = 0
for confusion_matrix in k_fold_results[0][f'{csv_path.split("/")[-2]}_train_val']['val_confusion_matricies']:
  tools.plot_confusion_matrix(confusion_matrix=confusion_matrix,
                              title=f'{csv_path}',
                              saving_path=os.path.join(saving_path,csv_path.split('/')[-1][:-4]+f'_conf_matrix_{count}.png'),
                              )
  count+=1



import custom.head as head
import custom.dataset as dataset
import os
import pickle
import custom.tools as tools
import torch
import tqdm
import custom.helper as helper

model_path = "history_run_samples_0_4_selftrained_ce_1_1_1_stride4_L1_366069_ATTENTIVE_JEPA_villi-Inspiron-16-Plus-7620_1750673405/1750673406842_VIDEOMAE_v2_G_NONE_NONE_SLIDING_WINDOW_ATTENTIVE_JEPA/train_ATTENTIVE_JEPA/k0_cross_val/k0_cross_val_sub_0/best_model_ep_4.pth"
config_path =os.path.join(*(model_path.split(os.sep)[:-4]),"k_fold_results.pkl")
with open(config_path, 'rb') as f:
  data = pickle.load(f)
head_params = data['config']['head_params']
att_head = head.AttentiveHeadJEPA(embed_dim=head_params['input_dim'],
                              num_classes=head_params['num_classes'],
                              num_heads=head_params['num_heads'],
                              num_cross_heads=head_params['num_cross_heads'],
                              dropout=head_params['dropout'],
                              attn_dropout=head_params['attn_dropout'],
                              residual_dropout=head_params['residual_dropout'],
                              mlp_ratio=head_params['mlp_ratio'],
                              pos_enc=head_params['pos_enc'],
                              grid_size_pos=head_params['T_S_S_shape'], # [T, S, S]
                              depth=head_params['depth'],
                              num_queries=head_params['num_queries'],
                              agg_method=head_params['agg_method'],
                              use_sdpa=True,
                              coral_loss=head_params['coral_loss'],
                              complete_block=head_params['complete_block'],
                              cross_block_after_transformers=head_params['cross_block_after_transformers'])

att_head.load_state_dict(torch.load(model_path))
csv_path = os.path.join(os.path.dirname(model_path),"train.csv")
savingfeatures_from_head_path =  os.path.join(os.path.dirname(model_path),f"head_features_{os.path.basename(csv_path).replace('.csv','')}")
print(f"CSV path: {csv_path}")
root_folder_features = "partA/video/features/all_front_giant_finetuned_1_1_1_stride4_interp_mirror.safetensors"
n_workers = 4
dataset_, loader_ = dataset.get_dataset_and_loader(batch_size=16,
                                            csv_path=csv_path,
                                            root_folder_features=root_folder_features,
                                            shuffle_training_batch=False,
                                            is_training=False,
                                            concatenate_temporal=False,
                                            dataset_type=tools.get_dataset_type(root_folder_features),
                                            prefetch_factor=2,
                                            backbone_dict=None,
                                            model=att_head, 
                                            is_coral_loss=False,
                                            soft_labels=0,
                                            label_smooth=0,
                                            n_workers=n_workers)

helper.dict_data = tools.load_dict_data(root_folder_features)
feature_list = []
y_list = []
subject_list = []
sample_id_list = []
count = 0
device = 'cuda'
att_head.to(device)
att_head.eval()

with torch.no_grad():
  for dict_batch_X, batch_y, batch_subjects,sample_id in tqdm.tqdm(loader_,total=len(loader_),desc=f'Feature extraction'):
    dict_batch_X = {key: value.to(device) for key, value in dict_batch_X.items()}
    batch_y = batch_y.to(device)
    # subject_batch_count[tmp] += 1
    dict_batch_X['list_sample_id'] = sample_id
    outputs,preds = att_head(**dict_batch_X,return_video_emb=True)
    if preds.shape[1] == 1: # if regression I don't need to keep dim 1 
      preds = preds.squeeze(1)
    feature_list.append(outputs.detach().cpu())
    y_list.append(batch_y.detach().cpu())
    subject_list.append(batch_subjects.detach().cpu())
    sample_id_list.append(sample_id.detach().cpu())
    # count += 1
    # if count == 5:
    #   break

dict_data = {
  'features': torch.cat(feature_list, dim=0),
  'labels': torch.cat(y_list, dim=0),
  'subjects': torch.cat(subject_list, dim=0),
  'sample_ids': torch.cat(sample_id_list, dim=0)
}
tools.save_dict_data(dict_data=dict_data,
                     saving_folder_path=savingfeatures_from_head_path,
                     save_as_safetensors=True)
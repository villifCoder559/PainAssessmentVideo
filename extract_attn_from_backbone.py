import custom.tools as tools
from custom.backbone import VideoBackbone
from torch.utils.data import DataLoader
import custom.dataset as dataset
import custom.helper as helper
import time
import torch

batch_size_feat_extraction = 1
n_workers = 1
model_type = helper.MODEL_TYPE.VIDEOMAE_v2_S
path_dataset = "/media/villi/TOSHIBA EXT/video_frontalized_interpolated_mirror"
path_labels = "partA/starting_point/samples.csv"

custom_ds = dataset.customDataset(path_dataset=path_dataset,
                                  sample_frame_strategy=helper.SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW,
                                  path_labels=path_labels)
backbone = VideoBackbone(model_type=model_type)
device = 'cuda'
dataloader = DataLoader(custom_ds, 
            batch_size=batch_size_feat_extraction,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=custom_ds._custom_collate_fn_extraction)
backbone.model.to(device)
backbone.model.eval()
count = 0
start = time.time()
for data, labels, subject_id,sample_id, path, list_sampled_frames in dataloader:
  data = data.to(device)
  with torch.no_grad():
    x,attn = backbone.forward_features(data,return_attn=True,return_embedding=False)
    print(f"Shape of attn: {attn.shape}")
    break
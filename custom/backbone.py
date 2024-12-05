from enum import Enum
from VideoMAEv2.models.modeling_finetune import vit_small_patch16_224, vit_base_patch16_224, vit_giant_patch14_224
from VideoMAEv2.models.modeling_pretrain import pretrain_videomae_giant_patch14_224
import os
import torch
import requests
from custom.helper import MODEL_TYPE

class backbone:
  def __init__(self,model_type, download_if_unavailable=False):
    assert model_type in MODEL_TYPE, f"Model type must be one of {MODEL_TYPE}."
    if not os.path.exists(model_type.value) and download_if_unavailable:
      self.download_model(model_type.value)
    else:
      assert os.path.exists(model_type.value), f"Model not found at {model_type.value}. Please set download_if_unavailable=True to download the model."
      
    if MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e == model_type:
      self.model = self._load_model_pretrained(model_type) # FIX: use load_model_finetune, this version doesn't have forward_features function
    else:
      self.model = self._load_model_finetune(model_type)
      self.tubelet_size = self.model.patch_embed.tubelet_size
      self.img_size = self.model.patch_embed.img_size[0] # [224, 224]
      self.frame_size = 16
      self.out_spatial_size = self.img_size / self.model.patch_embed.patch_size[0] # 224/16 = 14 
      self.embed_dim = self.model.embed_dim
    self.model_type = model_type
    
  def download_model(self,model_type):
    url="https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/"
    model_name = os.path.split(model_type)[-1]
    if model_type == MODEL_TYPE.VIDEOMAE_v2_S.value or model_type == MODEL_TYPE.VIDEOMAE_v2_B.value:
      url += "distill/"+model_name
    else:
      url += "mae-g/"+model_name
    print(url)
    weights_path = os.path.join('VideoMAEv2','pretrained', model_name)  
    
    # Download the file from the given URL
    response = requests.get(url, stream=True) # Stream the download for large files
    with open(weights_path, 'wb') as f:
      for chunk in response.iter_content(chunk_size=8192): # Download in chunks
        if chunk:
          f.write(chunk)    
                
  def _load_model_pretrained(self,model_type):
    if model_type == MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e:
      weights = torch.load(model_type.value)['model']
      new_weights = {}
      # remove decoder weights
      for key, value in weights.items():
        if 'encoder' in key and 'decoder' not in key:
          new_key = key.replace('encoder.', '')
          new_weights[new_key] = value
      del weights
      
      # load the model
      model = pretrain_videomae_giant_patch14_224(pretrained=False)
      model = model.encoder
      model.load_state_dict(new_weights)
    return model
  
  def _load_model_finetune(self,model_type):
    
    if model_type == MODEL_TYPE.VIDEOMAE_v2_S:
      kwargs = {'num_classes': 710}
      model = vit_small_patch16_224(pretrained=False, **kwargs)
      model.load_state_dict(torch.load(MODEL_TYPE.VIDEOMAE_v2_S.value)['module'])
    
    elif model_type == MODEL_TYPE.VIDEOMAE_v2_B:
      kwargs = {'num_classes': 710}
      model = vit_base_patch16_224(pretrained=False, **kwargs)
      model.load_state_dict(torch.load(MODEL_TYPE.VIDEOMAE_v2_B.value)['module'])
    
    elif model_type == MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e_K710_it_HMDB51_ft:
      kwargs = {'num_classes': 51}
      model = vit_giant_patch14_224(pretrained=False, **kwargs)
      model.load_state_dict(torch.load(MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e_K710_it_HMDB51_ft.value)['module'])
    
    self.remove_unwanted_layers(model)
    return model
  
  def remove_unwanted_layers(self,model):
    # Remove 'head', 'norm', and 'fc_norm' if they exist
    if hasattr(model, "head"):
      model.head = None
    if hasattr(model, "norm"):
      model.norm = None
    if hasattr(model, "fc_norm"):
      model.fc_norm = None
      
  def forward_features(self, x):
    """
    Forward pass to extract features from the input tensor.

    Args:
      x (torch.Tensor): Input tensor of shape [nr_clips, channels, nr_frames=16, H, W]

    Returns:
      emb (torch.Tensor): tensor of shape [batch_size, temporal_dim, patch_h, patch_w, emb_dim]
    """
    # x.shape = [B, C, T, H, W]
    num_frames = x.shape[2]
    # added return_embedding in the original code to catch the embedding
    feat = self.model.forward_features(x, return_embedding=True) # torch.Size([1, 1568, 768]) (VIDEOMAE_v2_B model)
    B = feat.shape[0]
    T = int(feat.shape[1] / (self.out_spatial_size ** 2))
    S = int(feat.shape[1] / (self.out_spatial_size * (num_frames / self.tubelet_size))) # 1568 / (14*8) = 14
    emb = feat.reshape(B, T, S, S, self.embed_dim)
    return emb # [1,1568,768] -> [1,8,14,14,768]



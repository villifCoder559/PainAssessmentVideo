import os
import torch
import sys
import yaml
from enum import Enum
from pathlib import Path

from VideoMAEv2.models.modeling_finetune import (
  vit_small_patch16_224,
  vit_base_patch16_224,
  vit_giant_patch14_224
)
from VideoMAEv2.models.modeling_pretrain import pretrain_videomae_giant_patch14_224, pretrain_videomae_base_patch16_224,pretrain_videomae_small_patch16_224
from transformers import ViTFeatureExtractor, ViTModel
from custom.helper import MODEL_TYPE, ModelTypeEntry
import torch.nn as nn

class BackboneBase(nn.Module):
  """Base class for feature extraction backbones."""
  
  def __init__(self):
    """Initialize the backbone base class."""
    super(BackboneBase,self).__init__()
    self.model = None
    self.model_type = None
    self.device = "cuda"
        
  def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    """Extract features from input. To be implemented by subclasses.
    
    Args:
      x: Input tensor
        
    Returns:
      Feature embeddings
    """
    pass
  
  def load_pretrained_weights(self):
    """Load pretrained weights from a file.
    
    Args:
      weights_path: Path to the weights file
    """
    pass

class VideoBackbone(BackboneBase):
  """Video feature extraction backbone using VideoMAEv2."""
  
  def __init__(
    self, 
    model_type,
    remove_head: bool = True,
    adapter_dict: dict = None,
    use_sdpa: bool = False,
    custom_model_path: bool = False,
  ):
    """Initialize the video backbone.
    
    Args:
      model_type: Type of model to use from MODEL_TYPE enum
      remove_head: Whether to remove classification head
      download_if_unavailable: Download model if not found locally
    
    Raises:
      AssertionError: If model type is invalid or model file not found
    """
    super().__init__()
    
    # Validate model type
    assert isinstance(model_type, ModelTypeEntry), f"Model type must be one of {MODEL_TYPE._entries}."

    # Check if model exists or needs downloading
    model_path = Path(model_type.value)
    self.model_type = model_type
    self.use_sdpa = use_sdpa
    if not model_path.exists():
      # if download_if_unavailable:
      #   print(f"Model not found at {model_path}. Downloading...")
      #   self._download_model(model_type)
      # else:
      raise FileNotFoundError(
        f"Model not found at {model_path}. "
        )
    if model_type in [MODEL_TYPE.VIDEOMAE_v2_S, MODEL_TYPE.VIDEOMAE_v2_B, MODEL_TYPE.VIDEOMAE_v2_G, MODEL_TYPE.VIDEOMAE_v2_G_unl]:
      if model_type == MODEL_TYPE.VIDEOMAE_v2_G_unl or custom_model_path:
        # Load pretrained model
        self.model = self._load_model_pretrained(model_type)
      else:
        self.model = self._load_model_finetune(model_type, remove_head=remove_head, use_sdpa=use_sdpa)
      
      # Freeze model parameters
      for param in self.model.parameters():
        param.requires_grad = False
      
      # Cache important model parameters for efficient access
      self.tubelet_size = self.model.patch_embed.tubelet_size
      self.img_size = self.model.patch_embed.img_size[0]  # [224, 224]
      self.patch_size = self.model.patch_embed.patch_size[0]  # 16
      self.out_spatial_size = self.img_size // self.patch_size  # 224/16 = 14
      self.embed_dim = self.model.embed_dim
      self.frame_size = 16  # Default frame size
      self.remove_head = remove_head
      if adapter_dict is not None:
        self.model.add_adapters(adapter_dict) # add_adapters defined in modeling_finetune.py
        
      
    elif model_type in [MODEL_TYPE.VJEPA2_G_384]:
      # IMPORT THE LIBRARY
      config_path = "vjepa2/configs/train/vitg16/cooldown-384px-64f.yaml"
      self.config_path = config_path
      jepa_model = self.load_jepa2_weights()
      self.model = jepa_model[0]
      
      # Cache important model parameters for efficient access
      self.tubelet_size = self.model.tubelet_size
      self.img_size = self.model.img_height  # height == widths
      self.patch_size = self.model.patch_size  # 16
      self.out_spatial_size = self.img_size // self.patch_size  # 224/16 = 14
      self.embed_dim = self.model.embed_dim
      self.frame_size = self.model.num_frames  # Default frame size
      if adapter_dict is not None:
        NotImplementedError(f"Fine-tuning type {adapter_dict} is not implemented for JEPA2 backbone.")

    self.adapter_dict = adapter_dict

  
  def load_jepa2_weights(self):
    if not sys.path.exists('vjepa2'):
      sys.path.insert(0, os.path.abspath('vjepa2'))
    with open (self.config_path,"r") as f:
      config = yaml.safe_load(f)
    import vjepa2.src.hub.backbones as backbones
    
    # Create model and weights for the encoder
    jepa_model = backbones.vjepa2_vit_giant_384(pretrained=False,**config)
    state_dict = torch.load(self.model_type.value, map_location='cpu', weights_only=True)
    del state_dict['predictor']
    encoder_state_dict = backbones._clean_backbone_key(state_dict["encoder"])
    res = jepa_model[0].load_state_dict(encoder_state_dict, strict=False)  # state_dict has pos_embed but we use RoPE
    print(res)
    return jepa_model

  def load_pretrained_weights(self):
    if self.model_type in [MODEL_TYPE.VIDEOMAE_v2_B, MODEL_TYPE.VIDEOMAE_v2_G, MODEL_TYPE.VIDEOMAE_v2_G_unl, MODEL_TYPE.VIDEOMAE_v2_S]:
      if self.model_type == MODEL_TYPE.VIDEOMAE_v2_G_unl:
        # Load pretrained model
        self.model = self._load_model_pretrained(self.model_type)
      else:
        self.model = self._load_model_finetune(self.model_type, remove_head=self.remove_head, use_sdpa=self.use_sdpa)
    elif self.model_type in [MODEL_TYPE.VJEPA2_G_384]:
      # Load JEPA2 model
      self.model = self.load_jepa2_weights()[0]
      
  def _load_model_pretrained(self, model_type):
    """Load a pretrained model.
    
    Args:
      model_type: Type of model to load
        
    Returns:
      Loaded model
    """
    # if model_type == MODEL_TYPE.VIDEOMAE_v2_G_unl:
    # Load weights
    checkpoint = torch.load(model_type.value, map_location='cpu')
    weights = checkpoint['model']
    # kwargs = {'init_ckpt':model_type.value,
    #           'pretrained':True}
    # Filter out decoder weights, keeping only encoder weights
    new_weights = {
      key.replace('encoder.', ''): value 
      for key, value in weights.items() 
      if 'encoder' in key and 'decoder' not in key
    }
    
    # Create model and load weights
    if model_type == MODEL_TYPE.VIDEOMAE_v2_G_unl:
      model = pretrain_videomae_giant_patch14_224(pretrained=False)
    elif model_type == MODEL_TYPE.VIDEOMAE_v2_S:
      model = pretrain_videomae_small_patch16_224(pretrained=False)
    elif model_type == MODEL_TYPE.VIDEOMAE_v2_B:
      model = pretrain_videomae_base_patch16_224(pretrained=False)
    elif model_type == MODEL_TYPE.VIDEOMAE_v2_G:
      model = pretrain_videomae_giant_patch14_224(pretrained=False)
    else:
      raise ValueError(f"Unsupported model type for pretrained loading: {model_type}")
    model = model.encoder
    model.load_state_dict(new_weights)
    
    return model
    
  def _load_model_finetune(self, model_type: Enum, remove_head: bool = True, use_sdpa: bool = False):
    """Load a fine-tuned model.
    
    Args:
      model_type: Type of model to load
      remove_head: Whether to remove the classification head
        
    Returns:
      Loaded model
    """
    model_map = {
      MODEL_TYPE.VIDEOMAE_v2_S: (vit_small_patch16_224, {'num_classes': 710}),
      MODEL_TYPE.VIDEOMAE_v2_B: (vit_base_patch16_224, {'num_classes': 710}),
      MODEL_TYPE.VIDEOMAE_v2_G: (vit_giant_patch14_224, {'num_classes': 710}),
      # MODEL_TYPE.VIDEOMAE_v2_G_unl: (vit_giant_patch14_224, None),
    }
    
    if model_type not in model_map:
      raise ValueError(f"Unsupported model type for fine-tuning: {model_type}")
    
    # Get model constructor and kwargs
    model_fn, kwargs = model_map[model_type]
    
    kwargs['use_sdpa'] = use_sdpa
    # Create model instance
    model = model_fn(pretrained=False, **kwargs)
    
    # Load weights
    checkpoint = torch.load(model_type.value, map_location=self.device,weights_only=True)
    model.load_state_dict(checkpoint['module'])
    
    # Remove classification head if requested
    if remove_head:
      self._remove_classification_layers(model)
    
    return model
  
  def _remove_classification_layers(self, model):
    """Remove classification layers from model.
    
    Args:
      model: Model to modify
    """
    # Remove classification and normalization layers
    for layer_name in ["head", "norm", "fc_norm"]:
      if hasattr(model, layer_name):
        setattr(model, layer_name, None)
        # print(f"Removed {layer_name} layer from model")
  
  @torch.no_grad()
  def forward_features(self, x: torch.Tensor, return_embedding: bool = True,return_attn: bool = False) -> torch.Tensor:
    """Extract features from video input. No preprocessing is applied, but the input tensor must be in the correct format.
    
    Args:
      x: Input tensor of shape [batch_size, channels, frames, height, width]
      return_embedding: Whether to return reshaped embeddings
        
    Returns:
      Feature embeddings
    """
    # Shape validation
    if x.dim() != 5:
      raise ValueError(f"Expected 5D input tensor [B,C,T,H,W], got {x.shape}")
    
    # Move model and input to device
    self.model.to(self.device)
    x = x.to(self.device)
    # Extract features
    self.model.eval()
    with torch.no_grad():
      # Forward pass
      # print(f'Free GPU memory: {torch.cuda.memory_reserved() / 1e9} GB')
      args = {'x':x, 'return_embedding':return_embedding, 'return_attn':return_attn}
      feat,attn = self.model.forward_features(**args)
    # Process and reshape features if needed
    if return_embedding:
      B = feat.shape[0]  # Batch size
      
      # Calculate dimensions
      T = int(feat.shape[1] / (self.out_spatial_size ** 2))  # Temporal dimension
      S = int(self.out_spatial_size)  # Spatial dimension
      
      feat = feat.reshape(B, T, S, S, self.embed_dim)# [B, T, S, S, embed_dim], e.g. [1, 8, 14, 14, 768]
    # feat = feat.to('cpu')
    # self.model.to('cpu') # Only for local testing
    if return_attn:
      attn = attn.to('cpu')
      return feat, attn  # Return both features and attention maps
    else:
      return feat  # Return raw features


class VitImageBackbone(BackboneBase):
  """Image feature extraction backbone using Vision Transformer."""
  
  def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k"):
    """Initialize the ViT image backbone.
    
    Args:
      model_name: Name of pretrained ViT model to use
    """
    super().__init__()
    
    # Load model and feature extractor
    print(f"Loading ViT model: {model_name}")
    self.model = ViTModel.from_pretrained(model_name)
    self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    self.model_type = MODEL_TYPE.ViT_image
    
    # Get patch size for spatial dimension calculations
    self.patch_size = 16  # Default for standard ViT models
    self.img_size = 224   # Default image size
    self.out_spatial_size = self.img_size // self.patch_size  # 14
    self.embed_dim = self.model.config.hidden_size
  
  @torch.no_grad()
  def forward_features(self, x: torch.Tensor) -> torch.Tensor:
    """Extract features from image input.
    
    Args:
      x: Input tensor of shape [batch_size, channels, 1, height, width]
        
    Returns:
      Feature embeddings
    """
    # Shape validation
    if x.dim() != 5:
      raise ValueError(f"Expected 5D input tensor [B,C,1,H,W], got {x.shape}")
    
    # Move to device
    self.model.to(self.device)
    x = x.to(self.device)
    
    # Remove singleton temporal dimension [B,C,1,H,W] -> [B,C,H,W]
    x = x.squeeze(2)
    
    # Extract features
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(x)
    
    # Get CLS token embeddings
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, embed_dim]
    
    # Optional: reshape to match video backbone output format
    # [B, embed_dim] -> [B, 1, 1, 1, embed_dim]
    B = cls_embedding.shape[0]
    reshaped_embedding = cls_embedding.reshape(B, 1, 1, 1, self.embed_dim)
    
    return reshaped_embedding
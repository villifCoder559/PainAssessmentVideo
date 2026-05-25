import os
import time
import torch
import sys
import yaml
from enum import Enum
from pathlib import Path

_VJEPA2_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vjepa2')
if _VJEPA2_ROOT not in sys.path:
  sys.path.insert(0, _VJEPA2_ROOT)

from VideoMAEv2.models.modeling_finetune import (
  vit_small_patch16_224,
  vit_base_patch16_224,
  vit_giant_patch14_224
)
from MAE_DFER import modeling_pretrain as dfer_modeling  
from VideoMAEv2.models.modeling_pretrain import pretrain_videomae_giant_patch14_224, pretrain_videomae_base_patch16_224,pretrain_videomae_small_patch16_224
from transformers import ViTFeatureExtractor, ViTModel
from custom.helper import MODEL_TYPE, ModelTypeEntry
import custom.helper as helper
import torch.nn as nn
from functools import partial


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
    model_type: ModelTypeEntry,
    remove_head: bool = True,
    adapter_dict: dict = None,
    use_sdpa: bool = False,
    freeze_backbone: bool = True,
    custom_model_path: bool = False,
    unfreeze_layers: int = 0,
  ):
    """Initialize the video backbone.

    Args:
      model_type:        Type of model to use from MODEL_TYPE enum.
      remove_head:       Whether to remove classification head.
      adapter_dict:      Optional adapter configuration dict.
      use_sdpa:          Whether to use scaled dot-product attention.
      freeze_backbone:   If True, freeze all backbone parameters.
      custom_model_path: Use a custom model path instead of the default.
      unfreeze_layers:   Number of last encoder blocks to unfreeze when
                         freeze_backbone=False. 0 means unfreeze all.

    Raises:
      AssertionError: If model type is invalid or model file not found.
      ValueError:     If unfreeze_layers exceeds the number of encoder blocks.
    """
    super().__init__()
    
    # Validate model type
    assert isinstance(model_type, ModelTypeEntry), f"Model type must be one of {MODEL_TYPE._entries}."

    # Check if model exists or needs downloading
    model_path = Path(model_type.value)
    self.model_type = model_type
    self.use_sdpa = use_sdpa
    # if not model_path.exists():
    #   # if download_if_unavailable:
    #   #   print(f"Model not found at {model_path}. Downloading...")
    #   #   self._download_model(model_type)
    #   # else:
    #   raise FileNotFoundError(
    #     f"Model not found at {model_path}. "
    #     )
    if model_type in [MODEL_TYPE.VIDEOMAE_v2_S, MODEL_TYPE.VIDEOMAE_v2_B, MODEL_TYPE.VIDEOMAE_v2_G, MODEL_TYPE.VIDEOMAE_v2_G_unl]:
      if model_type == MODEL_TYPE.VIDEOMAE_v2_G_unl or custom_model_path:
        # Load pretrained model
        self.model = self._load_model_pretrained(model_type)
      else:
        self.model = self._load_model_finetune(model_type, remove_head=remove_head, use_sdpa=use_sdpa)
      
      self.encoder_blocks = self.model.blocks
      self._apply_freeze_logic(freeze_backbone, unfreeze_layers)

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
        
    elif model_type == MODEL_TYPE.VJEPA_v2_1_B_384:
      if adapter_dict is not None:
        raise ValueError("Adapters are not supported for VJEPA 2.1 models.")
      self.model = self._load_vjepa2_1_vit_base(model_type, img_size=256, num_frames=64)
      self.tubelet_size = 2
      self.img_size = 256
      self.patch_size = 16
      self.out_spatial_size = self.img_size // self.patch_size  # 16
      self.embed_dim = 768
      self.frame_size = 64
      self.remove_head = remove_head
      self.encoder_blocks = getattr(self.model, 'blocks', None)
      self._apply_freeze_logic(freeze_backbone, unfreeze_layers)

    elif model_type in [MODEL_TYPE.VJEPA_v2_L_fpc64_256, MODEL_TYPE.VJEPA_v2_G_fpc64_384]:
      if adapter_dict is not None:
        raise ValueError("Adapters are not supported for JEPA2 models.")
      # Load JEPA2 model
      import transformers
      self.model = transformers.AutoModel.from_pretrained(model_type.value,cache_dir="/equilibrium/fvilli/PainAssessmentVideo/hugging_face_models")
      self.tubelet_size = self.model.config.tubelet_size
      self.img_size = self.model.config.image_size
      self.patch_size = self.model.config.patch_size
      self.out_spatial_size = self.img_size // self.patch_size  # 224/16 = 14
      self.embed_dim = self.model.config.hidden_size
      self.frame_size = self.model.config.frames_per_clip
      self.remove_head = remove_head
      self.encoder_blocks = None  # VJEPA2: block list not wired; fallback to all-or-nothing
      self._apply_freeze_logic(freeze_backbone, unfreeze_layers)
      
    elif model_type == MODEL_TYPE.DFER:
      if adapter_dict is not None:
        raise ValueError("Adapters are not supported for DFER model.")
      # Load DFER model
      dfer_args = {
        # "model_name": 'pretrain_videomae_base_dim512_no_depth_patch16_160',
        "encoder_depth": 16,
        "decoder_depth": 4,
        "epochs": 100,
        "lr": 3e-4,
        "num_workers": 20,
        "attn_type": "local_global",
        "part_win_size": (2, 5, 10),
        "lg_region_size": (2, 5, 10),
        "use_frame_diff_as_target": True,
        "mask_ratio": 0.9,
        "mask_type": "part_window",
        "input_size": 160,
        "batch_size": 32,
        "num_frames": 16,
        "sampling_rate": 4,
        "opt": "adamw",
        "opt_betas": (0.9, 0.95),
        "warmup_epochs": 5,
        "save_ckpt_freq": 10,
        "epochs": 100,
        "log_dir": "./output",
        "output_dir": "./output",
        "lr": 3e-4,
        "drop_path_rate": 0,
        "drop_block_rate": None,
        "lg_first_attn_type": "self",
        "lg_third_attn_type": "cross",
        "lg_attn_param_sharing_first_third": False,
        "lg_attn_param_sharing_all": False,
        "lg_no_second": False,
        "lg_no_third": False,
        "num_workers": 20,
        "attn_type": "local_global",
        "part_win_size": (2, 5, 10),
        "lg_region_size": (2, 5, 10),
        "use_frame_diff_as_target": True,
        "img_size": 160,
        "patch_size": 16,
        "encoder_embed_dim": 512,
        "encoder_num_heads":8,
        "encoder_num_classes":0,
        "decoder_num_classes": 1536,
        "decoder_embed_dim": 384,
        "decoder_num_heads": 6,
        "mlp_ratio": 4,
        "qkv_bias": True,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
      }
      self.model = dfer_modeling.PretrainVisionTransformer(**dfer_args)
      state_dict_pretrained = torch.load(model_type.value, map_location='cpu', weights_only=False) # weights_only=False to avoid errors
      self.model.load_state_dict(state_dict_pretrained['model'], strict=True)
      self.model = self.model.encoder
      self.tubelet_size = self.model.patch_embed.tubelet_size
      self.img_size = self.model.patch_embed.img_size[0]  # [160, 160]
      self.patch_size = self.model.patch_embed.patch_size[0]  # 16
      self.out_spatial_size = self.img_size // self.patch_size  # 160/16 = 10
      self.embed_dim = self.model.embed_dim
      self.frame_size = 16  # Default frame size
      self.encoder_blocks = self.model.blocks
      self._apply_freeze_logic(freeze_backbone, unfreeze_layers)
    else:
      raise ValueError(f"Unsupported model type: {model_type}")
    self.adapter_dict = adapter_dict


  def _apply_freeze_logic(self, freeze_backbone: bool, unfreeze_layers: int) -> None:
    """
    Apply parameter freezing/unfreezing logic to self.model.

    Args:
      freeze_backbone: If True, freeze all parameters (unfreeze_layers is ignored).
      unfreeze_layers: Number of last encoder blocks to unfreeze when
                       freeze_backbone=False. 0 means unfreeze all parameters.

    Raises:
      ValueError: If unfreeze_layers exceeds the total number of encoder blocks.
    """
    if freeze_backbone:
      for param in self.model.parameters():
        param.requires_grad = False
      print('VideoBackbone params are frozen (freeze_backbone=True).')
      return

    if unfreeze_layers == 0:
      for param in self.model.parameters():
        param.requires_grad = True
      print('VideoBackbone params are fully trainable (unfreeze_layers=0).')
      return

    # Partial unfreezing: freeze all first, then unfreeze last N blocks
    if self.encoder_blocks is None:
      # VJEPA2 fallback: no block list wired, unfreeze all
      for param in self.model.parameters():
        param.requires_grad = True
      print('VideoBackbone (VJEPA2): no block list found, unfreezing all params.')
      return

    n_blocks = len(self.encoder_blocks)
    if unfreeze_layers > n_blocks:
      raise ValueError(
        f"unfreeze_layers={unfreeze_layers} exceeds total encoder blocks={n_blocks}."
      )

    for param in self.model.parameters():
      param.requires_grad = False

    for block in self.encoder_blocks[-unfreeze_layers:]:
      for param in block.parameters():
        param.requires_grad = True

    print(f'VideoBackbone: last {unfreeze_layers}/{n_blocks} encoder blocks unfrozen.')

  def load_pretrained_weights(self):
    if self.model_type in [MODEL_TYPE.VIDEOMAE_v2_B, MODEL_TYPE.VIDEOMAE_v2_G, MODEL_TYPE.VIDEOMAE_v2_G_unl, MODEL_TYPE.VIDEOMAE_v2_S]:
      if self.model_type == MODEL_TYPE.VIDEOMAE_v2_G_unl:
        # Load pretrained model
        self.model = self._load_model_pretrained(self.model_type)
      else:
        self.model = self._load_model_finetune(self.model_type, remove_head=self.remove_head, use_sdpa=self.use_sdpa)
    # elif self.model_type in [MODEL_TYPE.VJEPA2_G_384]:
    #   # Load JEPA2 model
    #   self.model = self.load_jepa2_weights()[0]
      
  def _load_vjepa2_1_vit_base(self, model_type, img_size=256, num_frames=64):
    """
    Load V-JEPA 2.1 ViT-B encoder from a local .pt checkpoint.

    Args:
      model_type:  ModelTypeEntry whose .value is the local .pt path.
      img_size:    Spatial input size (RoPE handles resolution flexibly).
      num_frames:  Frames per clip consumed by the encoder.

    Returns:
      The encoder nn.Module with pretrained weights loaded (strict=True).
    """
    from vjepa2.app.vjepa_2_1.models import vision_transformer as vit_encoder
    from vjepa2.src.hub.backbones import _clean_backbone_key

    encoder = vit_encoder.vit_base(
      patch_size=16,
      img_size=(img_size, img_size),
      num_frames=num_frames,
      tubelet_size=2,
      use_sdpa=True,
      use_SiLU=False,
      wide_SiLU=True,
      uniform_power=True,
      use_rope=True,
      img_temporal_dim_size=1,
      interpolate_rope=True,
    )
    state_dict = torch.load(model_type.value, map_location='cpu', weights_only=False)
    encoder_state_dict = _clean_backbone_key(state_dict['ema_encoder'])
    encoder.load_state_dict(encoder_state_dict, strict=True)
    return encoder

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
  
  # @torch.no_grad()
  def forward_features(self, x: torch.Tensor, return_embedding: bool = True,return_attn: bool = False) -> torch.Tensor:
    """Extract features from video input. No preprocessing is applied, but the input tensor must be in the format [B,C,T,H,W].
    
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
    pid = os.getpid()
    t_setup = time.perf_counter()
    self.model.to(self.device)
    x = x.to(self.device)
    # Extract features
    if not self.model.training:
      self.model.eval()
    helper.time_profile_dict[f'{pid}_backbone_setup_time'] = helper.time_profile_dict.get(f'{pid}_backbone_setup_time', 0) + (time.perf_counter() - t_setup)

    # Forward pass
    t_encoder = time.perf_counter()
    if 'forward_features' in dir(self.model):
      args = {'x':x, 'return_embedding':return_embedding, 'return_attn':return_attn}
      feat,attn = self.model.forward_features(**args)
    else:
      feat = self.model(x)
      if 'last_hidden_state' in dir(feat):
        feat = feat.last_hidden_state
      attn = None
    if helper.PROFILING_GPU_SYNC and torch.cuda.is_available():
      torch.cuda.synchronize()
    helper.time_profile_dict[f'{pid}_backbone_encoder_time'] = helper.time_profile_dict.get(f'{pid}_backbone_encoder_time', 0) + (time.perf_counter() - t_encoder)

    # Reshape features
    t_reshape = time.perf_counter()
    if return_embedding:
      B = feat.shape[0]  # Batch size

      # Calculate dimensions
      T = int(feat.shape[1] / (self.out_spatial_size ** 2))  # Temporal dimension
      S = int(self.out_spatial_size)  # Spatial dimension

      feat = feat.reshape(B, T, S, S, self.embed_dim)# [B, T, S, S, embed_dim], e.g. [1, 8, 14, 14, 768]
    helper.time_profile_dict[f'{pid}_backbone_reshape_time'] = helper.time_profile_dict.get(f'{pid}_backbone_reshape_time', 0) + (time.perf_counter() - t_reshape)

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
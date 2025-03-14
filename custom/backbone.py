import os
import torch
import requests

from enum import Enum
from pathlib import Path
from tqdm import tqdm

from VideoMAEv2.models.modeling_finetune import (
  vit_small_patch16_224,
  vit_base_patch16_224,
  vit_giant_patch14_224
)
from VideoMAEv2.models.modeling_pretrain import pretrain_videomae_giant_patch14_224
from transformers import ViTFeatureExtractor, ViTModel
from custom.helper import MODEL_TYPE

class BackboneBase:
  """Base class for feature extraction backbones."""
  
  def __init__(self):
    """Initialize the backbone base class."""
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
    raise NotImplementedError("Subclasses must implement forward_features")


class VideoBackbone(BackboneBase):
  """Video feature extraction backbone using VideoMAEv2."""
  
  def __init__(
    self, 
    model_type: Enum,
    remove_head: bool = True,
    download_if_unavailable: bool = False
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
    assert model_type in MODEL_TYPE, f"Model type must be one of {list(MODEL_TYPE)}."
    
    # Check if model exists or needs downloading
    model_path = Path(model_type.value)
    if not model_path.exists():
      if download_if_unavailable:
        print(f"Model not found at {model_path}. Downloading...")
        self._download_model(model_type)
      else:
        raise FileNotFoundError(
          f"Model not found at {model_path}. "
          )
    
    # Initialize model based on type
    if model_type == MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e:
      raise NotImplementedError(
        "This version of the model doesn't have forward_features function. "
        "Use one of the fine-tuned models instead."
      )
    else:
      self.model = self._load_model_finetune(model_type, remove_head=remove_head)
        
      # Cache important model parameters for efficient access
      self.tubelet_size = self.model.patch_embed.tubelet_size
      self.img_size = self.model.patch_embed.img_size[0]  # [224, 224]
      self.patch_size = self.model.patch_embed.patch_size[0]  # 16
      self.out_spatial_size = self.img_size // self.patch_size  # 224/16 = 14
      self.embed_dim = self.model.embed_dim
      self.frame_size = 16  # Default frame size
        
    self.model_type = model_type
  
  def _download_model(self, model_type: Enum) -> None:
    """Download model weights from HuggingFace repository.
    
    Args:
      model_type: Type of model to download
    """
    url = "https://huggingface.co/OpenGVLab/VideoMAE2/resolve/main/"
    model_name = os.path.split(model_type.value)[-1]
    
    # Determine correct URL based on model type
    if model_type in (MODEL_TYPE.VIDEOMAE_v2_S, MODEL_TYPE.VIDEOMAE_v2_B):
      url += f"distill/{model_name}"
    else:
      url += f"mae-g/{model_name}"
        
    print(f"Downloading model from: {url}")
    
    # Create directory if it doesn't exist
    weights_dir = Path('VideoMAEv2/pretrained')
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / model_name
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise exception for HTTP errors
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {model_name}") as pbar:
      with open(weights_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
          if chunk:
            f.write(chunk)
            pbar.update(len(chunk))
    
    print(f"Model downloaded successfully to {weights_path}")
  
  def _load_model_pretrained(self, model_type: Enum):
    """Load a pretrained model.
    
    Args:
      model_type: Type of model to load
        
    Returns:
      Loaded model
    """
    if model_type == MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e:
      # Load weights
      checkpoint = torch.load(model_type.value, map_location=self.device)
      weights = checkpoint['model']
      
      # Filter out decoder weights, keeping only encoder weights
      new_weights = {
        key.replace('encoder.', ''): value 
        for key, value in weights.items() 
        if 'encoder' in key and 'decoder' not in key
      }
      
      # Create model and load weights
      model = pretrain_videomae_giant_patch14_224(pretrained=False)
      model = model.encoder
      model.load_state_dict(new_weights)
      
      return model
    
    return None
  
  def _load_model_finetune(self, model_type: Enum, remove_head: bool = True):
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
      MODEL_TYPE.VIDEOMAE_v2_G_pt_1200e_K710_it_HMDB51_ft: (vit_giant_patch14_224, {'num_classes': 51})
    }
    
    if model_type not in model_map:
      raise ValueError(f"Unsupported model type for fine-tuning: {model_type}")
    
    # Get model constructor and kwargs
    model_fn, kwargs = model_map[model_type]
    
    # Create model instance
    model = model_fn(pretrained=False, **kwargs)
    
    # Load weights
    checkpoint = torch.load(model_type.value, map_location=self.device)
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
  def forward_features(self, x: torch.Tensor, return_embedding: bool = True) -> torch.Tensor:
    """Extract features from video input.
    
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
      feat = self.model.forward_features(x, return_embedding=return_embedding)
    
    # Process and reshape features if needed
    if return_embedding:
      B = feat.shape[0]  # Batch size
      
      # Calculate dimensions
      T = int(feat.shape[1] / (self.out_spatial_size ** 2))  # Temporal dimension
      S = int(self.out_spatial_size)  # Spatial dimension
      
      feat = feat.reshape(B, T, S, S, self.embed_dim)# [B, T, S, S, embed_dim], e.g. [1, 8, 14, 14, 768]
    feat = feat.to('cpu')
    # self.model.to('cpu') # Only for local testing
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
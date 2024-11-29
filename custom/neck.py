from enum import Enum 
import os
import torch
from custom.helper import EMBEDDING_REDUCTION, CLIPS_REDUCTION


class neck:
  def __init__(self, embedding_reduction, clips_reduction):
    self.type_embedding_redcution = embedding_reduction
    self.type_clips_reduction = clips_reduction
    
    if embedding_reduction.value is None:
      self.embedding_reduction = None
    else:
      self.embedding_reduction = self.embedding_reduction_mean
      self.dim_embed_reduction = embedding_reduction.value
    
    if clips_reduction.value is None:
      self.clips_reduction = None
      
    else:
      self.clips_reduction = self.clips_reduction_mean
      self.dim_clips_reduction = clips_reduction.value
  
  def embedding_reduction_mean(self, embeddings):
    assert type(embeddings) == torch.Tensor, "Embeddings must be a tensor."
    return embeddings.mean(dim=self.dim_embed_reduction, keepdim=True)
  
  def clips_reduction_mean(self, clips):
    assert type(clips) == torch.Tensor, "Clips must be a tensor."
    return clips.mean(dim=self.dim_clips_reduction, keepdim=True)
  
    
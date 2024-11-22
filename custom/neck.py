from enum import Enum 
import os
import torch
import torch.nn as nn
from custom.helper import EMBEDDING_REDUCTION, CLIPS_REDUCTION


class neck:
  def __init__(self, embedding_reduction, clips_reduction):
    if embedding_reduction is not  None:
      assert embedding_reduction in EMBEDDING_REDUCTION, f"Embedding reduction must be one of {EMBEDDING_REDUCTION}."
    if clips_reduction is not None:
      assert clips_reduction in CLIPS_REDUCTION, f"Clips reduction must be one of {CLIPS_REDUCTION}."
    
    if embedding_reduction is None:
      self.embedding_reduction = None
    else:
      self.embedding_reduction = self.embedding_reduction_mean
      self.dim_embed_reduction = embedding_reduction.value
    
    if clips_reduction is None:
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

class GRU(nn.Module):
  def __init__(self,input_size,hidden_size,num_layers,output_size):
    super(GRU, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.gru = nn.GRU(input_size, hidden_size, num_layers)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, x):
    # x.shape = (seq_length, batch_size, input_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
    out, _ = self.gru(x, h0)
    out = self.fc(out[:, -1, :]) # get the last time step output
    return out
  
    
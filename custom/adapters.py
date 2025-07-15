import torch
import torch.nn as nn
import math
from torch import Tensor

## Adapter and PlainAdapter copied from paper "End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames" (https://arxiv.org/abs/2311.17241)
## Code -> https://github.com/sming256/OpenTAD/blob/main/opentad/models/backbones/vit_adapter.py#L19

class Adapter(nn.Module):
  def __init__(
      self,
      embed_dims: int,
      mlp_ratio: float = 0.25,
      kernel_size: int = 3,
      dilation: int = 1,
      temporal_size: int = 384,
  ) -> None:
    super().__init__()

    hidden_dims = int(embed_dims * mlp_ratio)

    # temporal depth-wise convolution
    self.temporal_size = temporal_size
    self.dwconv = nn.Conv1d(
        hidden_dims,
        hidden_dims,
        kernel_size=kernel_size,
        stride=1,
        padding=(kernel_size // 2) * dilation,
        dilation=dilation,
        groups=hidden_dims,
    )
    self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
    self.kernel_size = kernel_size
    self.hidden_dims = hidden_dims
    self.dilation = dilation
    self.embed_dims = embed_dims
    
    # adapter projection
    self.down_proj = nn.Linear(embed_dims, hidden_dims)
    self.act = nn.GELU()
    self.up_proj = nn.Linear(hidden_dims, embed_dims)
    self.gamma = nn.Parameter(torch.ones(1))
    
    # Initialize weights
    self.init_weights()
  
  def init_weights(self):
    """Initialize the weights of the adapter."""
    self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / self.kernel_size))
    self.dwconv.bias.data.zero_()
    self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / self.hidden_dims))
    self.conv.bias.data.zero_()
    torch.nn.init.trunc_normal_(self.down_proj.weight, std=0.02)
    torch.nn.init.constant_(self.down_proj.bias, 0)
    torch.nn.init.constant_(self.up_proj.weight, 0)  # the last projection layer is initialized to 0
    torch.nn.init.constant_(self.up_proj.bias, 0)
    self.gamma.data.fill_(1.0)  
  
  def forward(self, x: Tensor, h: int, w: int) -> Tensor:
    inputs = x

    # down and up projection
    x = self.down_proj(x)
    x = self.act(x)

    # temporal depth-wise convolution
    B, N, C = x.shape  # 48, 8*10*10, 384
    attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]
    attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
    attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
    attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
    attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
    attn = attn.reshape(B, N, C)
    x = x + attn

    x = self.up_proj(x)
    return x * self.gamma + inputs


class PlainAdapter(nn.Module):
  def __init__(
    self,
    embed_dims: int,
    mlp_ratio: float = 0.25,
    **kwargs,
    ) -> None:
    super().__init__()

    hidden_dims = int(embed_dims * mlp_ratio)

    # adapter projection
    self.down_proj = nn.Linear(embed_dims, hidden_dims)
    self.act = nn.GELU()
    self.up_proj = nn.Linear(hidden_dims, embed_dims)
    self.gamma = nn.Parameter(torch.ones(1))
    
    # Initialize weights
    self.init_weights()
    
    # torch.nn.init.trunc_normal_(self.down_proj, std=0.02, bias=0)
    # torch.nn.init.constant_(self.up_proj, 0)  # the last projection layer is initialized to 0

  def init_weights(self):
    """Initialize the weights of the adapter."""
    torch.nn.init.trunc_normal_(self.down_proj.weight, std=0.02)
    torch.nn.init.constant_(self.down_proj.bias, 0)
    torch.nn.init.constant_(self.up_proj.weight, 0)  # the last projection layer is initialized to 0
    torch.nn.init.constant_(self.up_proj.bias, 0)
    self.gamma.data.fill_(1.0)
  
  def forward(self, x: Tensor, h: int, w: int) -> Tensor:
    inputs = x
    # down and up projection
    x = self.down_proj(x)
    x = self.act(x)
    x = self.up_proj(x)
    return x * self.gamma + inputs
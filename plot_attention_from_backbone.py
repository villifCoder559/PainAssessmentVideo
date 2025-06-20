import custom.tools as tools
import custom.backbone as backbone
import custom.helper as helper
import custom.dataset as dataset
import torch
import matplotlib.pyplot as plt

  

def plot_attention(attn_tensor, title, saving_path,chunk_idx=0, figsize=(8, 4)):
  """
  attn_tensor: torch.Tensor of shape (num_chunks, N, N)
  title: str, plot title
  chunk_idx: int, which chunk to visualize
  """
  # detach & move to CPU numpy
  mat = attn_tensor[chunk_idx].cpu().numpy()
  
  plt.figure(figsize=figsize)
  plt.imshow(mat, aspect='equal',cmap='viridis')
  plt.colorbar()
  plt.title(f"{title} (chunk {chunk_idx})")
  plt.xlabel("key positions")
  plt.ylabel("query positions")
  plt.tight_layout()
  plt.savefig(saving_path)
    

def compare_attentions(temp_attn, space_attn, chunk_idx=0):
  T_pt = temp_attn.shape[1]
  S2   = space_attn.shape[1]

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  im1 = ax1.imshow(temp_attn[chunk_idx].cpu().numpy(), aspect='equal')
  ax1.set_title(f"Temporal (T_pt={T_pt})")
  ax1.set_xlabel("t′")
  ax1.set_ylabel("t")
  fig.colorbar(im1, ax=ax1)

  im2 = ax2.imshow(space_attn[chunk_idx].cpu().numpy(), aspect='equal')
  ax2.set_title(f"Spatial (S²={S2})")
  ax2.set_xlabel("patch x")
  ax2.set_ylabel("patch y")
  fig.colorbar(im2, ax=ax2)

  plt.tight_layout()
  plt.show()

import torch
import torch.nn.functional as F

def chunk_similarity(y: torch.Tensor, eps: float = 1e-8):
  """
  x: Tensor of shape [*, C]
  chunk_idx: which chunk to compute over
  returns: similarity matrix of shape [T*S*S, T*S*S]
  """

  feats = y.view(-1, y.shape[-1])          
  
  # 3a) For dot-product similarities:
  sim_dot = feats @ feats.t()     # [N, N]
  
  # 3b) For cosine similarities:
  feats_norm = feats / (feats.norm(dim=1, keepdim=True) + eps)
  sim_cos = feats_norm @ feats_norm.t()
  
  return sim_dot, sim_cos

def plot_similarity(cos_sim,saving_path):

  # Suppose sim is your similarity matrix, a PyTorch tensor of shape [N, N]
  # If it’s still a torch.Tensor, first move it to CPU and convert to NumPy:
  
  plt.figure(figsize=(10, 6))
  plt.imshow(cos_sim, aspect='auto')    # imshow uses a default colormap
  plt.colorbar(label='Similarity')
  plt.title('Similarity Heatmap')
  plt.xlabel('Vector Index')
  plt.ylabel('Vector Index')
  plt.tight_layout()
  plt.savefig(saving_path)


def main():
  video_path = "partA/video/video_frontalized_interpolated_mirror/071309_w_21/071309_w_21-PA4-062.mp4"
  chunk_size = 16
  frame_list = tools.get_list_frame_from_video_path(video_path)
  frame_list = torch.tensor(frame_list).permute(0,3,1,2) # shape (B,H,W,C) => (B,3,H,W)
  frame_list = dataset.customDataset.preprocess_images(frame_list)
  frame_list = torch.stack([frame_list[start:start + chunk_size] for start in range(0, len(frame_list)-chunk_size + 1, chunk_size)]) # [B,T=16,C,H,W]
  frame_list = frame_list.permute(0, 2, 1, 3, 4) # ->[B,C,T,H,W]

  video_backbone = backbone.VideoBackbone(
    model_type=helper.MODEL_TYPE.VIDEOMAE_v2_S
  )
  video_backbone.model.eval()
  video_backbone.model.to("cuda")
  attn_list = []
  feats_list = []
  for chunk in frame_list:
    with torch.no_grad():
      chunk = chunk.to("cuda")
      feats, attn = video_backbone.forward_features(chunk.unsqueeze(0), return_attn=True) # [B,C,T,H,W]
      attn_list.append(attn)
      feats_list.append(feats.detach().cpu())
  attn_list = torch.stack(attn_list).squeeze(dim=2) # [chunks,network_depth,1,nr_head,seq_len,seq_len], where seq_len = S*S*T
  feats_list = torch.cat(feats_list) # [chunks,T,S,S,C]
  if True:
    chunk_idx = 0
    _,cos_dot = chunk_similarity(feats_list[chunk_idx,0,:,:,:])
    plot_similarity(cos_dot,f"similarity_chunk_{chunk_idx}.png")
    
  else:
    attn_shape = attn_list.shape
    attn_list = attn_list.reshape(attn_shape[0],                # chunks
                                  attn_shape[1],                # depth
                                  attn_shape[2],                # nr_head
                                  chunk_size//video_backbone.tubelet_size,  # T
                                  video_backbone.out_spatial_size**2, # S*S
                                  chunk_size//video_backbone.tubelet_size,  # T
                                  video_backbone.out_spatial_size**2) # S*S
    attn_list = torch.mean(attn_list,dim=(1,2)) # mean in depth and num_head
    print(attn_list.shape)
    temporal_attention = torch.mean(attn_list,dim=(2,4))
    space_attention = torch.mean(attn_list,dim=(1,3))
    plot_attention(temporal_attention, "Temporal Attention","temporal_attention.png", chunk_idx=0, figsize=(6, 5))
    plot_attention(space_attention,    "Spatial Attention", "spatial_attention.png", chunk_idx=0, figsize=(5, 5))


if __name__ == "__main__":
  main()  
  
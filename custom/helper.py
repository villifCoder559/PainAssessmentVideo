from enum import Enum
import os
class SAMPLE_FRAME_STRATEGY(Enum):
  UNIFORM = 'uniform'
  SLIDING_WINDOW = 'sliding_window'
  CENTRAL_SAMPLING = 'central_sampling'
  RANDOM_SAMPLING = 'random_sampling' 
  
class MODEL_TYPE(Enum):
  VIDEOMAE_v2_S = os.path.join('VideoMAEv2','pretrained',"vit_s_k710_dl_from_giant.pth")
  VIDEOMAE_v2_B = os.path.join('VideoMAEv2','pretrained',"vit_b_k710_dl_from_giant.pth")
  VIDEOMAE_v2_G_pt_1200e = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e.pth")
  VIDEOMAE_v2_G_pt_1200e_K710_it_HMDB51_ft = os.path.join('VideoMAEv2','pretrained',"vit_g_hybrid_pt_1200e_k710_it_hmdb51_ft.pth")
  
class EMBEDDING_REDUCTION(Enum):  
  # [B,t,p,p,emb] -> [B,1,p,p,emb] ex: [3,8,14,14,768] -> [3,1,14,14,768]
  MEAN_TEMPORAL = (1) 
  # [B,t,p,p,emb] -> [B,t,1,1,emb] ex: [3,8,14,14,768] -> [3,8,1,1,768]
  MEAN_SPATIAL = (2,3) 
  # [B,t,p,p,emb] -> [B,1,1,1,emb] ex: [3,8,14,14,768] -> [3,1,1,1,768]
  MEAN_TEMPORAL_SPATIAL = (1,2,3)
  NONE = None
  
class CLIPS_REDUCTION(Enum): 
  # [B,t,p,p,emb] -> [1,t,p,p,emb] ex: [3,8,14,14,768] -> [1,8,14,14,768]
  MEAN = (0)
  NONE = None
  # MAX = "max"

class HEAD(Enum):
  SVR = 'SVR'
  GRU = 'GRU'


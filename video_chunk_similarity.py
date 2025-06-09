import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import custom.tools as tools
import numpy as np
import torch
import argparse
from torchmetrics.functional import pairwise_cosine_similarity

def plot_similarity_matrices(data_dict, output_dir='output'):
  """
  Given a data_dict with keys:
    - 'features': Tensor of shape [B, T, C, C, 2]
    - 'list_sample_id': List of length B with video IDs (strings)

  For each video b in batch, flatten its features to shape [N, 2] where N = T*C*C,
  compute a cosine similarity matrix of shape [N, N], and plot it as a heatmap.

  Args:
    data_dict (dict): Input dictionary.
    output_dir (str): Directory to save plots.
  """
  # Create output directory
  os.makedirs(output_dir, exist_ok=True)

  features = data_dict['features']  # expected torch.Tensor or numpy array
  list_sample_id = data_dict['list_sample_id']

  # Convert to torch tensor if needed
  if not torch.is_tensor(features):
    features = torch.tensor(features)

  channels = features.shape[-1]  # should be 2
  # assert channels == 2, f"Expected channel dim=2, got {channels}"
  unique_video_sample_ids = np.unique(list_sample_id)
  # cos_similarity = torch.nn.CosineSimilarity(dim=-1)
  
  for sample_id in unique_video_sample_ids: 
    mask_sample_id = (sample_id == list_sample_id)
    # Extract and flatten: [T, C1, C2, 2] -> [N, 2]
    feat = features[mask_sample_id]  # shape [T, C1, C2, 2]
    
    feat_flat = feat.reshape(-1, channels)  # [N, 2]

    sim_matrix = pairwise_cosine_similarity(feat_flat,feat_flat)  # Compute cosine similarity


    # Plot
    plt.figure(figsize=(8, 6))
    plt.title(f'Cosine Similarity for video {sample_id}')
    im = plt.imshow(sim_matrix, aspect='auto')
    plt.colorbar(im)
    plt.xlabel('Chunk index')
    plt.ylabel('Chunk index')

    # Save
    out_path = os.path.join(output_dir, f"cosine_similarity_{sample_id}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved similarity plot for video {sample_id} at {out_path}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Plot cosine similarity matrices for video chunk embeddings.')
  parser.add_argument('--dict_path', type=str, required=True, help="Path containing the dict")
  parser.add_argument('--output_dir', required=True,type=str, default='output',help='Directory to save the plots')
  args = parser.parse_args()

  # Load input
  data = tools.load_dict_data(args.dict_path)
  plot_similarity_matrices(data, args.output_dir)
 
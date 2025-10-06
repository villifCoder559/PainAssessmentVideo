import os
import tqdm
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import custom.tools as tools
import argparse

parser = argparse.ArgumentParser() 
parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder containing quadrant features')
parser.add_argument('--combined_root_folder', type=str, default=None, help='Path to the output folder for combined features')
root_folder = parser.parse_args().root_folder
combined_root_folder = parser.parse_args().combined_root_folder
if combined_root_folder is None:
  if root_folder.endswith('/'):
    root_folder = root_folder[:-1]
  combined_root_folder = root_folder + "_combined"

print(f"Loading features from {root_folder}")
print(f"Saving combined features to {combined_root_folder}")

if root_folder == combined_root_folder:
  raise ValueError("Root folder and combined root folder must be different")

quadrants_id = ['upper_left', 'upper_right', 'bottom_left', 'bottom_right']
csv_path = "partA/starting_point/samples.csv"

df = pd.read_csv(csv_path, sep='\t')
list_sample_name = df['sample_name'].to_list()
list_subject_name = df['subject_name'].to_list()
zip_subject_sample = list(zip(list_subject_name, list_sample_name))  # materialize list for executor


def process_sample(name, sample):
  feats_quadrant_path = {
    q: os.path.join(root_folder, q, name, f"{sample}${q}.safetensors")
    for q in quadrants_id
  }

  if not all(os.path.exists(p) for p in feats_quadrant_path.values()):
    raise FileNotFoundError(f"Features not found for sample {sample} in subject {name} in {q} quadrant")

  dict_feats = {q: tools.load_dict_data(feats_quadrant_path[q]) for q in quadrants_id}
  keys = list(dict_feats[quadrants_id[0]].keys())
  combined_feats = {}

  # compute quadrant tensor only once
  n_frames = dict_feats[quadrants_id[0]][keys[0]].shape[0]
  quadrant_tensor = np.repeat(quadrants_id, repeats=n_frames, axis=0)

  for k in keys:
    first = dict_feats[quadrants_id[0]][k]
    if isinstance(first, np.ndarray):
      combined_feats[k] = np.concatenate([dict_feats[q][k] for q in quadrants_id], axis=0)
    elif isinstance(first, torch.Tensor):
      combined_feats[k] = torch.cat([dict_feats[q][k] for q in quadrants_id], dim=0)
    else:
      raise ValueError(f"Unsupported data type for key {k}")

  combined_feats['quadrant'] = quadrant_tensor

  os.makedirs(os.path.join(combined_root_folder, name), exist_ok=True)
  tools.save_dict_data(combined_feats,
                       os.path.join(combined_root_folder, name, sample),
                       save_as_safetensors=True)


# run in parallel
with ProcessPoolExecutor(max_workers=8) as executor:
  futures = [
    executor.submit(process_sample, name, sample)
    for name, sample in zip_subject_sample
  ]
  for _ in tqdm.tqdm(futures, total=len(futures), desc="Combining quadrant features..."):
    _.result()  # propagate exceptions

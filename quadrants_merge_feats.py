import os
import tqdm
import torch
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import custom.tools as tools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder containing quadrant features')
parser.add_argument('--combined_root_folder', type=str, default=None, help='Path to the output folder for combined features')
parser.add_argument('--csv_path', type=str, default=None, help='Path to the CSV file with sample and subject names')
parser.add_argument('--nr_workers', type=int, default=4, help='Number of parallel workers to use')
parser.add_argument('--from_', type=int, default=None, help='Starting index of samples to process. Inclusive')
parser.add_argument('--to_', type=int, default=None, help='Ending index of samples to process. Exclusive')
args = parser.parse_args()

def process_sample(name, sample, root_folder, combined_root_folder, quadrants_id):
  feats_quadrant_path = {
    q: os.path.join(root_folder, q, name, f"{sample}${q}.safetensors")
    for q in quadrants_id
  }

  if not all(os.path.exists(p) for p in feats_quadrant_path.values()):
    raise FileNotFoundError(f"Features not found in path: {feats_quadrant_path}")

  dict_feats = {q: tools.load_dict_data(feats_quadrant_path[q]) for q in quadrants_id}
  keys = list(dict_feats[quadrants_id[0]].keys())
  combined_feats = {}

  n_frames = dict_feats[quadrants_id[0]][keys[0]].shape[0]
  quadrant_tensor = np.tile(np.array(quadrants_id), n_frames)

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
  tools.save_dict_data(
    combined_feats,
    os.path.join(combined_root_folder, name, sample),
    save_as_safetensors=True
  )
  return True

def main():
  csv_path = args.csv_path
  if args.csv_path is None:
    csv_path = "partA/starting_point/samples.csv"

  root_folder = args.root_folder
  combined_root_folder = args.combined_root_folder
  if combined_root_folder is None:
    if root_folder.endswith('/'):
      root_folder = root_folder[:-1]
    combined_root_folder = root_folder + "_combined"

  print(f"Loading features from {root_folder}")
  print(f"Saving combined features to {combined_root_folder}")

  if root_folder == combined_root_folder:
    raise ValueError("Root folder and combined root folder must be different")

  quadrants_id = ['upper_left', 'upper_right', 'bottom_left', 'bottom_right']

  df = pd.read_csv(csv_path, sep='\t', dtype={'subject_name': str, 'sample_name': str})
  if args.from_ is not None and args.to_ is not None:
    df = df.iloc[args.from_:args.to_]
  elif args.from_ is not None:
    df = df.iloc[args.from_:]
  elif args.to_ is not None:
    df = df.iloc[:args.to_]
    
  print(f"Processing {len(df)} samples from CSV: {csv_path}")
  list_sample_name = df['sample_name'].to_list()
  list_subject_name = df['subject_name'].to_list()

  zip_subject_sample = list(zip(list_subject_name, list_sample_name))
  
  with ProcessPoolExecutor(max_workers=args.nr_workers) as executor:
    futures = [
      executor.submit(process_sample, name, sample, root_folder, combined_root_folder, quadrants_id)
      for name, sample in zip_subject_sample
    ]
    for future in tqdm.tqdm(
      as_completed(futures),
      total=len(futures),
      desc="Combining quadrant features...",
    ):
      val = future.result()
      if val is not True:
        print(f"Error processing sample: {val}")

if __name__ == "__main__":
  main()

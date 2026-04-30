#!/usr/bin/env python3
"""
slim_pkl.py — retroactively reduce the size of xattn_embeds pkl files.

Two optimisations are applied:
  1. Strip config_model['results'] (~250 MB of k-fold training history that no
     consumer script reads from these files).
  2. --convert-embeddings: flatten the list-of-batch-tensors in
     video_embeddings['embeddings'] into a single float32 numpy array (small
     additional saving; also makes loading faster).

Usage
-----
  # single file, write <name>_slim.pkl next to original
  python slim_pkl.py --pkl path/to/file_xattn_embeds_123.pkl

  # single file, overwrite in place
  python slim_pkl.py --pkl path/to/file_xattn_embeds_123.pkl --inplace

  # whole directory tree
  python slim_pkl.py --pkl path/to/folder/ [--inplace] [--convert-embeddings]
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import tqdm


def slim_data(data: dict, convert_embeddings: bool = True) -> dict:
  """
  Return a slimmed copy of an xattn_embeds pkl dict.

  Args:
    data:               Dict loaded from an xattn_embeds pkl file.
    convert_embeddings: When True, flatten list-of-batch-tensors in
                        video_embeddings['embeddings'] to a single np.float32
                        array (format version 2).  Already-converted arrays
                        are left untouched.

  Returns:
    New dict with config_model['results'] removed and, optionally, embeddings
    converted to a numpy array.
  """
  slimmed = dict(data)

  # --- 1. Strip config_model['results'] ---
  if isinstance(slimmed.get('config_model'), dict) and 'results' in slimmed['config_model']:
    slimmed['config_model'] = {k: v for k, v in slimmed['config_model'].items() if k != 'results'}

  # --- 2. Optionally convert embeddings ---
  if convert_embeddings and isinstance(slimmed.get('video_embeddings'), dict):
    ve = slimmed['video_embeddings']
    raw = ve.get('embeddings')
    if raw is not None and not isinstance(raw, np.ndarray):
      flat = [desc.cpu().numpy() for batch_list in raw for desc in batch_list]
      slimmed['video_embeddings'] = {
        **ve,
        'embeddings': np.array(flat, dtype=np.float32),
        'pkl_format_version': 2,
      }

  return slimmed


def slim_file(pkl_path: str, inplace: bool, convert_embeddings: bool) -> None:
  """
  Load, slim, and save one pkl file.

  Args:
    pkl_path:           Absolute or relative path to the source pkl file.
    inplace:            When True, overwrite the source file; otherwise write
                        <stem>_slim.pkl in the same directory.
    convert_embeddings: Passed through to slim_data().
  """
  size_before = os.path.getsize(pkl_path) / 1e6
  with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

  slimmed = slim_data(data, convert_embeddings=convert_embeddings)

  if inplace:
    out_path = pkl_path
  else:
    p = Path(pkl_path)
    out_path = str(p.parent / f'{p.stem}_slim{p.suffix}')

  with open(out_path, 'wb') as f:
    pickle.dump(slimmed, f)

  size_after = os.path.getsize(out_path) / 1e6
  print(f'{pkl_path}  {size_before:.1f} MB → {size_after:.1f} MB  saved to {out_path}')


def find_xattn_pkls(root: str):
  """
  Recursively find all xattn_embeds pkl files under root.

  Args:
    root: Directory path to search.

  Returns:
    List of absolute path strings matching '*xattn_embeds_*.pkl'.
  """
  found = []
  for dirpath, _, files in os.walk(root):
    for fname in files:
      if fname.endswith('.pkl') and 'xattn_embeds_' in fname:
        found.append(os.path.join(dirpath, fname))
  return found


def main():
  parser = argparse.ArgumentParser(description="Slim down xattn_embeds pkl files.")
  parser.add_argument('--pkl', required=True,
                      help='Path to a pkl file or a directory to search recursively.')
  parser.add_argument('--inplace', action='store_true',
                      help='Overwrite original file (default: write <name>_slim.pkl).')
  parser.add_argument('--convert-embeddings', action='store_true',
                      help='Also flatten list-of-tensor embeddings to a single numpy array.')
  args = parser.parse_args()

  if os.path.isdir(args.pkl):
    pkl_files = find_xattn_pkls(args.pkl)
    print(f'Found {len(pkl_files)} xattn_embeds pkl files.')
    for p in tqdm.tqdm(pkl_files, desc='Slimming pkl files'):
      slim_file(p, inplace=args.inplace, convert_embeddings=args.convert_embeddings)
  else:
    slim_file(args.pkl, inplace=args.inplace, convert_embeddings=args.convert_embeddings)


if __name__ == '__main__':
  main()

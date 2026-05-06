#!/usr/bin/env python3
"""
Cross-space projection: project embeddings produced by an old model (e.g. UNBC-trained)
into the embedding space of a new model (e.g. BIOVID-trained) via anchor-based
interpolation, then classify the projected embeddings using the new model's final
linear layer.

Pipeline:
  1. Select K anchor samples from the new model domain (BIOVID).
  2. Extract K embeddings with old model (BIOVID features) → old_model_anchors (K, D_old).
  3. Extract K embeddings with new model (BIOVID features) → new_model_anchors (K, D_new).
  4. Extract N embeddings with old model (UNBC features)   → old_model_tensors (N, D_old).
  5. Compute similarity weights (N, K) in old model space.
  6. Project: projected (N, D_new) = weights @ new_model_anchors.
  7. Classify: new_model.head.linear(projected) → logits.
  8. Compute MAE + CCC metrics, save pkl.
"""
import argparse
import copy
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import custom.helper as helper
import custom.tools as tools
from custom.model import Model_Advanced
from log_cross_attention_from_model import clean_csv_from_augmentations

_SEED = 42

# (backbone_key, dataset_key) → pre-extracted features folder path
_FEATURES_MAP = {
  ('DFER',     'UNBC'):   'UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol',
  ('DFER',     'BIOVID'): 'partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol',
  ('VIDEOMAE', 'UNBC'):   'UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol',
  ('VIDEOMAE', 'BIOVID'): 'partA/video/features/VideoMaev2_S/spatial_pooled_features_Biovid_B_last143_stride16_interpol',
}


def _detect_backbone(features_path):
  """
  Detect backbone type from a pre-extracted features folder path.

  Args:
    features_path (str): Path to the features folder.

  Returns:
    str: 'DFER' or 'VIDEOMAE'.
  """
  return 'DFER' if 'DFER' in str(features_path).upper() else 'VIDEOMAE'


def _get_features_path(current_features_path, target_dataset):
  """
  Return the features folder path for the same backbone but a different target dataset.

  Args:
    current_features_path (str): Existing features path (used to infer backbone).
    target_dataset (str): 'UNBC' or 'BIOVID'.

  Returns:
    str: Features folder path for (backbone, target_dataset).
  """
  backbone = _detect_backbone(current_features_path)
  key = (backbone, target_dataset.upper())
  if key not in _FEATURES_MAP:
    raise ValueError(f'No features path in _FEATURES_MAP for backbone={backbone}, dataset={target_dataset}')
  return _FEATURES_MAP[key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_config(model_pth):
  """
  Load k_fold_results.pkl from 4 directory levels above the checkpoint file.

  Args:
    model_pth (str): Absolute path to the .pt/.pth model checkpoint.

  Returns:
    dict: Contents of k_fold_results.pkl (keys: config, model_advanced_params, results).
  """
  config_path = os.path.join(*Path(model_pth).parts[:-4], 'k_fold_results.pkl')
  with open(config_path, 'rb') as f:
    return pickle.load(f)


def _build_model(config_model):
  """
  Instantiate Model_Advanced from a loaded k_fold_results config.

  Args:
    config_model (dict): Loaded k_fold_results.pkl.

  Returns:
    Model_Advanced: Newly constructed model (head weights not yet loaded).
  """
  params = copy.deepcopy(config_model['model_advanced_params'])
  params['head_params']['skip_init_weights'] = True
  return Model_Advanced(**params)


def _resolve_split_csv(model_pth, split_name):
  """
  Convert a split name to its CSV path relative to a model checkpoint.

  Args:
    model_pth  (str): Path to model checkpoint file.
    split_name (str): One of 'train', 'val', 'test'.

  Returns:
    str: Absolute path to the CSV file.
  """
  if split_name == 'test':
    candidate = os.path.join(*Path(model_pth).parts[:-2], 'test.csv')
    if not os.path.exists(candidate):
      candidate = os.path.join(*Path(model_pth).parts[:-1], 'val.csv')
    return candidate
  return os.path.join(*Path(model_pth).parts[:-1], f'{split_name}.csv')


def _resolve_old_model_csvs(model_pth, split_or_all):
  """
  Resolve a split name (or 'all') to a list of old model CSV paths.

  Args:
    model_pth     (str): Path to old model checkpoint file.
    split_or_all  (str): 'train', 'val', 'test', or 'all'.

  Returns:
    List[str]: List of CSV file paths.
  """
  if split_or_all == 'all':
    return [_resolve_split_csv(model_pth, s) for s in ('train', 'val', 'test')]
  return [_resolve_split_csv(model_pth, split_or_all)]


def _extract_embeddings(model, model_pth, csv_path, config_model, features_path_override=None):
  """
  Run model inference and collect per-sample video embeddings via LOG_VIDEO_EMBEDDINGS.

  The csv_path must already be cleaned of augmented samples before calling this function.

  Args:
    model               (Model_Advanced): Instantiated model.
    model_pth           (str): Path to checkpoint weights (.pt/.pth).
    csv_path            (str): CSV file to run inference on (augmentation-free).
    config_model        (dict): Loaded k_fold_results.pkl for this model.
    features_path_override (str | None): If given, temporarily replaces
      model.path_to_extracted_features and model.dataset_type before inference
      and restores them afterward.

  Returns:
    dict with keys:
      'embeddings'  (np.ndarray): Shape (N, D), dtype float32.
      'labels'      (np.ndarray): Shape (N,),   dtype float32.
      'sample_ids'  (np.ndarray): Shape (N,),   dtype int64.
      'predictions' (np.ndarray): Shape (N,),   dtype float32.
  """
  original_path = model.path_to_extracted_features
  original_dtype = model.dataset_type
  if features_path_override is not None:
    model.path_to_extracted_features = features_path_override
    model.dataset_type = tools.get_dataset_type(features_path_override)

  # set_step_shift must match the active features path so augmentation filtering is correct
  helper.set_step_shift(model.path_to_extracted_features)

  # Clean the CSV once more (idempotent for already-clean CSVs, writes a _cleaned.csv copy)
  clean_csv = clean_csv_from_augmentations(csv_path)

  helper.init_log_video_embeddings()
  helper.LOG_VIDEO_EMBEDDINGS['enable'] = True
  helper.LOG_HISTORY_SAMPLE = True

  cfg = config_model['config']
  test_args = {
    'path_model_weights': model_pth,
    'state_dict': None,
    'csv_path': clean_csv,
    'criterion': cfg['criterion'],
    'is_test': True,
    'concatenate_temporal': cfg['concatenate_temp_dim'],
    'concatenate_quadrants': cfg['concatenate_quadrants'],
    'CCC_loss': cfg['CCC_loss'],
  }
  extra_kwargs = {k: v for k, v in cfg.items() if k not in test_args}
  extra_kwargs['split_chunks'] = 0

  model.test_pretrained_model(**test_args, **extra_kwargs)

  raw = helper.LOG_VIDEO_EMBEDDINGS
  # raw['embeddings'] is a list of (B, D) tensors, one per batch
  embeddings = np.concatenate([b.numpy() for b in raw['embeddings']], axis=0).astype(np.float32)

  result = {
    'embeddings':  embeddings,
    'labels':      np.array(raw['labels'],     dtype=np.float32),
    'sample_ids':  np.array(raw['sample_ids'], dtype=np.int64),
    'predictions': np.array(raw['predictions'], dtype=np.float32),
  }

  model.path_to_extracted_features = original_path
  model.dataset_type = original_dtype
  return result


def _align_by_sample_id(reference, to_align):
  """
  Reorder to_align so its rows match the sample_id order of reference.

  Args:
    reference (dict): Dict with 'sample_ids' key defining the target order.
    to_align  (dict): Dict whose numpy arrays will be reordered.

  Returns:
    dict: to_align with arrays reordered to match reference['sample_ids'].

  Raises:
    AssertionError: If any sample_id in reference is absent from to_align.
  """
  sid_to_idx = {sid: i for i, sid in enumerate(to_align['sample_ids'])}
  missing = set(reference['sample_ids'].tolist()) - set(to_align['sample_ids'].tolist())
  assert not missing, (
    f'{len(missing)} anchor sample_ids not found in the aligned dict: {missing}'
  )
  order = np.array([sid_to_idx[int(sid)] for sid in reference['sample_ids']])
  return {k: v[order] if isinstance(v, np.ndarray) else v for k, v in to_align.items()}


def _softmax(x):
  """
  Numerically stable row-wise softmax.

  Args:
    x (np.ndarray): Shape (N, K).

  Returns:
    np.ndarray: Shape (N, K), rows sum to 1.
  """
  shifted = x - x.max(axis=1, keepdims=True)
  exp_x = np.exp(shifted)
  return exp_x / exp_x.sum(axis=1, keepdims=True)


def _compute_weights(z, anchors, sim_type, method, temperature, sigma):
  """
  Compute interpolation weights between query embeddings and anchor embeddings.

  Args:
    z        (np.ndarray): Query embeddings,  shape (N, D).
    anchors  (np.ndarray): Anchor embeddings, shape (K, D).
    sim_type (str): Similarity/distance metric — 'cos', 'l1', 'l2', or 'l_inf'.
    method   (str): Weight computation — 'softmax' or 'rbf'.
    temperature (float): Temperature for softmax weighting (ignored for rbf).
    sigma    (float): Standard deviation for RBF kernel: exp(-d²/(2σ²)).

  Returns:
    np.ndarray: Weights, shape (N, K), rows sum to 1.
  """
  if sim_type == 'cos':
    # Normalize embeddings
    z_n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    a_n = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    # Compute cosine similarity
    sim = z_n @ a_n.T                         # (N, K), range [-1, 1]
    if method == 'softmax':
      scores = sim / temperature
    else:
      d = 1.0 - sim                           # cosine distance in [0, 2]
      scores = -(d ** 2) / (2.0 * sigma ** 2)
  else:
    from scipy.spatial.distance import cdist
    if sim_type == 'l_inf':
      d = cdist(z, anchors, metric='chebyshev')
    else:
      p_map = {'l1': 1, 'l2': 2}
      d = cdist(z, anchors, metric='minkowski', p=p_map[sim_type])
    if method == 'softmax':
      scores = -d / temperature               # negate: smaller distance → higher score
    else:
      scores = -(d ** 2) / (2.0 * sigma ** 2)

  return _softmax(scores)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def cross_space_projection(args):
  """
  Full cross-space projection pipeline.

  Args:
    args (argparse.Namespace): Parsed CLI arguments with fields:
      new_model_pth, old_model_pth, num_anchors, anchor_selection_type,
      csv_anchor_selection, old_model_csv, interpolation_similarity,
      weighting_method, temperature, rbf_sigma.

  Returns:
    str: Path to the saved results .pkl file.
  """
  np.random.seed(_SEED)
  random.seed(_SEED)

  uid = int(time.time())
  out_dir = os.path.join(os.getcwd(), 'Cross_projection', f'cross_space_projection_{uid}')
  os.makedirs(out_dir, exist_ok=True)
  print(f'[cross_space_projection] Output: {out_dir}')

  # --- Step 1: Load configs and build models ---
  print('Loading configs...')
  old_config = _load_config(args.old_model_pth)
  new_config = _load_config(args.new_model_pth)
  old_mt = old_config['model_advanced_params']['model_type']
  new_mt = new_config['model_advanced_params']['model_type']
  old_features_path = old_config['model_advanced_params']['features_folder_saving_path']
  new_features_path = new_config['model_advanced_params']['features_folder_saving_path']
  print(f'  old model type: {old_mt.name}  features: {old_features_path}')
  print(f'  new model type: {new_mt.name}  features: {new_features_path}')

  old_model = _build_model(old_config)
  new_model = _build_model(new_config)

  # --- Step 2: Select anchors from new model domain ---
  raw_anchor_csv = _resolve_split_csv(args.new_model_pth, args.csv_anchor_selection)
  assert os.path.isfile(raw_anchor_csv), f'Anchor CSV not found: {raw_anchor_csv}'

  helper.set_step_shift(new_features_path)
  clean_anchor_csv = clean_csv_from_augmentations(raw_anchor_csv)
  df_anchors_full = pd.read_csv(clean_anchor_csv, sep='\t', dtype={'sample_name': str})
  assert len(df_anchors_full) >= args.num_anchors, (
    f'Requested {args.num_anchors} anchors but only {len(df_anchors_full)} clean samples available.'
  )
  df_anchors = df_anchors_full.sample(n=args.num_anchors, random_state=_SEED).reset_index(drop=True)
  anchors_csv_path = os.path.join(out_dir, 'anchors.csv')
  df_anchors.to_csv(anchors_csv_path, index=False, sep='\t')
  print(f'Anchors ({args.num_anchors}) saved to {anchors_csv_path}')

  # --- Step 3: Build old model projection dataset ---
  raw_old_csvs = _resolve_old_model_csvs(args.old_model_pth, args.old_model_csv)
  helper.set_step_shift(old_features_path)
  dfs = []
  for p in raw_old_csvs:
    assert os.path.isfile(p), f'Old model CSV not found: {p}'
    clean_p = clean_csv_from_augmentations(p)
    dfs.append(pd.read_csv(clean_p, sep='\t', dtype={'sample_name': str}))
  df_old = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='sample_id')
  old_tensors_csv_path = os.path.join(out_dir, 'old_tensors.csv')
  df_old.to_csv(old_tensors_csv_path, index=False, sep='\t')
  print(f'Old model dataset ({len(df_old)} samples) saved to {old_tensors_csv_path}')

  # --- Step 4: Extract embeddings ---

  # 4.1 old model on BIOVID anchors: use same backbone but BIOVID feature folder
  # (old model was trained on VideoMAEv2-S; its head cannot process DFER features)
  biovid_features_for_old_model = _get_features_path(old_features_path, 'BIOVID')
  print(f'Extracting old_model_anchors with features override → {biovid_features_for_old_model}')
  old_model_anchors = _extract_embeddings(
    old_model, args.old_model_pth, anchors_csv_path, old_config,
    features_path_override=biovid_features_for_old_model,
  )
  print(f'  old_model_anchors: {old_model_anchors["embeddings"].shape}')

  # 4.2 new model on BIOVID anchors (no override — new_model already uses BIOVID features)
  print('Extracting new_model_anchors...')
  new_model_anchors = _extract_embeddings(
    new_model, args.new_model_pth, anchors_csv_path, new_config,
  )
  print(f'  new_model_anchors: {new_model_anchors["embeddings"].shape}')

  # 4.3 old model on old domain (UNBC) samples
  print('Extracting old_model_tensors...')
  old_model_tensors = _extract_embeddings(
    old_model, args.old_model_pth, old_tensors_csv_path, old_config,
  )
  print(f'  old_model_tensors: {old_model_tensors["embeddings"].shape}')

  # --- Step 5: Align anchor embeddings by sample_id ---
  new_model_anchors_aligned = _align_by_sample_id(old_model_anchors, new_model_anchors)

  # --- Step 6: Compute interpolation weights (N, K) in old model space ---
  print('Computing interpolation weights...')
  weights = _compute_weights(
    z=old_model_tensors['embeddings'],
    anchors=old_model_anchors['embeddings'],
    sim_type=args.interpolation_similarity,
    method=args.weighting_method,
    temperature=args.temperature,
    sigma=args.rbf_sigma,
  )
  print(f'  weights: {weights.shape}')

  # --- Step 7: Project into new model space (N, D_new) ---
  projected = weights @ new_model_anchors_aligned['embeddings'].astype(np.float32)
  # projected = (
  #   weights[:, :, np.newaxis] * new_model_anchors_aligned['embeddings'][np.newaxis, :, :]
  # ).sum(axis=1).astype(np.float32)
  print(f'  projected: {projected.shape}')

  new_model_tensors = {
    **old_model_tensors,
    'embeddings': projected,
    'weights':    weights.astype(np.float32),
  }

  # --- Step 8: Classify with new model's linear layer ---
  normalize_labels = bool(new_config['config'].get('normalize_labels', 0))
  max_label = new_config['config'].get('max_label', None)
  label_denorm = float(max_label) if normalize_labels and max_label else 1.0

  new_model.head.eval()
  device = next(new_model.head.linear.parameters()).device
  with torch.no_grad():
    logits = new_model.head.linear(
      torch.tensor(projected, dtype=torch.float32).to(device)
    ).cpu()
  predictions = (logits.numpy() * label_denorm).astype(np.float32)
  new_model_tensors['logits'] = logits.numpy().astype(np.float32)
  new_model_tensors['predictions'] = predictions

  # --- Metrics ---
  labels_true = old_model_tensors['labels']
  preds_flat = predictions.squeeze(-1) if predictions.ndim > 1 else predictions
  mae = float(np.mean(np.abs(preds_flat - labels_true)))
  ccc = float(tools.concordance_ccc(y_true=labels_true, y_pred=preds_flat))
  metrics = {'mae': mae, 'ccc': ccc}
  print(f'Metrics — MAE: {mae:.4f}  CCC: {ccc:.4f}')

  # --- Step 9: Save outputs ---
  config_logging = {
    'old_model_pth':           args.old_model_pth,
    'new_model_pth':           args.new_model_pth,
    'num_anchors':             args.num_anchors,
    'anchor_selection_type':   args.anchor_selection_type,
    'csv_anchor_selection':    args.csv_anchor_selection,
    'old_model_csv':           args.old_model_csv,
    'interpolation_similarity': args.interpolation_similarity,
    'weighting_method':        args.weighting_method,
    'temperature':             args.temperature,
    'rbf_sigma':               args.rbf_sigma,
    'uid':                     uid,
    'anchors_csv_path':        anchors_csv_path,
    'old_tensors_csv_path':    old_tensors_csv_path,
    'out_dir':                 out_dir,
  }

  dict_res = {
    'config_cross_space_projection': config_logging,
    'old_model_config': {k: v for k, v in old_config.items() if k != 'results'},
    'new_model_config': {k: v for k, v in new_config.items() if k != 'results'},
    'anchors_csv_path':             anchors_csv_path,
    'old_tensors_csv_path':         old_tensors_csv_path,
    'old_model_anchors_embeddings': old_model_anchors,
    'new_model_anchors_embeddings': new_model_anchors_aligned,
    'old_model_tensors':            old_model_tensors, # includes old embeddings, labels, sample_ids, predictions
    'new_model_tensors':            new_model_tensors, # includes projected embeddings, logits, predictions
    'metrics':                      metrics,
  }

  out_pkl = os.path.join(out_dir, f'results_{uid}.pkl')
  pkl_bytes = pickle.dumps(dict_res)
  print(f'Saving pkl ({len(pkl_bytes) / 1e6:.1f} MB) → {out_pkl}')
  with open(out_pkl, 'wb') as f:
    f.write(pkl_bytes)

  config_txt = os.path.join(out_dir, 'config_logging.txt')
  with open(config_txt, 'w') as f:
    for k, v in config_logging.items():
      f.write(f'{k}: {v}\n')

  print(f'Done. All outputs in {out_dir}')
  return out_pkl


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='Project old model embeddings into new model space via anchor interpolation.'
  )
  parser.add_argument('--new_model_pth', type=str, required=True,
                      help='Path to new model checkpoint (.pt/.pth)')
  parser.add_argument('--old_model_pth', type=str, required=True,
                      help='Path to old model checkpoint (.pt/.pth)')
  parser.add_argument('--num_anchors', type=int, required=True,
                      help='Number of anchor samples to select from new model domain')
  parser.add_argument('--anchor_selection_type', type=str, default='random',
                      help='Anchor selection strategy (currently only "random")')
  parser.add_argument('--csv_anchor_selection', type=str, required=True,
                      help='Split to select anchors from (train/val/test) — new model domain')
  parser.add_argument('--old_model_csv', type=str, required=True,
                      help='Split to project (train/val/test/all) — old model domain')
  parser.add_argument('--interpolation_similarity', type=str, default='cos',
                      choices=['cos', 'l1', 'l2', 'l_inf'],
                      help='Similarity/distance metric for weight computation')
  parser.add_argument('--weighting_method', type=str, required=True,
                      choices=['softmax', 'rbf'],
                      help='Method to convert similarities to interpolation weights')
  parser.add_argument('--temperature', type=float, default=1.0,
                      help='Temperature for softmax weighting (ignored for rbf)')
  parser.add_argument('--rbf_sigma', type=float, default=1.0,
                      help='Sigma for RBF kernel: exp(-d²/(2σ²)) (ignored for softmax)')
  cross_space_projection(parser.parse_args())

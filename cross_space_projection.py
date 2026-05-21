#!/usr/bin/env python3
"""
Cross-space projection: project embeddings produced by an old model into the embedding
space of a new model via anchor-based interpolation, then classify the projected
embeddings using the new model's final linear layer.

Datasets and backbones are auto-detected from each model's features folder path,
so any dataset pair supported by _FEATURES_MAP works (e.g. UNBC→BIOVID, BIOVID→UNBC,
AgeDB-split-A→AgeDB-split-B, etc.).

Pipeline:
  1. Select K anchor samples from the new model domain.
  2. Extract K embeddings with old model (new-domain features) → old_model_anchors (K, D_old).
  3. Extract K embeddings with new model (new-domain features) → new_model_anchors (K, D_new).
  4. Extract N embeddings with old model (old-domain features) → old_model_tensors (N, D_old).
  5. Compute similarity weights (N, K) in old model space.
  6. Project: projected (N, D_new) = weights @ new_model_anchors.
     (interpolation_similarity='linear': instead train a linear projector on all K
      anchor pairs, validated/tested on the new model's val.csv, then apply it.)
  7. Classify: new_model.head.linear(projected) → logits.
  8. Compute MAE + CCC metrics, save pkl.
"""
import argparse
import copy
import itertools
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import tqdm

import custom.helper as helper
import custom.tools as tools
from custom.model import Model_Advanced
from log_cross_attention_from_model import clean_csv_from_augmentations

_SEED = 42
_ZERO_ANCHOR_KEY    = (None,  0, None)  # anchor_cache sentinel for the num_anchors=0 identity case
_NEG_ONE_ANCHOR_KEY = (None, -1, None)  # anchor_cache sentinel for num_anchors=-1 oracle case


def _linear_projector_tag() -> str:
  """
  Build a short folder-name tag from the active LINEAR_PROJECTOR_CONFIG.

  Returns:
    str: Compact string like 'lr0.0001_bs64_adamw_wd0.0001_ep100_normT_mse_sp70-10-20'.
  """
  cfg = LINEAR_PROJECTOR_CONFIG
  sr  = cfg['split_ratios']
  sp  = f"{int(sr[0]*100)}-{int(sr[1]*100)}-{int(sr[2]*100)}"
  return (
    f"lr{cfg['lr']}"
    f"_bs{cfg['batch_size']}"
    f"_{cfg['optimizer']}"
    f"_wd{cfg['weight_decay']}"
    f"_ep{cfg['epochs']}"
    f"_norm{'T' if cfg['normalize_embeddings'] else 'F'}"
    f"_{cfg['loss']}"
    f"_sp{sp}"
  )

# Hyperparameters for the learned linear projector (interpolation_similarity='linear').
# Edit values here to change the training recipe; no CLI flag is exposed.
LINEAR_PROJECTOR_CONFIG = {
  'lr':                   1e-4,
  'batch_size':           64,
  'optimizer':            'adamw',   # 'adam' | 'adamw' | 'sgd'
  'weight_decay':         1e-4,
  'epochs':               300,
  'normalize_embeddings': True,
  'loss':                 'mse',     # 'mse' | 'mae' | 'cosine'
  # (train, val, test). The projector trains on ALL K anchors, so the train
  # entry is unused; val/test are a subject-disjoint split of the new model's
  # val.csv sized by these fractions of the val.csv row count (test absorbs
  # the remainder so every val.csv row is used).
  'split_ratios':         (0.0, 0.50, 0.50),
  'device':               'cuda',
  'num_workers':          4,
}

# (backbone_key, dataset_key) → pre-extracted features folder path
_FEATURES_MAP = {
  ('DFER',     'UNBC'):   'UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol',
  ('DFER',     'BIOVID'): 'partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol',
  ('DFER',     'AGEDB'):  'AgeDB/features/DFER/all_pooled_features_age',
  ('VIDEOMAE', 'AGEDB'):  'AgeDB/features/VideoMaev2_S/all_pooled_features_age',
  ('VIDEOMAE', 'UNBC'):   'UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol',
  ('VIDEOMAE', 'BIOVID'): 'partA/video/features/VideoMaev2_S/spatial_pooled_features_Biovid_B_last143_stride16_interpol',
}




def _detect_dataset(features_path: str) -> str:
  """
  Return the uppercase dataset name inferred from a features folder path.

  Args:
    features_path (str): Features folder path (e.g. model_advanced_params['features_folder_saving_path']).

  Returns:
    str: One of 'UNBC', 'BIOVID', 'AGEDB', 'CAER', 'MORPH'.

  Raises:
    ValueError: If no known dataset keyword is found in the path.
  """
  p = str(features_path).lower()
  if 'unbc' in p:
    return 'UNBC'
  if 'parta' in p or 'biovid' in p:
    return 'BIOVID'
  if 'agedb' in p:
    return 'AGEDB'
  if 'caer' in p:
    return 'CAER'
  if 'morph' in p:
    return 'MORPH'
  raise ValueError(f'Cannot detect dataset from features path: {features_path!r}')


def _detect_backbone(features_path: str) -> str:
  """
  Return the backbone key inferred from a features folder path.

  Args:
    features_path (str): Features folder path.

  Returns:
    str: One of 'DFER', 'VIDEOMAE', 'VJEPA2'.

  Raises:
    ValueError: If no known backbone keyword is found in the path.
  """
  p = str(features_path).upper()
  if 'DFER' in p:
    return 'DFER'
  if 'VIDEOMAE' in p:
    return 'VIDEOMAE'
  if 'VJEPA' in p or 'JEPA' in p:
    return 'VJEPA2'
  raise ValueError(f'Cannot detect backbone from features path: {features_path!r}')


def _get_features_path(current_features_path, target_dataset):
  """
  Return the features folder path for the same backbone but a different target dataset.

  Args:
    current_features_path (str): Existing features path (used to infer backbone via _detect_backbone).
    target_dataset (str): Target dataset name, e.g. 'UNBC', 'BIOVID', 'AGEDB'.

  Returns:
    str: Features folder path for (backbone, target_dataset).

  Raises:
    ValueError: If the (backbone, target_dataset) combo is absent from _FEATURES_MAP.
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


def _resolve_split_csv_strict(model_pth, split_name):
  """
  Resolve a split name to its CSV path, raising if the file does not exist.

  Unlike _resolve_split_csv, no test→val fallback is applied.

  Args:
    model_pth  (str): Path to model checkpoint file.
    split_name (str): One of 'train', 'val', 'test'.

  Returns:
    str: Absolute path to the CSV file.

  Raises:
    FileNotFoundError: If the expected CSV path does not exist.
  """
  p = os.path.join(*Path(model_pth).parts[:-1], f'{split_name}.csv')
  if not os.path.exists(p):
    raise FileNotFoundError(f'Split CSV not found (strict): {p}')
  return p


def _exc_split_paths(model_pth, exc_split):
  """
  Return the two CSV paths for all splits except exc_split, using strict resolution.

  Args:
    model_pth (str): Path to model checkpoint file.
    exc_split (str): The split to exclude; must be 'train', 'val', or 'test'.

  Returns:
    List[str]: Two CSV paths for the remaining splits, in (train, val, test) order.

  Raises:
    ValueError:           If exc_split is not one of 'train', 'val', 'test'.
    FileNotFoundError:    If either remaining split CSV is missing.
  """
  if exc_split not in ('train', 'val', 'test'):
    raise ValueError(f'exc_ requires train/val/test, got: {exc_split!r}')
  return [_resolve_split_csv_strict(model_pth, s)
          for s in ('train', 'val', 'test') if s != exc_split]


def _resolve_old_model_csvs(model_pth, split_or_all):
  """
  Resolve a split name (or 'all' / 'exc_{set}') to a list of old model CSV paths.

  Args:
    model_pth     (str): Path to old model checkpoint file.
    split_or_all  (str): 'train', 'val', 'test', 'all',
                         or 'exc_train' / 'exc_val' / 'exc_test'.

  Returns:
    List[str]: List of CSV file paths.
  """
  if split_or_all == 'all':
    return [_resolve_split_csv(model_pth, s) for s in ('train', 'val', 'test')]
  if split_or_all.startswith('exc_'):
    return _exc_split_paths(model_pth, split_or_all[4:])
  return [_resolve_split_csv(model_pth, split_or_all)]


def _resolve_anchor_csvs(model_pth, csv_sel):
  """
  Resolve an anchor split spec to a list of CSV paths.

  Args:
    model_pth (str): Path to model checkpoint file (new model).
    csv_sel   (str): 'train', 'val', 'test',
                     or 'exc_train' / 'exc_val' / 'exc_test'.

  Returns:
    List[str]: One or two CSV paths (two for exc_* specs).
  """
  if csv_sel.startswith('exc_'):
    return _exc_split_paths(model_pth, csv_sel[4:])
  return [_resolve_split_csv(model_pth, csv_sel)]


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


def _align_df_to_sample_ids(df, sample_ids, label):
  """
  Reorder a DataFrame's rows to match a given sample_id sequence.

  Args:
    df         (pd.DataFrame): Table with a 'sample_id' column.
    sample_ids (np.ndarray): Target sample_id order.
    label      (str): Human-readable name of df, used in the error message.

  Returns:
    pd.DataFrame: df reindexed to the sample_ids order, with index reset.

  Raises:
    AssertionError: If any sample_id in sample_ids is absent from df.
  """
  sid_col = df['sample_id'].astype(np.int64).to_numpy()
  target  = np.asarray(sample_ids, dtype=np.int64)
  sid_to_row = {int(s): i for i, s in enumerate(sid_col)}
  missing = set(target.tolist()) - set(sid_col.tolist())
  assert not missing, f'sample_ids missing from {label}: {missing}'
  order = np.array([sid_to_row[int(s)] for s in target], dtype=np.int64)
  return df.iloc[order].reset_index(drop=True)


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
# Linear projector helpers (interpolation_similarity='linear')
# ---------------------------------------------------------------------------

def _make_subject_disjoint_subsets(df, n_first, seed):
  """
  Partition a DataFrame into two subject-disjoint subsets, the first ~n_first rows.

  Unique subject_ids are shuffled (seeded) and assigned whole to subset A until it
  reaches n_first rows; every remaining subject goes to subset B. No subject_id appears
  in both subsets and every row lands in exactly one subset. Because subjects are
  assigned whole, the realised size of subset A is approximate.

  Args:
    df      (pd.DataFrame): Rows with a 'subject_id' column.
    n_first (int): Target row count for subset A (>= 1).
    seed    (int): RNG seed for the subject shuffle.

  Returns:
    tuple[np.ndarray, np.ndarray]: (idx_a, idx_b), positional index arrays into df.

  Raises:
    ValueError: If either subset ends up empty.
  """
  rng = np.random.default_rng(seed)
  subjects = df['subject_id'].to_numpy()
  unique_subjects = rng.permutation(np.unique(subjects))
  rows_a, rows_b = [], []
  count_a = 0
  for subj in unique_subjects:
    rows = np.where(subjects == subj)[0]
    if count_a < n_first:
      rows_a.append(rows)
      count_a += len(rows)
    else:
      rows_b.append(rows)
  idx_a = np.concatenate(rows_a) if rows_a else np.array([], dtype=np.int64)
  idx_b = np.concatenate(rows_b) if rows_b else np.array([], dtype=np.int64)
  if len(idx_a) == 0 or len(idx_b) == 0:
    raise ValueError(
      f'subject-disjoint split produced an empty subset '
      f'(|A|={len(idx_a)}, |B|={len(idx_b)}; target n_first={n_first}, '
      f'unique_subjects={len(unique_subjects)})'
    )
  return idx_a, idx_b


def _compute_norm_stats(X):
  """
  Compute per-feature mean and standard deviation.

  Args:
    X (np.ndarray): Shape (N, D).

  Returns:
    tuple[np.ndarray, np.ndarray]: (mean, std), each shape (D,). Std clamped to >=1e-8.
  """
  mean = X.mean(axis=0).astype(np.float32)
  std  = X.std(axis=0).astype(np.float32)
  std  = np.maximum(std, 1e-8)
  return mean, std


def _apply_norm(X, mean, std):
  """
  Apply per-feature normalization to a 2D array.

  Args:
    X    (np.ndarray): Shape (N, D).
    mean (np.ndarray): Shape (D,).
    std  (np.ndarray): Shape (D,), assumed > 0.

  Returns:
    np.ndarray: Shape (N, D), dtype float32.
  """
  return ((X - mean) / std).astype(np.float32)


def _build_projector_optimizer(params, cfg):
  """
  Build an optimizer for the linear projector according to LINEAR_PROJECTOR_CONFIG.

  Args:
    params (iterable): Iterable of nn.Parameter (typically projector.parameters()).
    cfg    (dict): LINEAR_PROJECTOR_CONFIG.

  Returns:
    torch.optim.Optimizer

  Raises:
    ValueError: If cfg['optimizer'] is unknown.
  """
  name = cfg['optimizer'].lower()
  if name == 'adam':
    return torch.optim.Adam(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
  if name == 'adamw':
    return torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
  if name == 'sgd':
    return torch.optim.SGD(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
  raise ValueError(f"Unknown linear-projector optimizer: {cfg['optimizer']!r}")


def _projector_selection_rule(loss_name):
  """
  Map a loss name to the (metric_key, is_better_fn, init_value) used to pick
  the best projector checkpoint by validation performance.

  The rule mirrors the loss the optimizer is minimizing:
    'mse'    → min va_mse
    'mae'    → min va_mae
    'cosine' → max va_cos  (loss is 1 - cos, so best = highest cos)

  Args:
    loss_name (str): 'mse' | 'mae' | 'cosine'.

  Returns:
    tuple[str, Callable[[float, float], bool], float]:
      (metric_key in {'mse','mae','cos'}, comparator new<>best, sentinel init).

  Raises:
    ValueError: If loss_name is unknown.
  """
  if loss_name == 'mse':
    return 'mse', (lambda new, best: new < best),  float('inf')
  if loss_name == 'mae':
    return 'mae', (lambda new, best: new < best),  float('inf')
  if loss_name == 'cosine':
    return 'cos', (lambda new, best: new > best), -float('inf')
  raise ValueError(f'Unknown linear-projector loss for selection: {loss_name!r}')


def _projector_loss_fn(name):
  """
  Resolve a loss name to a callable taking (pred, target) → scalar tensor.

  Args:
    name (str): 'mse' | 'mae' | 'cosine'.

  Returns:
    Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

  Raises:
    ValueError: If name is unknown.
  """
  import torch.nn.functional as F
  if name == 'mse':
    return F.mse_loss
  if name == 'mae':
    return F.l1_loss
  if name == 'cosine':
    return lambda p, t: 1.0 - F.cosine_similarity(p, t, dim=1).mean()
  raise ValueError(f'Unknown linear-projector loss: {name!r}')


def _eval_projector_batch_metrics(pred, target):
  """
  Compute per-batch MSE, MAE and mean cosine similarity.

  Args:
    pred   (torch.Tensor): Shape (B, D_new).
    target (torch.Tensor): Shape (B, D_new).

  Returns:
    tuple[float, float, float]: (mse, mae, mean_cosine_similarity).
  """
  import torch.nn.functional as F
  mse = F.mse_loss(pred, target).item()
  mae = F.l1_loss(pred, target).item()
  cos = F.cosine_similarity(pred, target, dim=1).mean().item()
  return mse, mae, cos


def _accum_projector_batch_metrics(pred, target, sums):
  """
  Accumulate per-batch MSE / MAE / mean-cosine-similarity into GPU scalar tensors
  in-place, without forcing a CUDA synchronization.

  Args:
    pred   (torch.Tensor): Shape (B, D_new).
    target (torch.Tensor): Shape (B, D_new).
    sums   (dict[str, torch.Tensor]): Pre-allocated zero scalar tensors with
      keys 'mse', 'mae', 'cos' on the same device as pred/target. Updated
      in-place.
  """
  import torch.nn.functional as F
  with torch.no_grad():
    sums['mse'] += F.mse_loss(pred, target)
    sums['mae'] += F.l1_loss(pred, target)
    sums['cos'] += F.cosine_similarity(pred, target, dim=1).mean()


def _extract_linear_val_pool(
  old_model, new_model, old_model_pth, new_model_pth,
  old_config, new_config, new_features_path, anchor_domain_features_for_old,
):
  """
  Extract the val.csv embedding pool used to validate/test the linear projector.

  The projector's validation and test sets are drawn from the new model's val.csv
  (the validation split from the new model's own training run), resolved next to
  new_model_pth. Both old- and new-model embeddings are extracted on the new-model
  domain, so the old model uses anchor_domain_features_for_old as a features override.

  Args:
    old_model     (Model_Advanced): Old model instance.
    new_model     (Model_Advanced): New model instance.
    old_model_pth (str): Path to old model checkpoint.
    new_model_pth (str): Path to new model checkpoint.
    old_config    (dict): Old model k_fold_results.pkl.
    new_config    (dict): New model k_fold_results.pkl.
    new_features_path (str): New model features folder (for set_step_shift).
    anchor_domain_features_for_old (str): Old-backbone features for the new-model
      dataset, used as the old model's features override.

  Returns:
    dict: keys 'old' (old-model embeddings dict), 'new' (new-model embeddings dict,
          aligned to 'old' by sample_id), 'df' (val.csv DataFrame), 'csv' (path str).
  """
  val_csv = _resolve_split_csv(new_model_pth, 'val')
  assert os.path.isfile(val_csv), f'New model val.csv not found: {val_csv}'
  helper.set_step_shift(new_features_path)
  df_val = pd.read_csv(
    clean_csv_from_augmentations(val_csv), sep='\t', dtype={'sample_name': str},
  )
  old_emb = _extract_embeddings(
    old_model, old_model_pth, val_csv, old_config,
    features_path_override=anchor_domain_features_for_old,
  )
  new_emb = _extract_embeddings(new_model, new_model_pth, val_csv, new_config)
  new_aligned = _align_by_sample_id(old_emb, new_emb)
  print(f'[linear val pool] new model val.csv ({len(df_val)} rows) → {val_csv}')
  return {'old': old_emb, 'new': new_aligned, 'df': df_val, 'csv': val_csv}


def _train_linear_projector(old_anchors, new_anchors, df_anch, val_pool, projector_dir, anchor_key_tag):
  """
  Train a learned linear projector mapping old anchor embeddings to new anchor embeddings.

  The projector is fit on *all* K anchor pairs selected from --csv_anchor_selection
  (the new-model domain). Its validation and test sets are a subject-disjoint split
  of the new model's val.csv (val_pool): with M = number of val.csv rows, val gets
  ~round(split_ratios[1] * M) rows for model selection and test absorbs the
  remainder, so val and test together cover every val.csv row. Model selection
  mirrors the loss the optimizer is minimizing (see _projector_selection_rule):
  val MSE for loss='mse', val MAE for loss='mae', and val cosine similarity
  (maximized) for loss='cosine'.

  Args:
    old_anchors    (dict): Output of _extract_embeddings on old model — keys
                           'embeddings' (K, D_old), 'sample_ids' (K,), 'labels' (K,).
    new_anchors    (dict): Output of _extract_embeddings on new model, already
                           aligned to old_anchors via _align_by_sample_id —
                           'embeddings' (K, D_new), 'sample_ids' (K,).
    df_anch        (pd.DataFrame): Anchor table written to disk (sample_id, subject_id,
                                   class_id columns required).
    val_pool       (dict): Output of _extract_linear_val_pool — keys 'old', 'new'
                           (embedding dicts on the new model's val.csv) and 'df'.
    projector_dir  (str): Output directory; contents created here:
                          split_training_stage/{train,val,test}.csv,
                          best_projector_{best_epoch}.pt.
    anchor_key_tag (str): Short slug used in log lines.

  Returns:
    dict: keys 'projector', 'norm_stats', 'splits', 'metrics', 'best_epoch',
          'best_val_mse' (legacy — always populated with MSE at best_epoch),
          'best_val_metric' (value of the metric used for selection),
          'best_val_metric_name' ('mse' | 'mae' | 'cos'), 'ckpt_path', 'config'.
  """
  from torch.utils.data import DataLoader, TensorDataset
  cfg = copy.deepcopy(LINEAR_PROJECTOR_CONFIG)
  device = torch.device(cfg['device'])

  os.makedirs(projector_dir, exist_ok=True)
  splits_dir = os.path.join(projector_dir, 'split_training_stage')
  os.makedirs(splits_dir, exist_ok=True)

  # --- Align df_anch / val.csv df row order with their embeddings' sample_ids ---
  df_anch_aligned = _align_df_to_sample_ids(df_anch, old_anchors['sample_ids'], 'anchor df')
  val_df_aligned  = _align_df_to_sample_ids(
    val_pool['df'], val_pool['old']['sample_ids'], 'val.csv df',
  )

  # --- Overlap check: anchors must not share sample_ids with the val.csv pool ---
  anchor_sids = set(old_anchors['sample_ids'].astype(np.int64).tolist())
  val_sids    = set(val_pool['old']['sample_ids'].astype(np.int64).tolist())
  overlap = anchor_sids & val_sids
  if overlap:
    raise ValueError(
      f"[linear_proj:{anchor_key_tag}] {len(overlap)} anchor sample_id(s) also appear "
      f"in the new model's val.csv: {sorted(overlap)[:10]}"
      f"{' ...' if len(overlap) > 10 else ''}. Pick a --csv_anchor_selection split "
      f"disjoint from 'val' (e.g. 'train')."
    )

  # --- Split val.csv subject-disjointly into val / test (test = remainder) ---
  M = len(val_pool['old']['embeddings'])
  n_val = round(cfg['split_ratios'][1] * M)
  if n_val < 1 or n_val >= M:
    raise ValueError(
      f"[linear_proj:{anchor_key_tag}] cannot build a val/test split from val.csv: "
      f"n_val={n_val} (val.csv rows={M}, split_ratios={cfg['split_ratios']})."
    )
  idx_va, idx_te = _make_subject_disjoint_subsets(val_df_aligned, n_val, _SEED)

  # --- Assemble split arrays: train = all anchors; val/test = val.csv subsets ---
  split_arrays = {
    'train': {
      'old':         old_anchors['embeddings'].astype(np.float32),
      'new':         new_anchors['embeddings'].astype(np.float32),
      'sample_ids':  old_anchors['sample_ids'].astype(np.int64),
      'subject_ids': df_anch_aligned['subject_id'].to_numpy(),
      'labels':      old_anchors['labels'].astype(np.float32),
    },
  }
  for name, idx in (('val', idx_va), ('test', idx_te)):
    split_arrays[name] = {
      'old':         val_pool['old']['embeddings'][idx].astype(np.float32),
      'new':         val_pool['new']['embeddings'][idx].astype(np.float32),
      'sample_ids':  val_pool['old']['sample_ids'][idx].astype(np.int64),
      'subject_ids': val_df_aligned['subject_id'].to_numpy()[idx],
      'labels':      val_pool['old']['labels'][idx].astype(np.float32),
    }

  splits_df = {
    'train': df_anch_aligned.reset_index(drop=True),
    'val':   val_df_aligned.iloc[idx_va].reset_index(drop=True),
    'test':  val_df_aligned.iloc[idx_te].reset_index(drop=True),
  }
  split_mode = 'anchors_train/valcsv_eval'

  print(f"  [linear_proj:{anchor_key_tag}] [{split_mode}] "
        f"train={len(split_arrays['train']['old'])} (all anchors)  "
        f"val={len(idx_va)} test={len(idx_te)} (subject-disjoint subsets of val.csv, {M} rows)")
  print(f"  [linear_proj:{anchor_key_tag}] unique subjects per split — "
        f"train={splits_df['train']['subject_id'].nunique()} "
        f"val={splits_df['val']['subject_id'].nunique()} "
        f"test={splits_df['test']['subject_id'].nunique()}")
  s_va = set(splits_df['val']['subject_id'].tolist())
  s_te = set(splits_df['test']['subject_id'].tolist())
  print(f"  [linear_proj:{anchor_key_tag}] val/test subject overlap: {len(s_va & s_te)}")

  # --- Persist split CSVs ---
  split_csv_paths = {}
  for name, sdf in splits_df.items():
    p = os.path.join(splits_dir, f'{name}.csv')
    sdf.to_csv(p, index=False, sep='\t')
    split_csv_paths[name] = p

  # --- Optional normalization (train-only stats) ---
  norm_stats = None
  if cfg['normalize_embeddings']:
    old_mean, old_std = _compute_norm_stats(split_arrays['train']['old'])
    new_mean, new_std = _compute_norm_stats(split_arrays['train']['new'])
    norm_stats = {
      'old_mean': old_mean, 'old_std': old_std,
      'new_mean': new_mean, 'new_std': new_std,
    }
    for name in ('train', 'val', 'test'):
      split_arrays[name]['old_norm'] = _apply_norm(split_arrays[name]['old'], old_mean, old_std)
      split_arrays[name]['new_norm'] = _apply_norm(split_arrays[name]['new'], new_mean, new_std)
  else:
    for name in ('train', 'val', 'test'):
      split_arrays[name]['old_norm'] = split_arrays[name]['old']
      split_arrays[name]['new_norm'] = split_arrays[name]['new']

  d_old = split_arrays['train']['old'].shape[1]
  d_new = split_arrays['train']['new'].shape[1]
  projector = torch.nn.Linear(d_old, d_new).to(device)
  optimizer = _build_projector_optimizer(projector.parameters(), cfg)
  loss_fn = _projector_loss_fn(cfg['loss'])

  # Pre-load every split to `device` once. The training tensors are tiny
  # (≤ a few MB), so the one-shot PCIe transfer is cheap and lets us drop
  # both per-batch .to(device) calls and DataLoader workers (which only pay
  # off for I/O-bound datasets, not in-memory regression).
  def _to_loader(name, shuffle):
    ds = TensorDataset(
      torch.from_numpy(split_arrays[name]['old_norm']).to(device),
      torch.from_numpy(split_arrays[name]['new_norm']).to(device),
    )
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=0, pin_memory=False)

  train_loader = _to_loader('train', shuffle=True)
  val_loader   = _to_loader('val',   shuffle=False)
  test_loader  = _to_loader('test',  shuffle=False)

  metrics = {'train': [], 'val': [], 'test': None}
  sel_key, is_better, sel_init = _projector_selection_rule(cfg['loss'])
  best_val_metric = sel_init
  best_epoch = -1
  best_state_dict = None

  for epoch in tqdm.tqdm(range(1, cfg['epochs'] + 1), desc=f'Linear projector training ({anchor_key_tag})'):
    projector.train()
    tr_sums = {k: torch.zeros((), device=device) for k in ('mse', 'mae', 'cos')}
    tr_n = 0
    for xb, yb in train_loader:
      pred = projector(xb)
      loss = loss_fn(pred, yb)
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      _accum_projector_batch_metrics(pred.detach(), yb, tr_sums)
      tr_n += 1

    projector.eval()
    va_sums = {k: torch.zeros((), device=device) for k in ('mse', 'mae', 'cos')}
    va_n = 0
    with torch.no_grad():
      for xb, yb in val_loader:
        pred = projector(xb)
        _accum_projector_batch_metrics(pred, yb, va_sums)
        va_n += 1

    # Single sync per epoch: stack the three GPU scalars and convert in one .tolist().
    tr_vals = torch.stack([tr_sums['mse'], tr_sums['mae'], tr_sums['cos']]) / max(tr_n, 1)
    va_vals = torch.stack([va_sums['mse'], va_sums['mae'], va_sums['cos']]) / max(va_n, 1)
    tr_mse, tr_mae, tr_cos = tr_vals.tolist()
    va_mse, va_mae, va_cos = va_vals.tolist()
    metrics['train'].append({'epoch': epoch, 'mse': tr_mse, 'mae': tr_mae, 'cos': tr_cos})
    metrics['val'].append(  {'epoch': epoch, 'mse': va_mse, 'mae': va_mae, 'cos': va_cos})

    va_sel = {'mse': va_mse, 'mae': va_mae, 'cos': va_cos}[sel_key]
    if is_better(va_sel, best_val_metric):
      best_val_metric = va_sel
      best_epoch = epoch
      best_state_dict = {k: v.detach().cpu().clone() for k, v in projector.state_dict().items()}

  # Snapshot the val row at best_epoch so we can preserve `best_val_mse` (legacy
  # field) regardless of which metric drove selection.
  best_val_row = (metrics['val'][best_epoch - 1] if best_epoch > 0
                  else {'mse': float('nan'), 'mae': float('nan'), 'cos': float('nan')})
  best_val_mse = best_val_row['mse']

  print(f"  [linear_proj:{anchor_key_tag}] best epoch: {best_epoch}  "
        f"best val {sel_key.upper()}: {best_val_metric:.6f}  "
        f"(loss='{cfg['loss']}')")

  # --- Restore best weights and run a single test evaluation ---
  projector.load_state_dict(best_state_dict)
  projector.eval()
  te_sums = {k: torch.zeros((), device=device) for k in ('mse', 'mae', 'cos')}
  te_n = 0
  with torch.no_grad():
    for xb, yb in test_loader:
      pred = projector(xb)
      _accum_projector_batch_metrics(pred, yb, te_sums)
      te_n += 1
  te_vals = torch.stack([te_sums['mse'], te_sums['mae'], te_sums['cos']]) / max(te_n, 1)
  te_mse, te_mae, te_cos = te_vals.tolist()
  test_metrics = {'mse': te_mse, 'mae': te_mae, 'cos': te_cos}
  metrics['test'] = test_metrics
  print(f"  [linear_proj:{anchor_key_tag}] test — MSE: {test_metrics['mse']:.6f}  "
        f"MAE: {test_metrics['mae']:.6f}  cos: {test_metrics['cos']:.6f}")

  # --- Persist best checkpoint ---
  ckpt_path = os.path.join(projector_dir, f'best_projector_{best_epoch}.pt')
  torch.save(best_state_dict, ckpt_path)

  # --- Project each split's embeddings using the best (now-loaded) projector ---
  projector_cpu = torch.nn.Linear(d_old, d_new)
  projector_cpu.load_state_dict(best_state_dict)
  projector_cpu.eval()

  splits_out = {}
  for name in ('train', 'val', 'test'):
    with torch.no_grad():
      pred_norm = projector_cpu(
        torch.from_numpy(split_arrays[name]['old_norm'])
      ).numpy().astype(np.float32)
    if norm_stats is not None:
      pred = (pred_norm * norm_stats['new_std']) + norm_stats['new_mean']
    else:
      pred = pred_norm
    splits_out[name] = {
      'df_path':     split_csv_paths[name],
      'projected':   pred.astype(np.float32),
      'target':      split_arrays[name]['new'].astype(np.float32),
      'sample_ids':  split_arrays[name]['sample_ids'].astype(np.int64),
      'subject_ids': split_arrays[name]['subject_ids'],
      'labels':      split_arrays[name]['labels'].astype(np.float32),
    }

  return {
    'projector':            projector_cpu,
    'norm_stats':           norm_stats,
    'splits':               splits_out,
    'metrics':              metrics,
    'best_epoch':           best_epoch,
    'best_val_mse':         float(best_val_mse),
    'best_val_metric':      float(best_val_metric),
    'best_val_metric_name': sel_key,
    'ckpt_path':            ckpt_path,
    'config':               cfg,
    'split_mode':           split_mode,
  }


def _apply_linear_projector(projector, norm_stats, old_embeddings):
  """
  Apply a trained linear projector to a batch of old-space embeddings, with
  optional input normalization and output un-normalization.

  Args:
    projector      (torch.nn.Linear): Trained projector (any device).
    norm_stats     (dict | None): {'old_mean','old_std','new_mean','new_std'} or None.
    old_embeddings (np.ndarray): Shape (N, D_old).

  Returns:
    np.ndarray: Projected embeddings, shape (N, D_new), dtype float32.
  """
  X = old_embeddings.astype(np.float32)
  if norm_stats is not None:
    X = _apply_norm(X, norm_stats['old_mean'], norm_stats['old_std'])
  device = next(projector.parameters()).device
  with torch.no_grad():
    pred = projector(torch.from_numpy(X).to(device)).cpu().numpy().astype(np.float32)
  if norm_stats is not None:
    pred = (pred * norm_stats['new_std']) + norm_stats['new_mean']
  return pred.astype(np.float32)


def _select_anchors(df_full, num_anchors, selection_type):
  """
  Select anchor samples from a DataFrame according to the given strategy.

  Args:
    df_full        (pd.DataFrame): Full pool of candidate anchor samples.
    num_anchors    (int): Anchors per stratum (or total for 'random').
    selection_type (str): One of:
      'random'               — sample num_anchors uniformly at random.
      'balance_class_random' — sample num_anchors per class_id;
                               total = num_anchors × num_classes.
      'balance_subject_random' — sample num_anchors per subject_id;
                               total = num_anchors × num_subjects.
      'balance_class_subject' — sample num_anchors per (class_id, subject_id) pair,
                               skipping empty combos (with a warning);
                               total ≤ num_anchors × num_classes × num_subjects.

  Returns:
    pd.DataFrame: Selected rows, reset index.

  Raises:
    ValueError: If any non-empty stratum has fewer than num_anchors samples,
                or if selection_type is unknown.
  """
  if selection_type == 'random':
    if len(df_full) < num_anchors:
      raise ValueError(
        f'random: only {len(df_full)} samples available but num_anchors={num_anchors}.'
      )
    return df_full.sample(n=num_anchors, random_state=_SEED).reset_index(drop=True)

  if selection_type == 'balance_class_random':
    dfs = []
    for cid in sorted(df_full['class_id'].unique()):
      df_cls = df_full[df_full['class_id'] == cid]
      if len(df_cls) < num_anchors:
        raise ValueError(
          f'balance_class_random: class_id={cid} has {len(df_cls)} samples '
          f'but num_anchors={num_anchors}.'
        )
      dfs.append(df_cls.sample(n=num_anchors, random_state=_SEED))
    return pd.concat(dfs, ignore_index=True)

  if selection_type == 'balance_subject_random':
    dfs = []
    for sid in sorted(df_full['subject_id'].unique()):
      df_subj = df_full[df_full['subject_id'] == sid]
      if len(df_subj) < num_anchors:
        raise ValueError(
          f'balance_subject_random: subject_id={sid} has {len(df_subj)} samples '
          f'but num_anchors={num_anchors}.'
        )
      dfs.append(df_subj.sample(n=num_anchors, random_state=_SEED))
    return pd.concat(dfs, ignore_index=True)

  if selection_type == 'balance_class_subject':
    dfs = []
    for cid in sorted(df_full['class_id'].unique()):
      for sid in sorted(df_full['subject_id'].unique()):
        df_cs = df_full[(df_full['class_id'] == cid) & (df_full['subject_id'] == sid)]
        if len(df_cs) == 0:
          print(f'  [balance_class_subject] Warning: skipping empty combo class_id={cid}, subject_id={sid}')
          continue
        if len(df_cs) < num_anchors:
          raise ValueError(
            f'balance_class_subject: class_id={cid}, subject_id={sid} has {len(df_cs)} samples '
            f'but num_anchors={num_anchors}.'
          )
        dfs.append(df_cs.sample(n=num_anchors, random_state=_SEED))
    if not dfs:
      raise ValueError('balance_class_subject: no valid (class_id, subject_id) combos found.')
    return pd.concat(dfs, ignore_index=True)

  raise ValueError(f'Unknown anchor_selection_type: {selection_type!r}')


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def _get_sampler(sampler_name, search_space):
  """
  Create an Optuna sampler.

  Args:
    sampler_name (str): 'tpe', 'random', or 'grid'.
    search_space (dict): Mapping param_name → list of values (used by GridSampler).

  Returns:
    optuna.samplers.BaseSampler
  """
  if sampler_name == 'grid':
    return optuna.samplers.GridSampler(search_space)
  if sampler_name == 'random':
    return optuna.samplers.RandomSampler()
  return optuna.samplers.TPESampler()


def _build_search_space(args):
  """
  Build Optuna search space dict from parsed CLI args.

  Args:
    args (argparse.Namespace): Parsed CLI args (hyper args are lists).

  Returns:
    dict: Mapping each of the 8 hyper param names to its candidate list.
  """
  return {
    'num_anchors':              args.num_anchors,
    'anchor_selection_type':    args.anchor_selection_type,
    'csv_anchor_selection':     args.csv_anchor_selection,
    'old_model_csv':            args.old_model_csv,
    'interpolation_similarity': args.interpolation_similarity,
    'weighting_method':         args.weighting_method,
    'temperature':              args.temperature,
    'rbf_sigma':                args.rbf_sigma,
  }


def _precompute_embeddings(
  old_model, new_model, old_model_pth, new_model_pth,
  old_config, new_config, old_features_path, new_features_path,
  args, precomputed_dir,
):
  """
  Pre-extract and cache all embeddings needed across Optuna trials.

  Anchor embeddings are keyed by (csv_anchor_selection, num_anchors, anchor_selection_type).
  When num_anchors=0, no anchor extraction is performed and _ZERO_ANCHOR_KEY is registered
  as a sentinel entry in anchor_cache.
  Old-model tensor embeddings are keyed by old_model_csv split name.

  Args:
    old_model           (Model_Advanced): Old model instance.
    new_model           (Model_Advanced): New model instance.
    old_model_pth       (str): Path to old model checkpoint.
    new_model_pth       (str): Path to new model checkpoint.
    old_config          (dict): Old model k_fold_results.pkl.
    new_config          (dict): New model k_fold_results.pkl.
    old_features_path   (str): Old model features folder.
    new_features_path   (str): New model features folder.
    args (argparse.Namespace): Parsed CLI args (hyper args are lists).
    precomputed_dir     (str): Directory for shared intermediate CSVs (anchors, tensors).

  Returns:
    tuple[dict, dict]:
      anchor_cache — (csv_anchor_selection, num_anchors, anchor_selection_type) →
          {'old': old_model_anchors, 'new': new_model_anchors_aligned, 'anchors_csv': str}
          _ZERO_ANCHOR_KEY → {'old': None, 'new': None, 'anchors_csv': None}
      tensor_cache — old_model_csv →
          {'old_tensors': old_model_tensors, 'old_tensors_csv': str}
  """
  anchor_cache = {}
  tensor_cache = {}
  new_dataset = _detect_dataset(new_features_path)
  old_backbone = _detect_backbone(old_features_path)
  new_backbone = _detect_backbone(new_features_path)
  if old_backbone != new_backbone:
    print(f'  WARNING: old backbone ({old_backbone}) differs from new backbone ({new_backbone}) — proceeding anyway')
  anchor_domain_features_for_old = _get_features_path(old_features_path, new_dataset)

  # --- val.csv pool for the linear projector (extracted once, reused per combo) ---
  linear_val_pool = None
  if 'linear' in args.interpolation_similarity:
    linear_val_pool = _extract_linear_val_pool(
      old_model, new_model, old_model_pth, new_model_pth,
      old_config, new_config, new_features_path, anchor_domain_features_for_old,
    )

  # --- Anchor embeddings (one extraction per unique combo) ---
  anchor_combos = list(itertools.product(
    args.csv_anchor_selection, args.num_anchors, args.anchor_selection_type,
  ))
  for csv_sel, num_anch, sel_type in anchor_combos:
    key = (csv_sel, num_anch, sel_type)
    if key in anchor_cache:
      continue
    if num_anch == 0:
      continue
    anchor_paths = _resolve_anchor_csvs(new_model_pth, csv_sel)
    helper.set_step_shift(new_features_path)
    anchor_dfs = []
    for p in anchor_paths:
      assert os.path.isfile(p), f'Anchor CSV not found: {p}'
      anchor_dfs.append(pd.read_csv(clean_csv_from_augmentations(p), sep='\t', dtype={'sample_name': str}))
    df_full = (
      pd.concat(anchor_dfs, ignore_index=True).drop_duplicates(subset='sample_id')
      if len(anchor_dfs) > 1 else anchor_dfs[0]
    )
    df_anch = _select_anchors(df_full, num_anch, sel_type)
    anchors_csv = os.path.join(precomputed_dir, f'anchors_{csv_sel}_{num_anch}_{sel_type}.csv')
    df_anch.to_csv(anchors_csv, index=False, sep='\t')
    print(f'[precompute] anchor key={key} ({len(df_anch)} rows) → {anchors_csv}')

    old_anch = _extract_embeddings(
      old_model, old_model_pth, anchors_csv, old_config,
      features_path_override=anchor_domain_features_for_old,
    )
    new_anch = _extract_embeddings(new_model, new_model_pth, anchors_csv, new_config)
    new_aligned = _align_by_sample_id(old_anch, new_anch)
    anchor_cache[key] = {
      'old': old_anch,
      'new': new_aligned,
      'anchors_csv': anchors_csv,
      'anchors_df': df_anch,
    }
    if 'linear' in args.interpolation_similarity:
      projector_dir = os.path.join(
        precomputed_dir, 'linear_projector', f'{csv_sel}_{num_anch}_{sel_type}',
      )
      anchor_cache[key]['projector'] = _train_linear_projector(
        old_anchors=old_anch,
        new_anchors=new_aligned,
        df_anch=df_anch,
        val_pool=linear_val_pool,
        projector_dir=projector_dir,
        anchor_key_tag=f'{csv_sel}_{num_anch}_{sel_type}',
      )

  if 0 in args.num_anchors:
    anchor_cache[_ZERO_ANCHOR_KEY] = {'old': None, 'new': None, 'anchors_csv': None}

  # --- Old-model tensor embeddings (one extraction per unique split) ---
  for old_csv_split in set(args.old_model_csv):
    if old_csv_split in tensor_cache:
      continue
    raw_csvs = _resolve_old_model_csvs(old_model_pth, old_csv_split)
    helper.set_step_shift(old_features_path)
    dfs = []
    for p in raw_csvs:
      assert os.path.isfile(p), f'Old model CSV not found: {p}'
      clean_p = clean_csv_from_augmentations(p)
      dfs.append(pd.read_csv(clean_p, sep='\t', dtype={'sample_name': str}))
    df_old = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='sample_id')
    old_tensors_csv = os.path.join(precomputed_dir, f'old_tensors_{old_csv_split}.csv')
    df_old.to_csv(old_tensors_csv, index=False, sep='\t')
    print(f'[precompute] old_tensors split={old_csv_split} ({len(df_old)} samples) → {old_tensors_csv}')
    tensor_cache[old_csv_split] = {
      'old_tensors': _extract_embeddings(old_model, old_model_pth, old_tensors_csv, old_config),
      'old_tensors_csv': old_tensors_csv,
    }

  # --- New-model oracle embeddings for num_anchors=-1 (one extraction per unique split) ---
  if -1 in args.num_anchors:
    old_dataset = _detect_dataset(old_features_path)
    new_features_for_old = _get_features_path(new_features_path, old_dataset)
    print(f'[precompute] num_anchors=-1 oracle: new model features override → {new_features_for_old}')
    for old_csv_split in set(args.old_model_csv):
      oracle_key = f'__oracle__{old_csv_split}'
      if oracle_key in tensor_cache:
        continue
      old_tensors_csv = tensor_cache[old_csv_split]['old_tensors_csv']
      oracle_raw = _extract_embeddings(
        new_model, new_model_pth, old_tensors_csv, new_config,
        features_path_override=new_features_for_old,
      )
      old_tensors = tensor_cache[old_csv_split]['old_tensors']
      tensor_cache[oracle_key] = _align_by_sample_id(old_tensors, oracle_raw)
      print(f'[precompute] oracle key={oracle_key!r} → {tensor_cache[oracle_key]["embeddings"].shape}')
    anchor_cache[_NEG_ONE_ANCHOR_KEY] = {'old': None, 'new': None, 'anchors_csv': None}

  return anchor_cache, tensor_cache


def _run_trial(trial_params, trial_number, anchor_cache, tensor_cache, new_model, new_config, trial_dir, uid):
  """
  Run a single cheap projection trial using pre-cached embeddings.

  Args:
    trial_params  (dict): Suggested hyper values for this trial (all 8 keys).
    trial_number  (int): Optuna trial number (logged in the saved result).
    anchor_cache  (dict): Pre-computed anchor embeddings from _precompute_embeddings.
    tensor_cache  (dict): Pre-computed old-model tensor embeddings.
    new_model     (Model_Advanced): New model instance (head.linear used for classification).
    new_config    (dict): New model k_fold_results.pkl.
    trial_dir     (str): Directory where results.pkl will be saved.
    uid           (int): Timestamp uid of the parent Optuna search run (saved in results.pkl).

  Returns:
    float: MAE for this trial.
  """
  old_model_tensors = tensor_cache[trial_params['old_model_csv']]['old_tensors']

  if trial_params['num_anchors'] == 0:
    d_old = old_model_tensors['embeddings'].shape[1]
    d_new = new_model.head.linear.in_features
    if d_old != d_new:
      raise ValueError(
        f'num_anchors=0 requires D_old == D_new, but got D_old={d_old}, D_new={d_new}'
      )
    projected = old_model_tensors['embeddings']
    print(f"  [trial {trial_number}] num_anchors=0 → projecting by identity (no interpolation)")
    weights   = np.zeros((len(projected), 0), dtype=np.float32)
  elif trial_params['num_anchors'] == -1:
    oracle_key = f'__oracle__{trial_params["old_model_csv"]}'
    projected  = tensor_cache[oracle_key]['embeddings']
    weights    = np.zeros((len(projected), 0), dtype=np.float32)
    print(f"  [trial {trial_number}] num_anchors=-1 → oracle (full new model on old domain)")
  else:
    anchor_key = (
      trial_params['csv_anchor_selection'],
      trial_params['num_anchors'],
      trial_params['anchor_selection_type'],
    )
    old_model_anchors         = anchor_cache[anchor_key]['old']
    new_model_anchors_aligned = anchor_cache[anchor_key]['new']
    if trial_params['interpolation_similarity'] == 'linear':
      bundle = anchor_cache[anchor_key]['projector']
      projected = _apply_linear_projector(
        bundle['projector'], bundle['norm_stats'],
        old_model_tensors['embeddings'],
      )
      weights = np.zeros((len(projected), 0), dtype=np.float32)
      print(f"  [trial {trial_number}] interpolation_similarity=linear → "
            f"applying learned projector (best_epoch={bundle['best_epoch']})")
    else:
      weights = _compute_weights(
        z=old_model_tensors['embeddings'],
        anchors=old_model_anchors['embeddings'],
        sim_type=trial_params['interpolation_similarity'],
        method=trial_params['weighting_method'],
        temperature=trial_params['temperature'],
        sigma=trial_params['rbf_sigma'],
      )
      projected = (weights @ new_model_anchors_aligned['embeddings'].astype(np.float32))

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
  preds_flat  = predictions.squeeze(-1) if predictions.ndim > 1 else predictions
  labels_true = old_model_tensors['labels']
  mae = float(np.mean(np.abs(preds_flat - labels_true)))
  ccc = float(tools.concordance_ccc(y_true=labels_true, y_pred=preds_flat))
  print(f'  [trial {trial_number}] MAE={mae:.4f}  CCC={ccc:.4f}  params={trial_params}')

  os.makedirs(trial_dir, exist_ok=True)
  trial_result = {
    'trial_params': trial_params,
    'trial_number': trial_number,
    'uid': uid,
    'new_model_tensors': {
      **old_model_tensors,
      'embeddings':  projected,
      'weights':     weights.astype(np.float32),
      'logits':      logits.numpy().astype(np.float32),
      'predictions': predictions,
    },
    'old_model_tensors': old_model_tensors,
    'metrics': {'mae': mae, 'ccc': ccc},
  }
  if trial_params['interpolation_similarity'] == 'linear' and trial_params['num_anchors'] not in (0, -1):
    anchor_key = (
      trial_params['csv_anchor_selection'],
      trial_params['num_anchors'],
      trial_params['anchor_selection_type'],
    )
    bundle = anchor_cache[anchor_key]['projector']
    trial_result['linear_projector'] = {
      'config':               bundle['config'],
      'norm_stats':           bundle['norm_stats'],
      'best_epoch':           bundle['best_epoch'],
      'best_val_mse':         bundle['best_val_mse'],
      'best_val_metric':      bundle['best_val_metric'],
      'best_val_metric_name': bundle['best_val_metric_name'],
      'ckpt_path':            bundle['ckpt_path'],
      'metrics':              bundle['metrics'],
      'splits':               bundle['splits'],
    }
  with open(os.path.join(trial_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(trial_result, f)
  return mae


def run_optuna(args):
  """
  Run an Optuna hyperparameter search over cross-space projection parameters.

  Models are built once; embeddings are pre-extracted and cached before the study starts
  so only the cheap projection + classification ops run per trial.

  Args:
    args (argparse.Namespace): Parsed CLI args. Hyper args are lists; model paths are strings.

  Returns:
    str: Path to the output directory containing study results.
  """
  np.random.seed(_SEED)
  random.seed(_SEED)
  optuna.logging.set_verbosity(optuna.logging.WARNING)

  uid = int(time.time())

  def _fmt(vals):
    return '-'.join(str(v) for v in vals)

  _lin_suffix = f'_lin_{_linear_projector_tag()}' if 'linear' in args.interpolation_similarity else ''
  args_tag = (
    f'K{_fmt(args.num_anchors)}'
    f'_{_fmt(args.anchor_selection_type)}'
    f'_{_fmt(args.csv_anchor_selection)}'
    f'_{_fmt(args.old_model_csv)}'
    f'_{_fmt(args.interpolation_similarity)}'
    f'_{_fmt(args.weighting_method)}'
    f'_t{_fmt(args.temperature)}'
    f'_s{_fmt(args.rbf_sigma)}'
    + _lin_suffix
  )
  tag_prefix = f'{args.run_tag}_' if args.run_tag else ''
  out_dir = os.path.join(
    os.getcwd(), 'Cross_projection', f'search_{tag_prefix}{args_tag}_{uid}',
  )
  precomputed_dir = os.path.join(out_dir, 'precomputed')
  os.makedirs(precomputed_dir, exist_ok=True)
  print(f'[run_optuna] Output: {out_dir}  n_trials={args.n_trials}  sampler={args.optuna_sampler}')

  print('Loading configs...')
  old_config = _load_config(args.old_model_pth)
  new_config  = _load_config(args.new_model_pth)
  old_features_path = old_config['model_advanced_params']['features_folder_saving_path']
  new_features_path = new_config['model_advanced_params']['features_folder_saving_path']
  old_model = _build_model(old_config)
  new_model  = _build_model(new_config)

  anchor_cache, tensor_cache = _precompute_embeddings(
    old_model, new_model, args.old_model_pth, args.new_model_pth,
    old_config, new_config, old_features_path, new_features_path,
    args, precomputed_dir,
  )

  search_space = _build_search_space(args)
  sampler = _get_sampler(args.optuna_sampler, search_space)
  study = optuna.create_study(direction='minimize', sampler=sampler)

  # Cache for num_anchors=0 and num_anchors=-1: all anchor-param combos produce the same
  # output per old_model_csv, so only the first trial per split actually runs.
  _zero_anchor_mae_cache:    dict = {}
  _neg_one_anchor_mae_cache: dict = {}
  # Cache for interpolation_similarity='linear': output is determined by
  # (anchor_key, old_model_csv); weighting_method/temperature/rbf_sigma are ignored.
  _linear_mae_cache:         dict = {}

  def objective(trial):
    params = {
      'num_anchors':              trial.suggest_categorical('num_anchors',              args.num_anchors),
      'anchor_selection_type':    trial.suggest_categorical('anchor_selection_type',    args.anchor_selection_type),
      'csv_anchor_selection':     trial.suggest_categorical('csv_anchor_selection',     args.csv_anchor_selection),
      'old_model_csv':            trial.suggest_categorical('old_model_csv',            args.old_model_csv),
      'interpolation_similarity': trial.suggest_categorical('interpolation_similarity', args.interpolation_similarity),
      'weighting_method':         trial.suggest_categorical('weighting_method',         args.weighting_method),
      'temperature':              trial.suggest_categorical('temperature',              args.temperature),
      'rbf_sigma':                trial.suggest_categorical('rbf_sigma',               args.rbf_sigma),
    }
    if params['num_anchors'] == 0:
      old_csv = params['old_model_csv']
      if old_csv in _zero_anchor_mae_cache:
        print(f'  [trial {trial.number}] num_anchors=0, old_model_csv={old_csv!r} — reusing cached result')
        return _zero_anchor_mae_cache[old_csv]
    if params['num_anchors'] == -1:
      old_csv = params['old_model_csv']
      if old_csv in _neg_one_anchor_mae_cache:
        print(f'  [trial {trial.number}] num_anchors=-1, old_model_csv={old_csv!r} — reusing cached result')
        return _neg_one_anchor_mae_cache[old_csv]
    if params['interpolation_similarity'] == 'linear' and params['num_anchors'] not in (0, -1):
      linear_key = (
        params['csv_anchor_selection'], params['num_anchors'],
        params['anchor_selection_type'], params['old_model_csv'],
      )
      if linear_key in _linear_mae_cache:
        print(f'  [trial {trial.number}] interpolation_similarity=linear, '
              f'key={linear_key} — reusing cached result')
        return _linear_mae_cache[linear_key]
    _proj_tag_t = (
      f'_lin_{_linear_projector_tag()}'
      if params['interpolation_similarity'] == 'linear'
      else f'_{params["weighting_method"]}_t{params["temperature"]}_s{params["rbf_sigma"]}'
    )
    trial_tag = (
      f'K{params["num_anchors"]}'
      f'_{params["anchor_selection_type"]}'
      f'_{params["csv_anchor_selection"]}'
      f'_{params["old_model_csv"]}'
      f'_{params["interpolation_similarity"]}'
      + _proj_tag_t
    )
    trial_dir = os.path.join(out_dir, f'cross_space_projection_{trial_tag}_{trial.number}')
    mae = _run_trial(params, trial.number, anchor_cache, tensor_cache, new_model, new_config, trial_dir, uid)
    if params['num_anchors'] == 0:
      _zero_anchor_mae_cache[params['old_model_csv']] = mae
    if params['num_anchors'] == -1:
      _neg_one_anchor_mae_cache[params['old_model_csv']] = mae
    if params['interpolation_similarity'] == 'linear' and params['num_anchors'] not in (0, -1):
      linear_key = (
        params['csv_anchor_selection'], params['num_anchors'],
        params['anchor_selection_type'], params['old_model_csv'],
      )
      _linear_mae_cache[linear_key] = mae
    return mae

  if args.n_trials is None:
    if args.optuna_sampler != 'grid':
      raise ValueError(
        '--n_trials=None (auto) is only supported with --optuna_sampler grid. '
        'Set --n_trials explicitly for tpe/random samplers.'
      )
    n_trials = 1
    for v in search_space.values():
      n_trials *= len(v)
    print(f'[run_optuna] n_trials=None + grid sampler → running all {n_trials} grid combinations')
  else:
    n_trials = args.n_trials

  study.optimize(objective, n_trials=n_trials)

  best = study.best_trial
  print(f'\n[run_optuna] Best trial #{best.number}: MAE={best.value:.4f}')
  print(f'  params: {best.params}')

  with open(os.path.join(out_dir, 'optuna_study.pkl'), 'wb') as f:
    pickle.dump(study, f)

  with open(os.path.join(out_dir, 'best_config.txt'), 'w') as f:
    f.write(f'script_cmd: {" ".join(sys.argv)}\n')
    f.write(f'best_trial: {best.number}\n')
    f.write(f'mae: {best.value:.6f}\n')
    for k, v in best.params.items():
      f.write(f'{k}: {v}\n')

  print(f'Done. All outputs in {out_dir}')
  return out_dir


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
  _proj_tag = (
    f'_lin_{_linear_projector_tag()}'
    if args.interpolation_similarity == 'linear'
    else f'_{args.weighting_method}_t{args.temperature}_s{args.rbf_sigma}'
  )
  args_tag = (
    f'K{args.num_anchors}'
    f'_{args.anchor_selection_type}'
    f'_{args.csv_anchor_selection}'
    f'_{args.old_model_csv}'
    f'_{args.interpolation_similarity}'
    + _proj_tag
  )
  tag_prefix = f'{args.run_tag}_' if args.run_tag else ''
  out_dir = os.path.join(os.getcwd(), 'Cross_projection', f'cross_space_projection_{tag_prefix}{args_tag}_{uid}')
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
  linear_bundle = None  # populated only when interpolation_similarity == 'linear'

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

  # 4.3 old model on old domain samples (always required)
  print('Extracting old_model_tensors...')
  old_model_tensors = _extract_embeddings(
    old_model, args.old_model_pth, old_tensors_csv_path, old_config,
  )
  print(f'  old_model_tensors: {old_model_tensors["embeddings"].shape}')

  if args.num_anchors == 0:
    # --- Identity projection: skip anchor steps, pass old features through unchanged ---
    d_old = old_model_tensors['embeddings'].shape[1]
    d_new = new_model.head.linear.in_features
    if d_old != d_new:
      raise ValueError(
        f'num_anchors=0 requires D_old == D_new, but got D_old={d_old}, D_new={d_new}'
      )
    print('[cross_space_projection] num_anchors=0 — identity projection (old features passed through)')
    projected                 = old_model_tensors['embeddings']
    weights                   = np.zeros((len(projected), 0), dtype=np.float32)
    old_model_anchors         = None
    new_model_anchors_aligned = None
    anchors_csv_path          = None
  elif args.num_anchors == -1:
    # --- Oracle: run full new model forward pass on old-domain samples using new backbone features ---
    old_dataset = _detect_dataset(old_features_path)
    new_features_for_old = _get_features_path(new_features_path, old_dataset)
    print(
      f'[cross_space_projection] num_anchors=-1 — oracle: full new model on old domain '
      f'(features override → {new_features_for_old})'
    )
    oracle_raw = _extract_embeddings(
      new_model, args.new_model_pth, old_tensors_csv_path, new_config,
      features_path_override=new_features_for_old,
    )
    new_model_oracle          = _align_by_sample_id(old_model_tensors, oracle_raw)
    projected                 = new_model_oracle['embeddings']
    weights                   = np.zeros((len(projected), 0), dtype=np.float32)
    old_model_anchors         = None
    new_model_anchors_aligned = None
    anchors_csv_path          = None
  else:
    # --- Step 2: Select anchors from new model domain ---
    anchor_paths = _resolve_anchor_csvs(args.new_model_pth, args.csv_anchor_selection)
    assert all(os.path.isfile(p) for p in anchor_paths), f'One or more anchor CSVs missing: {anchor_paths}'
    helper.set_step_shift(new_features_path)
    anchor_dfs = [
      pd.read_csv(clean_csv_from_augmentations(p), sep='\t', dtype={'sample_name': str})
      for p in anchor_paths
    ]
    df_anchors_full = (
      pd.concat(anchor_dfs, ignore_index=True).drop_duplicates(subset='sample_id')
      if len(anchor_dfs) > 1 else anchor_dfs[0]
    )
    df_anchors = _select_anchors(df_anchors_full, args.num_anchors, args.anchor_selection_type)
    anchors_csv_path = os.path.join(out_dir, 'anchors.csv')
    df_anchors.to_csv(anchors_csv_path, index=False, sep='\t')
    print(f'Anchors ({len(df_anchors)}) saved to {anchors_csv_path}')

    # 4.1 old model on new-domain anchors: use same backbone but new model's dataset feature folder
    new_dataset = _detect_dataset(new_features_path)
    old_backbone = _detect_backbone(old_features_path)
    new_backbone = _detect_backbone(new_features_path)
    if old_backbone != new_backbone:
      print(f'  WARNING: old backbone ({old_backbone}) differs from new backbone ({new_backbone}) — proceeding anyway')
    anchor_domain_features_for_old = _get_features_path(old_features_path, new_dataset)
    print(f'Extracting old_model_anchors with features override → {anchor_domain_features_for_old}')
    old_model_anchors = _extract_embeddings(
      old_model, args.old_model_pth, anchors_csv_path, old_config,
      features_path_override=anchor_domain_features_for_old,
    )
    print(f'  old_model_anchors: {old_model_anchors["embeddings"].shape}')

    # 4.2 new model on new-domain anchors (no override — new model already uses its own features)
    print('Extracting new_model_anchors...')
    new_model_anchors = _extract_embeddings(
      new_model, args.new_model_pth, anchors_csv_path, new_config,
    )
    print(f'  new_model_anchors: {new_model_anchors["embeddings"].shape}')

    # --- Step 5: Align anchor embeddings by sample_id ---
    new_model_anchors_aligned = _align_by_sample_id(old_model_anchors, new_model_anchors)

    if args.interpolation_similarity == 'linear':
      # --- Step 6 (linear): train projector on all anchors; val/test from val.csv ---
      projector_dir = os.path.join(out_dir, 'linear_projector')
      linear_val_pool = _extract_linear_val_pool(
        old_model, new_model, args.old_model_pth, args.new_model_pth,
        old_config, new_config, new_features_path, anchor_domain_features_for_old,
      )
      linear_bundle = _train_linear_projector(  # noqa: F841 — also persisted into dict_res below
        old_anchors=old_model_anchors,
        new_anchors=new_model_anchors_aligned,
        val_pool=linear_val_pool,
        df_anch=df_anchors,
        projector_dir=projector_dir,
        anchor_key_tag='single',
      )
      projected = _apply_linear_projector(
        linear_bundle['projector'], linear_bundle['norm_stats'],
        old_model_tensors['embeddings'],
      )
      weights = np.zeros((len(projected), 0), dtype=np.float32)
      print(f'  projected (linear): {projected.shape}')
    else:
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
    'script_cmd':              ' '.join(sys.argv),
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
  if linear_bundle is not None:
    dict_res['linear_projector'] = {
      'config':               linear_bundle['config'],
      'norm_stats':           linear_bundle['norm_stats'],
      'best_epoch':           linear_bundle['best_epoch'],
      'best_val_mse':         linear_bundle['best_val_mse'],
      'best_val_metric':      linear_bundle['best_val_metric'],
      'best_val_metric_name': linear_bundle['best_val_metric_name'],
      'ckpt_path':            linear_bundle['ckpt_path'],
      'metrics':              linear_bundle['metrics'],
      'splits':               linear_bundle['splits'],
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
  # --- Hyper args: accept one or more values; multiple values trigger Optuna ---
  parser.add_argument('--num_anchors', type=int, nargs='+', required=True,
                      help='Number of anchor samples (one or more values to sweep)')
  parser.add_argument('--anchor_selection_type', type=str, nargs='+', default=['random'],
                      choices=['random', 'balance_class_random', 'balance_subject_random', 'balance_class_subject'],
                      help='Anchor selection strategy (one or more values to sweep). '
                           'balance_* strategies use num_anchors per stratum.')
  parser.add_argument('--csv_anchor_selection', type=str, nargs='+', required=True,
                      help='Split(s) to select anchors from (train/val/test/exc_train/exc_val/exc_test) — new model domain')
  parser.add_argument('--old_model_csv', type=str, nargs='+', required=True,
                      help='Split(s) to project (train/val/test/all/exc_train/exc_val/exc_test) — old model domain')
  parser.add_argument('--interpolation_similarity', type=str, nargs='+', default=['cos'],
                      choices=['cos', 'l1', 'l2', 'l_inf', 'linear'],
                      help='Similarity/distance metric(s) for weight computation. '
                           "'linear' trains a learned nn.Linear(D_old, D_new) projector on the "
                           'anchor pairs (subject-disjoint split, see LINEAR_PROJECTOR_CONFIG); '
                           'weighting_method/temperature/rbf_sigma are ignored in that mode.')
  parser.add_argument('--weighting_method', type=str, nargs='+', default=None,
                      choices=['softmax', 'rbf'],
                      help='Method(s) to convert similarities to interpolation weights '
                           '(not required when --interpolation_similarity is linear)')
  parser.add_argument('--temperature', type=float, nargs='+', default=[1.0],
                      help='Temperature(s) for softmax weighting (ignored for rbf)')
  parser.add_argument('--rbf_sigma', type=float, nargs='+', default=[1.0],
                      help='Sigma value(s) for RBF kernel: exp(-d²/(2σ²)) (ignored for softmax)')
  # --- Optuna settings ---
  parser.add_argument('--n_trials', type=int, default=None,
                      help='Number of Optuna trials. If None (default) and --optuna_sampler grid, '
                           'runs the full grid automatically.')
  parser.add_argument('--optuna_sampler', type=str, default='grid',
                      choices=['tpe', 'random', 'grid'],
                      help='Optuna sampler (tpe, random, grid). Default is grid')
  parser.add_argument('--run_tag', type=str, default=None,
                      help='Optional label prepended to the output folder name for easy identification')

  args = parser.parse_args()
  if args.weighting_method is None:
    non_linear = [s for s in args.interpolation_similarity if s != 'linear']
    if non_linear:
      parser.error(
        '--weighting_method is required when --interpolation_similarity '
        f'includes a non-linear metric (got: {non_linear})'
      )
    args.weighting_method = ['softmax']  # dummy; never used in linear mode
  if 'linear' in args.interpolation_similarity and set(args.num_anchors) <= {0, -1}:
    parser.error(
      "interpolation_similarity='linear' requires at least one --num_anchors > 0 "
      "(linear mode trains a projector on anchor pairs; num_anchors=0/-1 use no anchors)"
    )
  _hyper_lists = [
    args.num_anchors, args.anchor_selection_type, args.csv_anchor_selection,
    args.old_model_csv, args.interpolation_similarity, args.weighting_method,
    args.temperature, args.rbf_sigma,
  ]
  use_optuna = any(len(v) > 1 for v in _hyper_lists) or (args.n_trials is not None and args.n_trials > 1)

  if use_optuna:
    run_optuna(args)
  else:
    # Unwrap single-element lists so cross_space_projection receives scalar args
    single_args = argparse.Namespace(**{
      k: (v[0] if isinstance(v, list) else v) for k, v in vars(args).items()
    })
    cross_space_projection(single_args)

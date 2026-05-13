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

import custom.helper as helper
import custom.tools as tools
from custom.model import Model_Advanced
from log_cross_attention_from_model import clean_csv_from_augmentations

_SEED = 42
_ZERO_ANCHOR_KEY    = (None,  0, None)  # anchor_cache sentinel for the num_anchors=0 identity case
_NEG_ONE_ANCHOR_KEY = (None, -1, None)  # anchor_cache sentinel for num_anchors=-1 oracle case

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
    anchor_cache[key] = {
      'old': old_anch,
      'new': _align_by_sample_id(old_anch, new_anch),
      'anchors_csv': anchors_csv,
    }

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

  args_tag = (
    f'K{_fmt(args.num_anchors)}'
    f'_{_fmt(args.anchor_selection_type)}'
    f'_{_fmt(args.csv_anchor_selection)}'
    f'_{_fmt(args.old_model_csv)}'
    f'_{_fmt(args.interpolation_similarity)}'
    f'_{_fmt(args.weighting_method)}'
    f'_t{_fmt(args.temperature)}'
    f'_s{_fmt(args.rbf_sigma)}'
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
    trial_tag = (
      f'K{params["num_anchors"]}'
      f'_{params["anchor_selection_type"]}'
      f'_{params["csv_anchor_selection"]}'
      f'_{params["old_model_csv"]}'
      f'_{params["interpolation_similarity"]}'
      f'_{params["weighting_method"]}'
      f'_t{params["temperature"]}'
      f'_s{params["rbf_sigma"]}'
    )
    trial_dir = os.path.join(out_dir, f'cross_space_projection_{trial_tag}_{trial.number}')
    mae = _run_trial(params, trial.number, anchor_cache, tensor_cache, new_model, new_config, trial_dir, uid)
    if params['num_anchors'] == 0:
      _zero_anchor_mae_cache[params['old_model_csv']] = mae
    if params['num_anchors'] == -1:
      _neg_one_anchor_mae_cache[params['old_model_csv']] = mae
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
  args_tag = (
    f'K{args.num_anchors}'
    f'_{args.anchor_selection_type}'
    f'_{args.csv_anchor_selection}'
    f'_{args.old_model_csv}'
    f'_{args.interpolation_similarity}'
    f'_{args.weighting_method}'
    f'_t{args.temperature}'
    f'_s{args.rbf_sigma}'
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
                      choices=['cos', 'l1', 'l2', 'l_inf'],
                      help='Similarity/distance metric(s) for weight computation')
  parser.add_argument('--weighting_method', type=str, nargs='+', required=True,
                      choices=['softmax', 'rbf'],
                      help='Method(s) to convert similarities to interpolation weights')
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

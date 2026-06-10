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
import hashlib
import itertools
import json
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
import yaml

import custom.helper as helper
import custom.tools as tools
from custom.model import Model_Advanced
from log_cross_attention_from_model import clean_csv_from_augmentations

_SEED = 42
_ZERO_ANCHOR_KEY    = (None,  0, None)  # anchor_cache sentinel for the num_anchors=0 identity case
_NEG_ONE_ANCHOR_KEY = (None, -1, None)  # anchor_cache sentinel for num_anchors=-1 oracle case


def _set_global_seed(seed):
  """
  Set the module-global RNG seed and seed every RNG the pipeline draws from.

  Reassigns the module global `_SEED` (which is read at call time by anchor
  selection — `_select_anchors`' `df.sample(random_state=_SEED)` — and the
  subject-disjoint splits via `_make_subject_disjoint_subsets`, and by the Optuna
  samplers in `_get_sampler`), then seeds Python's `random`, NumPy, and torch
  (CPU + all CUDA devices) so weight init and `shuffle=True` DataLoaders become
  reproducible. cudnn / `use_deterministic_algorithms` are intentionally left
  untouched (no speed penalty, no unsupported-op crashes).

  Args:
    seed (int): The seed value to apply globally.

  Returns:
    None.
  """
  global _SEED
  _SEED = seed
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# interpolation_similarity values that train/fit a projector mapping old→new space
# (as opposed to the anchor-weighted distance metrics cos/l1/l2/l_inf/geodesic).
# All require weighting_method='none' and num_anchors > 0.
_PROJECTOR_KINDS = ('linear', 'mlp', 'procrustes', 'linear_close')


def _projector_key(interp, mlp_activation):
  """
  Build the cache/folder key identifying a trained projector bundle.

  The 'mlp' kind trains one projector per activation, so its key embeds the
  activation; 'linear', 'procrustes' and 'linear_close' have no activation and
  key on the bare interp name (preserving backward-compatible cache keys and
  folder names).

  Args:
    interp         (str): interpolation_similarity value (a member of _PROJECTOR_KINDS).
    mlp_activation (str | None): Activation name; only consulted when interp == 'mlp'.

  Returns:
    str: 'mlp_<activation>' when interp == 'mlp', else interp unchanged.
  """
  return f'mlp_{mlp_activation}' if interp == 'mlp' else interp


def _projector_tag(kind: str, cfg=None) -> str:
  """
  Build a short folder-name tag from a projector recipe (LINEAR_PROJECTOR_CONFIG
  by default).

  Because procrustes and linear_close are closed-form solutions, their tag drops
  the SGD fields (lr/bs/optimizer/wd/epochs/loss) — two recipes that differ only
  in those fields therefore collapse to the same tag, which is exactly what makes
  this tag usable as a per-recipe bundle key that auto-dedupes redundant
  closed-form trainings (see _projector_bundle_key).

  Args:
    kind (str): 'linear', 'mlp', 'procrustes' or 'linear_close'. Procrustes and
      linear_close ignore the optimizer/lr/bs/wd/epochs/loss fields (closed-form
      solutions), so their tag only carries normalize_embeddings + split_ratios.
      'mlp' shares the trained-projector recipe with 'linear' but is prefixed
      'mlp_' (the swept activation is recorded separately in the run tag).
    cfg (dict | None): Projector recipe to tag. Defaults to LINEAR_PROJECTOR_CONFIG.

  Returns:
    str: Compact string like 'lr0.0001_bs64_adamw_wd0.0001_ep100_normT_mse_sp70-10-20'
         for kind='linear', the same prefixed with 'mlp_' for kind='mlp',
         'proc_normT_sp0-50-50' for kind='procrustes', or
         'linclose_rc1e-06_normT_sp0-50-50' for kind='linear_close'.
  """
  cfg = LINEAR_PROJECTOR_CONFIG if cfg is None else cfg
  sr  = cfg['split_ratios']
  sp  = f"{int(sr[0]*100)}-{int(sr[1]*100)}-{int(sr[2]*100)}"
  norm = 'T' if cfg['normalize_embeddings'] else 'F'
  if kind == 'procrustes':
    return f"proc_norm{norm}_sp{sp}"
  if kind == 'linear_close':
    return f"linclose_rc{cfg['closed_form_rcond']}_norm{norm}_sp{sp}"
  recipe = (
    f"lr{cfg['lr']}"
    f"_bs{cfg['batch_size']}"
    f"_{cfg['optimizer']}"
    f"_wd{cfg['weight_decay']}"
    f"_ep{cfg['epochs']}"
    f"_norm{norm}"
    f"_{cfg['loss']}"
    f"_sp{sp}"
  )
  return f"mlp_{recipe}" if kind == 'mlp' else recipe

# Hyperparameters for the learned linear projector (interpolation_similarity='linear').
# Edit values here to change the training recipe; no CLI flag is exposed.
LINEAR_PROJECTOR_CONFIG = {
  'lr':                   1e-4,
  'batch_size':           64,
  'optimizer':            'adamw',   # 'adam' | 'adamw' | 'sgd'
  'weight_decay':         0,
  'epochs':               300,
  'normalize_embeddings': True,
  'loss':                 'mse',     # 'mse' | 'mae' | 'cosine'
  # Closed-form OLS solve (interpolation_similarity='linear_close' only): singular
  # values of the centered anchor matrix below closed_form_rcond * max(sv) are
  # truncated (truncated-SVD least squares), bounding weight magnitude when the
  # anchors are ill-conditioned (small num_anchors relative to D_old, or collinear
  # features). Ignored by all other kinds.
  'closed_form_rcond':    1e-6,
  # (train, val, test). The projector trains on ALL K anchors, so the train
  # entry is unused; val/test are splits of the new model's
  # val.csv sized by these fractions of the val.csv row count (test absorbs
  # the remainder so every val.csv row is used).
  'split_ratios':         (0.0, 0.50, 0.50),
  'device':               'cuda',
  'num_workers':          4,
}

# Hyperparameters for the post-projection refinement stage (see
# _refine_projector_and_linear). After the projection is fixed, this second stage
# fine-tunes the new model's head.linear (and, in 'projector_linear' mode, the projector
# too) so that source-domain (old model / "model B") accuracy improves while the new
# model's own-domain accuracy is preserved. Edit the numeric values here; the on/off
# switch and the 'mode' are driven by the --refinement CLI flag (0/1/2).
#
# 'mode' (set from --refinement):
#   'projector_linear' (--refinement 2): jointly fine-tune the projector (old→new map)
#       AND a COPY of head.linear. Requires a trained projector, so it only runs for
#       projector kinds (interpolation_similarity in _PROJECTOR_KINDS) with num_anchors > 0.
#   'linear_only'      (--refinement 1): the projection is held FIXED and only a COPY of
#       head.linear is fine-tuned. Works for ALL interpolation_similarity values — the
#       projector kinds (frozen projector) AND the distance metrics (cos/l1/l2/l_inf/
#       geodesic, fixed interpolation), since no trainable projector is required.
#
# Loss = lambda_B * reg(linear(proj(emb_B)),     labels_B)        # improve source (model B)
#      + lambda_A * reg(linear(new_anchor_emb),  anchor_labels)   # preserve new-domain (model A)
# where emb_B are old-model embeddings on the old model's `refine_split` split, proj() is
# the (fixed in linear_only) old→new projection, new_anchor_emb are the REAL new-model
# anchor embeddings (D_new), reg() is `loss`, and predictions are compared in the real
# (denormalized) label scale.
REFINEMENT_CONFIG = {
  'enabled':            False,    # default off; toggle per-run via the --refinement CLI flag
  'mode':               'projector_linear',  # 'projector_linear' (--refinement 2) | 'linear_only' (--refinement 1)
  'lr_projector':       1e-4,     # learning rate for the projector params (projector_linear only)
  'lr_linear':          1e-5,     # smaller → change the linear layer only slightly
  'lambda_B':           1.0,      # weight of the source (model B) label term
  'lambda_A':           1.0,      # weight of the new-anchor preserve term
  'optimizer':          'adamw',  # 'adam' | 'adamw' | 'sgd'
  'weight_decay':       0,
  'epochs':             100,
  'loss':               'mse',    # reg(): 'mse' | 'mae' | 'huber'
  'batch_size':         64,
  'refine_split':       'train',  # old-model split providing emb_B (model B training set)
  'refine_val_split':   'val',    # old-model held-out split for the per-epoch source validation
  'refine_val_min_keep_frac': 0.1,  # if < this fraction of a val set survives leakage exclusion,
                                    # that term is dropped and selection falls back to train loss
  'new_eval_split':     'val',   # new-model split for the preserve eval (val fallback)
  'device':             'cuda',
  'report_after_refinement': True,  # headline MAE/CCC + saved preds reflect the refined model
}

# Fields of LINEAR_PROJECTOR_CONFIG / REFINEMENT_CONFIG that may be swept (given a
# list in the YAML config). Every other field is fixed (a scalar copied from the
# base dict). enabled/mode are intentionally absent from the refinement set: they
# stay driven by the --refinement CLI flag.
_PROJECTOR_SWEEPABLE = (
  'lr', 'batch_size', 'optimizer', 'weight_decay', 'epochs',
  'normalize_embeddings', 'loss',
)
_REFINEMENT_SWEEPABLE = (
  'lr_projector', 'lr_linear', 'lambda_B', 'lambda_A', 'optimizer',
  'weight_decay', 'epochs', 'loss', 'batch_size',
)

# --refinement flag → the refinement mode(s) to run. 0=off, 1=linear_only,
# 2=projector_linear, 3=BOTH (all applicable modes for the test). For 3, the
# concrete modes that actually run depend on the interpolation_similarity (see
# _applicable_refine_modes): projector_linear is dropped for distance metrics.
_REFINE_FLAG_TO_MODES = {
  0: [],
  1: ['linear_only'],
  2: ['projector_linear'],
  3: ['linear_only', 'projector_linear'],
}


def _applicable_refine_modes(flag, interp_sim, num_anchors):
  """
  Return the refinement modes that actually run for one concrete test config.

  Resolves the --refinement flag to its mode list (_REFINE_FLAG_TO_MODES) and
  drops modes that do not apply to this test: every mode needs num_anchors > 0,
  and 'projector_linear' additionally needs a projector kind (it fine-tunes the
  trained projector, which distance metrics do not have). So flag 3 yields both
  modes for a projector kind but collapses to ['linear_only'] for a distance
  metric, and any flag yields [] when num_anchors is 0 / -1.

  Args:
    flag       (int): The --refinement flag value (0/1/2/3).
    interp_sim (str): interpolation_similarity for the test (projector kind or
                      distance metric).
    num_anchors (int): Number of anchors for the test.

  Returns:
    list[str]: Ordered modes to run (subset of ['linear_only', 'projector_linear']).
  """
  if num_anchors in (0, -1):
    return []
  is_proj = interp_sim in _PROJECTOR_KINDS
  return [
    m for m in _REFINE_FLAG_TO_MODES.get(flag, [])
    if not (m == 'projector_linear' and not is_proj)
  ]


def _grid_refine_mode_ids(flag):
  """
  Return the values of the Optuna 'refine_mode' sweep axis for a grid run.

  The axis enumerates every candidate refinement mode the --refinement flag asks for
  (across all interps); per-trial pruning in objective() then keeps only the modes that
  apply to each trial's interp/anchors (via _applicable_refine_modes). When refinement is
  off the axis carries a single inert 'none' value so it never multiplies the grid.

  Args:
    flag (int): The --refinement flag value (0/1/2/3).

  Returns:
    list[str]: ['none'] when off, else the flag's mode list (1-2 entries).
  """
  return _REFINE_FLAG_TO_MODES.get(flag, []) or ['none']


def _expand_recipes(base_cfg, block, sweepable_fields, block_name):
  """
  Expand a config block into the full Cartesian product of its swept fields.

  Each sweepable field in `block` may be a scalar or a list; list-valued fields
  are crossed to form one complete recipe dict per combination (a deep copy of
  base_cfg with the swept values overridden). Scalar entries (and untouched
  base_cfg fields) stay constant across every recipe.

  Args:
    base_cfg         (dict): The module-global default config (LINEAR_PROJECTOR_CONFIG
                             or REFINEMENT_CONFIG) supplying every fixed field.
    block            (dict | None): The YAML block overriding/sweeping fields. None
                             or empty → a single recipe equal to base_cfg.
    sweepable_fields (tuple[str]): Field names allowed to be lists (_PROJECTOR_SWEEPABLE
                             or _REFINEMENT_SWEEPABLE).
    block_name       (str): 'linear_projector' | 'refinement', for error messages.

  Returns:
    list[dict]: One complete recipe dict per swept combination (≥ 1).

  Raises:
    ValueError: If a non-sweepable field is given a list, or an unknown field appears.
  """
  block = block or {}
  unknown = set(block) - set(base_cfg)
  if unknown:
    raise ValueError(
      f"[{block_name}] unknown field(s) {sorted(unknown)}; "
      f"valid fields: {sorted(base_cfg)}"
    )
  axis_names, axis_values = [], []
  fixed = {}
  for k, v in block.items():
    if isinstance(v, list):
      if k not in sweepable_fields:
        raise ValueError(
          f"[{block_name}] field {k!r} is not sweepable (got a list {v}); "
          f"sweepable fields: {sorted(sweepable_fields)}"
        )
      if len(v) == 0:
        raise ValueError(f"[{block_name}] field {k!r} was given an empty list")
      axis_names.append(k)
      axis_values.append(v)
    else:
      fixed[k] = v
  recipes = []
  for combo in itertools.product(*axis_values) if axis_values else [()]:
    recipe = copy.deepcopy(base_cfg)
    recipe.update(fixed)
    recipe.update(dict(zip(axis_names, combo)))
    recipes.append(recipe)
  return recipes


def _projector_recipe_id(cfg) -> str:
  """
  Build a kind-independent recipe id from a projector recipe's swept fields.

  Used as the value of the Optuna 'projector_config' axis and as a stable,
  human-readable handle for the recipe. Unlike _projector_tag this never drops
  fields, so two distinct recipes always get distinct ids regardless of kind.

  Args:
    cfg (dict): A projector recipe (complete LINEAR_PROJECTOR_CONFIG-shaped dict).

  Returns:
    str: e.g. 'lr0.0001_bs64_adamw_wd0_ep150_normT_mse'.
  """
  norm = 'T' if cfg['normalize_embeddings'] else 'F'
  return (
    f"lr{cfg['lr']}_bs{cfg['batch_size']}_{cfg['optimizer']}"
    f"_wd{cfg['weight_decay']}_ep{cfg['epochs']}_norm{norm}_{cfg['loss']}"
  )


def _projector_bundle_key(kind, activation, cfg) -> str:
  """
  Build the per-recipe key under which a trained projector bundle is cached in
  anchor_cache[...]['projectors'].

  Combines the kind/activation base key (_projector_key) with _projector_tag,
  which already collapses the SGD-only fields for procrustes/linear_close — so
  redundant closed-form recipes share one bundle while every distinct
  linear/mlp recipe gets its own.

  Args:
    kind       (str): Projector kind (a member of _PROJECTOR_KINDS).
    activation (str | None): mlp activation (only meaningful for kind='mlp').
    cfg        (dict): The projector recipe.

  Returns:
    str: e.g. 'linear__lr0.0001_bs64_adamw_wd0_ep150_normT_mse_sp0-50-50'.
  """
  return f"{_projector_key(kind, activation)}__{_projector_tag(kind, cfg)}"


def _refinement_tag(cfg) -> str:
  """
  Build a short id for a refinement recipe from its swept fields.

  Used both as the Optuna 'refinement_config' axis value and as the key under
  which a refinement bundle is cached (per projector bundle, and per distance
  metric). enabled/mode are excluded — they come from --refinement and are
  identical across every recipe in a run.

  Args:
    cfg (dict): A refinement recipe (complete REFINEMENT_CONFIG-shaped dict).

  Returns:
    str: e.g. 'reflrp0.0001_reflrl1e-05_lb1.0_la1.0_adamw_wd0_ep100_mse_bs64'.
  """
  return (
    f"reflrp{cfg['lr_projector']}_reflrl{cfg['lr_linear']}"
    f"_lb{cfg['lambda_B']}_la{cfg['lambda_A']}_{cfg['optimizer']}"
    f"_wd{cfg['weight_decay']}_ep{cfg['epochs']}_{cfg['loss']}_bs{cfg['batch_size']}"
  )


# (backbone_key, dataset_key) → pre-extracted features folder path
_FEATURES_MAP = {
  ('DFER',     'UNBC'):   'UNBC/video/features/DFER/spatial_pooled_features_UNBC_B_last143_stride16_interpol',
  ('DFER',     'BIOVID'): 'partA/video/features/DFER/spatial_pooled_features_Biovid_B_last143_stride16_interpol',
  ('DFER',     'AGEDB'):  'AgeDB/features/DFER/all_pooled_features_age',
  ('DFER',     'MORPH'):  'MORPH_2/features/DFER/all_pooled_features_MORPH',
  ('VIDEOMAE', 'UNBC'):   'UNBC/video/features/VideoMaev2_S/spatial_pooled_features_UNBC_B_last143_stride16_interpol',
  ('VIDEOMAE', 'BIOVID'): 'partA/video/features/VideoMaev2_S/spatial_pooled_features_Biovid_B_last143_stride16_interpol',
  ('VIDEOMAE', 'AGEDB'):  'AgeDB/features/VideoMaev2_S/all_pooled_features_age',
  ('VIDEOMAE', 'MORPH'):  'MORPH_2/features/VideoMaev2_S/all_pooled_features_MORPH',
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


def _resolve_test_csv_strict(model_pth):
  """
  Resolve the 'test' split CSV using the same lookup as _resolve_split_csv (one
  level above the checkpoint's fold-sub directory), but raise if test.csv is
  absent instead of silently falling back to val.csv.

  The cross-val checkpoint lives in a k*_cross_val_sub_* directory that holds only
  train.csv / val.csv; test.csv sits one level up in the k*_cross_val directory —
  hence parts[:-2] rather than the parts[:-1] used by _resolve_split_csv_strict.

  Args:
    model_pth (str): Path to model checkpoint file.

  Returns:
    str: Absolute path to test.csv.

  Raises:
    FileNotFoundError: If the expected test.csv does not exist (no val fallback).
  """
  p = os.path.join(*Path(model_pth).parts[:-2], 'test.csv')
  if not os.path.exists(p):
    raise FileNotFoundError(f'New-model test.csv not found (strict, no val fallback): {p}')
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


def _compute_weights(z, anchors, sim_type, sigma):
  """
  Compute interpolation weights between query embeddings and anchor embeddings
  using an RBF kernel on the chosen distance metric.

  Args:
    z        (np.ndarray): Query embeddings,  shape (N, D).
    anchors  (np.ndarray): Anchor embeddings, shape (K, D).
    sim_type (str): Distance metric — 'cos', 'l1', 'l2', 'l_inf', or 'geodesic'.
                    For 'cos', distance is `1 - cos_sim` in [0, 2].
                    For 'geodesic', a k-NN graph (k=5, Euclidean edges) is built
                    over anchor embeddings; all-pairs shortest paths are computed,
                    then each query is connected via its k nearest anchors.
                    Disconnected pairs fall back to direct L2.
    sigma    (float): Standard deviation for RBF kernel: exp(-d²/(2σ²)).

  Returns:
    np.ndarray: Weights, shape (N, K), rows sum to 1.
  """
  if sim_type == 'cos':
    z_n = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
    a_n = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-8)
    sim = z_n @ a_n.T                         # (N, K), range [-1, 1]
    d = 1.0 - sim                     # cosine distance in [0, 2], 0 when parallel, 1 when orthogonal, 2 when opposite
  elif sim_type == 'geodesic':
    from scipy.spatial.distance import cdist
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    _GEO_K = 5
    K = anchors.shape[0]
    k = min(_GEO_K, K - 1)

    # Build symmetric k-NN graph on anchors (Euclidean edge weights). Vectorized:
    # the k nearest anchors per row (excluding self at sorted column 0, distance 0).
    anchor_d = cdist(anchors, anchors, metric='euclidean')  # (K, K)
    graph = np.full((K, K), np.inf)
    np.fill_diagonal(graph, 0.0)
    nn_idx = np.argsort(anchor_d, axis=1)[:, 1:k + 1]        # (K, k)
    rows = np.repeat(np.arange(K), k)
    cols = nn_idx.ravel()
    vals = anchor_d[rows, cols]
    graph[rows, cols] = vals
    graph[cols, rows] = vals                                 # symmetrize

    # All-pairs shortest path on anchor graph
    geo_anchors = shortest_path(csr_matrix(graph), method='D', directed=False)  # (K, K)

    # Extend geodesic to queries: connect each query to its k nearest anchors.
    # Vectorized over all N queries — d_geo[i] = min over the k nearest anchors of
    # (d_direct[i, nbr] + geo_anchors[nbr, :]). argpartition gives the unordered
    # k-nearest set, which is sufficient since the result is a min over that set.
    d_direct = cdist(z, anchors, metric='euclidean')          # (N, K)
    N = z.shape[0]
    knn_idx = np.argpartition(d_direct, k - 1, axis=1)[:, :k]  # (N, k)
    q_rows = np.arange(N)
    d_geo = np.full((N, K), np.inf, dtype=d_direct.dtype)
    for j in range(k):                                        # k is small (<= 5), not N
      nbr = knn_idx[:, j]                                     # (N,)
      cand = d_direct[q_rows, nbr][:, None] + geo_anchors[nbr, :]  # (N, K)
      np.minimum(d_geo, cand, out=d_geo)

    # Fall back to direct L2 for unreachable anchors (disconnected components)
    inf_mask = ~np.isfinite(d_geo)
    if inf_mask.any():
      d_geo[inf_mask] = d_direct[inf_mask]

    d = d_geo
  else:
    from scipy.spatial.distance import cdist
    if sim_type == 'l_inf':
      d = cdist(z, anchors, metric='chebyshev')
    else:
      p_map = {'l1': 1, 'l2': 2}
      d = cdist(z, anchors, metric='minkowski', p=p_map[sim_type])

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


_ACTIVATIONS = {
  'gelu':       torch.nn.GELU,
  'relu':       torch.nn.ReLU,
  'silu':       torch.nn.SiLU,
  'leaky_relu': torch.nn.LeakyReLU,
}


def _build_projector_network(d_old, d_new, kind, activation):
  """
  Build the projector network mapping old-space embeddings (D_old) to new-space (D_new).

  Args:
    d_old      (int): Input (old model) embedding dimension.
    d_new      (int): Output (new model) embedding dimension.
    kind       (str): 'linear' → single nn.Linear(D_old, D_new);
                      'mlp'    → Linear(D_old, D_new) → activation → Linear(D_new, D_new),
                      a strict generalization of the linear projector (layer 1 is the
                      linear map, followed by a nonlinear refinement in the target space).
    activation (str | None): Activation name (a key of _ACTIVATIONS); only used for 'mlp'.

  Returns:
    torch.nn.Module: The projector network (output is unconstrained real-valued).

  Raises:
    ValueError: If kind is unknown or activation (for 'mlp') is not in _ACTIVATIONS.
  """
  if kind == 'linear':
    return torch.nn.Linear(d_old, d_new)
  if kind == 'mlp':
    if activation not in _ACTIVATIONS:
      raise ValueError(f'Unknown mlp activation: {activation!r} (choices: {list(_ACTIVATIONS)})')
    return torch.nn.Sequential(
      torch.nn.Linear(d_old, d_new),
      _ACTIVATIONS[activation](),
      torch.nn.Linear(d_new, d_new),
    )
  raise ValueError(f'Unknown projector network kind: {kind!r}')


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


def _train_linear_projector(old_anchors, new_anchors, df_anch, val_pool, projector_dir,
                            anchor_key_tag, kind='linear', activation='gelu', cfg=None):
  """
  Train a learned projector mapping old anchor embeddings to new anchor embeddings.

  Handles both the 'linear' projector (single nn.Linear) and the 'mlp' projector
  (Linear→activation→Linear); the network is built via _build_projector_network and
  the rest of the training loop is shared.

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
    kind           (str): 'linear' or 'mlp' (selects the projector network).
    activation     (str): Activation name for the 'mlp' network (ignored for 'linear').
    cfg            (dict | None): Projector recipe (LINEAR_PROJECTOR_CONFIG-shaped).
                           Defaults to the module-global LINEAR_PROJECTOR_CONFIG.

  Returns:
    dict: keys 'projector', 'norm_stats', 'splits', 'metrics', 'best_epoch',
          'best_val_mse' (legacy — always populated with MSE at best_epoch),
          'best_val_metric' (value of the metric used for selection),
          'best_val_metric_name' ('mse' | 'mae' | 'cos'), 'ckpt_path', 'config'.
  """
  from torch.utils.data import DataLoader, TensorDataset
  cfg = copy.deepcopy(LINEAR_PROJECTOR_CONFIG if cfg is None else cfg)
  cfg['mlp_activation'] = activation  # record which activation was used (None-effect for 'linear')
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
  projector = _build_projector_network(d_old, d_new, kind, activation).to(device)
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
  projector_cpu = _build_projector_network(d_old, d_new, kind, activation)
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
    'kind':                 kind,
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


# ---------------------------------------------------------------------------
# Post-projection refinement (REFINEMENT_CONFIG)
# ---------------------------------------------------------------------------

def _mae_micro_macro(preds, labels):
  """
  Compute micro- and macro-averaged MAE (mirrors cross_space_logs._compute_global_mae,
  re-implemented locally to avoid importing the heavy plotting module).

  Args:
    preds  (np.ndarray): Shape (N,) or (N, 1), predictions in the real label scale.
    labels (np.ndarray): Shape (N,), float ground-truth labels.

  Returns:
    tuple[float, float]: (micro_mae, macro_mae). micro = mean |pred-label| over all
      samples; macro = mean of per-rounded-class MAEs (each class weighted equally).
  """
  preds  = np.asarray(preds,  dtype=np.float32).reshape(-1)
  labels = np.asarray(labels, dtype=np.float32).reshape(-1)
  micro  = float(np.mean(np.abs(preds - labels)))
  labels_int = np.round(labels).astype(int)
  per_class  = [
    float(np.mean(np.abs(preds[labels_int == c] - labels[labels_int == c])))
    for c in np.unique(labels_int)
  ]
  macro = float(np.mean(per_class))
  return micro, macro


def _refine_reg_loss_fn(name):
  """
  Resolve the refinement regression-loss name to a callable (pred, target) → scalar.

  Args:
    name (str): 'mse' | 'mae' | 'huber'.

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
  if name == 'huber':
    return F.smooth_l1_loss
  raise ValueError(f'Unknown refinement reg loss: {name!r}')


def _build_refinement_optimizer(projector, linear, cfg, linear_only=False):
  """
  Build the optimizer for the refinement stage. In the default ('projector_linear') mode
  the projector and the linear layer get separate learning rates (cfg['lr_projector'],
  cfg['lr_linear']). In 'linear_only' mode the projection is frozen, so only the linear
  layer is optimized (at cfg['lr_linear']) and the projector group is omitted.

  Args:
    projector   (torch.nn.Module | None): The projector being refined. May be None when
                                          linear_only=True (the projection is fixed/external).
    linear      (torch.nn.Module): The (copied) new-model head.linear being refined.
    cfg         (dict): REFINEMENT_CONFIG.
    linear_only (bool): If True, optimize only the linear layer (single param group).

  Returns:
    torch.optim.Optimizer

  Raises:
    ValueError: If cfg['optimizer'] is unknown.
  """
  if linear_only:
    groups = [{'params': list(linear.parameters()), 'lr': cfg['lr_linear']}]
  else:
    groups = [
      {'params': list(projector.parameters()), 'lr': cfg['lr_projector']},
      {'params': list(linear.parameters()),    'lr': cfg['lr_linear']},
    ]
  name = cfg['optimizer'].lower()
  if name == 'adam':
    return torch.optim.Adam(groups, weight_decay=cfg['weight_decay'])
  if name == 'adamw':
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
  if name == 'sgd':
    return torch.optim.SGD(groups, weight_decay=cfg['weight_decay'])
  raise ValueError(f"Unknown refinement optimizer: {cfg['optimizer']!r}")


def _norm_tensors(norm_stats, device):
  """
  Pack projector normalization stats as no-grad float tensors on `device`.

  Args:
    norm_stats (dict | None): {'old_mean','old_std','new_mean','new_std'} or None.
    device     (torch.device): Target device.

  Returns:
    dict | None: Same keys with torch.Tensor values, or None when norm_stats is None
      (projector trained without normalization).
  """
  if norm_stats is None:
    return None
  return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
          for k, v in norm_stats.items()}


def _project_to_new_raw(projector, x_old, nt):
  """
  Differentiably map raw old-space embeddings to raw new-space embeddings via the
  projector, applying its fixed input normalization and output un-normalization (so the
  result is in the same raw space the new model's head.linear expects).

  Args:
    projector (torch.nn.Module): Projector mapping normalized old → normalized new.
    x_old     (torch.Tensor): Raw old-model embeddings, shape (B, D_old).
    nt        (dict | None): Output of _norm_tensors; None = no normalization.

  Returns:
    torch.Tensor: Raw new-space embeddings, shape (B, D_new).
  """
  x = (x_old - nt['old_mean']) / nt['old_std'] if nt is not None else x_old
  y = projector(x)
  if nt is not None:
    y = y * nt['new_std'] + nt['new_mean']
  return y


def _linear_preds(emb, linear, label_denorm):
  """
  Classify raw new-space embeddings with `linear` and return per-sample predictions.

  Args:
    emb          (np.ndarray): Shape (N, D_new), raw new-space embeddings.
    linear       (torch.nn.Module): Classifier (new model's head.linear, possibly refined).
    label_denorm (float): Multiplier mapping normalized logits to the real label scale.

  Returns:
    np.ndarray: Shape (N,), predictions in the real label scale.
  """
  device = next(linear.parameters()).device
  with torch.no_grad():
    logits = linear(torch.as_tensor(emb, dtype=torch.float32, device=device)).cpu().numpy()
  preds = (logits.squeeze(-1) if logits.ndim > 1 else logits) * label_denorm
  return np.asarray(preds, dtype=np.float32).reshape(-1)


def _eval_linear_mae(emb, linear, labels, label_denorm):
  """
  Classify raw new-space embeddings with `linear` and return (micro, macro) MAE.

  Args:
    emb          (np.ndarray): Shape (N, D_new), raw new-space embeddings.
    linear       (torch.nn.Module): Classifier (new model's head.linear, possibly refined).
    labels       (np.ndarray): Shape (N,), real-scale ground-truth labels.
    label_denorm (float): Multiplier mapping normalized logits to the real label scale.

  Returns:
    tuple[float, float]: (micro_mae, macro_mae).
  """
  return _mae_micro_macro(_linear_preds(emb, linear, label_denorm), labels)


def _eval_proj_mae(old_emb, projector, norm_stats, linear, labels, label_denorm):
  """
  Project old-space embeddings then classify with `linear`; return (micro, macro) MAE.

  Args:
    old_emb      (np.ndarray): Shape (N, D_old), raw old-model embeddings.
    projector    (torch.nn.Module): Old→new projector.
    norm_stats   (dict | None): Projector normalization stats.
    linear       (torch.nn.Module): Classifier (new model's head.linear, possibly refined).
    labels       (np.ndarray): Shape (N,), real-scale ground-truth labels.
    label_denorm (float): Multiplier mapping normalized logits to the real label scale.

  Returns:
    tuple[float, float]: (micro_mae, macro_mae).
  """
  projected = _apply_linear_projector(projector, norm_stats, old_emb)
  return _eval_linear_mae(projected, linear, labels, label_denorm)


def _guard_heldout_val(val_set, train_sample_ids, name, min_keep_frac):
  """
  Verify a validation set is disjoint from a training set (by sample_id) and, if not,
  drop the overlapping rows so the survivors form a genuinely held-out subset.

  Overlap is detected the same way as the anchor/val.csv check (_train_linear_projector):
  by intersecting integer sample_ids. This catches the realistic leakage cases — model B's
  val split sharing rows with its train split, or the new model's preserve-eval split sharing
  rows with the anchors used as the preservation training signal.

  Args:
    val_set          (dict): Validation embeddings dict with keys 'embeddings' (N, D),
                             'labels' (N,), 'sample_ids' (N,) (output of _extract_embeddings).
    train_sample_ids (np.ndarray): Sample ids used for training that the val set must not contain.
    name             (str): Short identifier for the check (used in the report and logs).
    min_keep_frac    (float): If the surviving fraction after exclusion is below this, the val
                             set is judged unusable and None is returned (status 'invalid').

  Returns:
    tuple[dict | None, dict]: (clean_val_set, report). clean_val_set is the original set when
      there is no overlap ('ok'), a row-filtered copy when overlap is removed and enough remain
      ('excluded'), or None when too few survive ('invalid'). report is a JSON-serializable dict
      describing the check (name, sizes, overlap/kept counts and fractions, status, and a capped
      list of overlapping sample_ids).
  """
  val_sids   = np.asarray(val_set['sample_ids']).astype(np.int64)
  train_sids = set(np.asarray(train_sample_ids).astype(np.int64).tolist())
  val_size   = int(val_sids.shape[0])
  keep_mask  = ~np.isin(val_sids, np.asarray(sorted(train_sids), dtype=np.int64))
  kept_count = int(keep_mask.sum())
  overlap_count = val_size - kept_count
  overlap_sids  = sorted(set(val_sids[~keep_mask].tolist()))
  kept_frac     = (kept_count / val_size) if val_size > 0 else 0.0

  report = {
    'name':               name,
    'val_size':           val_size,
    'overlap_count':      overlap_count,
    'overlap_frac':       (overlap_count / val_size) if val_size > 0 else 0.0,
    'kept_count':         kept_count,
    'kept_frac':          kept_frac,
    'overlap_sample_ids': overlap_sids[:100],  # capped to keep the report small
  }

  if overlap_count == 0:
    report['status'] = 'ok'
    return val_set, report
  if kept_frac < min_keep_frac:
    report['status'] = 'invalid'
    return None, report

  report['status'] = 'excluded'
  clean = dict(val_set)
  for k in ('embeddings', 'labels', 'sample_ids'):
    if k in clean:
      clean[k] = np.asarray(clean[k])[keep_mask]
  return clean, report


def _write_leakage_report(refine_dir, tag, checks, selection_used, min_keep_frac):
  """
  Write a validation-leakage report (JSON + human-readable TXT) into the experiment folder,
  but only when at least one check is not 'ok' — so the mere presence of the file flags that a
  held-out validation set was not respected during refinement.

  Args:
    refine_dir     (str): Refinement output directory (the files are written here).
    tag            (str): Short slug identifying this refinement run.
    checks         (list[dict]): Per-check report dicts from _guard_heldout_val.
    selection_used (str): Which selection criterion the run ended up using
                          ('balanced_val' or 'train_loss').
    min_keep_frac  (float): Threshold below which an excluded set is judged invalid.

  Returns:
    bool: True if a report was written (a leak existed), False otherwise.
  """
  if all(c.get('status') == 'ok' for c in checks):
    return False

  payload = {
    'tag':            tag,
    'min_keep_frac':  min_keep_frac,
    'selection_used': selection_used,
    'checks':         checks,
  }
  json_path = os.path.join(refine_dir, 'validation_leakage_report.json')
  txt_path  = os.path.join(refine_dir, 'validation_leakage_report.txt')
  with open(json_path, 'w') as f:
    json.dump(payload, f, indent=2)

  lines = [
    'VALIDATION LEAKAGE REPORT',
    f'  tag:            {tag}',
    f'  selection_used: {selection_used}',
    f'  min_keep_frac:  {min_keep_frac}',
    '',
  ]
  for c in checks:
    lines.append(f"  [{c['status'].upper()}] {c['name']}")
    lines.append(f"      val_size={c['val_size']}  overlap={c['overlap_count']} "
                 f"({c['overlap_frac']:.1%})  kept={c['kept_count']} ({c['kept_frac']:.1%})")
    if c['overlap_sample_ids']:
      shown = ', '.join(str(s) for s in c['overlap_sample_ids'])
      more  = ' ...' if c['overlap_count'] > len(c['overlap_sample_ids']) else ''
      lines.append(f"      overlapping sample_ids: {shown}{more}")
  with open(txt_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

  print(f"  [refine:LEAKAGE] {tag}: validation set(s) not held out — report written to "
        f"{json_path} (selection_used={selection_used})")
  for c in checks:
    if c['status'] != 'ok':
      print(f"    - {c['name']}: {c['status']} (overlap {c['overlap_count']}/{c['val_size']})")
  return True


def _refine_projector_and_linear(projector, norm_stats, head_linear,
                                 emb_B, labels_B, new_anchor_emb, anchor_labels,
                                 old_anchor_emb, label_denorm, refine_dir, tag, cfg=None,
                                 emb_B_val=None, labels_B_val=None,
                                 new_eval_emb=None, new_eval_labels=None,
                                 linear_only=False):
  """
  Second-stage refinement. By default ('projector_linear') it jointly fine-tunes the projector
  and a COPY of the new model's head.linear so source-domain (model B) accuracy improves while
  the new model's own-domain accuracy is preserved. In 'linear_only' mode the projection is held
  FIXED (no projector is trained) and only a COPY of head.linear is fine-tuned — in that case the
  caller has already mapped emb_B/emb_B_val into the NEW space (D_new), so projector=norm_stats=
  None and no projection is applied inside this function.

  Loss (per batch, real label scale):
    lambda_B * reg(linear(proj(emb_B)),    labels_B)        # improve source (model B)
  + lambda_A * reg(linear(new_anchor_emb), anchor_labels)   # preserve new-domain (model A)
  where proj() is the projector ('projector_linear') or the identity ('linear_only', emb_B is
  already in the new space).

  The originals are never mutated: "before" snapshots are deepcopies, and the trainable
  "after" copies are separate deepcopies.

  Epoch selection: when held-out validation sets are supplied (emb_B_val + new_eval_emb), the
  epoch with the lowest BALANCED validation loss (lambda_B * val_B + lambda_A * val_A, the same
  weighted objective as training but on held-out data) is restored. When they are omitted (the
  backward-compatible path), the epoch with the lowest total TRAINING loss is restored instead.
  All epochs are always run (no early stopping).

  Args:
    projector      (torch.nn.Module | None): Trained projector (old→new), start point. None when
                                 linear_only=True (the projection is fixed/applied by the caller).
    norm_stats     (dict | None): Projector normalization stats ({'old_mean',...} or None).
    head_linear    (torch.nn.Module): The new model's original head.linear (copied, not mutated).
    emb_B          (np.ndarray): Shape (N_B, D_old) raw old-model embeddings ('projector_linear'),
                                 or (N_B, D_new) already-projected embeddings ('linear_only'), on
                                 the model B training split.
    labels_B       (np.ndarray): Shape (N_B,), real-scale labels for emb_B.
    new_anchor_emb (np.ndarray): Shape (K, D_new), REAL new-model anchor embeddings (raw).
    anchor_labels  (np.ndarray): Shape (K,), real-scale anchor labels.
    old_anchor_emb (np.ndarray): Shape (K, D_old), old-model anchor embeddings (for the
                                 projector embedding-MSE drift metric only; ignored if linear_only).
    label_denorm   (float): Multiplier mapping normalized logits to the real label scale.
    refine_dir     (str): Directory for the checkpoints + refinement_metrics.csv.
    tag            (str): Short slug used in log lines / progress bar.
    cfg            (dict | None): REFINEMENT_CONFIG override (defaults to the module global).
    emb_B_val      (np.ndarray | None): Held-out model-B embeddings for the source (B) validation
                                 term — (Nv, D_old) for 'projector_linear', (Nv, D_new) for
                                 'linear_only'. None disables validation selection.
    labels_B_val   (np.ndarray | None): Shape (Nv,), real-scale labels for emb_B_val.
    new_eval_emb   (np.ndarray | None): Shape (Mv, D_new), held-out new-model embeddings for the
                                 preserve (A) validation term. None disables validation selection.
    new_eval_labels(np.ndarray | None): Shape (Mv,), real-scale labels for new_eval_emb.
    linear_only    (bool): If True, freeze the projection and fine-tune only head.linear.

  Returns:
    dict: keys 'projector_before', 'projector_after' (cpu nn.Modules, both None when linear_only),
      'linear_before', 'linear_after' (cpu nn.Modules), 'metrics' (per-epoch list), 'best_epoch',
      'best_val_total' (None when train-loss selection was used), 'selection_used'
      ('balanced_val' | 'train_loss'), 'proj_anchor_loss_before', 'proj_anchor_loss_after'
      (NaN when linear_only), 'ckpt_paths', 'config', 'tag'.
  """
  import torch.nn.functional as F
  from torch.utils.data import DataLoader, TensorDataset
  cfg = copy.deepcopy(REFINEMENT_CONFIG if cfg is None else cfg)
  device = torch.device(cfg['device'])
  os.makedirs(refine_dir, exist_ok=True)

  # Frozen "before" snapshots (deepcopies; the originals stay untouched). In linear_only
  # mode the projection is fixed/external, so there is no projector to snapshot or train.
  projector_before = None if linear_only else copy.deepcopy(projector).to('cpu').eval()
  linear_before    = copy.deepcopy(head_linear).to('cpu').eval()

  # Trainable "after" copies on device.
  projector_after = None if linear_only else copy.deepcopy(projector).to(device).train()
  linear_after    = copy.deepcopy(head_linear).to(device).train()

  nt        = _norm_tensors(norm_stats, device)
  reg       = _refine_reg_loss_fn(cfg['loss'])
  optimizer = _build_refinement_optimizer(projector_after, linear_after, cfg, linear_only=linear_only)

  # Anchor preserve tensors (real new-model embeddings → linear → labels).
  new_anchor_t = torch.as_tensor(new_anchor_emb, dtype=torch.float32, device=device)
  anchor_lab_t = torch.as_tensor(np.asarray(anchor_labels, np.float32).reshape(-1),
                                 dtype=torch.float32, device=device)

  # Source (model B) loader: raw old-model embeddings + labels, preloaded to device.
  emb_B_t = torch.as_tensor(emb_B, dtype=torch.float32, device=device)
  labB_t  = torch.as_tensor(np.asarray(labels_B, np.float32).reshape(-1),
                            dtype=torch.float32, device=device)
  loader  = DataLoader(TensorDataset(emb_B_t, labB_t), batch_size=cfg['batch_size'],
                       shuffle=True, num_workers=0, pin_memory=False)

  def _pred_real(linear, emb):
    out = linear(emb)
    out = out.squeeze(-1) if out.dim() > 1 else out
    return out * label_denorm

  def _proj_anchor_loss(proj):
    # Projector embedding-MSE in normalized space (matches the projector's own loss).
    dev = next(proj.parameters()).device
    xo  = torch.as_tensor(old_anchor_emb, dtype=torch.float32, device=dev)
    yn  = torch.as_tensor(new_anchor_emb, dtype=torch.float32, device=dev)
    if norm_stats is not None:
      xo = (xo - torch.as_tensor(norm_stats['old_mean'], dtype=torch.float32, device=dev)) \
              / torch.as_tensor(norm_stats['old_std'], dtype=torch.float32, device=dev)
      yn = (yn - torch.as_tensor(norm_stats['new_mean'], dtype=torch.float32, device=dev)) \
              / torch.as_tensor(norm_stats['new_std'], dtype=torch.float32, device=dev)
    with torch.no_grad():
      pred = proj(xo)
    return float(F.mse_loss(pred, yn).item())

  # Projector embedding-drift diagnostic is meaningless when the projection is frozen.
  proj_anchor_loss_before = float('nan') if linear_only else _proj_anchor_loss(projector_before)

  lam_b, lam_a = cfg['lambda_B'], cfg['lambda_A']

  # Held-out validation tensors (None → train-loss selection, backward compatible).
  has_val = emb_B_val is not None and new_eval_emb is not None
  if has_val:
    emb_B_val_t  = torch.as_tensor(emb_B_val, dtype=torch.float32, device=device)
    labB_val_t   = torch.as_tensor(np.asarray(labels_B_val, np.float32).reshape(-1),
                                   dtype=torch.float32, device=device)
    new_eval_t   = torch.as_tensor(new_eval_emb, dtype=torch.float32, device=device)
    new_eval_lab_t = torch.as_tensor(np.asarray(new_eval_labels, np.float32).reshape(-1),
                                     dtype=torch.float32, device=device)

  def _validate():
    # Per-epoch held-out evaluation mirroring the two training terms.
    if not linear_only:
      projector_after.eval()
    linear_after.eval()
    with torch.no_grad():
      # linear_only: emb_B_val is already in the new space, so the projection is the identity.
      proj_val = emb_B_val_t if linear_only else _project_to_new_raw(projector_after, emb_B_val_t, nt)
      val_B    = float(reg(_pred_real(linear_after, proj_val), labB_val_t).item())
      val_A    = float(reg(_pred_real(linear_after, new_eval_t), new_eval_lab_t).item())
    if linear_only:
      val_mae_micro_B, val_mae_macro_B = _eval_linear_mae(
        emb_B_val, linear_after, labels_B_val, label_denorm)
    else:
      val_mae_micro_B, val_mae_macro_B = _eval_proj_mae(
        emb_B_val, projector_after, norm_stats, linear_after, labels_B_val, label_denorm)
    val_mae_micro_A, val_mae_macro_A = _eval_linear_mae(
      new_eval_emb, linear_after, new_eval_labels, label_denorm)
    return {
      'val_B': val_B, 'val_A': val_A, 'val_total': lam_b * val_B + lam_a * val_A,
      'val_mae_micro_B': val_mae_micro_B, 'val_mae_macro_B': val_mae_macro_B,
      'val_mae_micro_A': val_mae_micro_A, 'val_mae_macro_A': val_mae_macro_A,
    }

  selection_used = 'balanced_val' if has_val else 'train_loss'
  metrics, best_score, best_epoch = [], float('inf'), -1
  best_val_total = None
  best_proj_sd = best_lin_sd = None

  for epoch in tqdm.tqdm(range(1, cfg['epochs'] + 1), desc=f'Refinement ({tag})'):
    if not linear_only:
      projector_after.train()
    linear_after.train()
    sums = {'total': 0.0, 'B': 0.0, 'A': 0.0}
    n_batches = 0
    for xb, yb in loader:
      # linear_only: xb is already in the new space (projection is fixed/external).
      proj_raw = xb if linear_only else _project_to_new_raw(projector_after, xb, nt)
      loss_b   = reg(_pred_real(linear_after, proj_raw), yb)
      loss_a   = reg(_pred_real(linear_after, new_anchor_t), anchor_lab_t)
      loss     = lam_b * loss_b + lam_a * loss_a
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      sums['total'] += float(loss.item())
      sums['B']     += float(loss_b.item())
      sums['A']     += float(loss_a.item())
      n_batches += 1
    n = max(n_batches, 1)
    row = {'epoch': epoch, 'loss_total': sums['total'] / n,
           'loss_B': sums['B'] / n, 'loss_A': sums['A'] / n}
    # Selection score: balanced held-out val loss when available, else train total loss.
    if has_val:
      row.update(_validate())
      score = row['val_total']
    else:
      score = row['loss_total']
    metrics.append(row)
    if score < best_score:
      best_score = score
      best_epoch = epoch
      best_val_total = row['val_total'] if has_val else None
      best_proj_sd = (None if linear_only else
                      {k: v.detach().cpu().clone() for k, v in projector_after.state_dict().items()})
      best_lin_sd  = {k: v.detach().cpu().clone() for k, v in linear_after.state_dict().items()}

  if best_lin_sd is not None:
    if not linear_only:
      projector_after.load_state_dict(best_proj_sd)
    linear_after.load_state_dict(best_lin_sd)
  if not linear_only:
    projector_after.eval()
  linear_after.eval()
  proj_anchor_loss_after = float('nan') if linear_only else _proj_anchor_loss(projector_after)

  if has_val:
    sel_str = f"val_total={best_val_total:.6f} (balanced_val)"
  else:
    sel_str = f"train_total={best_score:.6f} (train_loss)"
  print(f"  [refine:{tag}] best epoch {best_epoch}  {sel_str}  "
        f"proj_anchor_loss {proj_anchor_loss_before:.6f} → {proj_anchor_loss_after:.6f}")

  projector_after_cpu = None if linear_only else projector_after.to('cpu').eval()
  linear_after_cpu    = linear_after.to('cpu').eval()
  # In linear_only mode the projection is fixed, so no projector checkpoints are written.
  ckpt_paths = {
    'projector_before': None if linear_only else os.path.join(refine_dir, 'projector_before.pt'),
    'projector_after':  None if linear_only else os.path.join(refine_dir, 'projector_after.pt'),
    'linear_before':    os.path.join(refine_dir, 'linear_before.pt'),
    'linear_after':     os.path.join(refine_dir, 'linear_after.pt'),
  }
  if not linear_only:
    torch.save(projector_before.state_dict(),    ckpt_paths['projector_before'])
    torch.save(projector_after_cpu.state_dict(), ckpt_paths['projector_after'])
  torch.save(linear_before.state_dict(),       ckpt_paths['linear_before'])
  torch.save(linear_after_cpu.state_dict(),    ckpt_paths['linear_after'])
  pd.DataFrame(metrics).to_csv(os.path.join(refine_dir, 'refinement_metrics.csv'), index=False)

  return {
    'projector_before': projector_before,
    'projector_after':  projector_after_cpu,
    'linear_before':    linear_before,
    'linear_after':     linear_after_cpu,
    'metrics':          metrics,
    'best_epoch':       best_epoch,
    'best_val_total':   best_val_total,
    'selection_used':   selection_used,
    'proj_anchor_loss_before': proj_anchor_loss_before,
    'proj_anchor_loss_after':  proj_anchor_loss_after,
    'ckpt_paths':       ckpt_paths,
    'config':           cfg,
    'tag':              tag,
  }


def _run_refinement_stage(old_model, new_model, old_model_pth, new_model_pth,
                          old_config, new_config, old_features_path,
                          old_model_anchors, new_model_anchors_aligned, projector_bundle,
                          old_model_tensors, old_model_csv, label_denorm, refine_dir, tag,
                          emb_B=None, new_eval=None, new_test=None, emb_B_val=None,
                          mode=None, sim_type=None, rbf_sigma=None, cfg=None):
  """
  Orchestrate the refinement stage end to end: extract the model-B training embeddings
  and the new-model evaluation embeddings (unless supplied pre-computed), run
  _refine_projector_and_linear, and assemble the before/after metric block.

  Two modes (driven by REFINEMENT_CONFIG['mode'], overridable via `mode`):
    'projector_linear': jointly refine the trained projector + head.linear (projector_bundle
                        required).
    'linear_only':      hold the projection FIXED and refine only head.linear. The fixed
                        projection is either projector_bundle['projector'] (projector kinds) or
                        the distance-metric interpolation (sim_type/rbf_sigma over the anchors);
                        emb_B/emb_B_val are mapped into the new space here before refinement.

  Args:
    old_model, new_model         (Model_Advanced): Source and target models.
    old_model_pth, new_model_pth (str): Their checkpoints.
    old_config, new_config       (dict): Their k_fold_results configs.
    old_features_path            (str): Old model features folder (for restoring state_shift).
    old_model_anchors            (dict): _extract_embeddings output for old anchors
                                         (keys 'embeddings' (K,D_old), 'labels').
    new_model_anchors_aligned    (dict): Aligned new anchors (keys 'embeddings' (K,D_new)).
    projector_bundle             (dict | None): Output of a _train_*_projector (keys 'projector',
                                         'norm_stats', ...). None for distance-metric linear_only.
    old_model_tensors            (dict): Old embeddings on `old_model_csv` (the eval split).
    old_model_csv                (str): The split name being projected/evaluated.
    label_denorm                 (float): Logit→real-label multiplier.
    refine_dir                   (str): Output directory for checkpoints + logs.
    tag                          (str): Short slug for logs.
    emb_B  (dict | None): Pre-extracted model-B (old train) embeddings {'embeddings','labels'};
                          extracted here on old model's `refine_split` when None.
    new_eval (dict | None): Pre-extracted new-model eval embeddings {'embeddings','labels'};
                            extracted here on new model's `new_eval_split` when None.
    new_test (dict | None): Pre-extracted new-model TEST split embeddings
                            {'embeddings','labels','sample_ids'}; when None it is extracted here
                            via _resolve_test_csv_strict (raises if test.csv is missing — no val
                            fallback). Used only for the per-class before/after refinement diagnostic.
    emb_B_val (dict | None): Pre-extracted model-B held-out embeddings {'embeddings','labels',
                            'sample_ids'} on old model's `refine_val_split`; extracted here when None.
                            Used (after a leakage guard) for the per-epoch source validation term.
    mode      (str | None): 'projector_linear' | 'linear_only'. Defaults to REFINEMENT_CONFIG['mode'].
    sim_type  (str | None): Distance metric for the fixed interpolation (linear_only + no
                            projector_bundle), e.g. 'cos'/'l1'/'l2'/'l_inf'/'geodesic'.
    rbf_sigma (float | None): RBF sigma for that interpolation.
    cfg       (dict | None): Refinement recipe (REFINEMENT_CONFIG-shaped). The swept
                            numeric/loss fields drive _refine_projector_and_linear; the
                            fixed split fields are read here. Defaults to the global.

  Returns:
    dict: 'refine_bundle' (the _refine_projector_and_linear output), a flat 'metrics' dict ready
      to merge into the results pkl / summary.csv, and 'new_test_eval' — per-sample test-split
      predictions before/after refinement for the per-class improvement plot.
  """
  cfg = REFINEMENT_CONFIG if cfg is None else cfg
  mode = cfg['mode'] if mode is None else mode
  linear_only = (mode == 'linear_only')
  os.makedirs(refine_dir, exist_ok=True)

  # --- model-B training embeddings (old model, old domain, refine_split) ---
  if emb_B is None:
    b_csv = _resolve_split_csv(old_model_pth, cfg['refine_split'])
    assert os.path.isfile(b_csv), f'Refinement model-B split CSV not found: {b_csv}'
    helper.set_step_shift(old_features_path)
    emb_B = _extract_embeddings(old_model, old_model_pth, b_csv, old_config)

  # --- model-B held-out validation embeddings (old model, refine_val_split) ---
  if emb_B_val is None:
    bv_csv = _resolve_split_csv(old_model_pth, cfg['refine_val_split'])
    assert os.path.isfile(bv_csv), f'Refinement model-B val split CSV not found: {bv_csv}'
    helper.set_step_shift(old_features_path)
    emb_B_val = _extract_embeddings(old_model, old_model_pth, bv_csv, old_config)

  # --- new-model evaluation embeddings (new model, its own new_eval_split) ---
  if new_eval is None:
    e_csv = _resolve_split_csv(new_model_pth, cfg['new_eval_split'])
    assert os.path.isfile(e_csv), f'Refinement new-eval split CSV not found: {e_csv}'
    new_eval = _extract_embeddings(new_model, new_model_pth, e_csv, new_config)

  # --- new-model TEST split embeddings (for the per-class before/after diagnostic). The test
  #     split is required: no val fallback, so a missing test.csv raises instead of silently
  #     comparing against val. ---
  if new_test is None:
    t_csv = _resolve_test_csv_strict(new_model_pth)
    new_test = _extract_embeddings(new_model, new_model_pth, t_csv, new_config)

  # --- Held-out validation guard: the source-val split must be disjoint from emb_B (train),
  #     and the preserve-val split (new_eval) must be disjoint from the anchors (the term-A
  #     training signal). Overlap rows are auto-excluded; if too few survive the term is dropped
  #     and selection falls back to train loss. A report is written only when a leak exists. ---
  src_val,  rep_src  = _guard_heldout_val(
    emb_B_val, emb_B['sample_ids'], 'source_val (B val vs train)', cfg['refine_val_min_keep_frac'])
  prsv_val, rep_prsv = _guard_heldout_val(
    new_eval, old_model_anchors['sample_ids'], 'preserve_val (new_eval vs anchors)',
    cfg['refine_val_min_keep_frac'])
  use_val = src_val is not None and prsv_val is not None
  selection_used = 'balanced_val' if use_val else 'train_loss'
  _write_leakage_report(refine_dir, tag, [rep_src, rep_prsv], selection_used,
                        cfg['refine_val_min_keep_frac'])

  projector  = None if projector_bundle is None else projector_bundle['projector']
  norm_stats = None if projector_bundle is None else projector_bundle['norm_stats']
  head_linear = new_model.head.linear

  def _project_fixed(emb):
    """
    Map raw old-space embeddings (N, D_old) into the new space (N, D_new) using this run's
    FIXED projection: the trained projector when a bundle exists, else the distance-metric
    interpolation over the anchors. Used only in linear_only mode.
    """
    if projector_bundle is not None:
      return _apply_linear_projector(projector_bundle['projector'], norm_stats, emb)
    w = _compute_weights(emb, old_model_anchors['embeddings'], sim_type, rbf_sigma)
    return (w @ new_model_anchors_aligned['embeddings'].astype(np.float32)).astype(np.float32)

  if linear_only:
    # Pre-project emb_B / source-val into the new space; refine only head.linear.
    refine = _refine_projector_and_linear(
      projector=None, norm_stats=None, head_linear=head_linear,
      emb_B=_project_fixed(emb_B['embeddings']), labels_B=emb_B['labels'],
      new_anchor_emb=new_model_anchors_aligned['embeddings'],
      anchor_labels=old_model_anchors['labels'],
      old_anchor_emb=old_model_anchors['embeddings'],
      label_denorm=label_denorm, refine_dir=refine_dir, tag=tag,
      emb_B_val=(_project_fixed(src_val['embeddings']) if use_val else None),
      labels_B_val=(src_val['labels'] if use_val else None),
      new_eval_emb=(prsv_val['embeddings'] if use_val else None),
      new_eval_labels=(prsv_val['labels'] if use_val else None),
      linear_only=True, cfg=cfg,
    )
  else:
    refine = _refine_projector_and_linear(
      projector=projector, norm_stats=norm_stats, head_linear=head_linear,
      emb_B=emb_B['embeddings'], labels_B=emb_B['labels'],
      new_anchor_emb=new_model_anchors_aligned['embeddings'],
      anchor_labels=old_model_anchors['labels'],
      old_anchor_emb=old_model_anchors['embeddings'],
      label_denorm=label_denorm, refine_dir=refine_dir, tag=tag,
      emb_B_val=(src_val['embeddings'] if use_val else None),
      labels_B_val=(src_val['labels'] if use_val else None),
      new_eval_emb=(prsv_val['embeddings'] if use_val else None),
      new_eval_labels=(prsv_val['labels'] if use_val else None),
      cfg=cfg,
    )

  # --- before/after metrics ---
  # The new-model preserve eval + projector-drift loss depend only on the bundle (not on
  # old_model_csv). The old-on-csv metrics depend on the projected split, so they are
  # skipped here (left NaN) when old_model_tensors is None — the Optuna path fills them
  # per-trial against the stored refine bundle.
  if old_model_tensors is not None:
    old_emb = old_model_tensors['embeddings']
    old_lab = old_model_tensors['labels']
    if linear_only:
      old_proj = _project_fixed(old_emb)
      mae_old_micro_b, mae_old_macro_b = _eval_linear_mae(
        old_proj, refine['linear_before'], old_lab, label_denorm)
      mae_old_micro_a, mae_old_macro_a = _eval_linear_mae(
        old_proj, refine['linear_after'], old_lab, label_denorm)
    else:
      mae_old_micro_b, mae_old_macro_b = _eval_proj_mae(
        old_emb, refine['projector_before'], norm_stats, refine['linear_before'], old_lab, label_denorm)
      mae_old_micro_a, mae_old_macro_a = _eval_proj_mae(
        old_emb, refine['projector_after'], norm_stats, refine['linear_after'], old_lab, label_denorm)
  else:
    mae_old_micro_b = mae_old_macro_b = mae_old_micro_a = mae_old_macro_a = float('nan')
  mae_new_micro_b, mae_new_macro_b = _eval_linear_mae(
    new_eval['embeddings'], refine['linear_before'], new_eval['labels'], label_denorm)
  mae_new_micro_a, mae_new_macro_a = _eval_linear_mae(
    new_eval['embeddings'], refine['linear_after'], new_eval['labels'], label_denorm)

  # --- per-sample new-model test-split predictions (no projector: the test set is already in the
  #     new space) for the per-class before/after-refinement improvement plot. ---
  new_test_eval = {
    'split':        'test',
    'labels':       new_test['labels'].astype(np.float32),
    'sample_ids':   new_test['sample_ids'].astype(np.int64),
    'preds_before': _linear_preds(new_test['embeddings'], refine['linear_before'], label_denorm),
    'preds_after':  _linear_preds(new_test['embeddings'], refine['linear_after'], label_denorm),
  }

  metrics = {
    'refine_enabled':            True,
    'refine_mode':               mode,
    'refine_best_epoch':         refine['best_epoch'],
    'refine_best_val_total':     refine['best_val_total'],
    'refine_val_selection':      refine['selection_used'],
    'proj_anchor_loss_before':   refine['proj_anchor_loss_before'],
    'proj_anchor_loss_after':    refine['proj_anchor_loss_after'],
    'mae_micro_old_oncsv_before': mae_old_micro_b,
    'mae_macro_old_oncsv_before': mae_old_macro_b,
    'mae_micro_old_oncsv_after':  mae_old_micro_a,
    'mae_macro_old_oncsv_after':  mae_old_macro_a,
    'mae_micro_new_test_before':  mae_new_micro_b,
    'mae_macro_new_test_before':  mae_new_macro_b,
    'mae_micro_new_test_after':   mae_new_micro_a,
    'mae_macro_new_test_after':   mae_new_macro_a,
    'refine_old_model_csv':       old_model_csv,
    'refine_new_eval_split':      cfg['new_eval_split'],
    'projector_before_pth':       refine['ckpt_paths']['projector_before'],
    'projector_after_pth':        refine['ckpt_paths']['projector_after'],
    'linear_before_pth':          refine['ckpt_paths']['linear_before'],
    'linear_after_pth':           refine['ckpt_paths']['linear_after'],
  }
  return {'refine_bundle': refine, 'metrics': metrics, 'emb_B': emb_B, 'emb_B_val': emb_B_val,
          'new_eval': new_eval, 'new_test_eval': new_test_eval}


def _assert_linear_ckpt_matches_formula(weight, bias, x_norm, y_formula,
                                        kind, anchor_key_tag, rtol=1e-4):
  """
  Verify a saved nn.Linear (weight, bias) reproduces the closed-form affine map.

  Evaluates the nn.Linear map Y = x_norm @ weight.T + bias in float64 (so the
  check is deterministic and independent of GPU/TF32/backend float32 rounding)
  and compares it to the raw closed-form prediction y_formula with a scale-aware
  (relative) tolerance. This guards against real coding errors (wrong transpose /
  wrong bias formula) without false-failing on benign float32 differences. Both
  inputs derive from the same float32 closed-form solution, so the relative error
  is ~1e-6 in practice; rtol=1e-4 leaves headroom while still tripping on a
  genuine bug (which produces order-1+ relative error).

  Args:
    weight         (np.ndarray): nn.Linear weight, shape (D_new, D_old).
    bias           (np.ndarray): nn.Linear bias, shape (D_new,).
    x_norm         (np.ndarray): Input embeddings (normalized space), shape (N, D_old).
    y_formula      (np.ndarray): Raw closed-form prediction, shape (N, D_new).
    kind           (str):        'procrustes' | 'linear_close' (for the message).
    anchor_key_tag (str):        Short slug for the message.
    rtol           (float):      Relative tolerance on the max abs error.

  Raises:
    AssertionError: If the nn.Linear map and the formula disagree beyond rtol.
  """
  y_ref = np.asarray(y_formula, dtype=np.float64)
  if y_ref.size == 0:
    return
  x64 = np.asarray(x_norm, dtype=np.float64)
  y_lin = x64 @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias, dtype=np.float64)
  max_abs = float(np.max(np.abs(y_lin - y_ref)))
  scale = float(np.max(np.abs(y_ref)))
  rel = max_abs / scale if scale > 0 else max_abs
  assert rel < rtol, (
    f"[{kind}:{anchor_key_tag}] nn.Linear ckpt diverges from raw {kind} "
    f"formula (max abs err={max_abs:.3e}, rel err={rel:.3e}, rtol={rtol})"
  )


# ---------------------------------------------------------------------------
# Procrustes projector helpers (interpolation_similarity='procrustes')
# ---------------------------------------------------------------------------

def _fit_procrustes_solution(A, B):
  """
  Closed-form Orthogonal Procrustes fit with centering and isotropic scale.

  Finds (mu_A, mu_B, R, s) minimizing ||s * (A - mu_A) @ R + mu_B - B||_F over
  semi-orthogonal R (so R^T R = I when D_old >= D_new, else R R^T = I) and
  scalar s. Reflections are allowed (unconstrained orthogonal form), so
  det(R) is not constrained to +1.

  Steps:
    1. mu_A = A.mean(0),  mu_B = B.mean(0)
    2. A_c = A - mu_A,    B_c = B - mu_B
    3. M = A_c.T @ B_c                          shape (D_old, D_new)
    4. U, sigma, V_T = svd(M, full_matrices=False)
    5. R = U @ V_T                              shape (D_old, D_new)
    6. s = trace(Sigma) / ||A_c R||_F^2         optimal isotropic scale.
       NOTE: the often-quoted shortcut s = sum(sigma) / ||A_c||_F^2 holds only
       when R is square orthogonal (D_old == D_new); for the semi-orthogonal
       case we must use ||A_c R||_F^2 in the denominator.
    7. Y_hat = s * (X - mu_A) @ R + mu_B        (apply formula)
    8. Equivalent nn.Linear: weight = (s * R).T, bias = mu_B - s * mu_A @ R
       so that Y_hat = X @ weight.T + bias.

  Args:
    A (np.ndarray): Source anchors, shape (K, D_old), dtype float32.
    B (np.ndarray): Target anchors, shape (K, D_new), dtype float32.

  Returns:
    dict with keys:
      'mu_A'   (np.ndarray): Shape (D_old,).
      'mu_B'   (np.ndarray): Shape (D_new,).
      'R'      (np.ndarray): Shape (D_old, D_new), semi-orthogonal.
      'scale'  (float):      Optimal isotropic scale s.
      'weight' (np.ndarray): Shape (D_new, D_old), nn.Linear weight.
      'bias'   (np.ndarray): Shape (D_new,),       nn.Linear bias.
      'sigma'  (np.ndarray): Shape (min(D_old, D_new),), SVD singular values.
  """
  A = np.asarray(A, dtype=np.float32)
  B = np.asarray(B, dtype=np.float32)
  mu_A = A.mean(axis=0)
  mu_B = B.mean(axis=0)
  A_c = A - mu_A
  B_c = B - mu_B
  M = A_c.T @ B_c
  U, sigma, V_T = np.linalg.svd(M, full_matrices=False)
  R = (U @ V_T).astype(np.float32)
  AcR = A_c @ R
  denom = float((AcR ** 2).sum())
  scale = float(sigma.sum() / denom) if denom > 0 else 1.0
  weight = (scale * R).T.astype(np.float32)            # (D_new, D_old)
  bias   = (mu_B - scale * (mu_A @ R)).astype(np.float32)  # (D_new,)
  return {
    'mu_A':   mu_A.astype(np.float32),
    'mu_B':   mu_B.astype(np.float32),
    'R':      R,
    'scale':  scale,
    'weight': weight,
    'bias':   bias,
    'sigma':  sigma.astype(np.float32),
  }


def _train_procrustes_projector(old_anchors, new_anchors, df_anch, val_pool, projector_dir, anchor_key_tag, cfg=None):
  """
  Closed-form Orthogonal Procrustes projector that mirrors _train_linear_projector.

  The fit is closed-form on all K anchor pairs (no SGD, no epochs). To keep the
  returned bundle drop-in compatible with the existing _train_linear_projector
  bundle (so cross_space_logs.py plots work unchanged), metrics['train'] and
  metrics['val'] each carry a single 1-epoch entry, best_epoch=1, and the
  projector is saved as an nn.Linear state_dict whose forward exactly
  reproduces Y_hat = s * (X - mu_A) @ R + mu_B (in normalized space when
  normalize_embeddings is on).

  Args:
    old_anchors    (dict): Output of _extract_embeddings on old model.
    new_anchors    (dict): Output of _extract_embeddings on new model, aligned.
    df_anch        (pd.DataFrame): Anchor table with sample_id/subject_id/class_id.
    val_pool       (dict): Output of _extract_linear_val_pool.
    projector_dir  (str):  Output directory.
    anchor_key_tag (str):  Short slug used in log lines.
    cfg            (dict | None): Projector recipe; only normalize_embeddings /
                           split_ratios matter (closed-form). Defaults to the global.

  Returns:
    dict: Same shape as _train_linear_projector's return, plus:
      'kind'              ('procrustes')
      'procrustes_params' ({'mu_old', 'mu_new', 'scale', 'R'})
  """
  cfg = copy.deepcopy(LINEAR_PROJECTOR_CONFIG if cfg is None else cfg)
  device = torch.device(cfg['device'])

  os.makedirs(projector_dir, exist_ok=True)
  splits_dir = os.path.join(projector_dir, 'split_training_stage')
  os.makedirs(splits_dir, exist_ok=True)

  df_anch_aligned = _align_df_to_sample_ids(df_anch, old_anchors['sample_ids'], 'anchor df')
  val_df_aligned  = _align_df_to_sample_ids(
    val_pool['df'], val_pool['old']['sample_ids'], 'val.csv df',
  )

  anchor_sids = set(old_anchors['sample_ids'].astype(np.int64).tolist())
  val_sids    = set(val_pool['old']['sample_ids'].astype(np.int64).tolist())
  overlap = anchor_sids & val_sids
  if overlap:
    raise ValueError(
      f"[procrustes:{anchor_key_tag}] {len(overlap)} anchor sample_id(s) also appear "
      f"in the new model's val.csv: {sorted(overlap)[:10]}"
      f"{' ...' if len(overlap) > 10 else ''}. Pick a --csv_anchor_selection split "
      f"disjoint from 'val' (e.g. 'train')."
    )

  M = len(val_pool['old']['embeddings'])
  n_val = round(cfg['split_ratios'][1] * M)
  if n_val < 1 or n_val >= M:
    raise ValueError(
      f"[procrustes:{anchor_key_tag}] cannot build a val/test split from val.csv: "
      f"n_val={n_val} (val.csv rows={M}, split_ratios={cfg['split_ratios']})."
    )
  idx_va, idx_te = _make_subject_disjoint_subsets(val_df_aligned, n_val, _SEED)

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

  print(f"  [procrustes:{anchor_key_tag}] [{split_mode}] "
        f"train={len(split_arrays['train']['old'])} (all anchors, closed-form fit)  "
        f"val={len(idx_va)} test={len(idx_te)} (subject-disjoint subsets of val.csv, {M} rows)")
  print(f"  [procrustes:{anchor_key_tag}] unique subjects per split — "
        f"train={splits_df['train']['subject_id'].nunique()} "
        f"val={splits_df['val']['subject_id'].nunique()} "
        f"test={splits_df['test']['subject_id'].nunique()}")

  split_csv_paths = {}
  for name, sdf in splits_df.items():
    p = os.path.join(splits_dir, f'{name}.csv')
    sdf.to_csv(p, index=False, sep='\t')
    split_csv_paths[name] = p

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

  # --- Closed-form fit in (normalized) anchor space ---
  sol = _fit_procrustes_solution(
    split_arrays['train']['old_norm'],
    split_arrays['train']['new_norm'],
  )
  d_old = split_arrays['train']['old'].shape[1]
  d_new = split_arrays['train']['new'].shape[1]
  projector = torch.nn.Linear(d_old, d_new).to(device)
  with torch.no_grad():
    projector.weight.copy_(torch.from_numpy(sol['weight']))
    projector.bias.copy_(torch.from_numpy(sol['bias']))
  projector.eval()

  # Numerical equivalence check on training anchors (normalized space): the saved
  # weight/bias must encode Y_hat = s * (X - mu_A) @ R + mu_B. Done in float64 so it
  # is deterministic and not tripped by GPU/TF32 float32 rounding (see
  # _assert_linear_ckpt_matches_formula).
  y_formula = (
    sol['scale'] * (split_arrays['train']['old_norm'].astype(np.float64) - sol['mu_A'])
    @ sol['R'] + sol['mu_B']
  )
  _assert_linear_ckpt_matches_formula(
    sol['weight'], sol['bias'], split_arrays['train']['old_norm'],
    y_formula, 'procrustes', anchor_key_tag,
  )

  # --- Single-shot metrics on train/val/test ---
  def _eval(name):
    with torch.no_grad():
      pred = projector(torch.from_numpy(split_arrays[name]['old_norm']).to(device))
      tgt  = torch.from_numpy(split_arrays[name]['new_norm']).to(device)
      mse, mae, cos = _eval_projector_batch_metrics(pred, tgt)
    return {'mse': mse, 'mae': mae, 'cos': cos}

  tr_m = _eval('train')
  va_m = _eval('val')
  te_m = _eval('test')

  metrics = {
    'train': [{'epoch': 1, **tr_m}],
    'val':   [{'epoch': 1, **va_m}],
    'test':  te_m,
  }
  best_epoch = 1
  sel_key, _, _ = _projector_selection_rule(cfg['loss'])
  best_val_metric = va_m[sel_key]
  best_val_mse = va_m['mse']
  print(f"  [procrustes:{anchor_key_tag}] closed-form scale={sol['scale']:.4f}  "
        f"R.shape={tuple(sol['R'].shape)}  best_epoch=1  "
        f"val {sel_key.upper()}={best_val_metric:.6f}")
  print(f"  [procrustes:{anchor_key_tag}] test — MSE: {te_m['mse']:.6f}  "
        f"MAE: {te_m['mae']:.6f}  cos: {te_m['cos']:.6f}")

  ckpt_path = os.path.join(projector_dir, f'best_projector_{best_epoch}.pt')
  torch.save(projector.state_dict(), ckpt_path)

  projector_cpu = torch.nn.Linear(d_old, d_new)
  projector_cpu.load_state_dict(projector.state_dict())
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
    'kind':                 'procrustes',
    'procrustes_params': {
      'mu_old': sol['mu_A'],
      'mu_new': sol['mu_B'],
      'scale':  sol['scale'],
      'R':      sol['R'],
      'sigma':  sol['sigma'],
    },
  }


# ---------------------------------------------------------------------------
# Closed-form linear projector helpers (interpolation_similarity='linear_close')
# ---------------------------------------------------------------------------

def _fit_linear_closed_form(A, B, rcond=None):
  """
  Closed-form Ordinary Least Squares (OLS) fit of an affine map A -> B.

  Finds (W, b) minimizing the same Frobenius MSE that interpolation_similarity=
  'linear' descends by SGD, ||A @ W.T + b - B||_F^2, but in closed form. This is
  multivariate linear regression with an intercept. Using the center-then-solve
  form:

    1. mu_A = A.mean(0),  mu_B = B.mean(0)
    2. A_c = A - mu_A,    B_c = B - mu_B
    3. Wt = lstsq(A_c, B_c)        shape (D_old, D_new); = pinv(A_c) @ B_c, the
       minimum-norm solution when K < D_old (under-determined, small num_anchors).
    4. Y_hat = (A - mu_A) @ Wt + mu_B = A @ Wt + (mu_B - mu_A @ Wt)
    5. Equivalent nn.Linear: weight = Wt.T, bias = mu_B - mu_A @ Wt
       so that Y_hat = A @ weight.T + bias.

  This is the exact global minimizer of the pure MSE objective (no ridge/weight-
  decay term), solved via np.linalg.lstsq (SVD-based pseudo-inverse) for numerical
  stability over the normal equations. The solve runs in float64 and the result is
  cast back to float32. The rcond argument truncates small singular values of A_c
  (truncated-SVD least squares), which bounds the weight magnitude when A_c is
  ill-conditioned (small K relative to D_old, or collinear features) — still exact
  OLS on the retained well-conditioned subspace.

  Args:
    A     (np.ndarray):    Source anchors, shape (K, D_old).
    B     (np.ndarray):    Target anchors, shape (K, D_new).
    rcond (float | None):  Singular-value cutoff passed to np.linalg.lstsq
                           (relative to the largest singular value). None uses
                           NumPy's machine-precision default.

  Returns:
    dict with keys:
      'mu_A'   (np.ndarray): Shape (D_old,).
      'mu_B'   (np.ndarray): Shape (D_new,).
      'Wt'     (np.ndarray): Shape (D_old, D_new), the centered-space weight.
      'weight' (np.ndarray): Shape (D_new, D_old), nn.Linear weight.
      'bias'   (np.ndarray): Shape (D_new,),       nn.Linear bias.
      'rank'   (int):        Effective rank of A_c after rcond truncation.
  """
  A = np.asarray(A, dtype=np.float64)
  B = np.asarray(B, dtype=np.float64)
  mu_A = A.mean(axis=0)
  mu_B = B.mean(axis=0)
  A_c = A - mu_A
  B_c = B - mu_B
  Wt, _residuals, rank, _sv = np.linalg.lstsq(A_c, B_c, rcond=rcond)  # (D_old, D_new)
  weight = Wt.T.astype(np.float32)                 # (D_new, D_old)
  bias   = (mu_B - mu_A @ Wt).astype(np.float32)   # (D_new,)
  return {
    'mu_A':   mu_A.astype(np.float32),
    'mu_B':   mu_B.astype(np.float32),
    'Wt':     Wt.astype(np.float32),
    'weight': weight,
    'bias':   bias,
    'rank':   int(rank),
  }


def _train_linear_closed_projector(old_anchors, new_anchors, df_anch, val_pool, projector_dir, anchor_key_tag, cfg=None):
  """
  Closed-form OLS linear projector that mirrors _train_procrustes_projector.

  Produces the same kind of nn.Linear(D_old, D_new) map as interpolation_
  similarity='linear', but via the closed-form least-squares solution on all K
  anchor pairs (no SGD, no epochs). To keep the returned bundle drop-in
  compatible with _train_linear_projector (so cross_space_logs.py plots work
  unchanged), metrics['train'] and metrics['val'] each carry a single 1-epoch
  entry, best_epoch=1, and the projector is saved as an nn.Linear state_dict
  whose forward exactly reproduces Y_hat = (X - mu_A) @ Wt + mu_B (in normalized
  space when normalize_embeddings is on).

  Args:
    old_anchors    (dict): Output of _extract_embeddings on old model.
    new_anchors    (dict): Output of _extract_embeddings on new model, aligned.
    df_anch        (pd.DataFrame): Anchor table with sample_id/subject_id/class_id.
    val_pool       (dict): Output of _extract_linear_val_pool.
    projector_dir  (str):  Output directory.
    anchor_key_tag (str):  Short slug used in log lines.
    cfg            (dict | None): Projector recipe; only normalize_embeddings /
                           split_ratios / closed_form_rcond matter. Defaults to the global.

  Returns:
    dict: Same shape as _train_linear_projector's return, plus:
      'kind'               ('linear_close')
      'closed_form_params' ({'mu_old', 'mu_new', 'rank'})
  """
  cfg = copy.deepcopy(LINEAR_PROJECTOR_CONFIG if cfg is None else cfg)
  device = torch.device(cfg['device'])

  os.makedirs(projector_dir, exist_ok=True)
  splits_dir = os.path.join(projector_dir, 'split_training_stage')
  os.makedirs(splits_dir, exist_ok=True)

  df_anch_aligned = _align_df_to_sample_ids(df_anch, old_anchors['sample_ids'], 'anchor df')
  val_df_aligned  = _align_df_to_sample_ids(
    val_pool['df'], val_pool['old']['sample_ids'], 'val.csv df',
  )

  anchor_sids = set(old_anchors['sample_ids'].astype(np.int64).tolist())
  val_sids    = set(val_pool['old']['sample_ids'].astype(np.int64).tolist())
  overlap = anchor_sids & val_sids
  if overlap:
    raise ValueError(
      f"[linear_close:{anchor_key_tag}] {len(overlap)} anchor sample_id(s) also appear "
      f"in the new model's val.csv: {sorted(overlap)[:10]}"
      f"{' ...' if len(overlap) > 10 else ''}. Pick a --csv_anchor_selection split "
      f"disjoint from 'val' (e.g. 'train')."
    )

  M = len(val_pool['old']['embeddings'])
  n_val = round(cfg['split_ratios'][1] * M)
  if n_val < 1 or n_val >= M:
    raise ValueError(
      f"[linear_close:{anchor_key_tag}] cannot build a val/test split from val.csv: "
      f"n_val={n_val} (val.csv rows={M}, split_ratios={cfg['split_ratios']})."
    )
  idx_va, idx_te = _make_subject_disjoint_subsets(val_df_aligned, n_val, _SEED)

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

  print(f"  [linear_close:{anchor_key_tag}] [{split_mode}] "
        f"train={len(split_arrays['train']['old'])} (all anchors, closed-form fit)  "
        f"val={len(idx_va)} test={len(idx_te)} (subject-disjoint subsets of val.csv, {M} rows)")
  print(f"  [linear_close:{anchor_key_tag}] unique subjects per split — "
        f"train={splits_df['train']['subject_id'].nunique()} "
        f"val={splits_df['val']['subject_id'].nunique()} "
        f"test={splits_df['test']['subject_id'].nunique()}")

  split_csv_paths = {}
  for name, sdf in splits_df.items():
    p = os.path.join(splits_dir, f'{name}.csv')
    sdf.to_csv(p, index=False, sep='\t')
    split_csv_paths[name] = p

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

  # --- Closed-form OLS fit in (normalized) anchor space ---
  sol = _fit_linear_closed_form(
    split_arrays['train']['old_norm'],
    split_arrays['train']['new_norm'],
    rcond=cfg['closed_form_rcond'],
  )
  d_old = split_arrays['train']['old'].shape[1]
  d_new = split_arrays['train']['new'].shape[1]
  projector = torch.nn.Linear(d_old, d_new).to(device)
  with torch.no_grad():
    projector.weight.copy_(torch.from_numpy(sol['weight']))
    projector.bias.copy_(torch.from_numpy(sol['bias']))
  projector.eval()

  # Numerical equivalence check on training anchors (normalized space): the saved
  # weight/bias must encode Y_hat = (X - mu_A) @ Wt + mu_B. Done in float64 so it
  # is deterministic and not tripped by GPU/TF32 float32 rounding (see
  # _assert_linear_ckpt_matches_formula).
  y_formula = (
    (split_arrays['train']['old_norm'].astype(np.float64) - sol['mu_A']) @ sol['Wt']
    + sol['mu_B']
  )
  _assert_linear_ckpt_matches_formula(
    sol['weight'], sol['bias'], split_arrays['train']['old_norm'],
    y_formula, 'linear_close', anchor_key_tag,
  )

  # --- Single-shot metrics on train/val/test ---
  def _eval(name):
    with torch.no_grad():
      pred = projector(torch.from_numpy(split_arrays[name]['old_norm']).to(device))
      tgt  = torch.from_numpy(split_arrays[name]['new_norm']).to(device)
      mse, mae, cos = _eval_projector_batch_metrics(pred, tgt)
    return {'mse': mse, 'mae': mae, 'cos': cos}

  tr_m = _eval('train')
  va_m = _eval('val')
  te_m = _eval('test')

  metrics = {
    'train': [{'epoch': 1, **tr_m}],
    'val':   [{'epoch': 1, **va_m}],
    'test':  te_m,
  }
  best_epoch = 1
  sel_key, _, _ = _projector_selection_rule(cfg['loss'])
  best_val_metric = va_m[sel_key]
  best_val_mse = va_m['mse']
  print(f"  [linear_close:{anchor_key_tag}] closed-form OLS rank={sol['rank']}  "
        f"weight.shape={tuple(sol['weight'].shape)}  best_epoch=1  "
        f"val {sel_key.upper()}={best_val_metric:.6f}")
  print(f"  [linear_close:{anchor_key_tag}] test — MSE: {te_m['mse']:.6f}  "
        f"MAE: {te_m['mae']:.6f}  cos: {te_m['cos']:.6f}")

  ckpt_path = os.path.join(projector_dir, f'best_projector_{best_epoch}.pt')
  torch.save(projector.state_dict(), ckpt_path)

  projector_cpu = torch.nn.Linear(d_old, d_new)
  projector_cpu.load_state_dict(projector.state_dict())
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
    'kind':                 'linear_close',
    'closed_form_params': {
      'mu_old': sol['mu_A'],
      'mu_new': sol['mu_B'],
      'rank':   sol['rank'],
    },
  }


def _allocate_balanced(sizes, total_budget):
  """
  Compute per-stratum allocations that maximise the minimum count, using total_budget as
  a soft cap (coverage — at least 1 per stratum — is a hard requirement that may exceed it).

  Args:
    sizes        (dict): {stratum_key: available_sample_count}. All values must be >= 1.
    total_budget (int):  Desired total number of anchors across all strata.

  Returns:
    dict: {stratum_key: allocation_count} where sum(values) >= len(sizes) and
          each value is in [1, sizes[stratum_key]].
  """
  strata_sorted = sorted(sizes.keys(), key=lambda k: sizes[k])
  allocations = {}
  remaining_budget = total_budget
  for i, stratum in enumerate(strata_sorted):
    remaining_strata = len(strata_sorted) - i
    if remaining_budget < remaining_strata:
      # Coverage mode: budget exhausted before all strata get 1; give 1 to each remaining.
      for s in strata_sorted[i:]:
        allocations[s] = 1
      break
    target = remaining_budget // remaining_strata
    alloc = min(sizes[stratum], max(1, target))
    allocations[stratum] = alloc
    remaining_budget -= alloc
  return allocations


def _balance_random_anchors(df_full, num_anchors, key_col, unit, cap):
  """
  Balanced random anchor selection over a stratum column.

  Args:
    df_full     (pd.DataFrame): Candidate pool (must contain the `key_col` column).
    num_anchors (int): Total anchor budget.
    key_col     (str): Stratum column, 'class_id' or 'subject_id'.
    unit        (str): Log noun, 'class' or 'subject'.
    cap         (bool): If True and num_anchors < num_strata, return exactly
                        num_anchors anchors drawn from num_anchors distinct strata
                        (chosen uniformly at random, 1 sample each). If False (or
                        whenever num_anchors >= num_strata), use _allocate_balanced
                        (>=1 per stratum, may exceed the budget).

  Returns:
    pd.DataFrame: Selected rows, index reset.
  """
  strata = sorted(df_full[key_col].unique())
  if cap and num_anchors < len(strata):
    chosen = pd.Series(strata).sample(n=num_anchors, random_state=_SEED).tolist()
    dfs = [df_full[df_full[key_col] == s].sample(n=1, random_state=_SEED) for s in chosen]
    print(f'  [balance_{unit}_random_capped] under-budget: {num_anchors} distinct '
          f'{unit}(es) chosen uniformly (1 sample each)')
    return pd.concat(dfs, ignore_index=True)
  sizes = {s: len(df_full[df_full[key_col] == s]) for s in strata}
  allocations = _allocate_balanced(sizes, num_anchors)
  dfs = [df_full[df_full[key_col] == s].sample(n=allocations[s], random_state=_SEED) for s in strata]
  print(f'  [balance_{unit}_random] per-{unit} allocations: { {s: allocations[s] for s in strata} }')
  return pd.concat(dfs, ignore_index=True)


def _select_anchors(df_full, num_anchors, selection_type):
  """
  Select anchor samples from a DataFrame according to the given strategy.

  Args:
    df_full        (pd.DataFrame): Full pool of candidate anchor samples.
    num_anchors    (int): Total anchor budget for 'random', 'balance_class_random', and
                          'balance_subject_random'. Per-stratum count for 'balance_class_subject'.
    selection_type (str): One of:
      'random'               — sample num_anchors uniformly at random.
      'balance_class_random' — num_anchors is the total budget; at least 1 anchor per
                               class_id (hard requirement, may exceed budget); remaining
                               budget distributed to maximise the minimum per-class count.
      'balance_subject_random' — same as above but keyed on subject_id.
      'balance_class_random_capped' — like 'balance_class_random' when
                               num_anchors >= num_classes, but when num_anchors < num_classes
                               it returns EXACTLY num_anchors anchors from num_anchors distinct
                               class_ids chosen uniformly at random (1 sample each) instead of
                               overshooting the budget for full coverage.
      'balance_subject_random_capped' — same as above but keyed on subject_id.
      'balance_class_subject' — sample num_anchors per (class_id, subject_id) pair,
                               skipping empty combos (with a warning);
                               total ≤ num_anchors × num_classes × num_subjects.

  Returns:
    pd.DataFrame: Selected rows, reset index. Total rows may exceed num_anchors when
                  coverage forces it (balance_class/subject_random only); the *_capped
                  variants return exactly num_anchors when num_anchors < num_strata.

  Raises:
    ValueError: If any non-empty stratum has fewer than num_anchors samples
                (balance_class_subject only), or if selection_type is unknown.

  Note:
    df_full is sorted by sample_id (and the index reset) before any sampling so
    that the seeded, positional `df.sample(random_state=_SEED)` draws are
    reproducible across runs/scripts even when the upstream CSV read or concat
    produced the rows in a different order.
  """
  df_full = df_full.sort_values('sample_id').reset_index(drop=True)

  if selection_type == 'random':
    if len(df_full) < num_anchors:
      raise ValueError(
        f'random: only {len(df_full)} samples available but num_anchors={num_anchors}.'
      )
    return df_full.sample(n=num_anchors, random_state=_SEED).reset_index(drop=True)

  if selection_type == 'balance_class_random':
    return _balance_random_anchors(df_full, num_anchors, 'class_id', 'class', cap=False)

  if selection_type == 'balance_subject_random':
    return _balance_random_anchors(df_full, num_anchors, 'subject_id', 'subject', cap=False)

  if selection_type == 'balance_class_random_capped':
    return _balance_random_anchors(df_full, num_anchors, 'class_id', 'class', cap=True)

  if selection_type == 'balance_subject_random_capped':
    return _balance_random_anchors(df_full, num_anchors, 'subject_id', 'subject', cap=True)

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

  Note:
    All samplers are seeded with the module-global `_SEED` so the swept trial
    order / suggestions are reproducible across runs.
  """
  if sampler_name == 'grid':
    return optuna.samplers.GridSampler(search_space, seed=_SEED)
  if sampler_name == 'random':
    return optuna.samplers.RandomSampler(seed=_SEED)
  return optuna.samplers.TPESampler(seed=_SEED)


def _recipe_id_maps(args):
  """
  Build the id→recipe maps for the swept projector / refinement recipes.

  The ids are the Optuna 'projector_config' / 'refinement_config' axis values
  (insertion-ordered, deduplicated so a repeated YAML value doesn't duplicate a
  grid point). In CLI mode (no recipe sweep) each map has a single entry built
  from the module-global default config.

  Args:
    args (argparse.Namespace): Parsed args; may carry .projector_recipes /
      .refinement_recipes (lists of complete recipe dicts).

  Returns:
    tuple[dict, dict]: (projector_recipe_map, refinement_recipe_map),
      each mapping recipe id (str) → recipe dict.
  """
  proj_recipes = getattr(args, 'projector_recipes', None) or [LINEAR_PROJECTOR_CONFIG]
  ref_recipes  = getattr(args, 'refinement_recipes', None) or [REFINEMENT_CONFIG]
  proj_map, ref_map = {}, {}
  for r in proj_recipes:
    proj_map.setdefault(_projector_recipe_id(r), r)
  for r in ref_recipes:
    ref_map.setdefault(_refinement_tag(r), r)
  return proj_map, ref_map


def _build_search_space(args):
  """
  Build Optuna search space dict from parsed CLI args.

  Args:
    args (argparse.Namespace): Parsed CLI args (hyper args are lists).

  Returns:
    dict: Mapping each hyper param name to its candidate list. Includes the two
      bundled recipe axes 'projector_config' / 'refinement_config' (single-valued
      in CLI mode, so they don't multiply the grid) and the 'refine_mode' axis
      (>1 value only under --refinement 3; pruned per-trial to applicable modes).
  """
  proj_map, ref_map = _recipe_id_maps(args)
  return {
    'num_anchors':              args.num_anchors,
    'anchor_selection_type':    args.anchor_selection_type,
    'csv_anchor_selection':     args.csv_anchor_selection,
    'old_model_csv':            args.old_model_csv,
    'interpolation_similarity': args.interpolation_similarity,
    'mlp_activation':           args.mlp_activation,
    'weighting_method':         args.weighting_method,
    'rbf_sigma':                args.rbf_sigma,
    'projector_config':         list(proj_map),
    'refinement_config':        list(ref_map),
    'refine_mode':              _grid_refine_mode_ids(args.refinement),
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

  # Swept projector / refinement recipes (one element each in CLI mode → no sweep).
  projector_recipes  = getattr(args, 'projector_recipes',  None) or [LINEAR_PROJECTOR_CONFIG]
  refinement_recipes = getattr(args, 'refinement_recipes', None) or [REFINEMENT_CONFIG]
  multi_proj_recipe = len(projector_recipes) > 1
  multi_ref_recipe  = len(refinement_recipes) > 1

  # --- val.csv pool for projector modes (linear/mlp/procrustes), extracted once ---
  # Each (kind, activation, recipe) spec trains its own projector; 'mlp' expands over
  # the swept activations, 'linear'/'procrustes' carry activation=None, and every
  # projector recipe adds a variant — deduped by bundle key so procrustes/linear_close
  # (which ignore the SGD fields) don't retrain for recipes that collapse to the same tag.
  base_projector_specs = []
  for k in _PROJECTOR_KINDS:
    if k not in args.interpolation_similarity:
      continue
    if k == 'mlp':
      base_projector_specs.extend((k, act) for act in args.mlp_activation)
    else:
      base_projector_specs.append((k, None))
  projector_specs = []   # list of (kind, activation, recipe, bundle_key)
  _seen_bundle_keys = set()
  for kind, activation in base_projector_specs:
    for recipe in projector_recipes:
      bkey = _projector_bundle_key(kind, activation, recipe)
      if bkey in _seen_bundle_keys:
        continue
      _seen_bundle_keys.add(bkey)
      projector_specs.append((kind, activation, recipe, bkey))
  projector_val_pool = None
  if projector_specs:
    projector_val_pool = _extract_linear_val_pool(
      old_model, new_model, old_model_pth, new_model_pth,
      old_config, new_config, new_features_path, anchor_domain_features_for_old,
    )

  # --- Refinement-stage inputs (model-B train embeddings + new-model eval embeddings),
  #     extracted once and shared across every refinement bundle ---
  # Modes to precompute (driven by --refinement; empty when off). Every mode applies to a
  # projector bundle; distance metrics only support linear_only. Bundles are keyed by mode
  # so a single grid run (--refinement 3) can hold both. >1 mode → mode-suffixed sub-dirs.
  _grid_refine_modes = _REFINE_FLAG_TO_MODES.get(args.refinement, [])
  _multi_mode = len(_grid_refine_modes) > 1
  # Distance metrics in the sweep are refinable only in linear_only mode (no projector to train).
  _refine_distance_specs = (
    [m for m in args.interpolation_similarity if m not in _PROJECTOR_KINDS]
    if 'linear_only' in _grid_refine_modes else []
  )
  refine_emb_B = refine_emb_B_val = refine_new_eval = refine_new_test = None
  refine_label_denorm = 1.0
  if REFINEMENT_CONFIG['enabled'] and (projector_specs or _refine_distance_specs):
    _nl = bool(new_config['config'].get('normalize_labels', 0))
    _ml = new_config['config'].get('max_label', None)
    refine_label_denorm = float(_ml) if _nl and _ml else 1.0
    b_csv = _resolve_split_csv(old_model_pth, REFINEMENT_CONFIG['refine_split'])
    assert os.path.isfile(b_csv), f'Refinement model-B split CSV not found: {b_csv}'
    helper.set_step_shift(old_features_path)
    refine_emb_B = _extract_embeddings(old_model, old_model_pth, b_csv, old_config)
    bv_csv = _resolve_split_csv(old_model_pth, REFINEMENT_CONFIG['refine_val_split'])
    assert os.path.isfile(bv_csv), f'Refinement model-B val split CSV not found: {bv_csv}'
    helper.set_step_shift(old_features_path)
    refine_emb_B_val = _extract_embeddings(old_model, old_model_pth, bv_csv, old_config)
    e_csv = _resolve_split_csv(new_model_pth, REFINEMENT_CONFIG['new_eval_split'])
    assert os.path.isfile(e_csv), f'Refinement new-eval split CSV not found: {e_csv}'
    refine_new_eval = _extract_embeddings(new_model, new_model_pth, e_csv, new_config)
    # New-model TEST split for the per-class before/after diagnostic (strict: no val fallback).
    t_csv = _resolve_test_csv_strict(new_model_pth)
    refine_new_test = _extract_embeddings(new_model, new_model_pth, t_csv, new_config)
    print(f"[precompute] refinement inputs: "
          f"emb_B({REFINEMENT_CONFIG['refine_split']})={refine_emb_B['embeddings'].shape}  "
          f"emb_B_val({REFINEMENT_CONFIG['refine_val_split']})={refine_emb_B_val['embeddings'].shape}  "
          f"new_eval({REFINEMENT_CONFIG['new_eval_split']})={refine_new_eval['embeddings'].shape}  "
          f"new_test(test)={refine_new_test['embeddings'].shape}")

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
      'projectors': {},
      'refine_distance': {},  # (sim_type, rbf_sigma, mode, refine_tag) → linear_only refinement (distance metrics)
    }
    for kind, activation, recipe, bkey in projector_specs:
      pkey = _projector_key(kind, activation)
      projector_dir = os.path.join(
        precomputed_dir, f'{pkey}_projector', f'{csv_sel}_{num_anch}_{sel_type}',
      )
      # When projector recipes are swept, give each recipe its own subfolder so the
      # split CSVs / checkpoints don't collide. Single-recipe (CLI) runs keep the
      # original path unchanged.
      if multi_proj_recipe:
        projector_dir = os.path.join(projector_dir, _projector_tag(kind, recipe))
      common = dict(
        old_anchors=old_anch,
        new_anchors=new_aligned,
        df_anch=df_anch,
        val_pool=projector_val_pool,
        projector_dir=projector_dir,
        anchor_key_tag=f'{csv_sel}_{num_anch}_{sel_type}',
      )
      if kind == 'procrustes':
        bundle = _train_procrustes_projector(**common, cfg=recipe)
      elif kind == 'linear_close':
        bundle = _train_linear_closed_projector(**common, cfg=recipe)
      else:
        bundle = _train_linear_projector(**common, kind=kind, activation=activation, cfg=recipe)
      # Per-bundle refinement: one per (mode, swept refinement recipe), shared across
      # old_model_csv splits (the per-split old-on-csv before/after metrics are filled in
      # _run_trial). _run_refinement_stage deep-copies new_model.head.linear, so the shared
      # head is never mutated. Every applicable mode runs (linear_only and/or projector_linear,
      # per --refinement); bundles are keyed by (mode, rtag). In linear_only the projector is
      # frozen and only head.linear is refined.
      bundle['refinements'] = {}
      if REFINEMENT_CONFIG['enabled'] and refine_emb_B is not None:
        for _mode in _grid_refine_modes:
          for ref_recipe in refinement_recipes:
            rtag = _refinement_tag(ref_recipe)
            rdir = os.path.join(projector_dir, f'refinement_{_mode}' if _multi_mode else 'refinement')
            if multi_ref_recipe:
              rdir = os.path.join(rdir, rtag)
            bundle['refinements'][(_mode, rtag)] = _run_refinement_stage(
              old_model=old_model, new_model=new_model,
              old_model_pth=old_model_pth, new_model_pth=new_model_pth,
              old_config=old_config, new_config=new_config, old_features_path=old_features_path,
              old_model_anchors=old_anch, new_model_anchors_aligned=new_aligned,
              projector_bundle=bundle, old_model_tensors=None, old_model_csv=None,
              label_denorm=refine_label_denorm,
              refine_dir=rdir,
              tag=f'{csv_sel}_{num_anch}_{sel_type}_{bkey}_{_mode}_{rtag}', mode=_mode,
              emb_B=refine_emb_B, new_eval=refine_new_eval, new_test=refine_new_test,
              emb_B_val=refine_emb_B_val, cfg=ref_recipe,
            )
      anchor_cache[key]['projectors'][bkey] = bundle

    # --- Distance-metric linear_only refinements: one per (sim_type, rbf_sigma, refine_recipe)
    #     over these anchors (the fixed interpolation depends on sim_type/sigma). Cached and
    #     reused across old_model_csv splits; only built in linear_only mode. ---
    if REFINEMENT_CONFIG['enabled'] and refine_emb_B is not None and _refine_distance_specs:
      for sim_type in _refine_distance_specs:
        for sigma in args.rbf_sigma:
          for ref_recipe in refinement_recipes:
            rtag = _refinement_tag(ref_recipe)
            rd_dir = os.path.join(
              precomputed_dir, 'refine_distance', f'{csv_sel}_{num_anch}_{sel_type}',
              f'{sim_type}_s{sigma}')
            if multi_ref_recipe:
              rd_dir = os.path.join(rd_dir, rtag)
            anchor_cache[key]['refine_distance'][(sim_type, sigma, 'linear_only', rtag)] = _run_refinement_stage(
              old_model=old_model, new_model=new_model,
              old_model_pth=old_model_pth, new_model_pth=new_model_pth,
              old_config=old_config, new_config=new_config, old_features_path=old_features_path,
              old_model_anchors=old_anch, new_model_anchors_aligned=new_aligned,
              projector_bundle=None, old_model_tensors=None, old_model_csv=None,
              label_denorm=refine_label_denorm, refine_dir=rd_dir,
              tag=f'{csv_sel}_{num_anch}_{sel_type}_{sim_type}_s{sigma}_{rtag}',
              mode='linear_only', sim_type=sim_type, rbf_sigma=sigma,
              emb_B=refine_emb_B, new_eval=refine_new_eval, new_test=refine_new_test,
              emb_B_val=refine_emb_B_val, cfg=ref_recipe,
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


def _run_trial(trial_params, trial_number, anchor_cache, tensor_cache, new_model, new_config,
               trial_dir, uid, projector_recipe_map, refinement_recipe_map):
  """
  Run a single cheap projection trial using pre-cached embeddings.

  Args:
    trial_params  (dict): Suggested hyper values for this trial (all 10 keys,
                          including 'projector_config' / 'refinement_config').
    trial_number  (int): Optuna trial number (logged in the saved result).
    anchor_cache  (dict): Pre-computed anchor embeddings from _precompute_embeddings.
    tensor_cache  (dict): Pre-computed old-model tensor embeddings.
    new_model     (Model_Advanced): New model instance (head.linear used for classification).
    new_config    (dict): New model k_fold_results.pkl.
    trial_dir     (str): Directory where results.pkl will be saved.
    uid           (int): Timestamp uid of the parent Optuna search run (saved in results.pkl).
    projector_recipe_map  (dict): projector_config id → projector recipe dict.
    refinement_recipe_map (dict): refinement_config id → refinement recipe dict.

  Returns:
    float: MAE for this trial.
  """
  _trial_t0 = time.time()
  _proj_recipe = projector_recipe_map[trial_params['projector_config']]
  _ref_tag     = _refinement_tag(refinement_recipe_map[trial_params['refinement_config']])
  # Which refinement mode this trial applies (a swept axis under --refinement 3; a single
  # fixed value for 1/2; 'none' when off). Bundles were precomputed keyed by (mode, rtag).
  _rmode       = trial_params.get('refine_mode', 'none')
  old_model_tensors = tensor_cache[trial_params['old_model_csv']]['old_tensors']
  classify_linear = new_model.head.linear  # overridden below when refinement is applied

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
    if trial_params['interpolation_similarity'] in _PROJECTOR_KINDS:
      kind = trial_params['interpolation_similarity']
      bkey = _projector_bundle_key(kind, trial_params.get('mlp_activation'), _proj_recipe)
      bundle = anchor_cache[anchor_key]['projectors'][bkey]
      _refine = bundle.get('refinements', {}).get((_rmode, _ref_tag))
      _use_refined = (REFINEMENT_CONFIG['enabled'] and _refine is not None
                      and REFINEMENT_CONFIG['report_after_refinement'])
      _linear_only = _rmode == 'linear_only'
      # projector_linear refinement also adapts the projector (projector_after); linear_only keeps
      # the trained projector frozen and only swaps in the refined classifier.
      _proj_module = (_refine['refine_bundle']['projector_after']
                      if (_use_refined and not _linear_only) else bundle['projector'])
      projected = _apply_linear_projector(
        _proj_module, bundle['norm_stats'], old_model_tensors['embeddings'],
      )
      if _use_refined:
        classify_linear = _refine['refine_bundle']['linear_after']
      weights = np.zeros((len(projected), 0), dtype=np.float32)
      print(f"  [trial {trial_number}] interpolation_similarity={kind} → "
            f"applying {bkey} projector (best_epoch={bundle['best_epoch']})"
            f"{' + refinement' if _use_refined else ''}")
    else:
      weights = _compute_weights(
        z=old_model_tensors['embeddings'],
        anchors=old_model_anchors['embeddings'],
        sim_type=trial_params['interpolation_similarity'],
        sigma=trial_params['rbf_sigma'],
      )
      projected = (weights @ new_model_anchors_aligned['embeddings'].astype(np.float32))
      # Distance metrics support linear_only refinement (fixed interpolation, refined head.linear).
      if (REFINEMENT_CONFIG['enabled'] and _rmode == 'linear_only'
          and REFINEMENT_CONFIG['report_after_refinement']):
        _rd = anchor_cache[anchor_key]['refine_distance'].get(
          (trial_params['interpolation_similarity'], trial_params['rbf_sigma'], _rmode, _ref_tag))
        if _rd is not None:
          classify_linear = _rd['refine_bundle']['linear_after']
          print(f"  [trial {trial_number}] interpolation_similarity="
                f"{trial_params['interpolation_similarity']} → linear_only refinement applied")

  normalize_labels = bool(new_config['config'].get('normalize_labels', 0))
  max_label = new_config['config'].get('max_label', None)
  label_denorm = float(max_label) if normalize_labels and max_label else 1.0

  new_model.head.eval()
  classify_linear.eval()
  device = next(classify_linear.parameters()).device
  with torch.no_grad():
    logits = classify_linear(
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
  if (trial_params['interpolation_similarity'] in _PROJECTOR_KINDS
      and trial_params['num_anchors'] not in (0, -1)):
    anchor_key = (
      trial_params['csv_anchor_selection'],
      trial_params['num_anchors'],
      trial_params['anchor_selection_type'],
    )
    kind = trial_params['interpolation_similarity']
    bkey = _projector_bundle_key(kind, trial_params.get('mlp_activation'), _proj_recipe)
    bundle = anchor_cache[anchor_key]['projectors'][bkey]
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
      'kind':                 bundle.get('kind', kind),
      'procrustes_params':    bundle.get('procrustes_params'),
      'closed_form_params':   bundle.get('closed_form_params'),
    }
    refine = bundle.get('refinements', {}).get((_rmode, _ref_tag))
    if refine is not None:
      # Per-bundle metrics (new-test + projector-drift loss + ckpts) were computed once
      # in precompute; fill the per-split old-on-csv before/after here for this trial.
      rb = refine['refine_bundle']
      ns = bundle['norm_stats']
      old_emb, old_lab = old_model_tensors['embeddings'], old_model_tensors['labels']
      if _rmode == 'linear_only':
        # Projection is frozen; `projected` already holds the projector's output of old_emb.
        mi_b, ma_b = _eval_linear_mae(projected, rb['linear_before'], old_lab, label_denorm)
        mi_a, ma_a = _eval_linear_mae(projected, rb['linear_after'],  old_lab, label_denorm)
      else:
        mi_b, ma_b = _eval_proj_mae(old_emb, rb['projector_before'], ns, rb['linear_before'], old_lab, label_denorm)
        mi_a, ma_a = _eval_proj_mae(old_emb, rb['projector_after'],  ns, rb['linear_after'],  old_lab, label_denorm)
      trial_result['refinement'] = {
        **refine['metrics'],
        'mae_micro_old_oncsv_before': mi_b,
        'mae_macro_old_oncsv_before': ma_b,
        'mae_micro_old_oncsv_after':  mi_a,
        'mae_macro_old_oncsv_after':  ma_a,
        'refine_old_model_csv':       trial_params['old_model_csv'],
        'config':                     rb['config'],
        'per_epoch_metrics':          rb['metrics'],
        'new_test_eval':              refine.get('new_test_eval'),
      }
  elif (trial_params['interpolation_similarity'] not in _PROJECTOR_KINDS
        and trial_params['num_anchors'] not in (0, -1)
        and _rmode == 'linear_only'):
    # Distance-metric linear_only refinement: persist its before/after metrics for this trial.
    anchor_key = (
      trial_params['csv_anchor_selection'],
      trial_params['num_anchors'],
      trial_params['anchor_selection_type'],
    )
    _rd = anchor_cache[anchor_key]['refine_distance'].get(
      (trial_params['interpolation_similarity'], trial_params['rbf_sigma'], _rmode, _ref_tag))
    if _rd is not None:
      rb = _rd['refine_bundle']
      old_lab = old_model_tensors['labels']
      # `projected` is the fixed distance-metric projection of the old-on-csv split.
      mi_b, ma_b = _eval_linear_mae(projected, rb['linear_before'], old_lab, label_denorm)
      mi_a, ma_a = _eval_linear_mae(projected, rb['linear_after'],  old_lab, label_denorm)
      trial_result['refinement'] = {
        **_rd['metrics'],
        'mae_micro_old_oncsv_before': mi_b,
        'mae_macro_old_oncsv_before': ma_b,
        'mae_micro_old_oncsv_after':  mi_a,
        'mae_macro_old_oncsv_after':  ma_a,
        'refine_old_model_csv':       trial_params['old_model_csv'],
        'config':                     rb['config'],
        'per_epoch_metrics':          rb['metrics'],
        'new_test_eval':              _rd.get('new_test_eval'),
      }
  _runtime_min = (time.time() - _trial_t0) / 60.0
  trial_result['metrics']['runtime_min'] = _runtime_min
  print(f'  [trial {trial_number}] runtime: {_runtime_min:.2f} min')
  with open(os.path.join(trial_dir, 'results.pkl'), 'wb') as f:
    pickle.dump(trial_result, f)
  return mae


def run_optuna(args, out_root=None):
  """
  Run an Optuna hyperparameter search over cross-space projection parameters.

  Models are built once; embeddings are pre-extracted and cached before the study starts
  so only the cheap projection + classification ops run per trial.

  Args:
    args     (argparse.Namespace): Parsed CLI args. Hyper args are lists; model paths are strings.
    out_root (str | None):         Base directory for the output tree. Defaults to the current
                                   working directory when None, so callers can route outputs
                                   without changing cwd.

  Returns:
    str: Path to the output directory containing study results.
  """
  _set_global_seed(_SEED)
  optuna.logging.set_verbosity(optuna.logging.WARNING)

  uid = int(time.time())

  # Swept projector / refinement recipe maps (single-entry in CLI mode → no sweep).
  proj_map, ref_map = _recipe_id_maps(args)
  proj_ids, ref_ids = list(proj_map), list(ref_map)
  mode_ids = _grid_refine_mode_ids(args.refinement)  # refine_mode axis values
  multi_proj_recipe = len(proj_ids) > 1
  multi_ref_recipe  = len(ref_ids) > 1
  multi_mode        = len(mode_ids) > 1  # True only under --refinement 3

  def _fmt(vals):
    return '-'.join(str(v) for v in vals)

  _proj_suffix_parts = []
  for _k in _PROJECTOR_KINDS:
    if _k in args.interpolation_similarity:
      _proj_suffix_parts.append(_projector_tag(_k))
  _proj_suffix = ('_' + '_'.join(_proj_suffix_parts)) if _proj_suffix_parts else ''
  _act_suffix = (
    f'_act{_fmt(args.mlp_activation)}' if 'mlp' in args.interpolation_similarity else ''
  )
  _recipe_suffix = ''
  if multi_proj_recipe:
    _recipe_suffix += f'_proj{len(proj_ids)}'
  if multi_ref_recipe:
    _recipe_suffix += f'_ref{len(ref_ids)}'
  args_tag = (
    f'K{_fmt(args.num_anchors)}'
    f'_{_fmt(args.anchor_selection_type)}'
    f'_{_fmt(args.csv_anchor_selection)}'
    f'_{_fmt(args.old_model_csv)}'
    f'_{_fmt(args.interpolation_similarity)}'
    f'_{_fmt(args.weighting_method)}'
    f'_s{_fmt(args.rbf_sigma)}'
    + _proj_suffix
    + _act_suffix
    + _recipe_suffix
  )
  tag_prefix = f'{args.run_tag}_' if args.run_tag else ''
  if len(f'search_{tag_prefix}{args_tag}_{uid}') > 200:
    _hash = hashlib.md5(args_tag.encode()).hexdigest()[:16]
    args_tag = f'{args_tag[:60]}_{_hash}'
  base = out_root if out_root is not None else os.getcwd()
  out_dir = os.path.join(
    base, 'Cross_projection', f'search_{tag_prefix}{args_tag}_{uid}',
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
  # Cache for projector modes (linear/mlp/procrustes): output is determined by
  # (anchor_key, old_model_csv, projector_bundle_key, refinement_config, refine_mode);
  # weighting_method/rbf_sigma are ignored.
  _closed_form_mae_cache:    dict = {}

  def objective(trial):
    params = {
      'num_anchors':              trial.suggest_categorical('num_anchors',              args.num_anchors),
      'anchor_selection_type':    trial.suggest_categorical('anchor_selection_type',    args.anchor_selection_type),
      'csv_anchor_selection':     trial.suggest_categorical('csv_anchor_selection',     args.csv_anchor_selection),
      'old_model_csv':            trial.suggest_categorical('old_model_csv',            args.old_model_csv),
      'interpolation_similarity': trial.suggest_categorical('interpolation_similarity', args.interpolation_similarity),
      'mlp_activation':           trial.suggest_categorical('mlp_activation',            args.mlp_activation),
      'weighting_method':         trial.suggest_categorical('weighting_method',         args.weighting_method),
      'rbf_sigma':                trial.suggest_categorical('rbf_sigma',               args.rbf_sigma),
      'projector_config':         trial.suggest_categorical('projector_config',         proj_ids),
      'refinement_config':        trial.suggest_categorical('refinement_config',        ref_ids),
      'refine_mode':              trial.suggest_categorical('refine_mode',              mode_ids),
    }
    # Reject nonsensical combos: anchor-weighted similarities need rbf weighting,
    # projector similarities need 'none' (already auto-enforced at argparse for
    # pure sweeps, but mixed sweeps may still surface invalid pairs).
    interp = params['interpolation_similarity']
    wm     = params['weighting_method']
    if interp in ('cos', 'l1', 'l2', 'l_inf') and wm == 'none':
      raise optuna.TrialPruned(
        f"invalid combo: interpolation_similarity={interp!r} requires weighting_method='rbf'"
      )
    if interp in _PROJECTOR_KINDS and wm != 'none':
      raise optuna.TrialPruned(
        f"invalid combo: interpolation_similarity={interp!r} requires weighting_method='none'"
      )
    # mlp_activation only affects 'mlp'; collapse the axis for every other interp so it
    # does not multiply unrelated trials (only the first activation value is canonical).
    if interp != 'mlp' and params['mlp_activation'] != args.mlp_activation[0]:
      raise optuna.TrialPruned(
        f"mlp_activation={params['mlp_activation']!r} irrelevant for interp={interp!r}"
      )
    # projector_config only affects a trained-projector trial. Collapse it everywhere
    # else, and (within projector kinds) collapse recipes that map to the same bundle
    # key — procrustes/linear_close ignore the SGD fields, so those recipes are identical.
    _proj_relevant = interp in _PROJECTOR_KINDS and params['num_anchors'] not in (0, -1)
    if not _proj_relevant:
      if params['projector_config'] != proj_ids[0]:
        raise optuna.TrialPruned(
          f"projector_config={params['projector_config']!r} irrelevant for interp={interp!r}"
        )
    else:
      _act = params['mlp_activation']
      _this_bkey = _projector_bundle_key(interp, _act, proj_map[params['projector_config']])
      _canon = next(pid for pid in proj_ids
                    if _projector_bundle_key(interp, _act, proj_map[pid]) == _this_bkey)
      if params['projector_config'] != _canon:
        raise optuna.TrialPruned(
          f"projector_config={params['projector_config']!r} collapses to {_canon!r} for interp={interp!r}"
        )
    # --- Refinement axes (refine_mode then refinement_config) ---
    # refine_mode is a real axis only under --refinement 3 (≥2 candidate modes). Keep only
    # the modes that actually apply to this trial's interp/anchors; when refinement does not
    # run at all, collapse to the first id so the axis never spawns redundant trials.
    _appl_modes = (_applicable_refine_modes(args.refinement, interp, params['num_anchors'])
                   if REFINEMENT_CONFIG['enabled'] else [])
    if not _appl_modes:
      if params['refine_mode'] != mode_ids[0]:
        raise optuna.TrialPruned(
          f"refine_mode={params['refine_mode']!r} irrelevant (no refinement runs for this trial)"
        )
    elif params['refine_mode'] not in _appl_modes:
      raise optuna.TrialPruned(
        f"refine_mode={params['refine_mode']!r} not applicable for interp={interp!r}"
      )
    # refinement_config only matters when a refinement actually runs for this trial (an
    # applicable mode was selected). Collapse it otherwise.
    _ref_runs = bool(_appl_modes) and params['refine_mode'] in _appl_modes
    if not _ref_runs and params['refinement_config'] != ref_ids[0]:
      raise optuna.TrialPruned(
        f"refinement_config={params['refinement_config']!r} irrelevant for this trial"
      )
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
    if interp in _PROJECTOR_KINDS and params['num_anchors'] not in (0, -1):
      cf_key = (
        params['csv_anchor_selection'], params['num_anchors'],
        params['anchor_selection_type'], params['old_model_csv'],
        _projector_bundle_key(interp, params['mlp_activation'], proj_map[params['projector_config']]),
        params['refinement_config'], params['refine_mode'],
      )
      if cf_key in _closed_form_mae_cache:
        print(f'  [trial {trial.number}] interpolation_similarity={interp}, '
              f'key={cf_key} — reusing cached result')
        return _closed_form_mae_cache[cf_key]
    if interp in _PROJECTOR_KINDS:
      _proj_tag_t = f'_{_projector_tag(interp, proj_map[params["projector_config"]])}'
      if interp == 'mlp':
        _proj_tag_t += f'_{params["mlp_activation"]}'
    else:
      _proj_tag_t = f'_{params["weighting_method"]}_s{params["rbf_sigma"]}'
    if multi_ref_recipe and REFINEMENT_CONFIG['enabled']:
      _proj_tag_t += f'_{params["refinement_config"]}'
    if multi_mode and REFINEMENT_CONFIG['enabled']:
      _proj_tag_t += f'_{params["refine_mode"]}'
    trial_tag = (
      f'K{params["num_anchors"]}'
      f'_{params["anchor_selection_type"]}'
      f'_{params["csv_anchor_selection"]}'
      f'_{params["old_model_csv"]}'
      f'_{params["interpolation_similarity"]}'
      + _proj_tag_t
    )
    if len(f'trial{trial.number:04d}_{trial_tag}') > 200:
      _th = hashlib.md5(trial_tag.encode()).hexdigest()[:16]
      trial_tag = f'{trial_tag[:60]}_{_th}'
    trial_dir = os.path.join(out_dir, f'trial{trial.number:04d}_{trial_tag}')
    mae = _run_trial(params, trial.number, anchor_cache, tensor_cache, new_model, new_config,
                     trial_dir, uid, proj_map, ref_map)
    if params['num_anchors'] == 0:
      _zero_anchor_mae_cache[params['old_model_csv']] = mae
    if params['num_anchors'] == -1:
      _neg_one_anchor_mae_cache[params['old_model_csv']] = mae
    if interp in _PROJECTOR_KINDS and params['num_anchors'] not in (0, -1):
      cf_key = (
        params['csv_anchor_selection'], params['num_anchors'],
        params['anchor_selection_type'], params['old_model_csv'],
        _projector_bundle_key(interp, params['mlp_activation'], proj_map[params['projector_config']]),
        params['refinement_config'], params['refine_mode'],
      )
      _closed_form_mae_cache[cf_key] = mae
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

  # --- Grid-size preview: total trial cells + the expensive precompute trainings.
  #     (Pruning may skip some cells; projector/refinement counts match _precompute_embeddings.) ---
  _grid_total = 1
  for v in search_space.values():
    _grid_total *= len(v)
  _anchor_combos = list(dict.fromkeys(
    (c, n, s) for c in args.csv_anchor_selection
    for n in args.num_anchors for s in args.anchor_selection_type
    if n not in (0, -1)))
  _base_specs = []
  for _k in _PROJECTOR_KINDS:
    if _k not in args.interpolation_similarity:
      continue
    if _k == 'mlp':
      _base_specs.extend((_k, a) for a in args.mlp_activation)
    else:
      _base_specs.append((_k, None))
  _proj_bundles = {_projector_bundle_key(k, a, proj_map[pid])
                   for (k, a) in _base_specs for pid in proj_ids}
  n_proj_trainings = len(_anchor_combos) * len(_proj_bundles)
  n_ref_trainings = 0
  if REFINEMENT_CONFIG['enabled']:
    _grid_modes = _REFINE_FLAG_TO_MODES.get(args.refinement, [])
    # Every mode applies to a projector bundle; only linear_only applies to distance metrics.
    n_ref_trainings += n_proj_trainings * len(ref_ids) * len(_grid_modes)
    if 'linear_only' in _grid_modes:
      _dist_specs = [m for m in args.interpolation_similarity if m not in _PROJECTOR_KINDS]
      n_ref_trainings += len(_anchor_combos) * len(_dist_specs) * len(args.rbf_sigma) * len(ref_ids)
  print(
    f'[run_optuna] grid preview: full grid={_grid_total} trial cell(s) (pruning may skip some); '
    f'projector trainings={n_proj_trainings} ({len(_proj_bundles)} bundle(s) x '
    f'{len(_anchor_combos)} anchor combo(s)); refinement trainings={n_ref_trainings}  '
    f'[projector recipes={len(proj_ids)}, refinement recipes={len(ref_ids)}, '
    f"refinement modes={_REFINE_FLAG_TO_MODES.get(args.refinement, []) or ['off']}]"
  )

  _grid_t0 = time.time()
  study.optimize(objective, n_trials=n_trials)
  _grid_min = (time.time() - _grid_t0) / 60.0

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

  print(f'Total grid search time: {_grid_min:.2f} min ({n_trials} trials)')
  print(f'Done. All outputs in {out_dir}')
  return out_dir


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def cross_space_projection(args, out_root=None):
  """
  Full cross-space projection pipeline.

  Args:
    args     (argparse.Namespace): Parsed CLI arguments with fields:
      new_model_pth, old_model_pth, num_anchors, anchor_selection_type,
      csv_anchor_selection, old_model_csv, interpolation_similarity,
      weighting_method, rbf_sigma.
    out_root (str | None):         Base directory for the output tree. Defaults to the current
                                   working directory when None, so callers can route outputs
                                   without changing cwd.

  Returns:
    str: Path to the saved results .pkl file.
  """
  _set_global_seed(_SEED)

  _run_t0 = time.time()
  uid = int(time.time())
  # Single-run recipes: the YAML's first (only) recipe when present, else the global default.
  _proj_recipe = (args.projector_recipes[0] if getattr(args, 'projector_recipes', None) else None)
  _ref_recipe  = (args.refinement_recipes[0] if getattr(args, 'refinement_recipes', None) else None)
  if args.interpolation_similarity in _PROJECTOR_KINDS:
    _proj_tag = f'_{_projector_tag(args.interpolation_similarity, _proj_recipe)}'
    if args.interpolation_similarity == 'mlp':
      _proj_tag += f'_{args.mlp_activation}'
  else:
    _proj_tag = f'_{args.weighting_method}_s{args.rbf_sigma}'
  args_tag = (
    f'K{args.num_anchors}'
    f'_{args.anchor_selection_type}'
    f'_{args.csv_anchor_selection}'
    f'_{args.old_model_csv}'
    f'_{args.interpolation_similarity}'
    + _proj_tag
  )
  tag_prefix = f'{args.run_tag}_' if args.run_tag else ''
  base = out_root if out_root is not None else os.getcwd()
  out_dir = os.path.join(base, 'Cross_projection', f'cross_space_projection_{tag_prefix}{args_tag}_{uid}')
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
  linear_bundle = None  # populated only for projector modes (linear/mlp/procrustes)

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

    if args.interpolation_similarity in _PROJECTOR_KINDS:
      # --- Step 6 (projector): fit on all anchors; val/test from val.csv ---
      kind = args.interpolation_similarity
      activation = args.mlp_activation if kind == 'mlp' else None
      projector_dir = os.path.join(out_dir, f'{_projector_key(kind, activation)}_projector')
      projector_val_pool = _extract_linear_val_pool(
        old_model, new_model, args.old_model_pth, args.new_model_pth,
        old_config, new_config, new_features_path, anchor_domain_features_for_old,
      )
      common = dict(  # noqa: F841 — bundle also persisted into dict_res below
        old_anchors=old_model_anchors,
        new_anchors=new_model_anchors_aligned,
        val_pool=projector_val_pool,
        df_anch=df_anchors,
        projector_dir=projector_dir,
        anchor_key_tag='single',
      )
      if kind == 'procrustes':
        linear_bundle = _train_procrustes_projector(**common, cfg=_proj_recipe)
      elif kind == 'linear_close':
        linear_bundle = _train_linear_closed_projector(**common, cfg=_proj_recipe)
      else:
        linear_bundle = _train_linear_projector(**common, kind=kind, activation=activation, cfg=_proj_recipe)
      projected = _apply_linear_projector(
        linear_bundle['projector'], linear_bundle['norm_stats'],
        old_model_tensors['embeddings'],
      )
      weights = np.zeros((len(projected), 0), dtype=np.float32)
      print(f'  projected ({kind}): {projected.shape}')
    else:
      # --- Step 6: Compute interpolation weights (N, K) in old model space ---
      print('Computing interpolation weights...')
      weights = _compute_weights(
        z=old_model_tensors['embeddings'],
        anchors=old_model_anchors['embeddings'],
        sim_type=args.interpolation_similarity,
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

  # --- Step 8a: Optional post-projection refinement. --refinement selects which mode(s) run
  #     for this test (see _applicable_refine_modes): 1=linear_only (projection FIXED, refine a
  #     copy of head.linear; works for distance metrics too), 2=projector_linear (refine the
  #     projector + a copy of head.linear; projector kinds only), 3=ALL applicable modes in one
  #     run. Each mode trains independently into its own sub-dir; with >1 mode the headline
  #     predictions stay the pure projected output (there is no single "the refined" model). ---
  refine_results = {}  # mode -> _run_refinement_stage result
  classify_linear = new_model.head.linear
  _modes = (_applicable_refine_modes(args.refinement, args.interpolation_similarity, args.num_anchors)
            if REFINEMENT_CONFIG['enabled'] else [])
  if _modes:
    _is_projector_kind = args.interpolation_similarity in _PROJECTOR_KINDS
    _act = args.mlp_activation if args.interpolation_similarity == 'mlp' else None
    if _is_projector_kind:
      _refine_base = os.path.join(
        out_dir, f'{_projector_key(args.interpolation_similarity, _act)}_projector', 'refinement')
    else:
      _refine_base = os.path.join(
        out_dir, f'refinement_{args.interpolation_similarity}_{args.weighting_method}_s{args.rbf_sigma}')
    for _mode in _modes:
      # Suffix the dir / tag per mode only when several modes run, so single-mode
      # (--refinement 1/2) output paths stay byte-for-byte unchanged.
      refine_dir = f'{_refine_base}_{_mode}' if len(_modes) > 1 else _refine_base
      print(f'[cross_space_projection] Refinement stage (mode={_mode}) → {refine_dir}')
      rr = _run_refinement_stage(
        old_model=old_model, new_model=new_model,
        old_model_pth=args.old_model_pth, new_model_pth=args.new_model_pth,
        old_config=old_config, new_config=new_config, old_features_path=old_features_path,
        old_model_anchors=old_model_anchors, new_model_anchors_aligned=new_model_anchors_aligned,
        projector_bundle=linear_bundle, old_model_tensors=old_model_tensors,
        old_model_csv=args.old_model_csv, label_denorm=label_denorm,
        refine_dir=refine_dir, tag=(f'single_{_mode}' if len(_modes) > 1 else 'single'),
        mode=_mode,
        sim_type=(None if _is_projector_kind else args.interpolation_similarity),
        rbf_sigma=(None if _is_projector_kind else args.rbf_sigma),
        cfg=_ref_recipe,
      )
      refine_results[_mode] = rr
      rm = rr['metrics']
      print(f"  [{_mode}] refinement old-on-{args.old_model_csv} MAE micro: "
            f"{rm['mae_micro_old_oncsv_before']:.4f} → {rm['mae_micro_old_oncsv_after']:.4f}  |  "
            f"new-on-{REFINEMENT_CONFIG['new_eval_split']} MAE micro: "
            f"{rm['mae_micro_new_test_before']:.4f} → {rm['mae_micro_new_test_after']:.4f}")
    # Headline: only fold a refined model into the reported preds when exactly one mode ran.
    if REFINEMENT_CONFIG['report_after_refinement'] and len(_modes) == 1:
      _only_mode = _modes[0]
      _rb = refine_results[_only_mode]['refine_bundle']
      if _only_mode == 'linear_only':
        # Projection is held fixed; keep `projected` and only swap in the refined classifier.
        classify_linear = _rb['linear_after']
      else:
        projected = _apply_linear_projector(
          _rb['projector_after'], linear_bundle['norm_stats'], old_model_tensors['embeddings'])
        new_model_tensors['embeddings'] = projected
        classify_linear = _rb['linear_after']

  new_model.head.eval()
  classify_linear.eval()
  device = next(classify_linear.parameters()).device
  with torch.no_grad():
    logits = classify_linear(
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
    'mlp_activation':          args.mlp_activation,
    'weighting_method':        args.weighting_method,
    'rbf_sigma':               args.rbf_sigma,
    'uid':                     uid,
    'anchors_csv_path':        anchors_csv_path,
    'old_tensors_csv_path':    old_tensors_csv_path,
    'out_dir':                 out_dir,
    'script_cmd':              ' '.join(sys.argv),
  }

  _runtime_min = (time.time() - _run_t0) / 60.0
  metrics['runtime_min'] = _runtime_min

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
      'kind':                 linear_bundle.get('kind', args.interpolation_similarity),
      'procrustes_params':    linear_bundle.get('procrustes_params'),
      'closed_form_params':   linear_bundle.get('closed_form_params'),
    }
  if refine_results:
    def _refine_block(rr):
      """Assemble the per-mode refinement pkl block from a _run_refinement_stage result."""
      return {
        **rr['metrics'],
        'config':            rr['refine_bundle']['config'],
        'per_epoch_metrics': rr['refine_bundle']['metrics'],
        'new_test_eval':     rr.get('new_test_eval'),
      }
    if len(refine_results) == 1:
      # Single mode (--refinement 1/2): keep the legacy single-block schema unchanged.
      dict_res['refinement'] = _refine_block(next(iter(refine_results.values())))
    else:
      # Multiple modes (--refinement 3): one block per mode under 'refinements'.
      dict_res['refinements'] = {m: _refine_block(rr) for m, rr in refine_results.items()}
    # Self-contained before/after summary alongside the pkl (one row per mode).
    pd.DataFrame([rr['metrics'] for rr in refine_results.values()]).to_csv(
      os.path.join(out_dir, 'refinement_summary.csv'), index=False)

  out_pkl = os.path.join(out_dir, f'results_{uid}.pkl')
  pkl_bytes = pickle.dumps(dict_res)
  print(f'Saving pkl ({len(pkl_bytes) / 1e6:.1f} MB) → {out_pkl}')
  with open(out_pkl, 'wb') as f:
    f.write(pkl_bytes)

  config_txt = os.path.join(out_dir, 'config_logging.txt')
  with open(config_txt, 'w') as f:
    for k, v in config_logging.items():
      f.write(f'{k}: {v}\n')

  print(f'Done. All outputs in {out_dir} ({_runtime_min:.2f} min)')
  return out_pkl


# Top-level keys accepted in a --config YAML. The 8 sweep axes + run settings mirror
# the CLI flags; 'linear_projector' / 'refinement_config' are the swept-recipe blocks.
# Note: 'refinement' is the 0/1/2 on/off+mode flag (not the recipe block).
_ALLOWED_YAML_KEYS = {
  'new_model_pth', 'old_model_pth', 'num_anchors', 'anchor_selection_type',
  'csv_anchor_selection', 'old_model_csv', 'interpolation_similarity', 'mlp_activation',
  'weighting_method', 'rbf_sigma', 'n_trials', 'optuna_sampler', 'run_tag', 'refinement',
  'seed', 'linear_projector', 'refinement_config',
}
# YAML keys forwarded to argparse: list (nargs='+') axes vs scalar/string options.
_YAML_LIST_ARGS = (
  'num_anchors', 'anchor_selection_type', 'csv_anchor_selection', 'old_model_csv',
  'interpolation_similarity', 'mlp_activation', 'weighting_method', 'rbf_sigma',
)
_YAML_SCALAR_ARGS = ('new_model_pth', 'old_model_pth', 'run_tag', 'optuna_sampler',
                     'n_trials', 'refinement', 'seed')


def _yaml_to_argv(ycfg):
  """
  Translate a parsed --config YAML dict into an argv list for the main argparse parser.

  Reusing argparse means the YAML inherits every default, type and choices check the
  CLI enforces (and the required-arg errors when a key is missing). The two recipe
  blocks ('linear_projector' / 'refinement_config') are handled separately and are
  intentionally not emitted here.

  Args:
    ycfg (dict): Parsed YAML mapping (already validated against _ALLOWED_YAML_KEYS).

  Returns:
    list[str]: argv tokens, e.g. ['--num_anchors', '50', '100', '--optuna_sampler', 'grid'].
  """
  argv = []
  for key in _YAML_SCALAR_ARGS:
    if ycfg.get(key) is not None:
      argv += [f'--{key}', str(ycfg[key])]
  for key in _YAML_LIST_ARGS:
    if ycfg.get(key) is not None:
      vals = ycfg[key] if isinstance(ycfg[key], list) else [ycfg[key]]
      argv += [f'--{key}'] + [str(v) for v in vals]
  return argv


if __name__ == '__main__':
  # --- YAML-only detection: --config is mutually exclusive with every other flag ---
  _pre = argparse.ArgumentParser(add_help=False)
  _pre.add_argument('--config', type=str, default=None,
                    help='YAML file supplying ALL args (sole source; no other flags allowed).')
  _pre_args, _rest = _pre.parse_known_args()

  parser = argparse.ArgumentParser(
    description='Project old model embeddings into new model space via anchor interpolation.'
  )
  parser.add_argument('--config', type=str, default=None,
                      help='YAML file supplying ALL args. When given it is the SOLE source of '
                           'arguments — passing any other flag alongside it is an error. The YAML '
                           'mirrors these flags (lists stay lists) plus two swept-recipe blocks '
                           "'linear_projector' and 'refinement_config' whose fields may be scalars "
                           'or lists (lists are swept as a full Cartesian grid).')
  parser.add_argument('--new_model_pth', type=str, required=True,
                      help='Path to new model checkpoint (.pt/.pth)')
  parser.add_argument('--old_model_pth', type=str, required=True,
                      help='Path to old model checkpoint (.pt/.pth)')
  # --- Hyper args: accept one or more values; multiple values trigger Optuna ---
  parser.add_argument('--num_anchors', type=int, nargs='+', required=True,
                      help='Number of anchor samples (one or more values to sweep)')
  parser.add_argument('--anchor_selection_type', type=str, nargs='+', default=['random'],
                      choices=['random', 'balance_class_random', 'balance_subject_random',
                               'balance_class_random_capped', 'balance_subject_random_capped',
                               'balance_class_subject'],
                      help='Anchor selection strategy (one or more values to sweep). '
                           'balance_class/subject_random use num_anchors as total budget, guaranteeing >=1 anchor per stratum. '
                           'balance_class/subject_random_capped return exactly num_anchors anchors from num_anchors distinct '
                           'strata (1 sample each, uniformly chosen) when num_anchors < num_strata, else behave like the '
                           'non-capped balanced variant. '
                           'balance_class_subject uses num_anchors per (class, subject) pair (strict).')
  parser.add_argument('--csv_anchor_selection', type=str, nargs='+', required=True,
                      help='Split(s) to select anchors from (train/val/test/exc_train/exc_val/exc_test) — new model domain')
  parser.add_argument('--old_model_csv', type=str, nargs='+', required=True,
                      help='Split(s) to project (train/val/test/all/exc_train/exc_val/exc_test) — old model domain')
  parser.add_argument('--interpolation_similarity', type=str, nargs='+', default=['cos'],
                      choices=['cos', 'l1', 'l2', 'l_inf', 'linear', 'mlp', 'procrustes', 'linear_close', 'geodesic'],
                      help='Distance metric(s) for weight computation. '
                           "'cos' uses cosine distance d = 1 - |cos_sim| in [0, 1]. "
                           "'geodesic' builds a k-NN graph (k=5, Euclidean edges) over anchor "
                           'embeddings, runs all-pairs shortest path (scipy.sparse.csgraph), '
                           'then extends to queries via their k nearest anchors; disconnected '
                           'pairs fall back to direct L2. '
                           "'linear' trains a learned nn.Linear(D_old, D_new) projector on the "
                           'anchor pairs (subject-disjoint split, see LINEAR_PROJECTOR_CONFIG); '
                           "'mlp' trains Linear(D_old, D_new)->activation->Linear(D_new, D_new) "
                           '(same training recipe; activation set via --mlp_activation); '
                           "'procrustes' fits a closed-form centered + isotropically scaled "
                           'semi-orthogonal map on the same anchor pairs (no SGD); '
                           "'linear_close' fits the same nn.Linear(D_old, D_new) as 'linear' "
                           'but via the closed-form OLS (least-squares) solution instead of SGD '
                           '(exact minimizer of the pure MSE loss, no weight decay). For '
                           "'linear', 'mlp', 'procrustes' and 'linear_close', weighting_method "
                           "must be 'none' and rbf_sigma is ignored.")
  parser.add_argument('--mlp_activation', type=str, nargs='+', default=['gelu'],
                      choices=['gelu', 'relu', 'silu', 'leaky_relu'],
                      help="Activation(s) for the 'mlp' projector "
                           '(Linear(D_old,D_new)->act->Linear(D_new,D_new)). Sweepable: each '
                           'value becomes a separate Optuna trial. Ignored for non-mlp interps.')
  parser.add_argument('--weighting_method', type=str, nargs='+', default=['rbf'],
                      choices=['rbf', 'none'],
                      help="Method to convert distances to interpolation weights. "
                           "'rbf' = exp(-d^2/(2 sigma^2)) softmax. "
                           "'none' = no weighting (mandatory and auto-defaulted for "
                           "interpolation_similarity in {linear, mlp, procrustes}, which produce "
                           'projected embeddings directly).')
  parser.add_argument('--rbf_sigma', type=float, nargs='+', default=[1.0],
                      help='Sigma value(s) for RBF kernel: exp(-d²/(2σ²))')
  # --- Optuna settings ---
  parser.add_argument('--n_trials', type=int, default=None,
                      help='Number of Optuna trials. If None (default) and --optuna_sampler grid, '
                           'runs the full grid automatically.')
  parser.add_argument('--optuna_sampler', type=str, default='grid',
                      choices=['tpe', 'random', 'grid'],
                      help='Optuna sampler (tpe, random, grid). Default is grid')
  parser.add_argument('--run_tag', type=str, default=None,
                      help='Optional label prepended to the output folder name for easy identification')
  parser.add_argument('--seed', type=int, default=_SEED,
                      help='Global RNG seed for reproducibility (python random, numpy, torch, cuda). '
                           'Default 42.')
  parser.add_argument('--refinement', type=int, choices=[0, 1, 2, 3],
                      default=int(REFINEMENT_CONFIG['enabled']),
                      help='Post-projection refinement stage (see REFINEMENT_CONFIG). The value is '
                           'the number of layers refined: 0 = off; 1 = linear-only (the projection is '
                           'held FIXED and only a copy of the new model head.linear is fine-tuned — '
                           'works for ALL interpolation_similarity values, including the distance '
                           'metrics cos/l1/l2/l_inf/geodesic); 2 = projector+linear (jointly fine-tune '
                           'the projector + a copy of head.linear, projector kinds '
                           'linear/mlp/procrustes/linear_close only); 3 = ALL applicable modes in one '
                           'run (linear_only + projector_linear; projector_linear is skipped for '
                           'distance metrics, so 3 collapses to 1 there). 1/2/3 all require '
                           'num_anchors>0. Default off.')

  if _pre_args.config is not None:
    # --- YAML-only mode: the file is the sole source of args ---
    if _rest:
      _pre.error(f"--config is YAML-only; remove the other CLI flag(s): {_rest}")
    with open(_pre_args.config) as _f:
      _ycfg = yaml.safe_load(_f) or {}
    if not isinstance(_ycfg, dict):
      parser.error(f"--config {_pre_args.config!r} must contain a top-level mapping")
    _unknown = set(_ycfg) - _ALLOWED_YAML_KEYS
    if _unknown:
      parser.error(f"unknown top-level YAML key(s) {sorted(_unknown)}; "
                   f"allowed: {sorted(_ALLOWED_YAML_KEYS)}")
    if isinstance(_ycfg.get('refinement'), dict):
      parser.error("YAML key 'refinement' is the 0/1/2 on/off+mode flag; put the swept "
                   "refinement hyperparameters under 'refinement_config' instead")
    args = parser.parse_args(_yaml_to_argv(_ycfg))
    args.projector_recipes = _expand_recipes(
      LINEAR_PROJECTOR_CONFIG, _ycfg.get('linear_projector'), _PROJECTOR_SWEEPABLE, 'linear_projector')
    args.refinement_recipes = _expand_recipes(
      REFINEMENT_CONFIG, _ycfg.get('refinement_config'), _REFINEMENT_SWEEPABLE, 'refinement_config')
  else:
    args = parser.parse_args()
    args.projector_recipes  = [copy.deepcopy(LINEAR_PROJECTOR_CONFIG)]
    args.refinement_recipes = [copy.deepcopy(REFINEMENT_CONFIG)]

  REFINEMENT_CONFIG['enabled'] = (args.refinement != 0)
  # For 1/2 this is the single mode; for 3 (run all modes) it is only a vestigial default
  # (the standalone loop and the grid refine_mode axis both pass `mode=` explicitly, so the
  # actual mode never comes from here). Kept for back-compat with single-mode runs.
  REFINEMENT_CONFIG['mode'] = {1: 'linear_only', 2: 'projector_linear'}.get(
    args.refinement, 'projector_linear')
  # enabled/mode are driven by --refinement, never swept: stamp them onto every recipe.
  for _r in args.refinement_recipes:
    _r['enabled'] = REFINEMENT_CONFIG['enabled']
    _r['mode']    = REFINEMENT_CONFIG['mode']
  _closed_form_in_sweep = set(_PROJECTOR_KINDS) & set(args.interpolation_similarity)
  _anchor_weighted_in_sweep = {'cos', 'l1', 'l2', 'l_inf'} & set(args.interpolation_similarity)
  if _closed_form_in_sweep and set(args.num_anchors) <= {0, -1}:
    parser.error(
      f"interpolation_similarity={sorted(_closed_form_in_sweep)} requires at least one "
      f"--num_anchors > 0 (projector modes fit on anchor pairs; num_anchors=0/-1 use no anchors)"
    )
  # weighting_method normalization:
  #   - pure closed-form sweep: silently force ['none']
  #   - pure anchor-weighted sweep: forbid 'none'
  #   - mixed sweep: keep both, per-trial pruning handles invalid pairs
  if _closed_form_in_sweep and not _anchor_weighted_in_sweep:
    if args.weighting_method != ['none']:
      print(f"[argparse] interpolation_similarity={sorted(_closed_form_in_sweep)} → "
            f"forcing --weighting_method='none' (was {args.weighting_method})")
      args.weighting_method = ['none']
  elif _anchor_weighted_in_sweep and not _closed_form_in_sweep:
    if 'none' in args.weighting_method and 'rbf' not in args.weighting_method:
      parser.error(
        f"interpolation_similarity={sorted(_anchor_weighted_in_sweep)} requires "
        f"--weighting_method='rbf' (got {args.weighting_method!r}). 'none' is only valid "
        f"for linear/mlp/procrustes/linear_close."
      )
  # Apply the user's seed once, up front: this reassigns the module-global _SEED so
  # every call-time consumer (anchor sampling, subject-disjoint splits, Optuna sampler)
  # and the entry-point re-seeds in run_optuna / cross_space_projection use it.
  _set_global_seed(args.seed)

  _hyper_lists = [
    args.num_anchors, args.anchor_selection_type, args.csv_anchor_selection,
    args.old_model_csv, args.interpolation_similarity, args.mlp_activation,
    args.weighting_method, args.rbf_sigma,
  ]
  use_optuna = (
    any(len(v) > 1 for v in _hyper_lists)
    or (args.n_trials is not None and args.n_trials > 1)
    or len(args.projector_recipes) > 1
    or len(args.refinement_recipes) > 1
  )

  if use_optuna:
    run_optuna(args)
  else:
    # Unwrap single-element lists so cross_space_projection receives scalar args. The
    # recipe lists stay lists (cross_space_projection reads element [0] itself).
    _keep_lists = ('projector_recipes', 'refinement_recipes')
    single_args = argparse.Namespace(**{
      k: (v[0] if (isinstance(v, list) and k not in _keep_lists) else v)
      for k, v in vars(args).items()
    })
    cross_space_projection(single_args)

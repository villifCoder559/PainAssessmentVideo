"""Shared helpers for the cross-space LaTeX tables: method names and aggregate loading."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["linear", "mlp", "procrustes", "linear_close", "autoencoder"]


METHOD_NAMES = {
  "linear": "Linear",
  "mlp": "MLP",
  "procrustes": "Procrustes",
  "linear_close": "Linear (closed form)",
  "autoencoder": "EncDec",
}


MODEL_NAMES = {
  "VIDEOMAE_v2_S": "VMAEv2-S",
}


BASELINE_COLUMNS = [
  "srctest_mae_micro_old",
  "srctest_mae_macro_old",
  "newtest_mae_micro_before",
  "newtest_mae_macro_before",
]


STAGE_COLUMNS = {
  "projector_only": [
    "srctest_mae_micro_before",
    "srctest_mae_macro_before",
  ],
  "linear_only": [
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
  ],
  "projector_linear": [
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
  ],
  "random_projector_linear": [
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
  ],
}


RESULT_COLUMNS = {
  "projector_only": (
    "srctest_mae_micro_before",
    "srctest_mae_macro_before",
    "newtest_mae_micro_before",
    "newtest_mae_macro_before",
  ),
  "linear_only": (
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
  ),
  "projector_linear": (
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
  ),
  "random_projector_linear": (
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
  ),
}


def _path_list(value: object) -> list[str]:
  """Normalize a checkpoint value stored as a string or sequence."""
  if isinstance(value, (list, tuple)):
    return [str(item) for item in value]
  if isinstance(value, str):
    return [item for item in value.split(";") if item]
  raise ValueError(f"Expected checkpoint path(s), got {type(value).__name__}.")


def _load_pkl(path: Path) -> dict:
  """Load one trusted local aggregate PKL."""
  with path.open("rb") as handle:
    data = pickle.load(handle)
  if not isinstance(data, dict):
    raise ValueError(f"Aggregate PKL is not a dictionary: {path}")
  return data


def _global_config(checkpoint: str) -> dict:
  """Find and load the global model configuration above a checkpoint."""
  path = Path(checkpoint).expanduser()
  if not path.is_absolute():
    path = Path(__file__).resolve().parent / path
  for parent in path.parents:
    config_path = parent / "global_config.json"
    if config_path.is_file():
      with config_path.open(encoding="utf-8") as handle:
        return json.load(handle)
  raise ValueError(f"Could not find global_config.json for checkpoint: {checkpoint}")


def _dataset_name(config: dict) -> str:
  """Extract the trained dataset name from a model configuration."""
  value = config.get("path_csv_dataset") or config.get("path_video_dataset")
  if isinstance(value, (list, tuple)) and value:
    name = str(value[0])
  elif isinstance(value, str) and value:
    name = Path(value).parts[0]
  else:
    raise ValueError("Model config has no usable dataset path.")
  return "BIOVID" if name == "partA" else name


def _model_metadata(checkpoints: object) -> tuple[str, str]:
  """Return the consistent dataset and display model name for checkpoints."""
  metadata = set()
  for checkpoint in _path_list(checkpoints):
    config = _global_config(checkpoint)
    model_type = str(config.get("model_type") or "")
    if not model_type:
      raise ValueError(f"Model config has no model_type: {checkpoint}")
    metadata.add((_dataset_name(config), MODEL_NAMES.get(model_type, model_type)))
  if len(metadata) != 1:
    raise ValueError(f"Checkpoint model metadata is inconsistent: {sorted(metadata)}")
  return metadata.pop()


def _selected_aggregate(
  data: dict,
  projection: str,
  fake_distribution: str | None,
) -> bool:
  """Return whether an aggregate matches the requested projection kind."""
  config = data.get("config_cross_space_projection") or {}
  distribution = (
    data.get("fake_projection_distribution")
    or config.get("fake_projection_distribution")
  )
  metadata = data.get("fake_projection_metadata") or {}
  control = (
    data.get("fake_projection_control")
    or metadata.get("control")
    or config.get("fake_projection_control")
  )
  is_fake = bool(config.get("fake_projection") or distribution or control)
  if projection == "real":
    return not is_fake
  if projection == "fake_adapter":
    return is_fake and control == "fake_adapter"
  return (is_fake and control in (None, "fake_embeddings")
          and distribution == fake_distribution)


def _method_sort_key(method: str) -> tuple[int, str]:
  """Sort known projection methods like the reference table."""
  try:
    return METHOD_ORDER.index(method), method
  except ValueError:
    return len(METHOD_ORDER), method


def _root_summary_directions(
  root: Path,
  projection: str,
  stage: str,
  fake_distribution: str | None,
  *,
  skip_consistency_checks: bool = False,
  allow_unavailable: bool = False,
) -> list[dict]:
  """Load a stage, optionally retaining methods whose stage is unavailable."""
  summary_path = root / "aggregated_summary.csv"
  if not summary_path.is_file():
    raise ValueError(f"Missing consolidated summary: {summary_path}")
  summary = pd.read_csv(summary_path)
  required = [
    "source_pkl",
    "old_model_pth",
    "new_model_pth",
    "subtrial_index",
    "interpolation_similarity",
    "num_anchors",
    *BASELINE_COLUMNS,
  ]
  stage_required = [
    *STAGE_COLUMNS[stage],
    *(["refine_mode"] if stage != "projector_only" else []),
  ]
  if not allow_unavailable:
    required.extend(stage_required)
  stage_missing = [column for column in stage_required if column not in summary.columns]
  missing = [column for column in required if column not in summary.columns]
  if missing:
    raise ValueError(
      f"Missing summary columns in {summary_path}: {', '.join(missing)}"
    )
  rows = summary.loc[
    summary["subtrial_index"].astype(str).eq("AGGREGATE_MEAN")
  ]
  if allow_unavailable:
    # Retain one representative of malformed aggregates for validation below.
    without_mean = summary.loc[~summary["source_pkl"].isin(rows["source_pkl"])]
    rows = pd.concat([rows, without_mean.drop_duplicates("source_pkl")])
  if stage != "projector_only" and not allow_unavailable:
    rows = rows.loc[rows["refine_mode"].astype(str).eq(stage)]

  grouped: dict[tuple[tuple[str, str], tuple[str, str]], list[dict]] = {}
  pkl_cache: dict[Path, dict] = {}
  for row in rows.to_dict("records"):
    pkl_path = Path(str(row["source_pkl"]))
    if not pkl_path.is_absolute():
      pkl_path = root / pkl_path
    if not pkl_path.is_file():
      raise ValueError(
        f"Missing aggregate PKL referenced by {summary_path}: {pkl_path}"
      )
    if pkl_path not in pkl_cache:
      pkl_cache[pkl_path] = _load_pkl(pkl_path)
    data = pkl_cache[pkl_path]
    if not _selected_aggregate(data, projection, fake_distribution):
      continue
    if allow_unavailable and str(row["subtrial_index"]) != "AGGREGATE_MEAN":
      raise ValueError(f"Missing AGGREGATE_MEAN row for aggregate: {pkl_path}")
    config = data.get("config_cross_space_projection") or {}
    old_metadata = _model_metadata(row["old_model_pth"])
    new_metadata = _model_metadata(row["new_model_pth"])
    if (
      old_metadata != _model_metadata(config.get("old_model_pth"))
      or new_metadata != _model_metadata(config.get("new_model_pth"))
    ):
      raise ValueError(f"Checkpoint metadata mismatch for aggregate: {pkl_path}")
    row["method"] = str(row["interpolation_similarity"])
    row["source_pkl"] = str(pkl_path.relative_to(root))
    grouped.setdefault((old_metadata, new_metadata), []).append(row)

  directions = []
  for (old_metadata, new_metadata), direction_rows in grouped.items():
    by_method: dict[str, list[dict]] = {}
    for row in direction_rows:
      by_method.setdefault(row["method"], []).append(row)
    collapsed = []
    for method, method_rows in by_method.items():
      selected = method_rows if stage == "projector_only" else [
        row for row in method_rows if str(row.get("refine_mode")) == stage
      ]
      available = bool(selected) and not stage_missing
      if len(selected) > 1:
        if stage != "projector_only":
          raise ValueError(
            f"Duplicate aggregate method {method} for "
            f"{old_metadata[0]} -> {new_metadata[0]} in {summary_path}"
          )
        for column in (() if stage_missing else RESULT_COLUMNS[stage]):
          values = pd.to_numeric(
            pd.Series([item[column] for item in selected]), errors="coerce"
          )
          if not skip_consistency_checks and (
            values.isna().any() or not math.isclose(
              values.min(), values.max(), rel_tol=1e-7, abs_tol=1e-8
            )
          ):
            raise ValueError(
              f"Inconsistent projector_only column {column} in {summary_path}."
            )
      row = dict((selected or method_rows)[0])
      row["_stage_available"] = available
      collapsed.append(row)
    for column in BASELINE_COLUMNS:
      values = pd.to_numeric(
        pd.Series([row[column] for row in collapsed]), errors="coerce"
      )
      if not skip_consistency_checks and (
        values.isna().any() or not math.isclose(
          values.min(), values.max(), rel_tol=1e-7, abs_tol=1e-8
        )
      ):
        raise ValueError(
          f"Inconsistent baseline column {column} for "
          f"{old_metadata[0]} -> {new_metadata[0]} in {summary_path}"
        )
    collapsed.sort(key=lambda row: _method_sort_key(row["method"]))
    directions.append({
      "root": root,
      "source_dataset": old_metadata[0],
      "old_model": old_metadata[1],
      "target_dataset": new_metadata[0],
      "new_model": new_metadata[1],
      "rows": collapsed,
      "stage_available": all(row["_stage_available"] for row in collapsed),
      "stage_reason": "",
    })

  return sorted(
    directions,
    key=lambda direction: (
      tuple(sorted((direction["source_dataset"], direction["target_dataset"]))),
      direction["source_dataset"],
      direction["target_dataset"],
      direction["old_model"],
      direction["new_model"],
    ),
  )

"""Generate LaTeX tables from cross-projection aggregate results."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import time
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
STAGES = (
  "projector_only",
  "linear_only",
  "projector_linear",
  "random_projector_linear",
)
COMPARISON_STAGE_NAMES = {
  "projector_only": "Projector-only",
  "linear_only": "Linear-only refinement",
  "projector_linear": "Joint refinement",
}
STAGE_CAPTIONS = {
  "projector_only": "Projector-only cross-projection results",
  "linear_only": "Linear-only refinement cross-projection results",
  "projector_linear": "Projector-and-linear refinement cross-projection results",
  "random_projector_linear": (
    "Frozen-random-projector linear-head refinement cross-projection results"
  ),
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


def _fake_adapter_identity(config: dict, row: dict, stage: str) -> str:
  """Return the seed-invariant experiment identity used before averaging."""
  identity = {
    "old_model_pth": _path_list(config.get("old_model_pth")),
    "new_model_pth": _path_list(config.get("new_model_pth")),
    "method": row.get("method"),
    "num_anchors": row.get("num_anchors"),
    "refine_mode": stage if stage != "projector_only" else None,
  }
  for key in (
    "anchor_selection_type", "csv_anchor_selection", "old_model_csv",
    "remove_classes_greater", "mlp_activation", "mlp_num_layers",
    "weighting_method", "temperature", "rbf_sigma", "linear_projector",
    "refinement_config",
  ):
    if key in config:
      identity[key] = config[key]
  return json.dumps(identity, sort_keys=True, default=str)


def _load_direction(
  root: Path,
  projection: str,
  stage: str,
  fake_distribution: str | None,
  *,
  allow_unavailable: bool = False,
  skip_consistency_checks: bool = False,
) -> dict:
  """Load selected aggregate rows and model metadata from one experiment root."""
  rows = []
  unavailable = []
  old_metadata_values = set()
  new_metadata_values = set()
  for pkl_path in sorted(root.rglob("*.pkl")):
    if not pkl_path.parent.name.startswith("aggregated"):
      continue
    data = _load_pkl(pkl_path)
    if not _selected_aggregate(data, projection, fake_distribution):
      continue
    config = data.get("config_cross_space_projection") or {}
    summary_path = pkl_path.parent / "logs" / "summary.csv"
    if not summary_path.is_file():
      raise ValueError(f"Missing aggregate summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    if summary.empty:
      raise ValueError(f"Empty aggregate summary: {summary_path}")
    required = [
      "subtrial_index",
      "interpolation_similarity",
      "num_anchors",
      *BASELINE_COLUMNS,
    ]
    missing = [column for column in required if column not in summary.columns]
    if missing:
      raise ValueError(
        f"Missing summary columns in {summary_path}: {', '.join(missing)}"
      )
    aggregate_mean = summary.loc[
      summary["subtrial_index"].astype(str).eq("AGGREGATE_MEAN")
    ]
    stage_required = [
      *STAGE_COLUMNS[stage],
      *(["refine_mode"] if stage != "projector_only" else []),
    ]
    stage_missing = [
      column for column in stage_required if column not in summary.columns
    ]
    selected = aggregate_mean.iloc[:0]
    if stage != "projector_only" and "refine_mode" in aggregate_mean.columns:
      selected = aggregate_mean.loc[
        aggregate_mean["refine_mode"].astype(str).eq(stage)
      ]
      if len(selected) > 1:
        raise ValueError(
          f"Expected one AGGREGATE_MEAN/{stage} row in {summary_path}, "
          f"found {len(selected)}."
        )
    reason = None
    if stage_missing:
      reason = f"missing columns in {summary_path}: {', '.join(stage_missing)}"
    elif aggregate_mean.empty:
      reason = f"no AGGREGATE_MEAN row in {summary_path}"
    elif stage == "projector_only":
      for column in RESULT_COLUMNS[stage]:
        values = pd.to_numeric(aggregate_mean[column], errors="coerce")
        if not skip_consistency_checks and (
          values.isna().any() or not math.isclose(
            values.min(), values.max(), rel_tol=1e-7, abs_tol=1e-8
          )
        ):
          raise ValueError(
            f"Inconsistent projector_only column {column} in {summary_path}."
          )
      row = aggregate_mean.iloc[0].to_dict()
    else:
      if len(selected) != 1:
        reason = (
          f"expected one AGGREGATE_MEAN/{stage} row in {summary_path}, "
          f"found {len(selected)}"
        )
      else:
        row = selected.iloc[0].to_dict()
    if reason:
      if not allow_unavailable:
        if stage_missing:
          raise ValueError(
            f"Missing summary columns in {summary_path}: {', '.join(stage_missing)}"
          )
        raise ValueError(reason[0].upper() + reason[1:] + ".")
      unavailable.append(reason)
      row = (
        aggregate_mean.iloc[0].to_dict()
        if not aggregate_mean.empty else summary.iloc[0].to_dict()
      )
    method = str(row.get("interpolation_similarity")
                 or config.get("interpolation_similarity"))
    row.update({
      "method": method,
      "source_pkl": str(pkl_path.relative_to(root)),
      "_stage_available": reason is None,
    })
    if projection == "fake_adapter":
      metadata = data.get("fake_projection_metadata") or {}
      seed = metadata.get(
        "seed", data.get("fake_projection_seed", config.get("fake_projection_seed"))
      )
      if seed is None:
        raise ValueError(f"Missing fake-adapter seed in aggregate: {pkl_path}")
      row["_adapter_seed"] = str(seed)
      row["_adapter_identity"] = _fake_adapter_identity(config, row, stage)
    rows.append(row)
    old_metadata_values.add(_model_metadata(config.get("old_model_pth")))
    new_metadata_values.add(_model_metadata(config.get("new_model_pth")))

  if not rows:
    detail = (
      f"fake/{fake_distribution}" if projection == "fake" else projection
    )
    raise ValueError(f"No {detail} aggregate PKLs found under: {root}")
  if len(old_metadata_values) != 1 or len(new_metadata_values) != 1:
    raise ValueError(f"Conflicting model metadata across aggregates under: {root}")
  old_metadata = old_metadata_values.pop()
  new_metadata = new_metadata_values.pop()
  if projection == "fake_adapter":
    seed_rows = rows
    rows = []
    for method in sorted({row["method"] for row in seed_rows}, key=_method_sort_key):
      parts = [row for row in seed_rows if row["method"] == method]
      seeds = [part["_adapter_seed"] for part in parts]
      if len(seeds) != len(set(seeds)):
        raise ValueError(
          f"Duplicate fake-adapter seed for method {method} under: {root}"
        )
      if len({part["_adapter_identity"] for part in parts}) != 1:
        raise ValueError(
          f"Inconsistent fake-adapter configuration for method {method} under: {root}"
        )
      row = dict(parts[0])
      row["_stage_available"] = all(part["_stage_available"] for part in parts)
      numeric = set(BASELINE_COLUMNS)
      for columns in STAGE_COLUMNS.values():
        numeric.update(columns)
      for column in numeric:
        values = pd.to_numeric(
          pd.Series([part.get(column) for part in parts]), errors="coerce")
        if values.notna().any():
          row[column] = float(values.mean())
      row["source_pkl"] = ";".join(part["source_pkl"] for part in parts)
      row["adapter_seed_count"] = len(parts)
      row.pop("_adapter_seed", None)
      row.pop("_adapter_identity", None)
      rows.append(row)

  methods = [row["method"] for row in rows]
  duplicates = sorted({
    method for method in methods if methods.count(method) > 1
  })
  if duplicates:
    raise ValueError(
      "Duplicate aggregate method(s) under "
      f"{root}: {', '.join(duplicates)}"
    )
  for column in BASELINE_COLUMNS:
    values = pd.to_numeric(
      pd.Series([row[column] for row in rows]), errors="coerce"
    )
    if not skip_consistency_checks and (
      values.isna().any() or not math.isclose(
        values.min(), values.max(), rel_tol=1e-7, abs_tol=1e-8
      )
    ):
      raise ValueError(f"Inconsistent baseline column {column} under: {root}")
  rows.sort(key=lambda row: _method_sort_key(row["method"]))
  return {
    "root": root,
    "source_dataset": old_metadata[0],
    "old_model": old_metadata[1],
    "target_dataset": new_metadata[0],
    "new_model": new_metadata[1],
    "rows": rows,
    "stage_available": not unavailable,
    "stage_reason": "; ".join(unavailable),
  }


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


def _escape(value: object) -> str:
  """Escape text for ordinary LaTeX cells."""
  replacements = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
  }
  return "".join(replacements.get(char, char) for char in str(value))


def _metric(value: object, decimals: int) -> str:
  """Format one numeric metric to the requested number of decimal places."""
  return f"{float(value):.{decimals}f}"


def _metric_cells(
  values: dict[str, tuple[object, object]],
  datasets: tuple[str, ...],
  decimals: int,
) -> str:
  """Render MAE/macro-MAE pairs for every dataset, using X when absent."""
  cells = []
  for dataset in datasets:
    pair = values.get(dataset)
    cells.extend(
      ("X", "X")
      if pair is None
      else (_metric(value, decimals) for value in pair)
    )
  return " & ".join(cells)


def _render_direction(
  direction: dict,
  datasets: tuple[str, ...],
  stage: str,
  decimals: int,
  *,
  unavailable: bool = False,
) -> list[str]:
  """Render one projection direction and its two native baselines."""
  row_count = len(direction["rows"]) + 2
  lines = [
    rf"    \multirow{{{row_count}}}{{*}}{{\shortstack{{"
    rf"{_escape(direction['source_dataset'])} $\to$ "
    rf"{_escape(direction['target_dataset'])} \\",
    rf"    \footnotesize {_escape(direction['old_model'])} $\to$ "
    rf"{_escape(direction['new_model'])}}}}}",
  ]
  src_micro, src_macro, target_micro, target_macro = RESULT_COLUMNS[stage]
  for row in direction["rows"]:
    values = {} if unavailable else {
      direction["source_dataset"]: (row[src_micro], row[src_macro]),
      direction["target_dataset"]: (row[target_micro], row[target_macro]),
    }
    method = METHOD_NAMES.get(
      row["method"], row["method"].replace("_", " ").title()
    )
    lines.append(
      f"    & {_escape(method)} & {int(row['num_anchors'])} & "
      f"{_metric_cells(values, datasets, decimals)} "
      rf"\\ % ({row['source_pkl']})"
    )

  lines.append(rf"    \cmidrule(lr){{2-{3 + 2 * len(datasets)}}}")
  old_values = {} if unavailable else {
    direction["source_dataset"]: (
      direction["rows"][0]["srctest_mae_micro_old"],
      direction["rows"][0]["srctest_mae_macro_old"],
    ),
  }
  new_values = {} if unavailable else {
    direction["target_dataset"]: (
      direction["rows"][0]["newtest_mae_micro_before"],
      direction["rows"][0]["newtest_mae_macro_before"],
    ),
  }
  lines.extend([
    f"    & {_escape(direction['old_model'])} "
    f"({_escape(direction['source_dataset'])}) & X & "
    f"{_metric_cells(old_values, datasets, decimals)} "
    r"\\",
    f"    & {_escape(direction['new_model'])} "
    f"({_escape(direction['target_dataset'])}) & X & "
    f"{_metric_cells(new_values, datasets, decimals)} "
    r"\\",
  ])
  return lines


def _label_slug(value: str) -> str:
  """Convert a display value into a safe lowercase LaTeX label segment."""
  return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _filename_component(value: object) -> str:
  """Preserve display case while replacing filesystem-unsafe characters."""
  return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("._-")


def _default_output_path(
  first_root: Path,
  projection: str,
  stage: str,
  fake_distribution: str | None,
) -> Path:
  """Build the default output path from the first direction's model metadata."""
  for pkl_path in sorted(first_root.rglob("*.pkl")):
    if not pkl_path.parent.name.startswith("aggregated"):
      continue
    data = _load_pkl(pkl_path)
    if not _selected_aggregate(data, projection, fake_distribution):
      continue
    config = data.get("config_cross_space_projection") or {}
    old_dataset, old_model = _model_metadata(config.get("old_model_pth"))
    new_dataset, new_model = _model_metadata(config.get("new_model_pth"))
    config_name = "-".join(
      part for part in (
        projection,
        fake_distribution if projection == "fake" else None,
        stage,
      )
      if part
    )
    filename = "_".join((
      f"{_filename_component(old_model)}-{_filename_component(old_dataset)}",
      f"{_filename_component(new_model)}-{_filename_component(new_dataset)}",
      _filename_component(config_name),
      str(int(time.time())),
    ))
    return Path(__file__).resolve().parent / "z_latex_tables" / f"{filename}.tex"
  detail = f"fake/{fake_distribution}" if projection == "fake" else projection
  raise ValueError(f"No {detail} aggregate PKLs found under: {first_root}")


def _default_root_output_path(
  root: Path,
  projection: str,
  stage: str,
  fake_distribution: str | None,
) -> Path:
  """Build a default filename for a consolidated multi-direction root."""
  config_name = "-".join(
    part for part in (
      projection,
      fake_distribution if projection == "fake" else None,
      stage,
    )
    if part
  )
  filename = "_".join((
    _filename_component(root.name),
    _filename_component(config_name),
    str(int(time.time())),
  ))
  return Path(__file__).resolve().parent / "z_latex_tables" / f"{filename}.tex"


def _render_stage_table(
  directions: list[dict],
  datasets: tuple[str, ...],
  *,
  projection: str,
  stage: str,
  decimals: int,
  unavailable_all: bool = False,
) -> str:
  """Render one table for an already validated list of directions."""
  label = "_".join([
    "tab_cross_projection",
    *(_label_slug(dataset) for dataset in datasets),
    projection,
    stage,
  ])
  reasons = [
    f"{direction['source_dataset']} to {direction['target_dataset']}: "
    f"{direction['stage_reason']}"
    for direction in directions
    if not direction["stage_available"]
  ]
  caption = STAGE_CAPTIONS[stage]
  if projection == "fake_adapter":
    caption += " — random adapter control"
  if unavailable_all:
    caption += f" (unavailable: {'; '.join(reasons)})"
  column_end = 3 + 2 * len(datasets)
  header = " & ".join(
    rf"\multicolumn{{2}}{{c}}{{\textbf{{{_escape(dataset)}}}}}"
    for dataset in datasets
  )
  cmidrules = "".join(
    rf"\cmidrule(lr){{{start}-{start + 1}}}"
    for start in range(4, column_end + 1, 2)
  )
  metric_headers = " & ".join(
    (r"\textbf{MAE}", r"\textbf{macro-MAE}") * len(datasets)
  )
  lines = [
    r"\begin{table}[H]",
    r"\centering",
    rf"\caption{{{_escape(caption)}}}",
    rf"\label{{{label}}}",
  ]
  if len(datasets) > 2:
    lines.append(r"\resizebox{\textwidth}{!}{%")
  lines.extend([
    rf"\begin{{tabular}}{{clc{'cc' * len(datasets)}}}",
    r"    \toprule",
    f"    & & & {header} " + r"\\",
    f"    {cmidrules}",
    r"    \textbf{Direction} & \textbf{Mapping method} & "
    rf"\textbf{{anchors}} & {metric_headers} " + r"\\",
    r"    \midrule",
  ])
  for index, direction in enumerate(directions):
    if index:
      lines.append(
        r"    \midrule\midrule\midrule"
        if len(directions) == 2 else r"    \midrule"
      )
    lines.extend(_render_direction(
      direction,
      datasets,
      stage,
      decimals,
      unavailable=unavailable_all,
    ))
  lines.extend([r"    \bottomrule", r"\end{tabular}"])
  if len(datasets) > 2:
    lines.append("}")
  lines.append(r"\end{table}")
  return "\n".join(lines) + "\n"


def _generate_stage_table(
  first_root: str | Path,
  second_root: str | Path,
  *,
  projection: str,
  stage: str,
  decimals: int,
  fake_distribution: str | None = None,
  allow_unavailable: bool = False,
  skip_consistency_checks: bool = False,
) -> str:
  """Generate one stage's complete two-direction cross-projection table."""
  first = _load_direction(
    Path(first_root), projection, stage, fake_distribution,
    allow_unavailable=allow_unavailable,
    skip_consistency_checks=skip_consistency_checks,
  )
  second = _load_direction(
    Path(second_root), projection, stage, fake_distribution,
    allow_unavailable=allow_unavailable,
    skip_consistency_checks=skip_consistency_checks,
  )
  first_methods = {row["method"] for row in first["rows"]}
  second_methods = {row["method"] for row in second["rows"]}
  if first_methods != second_methods:
    missing = sorted(first_methods ^ second_methods)
    raise ValueError(
      "Projection method sets differ between directions: " + ", ".join(missing)
    )
  if (
    first["source_dataset"] != second["target_dataset"]
    or first["target_dataset"] != second["source_dataset"]
  ):
    raise ValueError(
      "Experiment roots are not inverse datasets: "
      f"{first['source_dataset']} -> {first['target_dataset']} and "
      f"{second['source_dataset']} -> {second['target_dataset']}."
    )
  unavailable = not first["stage_available"] or not second["stage_available"]
  return _render_stage_table(
    [first, second],
    (first["source_dataset"], first["target_dataset"]),
    projection=projection,
    stage=stage,
    decimals=decimals,
    unavailable_all=unavailable,
  )


def _generate_root_stage_table(
  root: str | Path,
  *,
  projection: str,
  stage: str,
  decimals: int,
  fake_distribution: str | None = None,
  skip_consistency_checks: bool = False,
) -> str:
  """Generate one stage table from a root-level consolidated summary."""
  directions = _root_summary_directions(
    Path(root),
    projection,
    stage,
    fake_distribution,
    skip_consistency_checks=skip_consistency_checks,
  )
  if not directions:
    return ""
  datasets = tuple(sorted({
    dataset
    for direction in directions
    for dataset in (
      direction["source_dataset"], direction["target_dataset"]
    )
  }))
  return _render_stage_table(
    directions,
    datasets,
    projection=projection,
    stage=stage,
    decimals=decimals,
  )


def _comparison_pair(
  row: dict | None, columns: tuple[str, str], decimals: int,
) -> str:
  """Format a comparison cell, rejecting malformed or non-finite metrics."""
  if row is None:
    return "X / X"
  values = []
  for column in columns:
    try:
      value = float(row[column])
    except (KeyError, TypeError, ValueError):
      raise ValueError(f"Invalid metric {column} in {row['source_pkl']}.") from None
    if not math.isfinite(value):
      raise ValueError(f"Non-finite metric {column} in {row['source_pkl']}.")
    values.append(_metric(value, decimals))
  return " / ".join(values)


def _generate_comparison_table(
  first_root: str | Path,
  second_root: str | Path | None,
  *,
  projection: str,
  stages: tuple[str, ...],
  fake_distribution: str | None,
  decimals: int,
  skip_consistency_checks: bool,
) -> str:
  """Merge selected stages by direction and method into one comparison table."""
  grouped = {}
  for stage in stages:
    if second_root is None:
      directions = _root_summary_directions(
        Path(first_root), projection, stage, fake_distribution,
        allow_unavailable=True, skip_consistency_checks=skip_consistency_checks,
      )
    else:
      directions = [
        _load_direction(
          Path(root), projection, stage, fake_distribution,
          allow_unavailable=True, skip_consistency_checks=skip_consistency_checks,
        )
        for root in (first_root, second_root)
      ]
      first, second = directions
      if (first['source_dataset'] != second['target_dataset']
          or first['target_dataset'] != second['source_dataset']):
        raise ValueError("Experiment roots are not inverse datasets.")
    for direction in directions:
      key = tuple(direction[name] for name in (
        "source_dataset", "target_dataset", "old_model", "new_model",
      ))
      if key not in grouped:
        grouped[key] = {**direction, "methods": {}, "baseline": direction['rows'][0]}
      merged = grouped[key]
      for row in direction['rows']:
        if str(row['subtrial_index']) != "AGGREGATE_MEAN":
          raise ValueError(f"Missing AGGREGATE_MEAN row in {row['source_pkl']}.")
        baseline = merged['baseline']
        # Validate every baseline even if repeated-value checks are disabled.
        _comparison_pair(row, tuple(BASELINE_COLUMNS[:2]), decimals)
        _comparison_pair(row, tuple(BASELINE_COLUMNS[2:]), decimals)
        if not skip_consistency_checks:
          for column in BASELINE_COLUMNS:
            if not math.isclose(float(row[column]), float(baseline[column]),
                                rel_tol=1e-7, abs_tol=1e-8):
              raise ValueError(f"Inconsistent baseline column {column} across stages.")
        try:
          anchors = float(row['num_anchors'])
        except (TypeError, ValueError):
          raise ValueError(f"Invalid anchor count in {row['source_pkl']}.") from None
        if not math.isfinite(anchors) or not anchors.is_integer() or anchors <= 0:
          raise ValueError(f"Invalid anchor count in {row['source_pkl']}.")
        method = merged['methods'].setdefault(row['method'], {
          "anchors": int(anchors), "stages": {}, "source_pkls": [],
        })
        if method['anchors'] != int(anchors):
          raise ValueError(
            f"Inconsistent anchor counts across stages for {row['method']}: "
            f"{direction['source_dataset']} -> {direction['target_dataset']}."
          )
        method['stages'][stage] = row if row['_stage_available'] else None
        if row['source_pkl'] not in method['source_pkls']:
          method['source_pkls'].append(row['source_pkl'])
  if not grouped:
    raise ValueError(f"No {projection} aggregate rows found under: {first_root}")

  native_stages = tuple(stage for stage in stages if stage != "projector_only")
  counts = {method['anchors'] for direction in grouped.values()
            for method in direction['methods'].values()}
  show_anchors = len(counts) > 1
  prefix_columns = 3 if show_anchors else 2
  source_width, target_width = 1 + len(stages), 1 + len(native_stages)
  source_start = prefix_columns + 1
  source_end = prefix_columns + source_width
  column_end = source_end + target_width
  headers = [r"\textbf{Direction}", r"\textbf{Map method}"]
  if show_anchors:
    headers.append(r"\textbf{Anchors}")
  for baseline_name, group_stages in (
    ("Native source baseline", stages), ("Native target baseline", native_stages),
  ):
    headers.extend(
      rf"\makecell{{\textbf{{{name}}} \\ \textbf{{(MAE / Macro-MAE)}}}}"
      for name in (baseline_name, *(COMPARISON_STAGE_NAMES[stage] for stage in group_stages))
    )
  roles = [""] * prefix_columns + [
    rf"\multicolumn{{{source_width}}}{{c}}{{\textbf{{Projected-source}}}}",
    rf"\multicolumn{{{target_width}}}{{c}}{{\textbf{{Native-target}}}}",
  ]
  datasets = sorted({dataset for direction in grouped.values()
                     for dataset in (direction['source_dataset'], direction['target_dataset'])})
  label = "_".join((
    "tab_cross_projection", *(_label_slug(dataset) for dataset in datasets),
    projection, "comparison", *stages,
    *((fake_distribution,) if fake_distribution else ()),
  ))
  lines = [
    r"\begin{table}[H]", r"\centering", r"\resizebox{\textwidth}{!}{%",
    rf"\begin{{tabular}}{{{'ll' + ('c' if show_anchors else '') + 'c' * (source_width + target_width)}}}",
    r"    \toprule", "    " + " & ".join(roles) + r" \\",
    rf"    \cmidrule(lr){{{source_start}-{source_end}}}"
    rf"\cmidrule(lr){{{source_end + 1}-{column_end}}}",
    "    " + " & ".join(headers) + r" \\", r"    \midrule",
  ]
  for index, direction in enumerate(grouped.values()):
    if index:
      lines.append(r"    \midrule")
    methods = sorted(direction['methods'], key=_method_sort_key)
    row_count = len(methods)
    lines.extend([
      rf"    \multirow{{{row_count}}}{{*}}{{\shortstack{{"
      rf"{_escape(direction['source_dataset'])} $\to$ "
      rf"{_escape(direction['target_dataset'])} \\",
      rf"    \footnotesize {_escape(direction['old_model'])} $\to$ "
      rf"{_escape(direction['new_model'])}}}}}",
    ])
    for method_index, method_name in enumerate(methods):
      method = direction['methods'][method_name]
      cells = [_escape(METHOD_NAMES.get(method_name, method_name.replace('_', ' ').title()))]
      if show_anchors:
        cells.append(str(method['anchors']))
      for baseline_columns, group_stages, result_offset in (
        (tuple(BASELINE_COLUMNS[:2]), stages, 0),
        (tuple(BASELINE_COLUMNS[2:]), native_stages, 2),
      ):
        value = _comparison_pair(direction['baseline'], baseline_columns, decimals)
        cells.append(rf"\multirow{{{row_count}}}{{*}}{{{value}}}" if method_index == 0 else "")
        cells.extend(
          _comparison_pair(method['stages'].get(stage),
                           RESULT_COLUMNS[stage][result_offset:result_offset + 2], decimals)
          for stage in group_stages
        )
      lines.append("    & " + " & ".join(cells) + r" \\ % ("
                   + ";".join(method['source_pkls']) + ")")
  caption = (
    "Comparison of " + ", ".join(COMPARISON_STAGE_NAMES[stage] for stage in stages)
    + ". Results are reported as MAE / Macro-MAE. Projected-source evaluates "
    "mapped source representations with the target regression head; native-target "
    "evaluates native target representations. Baselines report each original model "
    "on its own dataset."
  )
  if projection == "fake_adapter":
    caption += " Random adapter control."
  elif projection == "fake":
    caption += f" Fake embeddings control ({fake_distribution})."
  if not show_anchors:
    caption += f" All mapping methods use {next(iter(counts))} anchors."
  if any(row is None for direction in grouped.values()
         for method in direction['methods'].values() for row in method['stages'].values()):
    caption += " X / X denotes unavailable stage results."
  caption += " Lower values indicate better performance."
  lines.extend([
    r"    \bottomrule", r"\end{tabular}", "}",
    rf"\caption{{{_escape(caption)}}}", rf"\label{{{label}}}", r"\end{table}",
  ])
  return "\n".join(lines) + "\n"


def generate_table(
  first_root: str | Path,
  second_root: str | Path | None = None,
  *,
  projection: str,
  stage: str | None = None,
  compare_stages: tuple[str, ...] | list[str] | None = None,
  fake_distribution: str | None = None,
  decimals: int = 2,
  skip_consistency_checks: bool = False,
) -> str:
  """Generate separate stage tables, or one table comparing selected stages."""
  if not isinstance(decimals, int) or decimals < 0:
    raise ValueError("decimals must be non-negative.")
  if compare_stages is not None:
    if stage is not None:
      raise ValueError("stage and compare_stages are mutually exclusive.")
    if (not isinstance(compare_stages, (list, tuple))
        or len(compare_stages) not in (2, 3)
        or any(current not in COMPARISON_STAGE_NAMES for current in compare_stages)
        or len(set(compare_stages)) != len(compare_stages)):
      raise ValueError(
        "compare_stages must contain two or three distinct stages from "
        "projector_only, linear_only, projector_linear."
      )
    return _generate_comparison_table(
      first_root, second_root, projection=projection, stages=tuple(compare_stages),
      fake_distribution=fake_distribution, decimals=decimals,
      skip_consistency_checks=skip_consistency_checks,
    )
  if stage not in (*STAGES, "all"):
    raise ValueError(f"Unknown stage: {stage}")
  if second_root is None:
    stages = STAGES if stage == "all" else (stage,)
    tables = []
    for current in stages:
      table = _generate_root_stage_table(
        first_root,
        projection=projection,
        stage=current,
        decimals=decimals,
        fake_distribution=fake_distribution,
        skip_consistency_checks=skip_consistency_checks,
      )
      if table:
        tables.append(table)
    if not tables:
      detail = f"fake/{fake_distribution}" if projection == "fake" else projection
      raise ValueError(
        f"No {detail}/{stage} aggregate rows found in "
        f"{Path(first_root) / 'aggregated_summary.csv'}"
      )
    return "\n".join(tables)
  if stage != "all":
    return _generate_stage_table(
      first_root,
      second_root,
      projection=projection,
      stage=stage,
      decimals=decimals,
      fake_distribution=fake_distribution,
      skip_consistency_checks=skip_consistency_checks,
    )
  return "\n".join(
    _generate_stage_table(
      first_root,
      second_root,
      projection=projection,
      stage=current,
      decimals=decimals,
      fake_distribution=fake_distribution,
      allow_unavailable=True,
      skip_consistency_checks=skip_consistency_checks,
    )
    for current in STAGES
  )


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
    description="Generate a cross-projection LaTeX table."
  )
  parser.add_argument("first_root", type=Path)
  parser.add_argument(
    "second_root",
    type=Path,
    nargs="?",
    help="Inverse direction root; omit for consolidated-summary root mode.",
  )
  parser.add_argument(
    "--projection", choices=("real", "fake", "fake_adapter"), required=True)
  parser.add_argument(
    "--fake-distribution",
    choices=("matched_gaussian", "standard_normal"),
  )
  stage_options = parser.add_mutually_exclusive_group(required=True)
  stage_options.add_argument(
    "--stage",
    choices=(*STAGES, "all"),
  )
  stage_options.add_argument(
    "--compare-stages", nargs="+", choices=tuple(COMPARISON_STAGE_NAMES),
    help="Compare two or three selected stages side by side in one table.",
  )
  parser.add_argument("--decimals", type=int, default=2)
  parser.add_argument(
    "--skip-consistency-checks",
    action="store_true",
    help="Skip repeated projector-only and baseline metric consistency checks.",
  )
  parser.add_argument("--output", type=Path)
  return parser.parse_args()


def main() -> None:
  """Generate the requested table and write it to the resolved output path."""
  args = parse_args()
  if args.projection == "fake" and not args.fake_distribution:
    raise SystemExit("--fake-distribution is required for fake projections.")
  if args.projection != "fake" and args.fake_distribution:
    raise SystemExit("--fake-distribution is only valid for fake projections.")
  try:
    latex = generate_table(
      args.first_root,
      args.second_root,
      projection=args.projection,
      stage=args.stage,
      compare_stages=args.compare_stages,
      fake_distribution=args.fake_distribution,
      decimals=args.decimals,
      skip_consistency_checks=args.skip_consistency_checks,
    )
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
  try:
    output_stage = (
      "comparison-" + "-".join(args.compare_stages)
      if args.compare_stages is not None else args.stage
    )
    output = args.output or (
      _default_output_path(
        args.first_root,
        args.projection,
        output_stage,
        args.fake_distribution,
      )
      if args.second_root is not None
      else _default_root_output_path(
        args.first_root,
        args.projection,
        output_stage,
        args.fake_distribution,
      )
    )
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(latex, encoding="utf-8")
  print(f"Saved LaTeX table to: {output}")


if __name__ == "__main__":
  main()

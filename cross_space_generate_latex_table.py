"""Generate a two-direction LaTeX table from cross-projection aggregates."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import time
from pathlib import Path

import pandas as pd


METHOD_ORDER = ["linear", "mlp", "procrustes", "linear_close", "autoencoder"]
METHOD_NAMES = {
  "linear": "Linear (SGD)",
  "mlp": "MLP",
  "procrustes": "Procrustes",
  "linear_close": "Linear (closed form)",
  "autoencoder": "Autoencoder",
}
MODEL_NAMES = {
  "VIDEOMAE_v2_S": "VMAEv2-S",
}
STAGES = ("projector_only", "linear_only", "projector_linear")
STAGE_CAPTIONS = {
  "projector_only": "Projector-only cross-projection results",
  "linear_only": "Linear-only refinement cross-projection results",
  "projector_linear": "Projector-and-linear refinement cross-projection results",
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
    return str(value[0])
  if isinstance(value, str) and value:
    return Path(value).parts[0]
  raise ValueError("Model config has no usable dataset path.")


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
  is_fake = bool(config.get("fake_projection") or distribution)
  if projection == "real":
    return not is_fake
  return is_fake and distribution == fake_distribution


def _method_sort_key(method: str) -> tuple[int, str]:
  """Sort known projection methods like the reference table."""
  try:
    return METHOD_ORDER.index(method), method
  except ValueError:
    return len(METHOD_ORDER), method


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
    reason = None
    if stage_missing:
      reason = f"missing columns in {summary_path}: {', '.join(stage_missing)}"
    elif aggregate_mean.empty:
      reason = f"no AGGREGATE_MEAN row in {summary_path}"
    elif stage == "projector_only":
      for column in RESULT_COLUMNS[stage]:
        values = pd.to_numeric(aggregate_mean[column], errors="coerce")
        if not skip_consistency_checks and (
          values.isna().any() or values.max() - values.min() > 1e-10
        ):
          raise ValueError(
            f"Inconsistent projector_only column {column} in {summary_path}."
          )
      row = aggregate_mean.iloc[0].to_dict()
    else:
      selected = aggregate_mean.loc[
        aggregate_mean["refine_mode"].astype(str).eq(stage)
      ]
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
    })
    rows.append(row)
    old_metadata_values.add(_model_metadata(config.get("old_model_pth")))
    new_metadata_values.add(_model_metadata(config.get("new_model_pth")))

  if not rows:
    detail = (
      f"fake/{fake_distribution}" if projection == "fake" else "real"
    )
    raise ValueError(f"No {detail} aggregate PKLs found under: {root}")
  if len(old_metadata_values) != 1 or len(new_metadata_values) != 1:
    raise ValueError(f"Conflicting model metadata across aggregates under: {root}")
  old_metadata = old_metadata_values.pop()
  new_metadata = new_metadata_values.pop()
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
      values.isna().any() or values.max() - values.min() > 1e-10
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
  datasets: tuple[str, str],
  decimals: int,
) -> str:
  """Render two MAE/macro-MAE dataset pairs, using X when absent."""
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
  datasets: tuple[str, str],
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

  lines.append(r"    \cmidrule(lr){2-7}")
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
  detail = f"fake/{fake_distribution}" if projection == "fake" else "real"
  raise ValueError(f"No {detail} aggregate PKLs found under: {first_root}")


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
  datasets = (first["source_dataset"], first["target_dataset"])
  label = "_".join([
    "tab_cross_projection",
    _label_slug(datasets[0]),
    _label_slug(datasets[1]),
    projection,
    stage,
  ])
  unavailable = not first["stage_available"] or not second["stage_available"]
  reasons = [
    f"{direction['source_dataset']} to {direction['target_dataset']}: "
    f"{direction['stage_reason']}"
    for direction in (first, second)
    if not direction["stage_available"]
  ]
  caption = STAGE_CAPTIONS[stage]
  if unavailable:
    caption += f" (unavailable: {'; '.join(reasons)})"
  lines = [
    r"\begin{table}[H]",
    r"\centering",
    rf"\caption{{{_escape(caption)}}}",
    rf"\label{{{label}}}",
    r"\begin{tabular}{clccccc}",
    r"    \toprule",
    "    & & & "
    rf"\multicolumn{{2}}{{c}}{{\textbf{{{_escape(datasets[0])}}}}}"
    " & "
    rf"\multicolumn{{2}}{{c}}{{\textbf{{{_escape(datasets[1])}}}}} \\",
    r"    \cmidrule(lr){4-5}\cmidrule(lr){6-7}",
    r"    \textbf{Direction} & \textbf{Mapping method} & "
    r"\textbf{anchors} & \textbf{MAE} & \textbf{macro-MAE} & "
    r"\textbf{MAE} & \textbf{macro-MAE} \\",
    r"    \midrule",
    *_render_direction(
      first, datasets, stage, decimals, unavailable=unavailable
    ),
    r"    \midrule\midrule\midrule",
    *_render_direction(
      second, datasets, stage, decimals, unavailable=unavailable
    ),
    r"    \bottomrule",
    r"\end{tabular}",
    r"\end{table}",
  ]
  return "\n".join(lines) + "\n"


def generate_table(
  first_root: str | Path,
  second_root: str | Path,
  *,
  projection: str,
  stage: str,
  fake_distribution: str | None = None,
  decimals: int = 2,
  skip_consistency_checks: bool = False,
) -> str:
  """Generate one stage, or all available stages, as LaTeX tables."""
  if not isinstance(decimals, int) or decimals < 0:
    raise ValueError("decimals must be non-negative.")
  if stage not in (*STAGES, "all"):
    raise ValueError(f"Unknown stage: {stage}")
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
    description="Generate a two-direction cross-projection LaTeX table."
  )
  parser.add_argument("first_root", type=Path)
  parser.add_argument("second_root", type=Path)
  parser.add_argument("--projection", choices=("real", "fake"), required=True)
  parser.add_argument(
    "--fake-distribution",
    choices=("matched_gaussian", "standard_normal"),
  )
  parser.add_argument(
    "--stage",
    choices=(*STAGES, "all"),
    required=True,
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
  if args.projection == "real" and args.fake_distribution:
    raise SystemExit("--fake-distribution is only valid for fake projections.")
  try:
    latex = generate_table(
      args.first_root,
      args.second_root,
      projection=args.projection,
      stage=args.stage,
      fake_distribution=args.fake_distribution,
      decimals=args.decimals,
      skip_consistency_checks=args.skip_consistency_checks,
    )
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
  try:
    output = args.output or _default_output_path(
      args.first_root,
      args.projection,
      args.stage,
      args.fake_distribution,
    )
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(latex, encoding="utf-8")
  print(f"Saved LaTeX table to: {output}")


if __name__ == "__main__":
  main()

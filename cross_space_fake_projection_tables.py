#!/usr/bin/env python3
"""Summarize fitted-versus-random adapter results as CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from cross_space_generate_latex_table import _model_metadata


METHOD_ORDER = ("linear", "mlp", "procrustes", "linear_close", "autoencoder")
METHOD_NAMES = {
  "linear": "Linear",
  "mlp": "MLP",
  "procrustes": "Procrustes",
  "linear_close": "Linear (closed form)",
  "autoencoder": "Autoencoder",
}
MODE_NAMES = {
  "linear_only": "head_only",
  "projector_linear": "joint",
}
METRICS = {
  "micro_mae": (
    "trained_adapter_mae_micro", "random_adapter_mae_micro"),
  "macro_mae": (
    "trained_adapter_mae_macro", "random_adapter_mae_macro"),
}
CONFIG_COLUMNS = (
  "old_model_pth", "new_model_pth", "num_anchors",
  "anchor_selection_type", "csv_anchor_selection", "old_model_csv",
  "mlp_activation", "mlp_num_layers", "weighting_method", "rbf_sigma",
  "temperature", "projector_config", "refinement_config", "refine_mode",
)
SUMMARY_COLUMNS = (
  "direction", "projection", "refinement_mode", "metric", "fitted_mae",
  "random_mae_mean", "random_mae_seed_sd", "increase_pct", "delta_mae",
  "checkpoint_pair_count", "seed_count",
)


def _display_dataset(name: str) -> str:
  return "BioVid" if name.upper() == "BIOVID" else name


def _method_sort_key(method: str) -> tuple[int, str]:
  try:
    return METHOD_ORDER.index(method), method
  except ValueError:
    return len(METHOD_ORDER), method


def _input_csv(root: Path, fast_mode: bool) -> Path:
  suffix = "_fast" if fast_mode else ""
  return root / f"fake_adapter_seed_results{suffix}.csv"


def _add_directions(frame: pd.DataFrame) -> pd.DataFrame:
  missing = sorted({"old_model_pth", "new_model_pth"} - set(frame.columns))
  if missing:
    raise ValueError(f"Missing checkpoint columns: {', '.join(missing)}")
  metadata = {}
  for old_path, new_path in frame[["old_model_pth", "new_model_pth"]].drop_duplicates().itertuples(index=False):
    old_dataset, _ = _model_metadata(old_path)
    new_dataset, _ = _model_metadata(new_path)
    metadata[(old_path, new_path)] = (
      f"{_display_dataset(old_dataset)} -> {_display_dataset(new_dataset)}")
  frame = frame.copy()
  frame["_direction"] = [
    metadata[(old_path, new_path)]
    for old_path, new_path in frame[["old_model_pth", "new_model_pth"]].itertuples(index=False)
  ]
  return frame


def _exclude_datasets(frame: pd.DataFrame, datasets: set[str]) -> pd.DataFrame:
  excluded = {dataset.casefold() for dataset in datasets}
  if not excluded:
    return frame
  keep = frame["_direction"].map(
    lambda direction: not any(
      dataset.strip().casefold() in excluded
      for dataset in direction.split(" -> ")
    )
  )
  selected = frame.loc[keep].copy()
  if selected.empty:
    raise ValueError(
      f"No rows remain after excluding datasets: {', '.join(sorted(datasets))}")
  return selected


def _normalized_key_value(value: object) -> object:
  return None if pd.isna(value) else value


def _configuration_key(row: pd.Series) -> tuple[object, ...]:
  return tuple(_normalized_key_value(row.get(column)) for column in CONFIG_COLUMNS)


def _validated_rows(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
  required = {
    "status", "replay_error", "interpolation_similarity", "refinement_mode",
    "fake_projection_seed", "source_pkl_path", *CONFIG_COLUMNS,
    *(column for pair in METRICS.values() for column in pair),
  }
  missing = sorted(required - set(frame.columns))
  if missing:
    raise ValueError(f"Missing random-adapter columns under {root}: {', '.join(missing)}")
  if frame.empty:
    raise ValueError(f"Random-adapter results are empty under: {root}")
  if not frame["status"].astype(str).eq("success").all():
    raise ValueError(f"Non-successful random-adapter rows under: {root}")
  errors = frame["replay_error"].fillna("").astype(str).str.strip()
  if errors.ne("").any():
    raise ValueError(f"Random-adapter replay errors under: {root}")
  actual_modes = set(frame["refinement_mode"].astype(str))
  unknown_modes = sorted(actual_modes - set(MODE_NAMES))
  if unknown_modes:
    raise ValueError(f"Unsupported refinement modes under {root}: {', '.join(unknown_modes)}")
  missing_modes = sorted(set(MODE_NAMES) - actual_modes)
  if missing_modes:
    raise ValueError(
      f"Missing required refinement modes under {root}: {', '.join(missing_modes)}")
  numeric = ["fake_projection_seed", *(column for pair in METRICS.values() for column in pair)]
  for column in numeric:
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[column].isna().any() or not frame[column].map(math.isfinite).all():
      raise ValueError(f"Non-finite {column} values under: {root}")
  frame["fake_projection_seed"] = frame["fake_projection_seed"].astype(int)
  frame["_configuration_key"] = frame.apply(_configuration_key, axis=1)

  identity = [
    "interpolation_similarity", "refinement_mode", "fake_projection_seed",
    "_configuration_key",
  ]
  metrics = [column for pair in METRICS.values() for column in pair]
  for key, duplicates in frame.groupby(identity, dropna=False, sort=False):
    if len(duplicates) > 1 and any(duplicates[column].nunique(dropna=False) != 1 for column in metrics):
      raise ValueError(f"Conflicting duplicate random-adapter rows under {root}: {key}")
  frame = frame.drop_duplicates(identity).copy()

  all_seeds = set(frame["fake_projection_seed"])
  if len(all_seeds) < 2:
    raise ValueError(f"At least two random-adapter seeds are required under: {root}")
  pair_identity = ["interpolation_similarity", "refinement_mode", "_configuration_key"]
  for key, rows in frame.groupby(pair_identity, dropna=False, sort=False):
    seeds = set(rows["fake_projection_seed"])
    if seeds != all_seeds:
      raise ValueError(f"Incomplete seed coverage under {root}: {key}")
    for fitted, _ in METRICS.values():
      values = rows[fitted].to_numpy(dtype=float)
      # Each seed replays the fitted adapter on CPU in float32, so allow a few ULPs.
      if not all(math.isclose(values[0], value, rel_tol=1e-6, abs_tol=1e-12)
                 for value in values[1:]):
        raise ValueError(f"Seed-varying fitted metric {fitted} under {root}: {key}")
  return frame


def summarize_root(
  root: str | Path,
  fast_mode: bool = False,
  exclude_datasets: set[str] | None = None,
) -> pd.DataFrame:
  """Return one long-form fitted/random adapter summary for an experiment root."""
  root = Path(root).resolve()
  csv_path = _input_csv(root, fast_mode)
  if not csv_path.is_file():
    raise ValueError(f"Missing random-adapter results: {csv_path}")
  frame = _add_directions(pd.read_csv(csv_path))
  frame = _exclude_datasets(frame, exclude_datasets or set())
  frame = _validated_rows(frame, root)
  summaries = []
  grouped = frame.groupby(
    ["_direction", "interpolation_similarity", "refinement_mode"], sort=False)
  for (direction, method, mode), rows in grouped:
    pair_count = rows["_configuration_key"].nunique()
    seed_count = rows["fake_projection_seed"].nunique()
    for metric, (fitted_column, random_column) in METRICS.items():
      fitted = float(rows.groupby("_configuration_key")[fitted_column].first().mean())
      seed_means = rows.groupby("fake_projection_seed")[random_column].mean()
      random_mean = float(seed_means.mean())
      delta = random_mean - fitted
      summaries.append({
        "direction": direction,
        "projection": METHOD_NAMES.get(method, str(method).replace("_", " ").title()),
        "refinement_mode": MODE_NAMES[mode],
        "metric": metric,
        "fitted_mae": fitted,
        "random_mae_mean": random_mean,
        "random_mae_seed_sd": float(seed_means.std(ddof=1)),
        "increase_pct": math.nan if fitted == 0 else 100.0 * delta / fitted,
        "delta_mae": delta,
        "checkpoint_pair_count": int(pair_count),
        "seed_count": int(seed_count),
        "_method": method,
      })
  result = pd.DataFrame(summaries)
  direction_order = {
    value: index for index, value in enumerate(result["direction"].drop_duplicates())
  }
  result["_direction_order"] = result["direction"].map(direction_order)
  result["_method_order"] = result["_method"].map(_method_sort_key)
  result = result.sort_values(
    ["_direction_order", "_method_order", "refinement_mode", "metric"],
    kind="stable",
  )
  return result.loc[:, SUMMARY_COLUMNS].reset_index(drop=True)


def _latex_escape(value: object) -> str:
  replacements = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
  }
  return "".join(replacements.get(character, character) for character in str(value))


def _latex_direction(direction: str) -> str:
  source, target = direction.split(" -> ", maxsplit=1)
  return rf"{_latex_escape(source)} $\rightarrow$ {_latex_escape(target)}"


def _label_slug(value: object) -> str:
  return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _filename_slug(value: object) -> str:
  return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")


def _number(value: object, decimals: int) -> str:
  numeric = float(value)
  return "--" if not math.isfinite(numeric) else f"{numeric:.{decimals}f}"


def _render_table(
  frame: pd.DataFrame,
  mode: str,
  metric: str,
  label_prefix: str,
) -> str:
  selected = frame.loc[
    frame["refinement_mode"].eq(mode) & frame["metric"].eq(metric)]
  mode_title = "Head-only refinement" if mode == "head_only" else "Joint refinement"
  metric_title = "micro-MAE" if metric == "micro_mae" else "macro-MAE"
  lines = [
    r"\begin{table}[H]",
    r"\centering",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{llrrrr}",
    r"\toprule",
    r"\textbf{Direction} & \textbf{Projection} & \textbf{Fitted MAE} & "
    r"\textbf{Random MAE} & \textbf{Increase (\%)} & "
    r"\textbf{$\Delta$(MAE)} \\",
    r"\midrule",
  ]
  directions = selected["direction"].drop_duplicates().tolist()
  for direction_index, direction in enumerate(directions):
    rows = selected.loc[selected["direction"].eq(direction)]
    for row_index, row in enumerate(rows.itertuples(index=False)):
      direction_cell = (
        rf"\multirow{{{len(rows)}}}{{*}}{{{_latex_direction(direction)}}}"
        if row_index == 0 else ""
      )
      random = (
        rf"\({_number(row.random_mae_mean, 4)}"
        rf"\pm{_number(row.random_mae_seed_sd, 4)}\)"
      )
      lines.append(
        f"{direction_cell} & {_latex_escape(row.projection)} & "
        f"{_number(row.fitted_mae, 4)} & {random} & "
        f"{_number(row.increase_pct, 1)} & {_number(row.delta_mae, 4)} "
        r"\\"
      )
    if direction_index + 1 < len(directions):
      lines.append(r"\midrule")
  label = _label_slug(f"{label_prefix}_{mode}_{metric}")
  lines.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"}",
    rf"\caption{{Random-adapter ablation: {mode_title}, {metric_title}.}}",
    rf"\label{{tab:{label}}}",
    r"\end{table}",
  ])
  return "\n".join(lines)


def render_latex(frame: pd.DataFrame, label_prefix: str) -> str:
  """Render the four approved mode/metric tables from a long summary frame."""
  tables = [
    _render_table(frame, mode, metric, label_prefix)
    for mode in ("head_only", "joint")
    for metric in ("micro_mae", "macro_mae")
  ]
  return "\n\n".join(tables) + "\n"


def _create_run_directory(output_parent: Path) -> Path:
  output_parent.mkdir(parents=True, exist_ok=True)
  stem = f"random_adapter_ablation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
  candidate = output_parent / stem
  suffix = 1
  while candidate.exists():
    candidate = output_parent / f"{stem}_{suffix}"
    suffix += 1
  candidate.mkdir()
  return candidate


def generate(
  roots: list[str | Path],
  output_dir: str | Path,
  fast_mode: bool = False,
  exclude_datasets: set[str] | None = None,
) -> Path:
  """Generate per-direction and optional combined files in a unique run folder."""
  if not roots:
    raise ValueError("At least one experiment root is required.")
  resolved = [Path(root).resolve() for root in roots]
  summaries = [
    summarize_root(
      root, fast_mode=fast_mode, exclude_datasets=exclude_datasets)
    for root in resolved
  ]
  run_dir = _create_run_directory(Path(output_dir).resolve())
  for index, (root, summary) in enumerate(zip(resolved, summaries), start=1):
    stem = f"{index:02d}_{_filename_slug(root.name)}_random_adapter_ablation"
    summary.to_csv(run_dir / f"{stem}.csv", index=False)
    (run_dir / f"{stem}.tex").write_text(
      render_latex(summary, f"{run_dir.name}_{stem}"), encoding="utf-8")
  if len(summaries) > 1:
    combined = pd.concat(summaries, ignore_index=True)
    stem = "comparison_random_adapter_ablation"
    combined.to_csv(run_dir / f"{stem}.csv", index=False)
    (run_dir / f"{stem}.tex").write_text(
      render_latex(combined, f"{run_dir.name}_{stem}"), encoding="utf-8")
  return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Generate fitted-versus-random adapter CSV and LaTeX tables.")
  parser.add_argument("roots", nargs="+", type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument(
    "--fast-mode", action="store_true",
    help="Read fake_adapter_seed_results_fast.csv instead of the full-run file.")
  parser.add_argument(
    "--exclude-dataset", action="append", default=[], metavar="DATASET",
    help="Exclude rows whose source or target matches DATASET; repeat as needed.")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    run_dir = generate(
      args.roots,
      args.output_dir,
      fast_mode=args.fast_mode,
      exclude_datasets=set(args.exclude_dataset),
    )
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
  print(f"Saved random-adapter tables to: {run_dir}")
  return 0


if __name__ == "__main__":
  main()

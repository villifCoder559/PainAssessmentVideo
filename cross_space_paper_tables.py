#!/usr/bin/env python3
"""Render paper-format LaTeX tables from cross-space reproduction outputs.

Subcommands (see paper_reproduction_commands.txt for the full pipeline):
  anchor          Random vs quality anchors after joint refinement.
  refinement      Projector-only fitting vs joint projector--regressor refinement.
  closed-form     Anchor-only and refined errors of the Procrustes and OLS mappings.
  synthetic       Real vs standard-normal synthetic source embeddings.
  frozen          Learned projector vs frozen random projector (linear-head refinement).
  random-adapter  Fitted vs random-initialized projector under joint refinement.
  confusion       Row-normalized confusion matrices for one direction and method.

Inputs are produced by cross_space_logs.py --only_aggregated (aggregated_summary.csv)
and cross_space_fake_projection.py (aggregated_summary_fake.csv, fake-adapter PKLs).
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cross_space_fake_projection_tables as adapter_tables
from cross_space_generate_latex_table import (
  METHOD_NAMES,
  _method_sort_key,
  _model_metadata,
  _root_summary_directions,
)


DATASET_NAMES = {"BIOVID": "BioVid"}
# cross_space_fake_projection_tables reports display names; map them back to methods.
ADAPTER_METHODS = {name: method for method, name in adapter_tables.METHOD_NAMES.items()}
ADAPTER_DIRECTIONS = {
  adapter_tables._display_dataset(name): name for name in ("BIOVID", "MIntPAIN", "UNBC")
}


def _dataset(name: str) -> str:
  return DATASET_NAMES.get(name, name)


def _direction_cell(source: str, target: str) -> str:
  return rf"{_dataset(source)} $\rightarrow$ {_dataset(target)}"


def _direction_key(direction: tuple[str, str]) -> tuple:
  """Order directions like cross_space_generate_latex_table: dataset pair, then source."""
  return tuple(sorted(direction)), direction


def _method_name(method: str) -> str:
  return METHOD_NAMES.get(method, method.replace("_", " ").title())


def _number(value: object, decimals: int) -> str:
  value = float(value)
  if not math.isfinite(value):
    raise ValueError(f"Non-finite metric value: {value}")
  return f"{value:.{decimals}f}"


def _pair(values: tuple[object, object], decimals: int) -> str:
  return " / ".join(_number(value, decimals) for value in values)


def _change(new: object, reference: object) -> str:
  """Signed relative change in percent, computed before any display rounding."""
  change = 100.0 * (float(new) - float(reference)) / float(reference)
  return f"{round(change, 1) + 0.0:+.1f}"  # + 0.0 renders -0.0 as +0.0


def _slug(value: str) -> str:
  return re.sub(r"[^a-z0-9]+", "_", _dataset(value).lower()).strip("_")


def _render_table(
  columns: str,
  header: list[str],
  groups: list[tuple[str, list[list[str]]]],
  caption: str,
  label: str,
) -> str:
  """Render one table; each group is (direction cell, rows of remaining cells)."""
  lines = [
    r"\begin{table}[H]", r"\centering", r"\resizebox{\textwidth}{!}{%",
    rf"\begin{{tabular}}{{{columns}}}", r"\toprule", *header, r"\midrule",
  ]
  for index, (direction, rows) in enumerate(groups):
    if index:
      lines.append(r"\midrule")
    for row_index, cells in enumerate(rows):
      first = rf"\multirow{{{len(rows)}}}{{*}}{{{direction}}}" if row_index == 0 else ""
      lines.append(" & ".join([first, *cells]) + r" \\")
  lines += [
    r"\bottomrule", r"\end{tabular}", "}",
    rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}",
  ]
  return "\n".join(lines) + "\n"


def _stage_rows(root: str | Path, stage: str) -> dict[tuple[str, str], dict[str, dict]]:
  """Return {(source, target): {method: aggregate-mean row}} for one refinement stage."""
  rows = {}
  for direction in _root_summary_directions(Path(root), "real", stage, None):
    key = (direction["source_dataset"], direction["target_dataset"])
    if key in rows:
      raise ValueError(f"Duplicate direction {key[0]} -> {key[1]} under: {root}")
    rows[key] = {row["method"]: row for row in direction["rows"]}
  if not rows:
    raise ValueError(f"No {stage} aggregate rows found under: {root}")
  return rows


def _matched(first: dict, second: dict, names: tuple[str, str]) -> list[tuple[str, str]]:
  """Require two stage-row maps to cover the same directions and methods."""
  if set(first) != set(second):
    raise ValueError(
      f"Directions differ between {names[0]} ({sorted(first)}) and "
      f"{names[1]} ({sorted(second)})."
    )
  for direction in first:
    if set(first[direction]) != set(second[direction]):
      raise ValueError(
        f"Methods differ for {direction[0]} -> {direction[1]}: {names[0]} has "
        f"{sorted(first[direction])}, {names[1]} has {sorted(second[direction])}."
      )
  return sorted(first, key=_direction_key)


def anchor_table(
  random_root: str | Path,
  quality_root: str | Path,
  *,
  metric: str = "micro",
  decimals: int = 2,
) -> str:
  """Random vs quality anchors: projected-source and native-target error after joint refinement."""
  random_rows = _stage_rows(random_root, "projector_linear")
  quality_rows = _stage_rows(quality_root, "projector_linear")
  columns = (f"srctest_mae_{metric}_after", f"newtest_mae_{metric}_after")
  groups = []
  for direction in _matched(random_rows, quality_rows, ("random root", "quality root")):
    rows = []
    for method in sorted(random_rows[direction], key=_method_sort_key):
      random, quality = random_rows[direction][method], quality_rows[direction][method]
      rows.append([
        _method_name(method),
        *(_number(row[column], decimals)
          for column in columns for row in (random, quality)),
      ])
    groups.append((_direction_cell(*direction), rows))
  name = "MAE" if metric == "micro" else "Macro-MAE"
  header = [
    rf"&&\multicolumn{{2}}{{c}}{{\textbf{{Projected-source {name}}}$\downarrow$}}"
    rf"&\multicolumn{{2}}{{c}}{{\textbf{{Native-target {name}}}$\downarrow$}}\\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"\textbf{Direction} & \textbf{Projection} & \textbf{Random} & \textbf{Quality} "
    r"& \textbf{Random} & \textbf{Quality} \\",
  ]
  caption = (
    f"{name} comparison between class-balanced random and class-balanced quality-based "
    "anchor selection after task-aware refinement. Projected-source error measures "
    "preservation of the source task, whereas native-target error measures retention on "
    "native target representations. Lower values are better."
  )
  label = "tab:anchor_selection_mae" if metric == "micro" else "tab:anchor_selection_macro_mae"
  return _render_table("llcccc", header, groups, caption, label)


def refinement_table(root: str | Path, *, decimals: int = 2) -> str:
  """Projector-only fitting vs joint projector--regressor refinement (MAE / Macro-MAE)."""
  before = _stage_rows(root, "projector_only")
  joint = _stage_rows(root, "projector_linear")
  groups = []
  for direction in _matched(before, joint, ("projector-only stage", "joint stage")):
    rows = []
    for method in sorted(joint[direction], key=_method_sort_key):
      rows.append([_method_name(method), *(
        _pair((stage_row[f"{prefix}_mae_micro_{when}"], stage_row[f"{prefix}_mae_macro_{when}"]),
              decimals)
        for prefix in ("srctest", "newtest")
        for stage_row, when in ((before[direction][method], "before"),
                                (joint[direction][method], "after"))
      )])
    groups.append((_direction_cell(*direction), rows))
  stages = (
    r"& \makecell{\textbf{Projector-only} \\ \textbf{(MAE / Macro-MAE)}} "
    r"& \makecell{\textbf{Joint refinement} \\ \textbf{(MAE / Macro-MAE)}} "
  )
  header = [
    r"&&\multicolumn{2}{c}{\textbf{Projected-source}}"
    r"&\multicolumn{2}{c}{\textbf{Native-target}}\\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"\textbf{Direction} & \textbf{Map method} " + stages + stages + r"\\",
  ]
  caption = (
    "Comparison between projector-only optimization and joint projector--regressor "
    "refinement. Results are reported as MAE / Macro-MAE. Projected-source results measure "
    "performance after mapping source representations into the target space and evaluating "
    "them with the target linear regression head, whereas native-target results measure "
    "performance on the native target representations. Lower values indicate better "
    "performance."
  )
  return _render_table(
    "llcccc", header, groups, caption,
    "tab:comparison_proj_vs_projANDregressor_stage_mae_macro")


CLOSED_FORM_METHODS = ("procrustes", "linear_close")


def closed_form_table(root: str | Path, *, decimals: int = 3) -> str:
  """Anchor-only and refined errors for the closed-form mappings (Procrustes, OLS)."""
  before = _stage_rows(root, "projector_only")
  joint = _stage_rows(root, "projector_linear")
  groups = []
  for direction in _matched(before, joint, ("projector-only stage", "joint stage")):
    missing = [method for method in CLOSED_FORM_METHODS if method not in joint[direction]]
    if missing:
      raise ValueError(
        f"Missing closed-form methods for {direction[0]} -> {direction[1]}: {missing}")
    rows = []
    for method in CLOSED_FORM_METHODS:
      anchor, refined = before[direction][method], joint[direction][method]
      rows.append([
        _method_name(method),
        _pair((anchor["srctest_mae_micro_before"], anchor["srctest_mae_macro_before"]), decimals),
        _pair((refined["srctest_mae_micro_after"], refined["srctest_mae_macro_after"]), decimals),
        _pair((refined["newtest_mae_micro_after"], refined["newtest_mae_macro_after"]), decimals),
      ])
    groups.append((_direction_cell(*direction), rows))
  header = [
    r"\textbf{Direction} & \textbf{Mapping} & \textbf{Anchor-only source} "
    r"& \textbf{Refined source} & \textbf{Refined target} \\",
  ]
  caption = (
    "Closed-form mappings: MAE / Macro-MAE of the anchor-only fit and after joint "
    "refinement, averaged over the source--target checkpoint pairs."
  )
  return _render_table("llccc", header, groups, caption, "tab:closed_form")


def _synthetic_rows(root: str | Path) -> pd.DataFrame:
  path = Path(root) / "aggregated_summary_fake.csv"
  if not path.is_file():
    raise ValueError(f"Missing synthetic-control summary: {path}")
  frame = pd.read_csv(path)
  required = {
    "summary_row", "refinement_mode", "status", "distribution", "old_model_pth",
    "new_model_pth", "interpolation_similarity", "experiment", "success_count",
    "real_mae_micro", "fake_mae_micro", "real_mae_macro", "fake_mae_macro",
  }
  missing = sorted(required - set(frame.columns))
  if missing:
    raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")
  frame = frame.loc[
    frame["summary_row"].astype(str).isin(["MEAN", "RESULT"])
    & frame["refinement_mode"].astype(str).eq("projector_linear")
  ]
  if frame.empty:
    raise ValueError(f"No projector_linear MEAN rows in {path}")
  failed = frame.loc[~frame["status"].astype(str).eq("success"), "experiment"]
  if not failed.empty:
    raise ValueError(f"Non-successful synthetic-control rows in {path}: {', '.join(failed)}")
  distributions = set(frame["distribution"].astype(str))
  if distributions != {"standard_normal"}:
    raise ValueError(f"Expected standard_normal synthetic rows in {path}, got {sorted(distributions)}")
  frame = frame.copy()
  frame["_source"] = [_model_metadata(value)[0] for value in frame["old_model_pth"]]
  frame["_target"] = [_model_metadata(value)[0] for value in frame["new_model_pth"]]
  return frame


def synthetic_table(roots: list[str | Path], *, decimals: int = 2) -> str:
  """Real vs standard-normal synthetic source embeddings through the saved joint pipeline."""
  frame = pd.concat([_synthetic_rows(root) for root in roots], ignore_index=True)
  duplicates = frame.duplicated(["_source", "_target", "interpolation_similarity"])
  if duplicates.any():
    rows = frame.loc[duplicates, ["_source", "_target", "interpolation_similarity"]]
    raise ValueError(f"Duplicate synthetic-control rows: {rows.to_dict('records')}")
  counts = {
    f"{_dataset(row['_source'])}->{_dataset(row['_target'])} "
    f"{row['interpolation_similarity']}": int(row["success_count"])
    for row in frame.to_dict("records")
  }
  if len(set(counts.values())) > 1:
    print(f"warning: uneven checkpoint-pair counts: {counts}", file=sys.stderr)
  groups = []
  for direction, rows in sorted(
    frame.groupby(["_source", "_target"]), key=lambda item: _direction_key(item[0])
  ):
    source, target = direction
    cell = (
      rf"\shortstack{{{_direction_cell(source, target)} \\ "
      rf"\footnotesize evaluated on {_dataset(source)}}}"
    )
    cells = []
    for row in sorted(
      rows.to_dict("records"), key=lambda row: _method_sort_key(row["interpolation_similarity"])
    ):
      cells.append([
        _method_name(row["interpolation_similarity"]),
        *(_number(row[column], decimals) for column in (
          "real_mae_micro", "fake_mae_micro", "real_mae_macro", "fake_mae_macro")),
      ])
    groups.append((cell, cells))
  header = [
    r"\textbf{Direction} & \textbf{Mapping method} & \multicolumn{2}{c}{\textbf{MAE}} "
    r"& \multicolumn{2}{c}{\textbf{Macro-MAE}} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"& & \textbf{Real} & \textbf{Synthetic} & \textbf{Real} & \textbf{Synthetic} \\",
  ]
  caption = (
    "Comparison between real and standard-normal synthetic source representations. "
    "Projectors and linear regression heads are not retrained for the synthetic-input "
    "experiment."
  )
  return _render_table("clcccc", header, groups, caption, "tab:real_fake_synthetic_control")


def frozen_tables(
  learned_root: str | Path,
  frozen_root: str | Path,
  *,
  decimals: int = 3,
) -> str:
  """Learned projector + joint refinement vs frozen random projector + linear-head refinement."""
  learned = _stage_rows(learned_root, "projector_linear")
  frozen = _stage_rows(frozen_root, "random_projector_linear")
  directions = _matched(learned, frozen, ("learned root", "frozen root"))
  header = [
    r"\textbf{Direction} & \textbf{Projection} "
    r"& \makecell{\textbf{Learned projector + linear}\\ \textbf{(MAE / Macro-MAE)}} "
    r"& \makecell{\textbf{Frozen random + linear}\\ \textbf{(MAE / Macro-MAE)}} "
    r"& \makecell{\textbf{Relative change}\\ \textbf{(\%)}} \\",
  ]
  tables = []
  for side, prefix, description in (
    ("source", "srctest", "Projected-source"),
    ("target", "newtest", "Native-target"),
  ):
    columns = (f"{prefix}_mae_micro_after", f"{prefix}_mae_macro_after")
    groups = []
    for direction in directions:
      rows = []
      for method in sorted(learned[direction], key=_method_sort_key):
        fitted, random = learned[direction][method], frozen[direction][method]
        rows.append([
          _method_name(method),
          _pair(tuple(fitted[column] for column in columns), decimals),
          _pair(tuple(random[column] for column in columns), decimals),
          " / ".join(_change(random[column], fitted[column]) for column in columns),
        ])
      groups.append((_direction_cell(*direction), rows))
    caption = (
      f"{description} performance when jointly refining an anchor-trained projector and "
      "the target linear regression head, compared with refining only the linear head "
      "while keeping an identically structured randomly initialized projector frozen. "
      "Results are reported as MAE / Macro-MAE. Relative change is measured with respect "
      "to the learned-projector condition; positive values indicate worse performance "
      "for the frozen-random control."
    )
    tables.append(_render_table(
      "llccc", header, groups, caption, f"tab:projector_linear_vs_random_{side}"))
  return "\n".join(tables)


def random_adapter_table(
  roots: list[str | Path],
  *,
  fast_mode: bool = False,
  decimals: int = 2,
) -> str:
  """Fitted vs random-initialized projector under joint refinement (MAE / Macro-MAE)."""
  frame = pd.concat(
    [adapter_tables.summarize_root(root, fast_mode=fast_mode) for root in roots],
    ignore_index=True,
  )
  frame = frame.loc[frame["refinement_mode"].eq("joint")]
  if frame.empty:
    raise ValueError("No joint-refinement random-adapter rows found.")
  seed_counts = set(frame["seed_count"].astype(int))
  if len(seed_counts) != 1:
    raise ValueError(f"Inconsistent random-adapter seed counts: {sorted(seed_counts)}")
  cells = {}
  for row in frame.to_dict("records"):
    source, target = (
      ADAPTER_DIRECTIONS.get(name, name) for name in row["direction"].split(" -> ", 1))
    method = ADAPTER_METHODS.get(row["projection"], row["projection"])
    metrics = cells.setdefault((source, target), {}).setdefault(method, {})
    if row["metric"] in metrics:
      raise ValueError(f"Duplicate random-adapter row: {row['direction']} {method} {row['metric']}")
    metrics[row["metric"]] = row
  groups = []
  for direction in sorted(cells, key=_direction_key):
    rows = []
    for method in sorted(cells[direction], key=_method_sort_key):
      metrics = cells[direction][method]
      if set(metrics) != {"micro_mae", "macro_mae"}:
        raise ValueError(
          f"Missing MAE or Macro-MAE for {direction[0]} -> {direction[1]} {method}.")
      ordered = (metrics["micro_mae"], metrics["macro_mae"])
      rows.append([
        _method_name(method),
        _pair(tuple(row["fitted_mae"] for row in ordered), decimals),
        _pair(tuple(row["random_mae_mean"] for row in ordered), decimals),
        " / ".join(_change(row["random_mae_mean"], row["fitted_mae"]) for row in ordered),
      ])
    groups.append((_direction_cell(*direction), rows))
  header = [
    r"\textbf{Direction} & \textbf{Projection} "
    r"& \shortstack{\textbf{Original} \\ \textbf{(MAE / Macro-MAE) $\downarrow$}} "
    r"& \shortstack{\textbf{Random} \\ \textbf{(MAE / Macro-MAE) $\downarrow$}} "
    r"& \shortstack{\textbf{Relative change} \\ \textbf{(\%)}} \\",
  ]
  caption = (
    "Random-mapping ablation under joint refinement. Errors and relative percentage "
    "changes are reported in MAE / Macro-MAE order. Random-projector results are "
    f"averaged over {seed_counts.pop()} initialization seeds."
  )
  return _render_table(
    "llccc", header, groups, caption, "tab:random_adapter_ablation_joint_mae_macro_mae")


def _row_percentages(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
  """Row-normalized confusion matrix (percent) of rounded, clipped predictions."""
  true = np.clip(np.round(labels), 0, num_classes - 1).astype(np.int64)
  predicted = np.clip(np.round(predictions), 0, num_classes - 1).astype(np.int64)
  counts = np.zeros((num_classes, num_classes))
  np.add.at(counts, (true, predicted), 1)
  totals = counts.sum(axis=1, keepdims=True)
  return np.divide(100.0 * counts, totals, out=np.zeros_like(counts), where=totals > 0)


def _confusion_tabular(title: str, percentages: np.ndarray, width: str) -> list[str]:
  classes = range(len(percentages))
  lines = [
    rf"\begin{{minipage}}[t]{{{width}\textwidth}}", r"\centering",
    rf"\textbf{{{title}}}\par\smallskip",
    rf"\begin{{tabular}}{{r@{{\quad}}{'r' * len(percentages)}}}", r"\toprule",
    rf"& \multicolumn{{{len(percentages)}}}{{c}}{{Predicted level}} \\",
    "True & " + " & ".join(map(str, classes)) + r" \\", r"\midrule",
  ]
  lines += [
    f"{level} & " + " & ".join(f"{value:.1f}" for value in row) + r" \\"
    for level, row in zip(classes, percentages)
  ]
  return lines + [r"\bottomrule", r"\end{tabular}", r"\end{minipage}"]


CONFUSION_STAGES = {
  # stage: (replay prediction key, middle-matrix title suffix, caption phrase, label suffix)
  "joint": ("trained_adapter_predictions", "", "after joint refinement", ""),
  "projector_only": (
    "real_before_predictions", " (projector-only)",
    "after projector-only fitting, before refinement", "_projector_only"),
}


def confusion_figure(
  root: str | Path,
  *,
  method: str = "linear",
  seed: int = 42,
  fast_mode: bool = False,
  num_classes: int | None = None,
  stage: str = "joint",
) -> str:
  """Native-source, fitted and random-adapter confusion matrices for one method."""
  fitted_key, title_suffix, stage_phrase, label_suffix = CONFUSION_STAGES[stage]
  folder = "fake_adapter_random_init_fast" if fast_mode else "fake_adapter_random_init"
  base = Path(root) / folder / f"seed_{seed}"
  if not base.is_dir():
    raise ValueError(f"Missing random-adapter replay folder: {base}")
  pkls = sorted(
    path for path in base.rglob("results*.pkl")
    if not any(part.startswith("aggregated") for part in path.relative_to(base).parts)
  )
  labels, native, fitted, random = [], [], [], []
  directions = set()
  for path in pkls:
    with path.open("rb") as handle:
      data = pickle.load(handle)
    config = data.get("config_cross_space_projection") or {}
    if config.get("interpolation_similarity") != method:
      continue
    metadata = data.get("fake_projection_metadata") or {}
    if metadata.get("control") != "fake_adapter" or metadata.get("seed") != seed:
      raise ValueError(f"Not a seed-{seed} fake-adapter replay: {path}")
    evaluation = (data.get("fake_projection_evaluations") or {}).get("projector_linear")
    if evaluation is None:
      raise ValueError(f"Missing projector_linear replay in: {path}")
    old = data["old_model_tensors"]
    native_by_id = dict(zip(
      map(str, old["sample_ids"]), np.asarray(old["predictions"], dtype=float).reshape(-1)))
    sample_ids = [str(value) for value in evaluation["sample_ids"]]
    missing = [value for value in sample_ids if value not in native_by_id]
    if missing:
      raise ValueError(f"Native predictions missing for {len(missing)} samples in: {path}")
    labels.append(np.asarray(evaluation["labels"], dtype=float).reshape(-1))
    native.append(np.array([native_by_id[value] for value in sample_ids]))
    fitted.append(np.asarray(evaluation[fitted_key], dtype=float).reshape(-1))
    random.append(np.asarray(evaluation["random_adapter_predictions"], dtype=float).reshape(-1))
    directions.add((_model_metadata(config["old_model_pth"]), _model_metadata(config["new_model_pth"])))
  if not labels:
    raise ValueError(f"No {method} replay PKLs for seed {seed} under: {base}")
  if len(directions) != 1:
    raise ValueError(f"Replay PKLs under {base} mix directions: {sorted(directions)}")
  (source, old_model), (target, _) = directions.pop()
  labels = np.concatenate(labels)
  classes = num_classes or int(np.round(labels.max())) + 1
  matrices = [
    _row_percentages(labels, np.concatenate(predictions), classes)
    for predictions in (native, fitted, random)
  ]
  lines = [r"\begin{figure}[H]", r"\centering", r"\small", ""]
  lines += _confusion_tabular(f"Native source model ({old_model})", matrices[0], "0.60")
  lines += ["", r"\vspace{0.8em}", ""]
  lines += _confusion_tabular(
    f"Embeddings projected on target model{title_suffix}", matrices[1], "0.48")
  lines.append(r"\hfill")
  lines += _confusion_tabular(f"Random adapter projection (seed {seed})", matrices[2], "0.48")
  direction = rf"{_dataset(source)}$\rightarrow${_dataset(target)}"
  lines += [
    "",
    rf"\caption{{Row-normalized confusion matrices for the {_dataset(source)} source task "
    rf"and the {_method_name(method)} {direction} transfer ({stage_phrase}), pooled over "
    rf"{len(labels)} predictions from {len(native)} source--target checkpoint pairs. "
    r"Entries are percentages within each true pain level; continuous predictions are "
    rf"rounded and clipped to levels $0$--${classes - 1}$ only for visualization.}}",
    rf"\label{{fig:{_slug(source)}_{_slug(target)}_random_confusion{label_suffix}}}",
    r"\end{figure}",
  ]
  return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--output", type=Path, help="Write LaTeX here instead of stdout.")
  commands = parser.add_subparsers(dest="command", required=True)

  anchor = commands.add_parser("anchor", help="Random vs quality anchor selection.")
  anchor.add_argument("random_root", type=Path)
  anchor.add_argument("quality_root", type=Path)
  anchor.add_argument("--metric", choices=("micro", "macro"), default="micro")
  anchor.add_argument("--decimals", type=int, default=2)
  anchor.set_defaults(render=lambda args: anchor_table(
    args.random_root, args.quality_root, metric=args.metric, decimals=args.decimals))

  synthetic = commands.add_parser("synthetic", help="Real vs synthetic source embeddings.")
  synthetic.add_argument("roots", nargs="+", type=Path)
  synthetic.add_argument("--decimals", type=int, default=2)
  synthetic.set_defaults(render=lambda args: synthetic_table(args.roots, decimals=args.decimals))

  refinement = commands.add_parser("refinement", help="Projector-only vs joint refinement.")
  refinement.add_argument("root", type=Path)
  refinement.add_argument("--decimals", type=int, default=2)
  refinement.set_defaults(render=lambda args: refinement_table(args.root, decimals=args.decimals))

  closed_form = commands.add_parser("closed-form", help="Procrustes and OLS mappings.")
  closed_form.add_argument("root", type=Path)
  closed_form.add_argument("--decimals", type=int, default=3)
  closed_form.set_defaults(render=lambda args: closed_form_table(args.root, decimals=args.decimals))

  frozen = commands.add_parser("frozen", help="Learned vs frozen random projector.")
  frozen.add_argument("learned_root", type=Path)
  frozen.add_argument("frozen_root", type=Path)
  frozen.add_argument("--decimals", type=int, default=3)
  frozen.set_defaults(render=lambda args: frozen_tables(
    args.learned_root, args.frozen_root, decimals=args.decimals))

  adapter = commands.add_parser("random-adapter", help="Fitted vs random projector (joint).")
  adapter.add_argument("roots", nargs="+", type=Path)
  adapter.add_argument("--fast-mode", action="store_true")
  adapter.add_argument("--decimals", type=int, default=2)
  adapter.set_defaults(render=lambda args: random_adapter_table(
    args.roots, fast_mode=args.fast_mode, decimals=args.decimals))

  confusion = commands.add_parser("confusion", help="Row-normalized confusion matrices.")
  confusion.add_argument("root", type=Path)
  confusion.add_argument("--method", default="linear")
  confusion.add_argument("--seed", type=int, default=42)
  confusion.add_argument("--fast-mode", action="store_true")
  confusion.add_argument("--num-classes", type=int)
  confusion.add_argument("--stage", choices=tuple(CONFUSION_STAGES), default="joint",
                         help="Fitted predictions for the middle matrix.")
  confusion.set_defaults(render=lambda args: confusion_figure(
    args.root, method=args.method, seed=args.seed, fast_mode=args.fast_mode,
    num_classes=args.num_classes, stage=args.stage))
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
  args = parse_args(argv)
  try:
    latex = args.render(args)
  except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
  if args.output is None:
    sys.stdout.write(latex)
    return
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(latex, encoding="utf-8")
  print(f"Saved LaTeX to: {args.output}")


if __name__ == "__main__":
  main()

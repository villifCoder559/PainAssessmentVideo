#!/usr/bin/env python3
"""Compare the reproduced paper tables with the values printed in the manuscript.

Both sides are parsed from LaTeX: the paper tables from the manuscript .tex files and
the reproduced tables from z_latex_tables/paper_reproduction/ (written by
cross_space_paper_tables.py). A cell is reproduced when |reproduced - paper| is at most
0.01 for MAE / Macro-MAE values and 0.5 percentage points for relative changes and
confusion-matrix percentages; both values are compared at their printed precision.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PAPER_DIR = REPO_ROOT / "Cross_Space_Latent_Representation_Transfer_in_Pain_Score_Regression"
SUBMITTED = PAPER_DIR / "neurips_2026_wks_paper_submitted.tex"
REVISED = PAPER_DIR / "neurips_2026_wks_paper_9pages.tex"
REPRODUCED = REPO_ROOT / "z_latex_tables" / "paper_reproduction"

TOLERANCE = {"mae": 0.01, "pct": 0.5}
DATASETS = {"biovid": "BioVid", "mintpain": "MIntPAIN", "unbc": "UNBC"}
METHOD_ALIASES = {"OLS": "Linear (closed form)"}
DIRECTION = re.compile(r"([A-Za-z]+)\s*\$\\rightarrow\$\s*([A-Za-z]+)")


def _columns(*names: str, pct: tuple[str, ...] = ()) -> tuple[tuple[str, str], ...]:
  return tuple((name, "pct" if name in pct else "mae") for name in names)


MAE_MACRO = ("MAE", "Macro")
STAGE_PAIR = tuple(f"{stage} {metric}" for stage in ("Projector-only", "Joint") for metric in MAE_MACRO)
FROZEN_COLUMNS = _columns(
  "Learned MAE", "Learned Macro", "Frozen MAE", "Frozen Macro", "Δ% MAE", "Δ% Macro",
  pct=("Δ% MAE", "Δ% Macro"))


@dataclass(frozen=True)
class TableSpec:
  name: str
  paper: Path
  paper_labels: tuple[str, ...]
  reproduced: str
  reproduced_label: str
  columns: tuple[tuple[str, str], ...]


TABLES = (
  TableSpec(
    "Anchor selection, MAE (main results)", SUBMITTED, ("tab:quality_anchor_micro_mae",),
    "anchor_selection_mae.tex", "tab:anchor_selection_mae",
    _columns("Source Random", "Source Quality", "Target Random", "Target Quality")),
  TableSpec(
    "Anchor selection, Macro-MAE", SUBMITTED, ("tab:quality_anchor_macro_mae",),
    "anchor_selection_macro_mae.tex", "tab:anchor_selection_macro_mae",
    _columns("Source Random", "Source Quality", "Target Random", "Target Quality")),
  TableSpec(
    "Synthetic-embedding control", SUBMITTED,
    ("tab:real_fake_unbc_biovid", "tab:real_fake_mintpain_biovid", "tab:real_fake_mintpain_unbc"),
    "synthetic_embedding_control.tex", "tab:real_fake_synthetic_control",
    _columns("MAE Real", "MAE Synthetic", "Macro Real", "Macro Synthetic")),
  TableSpec(
    "Effect of task-aware refinement", SUBMITTED,
    ("tab:comparison_proj_vs_projANDregressor_stage_mae_macro",),
    "refinement_effect.tex", "tab:comparison_proj_vs_projANDregressor_stage_mae_macro",
    _columns(*(f"Source {name}" for name in STAGE_PAIR), *(f"Target {name}" for name in STAGE_PAIR))),
  TableSpec(
    "Random-mapping ablation", SUBMITTED, ("tab:random_adapter_ablation_joint_mae_macro_mae",),
    "random_adapter_ablation_joint.tex", "tab:random_adapter_ablation_joint_mae_macro_mae",
    _columns("Original MAE", "Original Macro", "Random MAE", "Random Macro", "Δ% MAE",
             "Δ% Macro", pct=("Δ% MAE", "Δ% Macro"))),
  TableSpec(
    "Frozen random projector, projected source", SUBMITTED,
    ("tab:projector_linear_vs_random_source",),
    "projector_linear_vs_random.tex", "tab:projector_linear_vs_random_source", FROZEN_COLUMNS),
  TableSpec(
    "Frozen random projector, native target", SUBMITTED,
    ("tab:projector_linear_vs_random_target",),
    "projector_linear_vs_random.tex", "tab:projector_linear_vs_random_target", FROZEN_COLUMNS),
  TableSpec(
    "Closed-form mappings (revised manuscript)", REVISED, ("tab:closed_form",),
    "closed_form.tex", "tab:closed_form",
    _columns("Anchor-only MAE", "Anchor-only Macro", "Refined source MAE",
             "Refined source Macro", "Refined target MAE", "Refined target Macro")),
)
MATRIX_NAMES = ("Native source model", "Projected on target model", "Random adapter (seed 42)")
FIGURES = (
  ("Confusion matrices, middle = joint refinement", SUBMITTED,
   "fig:mintpain_biovid_random_confusion",
   "mintpain_biovid_random_confusion.tex", "fig:mintpain_biovid_random_confusion"),
  ("Confusion matrices, middle = projector-only", SUBMITTED,
   "fig:mintpain_biovid_random_confusion",
   "mintpain_biovid_random_confusion_projector_only.tex",
   "fig:mintpain_biovid_random_confusion_projector_only"),
)


@dataclass(frozen=True)
class Row:
  direction: str
  method: str
  values: tuple[str, ...]


def _strip_comments(tex: str) -> str:
  return re.sub(r"(?<!\\)%.*", "", tex)


def _environment(tex: str, label: str, environment: str) -> str:
  """Return the text of the environment that contains \\label{label}."""
  tex = _strip_comments(tex)
  at = tex.find(rf"\label{{{label}}}")
  if at < 0:
    raise ValueError(f"Label not found: {label}")
  start = tex.rfind(rf"\begin{{{environment}}}", 0, at)
  end = tex.find(rf"\end{{{environment}}}", at)
  if start < 0 or end < 0:
    raise ValueError(f"No {environment} environment around label: {label}")
  return tex[start:end]


def _split(text: str, separator: str) -> list[str]:
  """Split at separator outside braces (so \\shortstack{a \\\\ b} stays one cell)."""
  parts, depth, start, index = [], 0, 0, 0
  while index < len(text):
    character = text[index]
    if character == "{":
      depth += 1
    elif character == "}":
      depth -= 1
    elif (depth == 0 and text.startswith(separator, index)
          and (separator != "&" or text[index - 1] != "\\")):
      parts.append(text[start:index])
      index += len(separator)
      start = index
      continue
    index += 1
  parts.append(text[start:])
  return parts


def _body_rows(tabular: str) -> list[list[str]]:
  """Data rows (lists of cells) between the first \\midrule and \\bottomrule."""
  body = re.split(r"\\midrule\b", tabular, maxsplit=1)
  if len(body) != 2:
    raise ValueError("Table without \\midrule")
  body = body[1].split(r"\bottomrule")[0].replace(r"\midrule", "")
  rows = [_split(row, "&") for row in _split(body, r"\\")]
  return [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]


def _values(cell: str) -> list[str]:
  cleaned = re.sub(r"[${}]|\\%", "", cell)
  values = [value.strip() for value in cleaned.split("/")]
  for value in values:
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", value):
      raise ValueError(f"Not a numeric cell: {cell!r}")
  return values


def parse_table(tex: str, label: str) -> list[Row]:
  """Rows of a direction/method table; a missing direction cell repeats the previous one."""
  rows, direction = [], None
  for cells in _body_rows(_environment(tex, label, "table")):
    match = DIRECTION.search(cells[0])
    if match:
      direction = " → ".join(DATASETS[name.lower()] for name in match.groups())
    elif cells[0]:
      raise ValueError(f"Unrecognized direction cell in {label}: {cells[0]!r}")
    if direction is None:
      raise ValueError(f"Row before any direction in {label}: {cells}")
    method = METHOD_ALIASES.get(cells[1], cells[1])
    values = tuple(value for cell in cells[2:] for value in _values(cell))
    rows.append(Row(direction, method, values))
  return rows


def parse_confusion(tex: str, label: str) -> list[list[list[str]]]:
  """The row-normalized matrices of a figure, in order (true level rows, percent cells)."""
  figure = _environment(tex, label, "figure")
  tabulars = re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", figure, flags=re.S)
  if not tabulars:
    raise ValueError(f"No tabular in figure: {label}")
  return [[row[1:] for row in _body_rows(tabular)] for tabular in tabulars]


def _cell(table, direction, method, column, kind, paper, reproduced) -> dict:
  delta = round(float(reproduced) - float(paper), 6) + 0.0
  return {
    "table": table, "direction": direction, "method": method, "column": column,
    "kind": kind, "paper": paper, "reproduced": reproduced, "delta": delta,
    "tolerance": TOLERANCE[kind],
    "within_tolerance": abs(delta) <= TOLERANCE[kind] + 1e-9,
  }


def compare_table(
  spec: TableSpec, paper_tex: str, reproduced_tex: str,
) -> tuple[list[dict], list[Row]]:
  """Cell comparisons for every paper row, plus reproduced rows the paper does not print."""
  paper_rows = [row for label in spec.paper_labels for row in parse_table(paper_tex, label)]
  reproduced = {}
  for row in parse_table(reproduced_tex, spec.reproduced_label):
    reproduced[(row.direction, row.method)] = row
  cells, seen = [], set()
  for row in paper_rows:
    key = (row.direction, row.method)
    if key in seen:
      raise ValueError(f"{spec.name}: duplicate paper row {key}")
    seen.add(key)
    if key not in reproduced:
      raise ValueError(f"{spec.name}: no reproduced row for {key[0]} {key[1]}")
    ours = reproduced[key].values
    if not len(row.values) == len(ours) == len(spec.columns):
      raise ValueError(
        f"{spec.name}: {key} has {len(row.values)} paper and {len(ours)} reproduced "
        f"values, expected {len(spec.columns)}")
    cells += [
      _cell(spec.name, *key, column, kind, paper, value)
      for (column, kind), paper, value in zip(spec.columns, row.values, ours)
    ]
  extra = [row for key, row in reproduced.items() if key not in seen]
  return cells, extra


def compare_figure(name: str, paper_tex: str, paper_label: str,
                   reproduced_tex: str, reproduced_label: str) -> list[dict]:
  paper = parse_confusion(paper_tex, paper_label)
  ours = parse_confusion(reproduced_tex, reproduced_label)
  if len(paper) != len(MATRIX_NAMES) or len(ours) != len(MATRIX_NAMES):
    raise ValueError(f"{name}: expected {len(MATRIX_NAMES)} matrices")
  cells = []
  for matrix, paper_rows, our_rows in zip(MATRIX_NAMES, paper, ours):
    if [len(row) for row in paper_rows] != [len(row) for row in our_rows]:
      raise ValueError(f"{name}: {matrix} shapes differ")
    for level, (paper_row, our_row) in enumerate(zip(paper_rows, our_rows)):
      cells += [
        _cell(name, matrix, f"true {level}", f"predicted {predicted}", "pct", paper, value)
        for predicted, (paper, value) in enumerate(zip(paper_row, our_row))
      ]
  return cells


def _summary(cells: list[dict]) -> dict:
  outside = [cell for cell in cells if not cell["within_tolerance"]]
  return {
    "cells": len(cells),
    "within": len(cells) - len(outside),
    "outside": len(outside),
    "max_abs_delta": max((abs(cell["delta"]) for cell in cells), default=0.0),
  }


def _decimals(*values: str) -> int:
  return max(len(value.partition(".")[2]) for value in values)


def _format_cell(cell: dict) -> str:
  if cell["delta"] == 0:
    return cell["paper"]
  delta = f"{cell['delta']:+.{_decimals(cell['paper'], cell['reproduced'])}f}"
  flag = "" if cell["within_tolerance"] else " ⚠"
  return f"{cell['paper']} → {cell['reproduced']} ({delta}){flag}"


def _markdown_grid(cells: list[dict], row_keys: tuple[str, str], headers: list[str]) -> list[str]:
  columns = list(dict.fromkeys(cell["column"] for cell in cells))
  grid: dict[tuple[str, str], dict[str, str]] = {}
  for cell in cells:
    grid.setdefault((cell[row_keys[0]], cell[row_keys[1]]), {})[cell["column"]] = _format_cell(cell)
  lines = [
    "| " + " | ".join([*headers, *columns]) + " |",
    "|" + "---|" * (len(headers) + len(columns)),
  ]
  for key, values in grid.items():
    lines.append("| " + " | ".join([*key, *(values[column] for column in columns)]) + " |")
  return lines


def _summary_line(cells: list[dict]) -> str:
  summary = _summary(cells)
  return (
    f"{summary['cells']} cells compared: {summary['within']} within tolerance, "
    f"{summary['outside']} outside; max |Δ| = {summary['max_abs_delta']:g}."
  )


def build_report() -> tuple[list[dict], str]:
  """Return all compared cells and the generated Markdown comparison sections."""
  texts = {path: path.read_text(encoding="utf-8") for path in (SUBMITTED, REVISED)}
  all_cells, sections, verdicts = [], [], []
  for spec in TABLES:
    reproduced_path = REPRODUCED / spec.reproduced
    cells, extra = compare_table(
      spec, texts[spec.paper], reproduced_path.read_text(encoding="utf-8"))
    all_cells += cells
    verdicts.append((spec.name, cells))
    lines = [
      f"### {spec.name}", "",
      f"Paper: `{'`, `'.join(spec.paper_labels)}` in `{spec.paper.name}`. "
      f"Reproduced: `{reproduced_path.relative_to(REPO_ROOT)}`.", "",
      _summary_line(cells), "",
      *_markdown_grid(cells, ("direction", "method"), ["Direction", "Method"]),
    ]
    if extra:
      columns = [name for name, _ in spec.columns]
      lines += [
        "", "Reproduced rows not printed in the paper:", "",
        "| " + " | ".join(["Direction", "Method", *columns]) + " |",
        "|" + "---|" * (2 + len(columns)),
        *("| " + " | ".join([row.direction, row.method, *row.values]) + " |" for row in extra),
      ]
    sections.append("\n".join(lines))
  for name, paper, paper_label, reproduced, reproduced_label in FIGURES:
    reproduced_path = REPRODUCED / reproduced
    cells = compare_figure(
      name, texts[paper], paper_label, reproduced_path.read_text(encoding="utf-8"),
      reproduced_label)
    all_cells += cells
    verdicts.append((name, cells))
    lines = [
      f"### {name}", "",
      f"Paper: `{paper_label}` in `{paper.name}`. "
      f"Reproduced: `{reproduced_path.relative_to(REPO_ROOT)}`.", "",
      _summary_line(cells),
    ]
    for matrix in MATRIX_NAMES:
      matrix_cells = [cell for cell in cells if cell["direction"] == matrix]
      lines += [
        "", f"**{matrix}** — {_summary_line(matrix_cells)}", "",
        *_markdown_grid(matrix_cells, ("direction", "method"), ["Matrix", "Row"]),
      ]
    sections.append("\n".join(lines))
  overview = [
    "| Comparison | Cells | Within tolerance | Outside | Max \\|Δ\\| |",
    "|---|---|---|---|---|",
  ]
  for name, cells in verdicts:
    summary = _summary(cells)
    overview.append(
      f"| {name} | {summary['cells']} | {summary['within']} | {summary['outside']} | "
      f"{summary['max_abs_delta']:g} |")
  return all_cells, "\n".join(overview) + "\n\n" + "\n\n".join(sections) + "\n"


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--csv", type=Path, help="Write one row per compared cell here.")
  parser.add_argument("--markdown", type=Path, help="Write the Markdown sections here.")
  args = parser.parse_args(argv)
  try:
    cells, markdown = build_report()
  except (OSError, ValueError) as exc:
    raise SystemExit(f"error: {exc}") from None
  if args.csv:
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
      writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
      writer.writeheader()
      writer.writerows(cells)
    print(f"Saved {len(cells)} cells to: {args.csv}", file=sys.stderr)
  if args.markdown:
    args.markdown.write_text(markdown, encoding="utf-8")
    print(f"Saved Markdown to: {args.markdown}", file=sys.stderr)
  else:
    sys.stdout.write(markdown)


if __name__ == "__main__":
  main()

#!/usr/bin/env python3
"""Anchor-budget sweep: projected-source MAE and Macro-MAE vs the number of anchors K.

Paper: fig:anchor_budget_sweep (projector-only stage). Runs each anchor-sweep YAML once per
value in its num_anchors list (output Cross_projection/<run_tag>/K<k>/, skipped if done),
summarizes the sweep with plot_anchor_sweep_mae.py (paper_outputs/anchor_budget/) and draws
the paper figure, paper_outputs/anchor_budget_sweep.png.
"""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ANCHOR_SWEEP_DIRECTIONS, CROSS_ROOT, OUT, YAML_ROOT,  # noqa: E402
                     cross_space_done, load_yaml, require_filled, run, setup, skip, temp_yaml)

SWEEP_ROOT = CROSS_ROOT / "anchor_sweep_rerun_2026_09_23"
# Paper figure: one column per direction (title, log y-axis), one row per metric.
PANELS = [("biovid-to-unbc", "BioVid → UNBC", False), ("mintpain-to-unbc", "MIntPAIN → UNBC", False),
          ("unbc-to-biovid", "UNBC → BioVid (log y)", True)]
METHODS = [("linear", "Linear", "#0072B2"), ("mlp", "MLP", "#D55E00"),
           ("autoencoder", "EncDec", "#009E73"), ("procrustes", "Procrustes", "#CC79A7"),
           ("linear_close", "OLS", "#666666")]


def plot_figure(summary_csv: Path, output: Path) -> None:
  """Projector-only projected-source MAE / Macro-MAE vs K, as in the paper appendix."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

  summary = pd.read_csv(summary_csv)
  # source_before (projector only) is the same in both refinement modes; keep one copy.
  summary = summary[summary.refinement_mode == "projector_linear"]
  fig, axes = plt.subplots(2, len(PANELS), figsize=(10.5, 5.7))
  for row, (metric, ylabel) in enumerate((("micro", "MAE (projected source)"),
                                          ("macro", "Macro-MAE (projected source)"))):
    for col, (direction, title, log_y) in enumerate(PANELS):
      ax = axes[row, col]
      data = summary[(summary.direction == direction) & (summary.metric == metric)]
      anchors = sorted(data.configured_anchors.unique())
      for method, label, color in METHODS:
        curve = data[(data.method == method) & (data.stage == "source_before")]
        if curve.empty:
          continue
        curve = curve.set_index("configured_anchors").reindex(anchors)["mean"]
        ax.plot(range(len(anchors)), curve.values, marker="o", markersize=3.5, linewidth=1.3,
                color=color, label=label)
      native = data[data.stage == "source_old"]["mean"]
      if not native.empty:
        ax.axhline(native.iloc[0], color="black", linestyle="--", linewidth=1, label="Native source")
      ax.set_xticks(range(len(anchors)), [str(k) for k in anchors],
                    fontsize=8 if len(anchors) > 5 else None)
      if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.8, 1, 1.5, 2, 3, 5, 10, 20]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_locator(NullLocator())
      ax.grid(alpha=0.25)
      if row == 0:
        ax.set_title(title)
      else:
        ax.set_xlabel("Anchors K")
      if col == 0:
        ax.set_ylabel(ylabel)
  handles = {}
  for ax in axes.flat:
    for handle, label in zip(*ax.get_legend_handles_labels()):
      handles.setdefault(label, handle)
  handles["Native source"] = handles.pop("Native source")  # last, as in the paper
  fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles))
  fig.tight_layout(rect=(0, 0.06, 1, 1))
  fig.savefig(output, dpi=150)
  plt.close(fig)
  print(f"Saved {output}")


def sweep_runs(path: Path) -> list[tuple[int, dict]]:
  """(K, single-K config) for every anchor budget of one sweep YAML."""
  config = load_yaml(path)
  base_tag = config["run_tag"].rstrip("/")
  runs = []
  for k in dict.fromkeys(config["num_anchors"]):
    run_config = copy.deepcopy(config)
    run_config["num_anchors"] = [k]
    run_config["run_tag"] = f"{base_tag}/K{k}"
    runs.append((k, run_config))
  return runs


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  if args.gpu == "cpu":
    sys.exit("error: the anchor sweep needs a GPU id")
  yamls = sorted(y for name in ANCHOR_SWEEP_DIRECTIONS
                 for y in (YAML_ROOT / "config_anchor_sweep" / name).glob("*.yaml"))
  pending = [y for y in yamls
             if args.force or not all(cross_space_done(c["run_tag"]) for _, c in sweep_runs(y))]
  require_filled(pending)
  with tempfile.TemporaryDirectory() as tmp:
    for path in yamls:
      for k, config in sweep_runs(path):
        if not args.force and cross_space_done(config["run_tag"]):
          skip(f"{path} K={k}")
          continue
        run([sys.executable, "cross_space_projection.py", "--config",
             str(temp_yaml(config, tmp, f"{path.stem}_K{k}.yaml"))], gpu=args.gpu)
  summary = OUT / "anchor_budget" / "data" / "anchor_sweep_summary.csv"
  if args.force or not summary.is_file():
    run([sys.executable, "plot_anchor_sweep_mae.py", "--input-root", str(SWEEP_ROOT),
         "--output-dir", str(OUT / "anchor_budget")])
  else:
    skip(summary)
  plot_figure(summary, OUT / "anchor_budget_sweep.png")
  return 0


if __name__ == "__main__":
  sys.exit(main())

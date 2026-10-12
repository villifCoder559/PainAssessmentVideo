#!/usr/bin/env python3
"""Frozen-representation baselines of the 12 training runs, plus constant predictors.

Paper: tab:literature_comparison_all_datasets, tab:baseline_summary_all_datasets.
Recomputes each fold's test metrics from the best checkpoints (extract_kfold_test_table.py
--raw, skipped if test_table_<metric>_raw.csv exists) and collects them in
paper_outputs/baselines.csv. Constant-predictor rows go to paper_outputs/baselines_constant.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (OUT, TRAIN_CONFIGS, load_yaml, newest_complete_run, run,  # noqa: E402
                     setup, skip)

# Constant predictors reported in the paper: config whose label CSV is used -> constant.
CONSTANTS = {"biovid_dfer": 2, "unbc_vas_dfer": 2, "unbc_opi0-5_dfer": 1, "mintpain_dfer": 0}
ACCURACY_DATASETS = ("mintpain", "xite")


def metrics_of(config: str) -> tuple[str, ...]:
  return ("mae", "accuracy") if config.startswith(ACCURACY_DATASETS) else ("mae",)


def summary_row(config: str, pkl: Path) -> dict:
  row = {"config": config, "run": str(pkl.parent)}
  mae = pd.read_csv(pkl.parent / "test_table_mae_raw.csv").set_index("fold")
  row.update(mae_mean=mae.at["mean", "test_MAE_raw"], mae_std=mae.at["std", "test_MAE_raw"],
             macro_mae=mae.at["mean", "test_MAE_macro"])
  if "accuracy" in metrics_of(config):
    acc = pd.read_csv(pkl.parent / "test_table_accuracy_raw.csv").set_index("fold")
    row.update(accuracy_pct=acc.at["mean", "test_accuracy_raw_pct"], macro_f1=acc.at["mean", "macro_f1"])
  return row


def constant_row(config: str, constant: int) -> dict:
  csv = load_yaml(TRAIN_CONFIGS / f"{config}.yaml")["csv"]
  labels = pd.read_csv(csv, sep="\t")["class_id"]
  error = (labels - constant).abs()
  return {"labels": csv, "constant": constant, "mae": error.mean(),
          "macro_mae": error.groupby(labels).mean().mean(),
          "accuracy_pct": 100 * (labels == constant).mean()}


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  configs = sorted(p.stem for p in TRAIN_CONFIGS.glob("*.yaml"))
  rows, missing = [], []
  for config in configs:
    found = newest_complete_run(config)
    if found is None:
      print(f"missing: no complete training run of {config} (python reproduce/train_all.py --only {config})")
      missing.append(config)
      continue
    pkl = found[0]
    for metric in metrics_of(config):
      table = pkl.parent / f"test_table_{metric}_raw.csv"
      if not args.force and table.is_file():
        skip(table)
        continue
      run([sys.executable, "extract_kfold_test_table.py", "--pkl", str(pkl), "--raw",
           "--metric", metric], gpu=args.gpu)
    rows.append(summary_row(config, pkl))
  OUT.mkdir(exist_ok=True)
  pd.DataFrame(rows).to_csv(OUT / "baselines.csv", index=False, float_format="%.4f")
  constants = pd.DataFrame([constant_row(c, k) for c, k in CONSTANTS.items()])
  constants.to_csv(OUT / "baselines_constant.csv", index=False, float_format="%.4f")
  print(pd.DataFrame(rows).drop(columns="run").to_string(index=False, float_format="%.2f"))
  print(constants.to_string(index=False, float_format="%.2f"))
  print(f"Saved {OUT / 'baselines.csv'} and {OUT / 'baselines_constant.csv'}")
  if missing:
    print(f"Incomplete: no run for {', '.join(missing)}")
  return 1 if missing else 0


if __name__ == "__main__":
  sys.exit(main())

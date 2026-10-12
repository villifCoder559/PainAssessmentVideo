#!/usr/bin/env python3
"""Cumulative predictions over growing prefixes of BioVid video 7678 (BioVid -> UNBC, MLP).

Paper: fig:cumulative_mean, fig:cumulative_per_target. Reruns the cross-space experiment on
the video cut at increasing lengths (plot_cumulative_predictions.py, needs a GPU), then adds
the face snapshots under each plot (plot_cumulative_snapshot_figure.py). Skipped if both
figures exist in paper_outputs/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (OUT, PAPER_SETS, YAML_ROOT, require_run, run,  # noqa: E402
                     run_cross_space_yaml, setup, skip)

SAMPLE = "7678"
SNAPSHOTS = ["47", "63", "95", "143"]
DIRECTION = "biovidVmae_to_unbcDfer_cross_validation"
EXPERIMENT_YAML = YAML_ROOT / "config_paper_tests_seed_42" / DIRECTION / "refinement3_mlp.yaml"
EXPERIMENT_ROOT = PAPER_SETS["config_paper_tests_seed_42"] / DIRECTION / "refinement3_mlp"
FIGURES = {"summary_full_range": "cumulative_biovid7678_mean_curves.png",
           "models_stage_2_full_range": "cumulative_biovid7678_task_aware_per_target.png"}


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  outputs = [OUT / name for name in FIGURES.values()]
  if not args.force and all(p.is_file() for p in outputs):
    skip(", ".join(map(str, outputs)))
    return 0
  run_cross_space_yaml(EXPERIMENT_YAML, args.gpu, force=False)
  native = require_run("biovid_dfer").parent
  run([sys.executable, "plot_cumulative_predictions.py", str(EXPERIMENT_ROOT), SAMPLE,
       "--origin", "source", "--refinement-stage", "2", "--compare-native-root", str(native),
       "--video", "1", "--snapshots", *SNAPSHOTS, "--dump-curves"], gpu=args.gpu)
  plots = sorted(Path("chunk_prediction_plots").glob(f"*/{SAMPLE}/mlp/*/stage_2/summary_full_range_*.png"),
                 key=lambda p: p.stat().st_mtime)
  plots = [p for p in plots if "_video_" not in p.name]
  if not plots:
    sys.exit("error: plot_cumulative_predictions.py wrote no summary_full_range_*.png")
  digest = plots[-1].stem.rsplit("_", 1)[1]
  folder = plots[-1].parent
  OUT.mkdir(exist_ok=True)
  for prefix, name in FIGURES.items():
    run([sys.executable, "plot_cumulative_snapshot_figure.py", str(folder / f"{prefix}_{digest}.png"),
         str(folder / f"{prefix}_video_{digest}_snapshots"), str(OUT / name)])
  print(f"Curve values: {folder / f'curves_{digest}.json'}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

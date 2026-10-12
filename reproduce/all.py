#!/usr/bin/env python3
"""Reproduce every paper result: train, fill the YAMLs, run cross-space, write all tables/figures.

Each step skips work whose outputs exist, so the script can be stopped and started again.
--force reruns every cross-space experiment, control and table; finished models are
retrained only with --retrain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fill_cross_space_checkpoints  # noqa: E402
import figure_anchor_budget  # noqa: E402
import figure_temporal_7678  # noqa: E402
import run_cross_space  # noqa: E402
import summarize_cross_space  # noqa: E402
import table_anchor_selection  # noqa: E402
import table_baselines  # noqa: E402
import table_frozen_projection  # noqa: E402
import table_random_mapping  # noqa: E402
import table_refinement  # noqa: E402
import table_synthetic_control  # noqa: E402
import table_video_statistics  # noqa: E402
import train_all  # noqa: E402
from _common import setup  # noqa: E402

PAPER_STEPS = [table_baselines, table_video_statistics, table_refinement, table_anchor_selection,
               table_synthetic_control, table_random_mapping, table_frozen_projection,
               figure_anchor_budget, figure_temporal_7678]


def main(argv: list[str] | None = None) -> int:
  parser = setup(__doc__.split("\n\n")[0])
  parser.add_argument("--retrain", action="store_true", help="Also retrain finished models.")
  args = parser.parse_args(argv)
  gpu = ["--gpu", args.gpu]
  force = ["--force"] if args.force else []
  train_all.main(gpu + (["--force"] if args.retrain else []))
  if fill_cross_space_checkpoints.main([]) != 0:
    return 1
  run_cross_space.main(gpu + force)
  summarize_cross_space.main(force)
  failed = []
  for step in PAPER_STEPS:
    print(f"\n=== {step.__name__} ===", flush=True)
    if step.main(gpu + force) != 0:
      failed.append(step.__name__)
  if failed:
    print(f"\nIncomplete: {', '.join(failed)} (see the messages above).")
    return 1
  print("\nAll paper tables and figures are in paper_outputs/.")
  return 0


if __name__ == "__main__":
  sys.exit(main())

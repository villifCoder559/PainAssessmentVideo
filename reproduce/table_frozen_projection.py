#!/usr/bin/env python3
"""Learned projection vs frozen random projection with linear-head refinement.

Paper: tab:projector_linear_vs_random_source and tab:projector_linear_vs_random_target
(both in one output file). Runs the missing main and ablation cross-space experiments first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PAPER_SETS, out_file, render_table, run_cross_space_sets,  # noqa: E402
                     setup, summarize_sets)

SETS = ["config_paper_tests_seed_42", "config_ablation_frozen_random_adapter"]


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  run_cross_space_sets(SETS, args.gpu, force=False)  # shared: reproduce/run_cross_space.py --force reruns
  summarize_sets(SETS, force=False)
  render_table(["frozen", *(str(PAPER_SETS[s]) for s in SETS)],
               out_file("tab:projector_linear_vs_random"))
  return 0


if __name__ == "__main__":
  sys.exit(main())

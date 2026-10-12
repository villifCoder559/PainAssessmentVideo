#!/usr/bin/env python3
"""Projector-only vs joint refinement, incl. Procrustes and OLS.

Paper: tab:comparison_proj_vs_projANDregressor_stage_mae_macro. Also writes the
closed-form-only table tab:closed_form. Runs the missing main cross-space experiments first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PAPER_SETS, out_file, render_table, run_cross_space_sets,  # noqa: E402
                     setup, summarize_sets)

SETS = ["config_paper_tests_seed_42"]


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  run_cross_space_sets(SETS, args.gpu, force=False)  # shared: reproduce/run_cross_space.py --force reruns
  summarize_sets(SETS, force=False)
  root = str(PAPER_SETS[SETS[0]])
  render_table(["refinement", root],
               out_file("tab:comparison_proj_vs_projANDregressor_stage_mae_macro"))
  render_table(["closed-form", root], out_file("tab:closed_form"))
  return 0


if __name__ == "__main__":
  sys.exit(main())

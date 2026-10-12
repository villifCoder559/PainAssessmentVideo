#!/usr/bin/env python3
"""Random vs quality-based anchor selection (MAE and Macro-MAE).

Paper: tab:quality_anchor_micro_mae, tab:quality_anchor_macro_mae. Runs the missing
cross-space experiments of the random and quality anchor sets first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PAPER_SETS, out_file, render_table, run_cross_space_sets,  # noqa: E402
                     setup, summarize_sets)

SETS = ["config_paper_tests_seed_42", "config_paper_tests_seed_42_quality"]


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  run_cross_space_sets(SETS, args.gpu, force=False)  # shared: reproduce/run_cross_space.py --force reruns
  summarize_sets(SETS, force=False)
  roots = [str(PAPER_SETS[s]) for s in SETS]
  for metric in ("micro", "macro"):
    render_table(["anchor", *roots, "--metric", metric],
                 out_file(f"tab:quality_anchor_{metric}_mae"))
  return 0


if __name__ == "__main__":
  sys.exit(main())

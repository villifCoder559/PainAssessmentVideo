#!/usr/bin/env python3
"""Real vs standard-normal synthetic source embeddings (synthetic-embedding control).

Paper: tab:real_fake_unbc_biovid, tab:real_fake_mintpain_biovid, tab:real_fake_mintpain_unbc
(one combined output table). Runs the missing main cross-space experiments and, for each
direction without aggregated_summary_fake.csv, the synthetic-embedding control first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (out_file, render_table, run_cross_space_sets, run_fake_projection,  # noqa: E402
                     setup, summarize_sets)

SETS = ["config_paper_tests_seed_42"]


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  run_cross_space_sets(SETS, args.gpu, force=False)  # shared: reproduce/run_cross_space.py --force reruns
  summarize_sets(SETS, force=False)
  roots = run_fake_projection("fake_embeddings", "aggregated_summary_fake.csv", args.gpu,
                              args.force, extra=("--distribution", "standard_normal"))
  render_table(["synthetic", *map(str, roots)], out_file("tab:real_fake_synthetic_control"))
  return 0


if __name__ == "__main__":
  sys.exit(main())

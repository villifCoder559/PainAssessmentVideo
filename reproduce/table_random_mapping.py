#!/usr/bin/env python3
"""Learned vs random cross-space mapping under joint refinement, and its confusion matrices.

Paper: tab:random_adapter_ablation_joint_mae_macro_mae and
fig:mintpain_biovid_random_confusion. Runs the missing main cross-space experiments and,
for each direction without aggregated_summary_fake_adapter.csv, the random-mapping
ablation (seeds 42-46) first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (PAPER_SETS, out_file, render_table, run_cross_space_sets,  # noqa: E402
                     run_fake_projection, setup, summarize_sets)

SETS = ["config_paper_tests_seed_42"]


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  run_cross_space_sets(SETS, args.gpu, force=False)  # shared: reproduce/run_cross_space.py --force reruns
  summarize_sets(SETS, force=False)
  roots = run_fake_projection("fake_adapter", "aggregated_summary_fake_adapter.csv",
                              args.gpu, args.force)
  render_table(["random-adapter", *map(str, roots)],
               out_file("tab:random_adapter_ablation_joint_mae_macro_mae"))
  mint_biovid = PAPER_SETS[SETS[0]] / "mintVmae_to_biovidDfer_cross_validation"
  render_table(["confusion", str(mint_biovid), "--method", "linear", "--seed", "42"],
               out_file("fig:mintpain_biovid_random_confusion"))
  return 0


if __name__ == "__main__":
  sys.exit(main())

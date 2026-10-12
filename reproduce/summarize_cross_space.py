#!/usr/bin/env python3
"""Summarize each cross-space output root into aggregated_summary.csv (used by the tables).

Runs cross_space_logs.py --only_aggregated --skip_umap on each root, unless its
aggregated_summary.csv is newer than every result (--force rewrites it).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PAPER_SETS, setup, summarize_sets  # noqa: E402


def main(argv: list[str] | None = None) -> int:
  parser = setup(__doc__.split("\n\n")[0], gpu=False)
  parser.add_argument("--sets", nargs="+", choices=list(PAPER_SETS), default=list(PAPER_SETS))
  args = parser.parse_args(argv)
  summarize_sets(args.sets, args.force)
  return 0


if __name__ == "__main__":
  sys.exit(main())

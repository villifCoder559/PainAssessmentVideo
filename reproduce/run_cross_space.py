#!/usr/bin/env python3
"""Run the paper cross-space experiments (every YAML of the three paper config sets).

Each YAML runs one projector method for one source -> target direction. YAMLs whose
aggregated result already exists under Cross_projection/<run_tag>/ are skipped (--force
reruns them). The YAMLs must point to your checkpoints first
(reproduce/fill_cross_space_checkpoints.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import PAPER_SETS, run_cross_space_sets, setup  # noqa: E402


def main(argv: list[str] | None = None) -> int:
  parser = setup(__doc__.split("\n\n")[0])
  parser.add_argument("--sets", nargs="+", choices=list(PAPER_SETS), default=list(PAPER_SETS))
  args = parser.parse_args(argv)
  run_cross_space_sets(args.sets, args.gpu, args.force)
  return 0


if __name__ == "__main__":
  sys.exit(main())

#!/usr/bin/env python3
"""Train the paper models: python train_model.py --config for each configs/train/*.yaml.

Configs that already have a complete run under runs/<config>/ are skipped (--force
trains them again). --only restricts the run to some configs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TRAIN_CONFIGS, newest_complete_run, run, setup, skip  # noqa: E402


def main(argv: list[str] | None = None) -> int:
  parser = setup(__doc__.split("\n\n")[0])
  parser.add_argument("--only", nargs="+", metavar="CONFIG",
                      help="Config names (file stems in configs/train/), e.g. biovid_dfer.")
  args = parser.parse_args(argv)
  configs = sorted(p.stem for p in TRAIN_CONFIGS.glob("*.yaml"))
  unknown = set(args.only or ()) - set(configs)
  if unknown:
    parser.error(f"unknown configs: {', '.join(sorted(unknown))}")
  for config in args.only or configs:
    if not args.force and newest_complete_run(config) is not None:
      skip(f"runs/{config}")
      continue
    run([sys.executable, "train_model.py", "--config", str(TRAIN_CONFIGS / f"{config}.yaml")],
        gpu=args.gpu)
  return 0


if __name__ == "__main__":
  sys.exit(main())

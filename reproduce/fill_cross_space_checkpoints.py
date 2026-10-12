#!/usr/bin/env python3
"""Point the cross-space YAMLs at your own training runs.

For every direction folder of the paper config sets in Cross_projection_yaml/, look up the
source and target training configs (configs/train/), take the newest complete 5-fold run of
each under runs/<config>/, and rewrite old_model_pth (source) and new_model_pth (target) in
place with the selected checkpoint of each fold. Comments and all other keys are kept.
Directions whose runs are missing are reported and left unchanged.

Usage:
  python reproduce/fill_cross_space_checkpoints.py [--dry-run] [--run CONFIG=RUN_DIR ...]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CONFIG_SETS, newest_complete_run, setup  # noqa: E402

DIRECTIONS = {name: pair for directions in CONFIG_SETS.values() for name, pair in directions.items()}
CROSS_SPACE_FOLDS = 5

_PATH_BLOCK = re.compile(r"^(?P<key>old_model_pth|new_model_pth):[ \t]*\[.*?\]", re.M | re.S)


def _format_list(key: str, paths: list[str]) -> str:
  indent = " " * (len(key) + 3)
  items = f",\n{indent}".join(f'"{p}"' for p in paths)
  return f"{key}: [{items}\n{indent[:-1]}]"


def fill_yaml_text(text: str, old_paths: list[str], new_paths: list[str]) -> str:
  """Replace the old_model_pth/new_model_pth lists of a YAML file, keeping everything else."""
  values = {"old_model_pth": old_paths, "new_model_pth": new_paths}
  found = set()

  def replace(match: re.Match) -> str:
    found.add(match["key"])
    return _format_list(match["key"], values[match["key"]])

  filled = _PATH_BLOCK.sub(replace, text)
  if found != set(values):
    raise ValueError(f"missing keys: {sorted(set(values) - found)}")
  before, after = yaml.safe_load(text), yaml.safe_load(filled)
  if after.get("old_model_pth") != old_paths or after.get("new_model_pth") != new_paths \
     or set(before) != set(after):
    raise ValueError("rewritten YAML does not parse back to the expected lists")
  return filled


def main(argv: list[str] | None = None) -> int:
  parser = setup(__doc__.split("\n\n")[0], gpu=False)
  parser.add_argument("--yaml-root", type=Path, default=Path("Cross_projection_yaml"))
  parser.add_argument("--runs-root", type=Path, default=Path("runs"))
  parser.add_argument("--run", action="append", default=[], metavar="CONFIG=RUN_DIR",
                      help="Use this history_run_* folder for CONFIG instead of the newest one.")
  parser.add_argument("--dry-run", action="store_true", help="Report only, do not write.")
  args = parser.parse_args(argv)

  overrides = {}
  for item in args.run:
    config, sep, run_dir = item.partition("=")
    if not sep or config not in {c for pair in DIRECTIONS.values() for c in pair}:
      raise SystemExit(f"error: bad --run {item!r} (expected CONFIG=RUN_DIR with a configs/train name)")
    overrides[config] = Path(run_dir)

  cache: dict[str, list[str] | None] = {}

  def checkpoints(config: str) -> list[str] | None:
    if config not in cache:
      found = newest_complete_run(config, args.runs_root, overrides.get(config),
                                  folds=CROSS_SPACE_FOLDS)
      cache[config] = None if found is None else found[1]
    return cache[config]

  errors = 0
  jobs = [(config_set, name, pair) for config_set, directions in CONFIG_SETS.items()
          for name, pair in directions.items()]
  for config_set, name, (source, target) in jobs:
    direction = args.yaml_root / config_set / name
    label = direction.relative_to(args.yaml_root)
    if not direction.is_dir():
      print(f"skipped  {label}: folder not found")
      continue
    missing = [c for c in (source, target) if checkpoints(c) is None]
    if missing:
      print(f"skipped  {label}: no complete run for {', '.join(missing)}")
      continue
    for yaml_path in sorted(direction.glob("*.yaml")):
      try:
        filled = fill_yaml_text(yaml_path.read_text(), checkpoints(source), checkpoints(target))
      except (ValueError, yaml.YAMLError) as exc:
        print(f"error    {yaml_path}: {exc}", file=sys.stderr)
        errors += 1
        continue
      if not args.dry_run:
        yaml_path.write_text(filled)
    print(f"{'would fill' if args.dry_run else 'filled'}  {label} ({source} -> {target})")
  return 1 if errors else 0


if __name__ == "__main__":
  sys.exit(main())

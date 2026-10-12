#!/usr/bin/env python3
"""Point the cross-space YAMLs at your own training runs.

For every direction folder under Cross_projection_yaml/, look up the source and target
training configs (configs/train/), take the newest complete 5-fold run of each under
runs/<config>/, and rewrite old_model_pth (source) and new_model_pth (target) in place
with the selected checkpoint of each fold. Comments and all other keys are kept.
Directions whose runs are missing are reported and left unchanged.

Usage (from the repository root):
  python fill_cross_space_checkpoints.py [--dry-run] [--run CONFIG=RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import yaml

NUM_FOLDS = 5

# Direction folder -> (source config, target config) in configs/train/.
PAPER_DIRECTIONS = {
  "unbcVmae_to_biovidDfer_cross_validation": ("unbc_opi0-4_vmae_s", "biovid_dfer"),
  "unbcVmae_to_mintDfer_cross_validation": ("unbc_opi0-4_vmae_s", "mintpain_dfer"),
  "biovidVmae_to_unbcDfer_cross_validation": ("biovid_vmae_s", "unbc_opi0-4_dfer"),
  "biovidVmae_to_mintDfer_cross_validation": ("biovid_vmae_s", "mintpain_dfer"),
  "mintVmae_to_biovidDfer_cross_validation": ("mintpain_vmae_s", "biovid_dfer"),
  "mintVmae_to_unbcDfer_cross_validation": ("mintpain_vmae_s", "unbc_opi0-4_dfer"),
}
ANCHOR_SWEEP_DIRECTIONS = {
  "config_pain_from_BiovidVmae_to_UNBCdfer-V2_cross_validation": ("biovid_vmae_s", "unbc_opi0-4_dfer"),
  "config_pain_from_MintVmae_to_unbcDfer_cross_validation": ("mintpain_vmae_s", "unbc_opi0-4_dfer"),
  "config_pain_from_unbcVmae_to_biovidDfer_cross_validation": ("unbc_opi0-4_vmae_s", "biovid_dfer"),
}
DIRECTIONS = {**PAPER_DIRECTIONS, **ANCHOR_SWEEP_DIRECTIONS}
# Config sets (folders of Cross_projection_yaml/) used in the paper.
CONFIG_SETS = {
  "config_paper_tests_seed_42": PAPER_DIRECTIONS,
  "config_paper_tests_seed_42_quality": PAPER_DIRECTIONS,
  "config_ablation_frozen_random_adapter": PAPER_DIRECTIONS,
  "config_anchor_sweep": ANCHOR_SWEEP_DIRECTIONS,
}

_PATH_BLOCK = re.compile(r"^(?P<key>old_model_pth|new_model_pth):[ \t]*\[.*?\]", re.M | re.S)


def checkpoints_of_run(pkl_path: Path) -> list[str] | None:
  """Selected checkpoint of each fold of one run, or None if the run is incomplete."""
  with open(pkl_path, "rb") as handle:
    results = pickle.load(handle).get("results", {})
  paths = []
  for k in range(NUM_FOLDS):
    final = results.get(f"k{k}_cross_val_final")
    if final is None:
      return None
    best = final["best_model"]
    fold, sub = best["fold_sub_fold_idx"]
    ckpt = (pkl_path.parent / "train_ATTENTIVE_JEPA" / f"k{fold}_cross_val"
            / f"k{fold}_cross_val_sub_{sub}" / f"best_model_ep_{best['best_model_idx']}.pt")
    if not ckpt.is_file():
      return None
    paths.append(ckpt.as_posix())
  return paths


def find_checkpoints(runs_root: Path, config: str, run_dir: Path | None = None) -> list[str] | None:
  """Checkpoints of the newest complete run of `config` (or of `run_dir` if given)."""
  pattern = "*/k_fold_results.pkl" if run_dir is not None else "history_run_*/*/k_fold_results.pkl"
  base = run_dir if run_dir is not None else runs_root / config
  pkls = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
  for pkl in pkls:
    paths = checkpoints_of_run(pkl)
    if paths is not None:
      return paths
  return None


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("--yaml-root", type=Path, default=Path("Cross_projection_yaml"))
  parser.add_argument("--runs-root", type=Path, default=Path("runs"))
  parser.add_argument("--run", action="append", default=[], metavar="CONFIG=RUN_DIR",
                      help="Use this history_run_* folder for CONFIG instead of the newest one.")
  parser.add_argument("--dry-run", action="store_true", help="Report only, do not write.")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv)
  overrides = {}
  for item in args.run:
    config, sep, run_dir = item.partition("=")
    if not sep or config not in {c for pair in DIRECTIONS.values() for c in pair}:
      raise SystemExit(f"error: bad --run {item!r} (expected CONFIG=RUN_DIR with a configs/train name)")
    overrides[config] = Path(run_dir)

  cache: dict[str, list[str] | None] = {}

  def checkpoints(config: str) -> list[str] | None:
    if config not in cache:
      cache[config] = find_checkpoints(args.runs_root, config, overrides.get(config))
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

"""Shared helpers for the reproduce/ scripts.

Every script runs from the repository root (all paths in this repo are relative to it),
runs a step only when its outputs are missing (--force reruns it) and writes paper
tables and figures to paper_outputs/.
"""

from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = Path("paper_outputs")
TRAIN_CONFIGS = Path("configs/train")
YAML_ROOT = Path("Cross_projection_yaml")
CROSS_ROOT = Path("Cross_projection")

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
# Config sets (folders of Cross_projection_yaml/) used in the paper -> their directions.
CONFIG_SETS = {
  "config_paper_tests_seed_42": PAPER_DIRECTIONS,
  "config_paper_tests_seed_42_quality": PAPER_DIRECTIONS,
  "config_ablation_frozen_random_adapter": PAPER_DIRECTIONS,
  "config_anchor_sweep": ANCHOR_SWEEP_DIRECTIONS,
}
# Config set -> output root under Cross_projection/ (the anchor sweep is handled separately).
PAPER_SETS = {
  "config_paper_tests_seed_42": CROSS_ROOT / "paper_tests_seed_42",
  "config_paper_tests_seed_42_quality": CROSS_ROOT / "paper_tests_seed_42_quality",
  "config_ablation_frozen_random_adapter": CROSS_ROOT / "ablation_paper_frozen_rnd_adapter",
}


def setup(description: str, gpu: bool = True, force: bool = True) -> argparse.ArgumentParser:
  """Change to the repository root and return a parser with the common flags."""
  os.chdir(REPO_ROOT)
  if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
  parser = argparse.ArgumentParser(description=description)
  if gpu:
    parser.add_argument("--gpu", default="0", help="GPU id, or 'cpu' (default: 0).")
  if force:
    parser.add_argument("--force", action="store_true", help="Rerun steps whose outputs exist.")
  return parser


def run(cmd: list[str], gpu: str | None = None, env: dict | None = None) -> None:
  """Print and run a command from the repository root; stop on failure."""
  full_env = {**os.environ, **(env or {})}
  if gpu is not None:
    full_env["CUDA_VISIBLE_DEVICES"] = "" if gpu == "cpu" else str(gpu)
  prefix = "" if gpu is None else f"CUDA_VISIBLE_DEVICES={full_env['CUDA_VISIBLE_DEVICES']} "
  print(f"\n$ {prefix}{shlex.join(cmd)}", flush=True)
  subprocess.run(cmd, check=True, env=full_env)


def skip(what: object) -> None:
  print(f"skip (done): {what}", flush=True)


def load_yaml(path: Path) -> dict:
  with open(path, encoding="utf-8") as handle:
    return yaml.safe_load(handle)


# ----- training runs -------------------------------------------------------------------

def num_folds(config: str) -> int:
  return int(load_yaml(REPO_ROOT / TRAIN_CONFIGS / f"{config}.yaml")["stop"][0])


def checkpoints_of_run(pkl_path: Path, folds: int) -> list[str] | None:
  """Selected checkpoint of each fold of one run, or None if the run is incomplete."""
  with open(pkl_path, "rb") as handle:
    results = pickle.load(handle).get("results", {})
  paths = []
  for k in range(folds):
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


def newest_complete_run(config: str, runs_root: Path = Path("runs"),
                        run_dir: Path | None = None, folds: int | None = None):
  """(k_fold_results.pkl, fold checkpoints) of the newest complete run of `config`.

  A run is complete when every fold of `stop[0]` has its final result and checkpoint.
  `run_dir` restricts the search to one history_run_* folder. Returns None if no run is
  complete.
  """
  folds = num_folds(config) if folds is None else folds
  base = run_dir if run_dir is not None else runs_root / config
  pattern = "*/k_fold_results.pkl" if run_dir is not None else "history_run_*/*/k_fold_results.pkl"
  for pkl in sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True):
    paths = checkpoints_of_run(pkl, folds)
    if paths is not None:
      return pkl, paths
  return None


def require_run(config: str) -> Path:
  """k_fold_results.pkl of the newest complete run of `config`, or stop with a hint."""
  found = newest_complete_run(config)
  if found is None:
    sys.exit(f"error: no complete training run of {config} under runs/{config}/; "
             f"run: python reproduce/train_all.py --only {config}")
  return found[0]


# ----- cross-space runs ----------------------------------------------------------------

def set_yamls(config_set: str) -> list[Path]:
  return sorted((YAML_ROOT / config_set).glob("*_cross_validation/*.yaml"))


def cross_space_done(run_tag: str) -> bool:
  """True if Cross_projection/<run_tag>/ holds an aggregated result pkl."""
  return any((CROSS_ROOT / run_tag).glob("aggregated_*/*.pkl"))


def require_filled(yamls: list[Path]) -> None:
  """Stop unless every checkpoint path in these YAMLs exists."""
  for path in yamls:
    config = load_yaml(path)
    for key in ("old_model_pth", "new_model_pth"):
      missing = [p for p in config[key] if not Path(p).is_file()]
      if missing:
        sys.exit(f"error: {path}: {key} points to missing checkpoints (e.g. {missing[0]}).\n"
                 "Train the models and fill the YAMLs first:\n"
                 "  python reproduce/train_all.py --gpu 0\n"
                 "  python reproduce/fill_cross_space_checkpoints.py")


def run_cross_space_yaml(path: Path, gpu: str, force: bool) -> None:
  config = load_yaml(path)
  if not force and cross_space_done(config["run_tag"]):
    skip(path)
    return
  require_filled([path])
  run([sys.executable, "cross_space_projection.py", "--config", str(path)], gpu=gpu)


def run_cross_space_sets(config_sets, gpu: str, force: bool) -> None:
  """Run every YAML of the given paper config sets that has no aggregated result yet."""
  yamls = [y for s in config_sets for y in set_yamls(s)]
  pending = yamls if force else [y for y in yamls if not cross_space_done(load_yaml(y)["run_tag"])]
  require_filled(pending)
  for path in yamls:
    run_cross_space_yaml(path, gpu, force)


def summarize_sets(config_sets, force: bool) -> None:
  """cross_space_logs.py --only_aggregated on each output root, unless its summary is current."""
  for config_set in config_sets:
    root = PAPER_SETS[config_set]
    summary = root / "aggregated_summary.csv"
    aggregates = list(root.glob("*/*/aggregated_*"))
    if not aggregates:
      sys.exit(f"error: no cross-space results under {root}; run reproduce/run_cross_space.py")
    if not force and summary.is_file() and \
       summary.stat().st_mtime >= max(a.stat().st_mtime for a in aggregates):
      skip(summary)
      continue
    run([sys.executable, "cross_space_logs.py", "--pkl_path", str(root),
         "--only_aggregated", "--skip_umap"])


def directions(root: Path) -> list[Path]:
  return sorted(p for p in root.glob("*_cross_validation") if p.is_dir())


def render_table(args: list[str], output: Path) -> None:
  """Write one cross_space_paper_tables.py table to paper_outputs/."""
  OUT.mkdir(exist_ok=True)
  run([sys.executable, "cross_space_paper_tables.py", "--output", str(output), *args])


def temp_yaml(config: dict, directory: str, name: str) -> Path:
  path = Path(directory) / name
  with open(path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
  return path



def out_file(label: str, suffix: str = ".tex") -> Path:
  """paper_outputs/<label with ':' replaced>.<suffix>, e.g. tab_closed_form.tex."""
  OUT.mkdir(exist_ok=True)
  return OUT / f"{label.replace(':', '_')}{suffix}"


def run_fake_projection(control: str, summary_name: str, gpu: str, force: bool,
                        extra: tuple[str, ...] = ()) -> list[Path]:
  """cross_space_fake_projection.py on every direction of the main set, unless already done."""
  roots = directions(PAPER_SETS["config_paper_tests_seed_42"])
  for root in roots:
    if not force and (root / summary_name).is_file():
      skip(root / summary_name)
      continue
    run([sys.executable, "cross_space_fake_projection.py", str(root), "--control", control, *extra],
        gpu=gpu)
  return roots

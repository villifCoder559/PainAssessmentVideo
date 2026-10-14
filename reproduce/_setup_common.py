"""Shared steps of the reproduce/setup_<dataset>.py scripts.

Each script turns the user's own copy of one dataset into the layout of README section 3:
label CSVs, preprocessed videos and the feature folders that training reads. The raw data
is symlinked, never copied. Every stage skips work whose outputs already exist, so an
interrupted run can simply be started again.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, load_yaml, run, setup, skip  # noqa: E402

FEATURE_CONFIGS = Path("configs/features")
REF_LANDMARKS = Path("partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl")
# extract_feature.py options shared by every paper feature folder (recorded in their config_dict.txt).
EXTRACT_ARGS = ["--emb_red", "spatial", "--backbone_type", "video", "--save_as_safetensors",
                "--stride_window", "16", "--stride_inside_window", "1", "--float_16", "--save_big_feature"]
CSV_COLUMNS = ["subject_id", "subject_name", "class_id", "class_name", "sample_id", "sample_name"]


def parser(description: str, stages: tuple[str, ...], raw_help: str):
  """Parser with the flags shared by every setup script; changes to the repository root."""
  p = setup(description, force=False)
  p.add_argument("--raw", type=Path, required=True, help=raw_help)
  p.add_argument("--root", type=Path, default=Path("."),
                 help="Data root that receives the dataset folders (default: the repository root).")
  p.add_argument("--stages", nargs="+", choices=stages, default=list(stages),
                 help=f"Stages to run, in order (default: all of {', '.join(stages)}).")
  p.add_argument("--backbones", nargs="+", choices=["DFER", "S"], default=["DFER", "S"],
                 help="Backbones whose features are extracted (default: both).")
  p.add_argument("--base_only", action="store_true",
                 help="Extract only the base feature folders, no augmented ones.")
  p.add_argument("--subjects", nargs="+", metavar="NAME",
                 help="Process only these subjects after the csv stage (quick test).")
  p.add_argument("--workers", type=int, default=4, help="Processes for video preprocessing.")
  return p


def resolve(args) -> None:
  """Make --raw and --root absolute and check that --raw exists."""
  args.raw = args.raw.expanduser().resolve()
  args.root = args.root.expanduser().resolve()
  if not args.raw.is_dir():
    sys.exit(f"error: --raw {args.raw} is not a folder")


def rel(path: Path) -> str:
  """Path relative to the repository root, as the repo scripts expect it.

  extract_feature.py infers the dataset and the augmentation from words in its paths, so
  the absolute part of a path outside the repository must not take part in that.
  """
  return os.path.relpath(path, REPO_ROOT)


def link(target: Path, link_path: Path) -> None:
  """Create link_path -> target, unless it already points there; stop on any other file."""
  target = Path(target)
  if link_path.is_symlink() or link_path.exists():
    if link_path.resolve() == target.resolve():
      return
    sys.exit(f"error: {link_path} exists and does not point to {target}; move it away first")
  link_path.parent.mkdir(parents=True, exist_ok=True)
  link_path.symlink_to(target)


def require(path: Path, hint: str) -> Path:
  if not path.exists():
    sys.exit(f"error: missing {path}\n{hint}")
  return path


def write_csv(frame: pd.DataFrame, path: Path) -> None:
  """Tab-separated label CSV, as the training code reads it."""
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, sep="\t", index=False)
  print(f"wrote {path} ({len(frame)} rows)")


def read_csv(path: Path) -> pd.DataFrame:
  return pd.read_csv(path, sep="\t", dtype={"sample_name": str, "subject_name": str})


def subset_csv(csv: Path, subjects: list[str] | None) -> Path:
  """`csv`, or a copy restricted to `subjects` (written next to it) for a quick test run."""
  if not subjects:
    return csv
  frame = read_csv(csv)
  unknown = set(subjects) - set(frame.subject_name)
  if unknown:
    sys.exit(f"error: subjects not in {csv}: {', '.join(sorted(unknown))}")
  out = csv.with_name(csv.stem + "_subset.csv")
  write_csv(frame[frame.subject_name.isin(subjects)], out)
  return out


def videos_of(csv: Path, folder: Path) -> list[Path]:
  """Expected <folder>/<subject_name>/<sample_name>.mp4 of every row of `csv`."""
  frame = read_csv(csv)
  return [folder / s / f"{n}.mp4" for s, n in zip(frame.subject_name, frame.sample_name)]


def require_videos(csv: Path, folder: Path) -> None:
  missing = [p for p in videos_of(csv, folder) if not p.is_file()]
  if missing:
    sys.exit(f"error: {len(missing)} of the videos listed in {csv} are missing in {folder}, "
             f"e.g. {missing[0]}")


def frontalize(videos: Path, csv: Path, out: Path, ref_landmarks: Path, workers: int) -> None:
  """extract_video_frontalized.py with frames padded to a multiple of 16 (README section 8)."""
  if all(p.is_file() for p in videos_of(csv, out)):
    skip(out)
    return
  require(ref_landmarks, "Build it first: python reproduce/setup_biovid.py --raw <BioVid PartA> "
          "--stages csv video landmarks")
  require_videos(csv, videos)
  run([sys.executable, "extract_video_frontalized.py", "--gv", "--vfp", rel(videos), "--csv", rel(csv),
       "--pfo", rel(out), "--prl", rel(ref_landmarks), "--interpolation_mod_chunk", "mirror_start_video",
       "16", "--workers", str(workers), "--skip_existing"])
  missing = [p for p in videos_of(csv, out) if not p.is_file()]
  if missing:
    sys.exit(f"error: frontalization failed for {len(missing)} videos (see the frontalize_errors_*.log "
             f"in {out}), e.g. {missing[0]}. Run the stage again to retry them.")


def feature_folders(config: dict, backbone: str, base_only: bool) -> list[tuple[str, list[str]]]:
  """(folder name, extract_feature.py flags) of every feature folder of one backbone."""
  base = config["base_name"]
  folders = [(base, [])]
  if base_only:
    return folders
  for name, spec in config["backbones"][backbone]["augmentations"].items():
    flags = [str(f) for f in spec["flags"]]
    folders.append((f"{base}_{name}", flags))
    folders += [(f"{base}_{name}${i}", flags) for i in range(spec["runs"] - 1)]
  return folders


def count_features(folder: Path) -> int:
  """Number of per-sample feature files the training code reads (no '$' in the file name)."""
  return sum(1 for p in folder.rglob("*.safetensors") if "$" not in p.name)


def extract_features(dataset: str, root: Path, labels: Path, backbones: list[str], base_only: bool,
                     gpu: str) -> None:
  """Every feature folder listed in configs/features/<dataset>.yaml that is not complete yet.

  Each folder is first written under <features>/<backbone>/.partial/ and renamed when its
  extraction finishes, so a folder with the final name is always complete.
  """
  config = load_yaml(FEATURE_CONFIGS / f"{dataset}.yaml")
  videos = root / config["videos"]
  require_videos(labels, videos)
  n_rows = len(read_csv(labels))
  for backbone in backbones:
    parent = root / config["features"] / config["backbones"][backbone]["folder"]
    for name, flags in feature_folders(config, backbone, base_only):
      final = parent / name
      if final.exists():
        found = count_features(final)
        if found >= n_rows:
          skip(final)
          continue
        sys.exit(f"error: {final} holds {found} of {n_rows} feature files; delete it and run again")
      partial = parent / ".partial" / name
      if partial.exists():
        shutil.rmtree(partial)
      partial.parent.mkdir(parents=True, exist_ok=True)
      run([sys.executable, "extract_feature.py", "--model_type", backbone, "--path_dataset", rel(videos),
           "--path_labels", rel(labels), "--saving_folder_path", rel(partial), *EXTRACT_ARGS, *flags],
          gpu=gpu)
      found = count_features(partial)
      if found < n_rows:
        sys.exit(f"error: {partial} holds {found} of {n_rows} feature files after extraction")
      partial.rename(final)
    if not base_only and config["backbones"][backbone]["augmentations"]:
      run([sys.executable, "check_augmentation_completeness.py", "--original_folder",
           rel(parent / config["base_name"])])
    shutil.rmtree(parent / ".partial", ignore_errors=True)


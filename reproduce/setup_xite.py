#!/usr/bin/env python3
"""Set up the X-ITE Pain challenge data from your own copy: CSVs, videos, features.

--raw holds train_data/ and test_data/ of the 2025 X-ITE Pain Challenge and the test labels
released by the organizers, Labels_test_data.csv (XITE/DATASET_STRUCTURE.md describes the
layout). Only the frontal face videos (fvf) are used.

Stages:
  csv          XITE/starting_point/samples.csv (train + labelled test rows) and the predefined
               splits XITE/starting_point/splits/: train_samples_21.csv (first 21 training
               subjects), val_samples_5.csv (last 5) and test.csv (the 4 test subjects)
  video        link XITE/video/video/<subject>/<sample>.mp4 to the fvf videos
  preprocess   frontalized videos padded to a multiple of 16 frames: XITE/video/video_frontalized,
               then XITE/check_frontalized_videos.py (needs the BioVid reference landmarks)
  features     the feature folders of configs/features/xite.yaml (GPU)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (CSV_COLUMNS, REF_LANDMARKS, REPO_ROOT, extract_features, frontalize, link,  # noqa: E402
                           parser, rel, require, resolve, run, subset_csv, write_csv)

sys.path.insert(0, str(REPO_ROOT / "XITE" / "starting_point"))
import prepare_xite  # noqa: E402

STAGES = ("csv", "video", "preprocess", "features")
TRAIN_SUBJECTS = 21  # the remaining 5 training subjects form the validation split


def test_labels(path: Path) -> dict[str, tuple[int, str]]:
  """{sample_name: (class_id, class_name)} from the organizers' Labels_test_data.csv."""
  labels = pd.read_csv(path)
  names = labels["Segment ID"].str.replace("_seg", "_", regex=False)
  classes = labels["True Label"].str.replace("_", "", regex=False)
  return {n: ({"PL1": 0, "PL2": 1}[c], c) for n, c in zip(names, classes)}


def labelled_rows(xite: Path, labels_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Training and test rows of prepare_xite.py, with the test labels filled in."""
  train, test = prepare_xite._discover_video_rows(xite)
  prepare_xite._assign_ids(train, test)
  prepare_xite._validate_expected_counts(train, test)
  labels = test_labels(labels_csv)
  missing = [row["sample_name"] for row in test if row["sample_name"] not in labels]
  if missing:
    sys.exit(f"error: {len(missing)} test samples have no label in {labels_csv}, e.g. {missing[0]}")
  for row in test:
    row["class_id"], row["class_name"] = labels[row["sample_name"]]
  return pd.DataFrame(train)[list(CSV_COLUMNS)], pd.DataFrame(test)[list(CSV_COLUMNS)]


def main(argv: list[str] | None = None) -> int:
  args = parser(__doc__.split("\n\n")[0], STAGES, "X-ITE folder with train_data/, test_data/, "
                "Labels_test_data.csv (see above).").parse_args(argv)
  resolve(args)
  xite = args.root / "XITE"
  start = xite / "starting_point"
  csv = start / "samples.csv"
  for name in ("train_data", "test_data"):
    link(require(args.raw / name, "--raw must hold train_data/ and test_data/"), xite / name)

  if "csv" in args.stages:
    labels_csv = require(args.raw / "Labels_test_data.csv", "the X-ITE test labels are required")
    prepare_xite.prepare_dataset(xite)  # train_samples.csv, test_samples.csv (unlabelled test rows)
    train, test = labelled_rows(xite, labels_csv)
    write_csv(pd.concat([train, test], ignore_index=True), csv)
    first = sorted(train.subject_name.unique(), key=prepare_xite._natural_key)[:TRAIN_SUBJECTS]
    write_csv(train[train.subject_name.isin(first)], start / "splits" / "train_samples_21.csv")
    write_csv(train[~train.subject_name.isin(first)], start / "splits" / "val_samples_5.csv")
    write_csv(test, start / "splits" / "test.csv")

  require(csv, "Run the csv stage first.")
  csv_run = subset_csv(csv, args.subjects)

  if "video" in args.stages:
    train, test = prepare_xite._discover_video_rows(xite)
    prepare_xite._organize_videos(xite, train + test)
    print(f"linked {len(train) + len(test)} videos in {xite / 'video' / 'video'}")

  if "preprocess" in args.stages:
    frontalize(xite / "video" / "video", csv_run, xite / "video" / "video_frontalized",
               args.root / REF_LANDMARKS, args.workers)
    run([sys.executable, "XITE/check_frontalized_videos.py", "--original-root", rel(xite / "video" / "video"),
         "--frontalized-root", rel(xite / "video" / "video_frontalized"), "--labels", rel(csv_run)])

  if "features" in args.stages:
    extract_features("xite", args.root, csv_run, args.backbones, args.base_only, args.gpu)
  return 0


if __name__ == "__main__":
  sys.exit(main())

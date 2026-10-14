#!/usr/bin/env python3
"""Set up BioVid Heat Pain Database Part A from your own copy: CSVs, videos, features.

--raw is the PartA folder of the official release: starting_point/samples.csv and
video/video/<subject_name>/<sample_name>.mp4 (a video/<subject_name>/ layout also works).

Stages:
  csv          partA/starting_point/samples.csv (the official file) and samples_adversarial.csv
               (same rows, subject_id counted from 0), the CSV used for training
  video        link partA/video/video to the raw videos
  landmarks    reference face geometry for frontalization (all datasets need it):
               partA/video/mean_face_landmarks_per_subject/all_subjects_mean_landmarks.pkl,
               always averaged over all 87 subjects (--subjects does not apply);
               about 23 min per subject per worker
  preprocess   frontalized videos padded to a multiple of 16 frames:
               partA/video/video_frontalized_interpolated_resolution_original
  features     the feature folders of configs/features/biovid.yaml (GPU)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (REF_LANDMARKS, extract_features, frontalize, link, parser, read_csv,  # noqa: E402
                           rel, require, require_videos, resolve, run, skip, subset_csv, write_csv)

STAGES = ("csv", "video", "landmarks", "preprocess", "features")


def main(argv: list[str] | None = None) -> int:
  args = parser(__doc__.split("\n\n")[0], STAGES, "BioVid PartA folder (see above).").parse_args(argv)
  resolve(args)
  part_a = args.root / "partA"
  csv = part_a / "starting_point" / "samples.csv"
  videos = part_a / "video" / "video"

  if "csv" in args.stages:
    raw_csv = require(args.raw / "starting_point" / "samples.csv", "--raw must be the BioVid PartA folder")
    csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_csv, csv)
    print(f"wrote {csv}")
    adversarial = read_csv(csv)
    adversarial["subject_id"] -= 1
    write_csv(adversarial, csv.with_name("samples_adversarial.csv"))

  require(csv, "Run the csv stage first.")
  csv_run = subset_csv(csv, args.subjects)

  if "video" in args.stages:
    raw_videos = args.raw / "video" / "video"
    link(raw_videos if raw_videos.is_dir() else args.raw / "video", videos)
    require_videos(csv_run, videos)

  if "landmarks" in args.stages:
    out = args.root / REF_LANDMARKS
    if out.is_file():
      skip(out)
    else:
      # The paper reference averages every subject; a subset mean would change all frontalizations.
      require_videos(csv, videos)
      run([sys.executable, "extract_landmarks.py", "--video_root", rel(videos), "--csv", rel(csv),
           "--out_folder", rel(out.parent), "--workers", str(args.workers)])

  if "preprocess" in args.stages:
    frontalize(videos, csv_run, part_a / "video" / "video_frontalized_interpolated_resolution_original",
               args.root / REF_LANDMARKS, args.workers)

  if "features" in args.stages:
    extract_features("biovid", args.root, csv_run, args.backbones, args.base_only, args.gpu)
  return 0


if __name__ == "__main__":
  sys.exit(main())

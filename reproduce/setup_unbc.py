#!/usr/bin/env python3
"""Set up UNBC-McMaster Shoulder Pain from your own copy: CSVs, videos, features.

--raw holds OPI/<subject>/<sequence>.txt and VAS/<subject>/<sequence>.txt (one pain score
per sequence) and WarpedVideos/<subject>/<sequence>/<sequence>.mp4 (or
WarpedVideos/<subject>/<sequence>.mp4).

Stages:
  csv          UNBC/starting_point/samples.csv (VAS 0-10), samples_OPI.csv (OPI 0-5),
               samples_OPI_0_to_4.csv (OPI 4 and 5 merged) and samples_adversarial.csv
               (VAS, subject_id counted from 0). Rows follow the paper's order, restored from
               the label-free UNBC/starting_point/sample_order_permutation.txt, so the
               training folds match the paper.
  video        link UNBC/video/WarpedVideos/<subject>/<sequence>.mp4 to the raw videos
  preprocess   face crop on a black background (UNBC/video/WarpedVideos_Cropped) and its copy
               padded to a multiple of 16 frames (UNBC/video/WarpedVideos_Cropped_interpolated_mirror)
  features     the feature folders of configs/features/unbc.yaml (GPU)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (REPO_ROOT, extract_features, link, parser, read_csv, require,  # noqa: E402
                           require_videos, resolve, skip, subset_csv, videos_of, write_csv)

STAGES = ("csv", "video", "preprocess", "features")
PERMUTATION = REPO_ROOT / "UNBC" / "starting_point" / "sample_order_permutation.txt"
COLUMNS = ["subject_id", "subject_name", "sample_id", "sample_name", "class_id", "class_name"]


def read_scores(folder: Path) -> dict[tuple[str, str], int]:
  """{(subject, sequence): score} from <folder>/<subject>/<sequence>.txt."""
  scores = {}
  for path in folder.glob("*/*.txt"):
    scores[(path.parent.name, path.stem)] = int(float(path.read_text().split()[0]))
  if not scores:
    sys.exit(f"error: no <subject>/<sequence>.txt label files in {folder}")
  return scores


def paper_order(sequences: list[tuple[str, str]]) -> list[tuple[str, str]]:
  """Sequences sorted by (subject, sequence), then reordered by the shipped permutation."""
  ordered = sorted(sequences)
  permutation = [int(line) for line in PERMUTATION.read_text().split()]
  if sorted(permutation) != list(range(len(ordered))):
    sys.exit(f"error: found {len(ordered)} UNBC sequences, but {PERMUTATION.name} orders "
             f"{len(permutation)}; --raw must hold the complete release")
  return [ordered[i] for i in permutation]


def build_csvs(order: list[tuple[str, str]], vas: dict, opi: dict) -> dict[str, pd.DataFrame]:
  """The four UNBC CSVs, all with the same rows and sample_id = row number."""
  def frame(subject_ids, labels, names):
    rows = [(sid, subject, k, sequence, label, name)
            for k, ((subject, sequence), sid, label, name) in enumerate(zip(order, subject_ids, labels, names), 1)]
    return pd.DataFrame(rows, columns=COLUMNS)

  prefix_ids = [int(subject[:3]) for subject, _ in order]
  first_seen = {}
  adversarial_ids = [first_seen.setdefault(subject, len(first_seen)) for subject, _ in order]
  vas_labels = [vas[key] for key in order]
  opi_labels = [opi[key] for key in order]
  opi_0_4 = [min(label, 4) for label in opi_labels]
  return {
    "samples.csv": frame(prefix_ids, vas_labels, [f"VAS_{v}" for v in vas_labels]),
    "samples_OPI.csv": frame(prefix_ids, opi_labels, [f"OPI_{v}" for v in opi_labels]),
    "samples_OPI_0_to_4.csv": frame(prefix_ids, opi_0_4, ["OPI_4_5" if v == 4 else f"OPI_{v}" for v in opi_0_4]),
    "samples_adversarial.csv": frame(adversarial_ids, vas_labels, [f"VAS_{v}" for v in vas_labels]),
  }


def raw_video(warped: Path, subject: str, sequence: str) -> Path:
  nested = warped / subject / sequence / f"{sequence}.mp4"
  return nested if nested.is_file() else warped / subject / f"{sequence}.mp4"


def crop_one(job: tuple[str, str, str]) -> str | None:
  """Crop and pad one video through temporary files; returns an error message or None."""
  from custom import unbc_crop  # already loaded by main() before the pool forked
  source, cropped, padded = job
  tmp_cropped, tmp_padded = cropped[:-4] + ".tmp.mp4", padded[:-4] + ".tmp.mp4"
  try:
    unbc_crop.crop_and_pad_video(source, tmp_cropped, tmp_padded)
    os.replace(tmp_cropped, cropped)
    os.replace(tmp_padded, padded)
    return None
  except Exception as error:  # reported together at the end of the stage
    return f"{source}: {error!r}"


def main(argv: list[str] | None = None) -> int:
  args = parser(__doc__.split("\n\n")[0], STAGES, "UNBC folder with OPI/, VAS/, WarpedVideos/ (see above).").parse_args(argv)
  resolve(args)
  unbc = args.root / "UNBC"
  csv = unbc / "starting_point" / "samples.csv"
  warped = unbc / "video" / "WarpedVideos"

  if "csv" in args.stages:
    vas = read_scores(require(args.raw / "VAS", "--raw must hold OPI/, VAS/ and WarpedVideos/"))
    opi = read_scores(require(args.raw / "OPI", "--raw must hold OPI/, VAS/ and WarpedVideos/"))
    if set(vas) != set(opi):
      sys.exit(f"error: OPI and VAS label files differ: {sorted(set(vas) ^ set(opi))[:5]}")
    for name, frame in build_csvs(paper_order(list(vas)), vas, opi).items():
      write_csv(frame, unbc / "starting_point" / name)

  require(csv, "Run the csv stage first.")
  csv_run = subset_csv(csv, args.subjects)

  if "video" in args.stages:
    raw_warped = require(args.raw / "WarpedVideos", "--raw must hold OPI/, VAS/ and WarpedVideos/")
    frame = read_csv(csv_run)
    for subject, sequence in zip(frame.subject_name, frame.sample_name):
      link(require(raw_video(raw_warped, subject, sequence), "a raw UNBC video is missing"),
           warped / subject / f"{sequence}.mp4")

  if "preprocess" in args.stages:
    require_videos(csv_run, warped)
    sources = videos_of(csv_run, warped)
    cropped = videos_of(csv_run, unbc / "video" / "WarpedVideos_Cropped")
    padded = videos_of(csv_run, unbc / "video" / "WarpedVideos_Cropped_interpolated_mirror")
    jobs = [(str(s), str(c), str(p)) for s, c, p in zip(sources, cropped, padded)
            if not (c.is_file() and p.is_file())]
    if not jobs:
      skip(unbc / "video" / "WarpedVideos_Cropped_interpolated_mirror")
    else:
      for _, c, p in jobs:
        Path(c).parent.mkdir(parents=True, exist_ok=True)
        Path(p).parent.mkdir(parents=True, exist_ok=True)
      # Load the crop module before forking: custom.helper starts a multiprocessing Manager
      # on import, which a pool worker is not allowed to do.
      from custom import unbc_crop  # noqa: F401
      print(f"cropping {len(jobs)} videos with {args.workers} processes")
      with mp.Pool(args.workers) as pool:
        errors = [e for e in pool.imap_unordered(crop_one, jobs) if e]
      if errors:
        sys.exit("error: cropping failed for:\n  " + "\n  ".join(errors))

  if "features" in args.stages:
    extract_features("unbc", args.root, csv_run, args.backbones, args.base_only, args.gpu)
  return 0


if __name__ == "__main__":
  sys.exit(main())

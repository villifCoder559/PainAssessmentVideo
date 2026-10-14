#!/usr/bin/env python3
"""Set up MIntPAIN from your own copy: CSV, videos, features.

--raw holds the subject folders of the release ("Sub1 <Name>"/Annotated_data_Sub01_Trial01/
Sub01_Trial01_Sweep01_Label0/RGB/*.jpg, ...), e.g. its Extracted_Data folder.

Stages:
  csv          MIntPAIN/starting_point/samples.csv (MIntPAIN/generate_videos.py)
  video        one 25 fps video per sweep from its RGB frames: MIntPAIN/video (needs ffmpeg)
  preprocess   frontalized videos padded to a multiple of 16 frames: MIntPAIN/video_frontalized
               (reference face: the BioVid landmarks shipped with the repository)
  features     the feature folders of configs/features/mintpain.yaml (GPU)
"""

from __future__ import annotations

import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _setup_common import (REPO_ROOT, extract_features, frontalize, parser, read_csv,  # noqa: E402
                           reference_landmarks, require, resolve, skip, subset_csv, videos_of)

sys.path.insert(0, str(REPO_ROOT / "MIntPAIN"))
import generate_videos  # noqa: E402

STAGES = ("csv", "video", "preprocess", "features")
FPS, CRF = 25, 10  # generate_videos.py defaults, used for the paper videos


def main(argv: list[str] | None = None) -> int:
  args = parser(__doc__.split("\n\n")[0], STAGES, "MIntPAIN folder with the Sub* folders (see above).").parse_args(argv)
  resolve(args)
  mintpain = args.root / "MIntPAIN"
  csv = mintpain / "starting_point" / "samples.csv"
  videos = mintpain / "video"
  clips = None

  if "csv" in args.stages:
    clips, empty = generate_videos.discover_clips(str(args.raw))
    if not clips:
      sys.exit(f"error: no Sub*_Trial*_Sweep*_Label* folders with RGB frames under {args.raw}")
    for sweep in empty:
      print(f"  [empty RGB, skipped] {sweep}")
    generate_videos.write_csv(generate_videos.build_csv_rows(clips), str(csv))
    print(f"wrote {csv} ({len(clips)} rows)")

  require(csv, "Run the csv stage first.")
  csv_run = subset_csv(csv, args.subjects)

  if "video" in args.stages:
    if shutil.which("ffmpeg") is None:
      sys.exit("error: ffmpeg not found on PATH")
    if all(p.is_file() for p in videos_of(csv_run, videos)):
      skip(videos)
    else:
      wanted = set(read_csv(csv_run).sample_name)
      clips = clips if clips is not None else generate_videos.discover_clips(str(args.raw))[0]
      tasks = [(c, str(videos), FPS, CRF, False) for c in clips if c["sample_name"] in wanted]
      print(f"encoding up to {len(tasks)} videos with {args.workers} processes")
      with ProcessPoolExecutor(max_workers=args.workers) as pool:
        failed = [f"{name}: {status}" for name, status in pool.map(generate_videos.encode_clip, tasks)
                  if status.startswith("failed")]
      if failed:
        sys.exit("error: encoding failed for:\n  " + "\n  ".join(failed))

  if "preprocess" in args.stages:
    frontalize(videos, csv_run, mintpain / "video_frontalized", reference_landmarks(args.root),
               args.workers)

  if "features" in args.stages:
    extract_features("mintpain", args.root, csv_run, args.backbones, args.base_only, args.gpu)
  return 0


if __name__ == "__main__":
  sys.exit(main())

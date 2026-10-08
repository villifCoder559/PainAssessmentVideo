#!/usr/bin/env python3
"""
Assemble MIntPAIN RGB frame sequences into per-clip .mp4 videos and emit a
samples.csv that mirrors partA/starting_point/samples.csv, so the existing
PainAssessmentVideo pipeline (extract_feature.py, train_model.py) can consume
MIntPAIN with no code changes.

Layout produced (mirrors partA/video/video_frontalized):
  MIntPAIN/video/Sub01/Sub01-L0-0101.mp4   # Sub01, Trial01, Sweep01, Label0
  ...
  MIntPAIN/starting_point/samples.csv      # subject_id subject_name class_id
                                           # class_name sample_id sample_name

Both trials of a subject are merged into one subject folder; the trial is
encoded in the filename to keep names unique. Class names are native L0..L4.
Videos keep the original 1920x1080 RGB resolution.
"""

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

SWEEP_RE = re.compile(r"Sub(\d+)_Trial(\d+)_Sweep(\d+)_Label(\d+)")


def parse_sweep(path):
  """
  Parse a MIntPAIN sweep folder path into its identifying fields.

  Args:
    path: Filesystem path to a sweep folder, whose basename looks like
          'Sub01_Trial01_Sweep01_Label0'.

  Returns:
    Dict with int keys 'sub_num', 'trial', 'sweep', 'label', or None if the
    basename does not match the expected sweep naming pattern.
  """
  m = SWEEP_RE.search(os.path.basename(path))
  if not m:
    return None
  return {
    "sub_num": int(m.group(1)),
    "trial": int(m.group(2)),
    "sweep": int(m.group(3)),
    "label": int(m.group(4)),
  }


def build_names(fields):
  """
  Derive the anonymized subject/sample naming fields for a parsed sweep.

  Args:
    fields: Dict from parse_sweep with 'sub_num', 'trial', 'sweep', 'label'.

  Returns:
    Dict adding 'subject_name' (e.g. 'Sub01'), 'class_name' (e.g. 'L0') and
    'sample_name' (e.g. 'Sub01-L0-0101'). sample_name uses only [A-Za-z0-9-].
  """
  subject_name = "Sub%02d" % fields["sub_num"]
  class_name = "L%d" % fields["label"]
  sample_name = "%s-%s-%02d%02d" % (
    subject_name, class_name, fields["trial"], fields["sweep"])
  return {"subject_name": subject_name, "class_name": class_name,
          "sample_name": sample_name}


def discover_clips(data_root):
  """
  Find every sweep folder under data_root and gather its RGB frames.

  Args:
    data_root: Path to MIntPAIN/Extracted_Data. The glob pattern '*/*/Sub*...'
               transparently traverses the space-containing subject dirs and
               the per-trial dirs.

  Returns:
    Tuple (clips, empty). 'clips' is a list of dicts with parsed fields, names,
    'sweep_dir', 'rgb_glob' and 'n_frames' (>0). 'empty' is a list of sweep dir
    paths that had no RGB frames (skipped).
  """
  pattern = os.path.join(data_root, "*", "*", "Sub*_Trial*_Sweep*_Label*")
  clips = []
  empty = []
  for sweep_dir in sorted(glob.glob(pattern)):
    if not os.path.isdir(sweep_dir):
      continue
    fields = parse_sweep(sweep_dir)
    if fields is None:
      continue
    rgb_dir = os.path.join(sweep_dir, "RGB")
    frames = sorted(glob.glob(os.path.join(rgb_dir, "RGB-*.jpg")))
    if not frames:
      empty.append(sweep_dir)
      continue
    clip = dict(fields)
    clip.update(build_names(fields))
    clip["sweep_dir"] = sweep_dir
    clip["rgb_glob"] = os.path.join(rgb_dir, "RGB-*.jpg")
    clip["n_frames"] = len(frames)
    clips.append(clip)
  return clips, empty


def build_csv_rows(clips):
  """
  Build samples.csv rows with a per-subject running sample_id.

  Args:
    clips: List of clip dicts from discover_clips.

  Returns:
    List of [subject_id, subject_name, class_id, class_name, sample_id,
    sample_name] rows. Within each subject, clips are ordered by
    (class_id, trial, sweep) so the CSV groups by class like the original
    samples.csv. sample_id is a single running counter across the whole CSV
    (1..N over all subjects), so no sample_id is duplicated.
  """
  by_subject = {}
  for c in clips:
    by_subject.setdefault(c["sub_num"], []).append(c)

  rows = []
  sample_id = 0
  for sub_num in sorted(by_subject):
    subject_clips = sorted(
      by_subject[sub_num], key=lambda c: (c["label"], c["trial"], c["sweep"]))
    for c in subject_clips:
      sample_id += 1
      rows.append([
        sub_num, c["subject_name"], c["label"], c["class_name"],
        sample_id, c["sample_name"],
      ])
  return rows


def write_csv(rows, out_csv):
  """
  Write the samples CSV (tab-separated) with the standard header.

  Args:
    rows:    List of row lists from build_csv_rows.
    out_csv: Destination path; parent dirs are created if missing.

  Returns:
    None.
  """
  os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
  with open(out_csv, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["subject_id", "subject_name", "class_id", "class_name",
                "sample_id", "sample_name"])
    w.writerows(rows)


def write_run_info(args, out_txt):
  """
  Dump the launch command and all resolved arguments to a txt file.

  Args:
    args:    The argparse Namespace returned by parse_args().
    out_txt: Destination path; parent dirs are created if missing.

  Returns:
    None. Writes the full command line followed by one 'key: value' line per
    argument, so a run can be reproduced from the file alone.
  """
  os.makedirs(os.path.dirname(os.path.abspath(out_txt)), exist_ok=True)
  with open(out_txt, "w") as f:
    f.write("command: python3 " + " ".join(sys.argv) + "\n\n")
    for key, value in sorted(vars(args).items()):
      f.write("%s: %s\n" % (key, value))


def encode_clip(args):
  """
  Encode one clip's RGB frames into an .mp4 via ffmpeg.

  Args:
    args: Tuple (clip, out_video, fps, crf, overwrite) where 'clip' is a clip
          dict from discover_clips, 'out_video' is the root output dir, 'fps'
          the output frame rate, 'crf' the x264 quality, and 'overwrite' a bool.

  Returns:
    Tuple (sample_name, status) where status is 'encoded', 'skipped' (already
    exists) or 'failed:<reason>'. ffmpeg reads frames with a glob pattern, so
    the lexical (== chronological) frame order is preserved.
  """
  clip, out_video, fps, crf, overwrite = args
  subject_dir = os.path.join(out_video, clip["subject_name"])
  out_path = os.path.join(subject_dir, clip["sample_name"] + ".mp4")

  if os.path.exists(out_path) and not overwrite:
    return (clip["sample_name"], "skipped")

  os.makedirs(subject_dir, exist_ok=True)
  cmd = [
    "ffmpeg", "-y", "-framerate", str(fps),
    "-pattern_type", "glob", "-i", clip["rgb_glob"],
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
    out_path,
  ]
  proc = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE)
  if proc.returncode != 0:
    tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
    reason = tail[-1] if tail else "ffmpeg error"
    return (clip["sample_name"], "failed:" + reason)
  return (clip["sample_name"], "encoded")


def main():
  """
  CLI entry point: discover clips, write the CSV, encode videos in parallel.

  Args:
    None (reads sys.argv via argparse).

  Returns:
    None. Exits non-zero if ffmpeg is unavailable or no clips are found.
  """
  here = os.path.dirname(os.path.abspath(__file__))
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--data_root", default=os.path.join(here, "Extracted_Data"))
  ap.add_argument("--out_video", default=os.path.join(here, "video"))
  ap.add_argument("--out_csv",
                  default=os.path.join(here, "starting_point", "samples.csv"))
  ap.add_argument("--fps", type=int, default=25)
  ap.add_argument("--crf", type=int, default=10, help="lower=better quality, 0=lossless, 51=worst")
  ap.add_argument("--workers", type=int, default=4)
  ap.add_argument("--overwrite", action="store_true",
                  help="re-encode clips whose .mp4 already exists")
  ap.add_argument("--only_csv", action="store_true",
                  help="discover + write CSV only, skip encoding")
  args = ap.parse_args()

  if shutil.which("ffmpeg") is None and not args.only_csv:
    sys.exit("ERROR: ffmpeg not found on PATH.")

  print("Discovering sweep folders under %s ..." % args.data_root)
  clips, empty = discover_clips(args.data_root)
  if not clips:
    sys.exit("ERROR: no clips found. Check --data_root.")
  subjects = sorted({c["subject_name"] for c in clips})
  print("Found %d clips across %d subjects (%d empty-RGB sweeps skipped)."
        % (len(clips), len(subjects), len(empty)))
  for d in empty:
    print("  [empty RGB, skipped] %s" % d)

  rows = build_csv_rows(clips)
  write_csv(rows, args.out_csv)
  print("Wrote %d rows -> %s" % (len(rows), args.out_csv))

  out_txt = os.path.join(args.out_video, "generate_videos_command.txt")
  write_run_info(args, out_txt)
  print("Wrote run info -> %s" % out_txt)

  if args.only_csv:
    print("only_csv: skipping encoding.")
    return

  tasks = [(c, args.out_video, args.fps, args.crf, args.overwrite)
           for c in clips]
  counts = {"encoded": 0, "skipped": 0, "failed": 0}
  failures = []
  with ProcessPoolExecutor(max_workers=args.workers) as ex:
    futures = [ex.submit(encode_clip, t) for t in tasks]
    bar = tqdm(as_completed(futures), total=len(tasks), desc="Encoding",
               unit="clip")
    for fut in bar:
      name, status = fut.result()
      if status.startswith("failed"):
        counts["failed"] += 1
        failures.append((name, status))
      else:
        counts[status] += 1
      bar.set_postfix(counts)

  print("\nDone. encoded=%d skipped-existing=%d failed=%d empty-skipped=%d"
        % (counts["encoded"], counts["skipped"], counts["failed"], len(empty)))
  for name, status in failures:
    print("  [FAILED] %s -> %s" % (name, status))


if __name__ == "__main__":
  main()

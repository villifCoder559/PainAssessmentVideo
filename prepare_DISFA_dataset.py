"""
Prepare the DISFA dataset for the pain/emotion assessment pipeline.

Given the original videos, their frontalized versions, and the per-subject Action
Unit annotations, this script:
  1. Validates that, for each video, the original and frontalized frame counts match
     the number of AU annotation lines (exact match), keeping only those videos.
  2. Extracts every frame of each frontalized video as a PNG image.
  3. (Optional) Emits a per-frame "chunk" video duplicating the frame N times, for
     feeding single images to a video backbone.
  4. Writes a labels CSV (same format as partA/starting_point/samples.csv) where each
     frame is labeled with the intensity of a target Action Unit.

Run:
  python3 prepare_DISFA_dataset.py --au_name au12 --make_image_videos
"""

import argparse
import csv
import os
import re

import cv2
import tqdm


def extract_subject(filename: str) -> str:
  """
  Extract the DISFA subject token from a video filename.

  Args:
    filename: Video file name, e.g. 'RightVideoSN001_Comp.avi'.

  Returns:
    The 'SN###' subject token (e.g. 'SN001').

  Raises:
    ValueError: If no 'SN###' token is found in the filename.
  """
  match = re.search(r"SN\d+", filename)
  if match is None:
    raise ValueError(f"No 'SN###' subject token found in '{filename}'")
  return match.group(0)


def count_frames(path: str) -> int:
  """
  Count the number of frames in a video file.

  Args:
    path: Path to the video file.

  Returns:
    Number of frames reported by cv2, or -1 if the video cannot be opened.
  """
  cap = cv2.VideoCapture(path)
  if not cap.isOpened():
    return -1
  n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  cap.release()
  return n


def load_au_intensities(path: str) -> list:
  """
  Load AU intensity labels from a DISFA annotation txt file.

  Args:
    path: Path to a 'SN###_au##.txt' file with lines 'frame,intensity'.

  Returns:
    List of intensities (int), index i holding the intensity for frame i+1.
  """
  intensities = []
  with open(path, "r") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      intensities.append(int(line.split(",")[1]))
  return intensities


def validate_videos(args) -> list:
  """
  Find frontalized videos whose original/frontalized frame counts match AU lines.

  Args:
    args: Parsed CLI arguments (see build_parser).

  Returns:
    List of dicts with keys: 'subject', 'front_path', 'intensities' for each
    video passing the exact frame-count match.
  """
  validated = []
  front_files = sorted(os.listdir(args.frontalized_folder))
  for front_name in tqdm.tqdm(front_files, desc="Validating videos"):
    front_path = os.path.join(args.frontalized_folder, front_name)
    if not os.path.isfile(front_path):
      continue
    stem = os.path.splitext(front_name)[0]

    # Locate the matching original video by stem (extension may differ).
    orig_path = None
    for orig_name in os.listdir(args.video_folder):
      if os.path.splitext(orig_name)[0] == stem:
        orig_path = os.path.join(args.video_folder, orig_name)
        break
    if orig_path is None:
      print(f"[skip] {front_name}: no matching original video")
      continue

    try:
      subject = extract_subject(front_name)
    except ValueError as e:
      print(f"[skip] {front_name}: {e}")
      continue

    au_path = os.path.join(args.action_units_folder, subject, f"{subject}_{args.au_name}.txt")
    if not os.path.isfile(au_path):
      print(f"[skip] {front_name}: missing AU file {au_path}")
      continue

    front_frames = count_frames(front_path)
    orig_frames = count_frames(orig_path)
    intensities = load_au_intensities(au_path)
    au_lines = len(intensities)

    if not (orig_frames == front_frames == au_lines):
      print(
        f"[skip] {front_name}: frame-count mismatch "
        f"(orig={orig_frames}, front={front_frames}, au={au_lines})"
      )
      continue

    validated.append({"subject": subject, "front_path": front_path, "intensities": intensities})
    print(f"[ok]   {front_name}: {front_frames} frames")

  return validated


def write_image_video(frame, path: str, n_frames: int) -> None:
  """
  Write a short video duplicating a single frame n_frames times.

  Args:
    frame:    Image frame (BGR ndarray) as returned by cv2.
    path:     Output .mp4 path.
    n_frames: Number of times to duplicate the frame.

  Returns:
    None.
  """
  height, width = frame.shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(path, fourcc, 30.0, (width, height))
  for _ in range(n_frames):
    writer.write(frame)
  writer.release()


def extract_frames(video, args) -> None:
  """
  Extract every frame of a validated video as PNG (and optional chunk videos).

  Args:
    video: Dict with 'subject' and 'front_path' (from validate_videos).
    args:  Parsed CLI arguments (see build_parser).

  Returns:
    None.
  """
  subject = video["subject"]
  img_dir = os.path.join(args.image_root, subject)
  os.makedirs(img_dir, exist_ok=True)
  if args.make_image_videos:
    vid_dir = os.path.join(args.video_image_root, subject)
    os.makedirs(vid_dir, exist_ok=True)

  cap = cv2.VideoCapture(video["front_path"])
  n_frames = len(video["intensities"])  # validated to equal the real frame count
  for idx in tqdm.tqdm(range(n_frames), desc=subject, leave=False):
    ok, frame = cap.read()
    if not ok:
      break
    frame_number = idx + 1  # 1-indexed to align with AU annotations
    stem = f"{subject}_frame_{frame_number}"

    img_path = os.path.join(img_dir, f"{stem}.png")
    if args.overwrite_png or not os.path.exists(img_path):
      cv2.imwrite(img_path, frame)

    if args.make_image_videos:
      vid_path = os.path.join(vid_dir, f"{stem}.mp4")
      if args.overwrite_png or not os.path.exists(vid_path):
        write_image_video(frame, vid_path, args.image_video_frames)
  cap.release()


def write_csv(validated: list, args) -> int:
  """
  Write the labels CSV (tab-separated) matching samples.csv format.

  Args:
    validated: List of validated video dicts (from validate_videos).
    args:      Parsed CLI arguments (see build_parser).

  Returns:
    Number of sample rows written.
  """
  os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
  sample_id = 0
  with open(args.csv_path, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["subject_id", "subject_name", "class_id", "class_name", "sample_id", "sample_name"])
    for video in validated:
      subject = video["subject"]
      subject_id = int(re.search(r"\d+", subject).group(0))
      for frame_number, intensity in enumerate(video["intensities"], start=1):
        sample_id += 1
        class_name = f"{args.au_name}_{intensity}"
        sample_name = f"{subject}_frame_{frame_number}"
        writer.writerow([subject_id, subject, intensity, class_name, sample_id, sample_name])
  return sample_id


def build_parser() -> argparse.ArgumentParser:
  """
  Build the command-line argument parser.

  Returns:
    Configured argparse.ArgumentParser.
  """
  parser = argparse.ArgumentParser(description="Prepare the DISFA dataset (frames + labels CSV).")
  parser.add_argument("--video_folder", default="DISFA/video", help="Folder with original videos.")
  parser.add_argument("--frontalized_folder", default="DISFA/video_frontalized", help="Folder with frontalized videos.")
  parser.add_argument("--image_root", default="DISFA/image_frontalized", help="Output root for per-frame PNGs.")
  parser.add_argument("--action_units_folder", default="DISFA/ActionUnits", help="Root of per-subject AU annotations.")
  parser.add_argument("--au_name", default="au12", help="Target Action Unit (e.g. au12).")
  parser.add_argument("--csv_path", default=None, help="Output CSV path (default DISFA/starting_point/{au_name}_samples.csv).")
  parser.add_argument("--overwrite_png", action="store_true", help="Re-decode/overwrite existing PNGs (and chunk videos).")
  parser.add_argument("--overwrite_csv", action="store_true", help="Overwrite the CSV if it already exists.")
  parser.add_argument("--make_image_videos", action="store_true", help="Also emit per-frame chunk videos.")
  parser.add_argument("--video_image_root", default="DISFA/video_image", help="Output root for chunk videos.")
  parser.add_argument("--image_video_frames", type=int, default=16, help="Frames per chunk video.")
  return parser


def main() -> None:
  """
  Entry point: validate videos, extract frames, and write the labels CSV.

  Returns:
    None.
  """
  args = build_parser().parse_args()
  if args.csv_path is None:
    args.csv_path = os.path.join("DISFA", "starting_point", f"{args.au_name}_samples.csv")

  if os.path.exists(args.csv_path) and not args.overwrite_csv:
    raise SystemExit(f"CSV already exists: {args.csv_path} (use --overwrite_csv to replace).")

  print("== Step 1: validating videos ==")
  validated = validate_videos(args)
  print(f"\n{len(validated)} video(s) passed validation.\n")
  if not validated:
    raise SystemExit("No videos passed validation; nothing to do.")

  print("== Step 2: extracting frames ==")
  for video in tqdm.tqdm(validated, desc="Extracting"):
    extract_frames(video, args)

  print("\n== Step 3: writing CSV ==")
  n_rows = write_csv(validated, args)
  print(f"Wrote {n_rows} rows to {args.csv_path}")


if __name__ == "__main__":
  main()

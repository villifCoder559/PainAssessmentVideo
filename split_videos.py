"""
split_videos.py — Horizontally split .mp4 videos into top and bottom halves.

Recursively scans a root folder for .mp4 files (skipping files with '$' in
their name), then for each video produces two new clips:
  <stem>$top.mp4  — upper portion
  <stem>$down.mp4 — lower portion

Output videos use the avc1 (H.264) codec with no audio and are written
next to their source file.

Usage:
  python3 split_videos.py --input_folder <path> [--ratio 0.5] [--workers 1]
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import tqdm


def split_video(filepath: str, ratio: float) -> None:
  """
  Split a single video into top and bottom halves using OpenCV.

  Reads every frame from the source file, slices it horizontally at
  `ratio * height`, and writes each slice to a separate VideoWriter.

  Args:
    filepath: Path to the source .mp4 file.
    ratio:    Fraction of height assigned to the top half (0 < ratio < 1).
  """
  cap = cv2.VideoCapture(filepath)
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {filepath}")

  width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps    = cap.get(cv2.CAP_PROP_FPS)
  fourcc = cv2.VideoWriter_fourcc(*"avc1")

  top_h    = int(height * ratio)
  top_h   -= top_h % 2       # H.264 requires even dimensions
  bottom_h = height - top_h
  bottom_h -= bottom_h % 2

  stem      = filepath[:-4]  # strip ".mp4"
  top_path  = stem + "$top.mp4"
  down_path = stem + "$down.mp4"

  top_writer  = cv2.VideoWriter(top_path,  fourcc, fps, (width, top_h))
  down_writer = cv2.VideoWriter(down_path, fourcc, fps, (width, bottom_h))

  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      top_writer.write(frame[:top_h, :])
      down_writer.write(frame[top_h:top_h + bottom_h, :])
  finally:
    cap.release()
    top_writer.release()
    down_writer.release()


def discover_videos(root: str) -> list[str]:
  """
  Recursively find .mp4 files under root that do not contain '$' in their name.

  Args:
    root: Root directory to search.

  Returns:
    Sorted list of absolute file paths.
  """
  videos = []
  for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
      if fname.endswith(".mp4") and "$" not in fname:
        videos.append(os.path.join(dirpath, fname))
  return sorted(videos)


def main() -> None:
  """
  Entry point: parse CLI arguments and process all discovered videos.
  """
  parser = argparse.ArgumentParser(
    description="Split .mp4 videos horizontally into top and bottom halves."
  )
  parser.add_argument(
    "--input_folder",
    required=True,
    help="Root folder to scan for .mp4 files.",
  )
  parser.add_argument(
    "--ratio",
    type=float,
    default=0.5,
    help="Fraction of height for the top half (default: 0.5).",
  )
  parser.add_argument(
    "--workers",
    type=int,
    default=1,
    help="Number of parallel worker threads (default: 1).",
  )
  args = parser.parse_args()

  if not (0 < args.ratio < 1):
    raise ValueError(f"--ratio must be between 0 and 1 (exclusive), got {args.ratio}")

  videos = discover_videos(args.input_folder)
  total = len(videos)
  print(f"Found {total} video(s) to process with {args.workers} worker(s).")

  with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {executor.submit(split_video, fp, args.ratio): fp for fp in videos}
    try:
      for fut in tqdm.tqdm(as_completed(futures), total=total, desc="Processing videos"):
        fut.result()
    except Exception:
      for f in futures:
        f.cancel()
      raise

  print("Done.")


if __name__ == "__main__":
  main()

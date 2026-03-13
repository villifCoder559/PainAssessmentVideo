"""
resize_videos.py — Batch-resize .mp4 videos under a root folder.

Usage:
  python resize_videos.py <root_folder> <WxH> [--workers N]

Example:
  python resize_videos.py partA/video/video 320x240 --workers 4

Output folder is created as a sibling of root_folder, named
"{root_folder_name}_{W}x{H}", preserving the original subfolder structure.
Existing output files are skipped.
"""

import argparse
import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Tuple

import cv2
from tqdm import tqdm


def parse_resolution(res_str: str) -> Tuple[int, int]:
  """
  Parse a resolution string of the form WxH.

  Args:
    res_str: Resolution string, e.g. "640x480".

  Returns:
    Tuple (width, height) as integers.

  Raises:
    ValueError: If the format is invalid or values are non-positive.
  """
  match = re.fullmatch(r"(\d+)[xX](\d+)", res_str.strip())
  if not match:
    raise ValueError(f"Resolution must be WxH (e.g. 640x480), got: '{res_str}'")
  w, h = int(match.group(1)), int(match.group(2))
  if w <= 0 or h <= 0:
    raise ValueError(f"Width and height must be positive, got: {w}x{h}")
  return w, h


def resize_video(args: Tuple[str, str, int, int]) -> str:
  """
  Resize a single video file and write it to the destination path.

  Args:
    args: Tuple of (src_path, dst_path, width, height).
      src_path: Absolute path to source .mp4 file.
      dst_path: Absolute path to destination .mp4 file.
      width:    Target frame width in pixels.
      height:   Target frame height in pixels.

  Returns:
    Status string: "skipped", "ok", or "error: <message>".
  """
  src_path, dst_path, width, height = args

  if os.path.exists(dst_path):
    return "skipped"

  os.makedirs(os.path.dirname(dst_path), exist_ok=True)

  cap = cv2.VideoCapture(src_path)
  if not cap.isOpened():
    return f"error: cannot open {src_path}"

  fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

  if not writer.isOpened():
    cap.release()
    return f"error: cannot create writer {dst_path}"

  while True:
    ret, frame = cap.read()
    if not ret:
      break
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    writer.write(resized)

  cap.release()
  writer.release()
  return "ok"


def main() -> None:
  """
  Entry point: parse arguments, discover .mp4 files, and resize in parallel.
  """
  parser = argparse.ArgumentParser(
    description="Batch-resize .mp4 videos preserving folder structure."
  )
  parser.add_argument("--root_folder", help="Root folder containing .mp4 files.")
  parser.add_argument("--resolution", help="Target resolution as WxH, e.g. 640x480.")
  parser.add_argument(
    "--workers",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of parallel worker processes (default: cpu count).",
  )
  args = parser.parse_args()

  root = Path(args.root_folder).resolve()
  if not root.is_dir():
    print(f"Error: '{root}' is not a directory.", file=sys.stderr)
    sys.exit(1)

  try:
    width, height = parse_resolution(args.resolution)
  except ValueError as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

  res_tag = f"{width}x{height}"
  out_root = root.parent / f"{root.name}_{res_tag}"

  mp4_files = sorted(root.rglob("*.mp4"))
  if not mp4_files:
    print(f"No .mp4 files found under '{root}'.")
    sys.exit(0)

  print(f"Source root : {root}")
  print(f"Output root : {out_root}")
  print(f"Resolution  : {width}x{height}")
  print(f"Files found : {len(mp4_files)}")
  print(f"Workers     : {args.workers}")
  print()

  task_args = []
  for src in mp4_files:
    rel = src.relative_to(root)
    dst = out_root / rel
    task_args.append((str(src), str(dst), width, height))

  results = []
  with multiprocessing.Pool(processes=args.workers) as pool:
    with tqdm(total=len(task_args), unit="video", desc="Resizing") as bar:
      for result in pool.imap_unordered(resize_video, task_args):
        results.append(result)
        bar.update(1)
        if result == "skipped":
          bar.set_postfix_str("last: skipped")
        elif result.startswith("error"):
          bar.set_postfix_str("last: ERROR")
        else:
          bar.set_postfix_str("last: ok")

  total = len(results)
  skipped = results.count("skipped")
  errors = sum(1 for r in results if r.startswith("error"))
  processed = total - skipped - errors

  print(f"\nDone. Total={total}  Processed={processed}  Skipped={skipped}  Errors={errors}")


if __name__ == "__main__":
  main()

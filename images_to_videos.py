"""
images_to_videos.py — Convert still images to 16-frame MP4 videos.

Recursively scans an input folder for image files (png, jpg, jpeg, bmp,
tiff, webp), then for each image produces an MP4 where that frame is
repeated --frames times. Output videos are written to --output_folder
preserving the original subfolder structure.

  Image/001_F/file.png  →  Video/001_F/file.mp4

Output uses the avc1 (H.264) codec at 25 FPS. Existing output files are
skipped. Processing is parallelised with ThreadPoolExecutor.

Usage:
  python3 images_to_videos.py \\
    --input_folder  Image/ \\
    --output_folder Video/ \\
    [--frames 16] [--fps 25] [--workers N]
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def discover_images(root: str) -> list[str]:
  """
  Recursively find image files under root.

  Args:
    root: Root directory to search.

  Returns:
    Sorted list of absolute file paths whose extension is in IMAGE_EXTENSIONS.
  """
  images = []
  for dirpath, _, filenames in os.walk(root):
    for fname in filenames:
      if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
        images.append(os.path.join(dirpath, fname))
  return sorted(images)


def image_to_video(
  img_path: str,
  src_root: str,
  dst_root: str,
  n_frames: int,
  fps: float,
) -> str:
  """
  Convert a single image to an n_frames-long MP4 by repeating the frame.

  Args:
    img_path: Absolute path to the source image file.
    src_root: Root of the source image tree (used to compute relative path).
    dst_root: Root of the destination video tree.
    n_frames: Number of times the frame is written (e.g. 16).
    fps:      Output video frame rate.

  Returns:
    Status string: "skipped", "ok", or "error: <message>".
  """
  rel = Path(img_path).relative_to(src_root)
  dst_path = str(Path(dst_root) / rel.with_suffix(".mp4"))

  if os.path.exists(dst_path):
    return "skipped"

  os.makedirs(os.path.dirname(dst_path), exist_ok=True)

  frame = cv2.imread(img_path)
  if frame is None:
    return f"error: cannot read image {img_path}"

  h, w = frame.shape[:2]
  w -= w % 2  # H.264 requires even dimensions
  h -= h % 2
  if frame.shape[1] != w or frame.shape[0] != h:
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

  fourcc = cv2.VideoWriter_fourcc(*"avc1")
  writer = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
  if not writer.isOpened():
    return f"error: cannot create writer {dst_path}"

  try:
    for _ in range(n_frames):
      writer.write(frame)
  finally:
    writer.release()

  return "ok"


def main() -> None:
  """
  Entry point: parse CLI arguments and convert all discovered images in parallel.
  """
  parser = argparse.ArgumentParser(
    description="Convert still images to repeated-frame MP4 videos, preserving folder structure."
  )
  parser.add_argument("--input_folder",  required=True, help="Root folder to scan for images.")
  parser.add_argument("--output_folder", required=True, help="Root folder for output MP4 files.")
  parser.add_argument("--frames",  type=int,   default=16,              help="Frames per video (default: 16).")
  parser.add_argument("--fps",     type=float, default=25.0,            help="Output frame rate (default: 25).")
  parser.add_argument("--workers", type=int,   default=os.cpu_count(),  help="Parallel worker threads (default: cpu count).")
  args = parser.parse_args()

  src_root = str(Path(args.input_folder).resolve())
  dst_root = str(Path(args.output_folder).resolve())

  images = discover_images(src_root)
  total = len(images)
  print(f"Found {total} image(s) to process with {args.workers} worker(s).")

  with ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = {
      executor.submit(image_to_video, fp, src_root, dst_root, args.frames, args.fps): fp
      for fp in images
    }
    results = []
    try:
      for fut in tqdm.tqdm(as_completed(futures), total=total, desc="Converting"):
        results.append(fut.result())
    except Exception:
      for f in futures:
        f.cancel()
      raise

  skipped = results.count("skipped")
  errors  = sum(1 for r in results if r.startswith("error"))
  processed = total - skipped - errors

  print(f"\nDone. Total={total}  Processed={processed}  Skipped={skipped}  Errors={errors}")


if __name__ == "__main__":
  main()

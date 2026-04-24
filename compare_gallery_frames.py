import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import cv2
import numpy as np


def parse_epoch(filename: str) -> float:
  """
  Extract a sortable epoch key from a PNG filename.

  Args:
    filename: PNG filename, e.g. 'model_epoch_100_train_...png' or
              'best_model_ep_66_train_...png'.

  Returns:
    Float epoch key. Regular epochs return int value; best-model files
    return epoch + 0.5 so they sort after the corresponding regular epoch.
    Unrecognised filenames return inf (sorted last).
  """
  m = re.search(r'model_epoch_(-?\d+)', filename)
  if m:
    return float(m.group(1))
  m = re.search(r'best_model_ep_(\d+)', filename)
  if m:
    return float(m.group(1)) + 0.5
  return float('inf')


def get_sorted_pngs(folder: str) -> list:
  """
  Return all PNG files in folder sorted by epoch key.

  Args:
    folder: Path to directory containing PNG files.

  Returns:
    List of Path objects sorted by parse_epoch, then alphabetically as
    tiebreaker.
  """
  paths = sorted(Path(folder).glob('*.png'), key=lambda p: (parse_epoch(p.name), p.name))
  return paths


def make_text_bar(width: int, height: int, left_text: str, right_text: str) -> np.ndarray:
  """
  Create a black bar with two labels centred in the left and right halves.

  Used for horizontal orientation.

  Args:
    width:      Total width of the bar in pixels.
    height:     Height of the bar in pixels.
    left_text:  Label drawn centred in the left half.
    right_text: Label drawn centred in the right half.

  Returns:
    np.ndarray of shape (height, width, 3), dtype uint8.
  """
  bar = np.zeros((height, width, 3), dtype=np.uint8)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.9
  thickness = 2
  color = (255, 255, 255)
  half = width // 2

  for text, x_offset in [(left_text, 0), (right_text, half)]:
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = x_offset + (half - tw) // 2
    y = (height + th) // 2
    cv2.putText(bar, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

  return bar


def make_separator_bar(width: int, height: int, top_text: str, bottom_text: str) -> np.ndarray:
  """
  Create a black bar with two labels stacked vertically.

  Used as the divider between the two images in vertical orientation.
  top_text labels image_1 (above the bar) and bottom_text labels image_2
  (below the bar).

  Args:
    width:       Width of the bar, matching the scaled image width.
    height:      Total height of the bar in pixels.
    top_text:    Label centred in the top half of the bar.
    bottom_text: Label centred in the bottom half of the bar.

  Returns:
    np.ndarray of shape (height, width, 3), dtype uint8.
  """
  bar = np.zeros((height, width, 3), dtype=np.uint8)
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.9
  thickness = 2
  color = (255, 255, 255)
  half = height // 2

  for text, y_offset in [(top_text, 0), (bottom_text, half)]:
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (width - tw) // 2
    y = y_offset + (half + th) // 2
    cv2.putText(bar, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

  return bar


def build_frame(
  img1: np.ndarray,
  img2: np.ndarray,
  scale: float,
  bar: np.ndarray,
  orientation: str,
) -> np.ndarray:
  """
  Resize both images and assemble the final frame with the label bar.

  Args:
    img1:        First image array (H, W, 3).
    img2:        Second image array (H, W, 3).
    scale:       Resize factor applied independently to each image.
    bar:         Pre-rendered label bar.
    orientation: 'horizontal' places images side-by-side with bar below;
                 'vertical' stacks images with bar in between.

  Returns:
    Assembled frame as np.ndarray (uint8, BGR).
  """
  h, w = img1.shape[:2]
  new_w, new_h = int(w * scale), int(h * scale)
  img1 = cv2.resize(img1, (new_w, new_h))
  img2 = cv2.resize(img2, (new_w, new_h))

  if orientation == 'horizontal':
    return np.concatenate([np.concatenate([img1, img2], axis=1), bar], axis=0)
  else:
    return np.concatenate([img1, bar, img2], axis=0)


def extract_run_tag(folder_path: str) -> str:
  """
  Extract a short human-readable tag from a folder path.

  Looks for a path component that starts with 'history_run_' and returns
  its last underscore-separated token (the codename). Falls back to the
  immediate parent directory name.

  Args:
    folder_path: Absolute or relative folder path string.

  Returns:
    Short tag string suitable for use in a filename.
  """
  parts = Path(folder_path).resolve().parts
  for part in parts:
    if part.startswith('history_run_'):
      tokens = part.split('_')
      # Last token is a numeric run-id timestamp; codename is the one before it
      for token in reversed(tokens):
        if token and not token.isdigit():
          return token
  return Path(folder_path).resolve().parent.name


def make_run_dirs(folder_1: str, folder_2: str) -> tuple:
  """
  Create a uniquely named output sub-folder inside each input folder.

  The folder name encodes both run tags and a timestamp so it is unique and
  self-documenting.  Both folders receive a sub-folder with the same name.

  Args:
    folder_1: Path to the first PNG folder.
    folder_2: Path to the second PNG folder.

  Returns:
    Tuple (dir_1, dir_2, stem) where dir_1/dir_2 are the created sub-folder
    paths and stem is the shared base name (used for the video filename).
  """
  tag1 = extract_run_tag(folder_1)
  tag2 = extract_run_tag(folder_2)
  ts = int(time.time())
  stem = f'comparison_{tag1}_vs_{tag2}_{ts}'
  dir_1 = os.path.join(folder_1, stem)
  dir_2 = os.path.join(folder_2, stem)
  os.makedirs(dir_1, exist_ok=True)
  os.makedirs(dir_2, exist_ok=True)
  return dir_1, dir_2, stem


def main() -> None:
  """
  Entry point: parse arguments, build side-by-side comparison video from
  two folders of PNG frames.
  """
  parser = argparse.ArgumentParser(
    description='Create a side-by-side comparison video from two PNG frame folders.'
  )
  parser.add_argument('--folder_1', required=True, help='Path to first PNG folder')
  parser.add_argument('--folder_2', required=True, help='Path to second PNG folder')
  parser.add_argument('--desc_1', default='Folder 1', help='Label for first folder')
  parser.add_argument('--desc_2', default='Folder 2', help='Label for second folder')
  parser.add_argument('--fps', type=float, default=4.0, help='Output video FPS (default: 4)')
  parser.add_argument('--scale', type=float, default=0.5, help='Scale factor per image (default: 0.5)')
  parser.add_argument(
    '--orientation', choices=['horizontal', 'vertical'], default='vertical',
    help='Layout: horizontal (left/right) or vertical (top/bottom, default)',
  )
  parser.add_argument('--output', default=None, help='Output video path (auto-generated if omitted)')
  args = parser.parse_args()

  files_1 = get_sorted_pngs(args.folder_1)
  files_2 = get_sorted_pngs(args.folder_2)

  if not files_1:
    raise ValueError(f'No PNG files found in: {args.folder_1}')
  if not files_2:
    raise ValueError(f'No PNG files found in: {args.folder_2}')

  print(f'Folder 1: {len(files_1)} frames')
  print(f'Folder 2: {len(files_2)} frames')

  n1_orig, n2_orig = len(files_1), len(files_2)
  n = min(n1_orig, n2_orig)
  if n1_orig != n2_orig:
    longer = 'folder_1' if n1_orig > n2_orig else 'folder_2'
    dropped = abs(n1_orig - n2_orig)
    print(f'Warning: {longer} has {dropped} extra frame(s) — dropping from the end to match ({n} frames used)')
    files_1 = files_1[:n]
    files_2 = files_2[:n]

  # Determine frame dimensions from the first available image
  sample = cv2.imread(str(files_1[0]))
  h, w = sample.shape[:2]
  scaled_w, scaled_h = int(w * args.scale), int(h * args.scale)

  if args.orientation == 'horizontal':
    bar_height = 50
    frame_w = scaled_w * 2
    frame_h = scaled_h + bar_height
    bar = make_text_bar(frame_w, bar_height, args.desc_1, args.desc_2)
  else:
    bar_height = 80
    frame_w = scaled_w
    frame_h = scaled_h * 2 + bar_height
    bar = make_separator_bar(frame_w, bar_height, args.desc_1, args.desc_2)

  if args.output:
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.basename(out_dir.rstrip('/'))
    extra_dirs = []
  else:
    out_dir, extra_dir, stem = make_run_dirs(args.folder_1, args.folder_2)
    extra_dirs = [extra_dir]

  video_name = f'{stem}.mp4'
  config_name = 'config.json'
  primary_video = os.path.join(out_dir, video_name)
  primary_config = os.path.join(out_dir, config_name)

  config = {
    'folder_1': os.path.abspath(args.folder_1),
    'folder_2': os.path.abspath(args.folder_2),
    'desc_1': args.desc_1,
    'desc_2': args.desc_2,
    'fps': args.fps,
    'scale': args.scale,
    'orientation': args.orientation,
    'n_frames_folder_1': n1_orig,
    'n_frames_folder_2': n2_orig,
    'n_frames_used': n,
    'frame_size': [frame_w, frame_h],
  }
  with open(primary_config, 'w') as f:
    json.dump(config, f, indent=2)
  print(f'Config: {primary_config}')

  fourcc = cv2.VideoWriter_fourcc(*'avc1')
  writer = cv2.VideoWriter(primary_video, fourcc, args.fps, (frame_w, frame_h))

  for f1, f2 in zip(files_1, files_2):
    img1 = cv2.imread(str(f1))
    img2 = cv2.imread(str(f2))
    frame = build_frame(img1, img2, args.scale, bar, args.orientation)
    writer.write(frame)

  writer.release()
  print(f'Video:  {primary_video}')

  for d in extra_dirs:
    shutil.copy2(primary_video, os.path.join(d, video_name))
    shutil.copy2(primary_config, os.path.join(d, config_name))
    print(f'Copied to: {d}/')


if __name__ == '__main__':
  main()

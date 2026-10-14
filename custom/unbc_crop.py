"""
Crop the UNBC-McMaster WarpedVideos to the face and pad them to a multiple of 16 frames.

The warped videos show the face on a white background. Each frame is cropped to the
bounding box of its non-white pixels, the remaining white pixels are set to black, and
every frame is resized to one size per video: the largest box of the video, rounded down
to even sides (the encoder needs even sizes). This reproduces the paper's
UNBC/video/WarpedVideos_Cropped up to video compression noise (mean absolute pixel
difference about 2.4 on 0-255). The padded copy prepends mirrored start frames until the
frame count is a multiple of 16 (UNBC/video/WarpedVideos_Cropped_interpolated_mirror).
"""

import os

import cv2
import numpy as np

import custom.tools as tools
from custom.faceExtractor import FaceExtractor

WHITE_THRESHOLD = 220  # grayscale value from which a pixel counts as background
FPS = 30
CHUNK = 16


def read_frames(video_path):
  """
  Read every frame of a video.

  Args:
    video_path: Path to the input video.

  Returns:
    list[np.ndarray]: BGR frames, each of shape (H, W, 3).
  """
  cap = cv2.VideoCapture(video_path)
  frames = []
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    frames.append(frame)
  cap.release()
  if not frames:
    raise ValueError(f'No frames read from {video_path}')
  return frames


def face_boxes(frames, white_threshold=WHITE_THRESHOLD):
  """
  Bounding box of the non-white pixels of each frame.

  Args:
    frames: BGR frames of one video.
    white_threshold: Grayscale value from which a pixel is background.

  Returns:
    list[tuple[int, int, int, int] | None]: (x1, y1, x2, y2) per frame, with x2/y2 the
    largest non-white column/row (the crop is frame[y1:y2, x1:x2]), or None for a frame
    without face pixels (some UNBC videos start with all-white frames).
  """
  boxes = []
  for frame in frames:
    ys, xs = np.where(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) < white_threshold)
    if len(xs) > 0 and xs.max() > xs.min() and ys.max() > ys.min():
      boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    else:
      boxes.append(None)
  return boxes


def crop_frames(frames, white_threshold=WHITE_THRESHOLD):
  """
  Crop each frame to its face box, blacken the background and resize to one size.

  Args:
    frames: BGR frames of one warped video.
    white_threshold: Grayscale value from which a pixel is background.

  Returns:
    list[np.ndarray]: BGR frames of shape (H, W, 3), with W and H the largest box width
    and height of the video rounded down to even values. Frames without face pixels are
    dropped.
  """
  kept = [(frame, box) for frame, box in zip(frames, face_boxes(frames, white_threshold))
          if box is not None]
  if not kept:
    raise ValueError('No frame contains face pixels')
  width = max(x2 - x1 for _, (x1, _, x2, _) in kept) // 2 * 2
  height = max(y2 - y1 for _, (_, y1, _, y2) in kept) // 2 * 2
  out = []
  for frame, (x1, y1, x2, y2) in kept:
    crop = frame[y1:y2, x1:x2].copy()
    crop[cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) >= white_threshold] = 0
    out.append(cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR))
  return out


def pad_frames(frames, chunk=CHUNK):
  """
  Prepend mirrored start frames until the frame count is a multiple of `chunk`.

  Args:
    frames: Frames of one video.
    chunk: Target multiple of the frame count.

  Returns:
    list[np.ndarray]: The padded frames (unchanged when already a multiple).
  """
  # apply_video_interpolation does not use the instance, so no landmark model is loaded.
  padded, _ = FaceExtractor.apply_video_interpolation(None, frames, chunk, mod='mirror_start_video')
  return padded


def crop_and_pad_video(video_path, cropped_path, padded_path):
  """
  Write the cropped and the cropped+padded copy of one warped video (both at 30 fps).

  Args:
    video_path: Input warped video.
    cropped_path: Output path of the cropped video.
    padded_path: Output path of the cropped video padded to a multiple of 16 frames.

  Returns:
    None.
  """
  cropped = crop_frames(read_frames(video_path))
  for path, frames in ((cropped_path, cropped), (padded_path, pad_frames(cropped))):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tools.generate_video_from_list_frame(list_frame=frames, path_video_output=path, fps=FPS,
                                         already_bgr=True)

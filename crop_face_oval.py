#!/usr/bin/env python3
"""
Standalone script: crop the face oval from a video (or a folder of videos)
using MediaPipe landmarks.

For each input video this detects facial landmarks with MediaPipe FaceMesh,
builds a mask from the FACEMESH_FACE_OVAL contour, blacks out everything outside
the oval, crops to the oval's bounding box, and writes a new video. Frames where
no face is detected are skipped.

The input may be a single video file or a folder searched recursively for .mp4
files; in folder mode the input subfolder tree is mirrored under the output
folder. By default the output keeps the source video's frame size; pass --size
to force square NxN frames.

Usage:
  python3 crop_face_oval.py input.mp4 -o output.mp4 --size 256
  python3 crop_face_oval.py MIntPAIN/video -o MIntPAIN/video_oval
"""

import argparse
import glob
import os

import cv2
import numpy as np
import mediapipe as mp


# Fractional margin added on each side of the full-range detector's face box before
# cropping the ROI that gets fed to FaceMesh (see --preprocess).
PREPROCESS_MARGIN = 0.6


def parse_args() -> argparse.Namespace:
  """
  Parse command-line arguments.

  Returns:
    Parsed arguments namespace with fields: input, output, size, padding,
    min_detection_confidence, min_tracking_confidence, preprocess.
  """
  p = argparse.ArgumentParser(description="Crop the face oval from a video using MediaPipe.")
  p.add_argument("input",
                 help="Path to an input video file, or a folder of .mp4 files "
                      "(searched recursively).")
  p.add_argument("-o", "--output", default=None,
                 help="Output path. For a file input: output video path "
                      "(default <input>_oval.mp4). For a folder input: output "
                      "folder (required); the input subfolder tree is mirrored.")
  p.add_argument("--size", type=int, default=None,
                 help="Side length (px) to force square output frames. "
                      "Default: keep the source video's original frame size.")
  p.add_argument("--padding", type=float, default=0.0,
                 help="Fractional margin added around the oval bounding box. Default: 0.0")
  p.add_argument("--min-detection-confidence", type=float, default=0.5,
                 help="MediaPipe min detection confidence. Default: 0.5")
  p.add_argument("--min-tracking-confidence", type=float, default=0.5,
                 help="MediaPipe min tracking confidence. Default: 0.5")
  p.add_argument("--preprocess", action="store_true",
                 help="Run a full-range face detector first to locate small/distant "
                      "faces, crop around them, then extract the oval on that region. "
                      "Use when FaceMesh alone skips frames (faces small in-frame).")
  return p.parse_args()


def oval_landmark_indices() -> list:
  """
  Compute the sorted unique landmark indices that define the face oval.

  Returns:
    List[int] of landmark indices from mediapipe FACEMESH_FACE_OVAL edges.
  """
  return sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})


def oval_crop(frame: np.ndarray, landmarks, oval_idx: list, padding: float):
  """
  Mask a frame to the face oval and crop to its bounding box.

  Args:
    frame:     BGR image. Shape: (H, W, 3).
    landmarks: MediaPipe NormalizedLandmarkList for one face.
    oval_idx:  Landmark indices tracing the face oval.
    padding:   Fractional margin added around the bounding box (0.0 = tight).

  Returns:
    Cropped BGR image with pixels outside the oval blacked out. Shape: (h, w, 3).
  """
  h, w = frame.shape[:2]
  pts = np.array(
    [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in oval_idx],
    dtype=np.int32,
  )
  pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
  pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

  hull = cv2.convexHull(pts)
  mask = np.zeros((h, w), dtype=np.uint8)
  cv2.fillConvexPoly(mask, hull, 255)
  masked = cv2.bitwise_and(frame, frame, mask=mask)

  x, y, bw, bh = cv2.boundingRect(hull)
  if padding > 0:
    px, py = int(bw * padding), int(bh * padding)
    x, y = max(0, x - px), max(0, y - py)
    bw, bh = min(w - x, bw + 2 * px), min(h - y, bh + 2 * py)

  return masked[y:y + bh, x:x + bw]


def detect_face_roi(frame: np.ndarray, face_detection, margin: float):
  """
  Locate a face with a full-range detector and crop a padded ROI around it.

  Used as a preprocessing step so FaceMesh (short-range only) can find faces that
  are small within the full frame. Since the oval is later built from whatever
  image FaceMesh sees, the returned ROI is fed directly to FaceMesh + oval_crop;
  no mapping back to full-frame coordinates is needed.

  Args:
    frame:          BGR image. Shape: (H, W, 3).
    face_detection: MediaPipe FaceDetection instance (expects full-range model).
    margin:         Fractional margin added on each side of the detected box.

  Returns:
    Cropped BGR ROI around the highest-confidence face. Shape: (h, w, 3).
    None if no face is detected.
  """
  h, w = frame.shape[:2]
  result = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  if not result.detections:
    return None
  box = result.detections[0].location_data.relative_bounding_box
  x0 = max(0, int((box.xmin - box.width * margin) * w))
  y0 = max(0, int((box.ymin - box.height * margin) * h))
  x1 = min(w, int((box.xmin + box.width * (1 + margin)) * w))
  y1 = min(h, int((box.ymin + box.height * (1 + margin)) * h))
  if x1 <= x0 or y1 <= y0:
    return None
  return frame[y0:y1, x0:x1]


def process_video(input_path: str, output_path: str, oval_idx: list, size,
                  padding: float, det_conf: float, trk_conf: float,
                  preprocess: bool = False) -> tuple:
  """
  Crop the face oval from every frame of one video and write the result.

  Args:
    input_path:  Path to the source video.
    output_path: Path to write the cropped video. Parent folders are created.
    oval_idx:    Landmark indices tracing the face oval (from oval_landmark_indices).
    size:        If an int, force square (size, size) output frames. If None,
                 keep the source video's original (width, height).
    padding:     Fractional margin added around the oval bounding box.
    det_conf:    MediaPipe min detection confidence.
    trk_conf:    MediaPipe min tracking confidence.
    preprocess:  If True, run a full-range face detector per frame and crop an ROI
                 around the face before FaceMesh (helps with small/distant faces).
                 Falls back to the full frame when the detector finds nothing.

  Returns:
    Tuple[int, int] of (written, skipped) frame counts.
  """
  cap = cv2.VideoCapture(input_path)
  if not cap.isOpened():
    raise SystemExit(f"Could not open video: {input_path}")
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  if fps <= 0:
    fps = 30.0

  if size is not None:
    out_w, out_h = size, size
  else:
    out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = None

  written, skipped = 0, 0
  # Full-range detector (stateless) to locate small faces before FaceMesh; None
  # when preprocessing is off.
  face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=det_conf) if preprocess else None
  # Fresh FaceMesh per video: its tracker holds state across frames and must
  # not leak between videos.
  with mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=det_conf,
    min_tracking_confidence=trk_conf,
  ) as face_mesh:
    while True:
      ok, frame = cap.read()
      if not ok:
        break
      # Fall back to the first decoded frame's shape if the capture props were 0.
      if out_w <= 0 or out_h <= 0:
        out_h, out_w = frame.shape[:2]
      if writer is None:
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
      # When preprocessing, run FaceMesh on the detected face ROI (falling back to
      # the full frame if the detector finds nothing); otherwise on the full frame.
      target = frame
      if face_detection is not None:
        roi = detect_face_roi(frame, face_detection, PREPROCESS_MARGIN)
        if roi is not None:
          target = roi
      result = face_mesh.process(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
      if not result.multi_face_landmarks:
        skipped += 1
        continue
      crop = oval_crop(target, result.multi_face_landmarks[0], oval_idx, padding)
      if crop.size == 0:
        skipped += 1
        continue
      writer.write(cv2.resize(crop, (out_w, out_h)))
      written += 1

  if face_detection is not None:
    face_detection.close()
  cap.release()
  if writer is not None:
    writer.release()
  return written, skipped


def main() -> None:
  """
  Run oval cropping over an input video or a folder of videos.

  Returns:
    None. Prints per-file and total counts of written vs skipped frames.
  """
  args = parse_args()
  oval_idx = oval_landmark_indices()

  if os.path.isdir(args.input):
    if not args.output:
      raise SystemExit("Folder input requires an output folder via -o/--output.")
    srcs = sorted(
      p for p in glob.glob(os.path.join(args.input, "**", "*.mp4"), recursive=True)
      if p.lower().endswith(".mp4")
    )
    if not srcs:
      raise SystemExit(f"No .mp4 files found under: {args.input}")

    total_written, total_skipped, empty_videos = 0, 0, 0
    for i, src in enumerate(srcs, start=1):
      rel = os.path.relpath(src, args.input)
      dst = os.path.join(args.output, rel)
      written, skipped = process_video(
        src, dst, oval_idx, args.size, args.padding,
        args.min_detection_confidence, args.min_tracking_confidence,
        args.preprocess,
      )
      if written == 0:
        empty_videos += 1
      total_written += written
      total_skipped += skipped
      print(f"[{i}/{len(srcs)}] {rel} | written {written} / skipped {skipped}")

    print(f"Done: {len(srcs)} videos -> {args.output} | "
          f"written {total_written} / skipped {total_skipped} frames | "
          f"{empty_videos} video(s) with no written frames")
  else:
    output = args.output or f"{os.path.splitext(args.input)[0]}_oval.mp4"
    written, skipped = process_video(
      args.input, output, oval_idx, args.size, args.padding,
      args.min_detection_confidence, args.min_tracking_confidence,
      args.preprocess,
    )
    print(f"Done: {output} | written {written} / skipped {skipped} frames")


if __name__ == "__main__":
  main()

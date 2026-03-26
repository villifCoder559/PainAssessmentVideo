"""
Key Frame Selector for BioVid Pain Prediction Pipeline
=======================================================
Adapted from Hossain & Muhammad (2019) "Emotion recognition using deep learning
approach from audio-visual emotional big data", Information Fusion 49, 69-78.

Changes vs. original paper:
- No Viola-Jones face detection (BioVid videos are already face-cropped)
- No video segmentation (whole video is processed at once)
- Three selection modes: MIN_INSTABILITY, MAX_INSTABILITY, HYBRID
- Selected key frames are forced to be a multiple of 16
- No frame can be selected more than once
- If the video is too short, all available frames are used
- Output is a new MP4 video composed of the selected key frames (RGB)

Selection modes
---------------
MIN_INSTABILITY  (original paper logic)
    Picks the most stable frame in each window — the one whose histogram
    changes least relative to its neighbours.  Good for capturing the
    settled apex of a clearly-held expression.  May miss brief, transient
    pain activations.

MAX_INSTABILITY
    Picks the most unstable frame in each window — the one with the highest
    chi-square distance from its neighbours.  Targets sudden facial changes
    (AU onsets/offsets) that are characteristic of spontaneous pain.

HYBRID
    Runs both passes independently on the full video and merges the results,
    allocating half the target count to MIN and half to MAX.  Gives the
    downstream model both a baseline resting-face context and the dynamic
    pain-relevant activations.  Recommended default for BioVid.

Usage
-----
  # Single video
  python3 key_frame_selector.py single \\
    --video_path data/sample.mp4 --output_path out/sample.mp4 --mode hybrid

  # Batch (whole dataset)
  python3 key_frame_selector.py batch \\
    --input_dir data/biovid/ --output_dir data/keyframes/ --mode max
"""

from __future__ import annotations

import argparse
import math
import cv2
import numpy as np
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Selection mode
# ---------------------------------------------------------------------------

class SelectionMode(str, Enum):
  """
  Strategy used to pick key frames from each sliding window.

  Attributes
  ----------
  MIN_INSTABILITY:
      Classic paper approach.  Selects the frame that changes the least
      within a window (lowest chi-square instability score).  Safe choice
      for posed, prolonged expressions.

  MAX_INSTABILITY:
      Pain-oriented approach.  Selects the frame that changes the most
      within a window (highest chi-square instability score).  Targets
      brief AU activations typical of spontaneous pain responses.

  HYBRID:
      Runs MIN and MAX passes independently, then merges them.  Half the
      final frames come from the MIN pool (baseline context) and half from
      the MAX pool (pain-relevant dynamics).  Recommended for BioVid.
  """
  MIN_INSTABILITY = "min"
  MAX_INSTABILITY = "max"
  HYBRID          = "hybrid"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _chi_square_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
  """
  Chi-square distance between two normalised histograms.

  d(h1, h2) = 0.5 * sum( (h1_i - h2_i)^2 / (h1_i + h2_i + eps) )

  Returns 0 for identical histograms and grows as they diverge.
  """
  eps = 1e-10
  return float(0.5 * np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps)))


def _frame_histogram(frame: np.ndarray, bins: int = 256) -> np.ndarray:
  """
  Compute a normalised grayscale histogram for a single BGR frame.
  Histograms are used only for comparing frames; the frames themselves
  are kept as RGB for the final output.
  """
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
  return cv2.normalize(hist, hist).flatten()


def _global_instability_scores(histograms: list[np.ndarray]) -> dict[int, float]:
  """
  Compute a global instability score for every frame using its two
  immediate neighbours (same formula as the paper, applied to the full
  video rather than a local window).

  Used to fill or trim the selected set after the windowed pass.
  """
  n = len(histograms)
  scores: dict[int, float] = {}
  for i in range(n):
    s = 0.0
    if i > 0:
      s += _chi_square_distance(histograms[i - 1], histograms[i])
    if i < n - 1:
      s += _chi_square_distance(histograms[i], histograms[i + 1])
    scores[i] = s
  return scores


# ---------------------------------------------------------------------------
# Single-mode windowed pass
# ---------------------------------------------------------------------------

def _windowed_pass(
  histograms: list[np.ndarray],
  window_half: int,
  step: int,
  min_temporal_gap: int,
  prefer_high: bool,
) -> tuple[set[int], dict[int, float]]:
  """
  One full sliding-window sweep over the video.

  Parameters
  ----------
  histograms       : pre-computed per-frame histograms
  window_half      : half the window size  ->  window = 2*window_half + 1
  step             : how many frames to advance the window each iteration
  min_temporal_gap : minimum frame distance between any two selected frames
  prefer_high      : if True  -> MAX mode (pick most unstable per window)
                     if False -> MIN mode (pick least unstable per window)

  Returns
  -------
  selected   : set of selected frame indices
  all_scores : dict mapping every visited frame to its local score
               (used later for fill / trim decisions)
  """
  total       = len(histograms)
  window_size = 2 * window_half + 1
  selected: set[int]           = set()
  all_scores: dict[int, float] = {}

  def _score_window(indices: list[int]) -> dict[int, float]:
    """Return local chi-square instability scores for all frames in the window."""
    scored: dict[int, float] = {}
    for j, idx in enumerate(indices):
      s = 0.0
      if j > 0:
        s += _chi_square_distance(histograms[indices[j - 1]], histograms[idx])
      if j < len(indices) - 1:
        s += _chi_square_distance(histograms[idx], histograms[indices[j + 1]])
      scored[idx] = s
      all_scores[idx] = max(all_scores.get(idx, 0.0), s)
    return scored

  def _pick(scored: dict[int, float]) -> None:
    """Add the best eligible candidate from a scored window to `selected`."""
    candidates = sorted(
      [
        i for i in scored
        if i not in selected
        and all(abs(i - s) >= min_temporal_gap for s in selected)
      ],
      key=lambda i: (-scored[i] if prefer_high else scored[i]),
    )
    if candidates:
      selected.add(candidates[0])

  pos = 0
  while pos + window_size <= total:
    _pick(_score_window(list(range(pos, pos + window_size))))
    pos += step

  # Leftover tail (shorter than a full window)
  if pos < total:
    tail = list(range(pos, total))
    if len(tail) >= 2:
      _pick(_score_window(tail))

  return selected, all_scores


# ---------------------------------------------------------------------------
# Fill / trim helpers  (mode-aware)
# ---------------------------------------------------------------------------

def _fill_to_target(
  selected: set[int],
  target: int,
  all_scores: dict[int, float],
  total_frames: int,
  prefer_high: bool,
  exclude: set[int] | None = None,
) -> set[int]:
  """
  Add frames from the unselected pool until `len(selected) == target`.

  The fill pool is sorted according to `prefer_high`:
  - MIN mode -> add the most stable   unselected frames first (low score)
  - MAX mode -> add the most unstable unselected frames first (high score)

  Parameters
  ----------
  exclude : optional set of frame indices to never add (used in HYBRID to
            prevent MAX fill from taking frames already owned by MIN pool)
  """
  needed = target - len(selected)
  if needed <= 0:
    return selected

  forbidden = selected | (exclude or set())
  unselected = sorted(
    [i for i in range(total_frames) if i not in forbidden],
    key=lambda i: (-all_scores.get(i, 0.0) if prefer_high else all_scores.get(i, 0.0)),
  )
  added = 0
  for idx in unselected:
    if added >= needed:
      break
    selected.add(idx)
    added += 1

  print(
    f"[KeyFrameSelector]   + added {added} supplementary frames "
    f"({'max' if prefer_high else 'min'} instability fill) -> {len(selected)} total."
  )
  return selected


def _trim_to_target(
  selected: set[int],
  target: int,
  all_scores: dict[int, float],
  prefer_high: bool,
) -> set[int]:
  """
  Remove frames from `selected` until `len(selected) == target`.

  Removes the frames *least consistent with the mode*:
  - MIN mode -> remove the most unstable frames (high score, least stable)
  - MAX mode -> remove the most stable frames   (low score, least dynamic)
  """
  excess = len(selected) - target
  if excess <= 0:
    return selected

  ordered = sorted(selected, key=lambda i: all_scores.get(i, 0.0))
  to_remove = ordered[-excess:] if not prefer_high else ordered[:excess]
  selected -= set(to_remove)

  print(
    f"[KeyFrameSelector]   - removed {excess} excess frames -> {len(selected)} total."
  )
  return selected


def _enforce_multiple_of_16(
  selected: set[int],
  all_scores: dict[int, float],
  total_frames: int,
  prefer_high: bool,
) -> list[int] | None:
  """
  Adjust `selected` to the nearest multiple of 16.

  Returns sorted list of indices, or None if the video cannot provide
  that many unique frames (caller should fall back to all frames).
  """
  n = len(selected)
  target = math.ceil(max(n, 1) / 16) * 16

  if target > total_frames:
    return None

  if n < target:
    selected = _fill_to_target(selected, target, all_scores, total_frames, prefer_high)
  elif n > target:
    selected = _trim_to_target(selected, target, all_scores, prefer_high)

  return sorted(selected)


# ---------------------------------------------------------------------------
# Hybrid pass  (internal)
# ---------------------------------------------------------------------------

def _hybrid_pass(
  histograms: list[np.ndarray],
  global_scores: dict[int, float],
  total: int,
  window_half: int,
  step: int,
  min_temporal_gap: int,
  hybrid_ratio: float,
) -> list[int] | None:
  """
  Run MIN and MAX windowed passes independently, then merge so that
  `hybrid_ratio` of the final frames come from MIN and
  `(1 - hybrid_ratio)` come from MAX.

  The two pools are kept strictly disjoint: a frame in the MIN pool
  is never duplicated in the MAX pool.

  After merging, the combined set is forced to the nearest multiple of 16.
  """
  if not 0 < hybrid_ratio < 1:
    raise ValueError(f"hybrid_ratio must be in (0, 1), got {hybrid_ratio!r}")

  # --- MIN pass --------------------------------------------------------
  min_selected, min_scores = _windowed_pass(
    histograms, window_half, step, min_temporal_gap, prefer_high=False
  )
  print(f"[KeyFrameSelector]   MIN pass -> {len(min_selected)} candidates")

  # --- MAX pass --------------------------------------------------------
  max_selected_raw, max_scores = _windowed_pass(
    histograms, window_half, step, min_temporal_gap, prefer_high=True
  )
  overlap      = max_selected_raw & min_selected
  max_selected = max_selected_raw - min_selected
  print(
    f"[KeyFrameSelector]   MAX pass -> {len(max_selected_raw)} candidates "
    f"({len(overlap)} overlap removed -> {len(max_selected)} unique MAX frames)"
  )

  all_scores = {**global_scores, **min_scores, **max_scores}

  # --- Determine final target and per-pool quotas ----------------------
  n_combined = len(min_selected) + len(max_selected)
  target     = math.ceil(max(n_combined, 1) / 16) * 16

  if target > total:
    return None

  n_min_target = round(target * hybrid_ratio)
  n_min_target = max(1, min(n_min_target, target - 1))
  n_max_target = target - n_min_target

  print(
    f"[KeyFrameSelector]   Hybrid target: {target} frames  "
    f"({n_min_target} MIN  +  {n_max_target} MAX)"
  )

  # --- Adjust MIN pool -------------------------------------------------
  if len(min_selected) < n_min_target:
    min_selected = _fill_to_target(
      min_selected, n_min_target, all_scores, total,
      prefer_high=False, exclude=max_selected,
    )
  elif len(min_selected) > n_min_target:
    min_selected = _trim_to_target(
      min_selected, n_min_target, all_scores, prefer_high=False
    )

  # --- Adjust MAX pool -------------------------------------------------
  if len(max_selected) < n_max_target:
    max_selected = _fill_to_target(
      max_selected, n_max_target, all_scores, total,
      prefer_high=True, exclude=min_selected,
    )
  elif len(max_selected) > n_max_target:
    max_selected = _trim_to_target(
      max_selected, n_max_target, all_scores, prefer_high=True
    )

  # Final safety: remove any accidental overlap (edge-case in fill)
  max_selected -= min_selected

  # Top up MAX if removing overlap created a shortfall
  remaining = set(range(total)) - min_selected - max_selected
  while len(max_selected) < n_max_target and remaining:
    best = max(remaining, key=lambda i: all_scores.get(i, 0.0))
    max_selected.add(best)
    remaining.discard(best)

  combined = min_selected | max_selected
  return sorted(combined)


# ---------------------------------------------------------------------------
# Public: select_key_frames
# ---------------------------------------------------------------------------

def select_key_frames(
  frames: list[np.ndarray],
  mode: SelectionMode = SelectionMode.HYBRID,
  window_half: int = 3,
  step: int = 4,
  min_temporal_gap: int = 2,
  hybrid_ratio: float = 0.5,
) -> list[int]:
  """
  Select key frames from a list of BGR frames using chi-square histogram
  distances inside a sliding window.

  Parameters
  ----------
  frames : list of np.ndarray
      All frames of the video (BGR, as returned by cv2).
  mode : SelectionMode
      Frame selection strategy.  Default: SelectionMode.HYBRID.
  window_half : int
      Half-size of the sliding window.  Default: 3  (window of 7).
  step : int
      How many frames to advance the window each iteration.  Default: 4.
  min_temporal_gap : int
      Minimum number of frames between any two selected frames.  Default: 2.
  hybrid_ratio : float
      Only used when mode == HYBRID.  Fraction of the final frame count
      drawn from the MIN pool.  Must be in (0, 1).  Default: 0.5.

  Returns
  -------
  list of int
      Sorted list of selected frame indices, guaranteed to be a multiple
      of 16 unless the video has fewer than 16 unique frames.
  """
  total = len(frames)

  if total < 16:
    print(
      f"[KeyFrameSelector] Video has only {total} frames — "
      "fewer than 16.  Returning all available frames."
    )
    return list(range(total))

  print(f"[KeyFrameSelector] Computing histograms for {total} frames ...")
  histograms    = [_frame_histogram(f) for f in frames]
  global_scores = _global_instability_scores(histograms)

  print(f"[KeyFrameSelector] Mode: {mode.value.upper()}")

  result: list[int] | None

  if mode == SelectionMode.MIN_INSTABILITY:
    selected, window_scores = _windowed_pass(
      histograms, window_half, step, min_temporal_gap, prefer_high=False
    )
    scores = {**global_scores, **window_scores}
    result = _enforce_multiple_of_16(selected, scores, total, prefer_high=False)

  elif mode == SelectionMode.MAX_INSTABILITY:
    selected, window_scores = _windowed_pass(
      histograms, window_half, step, min_temporal_gap, prefer_high=True
    )
    scores = {**global_scores, **window_scores}
    result = _enforce_multiple_of_16(selected, scores, total, prefer_high=True)

  elif mode == SelectionMode.HYBRID:
    result = _hybrid_pass(
      histograms=histograms,
      global_scores=global_scores,
      total=total,
      window_half=window_half,
      step=step,
      min_temporal_gap=min_temporal_gap,
      hybrid_ratio=hybrid_ratio,
    )

  else:
    raise ValueError(f"Unknown SelectionMode: {mode!r}")

  if result is None:
    print(
      f"[KeyFrameSelector] Cannot reach a multiple of 16 unique frames "
      f"(video has only {total} frames).  Returning all frames."
    )
    return list(range(total))

  assert len(result) % 16 == 0, f"Unexpected frame count: {len(result)}"
  print(f"[KeyFrameSelector] Done — {len(result)} key frames selected (multiple of 16).")
  return result


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video_frames(video_path: str | Path) -> tuple[list[np.ndarray], float]:
  """
  Load all frames from an MP4 file.

  Returns
  -------
  frames : list of np.ndarray  (BGR)
  fps    : float
  """
  cap = cv2.VideoCapture(str(video_path))
  if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

  fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  cap.release()

  if not frames:
    raise ValueError(f"No frames decoded from: {video_path}")
  return frames, fps


def write_video(
  frames: list[np.ndarray],
  indices: list[int],
  output_path: str | Path,
  fps: float = 25.0,
) -> None:
  """
  Write a new MP4 video containing only the frames at `indices`,
  in temporal order, using the original RGB colours.

  Parameters
  ----------
  frames      : full list of BGR frames from the original video
  indices     : sorted list of selected frame indices
  output_path : destination .mp4 file
  fps         : frame rate of the output video
  """
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  h, w   = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*"avc1")
  writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

  for idx in indices:
    writer.write(frames[idx])
  writer.release()

  print(
    f"[KeyFrameSelector] Output -> {output_path}  "
    f"({len(indices)} frames @ {fps:.2f} fps)"
  )


# ---------------------------------------------------------------------------
# Public pipeline entry-point
# ---------------------------------------------------------------------------

def process_video(
  video_path: str | Path,
  output_path: str | Path,
  mode: SelectionMode = SelectionMode.HYBRID,
  window_half: int = 3,
  step: int = 4,
  min_temporal_gap: int = 2,
  hybrid_ratio: float = 0.5,
  output_fps: float | None = None,
) -> list[int]:
  """
  Full pipeline: load -> select key frames -> write output video.

  Parameters
  ----------
  video_path       : path to the input BioVid MP4
  output_path      : path for the output key-frame MP4
  mode             : SelectionMode  (MIN_INSTABILITY | MAX_INSTABILITY | HYBRID)
  window_half      : half-size of the sliding window (default 3 -> window of 7)
  step             : window shift in frames (default 4)
  min_temporal_gap : minimum distance between two selected frames (default 2)
  hybrid_ratio     : MIN fraction for HYBRID mode, in (0, 1) (default 0.5)
  output_fps       : FPS of the output video; if None, inherits from input

  Returns
  -------
  list of int : selected frame indices (sorted, multiple of 16)
  """
  video_path = Path(video_path)
  print(f"\n{'='*60}")
  print(f"Processing : {video_path.name}")
  print(f"{'='*60}")

  frames, src_fps = load_video_frames(video_path)
  print(f"  Total frames : {len(frames)}  |  Source FPS : {src_fps:.2f}")

  key_indices = select_key_frames(
    frames,
    mode=mode,
    window_half=window_half,
    step=step,
    min_temporal_gap=min_temporal_gap,
    hybrid_ratio=hybrid_ratio,
  )

  write_video(
    frames=frames,
    indices=key_indices,
    output_path=output_path,
    fps=output_fps if output_fps is not None else src_fps,
  )

  return key_indices


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------

def process_dataset(
  input_dir: str | Path,
  output_dir: str | Path,
  pattern: str = "**/*.mp4",
  **kwargs,
) -> dict[str, list[int]]:
  """
  Process all MP4 files found under `input_dir` and save key-frame videos
  to the mirrored structure under `output_dir`.

  Parameters
  ----------
  input_dir  : root folder of the BioVid dataset
  output_dir : root folder for outputs
  pattern    : glob pattern to find videos  (default: "**/*.mp4")
  **kwargs   : forwarded verbatim to process_video()

  Returns
  -------
  dict mapping str(input_path) -> selected frame indices
  """
  input_dir  = Path(input_dir)
  output_dir = Path(output_dir)
  results: dict[str, list[int]] = {}

  video_paths = sorted(input_dir.glob(pattern))
  print(f"Found {len(video_paths)} video(s) under {input_dir}")

  for vp in video_paths:
    relative = vp.relative_to(input_dir)
    out_path = output_dir / relative
    try:
      indices = process_video(vp, out_path, **kwargs)
      results[str(vp)] = indices
    except Exception as exc:
      print(f"[ERROR] {vp.name}: {exc}")

  print(f"\nDone.  Processed {len(results)}/{len(video_paths)} video(s).")
  return results


# ---------------------------------------------------------------------------
# Argparse CLI
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
  """
  Add the shared algorithm arguments to a subparser.

  Args:
    parser: The argparse subparser to add arguments to.
  """
  parser.add_argument(
    "--mode", type=str, default="hybrid",
    choices=["min", "max", "hybrid"],
    help="Selection mode: min (most stable), max (most unstable), "
         "hybrid (both). Default: hybrid.",
  )
  parser.add_argument(
    "--window_half", type=int, default=3,
    help="Half-size of the sliding window (window = 2*W+1). Default: 3.",
  )
  parser.add_argument(
    "--step", type=int, default=4,
    help="Window advance in frames. Default: 4.",
  )
  parser.add_argument(
    "--min_temporal_gap", type=int, default=2,
    help="Minimum frame distance between selected frames. Default: 2.",
  )
  parser.add_argument(
    "--hybrid_ratio", type=float, default=0.5,
    help="Fraction of frames from MIN pool in hybrid mode (0-1). Default: 0.5.",
  )
  parser.add_argument(
    "--output_fps", type=float, default=None,
    help="FPS of the output video. Default: inherit from source.",
  )


def parse_args() -> argparse.Namespace:
  """
  Parse command-line arguments.

  Returns
  -------
  argparse.Namespace with the parsed arguments.
  """
  parser = argparse.ArgumentParser(
    description="Key Frame Selector for BioVid Pain Prediction Pipeline",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  # -- single video -----------------------------------------------------
  sp_single = subparsers.add_parser(
    "single", help="Process a single video file.",
  )
  sp_single.add_argument(
    "--video_path", type=str, required=True,
    help="Path to the input MP4 video.",
  )
  sp_single.add_argument(
    "--output_path", type=str, required=True,
    help="Path for the output key-frame MP4.",
  )
  _add_common_args(sp_single)

  # -- batch -------------------------------------------------------------
  sp_batch = subparsers.add_parser(
    "batch", help="Process all videos in a directory.",
  )
  sp_batch.add_argument(
    "--input_dir", type=str, required=True,
    help="Root folder containing input videos.",
  )
  sp_batch.add_argument(
    "--output_dir", type=str, required=True,
    help="Root folder for output key-frame videos.",
  )
  sp_batch.add_argument(
    "--pattern", type=str, default="**/*.mp4",
    help='Glob pattern to find videos. Default: "**/*.mp4".',
  )
  _add_common_args(sp_batch)

  return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  args = parse_args()
  mode = SelectionMode(args.mode)

  common_kwargs = dict(
    mode=mode,
    window_half=args.window_half,
    step=args.step,
    min_temporal_gap=args.min_temporal_gap,
    hybrid_ratio=args.hybrid_ratio,
    output_fps=args.output_fps,
  )

  if args.command == "single":
    indices = process_video(
      video_path=args.video_path,
      output_path=args.output_path,
      **common_kwargs,
    )
    print(f"\nKey frame indices: {indices}")

  elif args.command == "batch":
    # output_fps is not a kwarg of process_dataset -> pass via common_kwargs
    results = process_dataset(
      input_dir=args.input_dir,
      output_dir=args.output_dir,
      pattern=args.pattern,
      **common_kwargs,
    )

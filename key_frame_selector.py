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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
  window_log: list[dict] | None = None,
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
  window_log       : if not None, a mutable list to which a dict is appended
                     per window: {"indices", "scores", "picked"}.  Default None.

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

  def _pick(scored: dict[int, float], indices: list[int]) -> None:
    """Add the best eligible candidate from a scored window to `selected`."""
    candidates = sorted(
      [
        i for i in scored
        if i not in selected
        and all(abs(i - s) >= min_temporal_gap for s in selected)
      ],
      key=lambda i: (-scored[i] if prefer_high else scored[i]),
    )
    picked = candidates[0] if candidates else None
    if picked is not None:
      selected.add(picked)
    if window_log is not None:
      window_log.append({
        "indices": list(indices),
        "scores": dict(scored),
        "picked": picked,
      })

  pos = 0
  while pos + window_size <= total:
    indices = list(range(pos, pos + window_size))
    _pick(_score_window(indices), indices)
    pos += step

  # Leftover tail (shorter than a full window)
  if pos < total:
    tail = list(range(pos, total))
    if len(tail) >= 2:
      _pick(_score_window(tail), tail)

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
  window_log: dict | None = None,
) -> list[int] | None:
  """
  Run MIN and MAX windowed passes independently, then merge so that
  `hybrid_ratio` of the final frames come from MIN and
  `(1 - hybrid_ratio)` come from MAX.

  The two pools are kept strictly disjoint: a frame in the MIN pool
  is never duplicated in the MAX pool.

  After merging, the combined set is forced to the nearest multiple of 16.

  Args:
    histograms:       Pre-computed per-frame histograms.
    global_scores:    Global instability scores for all frames.
    total:            Total number of frames.
    window_half:      Half the window size.
    step:             Window advance in frames.
    min_temporal_gap: Minimum distance between selected frames.
    hybrid_ratio:     Fraction of final frames from MIN pool.
    window_log:       If not None, a mutable dict to populate with per-window
                      data from both passes and final selected sets.

  Returns:
    Sorted list of selected frame indices, or None if target exceeds total.
  """
  if not 0 < hybrid_ratio < 1:
    raise ValueError(f"hybrid_ratio must be in (0, 1), got {hybrid_ratio!r}")

  # --- MIN pass --------------------------------------------------------
  min_wlog = [] if window_log is not None else None
  min_selected, min_scores = _windowed_pass(
    histograms, window_half, step, min_temporal_gap, prefer_high=False,
    window_log=min_wlog,
  )
  print(f"[KeyFrameSelector]   MIN pass -> {len(min_selected)} candidates")

  # --- MAX pass --------------------------------------------------------
  max_wlog = [] if window_log is not None else None
  max_selected_raw, max_scores = _windowed_pass(
    histograms, window_half, step, min_temporal_gap, prefer_high=True,
    window_log=max_wlog,
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

  if window_log is not None:
    window_log["min_windows"] = min_wlog
    window_log["max_windows"] = max_wlog
    window_log["min_selected"] = set(min_selected)
    window_log["max_selected"] = set(max_selected)
    window_log["all_scores"] = all_scores
    window_log["global_scores"] = global_scores

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
  window_log: dict | None = None,
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
  window_log : dict or None
      If not None, a mutable dict that will be populated with per-window
      data for diagnostic logging.  Keys: min_windows, max_windows,
      min_selected, max_selected, all_scores, global_scores.

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
    if window_log is not None:
      histograms = [_frame_histogram(f) for f in frames]
      global_scores = _global_instability_scores(histograms)
      window_log["min_windows"] = None
      window_log["max_windows"] = None
      window_log["min_selected"] = set()
      window_log["max_selected"] = set()
      window_log["all_scores"] = global_scores
      window_log["global_scores"] = global_scores
    return list(range(total))

  print(f"[KeyFrameSelector] Computing histograms for {total} frames ...")
  histograms    = [_frame_histogram(f) for f in frames]
  global_scores = _global_instability_scores(histograms)

  print(f"[KeyFrameSelector] Mode: {mode.value.upper()}")

  result: list[int] | None

  if mode == SelectionMode.MIN_INSTABILITY:
    min_wlog = [] if window_log is not None else None
    selected, window_scores = _windowed_pass(
      histograms, window_half, step, min_temporal_gap, prefer_high=False,
      window_log=min_wlog,
    )
    scores = {**global_scores, **window_scores}
    result = _enforce_multiple_of_16(selected, scores, total, prefer_high=False)
    if window_log is not None:
      window_log["min_windows"] = min_wlog
      window_log["max_windows"] = None
      window_log["min_selected"] = set(selected)
      window_log["max_selected"] = set()
      window_log["all_scores"] = scores
      window_log["global_scores"] = global_scores

  elif mode == SelectionMode.MAX_INSTABILITY:
    max_wlog = [] if window_log is not None else None
    selected, window_scores = _windowed_pass(
      histograms, window_half, step, min_temporal_gap, prefer_high=True,
      window_log=max_wlog,
    )
    scores = {**global_scores, **window_scores}
    result = _enforce_multiple_of_16(selected, scores, total, prefer_high=True)
    if window_log is not None:
      window_log["min_windows"] = None
      window_log["max_windows"] = max_wlog
      window_log["min_selected"] = set()
      window_log["max_selected"] = set(selected)
      window_log["all_scores"] = scores
      window_log["global_scores"] = global_scores

  elif mode == SelectionMode.HYBRID:
    result = _hybrid_pass(
      histograms=histograms,
      global_scores=global_scores,
      total=total,
      window_half=window_half,
      step=step,
      min_temporal_gap=min_temporal_gap,
      hybrid_ratio=hybrid_ratio,
      window_log=window_log,
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
# Diagnostic plot helpers
# ---------------------------------------------------------------------------

_LOG_DPI = 100
_COLOR_MIN_SELECTED = "#E53935"
_COLOR_MAX_SELECTED = "#1E88E5"
_COLOR_UNSELECTED   = "#D0D0D0"
_TITLE_H_IN         = 0.6
_LABEL_H_IN         = 0.35
_TEXT_H_IN          = 0.30
_WINDOW_GAP_IN      = 0.25
_BORDER_WIDTH_PT    = 3.0


def _build_plot_windows(
  window_log: dict,
  mode: SelectionMode,
  total_frames: int,
) -> list[dict]:
  """
  Merge per-pass window lists into a single ordered list of window
  descriptors suitable for plotting.

  Args:
    window_log:   Populated window log dict from select_key_frames.
    mode:         Selection mode used.
    total_frames: Total number of frames in the video.

  Returns:
    Ordered list of dicts, each with keys: label, indices, scores,
    selected_min, selected_max.  The last entry may be an "Unassigned"
    section for frames not covered by any window.
  """
  min_windows  = window_log.get("min_windows") or []
  max_windows  = window_log.get("max_windows") or []
  min_selected = window_log.get("min_selected", set())
  max_selected = window_log.get("max_selected", set())
  global_scores = window_log.get("global_scores", {})

  windows: list[dict] = []
  covered: set[int] = set()

  if mode == SelectionMode.HYBRID:
    # MIN and MAX passes sweep the same ranges — pair them up.
    n = max(len(min_windows), len(max_windows))
    for i in range(n):
      mw = min_windows[i] if i < len(min_windows) else None
      xw = max_windows[i] if i < len(max_windows) else None
      indices = (mw or xw)["indices"]
      # merge scores from both passes for this window
      scores: dict[int, float] = {}
      if mw:
        scores.update(mw["scores"])
      if xw:
        for k, v in xw["scores"].items():
          scores[k] = max(scores.get(k, 0.0), v)
      first, last = indices[0], indices[-1]
      # Highlight only the frame picked from THIS window that survived to
      # the final selection — at most one red and one blue per window.
      min_pick = mw["picked"] if mw else None
      max_pick = xw["picked"] if xw else None
      win_sel_min = {min_pick} if min_pick is not None and min_pick in min_selected else set()
      win_sel_max = {max_pick} if max_pick is not None and max_pick in max_selected else set()
      windows.append({
        "label": f"Window {i + 1} [frames {first}-{last}]",
        "indices": indices,
        "scores": scores,
        "selected_min": win_sel_min,
        "selected_max": win_sel_max,
      })
      covered.update(indices)
  else:
    src = min_windows if mode == SelectionMode.MIN_INSTABILITY else max_windows
    for i, w in enumerate(src):
      first, last = w["indices"][0], w["indices"][-1]
      pick = w["picked"]
      if mode == SelectionMode.MIN_INSTABILITY:
        win_sel_min = {pick} if pick is not None and pick in min_selected else set()
        win_sel_max: set[int] = set()
      else:
        win_sel_min = set()
        win_sel_max = {pick} if pick is not None and pick in max_selected else set()
      windows.append({
        "label": f"Window {i + 1} [frames {first}-{last}]",
        "indices": w["indices"],
        "scores": w["scores"],
        "selected_min": win_sel_min,
        "selected_max": win_sel_max,
      })
      covered.update(w["indices"])

  # Unassigned frames — use global sets intersected with these indices,
  # so fill/trim frames still show their selection color.
  unassigned = sorted(set(range(total_frames)) - covered)
  if unassigned:
    unassigned_set = set(unassigned)
    windows.append({
      "label": "Unassigned frames",
      "indices": unassigned,
      "scores": {i: global_scores.get(i, 0.0) for i in unassigned},
      "selected_min": min_selected & unassigned_set,
      "selected_max": max_selected & unassigned_set,
    })

  return windows


def _render_window_plot(
  windows: list[dict],
  frames: list[np.ndarray],
  title: str,
  log_columns: int,
  log_frame_size: int,
  mode: SelectionMode,
) -> plt.Figure:
  """
  Render a diagnostic plot showing frame thumbnails in a grid for each
  window, with scores annotated and selection borders colored.

  Args:
    windows:        List of window descriptors to include in this plot.
    frames:         Full list of BGR video frames.
    title:          Title string for the top of the plot.
    log_columns:    Number of frame thumbnails per row.
    log_frame_size: Width in pixels of each thumbnail.
    mode:           Selection mode (controls border colors).

  Returns:
    A matplotlib Figure ready to be saved.
  """
  # --- compute thumbnail dimensions ---
  h_orig, w_orig = frames[0].shape[:2]
  aspect = h_orig / max(w_orig, 1)
  thumb_w = log_frame_size
  thumb_h = int(round(thumb_w * aspect))
  thumb_w_in = thumb_w / _LOG_DPI
  thumb_h_in = thumb_h / _LOG_DPI

  # --- figure dimensions ---
  fig_w_in = log_columns * thumb_w_in + 1.2  # left margin for labels
  total_h_in = _TITLE_H_IN
  for w in windows:
    n_rows = math.ceil(len(w["indices"]) / log_columns)
    total_h_in += _LABEL_H_IN + n_rows * (thumb_h_in + _TEXT_H_IN) + _WINDOW_GAP_IN

  fig = plt.figure(figsize=(fig_w_in, total_h_in), dpi=_LOG_DPI)
  fig.suptitle(title, fontsize=8, fontweight="bold", y=1 - _TITLE_H_IN / (2 * total_h_in))

  left_margin_in = 1.2
  cursor_y_in = total_h_in - _TITLE_H_IN  # top, below title

  for w in windows:
    # window label
    cursor_y_in -= _LABEL_H_IN
    fig.text(
      0.02, cursor_y_in / total_h_in + _LABEL_H_IN / (2 * total_h_in),
      w["label"], fontsize=7, fontweight="bold", va="center",
    )

    indices = w["indices"]
    scores  = w["scores"]
    sel_min = w["selected_min"]
    sel_max = w["selected_max"]

    for j, idx in enumerate(indices):
      col = j % log_columns
      row = j // log_columns
      x_in = left_margin_in + col * thumb_w_in
      y_in = cursor_y_in - (row + 1) * (thumb_h_in + _TEXT_H_IN)

      # axes position in figure fraction
      ax = fig.add_axes([
        x_in / fig_w_in,
        y_in / total_h_in,
        thumb_w_in / fig_w_in,
        thumb_h_in / total_h_in,
      ])

      # thumbnail
      resized = cv2.resize(frames[idx], (thumb_w, thumb_h))
      rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
      ax.imshow(rgb, aspect="auto")
      ax.set_xticks([])
      ax.set_yticks([])

      # border color
      if idx in sel_min:
        color = _COLOR_MIN_SELECTED
      elif idx in sel_max:
        color = _COLOR_MAX_SELECTED
      else:
        color = _COLOR_UNSELECTED

      for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(_BORDER_WIDTH_PT)

      # score text below thumbnail
      score_val = scores.get(idx, 0.0)
      fig.text(
        (x_in + thumb_w_in / 2) / fig_w_in,
        (y_in - 0.02) / total_h_in,
        f"F{idx}: {score_val:.4f}",
        fontsize=5, ha="center", va="top", color="black",
      )

    n_rows = math.ceil(len(indices) / log_columns)
    cursor_y_in -= n_rows * (thumb_h_in + _TEXT_H_IN) + _WINDOW_GAP_IN

  return fig


def _estimate_window_height(
  n_frames: int,
  log_columns: int,
  thumb_h_in: float,
) -> float:
  """
  Estimate the vertical space (inches) a single window section occupies.

  Args:
    n_frames:    Number of frames in the window.
    log_columns: Frames per row.
    thumb_h_in:  Thumbnail height in inches.

  Returns:
    Estimated height in inches.
  """
  n_rows = math.ceil(n_frames / log_columns)
  return _LABEL_H_IN + n_rows * (thumb_h_in + _TEXT_H_IN) + _WINDOW_GAP_IN


def _generate_log_plots(
  frames: list[np.ndarray],
  window_log: dict,
  video_name: str,
  mode: SelectionMode,
  window_half: int,
  step: int,
  min_temporal_gap: int,
  hybrid_ratio: float,
  log_root_folder: str | Path,
  log_columns: int = 7,
  log_max_height: int = 5000,
  log_frame_size: int = 150,
) -> None:
  """
  Generate diagnostic PNG plots showing per-window frame selection.

  Creates a subfolder under log_root_folder named with the video name
  and all parameters, then saves sequentially numbered PNGs.

  Args:
    frames:            Full list of BGR video frames.
    window_log:        Populated window log dict from select_key_frames.
    video_name:        Name of the video (without extension).
    mode:              Selection mode used.
    window_half:       Half window size used.
    step:              Step used.
    min_temporal_gap:  Temporal gap used.
    hybrid_ratio:      Hybrid ratio used.
    log_root_folder:   Root directory for log output.
    log_columns:       Frames per row in the grid.  Default: 7.
    log_max_height:    Max pixel height per PNG.  Default: 5000.
    log_frame_size:    Pixel width of each frame thumbnail.  Default: 150.
  """
  total = len(frames)

  # --- output folder ---
  folder_name = (
    f"{video_name}_mode_{mode.value}_window_half_{window_half}"
    f"_step_{step}_min_temporal_gap_{min_temporal_gap}"
    f"_hybrid_ratio_{hybrid_ratio}"
  )
  log_folder = Path(log_root_folder) / folder_name
  log_folder.mkdir(parents=True, exist_ok=True)

  # --- title string ---
  title = (
    f"{video_name}  |  mode={mode.value}  window_half={window_half}  "
    f"step={step}  min_temporal_gap={min_temporal_gap}  "
    f"hybrid_ratio={hybrid_ratio}"
  )

  # --- build window list ---
  all_windows = _build_plot_windows(window_log, mode, total)

  # --- compute thumbnail aspect ---
  h_orig, w_orig = frames[0].shape[:2]
  aspect = h_orig / max(w_orig, 1)
  thumb_h_in = int(round(log_frame_size * aspect)) / _LOG_DPI

  max_h_in = log_max_height / _LOG_DPI

  # --- pack windows into PNGs ---
  batch: list[dict] = []
  batch_h = _TITLE_H_IN
  seq = 1

  def _flush() -> None:
    """
    Render and save the current batch of windows as a single PNG.
    """
    nonlocal seq, batch, batch_h
    if not batch:
      return
    first_frame = batch[0]["indices"][0]
    last_frame  = batch[-1]["indices"][-1]
    fig = _render_window_plot(batch, frames, title, log_columns, log_frame_size, mode)
    fname = f"{seq:03d}_start_{first_frame}_end_{last_frame}.png"
    fig.savefig(str(log_folder / fname), dpi=_LOG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[KeyFrameSelector]   Saved {fname}")
    seq += 1
    batch = []
    batch_h = _TITLE_H_IN

  for w in all_windows:
    wh = _estimate_window_height(len(w["indices"]), log_columns, thumb_h_in)
    if batch and (batch_h + wh) > max_h_in:
      _flush()
    batch.append(w)
    batch_h += wh

  _flush()

  print(f"[KeyFrameSelector] Log plots saved to {log_folder}  ({seq - 1} file(s))")


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
  log_frames: bool = False,
  log_root_folder: str | Path = "logs",
  log_columns: int = 7,
  log_max_height: int = 5000,
  log_frame_size: int = 150,
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
  log_frames       : if True, generate diagnostic PNG plots.  Default: False.
  log_root_folder  : root directory for diagnostic output.  Default: "logs".
  log_columns      : frames per row in diagnostic grid.  Default: 7.
  log_max_height   : max pixel height per diagnostic PNG.  Default: 5000.
  log_frame_size   : thumbnail width in pixels.  Default: 150.

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

  wlog = {} if log_frames else None

  key_indices = select_key_frames(
    frames,
    mode=mode,
    window_half=window_half,
    step=step,
    min_temporal_gap=min_temporal_gap,
    hybrid_ratio=hybrid_ratio,
    window_log=wlog,
  )

  write_video(
    frames=frames,
    indices=key_indices,
    output_path=output_path,
    fps=output_fps if output_fps is not None else src_fps,
  )

  if log_frames and wlog:
    _generate_log_plots(
      frames=frames,
      window_log=wlog,
      video_name=video_path.stem,
      mode=mode,
      window_half=window_half,
      step=step,
      min_temporal_gap=min_temporal_gap,
      hybrid_ratio=hybrid_ratio,
      log_root_folder=log_root_folder,
      log_columns=log_columns,
      log_max_height=log_max_height,
      log_frame_size=log_frame_size,
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
  # -- logging / debugging ------------------------------------------------
  parser.add_argument(
    "--log_frames", action="store_true", default=False,
    help="Generate diagnostic plots showing per-window frame selection.",
  )
  parser.add_argument(
    "--log_root_folder", type=str, default="logs",
    help="Root directory for log output. Default: 'logs'.",
  )
  parser.add_argument(
    "--log_columns", type=int, default=7,
    help="Number of frame thumbnails per row in the diagnostic grid. Default: 7.",
  )
  parser.add_argument(
    "--log_max_height", type=int, default=5000,
    help="Maximum pixel height of each diagnostic PNG. Default: 5000.",
  )
  parser.add_argument(
    "--log_frame_size", type=int, default=150,
    help="Width in pixels of each frame thumbnail. Default: 150.",
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
    log_frames=args.log_frames,
    log_root_folder=args.log_root_folder,
    log_columns=args.log_columns,
    log_max_height=args.log_max_height,
    log_frame_size=args.log_frame_size,
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
      **common_kwargs)

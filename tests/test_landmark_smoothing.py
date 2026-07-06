"""
Tests for the temporal anti-jitter utilities: smooth_boxes (stabilized ROI / output
crop boxes) and LandmarkSmoother (savgol and kalman methods on (F, N, 3) landmarks).

Run with: pytest tests/test_landmark_smoothing.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.faceExtractor import LandmarkSmoother, smooth_boxes


FRAME_SHAPE = (480, 640)  # (H, W)


def make_moving_boxes(n, jitter=0, seed=0):
  """
  Build n boxes of constant size drifting linearly to the right, with optional jitter.

  Args:
    n:      Number of frames.
    jitter: Max absolute uniform noise (px) added independently to each corner.
    seed:   RNG seed.

  Returns:
    List of (x0, y0, x1, y1) int tuples inside FRAME_SHAPE.
  """
  rng = np.random.default_rng(seed)
  boxes = []
  for i in range(n):
    x0, y0 = 100 + i, 150
    box = np.array([x0, y0, x0 + 120, y0 + 160], dtype=np.float64)
    if jitter:
      box += rng.uniform(-jitter, jitter, size=4)
    boxes.append(tuple(box.astype(int)))
  return boxes


class TestSmoothBoxes:
  def test_constant_size_and_in_bounds(self):
    """All returned boxes share one size and stay inside the frame."""
    boxes = make_moving_boxes(60, jitter=8)
    smoothed = smooth_boxes(boxes, FRAME_SHAPE)
    h, w = FRAME_SHAPE
    sizes = {(x1 - x0, y1 - y0) for x0, y0, x1, y1 in smoothed}
    assert len(sizes) == 1
    for x0, y0, x1, y1 in smoothed:
      assert 0 <= x0 < x1 <= w
      assert 0 <= y0 < y1 <= h

  def test_reduces_center_jitter(self):
    """Smoothed centers move less frame-to-frame than jittery raw centers."""
    boxes = make_moving_boxes(60, jitter=8)
    smoothed = smooth_boxes(boxes, FRAME_SHAPE)

    def center_steps(bs):
      c = np.array([[(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0] for b in bs])
      return np.abs(np.diff(c, axis=0)).mean()

    assert center_steps(smoothed) < center_steps(boxes)

  def test_gaps_are_interpolated(self):
    """None entries are filled from neighbors instead of falling back to full frame."""
    boxes = make_moving_boxes(30)
    boxes[10] = None
    boxes[11] = None
    smoothed = smooth_boxes(boxes, FRAME_SHAPE)
    assert all(b is not None for b in smoothed)
    # gap centers should sit between the surrounding detections, not at the frame center
    x0, _, x1, _ = smoothed[10]
    assert 100 < (x0 + x1) / 2.0 < 100 + 30 + 120

  def test_all_none_returns_all_none(self):
    assert smooth_boxes([None] * 5, FRAME_SHAPE) == [None] * 5

  def test_empty_input(self):
    assert smooth_boxes([], FRAME_SHAPE) == []

  def test_short_sequence_no_crash(self):
    """Sequences shorter than the savgol window still get constant-size boxes."""
    boxes = make_moving_boxes(3, jitter=4)
    smoothed = smooth_boxes(boxes, FRAME_SHAPE)
    assert len(smoothed) == 3
    sizes = {(x1 - x0, y1 - y0) for x0, y0, x1, y1 in smoothed}
    assert len(sizes) == 1

  def test_boxes_tracking_moving_face(self):
    """The smoothed boxes follow the drifting face (no fixed crop losing it)."""
    boxes = make_moving_boxes(100)
    smoothed = smooth_boxes(boxes, FRAME_SHAPE)
    first_cx = (smoothed[0][0] + smoothed[0][2]) / 2.0
    last_cx = (smoothed[-1][0] + smoothed[-1][2]) / 2.0
    assert last_cx - first_cx > 80  # drifted ~99 px overall


def make_landmarks(num_frames=40, num_points=478, noise=0.005, seed=0):
  """
  Build noisy sinusoidal landmark trajectories.

  Args:
    num_frames: F.
    num_points: N.
    noise:      Std of gaussian noise added to the clean signal.
    seed:       RNG seed.

  Returns:
    (noisy, clean) arrays, both of shape (F, N, 3), values roughly in [0, 1].
  """
  rng = np.random.default_rng(seed)
  t = np.linspace(0, 2 * np.pi, num_frames)[:, None, None]
  base = rng.uniform(0.2, 0.8, size=(1, num_points, 3))
  clean = base + 0.05 * np.sin(t)
  noisy = clean + rng.normal(0, noise, size=clean.shape)
  return noisy, clean


class TestLandmarkSmoother:
  def test_savgol_preserves_shape(self):
    noisy, _ = make_landmarks()
    out = LandmarkSmoother(method='savgol', window_size=5).smooth(noisy)
    assert out.shape == noisy.shape

  def test_savgol_reduces_noise(self):
    noisy, clean = make_landmarks()
    out = LandmarkSmoother(method='savgol', window_size=5).smooth(noisy)
    err_before = np.abs(noisy - clean).mean()
    err_after = np.abs(out - clean).mean()
    assert err_after < err_before

  def test_savgol_short_video_returned_unchanged(self):
    noisy, _ = make_landmarks(num_frames=3)
    out = LandmarkSmoother(method='savgol', window_size=5).smooth(noisy)
    np.testing.assert_array_equal(out, noisy)

  def test_savgol_window_larger_than_video(self):
    noisy, _ = make_landmarks(num_frames=7)
    out = LandmarkSmoother(method='savgol', window_size=15).smooth(noisy)
    assert out.shape == noisy.shape

  def test_kalman_handles_3d_landmarks(self):
    """Kalman previously crashed assigning 2 values into a length-3 slot."""
    noisy, _ = make_landmarks(num_frames=10, num_points=20)
    out = LandmarkSmoother(method='kalman').smooth(noisy)
    assert out.shape == noisy.shape
    # z passes through unfiltered
    np.testing.assert_array_equal(out[..., 2], noisy[..., 2].astype(out.dtype))

  def test_invalid_method_raises(self):
    with pytest.raises(ValueError):
      LandmarkSmoother(method='bogus').smooth(np.zeros((10, 5, 3)))

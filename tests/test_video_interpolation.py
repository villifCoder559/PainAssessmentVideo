"""
Tests for FaceExtractor.apply_video_interpolation (video frame padding to a chunk multiple).

Covers mirror_start_video padding (including videos shorter than the required padding),
spread_linearly padding, no-op behavior on exact multiples, and invalid-mode errors.

Run with: pytest tests/test_video_interpolation.py -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.faceExtractor import FaceExtractor


def make_frames(n, h=4, w=4):
  """
  Build n distinct uint8 frames.

  Args:
    n: Number of frames.
    h: Frame height.
    w: Frame width.

  Returns:
    List of n arrays of shape (h, w, 3) where frame i is filled with value i.
  """
  return [np.full((h, w, 3), i, dtype=np.uint8) for i in range(n)]


def frame_values(frames):
  """
  Map frames created by make_frames back to their fill values.

  Args:
    frames: List of (h, w, 3) uint8 arrays.

  Returns:
    List of int fill values, one per frame.
  """
  return [int(f[0, 0, 0]) for f in frames]


def interpolate(frames, chunk_size, mod, fps=25, landmarks_list=None):
  """
  Call apply_video_interpolation without building a full FaceExtractor (no self usage).

  Args:
    frames:         List of input frames.
    chunk_size:     Chunk size to pad up to.
    mod:            Interpolation modality string.
    fps:            Frames per second for timestamp generation.
    landmarks_list: Optional per-frame landmarks to pad alongside frames.

  Returns:
    The tuple returned by apply_video_interpolation.
  """
  return FaceExtractor.apply_video_interpolation(None, frame_list=frames, chunk_size=chunk_size,
                                                 fps=fps, mod=mod, landmarks_list=landmarks_list)


class TestMirrorStartVideo:

  def test_pads_14_to_16_with_mirrored_start(self):
    frames = make_frames(14)
    new_frames, timestamps = interpolate(frames, 16, 'mirror_start_video')
    assert len(new_frames) == 16
    assert len(timestamps) == 16
    # symmetric mirror of the start: [f1, f0] + [f0..f13]
    assert frame_values(new_frames) == [1, 0] + list(range(14))

  def test_pads_7_to_16_when_padding_exceeds_length(self):
    frames = make_frames(7)
    new_frames, timestamps = interpolate(frames, 16, 'mirror_start_video')
    assert len(new_frames) == 16
    assert len(timestamps) == 16
    # symmetric (ping-pong) reflection before frame 0, as in np.pad(mode='symmetric')
    expected_prefix = list(np.pad(np.arange(7), (9, 0), mode='symmetric')[:9])
    assert frame_values(new_frames) == expected_prefix + list(range(7))

  def test_landmarks_padded_identically_to_frames(self):
    frames = make_frames(7)
    landmarks = [f'lm{i}' for i in range(7)]
    new_frames, timestamps, new_landmarks = interpolate(frames, 16, 'mirror_start_video',
                                                        landmarks_list=landmarks)
    assert len(new_landmarks) == 16
    assert [f'lm{v}' for v in frame_values(new_frames)] == new_landmarks

  def test_exact_multiple_is_noop(self):
    frames = make_frames(32)
    new_frames, timestamps = interpolate(frames, 16, 'mirror_start_video')
    assert len(new_frames) == 32
    assert frame_values(new_frames) == list(range(32))


class TestSpreadLinearly:

  def test_pads_14_to_16_without_raising(self):
    frames = make_frames(14)
    new_frames, timestamps = interpolate(frames, 16, 'spread_linearly')
    assert len(new_frames) == 16
    assert len(timestamps) == 16

  def test_pads_7_to_16(self):
    frames = make_frames(7)
    new_frames, timestamps = interpolate(frames, 16, 'spread_linearly')
    assert len(new_frames) == 16

  def test_landmarks_not_supported(self):
    frames = make_frames(14)
    with pytest.raises(NotImplementedError):
      interpolate(frames, 16, 'spread_linearly', landmarks_list=list(range(14)))


class TestInvalidMode:

  def test_unknown_mode_raises(self):
    frames = make_frames(14)
    with pytest.raises(ValueError):
      interpolate(frames, 16, 'not_a_mode')

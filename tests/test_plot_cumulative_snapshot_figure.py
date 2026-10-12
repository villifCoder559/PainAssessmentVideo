import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_cumulative_snapshot_figure as pcsf


def _synthetic_plot():
    gray = np.full((200, 300), 255.0)
    gray[20, 40:290] = 0      # top spine
    gray[180, 40:290] = 0     # bottom spine
    gray[20:181, 40] = 0      # left spine
    gray[20:181, 289] = 0     # right spine
    for col in (60, 120, 180):
        gray[181:189, col] = 0  # tick marks
    gray[178:190, 10:20] = 0    # y-tick label left of the axes, not a tick
    return gray


def test_axes_geometry_finds_spines_and_ticks_right_of_the_axis():
    assert pcsf.axes_geometry(_synthetic_plot()) == (20, 180, 40, [60, 120, 180])


def test_face_crop_keeps_the_panel_right_of_the_white_plot():
    composite = np.full((10, 30, 3), 255, dtype=np.uint8)
    composite[:, 20:] = 0
    assert pcsf.face_crop(composite).shape == (10, 10, 3)


def test_face_crop_rejects_a_snapshot_without_plot_panel():
    with pytest.raises(ValueError, match="white plot panel"):
        pcsf.face_crop(np.zeros((10, 30, 3), dtype=np.uint8))

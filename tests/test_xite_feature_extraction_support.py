import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from custom import helper


def test_xite_paths_select_global_non_overlapping_augmentation_ids():
    helper.set_step_shift("XITE/video/video_frontalized")

    assert helper.step_shift == 3597
    assert helper.transform_sample_id(3597, "hflip") == 7194

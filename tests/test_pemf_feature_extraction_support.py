from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from custom import helper


def test_pemf_paths_select_non_overlapping_augmentation_ids():
    helper.set_step_shift("PEMF/video/video_frontalized")

    assert helper.step_shift == 277
    assert helper.transform_sample_id(277, "hflip") == 554

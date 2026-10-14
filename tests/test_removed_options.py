import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.helper import REMOVED_OPTION_DEFAULTS, SAMPLE_FRAME_STRATEGY, drop_removed_options


def test_removed_option_with_default_is_dropped():
  assert drop_removed_options({'kept': 1, 'adversarial_head': 0}, 'test') == {'kept': 1}
  assert drop_removed_options({'kept': 1, 'lambda_center_loss': [0.0]}, 'test') == {'kept': 1}
  assert drop_removed_options({'sample_frame_strategy': SAMPLE_FRAME_STRATEGY.SLIDING_WINDOW}, 'test') == {}
  assert drop_removed_options({'xattn_mask': None, 'head_init_path': None}, 'test') == {}


def test_removed_option_with_other_value_raises():
  with pytest.raises(ValueError, match='adversarial_head'):
    drop_removed_options({'adversarial_head': 1}, 'head_params')
  with pytest.raises(ValueError, match='xattn_mask'):
    drop_removed_options({'xattn_mask': [[True]]}, 'config')


def test_unknown_keys_are_kept():
  assert drop_removed_options({'T_S_S_shape': [8, 1, 1], 'max_label': None}, 'test') == {
    'T_S_S_shape': [8, 1, 1], 'max_label': None}


def test_every_removed_option_default_is_accepted():
  assert drop_removed_options(dict(REMOVED_OPTION_DEFAULTS), 'test') == {}

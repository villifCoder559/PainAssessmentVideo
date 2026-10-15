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


def _model_advanced_args():
  # The option guard runs before anything is built, so placeholder values are enough.
  names = ('model_type embedding_reduction clips_reduction path_dataset path_labels head batch_size_training '
           'stride_window clip_length dict_augmented prefetch_factor features_folder_saving_path label_smooth '
           'n_workers new_csv_path load_dataset_in_memory').split()
  args = dict.fromkeys(names)
  args['head_params'] = {}
  return args


def test_model_advanced_rejects_unknown_option():
  from custom.model import Model_Advanced
  with pytest.raises(TypeError, match='not_an_option'):
    Model_Advanced(**_model_advanced_args(), not_an_option=1)

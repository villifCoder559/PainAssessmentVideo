import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_model import validate_arguments


def _args(**overrides):
  args = {'head': 'ATTENTIVE_JEPA', 'loss': ['L1'], 'pos_enc': [3], 'scheduler_name': ['cosine']}
  args.update(overrides)
  return args


@pytest.mark.parametrize('key, value', [('loss', ['huber']), ('pos_enc', [1]), ('scheduler_name', ['step'])])
def test_unsupported_values_fail_before_training(key, value):
  with pytest.raises(ValueError, match=key):
    validate_arguments(_args(**{key: value}))

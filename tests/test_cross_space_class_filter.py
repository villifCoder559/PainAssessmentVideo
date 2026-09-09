from pathlib import Path
import os
import runpy
import sys
from unittest import mock

import pandas as pd
import pytest


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with mock.patch('multiprocessing.Manager') as manager:
  manager.return_value.dict.return_value = {}
  import cross_space_projection as csp


def test_filter_classes_keeps_inclusive_boundary_and_removes_greater_rows():
  frame = pd.DataFrame({
    'class_id': [0.0, 7.0, 8.0, 12.0],
    'sample_id': [10, 11, 12, 13],
  })

  filtered = csp._filter_classes_by_max(frame, 7, Path('/datasets/train.csv'))

  assert filtered['sample_id'].tolist() == [10, 11]
  assert filtered['class_id'].tolist() == [0.0, 7.0]


def test_filter_classes_accepts_zero_cutoff():
  frame = pd.DataFrame({'class_id': [0, 1], 'sample_id': [10, 11]})

  filtered = csp._filter_classes_by_max(frame, 0, 'train.csv')

  assert filtered['sample_id'].tolist() == [10]


def test_filter_classes_disabled_does_not_require_class_id():
  frame = pd.DataFrame({'sample_id': [10, 11]})

  assert csp._filter_classes_by_max(frame, None, 'legacy.csv').equals(frame)


@pytest.mark.parametrize(
  ('frame', 'cutoff', 'message'),
  [
    (pd.DataFrame({'sample_id': [10]}), 7, r"missing required 'class_id'.*input\.csv"),
    (pd.DataFrame({'class_id': ['pain']}), 7, r"non-numeric 'class_id'.*input\.csv"),
    (pd.DataFrame({'class_id': [8, 9]}), 7, r'no rows with class_id <= 7.*input\.csv'),
    (pd.DataFrame({'class_id': [0]}), -1, r'non-negative.*-1'),
  ],
)
def test_filter_classes_rejects_invalid_or_empty_inputs(frame, cutoff, message):
  with pytest.raises(ValueError, match=message):
    csp._filter_classes_by_max(frame, cutoff, 'input.csv')


def test_yaml_forwards_class_cutoff_as_scalar_cli_argument():
  assert csp._yaml_to_argv({'remove_classes_greater': 7}) == [
    '--remove_classes_greater', '7',
  ]


def test_cli_help_exposes_class_cutoff(monkeypatch, capsys):
  script = Path(csp.__file__)
  monkeypatch.setattr(sys, 'argv', [str(script), '--help'])

  with pytest.raises(SystemExit) as exc:
    runpy.run_path(str(script), run_name='__main__')

  assert exc.value.code == 0
  assert '--remove_classes_greater' in capsys.readouterr().out

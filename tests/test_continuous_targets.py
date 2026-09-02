import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.targets import (
  TargetSpec,
  loss_uses_class_bins,
  prepare_batch_targets,
  resolve_target_spec,
  validate_primary_losses,
)


def test_pemf_targets_keep_float_values_and_use_seven_rounded_bins():
  values = np.array([5.20, 3.52, 0.28, 0.19, 6.26], dtype=np.float32)

  spec = TargetSpec.from_values(values)

  assert spec.target_min == pytest.approx(0.19)
  assert spec.target_max == pytest.approx(6.26)
  assert spec.bin_offset == 0
  assert spec.bin_count == 7
  np.testing.assert_array_equal(spec.to_bins(values), [5, 4, 0, 0, 6])
  assert np.float32(values[0]) == np.float32(5.20)


@pytest.mark.parametrize(
  ('values', 'expected_bins', 'offset', 'count'),
  [
    ([-1.5, -0.5, 0.5, 1.5], [0, 1, 3, 4], -2, 5),
    ([0.2, 2.2], [0, 2], 0, 3),
  ],
)
def test_bins_use_half_away_from_zero_and_cover_missing_integers(
  values, expected_bins, offset, count
):
  spec = TargetSpec.from_values(values)

  np.testing.assert_array_equal(spec.to_bins(values), expected_bins)
  assert spec.bin_offset == offset
  assert spec.bin_count == count


def test_normalization_round_trip_and_constant_range():
  spec = TargetSpec.from_values([0.19, 6.26], normalize=True)
  values = np.array([0.19, 3.52, 6.26], dtype=np.float32)
  np.testing.assert_allclose(spec.inverse(spec.normalize(values)), values, atol=1e-6)

  constant = TargetSpec.from_values([5.2, 5.2], normalize=True)
  np.testing.assert_array_equal(constant.normalize(values[:1] * 0 + 5.2), [0.0])
  np.testing.assert_array_equal(constant.inverse(np.array([0.0, 1.0])), [5.2, 5.2])


@pytest.mark.parametrize('values', [[1.0, np.nan], [1.0, np.inf], ['pain']])
def test_invalid_targets_are_rejected(values):
  with pytest.raises(ValueError, match='class_id.*numeric and finite'):
    TargetSpec.from_values(values)


def test_tensor_targets_preserve_float_and_make_long_bins():
  spec = TargetSpec.from_values([0.19, 6.26])
  targets = torch.tensor([5.20, 3.52, 0.28], dtype=torch.float32)

  bins = spec.to_bin_tensor(targets)

  assert targets.dtype == torch.float32
  assert targets.tolist() == pytest.approx([5.20, 3.52, 0.28])
  assert bins.dtype == torch.long
  assert bins.tolist() == [5, 4, 0]


def test_prediction_bins_are_clamped_to_configured_range():
  spec = TargetSpec.from_values([-1.2, 2.2])

  bins = spec.predictions_to_bins(torch.tensor([-9.0, -0.5, 0.5, 9.0]))

  assert bins.tolist() == [0, 0, 2, 3]


def test_csv_union_sets_bounds_and_serializable_metadata(tmp_path):
  paths = []
  for name, values in [('train', [0.19, 3.52]), ('val', [5.20]), ('test', [6.26])]:
    path = tmp_path / f'{name}.csv'
    pd.DataFrame({'class_id': values}).to_csv(path, sep='\t', index=False)
    paths.append(path)

  spec = TargetSpec.from_csv_paths(paths, normalize=True)
  restored = TargetSpec.from_metadata(spec.to_metadata())

  assert restored == spec
  assert spec.to_metadata() == {
    'target_min': pytest.approx(0.19),
    'target_max': pytest.approx(6.26),
    'normalization': 'min_max',
    'rounding': 'half_away_from_zero',
    'bin_offset': 0,
    'bin_count': 7,
    'bin_to_rounded_value': [0, 1, 2, 3, 4, 5, 6],
    'has_fractional_targets': True,
  }


@pytest.mark.parametrize(
  'loss_name', ['ce', 'ce_weight', 'cdw_ce', 'huber_ce', 'sim_loss', 'coral']
)
def test_fractional_targets_reject_class_only_primary_losses(loss_name):
  spec = TargetSpec.from_values([0.19, 5.20])

  with pytest.raises(ValueError, match='fractional class_id.*class-only'):
    validate_primary_losses([loss_name], spec)


def test_integer_targets_keep_classification_compatibility():
  spec = TargetSpec.from_values([0, 1, 2])

  validate_primary_losses(['ce'], spec)


def test_training_pops_bin_target_and_routes_exact_or_class_targets():
  spec = TargetSpec.from_values([0.19, 6.26], normalize=True)
  exact = torch.tensor([0.19, 5.20], dtype=torch.float32)

  regression_inputs = {'x': torch.ones(2, 1), 'class_targets': torch.tensor([0, 5])}
  regression, regression_bins = prepare_batch_targets(
    regression_inputs, exact, spec, use_exact_targets=True
  )
  assert 'class_targets' not in regression_inputs
  assert regression.dtype == torch.float32
  assert regression.tolist() == pytest.approx([0.0, (5.20 - 0.19) / (6.26 - 0.19)])
  assert regression_bins.tolist() == [0, 5]

  classification_inputs = {'x': torch.ones(2, 1), 'class_targets': torch.tensor([0, 5])}
  classification, classification_bins = prepare_batch_targets(
    classification_inputs, exact, spec, use_exact_targets=False
  )
  assert 'class_targets' not in classification_inputs
  assert classification.dtype == torch.long
  assert classification.tolist() == [0, 5]
  assert classification_bins.tolist() == [0, 5]


def test_classification_soft_targets_remain_compatible():
  spec = TargetSpec.from_values([0, 2])
  soft = torch.tensor([[0.8, 0.2, 0.0], [0.0, 0.2, 0.8]])
  inputs = {'class_targets': torch.tensor([0, 2])}

  loss_targets, class_targets = prepare_batch_targets(
    inputs, soft, spec, use_exact_targets=False
  )

  assert torch.equal(loss_targets, soft)
  assert class_targets.tolist() == [0, 2]


def test_runtime_target_spec_resolves_metadata_and_legacy_normalization():
  current = TargetSpec.from_values([0.19, 6.26], normalize=True)
  assert resolve_target_spec(metadata=current.to_metadata()) == current

  legacy = resolve_target_spec(normalize=True, legacy_max_label=5)
  assert legacy.inverse(torch.tensor([0.0, 1.0])).tolist() == [0.0, 5.0]


def test_disentangled_pain_loss_target_mode_detection():
  assert loss_uses_class_bins(torch.nn.CrossEntropyLoss())
  assert not loss_uses_class_bins(torch.nn.L1Loss())

from pathlib import Path
import sys

import pandas as pd
import pytest
import subprocess
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.targets import TargetSpec, huber_delta_for_optimization
from custom import tools
from cross_space_projection import _label_transform_from_config, _inverse_label_values
from train_model import configure_target_spec, get_class_weights


def _write(path, values):
  pd.DataFrame({'class_id': values}).to_csv(path, sep='\t', index=False)
  return str(path)


def test_ordinary_csv_configures_observed_target_metadata(tmp_path):
  csv_path = _write(tmp_path / 'samples.csv', [0.19, 5.20, 6.26])
  config = {
    'training_csv': csv_path,
    'predefined_csv_splits': None,
    'normalize_labels': 1,
    'loss': ['l1'],
  }

  spec = configure_target_spec(config)

  assert spec.target_min == pytest.approx(0.19)
  assert spec.target_max == pytest.approx(6.26)
  assert config['target_spec']['bin_count'] == 7


def test_predefined_splits_use_union_for_target_bounds(tmp_path):
  paths = {
    'train': _write(tmp_path / 'train.csv', [1.2, 3.0]),
    'val': _write(tmp_path / 'val.csv', [0.19]),
    'test': _write(tmp_path / 'test.csv', [6.26]),
  }
  config = {
    'training_csv': paths['train'],
    'predefined_csv_splits': paths,
    'normalize_labels': 0,
    'loss': ['l1'],
  }

  spec = configure_target_spec(config)

  assert (spec.target_min, spec.target_max) == pytest.approx((0.19, 6.26))


def test_config_rejects_fractional_targets_with_classification_loss(tmp_path):
  csv_path = _write(tmp_path / 'samples.csv', [0.19, 1.0])
  config = {
    'training_csv': csv_path,
    'predefined_csv_splits': None,
    'normalize_labels': 0,
    'loss': ['ce'],
  }

  with pytest.raises(ValueError, match='fractional class_id'):
    configure_target_spec(config)


def test_config_rejects_fractional_disentangled_pain_classification(tmp_path):
  csv_path = _write(tmp_path / 'samples.csv', [0.19, 1.0])
  config = {
    'training_csv': csv_path,
    'predefined_csv_splits': None,
    'normalize_labels': 0,
    'loss': None,
    'composite_loss': [None],
    'disent_loss_p_s': ['ce,ce'],
  }

  with pytest.raises(ValueError, match='fractional class_id'):
    configure_target_spec(config)


def test_huber_delta_is_interpreted_on_original_scale():
  normalized = TargetSpec.from_values([0.19, 6.26], normalize=True)
  raw = TargetSpec.from_values([0.19, 6.26], normalize=False)

  assert huber_delta_for_optimization(1.5, normalized) == pytest.approx(1.5 / 6.07)
  assert huber_delta_for_optimization(1.5, raw) == pytest.approx(1.5)


def test_weighted_classification_covers_missing_bins(tmp_path):
  csv_path = _write(tmp_path / 'samples.csv', [0, 2])
  spec = TargetSpec.from_values([0, 2])

  weights = get_class_weights(csv_path, spec)

  assert weights.tolist() == pytest.approx([2.0, 0.0, 2.0])


def test_csv_reader_does_not_truncate_continuous_targets(tmp_path):
  path = tmp_path / 'samples.csv'
  pd.DataFrame(
    [[1, 's1', 5.20, 'pain', 1, 'sample']],
    columns=['subject_id', 'subject_name', 'class_id', 'class_name', 'sample_id', 'sample_name'],
  ).to_csv(path, sep='\t', index=False)

  rows, _ = tools.get_array_from_csv(path)

  assert float(rows[0, 2]) == pytest.approx(5.20)


def test_regression_per_class_uses_bins_but_keeps_exact_loss_targets():
  spec = TargetSpec.from_values([0.19, 5.20])
  class_loss = torch.zeros(2, 2)
  class_accuracy = torch.zeros(2, 2)

  tools.compute_loss_per_class_(
    criterion=torch.nn.L1Loss(),
    unique_train_val_classes=torch.tensor([0, 5]),
    batch_y=torch.tensor([0.19, 5.20]),
    outputs=torch.tensor([0.29, 4.70]),
    class_loss=class_loss,
    class_accuracy=class_accuracy,
    class_targets=torch.tensor([0, 5]),
    target_spec=spec,
  )

  assert class_loss[0].tolist() == pytest.approx([0.10, 0.50])
  assert class_accuracy[0].tolist() == [1, 1]


def test_removed_ccc_loss_option_is_absent_from_cli_help():
  result = subprocess.run(
    [sys.executable, 'train_model.py', '--help'],
    cwd=Path(__file__).resolve().parents[1],
    capture_output=True,
    text=True,
    check=False,
  )

  assert result.returncode == 0, result.stderr
  assert '--add_CCC_loss' not in result.stdout


def test_cross_space_uses_affine_inverse_for_new_and_legacy_configs():
  spec = TargetSpec.from_values([0.19, 6.26], normalize=True)
  current = _label_transform_from_config({'config': {'target_spec': spec.to_metadata()}})
  legacy = _label_transform_from_config({
    'config': {'normalize_labels': 1, 'max_label': 5}
  })

  assert _inverse_label_values(torch.tensor([0.0, 1.0]), current).tolist() == pytest.approx([0.19, 6.26])
  assert _inverse_label_values(torch.tensor([0.0, 1.0]), legacy).tolist() == pytest.approx([0.0, 5.0])


def test_csv_subsampling_skips_missing_bins(monkeypatch):
  rows = pd.DataFrame({
    'subject_id': [1, 2],
    'subject_name': ['s1', 's2'],
    'class_id': [0.2, 2.2],
    'class_name': ['a', 'b'],
    'sample_id': [1, 2],
    'sample_name': ['x', 'y'],
  })
  captured = {}
  monkeypatch.setattr(
    tools,
    'get_array_from_csv',
    lambda _: (rows.to_numpy(), rows.columns.to_numpy()),
  )
  monkeypatch.setattr(
    pd.DataFrame,
    'to_csv',
    lambda self, *args, **kwargs: captured.setdefault('frame', self.copy()),
  )

  tools._generate_csv_subsampled('unused.csv', nr_samples_per_class=1)

  assert captured['frame']['class_id'].astype(float).tolist() == pytest.approx([0.2, 2.2])

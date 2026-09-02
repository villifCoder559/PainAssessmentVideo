from pathlib import Path
import sys

import pandas as pd
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.dataset import (
  AugmentedOnlyBatchSampler,
  SelectiveAugmentationBatchSampler,
  SubjectBatchSampler,
  balancedBatchSampler,
  customDataset,
  highly_optimized_custom_collate,
)
from custom.targets import TargetSpec


def _samples():
  return [
    {
      'features': torch.zeros(1, 1, 1, 1, 2),
      'labels': torch.tensor([5.20], dtype=torch.float32),
      'subject_id': torch.tensor([1]),
      'sample_id': 1,
    },
    {
      'features': torch.ones(1, 1, 1, 1, 2),
      'labels': torch.tensor([3.52], dtype=torch.float32),
      'subject_id': torch.tensor([2]),
      'sample_id': 2,
    },
  ]


def test_feature_collate_keeps_float_targets_and_attaches_long_bins():
  spec = TargetSpec.from_values([0.19, 6.26])

  result = highly_optimized_custom_collate(
    _samples(), pid=0, is_training=True, target_spec=spec
  )

  assert result['labels'].dtype == torch.float32
  assert result['labels'].tolist() == pytest.approx([5.20, 3.52])
  assert result['class_targets'].dtype == torch.long
  assert result['class_targets'].tolist() == [5, 4]


def test_raw_video_collate_keeps_four_items_and_puts_bins_in_input_dict():
  spec = TargetSpec.from_values([0.19, 6.26])
  dataset = customDataset.__new__(customDataset)
  dataset.smooth_labels = 0.0
  dataset.soft_labels = None
  dataset.coral_loss = False
  dataset.num_classes = spec.bin_count
  dataset.target_spec = spec
  batch = [
    {
      'features': torch.zeros(1, 3, 1, 2, 2),
      'labels': torch.tensor([5.20], dtype=torch.float32),
      'subject_id': torch.tensor([1]),
      'sample_id': torch.tensor(1),
    }
  ]

  inputs, targets, subjects, sample_ids = dataset._custom_collate(batch)

  assert len((inputs, targets, subjects, sample_ids)) == 4
  assert targets.dtype == torch.float32
  assert targets.item() == pytest.approx(5.20)
  assert inputs['class_targets'].dtype == torch.long
  assert inputs['class_targets'].tolist() == [5]


def test_raw_video_classification_transforms_use_bin_targets():
  spec = TargetSpec.from_values([0.19, 6.26])
  dataset = customDataset.__new__(customDataset)
  dataset.smooth_labels = 0.1
  dataset.soft_labels = None
  dataset.coral_loss = False
  dataset.num_classes = spec.bin_count
  dataset.target_spec = spec
  batch = [
    {
      'features': torch.zeros(1, 3, 1, 2, 2),
      'labels': torch.tensor([3.52], dtype=torch.float32),
      'subject_id': torch.tensor([1]),
      'sample_id': torch.tensor(1),
    }
  ]

  inputs, targets, _, _ = dataset._custom_collate(batch)

  assert inputs['class_targets'].tolist() == [4]
  assert targets.shape == (1, spec.bin_count)
  assert targets.argmax(dim=1).tolist() == [4]


def test_every_class_aware_sampler_uses_supplied_bins(monkeypatch):
  monkeypatch.setattr('custom.dataset.helper.step_shift', 100)
  frame = pd.DataFrame(
    {
      'sample_id': [1, 2, 3, 4, 101, 102, 103, 104],
      'class_id': [0.19, 0.28, 3.52, 3.60] * 2,
      'subject_id': [1, 2, 1, 2] * 2,
    }
  )
  bins = torch.tensor([0, 0, 4, 4] * 2).numpy()

  balanced = balancedBatchSampler(df=frame, class_bins=bins, batch_size=4, shuffle=True)
  selective = SelectiveAugmentationBatchSampler(
    df=frame,
    class_bins=bins,
    batch_size=2,
    shuffle=False,
    n_keep_augmentations=0,
    augmentation_strategy=1,
  )
  augmented = AugmentedOnlyBatchSampler(
    df=frame,
    class_bins=bins,
    batch_size=2,
    augmentations='hflip',
    balance_batch=False,
  )
  subject = SubjectBatchSampler(
    df=frame,
    class_bins=bins,
    batch_size=4,
    min_subjects_per_level=2,
    shuffle=False,
  )

  assert balanced.y_labels.tolist() == [0, 0, 4, 4] * 2
  assert set(selective.base_ids_per_class) == {0, 4}
  assert augmented.y_labels.tolist() == [0, 0, 4, 4]
  assert subject.pain_levels == [0, 4]

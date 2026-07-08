"""
Tests for SelectiveAugmentationBatchSampler keep_original parameter.
"""
import unittest
import pandas as pd
import numpy as np
from custom import helper
from custom.dataset import SelectiveAugmentationBatchSampler


def _make_df(n_originals=20, aug_types=('hflip', 'jitter', 'rotation')):
  """
  Builds a synthetic DataFrame mimicking the real dataset structure.

  Args:
    n_originals: Number of original samples (base_ids 1..n_originals).
    aug_types:   Tuple of augmentation type names to create for each original.

  Returns:
    pd.DataFrame with 'sample_id' and 'class_id' columns, indexed 0..N-1.
  """
  helper.step_shift = 100  # small shift for testing
  rows = []
  for base_id in range(1, n_originals + 1):
    rows.append({'sample_id': base_id, 'class_id': base_id % 4})
    for aug in aug_types:
      shift = helper.get_shift_for_sample_id(aug)
      rows.append({'sample_id': base_id + shift, 'class_id': base_id % 4})
  return pd.DataFrame(rows).reset_index(drop=True)


class TestKeepOriginalDefault(unittest.TestCase):
  """Verify keep_original=1.0 preserves the original behavior."""

  def setUp(self):
    self.df = _make_df(n_originals=10, aug_types=('hflip', 'jitter'))
    self.sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=1, augmentation_strategy=1,
      keep_original=1.0
    )

  def test_all_originals_present(self):
    """All original indices must appear in every epoch."""
    all_indices = [idx for batch in self.sampler for idx in batch]
    original_sample_ids = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    self.assertTrue(original_sample_ids.issubset(set(all_indices)))

  def test_augmentations_present(self):
    """With keep_original=1.0 and n_keep_augmentations=1, augmented indices must appear."""
    all_indices = [idx for batch in self.sampler for idx in batch]
    augmented_indices = set(self.df[self.df['sample_id'] > helper.step_shift].index)
    # At least some augmented indices should be present
    self.assertTrue(len(augmented_indices.intersection(set(all_indices))) > 0)


class TestKeepOriginalPartial(unittest.TestCase):
  """Verify keep_original < 1.0 splits originals and augmentations correctly."""

  def setUp(self):
    self.n_originals = 20
    self.aug_types = ('hflip', 'jitter', 'rotation')
    self.df = _make_df(n_originals=self.n_originals, aug_types=self.aug_types)

  def _collect_epoch(self, sampler):
    """Flatten all batches into a single list of indices for one epoch."""
    return [idx for batch in sampler for idx in batch]

  def test_correct_number_of_originals_kept(self):
    """The number of originals in an epoch should match floor(keep_original * N)."""
    keep_original = 0.5
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=2, augmentation_strategy=1,
      keep_original=keep_original
    )
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    originals_in_epoch = all_indices.intersection(original_indices)
    expected = int(keep_original * self.n_originals)
    self.assertEqual(len(originals_in_epoch), expected)

  def test_no_augmentations_for_kept_originals(self):
    """Originals that are kept must NOT have their augmentations in the epoch."""
    keep_original = 0.5
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=2, augmentation_strategy=1,
      keep_original=keep_original
    )
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    originals_in_epoch = all_indices.intersection(original_indices)

    # For each kept original, find its base_id and check no augmentation of that base_id appears
    for orig_idx in originals_in_epoch:
      base_id = self.df.loc[orig_idx, 'sample_id']
      for aug_type in self.aug_types:
        shift = helper.get_shift_for_sample_id(aug_type)
        aug_sample_id = base_id + shift
        aug_rows = self.df[self.df['sample_id'] == aug_sample_id]
        for aug_idx in aug_rows.index:
          self.assertNotIn(aug_idx, all_indices,
            f"Augmented idx {aug_idx} (type={aug_type}) of kept original base_id={base_id} should not be in epoch")

  def test_no_originals_for_augmentation_only_groups(self):
    """Non-selected originals must NOT appear in the epoch."""
    keep_original = 0.3
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=2, augmentation_strategy=1,
      keep_original=keep_original
    )
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)

    originals_in_epoch = all_indices.intersection(original_indices)
    excluded_originals = original_indices - originals_in_epoch

    # Excluded originals must not appear
    for orig_idx in excluded_originals:
      self.assertNotIn(orig_idx, all_indices)

  def test_augmentations_available_for_non_selected_originals(self):
    """Non-selected originals should have their augmentations available (some should appear)."""
    keep_original = 0.3
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=2, augmentation_strategy=1,
      keep_original=keep_original
    )
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)

    originals_in_epoch = all_indices.intersection(original_indices)
    excluded_original_base_ids = set()
    for orig_idx in (original_indices - originals_in_epoch):
      excluded_original_base_ids.add(self.df.loc[orig_idx, 'sample_id'])

    # At least some augmentations of excluded originals should appear
    aug_of_excluded_in_epoch = 0
    for base_id in excluded_original_base_ids:
      for aug_type in self.aug_types:
        shift = helper.get_shift_for_sample_id(aug_type)
        aug_sample_id = base_id + shift
        aug_rows = self.df[self.df['sample_id'] == aug_sample_id]
        for aug_idx in aug_rows.index:
          if aug_idx in all_indices:
            aug_of_excluded_in_epoch += 1
    self.assertGreater(aug_of_excluded_in_epoch, 0,
      "Augmentations of non-selected originals should appear in epoch")

  def test_different_originals_across_epochs(self):
    """Across multiple epochs, the set of kept originals should vary."""
    keep_original = 0.5
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=2, augmentation_strategy=1,
      keep_original=keep_original
    )
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)

    sets_of_originals = []
    for _ in range(10):
      all_indices = set(self._collect_epoch(sampler))
      originals_in_epoch = frozenset(all_indices.intersection(original_indices))
      sets_of_originals.append(originals_in_epoch)

    # With 20 originals keeping 10, across 10 epochs we should see variation
    unique_sets = set(sets_of_originals)
    self.assertGreater(len(unique_sets), 1,
      "The subset of kept originals should change across epochs")

  def test_keep_original_zero(self):
    """With keep_original=0, no originals should appear — only augmentations."""
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=2, augmentation_strategy=1,
      keep_original=0.0
    )
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    self.assertEqual(len(all_indices.intersection(original_indices)), 0,
      "No originals should appear when keep_original=0")
    # But augmentations should still be present
    self.assertGreater(len(all_indices), 0, "Augmentations should still appear")

  def test_strategy_2_works(self):
    """Verify keep_original works with augmentation_strategy=2."""
    sampler = SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=4, shuffle=False,
      n_keep_augmentations=1, augmentation_strategy=2,
      keep_original=0.5
    )
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    originals_in_epoch = all_indices.intersection(original_indices)
    expected = int(0.5 * self.n_originals)
    self.assertEqual(len(originals_in_epoch), expected)


class TestKeepOriginalValidation(unittest.TestCase):
  """Verify input validation for keep_original."""

  def test_invalid_keep_original_above_1(self):
    df = _make_df(n_originals=5, aug_types=('hflip',))
    with self.assertRaises(ValueError):
      SelectiveAugmentationBatchSampler(
        df=df, batch_size=2, shuffle=False,
        n_keep_augmentations=1, augmentation_strategy=1,
        keep_original=1.5
      )

  def test_invalid_keep_original_negative(self):
    df = _make_df(n_originals=5, aug_types=('hflip',))
    with self.assertRaises(ValueError):
      SelectiveAugmentationBatchSampler(
        df=df, batch_size=2, shuffle=False,
        n_keep_augmentations=1, augmentation_strategy=1,
        keep_original=-0.1
      )


def _make_imbalanced_df(class_sizes, aug_types=('hflip', 'jitter')):
  """
  Builds a synthetic DataFrame with a configurable number of original samples per class.

  Args:
    class_sizes: Dict mapping class_id -> number of original samples (base-id groups) for that class.
    aug_types:   Tuple of augmentation type names to create for each original.

  Returns:
    pd.DataFrame with 'sample_id' and 'class_id' columns, indexed 0..N-1.
  """
  helper.step_shift = 100  # small shift for testing
  rows = []
  base_id = 1
  for class_id, n_samples in class_sizes.items():
    for _ in range(n_samples):
      rows.append({'sample_id': base_id, 'class_id': class_id})
      for aug in aug_types:
        shift = helper.get_shift_for_sample_id(aug)
        rows.append({'sample_id': base_id + shift, 'class_id': class_id})
      base_id += 1
  return pd.DataFrame(rows).reset_index(drop=True)


def _groups_per_class(df, indices):
  """
  Counts the distinct base-id groups per class represented in a list of dataset indices.

  Args:
    df:      DataFrame with 'sample_id' and 'class_id' columns.
    indices: Iterable of dataset (df) indices from one epoch.

  Returns:
    Dict mapping class_id -> number of distinct base_ids seen.
  """
  seen = {}
  for idx in indices:
    sample_id = df.loc[idx, 'sample_id']
    base_id = sample_id if sample_id <= helper.step_shift else (sample_id - 1) % helper.step_shift + 1
    class_id = df.loc[idx, 'class_id']
    seen.setdefault(class_id, set()).add(base_id)
  return {class_id: len(base_ids) for class_id, base_ids in seen.items()}


class TestUndersampling(unittest.TestCase):
  """Verify the per-class undersampling of base-id groups (undersample_max_per_class)."""

  def setUp(self):
    self.class_sizes = {0: 4, 1: 12, 2: 20}
    self.aug_types = ('hflip', 'jitter')
    self.df = _make_imbalanced_df(self.class_sizes, aug_types=self.aug_types)

  def _make_sampler(self, undersample, keep_original=1.0, n_keep_augmentations=1, batch_size=8):
    """Builds a sampler on the imbalanced df with the given undersampling cap."""
    return SelectiveAugmentationBatchSampler(
      df=self.df, batch_size=batch_size, shuffle=False,
      n_keep_augmentations=n_keep_augmentations, augmentation_strategy=1,
      keep_original=keep_original,
      undersample_max_per_class=undersample
    )

  def _collect_epoch(self, sampler):
    """Flatten all batches into a single list of indices for one epoch."""
    return [idx for batch in sampler for idx in batch]

  def test_disabled_keeps_all_groups(self):
    """With undersample_max_per_class=0 every class keeps all its base-id groups (regression guard)."""
    sampler = self._make_sampler(undersample=0)
    counts = _groups_per_class(self.df, self._collect_epoch(sampler))
    self.assertEqual(counts, self.class_sizes)
    # All originals must appear, as in the default behavior
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    self.assertTrue(original_indices.issubset(all_indices))

  def test_absolute_cap_respected(self):
    """Over-represented classes are capped at N groups; smaller classes keep everything."""
    sampler = self._make_sampler(undersample=8)
    counts = _groups_per_class(self.df, self._collect_epoch(sampler))
    self.assertEqual(counts, {0: 4, 1: 8, 2: 8})

  def test_auto_minority_cap(self):
    """With -1 every class is capped at the minority class group count."""
    sampler = self._make_sampler(undersample=-1)
    counts = _groups_per_class(self.df, self._collect_epoch(sampler))
    min_count = min(self.class_sizes.values())
    self.assertEqual(counts, {class_id: min_count for class_id in self.class_sizes})

  def test_subset_redrawn_each_epoch(self):
    """The subsample of over-represented classes should vary across epochs."""
    sampler = self._make_sampler(undersample=5)
    class2_base_ids_per_epoch = []
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    for _ in range(10):
      epoch_indices = set(self._collect_epoch(sampler))
      base_ids = frozenset(self.df.loc[idx, 'sample_id']
                           for idx in epoch_indices.intersection(original_indices)
                           if self.df.loc[idx, 'class_id'] == 2)
      class2_base_ids_per_epoch.append(base_ids)
    self.assertGreater(len(set(class2_base_ids_per_epoch)), 1,
      "The per-class subsample should change across epochs")

  def test_composes_with_keep_original(self):
    """keep_original applies to the undersampled groups only."""
    keep_original = 0.5
    sampler = self._make_sampler(undersample=8, keep_original=keep_original)
    all_indices = set(self._collect_epoch(sampler))
    original_indices = set(self.df[self.df['sample_id'] <= helper.step_shift].index)
    originals_in_epoch = all_indices.intersection(original_indices)
    n_selected_groups = 4 + 8 + 8  # min(cap, class size) per class
    self.assertEqual(len(originals_in_epoch), int(keep_original * n_selected_groups))

  def test_len_matches_actual_batches(self):
    """__len__ must match the actual number of yielded batches (all groups have equal aug counts)."""
    sampler = self._make_sampler(undersample=8, n_keep_augmentations=1, batch_size=8)
    n_batches = sum(1 for _ in sampler)
    self.assertEqual(len(sampler), n_batches)

  def test_len_with_heterogeneous_aug_counts(self):
    """__len__ must match actual batches when classes have different augmentation densities."""
    helper.step_shift = 100
    rows = []
    base_id = 1
    for _ in range(20):  # class 0: originals only, no augmentations
      rows.append({'sample_id': base_id, 'class_id': 0})
      base_id += 1
    for _ in range(12):  # class 1: original + 2 augmentations
      rows.append({'sample_id': base_id, 'class_id': 1})
      for aug in ('hflip', 'jitter'):
        rows.append({'sample_id': base_id + helper.get_shift_for_sample_id(aug), 'class_id': 1})
      base_id += 1
    df = pd.DataFrame(rows).reset_index(drop=True)
    sampler = SelectiveAugmentationBatchSampler(
      df=df, batch_size=8, shuffle=False,
      n_keep_augmentations=1, augmentation_strategy=1,
      undersample_max_per_class=10
    )
    # class 0 contributes 10 * 1 sample, class 1 contributes 10 * 2 samples -> 30 samples -> 4 batches
    n_batches = sum(1 for _ in sampler)
    self.assertEqual(len(sampler), n_batches)

  def test_invalid_undersample_value(self):
    """Values below -1 must raise ValueError."""
    with self.assertRaises(ValueError):
      self._make_sampler(undersample=-2)


if __name__ == '__main__':
  unittest.main()

import unittest
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
  from custom.plot_aggregation import merge_confusion_matrices, weighted_means_by_label
except ModuleNotFoundError:
  merge_confusion_matrices = None
  weighted_means_by_label = None

try:
  from custom.plot_aggregation import confusion_axis_labels
except ImportError:
  confusion_axis_labels = None


class TestPlotAggregation(unittest.TestCase):
  def test_aggregation_module_is_available(self):
    self.assertIsNotNone(merge_confusion_matrices, 'custom.plot_aggregation is missing')

  def test_axis_label_helper_is_available(self):
    self.assertIsNotNone(confusion_axis_labels, 'confusion_axis_labels is missing')

  @unittest.skipIf(confusion_axis_labels is None, 'axis label helper is missing')
  def test_axis_labels_include_implicit_torchmetrics_gaps(self):
    labels = confusion_axis_labels(torch.zeros(3, 3), [0, 2])
    self.assertEqual(labels, (0, 1, 2))

  @unittest.skipIf(confusion_axis_labels is None, 'axis label helper is missing')
  def test_axis_labels_preserve_complete_explicit_labels(self):
    labels = confusion_axis_labels(torch.zeros(2, 2), ['none', 'pain'])
    self.assertEqual(labels, ('none', 'pain'))

  @unittest.skipIf(merge_confusion_matrices is None, 'aggregation module is missing')
  def test_missing_last_class_is_zero_padded_without_losing_counts(self):
    six_classes = torch.zeros(6, 6, dtype=torch.long)
    six_classes[4, 4] = 3
    six_classes[5, 2] = 4

    five_classes = torch.zeros(5, 5, dtype=torch.long)
    five_classes[4, 1] = 7

    state = merge_confusion_matrices(None, six_classes, [0, 1, 2, 3, 4, 5])
    result, labels = merge_confusion_matrices(state, five_classes, [0, 1, 2, 3, 4])

    self.assertEqual(labels, (0, 1, 2, 3, 4, 5))
    self.assertEqual(result[4, 1].item(), 7)
    self.assertEqual(result[4, 4].item(), 3)
    self.assertEqual(result[5, 2].item(), 4)
    self.assertEqual(result.sum().item(), 14)

  @unittest.skipIf(merge_confusion_matrices is None, 'aggregation module is missing')
  def test_reordered_labels_are_aligned_explicitly(self):
    first = torch.tensor([[5, 2], [3, 7]], dtype=torch.long)
    second = torch.tensor([[11, 13], [17, 19]], dtype=torch.long)

    state = merge_confusion_matrices(None, first, [2, 0])
    result, labels = merge_confusion_matrices(state, second, [0, 1])

    self.assertEqual(labels, (0, 1, 2))
    expected = torch.tensor([
      [18, 13, 3],
      [17, 19, 0],
      [2, 0, 5],
    ])
    self.assertTrue(torch.equal(result, expected))

  @unittest.skipIf(merge_confusion_matrices is None, 'aggregation module is missing')
  def test_merge_is_commutative_and_conserves_counts(self):
    first = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    second = torch.tensor([[5, 6], [7, 8]], dtype=torch.long)

    left = merge_confusion_matrices(
      merge_confusion_matrices(None, first, [0, 2]), second, [1, 2]
    )
    right = merge_confusion_matrices(
      merge_confusion_matrices(None, second, [1, 2]), first, [0, 2]
    )

    self.assertEqual(left[1], right[1])
    self.assertTrue(torch.equal(left[0], right[0]))
    self.assertEqual(left[0].sum().item(), first.sum().item() + second.sum().item())

  @unittest.skipIf(merge_confusion_matrices is None, 'aggregation module is missing')
  def test_rejects_labels_that_do_not_match_matrix_axes(self):
    with self.assertRaisesRegex(ValueError, 'labels'):
      merge_confusion_matrices(None, torch.zeros(2, 2), [0])

  @unittest.skipIf(weighted_means_by_label is None, 'aggregation module is missing')
  def test_weighted_means_use_sample_support_and_allow_missing_labels(self):
    result = weighted_means_by_label([
      ([0, 1], [1.0, 4.0], [2, 1]),
      ([0], [3.0], [6]),
    ])

    self.assertEqual(result, {0: 2.5, 1: 4.0})

  @unittest.skipIf(weighted_means_by_label is None, 'aggregation module is missing')
  def test_weighted_means_reject_non_positive_support(self):
    with self.assertRaisesRegex(ValueError, 'positive'):
      weighted_means_by_label([([0], [1.0], [0])])


if __name__ == '__main__':
  unittest.main()

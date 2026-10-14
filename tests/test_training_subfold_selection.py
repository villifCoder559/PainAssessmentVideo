import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.scripts import should_replace_best_subfold


class TestShouldReplaceBestSubfold(unittest.TestCase):
  def test_accuracy_prefers_higher_metric(self):
    current = {'val_metric_value': 0.70}

    self.assertTrue(
      should_replace_best_subfold(current, 0.90, 'val_accuracy')
    )
    self.assertFalse(
      should_replace_best_subfold(current, 0.60, 'val_accuracy')
    )

  def test_loss_prefers_lower_metric(self):
    current = {'val_metric_value': 0.30}

    self.assertTrue(should_replace_best_subfold(current, 0.20, 'val_loss'))
    self.assertFalse(should_replace_best_subfold(current, 0.40, 'val_loss'))

  def test_first_candidate_wins_and_ties_keep_it(self):
    self.assertTrue(
      should_replace_best_subfold(None, 0.70, 'val_accuracy')
    )
    self.assertFalse(
      should_replace_best_subfold(
        {'val_metric_value': 0.70}, 0.70, 'val_accuracy'
      )
    )


if __name__ == '__main__':
  unittest.main()

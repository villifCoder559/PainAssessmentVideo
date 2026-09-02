import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_model import configure_selection_policy


class TestConfigureSelectionPolicy(unittest.TestCase):
  def test_conflicting_cli_metric_is_replaced_everywhere(self):
    config = {
      'loss': ['ce'],
      'key_early_stopping': 'val_loss',
    }

    with self.assertWarnsRegex(UserWarning, 'Replacing --key_early_stopping'):
      metric = configure_selection_policy(config)

    self.assertEqual(metric, 'val_accuracy')
    self.assertEqual(config['key_early_stopping'], 'val_accuracy')
    self.assertEqual(config['target_metric_best_model'], 'val_accuracy')


if __name__ == '__main__':
  unittest.main()

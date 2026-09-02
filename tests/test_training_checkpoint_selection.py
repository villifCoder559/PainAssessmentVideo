import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.head import save_selected_checkpoints


class TestSaveSelectedCheckpoints(unittest.TestCase):
  def test_saves_distinct_best_and_last_epoch_states(self):
    best_state = {'weight': torch.tensor([1.0])}
    last_state = {'weight': torch.tensor([3.0])}

    with tempfile.TemporaryDirectory() as output_dir:
      saved_paths = save_selected_checkpoints(
        output_dir,
        best_model_state=best_state,
        best_model_epoch=1,
        last_model_state=last_state,
        last_model_epoch=3,
        save_best=True,
        save_last=True,
      )

      best_path = Path(output_dir, 'best_model_ep_1.pt')
      last_path = Path(output_dir, 'last_model_ep_3.pt')
      self.assertEqual(saved_paths, {
        'best': str(best_path),
        'last': str(last_path),
      })
      self.assertTrue(torch.equal(
        torch.load(best_path, weights_only=True)['weight'],
        torch.tensor([1.0]),
      ))
      self.assertTrue(torch.equal(
        torch.load(last_path, weights_only=True)['weight'],
        torch.tensor([3.0]),
      ))

  def test_best_only_does_not_create_a_last_checkpoint(self):
    with tempfile.TemporaryDirectory() as output_dir:
      saved_paths = save_selected_checkpoints(
        output_dir,
        best_model_state={'weight': torch.tensor([1.0])},
        best_model_epoch=1,
        last_model_state={'weight': torch.tensor([3.0])},
        last_model_epoch=3,
        save_best=True,
        save_last=False,
      )

      self.assertEqual(
        saved_paths,
        {'best': str(Path(output_dir, 'best_model_ep_1.pt'))},
      )
      self.assertFalse(Path(output_dir, 'last_model_ep_3.pt').exists())


if __name__ == '__main__':
  unittest.main()

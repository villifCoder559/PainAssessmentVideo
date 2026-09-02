import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import new_plot_res_from_server as plots


class TestPortableLossPlotCopies(unittest.TestCase):
  def test_grouped_plot_is_a_real_file_independent_of_original(self):
    copy_plot = getattr(plots, '_copy_plot', None)
    self.assertIsNotNone(copy_plot)

    with tempfile.TemporaryDirectory() as tmp_dir:
      root = Path(tmp_dir)
      original = root / 'model' / 'loss.png'
      grouped = root / 'loss_plots' / 'loss.png'
      original.parent.mkdir()
      grouped.parent.mkdir()
      original.write_bytes(b'new plot')

      stale_target = root / 'old' / 'loss.png'
      grouped.symlink_to(stale_target)
      copy_plot(original, grouped)
      original.unlink()

      self.assertFalse(grouped.is_symlink())
      self.assertEqual(grouped.read_bytes(), b'new plot')


if __name__ == '__main__':
  unittest.main()

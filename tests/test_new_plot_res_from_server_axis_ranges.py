import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import new_plot_res_from_server as plots


class TestLossAxisRange(unittest.TestCase):
  def test_xite_uses_fractional_loss_range(self):
    data = {
      'config': {
        'criterion': torch.nn.L1Loss(),
        'path_csv_dataset': ['XITE', 'samples.csv'],
      },
      'model_advanced_params': {
        'features_folder_saving_path': ['/features/xite'],
      },
    }

    self.assertEqual(plots._get_loss_y_lim_and_step(data), (1, 0.1))

  def test_xite_range_takes_precedence_over_criterion_range(self):
    data = {
      'config': {
        'criterion': torch.nn.MSELoss(),
        'path_csv_dataset': ['datasets', 'xite.csv'],
      },
      'model_advanced_params': {
        'features_folder_saving_path': ['/features/generic'],
      },
    }

    self.assertEqual(plots._get_loss_y_lim_and_step(data), (1, 0.1))

  def test_non_xite_keeps_existing_criterion_range(self):
    data = {
      'config': {
        'criterion': torch.nn.MSELoss(),
        'path_csv_dataset': ['datasets', 'biovid.csv'],
      },
      'model_advanced_params': {
        'features_folder_saving_path': ['/features/biovid'],
      },
    }

    self.assertEqual(plots._get_loss_y_lim_and_step(data), (15.1, 3))


class TestLossAccuracySubplotRange(unittest.TestCase):
  def test_validation_and_test_accuracy_axes_cover_zero_to_one(self):
    fig, ax = plots.plt.subplots()
    self.addCleanup(plots.plt.close, fig)

    plots._plot_loss_accuracy_subplot(
      ax=ax,
      train_losses=[0.8, 0.4],
      val_accuracy=[0.25, 0.75],
      point_accuracy={'epoch': 1, 'value': 0.6},
      y_lim_loss=1,
      step_lim=0.1,
    )

    self.assertEqual(tuple(ax.get_ylim()), (0.0, 1.0))
    accuracy_axes = [axis for axis in fig.axes if axis is not ax]
    self.assertEqual(len(accuracy_axes), 2)
    self.assertTrue(all(tuple(axis.get_ylim()) == (0.0, 1.0)
                        for axis in accuracy_axes))
    self.assertEqual(accuracy_axes[0].lines[0].get_ydata().tolist(), [0.6])
    self.assertEqual(accuracy_axes[1].lines[0].get_ydata().tolist(), [0.25, 0.75])


if __name__ == '__main__':
  unittest.main()

import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import new_plot_res_from_server as plots


class TestGroupedAccuracyAggregation(unittest.TestCase):
  def test_pools_confusion_counts_and_aligns_classes_across_subfolds(self):
    data = {
      'results': {
        'k0_cross_val_sub_0': {
          'train_val': {
            'best_model_idx': 2,
            'train_unique_y': torch.tensor([0, 1]),
            'train_confusion_matricies': {
              '2': torch.tensor([[8, 2], [1, 3]]),
            },
          },
        },
        'k0_cross_val_sub_1': {
          'train_val': {
            'best_model_idx': 1,
            'train_unique_y': torch.tensor([1, 2]),
            'train_confusion_matricies': {
              '1': torch.tensor([[1, 1], [0, 2]]),
            },
          },
        },
      },
    }

    grouped = plots.get_grouped_accuracies(data)

    self.assertEqual(set(grouped), {'k0'})
    self.assertAlmostEqual(grouped['k0']['train'][0], 0.8)
    self.assertAlmostEqual(grouped['k0']['train'][1], 4 / 6)
    self.assertAlmostEqual(grouped['k0']['train'][2], 1.0)
    self.assertEqual(grouped['k0']['val'], {})
    self.assertEqual(grouped['k0']['test'], {})

  def test_does_not_fall_back_to_sparse_accuracy_history(self):
    data = {
      'results': {
        'k2_cross_val_sub_0': {
          'train_val': {
            'best_model_idx': 1,
            'train_unique_y': torch.tensor([0, 2]),
            'list_train_accuracy_per_class': [
              torch.tensor([0.1, 0.2]),
              torch.tensor([0.5, 0.25]),
            ],
            'count_y_train': {0: 10, 2: 20},
          },
        },
      },
    }

    grouped = plots.get_grouped_accuracies(data)

    self.assertEqual(grouped['k2']['train'], {})

  def test_confusion_matrix_takes_precedence_over_stored_accuracy(self):
    data = {
      'results': {
        'k0_cross_val_sub_0': {
          'train_val': {
            'best_model_idx': 0,
            'train_unique_y': torch.tensor([0, 1]),
            'train_confusion_matricies': {
              '0': torch.tensor([[4, 1], [1, 1]]),
            },
            'list_train_accuracy_per_class': [torch.tensor([0.1, 0.1])],
            'count_y_train': {0: 100, 1: 100},
          },
        },
      },
    }

    grouped = plots.get_grouped_accuracies(data)

    self.assertAlmostEqual(grouped['k0']['train'][0], 0.8)
    self.assertAlmostEqual(grouped['k0']['train'][1], 0.5)

  def test_per_fold_accuracy_uses_confusion_recall_not_history_arrays(self):
    data = {
      'results': {
        'k0_cross_val_final': {
          'train_val': {
            'best_model_idx': 8,
            'train_unique_y': torch.tensor([0, 1]),
            'val_unique_y': torch.tensor([0, 1]),
            'train_confusion_matricies': {
              '8': torch.tensor([[15, 8], [5, 5]]),
            },
            'val_confusion_matricies': {
              8: torch.tensor([[19, 3], [15, 11]]),
            },
            'list_train_accuracy_per_class': [torch.tensor([0.01, 0.99])],
            'list_val_accuracy_per_class': [torch.tensor([0.02, 0.98])],
          },
          'test': {
            'test_unique_y': torch.tensor([0, 1]),
            'test_confusion_matrix': torch.tensor([[7, 3], [2, 8]]),
            'test_accuracy_per_class': torch.tensor([0.03, 0.97]),
          },
        },
      },
    }

    accuracies = plots.get_result_accuracies(
      data, 'k0_cross_val_final'
    )

    self.assertAlmostEqual(accuracies['train'][0], 15 / 23)
    self.assertAlmostEqual(accuracies['train'][1], 0.5)
    self.assertAlmostEqual(accuracies['val'][0], 19 / 22)
    self.assertAlmostEqual(accuracies['val'][1], 11 / 26)
    self.assertAlmostEqual(accuracies['test'][0], 0.7)
    self.assertAlmostEqual(accuracies['test'][1], 0.8)

  def test_omits_classes_with_zero_confusion_support(self):
    data = {
      'results': {
        'k0_cross_val_sub_0': {
          'train_val': {
            'best_model_idx': 0,
            'train_unique_y': torch.tensor([0, 1]),
            'train_confusion_matricies': {
              '0': torch.tensor([[0, 0], [1, 1]]),
            },
          },
        },
      },
    }

    grouped = plots.get_grouped_accuracies(data)

    self.assertEqual(grouped['k0']['train'], {1: 0.5})

  def test_final_test_does_not_fall_back_to_stored_accuracy_array(self):
    data = {
      'results': {
        'k0_cross_val_final': {
          'test': {
            'test_unique_y': torch.tensor([0, 1]),
            'test_accuracy_per_class': torch.tensor([0.5, 0.25]),
            'test_count_y': torch.tensor([2, 4]),
          },
        },
        'k1_cross_val_final': {
          'test': {
            'test_unique_y': torch.tensor([0, 1]),
            'test_accuracy_per_class': torch.tensor([1.0, 0.75]),
            'test_count_y': torch.tensor([6, 4]),
          },
        },
      },
    }

    grouped = plots.get_grouped_accuracies(data)

    self.assertEqual(grouped['final']['test'], {})


class TestGroupedAccuracyPlotting(unittest.TestCase):
  def test_renders_recall_as_fractions_on_fixed_zero_to_one_axis(self):
    fig, ax = plots.plt.subplots()
    self.addCleanup(plots.plt.close, fig)

    plots._plot_grouped_accuracy_per_class(
      ax,
      {0: 0.25, 2: 0.75},
      'Grouped TRAIN Accuracy',
    )

    self.assertEqual([bar.get_height() for bar in ax.patches], [0.25, 0.75])
    self.assertEqual(ax.get_ylim(), (0.0, 1.0))
    self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ['0', '2'])
    self.assertEqual(ax.get_ylabel(), 'Per-class accuracy (recall)')
    self.assertIn('Per-class accuracy (recall)', ax.get_title())

  def test_saves_validation_and_final_grouped_accuracy_figures(self):
    data = {
      'config': {
        'criterion': torch.nn.L1Loss(),
        'path_csv_dataset': ['MIntPAIN', 'samples.csv'],
      },
      'results': {
        'k0_cross_val_sub_0': {
          'train_val': {
            'best_model_idx': 0,
            'train_unique_y': torch.tensor([0, 1]),
            'val_unique_y': torch.tensor([0, 1]),
            'train_confusion_matricies': {
              '0': torch.tensor([[2, 0], [1, 1]]),
            },
            'val_confusion_matricies': {
              '0': torch.tensor([[1, 1], [0, 2]]),
            },
          },
        },
        'k0_cross_val_final': {
          'train_val': {
            'best_model_idx': 0,
            'train_unique_y': torch.tensor([0, 1]),
            'val_unique_y': torch.tensor([0, 1]),
            'train_confusion_matricies': {
              '0': torch.tensor([[3, 1], [0, 2]]),
            },
            'val_confusion_matricies': {
              '0': torch.tensor([[2, 1], [1, 2]]),
            },
          },
          'test': {
            'test_unique_y': torch.tensor([0, 1]),
            'test_confusion_matrix': torch.tensor([[2, 0], [1, 3]]),
          },
        },
      },
    }

    with tempfile.TemporaryDirectory() as output_root:
      plots.plot_grouped_accuracy_per_class(data, output_root, 'test42')

      output_names = {
        path.name for path in Path(output_root, 'test42').glob('*.png')
      }

    self.assertEqual(output_names, {
      'test42_grouped_accuracy_per_class_k0.png',
      'test42_grouped_accuracy_per_class_final.png',
    })

  def test_saves_available_split_when_other_split_is_missing(self):
    data = {
      'config': {
        'criterion': torch.nn.L1Loss(),
        'path_csv_dataset': ['MIntPAIN', 'samples.csv'],
      },
      'results': {
        'k3_cross_val_sub_0': {
          'train_val': {
            'best_model_idx': 0,
            'train_unique_y': torch.tensor([0]),
            'train_confusion_matricies': {'0': torch.tensor([[2]])},
          },
        },
      },
    }

    with tempfile.TemporaryDirectory() as output_root:
      plots.plot_grouped_accuracy_per_class(data, output_root, 'test7')

      output_path = Path(
        output_root, 'test7', 'test7_grouped_accuracy_per_class_k3.png'
      )
      self.assertTrue(output_path.is_file())

  def test_does_not_save_final_figure_from_stored_accuracy_array(self):
    data = {
      'config': {
        'criterion': torch.nn.L1Loss(),
        'path_csv_dataset': ['MIntPAIN', 'samples.csv'],
      },
      'results': {
        'k0_cross_val_final': {
          'test': {
            'test_unique_y': torch.tensor([0, 1]),
            'test_accuracy_per_class': torch.tensor([0.5, 0.25]),
            'test_count_y': torch.tensor([2, 4]),
          },
        },
      },
    }

    with tempfile.TemporaryDirectory() as output_root:
      plots.plot_grouped_accuracy_per_class(data, output_root, 'test8')

      output_path = Path(
        output_root, 'test8', 'test8_grouped_accuracy_per_class_final.png'
      )
      self.assertFalse(output_path.is_file())

  def test_skips_unbc_runs(self):
    data = {
      'config': {
        'criterion': torch.nn.L1Loss(),
        'path_csv_dataset': ['UNBC', 'samples.csv'],
      },
      'results': {},
    }

    with tempfile.TemporaryDirectory() as output_root:
      plots.plot_grouped_accuracy_per_class(data, output_root, 'test9')

      self.assertFalse(Path(output_root, 'test9').exists())

  def test_skips_criteria_without_classification_accuracy(self):
    data = {
      'config': {
        'criterion': plots.losses.RnCLoss(),
        'path_csv_dataset': ['MIntPAIN', 'samples.csv'],
      },
      'results': {},
    }

    with tempfile.TemporaryDirectory() as output_root:
      plots.plot_grouped_accuracy_per_class(data, output_root, 'test10')

      self.assertFalse(Path(output_root, 'test10').exists())


if __name__ == '__main__':
  unittest.main()

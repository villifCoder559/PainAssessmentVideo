import sys
import tempfile
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import new_plot_res_from_server as plots


class ImmediateExecutor:
  def __init__(self, max_workers=None):
    self.max_workers = max_workers

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    return False

  def submit(self, fn, arg):
    future = Future()
    try:
      future.set_result(fn(arg))
    except Exception as exc:
      future.set_exception(exc)
    return future


class TestPlotRunDetailsCsvMode(unittest.TestCase):
  def test_plot_only_dispatches_worker_without_touching_csvs(self):
    file_path = '/tmp/test1_run/k_fold_results.pkl'
    dict_args = {'loss_plot_type': 'loss', 'test_as_validation': 0}

    with tempfile.TemporaryDirectory() as output_root:
      summary_path = Path(output_root, 'summary.csv')
      summary_path.write_text('existing summary\n')

      with mock.patch.object(plots, 'ProcessPoolExecutor', ImmediateExecutor), \
           mock.patch.object(plots.tqdm, 'tqdm', side_effect=lambda iterable, **_: iterable), \
           mock.patch.object(plots, '_process_single_run', return_value=None) as process_run, \
           mock.patch.object(plots, 'generate_subject_class_loss_csv') as generate_loss_csvs:
        plots.plot_run_details(
          {file_path: object()}, output_root, False, dict_args,
          write_csv=False,
        )

      process_run.assert_called_once_with(
        (file_path, output_root, False, dict_args, False, False)
      )
      generate_loss_csvs.assert_not_called()
      self.assertEqual(summary_path.read_text(), 'existing summary\n')

  def test_plot_only_worker_keeps_saved_best_epoch_for_plotting(self):
    file_path = '/tmp/grid/test1_run/k_fold_results.pkl'
    data = {
      'results': {
        'k0_cross_val_sub_0': {
          'train_val': {'val_losses': [3.0, 1.0], 'best_model_idx': 0},
        },
        'k0_cross_val_final': {
          'train_val': {'val_losses': [3.0, 1.0], 'best_model_idx': 0},
        },
      },
      'config': {
        'path_csv_dataset': ['biovid.csv'],
        'model_type': SimpleNamespace(name='model'),
      },
      'time': 0,
    }
    plot_functions = [
      'clean_data',
      'plot_grouped_k_fold',
      'plot_losses',
      'plot_separated_losses_adversarial',
      'plot_hsic_per_epoch',
      'plot_grouped_accuracy_per_class',
      'plot_grouped_confusion_matrix',
      'plot_confusion_matrices',
      'plot_lr_wd_across_epochs',
      'plot_gradient_per_module',
      'plot_CCC_ICC_pearson',
      'plot_history_model_prediction',
      'plot_accuray_per_class_across_epochs',
      'link_attention_logs',
    ]

    with ExitStack() as stack:
      stack.enter_context(mock.patch.object(plots, 'load_results', return_value=data))
      generate_row = stack.enter_context(mock.patch.object(plots, 'generate_csv_row'))
      patched_plots = {
        name: stack.enter_context(mock.patch.object(plots, name))
        for name in plot_functions
      }
      result = plots._process_single_run((
        file_path,
        '/tmp/output',
        False,
        {'loss_plot_type': 'loss', 'test_as_validation': 0},
        False,
        False,
      ))

    self.assertIsNone(result)
    generate_row.assert_not_called()
    patched_plots['plot_grouped_k_fold'].assert_called_once()
    patched_plots['plot_grouped_accuracy_per_class'].assert_called_once()
    self.assertEqual(
      data['results']['k0_cross_val_sub_0']['train_val']['best_model_idx'],
      0,
    )
    self.assertEqual(
      data['results']['k0_cross_val_final']['train_val']['best_model_idx'],
      0,
    )

  def test_csv_preparation_cannot_mutate_shared_plotting_best_epoch(self):
    file_path = '/tmp/grid/test1_run/k_fold_results.pkl'
    shared_train_val = {
      'val_losses': [2.0] * 18 + [0.1],
      'best_model_idx': 8,
    }
    data = {
      'results': {
        'k0_cross_val_sub_0': {'train_val': shared_train_val},
        'k0_cross_val_final': {
          'train_val': shared_train_val,
          'test': {},
        },
      },
      'config': {
        'path_csv_dataset': ['biovid.csv'],
        'model_type': SimpleNamespace(name='model'),
      },
      'time': 0,
    }
    plotted_epochs = []

    def capture_csv_epoch(results, *_):
      self.assertEqual(
        results['k0_cross_val_sub_0']['train_val']['best_model_idx'], 18
      )
      self.assertEqual(
        results['k0_cross_val_final']['train_val']['best_model_idx'], 18
      )
      return {'test_id': 'test1'}

    def capture_plot_epoch(plot_data, *_args, **_kwargs):
      plotted_epochs.append(
        plot_data['results']['k0_cross_val_final']['train_val'][
          'best_model_idx'
        ]
      )

    plot_functions = [
      'clean_data',
      'plot_losses',
      'plot_separated_losses_adversarial',
      'plot_hsic_per_epoch',
      'plot_grouped_accuracy_per_class',
      'plot_grouped_confusion_matrix',
      'plot_confusion_matrices',
      'plot_lr_wd_across_epochs',
      'plot_gradient_per_module',
      'plot_CCC_ICC_pearson',
      'plot_history_model_prediction',
      'plot_accuray_per_class_across_epochs',
      'link_attention_logs',
    ]

    with ExitStack() as stack:
      stack.enter_context(mock.patch.object(plots, 'load_results', return_value=data))
      stack.enter_context(
        mock.patch.object(plots, 'generate_csv_row', side_effect=capture_csv_epoch)
      )
      stack.enter_context(
        mock.patch.object(
          plots, 'plot_grouped_k_fold', side_effect=capture_plot_epoch
        )
      )
      for name in plot_functions:
        stack.enter_context(mock.patch.object(plots, name))
      plots._process_single_run((
        file_path,
        '/tmp/output',
        False,
        {'loss_plot_type': 'loss', 'test_as_validation': 0},
        False,
        True,
      ))

    self.assertEqual(plotted_epochs, [8])
    self.assertEqual(shared_train_val['best_model_idx'], 8)

  def test_default_mode_still_writes_summary_csv(self):
    file_path = '/tmp/test1_run/k_fold_results.pkl'
    row = {'test_id': 'test1', 'metric': 1.0}

    with tempfile.TemporaryDirectory() as output_root, \
         mock.patch.object(plots, 'ProcessPoolExecutor', ImmediateExecutor), \
         mock.patch.object(plots.tqdm, 'tqdm', side_effect=lambda iterable, **_: iterable), \
         mock.patch.object(plots, '_process_single_run', return_value=row), \
         mock.patch.object(plots, 'generate_subject_class_loss_csv') as generate_loss_csvs:
      plots.plot_run_details(
        {file_path: object()},
        output_root,
        False,
        {'loss_plot_type': 'loss', 'test_as_validation': 0},
      )

      generate_loss_csvs.assert_called_once()
      self.assertEqual(
        Path(output_root, 'summary.csv').read_text().splitlines(),
        ['test_id,metric', 'test1,1.0'],
      )

  def test_loss_only_worker_does_not_dispatch_grouped_accuracy(self):
    file_path = '/tmp/grid/test1_run/k_fold_results.pkl'
    data = {
      'results': {
        'k0_cross_val_sub_0': {
          'train_val': {'val_losses': [1.0], 'best_model_idx': 0},
        },
      },
      'config': {
        'path_csv_dataset': ['biovid.csv'],
        'model_type': SimpleNamespace(name='model'),
      },
      'time': 0,
    }
    always_run_plots = [
      'clean_data',
      'plot_grouped_k_fold',
      'plot_losses',
      'plot_separated_losses_adversarial',
      'plot_hsic_per_epoch',
      'link_attention_logs',
    ]

    with ExitStack() as stack:
      stack.enter_context(mock.patch.object(plots, 'load_results', return_value=data))
      patched_always_run = {
        name: stack.enter_context(mock.patch.object(plots, name))
        for name in always_run_plots
      }
      grouped_accuracy = stack.enter_context(
        mock.patch.object(plots, 'plot_grouped_accuracy_per_class')
      )
      plots._process_single_run((
        file_path,
        '/tmp/output',
        False,
        {'loss_plot_type': 'loss', 'test_as_validation': 0},
        True,
        False,
      ))

    patched_always_run['plot_grouped_k_fold'].assert_called_once()
    grouped_accuracy.assert_not_called()


if __name__ == '__main__':
  unittest.main()

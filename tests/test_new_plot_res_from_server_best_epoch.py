import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from matplotlib.axes import Axes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import new_plot_res_from_server as plots


class TestBestEpochConfusionResolver(unittest.TestCase):
  def _data(self, result):
    return {
      'model_advanced_params': {'head': 'ATTENTIVE_JEPA'},
      'config': {'criterion': torch.nn.CrossEntropyLoss()},
      'results': {'k0_cross_val_sub_0': result},
    }

  def test_exact_lookup_accepts_string_and_integer_epoch_keys(self):
    string_matrix = torch.tensor([[3, 1], [0, 2]])
    integer_matrix = torch.tensor([[4, 0], [1, 1]])
    result = {
      'train_val': {
        'best_model_idx': 8,
        'train_confusion_matricies': {'8': string_matrix},
        'val_confusion_matricies': {8: integer_matrix},
      },
    }
    resolver = plots.BestEpochConfusionResolver(self._data(result), None)

    train = resolver.resolve('k0_cross_val_sub_0', 'train')
    val = resolver.resolve('k0_cross_val_sub_0', 'val')

    self.assertTrue(torch.equal(train.matrix, string_matrix))
    self.assertTrue(torch.equal(val.matrix, integer_matrix))
    self.assertEqual((train.requested_epoch, train.resolved_epoch), (8, 8))
    self.assertEqual((val.requested_epoch, val.resolved_epoch), (8, 8))
    self.assertEqual(train.source, 'stored')
    self.assertEqual(val.source, 'stored')

  def test_recomputes_missing_matrices_with_realistic_paths_and_one_model(self):
    result = {
      'train_val': {
        'best_model_idx': 8,
        'train_confusion_matricies': {},
        'val_confusion_matricies': {},
      },
      'test': {'test_confusion_matrix': None},
      'best_model': {'fold_sub_fold_idx': (0, 2)},
    }
    data = {
      'model_advanced_params': {'head': 'ATTENTIVE_JEPA'},
      'config': {'criterion': torch.nn.CrossEntropyLoss()},
      'results': {'k0_cross_val_final': result},
    }
    matrices = {
      'train': torch.tensor([[1, 2], [0, 3]]),
      'val': torch.tensor([[4, 0], [1, 2]]),
      'test': torch.tensor([[2, 1], [1, 5]]),
    }
    built_models = []
    evaluated = []

    def model_factory(_params):
      model = object()
      built_models.append(model)
      return model

    def evaluate_checkpoint(model, checkpoint, csv_path, split):
      evaluated.append((model, Path(checkpoint), Path(csv_path), split))
      return matrices[split]

    with tempfile.TemporaryDirectory() as run_dir:
      fold_dir = Path(
        run_dir, 'train_ATTENTIVE_JEPA', 'k0_cross_val'
      )
      subfold_dir = fold_dir / 'k0_cross_val_sub_2'
      subfold_dir.mkdir(parents=True)
      (subfold_dir / 'best_model_ep_8.pt').touch()
      (subfold_dir / 'train.csv').touch()
      (subfold_dir / 'val.csv').touch()
      (fold_dir / 'test.csv').touch()
      resolver = plots.BestEpochConfusionResolver(
        data,
        run_dir,
        model_factory=model_factory,
        evaluate_checkpoint=evaluate_checkpoint,
      )

      resolved = {
        split: resolver.resolve('k0_cross_val_final', split)
        for split in ('train', 'val', 'test')
      }

    self.assertEqual(len(built_models), 1)
    self.assertEqual(
      [(checkpoint.name, csv_path.name, split)
       for _, checkpoint, csv_path, split in evaluated],
      [
        ('best_model_ep_8.pt', 'train.csv', 'train'),
        ('best_model_ep_8.pt', 'val.csv', 'val'),
        ('best_model_ep_8.pt', 'test.csv', 'test'),
      ],
    )
    for split, resolution in resolved.items():
      self.assertTrue(torch.equal(resolution.matrix, matrices[split]))
      self.assertEqual(resolution.source, 'recomputed')
      self.assertEqual(resolution.resolved_epoch, 8)
    self.assertIn('evaluation-mode checkpoint', resolved['train'].description)

  def test_recompute_failure_uses_nearest_earlier_epoch_and_reports_it(self):
    result = {
      'train_val': {
        'best_model_idx': 8,
        'train_confusion_matricies': {
          '6': torch.tensor([[6]]),
          '10': torch.tensor([[10]]),
        },
      },
    }
    messages = []
    attempts = []

    def fail_evaluation(*args):
      attempts.append(args)
      raise RuntimeError('CUDA unavailable')

    with tempfile.TemporaryDirectory() as run_dir:
      fold_dir = Path(
        run_dir,
        'train_ATTENTIVE_JEPA',
        'k0_cross_val',
        'k0_cross_val_sub_0',
      )
      fold_dir.mkdir(parents=True)
      (fold_dir / 'best_model_ep_8.pt').touch()
      (fold_dir / 'train.csv').touch()
      resolver = plots.BestEpochConfusionResolver(
        self._data(result),
        run_dir,
        model_factory=lambda _params: object(),
        evaluate_checkpoint=fail_evaluation,
        diagnostic=messages.append,
      )

      first = resolver.resolve('k0_cross_val_sub_0', 'train')
      second = resolver.resolve('k0_cross_val_sub_0', 'train')

    self.assertIs(first, second)
    self.assertEqual(len(attempts), 1)
    self.assertEqual(first.source, 'nearest_stored')
    self.assertEqual((first.requested_epoch, first.resolved_epoch), (8, 6))
    self.assertTrue(torch.equal(first.matrix, torch.tensor([[6]])))
    self.assertIn('requested epoch 8', messages[0])
    self.assertIn('substituted stored epoch 6', messages[0])
    self.assertIn('CUDA unavailable', messages[0])

  def test_recomputation_uses_config_snapshot_taken_before_plot_mutations(self):
    result = {
      'train_val': {
        'best_model_idx': 8,
        'train_confusion_matricies': {},
      },
    }
    original_mask = torch.tensor([True, False, True])
    data = self._data(result)
    data['config']['xattn_mask'] = original_mask
    data['model_advanced_params']['concatenate_temporal'] = True
    factory_params = []
    evaluation_kwargs = []

    class FakeModel:
      def test_pretrained_model(self, **kwargs):
        evaluation_kwargs.append(kwargs)
        return {'test_confusion_matrix': torch.tensor([[2]])}

    def model_factory(params):
      factory_params.append(params)
      return FakeModel()

    with tempfile.TemporaryDirectory() as run_dir:
      subfold_dir = Path(
        run_dir,
        'train_ATTENTIVE_JEPA',
        'k0_cross_val',
        'k0_cross_val_sub_0',
      )
      subfold_dir.mkdir(parents=True)
      (subfold_dir / 'best_model_ep_8.pt').touch()
      (subfold_dir / 'train.csv').touch()
      resolver = plots.BestEpochConfusionResolver(
        data, run_dir, model_factory=model_factory
      )

      data['config']['xattn_mask'] = False
      data['model_advanced_params']['head'] = 'MUTATED'
      resolution = resolver.resolve('k0_cross_val_sub_0', 'train')

    self.assertEqual(resolution.source, 'recomputed')
    self.assertEqual(factory_params[0]['head'], 'ATTENTIVE_JEPA')
    self.assertTrue(torch.equal(
      evaluation_kwargs[0]['xattn_mask'], original_mask
    ))

  def test_none_recomputation_result_uses_nearest_stored_epoch(self):
    result = {
      'train_val': {
        'best_model_idx': 8,
        'train_confusion_matricies': {
          '7': torch.tensor([[7]]),
        },
      },
    }

    with tempfile.TemporaryDirectory() as run_dir:
      subfold_dir = Path(
        run_dir,
        'train_ATTENTIVE_JEPA',
        'k0_cross_val',
        'k0_cross_val_sub_0',
      )
      subfold_dir.mkdir(parents=True)
      (subfold_dir / 'best_model_ep_8.pt').touch()
      (subfold_dir / 'train.csv').touch()
      resolver = plots.BestEpochConfusionResolver(
        self._data(result),
        run_dir,
        model_factory=lambda _params: object(),
        evaluate_checkpoint=lambda *_args: None,
      )

      resolution = resolver.resolve('k0_cross_val_sub_0', 'train')

    self.assertEqual(resolution.source, 'nearest_stored')
    self.assertEqual(resolution.resolved_epoch, 7)

  def test_model_construction_failure_is_attempted_only_once_per_run(self):
    result = {
      'train_val': {
        'best_model_idx': 8,
        'train_confusion_matricies': {'7': torch.tensor([[7]])},
        'val_confusion_matricies': {'7': torch.tensor([[6]])},
      },
    }
    attempts = []

    def fail_factory(_params):
      attempts.append('attempt')
      raise RuntimeError('CUDA initialization failed')

    with tempfile.TemporaryDirectory() as run_dir:
      subfold_dir = Path(
        run_dir,
        'train_ATTENTIVE_JEPA',
        'k0_cross_val',
        'k0_cross_val_sub_0',
      )
      subfold_dir.mkdir(parents=True)
      (subfold_dir / 'best_model_ep_8.pt').touch()
      (subfold_dir / 'train.csv').touch()
      (subfold_dir / 'val.csv').touch()
      resolver = plots.BestEpochConfusionResolver(
        self._data(result), run_dir, model_factory=fail_factory
      )

      train = resolver.resolve('k0_cross_val_sub_0', 'train')
      val = resolver.resolve('k0_cross_val_sub_0', 'val')

    self.assertEqual(attempts, ['attempt'])
    self.assertEqual(train.source, 'nearest_stored')
    self.assertEqual(val.source, 'nearest_stored')


class TestBestEpochConfusionPlotting(unittest.TestCase):
  def _data(self, best_epoch=8, include_best=True):
    train_matrices = {'0': torch.tensor([[1, 1], [0, 2]])}
    val_matrices = {'0': torch.tensor([[2, 0], [1, 1]])}
    if include_best:
      train_matrices['8'] = torch.tensor([[3, 1], [1, 3]])
      val_matrices[8] = torch.tensor([[4, 0], [2, 2]])
    else:
      train_matrices.update({
        '6': torch.tensor([[6, 0], [1, 2]]),
        '10': torch.tensor([[10, 0], [1, 2]]),
      })
      val_matrices.update({
        '6': torch.tensor([[5, 1], [1, 2]]),
        '10': torch.tensor([[9, 1], [1, 2]]),
      })
    return {
      'config': {
        'criterion': torch.nn.L1Loss(),
        'validate': True,
        'path_csv_dataset': ['XITE', 'samples.csv'],
      },
      'model_advanced_params': {'head': 'ATTENTIVE_JEPA'},
      'results': {
        'k0_cross_val_final': {
          'train_val': {
            'best_model_idx': best_epoch,
            'train_confusion_matricies': train_matrices,
            'val_confusion_matricies': val_matrices,
          },
          'test': {
            'test_confusion_matrix': torch.tensor([[7, 1], [2, 6]]),
          },
        },
      },
    }

  def test_final_saved_best_figure_contains_train_val_and_test(self):
    data = self._data()
    titles = []

    def capture_plot(_matrix, ax, title):
      titles.append(title)
      ax.set_title(title)

    with tempfile.TemporaryDirectory() as output_root, \
         mock.patch.object(
           plots.tools, 'plot_confusion_matrix', side_effect=capture_plot
         ):
      plots.plot_confusion_matrices(data, output_root, 'run42')
      output_names = {
        path.name for path in Path(output_root, 'run42').glob('*.png')
      }

    self.assertIn(
      'run42_confusion_matrix_k0_cross_val_final_epoch_8.png',
      output_names,
    )
    best_titles = [title for title in titles if 'Best epoch 8' in title]
    self.assertEqual(len(best_titles), 3)
    self.assertTrue(any(title.startswith('TRAIN') for title in best_titles))
    self.assertTrue(any(title.startswith('VAL') for title in best_titles))
    self.assertTrue(any(title.startswith('TEST') for title in best_titles))

  def test_nearest_matrix_is_not_mislabeled_as_the_saved_best_epoch(self):
    data = self._data(include_best=False)
    titles = []
    resolver = plots.BestEpochConfusionResolver(data, None)

    def capture_plot(_matrix, ax, title):
      titles.append(title)
      ax.set_title(title)

    with tempfile.TemporaryDirectory() as output_root, \
         mock.patch.object(
           plots.tools, 'plot_confusion_matrix', side_effect=capture_plot
         ):
      plots.plot_confusion_matrices(
        data, output_root, 'run43', resolver=resolver
      )
      best_path = Path(
        output_root,
        'run43',
        'run43_confusion_matrix_k0_cross_val_final_epoch_8.png',
      )
      self.assertTrue(best_path.is_file())

    substituted = [title for title in titles if 'substituted stored epoch 6' in title]
    self.assertEqual(len(substituted), 2)
    self.assertTrue(all('Best epoch 8' in title for title in substituted))

  def test_final_best_figure_keeps_unavailable_split_as_explicit_panel(self):
    data = self._data()
    data['results']['k0_cross_val_final']['train_val'][
      'val_confusion_matricies'
    ] = {}

    def capture_plot(_matrix, ax, title):
      ax.set_title(title)

    with tempfile.TemporaryDirectory() as output_root, \
         mock.patch.object(
           plots.tools, 'plot_confusion_matrix', side_effect=capture_plot
         ):
      plots.plot_confusion_matrices(data, output_root, 'run44')
      output_path = Path(
        output_root,
        'run44',
        'run44_confusion_matrix_k0_cross_val_final_epoch_8.png',
      )
      image = plots.plt.imread(output_path)

    self.assertEqual(image.shape[:2], (1500, 500))

  def test_final_best_figure_always_requests_all_three_splits(self):
    original_set_title = Axes.set_title

    for case in ('validation_disabled', 'empty_test'):
      with self.subTest(case=case):
        data = self._data()
        unavailable_split = 'val' if case == 'validation_disabled' else 'test'
        if case == 'validation_disabled':
          data['config']['validate'] = False
          data['results']['k0_cross_val_final']['train_val'][
            'val_confusion_matricies'
          ] = {}
        else:
          data['results']['k0_cross_val_final']['test'] = {}

        requested_splits = []
        titles = []

        class RecordingResolver:
          def resolve(self, _key, split):
            requested_splits.append(split)
            if split == unavailable_split:
              return plots.ConfusionResolution(
                None, 8, None, 'unavailable',
                f'requested epoch 8 unavailable for {split}',
              )
            return plots.ConfusionResolution(
              torch.tensor([[1]]), 8, 8, 'stored', 'stored epoch 8'
            )

        def capture_title(ax, title, *args, **kwargs):
          titles.append(title)
          return original_set_title(ax, title, *args, **kwargs)

        def capture_plot(_matrix, ax, title):
          ax.set_title(title)

        with tempfile.TemporaryDirectory() as output_root, \
             mock.patch.object(Axes, 'set_title', new=capture_title), \
             mock.patch.object(
               plots.tools, 'plot_confusion_matrix', side_effect=capture_plot
             ):
          plots.plot_confusion_matrices(
            data, output_root, f'run_{case}', resolver=RecordingResolver()
          )
          best_path = Path(
            output_root,
            f'run_{case}',
            f'run_{case}_confusion_matrix_'
            'k0_cross_val_final_epoch_8.png',
          )
          image = plots.plt.imread(best_path)

        self.assertEqual(requested_splits, ['train', 'val', 'test'])
        self.assertEqual(image.shape[:2], (1500, 500))
        self.assertTrue(any(
          title.startswith(unavailable_split.upper())
          and 'unavailable' in title
          for title in titles
        ))


if __name__ == '__main__':
  unittest.main()

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import pandas as pd


def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_cross_space_logs():
    scipy_stats = _stub_module('scipy.stats')
    scipy = _stub_module('scipy', stats=scipy_stats)
    torchmetrics_classification = _stub_module(
        'torchmetrics.classification', MulticlassConfusionMatrix=object)
    torchmetrics = _stub_module('torchmetrics', classification=torchmetrics_classification)
    custom_tools = _stub_module(
        'custom.tools', concordance_ccc=lambda *args: None,
        plot_confusion_matrix=lambda *args, **kwargs: None)
    custom = _stub_module('custom', tools=custom_tools)
    reducted_plot = _stub_module(
        'new_plot_tsne_post_head', plot_reducted_embeddings=lambda *args, **kwargs: None)
    stubs = {
        'custom': custom,
        'custom.tools': custom_tools,
        'scipy': scipy,
        'scipy.stats': scipy_stats,
        'seaborn': _stub_module('seaborn'),
        'torch': _stub_module('torch'),
        'torchmetrics': torchmetrics,
        'torchmetrics.classification': torchmetrics_classification,
        'umap': _stub_module('umap'),
        'new_plot_tsne_post_head': reducted_plot,
    }
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'cross_space_logs.py'))
    spec = importlib.util.spec_from_file_location('cross_space_logs_aggregated_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


csl = _load_cross_space_logs()


class TestFindAggregatedPkls(unittest.TestCase):
    def test_finds_all_nested_pkls_only_below_aggregated_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = [
                root / 'run_b' / 'aggregated_2' / 'nested' / 'extra.pkl',
                root / 'run_a' / 'aggregated_1' / 'results_1.pkl',
                root / 'run_a' / 'aggregated_1' / 'nested_aggregated_3' / 'deep.pkl',
            ]
            excluded = [
                root / 'run_a' / 'subtrial' / 'results.pkl',
                root / 'run_b' / 'aggregated_2' / 'notes.txt',
            ]
            for path in expected + excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            found = csl._find_aggregated_pkls(root)

            self.assertEqual(found, sorted({str(path.resolve()) for path in expected}))

    def test_includes_pkls_when_root_itself_is_aggregated(self):
        with tempfile.TemporaryDirectory(prefix='aggregated_root_') as tmp:
            pkl = Path(tmp) / 'nested' / 'anything.pkl'
            pkl.parent.mkdir()
            pkl.touch()

            self.assertEqual(csl._find_aggregated_pkls(tmp), [str(pkl.resolve())])

    def test_rejects_invalid_or_empty_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / 'not-a-directory'
            file_path.touch()
            with self.assertRaisesRegex(ValueError, 'not a directory'):
                csl._find_aggregated_pkls(file_path)
            with self.assertRaisesRegex(ValueError, 'No .pkl files'):
                csl._find_aggregated_pkls(tmp)


class TestGenerateLogsAggregated(unittest.TestCase):
    @staticmethod
    def _write_summary_for(pkl_path, rows):
        out_dir = os.path.join(os.path.dirname(pkl_path), 'logs')
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
        return out_dir, None

    def test_processes_every_root_and_combines_generated_summary_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_1 = Path(tmp) / 'root_1'
            root_2 = Path(tmp) / 'root_2'
            pkl_1 = root_1 / 'run_b' / 'aggregated_2' / 'result_b.pkl'
            pkl_2 = root_1 / 'run_a' / 'aggregated_1' / 'result_a.pkl'
            pkl_3 = root_2 / 'aggregated_3' / 'result_c.pkl'
            for pkl in (pkl_1, pkl_2, pkl_3):
                pkl.parent.mkdir(parents=True, exist_ok=True)
                pkl.touch()

            def fake_generate(pkl_path, skip_umap=False):
                stem = Path(pkl_path).stem
                rows = ([{'refine_mode': 'linear', 'stat': 'mean'},
                         {'refine_mode': 'linear', 'stat': 'std'}]
                        if stem == 'result_a' else
                        [{'refine_mode': 'projected', 'stat': 'mean'}])
                return self._write_summary_for(pkl_path, rows)

            with mock.patch.object(csl, 'generate_logs', side_effect=fake_generate) as generate:
                result = csl.generate_logs_aggregated([root_1, root_2], skip_umap=True)

            self.assertEqual(result, [str(root_1), str(root_2)])
            self.assertEqual(
                [call.args[0] for call in generate.call_args_list],
                sorted([str(pkl_1.resolve()), str(pkl_2.resolve())]) + [str(pkl_3.resolve())],
            )
            self.assertTrue(all(call.kwargs == {'skip_umap': True}
                                for call in generate.call_args_list))

            combined_1 = pd.read_csv(root_1 / 'aggregated_summary.csv')
            self.assertEqual(len(combined_1), 3)
            self.assertEqual(
                set(combined_1['source_pkl']),
                {
                    os.path.join('run_a', 'aggregated_1', 'result_a.pkl'),
                    os.path.join('run_b', 'aggregated_2', 'result_b.pkl'),
                },
            )
            self.assertEqual(list(combined_1.columns)[0], 'source_pkl')
            self.assertEqual(len(pd.read_csv(root_2 / 'aggregated_summary.csv')), 1)

    def test_warns_and_continues_after_one_pickle_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / 'aggregated_1' / 'bad.pkl'
            good = root / 'aggregated_2' / 'good.pkl'
            for pkl in (bad, good):
                pkl.parent.mkdir(parents=True)
                pkl.touch()

            def fake_generate(pkl_path, skip_umap=False):
                if pkl_path == str(bad.resolve()):
                    raise RuntimeError('broken pickle')
                return self._write_summary_for(pkl_path, [{'mae': 0.5}])

            output = StringIO()
            with (
                mock.patch.object(csl, 'generate_logs', side_effect=fake_generate),
                redirect_stdout(output),
            ):
                csl.generate_logs_aggregated([root])

            self.assertIn('broken pickle', output.getvalue())
            combined = pd.read_csv(root / 'aggregated_summary.csv')
            self.assertEqual(combined['source_pkl'].tolist(),
                             [os.path.join('aggregated_2', 'good.pkl')])

    def test_uses_distinct_output_directories_for_pkls_in_the_same_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate_dir = root / 'aggregated_1'
            first = aggregate_dir / 'first.pkl'
            second = aggregate_dir / 'second.pkl'
            aggregate_dir.mkdir()
            first.touch()
            second.touch()

            def fake_generate(pkl_path, skip_umap=False, out_dir_override=None):
                self.assertIsNotNone(out_dir_override)
                os.makedirs(out_dir_override, exist_ok=True)
                pd.DataFrame([{'name': Path(pkl_path).stem}]).to_csv(
                    os.path.join(out_dir_override, 'summary.csv'), index=False)
                return out_dir_override, None

            with mock.patch.object(csl, 'generate_logs', side_effect=fake_generate) as generate:
                csl.generate_logs_aggregated([root])

            output_dirs = [call.kwargs['out_dir_override']
                           for call in generate.call_args_list]
            self.assertEqual(
                output_dirs,
                [str(aggregate_dir / 'logs_first'), str(aggregate_dir / 'logs_second')],
            )
            self.assertTrue(all(Path(path, 'summary.csv').is_file()
                                for path in output_dirs))
            combined = pd.read_csv(root / 'aggregated_summary.csv')
            self.assertEqual(set(combined['name']), {'first', 'second'})

    def test_raises_when_all_matched_pickles_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkl = root / 'aggregated_1' / 'bad.pkl'
            pkl.parent.mkdir(parents=True)
            pkl.touch()

            with (
                mock.patch.object(csl, 'generate_logs', side_effect=RuntimeError('broken')),
                self.assertRaisesRegex(RuntimeError, 'No aggregated summaries'),
            ):
                csl.generate_logs_aggregated([root])

            self.assertFalse((root / 'aggregated_summary.csv').exists())

    def test_failed_root_does_not_prevent_later_roots_from_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            failed_root = Path(tmp) / 'failed_root'
            good_root = Path(tmp) / 'good_root'
            failed_pkl = failed_root / 'aggregated_1' / 'bad.pkl'
            good_pkl = good_root / 'aggregated_2' / 'good.pkl'
            for pkl in (failed_pkl, good_pkl):
                pkl.parent.mkdir(parents=True)
                pkl.touch()

            def fake_generate(pkl_path, skip_umap=False):
                if pkl_path == str(failed_pkl.resolve()):
                    raise RuntimeError('broken')
                return self._write_summary_for(pkl_path, [{'mae': 0.25}])

            with (
                mock.patch.object(csl, 'generate_logs', side_effect=fake_generate),
                self.assertRaisesRegex(RuntimeError, str(failed_root)),
            ):
                csl.generate_logs_aggregated([failed_root, good_root])

            self.assertFalse((failed_root / 'aggregated_summary.csv').exists())
            self.assertTrue((good_root / 'aggregated_summary.csv').is_file())


class TestLogsOutDir(unittest.TestCase):
    def test_override_takes_precedence_over_format_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = os.path.join(tmp, 'custom_logs')
            self.assertEqual(
                csl._resolve_logs_out_dir({}, 'grid', '/run/trial/results.pkl', override),
                os.path.abspath(override),
            )
            data = {'config_cross_space_projection': {'out_dir': '/saved/run'}}
            self.assertEqual(
                csl._resolve_logs_out_dir(data, 'standalone', '/actual/results.pkl', override),
                os.path.abspath(override),
            )


class TestOnlyAggregatedCli(unittest.TestCase):
    def test_routes_multiple_roots_to_aggregated_batch(self):
        with mock.patch.object(csl, 'generate_logs_aggregated') as generate:
            csl.main([
                '--pkl_path', '/tmp/root-a', '/tmp/root-b',
                '--only_aggregated', '--skip_umap',
            ])

        generate.assert_called_once_with(['/tmp/root-a', '/tmp/root-b'], skip_umap=True)

    def test_rejects_incompatible_options(self):
        incompatible = [
            ['--plot_trials', '1'],
            ['--only_projector_plots'],
        ]
        for extra in incompatible:
            with self.subTest(extra=extra):
                with (
                    mock.patch.object(csl, 'generate_logs_aggregated') as generate,
                    self.assertRaises(SystemExit) as raised,
                    redirect_stderr(StringIO()),
                ):
                    csl.main(['--pkl_path', '/tmp/root', '--only_aggregated', *extra])
                self.assertEqual(raised.exception.code, 2)
                generate.assert_not_called()

    def test_converts_batch_validation_errors_to_cli_errors(self):
        with (
            mock.patch.object(csl, 'generate_logs_aggregated',
                              side_effect=ValueError('No .pkl files found')),
            self.assertRaises(SystemExit) as raised,
            redirect_stderr(StringIO()),
        ):
            csl.main(['--pkl_path', '/tmp/root', '--only_aggregated'])

        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

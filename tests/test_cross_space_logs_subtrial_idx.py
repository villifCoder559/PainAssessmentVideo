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
    spec = importlib.util.spec_from_file_location('cross_space_logs_subtrial_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


csl = _load_cross_space_logs()


def _make_subtrial(root, index, suffix='run', uid='1'):
    folder = Path(root) / f'cross_space_projection_subtrial_{index}_{suffix}'
    folder.mkdir(parents=True, exist_ok=True)
    pkl = folder / f'results_{uid}.pkl'
    pkl.touch()
    return pkl.resolve()


class TestFindSubtrialPkls(unittest.TestCase):
    def test_recurses_exactly_and_deduplicates_selectors_and_overlapping_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = root / 'nested' / 'experiment'
            expected_1_1 = _make_subtrial(experiment, '1_1')
            expected_2_3 = _make_subtrial(experiment, '2_3')
            _make_subtrial(experiment, '1_10')

            found, missing = csl._find_subtrial_pkls(
                [root, experiment], ['2_3', '1_1', '2_3'])

            self.assertEqual(found, sorted([str(expected_1_1), str(expected_2_3)]))
            self.assertEqual(missing, [])

    def test_accepts_a_matching_subtrial_as_the_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkl = _make_subtrial(tmp, '4_1')

            found, missing = csl._find_subtrial_pkls([pkl.parent], ['4_1'])

            self.assertEqual(found, [str(pkl)])
            self.assertEqual(missing, [])

    def test_rejects_invalid_selectors_roots_and_zero_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            not_dir = root / 'file'
            not_dir.touch()

            for selector in ('1', '1_2_3', 'a_2', '-1_2'):
                with self.subTest(selector=selector):
                    with self.assertRaisesRegex(ValueError, 'DIGITS_DIGITS'):
                        csl._find_subtrial_pkls([root], [selector])
            with self.assertRaisesRegex(ValueError, 'not a directory'):
                csl._find_subtrial_pkls([not_dir], ['1_2'])
            with self.assertRaisesRegex(ValueError, 'No subtrial results'):
                csl._find_subtrial_pkls([root], ['1_2'])


class TestGridCheckpointResolution(unittest.TestCase):
    def _grid_result(self, root):
        pkl = Path(root) / 'trial0000_example' / 'results.pkl'
        pkl.parent.mkdir(parents=True)
        pkl.touch()
        return pkl

    def test_uses_run_local_snapshot_after_temporary_yaml_is_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'search'
            pkl = self._grid_result(root)
            deleted_config = Path(tmp) / 'deleted-generated.yaml'
            (root / 'launch_config.yaml').write_text(
                'new_model_pth: snapshot-new.pt\nold_model_pth: snapshot-old.pt\n',
                encoding='utf-8',
            )
            (root / 'best_config.txt').write_text(
                f'script_cmd: python cross_space_projection.py --config {deleted_config}\n'
                'config_snapshot: launch_config.yaml\n',
                encoding='utf-8',
            )

            self.assertFalse(deleted_config.exists())
            self.assertEqual(
                csl._resolve_new_model_pth({}, 'grid', str(pkl)),
                'snapshot-new.pt',
            )
            self.assertEqual(
                csl._resolve_old_model_pth({}, 'grid', str(pkl)),
                'snapshot-old.pt',
            )

    def test_explicit_script_paths_take_precedence_over_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'search'
            pkl = self._grid_result(root)
            (root / 'launch_config.yaml').write_text(
                'new_model_pth: snapshot-new.pt\nold_model_pth: snapshot-old.pt\n',
                encoding='utf-8',
            )
            (root / 'best_config.txt').write_text(
                'script_cmd: python cross_space_projection.py '
                '--new_model_pth explicit-new.pt --old_model_pth explicit-old.pt\n'
                'config_snapshot: launch_config.yaml\n',
                encoding='utf-8',
            )

            self.assertEqual(
                csl._resolve_new_model_pth({}, 'grid', str(pkl)),
                'explicit-new.pt',
            )
            self.assertEqual(
                csl._resolve_old_model_pth({}, 'grid', str(pkl)),
                'explicit-old.pt',
            )

    def test_legacy_persistent_yaml_still_resolves_without_snapshot_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'search'
            pkl = self._grid_result(root)
            legacy_config = Path(tmp) / 'persistent.yaml'
            legacy_config.write_text(
                'new_model_pth: legacy-new.pt\nold_model_pth: legacy-old.pt\n',
                encoding='utf-8',
            )
            (root / 'best_config.txt').write_text(
                f'script_cmd: python cross_space_projection.py --config {legacy_config}\n',
                encoding='utf-8',
            )

            self.assertEqual(
                csl._resolve_new_model_pth({}, 'grid', str(pkl)),
                'legacy-new.pt',
            )
            self.assertEqual(
                csl._resolve_old_model_pth({}, 'grid', str(pkl)),
                'legacy-old.pt',
            )


class TestGenerateLogsSubtrialIndices(unittest.TestCase):
    def test_processes_matches_warns_for_missing_and_forwards_plot_modifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = _make_subtrial(root / 'a', '0_0', uid='bad')
            good = _make_subtrial(root / 'b', '0_0', uid='good')

            def fake_generate(pkl_path, **kwargs):
                if pkl_path == str(bad):
                    raise RuntimeError('broken pkl')
                return str(Path(pkl_path).parent / 'logs'), None

            output = StringIO()
            with (
                mock.patch.object(csl, 'generate_logs', side_effect=fake_generate) as generate,
                redirect_stdout(output),
            ):
                processed = csl.generate_logs_subtrial_indices(
                    [root], ['0_0', '4_4'], skip_umap=True,
                    only_projector_plots=True)

            self.assertEqual(processed, [str(good)])
            self.assertIn('4_4', output.getvalue())
            self.assertIn('broken pkl', output.getvalue())
            self.assertEqual(
                [call.args[0] for call in generate.call_args_list],
                sorted([str(bad), str(good)]),
            )
            self.assertTrue(all(
                call.kwargs == {
                    'skip_umap': True,
                    'only_projector_plots': True,
                }
                for call in generate.call_args_list
            ))

    def test_raises_when_every_matched_pickle_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_subtrial(root, '0_0')

            with (
                mock.patch.object(csl, 'generate_logs', side_effect=RuntimeError('broken')),
                self.assertRaisesRegex(RuntimeError, 'No subtrial logs'),
            ):
                csl.generate_logs_subtrial_indices([root], ['0_0'])


class TestSubtrialIndexCli(unittest.TestCase):
    def test_routes_multiple_roots_and_selectors(self):
        with mock.patch.object(csl, 'generate_logs_subtrial_indices') as generate:
            csl.main([
                '--pkl_path', '/tmp/root-a', '/tmp/root-b',
                '--subtrial_idx', '2_3', '4_1',
                '--skip_umap', '--only_projector_plots',
            ])

        generate.assert_called_once_with(
            ['/tmp/root-a', '/tmp/root-b'], ['2_3', '4_1'],
            skip_umap=True, only_projector_plots=True)

    def test_rejects_incompatible_modes(self):
        incompatible = [
            ['--plot_trials', '1'],
            ['--only_aggregated'],
        ]
        for extra in incompatible:
            with self.subTest(extra=extra):
                with (
                    mock.patch.object(csl, 'generate_logs_subtrial_indices') as generate,
                    self.assertRaises(SystemExit) as raised,
                    redirect_stderr(StringIO()),
                ):
                    csl.main([
                        '--pkl_path', '/tmp/root',
                        '--subtrial_idx', '2_3',
                        *extra,
                    ])
                self.assertEqual(raised.exception.code, 2)
                generate.assert_not_called()

    def test_converts_batch_errors_to_cli_errors(self):
        with (
            mock.patch.object(
                csl, 'generate_logs_subtrial_indices',
                side_effect=ValueError('No subtrial results found')),
            self.assertRaises(SystemExit) as raised,
            redirect_stderr(StringIO()),
        ):
            csl.main([
                '--pkl_path', '/tmp/root',
                '--subtrial_idx', '2_3',
            ])

        self.assertEqual(raised.exception.code, 2)


if __name__ == '__main__':
    unittest.main()

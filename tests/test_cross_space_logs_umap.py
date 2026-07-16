import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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
        'pandas': _stub_module('pandas'),
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
    spec = importlib.util.spec_from_file_location('cross_space_logs_umap_under_test', path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


csl = _load_cross_space_logs()


class TestUmapSplitImpact(unittest.TestCase):
    def test_projected_only_panel_uses_combined_panel_limits(self):
        reduced_both = np.array([
            [-10.0, -20.0],
            [10.0, 20.0],
            [0.0, 0.0],
            [2.0, 4.0],
            [-2.0, -4.0],
            [1.0, 2.0],
            [-1.0, -2.0],
        ])
        reduced_split = np.array([
            [100.0, 200.0],
            [110.0, 220.0],
            [105.0, 210.0],
            [102.0, 204.0],
            [108.0, 216.0],
        ])

        with (
            mock.patch.object(csl, '_compute_umap', side_effect=[reduced_both, reduced_split]),
            mock.patch.object(csl.plt, 'close'),
            mock.patch('matplotlib.figure.Figure.savefig'),
        ):
            csl.plot_umap_split_impact(
                projected_emb=np.zeros((2, 3), dtype=np.float32),
                projected_labels=np.array([0.0, 1.0]),
                split_emb=np.zeros((5, 3), dtype=np.float32),
                split_labels=np.arange(5, dtype=np.float32),
                split_name='train',
                out_dir='/tmp',
            )
            fig = csl.plt.gcf()

        combined_ax, projected_ax, split_ax = fig.axes[:3]
        self.assertEqual(projected_ax.get_xlim(), combined_ax.get_xlim())
        self.assertEqual(projected_ax.get_ylim(), combined_ax.get_ylim())
        self.assertNotEqual(split_ax.get_xlim(), combined_ax.get_xlim())
        self.assertNotEqual(split_ax.get_ylim(), combined_ax.get_ylim())
        csl.plt.close(fig)


if __name__ == '__main__':
    unittest.main()

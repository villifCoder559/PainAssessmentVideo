import unittest
import sys
import os
import tempfile

import numpy as np
import torch

# Add root to path to import cross_space_projection
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cross_space_projection import (
    _build_projector_network,
    _projector_recipe_id,
    _projector_tag,
    _projector_key,
    _projector_bundle_key,
    LINEAR_PROJECTOR_CONFIG,
)


def _cfg(**overrides):
    """A LINEAR_PROJECTOR_CONFIG copy with the given fields overridden."""
    c = dict(LINEAR_PROJECTOR_CONFIG)
    c.update(overrides)
    return c


class TestAutoencoderNetwork(unittest.TestCase):
    def test_layer_shapes(self):
        """autoencoder must be Linear(d_old, d_old//r) -> act -> Linear(d_old//r, d_new//r)
        -> act -> Linear(d_new//r, d_new) for the chosen ratio r."""
        d_old, d_new, r = 64, 32, 4
        net = _build_projector_network(d_old, d_new, 'autoencoder', 'gelu', encoder_ratio=r)
        linears = [m for m in net if isinstance(m, torch.nn.Linear)]
        acts = [m for m in net if isinstance(m, torch.nn.GELU)]
        self.assertEqual(len(linears), 3)
        self.assertEqual(len(acts), 2)
        self.assertEqual((linears[0].in_features, linears[0].out_features), (d_old, d_old // r))
        self.assertEqual((linears[1].in_features, linears[1].out_features), (d_old // r, d_new // r))
        self.assertEqual((linears[2].in_features, linears[2].out_features), (d_new // r, d_new))

    def test_forward_shape(self):
        """A forward pass on (B, d_old) returns (B, d_new)."""
        d_old, d_new = 48, 24
        net = _build_projector_network(d_old, d_new, 'autoencoder', 'relu', encoder_ratio=4)
        out = net(torch.randn(7, d_old))
        self.assertEqual(tuple(out.shape), (7, d_new))

    def test_hidden_dims_floored_to_one(self):
        """A ratio larger than the dim must floor the bottleneck to >= 1 (no zero-width layer)."""
        net = _build_projector_network(8, 8, 'autoencoder', 'gelu', encoder_ratio=999)
        out = net(torch.randn(3, 8))
        self.assertEqual(tuple(out.shape), (3, 8))

    def test_invalid_inputs(self):
        """Unknown activation and ratio < 1 must raise."""
        with self.assertRaises(ValueError):
            _build_projector_network(8, 8, 'autoencoder', 'not_an_act', encoder_ratio=4)
        with self.assertRaises(ValueError):
            _build_projector_network(8, 8, 'autoencoder', 'gelu', encoder_ratio=0)


class TestRatioDisambiguation(unittest.TestCase):
    """encoder_ratio must distinguish recipes/bundles for autoencoder but collapse for others."""

    def test_recipe_id_distinguishes_ratio(self):
        """Two recipes differing only in encoder_ratio must get distinct recipe ids
        (otherwise proj_map.setdefault would silently drop one)."""
        self.assertNotEqual(_projector_recipe_id(_cfg(encoder_ratio=2)),
                            _projector_recipe_id(_cfg(encoder_ratio=8)))

    def test_recipe_id_default_unchanged(self):
        """encoder_ratio == 4 (default) must leave the recipe id byte-for-byte unchanged."""
        base = dict(LINEAR_PROJECTOR_CONFIG)
        base.pop('encoder_ratio', None)
        self.assertEqual(_projector_recipe_id(base), _projector_recipe_id(_cfg(encoder_ratio=4)))

    def test_autoencoder_bundle_key_distinguishes_ratio(self):
        """For autoencoder, ratio variants get distinct bundle keys (one trained net each)."""
        k2 = _projector_bundle_key('autoencoder', 'gelu', _cfg(encoder_ratio=2))
        k8 = _projector_bundle_key('autoencoder', 'gelu', _cfg(encoder_ratio=8))
        self.assertNotEqual(k2, k8)

    def test_autoencoder_bundle_key_distinguishes_activation(self):
        """For autoencoder, activation is part of the bundle key (it is swept via --mlp_activation)."""
        kg = _projector_bundle_key('autoencoder', 'gelu', _cfg(encoder_ratio=4))
        kr = _projector_bundle_key('autoencoder', 'relu', _cfg(encoder_ratio=4))
        self.assertNotEqual(kg, kr)
        self.assertEqual(_projector_key('autoencoder', 'gelu'), 'autoencoder_gelu')

    def test_linear_bundle_key_ignores_ratio(self):
        """For non-autoencoder kinds the ratio is absent from the tag, so ratio variants
        collapse to a single trained bundle."""
        k2 = _projector_bundle_key('linear', None, _cfg(encoder_ratio=2))
        k8 = _projector_bundle_key('linear', None, _cfg(encoder_ratio=8))
        self.assertEqual(k2, k8)
        self.assertNotIn('er', _projector_tag('linear', _cfg(encoder_ratio=8)))


class TestCrossSpaceLogsAutoencoderRebuild(unittest.TestCase):
    """cross_space_logs must rebuild an autoencoder projector from a checkpoint.

    Regression guard: the recompute path used to coerce every kind not in
    ('linear', 'mlp') to 'linear', so an autoencoder checkpoint (a multi-layer
    Sequential) failed load_state_dict and the helper returned None — which in
    the dashboard silently drops the 'refined' row and shows the refined value
    under the 'projected' label.
    """

    def test_autoencoder_projector_ckpt_roundtrips(self):
        """A saved autoencoder projector must load through _apply_projector_ckpt
        (kind='autoencoder' honored, encoder_ratio threaded) and return (N, d_new)."""
        import cross_space_logs as csl
        d_old, d_new, r, n = 64, 32, 4, 5
        net = _build_projector_network(d_old, d_new, 'autoencoder', 'gelu', encoder_ratio=r)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            ckpt = f.name
        try:
            torch.save(net.state_dict(), ckpt)
            data = {
                'linear_projector': {
                    'kind': 'autoencoder',
                    'norm_stats': None,
                    'config': {'mlp_activation': 'gelu', 'mlp_num_layers': 1, 'encoder_ratio': r},
                },
                'new_model_tensors': {'embeddings': np.zeros((n, d_new), dtype=np.float32)},
            }
            old_emb = np.random.randn(n, d_old).astype(np.float32)
            out = csl._apply_projector_ckpt(data, old_emb, ckpt, stage_label='test')
            self.assertIsNotNone(out)  # pre-fix this is None (state_dict mismatch)
            self.assertEqual(tuple(np.asarray(out).shape), (n, d_new))
        finally:
            os.remove(ckpt)


if __name__ == '__main__':
    unittest.main()

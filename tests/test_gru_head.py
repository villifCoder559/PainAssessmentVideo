import os
import sys
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from custom.head import GRUHead
from custom import helper
from custom.dataset import highly_optimized_custom_collate, _custom_collate

torch.manual_seed(0)


@pytest.fixture
def default_kwargs():
  return dict(input_dim=8, hidden_size=16, num_layers=1)


@pytest.fixture
def small_seq():
  return torch.randn(2, 4, 8)


@pytest.fixture
def spatial_seq():
  return torch.randn(2, 4, 3, 3, 8)


# ---------- Initialization ----------

def test_init_regression_defaults(default_kwargs):
  h = GRUHead(**default_kwargs)
  assert h.is_classification is False
  assert h.linear.out_features == 1
  assert h.gru.bidirectional is False
  assert isinstance(h.input_norm, nn.Identity)
  assert isinstance(h.norm, nn.Identity)
  assert h.embedding_reduction == helper.EMBEDDING_REDUCTION.NONE


def test_init_classification_via_num_classes(default_kwargs):
  h = GRUHead(**default_kwargs, num_classes=5)
  assert h.linear.out_features == 5
  assert h.is_classification is True


def test_init_classification_via_output_size(default_kwargs):
  h = GRUHead(**default_kwargs, output_size=3)
  assert h.linear.out_features == 3
  assert h.is_classification is True


def test_bidirectional_doubles_out_dim(default_kwargs):
  h = GRUHead(**default_kwargs, bidirectional=True, layer_norm=True)
  assert h.linear.in_features == 32
  assert isinstance(h.norm, nn.LayerNorm)
  assert tuple(h.norm.normalized_shape) == (32,)


@pytest.mark.parametrize('layer_norm,expected', [
  (True, nn.LayerNorm),
  (False, nn.Identity),
])
def test_layer_norm_flag(default_kwargs, layer_norm, expected):
  h = GRUHead(**default_kwargs, layer_norm=layer_norm)
  assert isinstance(h.input_norm, expected)
  assert isinstance(h.norm, expected)


def test_dropout_zeroed_for_single_layer():
  h1 = GRUHead(input_dim=8, hidden_size=16, num_layers=1, dropout=0.5)
  assert h1.gru.dropout == 0.0
  h2 = GRUHead(input_dim=8, hidden_size=16, num_layers=2, dropout=0.5)
  assert h2.gru.dropout == 0.5


def test_training_init_hook_preserves_native_gru_parameters(default_kwargs):
  h = GRUHead(**default_kwargs)
  native_parameters = {name: value.detach().clone() for name, value in h.named_parameters()}

  h._initialize_weights(init_type='default')

  assert all(
    torch.equal(native_parameters[name], value)
    for name, value in h.named_parameters()
  )


# ---------- _reduce_to_sequence ----------

def test_reduce_passthrough_3d(default_kwargs, small_seq):
  h = GRUHead(**default_kwargs)
  out = h._reduce_to_sequence(small_seq)
  assert out is small_seq


def test_reduce_5d_mean_spatial(default_kwargs, spatial_seq):
  h = GRUHead(**default_kwargs, embedding_reduction=helper.EMBEDDING_REDUCTION.MEAN_SPATIAL)
  out = h._reduce_to_sequence(spatial_seq)
  assert out.shape == (2, 4, 8)
  assert torch.allclose(out, spatial_seq.mean(dim=(2, 3)))


def test_reduce_5d_mean_temporal(default_kwargs, spatial_seq):
  h = GRUHead(**default_kwargs, embedding_reduction=helper.EMBEDDING_REDUCTION.MEAN_TEMPORAL)
  out = h._reduce_to_sequence(spatial_seq)
  assert out.dim() == 3
  assert out.shape[0] == 2 and out.shape[2] == 8
  # After temporal mean (keepdim) + spatial loop-reduction.
  expected = spatial_seq.mean(dim=1, keepdim=True).mean(dim=2).mean(dim=2)
  assert torch.allclose(out, expected)


def test_reduce_5d_none_fallback(default_kwargs, spatial_seq):
  h = GRUHead(**default_kwargs, embedding_reduction=helper.EMBEDDING_REDUCTION.NONE)
  out = h._reduce_to_sequence(spatial_seq)
  assert out.shape == (2, 4, 8)


def test_reduce_high_dim_stress(default_kwargs):
  h = GRUHead(**default_kwargs)
  x = torch.randn(2, 4, 2, 2, 2, 8)
  out = h._reduce_to_sequence(x)
  assert out.shape == (2, 4, 8)


# ---------- Forward pass ----------

def test_forward_basic_shapes(default_kwargs, small_seq):
  h = GRUHead(**default_kwargs).eval()
  with torch.no_grad():
    r = h(small_seq)
  assert r['logits'].shape == (2, 1)
  assert r['embeddings'].shape == (2, 16)
  assert torch.isfinite(r['logits']).all()


def test_forward_valid_mask_uses_pack_padded(monkeypatch, default_kwargs, small_seq):
  spy = {'called': 0, 'lengths': None}
  orig = torch.nn.utils.rnn.pack_padded_sequence
  def wrapper(x, lengths, batch_first=False, enforce_sorted=True):
    spy['called'] += 1
    spy['lengths'] = lengths.clone()
    return orig(x, lengths, batch_first=batch_first, enforce_sorted=enforce_sorted)
  monkeypatch.setattr(torch.nn.utils.rnn, 'pack_padded_sequence', wrapper)

  h = GRUHead(**default_kwargs).eval()
  mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
  with torch.no_grad():
    r = h(small_seq, key_padding_mask=mask)
  assert spy['called'] == 1
  assert torch.equal(spy['lengths'], torch.tensor([3, 2]))
  assert r['logits'].shape == (2, 1)


def test_forward_mask_all_false_clamped_to_one(monkeypatch, default_kwargs, small_seq):
  spy = {'lengths': None}
  orig = torch.nn.utils.rnn.pack_padded_sequence
  def wrapper(x, lengths, **kw):
    spy['lengths'] = lengths.clone()
    return orig(x, lengths, **kw)
  monkeypatch.setattr(torch.nn.utils.rnn, 'pack_padded_sequence', wrapper)

  h = GRUHead(**default_kwargs).eval()
  mask = torch.zeros(2, 4, dtype=torch.bool)
  with torch.no_grad():
    r = h(small_seq, key_padding_mask=mask)
  assert torch.equal(spy['lengths'], torch.tensor([1, 1]))
  assert torch.isfinite(r['logits']).all()


def test_forward_invalid_mask_falls_back(monkeypatch, default_kwargs, small_seq):
  spy = {'called': 0}
  def wrapper(*a, **kw):
    spy['called'] += 1
    raise AssertionError('pack should not be called for invalid mask')
  monkeypatch.setattr(torch.nn.utils.rnn, 'pack_padded_sequence', wrapper)

  h = GRUHead(**default_kwargs).eval()
  bad_mask = torch.ones(2, 5, dtype=torch.bool)
  with torch.no_grad():
    r = h(small_seq, key_padding_mask=bad_mask)
  assert spy['called'] == 0
  assert r['logits'].shape == (2, 1)


def test_forward_bidirectional_output_shape(default_kwargs, small_seq):
  h = GRUHead(**default_kwargs, bidirectional=True).eval()
  # Patch gru.forward to return a deterministic h_n for concat verification.
  fixed_h_n = torch.arange(2 * 2 * 16, dtype=torch.float32).reshape(2, 2, 16)
  h.gru.forward = lambda *a, **kw: (None, fixed_h_n)
  with torch.no_grad():
    r = h(small_seq)
  assert r['embeddings'].shape == (2, 32)
  expected = torch.cat([fixed_h_n[-2], fixed_h_n[-1]], dim=-1)
  # norm is Identity here (layer_norm=False), so embeddings == concat directly.
  assert torch.allclose(r['embeddings'], expected)


def test_forward_return_video_emb_false(default_kwargs, small_seq):
  h = GRUHead(**default_kwargs).eval()
  with torch.no_grad():
    r = h(small_seq, return_video_emb=False)
  assert r['embeddings'] is None
  assert r['logits'].shape == (2, 1)


def test_forward_5d_input_through_reduce(default_kwargs, spatial_seq):
  h = GRUHead(**default_kwargs, embedding_reduction=helper.EMBEDDING_REDUCTION.MEAN_SPATIAL).eval()
  with torch.no_grad():
    r = h(spatial_seq)
  assert r['logits'].shape == (2, 1)


def test_forward_sequence_length_one(default_kwargs):
  h = GRUHead(**default_kwargs).eval()
  x = torch.randn(2, 1, 8)
  with torch.no_grad():
    r = h(x)
  assert r['logits'].shape == (2, 1)


# ---------- Backbone integration ----------

def test_backbone_called_when_provided(default_kwargs, small_seq):
  backbone = MagicMock()
  backbone.forward_features.return_value = small_seq
  h = GRUHead(**default_kwargs, backbone=backbone).eval()
  raw = torch.randn(2, 3, 16, 16)
  with torch.no_grad():
    h(raw)
  backbone.forward_features.assert_called_once()
  assert backbone.forward_features.call_args[0][0] is raw


def test_backbone_not_called_when_none(default_kwargs, small_seq):
  h = GRUHead(**default_kwargs).eval()
  assert h.backbone is None
  with torch.no_grad():
    r = h(small_seq)
  assert r['logits'].shape == (2, 1)


# ---------- Gradient flow ----------

def test_backward_produces_grads_all_trainable_params(default_kwargs, small_seq):
  h = GRUHead(**default_kwargs, layer_norm=True).train()
  r = h(small_seq)
  r['logits'].sum().backward()
  for name, p in h.named_parameters():
    if p.requires_grad:
      assert p.grad is not None, f'No grad for {name}'
      assert torch.isfinite(p.grad).all(), f'Non-finite grad for {name}'


# ---------- DataLoader / collate integration ----------

def _make_batch(B, nr_chunks, T, S, D, pid=0):
  """Return a list of B sample dicts mirroring _get_element output."""
  return [
    {
      'features': torch.randn(nr_chunks, T, S, S, D),
      'labels': torch.tensor([0]),
      'subject_id': torch.tensor([i]),
      'sample_id': i,
    }
    for i in range(B)
  ]


def _run_collate(batch, pid=0):
  """Call highly_optimized_custom_collate with the minimal required kwargs."""
  return highly_optimized_custom_collate(
    batch=batch,
    pid=pid,
    is_training=False,
    concatenate_temporal=False,
    smooth_labels=0.0,
    soft_labels_mat=None,
    coral_loss=False,
    split_chunks=False,
    concatenate_quadrants=False,
    xattn_mask=None,
    num_classes=2,
  )


@pytest.fixture
def dfer_batch():
  """
  Mimics a batch of DFER spatial-pooled features after MEAN_SPATIAL reduction
  in _get_element: shape [nr_chunks, T, 1, 1, D].
  """
  return _make_batch(B=4, nr_chunks=8, T=8, S=1, D=768)


@pytest.fixture
def dfer_batch_raw():
  """
  Mimics full spatial DFER features before reduction: [nr_chunks, T, 14, 14, D].
  Exercises the reshape path inside the collate.
  """
  return _make_batch(B=4, nr_chunks=2, T=4, S=14, D=768)


def test_collate_spatial_pooled_output_shape(dfer_batch):
  """
  For spatial-pooled features (S=1) the collate must produce x of shape
  (B, nr_chunks*T, D) — exactly what GRUHead forward expects as (B, T_seq, D).
  """
  B, nr_chunks, T, S, D = 4, 8, 8, 1, 768
  res = _run_collate(dfer_batch)
  x = res['features']
  assert x.shape == (B, nr_chunks * T * S * S, D)
  assert x.dtype == torch.float32


def test_collate_full_spatial_output_shape(dfer_batch_raw):
  """
  For full spatial features (S=14) collate flattens to (B, nr_chunks*T*S*S, D).
  GRUHead._reduce_to_sequence must then reduce this to (B, T_seq, D).
  """
  B, nr_chunks, T, S, D = 4, 2, 4, 14, 768
  res = _run_collate(dfer_batch_raw)
  x = res['features']
  assert x.shape == (B, nr_chunks * T * S * S, D)


def test_collate_mask_is_none_when_uniform_lengths(dfer_batch):
  """
  When all samples in a batch have identical feature shapes, no padding is
  needed and key_padding_mask must be None.
  """
  res = _run_collate(dfer_batch)
  assert res['key_padding_mask'] is None


def test_collate_mask_shape_when_variable_lengths():
  """
  When samples differ in nr_chunks, pad_sequence is used and the mask has
  shape (B, 1, 1, max_len).
  """
  batch = [
    {'features': torch.randn(nr_chunks, 4, 1, 1, 8), 'labels': torch.tensor([0]),
     'subject_id': torch.tensor([i]), 'sample_id': i}
    for i, nr_chunks in enumerate([2, 3, 4])
  ]
  res = _run_collate(batch)
  B = 3
  max_len = 4 * 4 * 1 * 1  # 4 chunks * T=4 * S=1 * S=1
  assert res['key_padding_mask'].shape == (B, 1, 1, max_len)


def test_custom_collate_returns_gru_dict_format():
  """
  _custom_collate must return ({'x': ..., 'key_padding_mask': ...}, labels, ...)
  when instance_model_name == GRUHead.
  """
  batch = _make_batch(B=2, nr_chunks=4, T=4, S=1, D=8)
  model = MagicMock(spec=GRUHead)
  model.training = False
  model.embedding_reduction = helper.EMBEDDING_REDUCTION.NONE
  inputs, labels, subject_id, sample_id = _custom_collate(
    batch=batch,
    instance_model_name=helper.INSTANCE_MODEL_NAME.GRUHead,
    concatenate_temporal=False,
    model=model,
    num_classes=2,
    smooth_labels=0.0,
    soft_labels_mat=None,
    coral_loss=False,
    concatenate_quadrants=False,
    xattn_mask=None,
    split_chunks=False,
  )
  assert 'x' in inputs
  assert 'key_padding_mask' in inputs
  assert inputs['x'].shape == (2, 4 * 4 * 1 * 1, 8)
  assert labels.shape == (2,)


def test_collate_to_gruhead_forward_end_to_end(dfer_batch):
  """
  Full pipeline: collate DFER spatial-pooled batch → GRUHead.forward.
  Logits must be (B, 1) with no NaN.
  """
  B, nr_chunks, T, S, D = 4, 8, 8, 1, 768
  res = _run_collate(dfer_batch)
  x = res['features']           # (B, nr_chunks*T, D) already 3D

  head = GRUHead(input_dim=D, hidden_size=64, num_layers=1).eval()
  with torch.no_grad():
    out = head(x, key_padding_mask=res['key_padding_mask'])

  assert out['logits'].shape == (B, 1)
  assert torch.isfinite(out['logits']).all()


def test_collate_4d_mask_bypassed_by_gruhead():
  """
  When the collate produces a 4-D key_padding_mask (B,1,1,T_seq), GRUHead
  must detect the wrong shape and fall back to plain gru(x) instead of
  pack_padded_sequence.  Logits must still be finite.
  """
  # Variable-length batch → produces a 4D mask.
  batch = [
    {'features': torch.randn(nr_c, 4, 1, 1, 8), 'labels': torch.tensor([0]),
     'subject_id': torch.tensor([i]), 'sample_id': i}
    for i, nr_c in enumerate([2, 3])
  ]
  res = _run_collate(batch)
  mask = res['key_padding_mask']
  assert mask is not None and mask.dim() == 4, 'precondition: mask is 4D'

  spy = {'called': 0}
  orig = torch.nn.utils.rnn.pack_padded_sequence
  def spy_pack(*a, **kw):
    spy['called'] += 1
    return orig(*a, **kw)

  import unittest.mock as mock
  with mock.patch.object(torch.nn.utils.rnn, 'pack_padded_sequence', spy_pack):
    head = GRUHead(input_dim=8, hidden_size=16, num_layers=1).eval()
    with torch.no_grad():
      out = head(res['features'], key_padding_mask=mask)

  assert spy['called'] == 0, 'pack_padded_sequence must not be called for 4D mask'
  assert torch.isfinite(out['logits']).all()


def test_collate_with_mean_spatial_reduction_matches_gruhead_input_dim():
  """
  The embedding_reduction applied in _get_element does NOT change D (last dim).
  GRUHead.input_dim must match the feature embedding dimension D, not T or S.
  This guards against accidentally wiring input_dim to the sequence length.
  """
  D = 768
  batch = _make_batch(B=2, nr_chunks=4, T=4, S=1, D=D)
  res = _run_collate(batch)
  x = res['features']
  _, _, feature_dim = x.shape
  assert feature_dim == D

  head = GRUHead(input_dim=feature_dim, hidden_size=32, num_layers=1).eval()
  with torch.no_grad():
    out = head(x)
  assert out['logits'].shape == (2, 1)

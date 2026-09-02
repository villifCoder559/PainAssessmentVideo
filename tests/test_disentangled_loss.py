from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.loss import DisentangledLoss, RnCLossV2


@pytest.fixture
def features():
  torch.manual_seed(0)
  return torch.randn(8, 16, requires_grad=True)


@pytest.mark.parametrize(
  'lambdas,active_slice,inactive_slice',
  [
    ((1.0, 0.0), slice(None, 8), slice(8, None)),
    ((0.0, 1.0), slice(8, None), slice(None, 8)),
  ],
)
def test_component_losses_isolate_gradients(
  features, lambdas, active_slice, inactive_slice
):
  criterion = DisentangledLoss(
    loss_fns=(nn.MSELoss(), nn.MSELoss()),
    split_idx=8,
    lambdas=lambdas,
  )

  loss, _ = criterion(
    features,
    target_pain=torch.randn(8, 8),
    target_subj=torch.randn(8, 8),
  )
  loss.backward()

  assert features.grad[:, active_slice].abs().sum() > 0
  assert features.grad[:, inactive_slice].abs().sum() == 0


def test_orthogonality_loss_reaches_both_feature_slices(features):
  criterion = DisentangledLoss(
    loss_fns=(nn.MSELoss(), nn.MSELoss()),
    split_idx=8,
    lambdas=(0.0, 0.0),
    ortho_lambda=1.0,
  )

  loss, log = criterion(
    features,
    target_pain=torch.randn(8, 8),
    target_subj=torch.randn(8, 8),
  )
  loss.backward()

  assert features.grad[:, :8].abs().sum() > 0
  assert features.grad[:, 8:].abs().sum() > 0
  assert log['ortho_loss'] > 0


def test_returns_trainable_loss_and_scalar_log_contract(features):
  criterion = DisentangledLoss(
    loss_fns=(nn.MSELoss(), nn.MSELoss()),
    split_idx=8,
    lambdas=(0.7, 0.3),
    ortho_lambda=0.5,
  )

  loss, log = criterion(
    features,
    target_pain=torch.randn(8, 8),
    target_subj=torch.randn(8, 8),
  )

  assert loss.ndim == 0
  assert loss.requires_grad
  assert set(log) == {
    'total_loss',
    'loss_pain',
    'loss_subj',
    'lambda_pain',
    'lambda_subj',
    'ortho_loss',
    'ortho_lambda',
  }
  assert log['lambda_pain'] == 0.7
  assert log['lambda_subj'] == 0.3


def test_supports_rnc_as_the_pain_component(features):
  criterion = DisentangledLoss(
    loss_fns=(RnCLossV2(), nn.MSELoss()),
    split_idx=8,
    lambdas=(1.0, 1.0),
    ortho_lambda=0.1,
  )

  loss, _ = criterion(
    features,
    target_pain=torch.randn(8, 1),
    target_subj=torch.randn(8, 8),
  )
  loss.backward()

  assert torch.isfinite(loss)
  assert torch.isfinite(features.grad).all()

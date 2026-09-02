from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.loss import RnCLossV2


def _reference_rnc_loss(features, labels, temperature=2.0):
  if labels.ndim == 1:
    labels = labels.unsqueeze(-1)
  label_differences = (labels[:, None, :] - labels[None, :, :]).abs().sum(-1)
  raw_logits = -(features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
  logits = raw_logits / temperature
  logits = logits - logits.max(dim=1, keepdim=True).values.detach()
  exponentials = logits.exp()
  count = len(features)
  diagonal = ~torch.eye(count, dtype=torch.bool, device=features.device)
  logits = logits[diagonal].view(count, count - 1)
  exponentials = exponentials[diagonal].view(count, count - 1)
  label_differences = label_differences[diagonal].view(count, count - 1)

  loss = features.new_zeros(())
  for candidate in range(count - 1):
    eligible = label_differences >= label_differences[:, candidate, None]
    log_probability = logits[:, candidate] - torch.log(
      (eligible * exponentials).sum(dim=-1)
    )
    loss = loss - log_probability.sum() / (count * (count - 1))
  return loss


def test_vectorized_loss_matches_direct_reference_and_gradients():
  torch.manual_seed(0)
  reference_features = torch.randn(8, 12, requires_grad=True)
  vectorized_features = reference_features.detach().clone().requires_grad_(True)
  labels = torch.tensor([[0.], [0.], [1.], [1.], [2.], [2.], [3.], [3.]])

  expected = _reference_rnc_loss(reference_features, labels)
  actual = RnCLossV2()(vectorized_features, labels)
  expected.backward()
  actual.backward()

  assert torch.allclose(actual, expected, atol=1e-6)
  assert torch.allclose(vectorized_features.grad, reference_features.grad, atol=1e-6)


def test_one_dimensional_labels_match_column_labels():
  torch.manual_seed(1)
  labels = torch.randn(8)
  one_dimensional_features = torch.randn(8, 12, requires_grad=True)
  column_features = one_dimensional_features.detach().clone().requires_grad_(True)
  criterion = RnCLossV2()

  one_dimensional_loss = criterion(one_dimensional_features, labels)
  column_loss = criterion(column_features, labels.unsqueeze(-1))
  one_dimensional_loss.backward()
  column_loss.backward()

  assert torch.allclose(one_dimensional_loss, column_loss, atol=1e-7)
  assert torch.allclose(one_dimensional_features.grad, column_features.grad, atol=1e-7)

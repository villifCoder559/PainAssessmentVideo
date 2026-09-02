"""Continuous target validation, normalization, and diagnostic binning."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


CLASS_ONLY_PRIMARY_LOSSES = frozenset({
  'ce', 'ce_weight', 'cdw_ce', 'huber_ce', 'sim_loss', 'coral',
})
SCALAR_REGRESSION_LOSSES = frozenset({'l1', 'l2', 'huber', 'logcosh'})
REPRESENTATION_LOSSES = frozenset({'contrastive_reg', 'only_contrastive', 'rncloss'})
CLASS_BIN_LOSS_TYPES = frozenset({
  'CrossEntropyLoss', 'CDW_CELoss', 'PHuberCrossEntropy', 'SimLoss',
})


def _validated_values(values):
  try:
    numeric = pd.to_numeric(pd.Series(np.atleast_1d(values)), errors='raise').to_numpy(dtype=np.float64)
  except (TypeError, ValueError) as exc:
    raise ValueError("class_id values must be numeric and finite") from exc
  if numeric.size == 0 or not np.isfinite(numeric).all():
    raise ValueError("class_id values must be numeric and finite")
  return numeric


def round_half_away_from_zero(values):
  """Round array-like values to integers without banker's rounding."""
  values = np.asarray(values, dtype=np.float64)
  return np.copysign(np.floor(np.abs(values) + 0.5), values).astype(np.int64)


@dataclass(frozen=True)
class TargetSpec:
  target_min: float
  target_max: float
  normalize_labels: bool = False
  has_fractional_targets: bool = False

  def __post_init__(self):
    bounds = _validated_values([self.target_min, self.target_max])
    if bounds[0] > bounds[1]:
      raise ValueError("target_min must not exceed target_max")

  @classmethod
  def from_values(cls, values, normalize=False):
    numeric = _validated_values(values)
    fractional = not np.equal(numeric, np.trunc(numeric)).all()
    return cls(float(numeric.min()), float(numeric.max()), bool(normalize), fractional)

  @classmethod
  def from_csv_paths(cls, paths, normalize=False):
    values = []
    for path in paths:
      frame = pd.read_csv(path, sep='\t')
      if 'class_id' not in frame:
        raise ValueError(f"CSV '{path}' is missing required column: class_id")
      values.extend(frame['class_id'].tolist())
    return cls.from_values(values, normalize=normalize)

  @classmethod
  def from_metadata(cls, metadata):
    return cls(
      target_min=metadata['target_min'],
      target_max=metadata['target_max'],
      normalize_labels=metadata.get('normalization') == 'min_max',
      has_fractional_targets=metadata.get('has_fractional_targets', False),
    )

  @property
  def rounded_min(self):
    return int(round_half_away_from_zero([self.target_min])[0])

  @property
  def rounded_max(self):
    return int(round_half_away_from_zero([self.target_max])[0])

  @property
  def bin_offset(self):
    return self.rounded_min

  @property
  def bin_count(self):
    return self.rounded_max - self.rounded_min + 1

  @property
  def span(self):
    return self.target_max - self.target_min

  def normalize(self, values):
    if not self.normalize_labels:
      return values
    if torch.is_tensor(values):
      if self.span == 0:
        return torch.zeros_like(values)
      return (values - self.target_min) / self.span
    values = np.asarray(values)
    if self.span == 0:
      return np.zeros_like(values, dtype=np.float64)
    return (values - self.target_min) / self.span

  def inverse(self, values):
    if not self.normalize_labels:
      return values
    if torch.is_tensor(values):
      if self.span == 0:
        return torch.full_like(values, self.target_min)
      return values * self.span + self.target_min
    values = np.asarray(values)
    if self.span == 0:
      return np.full_like(values, self.target_min, dtype=np.float64)
    return values * self.span + self.target_min

  def to_bins(self, values):
    rounded = round_half_away_from_zero(_validated_values(values))
    return rounded - self.bin_offset

  def to_bin_tensor(self, values):
    values = values if torch.is_tensor(values) else torch.as_tensor(values)
    rounded = torch.copysign(
      torch.floor(torch.abs(values.to(torch.float32)) + 0.5),
      values.to(torch.float32),
    ).to(torch.long)
    return rounded - self.bin_offset

  def predictions_to_bins(self, predictions):
    bins = self.to_bin_tensor(predictions)
    return bins.clamp(0, self.bin_count - 1)

  def to_metadata(self):
    return {
      'target_min': self.target_min,
      'target_max': self.target_max,
      'normalization': 'min_max' if self.normalize_labels else 'none',
      'rounding': 'half_away_from_zero',
      'bin_offset': self.bin_offset,
      'bin_count': self.bin_count,
      'bin_to_rounded_value': list(range(self.bin_offset, self.rounded_max + 1)),
      'has_fractional_targets': self.has_fractional_targets,
    }


def resolve_target_spec(
  metadata=None,
  values=None,
  normalize=False,
  legacy_max_label=None,
  fallback=None,
):
  """Resolve current metadata first, then legacy max-label normalization."""
  if isinstance(metadata, TargetSpec):
    return metadata
  if metadata:
    return TargetSpec.from_metadata(metadata)
  if normalize and legacy_max_label is not None and float(legacy_max_label) != 0:
    return TargetSpec(0.0, float(legacy_max_label), normalize_labels=True)
  if fallback is not None:
    if normalize and not fallback.normalize_labels:
      return TargetSpec(
        fallback.target_min,
        fallback.target_max,
        normalize_labels=True,
        has_fractional_targets=fallback.has_fractional_targets,
      )
    return fallback
  if values is not None:
    return TargetSpec.from_values(values, normalize=normalize)
  return TargetSpec(0.0, 1.0, normalize_labels=False)


def loss_uses_class_bins(loss):
  name = getattr(loss, '__name__', loss.__class__.__name__)
  return name in CLASS_BIN_LOSS_TYPES or name == 'coral_loss'


def validate_primary_losses(loss_names, target_spec):
  """Reject classification objectives that cannot consume fractional labels."""
  names = {str(name).lower() for name in (loss_names or []) if name is not None}
  invalid = sorted(names & CLASS_ONLY_PRIMARY_LOSSES)
  if target_spec.has_fractional_targets and invalid:
    raise ValueError(
      "fractional class_id targets cannot use class-only primary losses: "
      + ', '.join(invalid)
    )


def prepare_batch_targets(batch_inputs, batch_targets, target_spec, use_exact_targets):
  """Remove diagnostic bins from model inputs and select loss targets."""
  class_targets = batch_inputs.pop('class_targets', None)
  if class_targets is None:
    class_targets = (
      torch.argmax(batch_targets, dim=1)
      if batch_targets.ndim > 1
      else target_spec.to_bin_tensor(batch_targets)
    )
  class_targets = class_targets.long()
  if use_exact_targets:
    loss_targets = target_spec.normalize(batch_targets.float())
  elif batch_targets.ndim > 1:
    loss_targets = batch_targets
  else:
    loss_targets = class_targets
  return loss_targets, class_targets


def huber_delta_for_optimization(delta, target_spec):
  if delta is None or not target_spec.normalize_labels or target_spec.span == 0:
    return delta
  return float(delta) / target_spec.span

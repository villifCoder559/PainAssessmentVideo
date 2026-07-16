import torch


def _as_labels(labels):
  return tuple(label.item() if hasattr(label, 'item') else label for label in labels)


def _validate_matrix(matrix, labels):
  if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
    raise ValueError('confusion matrix must be square')
  if len(labels) != matrix.shape[0] or len(set(labels)) != len(labels):
    raise ValueError('labels must be unique and match the matrix axes')


def confusion_axis_labels(matrix, observed_labels):
  """Return labels for every TorchMetrics matrix axis, including implicit gaps."""
  observed_labels = _as_labels(observed_labels)
  if len(observed_labels) == matrix.shape[0]:
    return observed_labels
  implicit_labels = tuple(range(matrix.shape[0]))
  if set(observed_labels).issubset(implicit_labels):
    return implicit_labels
  raise ValueError('observed labels do not match the confusion matrix axes')


def merge_confusion_matrices(accum, matrix, labels):
  """Merge a confusion matrix into ``(counts, axis_labels)`` by label."""
  labels = _as_labels(labels)
  _validate_matrix(matrix, labels)
  if accum is None:
    return matrix.clone(), labels

  accum_matrix, accum_labels = accum
  accum_labels = _as_labels(accum_labels)
  _validate_matrix(accum_matrix, accum_labels)
  all_labels = tuple(sorted(set(accum_labels) | set(labels)))
  positions = {label: idx for idx, label in enumerate(all_labels)}
  merged = accum_matrix.new_zeros((len(all_labels), len(all_labels)))

  for source, source_labels in ((accum_matrix, accum_labels), (matrix, labels)):
    indices = torch.tensor(
      [positions[label] for label in source_labels], device=merged.device
    )
    merged[indices[:, None], indices[None, :]] += source.to(
      device=merged.device, dtype=merged.dtype
    )

  return merged, all_labels


def weighted_means_by_label(records):
  """Return sample-weighted means from ``(labels, means, supports)`` records."""
  totals = {}
  for labels, means, supports in records:
    labels, means, supports = list(labels), list(means), list(supports)
    if not (len(labels) == len(means) == len(supports)):
      raise ValueError('labels, means and supports must have matching lengths')
    for label, mean, support in zip(labels, means, supports):
      label = label.item() if hasattr(label, 'item') else label
      support = float(support)
      if support <= 0:
        raise ValueError('supports must be positive')
      weighted_sum, total_support = totals.get(label, (0.0, 0.0))
      totals[label] = weighted_sum + float(mean) * support, total_support + support
  return {label: weighted_sum / support for label, (weighted_sum, support) in totals.items()}

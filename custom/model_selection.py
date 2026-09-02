import warnings


CE_LOSS_NAMES = frozenset({'ce', 'ce_weight', 'cdw_ce', 'huber_ce'})
VALID_SELECTION_METRICS = frozenset({'val_accuracy', 'val_loss'})


def required_metric_for_losses(loss_names):
  normalized = [name.lower() for name in loss_names]
  metric_families = {
    'accuracy' if name in CE_LOSS_NAMES else 'loss'
    for name in normalized
  }
  if len(metric_families) > 1:
    raise ValueError(
      'A single Optuna study cannot mix accuracy-driven CE losses with '
      'loss-driven objectives. Run them as separate studies.'
    )
  if metric_families == {'accuracy'}:
    return 'val_accuracy'
  return 'val_loss'


def resolve_selection_metric(loss_names, requested_metric=None):
  required_metric = required_metric_for_losses(loss_names or [])
  if requested_metric is None or requested_metric == required_metric:
    return required_metric
  if requested_metric not in VALID_SELECTION_METRICS:
    raise ValueError(
      f'Invalid key for early stopping: {requested_metric}. '
      f'Expected one of {sorted(VALID_SELECTION_METRICS)}.'
    )
  warnings.warn(
    f'Replacing --key_early_stopping={requested_metric} with '
    f'{required_metric} because the selected loss family requires it.',
    UserWarning,
    stacklevel=2,
  )
  return required_metric


def is_better_metric(candidate, incumbent, metric_name):
  if metric_name == 'val_accuracy':
    return candidate > incumbent
  if metric_name == 'val_loss':
    return candidate < incumbent
  raise ValueError(
    f'Unsupported model-selection metric: {metric_name}. '
    f'Expected one of {sorted(VALID_SELECTION_METRICS)}.'
  )


def optuna_direction_for_metric(metric_name):
  if metric_name == 'val_accuracy':
    return 'maximize'
  if metric_name == 'val_loss':
    return 'minimize'
  raise ValueError(
    f'Unsupported model-selection metric: {metric_name}. '
    f'Expected one of {sorted(VALID_SELECTION_METRICS)}.'
  )


def threshold_pruner_kwargs(metric_name, threshold):
  if metric_name == 'val_accuracy':
    return {'lower': threshold}
  if metric_name == 'val_loss':
    return {'upper': threshold}
  raise ValueError(
    f'Unsupported model-selection metric: {metric_name}. '
    f'Expected one of {sorted(VALID_SELECTION_METRICS)}.'
  )


def best_metric_index(metric_values, metric_name):
  if not metric_values:
    raise ValueError('Cannot select a best model from an empty metric list.')
  best_index = 0
  for index, value in enumerate(metric_values[1:], start=1):
    if is_better_metric(value, metric_values[best_index], metric_name):
      best_index = index
  return best_index


def should_select_epoch(
  incumbent_metric,
  candidate_metric,
  metric_name,
  *,
  has_validation,
):
  if not has_validation:
    return True
  if incumbent_metric is None:
    return True
  return is_better_metric(candidate_metric, incumbent_metric, metric_name)

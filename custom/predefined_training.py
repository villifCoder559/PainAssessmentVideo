"""Single-run training support for predefined CSV splits."""

from copy import deepcopy
from pathlib import Path
import shutil


def run_predefined_split(
  model_advanced,
  predefined_csv_splits,
  augmented_train_csv_path,
  train_folder_path,
  seed_random_state,
  train_kwargs,
  test_kwargs,
  reduce_logs,
  set_seed,
):
  """Train once on fixed train/val/test CSVs and return fold-shaped results."""
  fold_path = Path(train_folder_path) / 'k0_cross_val'
  subfold_path = fold_path / 'k0_cross_val_sub_0'
  subfold_path.mkdir(parents=True, exist_ok=True)

  source_train_path = subfold_path / 'source_train.csv'
  train_path = subfold_path / 'train.csv'
  val_path = subfold_path / 'val.csv'
  test_path = fold_path / 'test.csv'
  shutil.copy2(predefined_csv_splits['train'], source_train_path)
  shutil.copy2(augmented_train_csv_path, train_path)
  shutil.copy2(predefined_csv_splits['val'], val_path)
  shutil.copy2(predefined_csv_splits['test'], test_path)

  set_seed(seed_random_state)
  dict_train = model_advanced.train(
    saving_path=str(subfold_path),
    train_csv_path=str(train_path),
    val_csv_path=str(val_path),
    **train_kwargs,
  )
  dict_results = dict_train['dict_results']
  best_epoch = dict_results['best_model_idx']
  best_model = {
    'best_model_idx': best_epoch,
    'best_model_state': deepcopy(dict_results['best_model_state']),
    'metric_for_stopping': train_kwargs['key_for_early_stopping'],
    'train_metric_value': dict_results['list_train_performance_metric'][best_epoch],
    'val_metric_value': dict_results['list_val_performance_metric'][best_epoch],
    'fold_sub_fold_idx': (0, 0),
  }

  state_dict = best_model['best_model_state']
  dict_test = model_advanced.test_pretrained_model(
    path_model_weights=None,
    state_dict=state_dict,
    csv_path=str(test_path),
    is_test=True,
    **test_kwargs,
  )

  dict_results['best_model_state'] = None
  best_model['best_model_state'] = None
  reduced_train = reduce_logs(dict_train)
  return {
    'k0_cross_val_sub_0': {
      'train_val': reduced_train,
      'test': dict_test,
    },
    'k0_cross_val_final': {
      'test': dict_test,
      'best_model': best_model,
      'train_val': reduced_train,
    },
  }

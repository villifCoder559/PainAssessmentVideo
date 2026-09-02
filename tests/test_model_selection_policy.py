import unittest
import sys
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custom.model_selection import (
  best_metric_index,
  is_better_metric,
  optuna_direction_for_metric,
  required_metric_for_losses,
  resolve_selection_metric,
  should_select_epoch,
  threshold_pruner_kwargs,
)


class TestRequiredMetricForLosses(unittest.TestCase):
  def test_ce_family_uses_validation_accuracy(self):
    for loss_name in ('ce', 'ce_weight', 'cdw_ce', 'huber_ce'):
      with self.subTest(loss_name=loss_name):
        self.assertEqual(
          required_metric_for_losses([loss_name]),
          'val_accuracy',
        )

  def test_mixed_metric_families_are_rejected(self):
    with self.assertRaisesRegex(
      ValueError,
      'separate studies',
    ):
      required_metric_for_losses(['ce', 'l1'])

  def test_every_other_loss_family_uses_validation_loss(self):
    for loss_names in (['l1'], ['l2'], ['huber'], ['sim_loss'], []):
      with self.subTest(loss_names=loss_names):
        self.assertEqual(required_metric_for_losses(loss_names), 'val_loss')


class TestResolveSelectionMetric(unittest.TestCase):
  def test_conflicting_valid_override_is_replaced_with_warning(self):
    with self.assertWarnsRegex(
      UserWarning,
      'Replacing --key_early_stopping=val_loss with val_accuracy',
    ):
      resolved = resolve_selection_metric(['ce'], 'val_loss')

    self.assertEqual(resolved, 'val_accuracy')

  def test_non_ce_conflicting_override_is_replaced_with_loss(self):
    with self.assertWarnsRegex(
      UserWarning,
      'Replacing --key_early_stopping=val_accuracy with val_loss',
    ):
      resolved = resolve_selection_metric(['l1'], 'val_accuracy')

    self.assertEqual(resolved, 'val_loss')

  def test_invalid_override_is_rejected_instead_of_replaced(self):
    with self.assertRaisesRegex(ValueError, 'Invalid key for early stopping'):
      resolve_selection_metric(['ce'], 'val_macro_precision')


class TestMetricDirection(unittest.TestCase):
  def test_comparison_maximizes_accuracy_and_minimizes_loss(self):
    self.assertTrue(is_better_metric(0.8, 0.7, 'val_accuracy'))
    self.assertFalse(is_better_metric(0.7, 0.8, 'val_accuracy'))
    self.assertTrue(is_better_metric(0.2, 0.3, 'val_loss'))
    self.assertFalse(is_better_metric(0.3, 0.2, 'val_loss'))

  def test_equal_metric_keeps_the_existing_selection(self):
    self.assertFalse(is_better_metric(0.8, 0.8, 'val_accuracy'))
    self.assertFalse(is_better_metric(0.2, 0.2, 'val_loss'))

  def test_optuna_direction_matches_the_selection_metric(self):
    self.assertEqual(optuna_direction_for_metric('val_accuracy'), 'maximize')
    self.assertEqual(optuna_direction_for_metric('val_loss'), 'minimize')

  def test_best_subfold_index_uses_the_metric_direction(self):
    self.assertEqual(
      best_metric_index([0.7, 0.9, 0.8], 'val_accuracy'),
      1,
    )
    self.assertEqual(
      best_metric_index([0.9, 0.4, 0.2], 'val_loss'),
      2,
    )

  def test_best_subfold_index_keeps_lower_index_on_a_tie(self):
    self.assertEqual(
      best_metric_index([0.9, 0.9], 'val_accuracy'),
      0,
    )

  def test_epoch_selection_is_metric_driven_with_validation(self):
    self.assertTrue(
      should_select_epoch(None, 0.50, 'val_accuracy', has_validation=True)
    )
    self.assertTrue(
      should_select_epoch(0.50, 0.80, 'val_accuracy', has_validation=True)
    )
    self.assertFalse(
      should_select_epoch(0.80, 0.70, 'val_accuracy', has_validation=True)
    )

  def test_epoch_selection_always_advances_without_validation(self):
    selected_epoch = None
    for epoch in range(3):
      if should_select_epoch(
        None, None, 'val_loss', has_validation=False
      ):
        selected_epoch = epoch

    self.assertEqual(selected_epoch, 2)

  def test_threshold_pruner_uses_the_metric_direction(self):
    def should_prune(metric_name, value):
      pruner = optuna.pruners.ThresholdPruner(
        **threshold_pruner_kwargs(metric_name, 0.2),
        n_warmup_steps=0,
      )
      study = optuna.create_study(
        direction=optuna_direction_for_metric(metric_name),
        pruner=pruner,
      )
      trial = study.ask()
      trial.report(value, step=0)
      return trial.should_prune()

    self.assertFalse(should_prune('val_loss', 0.1))
    self.assertTrue(should_prune('val_loss', 0.3))
    self.assertFalse(should_prune('val_accuracy', 0.3))
    self.assertTrue(should_prune('val_accuracy', 0.1))


if __name__ == '__main__':
  unittest.main()

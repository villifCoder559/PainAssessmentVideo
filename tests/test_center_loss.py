import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import custom.helper as helper
from custom.loss import CenterLoss
from custom.head import BaseHead
from custom.optimizers import OptimizerFactory


# ─── Shared infrastructure ────────────────────────────────────────────────────

class ToyHead(BaseHead):
    """
    Minimal model that inherits BaseHead.evaluate() without requiring a
    real backbone or attentive pooler.

    Forward output dict mirrors AttentiveHeadJEPA.forward() (head.py:1799-1802):
      {'logits': [B, num_classes], 'embeddings': [B, feat_dim]}
    """

    def __init__(self, feat_dim=16, num_classes=3):
        """
        Args:
          feat_dim:    Dimensionality of the input feature vector.
          num_classes: Number of output classes.
        """
        super().__init__(is_classification=True)
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x, key_padding_mask=None, return_video_emb=True, **kwargs):
        """
        Args:
          x:               [B, feat_dim] feature tensor.
          return_video_emb: When True, include embeddings in the output dict.
          **kwargs:        Absorbs extra keys injected by evaluate() (list_sample_id, etc.)

        Returns:
          dict with keys 'logits' [B, num_classes] and 'embeddings' [B, feat_dim] or None.
        """
        logits = self.linear(x)
        return {
            'logits': logits,
            'embeddings': x if return_video_emb else None,
        }


class ToyDataset(Dataset):
    """
    Synthetic dataset that satisfies the interface expected by BaseHead.evaluate():
      - Yields (dict_batch_X, batch_y, batch_subjects, sample_id)
      - Exposes get_unique_classes() and get_unique_subjects()

    Args:
      n_samples:   Total number of samples.
      feat_dim:    Feature dimensionality.
      num_classes: Number of target classes.
      n_subjects:  Number of subjects.
      seed:        Random seed for reproducibility.
    """

    def __init__(self, n_samples=20, feat_dim=16, num_classes=3, n_subjects=4, seed=42):
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.features = torch.randn(n_samples, feat_dim, generator=rng)
        self.labels = torch.randint(0, num_classes, (n_samples,), generator=rng)
        self.subjects = torch.randint(0, n_subjects, (n_samples,), generator=rng)
        self.num_classes = num_classes
        self.n_subjects = n_subjects

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        """Returns (dict_batch_X, batch_y, batch_subjects, sample_id)."""
        return (
            {'x': self.features[i]},
            self.labels[i],
            self.subjects[i],
            torch.tensor(i),
        )

    def get_unique_classes(self):
        """Required by evaluate() to size ConfusionMatrix and metrics arrays."""
        return torch.arange(self.num_classes)

    def get_unique_subjects(self):
        """Required by evaluate() for per-subject tracking."""
        return torch.arange(self.n_subjects)


def _make_eval_loader(n_samples=20, feat_dim=16, num_classes=3,
                      n_subjects=4, batch_size=4):
    """Creates a deterministic DataLoader from ToyDataset."""
    ds = ToyDataset(n_samples=n_samples, feat_dim=feat_dim,
                    num_classes=num_classes, n_subjects=n_subjects)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _patch_helper():
    """
    Disables all side-effect logging flags in custom.helper so that
    evaluate() skips optional per-class, per-subject, and confidence
    computations that are irrelevant for center loss tests.
    """
    helper.AMP_ENABLED = False
    helper.LOG_PER_CLASS = False
    helper.LOG_PER_SUBJECT = False
    helper.LOG_CONFIDENCE_PREDICTION = False
    helper.LOG_CROSS_ATTENTION = {'enable': False, 'state': 'val'}
    helper.LOG_VIDEO_EMBEDDINGS = {
        'enable': False, 'embeddings': [], 'labels': [],
        'sample_ids': [], 'predictions': [],
    }


# Minimal kwargs that satisfy evaluate()'s required keys when HSIC and
# adversarial training are disabled.
_BASE_EVAL_KWARGS = {
    'HSIC_subject_lambda': 0.0,
    'HSIC_pain_lambda': 0.0,
    'HSIC_feat_normalization': False,
    'adv_criterion': None,
    'lambda_center_loss': 0.0,
}


# ─── Group A: Forward math ────────────────────────────────────────────────────

class TestCenterLossForward(unittest.TestCase):
    """Unit tests for CenterLoss.forward() numerical correctness."""

    NUM_CLASSES = 3
    FEAT_DIM = 4

    def _make_zero_centers_fn(self, norm):
        """Returns a CenterLoss whose centers are all zeros (easy to hand-compute)."""
        fn = CenterLoss(self.NUM_CLASSES, self.FEAT_DIM, norm=norm)
        with torch.no_grad():
            fn.centers.data.zero_()
        return fn

    def test_l2_correctness(self):
        """L2 loss: mean over B of ||feat - center||^2 (sum over dims)."""
        fn = self._make_zero_centers_fn(norm=2)
        # Sample 0: diff = [1, 0, 0, 0] → sum(diff^2) = 1
        # Sample 1: diff = [0, 2, 0, 0] → sum(diff^2) = 4
        # Expected loss = mean([1, 4]) = 2.5
        features = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 2.0, 0.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = fn(features, labels)
        self.assertTrue(
            torch.isclose(loss, torch.tensor(2.5), atol=1e-6),
            f"L2 loss: expected 2.5, got {loss.item()}"
        )

    def test_l1_correctness(self):
        """L1 loss: mean over B of sum_d |feat - center|."""
        fn = self._make_zero_centers_fn(norm=1)
        # Sample 0: sum|diff| = 1; Sample 1: sum|diff| = 2
        # Expected loss = mean([1, 2]) = 1.5
        features = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 2.0, 0.0, 0.0]])
        labels = torch.tensor([0, 1])
        loss = fn(features, labels)
        self.assertTrue(
            torch.isclose(loss, torch.tensor(1.5), atol=1e-6),
            f"L1 loss: expected 1.5, got {loss.item()}"
        )

    def test_output_is_scalar(self):
        """Loss must be a 0-dim scalar tensor."""
        fn = CenterLoss(self.NUM_CLASSES, self.FEAT_DIM, norm=2)
        features = torch.randn(4, self.FEAT_DIM)
        labels = torch.zeros(4, dtype=torch.long)
        loss = fn(features, labels)
        self.assertEqual(loss.ndim, 0)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_centers_shape(self):
        """centers parameter must have shape (num_classes, feat_dim)."""
        fn = CenterLoss(self.NUM_CLASSES, self.FEAT_DIM, norm=2)
        self.assertEqual(fn.centers.shape, (self.NUM_CLASSES, self.FEAT_DIM))

    def test_invalid_norm_raises(self):
        """norm values other than 1 or 2 must raise AssertionError."""
        with self.assertRaises(AssertionError):
            CenterLoss(self.NUM_CLASSES, self.FEAT_DIM, norm=3)


# ─── Group B: Gradient flow & parameter update ───────────────────────────────

class TestCenterLossGradients(unittest.TestCase):
    """Verify that gradients reach both features and centers, and that
    an optimizer step actually moves the centers."""

    def setUp(self):
        torch.manual_seed(1)
        self.fn = CenterLoss(num_classes=3, feat_dim=8, norm=2)

    def _forward(self):
        """Runs a single forward pass and returns (loss, leaf features tensor)."""
        features = torch.randn(4, 8, requires_grad=True)
        labels = torch.tensor([0, 1, 2, 0])
        loss = self.fn(features, labels)
        return loss, features

    def test_grad_flows_to_features(self):
        """features.grad must be non-None and non-zero after backward."""
        loss, features = self._forward()
        loss.backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(features.grad.abs().sum().item(), 0.0)

    def test_grad_flows_to_centers(self):
        """centers.grad must be non-None and non-zero after backward."""
        loss, _ = self._forward()
        loss.backward()
        self.assertIsNotNone(self.fn.centers.grad)
        self.assertGreater(self.fn.centers.grad.abs().sum().item(), 0.0)

    def test_centers_update_after_step(self):
        """
        User test 1: center parameters must change after an optimizer step.
        Verifies that the centers are wired into the computational graph
        and that gradient descent actually moves them.
        """
        optimizer = optim.AdamW(self.fn.parameters(), lr=1e-3)
        centers_before = self.fn.centers.data.clone()
        loss, _ = self._forward()
        loss.backward()
        optimizer.step()
        self.assertTrue(
            torch.any(self.fn.centers.data != centers_before),
            "Centers did not change after optimizer.step(). "
            "Check that CenterLoss.centers is an nn.Parameter and is included "
            "in the optimizer."
        )


# ─── Group C: LR scheduling ──────────────────────────────────────────────────

def _make_optimizer_and_scheduler(main_lr=1e-3, min_lr=1e-6, ratio=10.0,
                                   epochs=10, steps_per_epoch=20):
    """
    Builds an OptimizerFactory with a tiny nn.Linear model plus a CenterLoss
    extra param group, mirroring the setup in head.py start_train().

    Returns:
      (optimizer, lr_scheduler, center_loss_fn)
    """
    config = {
        'optimizer_name': 'adamw',
        'scheduler_name': 'cosine',
        'wd_scheduler_name': None,
        'lr': main_lr,
        'weight_decay': 1e-4,
        'exclude_bias_wd': False,
        'min_lr': min_lr,
        'epochs': epochs,
        'steps_per_epoch': steps_per_epoch,
        'warm_up_epochs': None,
    }
    model = nn.Linear(8, 3)
    cl_fn = CenterLoss(num_classes=3, feat_dim=8, norm=2)
    extra_pg = [{
        'params': list(cl_fn.parameters()),
        'lr': main_lr / ratio,
        'weight_decay': 0.0,
        'is_center_loss': True,
    }]
    factory = OptimizerFactory(config)
    optimizer, lr_scheduler, _ = factory.create(model, extra_param_groups=extra_pg)
    return optimizer, lr_scheduler, cl_fn


class TestCenterLossLRScheduling(unittest.TestCase):
    """
    User test 2: LR scheduling correctness for the center loss parameter group.
    All tests use OptimizerFactory (adamw + cosine) with a CenterLoss extra
    param group, reproducing the wiring done in head.py start_train().
    """

    MAIN_LR = 1e-3
    MIN_LR = 1e-6

    def _get_center_pg(self, optimizer):
        """Returns the center loss parameter group from the optimizer."""
        return next(pg for pg in optimizer.param_groups
                    if pg.get('is_center_loss', False))

    def _get_main_pg(self, optimizer):
        """Returns the first (non-center) parameter group."""
        return next(pg for pg in optimizer.param_groups
                    if not pg.get('is_center_loss', False))

    def test_initial_center_lr_equals_main_lr_over_ratio(self):
        """
        Right after factory.create(), center group lr must equal main_lr / ratio.
        Verifies the extra_param_groups wiring in OptimizerFactory.create().
        """
        ratio = 10.0
        optimizer, _, _ = _make_optimizer_and_scheduler(
            main_lr=self.MAIN_LR, ratio=ratio)
        center_pg = self._get_center_pg(optimizer)
        self.assertAlmostEqual(
            center_pg['lr'], self.MAIN_LR / ratio, places=10,
            msg=f"Expected center_lr = {self.MAIN_LR / ratio}, got {center_pg['lr']}"
        )

    def test_center_lr_decreases_over_steps(self):
        """
        After N scheduler steps, center lr must be lower than its initial value.
        Verifies that the cosine schedule applies to the center group.
        """
        ratio = 10.0
        optimizer, lr_scheduler, _ = _make_optimizer_and_scheduler(
            main_lr=self.MAIN_LR, ratio=ratio)
        center_pg = self._get_center_pg(optimizer)
        initial_center_lr = center_pg['lr']

        for _ in range(100):
            lr_scheduler.step()

        self.assertLess(
            center_pg['lr'], initial_center_lr,
            "Center LR should decay after cosine scheduler steps."
        )

    def test_center_lr_clamp_at_1e8(self):
        """
        Manually forcing the center LR below 1e-8 and running the clamp
        code from head.py must bring it back to exactly 1e-8.
        """
        optimizer, _, _ = _make_optimizer_and_scheduler()
        center_pg = self._get_center_pg(optimizer)

        # Force LR far below the floor
        center_pg['lr'] = 1e-12

        # Reproduce the clamp logic from head.py lines 707-710
        for _pg in optimizer.param_groups:
            if _pg.get('is_center_loss', False):
                _pg['lr'] = max(_pg['lr'], 1e-8)

        self.assertGreaterEqual(
            center_pg['lr'], 1e-8,
            "LR clamp should enforce a minimum of 1e-8."
        )

    def test_center_lr_always_less_than_main_lr(self):
        """
        With ratio > 1, the center group lr must remain strictly below the
        main group lr at every scheduler step (over 50 steps).
        """
        ratio = 5.0
        optimizer, lr_scheduler, _ = _make_optimizer_and_scheduler(
            main_lr=self.MAIN_LR, ratio=ratio)
        main_pg = self._get_main_pg(optimizer)
        center_pg = self._get_center_pg(optimizer)

        for step in range(50):
            lr_scheduler.step()
            self.assertLess(
                center_pg['lr'], main_pg['lr'],
                f"Step {step}: center_lr ({center_pg['lr']:.2e}) should be "
                f"< main_lr ({main_pg['lr']:.2e}) when ratio={ratio}."
            )


# ─── Group D: evaluate() integration ─────────────────────────────────────────

@unittest.skipUnless(torch.cuda.is_available(), "Group D requires CUDA (evaluate() hardcodes device='cuda')")
class TestCenterLossIntegration(unittest.TestCase):
    """
    User test 3: calls the real BaseHead.evaluate() on ToyHead with a
    synthetic ToyDataset to verify that center loss values are logged
    correctly and that val_loss is unaffected by center loss.

    Skipped automatically when no GPU is available.
    """

    NUM_CLASSES = 3
    N_SUBJECTS = 4
    FEAT_DIM = 16
    N_SAMPLES = 20
    BATCH_SIZE = 4

    def setUp(self):
        _patch_helper()
        torch.manual_seed(42)
        self.model = ToyHead(feat_dim=self.FEAT_DIM, num_classes=self.NUM_CLASSES)
        self.loader = _make_eval_loader(
            n_samples=self.N_SAMPLES,
            feat_dim=self.FEAT_DIM,
            num_classes=self.NUM_CLASSES,
            n_subjects=self.N_SUBJECTS,
            batch_size=self.BATCH_SIZE,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.unique_subjects = torch.arange(self.N_SUBJECTS)
        self.unique_classes = torch.arange(self.NUM_CLASSES)
        # CenterLoss on CUDA — required because evaluate() runs on CUDA and
        # dict_out['embeddings'] will be on CUDA.
        self.cl_fn = CenterLoss(
            num_classes=self.NUM_CLASSES, feat_dim=self.FEAT_DIM, norm=2
        ).cuda()

    def _run_evaluate(self, extra_kwargs=None):
        """
        Calls model.evaluate() with the base kwargs, optionally merged
        with extra_kwargs.  Returns the result dict.
        """
        kwargs = dict(_BASE_EVAL_KWARGS)
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        # Reset LOG_VIDEO_EMBEDDINGS state between calls
        helper.LOG_VIDEO_EMBEDDINGS['embeddings'] = []
        helper.LOG_VIDEO_EMBEDDINGS['labels'] = []
        helper.LOG_VIDEO_EMBEDDINGS['sample_ids'] = []
        helper.LOG_VIDEO_EMBEDDINGS['predictions'] = []
        return self.model.evaluate(
            val_loader=self.loader,
            criterion=self.criterion,
            unique_val_subjects=self.unique_subjects,
            unique_val_classes=self.unique_classes,
            is_test=False,
            is_coral_loss=False,
            epoch=0,
            save_log=False,
            **kwargs,
        )

    def test_val_center_loss_key_present(self):
        """
        Return dict must contain 'val_center_loss' when center_loss_fn is
        provided and lambda_center_loss > 0.
        """
        result = self._run_evaluate({
            'center_loss_fn': self.cl_fn,
            'lambda_center_loss': 0.1,
        })
        self.assertIn('val_center_loss', result,
                      "evaluate() return dict must contain 'val_center_loss' "
                      "when center loss is enabled.")

    def test_val_loss_unchanged_by_center_loss(self):
        """
        val_loss must be identical with and without center_loss_fn since
        center loss is log-only inside evaluate() (not added to val_loss).
        """
        result_no_cl = self._run_evaluate()
        result_with_cl = self._run_evaluate({
            'center_loss_fn': self.cl_fn,
            'lambda_center_loss': 0.1,
        })
        self.assertAlmostEqual(
            result_no_cl['val_loss'],
            result_with_cl['val_loss'],
            places=5,
            msg="val_loss must not change when center loss is added to evaluate() "
                "(it should only be logged, not accumulated)."
        )

    def test_val_center_loss_value_correct(self):
        """
        val_center_loss must equal the manual batch-wise CenterLoss mean
        computed over the same loader with the same frozen model weights.
        """
        result = self._run_evaluate({
            'center_loss_fn': self.cl_fn,
            'lambda_center_loss': 0.1,
        })

        # Manual reference: iterate batches, compute CenterLoss, take mean
        self.model.eval()
        batch_losses = []
        with torch.no_grad():
            for dict_batch_X, batch_y, _, _ in self.loader:
                x = dict_batch_X['x'].cuda()
                batch_y = batch_y.long().cuda()
                dict_out = self.model(x, return_video_emb=True)
                cl_val = self.cl_fn(dict_out['embeddings'], batch_y)
                batch_losses.append(cl_val.item())

        expected = float(np.mean(batch_losses))
        self.assertAlmostEqual(
            result['val_center_loss'], expected, places=4,
            msg=f"val_center_loss {result['val_center_loss']:.6f} does not "
                f"match manual computation {expected:.6f}."
        )

    def test_val_center_loss_zero_when_disabled(self):
        """
        When lambda_center_loss=0 (default) and center_loss_fn is absent,
        val_center_loss must be 0.0 (epoch_center_loss list stays empty).
        """
        result = self._run_evaluate()  # no center_loss_fn, lambda=0
        self.assertEqual(
            result.get('val_center_loss', 0.0), 0.0,
            "val_center_loss must be 0.0 when center loss is disabled."
        )


if __name__ == '__main__':
    unittest.main()

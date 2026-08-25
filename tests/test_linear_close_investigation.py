import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.linear_close_investigation import (
    affine_metrics,
    compare_experiment_roots,
    checkpoint_diagnostics,
    fit_affine_adam,
    fit_affine_closed_form,
    fit_affine_ridge,
)
from analysis.linear_close_synthetic import (
    anchor_count_sweep,
    cross_domain_nuisance_experiment,
    scaling_experiment,
)


class ClosedFormSanityTest(unittest.TestCase):
    """Break caught: a transpose or intercept error in the diagnostic OLS implementation."""

    def test_recovers_known_well_determined_affine_mapping(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 6))
        weight = rng.normal(size=(6, 4))
        bias = np.array([3.0, -2.0, 0.5, 7.0])
        y = x @ weight + bias

        fit = fit_affine_closed_form(x, y, rcond=None)

        np.testing.assert_allclose(fit.weight, weight, atol=1e-10)
        np.testing.assert_allclose(fit.bias, bias, atol=1e-10)
        self.assertEqual(fit.rank, 6)
        self.assertLess(affine_metrics(fit, x, y)["mse"], 1e-20)

    def test_intercept_is_necessary_for_translated_mapping(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(120, 5))
        weight = rng.normal(size=(5, 3))
        bias = np.array([10.0, -8.0, 4.0])
        y = x @ weight + bias

        with_intercept = fit_affine_closed_form(x, y, intercept=True, rcond=None)
        without_intercept = fit_affine_closed_form(x, y, intercept=False, rcond=None)

        self.assertLess(affine_metrics(with_intercept, x, y)["mse"], 1e-20)
        self.assertGreater(affine_metrics(without_intercept, x, y)["mse"], 40.0)

    def test_production_solution_matches_numpy_torch_and_sklearn(self):
        """A solver/materialization bug would disagree on held-out predictions."""
        from sklearn.linear_model import LinearRegression

        with mock.patch("multiprocessing.Manager") as manager:
            manager.return_value.dict.return_value = {}
            from cross_space_projection import _fit_linear_closed_form

        rng = np.random.default_rng(19)
        x = rng.normal(size=(90, 12)).astype(np.float32)
        y = rng.normal(size=(90, 7)).astype(np.float32)
        probe = rng.normal(size=(25, 12)).astype(np.float32)

        production = _fit_linear_closed_form(x, y, rcond=None)
        independent = fit_affine_closed_form(x, y, rcond=None)
        sklearn_fit = LinearRegression().fit(x, y)
        augmented = torch.column_stack((torch.from_numpy(x).double(), torch.ones(len(x), 1)))
        torch_coeff = torch.linalg.lstsq(augmented, torch.from_numpy(y).double()).solution

        production_prediction = probe @ production["weight"].T + production["bias"]
        np.testing.assert_allclose(production_prediction, independent.predict(probe), atol=2e-6)
        np.testing.assert_allclose(production_prediction, sklearn_fit.predict(probe), atol=2e-5)
        np.testing.assert_allclose(
            production_prediction,
            torch.column_stack((torch.from_numpy(probe).double(), torch.ones(len(probe), 1)))
            .matmul(torch_coeff).numpy(),
            atol=2e-6,
        )

    def test_checkpoint_guard_accepts_exact_formula_parameter_quantization(self):
        """Large cancellation must not reject correctly quantized OLS parameters."""
        with mock.patch("multiprocessing.Manager") as manager:
            manager.return_value.dict.return_value = {}
            from cross_space_projection import (
                _assert_linear_ckpt_matches_quantized_formula,
                _fit_linear_closed_form,
            )

        rng = np.random.default_rng(0)
        left, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        right, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        singular_values = np.geomspace(1.0, 1e-5, 12)
        x = (0.01 + (left * singular_values) @ right.T).astype(np.float32)
        y = rng.normal(size=(12, 20)).astype(np.float32)

        solution = _fit_linear_closed_form(x, y, rcond=1e-5)

        x64 = x.astype(np.float64)
        y64 = y.astype(np.float64)
        x_mean = x64.mean(axis=0)
        y_mean = y64.mean(axis=0)
        expected_weight_t = np.linalg.lstsq(
            x64 - x_mean, y64 - y_mean, rcond=1e-5,
        )[0]
        expected_weight = expected_weight_t.T
        expected_bias = y_mean - x_mean @ expected_weight_t
        expected_prediction = (x64 - x_mean) @ expected_weight_t + y_mean
        checkpoint_prediction = (
            x64 @ solution["weight"].astype(np.float64).T
            + solution["bias"].astype(np.float64)
        )
        relative_output_error = (
            np.max(np.abs(checkpoint_prediction - expected_prediction))
            / np.max(np.abs(expected_prediction))
        )

        self.assertGreater(relative_output_error, 1e-4)
        np.testing.assert_allclose(
            solution["weight_float64"], expected_weight, rtol=0, atol=1e-12,
        )
        np.testing.assert_allclose(
            solution["bias_float64"], expected_bias, rtol=0, atol=1e-12,
        )
        _assert_linear_ckpt_matches_quantized_formula(
            solution["weight"],
            solution["bias"],
            solution["weight_float64"],
            solution["bias_float64"],
            "linear_close",
            "cancellation_regression",
        )

        wrong_bias = solution["bias"].copy()
        wrong_bias[0] = np.nextafter(wrong_bias[0], np.float32(np.inf))
        with self.assertRaises(AssertionError):
            _assert_linear_ckpt_matches_quantized_formula(
                solution["weight"],
                wrong_bias,
                solution["weight_float64"],
                solution["bias_float64"],
                "linear_close",
                "cancellation_regression",
            )


class RegularizationSanityTest(unittest.TestCase):
    """Break caught: ridge accidentally regularizes the intercept or increases unstable weights."""

    def test_ridge_reduces_weight_norm_in_underdetermined_noisy_fit(self):
        rng = np.random.default_rng(23)
        x_train = rng.normal(size=(20, 80))
        true_weight = rng.normal(size=(80, 5)) / np.sqrt(80)
        bias = np.arange(5, dtype=float)
        y_train = x_train @ true_weight + bias + rng.normal(scale=0.5, size=(20, 5))

        ols = fit_affine_closed_form(x_train, y_train, rcond=None)
        ridge = fit_affine_ridge(x_train, y_train, alpha=20.0)

        self.assertLess(np.linalg.norm(ridge.weight), np.linalg.norm(ols.weight))
        np.testing.assert_allclose(ridge.bias, y_train.mean(0) - x_train.mean(0) @ ridge.weight)

    def test_zero_ridge_matches_minimum_norm_ols_when_underdetermined(self):
        """Break caught: normal equations choose an unstable non-pseudoinverse solution."""
        rng = np.random.default_rng(29)
        x = rng.normal(size=(12, 30))
        y = rng.normal(size=(12, 4))

        ols = fit_affine_closed_form(x, y, rcond=None)
        zero_ridge = fit_affine_ridge(x, y, alpha=0.0)

        np.testing.assert_allclose(zero_ridge.weight, ols.weight, atol=1e-10)
        np.testing.assert_allclose(zero_ridge.bias, ols.bias, atol=1e-10)

    def test_learned_affine_map_recovers_well_determined_mapping(self):
        rng = np.random.default_rng(31)
        x = rng.normal(size=(160, 4)).astype(np.float32)
        weight = rng.normal(size=(4, 3)).astype(np.float32)
        bias = np.array([1.0, -2.0, 0.5], dtype=np.float32)
        y = x @ weight + bias

        result = fit_affine_adam(
            x[:120], y[:120], x[120:], y[120:],
            lr=0.03, epochs=500, batch_size=120, seed=5,
        )

        self.assertLess(result.val_mse, 1e-7)
        self.assertGreater(result.best_epoch, 1)


class SafeArtifactDiagnosticTest(unittest.TestCase):
    """Break caught: mismatched subtrials or checkpoints are silently compared as paired."""

    def test_checkpoint_diagnostics_reports_parameter_norms_and_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projector.pt"
            torch.save({
                "weight": torch.tensor([[3.0, 0.0], [0.0, 4.0]]),
                "bias": torch.tensor([6.0, 8.0]),
            }, path)

            result = checkpoint_diagnostics(path)

        self.assertEqual(result["weight_shape"], [2, 2])
        self.assertEqual(result["weight_rank"], 2)
        self.assertAlmostEqual(result["weight_norm"], 5.0)
        self.assertAlmostEqual(result["bias_norm"], 10.0)

    def test_checkpoint_diagnostics_rejects_missing_bias(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "projector.pt"
            torch.save({"weight": torch.eye(2), "unrelated": torch.zeros(1)}, path)

            with self.assertRaisesRegex(ValueError, "lacks weight/bias"):
                checkpoint_diagnostics(path)

    def test_experiment_comparison_verifies_csv_identity_and_pairs_by_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            close_root = Path(tmp) / "close"
            linear_root = Path(tmp) / "linear"
            close_dir = close_root / "cross_space_projection_subtrial_2_3_close"
            linear_dir = linear_root / "cross_space_projection_subtrial_2_3_linear"
            for run_dir, method, weight_scale in (
                (close_dir, "linear_close", 5.0),
                (linear_dir, "linear", 1.0),
            ):
                (run_dir / f"{method}_projector" / "split_training_stage").mkdir(parents=True)
                (run_dir / "logs").mkdir()
                for name in ("anchors.csv", "old_tensors.csv"):
                    (run_dir / name).write_text("sample_id\n10\n", encoding="utf-8")
                for split in ("train", "val", "test"):
                    (run_dir / f"{method}_projector" / "split_training_stage" / f"{split}.csv").write_text(
                        "sample_id\n10\n", encoding="utf-8"
                    )
                summary = pd.DataFrame([{
                    "mae": 4.0 if method == "linear_close" else 1.0,
                    "mae_micro_old_oncsv_before": 4.0 if method == "linear_close" else 1.0,
                    "refine_mode": "projector_linear",
                    "proj_anchor_loss_before": 0.0 if method == "linear_close" else 0.2,
                    "lp_best_epoch": 1 if method == "linear_close" else 50,
                }])
                if method == "linear_close":
                    summary.to_csv(run_dir / "logs" / "summary.csv", index=False)
                    pd.DataFrame({"l2": [8.0, 10.0], "cos_sim": [0.5, 0.7]}).to_csv(
                        run_dir / "logs" / "embedding_reconstruction_test_projected.csv", index=False
                    )
                else:
                    # Most historical subtrials have pipeline artifacts but no generated logs.
                    summary.drop(columns=["mae", "lp_best_epoch"]).to_csv(
                        run_dir / "refinement_summary.csv", index=False
                    )
                torch.save({
                    "weight": torch.eye(2) * weight_scale,
                    "bias": torch.zeros(2),
                }, run_dir / f"{method}_projector" / "best_projector_1.pt")

            rows = compare_experiment_roots(close_root, linear_root)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["new_idx"], row["old_idx"]), (2, 3))
        self.assertTrue(row["anchors_equal"])
        self.assertTrue(row["all_splits_equal"])
        self.assertEqual(row["close_mae"], 4.0)
        self.assertEqual(row["linear_mae"], 1.0)
        self.assertGreater(row["close_weight_norm"], row["linear_weight_norm"])
        self.assertEqual(row["close_reconstruction_l2_mean"], 9.0)


class ProgressiveSyntheticTest(unittest.TestCase):
    """Break caught: reported failure transitions cannot be regenerated."""

    def test_domain_nuisance_breaks_ols_but_not_ridge(self):
        rows = cross_domain_nuisance_experiment(seed=101)

        self.assertGreater(rows[-1]["ols_test_mse"], 100 * rows[0]["ols_test_mse"])
        self.assertGreater(rows[-1]["ols_test_mse"], 1000 * rows[-1]["ridge_test_mse"])
        self.assertLess(rows[-1]["ridge_test_mse"], 0.02)
        self.assertGreater(rows[-1]["ols_test_mse"], 30 * rows[-1]["adam_test_mse"])
        self.assertLess(rows[-1]["adam_weight_norm"], rows[-1]["ols_weight_norm"] / 10)
        self.assertGreater(rows[0]["ols_weight_norm"], 10 * rows[0]["ridge_weight_norm"])

    def test_anchor_sweep_has_interpolation_peak_and_ridge_controls_it(self):
        rows = anchor_count_sweep(seed=103)
        by_count = {row["anchors"]: row for row in rows}

        peak = by_count[80]
        self.assertGreater(peak["ols_test_mse"], 10 * by_count[40]["ols_test_mse"])
        self.assertGreater(peak["ols_test_mse"], 10 * by_count[160]["ols_test_mse"])
        self.assertLess(peak["ridge_test_mse"], peak["ols_test_mse"] / 10)

    def test_standardization_recovers_badly_scaled_mapping(self):
        result = scaling_experiment(seed=107)

        self.assertGreater(result["raw_test_mse"], 1000 * result["standardized_test_mse"])
        self.assertLess(result["raw_rank"], result["dimension"])
        self.assertEqual(result["standardized_rank"], result["dimension"])


if __name__ == "__main__":
    unittest.main()

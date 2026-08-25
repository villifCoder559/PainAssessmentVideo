import csv
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.compare_dfer_vmaes_mae import (
    analyze_folds,
    analyze_subjects,
    extract_paired_mae,
    holm_adjust,
    write_analysis_outputs,
)


def make_result(delta=0.0, reverse_first_fold=False, stored_offset=0.0):
    first_ids = np.array([10, 11])
    first_mae = np.array([1.0, 2.0]) + delta
    first_counts = np.array([1, 2])
    if reverse_first_fold:
        first_ids = first_ids[::-1]
        first_mae = first_mae[::-1]
        first_counts = first_counts[::-1]

    return {
        "config": {"criterion": "L1Loss()"},
        "results": {
            "k0_cross_val_final": {
                "test": {
                    "test_unique_subject_ids": first_ids,
                    "test_loss_per_subject": first_mae,
                    "test_count_subject_ids": first_counts,
                    "test_l1_error": (5.0 + 3.0 * delta) / 3.0 + stored_offset,
                }
            },
            "k1_cross_val_final": {
                "test": {
                    "test_unique_subject_ids": np.array([12, 13]),
                    "test_loss_per_subject": np.array([1.5, 1.0]) + delta,
                    "test_count_subject_ids": np.array([2, 2]),
                    "test_l1_error": 1.25 + delta + stored_offset,
                }
            },
            "k0_cross_val_sub_0": {},
        },
    }


class TestPairedMaeExtraction(unittest.TestCase):
    def test_aligns_subjects_and_reconstructs_sample_weighted_fold_mae(self):
        paired = extract_paired_mae(
            "Synthetic",
            make_result(),
            make_result(delta=0.2, reverse_first_fold=True, stored_offset=0.5),
        )

        np.testing.assert_allclose(paired.subject_dfer, [1.0, 2.0, 1.5, 1.0])
        np.testing.assert_allclose(paired.subject_vmaes, [1.2, 2.2, 1.7, 1.2])
        np.testing.assert_allclose(paired.fold_dfer, [5.0 / 3.0, 1.25])
        np.testing.assert_allclose(paired.fold_vmaes, [5.0 / 3.0 + 0.2, 1.45])
        self.assertEqual(paired.fold_labels, ("k0_cross_val_final", "k1_cross_val_final"))
        self.assertEqual(len(paired.audit_rows), 4)
        self.assertTrue(all(row["matches_stored"] for row in paired.audit_rows[:2]))
        self.assertTrue(all(not row["matches_stored"] for row in paired.audit_rows[2:]))

    def test_rejects_subject_mismatch(self):
        vmaes = make_result(delta=0.2)
        vmaes["results"]["k0_cross_val_final"]["test"]["test_unique_subject_ids"][0] = 99

        with self.assertRaisesRegex(ValueError, "subject IDs differ"):
            extract_paired_mae("Synthetic", make_result(), vmaes)

    def test_rejects_subject_reused_across_test_folds(self):
        dfer = make_result()
        vmaes = make_result(delta=0.2)
        for data in (dfer, vmaes):
            data["results"]["k1_cross_val_final"]["test"]["test_unique_subject_ids"][0] = 10

        with self.assertRaisesRegex(ValueError, "multiple final folds"):
            extract_paired_mae("Synthetic", dfer, vmaes)

    def test_rejects_non_l1_configuration(self):
        dfer = make_result()
        dfer["config"]["criterion"] = "MSELoss()"

        with self.assertRaisesRegex(ValueError, "L1Loss"):
            extract_paired_mae("Synthetic", dfer, make_result(delta=0.2))

    def test_rejects_loss_names_that_only_contain_l1loss(self):
        for invalid_name in ("SmoothL1Loss()", "NotL1Loss()"):
            with self.subTest(invalid_name=invalid_name):
                dfer = make_result()
                dfer["config"]["criterion"] = invalid_name
                with self.assertRaisesRegex(ValueError, "L1Loss"):
                    extract_paired_mae("Synthetic", dfer, make_result(delta=0.2))

    def test_rejects_negative_mae_values(self):
        vmaes = make_result(delta=0.2)
        vmaes["results"]["k0_cross_val_final"]["test"]["test_loss_per_subject"][0] = -0.1

        with self.assertRaisesRegex(ValueError, "non-negative"):
            extract_paired_mae("Synthetic", make_result(), vmaes)

    def test_rejects_fold_and_sample_count_mismatches(self):
        vmaes = make_result(delta=0.2)
        vmaes["results"]["k1_cross_val_final"]["test"]["test_count_subject_ids"][0] = 3
        with self.assertRaisesRegex(ValueError, "sample counts differ"):
            extract_paired_mae("Synthetic", make_result(), vmaes)

        vmaes = make_result(delta=0.2)
        vmaes["results"]["k2_cross_val_final"] = vmaes["results"].pop("k1_cross_val_final")
        with self.assertRaisesRegex(ValueError, "final fold keys differ"):
            extract_paired_mae("Synthetic", make_result(), vmaes)

    def test_rejects_subject_count_total_that_disagrees_with_class_total(self):
        dfer = make_result()
        vmaes = make_result(delta=0.2)
        for data in (dfer, vmaes):
            data["results"]["k0_cross_val_final"]["test"]["test_count_y"] = np.array([1, 1])

        with self.assertRaisesRegex(ValueError, "subject and class sample totals differ"):
            extract_paired_mae("Synthetic", dfer, vmaes)

    def test_rejects_nonfinite_subject_mae(self):
        vmaes = make_result(delta=0.2)
        vmaes["results"]["k0_cross_val_final"]["test"]["test_loss_per_subject"][0] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            extract_paired_mae("Synthetic", make_result(), vmaes)


class TestStatistics(unittest.TestCase):
    def test_holm_adjustment_preserves_input_order(self):
        np.testing.assert_allclose(holm_adjust([0.04, 0.001, 0.7]), [0.08, 0.003, 0.7])

    def test_subject_analysis_reports_paired_mean_effect_and_deterministic_interval(self):
        dfer = np.array([1.0, 1.1, 0.9, 1.2, 1.0, 0.8])
        vmaes = dfer + np.array([0.2, 0.1, 0.3, 0.2, 0.4, 0.1])

        first = analyze_subjects("Synthetic", dfer, vmaes, seed=7, resamples=999)
        second = analyze_subjects("Synthetic", dfer, vmaes, seed=7, resamples=999)

        self.assertAlmostEqual(first["mean_difference_vmaes_minus_dfer"], 0.2166666667)
        self.assertGreater(first["cohen_dz"], 0)
        self.assertEqual(first["bootstrap_ci95_low"], second["bootstrap_ci95_low"])
        self.assertEqual(first["signflip_p_raw"], second["signflip_p_raw"])

    def test_fold_analysis_uses_corrected_resampled_variance(self):
        dfer = np.array([1.0, 1.1, 0.9, 1.2, 1.0])
        vmaes = dfer + np.array([0.1, 0.2, 0.15, 0.05, 0.1])

        row = analyze_folds("Synthetic", dfer, vmaes)
        differences = vmaes - dfer
        expected = differences.mean() / (
            differences.std(ddof=1) * np.sqrt(1 / 5 + 1 / 4)
        )

        self.assertAlmostEqual(row["corrected_t_statistic"], expected)
        self.assertEqual(row["exact_signflip_p_raw"], 0.0625)

    def test_fold_analysis_requires_exactly_five_pairs(self):
        with self.assertRaisesRegex(ValueError, "exactly five"):
            analyze_folds("Synthetic", [1.0, 1.1, 0.9, 1.2], [1.1, 1.2, 1.0, 1.3])


class TestOutputs(unittest.TestCase):
    def test_direct_script_execution_adds_repository_root_to_import_path(self):
        repository_root = Path(__file__).resolve().parents[1]
        script = repository_root / "analysis" / "compare_dfer_vmaes_mae.py"
        original_path = sys.path.copy()
        try:
            sys.path[:] = [
                entry
                for entry in sys.path
                if not entry or Path(entry).resolve() != repository_root
            ]
            runpy.run_path(str(script), run_name="direct_execution_test")
            self.assertIn(str(repository_root), sys.path)
        finally:
            sys.path[:] = original_path

    def test_writes_three_csv_files_and_markdown_report(self):
        subject_rows = [{"dataset": "Synthetic", "paired_t_p_raw": 0.01}]
        fold_rows = [{"dataset": "Synthetic", "corrected_t_p_raw": 0.2}]
        audit_rows = [{"dataset": "Synthetic", "fold": "k0", "model": "DFER"}]

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp, "results")
            report_path = Path(tmp, "report.md")
            write_analysis_outputs(subject_rows, fold_rows, audit_rows, output_dir, report_path)

            self.assertTrue(Path(output_dir, "subject_results.csv").is_file())
            self.assertTrue(Path(output_dir, "fold_results.csv").is_file())
            self.assertTrue(Path(output_dir, "metric_audit.csv").is_file())
            self.assertTrue(report_path.is_file())
            with Path(output_dir, "subject_results.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["dataset"], "Synthetic")
            report = report_path.read_text()
            self.assertIn("DFER vs VMAE-S", report)
            self.assertIn("Confidence intervals are unadjusted", report)
            self.assertIn("## Sensitivity analyses", report)
            self.assertIn("## Conclusion", report)
            self.assertIn("five-fold results", report)
            self.assertNotIn("five-model", report)


if __name__ == "__main__":
    unittest.main()

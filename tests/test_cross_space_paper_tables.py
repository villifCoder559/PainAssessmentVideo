import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cross_space_paper_tables as paper_tables


BASELINES = {
  "srctest_mae_micro_old": 0.80,
  "srctest_mae_macro_old": 0.90,
  "newtest_mae_micro_before": 1.12,
  "newtest_mae_macro_before": 1.12,
}


def _write_model(root: Path, name: str, dataset: str, model_type: str) -> str:
  """Create a minimal model checkpoint tree and return its checkpoint path."""
  run = root / "models" / name
  checkpoint = run / "train" / "best_model.pt"
  checkpoint.parent.mkdir(parents=True)
  checkpoint.touch()
  (run / "global_config.json").write_text(
    json.dumps({
      "model_type": model_type,
      "path_csv_dataset": [dataset, "starting_point", "samples.csv"],
    }),
    encoding="utf-8",
  )
  return str(checkpoint)


def _write_stage_root(
  root: Path, old_model: str, new_model: str, rows: dict, before: dict | None = None,
) -> Path:
  """Write aggregate PKLs plus aggregated_summary.csv; rows maps (method, mode) to metrics.

  before optionally maps method to its projector-only (srctest micro, macro) errors.
  """
  frames = []
  for (method, mode), (src_micro, src_macro, new_micro, new_macro) in rows.items():
    projector_only = {}
    if before:
      projector_only = dict(zip(
        ("srctest_mae_micro_before", "srctest_mae_macro_before"), before[method]))
    aggregate = root / f"refinement_{method}" / "aggregated_1"
    aggregate.mkdir(parents=True, exist_ok=True)
    pkl_path = aggregate / "results.pkl"
    with pkl_path.open("wb") as handle:
      pickle.dump({"config_cross_space_projection": {
        "old_model_pth": [old_model], "new_model_pth": [new_model],
        "interpolation_similarity": method,
      }}, handle)
    frames.append({
      "source_pkl": str(pkl_path.relative_to(root)),
      "old_model_pth": old_model,
      "new_model_pth": new_model,
      "subtrial_index": "AGGREGATE_MEAN",
      "interpolation_similarity": method,
      "num_anchors": 100,
      "refine_mode": mode,
      "srctest_mae_micro_after": src_micro,
      "srctest_mae_macro_after": src_macro,
      "newtest_mae_micro_after": new_micro,
      "newtest_mae_macro_after": new_macro,
      **projector_only,
      **BASELINES,
    })
  root.mkdir(parents=True, exist_ok=True)
  pd.DataFrame(frames).to_csv(root / "aggregated_summary.csv", index=False)
  return root


def _body_rows(latex: str) -> list[str]:
  """Return table data rows with the direction cell stripped."""
  return [
    line.split("&", 1)[1].strip() for line in latex.splitlines()
    if line.endswith(r"\\") and "&" in line and r"\textbf" not in line
  ]


class PaperTablesTest(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.root = Path(self.tmp.name)
    self.unbc = _write_model(self.root, "unbc", "UNBC", "VIDEOMAE_v2_S")
    self.biovid = _write_model(self.root, "biovid", "partA", "DFER")

  def tearDown(self):
    self.tmp.cleanup()

  def test_anchor_table_pairs_random_and_quality_columns(self):
    random = _write_stage_root(self.root / "random", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (0.85, 0.91, 1.12, 1.13),
      ("linear", "linear_only"): (9.0, 9.0, 9.0, 9.0),
      ("procrustes", "projector_linear"): (0.84, 0.90, 1.11, 1.12),
    })
    quality = _write_stage_root(self.root / "quality", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (0.82, 0.88, 1.10, 1.14),
      ("procrustes", "projector_linear"): (0.83, 0.89, 1.15, 1.16),
    })
    latex = paper_tables.anchor_table(random, quality)
    self.assertIn(r"UNBC $\rightarrow$ BioVid", latex)
    self.assertEqual(_body_rows(latex), [
      r"Linear & 0.85 & 0.82 & 1.12 & 1.10 \\",
      r"Procrustes & 0.84 & 0.83 & 1.11 & 1.15 \\",
    ])
    macro = paper_tables.anchor_table(random, quality, metric="macro")
    self.assertIn(r"Linear & 0.91 & 0.88 & 1.13 & 1.14 \\", macro)
    self.assertIn(r"\label{tab:anchor_selection_macro_mae}", macro)

  def test_anchor_table_rejects_missing_quality_method(self):
    random = _write_stage_root(self.root / "random", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (0.85, 0.91, 1.12, 1.13),
      ("linear_close", "projector_linear"): (0.94, 0.96, 1.12, 1.12),
    })
    quality = _write_stage_root(self.root / "quality", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (0.82, 0.88, 1.10, 1.14),
    })
    with self.assertRaisesRegex(ValueError, "Methods differ"):
      paper_tables.anchor_table(random, quality)

  def test_refinement_table_pairs_projector_only_and_joint_stages(self):
    root = _write_stage_root(self.root / "stages", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (0.85, 0.91, 1.11, 1.13),
      ("linear", "linear_only"): (9.0, 9.0, 9.0, 9.0),
    }, before={"linear": (0.97, 1.00)})
    latex = paper_tables.refinement_table(root)
    self.assertEqual(_body_rows(latex), [
      r"Linear & 0.97 / 1.00 & 0.85 / 0.91 & 1.12 / 1.12 & 1.11 / 1.13 \\",
    ])
    self.assertIn(r"\label{tab:comparison_proj_vs_projANDregressor_stage_mae_macro}", latex)

  def test_closed_form_table_lists_procrustes_and_ols_only(self):
    root = _write_stage_root(self.root / "stages", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (0.85, 0.91, 1.12, 1.12),
      ("linear_close", "projector_linear"): (0.944, 0.964, 1.217, 1.217),
      ("procrustes", "projector_linear"): (0.843, 0.915, 1.122, 1.122),
    }, before={"linear": (0.97, 1.0), "linear_close": (5.742, 4.897),
               "procrustes": (1.079, 1.078)})
    latex = paper_tables.closed_form_table(root)
    self.assertEqual(_body_rows(latex), [
      r"Procrustes & 1.079 / 1.078 & 0.843 / 0.915 & 1.122 / 1.122 \\",
      r"Linear (closed form) & 5.742 / 4.897 & 0.944 / 0.964 & 1.217 / 1.217 \\",
    ])
    without = _write_stage_root(self.root / "without", self.unbc, self.biovid, {
      ("procrustes", "projector_linear"): (0.843, 0.915, 1.122, 1.122),
    }, before={"procrustes": (1.079, 1.078)})
    with self.assertRaisesRegex(ValueError, "Missing closed-form methods"):
      paper_tables.closed_form_table(without)

  def test_frozen_tables_report_signed_relative_change(self):
    learned = _write_stage_root(self.root / "learned", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (1.0, 2.0, 1.0, 1.0),
    })
    frozen = _write_stage_root(self.root / "frozen", self.unbc, self.biovid, {
      ("linear", "random_projector_linear"): (1.5, 1.8, 1.0, 1.0004),
    })
    source, target = paper_tables.frozen_tables(learned, frozen).split(r"\end{table}")[:2]
    self.assertIn(r"Linear & 1.000 / 2.000 & 1.500 / 1.800 & +50.0 / -10.0 \\", source)
    self.assertIn(r"\label{tab:projector_linear_vs_random_source}", source)
    self.assertIn(r"Linear & 1.000 / 1.000 & 1.000 / 1.000 & +0.0 / +0.0 \\", target)
    self.assertIn(r"\label{tab:projector_linear_vs_random_target}", target)

  def test_frozen_tables_require_random_projector_stage(self):
    learned = _write_stage_root(self.root / "learned", self.unbc, self.biovid, {
      ("linear", "projector_linear"): (1.0, 2.0, 1.0, 1.0),
    })
    with self.assertRaises(ValueError):
      paper_tables.frozen_tables(learned, learned)

  def _write_fake_summary(self, name: str = "synthetic", **overrides) -> Path:
    root = self.root / name
    root.mkdir()
    rows = []
    for method, real, fake in (("mlp", 0.85, 1.36), ("linear", 0.84, 1.43)):
      for mode, offset in (("projector_linear", 0.0), ("linear_only", 5.0)):
        for summary_row, scale in (("MEAN", 1.0), ("STD", 0.01)):
          rows.append({
            "experiment": f"refinement3_{method}", "refinement_mode": mode,
            "summary_row": summary_row, "status": "success",
            "distribution": "standard_normal", "success_count": 25,
            "old_model_pth": self.unbc, "new_model_pth": self.biovid,
            "interpolation_similarity": method,
            "real_mae_micro": (real + offset) * scale,
            "fake_mae_micro": (fake + offset) * scale,
            "real_mae_macro": (real + 0.06 + offset) * scale,
            "fake_mae_macro": (fake + 0.6 + offset) * scale,
            **overrides,
          })
    pd.DataFrame(rows).to_csv(root / "aggregated_summary_fake.csv", index=False)
    return root

  def test_synthetic_table_uses_joint_mean_rows(self):
    latex = paper_tables.synthetic_table([self._write_fake_summary()])
    self.assertIn("evaluated on UNBC", latex)
    self.assertEqual(_body_rows(latex), [
      r"Linear & 0.84 & 1.43 & 0.90 & 2.03 \\",
      r"MLP & 0.85 & 1.36 & 0.91 & 1.96 \\",
    ])

  def test_synthetic_table_rejects_failed_or_other_distribution(self):
    with self.assertRaisesRegex(ValueError, "Non-successful"):
      paper_tables.synthetic_table([self._write_fake_summary("failed", status="partial")])
    with self.assertRaisesRegex(ValueError, "standard_normal"):
      paper_tables.synthetic_table(
        [self._write_fake_summary("gaussian", distribution="matched_gaussian")])

  def _adapter_summary(
    self, metrics=("micro_mae", "macro_mae"), direction="MIntPAIN -> BioVid",
  ) -> pd.DataFrame:
    values = {"micro_mae": (1.30, 1.28), "macro_mae": (1.39, 2.00)}
    rows = []
    for mode in ("head_only", "joint"):
      for metric in metrics:
        fitted, random = values[metric]
        rows.append({
          "direction": direction, "projection": "Autoencoder",
          "refinement_mode": mode, "metric": metric, "fitted_mae": fitted,
          "random_mae_mean": random if mode == "joint" else 9.0,
          "random_mae_seed_sd": 0.01, "increase_pct": 0.0, "delta_mae": 0.0,
          "checkpoint_pair_count": 25, "seed_count": 5,
        })
    return pd.DataFrame(rows)

  def test_random_adapter_table_uses_joint_rows(self):
    summaries = [self._adapter_summary(), self._adapter_summary(direction="UNBC -> BioVid")]
    with mock.patch.object(
      paper_tables.adapter_tables, "summarize_root", side_effect=summaries,
    ) as summarize:
      latex = paper_tables.random_adapter_table(["a", "b"], fast_mode=True)
    self.assertEqual(summarize.call_count, 2)
    summarize.assert_called_with("b", fast_mode=True)
    self.assertIn(r"MIntPAIN $\rightarrow$ BioVid", latex)
    self.assertIn(r"UNBC $\rightarrow$ BioVid", latex)
    self.assertEqual(
      _body_rows(latex), [r"EncDec & 1.30 / 1.39 & 1.28 / 2.00 & -1.5 / +43.9 \\"] * 2)
    self.assertIn("averaged over 5 initialization seeds", latex)

  def test_random_adapter_table_requires_both_metrics(self):
    with mock.patch.object(
      paper_tables.adapter_tables, "summarize_root",
      return_value=self._adapter_summary(metrics=("micro_mae",)),
    ):
      with self.assertRaisesRegex(ValueError, "Missing MAE or Macro-MAE"):
        paper_tables.random_adapter_table(["a"])

  def _write_replay(self, folder: str, method: str, seed: int = 42) -> None:
    path = (
      self.root / "direction" / "fake_adapter_random_init" / "seed_42" / folder / "results_1.pkl")
    path.parent.mkdir(parents=True)
    with path.open("wb") as handle:
      pickle.dump({
        "config_cross_space_projection": {
          "old_model_pth": self.unbc, "new_model_pth": self.biovid,
          "interpolation_similarity": method,
        },
        "fake_projection_metadata": {"control": "fake_adapter", "seed": seed},
        # Native predictions are stored in a different sample order on purpose.
        "old_model_tensors": {
          "sample_ids": np.array([13, 12, 11, 10]),
          "predictions": np.array([1.2, 0.9, 0.4, 0.1]),
        },
        "fake_projection_evaluations": {"projector_linear": {
          "sample_ids": np.array([10, 11, 12, 13]),
          "labels": np.array([0.0, 0.0, 1.0, 1.0]),
          "trained_adapter_predictions": np.array([0.2, 1.1, 0.8, 1.4]),
          "real_before_predictions": np.array([1.2, 1.1, 0.8, 1.4]),
          "random_adapter_predictions": np.array([0.0, 0.1, -0.3, 0.2]),
        }},
      }, handle)

  def test_confusion_figure_pools_rows_and_aligns_native_predictions(self):
    self._write_replay("refinement3_linear/pair_0", "linear")
    self._write_replay("refinement3_linear/pair_1", "linear")
    self._write_replay("refinement3_mlp/pair_0", "mlp")
    self._write_replay("refinement3_linear/aggregated_fake", "linear")
    latex = paper_tables.confusion_figure(self.root / "direction", method="linear")
    tables = latex.split(r"\end{tabular}")[:3]
    native, fitted, random = (
      [line for line in table.splitlines() if line[:1].isdigit()] for table in tables)
    self.assertEqual(native, [r"0 & 100.0 & 0.0 \\", r"1 & 0.0 & 100.0 \\"])
    self.assertEqual(fitted, [r"0 & 50.0 & 50.0 \\", r"1 & 0.0 & 100.0 \\"])
    self.assertEqual(random, [r"0 & 100.0 & 0.0 \\", r"1 & 100.0 & 0.0 \\"])
    self.assertIn("from 2 source--target checkpoint pairs", latex)
    self.assertIn(r"\label{fig:unbc_biovid_random_confusion}", latex)

  def test_confusion_figure_projector_only_stage_uses_pre_refinement_predictions(self):
    self._write_replay("refinement3_linear/pair_0", "linear")
    latex = paper_tables.confusion_figure(
      self.root / "direction", method="linear", stage="projector_only")
    fitted = [
      line for line in latex.split(r"\end{tabular}")[1].splitlines() if line[:1].isdigit()]
    self.assertEqual(fitted, [r"0 & 0.0 & 100.0 \\", r"1 & 0.0 & 100.0 \\"])
    self.assertIn("(projector-only)", latex)
    self.assertIn(r"\label{fig:unbc_biovid_random_confusion_projector_only}", latex)

  def test_confusion_figure_requires_matching_replays(self):
    with self.assertRaisesRegex(ValueError, "Missing random-adapter replay folder"):
      paper_tables.confusion_figure(self.root / "direction")
    self._write_replay("refinement3_mlp/pair_0", "mlp")
    with self.assertRaisesRegex(ValueError, "No linear replay PKLs"):
      paper_tables.confusion_figure(self.root / "direction", method="linear")


if __name__ == "__main__":
  unittest.main()

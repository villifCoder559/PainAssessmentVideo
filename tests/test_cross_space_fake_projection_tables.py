import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cross_space_fake_projection_tables as tables


class TestCrossSpaceFakeProjectionTables(unittest.TestCase):
  def _write_model(self, workspace, name, dataset, model_type):
    run = workspace / "models" / name
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

  def _row(self, source, old_model, new_model, seed, mode, fitted, random):
    return {
      "status": "success",
      "replay_error": "",
      "old_model_pth": old_model,
      "new_model_pth": new_model,
      "interpolation_similarity": "linear",
      "refinement_mode": mode,
      "fake_projection_seed": seed,
      "source_pkl_path": source,
      "num_anchors": 100,
      "anchor_selection_type": "balance_class_random",
      "csv_anchor_selection": "train",
      "old_model_csv": "test",
      "mlp_activation": "gelu",
      "mlp_num_layers": 1,
      "weighting_method": "none",
      "rbf_sigma": 1.0,
      "temperature": 1.0,
      "projector_config": "",
      "refinement_config": "",
      "refine_mode": "",
      "trained_adapter_mae_micro": fitted[0],
      "random_adapter_mae_micro": random[0],
      "trained_adapter_mae_macro": fitted[1],
      "random_adapter_mae_macro": random[1],
    }

  def test_summarizes_unique_pairs_and_seed_variability_for_both_metrics(self):
    """Catch seed-row double weighting and incorrect across-seed SD."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      root = workspace / "biovid_to_unbc"
      root.mkdir()
      old_model_1 = self._write_model(
        workspace, "biovid_vmae_1", "partA", "VIDEOMAE_v2_S")
      old_model_2 = self._write_model(
        workspace, "biovid_vmae_2", "partA", "VIDEOMAE_v2_S")
      new_model = self._write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      rows = []
      values = {
        42: (((1.0, 2.0), (2.0, 3.0)), ((3.0, 4.0), (4.0, 5.0))),
        43: (((1.0, 2.0), (4.0, 5.0)), ((3.0, 4.0), (6.0, 7.0))),
      }
      for seed, (first, second) in values.items():
        for mode in ("linear_only", "projector_linear"):
          rows.append(self._row(
            "/runs/pair_1/results.pkl", old_model_1, new_model, seed,
            mode, *first))
          rows.append(self._row(
            "/runs/pair_2/results.pkl", old_model_2, new_model, seed,
            mode, *second))
      # An identical rerun of pair 1 must not give that pair extra weight.
      rows.extend([
        {**row, "source_pkl_path": row["source_pkl_path"].replace(
          "pair_1", "pair_1_rerun")}
        for row in rows if "pair_1" in row["source_pkl_path"]
      ])
      pd.DataFrame(rows).to_csv(
        root / "fake_adapter_seed_results_fast.csv", index=False)

      summary = tables.summarize_root(root, fast_mode=True)

    self.assertEqual(set(summary["metric"]), {"micro_mae", "macro_mae"})
    head_only = summary.loc[summary["refinement_mode"].eq("head_only")]
    micro = head_only.loc[head_only["metric"].eq("micro_mae")].iloc[0]
    macro = head_only.loc[head_only["metric"].eq("macro_mae")].iloc[0]
    self.assertEqual(micro["direction"], "BioVid -> UNBC")
    self.assertEqual(micro["projection"], "Linear")
    self.assertEqual(micro["refinement_mode"], "head_only")
    self.assertEqual(micro["checkpoint_pair_count"], 2)
    self.assertEqual(micro["seed_count"], 2)
    self.assertEqual(micro["fitted_mae"], 2.0)
    self.assertEqual(micro["random_mae_mean"], 4.0)
    self.assertTrue(math.isclose(
      micro["random_mae_seed_sd"], math.sqrt(2), rel_tol=1e-12))
    self.assertEqual(micro["delta_mae"], 2.0)
    self.assertEqual(micro["increase_pct"], 100.0)
    self.assertEqual(macro["fitted_mae"], 3.0)
    self.assertEqual(macro["random_mae_mean"], 5.0)
    self.assertTrue(math.isclose(macro["increase_pct"], 200 / 3, rel_tol=1e-12))

  def _write_simple_experiment(self, root, old_model, new_model):
    rows = []
    for mode, fitted, random_by_seed in (
      ("linear_only", (1.0, 2.0), {42: (2.0, 3.0), 43: (4.0, 5.0)}),
      ("projector_linear", (0.5, 1.5), {42: (1.0, 2.0), 43: (2.0, 4.0)}),
    ):
      for seed, random in random_by_seed.items():
        rows.append(self._row(
          f"/runs/{root.name}/results.pkl", old_model, new_model, seed,
          mode, fitted, random))
    root.mkdir()
    pd.DataFrame(rows).to_csv(
      root / "fake_adapter_seed_results_fast.csv", index=False)

  def test_generates_unique_self_contained_per_direction_and_combined_outputs(self):
    """Catch overwrites, missing output pairs, and malformed four-table LaTeX."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      biovid_vmae = self._write_model(
        workspace, "biovid_vmae", "partA", "VIDEOMAE_v2_S")
      unbc_dfer = self._write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      unbc_vmae = self._write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S")
      biovid_dfer = self._write_model(workspace, "biovid_dfer", "partA", "DFER")
      first = workspace / "biovid_to_unbc"
      second = workspace / "unbc_to_biovid"
      self._write_simple_experiment(first, biovid_vmae, unbc_dfer)
      self._write_simple_experiment(second, unbc_vmae, biovid_dfer)
      output_parent = workspace / "tables"

      run = tables.generate([first, second], output_parent, fast_mode=True)
      second_run = tables.generate([first], output_parent, fast_mode=True)

      self.assertNotEqual(run, second_run)
      self.assertRegex(run.name, r"^random_adapter_ablation_\d{8}_\d{6}_\d{6}(?:_\d+)?$")
      self.assertEqual(run.parent, output_parent)
      self.assertEqual(
        {path.name for path in run.iterdir()},
        {
          "01_biovid_to_unbc_random_adapter_ablation.csv",
          "01_biovid_to_unbc_random_adapter_ablation.tex",
          "02_unbc_to_biovid_random_adapter_ablation.csv",
          "02_unbc_to_biovid_random_adapter_ablation.tex",
          "comparison_random_adapter_ablation.csv",
          "comparison_random_adapter_ablation.tex",
        },
      )
      combined = pd.read_csv(run / "comparison_random_adapter_ablation.csv")
      latex = (run / "comparison_random_adapter_ablation.tex").read_text(
        encoding="utf-8")

    self.assertEqual(len(combined), 8)
    self.assertEqual(
      combined["direction"].drop_duplicates().tolist(),
      ["BioVid -> UNBC", "UNBC -> BioVid"],
    )
    self.assertEqual(latex.count(r"\begin{table}[H]"), 4)
    self.assertIn(r"\textbf{Direction}", latex)
    self.assertIn(r"\textbf{Fitted MAE}", latex)
    self.assertIn(r"\textbf{Random MAE}", latex)
    self.assertIn(r"\textbf{Increase (\%)}", latex)
    self.assertIn(r"\(3.0000\pm1.4142\)", latex)
    self.assertIn(r"BioVid $\rightarrow$ UNBC", latex)
    labels = [line for line in latex.splitlines() if line.startswith(r"\label{")]
    self.assertEqual(len(labels), 4)
    self.assertEqual(len(set(labels)), 4)

  def test_rejects_missing_refinement_mode_before_creating_outputs(self):
    """Catch generation of empty joint/head-only tables from incomplete input."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      old_model = self._write_model(
        workspace, "biovid_vmae", "partA", "VIDEOMAE_v2_S")
      new_model = self._write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      root = workspace / "incomplete"
      self._write_simple_experiment(root, old_model, new_model)
      csv_path = root / "fake_adapter_seed_results_fast.csv"
      frame = pd.read_csv(csv_path)
      frame.loc[frame["refinement_mode"].eq("linear_only")].to_csv(
        csv_path, index=False)
      output_parent = workspace / "tables"

      with self.assertRaisesRegex(ValueError, "required refinement modes"):
        tables.generate([root], output_parent, fast_mode=True)

      self.assertFalse(output_parent.exists())

  def test_excludes_requested_dataset_from_a_mixed_experiment_root(self):
    """Catch PEMF rows leaking into summaries, validation, or averages."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      biovid_vmae = self._write_model(
        workspace, "biovid_vmae", "partA", "VIDEOMAE_v2_S")
      unbc_dfer = self._write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      unbc_vmae = self._write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S")
      pemf_dfer = self._write_model(workspace, "pemf_dfer", "PEMF", "DFER")
      mixed = workspace / "mixed"
      other = workspace / "other"
      self._write_simple_experiment(mixed, biovid_vmae, unbc_dfer)
      self._write_simple_experiment(other, unbc_vmae, pemf_dfer)
      filename = "fake_adapter_seed_results_fast.csv"
      pd.concat([
        pd.read_csv(mixed / filename),
        pd.read_csv(other / filename),
      ], ignore_index=True).to_csv(mixed / filename, index=False)

      summary = tables.summarize_root(
        mixed, fast_mode=True, exclude_datasets={"PEMF"})

    self.assertEqual(
      summary["direction"].drop_duplicates().tolist(),
      ["BioVid -> UNBC"],
    )
    self.assertEqual(len(summary), 4)

  def test_rejects_invalid_incomplete_and_conflicting_rows(self):
    """Catch silent omission or averaging of untrustworthy replay records."""
    cases = (
      (
        "failed status",
        lambda frame: frame.assign(status=["error", *frame["status"].iloc[1:]]),
        "Non-successful",
      ),
      (
        "replay error",
        lambda frame: frame.assign(
          replay_error=["checkpoint missing", *frame["replay_error"].iloc[1:]]),
        "replay errors",
      ),
      (
        "non-finite metric",
        lambda frame: frame.assign(
          random_adapter_mae_micro=[
            "not-a-number", *frame["random_adapter_mae_micro"].iloc[1:]]),
        "Non-finite",
      ),
      (
        "incomplete seeds",
        lambda frame: frame.drop(frame.index[
          frame["refinement_mode"].eq("linear_only")
          & frame["fake_projection_seed"].eq(43)
        ]),
        "Incomplete seed coverage",
      ),
      (
        "seed-varying fitted metric",
        lambda frame: frame.assign(
          trained_adapter_mae_micro=frame["trained_adapter_mae_micro"].where(
            ~(
              frame["refinement_mode"].eq("linear_only")
              & frame["fake_projection_seed"].eq(43)
            ),
            frame["trained_adapter_mae_micro"] + 0.1,
          )),
        "Seed-varying fitted metric",
      ),
      (
        "conflicting duplicate",
        lambda frame: pd.concat([
          frame,
          frame.iloc[[0]].assign(
            random_adapter_mae_micro=frame.iloc[0]["random_adapter_mae_micro"] + 0.5),
        ], ignore_index=True),
        "Conflicting duplicate",
      ),
    )
    for name, mutate, message in cases:
      with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        old_model = self._write_model(
          workspace, "biovid_vmae", "partA", "VIDEOMAE_v2_S")
        new_model = self._write_model(workspace, "unbc_dfer", "UNBC", "DFER")
        root = workspace / "experiment"
        self._write_simple_experiment(root, old_model, new_model)
        csv_path = root / "fake_adapter_seed_results_fast.csv"
        mutate(pd.read_csv(csv_path)).to_csv(csv_path, index=False)

        with self.assertRaisesRegex(ValueError, message):
          tables.summarize_root(root, fast_mode=True)


if __name__ == "__main__":
  unittest.main()

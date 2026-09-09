import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cross_space_generate_latex_table as latex_table

generate_table = latex_table.generate_table


REFERENCE_TABLE = REPO_ROOT / "z_latex_tables" / "mint_unbc_table"
REFERENCE_ROOTS = (
  REPO_ROOT / "Cross_projection" / "unbcVMAE-mintDFER",
  REPO_ROOT / "Cross_projection" / "mintVMAE-unbcDFER",
)


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


def _summary_row(
  method: str,
  mode: str,
  values: tuple[float, ...],
  *,
  projected: tuple[float, float] | None = None,
) -> dict:
  """Build one aggregate-mean summary row with the production metric schema."""
  (
    src_micro,
    src_macro,
    target_micro,
    target_macro,
    old_micro,
    old_macro,
    native_micro,
    native_macro,
  ) = values
  projected_micro, projected_macro = projected or (src_micro, src_macro)
  return {
    "subtrial_index": "AGGREGATE_MEAN",
    "refine_mode": mode,
    "interpolation_similarity": method,
    "num_anchors": 100,
    "srctest_mae_micro_before": projected_micro,
    "srctest_mae_macro_before": projected_macro,
    "srctest_mae_micro_after": src_micro,
    "srctest_mae_macro_after": src_macro,
    "newtest_mae_micro_after": target_micro,
    "newtest_mae_macro_after": target_macro,
    "srctest_mae_micro_old": old_micro,
    "srctest_mae_macro_old": old_macro,
    "newtest_mae_micro_before": native_micro,
    "newtest_mae_macro_before": native_macro,
  }


def _write_aggregate(
  root: Path,
  method: str,
  old_model: str,
  new_model: str,
  rows: list[dict],
  *,
  fake_distribution: str | None = None,
  fake_control: str | None = None,
  adapter_seed: int | None = None,
  aggregate_name: str | None = None,
) -> Path:
  """Create one aggregate PKL and its adjacent summary CSV."""
  base = root
  if fake_control == "fake_adapter":
    base = base / "fake_adapter_random_init" / f"seed_{adapter_seed}"
  elif fake_distribution:
    base = base / f"fake_projection_{fake_distribution}"
  aggregate = (
    base
    / f"refinement3_{method}-cross-validation"
    / (
      aggregate_name
      or ("aggregated_fake" if fake_distribution or fake_control else "aggregated_1")
    )
  )
  aggregate.mkdir(parents=True)
  data = {
    "config_cross_space_projection": {
      "old_model_pth": [old_model],
      "new_model_pth": [new_model],
      "interpolation_similarity": method,
      "fake_projection": bool(fake_distribution),
      "fake_projection_distribution": fake_distribution,
      "fake_projection_control": fake_control,
    },
  }
  if fake_distribution:
    data["fake_projection_distribution"] = fake_distribution
  if fake_control:
    data["fake_projection_control"] = fake_control
    data["fake_projection_metadata"] = {
      "control": fake_control,
      "initialization": "pytorch_default",
      "seed": adapter_seed,
    }
  pkl_path = aggregate / "results.pkl"
  with pkl_path.open("wb") as handle:
    pickle.dump(data, handle)
  logs = aggregate / "logs"
  logs.mkdir()
  pd.DataFrame(rows).to_csv(logs / "summary.csv", index=False)
  return pkl_path


def _write_root_summary(root: Path, pkl_paths: list[Path]) -> None:
  """Build the production-shaped consolidated summary used by root mode."""
  frames = []
  for pkl_path in pkl_paths:
    with pkl_path.open("rb") as handle:
      config = pickle.load(handle)["config_cross_space_projection"]
    frame = pd.read_csv(pkl_path.parent / "logs" / "summary.csv")
    frame["source_pkl"] = str(pkl_path.relative_to(root))
    frame["old_model_pth"] = ";".join(config["old_model_pth"])
    frame["new_model_pth"] = ";".join(config["new_model_pth"])
    frames.append(frame)
  pd.concat(frames, ignore_index=True).to_csv(
    root / "aggregated_summary.csv", index=False
  )


def _canonical_rows(latex: str) -> list[list[tuple[str, ...]]]:
  """Extract semantic data rows from both rendered table sections."""
  sections = latex.split(r"\midrule\midrule\midrule")
  parsed = []
  for section in sections:
    rows = []
    for line in section.splitlines():
      stripped = line.strip()
      if not stripped.startswith("&"):
        continue
      fields = [field.strip() for field in stripped.split("%", 1)[0].split("&")]
      if len(fields) != 7 or fields[2] not in {"100", "X"}:
        continue
      fields[-1] = fields[-1].removesuffix(r"\\").strip()
      fields[1] = (
        fields[1]
        .replace("Linear layer (SGD)", "Linear")
        .replace("Linear (SGD)", "Linear")
        .replace("Autoencoder", "EncDec")
        .replace("Linear layer (closed form)", "Linear (closed form)")
        .replace("VMae-S", "VMAEv2-S")
      )
      rows.append(tuple(fields[1:]))
    parsed.append(rows)
  return parsed


class TestGenerateTable(unittest.TestCase):
  def test_generates_reference_shaped_inverse_table(self):
    """Place source and target results under fixed columns in both directions."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "unbc_to_mint"
      second = workspace / "mint_to_unbc"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")

      first_values = {
        "linear": (0.87, 0.90, 1.27, 1.27, 0.80, 0.90, 1.14, 1.45),
        "mlp": (0.84, 0.91, 1.28, 1.27, 0.80, 0.90, 1.14, 1.45),
      }
      second_values = {
        "linear": (1.30, 1.38, 0.80, 0.87, 1.29, 1.60, 0.81, 0.87),
        "mlp": (1.30, 1.40, 0.80, 0.87, 1.29, 1.60, 0.81, 0.87),
      }
      for method, values in first_values.items():
        _write_aggregate(
          first,
          method,
          unbc_vmae,
          mint_dfer,
          [_summary_row(method, "projector_linear", values)],
        )
      for method, values in second_values.items():
        _write_aggregate(
          second,
          method,
          mint_vmae,
          unbc_dfer,
          [_summary_row(method, "projector_linear", values)],
        )

      latex = generate_table(
        first,
        second,
        projection="real",
        stage="projector_linear",
      )

    self.assertIn(
      r"\multicolumn{2}{c}{\textbf{UNBC}}"
      r" & \multicolumn{2}{c}{\textbf{MIntPAIN}}",
      latex,
    )
    self.assertIn(
      r"\shortstack{UNBC $\to$ MIntPAIN \\"
      "\n"
      r"    \footnotesize VMAEv2-S $\to$ DFER}",
      latex,
    )
    self.assertIn(
      r"Linear & 100 & 0.87 & 0.90 & 1.27 & 1.27", latex
    )
    self.assertIn(
      r"Linear & 100 & 0.80 & 0.87 & 1.30 & 1.38", latex
    )
    self.assertIn(r"VMAEv2-S (UNBC) & X & 0.80 & 0.90 & X & X", latex)
    self.assertIn(r"DFER (MIntPAIN) & X & X & X & 1.14 & 1.45", latex)
    first_section, second_section = latex.split(r"\midrule\midrule\midrule")
    self.assertLess(first_section.index("Linear"), first_section.index("MLP"))
    self.assertLess(second_section.index("Linear"), second_section.index("MLP"))

  def test_random_projector_linear_stage_uses_post_refinement_metrics(self):
    """Catch selecting the new mode with the legacy projector-linear stage."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "unbc_to_mint"
      second = workspace / "mint_to_unbc"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      values = (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1)
      for root, old_model, new_model in (
        (first, unbc_vmae, mint_dfer),
        (second, mint_vmae, unbc_dfer),
      ):
        _write_aggregate(
          root,
          "linear",
          old_model,
          new_model,
          [_summary_row("linear", "random_projector_linear", values)],
        )

      latex = generate_table(
        first,
        second,
        projection="real",
        stage="random_projector_linear",
      )

    self.assertIn("Frozen-random-projector linear-head refinement", latex)
    self.assertIn("Linear & 100 & 0.61 & 0.62 & 0.63 & 0.64", latex)

  def test_one_root_renders_three_datasets_with_inverse_directions_paired(self):
    """Catch treating a consolidated root as one conflicting direction."""
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      datasets = ("BIOVID", "MIntPAIN", "UNBC")
      old_models = {
        dataset: _write_model(
          root, f"{dataset}_old", dataset, "VIDEOMAE_v2_S"
        )
        for dataset in datasets
      }
      new_models = {
        dataset: _write_model(root, f"{dataset}_new", dataset, "DFER")
        for dataset in datasets
      }
      values = (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1)
      pkl_paths = []
      directions = (
        ("UNBC", "MIntPAIN"),
        ("BIOVID", "UNBC"),
        ("MIntPAIN", "BIOVID"),
        ("MIntPAIN", "UNBC"),
        ("UNBC", "BIOVID"),
        ("BIOVID", "MIntPAIN"),
      )
      for source, target in directions:
        direction_root = root / f"cross-validation_{source}-to-{target}"
        pkl_paths.append(_write_aggregate(
          direction_root,
          "linear",
          old_models[source],
          new_models[target],
          [_summary_row("linear", "random_projector_linear", values)],
        ))
      pkl_paths.append(_write_aggregate(
        root / "cross-validation_BIOVID-to-MIntPAIN",
        "mlp",
        old_models["BIOVID"],
        new_models["MIntPAIN"],
        [_summary_row("mlp", "random_projector_linear", values)],
      ))
      _write_root_summary(root, pkl_paths)

      latex = generate_table(
        root,
        projection="real",
        stage="random_projector_linear",
      )

    self.assertIn(r"\resizebox{\textwidth}{!}{%", latex)
    self.assertIn(r"\begin{tabular}{clccccccc}", latex)
    for dataset in datasets:
      self.assertIn(
        rf"\multicolumn{{2}}{{c}}{{\textbf{{{dataset}}}}}", latex
      )
    self.assertIn(
      r"\multicolumn{2}{c}{\textbf{UNBC}} \\",
      latex,
    )
    ordered_directions = (
      r"BIOVID $\to$ MIntPAIN",
      r"MIntPAIN $\to$ BIOVID",
      r"BIOVID $\to$ UNBC",
      r"UNBC $\to$ BIOVID",
      r"MIntPAIN $\to$ UNBC",
      r"UNBC $\to$ MIntPAIN",
    )
    positions = [latex.index(direction) for direction in ordered_directions]
    self.assertEqual(positions, sorted(positions))
    first_direction = latex.split(ordered_directions[1], 1)[0]
    self.assertIn("Linear & 100", first_direction)
    self.assertIn("MLP & 100", first_direction)

  def test_one_root_all_skips_stage_tables_without_matching_directions(self):
    """Catch emitting empty or placeholder legacy tables for a new-mode root."""
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      old_model = _write_model(root, "old", "UNBC", "VIDEOMAE_v2_S")
      new_model = _write_model(root, "new", "MIntPAIN", "DFER")
      values = (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1)
      pkl_path = _write_aggregate(
        root / "cross-validation_UNBC-to-MIntPAIN",
        "linear",
        old_model,
        new_model,
        [_summary_row("linear", "random_projector_linear", values)],
      )
      _write_root_summary(root, [pkl_path])

      latex = generate_table(root, projection="real", stage="all")

    self.assertEqual(latex.count(r"\begin{table}[H]"), 2)
    self.assertIn("Projector-only cross-projection results", latex)
    self.assertIn("Frozen-random-projector linear-head refinement", latex)
    self.assertNotIn("Linear-only refinement cross-projection results", latex)
    self.assertNotIn("Projector-and-linear refinement cross-projection results", latex)

  def test_one_root_filters_rows_using_referenced_pkl_projection_metadata(self):
    """Catch trusting a mixed consolidated CSV without checking its PKLs."""
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      old_model = _write_model(root, "old", "UNBC", "VIDEOMAE_v2_S")
      new_model = _write_model(root, "new", "MIntPAIN", "DFER")
      values = (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1)
      direction_root = root / "cross-validation_UNBC-to-MIntPAIN"
      real_pkl = _write_aggregate(
        direction_root,
        "linear",
        old_model,
        new_model,
        [_summary_row("linear", "random_projector_linear", values)],
      )
      fake_pkl = _write_aggregate(
        direction_root,
        "mlp",
        old_model,
        new_model,
        [_summary_row("mlp", "random_projector_linear", values)],
        fake_distribution="standard_normal",
      )
      _write_root_summary(root, [real_pkl, fake_pkl])

      latex = generate_table(
        root,
        projection="real",
        stage="random_projector_linear",
      )

    self.assertIn("Linear & 100", latex)
    self.assertNotIn("MLP & 100", latex)

  def test_rejects_incomplete_method_set(self):
    """Reject a direction that is missing a method present in its inverse."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      for method in ("linear", "mlp"):
        _write_aggregate(
          first,
          method,
          unbc_vmae,
          mint_dfer,
          [_summary_row(method, "projector_linear", values)],
        )
      _write_aggregate(
        second,
        "linear",
        mint_vmae,
        unbc_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )

      with self.assertRaisesRegex(ValueError, "method sets differ.*mlp"):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )

  def test_rejects_non_inverse_datasets(self):
    """Reject roots whose trained datasets are not opposite directions."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      biovid_vmae = _write_model(
        workspace, "biovid_vmae", "BioVid", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      _write_aggregate(
        first,
        "linear",
        unbc_vmae,
        mint_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )
      _write_aggregate(
        second,
        "linear",
        biovid_vmae,
        unbc_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )

      with self.assertRaisesRegex(ValueError, "inverse datasets"):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )

  def test_rejects_duplicate_methods(self):
    """Reject multiple selected aggregates for the same method."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      for aggregate_name in ("aggregated_1", "aggregated_2"):
        _write_aggregate(
          first,
          "linear",
          unbc_vmae,
          mint_dfer,
          [_summary_row("linear", "projector_linear", values)],
          aggregate_name=aggregate_name,
        )
      _write_aggregate(
        second,
        "linear",
        mint_vmae,
        unbc_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )

      with self.assertRaisesRegex(ValueError, "Duplicate aggregate.*linear"):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )

  def test_metric_consistency_tolerates_roundoff_but_rejects_invalid_values(self):
    """Accept float32-scale noise without hiding missing or conflicting metrics."""
    for check in ("projector_only", "baseline"):
      for first_value, second_value, accepted in (
        (4.38866149044037, 4.38866148662567, True),
        (4.0, 4.0000002, True),
        (0.0, 5e-9, True),
        (4.0, 4.01, False),
        (4.0, float("nan"), False),
      ):
        with self.subTest(check=check, values=(first_value, second_value)):
          with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = _write_model(root, "old", "UNBC", "VIDEOMAE_v2_S")
            new = _write_model(root, "new", "MIntPAIN", "DFER")
            rows = [
              _summary_row("linear", mode, (0.6,) * 8)
              for mode in ("linear_only", "projector_linear")
            ]
            column = (
              "srctest_mae_macro_before" if check == "projector_only"
              else "srctest_mae_micro_old"
            )
            for row, value in zip(rows, (first_value, second_value)):
              row[column] = value
            if check == "projector_only":
              _write_aggregate(root, "linear", old, new, rows)
            else:
              rows[1]["interpolation_similarity"] = "mlp"
              rows[1]["refine_mode"] = "linear_only"
              for method, row in zip(("linear", "mlp"), rows):
                _write_aggregate(root, method, old, new, [row])
            stage = "projector_only" if check == "projector_only" else "linear_only"
            if accepted:
              direction = latex_table._load_direction(root, "real", stage, None)
              self.assertEqual(direction["rows"][0][column], first_value)
              self.assertTrue(direction["stage_available"])
            else:
              with self.assertRaisesRegex(ValueError, f"Inconsistent {check}"):
                latex_table._load_direction(root, "real", stage, None)

  def test_rejects_inconsistent_baselines(self):
    """Reject method summaries that disagree about a native baseline."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      first_values = {
        "linear": (0.8, 0.9, 1.2, 1.3, 0.70, 0.8, 1.0, 1.1),
        "mlp": (0.8, 0.9, 1.2, 1.3, 0.75, 0.8, 1.0, 1.1),
      }
      second_values = (1.2, 1.3, 0.8, 0.9, 1.0, 1.1, 0.7, 0.8)
      for method, values in first_values.items():
        _write_aggregate(
          first,
          method,
          unbc_vmae,
          mint_dfer,
          [_summary_row(method, "projector_linear", values)],
        )
        _write_aggregate(
          second,
          method,
          mint_vmae,
          unbc_dfer,
          [_summary_row(method, "projector_linear", second_values)],
        )

      with self.assertRaisesRegex(
        ValueError, "Inconsistent baseline.*srctest_mae_micro_old"
      ):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )
      latex = generate_table(
        first,
        second,
        projection="real",
        stage="projector_linear",
        skip_consistency_checks=True,
      )

    self.assertIn(r"\begin{table}[H]", latex)

  def test_rejects_conflicting_model_metadata_within_root(self):
    """Reject aggregate methods that point at different trained datasets."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      biovid_vmae = _write_model(
        workspace, "biovid_vmae", "BioVid", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      biovid_dfer = _write_model(workspace, "biovid_dfer", "BioVid", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      for method, old_model in (
        ("linear", unbc_vmae),
        ("mlp", biovid_vmae),
      ):
        _write_aggregate(
          first,
          method,
          old_model,
          mint_dfer,
          [_summary_row(method, "projector_linear", values)],
        )
        _write_aggregate(
          second,
          method,
          mint_vmae,
          biovid_dfer,
          [_summary_row(method, "projector_linear", values)],
        )

      with self.assertRaisesRegex(ValueError, "Conflicting model metadata"):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )

  def test_filters_projection_distribution_and_stage(self):
    """Select only the requested real/fake projection and refinement strategy."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      real_linear_only = (0.51, 0.52, 0.53, 0.54, 0.7, 0.8, 1.0, 1.1)
      real_projector = (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1)
      fake_linear_only = (0.81, 0.82, 0.83, 0.84, 0.7, 0.8, 1.0, 1.1)
      fake_projector = (0.91, 0.92, 0.93, 0.94, 0.7, 0.8, 1.0, 1.1)
      for root, old_model, new_model in (
        (first, unbc_vmae, mint_dfer),
        (second, mint_vmae, unbc_dfer),
      ):
        _write_aggregate(
          root,
          "linear",
          old_model,
          new_model,
          [
            _summary_row("linear", "linear_only", real_linear_only),
            _summary_row("linear", "projector_linear", real_projector),
          ],
        )
        _write_aggregate(
          root,
          "linear",
          old_model,
          new_model,
          [
            _summary_row("linear", "linear_only", fake_linear_only),
            _summary_row("linear", "projector_linear", fake_projector),
          ],
          fake_distribution="standard_normal",
        )

      real = generate_table(
        first,
        second,
        projection="real",
        stage="linear_only",
      )
      fake = generate_table(
        first,
        second,
        projection="fake",
        fake_distribution="standard_normal",
        stage="projector_linear",
      )

    self.assertIn("0.51 & 0.52 & 0.53 & 0.54", real)
    self.assertNotIn("0.61 & 0.62 & 0.63 & 0.64", real)
    self.assertIn("0.91 & 0.92 & 0.93 & 0.94", fake)
    self.assertIn("fake_projection_standard_normal", fake)
    self.assertNotIn("0.81 & 0.82 & 0.83 & 0.84", fake)

  def test_fake_adapter_selection_averages_seed_aggregates(self):
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      first_old = _write_model(workspace, "first_old", "BioVid", "VIDEOMAE_v2_S")
      first_new = _write_model(workspace, "first_new", "MIntPAIN", "DFER")
      second_old = _write_model(workspace, "second_old", "MIntPAIN", "VIDEOMAE_v2_S")
      second_new = _write_model(workspace, "second_new", "BioVid", "DFER")
      for root, old_model, new_model in (
        (first, first_old, first_new),
        (second, second_old, second_new),
      ):
        for seed, micro in ((42, 0.8), (43, 1.0)):
          values = (micro, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
          _write_aggregate(
            root, "linear", old_model, new_model,
            [_summary_row("linear", "projector_linear", values)],
            fake_control="fake_adapter", adapter_seed=seed,
          )

      latex = generate_table(
        first, second, projection="fake_adapter", stage="projector_linear")

    self.assertIn("0.90 & 0.90 & 1.20 & 1.30", latex)
    self.assertIn("fake_adapter_random_init", latex)

  def test_fake_adapter_rejects_duplicate_aggregate_for_one_seed(self):
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      first_old = _write_model(workspace, "first_old", "BioVid", "VIDEOMAE_v2_S")
      first_new = _write_model(workspace, "first_new", "MIntPAIN", "DFER")
      second_old = _write_model(workspace, "second_old", "MIntPAIN", "VIDEOMAE_v2_S")
      second_new = _write_model(workspace, "second_new", "BioVid", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      for aggregate_name in ("aggregated_fake_a", "aggregated_fake_b"):
        _write_aggregate(
          first, "linear", first_old, first_new,
          [_summary_row("linear", "projector_linear", values)],
          fake_control="fake_adapter", adapter_seed=42,
          aggregate_name=aggregate_name,
        )
      _write_aggregate(
        second, "linear", second_old, second_new,
        [_summary_row("linear", "projector_linear", values)],
        fake_control="fake_adapter", adapter_seed=42,
      )

      with self.assertRaisesRegex(ValueError, "Duplicate fake-adapter seed"):
        generate_table(
          first, second, projection="fake_adapter", stage="projector_linear")

  def test_fake_adapter_rejects_inconsistent_seed_configuration(self):
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      first_old = _write_model(workspace, "first_old", "BioVid", "VIDEOMAE_v2_S")
      first_new = _write_model(workspace, "first_new", "MIntPAIN", "DFER")
      second_old = _write_model(workspace, "second_old", "MIntPAIN", "VIDEOMAE_v2_S")
      second_new = _write_model(workspace, "second_new", "BioVid", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      for root, old_model, new_model in (
        (first, first_old, first_new),
        (second, second_old, second_new),
      ):
        for seed, num_anchors in ((42, 100), (43, 50)):
          row = _summary_row("linear", "projector_linear", values)
          row["num_anchors"] = num_anchors
          _write_aggregate(
            root, "linear", old_model, new_model, [row],
            fake_control="fake_adapter", adapter_seed=seed,
          )

      with self.assertRaisesRegex(
        ValueError, "Inconsistent fake-adapter configuration",
      ):
        generate_table(
          first, second, projection="fake_adapter", stage="projector_linear")

  def test_projector_only_uses_matching_before_values(self):
    """Collapse matching pre-refinement rows instead of selecting either refinement."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      for root, old_model, new_model in (
        (first, unbc_vmae, mint_dfer),
        (second, mint_vmae, unbc_dfer),
      ):
        _write_aggregate(
          root,
          "linear",
          old_model,
          new_model,
          [
            _summary_row(
              "linear", "linear_only",
              (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1),
              projected=(0.71, 0.72),
            ),
            _summary_row(
              "linear", "projector_linear",
              (0.51, 0.52, 0.53, 0.54, 0.7, 0.8, 1.0, 1.1),
              projected=(0.71, 0.72),
            ),
          ],
        )

      latex = generate_table(
        first,
        second,
        projection="real",
        stage="projector_only",
        decimals=3,
      )

    self.assertIn("Linear & 100 & 0.710 & 0.720 & 1.000 & 1.100", latex)
    self.assertNotIn("0.610 & 0.620", latex)
    self.assertNotIn("0.510 & 0.520", latex)

  def test_projector_only_rejects_disagreeing_before_values(self):
    """Reject ambiguous projector-only values duplicated across refinement rows."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      rows = [
        _summary_row(
          "linear", "linear_only",
          (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1),
          projected=(0.71, 0.72),
        ),
        _summary_row(
          "linear", "projector_linear",
          (0.51, 0.52, 0.53, 0.54, 0.7, 0.8, 1.0, 1.1),
          projected=(0.81, 0.72),
        ),
      ]
      _write_aggregate(first, "linear", unbc_vmae, mint_dfer, rows)
      _write_aggregate(second, "linear", mint_vmae, unbc_dfer, rows)

      with self.assertRaisesRegex(ValueError, "Inconsistent projector_only"):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_only",
        )
      latex = generate_table(
        first,
        second,
        projection="real",
        stage="projector_only",
        skip_consistency_checks=True,
      )

    self.assertIn(r"\begin{table}[H]", latex)

  def test_all_stages_renders_three_distinct_tables(self):
    """Render projector-only and both refinement modes into one output."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      rows = [
        _summary_row(
          "linear", "linear_only",
          (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1),
          projected=(0.71, 0.72),
        ),
        _summary_row(
          "linear", "projector_linear",
          (0.51, 0.52, 0.53, 0.54, 0.7, 0.8, 1.0, 1.1),
          projected=(0.71, 0.72),
        ),
      ]
      _write_aggregate(first, "linear", unbc_vmae, mint_dfer, rows)
      _write_aggregate(second, "linear", mint_vmae, unbc_dfer, rows)

      latex = generate_table(
        first,
        second,
        projection="real",
        stage="all",
        decimals=1,
      )

    self.assertEqual(latex.count(r"\begin{table}[H]"), 4)
    captions = [
      "Projector-only cross-projection results",
      "Linear-only refinement cross-projection results",
      "Projector-and-linear refinement cross-projection results",
      "Frozen-random-projector linear-head refinement cross-projection results",
    ]
    self.assertEqual([latex.index(caption) for caption in captions],
                     sorted(latex.index(caption) for caption in captions))
    for stage, caption in zip(
      latex_table.STAGES,
      captions,
    ):
      self.assertIn(f"real_{stage}", latex)
      self.assertLess(
        latex.index(rf"\caption{{{caption}"),
        latex.index(f"real_{stage}"),
      )
    self.assertIn("0.7 & 0.7 & 1.0 & 1.1", latex)
    self.assertIn("0.6 & 0.6 & 0.6 & 0.6", latex)
    self.assertIn("0.5 & 0.5 & 0.5 & 0.5", latex)

  def test_all_stages_uses_x_placeholder_for_unavailable_stage(self):
    """Keep discovered methods but blank every metric for an incomplete stage."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      rows = [
        _summary_row(
          "linear", "linear_only",
          (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1),
          projected=(0.71, 0.72),
        ),
      ]
      _write_aggregate(first, "linear", unbc_vmae, mint_dfer, rows)
      _write_aggregate(second, "linear", mint_vmae, unbc_dfer, rows)

      latex = generate_table(
        first,
        second,
        projection="real",
        stage="all",
      )

    self.assertIn(
      "Projector-and-linear refinement cross-projection results "
      "(unavailable:",
      latex,
    )
    projector_linear = latex.split(
      r"\label{tab_cross_projection_unbc_mintpain_real_projector_linear}"
    )[1]
    self.assertIn("Linear & 100 & X & X & X & X", projector_linear)

  def test_rejects_negative_decimal_count(self):
    """Require a non-negative output precision."""
    with self.assertRaisesRegex(ValueError, "decimals must be non-negative"):
      generate_table(
        "first",
        "second",
        projection="real",
        stage="projector_only",
        decimals=-1,
      )

  @unittest.skipUnless(
    REFERENCE_TABLE.is_file() and all(root.is_dir() for root in REFERENCE_ROOTS),
    "reference table or experiment artifacts are unavailable",
  )
  def test_real_projector_linear_reproduces_mint_unbc_reference_table(self):
    """Reproduce the supplied table structure and values within manual rounding."""
    latex = generate_table(
      *REFERENCE_ROOTS,
      projection="real",
      stage="projector_linear",
    )
    expected = _canonical_rows(REFERENCE_TABLE.read_text(encoding="utf-8"))
    actual = _canonical_rows(latex)

    self.assertEqual(len(actual), len(expected))
    for actual_section, expected_section in zip(actual, expected):
      self.assertEqual(len(actual_section), len(expected_section))
      for actual_row, expected_row in zip(actual_section, expected_section):
        self.assertEqual(actual_row[:2], expected_row[:2])
        for actual_value, expected_value in zip(actual_row[2:], expected_row[2:]):
          if expected_value == "X":
            self.assertEqual(actual_value, "X")
          else:
            self.assertAlmostEqual(
              float(actual_value), float(expected_value), delta=0.011
            )

  def test_cli_reports_data_errors_without_traceback(self):
    """Report invalid experiment data as a concise CLI error."""
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      result = subprocess.run(
        [
          sys.executable,
          str(REPO_ROOT / "cross_space_generate_latex_table.py"),
          str(root),
          str(root),
          "--projection",
          "real",
          "--stage",
          "projector_linear",
          "--output",
          str(root / "table.tex"),
        ],
        capture_output=True,
        text=True,
        check=False,
      )

    self.assertNotEqual(result.returncode, 0)
    self.assertIn("No real aggregate PKLs", result.stderr)
    self.assertNotIn("Traceback", result.stderr)

  def test_cli_defaults_output_to_named_z_latex_table(self):
    """Derive a timestamped output path from the first projection direction."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      _write_aggregate(
        first,
        "linear",
        unbc_vmae,
        mint_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )
      _write_aggregate(
        second,
        "linear",
        mint_vmae,
        unbc_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )
      expected = (
        workspace / "z_latex_tables"
        / "VMAEv2-S-UNBC_DFER-MIntPAIN_real-projector_linear_1234567890.tex"
      )
      argv = [
        "cross_space_generate_latex_table.py",
        str(first),
        str(second),
        "--projection",
        "real",
        "--stage",
        "projector_linear",
      ]
      with (
        mock.patch.object(latex_table, "__file__", str(workspace / "script.py")),
        mock.patch.object(latex_table.time, "time", return_value=1234567890),
        mock.patch.object(sys, "argv", argv),
      ):
        latex_table.main()

      self.assertTrue(expected.is_file())

  def test_cli_accepts_one_root_and_uses_a_root_based_default_filename(self):
    """Catch requiring a second positional root in consolidated-summary mode."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      root = workspace / "experiment_root"
      root.mkdir()
      old_model = _write_model(root, "old", "UNBC", "VIDEOMAE_v2_S")
      new_model = _write_model(root, "new", "MIntPAIN", "DFER")
      values = (0.61, 0.62, 0.63, 0.64, 0.7, 0.8, 1.0, 1.1)
      pkl_path = _write_aggregate(
        root / "cross-validation_UNBC-to-MIntPAIN",
        "linear",
        old_model,
        new_model,
        [_summary_row("linear", "random_projector_linear", values)],
      )
      _write_root_summary(root, [pkl_path])
      expected = (
        workspace / "z_latex_tables"
        / "experiment_root_real-random_projector_linear_1234567890.tex"
      )
      argv = [
        "cross_space_generate_latex_table.py",
        str(root),
        "--projection",
        "real",
        "--stage",
        "random_projector_linear",
      ]
      with (
        mock.patch.object(latex_table, "__file__", str(workspace / "script.py")),
        mock.patch.object(latex_table.time, "time", return_value=1234567890),
        mock.patch.object(sys, "argv", argv),
      ):
        latex_table.main()

      self.assertTrue(expected.is_file())

  def test_cli_rejects_invalid_fake_distribution_combinations(self):
    """Require a fake distribution exactly when fake projection is selected."""
    cases = [
      (
        ["--projection", "fake"],
        "--fake-distribution is required for fake projections",
      ),
      (
        [
          "--projection",
          "real",
          "--fake-distribution",
          "standard_normal",
        ],
        "--fake-distribution is only valid for fake projections",
      ),
    ]
    for projection_args, message in cases:
      with self.subTest(projection_args=projection_args):
        result = subprocess.run(
          [
            sys.executable,
            str(REPO_ROOT / "cross_space_generate_latex_table.py"),
            "first",
            "second",
            *projection_args,
            "--stage",
            "projector_linear",
            "--output",
            "table.tex",
          ],
          capture_output=True,
          text=True,
          check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

  def test_rejects_missing_or_malformed_adjacent_summary(self):
    """Reject an aggregate whose logging summary is absent or malformed."""
    with tempfile.TemporaryDirectory() as tmp:
      workspace = Path(tmp)
      first = workspace / "first"
      second = workspace / "second"
      unbc_vmae = _write_model(
        workspace, "unbc_vmae", "UNBC", "VIDEOMAE_v2_S"
      )
      mint_dfer = _write_model(workspace, "mint_dfer", "MIntPAIN", "DFER")
      mint_vmae = _write_model(
        workspace, "mint_vmae", "MIntPAIN", "VIDEOMAE_v2_S"
      )
      unbc_dfer = _write_model(workspace, "unbc_dfer", "UNBC", "DFER")
      values = (0.8, 0.9, 1.2, 1.3, 0.7, 0.8, 1.0, 1.1)
      _write_aggregate(
        first,
        "linear",
        unbc_vmae,
        mint_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )
      _write_aggregate(
        second,
        "linear",
        mint_vmae,
        unbc_dfer,
        [_summary_row("linear", "projector_linear", values)],
      )
      summary_path = next(first.rglob("summary.csv"))
      summary_path.unlink()

      with self.assertRaisesRegex(ValueError, "Missing aggregate summary"):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )

      malformed = _summary_row("linear", "projector_linear", values)
      malformed.pop("refine_mode")
      pd.DataFrame([malformed]).to_csv(summary_path, index=False)
      with self.assertRaisesRegex(
        ValueError, "Missing summary columns.*refine_mode"
      ):
        generate_table(
          first,
          second,
          projection="real",
          stage="projector_linear",
        )


if __name__ == "__main__":
  unittest.main()

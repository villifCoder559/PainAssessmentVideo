from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from analysis.analyze_anchor_sweep_mae import (
    METHODS,
    _read_controlled_ablation,
    artifact_record,
    build_projector,
    compute_anchor_effects,
    compute_method_dispersion,
    evaluate_checkpoint_pairs,
    mae_micro_macro,
    render_analysis_report,
    render_investigation_report,
    run_synthetic_probe,
    select_best_and_efficient_k,
    validate_artifact_record,
    validate_cell_alignment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "analysis" / "configs"


def test_micro_and_macro_mae_use_sample_and_rounded_label_weighting():
    labels = np.array([0.1, 0.2, 0.4, 1.2])
    predictions = np.array([0.1, 1.2, 2.4, 3.2])

    micro, macro = mae_micro_macro(labels, predictions)

    assert micro == pytest.approx(1.25)
    assert macro == pytest.approx(((0.0 + 1.0 + 2.0) / 3.0 + 2.0) / 2.0)


def test_best_and_one_percent_efficient_k_prefer_smallest_eligible_k():
    best, efficient = select_best_and_efficient_k(
        {5: 1.02, 25: 1.009, 100: 1.0, 500: 1.004}
    )

    assert best == 100
    assert efficient == 25


def _anchor_fixture() -> pd.DataFrame:
    rows = []
    # Pair 0 improves at every step; pair 1 improves overall but has a reversal.
    values = {
        (0, 5): 4.0,
        (0, 25): 3.0,
        (0, 100): 1.0,
        (1, 5): 5.0,
        (1, 25): 6.0,
        (1, 100): 4.0,
    }
    for pair in (0, 1):
        for anchors in (5, 25, 100):
            rows.append(
                {
                    "direction": "a-to-b",
                    "method": "linear",
                    "refinement_mode": "linear_only",
                    "configured_anchors": anchors,
                    "new_idx": pair,
                    "old_idx": 0,
                    "source_after_micro": values[pair, anchors],
                    "target_after_micro": values[pair, anchors] + 1.0,
                    "source_after_macro": values[pair, anchors] + 0.25,
                    "target_after_macro": values[pair, anchors] + 1.25,
                }
            )
    return pd.DataFrame(rows)


def test_anchor_effects_use_lower_mae_as_improvement_and_measure_consistency():
    result = compute_anchor_effects(_anchor_fixture())
    row = result[
        (result.metric == "micro") & (result.stage == "source_after")
    ].iloc[0]

    assert row.mean_at_smallest_k == pytest.approx(4.5)
    assert row.mean_at_largest_k == pytest.approx(2.5)
    assert row.absolute_change_largest_minus_smallest == pytest.approx(-2.0)
    assert row.improvement_smallest_minus_largest == pytest.approx(2.0)
    assert row.fold_pair_improvement_fraction == pytest.approx(1.0)
    assert row.adjacent_step_improvement_fraction == pytest.approx(0.75)
    assert row.fold_pair_change_sd > 0
    assert row.method_rank_at_largest_k == 1
    assert bool(row.micro_macro_endpoint_sign_agreement)


def test_dispersion_distinguishes_exact_equality_from_display_rounding():
    rows = []
    for method, before, after, macro_after in (
        ("linear", 2.0, 1.0041, 1.2),
        ("mlp", 2.2, 1.0049, 1.3),
        ("autoencoder", 2.4, 1.0044, 1.1),
        ("procrustes", 2.6, 1.0046, 1.4),
    ):
        rows.append(
            {
                "direction": "a-to-b",
                "method": method,
                "configured_anchors": 100,
                "refinement_mode": "projector_linear",
                "new_idx": 0,
                "old_idx": 0,
                "source_before_micro": before,
                "source_after_micro": after,
                "source_before_macro": before + 0.1,
                "source_after_macro": macro_after,
                "target_before_micro": before + 1.0,
                "target_after_micro": after + 1.0,
                "target_before_macro": before + 1.1,
                "target_after_macro": macro_after + 1.0,
            }
        )

    result = compute_method_dispersion(pd.DataFrame(rows), display_decimals=2)
    source_micro = result[
        (result.domain == "source") & (result.metric == "micro")
    ].iloc[0]

    assert source_micro.range_before == pytest.approx(0.6)
    assert source_micro.range_after == pytest.approx(0.0008)
    assert not bool(source_micro.full_precision_equal_after)
    assert bool(source_micro.display_rounded_equal_after)
    assert not bool(source_micro.micro_macro_ranking_agree_after)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "misaligned"])
def test_cell_alignment_rejects_missing_duplicate_and_fold_misalignment(fault):
    rows = [
        {"method": method, "new_idx": new, "old_idx": old}
        for method in METHODS
        for new in (0, 1)
        for old in (0, 1)
    ]
    frame = pd.DataFrame(rows)
    if fault == "missing":
        frame = frame.iloc[:-1]
    elif fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    else:
        frame.loc[frame.index[-1], "new_idx"] = 2

    with pytest.raises(ValueError, match="cell alignment"):
        validate_cell_alignment(frame, methods=METHODS, folds=(0, 1))


def test_artifact_hash_guard_detects_changed_file(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    record = artifact_record(path)

    assert record["sha256"] == hashlib.sha256(b"first").hexdigest()
    validate_artifact_record(record)
    path.write_bytes(b"second")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifact_record(record)


@pytest.mark.parametrize(
    ("kind", "expected_keys"),
    [
        ("linear", {"weight", "bias"}),
        ("mlp", {"0.weight", "0.bias", "2.weight", "2.bias"}),
        (
            "autoencoder",
            {"0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias"},
        ),
        ("procrustes", {"weight", "bias"}),
    ],
)
def test_builds_all_checkpoint_architectures(kind, expected_keys):
    projector = build_projector(
        kind, d_old=8, d_new=12, activation="gelu", num_layers=1, encoder_ratio=4
    )

    assert set(projector.state_dict()) == expected_keys
    assert projector(torch.zeros(3, 8)).shape == (3, 12)


def test_checkpoint_pair_evaluation_applies_normalization_and_label_denormalization():
    projectors = {}
    heads = {}
    for name, scale in (("before", 1.0), ("after", 2.0)):
        projector = torch.nn.Linear(1, 1)
        head = torch.nn.Linear(1, 1)
        with torch.no_grad():
            projector.weight.fill_(scale)
            projector.bias.zero_()
            head.weight.fill_(scale)
            head.bias.zero_()
        projectors[name] = projector
        heads[name] = head
    norm_stats = {
        "old_mean": np.array([1.0], np.float32),
        "old_std": np.array([2.0], np.float32),
        "new_mean": np.array([10.0], np.float32),
        "new_std": np.array([3.0], np.float32),
    }

    result = evaluate_checkpoint_pairs(
        np.array([[3.0]], np.float32),
        np.array([22.0], np.float32),
        projectors,
        heads,
        norm_stats=norm_stats,
        label_denorm=2.0,
    )

    # before projector: ((3-1)/2)*1*3+10 = 13; before head *2 = 26.
    assert result["P_before_H_before"]["prediction_l2"] == pytest.approx(26.0)
    assert result["P_before_H_before"]["micro_mae"] == pytest.approx(4.0)
    assert result["P_after_H_after"]["micro_mae"] == pytest.approx(42.0)
    assert result["P_after_H_after"]["prediction_l2_from_baseline"] == pytest.approx(38.0)


def test_checkpoint_pair_evaluation_reconstructs_stored_source_and_target_mae():
    projector = torch.nn.Linear(1, 1)
    head = torch.nn.Linear(1, 1)
    with torch.no_grad():
        projector.weight.fill_(1.0)
        projector.bias.zero_()
        head.weight.fill_(1.0)
        head.bias.zero_()
    reconstructed = evaluate_checkpoint_pairs(
        np.array([[0.0], [2.0]], np.float32),
        np.array([0.0, 1.0], np.float32),
        {"before": projector, "after": projector},
        {"before": head, "after": head},
    )
    target_micro, target_macro = mae_micro_macro(
        np.array([0.0, 1.0]), np.array([0.5, 1.5])
    )

    assert reconstructed["P_before_H_before"]["micro_mae"] == pytest.approx(0.5)
    assert reconstructed["P_before_H_before"]["macro_mae"] == pytest.approx(0.5)
    assert target_micro == pytest.approx(0.5)
    assert target_macro == pytest.approx(0.5)


def test_synthetic_probe_is_deterministic_and_joint_training_compresses_dispersion():
    first = run_synthetic_probe(seed=42)
    second = run_synthetic_probe(seed=42)

    assert first == second
    assert first["linear_only_range"] > 0.05
    assert first["projector_linear_range"] < first["linear_only_range"] * 0.25


def test_controlled_ablation_reader_accepts_optuna_single_refinement_schema(tmp_path):
    trial = (
        tmp_path
        / "unbc_to_biovid_fold00_k100"
        / "search"
        / "trial0000_linear_projector_linear"
        / "results.pkl"
    )
    trial.parent.mkdir(parents=True)
    payload = {
        "trial_params": {
            "num_anchors": 100,
            "interpolation_similarity": "linear",
            "refine_mode": "projector_linear",
        },
        "refinement": {
            "refine_mode": "projector_linear",
            "mae_micro_old_oncsv_after": 0.7,
            "mae_macro_old_oncsv_after": 0.8,
            "mae_micro_new_test_after": 0.9,
            "mae_macro_new_test_after": 1.0,
        },
    }
    with trial.open("wb") as stream:
        pickle.dump(payload, stream)

    result = _read_controlled_ablation(tmp_path)
    row = result[
        (result.evidence_type == "matched_gpu")
        & (result.direction == "unbc-to-biovid")
        & (result.method == "linear")
        & (result.refinement_mode == "projector_linear")
    ].iloc[0]

    assert row.status == "complete"
    assert row.source_after_micro == pytest.approx(0.7)
    assert Path(row.result_path) == trial.resolve()


def test_ablation_yamls_match_required_recipe_and_production_yaml_contract():
    configs = [
        yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))
        for name in (
            "refinement_ablation_biovid_to_mintpain_fold00_k100.yaml",
            "refinement_ablation_unbc_to_biovid_fold00_k100.yaml",
        )
    ]

    for config in configs:
        assert config["num_anchors"] == [100]
        assert config["seed"] == [42]
        assert config["anchor_selection_type"] == ["balance_class_random"]
        assert config["csv_anchor_selection"] == ["train"]
        assert config["old_model_csv"] == ["test"]
        assert config["interpolation_similarity"] == list(METHODS)
        assert config["refinement"] == 3
        assert len(config["new_model_pth"]) == len(config["old_model_pth"]) == 1
        assert config["linear_projector"] == {
            "lr": [1e-5],
            "batch_size": [64],
            "optimizer": "adamw",
            "weight_decay": 0,
            "epochs": 750,
            "normalize_embeddings": [False],
            "loss": ["mse"],
            "encoder_ratio": 4,
        }
        assert config["refinement_config"] == {
            "lr_projector": [1e-4],
            "lr_linear": [1e-4],
            "lambda_B": [1e-4],
            "lambda_A": [1e-3],
            "optimizer": "adamw",
            "weight_decay": 0,
            "epochs": 150,
            "loss": ["mse"],
            "batch_size": 64,
        }
    assert configs[0]["refinement_config"] == configs[1]["refinement_config"]


def test_reports_render_required_sections_and_dependency_warning():
    effects = compute_anchor_effects(_anchor_fixture())
    dispersion_source = []
    for method, offset in zip(METHODS, (0.0, 0.1, 0.2, 0.3)):
        for row in _anchor_fixture().to_dict("records"):
            row = dict(row)
            row["method"] = method
            for key in (
                "source_after_micro",
                "target_after_micro",
                "source_after_macro",
                "target_after_macro",
            ):
                row[key] += offset
            row.update(
                source_before_micro=row["source_after_micro"] + 1,
                source_before_macro=row["source_after_macro"] + 1,
                target_before_micro=row["target_after_micro"] + 1,
                target_before_macro=row["target_after_macro"] + 1,
            )
            dispersion_source.append(row)
    dispersion = compute_method_dispersion(pd.DataFrame(dispersion_source))

    analysis = render_analysis_report(effects, input_rows=10_000, selected_rows=8_000)
    investigation = render_investigation_report(
        dispersion,
        checkpoint_diagnostics=pd.DataFrame(),
        controlled_ablation=pd.DataFrame(),
        synthetic_probe=run_synthetic_probe(seed=42),
    )

    for heading in (
        "# Anchor-Sweep MAE Analysis",
        "## Methodology and coverage",
        "## Headline findings",
        "## Direction-level findings",
        "## Micro versus macro MAE",
        "## Source/target trade-offs",
        "## Robustness and limitations",
    ):
        assert heading in analysis
    for heading in (
        "# Anchor-Sweep Refinement-Mode Investigation",
        "## Observed phenomenon",
        "## Exact-equality audit",
        "## Implementation-derived objective",
        "## Hypothesis ledger",
        "## Checkpoint decomposition",
        "## Deterministic CPU probe",
        "## Matched GPU ablations",
        "## Evidence synthesis and limitations",
    ):
        assert heading in investigation
    assert "dependent Cartesian product" in analysis
    assert "dependent Cartesian product" in investigation
    assert "no p-values" in analysis


def test_reports_surface_traceable_audit_values_and_direction_limited_evidence():
    analysis_rows = []
    for method, offset in zip(METHODS, (0.0, 0.1, 0.2, 0.3)):
        for row in _anchor_fixture().to_dict("records"):
            row = dict(row)
            row["method"] = method
            for key in (
                "source_after_micro",
                "target_after_micro",
                "source_after_macro",
                "target_after_macro",
            ):
                row[key] += offset
            row.update(
                source_before_micro=row["source_after_micro"] + 1,
                source_before_macro=row["source_after_macro"] + 1,
                target_before_micro=row["target_after_micro"] + 1,
                target_before_macro=row["target_after_macro"] + 1,
            )
            analysis_rows.append(row)
    source = pd.DataFrame(analysis_rows)
    effects = compute_anchor_effects(source)
    dispersion = compute_method_dispersion(source)

    checkpoints = pd.DataFrame(
        [
            {
                "projector_checkpoints_distinct": True,
                "head_checkpoints_distinct": True,
                "source_predictions_distinct": True,
                "target_predictions_distinct": True,
                "source_sample_ids_aligned_across_methods": True,
                "target_sample_ids_aligned_across_methods": True,
                "source_after_cross_method_unique_prediction_hashes": 4,
                "target_after_cross_method_unique_prediction_hashes": 4,
                "stored_source_before_micro": 1.0,
                "reconstructed_source_before_micro": 1.0 + 1e-7,
                "stored_source_after_micro": 0.9,
                "reconstructed_source_after_micro": 0.9,
                "stored_source_before_macro": 1.1,
                "reconstructed_source_before_macro": 1.1,
                "stored_source_after_macro": 1.0,
                "reconstructed_source_after_macro": 1.0,
                "stored_target_before_micro": 1.2,
                "reconstructed_target_before_micro": 1.2,
                "stored_target_after_micro": 1.1,
                "reconstructed_target_after_micro": 1.1,
                "stored_target_before_macro": 1.3,
                "reconstructed_target_before_macro": 1.3,
                "stored_target_after_macro": 1.2,
                "reconstructed_target_after_macro": 1.2,
            }
        ]
    )
    controlled_rows = []
    for direction in ("a-to-b", "c-to-d"):
        for method, frozen, joint in zip(
            METHODS,
            (1.0, 1.1, 1.2, 1.3),
            (0.9, 0.91, 0.92, 0.93),
        ):
            for mode, value, mode_range in (
                ("linear_only", frozen, 0.3),
                ("projector_linear", joint, 0.03),
            ):
                controlled_rows.append(
                    {
                        "evidence_type": "matched_gpu",
                        "direction": direction,
                        "method": method,
                        "refinement_mode": mode,
                        "status": "complete",
                        "source_after_micro": value,
                        "cross_method_range_source_after_micro": mode_range,
                        "projector_linear_to_linear_only_range_ratio": 0.1,
                    }
                )
    controlled = pd.DataFrame(controlled_rows)

    analysis = render_analysis_report(effects, input_rows=10_000, selected_rows=8_000)
    investigation = render_investigation_report(
        dispersion,
        checkpoint_diagnostics=checkpoints,
        controlled_ablation=controlled,
        synthetic_probe=run_synthetic_probe(seed=42),
    )

    for display_name in ("Linear", "MLP", "Autoencoder", "Procrustes"):
        assert f"| {display_name} |" in analysis
        assert f"| {display_name} |" in investigation
    assert "Coverage counts are recorded in `run_metadata.json`" in analysis
    assert "0 of 12 after-refinement cells are exactly equal" in investigation
    assert "1/1 projector checkpoints" in investigation
    assert "maximum absolute reconstruction difference is 1.00e-07" in investigation
    assert "| a-to-b | Linear | 1.000000 | 0.900000 |" in investigation
    assert (
        "| Frozen and jointly trained refinement update different parameter sets | supported |"
        in investigation
    )
    assert (
        "| Joint training uniformly erases all method differences | rejected |"
        in investigation
    )
    assert (
        "| One universal causal explanation covers every direction | inconclusive |"
        in investigation
    )
    assert "two controlled directions" in investigation

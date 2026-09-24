from unittest import mock
from pathlib import Path
import sys
from types import SimpleNamespace
import pickle

import numpy as np
import pandas as pd
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# custom.helper starts a multiprocessing Manager while cross_space_projection is
# imported. These tests exercise only the pure refinement-selection helpers.
with mock.patch("multiprocessing.Manager") as manager:
    manager.return_value.dict.return_value = {}
    import cross_space_projection as csp

with mock.patch.dict(sys.modules, {"umap": SimpleNamespace(UMAP=object)}):
    import cross_space_logs as csl


def _source_inference(sample_ids, labels, predictions):
    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    return {
        "sample_ids": sample_ids,
        "labels": np.asarray(labels, dtype=np.float32),
        "predictions": np.asarray(predictions, dtype=np.float32),
        "embeddings": np.column_stack((sample_ids, sample_ids + 100)).astype(np.float32),
    }


def test_default_refinement_budget_uses_num_anchors_and_source_quality_scores():
    candidates = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 0, 1, 1, 1],
        }
    )
    source = _source_inference(
        sample_ids=[1, 2, 3, 4, 5, 6],
        labels=[0, 0, 0, 0, 0, 0],
        predictions=[0.8, 0.1, 0.4, 0.5, 0.2, 0.9],
    )

    selected_df, selected_source, metadata = csp._select_refinement_source_subset(
        candidates,
        source,
        num_refinement_samples=0,
        num_anchors=2,
        selection_type="balance_class_quality",
    )

    assert selected_df["sample_id"].tolist() == [2, 5]
    assert selected_source["sample_ids"].tolist() == [2, 5]
    assert selected_source["embeddings"][:, 0].tolist() == [2.0, 5.0]
    assert metadata == {
        "num_refinement_samples": 0,
        "refinement_sample_budget": 2,
        "num_refinement_samples_real": 2,
    }


def test_full_refinement_sentinel_keeps_every_source_sample():
    candidates = pd.DataFrame(
        {"sample_id": [30, 10, 20], "class_id": [1, 0, 0]}
    )
    source = _source_inference(
        sample_ids=[30, 10, 20],
        labels=[3, 1, 2],
        predictions=[3, 1, 2],
    )

    selected_df, selected_source, metadata = csp._select_refinement_source_subset(
        candidates,
        source,
        num_refinement_samples=-1,
        num_anchors=2,
        selection_type="random",
    )

    assert selected_df["sample_id"].tolist() == [30, 10, 20]
    assert selected_source is source
    assert metadata == {
        "num_refinement_samples": -1,
        "refinement_sample_budget": 3,
        "num_refinement_samples_real": 3,
    }


def test_positive_refinement_budget_preserves_balanced_strategy_overshoot():
    candidates = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 1, 1, 2, 2],
        }
    )
    source = _source_inference(
        sample_ids=[1, 2, 3, 4, 5, 6],
        labels=[0, 0, 1, 1, 2, 2],
        predictions=[0, 0, 1, 1, 2, 2],
    )

    selected_df, selected_source, metadata = csp._select_refinement_source_subset(
        candidates,
        source,
        num_refinement_samples=2,
        num_anchors=99,
        selection_type="balance_class_random",
    )

    assert len(selected_df) == 3
    assert selected_df["class_id"].tolist() == [0, 1, 2]
    assert selected_source["sample_ids"].tolist() == selected_df["sample_id"].tolist()
    assert metadata["refinement_sample_budget"] == 2
    assert metadata["num_refinement_samples_real"] == 3


@pytest.mark.parametrize("value", [-2, -10])
def test_refinement_sample_count_rejects_values_below_minus_one(value):
    with pytest.raises(ValueError, match="num_refinement_samples.*-1, 0, or a positive"):
        csp._validate_num_refinement_samples([0, value, 5])


def test_refinement_stage_trains_on_selected_source_rows_and_saves_audit_csv(tmp_path):
    source_df = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 0, 1, 1, 1],
        }
    )
    source = _source_inference(
        sample_ids=[1, 2, 3, 4, 5, 6],
        labels=[0, 0, 0, 1, 1, 1],
        predictions=[0.8, 0.1, 0.4, 1.5, 1.2, 1.9],
    )
    source_val = _source_inference(
        sample_ids=[20, 21], labels=[0, 1], predictions=[0, 1]
    )
    anchors_old = _source_inference(
        sample_ids=[100, 101], labels=[0, 1], predictions=[0, 1]
    )
    anchors_new = _source_inference(
        sample_ids=[100, 101], labels=[0, 1], predictions=[0, 1]
    )
    new_eval = _source_inference(
        sample_ids=[200, 201], labels=[0, 1], predictions=[0, 1]
    )
    new_test = _source_inference(
        sample_ids=[300, 301], labels=[0, 1], predictions=[0, 1]
    )
    new_model = SimpleNamespace(
        head=SimpleNamespace(linear=torch.nn.Linear(2, 1, bias=True))
    )
    cfg = dict(csp.REFINEMENT_CONFIG)
    cfg.update(
        {
            "enabled": True,
            "mode": "linear_only",
            "device": "cpu",
            "epochs": 1,
            "batch_size": 2,
            "optimizer": "sgd",
            "lr_linear": 1e-3,
        }
    )

    result = csp._run_refinement_stage(
        old_model=object(),
        new_model=new_model,
        old_model_pth="old.pt",
        new_model_pth="new.pt",
        old_config={},
        new_config={},
        old_features_path="unused",
        old_model_anchors=anchors_old,
        new_model_anchors_aligned=anchors_new,
        projector_bundle=None,
        old_model_tensors=None,
        old_model_csv="test",
        label_denorm=1.0,
        refine_dir=str(tmp_path),
        tag="sampled",
        emb_B=source,
        emb_B_val=source_val,
        new_eval=new_eval,
        new_test=new_test,
        mode="linear_only",
        sim_type="l2",
        rbf_sigma=1.0,
        cfg=cfg,
        source_df=source_df,
        num_refinement_samples=0,
        num_anchors=2,
        anchor_selection_type="balance_class_quality",
    )

    assert result["emb_B"]["sample_ids"].tolist() == [2, 5]
    audit_path = Path(result["metrics"]["refinement_samples_csv_path"])
    assert audit_path == tmp_path / "refinement_source_samples.csv"
    assert pd.read_csv(audit_path, sep="\t")["sample_id"].tolist() == [2, 5]
    assert result["metrics"]["num_refinement_samples"] == 0
    assert result["metrics"]["refinement_sample_budget"] == 2
    assert result["metrics"]["num_refinement_samples_real"] == 2


def test_yaml_forwards_refinement_sample_count_as_a_sweep_axis():
    assert csp._yaml_to_argv({"num_refinement_samples": [0, -1, 8]}) == [
        "--num_refinement_samples",
        "0",
        "-1",
        "8",
    ]


def test_search_space_contains_refinement_sample_count_axis():
    args = SimpleNamespace(
        num_anchors=[4, 8],
        num_refinement_samples=[0, -1, 12],
        anchor_selection_type=["random"],
        csv_anchor_selection=["train"],
        old_model_csv=["test"],
        interpolation_similarity=["linear"],
        mlp_activation=["gelu"],
        mlp_num_layers=[1],
        weighting_method=["none"],
        rbf_sigma=[1.0],
        refinement=2,
        projector_recipes=[csp.LINEAR_PROJECTOR_CONFIG],
        refinement_recipes=[csp.REFINEMENT_CONFIG],
    )

    assert csp._build_search_space(args)["num_refinement_samples"] == [0, -1, 12]


def test_grid_precompute_separates_refinement_bundles_by_source_sample_count(tmp_path):
    rows = pd.DataFrame(
        {
            "sample_id": [1, 2, 3, 4, 5, 6],
            "class_id": [0, 0, 0, 1, 1, 1],
            "subject_id": [10, 10, 11, 11, 12, 12],
        }
    )
    data_csv = tmp_path / "data.csv"
    rows.to_csv(data_csv, index=False, sep="\t")

    def fake_extract(_model, _checkpoint, csv_path, _config, **_kwargs):
        frame = pd.read_csv(csv_path, sep="\t")
        sample_ids = frame["sample_id"].to_numpy(dtype=np.int64)
        labels = frame["class_id"].to_numpy(dtype=np.float32)
        return {
            "sample_ids": sample_ids,
            "labels": labels,
            "predictions": labels.copy(),
            "embeddings": np.column_stack((sample_ids, sample_ids + 100)).astype(np.float32),
        }

    def fake_refinement(**kwargs):
        return {"num_refinement_samples": kwargs.get("num_refinement_samples")}

    args = SimpleNamespace(
        projector_recipes=[csp.LINEAR_PROJECTOR_CONFIG],
        refinement_recipes=[csp.REFINEMENT_CONFIG],
        interpolation_similarity=["cos"],
        mlp_activation=["gelu"],
        mlp_num_layers=[1],
        refinement=1,
        csv_anchor_selection=["train"],
        num_anchors=[2],
        num_refinement_samples=[0, 3],
        anchor_selection_type=["random"],
        old_model_csv=[],
        rbf_sigma=[1.0],
    )

    with (
        mock.patch.dict(csp.REFINEMENT_CONFIG, {"enabled": True}),
        mock.patch.object(csp, "_detect_dataset", return_value="NEW"),
        mock.patch.object(csp, "_detect_backbone", return_value="BACKBONE"),
        mock.patch.object(csp, "_get_features_path", return_value="old-backbone-new-domain"),
        mock.patch.object(csp, "_resolve_anchor_csvs", return_value=[str(data_csv)]),
        mock.patch.object(csp, "_resolve_split_csv", return_value=str(data_csv)),
        mock.patch.object(csp, "_resolve_test_csv_strict", return_value=str(data_csv)),
        mock.patch.object(csp, "clean_csv_from_augmentations", side_effect=lambda path: path),
        mock.patch.object(csp.helper, "set_step_shift"),
        mock.patch.object(csp, "_extract_embeddings", side_effect=fake_extract),
        mock.patch.object(csp, "_label_transform_from_config", return_value=1.0),
        mock.patch.object(csp, "_run_refinement_stage", side_effect=fake_refinement),
    ):
        anchor_cache, _ = csp._precompute_embeddings(
            object(),
            object(),
            "old-model.pt",
            "new-model.pt",
            {"config": {}},
            {"config": {}},
            "old-features",
            "new-features",
            args,
            str(tmp_path),
        )

    refinement_cache = anchor_cache[("train", 2, "random")]["refine_distance"]
    assert {key[-1] for key in refinement_cache} == {0, 3}
    assert {value["num_refinement_samples"] for value in refinement_cache.values()} == {0, 3}


def test_trial_uses_refinement_bundle_for_selected_source_sample_count(tmp_path):
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    labels = np.array([0.0, 1.0], dtype=np.float32)
    old_tensors = {
        "embeddings": embeddings,
        "labels": labels,
        "sample_ids": np.array([10, 11], dtype=np.int64),
        "predictions": labels.copy(),
    }
    linear_before = torch.nn.Linear(2, 1, bias=False)
    linear_three = torch.nn.Linear(2, 1, bias=False)
    linear_eight = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        linear_before.weight.zero_()
        linear_three.weight[:] = torch.tensor([[1.0, 0.0]])
        linear_eight.weight[:] = torch.tensor([[0.0, 1.0]])

    refinement_recipe = dict(csp.REFINEMENT_CONFIG)
    refinement_tag = csp._refinement_tag(refinement_recipe)

    def cached_refinement(linear_after, count):
        return {
            "refine_bundle": {
                "linear_before": linear_before,
                "linear_after": linear_after,
                "config": refinement_recipe,
                "metrics": [],
            },
            "metrics": {"num_refinement_samples": count},
            "new_test_eval": None,
        }

    anchor_key = ("train", 2, "random")
    anchor_cache = {
        anchor_key: {
            "old": {"embeddings": embeddings},
            "new": {"embeddings": embeddings},
            "projectors": {},
            "random_projectors": {},
            "refine_distance": {
                ("l2", 0.2, "linear_only", refinement_tag, 3): cached_refinement(
                    linear_three, 3
                ),
                ("l2", 0.2, "linear_only", refinement_tag, 8): cached_refinement(
                    linear_eight, 8
                ),
            },
        }
    }
    tensor_cache = {
        "test": {"old_tensors": old_tensors, "old_tensors_csv": "unused.csv"}
    }
    head = torch.nn.Module()
    head.linear = linear_before
    model = SimpleNamespace(head=head)
    params = {
        "num_anchors": 2,
        "num_refinement_samples": 8,
        "anchor_selection_type": "random",
        "csv_anchor_selection": "train",
        "old_model_csv": "test",
        "interpolation_similarity": "l2",
        "mlp_activation": "gelu",
        "mlp_num_layers": 1,
        "weighting_method": "rbf",
        "rbf_sigma": 0.2,
        "projector_config": "projector",
        "refinement_config": "refinement",
        "refine_mode": "linear_only",
    }

    with mock.patch.dict(
        csp.REFINEMENT_CONFIG,
        {"enabled": True, "report_after_refinement": True},
    ):
        csp._run_trial(
            params,
            0,
            anchor_cache,
            tensor_cache,
            model,
            {"config": {"normalize_labels": 0}},
            str(tmp_path),
            1,
            {"projector": dict(csp.LINEAR_PROJECTOR_CONFIG)},
            {"refinement": refinement_recipe},
        )

    with (tmp_path / "results.pkl").open("rb") as stream:
        result = pickle.load(stream)
    assert result["refinement"]["num_refinement_samples"] == 8


def test_summary_metadata_preserves_refinement_sample_count():
    params = csl._synth_trial_params_from_cfg({"num_refinement_samples": 12})
    assert params["num_refinement_samples"] == 12

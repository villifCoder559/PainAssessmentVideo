from unittest import mock
from pathlib import Path
import sys
from types import SimpleNamespace

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
            "optimizer": "adamw",
            "lr_linear": 1e-3,
        }
    )
    identity = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        identity.weight[:] = torch.eye(2)

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
        projector_bundle={"projector": identity, "norm_stats": None, "kind": "linear"},
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


def test_summary_metadata_preserves_refinement_sample_count():
    params = csl._synth_trial_params_from_cfg({"num_refinement_samples": 12})
    assert params["num_refinement_samples"] == 12


def test_refinement_stage_default_selector_is_supported():
    import inspect
    default = inspect.signature(csp._run_refinement_stage).parameters['anchor_selection_type'].default
    df = pd.DataFrame({'sample_id': range(4), 'class_id': [0, 0, 1, 1]})
    assert len(csp._select_anchors(df, 2, default)) == 2


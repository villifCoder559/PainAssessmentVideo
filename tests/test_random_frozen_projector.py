import copy
import os
import pickle
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


with mock.patch("multiprocessing.Manager") as manager:
    manager.return_value.dict.return_value = {}
    import cross_space_projection as csp


def _projector_cfg(**overrides):
    cfg = copy.deepcopy(csp.LINEAR_PROJECTOR_CONFIG)
    cfg.update({"device": "cpu", "normalize_embeddings": True})
    cfg.update(overrides)
    return cfg


def _refinement_cfg(**overrides):
    cfg = copy.deepcopy(csp.REFINEMENT_CONFIG)
    cfg.update({
        "device": "cpu",
        "epochs": 3,
        "batch_size": 4,
        "lr_linear": 0.05,
        "optimizer": "sgd",
        "weight_decay": 0,
        "lambda_B": 1.0,
        "lambda_A": 1.0,
        "loss": "mse",
    })
    cfg.update(overrides)
    return cfg


def _anchors(seed=0, n=8, d_old=4, d_new=3):
    rng = np.random.default_rng(seed)
    return (
        {"embeddings": rng.normal(size=(n, d_old)).astype(np.float32)},
        {"embeddings": rng.normal(size=(n, d_new)).astype(np.float32)},
    )


def test_refinement_flags_add_random_only_and_all_modes():
    assert csp._REFINE_FLAG_TO_MODES[4] == ["random_projector_linear"]
    assert csp._REFINE_FLAG_TO_MODES[5] == [
        "linear_only",
        "projector_linear",
        "random_projector_linear",
    ]
    assert csp._REFINE_FLAG_TO_MODES[3] == ["linear_only", "projector_linear"]


def test_random_mode_applies_only_to_randomizable_projectors():
    for kind in ("linear", "mlp", "autoencoder"):
        assert csp._applicable_refine_modes(4, kind, 10) == ["random_projector_linear"]
    for kind in ("procrustes", "linear_close", "cos", "geodesic"):
        assert csp._applicable_refine_modes(4, kind, 10) == []
    assert csp._applicable_refine_modes(4, "linear", 0) == []


def test_all_mode_keeps_legacy_fallbacks_for_nonrandom_projectors():
    assert csp._applicable_refine_modes(5, "linear", 10) == [
        "linear_only",
        "projector_linear",
        "random_projector_linear",
    ]
    assert csp._applicable_refine_modes(5, "procrustes", 10) == [
        "linear_only",
        "projector_linear",
    ]
    assert csp._applicable_refine_modes(5, "cos", 10) == ["linear_only"]


def test_only_random_mode_skips_learned_projector_fit():
    assert not csp._should_fit_projector(4)
    for flag in (0, 1, 2, 3, 5):
        assert csp._should_fit_projector(flag)


def test_random_bundle_is_deterministic_frozen_normalized_and_rng_isolated():
    old, new = _anchors()
    cfg = _projector_cfg()

    torch.manual_seed(99)
    expected_next = torch.rand(5)
    torch.manual_seed(99)
    first = csp._build_random_projector_bundle(
        old, new, kind="linear", activation=None, num_layers=1, cfg=cfg, seed=7,
    )
    actual_next = torch.rand(5)
    second = csp._build_random_projector_bundle(
        old, new, kind="linear", activation=None, num_layers=1, cfg=cfg, seed=7,
    )

    assert torch.equal(actual_next, expected_next)
    assert first["projector_trained"] is False
    assert first["random_seed"] == second["random_seed"]
    assert first["norm_stats"] is not None
    assert np.allclose(first["norm_stats"]["old_mean"], old["embeddings"].mean(axis=0))
    assert np.allclose(first["norm_stats"]["new_mean"], new["embeddings"].mean(axis=0))
    assert all(not p.requires_grad for p in first["projector"].parameters())
    for key, value in first["projector"].state_dict().items():
        assert torch.equal(value, second["projector"].state_dict()[key])


def test_random_bundle_key_ignores_training_only_recipe_fields():
    cfg_a = _projector_cfg(lr=1e-4, epochs=10, loss="mse")
    cfg_b = _projector_cfg(lr=1e-2, epochs=999, loss="mae")
    assert csp._random_projector_bundle_key("linear", None, cfg_a) == \
        csp._random_projector_bundle_key("linear", None, cfg_b)


def test_random_bundle_supports_mlp_and_autoencoder_shapes():
    old, new = _anchors(d_old=8, d_new=6)
    for kind in ("mlp", "autoencoder"):
        bundle = csp._build_random_projector_bundle(
            old, new, kind=kind, activation="relu", num_layers=2,
            cfg=_projector_cfg(encoder_ratio=2), seed=5,
        )
        projected = csp._apply_linear_projector(
            bundle["projector"], bundle["norm_stats"], old["embeddings"],
        )
        assert projected.shape == (len(old["embeddings"]), 6)
        assert all(not parameter.requires_grad for parameter in bundle["projector"].parameters())


def test_random_bundle_uses_current_global_seed_when_seed_is_omitted():
    old, new = _anchors()
    original_seed = csp._SEED
    try:
        csp._SEED = 123
        implicit = csp._build_random_projector_bundle(
            old, new, kind="linear", activation=None, cfg=_projector_cfg(),
        )
        explicit = csp._build_random_projector_bundle(
            old, new, kind="linear", activation=None, cfg=_projector_cfg(), seed=123,
        )
    finally:
        csp._SEED = original_seed
    assert implicit["random_seed"] == explicit["random_seed"]


def test_random_projector_stays_identical_while_head_updates(tmp_path):
    old, new = _anchors(n=12)
    bundle = csp._build_random_projector_bundle(
        old, new, kind="linear", activation=None, num_layers=1,
        cfg=_projector_cfg(), seed=11,
    )
    before = {k: v.clone() for k, v in bundle["projector"].state_dict().items()}
    head = torch.nn.Linear(3, 1)
    head_before = {k: v.clone() for k, v in head.state_dict().items()}
    labels = np.linspace(0, 1, 12, dtype=np.float32)

    result = csp._refine_projector_and_linear(
        projector=bundle["projector"], norm_stats=bundle["norm_stats"],
        head_linear=head, emb_B=old["embeddings"], labels_B=labels,
        new_anchor_emb=new["embeddings"], anchor_labels=labels,
        old_anchor_emb=old["embeddings"], label_denorm=1.0,
        refine_dir=str(tmp_path), tag="random-frozen",
        cfg=_refinement_cfg(), frozen_projector=True,
    )

    for key, value in result["projector_after"].state_dict().items():
        assert torch.equal(value, before[key])
    assert all(p.grad is None for p in result["projector_after"].parameters())
    assert any(
        not torch.equal(value, head_before[key])
        for key, value in result["linear_after"].state_dict().items()
    )
    assert result["proj_anchor_loss_before"] == result["proj_anchor_loss_after"]
    assert result["ckpt_paths"]["projector_before"] == \
        result["ckpt_paths"]["projector_after"]


def test_optuna_trial_uses_random_bundle_and_refined_head(tmp_path):
    old, new = _anchors(n=6, d_old=2, d_new=2)
    old.update({
        "labels": np.linspace(0, 1, 6, dtype=np.float32),
        "sample_ids": np.arange(6, dtype=np.int64),
    })
    new.update({"sample_ids": old["sample_ids"]})
    recipe = _projector_cfg()
    refinement_recipe = _refinement_cfg(epochs=1)
    refinement_tag = csp._refinement_tag(refinement_recipe)
    random_bundle = csp._build_random_projector_bundle(
        old, new, kind="linear", activation=None, cfg=recipe, seed=13,
    )
    head_before = torch.nn.Linear(2, 1)
    head_after = copy.deepcopy(head_before)
    with torch.no_grad():
        head_after.weight.add_(0.25)
    refine_bundle = {
        "projector_before": copy.deepcopy(random_bundle["projector"]),
        "projector_after": copy.deepcopy(random_bundle["projector"]),
        "linear_before": head_before,
        "linear_after": head_after,
        "config": refinement_recipe,
        "metrics": [],
    }
    refine_metrics = {
        "refine_enabled": True,
        "refine_mode": "random_projector_linear",
        "projector_random_init": True,
    }
    random_bundle["refinements"] = {
        ("random_projector_linear", refinement_tag): {
            "refine_bundle": refine_bundle,
            "metrics": refine_metrics,
            "new_test_eval": None,
        },
        ("random_projector_linear", refinement_tag, 3): {
            "refine_bundle": refine_bundle,
            "metrics": {**refine_metrics, "projector_before_pth": "chosen.pt"},
            "new_test_eval": None,
        },
    }
    random_bundle["ckpt_path"] = "other-count.pt"
    key = ("train", 6, "random")
    random_key = csp._random_projector_bundle_key("linear", None, recipe)
    anchor_cache = {
        key: {
            "old": old,
            "new": new,
            "projectors": {},
            "random_projectors": {random_key: random_bundle},
            "refine_distance": {},
        },
    }
    old_tensors = {
        **old,
        "predictions": np.zeros(6, dtype=np.float32),
    }
    tensor_cache = {"test": {"old_tensors": old_tensors, "old_tensors_csv": "unused"}}
    head_module = torch.nn.Module()
    head_module.linear = head_before
    model = SimpleNamespace(head=head_module)
    params = {
        "num_anchors": 6,
        "num_refinement_samples": 3,
        "anchor_selection_type": "random",
        "csv_anchor_selection": "train",
        "old_model_csv": "test",
        "interpolation_similarity": "linear",
        "mlp_activation": "gelu",
        "mlp_num_layers": 1,
        "weighting_method": "none",
        "rbf_sigma": 1.0,
        "projector_config": "projector",
        "refinement_config": "refinement",
        "refine_mode": "random_projector_linear",
    }
    enabled_before = csp.REFINEMENT_CONFIG["enabled"]
    csp.REFINEMENT_CONFIG["enabled"] = True
    try:
        csp._run_trial(
            params, 0, anchor_cache, tensor_cache, model,
            {"config": {"normalize_labels": 0}}, str(tmp_path), 1,
            {"projector": recipe}, {"refinement": refinement_recipe},
        )
    finally:
        csp.REFINEMENT_CONFIG["enabled"] = enabled_before

    with (tmp_path / "results.pkl").open("rb") as stream:
        result = pickle.load(stream)
    expected_projection = csp._apply_linear_projector(
        random_bundle["projector"], random_bundle["norm_stats"], old["embeddings"],
    )
    np.testing.assert_allclose(result["new_model_tensors"]["embeddings"], expected_projection)
    assert result["linear_projector"]["projector_trained"] is False
    assert result["linear_projector"]["random_seed"] == random_bundle["random_seed"]
    assert result["linear_projector"]["ckpt_path"] == "chosen.pt"
    assert result["refinement"]["refine_mode"] == "random_projector_linear"

import copy
import os
import sys
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
        "optimizer": "adamw",
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


def test_refinement_flags_select_paper_modes():
    assert csp._applicable_refine_modes(4) == ["random_projector_linear"]
    assert csp._applicable_refine_modes(3) == ["linear_only", "projector_linear"]


def test_random_procrustes_is_frozen_data_free_semi_orthogonal_rotation():
    for d_old, d_new in ((8, 5), (5, 8), (6, 6)):
        old, new = _anchors(d_old=d_old, d_new=d_new)
        cfg = _projector_cfg(normalize_embeddings=False)
        bundle = csp._build_random_projector_bundle(
            old, new, kind="procrustes", activation=None, cfg=cfg, seed=3,
        )
        again = csp._build_random_projector_bundle(
            old, new, kind="procrustes", activation=None, cfg=cfg, seed=3,
        )
        other = csp._build_random_projector_bundle(
            old, new, kind="procrustes", activation=None, cfg=cfg, seed=4,
        )
        proj = bundle["projector"]
        R = proj.weight.detach().numpy().T.astype(np.float64)
        gram = R.T @ R if d_old >= d_new else R @ R.T
        assert np.allclose(gram, np.eye(min(d_old, d_new)), atol=1e-5)
        assert np.allclose(R, bundle["procrustes_params"]["R"])
        assert bundle["procrustes_params"]["scale"] == 1.0
        assert not proj.bias.detach().any()
        assert all(not p.requires_grad for p in proj.parameters())
        assert torch.equal(proj.weight, again["projector"].weight)
        assert not torch.equal(proj.weight, other["projector"].weight)
        # Data-free: the map ignores the anchor contents.
        old2, new2 = _anchors(seed=11, d_old=d_old, d_new=d_new)
        swapped = csp._build_random_projector_bundle(
            old2, new2, kind="procrustes", activation=None, cfg=cfg, seed=3,
        )
        assert torch.equal(proj.weight, swapped["projector"].weight)


def test_random_linear_close_is_frozen_linear_with_its_own_draw():
    old, new = _anchors(d_old=8, d_new=6)
    cfg = _projector_cfg()
    close = csp._build_random_projector_bundle(
        old, new, kind="linear_close", activation=None, cfg=cfg, seed=5,
    )
    linear = csp._build_random_projector_bundle(
        old, new, kind="linear", activation=None, cfg=cfg, seed=5,
    )
    assert isinstance(close["projector"], torch.nn.Linear)
    assert close["projector"].weight.shape == (6, 8)
    assert all(not p.requires_grad for p in close["projector"].parameters())
    assert "procrustes_params" not in close
    assert not torch.equal(close["projector"].weight, linear["projector"].weight)


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


def test_random_bundle_supports_mlp_and_autoencoder_shapes():
    old, new = _anchors(d_old=8, d_new=6)
    for kind in ("mlp", "autoencoder"):
        bundle = csp._build_random_projector_bundle(
            old, new, kind=kind, activation="gelu", num_layers=2,
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


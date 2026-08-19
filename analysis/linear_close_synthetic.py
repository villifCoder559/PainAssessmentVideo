#!/usr/bin/env python3
"""Deterministic synthetic regimes for the linear-projector investigation.

Each function isolates one factor and returns JSON-serializable measurements.  The
suite is intentionally independent of the experiment pipeline and never writes a
checkpoint or changes a production configuration.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.linear_close_investigation import (
    affine_metrics,
    fit_affine_adam,
    fit_affine_closed_form,
    fit_affine_ridge,
)


def _orthonormal_loadings(rng: np.random.Generator, dimension: int, rank: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(dimension, rank)))[0]


def _mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def known_mapping_experiment(seed: int = 89) -> dict[str, float]:
    """Well-determined, noiseless ``Y=XW+b``: OLS and converged Adam agree."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(300, 8)).astype(np.float32)
    weight = rng.normal(size=(8, 6)).astype(np.float32)
    bias = rng.normal(size=6).astype(np.float32)
    y = x @ weight + bias
    ols = fit_affine_closed_form(x[:200], y[:200], rcond=None)
    adam = fit_affine_adam(
        x[:200], y[:200], x[200:], y[200:],
        lr=0.03, epochs=500, batch_size=200, seed=seed,
    )
    return {
        "ols_test_mse": _mse(ols.predict(x[200:]), y[200:]),
        "adam_test_mse": _mse(adam.predict(x[200:]), y[200:]),
        "maximum_prediction_difference": float(
            np.max(np.abs(ols.predict(x[200:]) - adam.predict(x[200:])))
        ),
        "adam_best_epoch": adam.best_epoch,
    }


def sufficient_anchor_experiment(seed: int = 91) -> dict[str, float]:
    """High-dimensional but overdetermined noisy mapping."""
    rng = np.random.default_rng(seed)
    dimension, output_dimension = 384, 16
    true_weight = rng.normal(size=(dimension, output_dimension)) / np.sqrt(dimension)
    bias = rng.normal(size=output_dimension)
    x_train = rng.normal(size=(800, dimension))
    x_test = rng.normal(size=(300, dimension))
    y_train = x_train @ true_weight + bias + rng.normal(
        scale=0.1, size=(len(x_train), output_dimension),
    )
    y_test = x_test @ true_weight + bias
    ols = fit_affine_closed_form(x_train, y_train, rcond=None)
    ridge = fit_affine_ridge(x_train, y_train, alpha=10.0)
    return {
        "anchors": len(x_train),
        "dimension": dimension,
        "ols_test_mse": _mse(ols.predict(x_test), y_test),
        "ridge_test_mse": _mse(ridge.predict(x_test), y_test),
        "ols_rank": ols.rank,
    }


def underdetermined_experiment(seed: int = 97) -> dict[str, float]:
    """Few-anchor, high-dimensional exact mapping with unidentified directions."""
    rng = np.random.default_rng(seed)
    anchors, dimension, output_dimension = 100, 384, 16
    true_weight = rng.normal(size=(dimension, output_dimension)) / np.sqrt(dimension)
    bias = rng.normal(size=output_dimension)
    x_train = rng.normal(size=(anchors, dimension))
    x_test = rng.normal(size=(300, dimension))
    y_train = x_train @ true_weight + bias
    y_test = x_test @ true_weight + bias
    ols = fit_affine_closed_form(x_train, y_train, rcond=None)
    ridge = fit_affine_ridge(x_train, y_train, alpha=10.0)
    return {
        "anchors": anchors,
        "dimension": dimension,
        "ols_anchor_mse": affine_metrics(ols, x_train, y_train)["mse"],
        "ols_test_mse": _mse(ols.predict(x_test), y_test),
        "ridge_test_mse": _mse(ridge.predict(x_test), y_test),
        "ols_rank": ols.rank,
    }


def low_rank_noise_experiment(seed: int = 99) -> dict[str, float]:
    """Correlated low-rank anchors with weak noisy directions."""
    rng = np.random.default_rng(seed)
    anchors, dimension, latent_rank, output_dimension = 100, 384, 20, 16
    loading = _orthonormal_loadings(rng, dimension, latent_rank)
    target = rng.normal(size=(latent_rank, output_dimension)) / np.sqrt(latent_rank)
    z_train = rng.normal(size=(anchors, latent_rank))
    z_test = rng.normal(size=(300, latent_rank))
    x_train = z_train @ loading.T + 1e-3 * rng.normal(size=(anchors, dimension))
    x_test = z_test @ loading.T + 1e-3 * rng.normal(size=(300, dimension))
    y_train = z_train @ target + 0.05 * rng.normal(size=(anchors, output_dimension))
    y_test = z_test @ target
    ols = fit_affine_closed_form(x_train, y_train, rcond=1e-5)
    truncated = fit_affine_closed_form(x_train, y_train, rcond=1e-2)
    ridge = fit_affine_ridge(x_train, y_train, alpha=1.0)
    return {
        "ols_anchor_mse": affine_metrics(ols, x_train, y_train)["mse"],
        "ols_test_mse": _mse(ols.predict(x_test), y_test),
        "ols_weight_norm": float(np.linalg.norm(ols.weight)),
        "ols_rank": ols.rank,
        "truncated_anchor_mse": affine_metrics(truncated, x_train, y_train)["mse"],
        "truncated_test_mse": _mse(truncated.predict(x_test), y_test),
        "truncated_weight_norm": float(np.linalg.norm(truncated.weight)),
        "truncated_rank": truncated.rank,
        "ridge_test_mse": _mse(ridge.predict(x_test), y_test),
        "ridge_weight_norm": float(np.linalg.norm(ridge.weight)),
    }


def cross_domain_nuisance_experiment(seed: int = 101) -> list[dict[str, float]]:
    """Hold latent signal fixed while increasing only deployment nuisance scale."""
    rng = np.random.default_rng(seed)
    anchors, dimension, latent_rank, output_dimension = 60, 80, 10, 12
    loading = _orthonormal_loadings(rng, dimension, latent_rank)
    target = rng.normal(size=(latent_rank, output_dimension)) / np.sqrt(latent_rank)
    head = rng.normal(size=output_dimension) / np.sqrt(output_dimension)
    z_train = rng.normal(size=(anchors, latent_rank))
    x_train = z_train @ loading.T + 1e-3 * rng.normal(size=(anchors, dimension))
    y_train = z_train @ target + 0.05 * rng.normal(size=(anchors, output_dimension))
    ols = fit_affine_closed_form(x_train, y_train, rcond=1e-5)
    ridge = fit_affine_ridge(x_train, y_train, alpha=0.1)
    z_val = rng.normal(size=(120, latent_rank))
    x_val = z_val @ loading.T + 1e-3 * rng.normal(size=(120, dimension))
    y_val = z_val @ target
    adam = fit_affine_adam(
        x_train, y_train, x_val, y_val,
        lr=1e-5, epochs=750, batch_size=64, seed=seed,
    )
    z_test = rng.normal(size=(500, latent_rank))
    nuisance = rng.normal(size=(500, dimension))
    y_test = z_test @ target
    rows = []
    for nuisance_scale in (1e-3, 0.05, 0.1):
        x_test = z_test @ loading.T + nuisance_scale * nuisance
        ols_prediction = ols.predict(x_test)
        ridge_prediction = ridge.predict(x_test)
        adam_prediction = adam.predict(x_test)
        rows.append({
            "nuisance_scale": nuisance_scale,
            "ols_test_mse": _mse(ols_prediction, y_test),
            "ridge_test_mse": _mse(ridge_prediction, y_test),
            "ols_head_mae": float(np.mean(np.abs((ols_prediction - y_test) @ head))),
            "ridge_head_mae": float(np.mean(np.abs((ridge_prediction - y_test) @ head))),
            "adam_test_mse": _mse(adam_prediction, y_test),
            "adam_head_mae": float(np.mean(np.abs((adam_prediction - y_test) @ head))),
            "ols_weight_norm": float(np.linalg.norm(ols.weight)),
            "ridge_weight_norm": float(np.linalg.norm(ridge.weight)),
            "adam_weight_norm": float(np.linalg.norm(adam.fit.weight)),
            "adam_best_epoch": adam.best_epoch,
        })
    return rows


def anchor_count_sweep(seed: int = 103) -> list[dict[str, float]]:
    """Expose the OLS interpolation peak while changing only anchor count."""
    rng = np.random.default_rng(seed)
    dimension, latent_rank, output_dimension = 80, 12, 10
    loading = _orthonormal_loadings(rng, dimension, latent_rank)
    target = rng.normal(size=(latent_rank, output_dimension)) / np.sqrt(latent_rank)
    z_pool = rng.normal(size=(320, latent_rank))
    x_pool = z_pool @ loading.T + 1e-2 * rng.normal(size=(320, dimension))
    y_pool = z_pool @ target + 0.1 * rng.normal(size=(320, output_dimension))
    z_test = rng.normal(size=(500, latent_rank))
    x_test = z_test @ loading.T + 1e-2 * rng.normal(size=(500, dimension))
    y_test = z_test @ target
    rows = []
    for anchors in (20, 40, 80, 160, 320):
        ols = fit_affine_closed_form(x_pool[:anchors], y_pool[:anchors], rcond=1e-5)
        ridge = fit_affine_ridge(x_pool[:anchors], y_pool[:anchors], alpha=1.0)
        rows.append({
            "anchors": anchors,
            "ols_test_mse": _mse(ols.predict(x_test), y_test),
            "ridge_test_mse": _mse(ridge.predict(x_test), y_test),
            "ols_weight_norm": float(np.linalg.norm(ols.weight)),
            "ridge_weight_norm": float(np.linalg.norm(ridge.weight)),
            "ols_rank": ols.rank,
        })
    return rows


def scaling_experiment(seed: int = 107) -> dict[str, float]:
    """Demonstrate that a relative SVD cutoff depends on coordinate scale."""
    rng = np.random.default_rng(seed)
    dimension, output_dimension = 12, 5
    scale = np.geomspace(1.0, 1e-8, dimension)
    weight = rng.normal(size=(dimension, output_dimension))
    bias = rng.normal(size=output_dimension)
    latent_train = rng.normal(size=(240, dimension))
    latent_test = rng.normal(size=(300, dimension))
    x_train = latent_train * scale
    x_test = latent_test * scale
    y_train = latent_train @ weight + bias
    y_test = latent_test @ weight + bias
    raw = fit_affine_closed_form(x_train, y_train, rcond=1e-5)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    standardized = fit_affine_closed_form(
        (x_train - mean) / std, y_train, rcond=1e-5,
    )
    return {
        "dimension": dimension,
        "raw_rank": raw.rank,
        "standardized_rank": standardized.rank,
        "raw_test_mse": _mse(raw.predict(x_test), y_test),
        "standardized_test_mse": _mse(
            standardized.predict((x_test - mean) / std), y_test,
        ),
    }


def run_suite() -> dict:
    """Run every documented regime and return a serializable result tree."""
    return {
        "known_mapping": known_mapping_experiment(),
        "sufficient_anchors": sufficient_anchor_experiment(),
        "underdetermined": underdetermined_experiment(),
        "low_rank_noise": low_rank_noise_experiment(),
        "cross_domain_nuisance": cross_domain_nuisance_experiment(),
        "anchor_count": anchor_count_sweep(),
        "scaling": scaling_experiment(),
    }


if __name__ == "__main__":
    print(json.dumps(run_suite(), indent=2, sort_keys=True))

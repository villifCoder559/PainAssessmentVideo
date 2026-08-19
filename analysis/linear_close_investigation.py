#!/usr/bin/env python3
"""Controlled diagnostics for the learned and closed-form affine projectors.

This module intentionally does not import :mod:`cross_space_projection`: importing
the training pipeline initializes process-level logging state.  The formulas here
are small, independently testable reproductions used to pressure-test the method;
the investigation document cross-checks them against the production implementation
and saved checkpoints.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class AffineFit:
    """Affine prediction ``x @ weight + bias`` plus solver diagnostics."""

    weight: np.ndarray
    bias: np.ndarray
    rank: int
    singular_values: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.weight + self.bias


@dataclass(frozen=True)
class AdamFitResult:
    """Validation-selected affine fit produced by the learned projector recipe."""

    fit: AffineFit
    best_epoch: int
    train_mse: float
    val_mse: float
    initial_weight_norm: float
    initial_bias_norm: float
    delta_weight_norm: float
    delta_bias_norm: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.fit.predict(x)


def fit_affine_closed_form(
    x: np.ndarray,
    y: np.ndarray,
    *,
    intercept: bool = True,
    rcond: float | None = None,
) -> AffineFit:
    """Fit multivariate OLS using the same centered SVD solve as ``linear_close``."""
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.ndim != 2 or y64.ndim != 2 or x64.shape[0] != y64.shape[0]:
        raise ValueError("x and y must be 2-D arrays with the same row count")

    if intercept:
        x_mean = x64.mean(axis=0)
        y_mean = y64.mean(axis=0)
        x_fit = x64 - x_mean
        y_fit = y64 - y_mean
    else:
        x_mean = np.zeros(x64.shape[1], dtype=np.float64)
        y_mean = np.zeros(y64.shape[1], dtype=np.float64)
        x_fit = x64
        y_fit = y64

    weight, _residuals, rank, singular_values = np.linalg.lstsq(
        x_fit, y_fit, rcond=rcond,
    )
    bias = y_mean - x_mean @ weight
    return AffineFit(weight, bias, int(rank), singular_values)


def fit_affine_ridge(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
    intercept: bool = True,
) -> AffineFit:
    """Fit centered ridge regression without penalizing the intercept."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.ndim != 2 or y64.ndim != 2 or x64.shape[0] != y64.shape[0]:
        raise ValueError("x and y must be 2-D arrays with the same row count")

    if intercept:
        x_mean = x64.mean(axis=0)
        y_mean = y64.mean(axis=0)
        x_fit = x64 - x_mean
        y_fit = y64 - y_mean
    else:
        x_mean = np.zeros(x64.shape[1], dtype=np.float64)
        y_mean = np.zeros(y64.shape[1], dtype=np.float64)
        x_fit = x64
        y_fit = y64

    left, singular_values, right_t = np.linalg.svd(x_fit, full_matrices=False)
    if alpha == 0:
        cutoff = (
            np.finfo(x_fit.dtype).eps * max(x_fit.shape) * singular_values[0]
            if singular_values.size else 0.0
        )
        factors = np.divide(
            1.0,
            singular_values,
            out=np.zeros_like(singular_values),
            where=singular_values > cutoff,
        )
    else:
        factors = singular_values / (singular_values**2 + alpha)
    weight = (right_t.T * factors) @ (left.T @ y_fit)
    bias = y_mean - x_mean @ weight
    rank = np.linalg.matrix_rank(x_fit)
    return AffineFit(weight, bias, int(rank), singular_values)


def fit_affine_adam(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lr: float = 1e-5,
    epochs: int = 750,
    batch_size: int = 64,
    weight_decay: float = 0.0,
    seed: int = 42,
) -> AdamFitResult:
    """Reproduce the validation-selected ``nn.Linear``/AdamW estimator on arrays."""
    import torch
    import torch.nn.functional as functional

    train_x = torch.as_tensor(np.asarray(x_train, np.float32))
    train_y = torch.as_tensor(np.asarray(y_train, np.float32))
    val_x = torch.as_tensor(np.asarray(x_val, np.float32))
    val_y = torch.as_tensor(np.asarray(y_val, np.float32))
    if train_x.ndim != 2 or train_y.ndim != 2 or train_x.shape[0] != train_y.shape[0]:
        raise ValueError("training arrays must be 2-D with the same row count")
    if val_x.ndim != 2 or val_y.ndim != 2 or val_x.shape[0] != val_y.shape[0]:
        raise ValueError("validation arrays must be 2-D with the same row count")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    torch.manual_seed(seed)
    projector = torch.nn.Linear(train_x.shape[1], train_y.shape[1])
    initial_weight = projector.weight.detach().clone()
    initial_bias = projector.bias.detach().clone()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_epoch = -1
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        projector.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            prediction = projector(train_x[indices])
            loss = functional.mse_loss(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        projector.eval()
        with torch.no_grad():
            val_mse = float(functional.mse_loss(projector(val_x), val_y))
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in projector.state_dict().items()}

    projector.load_state_dict(best_state)
    projector.eval()
    with torch.no_grad():
        train_mse = float(functional.mse_loss(projector(train_x), train_y))
    final_weight = projector.weight.detach()
    final_bias = projector.bias.detach()
    weight = final_weight.T.numpy().astype(np.float64)
    bias = final_bias.numpy().astype(np.float64)
    singular_values = np.linalg.svd(weight, compute_uv=False)
    fit = AffineFit(weight, bias, int(np.linalg.matrix_rank(weight)), singular_values)
    return AdamFitResult(
        fit=fit,
        best_epoch=best_epoch,
        train_mse=train_mse,
        val_mse=best_val,
        initial_weight_norm=float(torch.linalg.vector_norm(initial_weight)),
        initial_bias_norm=float(torch.linalg.vector_norm(initial_bias)),
        delta_weight_norm=float(torch.linalg.vector_norm(final_weight - initial_weight)),
        delta_bias_norm=float(torch.linalg.vector_norm(final_bias - initial_bias)),
    )


def affine_metrics(fit: AffineFit, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Return prediction error and parameter-size diagnostics for an affine fit."""
    residual = fit.predict(x) - np.asarray(y, dtype=np.float64)
    return {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "weight_norm": float(np.linalg.norm(fit.weight)),
        "bias_norm": float(np.linalg.norm(fit.bias)),
    }


def checkpoint_diagnostics(path: str | Path, *, rank_rtol: float = 1e-5) -> dict:
    """Read a projector state dict safely and report scale/rank diagnostics."""
    import torch

    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not {"weight", "bias"} <= set(state):
        raise ValueError(f"checkpoint lacks weight/bias tensors: {path}")
    weight = state["weight"].detach().to(dtype=torch.float64)
    bias = state["bias"].detach().to(dtype=torch.float64)
    singular_values = torch.linalg.svdvals(weight)
    largest = float(singular_values[0]) if singular_values.numel() else 0.0
    threshold = rank_rtol * largest
    rank = int((singular_values > threshold).sum()) if largest else 0
    smallest_retained = float(singular_values[rank - 1]) if rank else 0.0
    return {
        "weight_shape": list(weight.shape),
        "weight_norm": float(torch.linalg.vector_norm(weight)),
        "bias_norm": float(torch.linalg.vector_norm(bias)),
        "weight_rank": rank,
        "largest_singular_value": largest,
        "smallest_retained_singular_value": smallest_retained,
        "retained_condition_number": (
            largest / smallest_retained if smallest_retained > 0 else float("inf")
        ),
    }


_SUBTRIAL_RE = re.compile(r"^cross_space_projection_subtrial_(\d+)_(\d+)_")


def _discover_subtrials(root: str | Path) -> dict[tuple[int, int], Path]:
    found: dict[tuple[int, int], Path] = {}
    for path in sorted(Path(root).glob("cross_space_projection_subtrial_*")):
        match = _SUBTRIAL_RE.match(path.name)
        if not match or not path.is_dir():
            continue
        key = (int(match.group(1)), int(match.group(2)))
        if key in found:
            raise ValueError(f"duplicate subtrial {key} under {root}")
        found[key] = path
    return found


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _single_path(paths, label: str) -> Path:
    matches = list(paths)
    if len(matches) != 1:
        raise ValueError(f"expected one {label}, found {len(matches)}")
    return matches[0]


def _summary_row(run_dir: Path) -> dict[str, str]:
    summary_path = run_dir / "logs" / "summary.csv"
    if not summary_path.exists():
        summary_path = run_dir / "refinement_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty summary.csv under {run_dir}")
    return next((row for row in rows if row.get("refine_mode") == "projector_linear"), rows[0])


def _mean_csv_columns(path: Path, columns: tuple[str, ...]) -> dict[str, float]:
    if not path.exists():
        return {column: float("nan") for column in columns}
    values = {column: [] for column in columns}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for column in columns:
                values[column].append(float(row[column]))
    return {column: float(np.mean(column_values)) for column, column_values in values.items()}


def _float_or_nan(value: str | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def _compare_subtrial_pair(close_dir: Path, linear_dir: Path) -> dict:
    close_projector = _single_path(close_dir.glob("*_projector"), "close projector directory")
    linear_projector = _single_path(linear_dir.glob("*_projector"), "linear projector directory")
    close_ckpt = _single_path(close_projector.glob("best_projector_*.pt"), "close checkpoint")
    linear_ckpt = _single_path(linear_projector.glob("best_projector_*.pt"), "linear checkpoint")
    close_stats = checkpoint_diagnostics(close_ckpt)
    linear_stats = checkpoint_diagnostics(linear_ckpt)
    close_summary = _summary_row(close_dir)
    linear_summary = _summary_row(linear_dir)
    close_reconstruction = _mean_csv_columns(
        close_dir / "logs" / "embedding_reconstruction_test_projected.csv", ("l2", "cos_sim")
    )
    linear_reconstruction = _mean_csv_columns(
        linear_dir / "logs" / "embedding_reconstruction_test_projected.csv", ("l2", "cos_sim")
    )
    close_epoch = float(close_ckpt.stem.rsplit("_", 1)[-1])
    linear_epoch = float(linear_ckpt.stem.rsplit("_", 1)[-1])

    file_equal = {
        "anchors_equal": _sha256(close_dir / "anchors.csv") == _sha256(linear_dir / "anchors.csv"),
        "old_tensors_equal": (
            _sha256(close_dir / "old_tensors.csv") == _sha256(linear_dir / "old_tensors.csv")
        ),
    }
    split_equal = {}
    for split in ("train", "val", "test"):
        split_equal[split] = (
            _sha256(close_projector / "split_training_stage" / f"{split}.csv")
            == _sha256(linear_projector / "split_training_stage" / f"{split}.csv")
        )

    return {
        **file_equal,
        "all_splits_equal": all(split_equal.values()),
        **{f"{split}_split_equal": value for split, value in split_equal.items()},
        "close_mae": _float_or_nan(
            close_summary.get("mae") or close_summary.get("mae_micro_old_oncsv_before")
        ),
        "linear_mae": _float_or_nan(
            linear_summary.get("mae") or linear_summary.get("mae_micro_old_oncsv_before")
        ),
        "close_anchor_mse": _float_or_nan(close_summary.get("proj_anchor_loss_before")),
        "linear_anchor_mse": _float_or_nan(linear_summary.get("proj_anchor_loss_before")),
        "close_best_epoch": _float_or_nan(close_summary.get("lp_best_epoch")) if close_summary.get("lp_best_epoch") else close_epoch,
        "linear_best_epoch": _float_or_nan(linear_summary.get("lp_best_epoch")) if linear_summary.get("lp_best_epoch") else linear_epoch,
        "close_reconstruction_l2_mean": close_reconstruction["l2"],
        "linear_reconstruction_l2_mean": linear_reconstruction["l2"],
        "close_reconstruction_cos_mean": close_reconstruction["cos_sim"],
        "linear_reconstruction_cos_mean": linear_reconstruction["cos_sim"],
        **{f"close_{key}": value for key, value in close_stats.items()},
        **{f"linear_{key}": value for key, value in linear_stats.items()},
    }


def compare_experiment_roots(close_root: str | Path, linear_root: str | Path) -> list[dict]:
    """Pair saved subtrials by fold indices and safely compare their artifacts."""
    close_trials = _discover_subtrials(close_root)
    linear_trials = _discover_subtrials(linear_root)
    if set(close_trials) != set(linear_trials):
        raise ValueError(
            "subtrial keys differ: "
            f"close_only={sorted(set(close_trials) - set(linear_trials))}, "
            f"linear_only={sorted(set(linear_trials) - set(close_trials))}"
        )
    rows = []
    for new_idx, old_idx in sorted(close_trials):
        row = _compare_subtrial_pair(
            close_trials[(new_idx, old_idx)], linear_trials[(new_idx, old_idx)]
        )
        rows.append({"new_idx": new_idx, "old_idx": old_idx, **row})
    return rows

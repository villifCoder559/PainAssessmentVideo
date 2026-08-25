#!/usr/bin/env python3
"""Analyze MAE across cross-projection anchor-count sweeps."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


DIRECTION_ALIASES = {
    "bioVmae_to_mintDfer": ("BioVid", "MIntPain"),
    "cross-validation_BioVmae-unbcDFER_v2": ("BioVid", "UNBC"),
    "mintVMAE-bioDFER": ("MIntPain", "BioVid"),
    "mintVMAE-unbcDFER": ("MIntPain", "UNBC"),
    "unbcVMAE-mintDFER": ("UNBC", "MIntPain"),
    "unbcVmae_to_biovidDfer": ("UNBC", "BioVid"),
}

METHODS = ("linear", "linear_close", "mlp", "autoencoder", "procrustes")
REFINEMENT_MODES = ("linear_only", "projector_linear")
PLOT_REFINEMENT_STAGES = ("projector_only", *REFINEMENT_MODES)
CACHE_SCHEMA_VERSION = 1
DEFAULT_INPUT_ROOT = Path("Cross_projection/anchor_sweep")
DEFAULT_OUTPUT_DIR = Path("analysis/results/anchor_sweep_mae")
TQDM_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
    "[elapsed {elapsed}, ETA {remaining}, {rate_fmt}]"
)

METHOD_LABELS = {
    "linear": "Linear",
    "linear_close": "Linear close",
    "mlp": "MLP",
    "autoencoder": "Autoencoder",
    "procrustes": "Procrustes",
}
METHOD_COLORS = {
    "linear": "#0072B2",
    "linear_close": "#56B4E9",
    "mlp": "#D55E00",
    "autoencoder": "#009E73",
    "procrustes": "#CC79A7",
}
PLOT_STAGE_LABELS = {
    "projector_only": "Projector only",
    "linear_only": "Linear only",
    "projector_linear": "Projector + linear",
}


@dataclass(frozen=True)
class SweepGroup:
    direction_alias: str
    source_dataset: str
    target_dataset: str
    method: str
    configured_anchors: int
    k_dir: Path
    aggregate_candidates: tuple[Path, ...]
    aggregate_path: Path | None
    aggregate_uid: int | None

    @property
    def direction_slug(self) -> str:
        return f"{self.source_dataset.lower()}-to-{self.target_dataset.lower()}"


@dataclass(frozen=True)
class ValidationResult:
    group: SweepGroup
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    valid_pairs: tuple[tuple[int, int], ...]
    valid_entries: tuple[tuple[int, int, Path], ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SubtrialMetric:
    direction_alias: str
    direction: str
    source_dataset: str
    target_dataset: str
    method: str
    configured_anchors: int
    actual_anchors: int
    anchor_count_mismatch: bool
    new_idx: int
    old_idx: int
    refinement_mode: str
    source_old_micro: float
    source_before_micro: float
    source_after_micro: float
    target_before_micro: float
    target_after_micro: float
    source_improvement_micro: float
    target_preservation_change_micro: float
    source_old_macro: float
    source_before_macro: float
    source_after_macro: float
    target_before_macro: float
    target_after_macro: float
    source_improvement_macro: float
    target_preservation_change_macro: float
    aggregate_path: str
    subtrial_path: str


@dataclass(frozen=True)
class PlotOptions:
    output_dir: Path
    metrics: tuple[str, ...] = ("micro", "macro")
    anchor_scales: tuple[str, ...] = ("log", "categorical")
    plots: tuple[str, ...] = (
        "stages",
        "methods",
        "improvement",
        "heatmap",
        "distribution",
    )
    formats: tuple[str, ...] = ("png", "pdf")
    dpi: int = 300
    show_std: bool = False
    exclude_linear_close: bool = False
    show_progress: bool | None = None


def _progress_enabled() -> bool:
    """Show progress by default only in an interactive terminal."""
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def _progress(
    iterable: Any = None,
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    enabled: bool,
) -> Any:
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=not enabled,
        dynamic_ncols=True,
        bar_format=TQDM_BAR_FORMAT,
        file=sys.stderr,
    )


def _folder_method(path: Path) -> str:
    name = path.name.lower()
    for method in sorted(METHODS, key=len, reverse=True):
        if method in name:
            return method
    return name


def _aggregate_directory_uid(path: Path) -> int:
    match = re.fullmatch(r"aggregated_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Invalid aggregate directory name: {path}")
    return int(match.group(1))


def _aggregate_uid(path: Path) -> int:
    return _aggregate_directory_uid(path.parent)


def _compatible_aggregate_result(path: Path) -> bool:
    match = re.fullmatch(r"results_(\d+)\.pkl", path.name)
    if match is None:
        return False
    directory_uid = _aggregate_uid(path)
    return int(match.group(1)) in {directory_uid, directory_uid + 1}


class _IgnoredPickleObject:
    """State container for irrelevant legacy classes embedded in trusted PKLs."""

    def __new__(cls, *_args: object, **_kwargs: object) -> "_IgnoredPickleObject":
        return object.__new__(cls)

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["state"] = state


_IGNORED_PICKLE_CLASSES: dict[tuple[str, str], type[_IgnoredPickleObject]] = {}


class _TrustedResultsUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        # Historical configs contain model/helper instances. Their state is not
        # used here, and importing those modules starts training-side services.
        if module.startswith("custom."):
            return _IGNORED_PICKLE_CLASSES.setdefault(
                (module, name), type(name, (_IgnoredPickleObject,), {})
            )
        return super().find_class(module, name)


def _load_pickle(path: Path) -> object:
    with path.open("rb") as stream:
        return _TrustedResultsUnpickler(stream).load()


def _configured_anchor_value(value: object) -> int | None:
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return None


def _manifest_path(aggregate_path: Path, value: object) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = aggregate_path.parent / path
    return path.resolve()


def _finite_vector(value: object, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _mae_pair(labels: object, predictions: object, label: str) -> tuple[float, float]:
    y_true = _finite_vector(labels, f"{label} labels")
    y_pred = _finite_vector(predictions, f"{label} predictions")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{label} prediction/label lengths differ ({len(y_pred)} != {len(y_true)})"
        )
    if len(y_true) == 0:
        raise ValueError(f"{label} arrays are empty")
    errors = np.abs(y_true - y_pred)
    rounded = np.rint(y_true).astype(int)
    per_class = [float(errors[rounded == value].mean()) for value in np.unique(rounded)]
    return float(errors.mean()), float(np.mean(per_class))


def _validate_subtrial(path: Path, group: SweepGroup) -> list[str]:
    errors: list[str] = []
    try:
        payload = _load_pickle(path)
    except Exception as error:  # trusted input can still be corrupt/incompatible
        return [f"Could not read referenced PKL {path}: {error}"]
    if not isinstance(payload, dict):
        return [f"Subtrial PKL is not a mapping: {path}"]
    config = payload.get("config_cross_space_projection", {})
    if not isinstance(config, dict):
        errors.append(f"Subtrial configuration is not a mapping: {path}")
    else:
        method = config.get("interpolation_similarity")
        anchors = _configured_anchor_value(config.get("num_anchors"))
        if method not in METHODS:
            errors.append(
                f"Unsupported interpolation_similarity {method!r}; expected one of {', '.join(METHODS)}"
            )
        if method != group.method:
            errors.append(
                f"Parent method {group.method!r} is inconsistent with subtrial method {method!r}: {path}"
            )
        if anchors != group.configured_anchors:
            errors.append(
                f"Parent K{group.configured_anchors} is inconsistent with subtrial K{anchors}: {path}"
            )
    old = payload.get("old_model_tensors", {})
    if not isinstance(old, dict):
        errors.append(f"old_model_tensors is missing or invalid: {path}")
    else:
        try:
            _mae_pair(old.get("labels"), old.get("predictions"), "source-test old baseline")
        except ValueError as error:
            errors.append(f"{path}: {error}")

    refinements = payload.get("refinements", {})
    if not isinstance(refinements, dict):
        refinements = {}
    projector_only_before: dict[str, dict[str, float]] = {}
    for mode in ("linear_only", "projector_linear"):
        block = refinements.get(mode)
        if not isinstance(block, dict):
            errors.append(f"Missing {mode} refinement block: {path}")
            continue
        mode_before: dict[str, float] = {}
        for key in (
            "mae_micro_old_oncsv_before",
            "mae_macro_old_oncsv_before",
            "mae_micro_old_oncsv_after",
            "mae_macro_old_oncsv_after",
        ):
            try:
                value = _finite_nonnegative(block.get(key), f"{mode}.{key}")
                if key.endswith("_before"):
                    metric = "micro" if "_micro_" in key else "macro"
                    mode_before[f"source_before_{metric}"] = value
            except ValueError as error:
                errors.append(f"{path}: {error}")
        evaluation = block.get("new_test_eval", {})
        if not isinstance(evaluation, dict):
            errors.append(f"Missing {mode}.new_test_eval: {path}")
            continue
        for stage in ("before", "after"):
            try:
                micro, macro = _mae_pair(
                    evaluation.get("labels"),
                    evaluation.get(f"preds_{stage}"),
                    f"{mode} target-test {stage}",
                )
                if stage == "before":
                    mode_before["target_before_micro"] = micro
                    mode_before["target_before_macro"] = macro
            except ValueError as error:
                errors.append(f"{path}: {error}")
        projector_only_before[mode] = mode_before
    if all(mode in projector_only_before for mode in REFINEMENT_MODES):
        for column in (
            "source_before_micro",
            "source_before_macro",
            "target_before_micro",
            "target_before_macro",
        ):
            values = [projector_only_before[mode].get(column) for mode in REFINEMENT_MODES]
            if all(value is not None for value in values) and not np.isclose(
                float(values[0]), float(values[1]), rtol=1e-9, atol=1e-12
            ):
                errors.append(
                    f"{path}: Inconsistent projector-only {column} values between "
                    f"linear_only and projector_linear: {values}"
                )
    try:
        anchors_path = _anchors_path(payload, path)
        pd.read_csv(anchors_path)
    except Exception as error:
        errors.append(f"{path}: {error}")
    return errors


def discover_groups(
    input_root: str | Path,
    aggregate_policy: Literal["error", "latest"] = "error",
) -> list[SweepGroup]:
    """Discover direction/method/K groups and select aggregate candidates."""
    if aggregate_policy not in {"error", "latest"}:
        raise ValueError(f"Unknown aggregate policy: {aggregate_policy}")
    root = Path(input_root)
    if not root.is_dir():
        raise ValueError(f"Input root does not exist or is not a directory: {root}")

    groups: list[SweepGroup] = []
    direction_order = {alias: index for index, alias in enumerate(DIRECTION_ALIASES)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    for direction_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        alias = direction_dir.name
        if alias not in DIRECTION_ALIASES:
            known = ", ".join(DIRECTION_ALIASES)
            raise ValueError(f"Unknown direction alias {alias!r}; expected one of: {known}")
        source, target = DIRECTION_ALIASES[alias]
        for method_dir in sorted(path for path in direction_dir.iterdir() if path.is_dir()):
            for k_dir in sorted(path for path in method_dir.iterdir() if path.is_dir()):
                match = re.fullmatch(r"K(\d+)", k_dir.name)
                if match is None:
                    continue
                anchors = int(match.group(1))
                aggregate_dirs = tuple(
                    sorted(
                        (
                            path
                            for path in k_dir.iterdir()
                            if path.is_dir()
                            and re.fullmatch(r"aggregated_\d+", path.name)
                        ),
                        key=_aggregate_directory_uid,
                    )
                )
                if aggregate_policy == "latest" and aggregate_dirs:
                    active_dirs = (max(aggregate_dirs, key=_aggregate_directory_uid),)
                else:
                    active_dirs = aggregate_dirs
                raw_candidates = tuple(
                    sorted(
                        (
                            path.resolve()
                            for aggregate_dir in active_dirs
                            for path in aggregate_dir.glob("results_*.pkl")
                        ),
                        key=lambda path: (_aggregate_uid(path), path.name),
                    )
                )
                candidates = (
                    ()
                    if len(raw_candidates) == 1
                    and not _compatible_aggregate_result(raw_candidates[0])
                    else raw_candidates
                )
                selected: Path | None
                if len(candidates) == 1:
                    selected = candidates[0]
                else:
                    selected = None

                method = _folder_method(method_dir)
                if selected is not None:
                    try:
                        aggregate = _load_pickle(selected)
                    except Exception:
                        # Discovery must retain corrupt/incompatible candidates so
                        # validation can report them instead of aborting the scan.
                        aggregate = None
                    if isinstance(aggregate, dict):
                        config = aggregate.get("config_cross_space_projection", {})
                        if isinstance(config, dict):
                            method = str(config.get("interpolation_similarity", method))
                groups.append(
                    SweepGroup(
                        direction_alias=alias,
                        source_dataset=source,
                        target_dataset=target,
                        method=method,
                        configured_anchors=anchors,
                        k_dir=k_dir.resolve(),
                        aggregate_candidates=candidates,
                        aggregate_path=selected,
                        aggregate_uid=_aggregate_uid(selected) if selected else None,
                    )
                )

    return sorted(
        groups,
        key=lambda group: (
            direction_order[group.direction_alias],
            method_order.get(group.method, len(method_order)),
            group.method,
            group.configured_anchors,
        ),
    )


def validate_group(group: SweepGroup, allow_incomplete: bool = False) -> ValidationResult:
    """Validate one aggregate manifest and all subtrials it references."""
    errors: list[str] = []
    warnings: list[str] = []
    if not group.aggregate_candidates:
        errors.append(f"Missing aggregate candidate under {group.k_dir}")
    elif len(group.aggregate_candidates) > 1 and group.aggregate_path is None:
        errors.append(
            f"Duplicate aggregate candidates under {group.k_dir}: "
            + ", ".join(str(path) for path in group.aggregate_candidates)
        )
    if group.aggregate_path is None:
        return ValidationResult(group, tuple(errors), tuple(warnings), (), ())

    try:
        aggregate = _load_pickle(group.aggregate_path)
    except Exception as error:
        errors.append(f"Could not read aggregate PKL {group.aggregate_path}: {error}")
        return ValidationResult(group, tuple(errors), tuple(warnings), (), ())
    if not isinstance(aggregate, dict):
        errors.append(f"Aggregate PKL is not a mapping: {group.aggregate_path}")
        return ValidationResult(group, tuple(errors), tuple(warnings), (), ())

    config = aggregate.get("config_cross_space_projection", {})
    if not isinstance(config, dict):
        errors.append(f"Aggregate configuration is not a mapping: {group.aggregate_path}")
    else:
        method = config.get("interpolation_similarity")
        anchors = _configured_anchor_value(config.get("num_anchors"))
        if method != group.method:
            errors.append(
                f"Parent method {group.method!r} is inconsistent with aggregate method {method!r}"
            )
        if anchors != group.configured_anchors:
            errors.append(
                f"Parent K{group.configured_anchors} is inconsistent with aggregate K{anchors}"
            )

    paths = aggregate.get("subtrial_pkls")
    rows = aggregate.get("subtrials")
    count = aggregate.get("n_subtrials")
    if not isinstance(paths, (list, tuple)) or not isinstance(rows, (list, tuple)):
        errors.append("Aggregate manifest requires list-valued subtrial_pkls and subtrials")
        return ValidationResult(group, tuple(errors), tuple(warnings), (), ())
    if count != len(paths) or count != len(rows):
        errors.append(
            f"Aggregate manifest lengths disagree: n_subtrials={count}, "
            f"subtrial_pkls={len(paths)}, subtrials={len(rows)}"
        )

    seen: set[tuple[int, int]] = set()
    valid_entries: list[tuple[int, int, Path]] = []
    for index, (path_value, row) in enumerate(zip(paths, rows)):
        if not isinstance(row, dict):
            errors.append(f"Subtrial manifest row {index} is not a mapping")
            continue
        pair = (row.get("new_idx"), row.get("old_idx"))
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in pair):
            errors.append(f"Subtrial manifest row {index} has invalid fold pair {pair}")
            continue
        typed_pair = (int(pair[0]), int(pair[1]))
        if typed_pair in seen:
            errors.append(f"Duplicate fold pair {typed_pair} in {group.aggregate_path}")
            continue
        seen.add(typed_pair)
        if typed_pair not in {(new, old) for new in range(5) for old in range(5)}:
            errors.append(f"Fold pair {typed_pair} is outside the expected 0..4 grid")
            continue
        path = _manifest_path(group.aggregate_path, path_value)
        if not path.is_file():
            message = f"Missing referenced PKL for fold pair {typed_pair}: {path}"
            if allow_incomplete:
                warnings.append(message)
            else:
                errors.append(message)
            continue
        subtrial_errors = _validate_subtrial(path, group)
        if subtrial_errors:
            if allow_incomplete:
                warnings.extend(
                    f"Skipping invalid fold pair {typed_pair}: {message}"
                    for message in subtrial_errors
                )
            else:
                errors.extend(subtrial_errors)
            continue
        valid_entries.append((*typed_pair, path))

    expected = {(new, old) for new in range(5) for old in range(5)}
    unique_valid = {(new, old) for new, old, _path in valid_entries}
    if unique_valid != expected:
        message = (
            f"Expected 25 unique fold pairs, found {len(unique_valid)}; "
            f"missing={sorted(expected - unique_valid)}"
        )
        if allow_incomplete and valid_entries:
            warnings.append(f"Retaining incomplete group: {message}")
        else:
            errors.append(message)
    if count != 25:
        message = f"n_subtrials must be 25 in strict mode, found {count}"
        if allow_incomplete and valid_entries:
            warnings.append(f"Retaining incomplete group: {message}")
        else:
            errors.append(message)

    return ValidationResult(
        group=group,
        errors=tuple(dict.fromkeys(errors)),
        warnings=tuple(dict.fromkeys(warnings)),
        valid_pairs=tuple((new, old) for new, old, _path in valid_entries),
        valid_entries=tuple(valid_entries),
    )


def _anchors_path(payload: dict[str, Any], subtrial_path: Path) -> Path:
    config = payload.get("config_cross_space_projection", {})
    value = payload.get("anchors_csv_path")
    if value is None and isinstance(config, dict):
        value = config.get("anchors_csv_path")
    if value is not None:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = subtrial_path.parent / candidate
        if candidate.is_file():
            return candidate.resolve()
    fallback = subtrial_path.parent / "anchors.csv"
    if fallback.is_file():
        return fallback.resolve()
    raise ValueError(f"Missing anchors.csv for subtrial {subtrial_path}")


def extract_group(group: SweepGroup, allow_incomplete: bool = False) -> list[SubtrialMetric]:
    """Extract wide MAE rows for every valid fold pair and refinement mode."""
    validation = validate_group(group, allow_incomplete=allow_incomplete)
    if not validation.valid:
        raise ValueError("; ".join(validation.errors))
    if group.aggregate_path is None:  # narrowed by validation, retained for typing
        raise ValueError(f"No selected aggregate for {group.k_dir}")

    metrics: list[SubtrialMetric] = []
    for new_idx, old_idx, subtrial_path in validation.valid_entries:
        payload = _load_pickle(subtrial_path)
        assert isinstance(payload, dict)
        old = payload["old_model_tensors"]
        source_old_micro, source_old_macro = _mae_pair(
            old["labels"], old["predictions"], "source-test old baseline"
        )
        anchors_path = _anchors_path(payload, subtrial_path)
        actual_anchors = len(pd.read_csv(anchors_path))
        refinements = payload["refinements"]
        for mode in ("linear_only", "projector_linear"):
            block = refinements[mode]
            evaluation = block["new_test_eval"]
            target_before_micro, target_before_macro = _mae_pair(
                evaluation["labels"], evaluation["preds_before"], f"{mode} target-test before"
            )
            target_after_micro, target_after_macro = _mae_pair(
                evaluation["labels"], evaluation["preds_after"], f"{mode} target-test after"
            )
            source_before_micro = _finite_nonnegative(
                block["mae_micro_old_oncsv_before"], f"{mode}.mae_micro_old_oncsv_before"
            )
            source_before_macro = _finite_nonnegative(
                block["mae_macro_old_oncsv_before"], f"{mode}.mae_macro_old_oncsv_before"
            )
            source_after_micro = _finite_nonnegative(
                block["mae_micro_old_oncsv_after"], f"{mode}.mae_micro_old_oncsv_after"
            )
            source_after_macro = _finite_nonnegative(
                block["mae_macro_old_oncsv_after"], f"{mode}.mae_macro_old_oncsv_after"
            )
            metrics.append(
                SubtrialMetric(
                    direction_alias=group.direction_alias,
                    direction=group.direction_slug,
                    source_dataset=group.source_dataset,
                    target_dataset=group.target_dataset,
                    method=group.method,
                    configured_anchors=group.configured_anchors,
                    actual_anchors=actual_anchors,
                    anchor_count_mismatch=actual_anchors != group.configured_anchors,
                    new_idx=new_idx,
                    old_idx=old_idx,
                    refinement_mode=mode,
                    source_old_micro=source_old_micro,
                    source_before_micro=source_before_micro,
                    source_after_micro=source_after_micro,
                    target_before_micro=target_before_micro,
                    target_after_micro=target_after_micro,
                    source_improvement_micro=source_before_micro - source_after_micro,
                    target_preservation_change_micro=target_before_micro - target_after_micro,
                    source_old_macro=source_old_macro,
                    source_before_macro=source_before_macro,
                    source_after_macro=source_after_macro,
                    target_before_macro=target_before_macro,
                    target_after_macro=target_after_macro,
                    source_improvement_macro=source_before_macro - source_after_macro,
                    target_preservation_change_macro=target_before_macro - target_after_macro,
                    aggregate_path=str(group.aggregate_path),
                    subtrial_path=str(subtrial_path),
                )
            )
    return metrics


_IDENTITY_COLUMNS = [
    "direction_alias",
    "direction",
    "source_dataset",
    "target_dataset",
    "method",
    "configured_anchors",
    "refinement_mode",
]
_STAGES = (
    "source_old",
    "source_before",
    "source_after",
    "target_before",
    "target_after",
    "source_improvement",
    "target_preservation_change",
)


def summarize_subtrials(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute equal-weight means and sample SDs over fold-pair rows."""
    required = set(_IDENTITY_COLUMNS)
    required.update(f"{stage}_{metric}" for stage in _STAGES for metric in ("micro", "macro"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Subtrial frame is missing columns: {', '.join(missing)}")
    records: list[dict[str, Any]] = []
    grouped = frame.groupby(_IDENTITY_COLUMNS, sort=False, dropna=False)
    for identity, group_frame in grouped:
        base = dict(zip(_IDENTITY_COLUMNS, identity))
        audit = {
            "actual_anchors_min": int(group_frame["actual_anchors"].min())
            if "actual_anchors" in group_frame
            else np.nan,
            "actual_anchors_max": int(group_frame["actual_anchors"].max())
            if "actual_anchors" in group_frame
            else np.nan,
            "anchor_mismatch_count": int(group_frame["anchor_count_mismatch"].astype(bool).sum())
            if "anchor_count_mismatch" in group_frame
            else 0,
        }
        for metric in ("micro", "macro"):
            for stage in _STAGES:
                values = pd.to_numeric(group_frame[f"{stage}_{metric}"], errors="raise")
                records.append(
                    {
                        **base,
                        **audit,
                        "metric": metric,
                        "stage": stage,
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "n": int(values.count()),
                    }
                )
    columns = _IDENTITY_COLUMNS + [
        "actual_anchors_min",
        "actual_anchors_max",
        "anchor_mismatch_count",
        "metric",
        "stage",
        "mean",
        "sd",
        "n",
    ]
    return pd.DataFrame.from_records(records, columns=columns).sort_values(
        ["direction", "method", "configured_anchors", "refinement_mode", "metric", "stage"],
        kind="stable",
        ignore_index=True,
    )


def select_best_anchors(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the smallest K tied for minimum final source MAE."""
    final = summary.loc[summary["stage"] == "source_after"].copy()
    keys = [
        "direction_alias",
        "direction",
        "source_dataset",
        "target_dataset",
        "method",
        "refinement_mode",
        "metric",
    ]
    selected: list[pd.Series] = []
    for _identity, group_frame in final.groupby(keys, sort=False, dropna=False):
        minimum = float(group_frame["mean"].min())
        tied = group_frame.loc[
            np.isclose(group_frame["mean"].to_numpy(dtype=float), minimum, rtol=1e-9, atol=1e-12)
        ]
        selected.append(tied.sort_values("configured_anchors", kind="stable").iloc[0])
    if not selected:
        return final.reset_index(drop=True)
    return pd.DataFrame(selected).reset_index(drop=True)


def _axis_x(axis: Any, anchors: list[int], scale: str) -> tuple[np.ndarray, dict[int, float]]:
    ordered = sorted(set(int(value) for value in anchors))
    if scale == "log":
        if any(value <= 0 for value in ordered):
            raise ValueError("Log anchor scale requires positive K values")
        positions = {value: float(value) for value in ordered}
        axis.set_xscale("log")
        axis.set_xticks(ordered)
        axis.set_xticklabels([str(value) for value in ordered])
    elif scale == "categorical":
        positions = {value: float(index) for index, value in enumerate(ordered)}
        axis.set_xticks(range(len(ordered)))
        axis.set_xticklabels([str(value) for value in ordered])
    else:
        raise ValueError(f"Unknown anchor scale: {scale}")
    return np.asarray([positions[value] for value in ordered]), positions


def _plot_summary_line(
    axis: Any,
    rows: pd.DataFrame,
    positions: dict[int, float],
    *,
    label: str,
    color: str,
    marker: str = "o",
    show_std: bool = False,
    clip_std_at_zero: bool = True,
) -> None:
    if rows.empty:
        return
    ordered = rows.sort_values("configured_anchors")
    x = np.asarray([positions[int(value)] for value in ordered["configured_anchors"]])
    mean = ordered["mean"].to_numpy(dtype=float)
    sd = ordered["sd"].fillna(0.0).to_numpy(dtype=float)
    axis.plot(x, mean, marker=marker, linewidth=1.8, markersize=4.5, label=label, color=color)
    if show_std:
        lower = np.maximum(0.0, mean - sd) if clip_std_at_zero else mean - sd
        axis.fill_between(x, lower, mean + sd, color=color, alpha=0.16)


def _summary_rows_for_plot_stage(
    summary: pd.DataFrame,
    plot_stage: str,
    metric_stage: str,
) -> pd.DataFrame:
    """Select one summary curve for a displayed refinement stage."""
    if plot_stage != "projector_only":
        return summary[
            (summary["refinement_mode"] == plot_stage)
            & (summary["stage"] == metric_stage)
        ]

    before_stage = {
        "source_after": "source_before",
        "target_after": "target_before",
    }.get(metric_stage, metric_stage)
    candidates = summary[
        summary["refinement_mode"].isin(REFINEMENT_MODES)
        & (summary["stage"] == before_stage)
    ]
    if candidates.empty:
        return candidates.copy()

    identity_columns = [
        column
        for column in (
            "direction_alias",
            "direction",
            "source_dataset",
            "target_dataset",
            "method",
            "configured_anchors",
            "metric",
            "stage",
        )
        if column in candidates.columns
    ]
    collapsed: list[pd.Series] = []
    for identity, group in candidates.groupby(
        identity_columns, sort=False, dropna=False
    ):
        for column in ("mean", "sd", "n"):
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="raise").to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=1e-9, atol=1e-12, equal_nan=True):
                raise ValueError(
                    f"Inconsistent projector-only {column} values for {identity}: "
                    f"{values.tolist()}"
                )
        preferred = group[group["refinement_mode"] == "linear_only"]
        row = (preferred if not preferred.empty else group).iloc[0].copy()
        row["refinement_mode"] = "projector_only"
        collapsed.append(row)
    return pd.DataFrame(collapsed).reset_index(drop=True)


def _finish_axis(axis: Any, *, zero_line: bool = False) -> None:
    if zero_line:
        axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.set_xlabel("number of anchors")
    axis.tick_params(axis="y", labelleft=True)


def _save_figure(figure: Any, base: Path, options: PlotOptions) -> list[Path]:
    paths: list[Path] = []
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in options.formats:
        path = base.with_suffix(f".{extension}")
        figure.savefig(path, dpi=options.dpi, bbox_inches="tight")
        paths.append(path)
    return paths


def _stage_figure(
    summary: pd.DataFrame,
    direction: str,
    method: str,
    metric: str,
    scale: str,
    show_std: bool = False,
) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row", squeeze=False)
    selected = summary[
        (summary["direction"] == direction)
        & (summary["method"] == method)
        & (summary["metric"] == metric)
    ]
    anchors = sorted(selected["configured_anchors"].astype(int).unique())
    refinement_specs = (
        (("source_old", "Old native", "#666666"), ("source_before", "Projected", "#E69F00"), ("source_after", "Refined", "#009E73")),
        (("target_before", "Original", "#0072B2"), ("target_after", "After refinement", "#D55E00")),
    )
    projector_only_specs = (
        refinement_specs[0][:-1],
        refinement_specs[1][:1],
    )
    for column, plot_stage in enumerate(PLOT_REFINEMENT_STAGES):
        stage_specs = (
            projector_only_specs if plot_stage == "projector_only" else refinement_specs
        )
        for row_index, specs in enumerate(stage_specs):
            axis = axes[row_index, column]
            _unused, positions = _axis_x(axis, anchors, scale)
            for stage, label, color in specs:
                rows = _summary_rows_for_plot_stage(selected, plot_stage, stage)
                _plot_summary_line(
                    axis,
                    rows,
                    positions,
                    label=label,
                    color=color,
                    show_std=show_std,
                )
            _finish_axis(axis)
            axis.set_title(PLOT_STAGE_LABELS[plot_stage])
            if column == 0:
                prefix = "Source-test" if row_index == 0 else "Target-test"
                axis.set_ylabel(f"{prefix} {metric}-MAE (lower is better)")
            axis.legend(fontsize=8)
    figure.suptitle(f"{direction}: {METHOD_LABELS.get(method, method)} stage MAE")
    figure.tight_layout()
    return figure


def _methods_figure(
    summary: pd.DataFrame,
    direction: str,
    metric: str,
    scale: str,
    show_std: bool = False,
) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 8), sharey="row", squeeze=False)
    selected = summary[
        (summary["direction"] == direction)
        & (summary["metric"] == metric)
    ]
    anchors = sorted(selected["configured_anchors"].astype(int).unique())
    source_dataset = str(selected["source_dataset"].iloc[0])
    target_dataset = str(selected["target_dataset"].iloc[0])
    stage_rows = (("source_after", source_dataset), ("target_after", target_dataset))
    for row_index, (stage, dataset_name) in enumerate(stage_rows):
        for column, plot_stage in enumerate(PLOT_REFINEMENT_STAGES):
            axis = axes[row_index, column]
            _unused, positions = _axis_x(axis, anchors, scale)
            for method in METHODS:
                method_rows = selected[selected["method"] == method]
                rows = _summary_rows_for_plot_stage(method_rows, plot_stage, stage)
                _plot_summary_line(
                    axis,
                    rows,
                    positions,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    show_std=show_std,
                )
            _finish_axis(axis)
            axis.set_title(PLOT_STAGE_LABELS[plot_stage])
            if column == 0:
                axis.set_ylabel(f"{dataset_name} {metric}-MAE")
            axis.legend(fontsize=8)
    figure.suptitle(f"{direction}: MAE by method and refinement stage")
    figure.tight_layout()
    return figure


def _improvement_figure(
    summary: pd.DataFrame,
    direction: str,
    metric: str,
    scale: str,
    show_std: bool = False,
) -> Any:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row", squeeze=False)
    selected = summary[(summary["direction"] == direction) & (summary["metric"] == metric)]
    anchors = sorted(selected["configured_anchors"].astype(int).unique())
    stage_rows = (
        ("source_improvement", "Source projected − refined MAE"),
        ("target_preservation_change", "Target before − after MAE"),
    )
    for column, mode in enumerate(REFINEMENT_MODES):
        for row_index, (stage, ylabel) in enumerate(stage_rows):
            axis = axes[row_index, column]
            _unused, positions = _axis_x(axis, anchors, scale)
            for method in METHODS:
                rows = selected[
                    (selected["refinement_mode"] == mode)
                    & (selected["method"] == method)
                    & (selected["stage"] == stage)
                ]
                _plot_summary_line(
                    axis,
                    rows,
                    positions,
                    label=METHOD_LABELS[method],
                    color=METHOD_COLORS[method],
                    show_std=show_std,
                    clip_std_at_zero=False,
                )
            _finish_axis(axis, zero_line=True)
            axis.set_title(mode.replace("_", " "))
            if column == 0:
                axis.set_ylabel(f"{ylabel}\n(positive is improvement)")
            axis.legend(fontsize=8)
    figure.suptitle(f"{direction}: refinement changes ({metric}-MAE)")
    figure.tight_layout()
    return figure


def _heatmap_figure(summary: pd.DataFrame, direction: str, metric: str) -> Any:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    selected = summary[
        (summary["direction"] == direction)
        & (summary["metric"] == metric)
    ]
    methods = [method for method in METHODS if method in set(selected["method"])]
    anchors = sorted(selected["configured_anchors"].astype(int).unique())
    matrices: list[np.ndarray] = []
    for plot_stage in PLOT_REFINEMENT_STAGES:
        matrix = np.full((len(methods), len(anchors)), np.nan)
        for row_index, method in enumerate(methods):
            for column, anchors_value in enumerate(anchors):
                method_rows = selected[
                    (selected["method"] == method)
                    & (selected["configured_anchors"] == anchors_value)
                ]
                match = _summary_rows_for_plot_stage(
                    method_rows, plot_stage, "source_after"
                )
                if not match.empty:
                    matrix[row_index, column] = float(match.iloc[0]["mean"])
        matrices.append(matrix)
    finite = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices])
    vmin, vmax = (float(finite.min()), float(finite.max())) if len(finite) else (0.0, 1.0)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    figure, axes = plt.subplots(
        1, 3, figsize=(max(14, len(anchors) * 1.8), 4.8), squeeze=False
    )
    image = None
    for column, (plot_stage, matrix) in enumerate(
        zip(PLOT_REFINEMENT_STAGES, matrices)
    ):
        axis = axes[0, column]
        image = axis.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        axis.set_xticks(range(len(anchors)), labels=[str(value) for value in anchors])
        axis.set_yticks(range(len(methods)), labels=[METHOD_LABELS.get(value, value) for value in methods])
        axis.set_xlabel("number of anchors")
        axis.set_title(PLOT_STAGE_LABELS[plot_stage])
        for row_index in range(len(methods)):
            finite_columns = np.flatnonzero(np.isfinite(matrix[row_index]))
            if len(finite_columns):
                values = matrix[row_index, finite_columns]
                best_column = int(finite_columns[np.argmin(values)])
                axis.add_patch(
                    Rectangle(
                        (best_column - 0.48, row_index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="white",
                        linewidth=2.2,
                    )
                )
            for anchor_index in finite_columns:
                axis.text(
                    anchor_index,
                    row_index,
                    f"{matrix[row_index, anchor_index]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), label=f"Final source {metric}-MAE (lower is better)")
    figure.suptitle(f"{direction}: final source MAE heatmap")
    figure.subplots_adjust(top=0.82, bottom=0.14, wspace=0.28)
    return figure


def _subtrial_rows_for_plot_stage(
    subtrials: pd.DataFrame,
    plot_stage: str,
    value_column: str,
) -> tuple[pd.DataFrame, str]:
    if plot_stage != "projector_only":
        return subtrials[subtrials["refinement_mode"] == plot_stage], value_column

    before_column = value_column.replace("source_after_", "source_before_").replace(
        "target_after_", "target_before_"
    )
    candidates = subtrials[subtrials["refinement_mode"].isin(REFINEMENT_MODES)]
    if candidates.empty:
        return candidates.copy(), before_column
    identity_columns = [
        column
        for column in (
            "direction_alias",
            "direction",
            "source_dataset",
            "target_dataset",
            "method",
            "configured_anchors",
            "actual_anchors",
            "new_idx",
            "old_idx",
        )
        if column in candidates.columns
    ]
    collapsed: list[pd.Series] = []
    for identity, group in candidates.groupby(
        identity_columns, sort=False, dropna=False
    ):
        values = pd.to_numeric(group[before_column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(values, values[0], rtol=1e-9, atol=1e-12, equal_nan=True):
            raise ValueError(
                f"Inconsistent projector-only {before_column} values for {identity}: "
                f"{values.tolist()}"
            )
        preferred = group[group["refinement_mode"] == "linear_only"]
        row = (preferred if not preferred.empty else group).iloc[0].copy()
        row["refinement_mode"] = "projector_only"
        collapsed.append(row)
    return pd.DataFrame(collapsed).reset_index(drop=True), before_column


def _distribution_figure(subtrials: pd.DataFrame, direction: str, method: str, metric: str) -> Any:
    import matplotlib.pyplot as plt

    selected = subtrials[
        (subtrials["direction"] == direction) & (subtrials["method"] == method)
    ]
    anchors = sorted(selected["configured_anchors"].astype(int).unique())
    value_column = f"source_after_{metric}"
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True, squeeze=False)
    generator = np.random.default_rng(0)
    for column, plot_stage in enumerate(PLOT_REFINEMENT_STAGES):
        axis = axes[0, column]
        stage_rows, stage_value_column = _subtrial_rows_for_plot_stage(
            selected, plot_stage, value_column
        )
        values = [
            stage_rows[stage_rows["configured_anchors"] == anchor][
                stage_value_column
            ].to_numpy(dtype=float)
            for anchor in anchors
        ]
        axis.boxplot(values, positions=np.arange(len(anchors)), widths=0.55, showfliers=False)
        for position, observations in enumerate(values):
            jitter = generator.uniform(-0.14, 0.14, size=len(observations))
            axis.scatter(
                position + jitter,
                observations,
                s=14,
                alpha=0.65,
                color=METHOD_COLORS.get(method, "#0072B2"),
                edgecolors="none",
            )
        axis.set_xticks(range(len(anchors)), labels=[str(value) for value in anchors])
        axis.set_xlabel("number of anchors")
        axis.set_title(PLOT_STAGE_LABELS[plot_stage])
        axis.grid(True, axis="y", alpha=0.25)
        axis.tick_params(axis="y", labelleft=True)
    axes[0, 0].set_ylabel(f"Source-test {metric}-MAE")
    figure.suptitle(f"{direction}: {METHOD_LABELS.get(method, method)} fold-pair distributions")
    figure.tight_layout()
    return figure


def _validate_projector_only_before_values(subtrials: pd.DataFrame) -> None:
    for dataset_role in ("source", "target"):
        for metric in ("micro", "macro"):
            column = f"{dataset_role}_before_{metric}"
            if column in subtrials.columns:
                _subtrial_rows_for_plot_stage(subtrials, "projector_only", column)


def _plot_figure_total(summary: pd.DataFrame, options: PlotOptions) -> int:
    total = 0
    line_families = set(options.plots) & {"stages", "methods", "improvement"}
    for metric in options.metrics:
        for direction in summary["direction"].unique():
            selected = summary[
                (summary["direction"] == direction) & (summary["metric"] == metric)
            ]
            method_count = sum(method in set(selected["method"]) for method in METHODS)
            for _scale in options.anchor_scales:
                if "stages" in line_families:
                    total += method_count
                if "methods" in line_families:
                    total += 1
                if "improvement" in line_families:
                    total += 1
            if "heatmap" in options.plots:
                total += 1
            if "distribution" in options.plots:
                total += method_count
    return total


def render_plots(
    subtrials: pd.DataFrame,
    summary: pd.DataFrame,
    options: PlotOptions,
) -> list[Path]:
    """Render the selected plot families and return every written path."""
    if options.exclude_linear_close:
        subtrials = subtrials[subtrials["method"] != "linear_close"].reset_index(
            drop=True
        )
        summary = summary[summary["method"] != "linear_close"].reset_index(drop=True)
    _validate_projector_only_before_values(subtrials)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    directions = sorted(summary["direction"].unique())
    line_families = set(options.plots) & {"stages", "methods", "improvement"}
    progress = _progress(
        total=_plot_figure_total(summary, options),
        desc="Rendering figures",
        unit="figure",
        enabled=_progress_enabled() if options.show_progress is None else options.show_progress,
    )
    for metric in options.metrics:
        figure_dir = Path(options.output_dir) / "figures" / metric
        for direction in directions:
            direction_summary = summary[
                (summary["direction"] == direction) & (summary["metric"] == metric)
            ]
            methods = [method for method in METHODS if method in set(direction_summary["method"])]
            for scale in options.anchor_scales:
                if "stages" in line_families:
                    for method in methods:
                        figure = _stage_figure(
                            summary,
                            direction,
                            method,
                            metric,
                            scale,
                            show_std=options.show_std,
                        )
                        paths.extend(
                            _save_figure(
                                figure,
                                figure_dir / f"stages__{direction}__{method}__{scale}",
                                options,
                            )
                        )
                        plt.close(figure)
                        progress.update(1)
                if "methods" in line_families:
                    figure = _methods_figure(
                        summary,
                        direction,
                        metric,
                        scale,
                        show_std=options.show_std,
                    )
                    paths.extend(
                        _save_figure(
                            figure, figure_dir / f"methods__{direction}__{scale}", options
                        )
                    )
                    plt.close(figure)
                    progress.update(1)
                if "improvement" in line_families:
                    figure = _improvement_figure(
                        summary,
                        direction,
                        metric,
                        scale,
                        show_std=options.show_std,
                    )
                    paths.extend(
                        _save_figure(
                            figure, figure_dir / f"improvement__{direction}__{scale}", options
                        )
                    )
                    plt.close(figure)
                    progress.update(1)
            if "heatmap" in options.plots:
                figure = _heatmap_figure(summary, direction, metric)
                paths.extend(
                    _save_figure(figure, figure_dir / f"heatmap__{direction}", options)
                )
                plt.close(figure)
                progress.update(1)
            if "distribution" in options.plots:
                for method in methods:
                    figure = _distribution_figure(subtrials, direction, method, metric)
                    paths.extend(
                        _save_figure(
                            figure,
                            figure_dir / f"distribution__{direction}__{method}",
                            options,
                        )
                    )
                    plt.close(figure)
                    progress.update(1)
    progress.close()
    return paths


def _cache_base(output_dir: Path, group: SweepGroup) -> Path:
    return output_dir / "cache" / (
        f"{group.direction_slug}__{group.method}__K{group.configured_anchors}"
    )


def _file_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _group_signature(group: SweepGroup) -> dict[str, Any]:
    if group.aggregate_path is None:
        raise ValueError(f"Cannot sign group without a selected aggregate: {group.k_dir}")
    aggregate = _load_pickle(group.aggregate_path)
    if not isinstance(aggregate, dict):
        raise ValueError(f"Aggregate PKL is not a mapping: {group.aggregate_path}")
    values = aggregate.get("subtrial_pkls", [])
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"subtrial_pkls is not a list: {group.aggregate_path}")
    files = [_file_signature(group.aggregate_path)]
    for value in values:
        subtrial_path = _manifest_path(group.aggregate_path, value)
        files.append(_file_signature(subtrial_path))
        if subtrial_path.is_file():
            try:
                payload = _load_pickle(subtrial_path)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                try:
                    anchors_path = _anchors_path(payload, subtrial_path)
                except ValueError:
                    anchors_path = subtrial_path.parent / "anchors.csv"
                files.append(_file_signature(anchors_path))
    payload = {"schema_version": CACHE_SCHEMA_VERSION, "files": files}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**payload, "digest": hashlib.sha256(encoded).hexdigest()}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _cache_matches(base: Path, group: SweepGroup, signature: dict[str, Any]) -> bool:
    csv_path = base.with_suffix(".csv")
    metadata = _read_json(base.with_suffix(".json"))
    return bool(
        csv_path.is_file()
        and metadata
        and metadata.get("schema_version") == CACHE_SCHEMA_VERSION
        and metadata.get("signature") == signature
        and metadata.get("identity")
        == {
            "direction": group.direction_slug,
            "method": group.method,
            "configured_anchors": group.configured_anchors,
        }
    )


def _write_group_cache(
    base: Path,
    group: SweepGroup,
    signature: dict[str, Any],
    frame: pd.DataFrame,
    validation: ValidationResult,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base.with_suffix(".csv"), index=False)
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "identity": {
            "direction": group.direction_slug,
            "method": group.method,
            "configured_anchors": group.configured_anchors,
        },
        "signature": signature,
        "rows": len(frame),
        "validation_warnings": list(validation.warnings),
    }
    with base.with_suffix(".json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _validation_records(result: ValidationResult) -> list[dict[str, Any]]:
    group = result.group
    base = {
        "direction_alias": group.direction_alias,
        "direction": group.direction_slug,
        "method": group.method,
        "configured_anchors": group.configured_anchors,
        "k_dir": str(group.k_dir),
        "aggregate_path": str(group.aggregate_path) if group.aggregate_path else "",
    }
    records = [{**base, "severity": "error", "message": value} for value in result.errors]
    records.extend({**base, "severity": "warning", "message": value} for value in result.warnings)
    if not records:
        records.append({**base, "severity": "info", "message": "Group validation passed"})
    return records


def _global_validation_records(
    groups: list[SweepGroup], allow_incomplete: bool
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identities: dict[tuple[str, str, int], list[SweepGroup]] = {}
    for group in groups:
        identities.setdefault(
            (group.direction_slug, group.method, group.configured_anchors), []
        ).append(group)
    for identity, matches in identities.items():
        if len(matches) > 1:
            records.append(
                {
                    "direction_alias": "",
                    "direction": identity[0],
                    "method": identity[1],
                    "configured_anchors": identity[2],
                    "k_dir": " | ".join(str(group.k_dir) for group in matches),
                    "aggregate_path": "",
                    "severity": "error",
                    "message": f"Duplicate direction/method/K identity {identity}",
                }
            )
    for direction in sorted({group.direction_slug for group in groups}):
        by_method = {
            method: {
                group.configured_anchors
                for group in groups
                if group.direction_slug == direction and group.method == method
            }
            for method in {group.method for group in groups if group.direction_slug == direction}
        }
        grids = {tuple(sorted(values)) for values in by_method.values()}
        if len(by_method) > 1 and len(grids) > 1:
            severity = "warning" if allow_incomplete else "error"
            records.append(
                {
                    "direction_alias": "",
                    "direction": direction,
                    "method": "",
                    "configured_anchors": "",
                    "k_dir": "",
                    "aggregate_path": "",
                    "severity": severity,
                    "message": "Inconsistent anchor grids between methods: "
                    + "; ".join(
                        f"{method}={sorted(values)}" for method, values in sorted(by_method.items())
                    ),
                }
            )
    return records


def _direction_slugs() -> tuple[str, ...]:
    return tuple(
        f"{source.lower()}-to-{target.lower()}" for source, target in DIRECTION_ALIASES.values()
    )


def _expand_choice(value: str, all_values: tuple[str, ...]) -> tuple[str, ...]:
    return all_values if value == "both" else (value,)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and plot descriptive micro/macro MAE across anchor sweeps. "
            "Fold-pair SDs are descriptive; no inferential independence is assumed."
        )
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--direction", nargs="+", metavar="SLUG")
    parser.add_argument("--method", nargs="+", choices=METHODS)
    parser.add_argument("--anchors", nargs="+", type=int, metavar="K")
    parser.add_argument("--metric", choices=("micro", "macro", "both"), default="both")
    parser.add_argument(
        "--anchor-scale", choices=("log", "categorical", "both"), default="both"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=("stages", "methods", "improvement", "heatmap", "distribution", "all"),
        default=("all",),
    )
    parser.add_argument("--format", choices=("png", "pdf", "both"), default="both")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show-std",
        action="store_true",
        help="Show descriptive ±1 sample-SD ribbons on line plots (hidden by default).",
    )
    parser.add_argument(
        "--exclude-linear-close",
        action="store_true",
        help="Exclude the linear_close method from plots while retaining it in data tables.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--aggregate-policy",
        choices=("error", "latest"),
        default="error",
        help=(
            "Resolve multiple aggregated_<UID>/results_<UID>.pkl candidates within a K "
            "directory: 'error' (default) requires exactly one aggregate and reports "
            "duplicates as a validation error; 'latest' selects the highest numeric "
            "aggregate UID."
        ),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--extract-only", action="store_true")
    mode.add_argument("--plot-only", action="store_true")
    return parser


def _filter_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    selected = frame
    if args.direction:
        selected = selected[selected["direction"].isin(args.direction)]
    if args.method:
        selected = selected[selected["method"].isin(args.method)]
    if args.anchors:
        selected = selected[selected["configured_anchors"].isin(args.anchors)]
    return selected.reset_index(drop=True)


def _load_plot_only_cache(output_dir: Path, args: argparse.Namespace) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cache_dir = output_dir / "cache"
    for metadata_path in sorted(cache_dir.glob("*.json")):
        metadata = _read_json(metadata_path)
        if not metadata or metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            continue
        identity = metadata.get("identity", {})
        if args.direction and identity.get("direction") not in args.direction:
            continue
        if args.method and identity.get("method") not in args.method:
            continue
        if args.anchors and identity.get("configured_anchors") not in args.anchors:
            continue
        csv_path = metadata_path.with_suffix(".csv")
        if csv_path.is_file():
            frames.append(pd.read_csv(csv_path))
    if not frames:
        raise ValueError(
            f"No compatible schema-v{CACHE_SCHEMA_VERSION} group caches matched in {cache_dir}"
        )
    return pd.concat(frames, ignore_index=True)


def _write_analysis_tables(output_dir: Path, subtrials: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    subtrials.to_csv(data_dir / "anchor_sweep_subtrials.csv", index=False)
    summary = summarize_subtrials(subtrials)
    summary.to_csv(data_dir / "anchor_sweep_summary.csv", index=False)
    best = select_best_anchors(summary)
    best.to_csv(data_dir / "anchor_sweep_best_anchors.csv", index=False)
    return summary, best


def _write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    subtrials: pd.DataFrame | None,
    figures: Sequence[Path],
    status: str,
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    metadata = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "arguments": arguments,
        "subtrial_rows": 0 if subtrials is None else len(subtrials),
        "fold_pair_interpretation": (
            "Each new-fold x old-fold combination is equally weighted. Sample SD is descriptive; "
            "the combinations are not treated as independent observations for inference."
        ),
        "figures": [str(path) for path in figures],
    }
    with (data_dir / "run_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    progress_enabled = _progress_enabled()
    known_directions = set(_direction_slugs())
    if args.direction:
        unknown = sorted(set(args.direction) - known_directions)
        if unknown:
            parser.error(
                f"unknown direction slug(s): {', '.join(unknown)}; expected: "
                + ", ".join(_direction_slugs())
            )
    if args.anchors and any(value < 1 for value in args.anchors):
        parser.error("--anchors values must be positive integers")
    if args.dpi < 1 or args.workers < 1:
        parser.error("--dpi and --workers must be positive integers")
    if args.plot_only and args.refresh_cache:
        parser.error("--refresh-cache cannot be combined with --plot-only")
    if (
        args.exclude_linear_close
        and not args.extract_only
        and args.method
        and "linear_close" in args.method
    ):
        parser.error(
            "--exclude-linear-close cannot be combined with --method containing linear_close"
        )

    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    validation_records: list[dict[str, Any]] = []

    if args.plot_only:
        try:
            subtrials = _load_plot_only_cache(output_dir, args)
        except ValueError as error:
            parser.error(str(error))
        validation_records.append(
            {
                "direction_alias": "",
                "direction": "",
                "method": "",
                "configured_anchors": "",
                "k_dir": "",
                "aggregate_path": "",
                "severity": "info",
                "message": "Plot-only mode trusted compatible per-group caches; PKLs were not scanned",
            }
        )
    else:
        try:
            groups = discover_groups(args.input_root, args.aggregate_policy)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        groups = [
            group
            for group in groups
            if (not args.direction or group.direction_slug in args.direction)
            and (not args.method or group.method in args.method)
            and (not args.anchors or group.configured_anchors in args.anchors)
        ]
        if not groups:
            validation_records.append(
                {
                    "direction_alias": "",
                    "direction": "",
                    "method": "",
                    "configured_anchors": "",
                    "k_dir": "",
                    "aggregate_path": "",
                    "severity": "error",
                    "message": "No sweep groups matched the selected filters",
                }
            )
            pd.DataFrame(validation_records).to_csv(
                data_dir / "validation_report.csv", index=False
            )
            _write_metadata(output_dir, args, None, (), "validation_failed")
            return 1

        validations = [
            validate_group(group, allow_incomplete=args.allow_incomplete)
            for group in _progress(
                groups,
                desc="Validating groups",
                unit="group",
                enabled=progress_enabled,
            )
        ]
        for result in validations:
            validation_records.extend(_validation_records(result))
        validation_records.extend(_global_validation_records(groups, args.allow_incomplete))
        report = pd.DataFrame(validation_records)
        report.to_csv(data_dir / "validation_report.csv", index=False)
        if (report["severity"] == "error").any():
            _write_metadata(output_dir, args, None, (), "validation_failed")
            print(
                f"Validation failed; see {data_dir / 'validation_report.csv'}",
                file=sys.stderr,
            )
            return 1

        signatures: dict[SweepGroup, dict[str, Any]] = {}
        frames_by_group: dict[SweepGroup, pd.DataFrame] = {}
        stale: list[tuple[SweepGroup, ValidationResult]] = []
        group_validations = zip(groups, validations)
        for group, validation in _progress(
            group_validations,
            total=len(groups),
            desc="Checking caches",
            unit="group",
            enabled=progress_enabled,
        ):
            signatures[group] = _group_signature(group)
            base = _cache_base(output_dir, group)
            if not args.refresh_cache and _cache_matches(base, group, signatures[group]):
                frames_by_group[group] = pd.read_csv(base.with_suffix(".csv"))
            else:
                stale.append((group, validation))

        if stale:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(extract_group, group, args.allow_incomplete): (group, validation)
                    for group, validation in stale
                }
                extraction_progress = _progress(
                    total=len(futures),
                    desc="Extracting groups",
                    unit="group",
                    enabled=progress_enabled,
                )
                for future in as_completed(futures):
                    group, validation = futures[future]
                    frame = pd.DataFrame(asdict(row) for row in future.result())
                    _write_group_cache(
                        _cache_base(output_dir, group),
                        group,
                        signatures[group],
                        frame,
                        validation,
                    )
                    frames_by_group[group] = frame
                    extraction_progress.update(1)
                extraction_progress.close()
        subtrials = pd.concat([frames_by_group[group] for group in groups], ignore_index=True)

    subtrials = _filter_frame(subtrials, args)
    if subtrials.empty:
        print("error: no cached/extracted rows matched the filters", file=sys.stderr)
        _write_metadata(output_dir, args, subtrials, (), "no_rows")
        return 1
    try:
        _validate_projector_only_before_values(subtrials)
    except ValueError as error:
        validation_records.append(
            {
                "direction_alias": "",
                "direction": "",
                "method": "",
                "configured_anchors": "",
                "k_dir": "",
                "aggregate_path": "",
                "severity": "error",
                "message": str(error),
            }
        )
        pd.DataFrame(validation_records).to_csv(
            data_dir / "validation_report.csv", index=False
        )
        _write_metadata(
            output_dir,
            args,
            subtrials,
            (),
            "projector_only_validation_failed",
        )
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.plot_only:
        pd.DataFrame(validation_records).to_csv(data_dir / "validation_report.csv", index=False)
    summary, _best = _write_analysis_tables(output_dir, subtrials)
    figures: list[Path] = []
    if not args.extract_only:
        plots = (
            ("stages", "methods", "improvement", "heatmap", "distribution")
            if "all" in args.plots
            else tuple(dict.fromkeys(args.plots))
        )
        options = PlotOptions(
            output_dir=output_dir,
            metrics=_expand_choice(args.metric, ("micro", "macro")),
            anchor_scales=_expand_choice(args.anchor_scale, ("log", "categorical")),
            plots=plots,
            formats=_expand_choice(args.format, ("png", "pdf")),
            dpi=args.dpi,
            show_std=args.show_std,
            exclude_linear_close=args.exclude_linear_close,
            show_progress=progress_enabled,
        )
        figures = render_plots(subtrials, summary, options)
    _write_metadata(output_dir, args, subtrials, figures, "complete")
    print(
        f"Wrote {len(subtrials)} subtrial-mode rows and {len(figures)} figures to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

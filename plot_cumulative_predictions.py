#!/usr/bin/env python3
"""Plot cumulative pain predictions for consecutive video prefixes."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import math
from dataclasses import dataclass, replace
import pickle
import re
import shlex
import sys
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import numpy as np


CHUNK_SIZE = 16
STAGE_TO_MODE = {
    1: "linear_only",
    2: "projector_linear",
    4: "random_projector_linear",
}
SUPPORTED_PROJECTORS = frozenset(
    {"linear", "mlp", "autoencoder", "procrustes", "linear_close"}
)
SUPPORTED_DATASETS = (
    "UNBC", "BIOVID", "MINTPAIN", "AGEDB", "CAER", "MORPH", "PEMF",
)
DATASET_MARKERS = {
    "UNBC": ("unbc",),
    "BIOVID": ("parta", "biovid"),
    "MINTPAIN": ("mint",),
    "AGEDB": ("agedb",),
    "CAER": ("caer",),
    "MORPH": ("morph",),
    "PEMF": ("pemf",),
}


@dataclass(frozen=True)
class NativeSelection:
    root: Path
    config: dict
    fold: int
    subfold: int
    epoch: int
    checkpoint: Path
    test_csv: Path
    row: dict
    video_path: Path


@dataclass(frozen=True)
class CrossVariant:
    new_idx: int
    old_idx: int
    new_model_pth: Path
    old_model_pth: Path
    result_path: Path
    result: dict
    refinement: dict


@dataclass(frozen=True)
class CrossSelection:
    root: Path
    result_path: Path
    origin: str
    stage: int
    fixed_index: int
    fixed_model_pth: Path
    test_csv: Path
    row: dict
    variants: tuple[CrossVariant, ...]


@dataclass(frozen=True)
class CurveSet:
    summary: dict[str, Sequence[float]]
    detail: dict[str, Sequence[float]]
    full_range_summary: dict[str, Sequence[float]] | None = None


@dataclass(frozen=True)
class InferenceResult:
    curves: CurveSet
    frame_count: int
    ground_truth: float
    metadata: str
    origin_dataset: str
    origin_model: str
    source_dataset: str | None = None
    source_model: str | None = None
    target_dataset: str | None = None
    target_model: str | None = None


@dataclass
class RuntimeModel:
    config: dict
    checkpoint: Path
    owner: object
    backbone: object
    head: object


def chunk_frames(frames: Sequence, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Split consecutive frames and pad the tail by repeating its final frame."""
    frames = np.asarray(frames)
    if len(frames) == 0:
        raise ValueError("video contains no frames")
    missing = (-len(frames)) % chunk_size
    if missing:
        frames = np.concatenate(
            [frames, np.repeat(frames[-1:], missing, axis=0)], axis=0)
    return frames.reshape(-1, chunk_size, *frames.shape[1:])


def cumulative_prefixes(chunks: Sequence) -> list:
    return [chunks[: end + 1] for end in range(len(chunks))]


def cumulative_frame_ranges(
    frame_count: int, chunk_size: int = CHUNK_SIZE,
) -> list[tuple[int, int]]:
    if frame_count <= 0:
        return []
    return [
        (0, min(frame_count - 1, (index + 1) * chunk_size - 1))
        for index in range((frame_count + chunk_size - 1) // chunk_size)
    ]


def refinement_mode(stage: int) -> str:
    try:
        return STAGE_TO_MODE[stage]
    except KeyError as exc:
        raise ValueError(f"unsupported refinement stage: {stage}") from exc


def mean_curve(curves: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(curve, dtype=np.float64) for curve in curves]
    if not arrays:
        raise ValueError("cannot average an empty curve collection")
    if len({len(curve) for curve in arrays}) != 1:
        raise ValueError("curves must have the same length")
    return np.mean(np.stack(arrays), axis=0)


def source_curve_set(
    native: Sequence[float],
    plain_by_target: dict[int, Sequence[float]],
    refined_by_target: dict[int, Sequence[float]],
    *,
    stage: int,
    source_label: str,
    target_label: str,
) -> CurveSet:
    projection_label = f"{source_label} → {target_label}"
    refinement_label = (
        "Task-aware ref. (mean)" if stage == 2
        else f"Mean stage {stage} refinement"
    )
    return CurveSet(
        summary={
            f"Source model — {source_label}": native,
            f"Projection-only (mean) — {projection_label}": mean_curve(
                list(plain_by_target.values())
            ),
            f"{refinement_label} — {projection_label}": mean_curve(
                list(refined_by_target.values())
            ),
        },
        detail={
            f"Target new_idx={index} — {target_label}": values
            for index, values in sorted(refined_by_target.items())
        },
    )


def target_curve_set(
    native: Sequence[float],
    refined_by_source: dict[int, Sequence[float]],
    *,
    stage: int,
    source_label: str,
    target_label: str,
) -> CurveSet:
    projection_label = f"{source_label} → {target_label}"
    detail = {f"Native target — {target_label}": native}
    detail.update({
        f"Source old_idx={index} — {projection_label}": values
        for index, values in sorted(refined_by_source.items())
    })
    return CurveSet(
        summary={
            f"Native target — {target_label}": native,
            f"Mean refined linear (stage {stage}) — {projection_label}": mean_curve(
                list(refined_by_source.values())
            ),
        },
        detail=detail,
    )


def prediction_title(
    sample_id: str,
    origin: str,
    result: InferenceResult,
) -> str:
    if origin == "native":
        return (
            f"{result.origin_dataset} · Sample {sample_id} · "
            f"Native {result.origin_model}\n{result.metadata}"
        )
    source = f"{result.source_model} ({result.source_dataset})"
    target = f"{result.target_model} ({result.target_dataset})"
    return (
        f"{result.origin_dataset} · Sample {sample_id} · "
        f"Origin {origin}: {result.origin_model}\n"
        f"Cross-space: {source} → {target} · {result.metadata}"
    )


def presentation_title(
    sample_id: str,
    origin: str,
    result: InferenceResult,
    y_max: float,
    *,
    stage: int | None,
) -> str:
    model = (
        "VideoMAE vs DFER" if result.origin_model == "VideoMAE_DFER"
        else result.origin_model
    )
    title = (
        f"Sample {sample_id} ({result.origin_dataset}) · "
        f"Origin-{origin}: {model} · labels range: 0-{y_max:g}"
    )
    if origin == "native":
        return title
    stage_label = {
        1: "projector-only",
        2: "task-aware refin.",
        4: "Stage 4 refin.",
    }[stage]
    return (
        f"{title}\nCross-space: {result.source_model} ({result.source_dataset}) "
        f"-> {result.target_model} ({result.target_dataset}) · {stage_label}"
    )


def combine_native_results(
    first: InferenceResult,
    second: InferenceResult,
    first_axis_max: float,
    second_axis_max: float,
) -> tuple[InferenceResult, float]:
    by_model = {result.origin_model: result for result in (first, second)}
    if set(by_model) != {"VideoMAE", "DFER"}:
        raise ValueError("native comparison requires one VideoMAE and one DFER model")
    if first.origin_dataset != second.origin_dataset:
        raise ValueError("native comparison models must use the same dataset")
    if first.frame_count != second.frame_count:
        raise ValueError("native comparison models must use the same frame count")
    if not math.isclose(first.ground_truth, second.ground_truth):
        raise ValueError("native comparison models must use the same ground truth")
    if not math.isclose(first_axis_max, second_axis_max):
        raise ValueError("native comparison models must use the same prediction range")

    curves = {}
    for model in ("VideoMAE", "DFER"):
        summary = by_model[model].curves.summary
        if len(summary) != 1:
            raise ValueError(f"native {model} inference must produce exactly one curve")
        curves[f"{model} (native)"] = next(iter(summary.values()))
    if len({len(curve) for curve in curves.values()}) != 1:
        raise ValueError("native comparison models must use the same prefix count")
    curve_set = CurveSet(summary=curves, detail=curves)
    return InferenceResult(
        curves=curve_set,
        frame_count=first.frame_count,
        ground_truth=first.ground_truth,
        metadata="",
        origin_dataset=first.origin_dataset,
        origin_model="VideoMAE_DFER",
    ), first_axis_max


def validate_native_comparison_selections(
    first: NativeSelection,
    second: NativeSelection,
) -> None:
    identity = ("sample_id", "subject_name", "sample_name")
    if any(first.row.get(key) != second.row.get(key) for key in identity):
        raise ValueError("native comparison experiments must refer to the same sample")


def native_comparison_title(
    sample_id: str,
    result: InferenceResult,
    y_max: float,
) -> str:
    return (
        f"VideoMAE vs DFER · {result.origin_dataset} · Sample {sample_id} · "
        f"Full range 0–{y_max:g}"
    )


def prediction_figure_title(
    title: str, suffix: str, *, debug_title: bool = True,
) -> str:
    if not debug_title:
        return title
    first, separator, remainder = title.partition("\n")
    return f"{first} · {suffix}{separator}{remainder}"


def target_refined_predictions(
    native_embeddings: Sequence,
    refined_linears: Sequence[tuple[int, object, dict]],
    *,
    predict_linear,
) -> dict[int, list[float]]:
    """Apply target heads directly; target-origin inference never uses a projector."""
    return {
        index: [predict_linear(linear, embedding, config) for embedding in native_embeddings]
        for index, linear, config in refined_linears
    }


def invocation_output_dir(
    output_root: str | Path,
    *,
    sample_id: str,
    origin: str,
    stage: int | None,
    origin_dataset: str,
    origin_model: str,
    source_dataset: str | None = None,
    source_model: str | None = None,
    target_dataset: str | None = None,
    target_model: str | None = None,
    projector_kind: str | None = None,
) -> Path:
    origin_identity = f"{origin_model}_{origin_dataset}".lower()
    if origin == "native":
        return Path(output_root) / origin_identity / str(sample_id) / origin
    if not projector_kind:
        raise ValueError("cross-projection output requires a projector kind")
    cross_identity = (
        f"{source_model}_{source_dataset}_to_{target_dataset}_{target_model}".lower()
    )
    return (
        Path(output_root) / cross_identity / str(sample_id) / projector_kind
        / origin_identity / f"stage_{stage}"
    )


def invocation_digest(argv: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(argv).encode()).hexdigest()[:12]


def inverse_target(values, model_config: dict):
    cfg = model_config.get("config", model_config)
    metadata = cfg.get("target_spec") or {}
    normalized = metadata.get("normalization") == "min_max"
    if not metadata:
        normalized = bool(cfg.get("normalize_labels", 0))
    array = np.asarray(values, dtype=np.float64)
    if not normalized:
        return array
    lower = float(metadata.get("target_min", 0.0))
    upper = float(metadata.get("target_max", cfg.get("max_label", 1.0)))
    return array * (upper - lower) + lower


def dataset_axis_max(model_config: dict, test_csv: str | Path) -> float:
    cfg = model_config.get("config", model_config)
    target_max = (cfg.get("target_spec") or {}).get("target_max")
    if target_max is None:
        target_max = cfg.get("max_label")
    if target_max is not None:
        return float(target_max)

    values = []
    for path in sorted(Path(test_csv).parent.parent.glob("k*_cross_val/test.csv")):
        with path.open(newline="") as stream:
            values.extend(
                float(row["class_id"])
                for row in csv.DictReader(stream, delimiter="\t")
                if row.get("class_id", "").strip()
            )
    if not values:
        raise ValueError("cannot determine target maximum from config or test CSVs")
    return max(values)


def require_cuda(torch_module=None) -> None:
    if torch_module is None:
        import torch as torch_module
    if not torch_module.cuda.is_available():
        raise RuntimeError("CUDA is required for cumulative video inference")


def decode_video(video_path: str | Path, *, cv2_module=None) -> np.ndarray:
    if cv2_module is None:
        import cv2 as cv2_module
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"video cannot be opened: {video_path}")
    frames = []
    try:
        while True:
            available, frame = capture.read()
            if not available:
                break
            frames.append(cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {video_path}")
    return np.stack(frames)


def _prediction_value(logits, head, model_config: dict, torch_module) -> float:
    cfg = model_config.get("config", model_config)
    params = model_config.get("model_advanced_params") or {}
    head_params = params.get("head_params") or cfg.get("head_params") or {}
    if head_params.get("coral_loss"):
        return float((torch_module.sigmoid(logits) > 0.5).sum().item())
    if getattr(head, "is_classification", False):
        return float(torch_module.argmax(logits, dim=-1).reshape(-1)[0].item())
    flat = logits.reshape(-1)
    if flat.numel() != 1:
        raise ValueError(f"regression head returned {flat.numel()} values")
    return float(inverse_target([flat[0].item()], model_config)[0])


def run_head_prefixes(
    head,
    clip_features,
    model_config: dict,
    *,
    torch_module=None,
    device="cuda",
) -> tuple[list[float], list[np.ndarray]]:
    if torch_module is None:
        import torch as torch_module
    if hasattr(head, "eval"):
        head.eval()
    predictions, embeddings = [], []
    with torch_module.inference_mode():
        for end in range(1, len(clip_features) + 1):
            prefix = clip_features[:end].reshape(1, -1, clip_features.shape[-1])
            prefix = prefix.to(device=device, dtype=torch_module.float32)
            output = head(x=prefix, key_padding_mask=None, return_video_emb=True)
            predictions.append(
                _prediction_value(output["logits"], head, model_config, torch_module)
            )
            embeddings.append(
                output["embeddings"].reshape(-1).detach().cpu().numpy().astype(np.float32)
            )
    return predictions, embeddings


def build_projector_network(
    input_dim: int,
    output_dim: int,
    kind: str,
    *,
    activation: str | None = None,
    num_layers: int = 1,
    encoder_ratio: int = 4,
    torch_module=None,
):
    if torch_module is None:
        import torch as torch_module
    if kind not in SUPPORTED_PROJECTORS:
        raise ValueError(f"unsupported projector type: {kind!r}")
    if kind in {"linear", "procrustes", "linear_close"}:
        return torch_module.nn.Linear(input_dim, output_dim)
    activations = {
        "gelu": torch_module.nn.GELU,
        "relu": torch_module.nn.ReLU,
        "silu": torch_module.nn.SiLU,
        "leaky_relu": torch_module.nn.LeakyReLU,
    }
    if activation not in activations:
        raise ValueError(f"unsupported projector activation: {activation!r}")
    active = activations[activation]
    if kind == "mlp":
        if num_layers < 1:
            raise ValueError("MLP projector layers must be positive")
        layers = [torch_module.nn.Linear(input_dim, output_dim)]
        for _ in range(num_layers):
            layers += [active(), torch_module.nn.Linear(output_dim, output_dim)]
        return torch_module.nn.Sequential(*layers)
    if encoder_ratio < 1:
        raise ValueError("autoencoder ratio must be positive")
    hidden_in = max(1, input_dim // encoder_ratio)
    hidden_out = max(1, output_dim // encoder_ratio)
    return torch_module.nn.Sequential(
        torch_module.nn.Linear(input_dim, hidden_in),
        active(),
        torch_module.nn.Linear(hidden_in, hidden_out),
        active(),
        torch_module.nn.Linear(hidden_out, output_dim),
    )


def apply_projector(
    projector,
    embedding,
    norm_stats: dict | None,
    *,
    torch_module=None,
    device="cuda",
) -> np.ndarray:
    if torch_module is None:
        import torch as torch_module
    value = torch_module.as_tensor(embedding, dtype=torch_module.float32, device=device)
    if norm_stats is not None:
        old_mean = torch_module.as_tensor(
            norm_stats["old_mean"], dtype=torch_module.float32, device=device)
        old_std = torch_module.as_tensor(
            norm_stats["old_std"], dtype=torch_module.float32, device=device)
        value = (value - old_mean) / old_std
    with torch_module.inference_mode():
        projected = projector(value)
    if norm_stats is not None:
        new_mean = torch_module.as_tensor(
            norm_stats["new_mean"], dtype=torch_module.float32, device=device)
        new_std = torch_module.as_tensor(
            norm_stats["new_std"], dtype=torch_module.float32, device=device)
        projected = projected * new_std + new_mean
    return projected.detach().cpu().numpy().astype(np.float32)


def _artifact_path(variant: CrossVariant, stored: str | Path) -> Path:
    path = _stored_path(stored, relative_to=variant.result_path.parent)
    if path.exists():
        return path
    config = variant.result.get("config_cross_space_projection") or {}
    old_base = config.get("out_dir")
    if old_base:
        try:
            suffix = Path(stored).relative_to(old_base)
        except ValueError:
            pass
        else:
            relocated = variant.result_path.parent / suffix
            if relocated.exists():
                return relocated.resolve()
    raise FileNotFoundError(f"artifact unavailable: {stored}")


def selected_projector_path(variant: CrossVariant, stage: int) -> Path:
    if stage == 1:
        stored = (variant.result.get("linear_projector") or {}).get("ckpt_path")
    else:
        stored = variant.refinement.get("projector_after_pth")
    if not stored:
        raise FileNotFoundError(f"stage {stage} has no projector checkpoint")
    return _artifact_path(variant, stored)


def _config_for_checkpoint(checkpoint: Path, embedded: dict | None) -> dict:
    if embedded and embedded.get("model_advanced_params"):
        return embedded
    config_path = checkpoint.parents[3] / "k_fold_results.pkl"
    if not config_path.is_file():
        raise FileNotFoundError(f"model configuration unavailable: {config_path}")
    return _load_pickle(config_path)


def _model_type_name(model_type) -> str:
    return str(getattr(model_type, "name", model_type)).upper()


def model_dataset_identity(model_config: dict) -> tuple[str, str]:
    params = model_config.get("model_advanced_params") or {}
    cfg = model_config.get("config") or {}
    model_type = _model_type_name(params.get("model_type"))
    if model_type == "DFER":
        model = "DFER"
    elif model_type.startswith("VIDEOMAE"):
        model = "VideoMAE"
    else:
        raise ValueError(f"unsupported backbone: {model_type!r}")

    paths = " ".join(str(value) for value in (
        params.get("features_folder_saving_path"),
        params.get("path_dataset"),
        cfg.get("path_video_dataset"),
    ) if value).lower()
    for dataset, markers in DATASET_MARKERS.items():
        if any(marker in paths for marker in markers):
            return dataset, model
    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(
        f"Cannot detect dataset from stored model paths; supported datasets: {supported}"
    )


def load_runtime_model(
    checkpoint: str | Path,
    embedded_config: dict | None = None,
) -> RuntimeModel:
    checkpoint = _stored_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"model checkpoint unavailable: {checkpoint}")
    config = _config_for_checkpoint(checkpoint, embedded_config)
    params = config["model_advanced_params"]
    model_type = params["model_type"]
    model_name = _model_type_name(model_type)
    if model_name != "DFER" and not model_name.startswith("VIDEOMAE"):
        raise ValueError(
            f"unsupported backbone {model_name!r}; only DFER and VideoMAE are supported"
        )
    pretrained = getattr(model_type, "value", None)
    if pretrained:
        pretrained_path = _stored_path(pretrained)
        if not pretrained_path.is_file():
            raise FileNotFoundError(
                f"local pretrained backbone weights unavailable: {pretrained_path}"
            )
    if params.get("concatenate_temporal") or (
        config.get("config") or {}
    ).get("concatenate_quadrants"):
        raise ValueError("temporal or quadrant concatenation is unsupported")

    from cross_space_projection import _build_model

    owner = _build_model(config)
    owner.head.load_state_weights(str(checkpoint))
    owner.head.to("cuda").eval()
    backbone = owner.backbone
    backbone.device = "cuda"
    backbone.model.to("cuda").eval()
    # Prefix heads consume already-extracted clip features. If this run trained
    # end-to-end, the checkpoint has now restored the shared backbone weights.
    owner.head.backbone = None
    return RuntimeModel(config, checkpoint, owner, backbone, owner.head)


def _embedding_reduction(runtime: RuntimeModel, features, torch_module):
    reduction = runtime.config["model_advanced_params"].get("embedding_reduction")
    name = str(getattr(reduction, "name", reduction)).upper()
    value = getattr(reduction, "value", None)
    if reduction is None or name in {"NONE", "EMBEDDING_REDUCTION.NONE"}:
        return features
    if "SPATIAL_MASKED" in name:
        from extract_feature import masked_spatial_mean
        return masked_spatial_mean(features)
    if "ADAPTIVE_POOLING" in name:
        raise ValueError("adaptive 3-D pooling metadata is unavailable in model artifacts")
    if value is None and isinstance(reduction, (tuple, list)):
        value = tuple(reduction)
    if value is None:
        names = {
            "SPATIAL": (2, 3),
            "TEMPORAL": (1,),
            "ALL": (1, 2, 3),
        }
        value = next((axes for key, axes in names.items() if key in name), None)
    if value is None:
        raise ValueError(f"unsupported embedding reduction: {reduction!r}")
    return torch_module.mean(features, dim=tuple(value), keepdim=True)


def extract_clip_features(
    runtime: RuntimeModel,
    chunks: np.ndarray,
    batch_size: int,
):
    import torch
    from custom.dataset import customDataset

    outputs = []
    image_size = int(runtime.backbone.img_size)
    dataset = getattr(runtime.owner, "dataset", None)
    image_mean = getattr(dataset, "image_mean", None)
    image_std = getattr(dataset, "image_std", None)
    for start in range(0, len(chunks), batch_size):
        raw = np.ascontiguousarray(chunks[start:start + batch_size])
        batch, frames, height, width, channels = raw.shape
        tensor = torch.from_numpy(raw).permute(0, 1, 4, 2, 3).reshape(
            batch * frames, channels, height, width
        )
        tensor = customDataset.preprocess_images(
            tensor,
            crop_size=(image_size, image_size),
            image_mean=image_mean,
            image_std=image_std,
        )
        tensor = tensor.reshape(
            batch, frames, channels, image_size, image_size
        ).permute(0, 2, 1, 3, 4)
        with torch.inference_mode():
            features = runtime.backbone.forward_features(tensor)
            features = _embedding_reduction(runtime, features, torch)
        outputs.append(features.detach().cpu())
    return torch.cat(outputs, dim=0)


def _linear_input_dim(linear) -> int:
    if hasattr(linear, "in_features"):
        return int(linear.in_features)
    for module in linear.modules():
        if hasattr(module, "in_features"):
            return int(module.in_features)
    state = linear.state_dict()
    for key, value in state.items():
        if key.endswith("weight") and value.ndim == 2:
            return int(value.shape[1])
    raise ValueError("cannot determine target linear input dimension")


def _load_state(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def load_refined_linear(base_linear, variant: CrossVariant):
    stored = variant.refinement.get("linear_after_pth")
    if not stored:
        raise FileNotFoundError("refined linear checkpoint is absent")
    linear = copy.deepcopy(base_linear)
    linear.load_state_dict(_load_state(_artifact_path(variant, stored)), strict=True)
    return linear.to("cuda").eval()


def load_projector_for_variant(
    variant: CrossVariant,
    stage: int,
    input_dim: int,
    output_dim: int,
):
    import torch

    bundle = variant.result.get("linear_projector") or {}
    config = bundle.get("config") or {}
    kind = bundle.get("kind") or (
        variant.result.get("config_cross_space_projection") or {}
    ).get("interpolation_similarity")
    projector = build_projector_network(
        input_dim,
        output_dim,
        kind,
        activation=config.get("mlp_activation"),
        num_layers=int(config.get("mlp_num_layers", 1)),
        encoder_ratio=int(config.get("encoder_ratio", 4)),
        torch_module=torch,
    )
    projector.load_state_dict(
        _load_state(selected_projector_path(variant, stage)), strict=True
    )
    return projector.to("cuda").eval(), bundle.get("norm_stats")


def predict_linear_value(
    linear,
    embedding,
    model_config: dict,
    reference_head,
) -> float:
    import torch

    value = torch.as_tensor(embedding, dtype=torch.float32, device="cuda")
    with torch.inference_mode():
        logits = linear(value.reshape(1, -1))
    return _prediction_value(logits, reference_head, model_config, torch)


def _embedded_model_config(variant: CrossVariant, side: str) -> dict | None:
    value = variant.result.get(f"{side}_model_config")
    return value if isinstance(value, dict) else None


def cross_model_identities(
    selection: CrossSelection,
) -> tuple[tuple[str, str], tuple[str, str]]:
    variant = selection.variants[0]
    old_config = _config_for_checkpoint(
        variant.old_model_pth,
        _embedded_model_config(variant, "old"),
    )
    new_config = _config_for_checkpoint(
        variant.new_model_pth,
        _embedded_model_config(variant, "new"),
    )
    return model_dataset_identity(old_config), model_dataset_identity(new_config)


def resolve_selection_video(selection: CrossSelection) -> Path:
    variant = selection.variants[0]
    side = "old" if selection.origin == "source" else "new"
    config = _config_for_checkpoint(
        selection.fixed_model_pth,
        _embedded_model_config(variant, side),
    )
    return _video_path(config, selection.row)


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as stream:
        return pickle.load(stream)


def _head_name(value) -> str:
    value = getattr(value, "value", value)
    return str(value).split(".")[-1]


def _sample_key(value) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _sample_row(csv_path: Path, sample_id: str) -> dict | None:
    if not csv_path.is_file():
        raise FileNotFoundError(f"test CSV unavailable: {csv_path}")
    with csv_path.open(newline="") as stream:
        rows = [
            row for row in csv.DictReader(stream, delimiter="\t")
            if _sample_key(row.get("sample_id", "")) == _sample_key(sample_id)
        ]
    if len(rows) > 1:
        raise ValueError(f"sample {sample_id} occurs multiple times in {csv_path}")
    return rows[0] if rows else None


def _stored_path(value, *, relative_to: Path | None = None) -> Path:
    path = Path(value)
    candidates = [path]
    if not path.is_absolute():
        candidates += [Path(__file__).resolve().parent / path]
        if relative_to is not None:
            candidates.append(relative_to / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[-1].resolve()


def _video_path(config: dict, row: dict) -> Path:
    params = config.get("model_advanced_params") or {}
    cfg = config.get("config") or {}
    if row.get("video_path"):
        path = _stored_path(row["video_path"])
    else:
        dataset = params.get("path_dataset") or cfg.get("path_video_dataset")
        if not dataset:
            raise ValueError("model configuration has no video dataset path")
        extension = params.get("video_extension", ".mp4")
        path = _stored_path(dataset) / row.get("subject_name", "") / (
            row["sample_name"] + extension
        )
    if "$" in row.get("sample_name", ""):
        raise ValueError("only original, non-augmented samples are supported")
    if not path.is_file():
        raise FileNotFoundError(f"video unavailable: {path}")
    return path


def resolve_native_experiment(
    experiment_root: str | Path,
    sample_id: str,
    *,
    load_pickle=_load_pickle,
) -> NativeSelection:
    root = Path(experiment_root).resolve()
    result_path = root if root.is_file() else root / "k_fold_results.pkl"
    if result_path.name != "k_fold_results.pkl" or not result_path.is_file():
        raise FileNotFoundError(f"native result unavailable: {result_path}")
    root = result_path.parent
    data = load_pickle(result_path)
    head = _head_name(data["model_advanced_params"]["head"])
    matches = []
    for key, result in data.get("results", {}).items():
        match = re.fullmatch(r"k(\d+)_cross_val_final", key)
        if not match:
            continue
        fold = int(match.group(1))
        test_csv = root / f"train_{head}" / f"k{fold}_cross_val" / "test.csv"
        row = _sample_row(test_csv, sample_id)
        if row is not None:
            matches.append((fold, result, test_csv, row))
    if not matches:
        raise ValueError(f"sample {sample_id} is not in any native test fold")
    if len(matches) != 1:
        raise ValueError(f"sample {sample_id} occurs in multiple test folds")
    fold, result, test_csv, row = matches[0]
    best = result.get("best_model") or {}
    epoch = int(best["best_model_idx"])
    fold_subfold = best.get("fold_sub_fold_idx")
    if fold_subfold is None:
        raise ValueError(f"selected subfold is absent for fold {fold}")
    subfold = int(fold_subfold[1])
    checkpoint = (
        test_csv.parent / f"k{fold}_cross_val_sub_{subfold}"
        / f"best_model_ep_{epoch}.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint unavailable: {checkpoint}")
    return NativeSelection(
        root=root,
        config=data,
        fold=fold,
        subfold=subfold,
        epoch=epoch,
        checkpoint=checkpoint,
        test_csv=test_csv,
        row=row,
        video_path=_video_path(data, row),
    )


def _cross_result_path(root: Path) -> Path:
    if root.is_file():
        return root
    aggregate = sorted(
        path
        for folder in root.iterdir() if folder.is_dir() and folder.name.startswith("aggregated_")
        for path in folder.glob("results*.pkl")
    )
    direct = sorted(root.glob("results*.pkl"))
    candidates = aggregate + direct
    if len(candidates) != 1:
        listed = ", ".join(str(path) for path in candidates) or "none"
        raise ValueError(f"ambiguous cross result pickles: {listed}")
    return candidates[0]


def _cross_records(result_path: Path, aggregate: dict, load_pickle) -> list[dict]:
    if not aggregate.get("aggregated"):
        config = aggregate.get("config_cross_space_projection") or {}
        new_path = config.get("new_model_pth")
        old_path = config.get("old_model_pth")
        return [{
            "new_idx": int(aggregate.get("new_idx", _checkpoint_fold(new_path))),
            "old_idx": int(aggregate.get("old_idx", _checkpoint_fold(old_path))),
            "new_model_pth": new_path,
            "old_model_pth": old_path,
            "result_path": result_path,
            "result": aggregate,
        }]
    metadata = aggregate.get("subtrials") or []
    paths = aggregate.get("subtrial_pkls") or []
    if len(metadata) != len(paths) or not metadata:
        raise ValueError("aggregate has inconsistent subtrial metadata")
    records = []
    for item, stored in zip(metadata, paths):
        path = _stored_path(stored, relative_to=result_path.parent)
        if not path.is_file():
            raise FileNotFoundError(f"subtrial result unavailable: {path}")
        records.append({
            **item,
            "new_idx": int(item["new_idx"]),
            "old_idx": int(item["old_idx"]),
            "result_path": path,
            "result": None,
        })
    return records


def _checkpoint_fold(path) -> int:
    if not path:
        return 0
    match = re.search(r"(?:^|[/\\])k(\d+)_cross_val(?:[/\\]|$)", str(path))
    return int(match.group(1)) if match else 0


def _refinement_blocks(result: dict) -> dict[str, dict]:
    if isinstance(result.get("refinements"), dict):
        return result["refinements"]
    block = result.get("refinement")
    if isinstance(block, dict) and block.get("refine_mode"):
        return {block["refine_mode"]: block}
    return {}


def _available_stages(records: Sequence[dict]) -> list[int]:
    if not records:
        return []
    available = set(STAGE_TO_MODE)
    for record in records:
        modes = set(_refinement_blocks(record["result"]))
        available &= {stage for stage, mode in STAGE_TO_MODE.items() if mode in modes}
    return sorted(available)


def _model_path(record: dict, side: str) -> Path:
    value = record.get(f"{side}_model_pth")
    if not value:
        value = ((record.get("result") or {}).get(
            "config_cross_space_projection") or {}).get(
            f"{side}_model_pth"
        )
    if not value:
        raise ValueError(f"subtrial has no {side}_model_pth")
    return _stored_path(value)


def resolve_cross_experiment(
    experiment_root: str | Path,
    sample_id: str,
    origin: str,
    stage: int = 2,
    *,
    load_pickle=_load_pickle,
) -> CrossSelection:
    if origin not in {"source", "target"}:
        raise ValueError("cross experiments require origin 'source' or 'target'")
    mode = refinement_mode(stage)
    root = Path(experiment_root).resolve()
    result_path = _cross_result_path(root)
    records = _cross_records(result_path, load_pickle(result_path), load_pickle)
    side = "old" if origin == "source" else "new"
    by_index = {}
    for record in records:
        index = int(record[f"{side}_idx"])
        model_path = _model_path(record, side)
        previous = by_index.setdefault(index, model_path)
        if previous != model_path:
            raise ValueError(f"{side} index {index} maps to multiple model checkpoints")
    memberships = []
    for index, model_path in sorted(by_index.items()):
        test_csv = model_path.parents[1] / "test.csv"
        row = _sample_row(test_csv, sample_id)
        if row is not None:
            memberships.append((index, model_path, test_csv, row))
    if not memberships:
        raise ValueError(f"sample {sample_id} is not in any {origin} test fold")
    if len(memberships) != 1:
        raise ValueError(f"sample {sample_id} occurs in multiple {origin} test folds")
    fixed_index, fixed_model, test_csv, row = memberships[0]
    selected = [record for record in records if int(record[f"{side}_idx"]) == fixed_index]
    for record in selected:
        if record["result"] is None:
            record["result"] = load_pickle(record["result_path"])
        result = record["result"]
        config = result.get("config_cross_space_projection") or {}
        kind = (result.get("linear_projector") or {}).get(
            "kind", config.get("interpolation_similarity")
        )
        if kind not in SUPPORTED_PROJECTORS:
            raise ValueError(f"unsupported projector type: {kind!r}")
    available = _available_stages(selected)
    if stage not in available:
        display = ", ".join(map(str, available)) or "none"
        modes = ", ".join(STAGE_TO_MODE[item] for item in available) or "none"
        raise ValueError(
            f"refinement stage {stage} is unavailable; available stages: {display}; "
            f"available modes: {modes}"
        )
    variants = []
    for record in selected:
        block = _refinement_blocks(record["result"])[mode]
        if not block.get("linear_after_pth"):
            raise FileNotFoundError(f"stage {stage} has no refined linear checkpoint")
        if origin == "source":
            projector_key = (
                "projector_after_pth" if stage in {2, 4} else None
            )
            projector_path = (
                block.get(projector_key) if projector_key
                else (record["result"].get("linear_projector") or {}).get("ckpt_path")
            )
            if not projector_path:
                raise FileNotFoundError(f"stage {stage} has no projector checkpoint")
        variant = CrossVariant(
            new_idx=int(record["new_idx"]),
            old_idx=int(record["old_idx"]),
            new_model_pth=_model_path(record, "new"),
            old_model_pth=_model_path(record, "old"),
            result_path=record["result_path"],
            result=record["result"],
            refinement=block,
        )
        _artifact_path(variant, block["linear_after_pth"])
        if origin == "source":
            selected_projector_path(variant, stage)
        variants.append(variant)
    variants.sort(key=lambda item: (item.new_idx, item.old_idx))
    return CrossSelection(
        root=root,
        result_path=result_path,
        origin=origin,
        stage=stage,
        fixed_index=fixed_index,
        fixed_model_pth=fixed_model,
        test_csv=test_csv,
        row=row,
        variants=tuple(variants),
    )


def make_prediction_figure(
    curves: dict[str, Sequence[float]],
    frame_ranges: Sequence[tuple[int, int]],
    ground_truth: float,
    title: str,
    *,
    y_max: float | None = None,
):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6.5))
    x_values = np.arange(len(frame_ranges))
    for label, values in curves.items():
        if len(values) != len(frame_ranges):
            raise ValueError(f"curve {label!r} does not match the number of prefixes")
        axis.plot(x_values, values, marker="o", linewidth=1.8, label=label)
    axis.axhline(
        ground_truth,
        color="black",
        linestyle="--",
        linewidth=1.3,
        label=f"Ground truth = {ground_truth:g}",
    )
    axis.set_title(title)
    axis.set_xlabel("End frame")
    axis.set_ylabel("Pain prediction")
    if y_max is not None:
        axis.set_ylim(0, y_max)
    axis.set_xticks(x_values)
    axis.set_xticklabels(
        [str(end) for _, end in frame_ranges],
        rotation=45, ha="right", rotation_mode="anchor",
    )
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    return figure


def save_prediction_plots(
    curves: CurveSet,
    frame_ranges: Sequence[tuple[int, int]],
    ground_truth: float,
    title: str,
    output_dir: str | Path,
    *,
    stage: int,
    digest: str,
    y_max: float,
    debug_title: bool = False,
) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / f"summary_{digest}.png",
        output_dir / f"summary_full_range_{digest}.png",
        output_dir / f"models_stage_{stage}_{digest}.png",
        output_dir / f"models_stage_{stage}_full_range_{digest}.png",
    )
    full_range_summary = (
        curves.full_range_summary
        if curves.full_range_summary is not None else curves.summary
    )
    for path, data, suffix, axis_max in (
        (paths[0], curves.summary, "Summary", None),
        (paths[1], full_range_summary, f"Summary, full range 0–{y_max:g}", y_max),
        (paths[2], curves.detail, f"Per-model detail, stage {stage}", None),
        (
            paths[3],
            curves.detail,
            f"Per-model detail, stage {stage}, full range 0–{y_max:g}",
            y_max,
        ),
    ):
        figure = make_prediction_figure(
            data,
            frame_ranges,
            ground_truth,
            prediction_figure_title(title, suffix, debug_title=debug_title),
            y_max=axis_max,
        )
        figure.savefig(path, dpi=200)
        plt.close(figure)
    return paths


def save_native_comparison_plot(
    curves: CurveSet,
    frame_ranges: Sequence[tuple[int, int]],
    ground_truth: float,
    title: str,
    output_dir: str | Path,
    *,
    digest: str,
    y_max: float,
) -> Path:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"native_comparison_full_range_{digest}.png"
    figure = make_prediction_figure(
        curves.summary,
        frame_ranges,
        ground_truth,
        title,
        y_max=y_max,
    )
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def cursor_positions(
    frame_ranges: Sequence[tuple[int, int]], frame_count: int,
) -> np.ndarray:
    """Map each source frame to its interpolated cumulative-chunk position."""
    if frame_count < 1:
        return np.empty(0, dtype=np.float64)
    if not frame_ranges:
        raise ValueError("video frames have no cumulative frame ranges")
    endpoints = np.asarray([end for _, end in frame_ranges], dtype=np.float64)
    if len(endpoints) != len(set(endpoints)) or endpoints[-1] != frame_count - 1:
        raise ValueError("cumulative frame ranges must end at the final source frame")
    return np.interp(
        np.arange(frame_count), endpoints, np.arange(len(endpoints), dtype=np.float64)
    )


def _source_frames_and_fps(
    video_path: str | Path,
    *,
    cv2_module=None,
) -> tuple[list[np.ndarray], float]:
    if cv2_module is None:
        import cv2 as cv2_module
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"video cannot be opened: {video_path}")
    try:
        fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        frames = []
        while True:
            available, frame = capture.read()
            if not available:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"video contains no frames: {video_path}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"video has invalid source FPS: {fps}")
    return frames, fps


def _full_range_plot_image(
    curves: dict[str, Sequence[float]],
    frame_ranges: Sequence[tuple[int, int]],
    ground_truth: float,
    title: str,
    y_max: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    import matplotlib.pyplot as plt

    figure = make_prediction_figure(
        curves, frame_ranges, ground_truth, title, y_max=y_max)
    axis = figure.axes[0]
    axis.set_ylim(-0.25, y_max + 0.25)
    figure.set_size_inches(12, 6.5)
    figure.set_dpi(100)
    figure.canvas.draw()
    image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    lower, upper = axis.get_ylim()
    bottom = int(round(image.shape[0] - axis.transData.transform((0, lower))[1]))
    top = int(round(image.shape[0] - axis.transData.transform((0, upper))[1]))
    cursor_start, cursor_end = (
        int(round(axis.transData.transform((index, 0))[0]))
        for index in (0, len(frame_ranges) - 1)
    )
    plt.close(figure)
    return image, (top, bottom, cursor_start, cursor_end)


def _draw_cursor(
    image: np.ndarray,
    cursor_x: float,
    axis_pixels: tuple[int, int, int, int],
    frame_ranges: Sequence[tuple[int, int]],
    *,
    cv2_module,
) -> None:
    top, bottom, start, end = axis_pixels
    maximum = max(1, len(frame_ranges) - 1)
    x = int(round(start + (end - start) * cursor_x / maximum))
    cv2_module.line(image, (x, top), (x, bottom), (255, 0, 0), 2)
    cv2_module.circle(image, (x, bottom), 5, (255, 0, 0), -1)


def _composite_frame(
    plot_rgb: np.ndarray,
    source_frame: np.ndarray,
    source_fps: float,
    playback_fps: float,
    speed: float,
    *,
    cv2_module,
) -> np.ndarray:
    footer_height = 96
    plot_height, plot_width = plot_rgb.shape[:2]
    source_height = plot_height - footer_height
    scale = source_height / source_frame.shape[0]
    source_width = max(1, round(source_frame.shape[1] * scale))
    pane_width = source_width + source_width % 2
    composite = np.zeros((plot_height, plot_width + pane_width, 3), dtype=np.uint8)
    composite[:, :plot_width] = plot_rgb[..., ::-1]
    composite[:source_height, plot_width:plot_width + source_width] = cv2_module.resize(
        source_frame, (source_width, source_height), interpolation=cv2_module.INTER_AREA
    )
    text = (
        f"Source FPS: {source_fps:g}",
        f"Playback FPS: {playback_fps:g}",
        f"Speed: {speed:g}x",
    )
    for index, line in enumerate(text):
        cv2_module.putText(
            composite, line, (plot_width + 18, source_height + 25 + index * 23),
            cv2_module.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
            cv2_module.LINE_AA,
        )
    return composite


def _validated_video_speed(value) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("video speed must be a number") from exc
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("video speed must be positive and finite")
    return speed


def _verify_encoded_video(
    path: Path,
    expected_frames: int,
    expected_fps: float,
    *,
    cv2_module,
) -> None:
    capture = cv2_module.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"encoded video validation failed: cannot open {path}")
    try:
        fps = float(capture.get(cv2_module.CAP_PROP_FPS))
        if not math.isclose(fps, expected_fps, rel_tol=0.01, abs_tol=0.01):
            raise RuntimeError(
                f"encoded video validation failed: expected FPS {expected_fps:g}, "
                f"got {fps:g}"
            )
        frame_count = 0
        while capture.read()[0]:
            frame_count += 1
    finally:
        capture.release()
    if frame_count != expected_frames:
        raise RuntimeError(
            f"encoded video validation failed: expected {expected_frames} frames, "
            f"got {frame_count}"
        )


def _write_synchronized_videos(
    plots: Sequence[tuple[np.ndarray, tuple[int, int, int, int]]],
    frame_ranges: Sequence[tuple[int, int]],
    *,
    source_video: str | Path,
    paths: Sequence[Path],
    speed: float,
) -> tuple[Path, ...]:
    import cv2

    frames, source_fps = _source_frames_and_fps(source_video, cv2_module=cv2)
    positions = cursor_positions(frame_ranges, len(frames))
    playback_fps = source_fps * speed
    first = _composite_frame(
        plots[0][0], frames[0], source_fps, playback_fps, speed, cv2_module=cv2)
    codec = cv2.VideoWriter_fourcc(*"avc1")
    temporary_paths = tuple(
        path.with_name(f".{path.stem}.{uuid4().hex}.tmp.mp4") for path in paths
    )
    writers = []
    try:
        for path in temporary_paths:
            writers.append(
                cv2.VideoWriter(str(path), codec, playback_fps, first.shape[1::-1])
            )
        if not all(writer.isOpened() for writer in writers):
            raise RuntimeError("OpenCV could not open H.264 video writers")
        for source_frame, position in zip(frames, positions):
            for writer, plot in zip(writers, plots):
                plot_image = plot[0].copy()
                _draw_cursor(
                    plot_image, position, plot[1], frame_ranges, cv2_module=cv2)
                writer.write(_composite_frame(
                    plot_image, source_frame, source_fps, playback_fps, speed,
                    cv2_module=cv2))
        for writer in writers:
            writer.release()
        for path in temporary_paths:
            _verify_encoded_video(
                path, len(frames), playback_fps, cv2_module=cv2)
        for temporary, path in zip(temporary_paths, paths):
            temporary.replace(path)
    except Exception:
        for writer in writers:
            try:
                writer.release()
            except Exception:
                pass
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        raise
    return tuple(paths)


def save_prediction_videos(
    curves: CurveSet,
    frame_ranges: Sequence[tuple[int, int]],
    ground_truth: float,
    title: str,
    *,
    source_video: str | Path,
    output_dir: str | Path,
    stage: int,
    digest: str,
    y_max: float,
    speed: float,
    debug_title: bool = False,
) -> tuple[Path, Path]:
    """Write synchronized summary and per-model H.264 composites atomically."""
    speed = _validated_video_speed(speed)
    full_range_summary = (
        curves.full_range_summary
        if curves.full_range_summary is not None else curves.summary
    )
    plots = (
        _full_range_plot_image(
            full_range_summary, frame_ranges, ground_truth,
            prediction_figure_title(
                title, f"Summary, full range 0–{y_max:g}",
                debug_title=debug_title), y_max),
        _full_range_plot_image(
            curves.detail, frame_ranges, ground_truth,
            prediction_figure_title(
                title,
                f"Per-model detail, stage {stage}, full range 0–{y_max:g}",
                debug_title=debug_title,
            ),
            y_max,
        ),
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        output_dir / f"summary_full_range_video_{digest}.mp4",
        output_dir / f"models_stage_{stage}_full_range_video_{digest}.mp4",
    )
    return _write_synchronized_videos(
        plots,
        frame_ranges,
        source_video=source_video,
        paths=paths,
        speed=speed,
    )


def save_native_comparison_video(
    curves: CurveSet,
    frame_ranges: Sequence[tuple[int, int]],
    ground_truth: float,
    title: str,
    *,
    source_video: str | Path,
    output_dir: str | Path,
    digest: str,
    y_max: float,
    speed: float,
) -> Path:
    speed = _validated_video_speed(speed)
    plot = _full_range_plot_image(
        curves.summary, frame_ranges, ground_truth, title, y_max)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"native_comparison_full_range_video_{digest}.mp4"
    return _write_synchronized_videos(
        (plot,),
        frame_ranges,
        source_video=source_video,
        paths=(path,),
        speed=speed,
    )[0]


def save_script_snapshot(
    output_dir: str | Path,
    argv: Sequence[str],
    *,
    digest: str,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["python3", Path(__file__).name, *argv])
    snapshot = output_dir / f"plot_cumulative_predictions_{digest}.txt"
    snapshot.write_text(f"{command}\n", encoding="utf-8")
    return snapshot


def _positive_finite_speed(value: str) -> float:
    try:
        return _validated_video_speed(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot cumulative pain predictions from consecutive 16-frame video prefixes."
        )
    )
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("sample_id")
    parser.add_argument(
        "--compare-native-root", type=Path,
        help="native experiment for the other model on the sample's dataset",
    )
    parser.add_argument(
        "--target-native-root", type=Path,
        help="native target-model experiment trained on the source dataset",
    )
    parser.add_argument("--origin", choices=("source", "target"))
    parser.add_argument("--refinement-stage", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--debug_title", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("chunk_prediction_plots"))
    parser.add_argument("--backbone-batch-size", type=int, default=8)
    parser.add_argument(
        "--video",
        dest="video_speed",
        nargs="?",
        const=1.0,
        type=_positive_finite_speed,
        default=None,
        metavar="SPEED",
    )
    return parser


def _experiment_kind(path: Path) -> str:
    if (path.is_file() and path.name == "k_fold_results.pkl") or (
        path.is_dir() and (path / "k_fold_results.pkl").is_file()
    ):
        return "native"
    return "cross"


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    native_inference=None,
    cross_inference=None,
) -> tuple[Path, ...]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(raw_argv)
    if args.backbone_batch_size < 1:
        raise ValueError("--backbone-batch-size must be positive")
    video_enabled = args.video_speed is not None
    kind = _experiment_kind(args.experiment_root)
    comparison_enabled = args.compare_native_root is not None
    native_comparison_enabled = comparison_enabled and kind == "native"
    if comparison_enabled and _experiment_kind(args.compare_native_root) != "native":
        if kind == "native":
            raise ValueError("--compare-native-root requires two native experiments")
        raise ValueError("--compare-native-root requires a native experiment")
    if args.target_native_root is not None and (
        kind != "cross" or args.origin != "source"
    ):
        raise ValueError("--target-native-root requires a source-origin cross experiment")
    if comparison_enabled and args.target_native_root is not None:
        raise ValueError("--compare-native-root and --target-native-root are exclusive")
    if kind == "native":
        if args.origin is not None:
            raise ValueError("--origin is only valid for cross-projection experiments")
        require_cuda()
        selection = resolve_native_experiment(args.experiment_root, args.sample_id)
        y_max = dataset_axis_max(selection.config, selection.test_csv)
        inference = native_inference or infer_native
        if comparison_enabled:
            comparison_selection = resolve_native_experiment(
                args.compare_native_root, args.sample_id
            )
            validate_native_comparison_selections(selection, comparison_selection)
            comparison_y_max = dataset_axis_max(
                comparison_selection.config, comparison_selection.test_csv
            )
            result = inference(selection, args.backbone_batch_size)
            comparison_result = inference(
                comparison_selection, args.backbone_batch_size
            )
            result, y_max = combine_native_results(
                result, comparison_result, y_max, comparison_y_max
            )
        else:
            result = inference(selection, args.backbone_batch_size)
        origin = "native"
        source_video = selection.video_path if video_enabled else None
    else:
        if args.origin is None:
            raise ValueError("--origin is required for cross-projection experiments")
        require_cuda()
        selection = resolve_cross_experiment(
            args.experiment_root,
            args.sample_id,
            args.origin,
            args.refinement_stage,
        )
        fixed_side = "old" if args.origin == "source" else "new"
        fixed_config = _config_for_checkpoint(
            selection.fixed_model_pth,
            _embedded_model_config(selection.variants[0], fixed_side),
        )
        y_max = dataset_axis_max(fixed_config, selection.test_csv)
        inference = cross_inference or infer_cross
        result = inference(selection, args.backbone_batch_size)
        cross_native_root = args.compare_native_root or args.target_native_root
        if cross_native_root is not None:
            missing_side = "target" if args.origin == "source" else "source"
            expected_model = (
                result.target_model if args.origin == "source" else result.source_model
            )
            native_selection = resolve_native_experiment(
                cross_native_root, args.sample_id
            )
            validate_native_comparison_selections(selection, native_selection)
            native_dataset, native_model = model_dataset_identity(
                native_selection.config
            )
            if (native_dataset, native_model) != (
                result.origin_dataset, expected_model
            ):
                raise ValueError(
                    f"{missing_side} native experiment must use the {args.origin} "
                    f"dataset and {missing_side} model architecture"
                )
            native_result = (native_inference or infer_native)(
                native_selection, args.backbone_batch_size
            )
            if (native_result.origin_dataset, native_result.origin_model) != (
                native_dataset, native_model
            ):
                raise ValueError(
                    f"{missing_side} native inference has the wrong model identity"
                )
            if native_result.frame_count != result.frame_count:
                raise ValueError(
                    f"{missing_side} native inference has a different frame count"
                )
            if not math.isclose(native_result.ground_truth, result.ground_truth):
                raise ValueError(
                    f"{missing_side} native inference has a different ground truth"
                )
            if len(native_result.curves.summary) != 1:
                raise ValueError(
                    f"{missing_side} native inference must produce exactly one curve"
                )
            native_curve = next(iter(native_result.curves.summary.values()))
            if len(native_curve) != len(next(iter(result.curves.summary.values()))):
                raise ValueError(
                    f"{missing_side} native inference has a different prefix count"
                )
            comparison_summary = {
                **result.curves.summary,
                f"{missing_side.title()} model — {native_model} "
                f"({native_dataset})": native_curve,
            }
            result = replace(result, curves=replace(
                result.curves,
                summary=(
                    comparison_summary if comparison_enabled else result.curves.summary
                ),
                full_range_summary=comparison_summary,
            ))
        origin = args.origin
        source_video = resolve_selection_video(selection) if video_enabled else None
    projector_kind = None if kind == "native" else (
        (selection.variants[0].result.get("linear_projector") or {}).get("kind")
        or (selection.variants[0].result.get(
            "config_cross_space_projection") or {}).get("interpolation_similarity")
    )
    digest = invocation_digest(raw_argv)
    output_dir = invocation_output_dir(
        args.output_root,
        sample_id=args.sample_id,
        origin=origin,
        stage=None if kind == "native" else args.refinement_stage,
        origin_dataset=result.origin_dataset,
        origin_model=result.origin_model,
        source_dataset=result.source_dataset,
        source_model=result.source_model,
        target_dataset=result.target_dataset,
        target_model=result.target_model,
        projector_kind=projector_kind,
    )
    ranges = cumulative_frame_ranges(result.frame_count)
    if native_comparison_enabled:
        title = (
            native_comparison_title(args.sample_id, result, y_max)
            if args.debug_title else presentation_title(
                args.sample_id, origin, result, y_max, stage=None
            )
        )
        paths = (save_native_comparison_plot(
            result.curves,
            ranges,
            result.ground_truth,
            title,
            output_dir,
            digest=digest,
            y_max=y_max,
        ),)
    else:
        title = (
            prediction_title(args.sample_id, origin, result)
            if args.debug_title else presentation_title(
                args.sample_id, origin, result, y_max,
                stage=None if origin == "native" else args.refinement_stage,
            )
        )
        paths = save_prediction_plots(
            result.curves,
            ranges,
            result.ground_truth,
            title,
            output_dir,
            stage=args.refinement_stage,
            digest=digest,
            y_max=y_max,
            debug_title=args.debug_title,
        )
    if video_enabled:
        if native_comparison_enabled:
            paths += (save_native_comparison_video(
                result.curves,
                ranges,
                result.ground_truth,
                title,
                source_video=source_video,
                output_dir=output_dir,
                digest=digest,
                y_max=y_max,
                speed=args.video_speed,
            ),)
        else:
            paths += save_prediction_videos(
                result.curves,
                ranges,
                result.ground_truth,
                title,
                source_video=source_video,
                output_dir=output_dir,
                stage=args.refinement_stage,
                digest=digest,
                y_max=y_max,
                speed=args.video_speed,
                debug_title=args.debug_title,
            )
    snapshot = save_script_snapshot(output_dir, raw_argv, digest=digest)
    for path in paths:
        print(f"Saved {path}")
    print(f"Saved {snapshot}")
    return paths


def infer_native(selection: NativeSelection, backbone_batch_size: int) -> InferenceResult:
    dataset, model = model_dataset_identity(selection.config)
    runtime = load_runtime_model(selection.checkpoint, selection.config)
    frames = decode_video(selection.video_path)
    features = extract_clip_features(
        runtime, chunk_frames(frames), backbone_batch_size
    )
    predictions, _ = run_head_prefixes(runtime.head, features, runtime.config)
    curves = {f"Native {model} ({dataset})": predictions}
    return InferenceResult(
        curves=CurveSet(summary=curves, detail=curves),
        frame_count=len(frames),
        ground_truth=float(selection.row["class_id"]),
        metadata=(
            f"native fold {selection.fold}, subfold {selection.subfold}, "
            f"epoch {selection.epoch}"
        ),
        origin_dataset=dataset,
        origin_model=model,
    )


def infer_cross(selection: CrossSelection, backbone_batch_size: int) -> InferenceResult:
    import torch

    first = selection.variants[0]
    (source_dataset, source_model), (target_dataset, target_model) = (
        cross_model_identities(selection)
    )
    source_label = f"{source_model} ({source_dataset})"
    target_label = f"{target_model} ({target_dataset})"
    fixed_side = "old" if selection.origin == "source" else "new"
    fixed_config = _embedded_model_config(first, fixed_side)
    runtime = load_runtime_model(selection.fixed_model_pth, fixed_config)
    frames = decode_video(resolve_selection_video(selection))
    features = extract_clip_features(
        runtime, chunk_frames(frames), backbone_batch_size
    )
    native_predictions, prefix_embeddings = run_head_prefixes(
        runtime.head, features, runtime.config
    )

    if selection.origin == "target":
        refined = []
        for variant in selection.variants:
            refined.append((
                variant.old_idx,
                load_refined_linear(runtime.head.linear, variant),
                runtime.config,
            ))
        refined_predictions = target_refined_predictions(
            prefix_embeddings,
            refined,
            predict_linear=lambda linear, embedding, config: predict_linear_value(
                linear, embedding, config, runtime.head
            ),
        )
        curves = target_curve_set(
            native_predictions,
            refined_predictions,
            stage=selection.stage,
            source_label=source_label,
            target_label=target_label,
        )
        metadata = (
            f"target fold/new_idx {selection.fixed_index}, stage {selection.stage}, "
            f"source old_idx variants "
            f"{[variant.old_idx for variant in selection.variants]}"
        )
    else:
        plain_predictions = {}
        refined_predictions = {}
        source_dim = int(np.asarray(prefix_embeddings[0]).size)
        for variant in selection.variants:
            target_config = _embedded_model_config(variant, "new")
            target_runtime = load_runtime_model(
                variant.new_model_pth, target_config
            )
            target_dim = _linear_input_dim(target_runtime.head.linear)
            plain_projector, norm_stats = load_projector_for_variant(
                variant, 1, source_dim, target_dim
            )
            if selection.stage == 1:
                selected_projector = plain_projector
            else:
                selected_projector, _ = load_projector_for_variant(
                    variant, selection.stage, source_dim, target_dim
                )
            refined_linear = load_refined_linear(
                target_runtime.head.linear, variant
            )
            plain_predictions[variant.new_idx] = [
                predict_linear_value(
                    target_runtime.head.linear,
                    apply_projector(
                        plain_projector, embedding, norm_stats,
                        torch_module=torch,
                    ),
                    target_runtime.config,
                    target_runtime.head,
                )
                for embedding in prefix_embeddings
            ]
            refined_predictions[variant.new_idx] = [
                predict_linear_value(
                    refined_linear,
                    apply_projector(
                        selected_projector, embedding, norm_stats,
                        torch_module=torch,
                    ),
                    target_runtime.config,
                    target_runtime.head,
                )
                for embedding in prefix_embeddings
            ]
            del target_runtime, plain_projector, selected_projector, refined_linear
            torch.cuda.empty_cache()
        curves = source_curve_set(
            native_predictions,
            plain_predictions,
            refined_predictions,
            stage=selection.stage,
            source_label=source_label,
            target_label=target_label,
        )
        metadata = (
            f"source fold/old_idx {selection.fixed_index}, stage {selection.stage}, "
            f"target new_idx variants "
            f"{[variant.new_idx for variant in selection.variants]}"
        )
    return InferenceResult(
        curves=curves,
        frame_count=len(frames),
        ground_truth=float(selection.row["class_id"]),
        metadata=metadata,
        origin_dataset=(
            source_dataset if selection.origin == "source" else target_dataset
        ),
        origin_model=(source_model if selection.origin == "source" else target_model),
        source_dataset=source_dataset,
        source_model=source_model,
        target_dataset=target_dataset,
        target_model=target_model,
    )


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

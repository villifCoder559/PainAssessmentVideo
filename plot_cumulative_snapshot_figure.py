#!/usr/bin/env python3
"""Paper figure: a cumulative-prediction plot with snapshot frames below it.

Takes a full-range plot and the snapshot folder written by
plot_cumulative_predictions.py (--video --snapshots), drops the plot title, and
places the snapshot faces under the plot with an arrow from each x-axis tick.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence

import numpy as np

CHUNK_SIZE = 16
CM = 1 / 2.54
INK, ARROW = "#1a1a19", "#6b6a63"


def axes_geometry(gray: np.ndarray) -> tuple[int, int, int, list[int]]:
    """Return (top spine row, bottom spine row, left spine col, tick columns)."""
    height, width = gray.shape
    dark = gray < 80
    rows = np.flatnonzero(dark.sum(1) > 0.6 * width)
    cols = np.flatnonzero(dark.sum(0) > 0.6 * height)
    if len(rows) < 2 or not len(cols):
        raise ValueError("plot axes not found")
    top, bottom, left = int(rows[0]), int(rows[-1]), int(cols[0])
    below = (gray[bottom + 3:bottom + 8] < 100).all(0)
    below[:left + 1] = False
    marks = np.flatnonzero(below)
    groups = np.split(marks, np.flatnonzero(np.diff(marks) > 1) + 1)
    return top, bottom, left, [int(group.mean()) for group in groups if len(group)]


def face_crop(snapshot: np.ndarray) -> np.ndarray:
    """Cut the source-frame panel from a plot|frame composite snapshot."""
    white = (snapshot[0] > 245).all(1)
    boundary = int(np.argmin(white))
    if boundary == 0:
        raise ValueError("snapshot has no white plot panel on the left")
    return snapshot[:, boundary:]


def snapshot_frames(folder: Path) -> list[int]:
    frames = sorted(
        int(match.group(1)) for path in folder.glob("frame_*.png")
        if (match := re.fullmatch(r"frame_(\d+)\.png", path.name))
    )
    if not frames:
        raise FileNotFoundError(f"no frame_*.png snapshots in {folder}")
    return frames


def make_figure(
    plot_path: Path,
    snapshot_dir: Path,
    *,
    width_cm: float = 14.0,
    face_height_cm: float = 2.3,
    title_margin: int = 18,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    from PIL import Image

    plot = np.asarray(Image.open(plot_path).convert("RGB"))
    top, bottom, left, ticks = axes_geometry(plot.mean(2))
    crop_top = max(0, top - title_margin)
    plot = plot[crop_top:].copy()
    # blank title remnants above the axes; keep the top y-tick label on the left
    plot[:top - crop_top - 1, left + 1:] = 255
    frames = snapshot_frames(snapshot_dir)
    faces = [
        face_crop(np.asarray(Image.open(snapshot_dir / f"frame_{frame:05d}.png")
                             .convert("RGB")))
        for frame in frames
    ]
    for frame in frames:
        if frame // CHUNK_SIZE >= len(ticks):
            raise ValueError(f"frame {frame} is beyond the last plotted chunk")

    plot_h = width_cm * plot.shape[0] / plot.shape[1]
    face_w = face_height_cm * faces[0].shape[1] / faces[0].shape[0]
    gap_h, label_h = 0.75, 0.45
    height_cm = plot_h + gap_h + face_height_cm + label_h
    figure = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    plot_ax = figure.add_axes([0, 1 - plot_h / height_cm, 1, plot_h / height_cm])
    plot_ax.imshow(plot)
    plot_ax.set_axis_off()

    slot = width_cm / len(frames)
    for index, (frame, face) in enumerate(zip(frames, faces)):
        left = (index + 0.5) * slot - face_w / 2
        ax = figure.add_axes([
            left / width_cm, label_h / height_cm,
            face_w / width_cm, face_height_cm / height_cm,
        ])
        ax.imshow(face)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(ARROW)
            spine.set_linewidth(0.5)
        ax.set_xlabel(f"frame {frame}", fontsize=7, color=INK, labelpad=1.5)
        figure.add_artist(ConnectionPatch(
            xyA=(ticks[frame // CHUNK_SIZE], bottom - crop_top + 60),
            coordsA=plot_ax.transData,
            xyB=(0.5, 1.0), coordsB=ax.transAxes,
            arrowstyle="-|>", mutation_scale=6, lw=0.6, color=ARROW,
            shrinkA=0, shrinkB=1,
        ))
    return figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("plot", type=Path, help="*_full_range_<digest>.png")
    parser.add_argument("snapshot_dir", type=Path, help="*_video_<digest>_snapshots")
    parser.add_argument("output", type=Path, nargs="+", help="output .png/.pdf paths")
    parser.add_argument("--width-cm", type=float, default=14.0)
    parser.add_argument("--face-height-cm", type=float, default=2.3)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    figure = make_figure(
        args.plot, args.snapshot_dir,
        width_cm=args.width_cm, face_height_cm=args.face_height_cm,
    )
    for path in args.output:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=args.dpi)
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

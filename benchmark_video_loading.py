"""
benchmark_video_loading.py — Compare cv2 vs PyAV frame-decoding speed.

Usage:
  python3 benchmark_video_loading.py --root_folder <path> --n <int> [--seed <int>]
"""

import argparse
import glob
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Tuple
import tqdm
import cv2
import av
import numpy as np
import psutil


# ---------------------------------------------------------------------------
# Decoder functions
# ---------------------------------------------------------------------------

def read_all_frames_cv2(path: str) -> Tuple[List[np.ndarray], int]:
  """
  Read all frames from a video file using OpenCV (cv2).

  Args:
    path: Absolute or relative path to an .mp4 video file.

  Returns:
    Tuple of (frames, n_frames) where frames is a list of BGR numpy arrays
    and n_frames is the total count of decoded frames.
  """
  cap = cv2.VideoCapture(path)
  frames = []
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  cap.release()
  return frames, len(frames)


def read_all_frames_pyav(path: str) -> Tuple[List[np.ndarray], int]:
  """
  Read all frames from a video file using PyAV (av).

  Args:
    path: Absolute or relative path to an .mp4 video file.

  Returns:
    Tuple of (frames, n_frames) where frames is a list of BGR numpy arrays
    (converted from YUV via bgr24) and n_frames is the total count of decoded frames.
  """
  frames = []
  with av.open(path) as container:
    for frame in container.decode(video=0):
      frames.append(frame.to_ndarray(format="bgr24"))
  return frames, len(frames)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VideoResult:
  """
  Timing and frame count for a single video.

  Attributes:
    path:     Path to the video file.
    n_frames: Number of frames decoded.
    elapsed:  Wall time in seconds.
    fps:      Decoded frames per second.
  """
  path: str
  n_frames: int
  elapsed: float
  fps: float


@dataclass
class BenchmarkResult:
  """
  Aggregate benchmark result for one library over N videos.

  Attributes:
    library:        Name of the decoding library.
    videos:         Per-video results.
    total_time:     Total wall time across all N videos (s).
    mean_time:      Mean wall time per video (s).
    std_time:       Std-dev of per-video wall times (s).
    total_frames:   Total frames decoded.
    throughput_fps: total_frames / total_time.
    peak_rss_mb:    Peak RSS memory increase (MB) over the baseline.
  """
  library: str
  videos: List[VideoResult] = field(default_factory=list)
  total_time: float = 0.0
  mean_time: float = 0.0
  std_time: float = 0.0
  total_frames: int = 0
  throughput_fps: float = 0.0
  peak_rss_mb: float = 0.0


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark(
  library_name: str,
  warm_up_path: str,
  paths: List[str],
  reader_fn: Callable[[str], Tuple[List[np.ndarray], int]],
) -> BenchmarkResult:
  """
  Run a decoding benchmark for a single library over a list of video paths.

  A warm-up video is decoded first (result not recorded) to prime filesystem
  caches and avoid cold-start bias.

  Args:
    library_name: Human-readable name of the library (e.g. "cv2", "PyAV").
    warm_up_path: Path to the video used for warm-up (not included in metrics).
    paths:        List of N video paths to benchmark.
    reader_fn:    Function that decodes a video and returns (frames, n_frames).

  Returns:
    BenchmarkResult populated with per-video and aggregate metrics.
  """
  result = BenchmarkResult(library=library_name)
  proc = psutil.Process()

  # Warm-up run — not timed
  print(f"  [{library_name}] warming up on {Path(warm_up_path).name} ...")
  reader_fn(warm_up_path)

  rss_baseline = proc.memory_info().rss
  rss_peak = rss_baseline

  for i, path in tqdm.tqdm(enumerate(paths), total=len(paths), desc=f"Decoding with {library_name}"):
    t0 = time.perf_counter()
    frames, n_frames = reader_fn(path)
    elapsed = time.perf_counter() - t0

    # Sample RSS while frames are still in memory
    rss_now = proc.memory_info().rss
    if rss_now > rss_peak:
      rss_peak = rss_now

    # Free frames immediately to bound memory between iterations
    del frames

    fps = n_frames / elapsed if elapsed > 0 else 0.0
    result.videos.append(VideoResult(
      path=path,
      n_frames=n_frames,
      elapsed=elapsed,
      fps=fps,
    ))
    # print(f"  [{library_name}] {i + 1}/{len(paths)} — {Path(path).name}: "
    #       f"{n_frames} frames in {elapsed:.3f}s ({fps:.1f} fps)")

  times = [v.elapsed for v in result.videos]
  result.total_time = sum(times)
  result.mean_time = statistics.mean(times)
  result.std_time = statistics.stdev(times) if len(times) > 1 else 0.0
  result.total_frames = sum(v.n_frames for v in result.videos)
  result.throughput_fps = result.total_frames / result.total_time if result.total_time > 0 else 0.0
  result.peak_rss_mb = (rss_peak - rss_baseline) / (1024 ** 2)

  return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(cv2_res: BenchmarkResult, pyav_res: BenchmarkResult) -> None:
  """
  Print a side-by-side summary table and per-video breakdown for both libraries.

  Args:
    cv2_res:  BenchmarkResult for the cv2 library.
    pyav_res: BenchmarkResult for the PyAV library.
  """
  col = 22
  print("\n" + "=" * 70)
  print("SUMMARY")
  print("=" * 70)
  header = f"{'Metric':<30} {'cv2':>{col}} {'PyAV':>{col}}"
  print(header)
  print("-" * 70)

  def row(label, cv2_val, pyav_val, fmt=".3f"):
    print(f"{label:<30} {cv2_val:>{col}{fmt}} {pyav_val:>{col}{fmt}}")

  row("Total wall time (s)",      cv2_res.total_time,     pyav_res.total_time)
  row("Mean time per video (s)",  cv2_res.mean_time,      pyav_res.mean_time)
  row("Std-dev per video (s)",    cv2_res.std_time,       pyav_res.std_time)
  row("Total frames decoded",     cv2_res.total_frames,   pyav_res.total_frames,  fmt="d")
  row("Throughput (fps)",         cv2_res.throughput_fps, pyav_res.throughput_fps, fmt=".1f")
  row("Peak RSS delta (MB)",      cv2_res.peak_rss_mb,    pyav_res.peak_rss_mb)
  print("=" * 70)

  # for res in (cv2_res, pyav_res):
  #   print(f"\nPer-video breakdown [{res.library}]:")
  #   print(f"  {'#':<5} {'frames':>7} {'time (s)':>10} {'fps':>8}  path")
  #   print(f"  {'-'*5} {'-'*7} {'-'*10} {'-'*8}  {'-'*30}")
  #   for i, v in enumerate(res.videos):
  #     name = Path(v.path).name
  #     print(f"  {i+1:<5} {v.n_frames:>7} {v.elapsed:>10.3f} {v.fps:>8.1f}  {name}")


def save_json(cv2_res: BenchmarkResult, pyav_res: BenchmarkResult, path: str) -> None:
  """
  Save benchmark results to a JSON file.

  Args:
    cv2_res:  BenchmarkResult for cv2.
    pyav_res: BenchmarkResult for PyAV.
    path:     Output JSON file path.
  """
  def serialise(r: BenchmarkResult) -> dict:
    return {
      "library": r.library,
      "total_time_s": r.total_time,
      "mean_time_s": r.mean_time,
      "std_time_s": r.std_time,
      "total_frames": r.total_frames,
      "throughput_fps": r.throughput_fps,
      "peak_rss_delta_mb": r.peak_rss_mb,
      "videos": [
        {"path": v.path, "n_frames": v.n_frames, "elapsed_s": v.elapsed, "fps": v.fps}
        for v in r.videos
      ],
    }

  with open(path, "w") as f:
    json.dump({"cv2": serialise(cv2_res), "pyav": serialise(pyav_res)}, f, indent=2)
  print(f"\nResults saved to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
  """
  Parse CLI arguments, discover videos, run the benchmark, and report results.
  """
  parser = argparse.ArgumentParser(
    description="Benchmark cv2 vs PyAV frame-decoding speed on N randomly sampled videos."
  )
  parser.add_argument("--root_folder", required=True,
                      help="Root directory; all .mp4 files under it are collected recursively.")
  parser.add_argument("--n", type=int, required=True,
                      help="Number of videos to benchmark (randomly sampled without replacement).")
  parser.add_argument("--seed", type=int, default=42,
                      help="RNG seed for reproducibility (default: 42).")
  parser.add_argument("--save_json", type=str, default="benchmark_results.json",
                      help="Path for JSON output (default: benchmark_results.json). Pass '' to skip.")
  args = parser.parse_args()

  # 1. Discover videos
  pattern = str(Path(args.root_folder) / "**" / "*.mp4")
  all_paths = sorted(glob.glob(pattern, recursive=True))
  print(f"Found {len(all_paths)} .mp4 files under '{args.root_folder}'.")

  needed = args.n + 1  # N benchmark videos + 1 warm-up
  if len(all_paths) < needed:
    raise ValueError(
      f"Need at least {needed} videos (N={args.n} + 1 warm-up), "
      f"but only {len(all_paths)} found."
    )

  # 2. Sample
  rng = random.Random(args.seed)
  sampled = rng.sample(all_paths, needed)
  warm_up_path = sampled[0]
  bench_paths = sampled[1:]
  print(f"Warm-up video : {Path(warm_up_path).name}")
  print(f"Benchmark videos: {args.n} (seed={args.seed})\n")

  # 3. Run benchmarks
  print("--- cv2 ---")
  cv2_res = benchmark("cv2", warm_up_path, bench_paths, read_all_frames_cv2)

  print("\n--- PyAV ---")
  pyav_res = benchmark("PyAV", warm_up_path, bench_paths, read_all_frames_pyav)

  # 4. Report
  print_summary(cv2_res, pyav_res)

  if args.save_json:
    save_json(cv2_res, pyav_res, args.save_json)


if __name__ == "__main__":
  main()

import cv2
import os
import sys
import numpy as np
import csv
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool, cpu_count
import logging
from typing import Optional, Dict, List, Tuple
import time
from functools import wraps
from scipy import stats  # <--- [CHANGE 1] Added import for statistical tests

# Configure logging with detailed timing info
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================
# TIMING DECORATOR
# ============================================
def timed_function(func):
    """Decorator to log execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"⏱️  {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper


# ============================================
# VIDEO ANALYSIS WITH OPTIMIZATIONS
# ============================================
def analyze_video(
    video_path: str,
    frame_skip: int = 1,
    resize_dim: int = 256,
    of_dim: int = 256,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Analyze video with temporal dynamics metrics.
    """
    total_start = time.time()
    video_name = os.path.basename(video_path)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return None
    
    ret, prev = cap.read()
    if not ret:
        logger.warning(f"Cannot read first frame: {video_path}")
        cap.release()
        return None
    
    # Convert and resize
    prev_small = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_small = cv2.resize(prev_small, (resize_dim, resize_dim)).astype(np.float32) / 255.0
    
    # For optical flow
    prev_flow = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_flow = cv2.resize(prev_flow, (of_dim, of_dim)).astype(np.float32) / 255.0
    
    frame_count = 0
    frame_diffs = []
    cosine_sims = []
    flow_mags = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % frame_skip != 0:
                continue
            
            # Small resolution for similarity metrics
            frame_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_small = cv2.resize(frame_small, (resize_dim, resize_dim)).astype(np.float32) / 255.0
            
            # Frame difference (L1 distance)
            # Store value for mean calculation later
            diff = np.mean(np.abs(prev_small - frame_small))
            frame_diffs.append(diff)
            
            # Cosine similarity
            f1 = prev_small.flatten().reshape(1, -1)
            f2 = frame_small.flatten().reshape(1, -1)
            cosine_sims.append(cosine_similarity(f1, f2)[0, 0])
            
            # Optical flow
            frame_flow = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_flow = cv2.resize(frame_flow, (of_dim, of_dim)).astype(np.float32) / 255.0
            
            flow = cv2.calcOpticalFlowFarneback(
                prev_flow, frame_flow, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mags.append(np.mean(mag))
            
            prev_small = frame_small
            prev_flow = frame_flow
    
    finally:
        cap.release()
    
    if len(frame_diffs) == 0:
        logger.warning(f"No frames processed: {video_path}")
        return None
    
    total_time = time.time() - total_start
    
    return {
        "frame_diff": float(np.mean(frame_diffs)),
        "frame_diff_list": frame_diffs,  # <--- [CHANGE 2] Return the raw list
        "cosine_sim": float(np.mean(cosine_sims)),
        "optical_flow": float(np.mean(flow_mags)),
        "frame_count": frame_count,
        "process_time": total_time
    }


# ============================================
# FIND VIDEOS
# ============================================
def find_videos(root: str) -> List[str]:
    """Recursively find all MP4 files in directory tree."""
    videos = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".mp4") and '$' not in f:
                full_path = os.path.join(dirpath, f)
                if os.path.isfile(full_path):
                    videos.append(full_path)
    return sorted(videos)


# ============================================
# PROCESS VIDEO PAIR
# ============================================
def process_pair(args: Tuple[str, str, str, int, int, int]) -> Optional[Dict]:
    """Compare metrics between original and permuted video."""
    orig_video, original_root, permuted_root, frame_skip, resize_dim, of_dim = args
    
    rel_path = os.path.relpath(orig_video, original_root)
    perm_video = os.path.join(permuted_root, rel_path)
    
    if not os.path.exists(perm_video):
        logger.warning(f"Permuted video not found: {perm_video}")
        return None
    
    # Analyze original
    orig_metrics = analyze_video(
        orig_video,
        frame_skip=frame_skip,
        resize_dim=resize_dim,
        of_dim=of_dim
    )
    
    if orig_metrics is None: return None
    
    # Analyze permuted
    perm_metrics = analyze_video(
        perm_video,
        frame_skip=frame_skip,
        resize_dim=resize_dim,
        of_dim=of_dim
    )
    
    if perm_metrics is None: return None
    
    # --- [CHANGE 3] PERFORM PER-VIDEO T-TEST ---
    # We compare the list of transitions (frame diffs) from the Original video
    # vs the list of transitions from the Permuted video.
    # We use ttest_ind (independent) because the permutation destroys the 1-to-1 mapping of transitions.
    # We use equal_var=False (Welch's t-test) because permuted variance is usually much higher.
    orig_diffs = orig_metrics["frame_diff_list"]
    perm_diffs = perm_metrics["frame_diff_list"]
    if len(orig_diffs) < 2 or len(perm_diffs) < 2:
        p_val = np.nan
    else:
        _, p_val = stats.ttest_ind(
            orig_diffs,
            perm_diffs,
            equal_var=False,
            nan_policy='omit'
        )
        if not np.isfinite(p_val):
            p_val = np.nan
    
    return {
        "video": rel_path,
        "orig_frame_diff": orig_metrics.get("frame_diff", np.nan),
        "perm_frame_diff": perm_metrics.get("frame_diff", np.nan),
        "orig_cosine": orig_metrics.get("cosine_sim", np.nan),
        "perm_cosine": perm_metrics.get("cosine_sim", np.nan),
        "orig_flow": orig_metrics.get("optical_flow", np.nan),
        "perm_flow": perm_metrics.get("optical_flow", np.nan),
        "frame_count": orig_metrics.get("frame_count", np.nan),
        "process_time_orig": orig_metrics.get("process_time", np.nan),
        "process_time_perm": perm_metrics.get("process_time", np.nan),
        "transition_p_val": p_val  # <--- Store the p-value
    }


# ============================================
# SAVE CSV
# ============================================
def save_csv(results: List[Dict], output_file: str) -> None:
    fieldnames = [
        "video", 
        "frame_count",
        "orig_frame_diff", "perm_frame_diff",
        "orig_cosine", "perm_cosine",
        "orig_flow", "perm_flow",
        "process_time_orig", "process_time_perm",
        "transition_p_val"  # <--- [CHANGE 4] Add to CSV header
    ]
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"Results saved to: {output_file}")


# ============================================
# SUMMARY STATISTICS
# ============================================
def summarize(results: List[Dict]) -> None:
    """Print summary statistics."""
    
    if not results:
        print("No results to summarize.")
        return
        
    orig_fd = np.array([r["orig_frame_diff"] for r in results])
    perm_fd = np.array([r["perm_frame_diff"] for r in results])
    p_vals = np.array([r["transition_p_val"] for r in results]) # Get p-values
    
    print("\n" + "=" * 60)
    print("GLOBAL SUMMARY")
    print("=" * 60 + "\n")
    
    print(f"Total videos analyzed: {len(results)}")
    
    # [CHANGE 5] Add Summary of Significance
    # How many videos are individually statistically broken?
    # We use 0.05 as the standard alpha threshold.
    valid_mask = np.isfinite(p_vals)
    valid_tests = int(np.sum(valid_mask))
    significant_count = int(np.sum((p_vals < 0.05) & valid_mask))
    significant_pct = (significant_count / valid_tests * 100.0) if valid_tests > 0 else 0.0
    print(f"Videos with broken dynamics (p < 0.05): {significant_count}/{valid_tests} ({significant_pct:.1f}%)\n")
    
    print("Average Metrics:")
    print(f"  Frame Diff (Orig): {np.mean(orig_fd):.4f}")
    print(f"  Frame Diff (Perm): {np.mean(perm_fd):.4f}")
    mean_orig_fd = np.mean(orig_fd)
    mean_perm_fd = np.mean(perm_fd)
    fold_change = (mean_perm_fd / mean_orig_fd) if mean_orig_fd != 0 else np.nan
    print(f"  Avg Fold Change:   {fold_change:.2f}x")
    print("=" * 60)


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":

    original_root = "partA/video/video_frontalized_interpolated_resolution_original"
    permuted_root = "partA/video/video_frontalized_interpolated_resolution_original_framepermute"
    output_csv = "temporal_dynamics_results_p_val_frame.csv"
    
    FRAME_SKIP = 1  
    RESIZE_DIM = 256  
    OF_DIM = 256  
    
    if not os.path.isdir(original_root):
        print(f"ERROR: Original video directory not found: {original_root}")
        sys.exit(1)
    
    if not os.path.isdir(permuted_root):
        print(f"ERROR: Permuted video directory not found: {permuted_root}")
        sys.exit(1)
    
    print(f"Searching for videos in: {original_root}")
    original_videos = find_videos(original_root)
    
    if not original_videos:
        print("ERROR: No MP4 files found.")
        sys.exit(1)
    
    print(f"Found {len(original_videos)} videos. Processing...\n")
    
    args = [(v, original_root, permuted_root, FRAME_SKIP, RESIZE_DIM, OF_DIM) for v in original_videos]
    
    num_workers = max(1, cpu_count() - 1)
    script_start = time.time()
    
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_pair, args),
                total=len(args),
                desc="Processing video pairs"
            )
        )
    
    results = [r for r in results if r is not None]
    
    if not results:
        print("ERROR: No results generated.")
        sys.exit(1)
    
    save_csv(results, output_csv)
    summarize(results)
    
    print(f"Total execution time: {time.time() - script_start:.2f}s")
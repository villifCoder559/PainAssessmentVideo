#!/usr/bin/env python3
"""Run smoke_test.sh on one GPU and check that the GPU was really used.

Device selection silently falls back to CPU when CUDA is unusable, so a PASS alone does
not prove GPU use. This script fails if any extraction or training log lacks the GPU
markers, or if any log contains a CUDA error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import run, setup  # noqa: E402

CUDA_ERRORS = re.compile(
  r"CUDA error|device-side assert|CUBLAS_STATUS|CUDNN_STATUS|out of memory|same device", re.I)


def gpu_problems(logs: Path) -> list[str]:
  """Log files that show the GPU was not used, or that contain CUDA errors."""
  problems = []
  for pattern, marker in (("extract_*.log", "extracting features using.... cuda"),
                          ("train_*.log", "GPU memory peak")):
    problems += [f"{log}: no '{marker}'" for log in sorted(logs.glob(pattern))
                 if marker not in log.read_text(errors="replace")]
  problems += [f"{log}: CUDA error" for log in sorted(logs.glob("*.log"))
               if CUDA_ERRORS.search(log.read_text(errors="replace"))]
  return problems


def main(argv: list[str] | None = None) -> int:
  parser = setup(__doc__.split("\n\n")[0], force=False)
  parser.add_argument("--full", action="store_true",
                      help="Also all 12 training configs and VideoMAEv2-S extraction (SMOKE_FULL=1).")
  args = parser.parse_args(argv)
  if args.gpu == "cpu":
    parser.error("use 'bash smoke_test.sh' for the CPU smoke test")
  run(["bash", "smoke_test.sh"], env={"SMOKE_DEVICE": args.gpu, "SMOKE_FULL": "1" if args.full else "0"})
  problems = gpu_problems(Path("smoke_out/logs"))
  for problem in problems:
    print(f"GPU CHECK FAIL  {problem}")
  print("GPU CHECK " + ("FAIL" if problems else "PASS"))
  return 1 if problems else 0


if __name__ == "__main__":
  sys.exit(main())

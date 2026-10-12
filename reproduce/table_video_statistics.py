#!/usr/bin/env python3
"""Video-level MAE of MAE-DFER vs VideoMAEv2-S, with paired t-tests and Holm correction.

Paper: tab:video-mae-performance, tab:video-mae-significance. Runs
statistical_compare_kfold_results.py --recompute once per dataset (re-infers the per-video
predictions of every fold on the GPU; skipped if its CSV exists), then applies the Holm
correction across the four dataset comparisons.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT, out_file, require_run, run, setup, skip  # noqa: E402

# Dataset name in the table -> (MAE-DFER config, VideoMAEv2-S config).
PAIRS = {
  "BioVid": ("biovid_dfer", "biovid_vmae_s"),
  "MIntPAIN": ("mintpain_dfer", "mintpain_vmae_s"),
  "UNBC (OPI 0--4)": ("unbc_opi0-4_dfer", "unbc_opi0-4_vmae_s"),
  "X-ITE": ("xite_dfer", "xite_vmae_s"),
}


def _p(value: float) -> str:
  if value >= 1e-3:
    return f"{value:.6f}".rstrip("0")
  mantissa, exponent = f"{value:.3e}".split("e")
  return rf"{mantissa}\times10^{{{int(exponent)}}}"


def render(rows: list[dict]) -> tuple[str, str]:
  perf = [r"\begin{tabular}{lrrrr}", r"\toprule",
          r"Dataset & Videos & MAE-DFER & VideoMAE~V2-S & \(\Delta\) \\", r"\midrule"]
  sig = [r"\begin{tabular}{lrr}", r"\toprule",
         r"Dataset & Raw \(p\) & Holm-adjusted \(p\) \\", r"\midrule"]
  for r in rows:
    perf.append(f"{r['dataset']} & {r['n_pairs']} & {r['dfer']:.4f} & {r['vmae']:.4f} & "
                rf"{r['vmae'] - r['dfer']:.4f} \\")
    star = r"\,{}^{*}" if r["p_holm"] < 0.05 else ""
    sig.append(rf"{r['dataset']} & \({_p(r['p_raw'])}\) & \({_p(r['p_holm'])}{star}\) \\")
  end = [r"\bottomrule", r"\end{tabular}", ""]
  return "\n".join(perf + end), "\n".join(sig + end)


def main(argv: list[str] | None = None) -> int:
  args = setup(__doc__.split("\n\n")[0]).parse_args(argv)
  from statistical_compare_kfold_results import holm_adjust
  rows = []
  for dataset, (dfer, vmae) in PAIRS.items():
    out = OUT / "statistics" / f"{dfer}_vs_{vmae}_video_mae.csv"
    if not args.force and out.is_file():
      skip(out)
    else:
      run([sys.executable, "statistical_compare_kfold_results.py",
           "--pkl_path_0", str(require_run(dfer)), "--pkl_path_1", str(require_run(vmae)),
           "--analysis_level", "video", "--measure", "mae", "--recompute", "--out", str(out)],
          gpu=args.gpu)
    result = pd.read_csv(out).query("metric == 'mae'").iloc[0]
    rows.append({"dataset": dataset, "n_pairs": int(result["n_pairs"]),
                 "dfer": result["pkl_0_mean"], "vmae": result["pkl_1_mean"],
                 "p_raw": result["ttest_p_raw"]})
  for row, adjusted in zip(rows, holm_adjust([r["p_raw"] for r in rows])):
    row["p_holm"] = float(adjusted)
  performance, significance = render(rows)
  for label, text in (("tab:video-mae-performance", performance),
                      ("tab:video-mae-significance", significance)):
    out_file(label).write_text(text)
    print(f"Saved LaTeX to: {out_file(label)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())

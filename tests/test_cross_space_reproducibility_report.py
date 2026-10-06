import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cross_space_reproducibility_report as report


PAPER_TABLE = r"""
\begin{table}[H]
\begin{tabular}{llccc}
\toprule
\textbf{Direction} & \shortstack{\textbf{Original} \\ \textbf{(MAE)}} & x & y \\
\cmidrule(lr){3-4}
\midrule
\multirow{2}{*}{
    \shortstack{
        UNBC $\rightarrow$ BioVid \\
        \footnotesize evaluated on UNBC
    }
}
& Linear
& 0.85 / 0.91
& -2.00 / 44.10 \\
% & MLP & 9.99 / 9.99 & 9.99 / 9.99 \\
& EncDec & 0.83 / 0.90 & +1.5 / +0.0 \\
\midrule
MIntPAIN $\rightarrow$ UNBC & OLS & 1.332 / 1.436 & 0.0 / 0.0 \\
\bottomrule
\end{tabular}
\caption{Example.}
\label{tab:example}
\end{table}
"""

REPRODUCED_TABLE = r"""
\begin{table}[H]
\begin{tabular}{llccc}
\toprule
\textbf{Direction} & a & b \\
\midrule
\multirow{3}{*}{UNBC $\rightarrow$ BioVid} & Linear & 0.86 / 0.91 & -2.0 / 44.1 \\
 & Procrustes & 0.88 / 0.96 & -6.1 / +56.5 \\
 & EncDec & 0.83 / 0.92 & +1.5 / +0.0 \\
\midrule
\multirow{1}{*}{MIntPAIN $\rightarrow$ UNBC} & Linear (closed form) & 1.332 / 1.436 & +0.4 / +0.0 \\
\bottomrule
\end{tabular}
\label{tab:ours}
\end{table}
"""

SPEC = report.TableSpec(
  "Example", Path("paper.tex"), ("tab:example",), "ours.tex", "tab:ours",
  report._columns("MAE", "Macro", "Δ% MAE", "Δ% Macro", pct=("Δ% MAE", "Δ% Macro")),
)


class ParseTableTest(unittest.TestCase):
  def test_parses_shortstack_multiline_rows_aliases_and_signed_values(self):
    rows = report.parse_table(PAPER_TABLE, "tab:example")
    self.assertEqual(rows, [
      report.Row("UNBC → BioVid", "Linear", ("0.85", "0.91", "-2.00", "44.10")),
      report.Row("UNBC → BioVid", "EncDec", ("0.83", "0.90", "+1.5", "+0.0")),
      report.Row("MIntPAIN → UNBC", "Linear (closed form)", ("1.332", "1.436", "0.0", "0.0")),
    ])

  def test_rejects_missing_label_and_non_numeric_cells(self):
    with self.assertRaisesRegex(ValueError, "Label not found"):
      report.parse_table(PAPER_TABLE, "tab:missing")
    with self.assertRaisesRegex(ValueError, "Not a numeric cell"):
      report.parse_table(PAPER_TABLE.replace("0.83 / 0.90", "n/a"), "tab:example")


class CompareTableTest(unittest.TestCase):
  def test_applies_mae_and_percentage_tolerances(self):
    cells, extra = report.compare_table(SPEC, PAPER_TABLE, REPRODUCED_TABLE)
    by_key = {(cell["method"], cell["column"]): cell for cell in cells}
    self.assertEqual(len(cells), 12)
    self.assertAlmostEqual(by_key[("Linear", "MAE")]["delta"], 0.01)
    self.assertTrue(by_key[("Linear", "MAE")]["within_tolerance"])
    self.assertEqual(by_key[("Linear", "Δ% Macro")]["delta"], 0.0)
    self.assertFalse(by_key[("EncDec", "Macro")]["within_tolerance"])  # +0.02 > 0.01
    self.assertTrue(by_key[("Linear (closed form)", "Δ% MAE")]["within_tolerance"])  # 0.4 <= 0.5
    self.assertEqual([row.method for row in extra], ["Procrustes"])

  def test_flags_percentage_outside_tolerance(self):
    cells, _ = report.compare_table(
      SPEC, PAPER_TABLE, REPRODUCED_TABLE.replace("+0.4 / +0.0", "+0.4 / +0.6"))
    cell = next(cell for cell in cells
                if cell["method"] == "Linear (closed form)" and cell["column"] == "Δ% Macro")
    self.assertFalse(cell["within_tolerance"])

  def test_rejects_paper_row_without_reproduced_row(self):
    with self.assertRaisesRegex(ValueError, "no reproduced row for UNBC → BioVid EncDec"):
      report.compare_table(
        SPEC, PAPER_TABLE,
        REPRODUCED_TABLE.replace(r" & EncDec & 0.83 / 0.92 & +1.5 / +0.0 \\", ""))


FIGURE = r"""
\begin{figure}[H]
\begin{tabular}{r@{\quad}rr}
\toprule
& \multicolumn{2}{c}{Predicted level} \\
True & 0 & 1 \\
\midrule
0 & 58.3 & 41.7 \\
1 & 54.1 & 45.9 \\
\bottomrule
\end{tabular}
\begin{tabular}{r@{\quad}rr}
\toprule
True & 0 & 1 \\
\midrule
0 & 100.0 & 0.0 \\
1 & 100.0 & 0.0 \\
\bottomrule
\end{tabular}
\caption{Example.}
\label{fig:example}
\end{figure}
"""


class ParseConfusionTest(unittest.TestCase):
  def test_returns_each_matrix_in_order(self):
    self.assertEqual(report.parse_confusion(FIGURE, "fig:example"), [
      [["58.3", "41.7"], ["54.1", "45.9"]],
      [["100.0", "0.0"], ["100.0", "0.0"]],
    ])


if __name__ == "__main__":
  unittest.main()

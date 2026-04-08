"""
Hyperparameter Grid Search Statistical Analysis
================================================
Analyzes a summary.csv from a hyperparameter grid search using:
  - Auto-detection of hyperparameter columns (overridable via CLI)
  - Kruskal-Wallis (3+ level factors) / Mann-Whitney U (2-level factors)
  - Main-effects ANOVA (manual Type-II SS, no statsmodels needed)
  - Shapiro-Wilk on residuals
  - Pairwise post-hoc with Bonferroni correction

Usage:
  # Fully automatic — detect factors from the CSV
  python analyze_grid_search.py --csv summary.csv --metric mean_val_l1_error_best_epochs_k0

  # Exclude specific auto-detected columns
  python analyze_grid_search.py --csv summary.csv --exclude-cols time_min random_state

  # Skip auto-detection entirely and specify factors manually
  python analyze_grid_search.py --csv summary.csv --factors learning_rate ATTENTIVE_JEPA.dropout

  # Post-hoc on all / significant / explicit factors
  python analyze_grid_search.py --csv summary.csv --posthoc-factors all
  python analyze_grid_search.py --csv summary.csv --posthoc-factors significant
  python analyze_grid_search.py --csv summary.csv --posthoc-factors learning_rate

Requirements: pandas, numpy, scipy  (no statsmodels)
"""

import argparse
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV    = "summary.csv"
DEFAULT_METRIC = "mean_val_l1_error_best_epochs_k0"

# Patterns that identify output/result columns, not hyperparameters.
# Columns whose name matches any of these are excluded from auto-detection.
OUTPUT_PATTERNS = [
  r"^mean_train",
  r"^mean_val",
  r"^total_mean",
  r"^total_median",
  r"^best_epoch",
  r"^final_best",
  r"^time_",
  r"_loss$",
  r"_accuracy$",
  r"_error$",
  r"_rmse$",
  r"^test_id$",
  r"^random_state$",
  r"^real_k_fold$",
  r"^k_fold",
  r"steps_per_epoch",
  r"clip_grad_norm",
  r"debug_",
]

# Values that look like Python object reprs or serialized dicts/lists —
# these columns cannot be used as ANOVA factors.
COMPLEX_VALUE_RE = re.compile(r"^[{\[<]")


# ---------------------------------------------------------------------------
# Factor auto-detection
# ---------------------------------------------------------------------------

def _looks_complex(series: pd.Series) -> bool:
  """Return True if any non-null value looks like a dict, list, or object repr."""
  sample = series.dropna().astype(str)
  return sample.apply(lambda v: bool(COMPLEX_VALUE_RE.match(v))).any()


def _matches_output_pattern(col: str) -> bool:
  return any(re.search(pat, col) for pat in OUTPUT_PATTERNS)


def auto_detect_factors(
  df: pd.DataFrame,
  metric: str,
  max_levels: int = 15,
  min_levels: int = 2,
  exclude_cols: list[str] | None = None,
) -> dict[str, str]:
  """
  Scan all columns and return a {col_name: col_name} dict of likely
  hyperparameter columns, applying the following filters in order:

    1. Skip the metric column itself.
    2. Skip columns explicitly excluded via --exclude-cols.
    3. Skip columns whose name matches an output/result pattern.
    4. Skip columns with fewer than min_levels or more than max_levels
       unique non-null values (constant or continuous).
    5. Skip columns whose values look like serialized Python objects
       (dicts, lists, object reprs).
  """
  exclude_set = set(exclude_cols or [])
  factors = {}

  for col in df.columns:
    if col == metric:
      continue
    if col in exclude_set:
      continue
    if _matches_output_pattern(col):
      continue

    n_unique = df[col].nunique(dropna=True)
    if not (min_levels <= n_unique <= max_levels):
      continue

    if _looks_complex(df[col]):
      continue

    factors[col] = col

  return factors


def print_detected_factors(factors: dict[str, str]) -> None:
  print("=" * 65)
  print("0. AUTO-DETECTED FACTORS")
  print("   (use --exclude-cols or --factors to override)")
  print("=" * 65)
  if not factors:
    print("  ⚠  No factors detected. Use --factors to specify them manually.")
    return
  for col in factors:
    print(f"  {col}")
  print()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
  csv_path: str,
  metric: str,
  factors: dict[str, str],
) -> pd.DataFrame:
  df = pd.read_csv(csv_path)
  missing = [col for col in factors.values() if col not in df.columns]
  if missing:
    raise ValueError(f"Factor columns not found in CSV: {missing}")
  if metric not in df.columns:
    raise ValueError(f"Metric column '{metric}' not found in CSV.")
  print(f"Loaded {len(df)} rows from '{csv_path}'")
  print(f"Metric  : {metric}")
  print(f"  count={len(df[metric])}, mean={df[metric].mean():.4f}, "
      f"std={df[metric].std():.4f}, "
      f"min={df[metric].min():.4f}, max={df[metric].max():.4f}\n")
  return df


# ---------------------------------------------------------------------------
# 1. Factor-level summary
# ---------------------------------------------------------------------------

def factor_summary(
  df: pd.DataFrame,
  metric: str,
  factors: dict[str, str],
) -> None:
  print("=" * 65)
  print("1. FACTOR-LEVEL SUMMARY  (count / mean / std)")
  print("=" * 65)
  for name, col in factors.items():
    tbl = df.groupby(col)[metric].agg(["count", "mean", "std"]).round(4)
    tbl.index.name = name
    print(tbl.to_string())
    print()


# ---------------------------------------------------------------------------
# 2. Non-parametric marginal tests
# ---------------------------------------------------------------------------

def nonparametric_tests(
  df: pd.DataFrame,
  metric: str,
  factors: dict[str, str],
) -> dict[str, float]:
  """Run marginal non-parametric tests. Returns {factor_name: p_value}."""
  print("=" * 65)
  print("2. NON-PARAMETRIC MARGINAL TESTS")
  print("   (Kruskal-Wallis for 3+ levels, Mann-Whitney U for 2 levels)")
  print("=" * 65)
  fmt = "{:<30} {:<20} {:>10} {:>10}  {}"
  print(fmt.format("Factor", "Test", "stat", "p", "sig"))
  print("-" * 75)

  pvalues = {}
  for name, col in factors.items():
    groups = [g[metric].values for _, g in df.groupby(col)]
    if len(groups) < 2:
      continue
    if len(groups) == 2:
      stat, p = stats.mannwhitneyu(*groups, alternative="two-sided")
      test = "Mann-Whitney U"
    else:
      stat, p = stats.kruskal(*groups)
      test = "Kruskal-Wallis"
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(fmt.format(name[:30], test, f"{stat:.3f}", f"{p:.4f}", sig))
    pvalues[name] = p
  print()
  return pvalues


# ---------------------------------------------------------------------------
# 3. ANOVA helpers
# ---------------------------------------------------------------------------

def dummy_encode(series: pd.Series, name: str) -> dict:
  """Drop-first dummy encoding. Returns {col_label: array}."""
  vals = sorted(series.unique())
  return {
    f"{name}={v}": (series == v).astype(float).values
    for v in vals[1:]
  }


def ols_residual_ss(X: np.ndarray, y: np.ndarray):
  """Return (SS_residuals, coefficients) via least squares."""
  b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
  ss = np.sum((y - X @ b) ** 2)
  return ss, b


def anova_type2(
  df: pd.DataFrame,
  metric: str,
  factors: dict[str, str],
) -> np.ndarray:
  """
  Type-II SS ANOVA: for each factor, compare the full model to the
  model with that factor removed. SS_factor = SS_reduced - SS_full.
  Returns residuals for the normality check.
  """
  print("=" * 65)
  print("3. MAIN-EFFECTS ANOVA  (Type-II SS, manual implementation)")
  print("   Assumption: no interactions (interactions treated as error)")
  print("=" * 65)

  y = df[metric].values
  n = len(y)

  dm = {"intercept": np.ones(n)}
  factor_cols: dict[str, list[str]] = {}
  for fname, col in factors.items():
    dummies = dummy_encode(df[col], fname)
    factor_cols[fname] = list(dummies.keys())
    dm.update(dummies)

  col_names  = list(dm.keys())
  X_full     = np.column_stack(list(dm.values()))
  ss_full, b_full = ols_residual_ss(X_full, y)
  df_resid   = n - X_full.shape[1]
  ms_resid   = ss_full / df_resid

  fmt = "{:<30} {:>10} {:>4} {:>10} {:>8} {:>8}  {}"
  print(fmt.format("Factor", "SS", "df", "MS", "F", "p", "sig"))
  print("-" * 75)

  all_idx = list(range(len(col_names)))
  for fname, fcols in factor_cols.items():
    remove_idx = [col_names.index(c) for c in fcols]
    keep_idx   = [i for i in all_idx if i not in remove_idx]
    X_reduced  = X_full[:, keep_idx]
    ss_reduced, _ = ols_residual_ss(X_reduced, y)
    ss_factor  = ss_reduced - ss_full
    df_factor  = len(fcols)
    ms_factor  = ss_factor / df_factor
    F          = ms_factor / ms_resid
    p          = 1 - stats.f.cdf(F, df_factor, df_resid)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns" # ns = not significant
    print(fmt.format(fname[:30], f"{ss_factor:.6f}", df_factor,
             f"{ms_factor:.6f}", f"{F:.3f}", f"{p:.4f}", sig))

  print(fmt.format("Residuals", f"{ss_full:.6f}", df_resid,
           f"{ms_resid:.6f}", "", "", ""))

  ss_total = np.sum((y - y.mean()) ** 2)
  r2 = 1 - ss_full / ss_total
  print(f"\nR² = {r2:.4f}  |  Residual df = {df_resid}")
  print()
  return y - X_full @ b_full


# ---------------------------------------------------------------------------
# 4. Shapiro-Wilk on residuals
# ---------------------------------------------------------------------------

def check_normality(residuals: np.ndarray) -> None:
  print("=" * 65)
  print("4. SHAPIRO-WILK ON ANOVA RESIDUALS")
  print("   (checks whether ANOVA normality assumption holds)")
  print("=" * 65)
  w, p = stats.shapiro(residuals)
  verdict = "PASS → ANOVA valid" if p > 0.05 else "FAIL → prefer non-parametric results"
  print(f"  W = {w:.4f},  p = {p:.4f}  →  {verdict}\n")


# ---------------------------------------------------------------------------
# 5. Collinearity check
# ---------------------------------------------------------------------------

def collinearity_check(
  df: pd.DataFrame,
  factors: dict[str, str],
) -> None:
  print("=" * 65)
  print("5. COLLINEARITY CHECK")
  print("   (perfect collinearity makes factor effects inseparable)")
  print("=" * 65)
  items = list(factors.items())
  found_any = False
  for (na, ca), (nb, cb) in combinations(items, 2):
    ct = pd.crosstab(df[ca], df[cb])
    is_collinear = (
      (ct > 0).sum(axis=1).max() == 1
      and (ct > 0).sum(axis=0).max() == 1
    )
    if is_collinear:
      found_any = True
      print(f"  ⚠  '{na}' and '{nb}' are perfectly collinear.")
      print(f"     Their effects cannot be separated in ANOVA.")
      print(f"     Crosstab:\n{ct}\n")
  if not found_any:
    print("  No perfect collinearity detected.\n")


# ---------------------------------------------------------------------------
# 6. Post-hoc pairwise tests
# ---------------------------------------------------------------------------

def posthoc(
  df: pd.DataFrame,
  metric: str,
  factor_name: str,
  factors: dict[str, str],
) -> None:
  col   = factors[factor_name]
  vals  = sorted(df[col].unique())
  pairs = list(combinations(vals, 2))
  alpha_bonf = 0.05 / len(pairs)

  print("=" * 65)
  print(f"6. POST-HOC: '{factor_name}'  (pairwise Mann-Whitney + Bonferroni)")
  print(f"   Bonferroni α = 0.05 / {len(pairs)} = {alpha_bonf:.4f}")
  print("=" * 65)
  fmt = "  {:>14} vs {:>14}:  p={:.5f}  {}  |  means {:.4f} vs {:.4f}"
  for a, b in pairs:
    g1 = df[df[col] == a][metric]
    g2 = df[df[col] == b][metric]
    _, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    sig = "* (sig)" if p < alpha_bonf else "ns"
    print(fmt.format(str(a), str(b), p, sig, g1.mean(), g2.mean()))
  print()


def resolve_posthoc_factors(
  requested: list[str],
  pvalues: dict[str, float],
  factors: dict[str, str],
  alpha: float,
) -> list[str]:
  """Translate --posthoc-factors into a concrete list of factor names."""
  if requested == ["all"]:
    return list(factors.keys())
  if requested == ["significant"]:
    chosen = [f for f, p in pvalues.items() if p < alpha]
    if not chosen:
      print(f"  No factors reached p < {alpha}. No post-hoc tests will run.\n")
    return chosen
  valid, invalid = [], []
  for name in requested:
    (valid if name in factors else invalid).append(name)
  if invalid:
    print(f"  Warning: unknown factor(s) ignored: {invalid}")
  return valid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
  p = argparse.ArgumentParser(
    description="Statistical analysis of a hyperparameter grid search CSV.",
    formatter_class=argparse.RawTextHelpFormatter,
  )
  p.add_argument("--csv",    required=True,    help="Path to summary CSV")
  p.add_argument("--metric", default=DEFAULT_METRIC, help="Target metric column name")

  # Factor selection
  p.add_argument(
    "--factors",
    nargs="+",
    default=None,
    metavar="COL",
    help=(
      "Explicit CSV column names to use as factors.\n"
      "Skips auto-detection entirely.\n"
      "Example: --factors learning_rate ATTENTIVE_JEPA.dropout reg_lambda_L2"
    ),
  )
  p.add_argument(
    "--exclude-cols",
    nargs="+",
    default=["reg_lambda_L2", "early_stopping","lr","learning_rate","regularization_lambda_L2"],
    metavar="COL",
    help="Columns to exclude from auto-detection (ignored if --factors is set).",
  )
  p.add_argument(
    "--max-levels",
    type=int,
    default=15,
    help="Max unique values for a column to qualify as a factor (default: 15).",
  )
  p.add_argument(
    "--min-levels",
    type=int,
    default=2,
    help="Min unique values for a column to qualify as a factor (default: 2).",
  )

  # Post-hoc
  p.add_argument(
    "--posthoc-factors",
    nargs="+",
    default=["significant"],
    metavar="FACTOR",
    help=(
      "Which factors to run post-hoc on:\n"
      "  'significant' — factors with p < --posthoc-alpha (default)\n"
      "  'all'         — every detected factor\n"
      "  <col names>   — explicit list"
    ),
  )
  p.add_argument(
    "--posthoc-alpha",
    type=float,
    default=0.05,
    help="p-value threshold for 'significant' mode (default: 0.05).",
  )
  return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  args = parse_args()

  # --- Build the factors dict ---
  if args.factors:
    # Manual override: use exactly the columns the user listed.
    factors = {col: col for col in args.factors}
    print("=" * 65)
    print("0. FACTORS  (manually specified via --factors)")
    print("=" * 65)
    for col in factors:
      print(f"  {col}")
    print()
  else:
    # Auto-detect: read the CSV once just to scan column names and values.
    raw = pd.read_csv(args.csv)
    factors = auto_detect_factors(
      raw,
      metric=args.metric,
      max_levels=args.max_levels,
      min_levels=args.min_levels,
      exclude_cols=args.exclude_cols,
    )
    print_detected_factors(factors)

  if not factors:
    raise SystemExit("No factors to analyze. Exiting.")

  # --- Load and validate ---
  df = load_data(args.csv, args.metric, factors)

  # --- Analysis pipeline ---
  factor_summary(df, args.metric, factors)
  pvalues = nonparametric_tests(df, args.metric, factors)
  collinearity_check(df, factors)
  residuals = anova_type2(df, args.metric, factors)
  check_normality(residuals)

  factors_to_test = resolve_posthoc_factors(
    args.posthoc_factors, pvalues, factors, args.posthoc_alpha
  )
  for factor in factors_to_test:
    posthoc(df, args.metric, factor, factors)


if __name__ == "__main__":
  main()
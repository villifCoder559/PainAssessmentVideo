"""Generate LaTeX projection tables and a tidy summary CSV.

Expected input columns
----------------------
- source_pkl
- interpolation_similarity
- refine_mode
- subtrial_index
- srctest_mae_micro_old
- srctest_mae_macro_old
- srctest_mae_micro_after
- srctest_mae_macro_after
- newtest_mae_micro_before
- newtest_mae_macro_before
- newtest_mae_micro_after
- newtest_mae_macro_after

The input may contain ``AGGREGATE_MEAN``/``AGGREGATE_STD`` rows. Only the
mean rows are used. If aggregate rows are absent, the script computes means
from the available rows for each configuration.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "source_pkl",
    "interpolation_similarity",
    "refine_mode",
    "subtrial_index",
    "srctest_mae_micro_old",
    "srctest_mae_macro_old",
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_before",
    "newtest_mae_macro_before",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
}

METRIC_COLUMNS = [
    "srctest_mae_micro_old",
    "srctest_mae_macro_old",
    "srctest_mae_micro_after",
    "srctest_mae_macro_after",
    "newtest_mae_micro_before",
    "newtest_mae_macro_before",
    "newtest_mae_micro_after",
    "newtest_mae_macro_after",
]

# Known methods are shown in this order. Unknown methods are appended
# alphabetically, so the script remains usable with future projection methods.
METHOD_ORDER = [
    "procrustes",
    "linear_close",
    "linear",
    "mlp",
    "autoencoder",
]

METHOD_DISPLAY_NAMES = {
    "procrustes": "Procrustes",
    "linear_close": "Linear layer (closed form)",
    "linear": "Linear layer (SGD)",
    "mlp": "MLP",
    "autoencoder": "Autoencoder",
}

REFINE_MODE_DESCRIPTIONS = {
    "linear_only": "only the regressor head is fine-tuned",
    "projector_linear": "both the projector and the regressor head are fine-tuned",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one LaTeX table per refine_mode and save the underlying "
            "tidy mean-results DataFrame as CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to aggregated_summary_filtered.csv",
    )
    parser.add_argument(
        "--latex-output",
        type=Path,
        default=Path("projection_results_means_only_with_comments.tex"),
        help=(
            "Output .tex filename. It is always written in the same directory "
            "as input_csv (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--df-output",
        type=Path,
        default=Path("projection_results_means_only.csv"),
        help=(
            "Output tidy summary CSV filename. It is always written in the "
            "same directory as input_csv (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--comment-prefix",
        default="cross-validation_BioVmae-unbcDFER_v2",
        help="Prefix placed before the source_pkl folder in LaTeX comments",
    )
    parser.add_argument(
        "--test-date",
        default="test_Jul_16_26",
        help="Date/tag placed in every recovery comment",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places in LaTeX and saved CSV (default: 2)",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            "The input CSV is missing required columns: " + ", ".join(missing)
        )


def output_path_next_to_input(input_csv: Path, output: Path) -> Path:
    """Return an output path in the same directory as the input CSV.

    Any directory component supplied for ``output`` is discarded; only its
    filename is retained.
    """
    resolved_input = input_csv.expanduser().resolve()
    return resolved_input.parent / output.name


def source_folder(source_pkl: object) -> str:
    """Return source_pkl without the final results_*.pkl filename."""
    value = str(source_pkl).replace("\\", "/").strip()
    return value.rsplit("/", 1)[0] if "/" in value else value


def extract_num_anchors(source_pkl: object) -> int:
    """Extract the anchor count from names such as refinement3_mlp_250/..."""
    folder = source_folder(source_pkl).split("/", 1)[0]
    match = re.search(r"_(\d+)$", folder)
    if not match:
        raise ValueError(
            f"Could not extract num_anchors from source_pkl={source_pkl!r}. "
            "Expected the first folder to end in an integer, e.g. "
            "'refinement3_mlp_250'."
        )
    return int(match.group(1))


def select_or_compute_means(df: pd.DataFrame) -> pd.DataFrame:
    """Select AGGREGATE_MEAN rows, or compute configuration means if absent."""
    work = df.copy()
    for column in METRIC_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    aggregate_mask = (
        work["subtrial_index"].astype(str).str.upper().eq("AGGREGATE_MEAN")
    )
    if aggregate_mask.any():
        means = work.loc[aggregate_mask].copy()
    else:
        group_columns = [
            "source_pkl",
            "interpolation_similarity",
            "refine_mode",
        ]
        means = (
            work.groupby(group_columns, dropna=False, as_index=False)[METRIC_COLUMNS]
            .mean()
        )
        means["subtrial_index"] = "COMPUTED_MEAN"

    if means.empty:
        raise ValueError("No usable rows were found in the input CSV.")

    means["num_anchors"] = means["source_pkl"].map(extract_num_anchors)
    means["source_pkl_folder"] = means["source_pkl"].map(source_folder)
    return means


def scalar_baseline(series: pd.Series, name: str, tolerance: float = 1e-10) -> float:
    """Return a baseline value and warn if it is not constant across rows."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"No numeric values found for baseline column {name!r}.")

    # Baselines should be repeated constants in the aggregated file. Using the
    # mean makes the script tolerant to tiny floating-point differences.
    if values.max() - values.min() > tolerance:
        print(
            f"Warning: baseline column {name!r} is not constant; "
            f"using its mean ({values.mean():.6f})."
        )
    return float(values.mean())


def method_sort_key(method: str) -> tuple[int, str]:
    method = str(method)
    try:
        return METHOD_ORDER.index(method), method
    except ValueError:
        return len(METHOD_ORDER), method


def latex_escape_identifier(value: object) -> str:
    """Escape underscores for use in LaTeX text and labels."""
    return str(value).replace("_", r"\_")


def build_summary_dataframe(
    means: pd.DataFrame,
    comment_prefix: str,
    test_date: str,
    decimals: int,
) -> pd.DataFrame:
    """Create the tidy DataFrame used to write both CSV and LaTeX."""
    dfer_micro = scalar_baseline(
        means["newtest_mae_micro_before"], "newtest_mae_micro_before"
    )
    dfer_macro = scalar_baseline(
        means["newtest_mae_macro_before"], "newtest_mae_macro_before"
    )
    videomae_micro = scalar_baseline(
        means["srctest_mae_micro_old"], "srctest_mae_micro_old"
    )
    videomae_macro = scalar_baseline(
        means["srctest_mae_macro_old"], "srctest_mae_macro_old"
    )

    records: list[dict[str, object]] = []
    refine_modes = list(dict.fromkeys(means["refine_mode"].astype(str)))

    for refine_mode in refine_modes:
        mode_df = means.loc[means["refine_mode"].astype(str).eq(refine_mode)].copy()
        methods = sorted(
            mode_df["interpolation_similarity"].astype(str).unique(),
            key=method_sort_key,
        )

        for method in methods:
            method_df = mode_df.loc[
                mode_df["interpolation_similarity"].astype(str).eq(method)
            ].sort_values("num_anchors")

            for _, row in method_df.iterrows():
                recovery_path = (
                    f"{comment_prefix.rstrip('/')}/"
                    f"{str(row['source_pkl_folder']).lstrip('/')}"
                )
                records.append(
                    {
                        "refine_mode": refine_mode,
                        "row_type": "projection",
                        "interpolation_similarity": method,
                        "projection_method": METHOD_DISPLAY_NAMES.get(
                            method, method.replace("_", " ").title()
                        ),
                        "num_anchors": int(row["num_anchors"]),
                        "unbc_mae": row["newtest_mae_micro_after"],
                        "unbc_macro_mae": row["newtest_mae_macro_after"],
                        "biovid_mae": row["srctest_mae_micro_after"],
                        "biovid_macro_mae": row["srctest_mae_macro_after"],
                        "source_pkl": row["source_pkl"],
                        "source_pkl_folder": row["source_pkl_folder"],
                        "recovery_comment": (
                            f"{recovery_path}, date: {test_date}"
                        ),
                    }
                )

        records.extend(
            [
                {
                    "refine_mode": refine_mode,
                    "row_type": "baseline",
                    "interpolation_similarity": pd.NA,
                    "projection_method": "DFER (UNBC)",
                    "num_anchors": pd.NA,
                    "unbc_mae": dfer_micro,
                    "unbc_macro_mae": dfer_macro,
                    "biovid_mae": pd.NA,
                    "biovid_macro_mae": pd.NA,
                    "source_pkl": pd.NA,
                    "source_pkl_folder": pd.NA,
                    "recovery_comment": pd.NA,
                },
                {
                    "refine_mode": refine_mode,
                    "row_type": "baseline",
                    "interpolation_similarity": pd.NA,
                    "projection_method": "VideoMAE (BioVid)",
                    "num_anchors": pd.NA,
                    "unbc_mae": pd.NA,
                    "unbc_macro_mae": pd.NA,
                    "biovid_mae": videomae_micro,
                    "biovid_macro_mae": videomae_macro,
                    "source_pkl": pd.NA,
                    "source_pkl_folder": pd.NA,
                    "recovery_comment": pd.NA,
                },
            ]
        )

    summary = pd.DataFrame.from_records(records)
    metric_columns = [
        "unbc_mae",
        "unbc_macro_mae",
        "biovid_mae",
        "biovid_macro_mae",
    ]
    summary[metric_columns] = (
        summary[metric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .astype("Float64")
        .round(decimals)
    )
    summary["num_anchors"] = pd.to_numeric(
        summary["num_anchors"], errors="coerce"
    ).astype("Int64")
    return summary


def format_metric(value: object, decimals: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{float(value):.{decimals}f}"


def format_anchor(value: object) -> str:
    if pd.isna(value):
        return "--"
    return str(int(value))


def table_caption(refine_mode: str) -> str:
    description = REFINE_MODE_DESCRIPTIONS.get(
        refine_mode,
        "the selected projector/refinement components are fine-tuned",
    )
    mode_tex = latex_escape_identifier(refine_mode)
    return (
        "Projection results for the backbone-shift setting with the "
        f"\\texttt{{{mode_tex}}} refinement mode ({description}). "
        "Values are reported as mean values."
    )


def build_latex_table(
    mode_df: pd.DataFrame,
    refine_mode: str,
    decimals: int,
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{table_caption(refine_mode)}}}",
        rf"\label{{tab:unbc_biovid_full_shift_{refine_mode}_results}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccc}",
        r"    \toprule",
        r"    & & \multicolumn{2}{c}{\textbf{UNBC}} & \multicolumn{2}{c}{\textbf{BioVid}} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"    \textbf{Projection method} & \textbf{Anchors} & \textbf{MAE} & \textbf{Macro-MAE} & \textbf{MAE} & \textbf{Macro-MAE} \\",
        r"    \midrule",
    ]

    projection_df = mode_df.loc[mode_df["row_type"].eq("projection")].copy()
    methods = list(dict.fromkeys(projection_df["interpolation_similarity"].astype(str)))

    for method_index, method in enumerate(methods):
        method_rows = projection_df.loc[
            projection_df["interpolation_similarity"].astype(str).eq(method)
        ].sort_values("num_anchors")

        for _, row in method_rows.iterrows():
            lines.append(
                "    "
                f"{row['projection_method']} & {format_anchor(row['num_anchors'])} & "
                f"{format_metric(row['unbc_mae'], decimals)} & "
                f"{format_metric(row['unbc_macro_mae'], decimals)} & "
                f"{format_metric(row['biovid_mae'], decimals)} & "
                f"{format_metric(row['biovid_macro_mae'], decimals)} "
                rf"\\ % ({row['recovery_comment']})"
            )

        if method_index < len(methods) - 1:
            lines.append(r"    \addlinespace")

    lines.append(r"    \midrule\midrule")
    baseline_df = mode_df.loc[mode_df["row_type"].eq("baseline")]
    for _, row in baseline_df.iterrows():
        lines.append(
            "    "
            f"{row['projection_method']} & {format_anchor(row['num_anchors'])} & "
            f"{format_metric(row['unbc_mae'], decimals)} & "
            f"{format_metric(row['unbc_macro_mae'], decimals)} & "
            f"{format_metric(row['biovid_mae'], decimals)} & "
            f"{format_metric(row['biovid_macro_mae'], decimals)} "
            r"\\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def build_latex_document(summary_df: pd.DataFrame, decimals: int) -> str:
    tables: list[str] = []
    for refine_mode in dict.fromkeys(summary_df["refine_mode"].astype(str)):
        mode_df = summary_df.loc[
            summary_df["refine_mode"].astype(str).eq(refine_mode)
        ]
        tables.append(build_latex_table(mode_df, refine_mode, decimals))
    return "\n\n".join(tables) + "\n"


def generate_outputs(
    input_csv: Path,
    latex_output: Path,
    df_output: Path,
    comment_prefix: str = "cross-validation_BioVmae-unbcDFER_v2",
    test_date: str = "test_Jul_16_26",
    decimals: int = 2,
) -> pd.DataFrame:
    """Generate both output files and return the tidy summary DataFrame."""
    if decimals < 0:
        raise ValueError("decimals must be non-negative.")

    input_csv = input_csv.expanduser().resolve()
    latex_output = output_path_next_to_input(input_csv, latex_output)
    df_output = output_path_next_to_input(input_csv, df_output)

    raw_df = pd.read_csv(input_csv)
    validate_columns(raw_df)
    means = select_or_compute_means(raw_df)
    summary_df = build_summary_dataframe(
        means=means,
        comment_prefix=comment_prefix,
        test_date=test_date,
        decimals=decimals,
    )
    latex = build_latex_document(summary_df, decimals=decimals)

    latex_output.parent.mkdir(parents=True, exist_ok=True)
    df_output.parent.mkdir(parents=True, exist_ok=True)
    latex_output.write_text(latex, encoding="utf-8")
    summary_df.to_csv(df_output, index=False, float_format=f"%.{decimals}f")
    return summary_df


def main() -> None:
    args = parse_args()
    resolved_input = args.input_csv.expanduser().resolve()
    resolved_latex_output = output_path_next_to_input(
        resolved_input, args.latex_output
    )
    resolved_df_output = output_path_next_to_input(
        resolved_input, args.df_output
    )

    summary_df = generate_outputs(
        input_csv=resolved_input,
        latex_output=resolved_latex_output,
        df_output=resolved_df_output,
        comment_prefix=args.comment_prefix,
        test_date=args.test_date,
        decimals=args.decimals,
    )
    print(f"Saved LaTeX tables to: {resolved_latex_output}")
    print(f"Saved summary DataFrame to: {resolved_df_output}")
    print(f"Generated {len(summary_df)} rows across "
          f"{summary_df['refine_mode'].nunique()} refine_mode values.")


if __name__ == "__main__":
    main()
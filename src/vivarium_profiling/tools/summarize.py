from pathlib import Path

import pandas as pd
from loguru import logger

from vivarium_profiling.tools.extraction import ExtractionConfig
from vivarium_profiling.tools.plotting import (
    create_figures,
    plot_bottleneck_fractions,
)

"""Benchmark summarization and visualization utilities."""

BASELINE = "model_spec_baseline.yaml"  # Default baseline model spec name
BASE_SUMMARIZE_COLUMNS = ["rt_s", "mem_mb", "rt_non_run_s"]


def summarize(
    raw: pd.DataFrame, output_dir: Path, config: ExtractionConfig | None = None
) -> pd.DataFrame:
    """Summarize benchmark results with statistics and percent differences.

    Parameters
    ----------
    raw
        Raw benchmark results DataFrame with columns: model_spec, run, rt_s, mem_mb,
        and metric columns from extraction config.
    output_dir
        Directory to save summary.csv.
    config
        Extraction configuration defining metrics to summarize. If None, uses default.

    Returns
    -------
        Summary DataFrame with aggregated statistics and percent differences.

    """
    if config is None:
        config = ExtractionConfig()

    bottleneck_names = [p.name for p in config.patterns if p.extract_cumtime]

    summary = raw.copy()
    summary["rt_non_run_s"] = summary["rt_s"] - summary["rt_run_s"]

    # Calculate bottleneck fractions of run() time
    for bottleneck in bottleneck_names:
        cumtime_col = f"{bottleneck}_cumtime"
        if cumtime_col in summary.columns:
            summary[f"{bottleneck}_fraction"] = summary[cumtime_col] / summary["rt_run_s"]

    agg_dict = {}

    metric_columns = BASE_SUMMARIZE_COLUMNS + config.metric_columns
    fraction_columns = [f"{bn}_fraction" for bn in bottleneck_names]

    for col in metric_columns + fraction_columns:
        if col in summary.columns:
            agg_dict[f"{col}_mean"] = (col, "mean")
            agg_dict[f"{col}_median"] = (col, "median")
            agg_dict[f"{col}_std"] = (col, "std")
            agg_dict[f"{col}_min"] = (col, "min")
            agg_dict[f"{col}_max"] = (col, "max")

    summary = summary.groupby("model_spec").agg(**agg_dict).reset_index()

    # Fill NaN values in std columns with 0 (occurs with single observations)
    std_cols = [col for col in summary.columns if col.endswith("_std")]
    summary[std_cols] = summary[std_cols].fillna(0.0)

    # Calculate percent differences from baseline (median values)
    baseline_mask = summary["model_spec"].str.endswith(BASELINE)
    median_cols = [col for col in summary.columns if col.endswith("_median")]

    for median_col in median_cols:
        baseline_value = summary.loc[baseline_mask, median_col].values[0]
        pdiff_col = median_col.replace("_median", "_pdiff")
        summary[pdiff_col] = (summary[median_col] - baseline_value) / baseline_value * 100

    # Move the baseline row to the top
    summary = pd.concat(
        [
            summary.loc[summary["model_spec"].str.endswith(BASELINE)],
            summary.loc[~summary["model_spec"].str.endswith(BASELINE)],
        ]
    ).reset_index(drop=True)

    # Add model col for readability
    value_cols = [col for col in summary.columns if col != "model_spec"]
    summary["model"] = (
        summary["model_spec"]
        .str.split("/")
        .str[-1]
        .str.replace(".yaml", "")
        .str.replace("model_spec_", "")
    )
    summary = summary[["model_spec", "model"] + value_cols]
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved summary.csv")

    if summary.isna().any().any():
        logger.warning("Unexpected NaNs found in summary data.")

    return summary


def run_summarize_analysis(
    benchmark_results_filepath: Path, config: ExtractionConfig | None = None
) -> None:
    """Main function to run full summarize analysis pipeline.

    Parameters
    ----------
    benchmark_results_filepath
        Path to benchmark_results.csv file.
    config
        Extraction configuration. If None, uses default.

    """
    if config is None:
        config = ExtractionConfig()

    output_dir = benchmark_results_filepath.parent
    print(f"\nProcessing benchmark results from {benchmark_results_filepath}")
    print(f"Summarizing results to {output_dir}\n")

    raw = pd.read_csv(benchmark_results_filepath)
    if raw.isna().any().any():
        raise ValueError("NaNs found in raw data.")

    summary = summarize(raw, output_dir, config)

    create_figures(
        summary, output_dir, "performance_analysis", "rt_s", "mem_mb", "rt_s_pdiff"
    )

    create_figures(
        summary,
        output_dir,
        "runtime_analysis_setup",
        "rt_setup_s",
        None,
        "rt_setup_s_pdiff",
    )
    create_figures(
        summary,
        output_dir,
        "runtime_analysis_initialize_simulants",
        "rt_initialize_simulants_s",
        None,
        "rt_initialize_simulants_s_pdiff",
    )
    create_figures(
        summary, output_dir, "runtime_analysis_run", "rt_run_s", None, "rt_run_s_pdiff"
    )
    create_figures(
        summary,
        output_dir,
        "runtime_analysis_finalize",
        "rt_finalize_s",
        None,
        "rt_finalize_s_pdiff",
    )
    create_figures(
        summary,
        output_dir,
        "runtime_analysis_report",
        "rt_report_s",
        None,
        "rt_report_s_pdiff",
    )
    create_figures(
        summary,
        output_dir,
        "runtime_analysis_non_run",
        "rt_non_run_s",
        None,
        "rt_non_run_s_pdiff",
    )

    create_figures(
        summary,
        output_dir,
        "bottleneck_runtime_analysis_gather_results",
        "gather_results_cumtime",
        None,
        "gather_results_cumtime_pdiff",
    )
    create_figures(
        summary,
        output_dir,
        "bottleneck_runtime_analysis_pipeline_call",
        "pipeline_call_cumtime",
        None,
        "pipeline_call_cumtime_pdiff",
    )
    create_figures(
        summary,
        output_dir,
        "bottleneck_runtime_analysis_population_get",
        "population_get_cumtime",
        None,
        "population_get_cumtime_pdiff",
    )

    plot_bottleneck_fractions(summary, output_dir, "median")

    print("\n*** FINISHED ***")

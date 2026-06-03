"""
Plot one or more metrics from a wandb run as line charts.

Usage:
    python scripts/plot_wandb_metrics.py \
        --run_id nqfpdn2u \
        --metrics "reward_extra/acc/mean" "critic/rewards/mean" \
        --output figures/merged_run_metrics.png

All config lives in the argparse section below; the rest is reusable.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def fetch_run_history(entity: str, project: str, run_id: str, metrics: list[str],
                      timeout: int = 120, max_retries: int = 3) -> pd.DataFrame:
    api = wandb.Api(timeout=timeout)
    run = api.run(f"{entity}/{project}/{run_id}")
    total = run.lastHistoryStep + 1
    for attempt in range(1, max_retries + 1):
        try:
            df = run.history(samples=total + 100, keys=["_step"] + metrics, pandas=True)
            df = df.sort_values("_step").reset_index(drop=True)
            return df, run.name
        except Exception as e:
            if attempt == max_retries:
                raise
            print(f"Attempt {attempt} failed ({e}), retrying...")
    raise RuntimeError("unreachable")


def smooth(series: pd.Series, window: int) -> pd.Series:
    """Simple rolling-mean smoothing; returns original if window <= 1."""
    if window <= 1:
        return series
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_metrics(
    df: pd.DataFrame,
    metrics: list[str],
    run_name: str,
    output_path: str,
    smooth_window: int = 1,
    labels: list[str] | None = None,
    title: str | None = None,
    figsize: tuple = (10, 5),
    split_step: int | None = None,
    forecast_steps: int = 0,
    forecast_noise: float = 0.02,
    extra_series: list[tuple] | None = None,
    colors: list[str] | None = None,
    alpha: float = 1.0,
):
    """
    Plot multiple metrics against step on a single axes.

    Args:
        df:             DataFrame with '_step' column and one column per metric.
        metrics:        List of column names to plot.
        run_name:       Used as default title.
        output_path:    Path to save the figure (PNG/PDF/SVG).
        smooth_window:  Rolling-mean window size (1 = no smoothing).
        labels:         Custom legend labels; defaults to metric names.
        title:          Figure title; defaults to run_name.
        figsize:        Matplotlib figure size.
        split_step:     If given, draw a vertical dashed line at this step
                        to mark where the two runs were joined.
        forecast_steps: Number of future steps to forecast via linear fit.
        forecast_noise: Std dev of Gaussian noise added to forecast (relative
                        to the y range of each metric).
        extra_series:   Optional list of (x, y_raw, label) tuples from other
                        runs to overlay on the same axes.
    """
    if labels is None:
        labels = metrics

    fig, ax = plt.subplots(figsize=figsize)

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        if metric not in df.columns:
            print(f"Warning: metric '{metric}' not found, skipping.")
            continue
        color = colors[i] if colors and i < len(colors) else None
        x = df["_step"]
        y = smooth(df[metric], smooth_window)
        line, = ax.plot(x, y, linewidth=1.5, label=label, color=color, alpha=alpha)
        color = line.get_color()
        # faint raw line when smoothing is on
        if smooth_window > 1:
            ax.plot(x, df[metric], alpha=0.2, linewidth=0.6, color=color)

        # --- linear forecast ---
        if forecast_steps > 0:
            valid = y.dropna()
            if len(valid) >= 2:
                x_fit = x.loc[valid.index].values
                y_fit = valid.values
                coeffs = np.polyfit(x_fit, y_fit, 1)
                last_step = int(x_fit[-1])
                x_pred = np.arange(last_step, last_step + forecast_steps + 1)
                y_pred = np.polyval(coeffs, x_pred)
                y_range = y_fit.max() - y_fit.min() if y_fit.max() != y_fit.min() else abs(y_fit.mean()) + 1e-6
                noise = np.random.randn(len(x_pred)) * forecast_noise * y_range
                ax.plot(x_pred, y_pred + noise, linewidth=1.5, linestyle="--",
                        color=color, alpha=0.8, label=f"{label} (forecast)")

    # --- overlay extra runs ---
    for (x_extra, y_raw_extra, label_extra) in (extra_series or []):
        y_extra = smooth(pd.Series(y_raw_extra), smooth_window)
        line, = ax.plot(x_extra, y_extra, linewidth=1.5, linestyle="--", label=label_extra)
        if smooth_window > 1:
            ax.plot(x_extra, y_raw_extra, alpha=0.2, linewidth=0.6, color=line.get_color())

    if split_step is not None:
        ax.axvline(x=split_step, color="gray", linestyle="--", linewidth=1,
                   label=f"run join (step {split_step})")

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title or run_name, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics from a wandb run.")
    parser.add_argument("--entity",   default="wxk20040223-tsinghua-university")
    parser.add_argument("--project",  default="verl_grpo_rubrics_verif")
    parser.add_argument("--run_id",   default="nqfpdn2u",
                        help="wandb run ID")
    parser.add_argument("--metrics",  nargs="+",
                        default=["reward_extra/acc/mean", "critic/rewards/mean"],
                        help="Metric keys to plot")
    parser.add_argument("--labels",   nargs="+", default=None,
                        help="Legend labels (same order as --metrics)")
    parser.add_argument("--smooth",   type=int, default=5,
                        help="Rolling-mean window (1 = off)")
    parser.add_argument("--split_step", type=int, default=None,
                        help="Step where runs were joined; draws a vertical line (default: off)")
    parser.add_argument("--output",   default="figures/merged_run_metrics.png",
                        help="Output file path")
    parser.add_argument("--title",    default=None)
    parser.add_argument("--forecast_steps", type=int, default=0,
                        help="Number of future steps to forecast (0 = off)")
    parser.add_argument("--forecast_noise", type=float, default=0.02,
                        help="Noise std dev relative to each metric's y range")
    parser.add_argument("--colors",  nargs="+", default=None,
                        help="Line colors (same order as --metrics), e.g. blue green")
    parser.add_argument("--alpha",   type=float, default=1.0,
                        help="Line opacity (0.0–1.0)")
    parser.add_argument("--extra_runs", nargs="+", default=[],
                        metavar="RUN_ID:METRIC",
                        help="Extra run/metric pairs to overlay, e.g. "
                             "yp2gao0s:reward_extra/acc/mean")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Fetching run {args.run_id} ...")
    df, run_name = fetch_run_history(
        args.entity, args.project, args.run_id, args.metrics
    )
    print(f"  {len(df)} steps, columns: {list(df.columns)}")

    # --- fetch extra overlay runs ---
    extra_series = []
    for spec in args.extra_runs:
        run_id_extra, metric_extra = spec.split(":", 1)
        print(f"Fetching extra run {run_id_extra} / {metric_extra} ...")
        df_extra, name_extra = fetch_run_history(
            args.entity, args.project, run_id_extra, [metric_extra]
        )
        print(f"  {len(df_extra)} steps")
        if metric_extra in df_extra.columns:
            x_e = df_extra["_step"].values
            y_e = df_extra[metric_extra].values
            extra_series.append((x_e, y_e, f"{name_extra} / {metric_extra}"))
        else:
            print(f"  Warning: metric '{metric_extra}' not in run {run_id_extra}, skipping.")

    plot_metrics(
        df=df,
        metrics=args.metrics,
        run_name=run_name,
        output_path=args.output,
        smooth_window=args.smooth,
        labels=args.labels,
        title=args.title,
        split_step=args.split_step,
        forecast_steps=args.forecast_steps,
        forecast_noise=args.forecast_noise,
        extra_series=extra_series,
        colors=args.colors,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()

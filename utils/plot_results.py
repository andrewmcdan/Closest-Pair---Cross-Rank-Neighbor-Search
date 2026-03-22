#!/usr/bin/env python3
"""
Create benchmark plots from closest-pair results CSV.

Outputs PNG plots to ./data by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _pick_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"Missing expected columns. Tried: {list(candidates)}")
    return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    points_col = _pick_column(df, ["reported_points", "requested_points"])
    threads_col = _pick_column(df, ["reported_threads", "requested_threads"])
    crns_serial_col = _pick_column(df, ["crns_serial_ms", "custom_algorithm_serial_ms"])
    crns_parallel_col = _pick_column(
        df,
        ["crns_parallel_ms", "custom_algorithm_parallel_ms", "custom_algorithm_ms"],
    )
    dnc_serial_col = _pick_column(df, ["divide_conquer_serial_ms"], required=False)
    dnc_parallel_col = _pick_column(df, ["divide_conquer_parallel_ms"], required=False)

    rename_map = {
        points_col: "points",
        threads_col: "threads",
        crns_serial_col: "crns_serial_ms",
        crns_parallel_col: "crns_parallel_ms",
    }
    if dnc_serial_col:
        rename_map[dnc_serial_col] = "dnc_serial_ms"
    if dnc_parallel_col:
        rename_map[dnc_parallel_col] = "dnc_parallel_ms"

    df = df.rename(columns=rename_map)
    _coerce_numeric(
        df,
        ["points", "threads", "crns_serial_ms", "crns_parallel_ms", "dnc_serial_ms", "dnc_parallel_ms"],
    )

    df = df.dropna(subset=["points", "threads", "crns_serial_ms", "crns_parallel_ms"])
    df["points"] = df["points"].astype(int)
    df["threads"] = df["threads"].astype(int)
    return df


def _build_runtime_curves(df: pd.DataFrame, thread_focus: int) -> pd.DataFrame:
    focused = df[df["threads"] == thread_focus].copy()
    if focused.empty:
        raise ValueError(f"No rows found for thread count {thread_focus}.")

    series = [
        ("CRNS serial", "crns_serial_ms"),
        ("CRNS parallel", "crns_parallel_ms"),
    ]
    if "dnc_serial_ms" in focused.columns:
        series.append(("D&C serial", "dnc_serial_ms"))
    if "dnc_parallel_ms" in focused.columns:
        series.append(("D&C parallel", "dnc_parallel_ms"))

    records: list[dict[str, float | int | str]] = []
    for label, col in series:
        tmp = (
            focused[["points", col]]
            .dropna()
            .groupby("points", as_index=False)
            .median(numeric_only=True)
            .rename(columns={col: "runtime_ms"})
        )
        for _, r in tmp.iterrows():
            records.append(
                {
                    "points": int(r["points"]),
                    "algorithm": label,
                    "runtime_ms": float(r["runtime_ms"]),
                }
            )
    return pd.DataFrame.from_records(records)


def _plot_runtime_by_points(runtime_df: pd.DataFrame, out_path: Path, thread_focus: int) -> None:
    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=runtime_df,
        x="points",
        y="runtime_ms",
        hue="algorithm",
        marker="o",
        linewidth=2.2,
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points (log scale)")
    plt.ylabel("Runtime (ms, log scale)")
    plt.title(f"Closest-Pair Runtime vs Problem Size (threads={thread_focus})")
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _build_parallel_speedup(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for points, group in df.groupby("points"):
        t1 = group[group["threads"] == 1]
        if t1.empty:
            continue

        base_crns = float(t1["crns_parallel_ms"].median())
        base_dnc = None
        if "dnc_parallel_ms" in group.columns and not t1["dnc_parallel_ms"].dropna().empty:
            base_dnc = float(t1["dnc_parallel_ms"].median())

        agg = group.groupby("threads", as_index=False).median(numeric_only=True)
        for _, row in agg.iterrows():
            threads = int(row["threads"])
            crns_speedup = base_crns / float(row["crns_parallel_ms"])
            records.append(
                {
                    "points": int(points),
                    "threads": threads,
                    "algorithm": "CRNS parallel",
                    "speedup_vs_1t": crns_speedup,
                }
            )
            if base_dnc is not None and not np.isnan(row.get("dnc_parallel_ms", np.nan)):
                dnc_speedup = base_dnc / float(row["dnc_parallel_ms"])
                records.append(
                    {
                        "points": int(points),
                        "threads": threads,
                        "algorithm": "D&C parallel",
                        "speedup_vs_1t": dnc_speedup,
                    }
                )
    return pd.DataFrame.from_records(records)


def _plot_parallel_speedup(speedup_df: pd.DataFrame, out_path: Path) -> None:
    if speedup_df.empty:
        return

    plot_df = speedup_df.copy()
    plot_df["series"] = plot_df["algorithm"] + " (n=" + plot_df["points"].astype(str) + ")"

    plt.figure(figsize=(11, 7))
    sns.lineplot(
        data=plot_df,
        x="threads",
        y="speedup_vs_1t",
        hue="series",
        marker="o",
        linewidth=2.0,
    )
    plt.xlabel("Threads")
    plt.ylabel("Speedup vs 1 thread")
    plt.title("Parallel Scaling: Speedup vs Thread Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _build_ratio_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if "dnc_parallel_ms" not in df.columns:
        return pd.DataFrame()

    agg = df.groupby(["points", "threads"], as_index=False).median(numeric_only=True)
    agg = agg.dropna(subset=["crns_parallel_ms", "dnc_parallel_ms"])
    agg["ratio_dnc_over_crns"] = agg["dnc_parallel_ms"] / agg["crns_parallel_ms"]
    heat = agg.pivot(index="points", columns="threads", values="ratio_dnc_over_crns")
    return heat.sort_index()


def _plot_ratio_heatmap(heat: pd.DataFrame, out_path: Path) -> None:
    if heat.empty:
        return

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "D&C parallel time / CRNS parallel time"},
    )
    plt.title("Relative Efficiency Heatmap (>1 means CRNS parallel is faster)")
    plt.xlabel("Threads")
    plt.ylabel("Points")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def _build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["points", "threads"], as_index=False).median(numeric_only=True)
    if "dnc_serial_ms" in agg.columns:
        agg["serial_ratio_dnc_over_crns"] = agg["dnc_serial_ms"] / agg["crns_serial_ms"]
    if "dnc_parallel_ms" in agg.columns:
        agg["parallel_ratio_dnc_over_crns"] = agg["dnc_parallel_ms"] / agg["crns_parallel_ms"]
    return agg.sort_values(["points", "threads"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CRNS efficiency from benchmark CSV.")
    parser.add_argument("--input", default="results.csv", help="Input CSV path.")
    parser.add_argument("--outdir", default="data", help="Directory for generated plots.")
    parser.add_argument(
        "--thread-focus",
        type=int,
        default=None,
        help="Thread count used for the runtime-vs-points figure. Defaults to max thread count found.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    df = _prepare_dataframe(input_path)
    thread_focus = int(df["threads"].max()) if args.thread_focus is None else args.thread_focus

    runtime_df = _build_runtime_curves(df, thread_focus=thread_focus)
    _plot_runtime_by_points(runtime_df, outdir / f"runtime_by_points_t{thread_focus}.png", thread_focus)

    speedup_df = _build_parallel_speedup(df)
    _plot_parallel_speedup(speedup_df, outdir / "parallel_speedup_vs_threads.png")

    heat_df = _build_ratio_heatmap(df)
    _plot_ratio_heatmap(heat_df, outdir / "crns_vs_dnc_parallel_ratio_heatmap.png")

    summary = _build_summary_table(df)
    summary.to_csv(outdir / "summary_median_by_points_threads.csv", index=False)

    print(f"Loaded rows: {len(df)}")
    print(f"Output directory: {outdir}")
    print("Generated files:")
    print(f"- {outdir / f'runtime_by_points_t{thread_focus}.png'}")
    print(f"- {outdir / 'parallel_speedup_vs_threads.png'}")
    if not heat_df.empty:
        print(f"- {outdir / 'crns_vs_dnc_parallel_ratio_heatmap.png'}")
    print(f"- {outdir / 'summary_median_by_points_threads.csv'}")


if __name__ == "__main__":
    main()

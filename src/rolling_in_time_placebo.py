"""
Rolling in-time placebo: SCM gap path for every possible pre-treatment year.

Addresses Reviewer Comment 3.2: "perform a rolling in-time placebo analysis for
every possible year in the pre-treatment period (e.g., 2000-2009)" to demonstrate
that the observed divergence is unique to the actual treatment year and not
pre-existing diverging trends.

For each candidate treatment year T we re-estimate SCM with pre-treatment data
through T-1 and plot the resulting gap path. Placebo years (T != actual treatment)
should show no sustained divergence; the actual treatment year should stand out.
"""
import os
from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth

import nz_util
from scm_common import (
    ANALYSIS_END_YEAR,
    CHILE_TREATED,
    NZ_TREATED,
    build_chile_dataprep,
    build_nz_dataprep,
    rmspe,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

NZ_ACTUAL_TREATMENT = 2011
NZ_PRE_YEARS = list(range(2000, 2010))  # 2000-2009, every pre-treatment year
CHILE_ACTUAL_TREATMENT = 2010
CHILE_PRE_YEARS = list(range(1990, 2010))  # 1990-2009


def _run_rolling_placebo(
    country: str,
    df: pd.DataFrame,
    dataprep_builder: Callable[[pd.DataFrame, int], Dataprep],
    candidate_years: list[int],
    actual_treatment_year: int,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    fit_start_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each candidate year T, fit SCM with treatment_year=T; return gap paths and summary."""
    all_years = list(range(int(df[time_col].min()), min(int(df[time_col].max()) + 1, ANALYSIS_END_YEAR + 1)))
    gap_paths: dict[int, pd.Series] = {}
    summary_rows: list[dict] = []

    for T in candidate_years:
        pre_years = [y for y in all_years if fit_start_year <= y < T]
        if len(pre_years) < 1:
            continue
        dataprep = dataprep_builder(df, T)
        synth = Synth()
        try:
            synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="equal" if country == "New Zealand" else "ols")
        except Exception:
            continue
        z0, z1 = synth.dataprep.make_outcome_mats(time_period=all_years)
        synthetic = pd.Series(np.asarray(synth._synthetic(z0)).flatten().astype(float), index=all_years)
        treated = pd.Series(np.asarray(z1).flatten().astype(float), index=all_years)
        gap_pct = (treated - synthetic) / synthetic.replace(0, np.nan) * 100.0
        gap_paths[T] = gap_pct

        post_mask = (np.array(all_years) >= T) & (np.array(all_years) <= ANALYSIS_END_YEAR)
        pre_mask = (np.array(all_years) >= fit_start_year) & (np.array(all_years) < T)
        rmspe_pre = rmspe(gap_pct.iloc[pre_mask]) if pre_mask.any() else np.nan
        rmspe_post = rmspe(gap_pct.iloc[post_mask]) if post_mask.any() else np.nan
        ratio = rmspe_post / rmspe_pre if np.isfinite(rmspe_pre) and rmspe_pre > 0 else np.nan
        mean_post = float(gap_pct.iloc[post_mask].mean()) if post_mask.any() else np.nan
        summary_rows.append({
            "Country": country,
            "CandidateTreatmentYear": T,
            "IsActual": T == actual_treatment_year,
            "MeanPostGapPct": mean_post,
            "RMSPE_Pre": rmspe_pre,
            "RMSPE_Post": rmspe_post,
            "RMSPE_Ratio": ratio,
        })

    paths_df = pd.DataFrame(gap_paths).reindex(all_years)
    summary_df = pd.DataFrame(summary_rows)
    return paths_df, summary_df


def _plot_rolling_placebo(
    paths_df: pd.DataFrame,
    actual_year: int,
    country_label: str,
    output_path: str,
) -> None:
    """One panel: each line = gap path when treatment is assumed in year T; actual year highlighted."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x_vals = np.asarray(paths_df.index, dtype=float)
    for col in paths_df.columns:
        is_actual = col == actual_year
        ax.plot(
            x_vals,
            paths_df[col].to_numpy(),
            color="red" if is_actual else "gray",
            linewidth=2.5 if is_actual else 0.9,
            alpha=0.9 if is_actual else 0.5,
            label=f"Treatment = {col}" if is_actual else None,
        )
    ax.axvline(actual_year, color="red", linestyle="--", linewidth=1, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap (%)")
    ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    ax.set_title(f"Rolling in-time placebo: {country_label} (each line = gap path if treatment in that year)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_rolling_in_time_placebo(output_dir: str = FIGURES_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    nz_df = nz_util.clean_data_for_synthetic_control().copy()
    nz_df["Tertiary Share"] = nz_df["Tertiary"] / nz_df["Population"]
    chile_df = pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv"))

    nz_candidates = NZ_PRE_YEARS + [NZ_ACTUAL_TREATMENT]
    chile_candidates = CHILE_PRE_YEARS + [CHILE_ACTUAL_TREATMENT]

    nz_paths, nz_summary = _run_rolling_placebo(
        country="New Zealand",
        df=nz_df,
        dataprep_builder=build_nz_dataprep,
        candidate_years=nz_candidates,
        actual_treatment_year=NZ_ACTUAL_TREATMENT,
        unit_col="Region",
        time_col="Year",
        outcome_col="GDP per capita",
        fit_start_year=2000,
    )
    chile_paths, chile_summary = _run_rolling_placebo(
        country="Chile",
        df=chile_df,
        dataprep_builder=build_chile_dataprep,
        candidate_years=chile_candidates,
        actual_treatment_year=CHILE_ACTUAL_TREATMENT,
        unit_col="region_name",
        time_col="year",
        outcome_col="gdp_cap",
        fit_start_year=1990,
    )

    os.makedirs(output_dir, exist_ok=True)
    nz_summary.to_csv(os.path.join(output_dir, "rolling_in_time_placebo_nz_summary.csv"), index=False)
    chile_summary.to_csv(os.path.join(output_dir, "rolling_in_time_placebo_chile_summary.csv"), index=False)
    combined = pd.concat([nz_summary, chile_summary], ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "rolling_in_time_placebo_summary.csv"), index=False)

    _plot_rolling_placebo(
        nz_paths,
        NZ_ACTUAL_TREATMENT,
        "New Zealand (Canterbury)",
        os.path.join(output_dir, "rolling_in_time_placebo_nz.png"),
    )
    _plot_rolling_placebo(
        chile_paths,
        CHILE_ACTUAL_TREATMENT,
        "Chile (Maule)",
        os.path.join(output_dir, "rolling_in_time_placebo_chile.png"),
    )

    # Two-panel figure for manuscript (separate y-axes: NZ ~0–10%, Chile ~-45–5%)
    # Placebo lines: one distinct color per year (by index); actual treatment in red; colorbar with integer ticks.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    cmap = mcm.get_cmap("viridis")
    for ax, paths_df, actual_year, title in [
        (axes[0], nz_paths, NZ_ACTUAL_TREATMENT, "New Zealand (Canterbury)"),
        (axes[1], chile_paths, CHILE_ACTUAL_TREATMENT, "Chile (Maule)"),
    ]:
        placebo_years = sorted(c for c in paths_df.columns if c != actual_year)
        n_placebo = len(placebo_years)
        x_vals = np.asarray(paths_df.index, dtype=float)
        # Plot placebos first (sorted), each with a distinct color by index over full colormap range.
        for i, col in enumerate(placebo_years):
            t = i / max(1, n_placebo - 1)  # 0 to 1 over placebo years
            ax.plot(
                x_vals,
                paths_df[col].to_numpy(),
                color=cmap(t),
                linewidth=0.9,
                alpha=0.75,
            )
        # Plot actual treatment year on top.
        if actual_year in paths_df.columns:
            ax.plot(
                x_vals,
                paths_df[actual_year].to_numpy(),
                color="red",
                linewidth=2.5,
                alpha=0.95,
            )
        ax.axvline(actual_year, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Year")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(mtick.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
        if n_placebo > 0:
            norm = mcolors.Normalize(vmin=min(placebo_years), vmax=max(placebo_years))
            sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.7, aspect=25)
            cbar.set_label("Placebo treatment year", fontsize=8)
            cbar.set_ticks(placebo_years)
            cbar.ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%d"))
    axes[0].set_ylabel("Gap (%)")
    fig.suptitle("Rolling in-time placebo: gap path by candidate treatment year (colorbar) vs actual treatment year (red)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(output_dir, "rolling_in_time_placebo_paths.png"), dpi=220)
    plt.close(fig)

    return combined, nz_paths


if __name__ == "__main__":
    summary, _ = run_rolling_in_time_placebo()
    print(summary.to_string(index=False))

import os

import pandas as pd
from matplotlib import pyplot as plt
from typing import Optional
import matplotlib.ticker as mtick

# Project root and canonical figures output directory (used by main.tex)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)


def synth_plot(synth, time_period, treatment_time) -> None:
        """Plot the outcome variable over time for the treated unit and the
        synthetic control.
        """
        Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = synth._synthetic(Z0=Z0)
        plt.plot(Z1, color="red", linewidth=1.5, label=Z1.name)
        plt.plot(
            ts_synthetic,
            color="red",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic",
        )
        if synth.dataprep is not None:
            plt.ylabel(synth.dataprep.dependent)
        if treatment_time:
            plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        plt.show()


def synth_plot_nz(synth, time_period, treatment_time, filename=None) -> None:
        """Plot the outcome variable over time for the treated unit and the
        synthetic control.
        """
        Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = synth._synthetic(Z0=Z0)
        plt.plot(Z1/1000, color="red", linewidth=1.5, label=Z1.name)
        plt.plot(
            ts_synthetic/1000,
            color="red",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic Control",
        )
        plt.ylabel('GDP per Capita (Thousands of NZD)')
        if treatment_time:
            plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename))
        plt.show()
        rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
        return rv



def synth_plot_sector(synth, time_period, treatment_time, filename=None, sector="NZD") -> None:
    """Plot the outcome variable over time for the treated unit and the
    synthetic control, with y-axis expressed as a percentage including tick marks.
    """
    Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
    ts_synthetic = synth._synthetic(Z0=Z0)
    
    # Convert to percentage
    Z1_percent = Z1 * 100
    ts_synthetic_percent = ts_synthetic * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(Z1_percent, color="red", linewidth=1.5, label=Z1.name)
    ax.plot(
        ts_synthetic_percent,
        color="red",
        linewidth=1,
        linestyle="dashed",
        label="Synthetic Control",
    )
    
    ax.set_ylabel(f'Sectoral share of GDP ({sector})')
    ax.set_ylim(bottom=0)  # Set y-axis to start at 0%
    
    # Format y-axis ticks as percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    if treatment_time:
        ax.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    
    ax.legend()
    #ax.set_title(f'Sectoral Share of GDP for {sector}')
    
    if filename:
        plt.savefig(os.path.join(FIGURES_DIR, filename))
    
    plt.show()
    
    rv = pd.concat([ts_synthetic_percent, Z1_percent], axis=1).rename(columns={0: 'Synthetic Control'})
    return rv

def synth_plot_chile(synth, time_period, treatment_time, filename=None) -> None:
        """Plot the outcome variable over time for the treated unit and the
        synthetic control.
        """
        Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
        ts_synthetic = synth._synthetic(Z0=Z0)
        plt.plot(Z1/1000000, color="red", linewidth=1.5, label=Z1.name)
        plt.plot(
            ts_synthetic/1000000,
            color="red",
            linewidth=1,
            linestyle="dashed",
            label="Synthetic Control",
        )
        plt.ylabel('GDP per Capita (Millions of CLP)')
        if treatment_time:
            plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename))
        plt.show()
        rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
        return rv


def synth_plot_population(synth, time_period, treatment_time, filename=None, divide_by=1000, ylabel='Population (thousands)') -> None:
    """Plot population or similar outcome for treated unit and synthetic control."""
    Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
    ts_synthetic = synth._synthetic(Z0=Z0)
    plt.plot(Z1 / divide_by, color="red", linewidth=1.5, label=Z1.name)
    plt.plot(
        ts_synthetic / divide_by,
        color="red",
        linewidth=1,
        linestyle="dashed",
        label="Synthetic Control",
    )
    plt.ylabel(ylabel)
    if treatment_time:
        plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.legend()
    if filename:
        plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.show()
    rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
    return rv


def synth_plot_gdp_total(synth, time_period, treatment_time, filename=None, divide_by=1e9, ylabel='Total GDP (billions)') -> None:
    """Plot total GDP or similar outcome for treated unit and synthetic control."""
    Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
    ts_synthetic = synth._synthetic(Z0=Z0)
    plt.plot(Z1 / divide_by, color="red", linewidth=1.5, label=Z1.name)
    plt.plot(
        ts_synthetic / divide_by,
        color="red",
        linewidth=1,
        linestyle="dashed",
        label="Synthetic Control",
    )
    plt.ylabel(ylabel)
    if treatment_time:
        plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.legend()
    if filename:
        plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.show()
    rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
    return rv


def gap_plot(
    synth,
    time_period,
    treatment_time: Optional[int] = None,
    filename: Optional[str] = None,
) -> None:
    """Plot the percent gap (actual - synthetic) / synthetic * 100 for the treated unit only."""
    Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
    ts_synthetic = synth._synthetic(Z0=Z0)
    gap_pct = (Z1 - ts_synthetic) / ts_synthetic * 100
    gap_pct = gap_pct[gap_pct.index.isin(time_period)]
    plt.plot(gap_pct, color="red", linewidth=1.5, label="Gap (%)")
    if treatment_time:
        plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.axhline(y=0, color="black")
    plt.ylabel("Gap (%)")
    plt.legend()
    if filename:
        plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.show()


def placebo_plot(
        placebo,
        time_period,
        grid: bool = False,
        treatment_time: Optional[int] = None,
        mspe_threshold: Optional[float] = None,
        exclude_units: Optional[list] = None,
        divide_by: Optional[int] = 1000,
        y_axis_label: Optional[str] = None,
        y_axis_limit: Optional[int] = 10,
        filename: Optional[str] = None
    ):
        """Plot the gaps between the treated unit and the synthetic control
        for each placebo test.

        Parameters
        ----------
        time_period : Iterable | pandas.Series | dict, optional
            Time range to plot, if none is supplied then the time range used
            is the time period over which the optimisation happens, by default
            None
        grid : bool, optional
            Whether or not to plot a grid, by default True
        treatment_time : int, optional
            If supplied, plot a vertical line at the time period that the
            treatment time occurred, by default None
        mspe_threshold : float, optional
            Remove any non-treated units whose MSPE pre-treatment is :math:`>`
            mspe_threshold :math:`\\times` the MSPE of the treated unit pre-treatment.
            This serves to exclude any non-treated units whose synthetic control
            had a poor pre-treatment match to the actual relative to how the
            actual treated unit matched pre-treatment.

        Raises
        ------
        ValueError
            if no placebo test has been run yet
        ValueError
            if `mspe_threshold` is supplied but `treatment_year` is not.
        """
        if placebo.gaps is None:
            raise ValueError("No gaps available; run a placebo test first.")
        time_period = time_period if time_period is not None else placebo.time_optimize_ssr

        gaps = placebo.gaps.drop(columns=exclude_units) if exclude_units else placebo.gaps

        if mspe_threshold:
            if not treatment_time:
                raise ValueError("Need `treatment_time` to use `mspe_threshold`.")
            pre_mspe = gaps.loc[:treatment_time].pow(2).sum(axis=0)
            pre_mspe_treated = placebo.treated_gap.loc[:treatment_time].pow(2).sum(axis=0)
            keep = pre_mspe[pre_mspe < mspe_threshold * pre_mspe_treated].index
            placebo_gaps = gaps[gaps.index.isin(time_period)][keep]
        else:
            placebo_gaps = gaps[gaps.index.isin(time_period)]
        plt.plot(placebo_gaps/divide_by, color="black", alpha=0.1)
        plt.plot(placebo.treated_gap[placebo.treated_gap.index.isin(time_period)]/divide_by, color="red", alpha=1.0)
        if treatment_time:
            plt.axvline(x=treatment_time, ymin=0.05, ymax=0.95, color='black', linestyle="dashed")
        plt.axhline(y=0, color="black")
        plt.grid(grid)
        plt.ylim(-y_axis_limit, y_axis_limit)
        plt.ylabel(y_axis_label)
        if filename:
            plt.savefig(os.path.join(FIGURES_DIR, filename))
        plt.show()


def in_time_placebo_plot(
    placebo_gaps: dict,
    actual_gap: pd.Series,
    time_period,
    actual_treatment_year: int,
    divide_by: float = 1000,
    y_axis_label: str = "Gap",
    y_axis_limit: Optional[float] = 10,
    filename: Optional[str] = None,
) -> None:
    """Plot in-time placebo gaps (gray) vs actual treated gap (red).
    placebo_gaps: dict mapping fake_treatment_year -> gap Series
    actual_gap: gap Series from the real treatment analysis
    """
    time_list = list(time_period)
    for _, gap_series in placebo_gaps.items():
        g = gap_series[gap_series.index.isin(time_list)]
        if len(g) > 0:
            plt.plot(g.index, g.values / divide_by, color="black", alpha=0.15, linewidth=0.8)
    g_act = actual_gap[actual_gap.index.isin(time_list)]
    if len(g_act) > 0:
        plt.plot(g_act.index, g_act.values / divide_by, color="red", linewidth=1.5, label="Treated")
    plt.axvline(x=actual_treatment_year, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.axhline(y=0, color="black")
    plt.ylabel(y_axis_label)
    plt.ylim(-y_axis_limit, y_axis_limit)
    plt.legend()
    if filename:
        plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.show()
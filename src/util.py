import os

import pandas as pd
import numpy as np
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


def compute_uniform_confidence_bands(
    placebo,
    treatment_time: int,
    alpha: float = 0.1,
    mspe_threshold: Optional[float] = None,
    exclude_units: Optional[list] = None,
) -> tuple:
    """
    Compute uniform confidence bands for SCM treatment effects following 
    Firpo & Possebom (2018).
    
    The approach constructs confidence bands that are valid uniformly across all
    post-treatment periods by using the maximum absolute standardized gap from
    the placebo distribution.
    
    Parameters
    ----------
    placebo : PlaceboTest
        A fitted placebo test object from pysyncon
    treatment_time : int
        The period when treatment occurred
    alpha : float, optional
        Significance level (default 0.1 for 90% confidence)
    mspe_threshold : float, optional
        Remove any non-treated units whose MSPE pre-treatment is >
        mspe_threshold × the MSPE of the treated unit pre-treatment.
    exclude_units : list, optional
        List of unit names to exclude from placebo distribution
        
    Returns
    -------
    tuple
        (lower_band, upper_band) as pandas Series indexed by time period
    """
    if placebo.gaps is None:
        raise ValueError("No gaps available; run a placebo test first.")
    
    gaps = placebo.gaps.copy()
    if exclude_units:
        gaps = gaps.drop(columns=exclude_units, errors='ignore')
    
    # Filter placebos by pre-treatment MSPE if threshold specified
    if mspe_threshold:
        pre_mspe = gaps.loc[:treatment_time].pow(2).sum(axis=0)
        pre_mspe_treated = placebo.treated_gap.loc[:treatment_time].pow(2).sum(axis=0)
        keep = pre_mspe[pre_mspe < mspe_threshold * pre_mspe_treated].index
        gaps = gaps[keep]
    
    # Compute RMSPE for standardization (pre-treatment period)
    pre_rmspe = np.sqrt(gaps.loc[:treatment_time].pow(2).mean(axis=0))
    pre_rmspe_treated = np.sqrt(placebo.treated_gap.loc[:treatment_time].pow(2).mean(axis=0))
    
    # Standardize gaps by pre-treatment RMSPE
    standardized_gaps = gaps.div(pre_rmspe, axis=1)
    
    # For post-treatment period, compute maximum absolute standardized gap for each placebo
    post_treatment_gaps = standardized_gaps.loc[treatment_time:]
    max_abs_gaps = post_treatment_gaps.abs().max(axis=0)
    
    # Compute critical value as the (1-alpha) quantile of the max distribution
    critical_value = np.quantile(max_abs_gaps, 1 - alpha)
    
    # Construct uniform confidence bands: treated_gap ± critical_value × RMSPE_treated
    treated_gap_post = placebo.treated_gap.loc[treatment_time:]
    band_width = critical_value * pre_rmspe_treated
    
    lower_band = treated_gap_post - band_width
    upper_band = treated_gap_post + band_width
    
    return lower_band, upper_band


def gap_plot_with_uniform_ci(
    synth,
    placebo,
    time_period,
    treatment_time: int,
    alpha: float = 0.1,
    mspe_threshold: Optional[float] = None,
    exclude_units: Optional[list] = None,
    divide_by: float = 1.0,
    y_axis_label: str = "Gap",
    filename: Optional[str] = None,
) -> None:
    """
    Plot the gap between treated and synthetic control with uniform confidence bands.
    
    Parameters
    ----------
    synth : Synth
        Fitted synthetic control object
    placebo : PlaceboTest
        Fitted placebo test object
    time_period : range or list
        Time periods to plot
    treatment_time : int
        The period when treatment occurred
    alpha : float, optional
        Significance level for confidence bands (default 0.1 for 90% CI)
    mspe_threshold : float, optional
        MSPE filtering threshold for placebos
    exclude_units : list, optional
        List of units to exclude from placebo distribution
    divide_by : float, optional
        Scaling factor for y-axis (default 1.0)
    y_axis_label : str, optional
        Label for y-axis
    filename : str, optional
        If provided, save figure to this filename
    """
    # Compute the treated gap
    Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=time_period)
    ts_synthetic = synth._synthetic(Z0=Z0)
    gap = Z1 - ts_synthetic
    gap = gap[gap.index.isin(time_period)]
    
    # Compute uniform confidence bands
    lower_band, upper_band = compute_uniform_confidence_bands(
        placebo=placebo,
        treatment_time=treatment_time,
        alpha=alpha,
        mspe_threshold=mspe_threshold,
        exclude_units=exclude_units,
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot confidence band
    post_treatment_periods = [t for t in time_period if t >= treatment_time]
    ax.fill_between(
        post_treatment_periods,
        lower_band / divide_by,
        upper_band / divide_by,
        alpha=0.2,
        color='gray',
        label=f'{int((1-alpha)*100)}% Uniform CI'
    )
    
    # Plot the gap
    ax.plot(gap / divide_by, color="red", linewidth=1.5, label="Treatment Gap")
    
    # Add reference lines
    ax.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    ax.axhline(y=0, color="black", linewidth=0.8)
    
    ax.set_ylabel(y_axis_label)
    ax.legend()
    ax.grid(alpha=0.3)
    
    if filename:
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
    plt.show()
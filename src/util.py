import pandas as pd
from matplotlib import pyplot as plt
from typing import Optional


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
            plt.savefig(f"../output/{filename}")
        plt.show()
        rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
        return rv

def synth_plot_sector(synth, time_period, treatment_time, filename=None, sector="NZD") -> None:
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
            label="Synthetic Control",
        )
        plt.ylabel(f'GDP per Capita ({sector})')
        if treatment_time:
            plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
        plt.legend()
        if filename:
            plt.savefig(f"../output/{filename}")
        plt.show()
        rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
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
            plt.savefig(f"../output/{filename}")
        plt.show()
        rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
        return rv


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
            plt.savefig(f"../output/{filename}")
        plt.show()
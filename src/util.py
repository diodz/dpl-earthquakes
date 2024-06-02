import pandas as pd
from matplotlib import pyplot as plt


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


def synth_plot_nz(synth, time_period, treatment_time) -> None:
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
        plt.show()
        rv = pd.concat([ts_synthetic, Z1], axis=1).rename(columns={0: 'Synthetic Control'})
        return rv
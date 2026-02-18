"""
Generate main SCM figures for the article: GDP paths, gap, placebos, and jackknife.

Replaces the logic previously in notebooks/Maule SCM.ipynb and notebooks/Canterbury SCM.ipynb.
All figures are written to article_assets/ (used by main.tex).
"""
from __future__ import annotations

import os
import warnings

# Non-interactive backend for batch runs (e.g. run_analysis.py)
import matplotlib
matplotlib.use("Agg")

from pysyncon import Synth
from pysyncon.utils import PlaceboTest

import nz_util
import process_chile_gdp_data as pcd
import util
from scm_common import (
    ANALYSIS_END_YEAR,
    CHILE_ACTUAL_TREATMENT,
    NZ_ACTUAL_TREATMENT,
    build_chile_dataprep,
    build_nz_dataprep,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Time ranges for plotting (aligned with analysis end)
NZ_TIME = range(2000, ANALYSIS_END_YEAR + 1)
CHILE_TIME = range(1990, ANALYSIS_END_YEAR + 1)


def _run_nz(output_dir: str = FIGURES_DIR) -> None:
    """Build NZ dataprep, fit SCM, run placebo test, and save all NZ figures."""
    warnings.simplefilter("ignore", category=FutureWarning)
    df = nz_util.clean_data_for_synthetic_control().copy()
    df["Tertiary Share"] = df["Tertiary"] / df["Population"]
    dataprep = build_nz_dataprep(df, NZ_ACTUAL_TREATMENT)

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="equal")

    util.synth_plot_nz(
        synth,
        time_period=NZ_TIME,
        treatment_time=NZ_ACTUAL_TREATMENT,
        filename="nz_gdp_paths.png",
    )
    util.gap_plot(
        synth,
        time_period=NZ_TIME,
        treatment_time=NZ_ACTUAL_TREATMENT,
        filename="nz_gap.png",
    )

    placebo_test = PlaceboTest()
    placebo_test.fit(
        dataprep=dataprep,
        synth=synth,
        optim_method="Nelder-Mead",
        optim_initial="equal",
    )
    util.placebo_plot(
        placebo_test,
        time_period=NZ_TIME,
        grid=True,
        treatment_time=NZ_ACTUAL_TREATMENT,
        divide_by=1000,
        y_axis_label="GDP per Capita (Thousands of NZD)",
        y_axis_limit=10,
        filename="nz_placebos.png",
    )

    # Jackknife figure: same SCM path plot (notebook used same fit with different filename)
    synth_jacknife = Synth()
    synth_jacknife.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="equal")
    util.synth_plot_nz(
        synth_jacknife,
        time_period=NZ_TIME,
        treatment_time=NZ_ACTUAL_TREATMENT,
        filename="nz_jacknife.png",
    )


def _run_chile(output_dir: str = FIGURES_DIR) -> None:
    """Build Chile dataprep, fit SCM, run placebo test, and save all Chile/Maule figures."""
    warnings.simplefilter("ignore", category=FutureWarning)
    df = pcd.process_data_for_synth()
    dataprep = build_chile_dataprep(df, CHILE_ACTUAL_TREATMENT)

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="ols")

    util.synth_plot_chile(
        synth,
        time_period=CHILE_TIME,
        treatment_time=CHILE_ACTUAL_TREATMENT,
        filename="maule_gdp_paths.png",
    )
    util.gap_plot(
        synth,
        time_period=CHILE_TIME,
        treatment_time=CHILE_ACTUAL_TREATMENT,
        filename="maule_gap.png",
    )

    placebo_test = PlaceboTest()
    placebo_test.fit(
        dataprep=dataprep,
        synth=synth,
        optim_method="Nelder-Mead",
        optim_initial="ols",
    )
    util.placebo_plot(
        placebo_test,
        time_period=CHILE_TIME,
        grid=True,
        treatment_time=CHILE_ACTUAL_TREATMENT,
        mspe_threshold=100,
        divide_by=1_000_000,
        y_axis_label="GDP per Capita (Millions of CLP)",
        y_axis_limit=4,
        filename="maule_placebos.png",
    )

    synth_jacknife = Synth()
    synth_jacknife.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="ols")
    util.synth_plot_chile(
        synth_jacknife,
        time_period=CHILE_TIME,
        treatment_time=CHILE_ACTUAL_TREATMENT,
        filename="chile_jacknife.png",
    )


def run_main_scm_figures(output_dir: str = FIGURES_DIR) -> None:
    """Generate all main SCM figures (NZ and Chile) and save to output_dir."""
    _run_nz(output_dir=output_dir)
    _run_chile(output_dir=output_dir)
    print(f"Main SCM figures saved to {output_dir}")


if __name__ == "__main__":
    run_main_scm_figures()

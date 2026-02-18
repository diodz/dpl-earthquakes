import os
import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest

import nz_util
import process_chile_gdp_data as pcd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)


CHILE_TREATED = "VII Del Maule"
CHILE_CONTROLS = [
    "I De Tarapacá",
    "II De Antofagasta",
    "III De Atacama",
    "IV De Coquimbo",
    "V De Valparaíso",
    "RMS Región Metropolitana de Santiago",
    "VI Del Libertador General Bernardo OHiggins",
    "IX De La Araucanía",
    "X De Los Lagos",
    "XI Aysén del General Carlos Ibáñez del Campo",
    "XII De Magallanes y de la Antártica Chilena",
]

NZ_TREATED = "Canterbury"
NZ_CONTROLS = [
    "Auckland",
    "Bay of Plenty",
    "Gisborne",
    "Hawke's Bay",
    "Manawatu-Whanganui",
    "Marlborough",
    "Northland",
    "Otago",
    "Southland",
    "Taranaki",
    "Tasman/Nelson",
    "Waikato",
    "Wellington",
    "West Coast",
]

CHILE_SCM_OPTIONS = {"optim_method": "Nelder-Mead", "optim_initial": "ols"}
NZ_SCM_OPTIONS = {"optim_method": "Nelder-Mead", "optim_initial": "equal"}

ALPHA = 0.05
# Pre-fit placebo retention threshold (× treated pre-RMSPE); used for both Chile and NZ sensitivity.
MSPE_SENSITIVITY = [2.0, 5.0, 10.0]


@dataclass
class UniformScenario:
    country: str
    scenario: str
    treatment_year: int
    mspe_threshold: float
    alpha: float
    intervals: pd.DataFrame
    treated_pre_rmspe: float
    critical_value: float
    retained_placebos: list[str]


def _build_chile_dataprep(df: pd.DataFrame) -> Dataprep:
    return Dataprep(
        foo=df,
        predictors=[
            "agropecuario",
            "pesca",
            "mineria",
            "industria_m",
            "electricidad",
            "construccion",
            "comercio",
            "transporte",
            "servicios_financieros",
            "vivienda",
            "personales",
            "publica",
        ],
        predictors_op="mean",
        time_predictors_prior=range(2005, 2009),
        special_predictors=[
            ("gdp_cap", range(2005, 2009), "mean"),
            ("ed_superior_cap", range(2008, 2009), "mean"),
        ],
        dependent="gdp_cap",
        unit_variable="region_name",
        time_variable="year",
        treatment_identifier=CHILE_TREATED,
        controls_identifier=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, 2009),
    )


def _build_nz_dataprep(df: pd.DataFrame, treatment_year: int) -> Dataprep:
    if treatment_year == 2010:
        time_predictors_prior = range(2005, 2009)
        special_predictors = [
            ("GDP per capita", range(2005, 2009), "mean"),
            ("Tertiary Share", range(2008, 2009), "mean"),
        ]
        time_optimize_ssr = range(2000, 2010)
    elif treatment_year == 2011:
        # Shift the predictor window one year later when treatment starts in 2011.
        time_predictors_prior = range(2006, 2011)
        special_predictors = [
            ("GDP per capita", range(2006, 2011), "mean"),
            ("Tertiary Share", range(2010, 2011), "mean"),
        ]
        time_optimize_ssr = range(2000, 2011)
    else:
        raise ValueError(f"Unsupported New Zealand treatment year: {treatment_year}")

    return Dataprep(
        foo=df,
        predictors=nz_util.SECTORIAL_GDP_VARIABLES,
        predictors_op="mean",
        time_predictors_prior=time_predictors_prior,
        special_predictors=special_predictors,
        dependent="GDP per capita",
        unit_variable="Region",
        time_variable="Year",
        treatment_identifier=NZ_TREATED,
        controls_identifier=NZ_CONTROLS,
        time_optimize_ssr=time_optimize_ssr,
    )


def _run_placebo_gaps(dataprep: Dataprep, scm_options: dict[str, Any]) -> tuple[pd.Series, pd.DataFrame]:
    synth = Synth()
    synth.fit(dataprep=dataprep, **scm_options)

    placebo_test = PlaceboTest()
    placebo_test.fit(
        dataprep=dataprep,
        scm=synth,
        scm_options=scm_options,
        verbose=False,
    )
    return placebo_test.treated_gap.astype(float), placebo_test.gaps.astype(float)


def _rmspe(series: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(series.to_numpy(dtype=float)))))


def _convert_gap_levels_to_percent(
    treated_gap_levels: pd.Series,
    placebo_gap_levels: pd.DataFrame,
    source_df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated_unit: str,
) -> tuple[pd.Series, pd.DataFrame]:
    outcomes = source_df.pivot_table(index=time_col, columns=unit_col, values=outcome_col, aggfunc="first")
    common_years = treated_gap_levels.index.intersection(outcomes.index)
    treated_gap_levels = treated_gap_levels.loc[common_years].astype(float)
    placebo_gap_levels = placebo_gap_levels.loc[common_years].astype(float)

    treated_actual = outcomes.loc[common_years, treated_unit].astype(float)
    treated_synthetic = treated_actual - treated_gap_levels
    treated_gap_pct = 100.0 * treated_gap_levels.divide(treated_synthetic.replace(0, np.nan))

    placebo_gap_pct = pd.DataFrame(index=common_years)
    for unit in placebo_gap_levels.columns:
        actual = outcomes.loc[common_years, unit].astype(float)
        synthetic = actual - placebo_gap_levels[unit]
        placebo_gap_pct[unit] = 100.0 * placebo_gap_levels[unit].divide(synthetic.replace(0, np.nan))

    treated_gap_pct = treated_gap_pct.replace([np.inf, -np.inf], np.nan).dropna()
    placebo_gap_pct = placebo_gap_pct.replace([np.inf, -np.inf], np.nan)
    placebo_gap_pct = placebo_gap_pct.loc[treated_gap_pct.index]
    return treated_gap_pct, placebo_gap_pct


def _uniform_intervals_from_placebos(
    treated_gap_pct: pd.Series,
    placebo_gaps_pct: pd.DataFrame,
    treatment_year: int,
    mspe_threshold: float,
    alpha: float,
) -> tuple[pd.DataFrame, float, float, list[str]]:
    pre_mask = treated_gap_pct.index < treatment_year
    post_mask = treated_gap_pct.index >= treatment_year
    if pre_mask.sum() == 0 or post_mask.sum() == 0:
        raise ValueError("Need both pre and post periods to compute uniform confidence sets.")

    treated_pre_rmspe = _rmspe(treated_gap_pct.loc[pre_mask])
    if treated_pre_rmspe <= 0:
        raise ValueError("Treated pre-period RMSPE must be positive.")

    placebo_pre_rmspe = placebo_gaps_pct.loc[pre_mask].apply(_rmspe, axis=0)
    retained = placebo_pre_rmspe[
        (placebo_pre_rmspe > 0) & (placebo_pre_rmspe <= mspe_threshold * treated_pre_rmspe)
    ].index.tolist()
    if len(retained) == 0:
        raise ValueError(f"No placebo units satisfy MSPE threshold={mspe_threshold}.")

    standardized_placebo_post = placebo_gaps_pct.loc[post_mask, retained].divide(
        placebo_pre_rmspe.loc[retained], axis=1
    )
    sup_stats = standardized_placebo_post.abs().max(axis=0).to_numpy(dtype=float)
    critical_value = float(np.quantile(sup_stats, 1.0 - alpha, method="higher"))

    half_width = critical_value * treated_pre_rmspe
    intervals = pd.DataFrame(index=treated_gap_pct.index)
    intervals.index.name = "Year"
    intervals["treated_gap_pct"] = treated_gap_pct
    intervals["lower_bound_pct"] = np.nan
    intervals["upper_bound_pct"] = np.nan
    intervals.loc[post_mask, "lower_bound_pct"] = treated_gap_pct.loc[post_mask] - half_width
    intervals.loc[post_mask, "upper_bound_pct"] = treated_gap_pct.loc[post_mask] + half_width
    return intervals, treated_pre_rmspe, critical_value, retained


def _plot_uniform_panel(
    ax: plt.Axes,
    scenario: UniformScenario,
    title: str,
    ylabel: str = "Gap (%)",
) -> None:
    intervals = scenario.intervals
    years = intervals.index.to_numpy(dtype=int)
    post_mask = years >= scenario.treatment_year

    ax.plot(
        years,
        intervals["treated_gap_pct"].to_numpy(dtype=float),
        color="#d62728",
        linewidth=1.8,
        label="Treated gap",
    )
    ax.fill_between(
        years[post_mask],
        intervals.loc[post_mask, "lower_bound_pct"].to_numpy(dtype=float),
        intervals.loc[post_mask, "upper_bound_pct"].to_numpy(dtype=float),
        color="#1f77b4",
        alpha=0.2,
        label=f"{int((1.0 - scenario.alpha) * 100)}% uniform interval",
    )
    ax.axvline(scenario.treatment_year, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _scenario_to_long_df(scenario: UniformScenario) -> pd.DataFrame:
    long_df = scenario.intervals.reset_index().rename(
        columns={
            "Year": "Year",
            "treated_gap_pct": "TreatedGapPct",
            "lower_bound_pct": "LowerBoundPct",
            "upper_bound_pct": "UpperBoundPct",
        }
    )
    long_df["Country"] = scenario.country
    long_df["Scenario"] = scenario.scenario
    long_df["TreatmentYear"] = scenario.treatment_year
    long_df["MSPEThreshold"] = scenario.mspe_threshold
    long_df["Alpha"] = scenario.alpha
    long_df["TreatedPreRMSPEPct"] = scenario.treated_pre_rmspe
    long_df["CriticalValue"] = scenario.critical_value
    long_df["NPlacebosRetained"] = len(scenario.retained_placebos)
    long_df["RetainedPlacebos"] = ";".join(scenario.retained_placebos)
    return long_df


def _scenario_summary_row(scenario: UniformScenario) -> dict[str, Any]:
    post_df = scenario.intervals[scenario.intervals.index >= scenario.treatment_year]
    excludes_zero = (
        (post_df["lower_bound_pct"] > 0) | (post_df["upper_bound_pct"] < 0)
    ).astype(int)
    return {
        "Country": scenario.country,
        "Scenario": scenario.scenario,
        "TreatmentYear": scenario.treatment_year,
        "MSPEThreshold": scenario.mspe_threshold,
        "Alpha": scenario.alpha,
        "TreatedPreRMSPEPct": scenario.treated_pre_rmspe,
        "CriticalValue": scenario.critical_value,
        "NPlacebosRetained": len(scenario.retained_placebos),
        "RetainedPlacebos": ";".join(scenario.retained_placebos),
        "PostYears": int(post_df.shape[0]),
        "SharePostYearsExcludingZero": float(excludes_zero.mean()),
        "AllPostYearsExcludeZero": bool(excludes_zero.all()),
    }


def run_uniform_confidence_analysis(output_dir: str = FIGURES_DIR) -> pd.DataFrame:
    warnings.filterwarnings("ignore", category=FutureWarning)
    os.makedirs(output_dir, exist_ok=True)

    scenarios: list[UniformScenario] = []

    # Chile baseline placebos, then vary pre-fit filters.
    chile_df = pcd.process_data_for_synth()
    chile_dataprep = _build_chile_dataprep(chile_df)
    chile_treated_gap, chile_placebo_gaps = _run_placebo_gaps(chile_dataprep, CHILE_SCM_OPTIONS)
    chile_treated_gap, chile_placebo_gaps = _convert_gap_levels_to_percent(
        treated_gap_levels=chile_treated_gap,
        placebo_gap_levels=chile_placebo_gaps,
        source_df=chile_df,
        unit_col="region_name",
        time_col="year",
        outcome_col="gdp_cap",
        treated_unit=CHILE_TREATED,
    )
    for threshold in MSPE_SENSITIVITY:
        intervals, pre_rmspe, critical, retained = _uniform_intervals_from_placebos(
            treated_gap_pct=chile_treated_gap,
            placebo_gaps_pct=chile_placebo_gaps,
            treatment_year=2010,
            mspe_threshold=threshold,
            alpha=ALPHA,
        )
        scenarios.append(
            UniformScenario(
                country="Chile",
                scenario=f"chile_fit_threshold_{int(threshold)}x",
                treatment_year=2010,
                mspe_threshold=threshold,
                alpha=ALPHA,
                intervals=intervals,
                treated_pre_rmspe=pre_rmspe,
                critical_value=critical,
                retained_placebos=retained,
            )
        )

    # New Zealand: fixed treatment year 2011, vary pre-fit threshold (same as Chile).
    nz_df = nz_util.clean_data_for_synthetic_control().copy()
    nz_df["Tertiary Share"] = nz_df["Tertiary"] / nz_df["Population"]
    nz_dataprep = _build_nz_dataprep(nz_df, treatment_year=2011)
    nz_treated_gap, nz_placebo_gaps = _run_placebo_gaps(nz_dataprep, NZ_SCM_OPTIONS)
    nz_treated_gap, nz_placebo_gaps = _convert_gap_levels_to_percent(
        treated_gap_levels=nz_treated_gap,
        placebo_gap_levels=nz_placebo_gaps,
        source_df=nz_df,
        unit_col="Region",
        time_col="Year",
        outcome_col="GDP per capita",
        treated_unit=NZ_TREATED,
    )
    for threshold in MSPE_SENSITIVITY:
        intervals, pre_rmspe, critical, retained = _uniform_intervals_from_placebos(
            treated_gap_pct=nz_treated_gap,
            placebo_gaps_pct=nz_placebo_gaps,
            treatment_year=2011,
            mspe_threshold=threshold,
            alpha=ALPHA,
        )
        scenarios.append(
            UniformScenario(
                country="New Zealand",
                scenario=f"nz_fit_threshold_{int(threshold)}x",
                treatment_year=2011,
                mspe_threshold=threshold,
                alpha=ALPHA,
                intervals=intervals,
                treated_pre_rmspe=pre_rmspe,
                critical_value=critical,
                retained_placebos=retained,
            )
        )

    scenario_map = {scenario.scenario: scenario for scenario in scenarios}
    chile_baseline = scenario_map["chile_fit_threshold_10x"]
    nz_baseline = scenario_map["nz_fit_threshold_10x"]

    # Main two-country figure.
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    _plot_uniform_panel(
        axes[0],
        chile_baseline,
        title="Chile (Maule): 95% uniform confidence set",
        ylabel="GDP per capita gap (%)",
    )
    _plot_uniform_panel(
        axes[1],
        nz_baseline,
        title="New Zealand (Canterbury): 95% uniform confidence set",
        ylabel="",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(output_dir, "scm_uniform_confidence_sets.png"), dpi=220)
    plt.close(fig)

    # Chile fit-threshold sensitivity figure.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for axis, threshold in zip(axes, MSPE_SENSITIVITY, strict=True):
        scenario = scenario_map[f"chile_fit_threshold_{int(threshold)}x"]
        _plot_uniform_panel(
            axis,
            scenario,
            title=f"Chile: pre-fit threshold {int(threshold)}x",
            ylabel="GDP per capita gap (%)" if threshold == MSPE_SENSITIVITY[0] else "",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "chile_uniform_threshold_sensitivity.png"), dpi=220)
    plt.close(fig)

    # New Zealand fit-threshold sensitivity figure (same design as Chile).
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for axis, threshold in zip(axes, MSPE_SENSITIVITY, strict=True):
        scenario = scenario_map[f"nz_fit_threshold_{int(threshold)}x"]
        _plot_uniform_panel(
            axis,
            scenario,
            title=f"New Zealand: pre-fit threshold {int(threshold)}x",
            ylabel="GDP per capita gap (%)" if threshold == MSPE_SENSITIVITY[0] else "",
        )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "nz_uniform_threshold_sensitivity.png"), dpi=220)
    plt.close(fig)

    # Export tidy outputs for reproducibility.
    long_df = pd.concat([_scenario_to_long_df(scenario) for scenario in scenarios], ignore_index=True)
    long_df.to_csv(os.path.join(output_dir, "scm_uniform_confidence_sets.csv"), index=False)

    summary_df = pd.DataFrame([_scenario_summary_row(scenario) for scenario in scenarios])
    summary_df.to_csv(os.path.join(output_dir, "scm_uniform_confidence_summary.csv"), index=False)
    return summary_df


if __name__ == "__main__":
    summary = run_uniform_confidence_analysis()
    print(summary.to_string(index=False))

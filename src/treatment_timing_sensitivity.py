import os
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest

import nz_util

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

ANALYSIS_END_YEAR = 2019
NZ_COMMON_POST_START = 2011
CHILE_COMMON_POST_START = 2010

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
CHILE_PREDICTORS = [
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
]


@dataclass(frozen=True)
class TimingScenario:
    label: str
    treatment_year: int
    preferred: bool = False


NZ_SCENARIOS = [
    TimingScenario("2010 start", 2010, preferred=False),
    TimingScenario("2011 start (preferred)", 2011, preferred=True),
    TimingScenario("2012 sequence-aware start", 2012, preferred=False),
]
CHILE_SCENARIOS = [
    TimingScenario("2009 placebo start", 2009, preferred=False),
    TimingScenario("2010 baseline start", 2010, preferred=True),
    TimingScenario("2011 placebo start", 2011, preferred=False),
]


def _rmspe(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(valid))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return float("nan")
    return float(numerator / denominator)


def _build_nz_dataprep(df: pd.DataFrame, treatment_year: int) -> Dataprep:
    predictors_window = range(max(2000, treatment_year - 5), treatment_year)
    tertiary_window = range(max(2000, treatment_year - 2), treatment_year)
    return Dataprep(
        foo=df,
        predictors=nz_util.SECTORIAL_GDP_VARIABLES,
        predictors_op="mean",
        time_predictors_prior=predictors_window,
        special_predictors=[
            ("GDP per capita", predictors_window, "mean"),
            ("Tertiary Share", tertiary_window, "mean"),
        ],
        dependent="GDP per capita",
        unit_variable="Region",
        time_variable="Year",
        treatment_identifier=NZ_TREATED,
        controls_identifier=NZ_CONTROLS,
        time_optimize_ssr=range(2000, treatment_year),
    )


def _build_chile_dataprep(df: pd.DataFrame, treatment_year: int) -> Dataprep:
    predictors_window = range(max(1990, treatment_year - 5), treatment_year)
    education_window = range(max(1990, treatment_year - 2), treatment_year)
    return Dataprep(
        foo=df,
        predictors=CHILE_PREDICTORS,
        predictors_op="mean",
        time_predictors_prior=predictors_window,
        special_predictors=[
            ("gdp_cap", predictors_window, "mean"),
            ("ed_superior_cap", education_window, "mean"),
        ],
        dependent="gdp_cap",
        unit_variable="region_name",
        time_variable="year",
        treatment_identifier=CHILE_TREATED,
        controls_identifier=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, treatment_year),
    )


def _evaluate_scenario(
    country: str,
    scenario: TimingScenario,
    df: pd.DataFrame,
    dataprep_builder: Callable[[pd.DataFrame, int], Dataprep],
    treated: str,
    controls: list[str],
    unit_col: str,
    time_col: str,
    outcome_col: str,
    optim_method: str,
    optim_initial: str,
    common_post_start: int,
    fit_start_year: int,
) -> tuple[dict, list[dict], list[dict]]:
    dataprep = dataprep_builder(df, scenario.treatment_year)
    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method=optim_method, optim_initial=optim_initial)

    all_years = list(range(int(df[time_col].min()), int(df[time_col].max()) + 1))
    z0, z1 = synth.dataprep.make_outcome_mats(time_period=all_years)
    synthetic = pd.Series(np.asarray(synth._synthetic(z0)).flatten().astype(float), index=all_years)
    treated_outcome = pd.Series(np.asarray(z1).flatten().astype(float), index=all_years)
    treated_gap_level = treated_outcome - synthetic

    pivot = (
        df[df[unit_col].isin([treated] + controls)]
        .pivot(index=time_col, columns=unit_col, values=outcome_col)
        .reindex(all_years)
    )
    treated_gap_pct = treated_gap_level / synthetic * 100.0

    placebo_test = PlaceboTest()
    placebo_test.fit(
        dataprep=dataprep,
        scm=synth,
        scm_options={"optim_method": optim_method, "optim_initial": optim_initial},
    )
    placebo_gaps_level = placebo_test.gaps.reindex(all_years)

    placebo_gap_pct = pd.DataFrame(index=all_years)
    placebo_rows: list[dict] = []
    year_index = np.asarray(all_years, dtype=int)
    pre_mask = (year_index >= fit_start_year) & (year_index < scenario.treatment_year)
    post_mask = (year_index >= scenario.treatment_year) & (year_index <= ANALYSIS_END_YEAR)
    common_mask = (year_index >= common_post_start) & (year_index <= ANALYSIS_END_YEAR)

    for unit in placebo_gaps_level.columns:
        unit_actual = pivot[unit].astype(float)
        unit_synthetic = unit_actual - placebo_gaps_level[unit]
        unit_gap_pct = placebo_gaps_level[unit] / unit_synthetic * 100.0
        placebo_gap_pct[unit] = unit_gap_pct
        unit_values = unit_gap_pct.to_numpy(dtype=float)
        unit_pre = _rmspe(unit_values[pre_mask])
        unit_post = _rmspe(unit_values[post_mask])
        placebo_rows.append(
            {
                "Country": country,
                "ScenarioLabel": scenario.label,
                "TreatmentYear": scenario.treatment_year,
                "Unit": unit,
                "PreRMSPEPct": unit_pre,
                "PostRMSPEPct": unit_post,
                "RMSPERatio": _safe_ratio(unit_post, unit_pre),
                "MeanGapPctCommonWindow": float(np.nanmean(unit_values[common_mask])),
            }
        )

    treated_values = treated_gap_pct.to_numpy(dtype=float)
    treated_pre = _rmspe(treated_values[pre_mask])
    treated_post = _rmspe(treated_values[post_mask])
    treated_ratio = _safe_ratio(treated_post, treated_pre)
    treated_mean_common = float(np.nanmean(treated_values[common_mask]))

    placebo_ratio_values = np.asarray([row["RMSPERatio"] for row in placebo_rows], dtype=float)
    placebo_ratio_values = placebo_ratio_values[np.isfinite(placebo_ratio_values)]
    ratio_rank = int(np.sum(placebo_ratio_values >= treated_ratio)) + 1
    ratio_p_value = float(ratio_rank / (placebo_ratio_values.size + 1))

    placebo_mean_values = np.asarray(
        [row["MeanGapPctCommonWindow"] for row in placebo_rows], dtype=float
    )
    placebo_mean_values = placebo_mean_values[np.isfinite(placebo_mean_values)]
    if treated_mean_common >= 0:
        mean_rank = int(np.sum(placebo_mean_values >= treated_mean_common)) + 1
    else:
        mean_rank = int(np.sum(placebo_mean_values <= treated_mean_common)) + 1
    mean_p_value = float(mean_rank / (placebo_mean_values.size + 1))

    summary_row = {
        "Country": country,
        "ScenarioLabel": scenario.label,
        "TreatmentYear": scenario.treatment_year,
        "PreferredConvention": scenario.preferred,
        "PreRMSPEPct": treated_pre,
        "PostRMSPEPct": treated_post,
        "RMSPERatio": treated_ratio,
        "RMSPERatioPlaceboRank": ratio_rank,
        "RMSPERatioOneSidedPValue": ratio_p_value,
        "MeanGapPctCommonWindow": treated_mean_common,
        "MeanGapPlaceboRank": mean_rank,
        "MeanGapOneSidedPValue": mean_p_value,
        "PostWindowStart": scenario.treatment_year,
        "PostWindowEnd": ANALYSIS_END_YEAR,
        "CommonWindowStart": common_post_start,
        "CommonWindowEnd": ANALYSIS_END_YEAR,
        "NumPlaceboUnits": int(placebo_ratio_values.size),
    }

    gap_rows = [
        {
            "Country": country,
            "ScenarioLabel": scenario.label,
            "TreatmentYear": scenario.treatment_year,
            "PreferredConvention": scenario.preferred,
            "Year": int(year),
            "TreatedOutcome": float(treated_outcome.loc[year]),
            "SyntheticOutcome": float(synthetic.loc[year]),
            "GapLevel": float(treated_gap_level.loc[year]),
            "GapPct": float(treated_gap_pct.loc[year]),
        }
        for year in all_years
    ]
    return summary_row, gap_rows, placebo_rows


def _plot_gap_paths(gap_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    country_configs = [
        ("New Zealand (Canterbury)", "New Zealand", NZ_COMMON_POST_START),
        ("Chile (Maule)", "Chile", CHILE_COMMON_POST_START),
    ]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    for axis, (title, country, common_start) in zip(axes, country_configs, strict=True):
        country_df = gap_df[gap_df["Country"] == country]
        scenario_order = (
            country_df[["ScenarioLabel", "TreatmentYear", "PreferredConvention"]]
            .drop_duplicates()
            .sort_values("TreatmentYear")
        )
        for color, (_, row) in zip(colors, scenario_order.iterrows(), strict=False):
            scenario_label = row["ScenarioLabel"]
            treatment_year = int(row["TreatmentYear"])
            preferred = bool(row["PreferredConvention"])
            line_df = country_df[country_df["ScenarioLabel"] == scenario_label]
            axis.plot(
                line_df["Year"],
                line_df["GapPct"],
                label=scenario_label,
                color=color,
                linewidth=2.5 if preferred else 1.7,
                linestyle="-" if preferred else "--",
                alpha=0.95,
            )
            axis.axvline(treatment_year, color=color, linestyle=":", linewidth=0.9, alpha=0.5)

        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.axvspan(common_start, ANALYSIS_END_YEAR, color="#efefef", alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("Year")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Gap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_ratio_summary(summary_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    country_configs = [("New Zealand", axes[0]), ("Chile", axes[1])]

    for country, axis in country_configs:
        subset = summary_df[summary_df["Country"] == country].sort_values("TreatmentYear")
        x = np.arange(subset.shape[0])
        bars = axis.bar(x, subset["RMSPERatio"], color="#4c78a8", alpha=0.85)
        for idx, (_, row) in enumerate(subset.iterrows()):
            axis.text(
                idx,
                row["RMSPERatio"] + 0.15,
                f"rank {int(row['RMSPERatioPlaceboRank'])}/{int(row['NumPlaceboUnits']) + 1}\np={row['RMSPERatioOneSidedPValue']:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
            if bool(row["PreferredConvention"]):
                bars[idx].set_color("#d62728")
        axis.set_xticks(x)
        axis.set_xticklabels(subset["ScenarioLabel"], rotation=12, ha="right")
        axis.set_title(country)
        axis.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Post/Pre RMSPE ratio")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_treatment_timing_sensitivity(output_dir: str = FIGURES_DIR) -> pd.DataFrame:
    nz_df = nz_util.clean_data_for_synthetic_control().copy()
    nz_df["Tertiary Share"] = nz_df["Tertiary"] / nz_df["Population"]
    chile_df = pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv"))

    summary_rows: list[dict] = []
    gap_rows: list[dict] = []
    placebo_rows: list[dict] = []

    for scenario in NZ_SCENARIOS:
        summary, gaps, placebos = _evaluate_scenario(
            country="New Zealand",
            scenario=scenario,
            df=nz_df,
            dataprep_builder=_build_nz_dataprep,
            treated=NZ_TREATED,
            controls=NZ_CONTROLS,
            unit_col="Region",
            time_col="Year",
            outcome_col="GDP per capita",
            optim_method="Nelder-Mead",
            optim_initial="equal",
            common_post_start=NZ_COMMON_POST_START,
            fit_start_year=2000,
        )
        summary_rows.append(summary)
        gap_rows.extend(gaps)
        placebo_rows.extend(placebos)

    for scenario in CHILE_SCENARIOS:
        summary, gaps, placebos = _evaluate_scenario(
            country="Chile",
            scenario=scenario,
            df=chile_df,
            dataprep_builder=_build_chile_dataprep,
            treated=CHILE_TREATED,
            controls=CHILE_CONTROLS,
            unit_col="region_name",
            time_col="year",
            outcome_col="gdp_cap",
            optim_method="Nelder-Mead",
            optim_initial="ols",
            common_post_start=CHILE_COMMON_POST_START,
            fit_start_year=1990,
        )
        summary_rows.append(summary)
        gap_rows.extend(gaps)
        placebo_rows.extend(placebos)

    summary_df = pd.DataFrame(summary_rows).sort_values(["Country", "TreatmentYear"])
    gap_df = pd.DataFrame(gap_rows).sort_values(["Country", "ScenarioLabel", "Year"])
    placebo_df = pd.DataFrame(placebo_rows).sort_values(["Country", "ScenarioLabel", "Unit"])

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "timing_sensitivity_summary.csv")
    gap_path = os.path.join(output_dir, "timing_sensitivity_gaps.csv")
    placebo_path = os.path.join(output_dir, "timing_sensitivity_placebo_distribution.csv")
    nz_summary_path = os.path.join(output_dir, "nz_timing_sensitivity_summary.csv")
    chile_summary_path = os.path.join(output_dir, "chile_timing_sensitivity_summary.csv")
    gap_fig_path = os.path.join(output_dir, "timing_sensitivity_gap_paths.png")
    ratio_fig_path = os.path.join(output_dir, "timing_sensitivity_rmspe_ratios.png")

    summary_df.to_csv(summary_path, index=False)
    gap_df.to_csv(gap_path, index=False)
    placebo_df.to_csv(placebo_path, index=False)
    summary_df[summary_df["Country"] == "New Zealand"].to_csv(nz_summary_path, index=False)
    summary_df[summary_df["Country"] == "Chile"].to_csv(chile_summary_path, index=False)

    _plot_gap_paths(gap_df, gap_fig_path)
    _plot_ratio_summary(summary_df, ratio_fig_path)
    return summary_df


if __name__ == "__main__":
    result = run_treatment_timing_sensitivity()
    print(result.to_string(index=False))

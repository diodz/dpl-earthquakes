import os
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest

import nz_util

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSETS_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(_ASSETS_DIR, exist_ok=True)

CHILE_TREATED = "VII Del Maule"
CHILE_CONTROLS_BASELINE = [
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

NZ_TREATED = "Canterbury"
NZ_CONTROLS_BASELINE = [
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

ANALYSIS_END_YEAR = 2019


@dataclass(frozen=True)
class SpilloverScenario:
    country: str
    scenario_id: str
    scenario_label: str
    exclusions: list[str]


SCENARIOS = [
    SpilloverScenario(
        country="Chile",
        scenario_id="baseline",
        scenario_label="Baseline (Biobio excluded)",
        exclusions=[],
    ),
    SpilloverScenario(
        country="Chile",
        scenario_id="exclude_ohiggins",
        scenario_label="Ring 1: exclude O'Higgins",
        exclusions=["VI Del Libertador General Bernardo OHiggins"],
    ),
    SpilloverScenario(
        country="Chile",
        scenario_id="exclude_ohiggins_araucania",
        scenario_label="Ring 2: exclude O'Higgins + Araucania",
        exclusions=[
            "VI Del Libertador General Bernardo OHiggins",
            "IX De La Araucanía",
        ],
    ),
    SpilloverScenario(
        country="New Zealand",
        scenario_id="baseline",
        scenario_label="Baseline donor pool",
        exclusions=[],
    ),
    SpilloverScenario(
        country="New Zealand",
        scenario_id="exclude_auckland_wellington",
        scenario_label="Ring 1: exclude Auckland + Wellington",
        exclusions=["Auckland", "Wellington"],
    ),
    SpilloverScenario(
        country="New Zealand",
        scenario_id="exclude_metro_corridor",
        scenario_label="Ring 2: North-Island corridor exclusion",
        exclusions=[
            "Auckland",
            "Wellington",
            "Waikato",
            "Bay of Plenty",
            "Hawke's Bay",
            "Manawatu-Whanganui",
        ],
    ),
]


def _rmspe(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(valid))))


def _safe_pct_gap(actual: pd.Series, synthetic: pd.Series) -> pd.Series:
    return 100.0 * (actual - synthetic) / synthetic.replace(0, np.nan)


def _prepare_chile_outcome_data() -> pd.DataFrame:
    path = os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv")
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    numeric_cols = ["gdp_cap"] + CHILE_PREDICTORS + ["ed_superior_cap"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prepare_nz_outcome_data() -> pd.DataFrame:
    path = os.path.join(_PROJECT_ROOT, "inter", "nz.csv")
    df = pd.read_csv(path)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["GDP per capita"] = pd.to_numeric(df["GDP per capita"], errors="coerce")
    df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
    df["Tertiary"] = pd.to_numeric(df["Tertiary"], errors="coerce")
    df["Tertiary Share"] = df["Tertiary"] / df["Population"]
    return df


def _prepare_chile_flow_data() -> pd.DataFrame:
    path = os.path.join(_PROJECT_ROOT, "data", "scm_2010.csv")
    df = pd.read_csv(path)
    name_map = {
        "tarapaca": "I De Tarapacá",
        "antofagasta": "II De Antofagasta",
        "atacama": "III De Atacama",
        "coquimbo": "IV De Coquimbo",
        "valparaiso": "V De Valparaíso",
        "metropolitana": "RMS Región Metropolitana de Santiago",
        "ohiggins": "VI Del Libertador General Bernardo OHiggins",
        "maule": "VII Del Maule",
        "biobio": "VIII Del Biobío",
        "araucania": "IX De La Araucanía",
        "loslagos": "X De Los Lagos",
        "aysen": "XI Aysén del General Carlos Ibáñez del Campo",
        "magallanes": "XII De Magallanes y de la Antártica Chilena",
    }
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["region_name"] = df["nom_region"].map(name_map)
    numeric_cols = [
        "population",
        "gdp_pc",
        "construcción",
        "industria manufacturera",
        "comercio, restaurantes y hoteles",
        "transporte, información y comunicaciones",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_chile_dataprep(df: pd.DataFrame, controls: list[str]) -> Dataprep:
    return Dataprep(
        foo=df,
        predictors=CHILE_PREDICTORS,
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
        controls_identifier=controls,
        time_optimize_ssr=range(1990, 2009),
    )


def _build_nz_dataprep(df: pd.DataFrame, controls: list[str]) -> Dataprep:
    return Dataprep(
        foo=df,
        predictors=nz_util.SECTORIAL_GDP_VARIABLES,
        predictors_op="mean",
        time_predictors_prior=range(2005, 2009),
        special_predictors=[
            ("GDP per capita", range(2005, 2009), "mean"),
            ("Tertiary Share", range(2008, 2009), "mean"),
        ],
        dependent="GDP per capita",
        unit_variable="Region",
        time_variable="Year",
        treatment_identifier=NZ_TREATED,
        controls_identifier=controls,
        time_optimize_ssr=range(2000, 2010),
    )


def _evaluate_scenario(
    scenario: SpilloverScenario,
    chile_df: pd.DataFrame,
    nz_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series]:
    if scenario.country == "Chile":
        controls = [c for c in CHILE_CONTROLS_BASELINE if c not in scenario.exclusions]
        dataprep = _build_chile_dataprep(chile_df, controls)
        treated = CHILE_TREATED
        dependent = "gdp_cap"
        unit_col = "region_name"
        time_col = "year"
        treatment_year = 2010
        fit_options = {"optim_method": "Nelder-Mead", "optim_initial": "ols"}
        source = chile_df
    elif scenario.country == "New Zealand":
        controls = [c for c in NZ_CONTROLS_BASELINE if c not in scenario.exclusions]
        dataprep = _build_nz_dataprep(nz_df, controls)
        treated = NZ_TREATED
        dependent = "GDP per capita"
        unit_col = "Region"
        time_col = "Year"
        treatment_year = 2011
        fit_options = {"optim_method": "Nelder-Mead", "optim_initial": "equal"}
        source = nz_df
    else:
        raise ValueError(f"Unsupported country: {scenario.country}")

    synth = Synth()
    synth.fit(dataprep=dataprep, **fit_options)

    years = np.arange(int(source[time_col].min()), int(source[time_col].max()) + 1)
    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    synthetic = pd.Series(np.asarray(synth._synthetic(z0)).flatten().astype(float), index=years)
    actual = pd.Series(np.asarray(z1).flatten().astype(float), index=years)
    gap_pct = _safe_pct_gap(actual, synthetic)

    pre_mask = years < treatment_year
    post_mask = (years >= treatment_year) & (years <= ANALYSIS_END_YEAR)
    pre_rmspe = _rmspe(gap_pct.to_numpy(dtype=float)[pre_mask])
    post_rmspe = _rmspe(gap_pct.to_numpy(dtype=float)[post_mask])
    ratio = post_rmspe / pre_rmspe if np.isfinite(pre_rmspe) and pre_rmspe > 0 else float("nan")
    mean_post_gap = float(np.nanmean(gap_pct.to_numpy(dtype=float)[post_mask]))

    placebo_test = PlaceboTest()
    placebo_test.fit(dataprep=dataprep, scm=synth, scm_options=fit_options, verbose=False)
    placebo_gaps = placebo_test.gaps.reindex(years).astype(float)
    pivot = (
        source[source[unit_col].isin([treated] + controls)]
        .pivot(index=time_col, columns=unit_col, values=dependent)
        .reindex(years)
    )

    placebo_ratios: list[float] = []
    placebo_means: list[float] = []
    for unit in placebo_gaps.columns:
        unit_actual = pivot[unit].astype(float)
        unit_synthetic = unit_actual - placebo_gaps[unit]
        unit_gap_pct = _safe_pct_gap(unit_actual, unit_synthetic).to_numpy(dtype=float)
        unit_pre = _rmspe(unit_gap_pct[pre_mask])
        unit_post = _rmspe(unit_gap_pct[post_mask])
        if np.isfinite(unit_pre) and unit_pre > 0 and np.isfinite(unit_post):
            placebo_ratios.append(unit_post / unit_pre)
        unit_mean = float(np.nanmean(unit_gap_pct[post_mask]))
        if np.isfinite(unit_mean):
            placebo_means.append(unit_mean)

    ratio_rank = int(np.sum(np.asarray(placebo_ratios) >= ratio)) + 1
    ratio_p = float(ratio_rank / (len(placebo_ratios) + 1))
    if mean_post_gap >= 0:
        mean_rank = int(np.sum(np.asarray(placebo_means) >= mean_post_gap)) + 1
    else:
        mean_rank = int(np.sum(np.asarray(placebo_means) <= mean_post_gap)) + 1
    mean_p = float(mean_rank / (len(placebo_means) + 1))

    weights = synth.weights().sort_values(ascending=False)
    top_donor = str(weights.index[0])
    top_weight = float(weights.iloc[0])

    summary = {
        "Country": scenario.country,
        "ScenarioID": scenario.scenario_id,
        "ScenarioLabel": scenario.scenario_label,
        "ExcludedDonors": "; ".join(scenario.exclusions) if scenario.exclusions else "(none)",
        "DonorCount": len(controls),
        "TreatmentYear": treatment_year,
        "MeanPostGapPct": mean_post_gap,
        "PreRMSPEPct": pre_rmspe,
        "PostRMSPEPct": post_rmspe,
        "PostPreRMSPERatio": ratio,
        "RMSPERatioPlaceboRank": ratio_rank,
        "RMSPERatioPValue": ratio_p,
        "MeanGapPlaceboRank": mean_rank,
        "MeanGapPValue": mean_p,
        "TopDonor": top_donor,
        "TopDonorWeight": top_weight,
    }

    gap_df = pd.DataFrame(
        {
            "Country": scenario.country,
            "ScenarioID": scenario.scenario_id,
            "ScenarioLabel": scenario.scenario_label,
            "Year": years.astype(int),
            "GapPct": gap_pct.to_numpy(dtype=float),
        }
    )
    return summary, gap_df, weights


def _mean_annual_growth(series: pd.Series) -> float:
    return float(series.pct_change().dropna().mean() * 100.0)


def _corr_growth(a: pd.Series, b: pd.Series) -> float:
    a_growth = a.pct_change()
    b_growth = b.pct_change()
    return float(a_growth.corr(b_growth))


def _build_flow_diagnostics(
    chile_flow: pd.DataFrame,
    nz_df: pd.DataFrame,
    chile_baseline_weights: pd.Series,
    nz_baseline_weights: pd.Series,
) -> pd.DataFrame:
    chile_focus = [
        CHILE_TREATED,
        "VI Del Libertador General Bernardo OHiggins",
        "VIII Del Biobío",
        "IX De La Araucanía",
        "RMS Región Metropolitana de Santiago",
    ]
    nz_focus = [
        NZ_TREATED,
        "Auckland",
        "Wellington",
        "Waikato",
        "West Coast",
    ]

    rows: list[dict[str, Any]] = []

    treated_chile = (
        chile_flow[chile_flow["region_name"] == CHILE_TREATED]
        .set_index("year")
        .sort_index()["gdp_pc"]
        .loc[1991:2009]
    )
    for region in chile_focus:
        work = chile_flow[chile_flow["region_name"] == region].set_index("year").sort_index()
        pop_pre = work.loc[2005:2009, "population"]
        pop_post = work.loc[2010:2015, "population"]
        cons_pre = float(work.loc[2005:2009, "construcción"].mean())
        cons_post = float(work.loc[2010:2015, "construcción"].mean())
        trad_pre = float(
            work.loc[2005:2009, [
                "industria manufacturera",
                "comercio, restaurantes y hoteles",
                "transporte, información y comunicaciones",
            ]].sum(axis=1).mean()
        )
        corr_pre = _corr_growth(treated_chile, work.loc[1991:2009, "gdp_pc"])
        rows.append(
            {
                "Country": "Chile",
                "Region": region,
                "BaselineWeight": float(chile_baseline_weights.get(region, 0.0)),
                "PopulationGrowthPrePct": _mean_annual_growth(pop_pre),
                "PopulationGrowthPostPct": _mean_annual_growth(pop_post),
                "PopulationGrowthDeltaPP": _mean_annual_growth(pop_post) - _mean_annual_growth(pop_pre),
                "ConstructionSharePrePct": cons_pre,
                "ConstructionSharePostPct": cons_post,
                "ConstructionShareDeltaPP": cons_post - cons_pre,
                "TradablesSharePrePct": trad_pre,
                "GDPPerCapGrowthCorrPre": corr_pre,
            }
        )

    treated_nz = (
        nz_df[nz_df["Region"] == NZ_TREATED]
        .set_index("Year")
        .sort_index()["GDP per capita"]
        .loc[2001:2010]
    )
    for region in nz_focus:
        work = nz_df[nz_df["Region"] == region].set_index("Year").sort_index()
        pop_pre = work.loc[2005:2010, "Population"]
        pop_post = work.loc[2011:2019, "Population"]
        cons_pre = float(work.loc[2005:2010, "Construction"].mean() * 100.0)
        cons_post = float(work.loc[2011:2019, "Construction"].mean() * 100.0)
        trad_pre = float(
            work.loc[2005:2010, [
                "Manufacturing",
                "Wholesale Trade",
                "Transport, Postal and Warehousing",
            ]].sum(axis=1).mean() * 100.0
        )
        corr_pre = _corr_growth(treated_nz, work.loc[2001:2010, "GDP per capita"])
        rows.append(
            {
                "Country": "New Zealand",
                "Region": region,
                "BaselineWeight": float(nz_baseline_weights.get(region, 0.0)),
                "PopulationGrowthPrePct": _mean_annual_growth(pop_pre),
                "PopulationGrowthPostPct": _mean_annual_growth(pop_post),
                "PopulationGrowthDeltaPP": _mean_annual_growth(pop_post) - _mean_annual_growth(pop_pre),
                "ConstructionSharePrePct": cons_pre,
                "ConstructionSharePostPct": cons_post,
                "ConstructionShareDeltaPP": cons_post - cons_pre,
                "TradablesSharePrePct": trad_pre,
                "GDPPerCapGrowthCorrPre": corr_pre,
            }
        )

    return pd.DataFrame(rows)


def _plot_gap_paths(gaps: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=False)
    country_config = [
        ("Chile", 2010, axes[0]),
        ("New Zealand", 2011, axes[1]),
    ]

    colors = {
        "baseline": "black",
        "exclude_ohiggins": "#1f77b4",
        "exclude_ohiggins_araucania": "#2ca02c",
        "exclude_auckland_wellington": "#1f77b4",
        "exclude_metro_corridor": "#2ca02c",
    }

    for country, treatment_year, ax in country_config:
        sub = gaps[gaps["Country"] == country].copy()
        for scenario_id, scenario_frame in sub.groupby("ScenarioID"):
            label = scenario_frame["ScenarioLabel"].iloc[0]
            lw = 2.4 if scenario_id == "baseline" else 2.0
            ax.plot(
                scenario_frame["Year"],
                scenario_frame["GapPct"],
                label=label,
                color=colors.get(scenario_id, None),
                linewidth=lw,
            )
        ax.axhline(0.0, color="gray", linewidth=0.9, linestyle=":")
        ax.axvline(treatment_year, color="gray", linewidth=1.0, linestyle="--")
        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.set_ylabel("Treated minus synthetic gap (%)")
        ax.grid(alpha=0.2, linewidth=0.5)
        ax.legend(loc="best", fontsize=8, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(_ASSETS_DIR, "spillover_sensitivity_paths.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    chile_outcome = _prepare_chile_outcome_data()
    nz_outcome = _prepare_nz_outcome_data()
    chile_flow = _prepare_chile_flow_data()

    summary_rows: list[dict[str, Any]] = []
    gap_frames: list[pd.DataFrame] = []
    baseline_weights: dict[str, pd.Series] = {}

    for scenario in SCENARIOS:
        summary, gaps, weights = _evaluate_scenario(scenario, chile_outcome, nz_outcome)
        summary_rows.append(summary)
        gap_frames.append(gaps)
        if scenario.scenario_id == "baseline":
            baseline_weights[scenario.country] = weights

    summary_df = pd.DataFrame(summary_rows)
    for country in summary_df["Country"].unique():
        baseline_value = summary_df[
            (summary_df["Country"] == country) & (summary_df["ScenarioID"] == "baseline")
        ]["MeanPostGapPct"].iloc[0]
        mask = summary_df["Country"] == country
        summary_df.loc[mask, "DeltaVsBaselinePP"] = summary_df.loc[mask, "MeanPostGapPct"] - baseline_value

    gaps_df = pd.concat(gap_frames, ignore_index=True)
    flow_df = _build_flow_diagnostics(
        chile_flow=chile_flow,
        nz_df=nz_outcome,
        chile_baseline_weights=baseline_weights["Chile"],
        nz_baseline_weights=baseline_weights["New Zealand"],
    )

    summary_df.to_csv(
        os.path.join(_ASSETS_DIR, "spillover_sensitivity_summary.csv"),
        index=False,
    )
    gaps_df.to_csv(
        os.path.join(_ASSETS_DIR, "spillover_gap_paths.csv"),
        index=False,
    )
    flow_df.to_csv(
        os.path.join(_ASSETS_DIR, "spillover_flow_diagnostics.csv"),
        index=False,
    )
    _plot_gap_paths(gaps_df)

    print("Saved spillover diagnostics outputs:")
    print("- article_assets/spillover_sensitivity_summary.csv")
    print("- article_assets/spillover_gap_paths.csv")
    print("- article_assets/spillover_flow_diagnostics.csv")
    print("- article_assets/spillover_sensitivity_paths.png")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

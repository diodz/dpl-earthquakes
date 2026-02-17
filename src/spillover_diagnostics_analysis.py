import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth

import nz_util

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)


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
NZ_TREATMENT_YEAR = 2011
NZ_START_YEAR = 2000
NZ_END_YEAR = 2019
NZ_PRE_PROXY_YEARS = list(range(2005, 2011))
NZ_POST_PROXY_YEARS = list(range(2011, 2020))

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
CHILE_TREATMENT_YEAR = 2010
CHILE_START_YEAR = 1990
CHILE_END_YEAR = 2019
CHILE_PRE_PROXY_YEARS = list(range(2005, 2010))
CHILE_POST_PROXY_YEARS = list(range(2010, 2015))


CHILE_RISK_GROUP = {
    "VI Del Libertador General Bernardo OHiggins": "Adjacent",
    "V De Valparaíso": "Near-ring",
    "RMS Región Metropolitana de Santiago": "Near-ring",
    "IX De La Araucanía": "Near-ring",
}

NZ_RISK_GROUP = {
    "Auckland": "Network hub",
    "Wellington": "Network hub",
    "West Coast": "Adjacent ring",
    "Otago": "Adjacent ring",
    "Marlborough": "Adjacent ring",
    "Tasman/Nelson": "Adjacent ring",
}

RISK_COLOR = {
    "Treated": "#d62728",
    "Adjacent": "#e6550d",
    "Near-ring": "#fd8d3c",
    "Adjacent ring": "#ff9896",
    "Network hub": "#9467bd",
    "Other donor": "#7f7f7f",
}


@dataclass(frozen=True)
class ExclusionScenario:
    country: str
    order: int
    label: str
    description: str
    excluded_controls: tuple[str, ...]


SCENARIOS: list[ExclusionScenario] = [
    ExclusionScenario(
        country="Chile",
        order=1,
        label="Baseline donor pool",
        description="No additional exclusion beyond main specification.",
        excluded_controls=(),
    ),
    ExclusionScenario(
        country="Chile",
        order=2,
        label="Exclude adjacent O'Higgins",
        description="Drops VI (O'Higgins) as the closest unaffected neighbor.",
        excluded_controls=("VI Del Libertador General Bernardo OHiggins",),
    ),
    ExclusionScenario(
        country="Chile",
        order=3,
        label="Exclude adjacent + near-ring",
        description="Drops O'Higgins plus the southern near-ring (Araucania).",
        excluded_controls=(
            "VI Del Libertador General Bernardo OHiggins",
            "IX De La Araucanía",
        ),
    ),
    ExclusionScenario(
        country="New Zealand",
        order=1,
        label="Baseline donor pool",
        description="No additional exclusion beyond main specification.",
        excluded_controls=(),
    ),
    ExclusionScenario(
        country="New Zealand",
        order=2,
        label="Exclude Auckland and Wellington",
        description="Network spillover check requested by reviewers.",
        excluded_controls=("Auckland", "Wellington"),
    ),
    ExclusionScenario(
        country="New Zealand",
        order=3,
        label="Exclude adjacent South Island ring",
        description="Drops West Coast, Otago, Marlborough, Tasman/Nelson.",
        excluded_controls=("West Coast", "Otago", "Marlborough", "Tasman/Nelson"),
    ),
    ExclusionScenario(
        country="New Zealand",
        order=4,
        label="Exclude ring + network hubs",
        description="Strictest NZ donor pool: adjacent ring plus Auckland/Wellington.",
        excluded_controls=(
            "West Coast",
            "Otago",
            "Marlborough",
            "Tasman/Nelson",
            "Auckland",
            "Wellington",
        ),
    ),
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


def _safe_corr(lhs: pd.Series, rhs: pd.Series) -> float:
    joined = pd.concat([lhs, rhs], axis=1).dropna()
    if joined.shape[0] < 3:
        return float("nan")
    return float(joined.corr().iloc[0, 1])


def _build_nz_dataprep(df: pd.DataFrame, controls: list[str]) -> Dataprep:
    return Dataprep(
        foo=df,
        predictors=nz_util.SECTORIAL_GDP_VARIABLES,
        predictors_op="mean",
        time_predictors_prior=range(2006, 2011),
        special_predictors=[
            ("GDP per capita", range(2006, 2011), "mean"),
            ("Tertiary Share", range(2010, 2011), "mean"),
        ],
        dependent="GDP per capita",
        unit_variable="Region",
        time_variable="Year",
        treatment_identifier=NZ_TREATED,
        controls_identifier=controls,
        time_optimize_ssr=range(2000, 2011),
    )


def _build_chile_dataprep(df: pd.DataFrame, controls: list[str]) -> Dataprep:
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
        controls_identifier=controls,
        time_optimize_ssr=range(1990, 2009),
    )


def _evaluate_scenario(
    *,
    scenario: ExclusionScenario,
    nz_df: pd.DataFrame,
    chile_df: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    if scenario.country == "New Zealand":
        controls = [region for region in NZ_CONTROLS if region not in scenario.excluded_controls]
        dataprep = _build_nz_dataprep(nz_df, controls)
        years = list(range(NZ_START_YEAR, NZ_END_YEAR + 1))
        treatment_year = NZ_TREATMENT_YEAR
        optim_method = "Nelder-Mead"
        optim_initial = "equal"
    else:
        controls = [region for region in CHILE_CONTROLS if region not in scenario.excluded_controls]
        dataprep = _build_chile_dataprep(chile_df, controls)
        years = list(range(CHILE_START_YEAR, CHILE_END_YEAR + 1))
        treatment_year = CHILE_TREATMENT_YEAR
        optim_method = "Nelder-Mead"
        optim_initial = "ols"

    if len(controls) < 2:
        raise ValueError(f"Scenario {scenario.label} has too few donors: {len(controls)}.")

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method=optim_method, optim_initial=optim_initial)

    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    synthetic = pd.Series(np.asarray(synth._synthetic(z0)).flatten().astype(float), index=years)
    treated = pd.Series(np.asarray(z1).flatten().astype(float), index=years)
    gap_pct = (treated - synthetic) / synthetic * 100.0

    year_index = np.asarray(years, dtype=int)
    pre_mask = year_index < treatment_year
    post_mask = year_index >= treatment_year

    pre_rmspe = _rmspe(gap_pct.to_numpy(dtype=float)[pre_mask])
    post_rmspe = _rmspe(gap_pct.to_numpy(dtype=float)[post_mask])
    mean_post_gap = float(np.nanmean(gap_pct.to_numpy(dtype=float)[post_mask]))

    summary_row = {
        "Country": scenario.country,
        "ScenarioOrder": scenario.order,
        "ScenarioLabel": scenario.label,
        "ScenarioDescription": scenario.description,
        "ExcludedControls": ";".join(scenario.excluded_controls),
        "NumControls": len(controls),
        "MeanPostGapPct": mean_post_gap,
        "PreRMSPEPct": pre_rmspe,
        "PostRMSPEPct": post_rmspe,
        "PostPreRMSPERatio": _safe_ratio(post_rmspe, pre_rmspe),
        "TreatmentYear": treatment_year,
        "PostWindowStart": treatment_year,
        "PostWindowEnd": years[-1],
    }
    gap_rows = [
        {
            "Country": scenario.country,
            "ScenarioOrder": scenario.order,
            "ScenarioLabel": scenario.label,
            "Year": int(year),
            "GapPct": float(gap_pct.loc[year]),
            "Treated": float(treated.loc[year]),
            "Synthetic": float(synthetic.loc[year]),
        }
        for year in years
    ]
    return summary_row, gap_rows


def _compute_channel_diagnostics(
    *,
    country: str,
    df: pd.DataFrame,
    unit_col: str,
    year_col: str,
    outcome_col: str,
    population_col: str,
    labor_col: str,
    labor_proxy_label: str,
    treated: str,
    controls: list[str],
    pre_years: list[int],
    post_years: list[int],
    risk_group_map: dict[str, str],
    population_source: str,
    labor_source: str,
    io_source: str,
) -> pd.DataFrame:
    work = df[df[unit_col].isin([treated] + controls)].copy()
    work = work.sort_values([unit_col, year_col])

    outcome_wide = work.pivot(index=year_col, columns=unit_col, values=outcome_col).sort_index()
    population_wide = work.pivot(index=year_col, columns=unit_col, values=population_col).sort_index()
    labor_wide = work.pivot(index=year_col, columns=unit_col, values=labor_col).sort_index()

    outcome_growth = outcome_wide.pct_change(fill_method=None) * 100.0
    population_growth = population_wide.pct_change(fill_method=None) * 100.0

    rows: list[dict] = []
    all_units = [treated] + controls
    for unit in all_units:
        risk_group = "Treated" if unit == treated else risk_group_map.get(unit, "Other donor")
        corr = _safe_corr(
            outcome_growth.loc[outcome_growth.index.isin(pre_years), treated],
            outcome_growth.loc[outcome_growth.index.isin(pre_years), unit],
        )
        pop_pre = float(population_growth.loc[population_growth.index.isin(pre_years), unit].mean())
        pop_post = float(population_growth.loc[population_growth.index.isin(post_years), unit].mean())
        pop_shift = pop_post - pop_pre

        labor_pre = float(labor_wide.loc[labor_wide.index.isin(pre_years), unit].mean())
        labor_post = float(labor_wide.loc[labor_wide.index.isin(post_years), unit].mean())
        labor_shift_pp = (labor_post - labor_pre) * 100.0

        rows.append(
            {
                "Country": country,
                "Region": unit,
                "RiskGroup": risk_group,
                "PreTreatmentGDPGrowthCorrelation": corr,
                "PopulationGrowthPrePct": pop_pre,
                "PopulationGrowthPostPct": pop_post,
                "PopulationGrowthShiftPctPoints": pop_shift,
                "LaborProxyLabel": labor_proxy_label,
                "LaborProxyPreLevel": labor_pre,
                "LaborProxyPostLevel": labor_post,
                "LaborProxyShiftPctPoints": labor_shift_pp,
                "PreWindow": f"{min(pre_years)}-{max(pre_years)}",
                "PostWindow": f"{min(post_years)}-{max(post_years)}",
                "PopulationSource": population_source,
                "LaborSource": labor_source,
                "TradeIOProxySource": io_source,
            }
        )

    out = pd.DataFrame(rows)
    donor_mask = out["RiskGroup"] != "Treated"
    donor_pop_median = float(out.loc[donor_mask, "PopulationGrowthShiftPctPoints"].median())
    donor_labor_median = float(out.loc[donor_mask, "LaborProxyShiftPctPoints"].median())
    out["PopulationShiftVsDonorMedianPctPoints"] = (
        out["PopulationGrowthShiftPctPoints"] - donor_pop_median
    )
    out["LaborShiftVsDonorMedianPctPoints"] = (
        out["LaborProxyShiftPctPoints"] - donor_labor_median
    )
    return out.sort_values(["Country", "RiskGroup", "Region"]).reset_index(drop=True)


def _plot_exclusion_effects(summary_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, country in zip(axes, ["Chile", "New Zealand"], strict=True):
        subset = summary_df[summary_df["Country"] == country].sort_values("ScenarioOrder")
        colors = ["#d62728" if "Baseline" in label else "#4c78a8" for label in subset["ScenarioLabel"]]
        bars = axis.bar(np.arange(subset.shape[0]), subset["MeanPostGapPct"], color=colors, alpha=0.9)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_title(country)
        axis.set_xticks(np.arange(subset.shape[0]))
        axis.set_xticklabels(subset["ScenarioLabel"], rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, subset["MeanPostGapPct"], strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + (0.15 if value >= 0 else -0.15),
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=8,
            )
    axes[0].set_ylabel("Mean post-treatment gap (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_exclusion_paths(gaps_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    treatment_year_map = {"Chile": CHILE_TREATMENT_YEAR, "New Zealand": NZ_TREATMENT_YEAR}

    for axis, country in zip(axes, ["Chile", "New Zealand"], strict=True):
        subset = gaps_df[gaps_df["Country"] == country]
        scenario_order = (
            subset[["ScenarioOrder", "ScenarioLabel"]]
            .drop_duplicates()
            .sort_values("ScenarioOrder")
            .reset_index(drop=True)
        )
        for _, row in scenario_order.iterrows():
            scenario_subset = subset[subset["ScenarioLabel"] == row["ScenarioLabel"]]
            is_baseline = "Baseline" in row["ScenarioLabel"]
            axis.plot(
                scenario_subset["Year"],
                scenario_subset["GapPct"],
                label=row["ScenarioLabel"],
                linewidth=2.2 if is_baseline else 1.5,
                linestyle="-" if is_baseline else "--",
                alpha=0.95,
            )
        axis.axvline(treatment_year_map[country], color="black", linestyle=":", linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_title(country)
        axis.set_xlabel("Year")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Gap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_flow_proxies(channel_df: pd.DataFrame, output_path: str) -> None:
    donor_df = channel_df[channel_df["RiskGroup"] != "Treated"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")
    configs = [("Chile", "PopulationGrowthShiftPctPoints"), ("New Zealand", "PopulationGrowthShiftPctPoints")]
    labor_configs = [("Chile", "LaborProxyShiftPctPoints"), ("New Zealand", "LaborProxyShiftPctPoints")]

    for col_idx, (country, metric) in enumerate(configs):
        axis = axes[0, col_idx]
        subset = donor_df[donor_df["Country"] == country].copy()
        subset = subset.sort_values(["RiskGroup", "Region"])
        colors = [RISK_COLOR.get(group, "#7f7f7f") for group in subset["RiskGroup"]]
        axis.bar(subset["Region"], subset[metric], color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_title(f"{country}: migration proxy")
        axis.set_ylabel("Population growth shift (p.p.)")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=65, labelsize=8)

    for col_idx, (country, metric) in enumerate(labor_configs):
        axis = axes[1, col_idx]
        subset = donor_df[donor_df["Country"] == country].copy()
        subset = subset.sort_values(["RiskGroup", "Region"])
        colors = [RISK_COLOR.get(group, "#7f7f7f") for group in subset["RiskGroup"]]
        axis.bar(subset["Region"], subset[metric], color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_title(f"{country}: labor proxy")
        axis.set_ylabel("Labor proxy shift (p.p.)")
        axis.grid(axis="y", alpha=0.2)
        axis.tick_params(axis="x", rotation=65, labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_spillover_diagnostics(output_dir: str = FIGURES_DIR) -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    nz_df = nz_util.clean_data_for_synthetic_control().copy()
    nz_df["Tertiary Share"] = nz_df["Tertiary"] / nz_df["Population"]
    chile_df = pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv"))

    summary_rows: list[dict] = []
    gap_rows: list[dict] = []
    for scenario in SCENARIOS:
        summary, gaps = _evaluate_scenario(scenario=scenario, nz_df=nz_df, chile_df=chile_df)
        summary_rows.append(summary)
        gap_rows.extend(gaps)

    summary_df = pd.DataFrame(summary_rows).sort_values(["Country", "ScenarioOrder"])
    gap_df = pd.DataFrame(gap_rows).sort_values(["Country", "ScenarioOrder", "Year"])

    baseline_by_country = (
        summary_df[summary_df["ScenarioLabel"] == "Baseline donor pool"]
        .set_index("Country")["MeanPostGapPct"]
        .to_dict()
    )
    summary_df["BaselineMeanPostGapPct"] = summary_df["Country"].map(baseline_by_country)
    summary_df["DeltaVsBaselinePctPoints"] = (
        summary_df["MeanPostGapPct"] - summary_df["BaselineMeanPostGapPct"]
    )

    strictest = summary_df.sort_values(["Country", "ScenarioOrder"], ascending=[True, False])
    strictest = strictest.groupby("Country", as_index=False).head(1)
    strictest = strictest.rename(
        columns={
            "ScenarioLabel": "MostDemandingScenario",
            "MeanPostGapPct": "MostDemandingScenarioMeanPostGapPct",
        }
    )[
        [
            "Country",
            "MostDemandingScenario",
            "MostDemandingScenarioMeanPostGapPct",
            "DeltaVsBaselinePctPoints",
            "NumControls",
        ]
    ]

    nz_channels = _compute_channel_diagnostics(
        country="New Zealand",
        df=nz_df,
        unit_col="Region",
        year_col="Year",
        outcome_col="GDP per capita",
        population_col="Population",
        labor_col="Construction",
        labor_proxy_label="Construction share of regional GDP",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        pre_years=NZ_PRE_PROXY_YEARS,
        post_years=NZ_POST_PROXY_YEARS,
        risk_group_map=NZ_RISK_GROUP,
        population_source="Stats NZ regional GDP/population release",
        labor_source="Stats NZ industry shares (construction)",
        io_source="Pre-treatment GDP growth comovement proxy",
    )
    chile_channels = _compute_channel_diagnostics(
        country="Chile",
        df=chile_df,
        unit_col="region_name",
        year_col="year",
        outcome_col="gdp_cap",
        population_col="population",
        labor_col="ed_superior_cap",
        labor_proxy_label="Tertiary enrollment per capita (labor-supply proxy)",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        pre_years=CHILE_PRE_PROXY_YEARS,
        post_years=CHILE_POST_PROXY_YEARS,
        risk_group_map=CHILE_RISK_GROUP,
        population_source="INE regional population (processed in inter/processed_chile.csv)",
        labor_source="MINEDUC tertiary enrollment per capita (processed series)",
        io_source="Pre-treatment GDP growth comovement proxy",
    )
    channel_df = pd.concat([chile_channels, nz_channels], ignore_index=True)

    channel_group_summary = (
        channel_df[channel_df["RiskGroup"] != "Treated"]
        .groupby(["Country", "RiskGroup"], as_index=False)
        .agg(
            MeanGDPGrowthCorrelation=("PreTreatmentGDPGrowthCorrelation", "mean"),
            MeanPopulationShiftPctPoints=("PopulationGrowthShiftPctPoints", "mean"),
            MeanLaborShiftPctPoints=("LaborProxyShiftPctPoints", "mean"),
            Regions=("Region", lambda x: ";".join(sorted(x))),
        )
    )

    summary_df.to_csv(os.path.join(output_dir, "spillover_exclusion_summary.csv"), index=False)
    gap_df.to_csv(os.path.join(output_dir, "spillover_exclusion_gaps.csv"), index=False)
    strictest.to_csv(
        os.path.join(output_dir, "spillover_baseline_vs_strictest_summary.csv"), index=False
    )
    channel_df.to_csv(os.path.join(output_dir, "spillover_channel_diagnostics.csv"), index=False)
    channel_group_summary.to_csv(
        os.path.join(output_dir, "spillover_channel_group_summary.csv"), index=False
    )

    _plot_exclusion_effects(
        summary_df, os.path.join(output_dir, "spillover_exclusion_effects.png")
    )
    _plot_exclusion_paths(gap_df, os.path.join(output_dir, "spillover_exclusion_gap_paths.png"))
    _plot_flow_proxies(channel_df, os.path.join(output_dir, "spillover_flow_proxies.png"))

    return {
        "spillover_summary": summary_df,
        "spillover_gaps": gap_df,
        "spillover_strictest": strictest,
        "spillover_channels": channel_df,
        "spillover_channel_groups": channel_group_summary,
    }


if __name__ == "__main__":
    outputs = run_spillover_diagnostics()
    print(outputs["spillover_summary"].to_string(index=False))

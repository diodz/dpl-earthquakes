import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth
from pysyncon.utils import PlaceboTest

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

CHILE_TREATED = "maule"
CHILE_CONTROLS = [
    "tarapaca",
    "antofagasta",
    "atacama",
    "coquimbo",
    "valparaiso",
    "metropolitana",
    "ohiggins",
    "araucania",
    "loslagos",
    "aysen",
    "magallanes",
]
CHILE_TREATMENT_YEAR = 2010
CHILE_START_YEAR = 1990
CHILE_END_YEAR = 2015


@dataclass
class ModelResult:
    country: str
    outcome: str
    scale: str
    years: list[int]
    treated: pd.Series
    synthetic: pd.Series
    gap: pd.Series
    mean_post_gap: float
    mean_post_gap_pct_of_synthetic: float
    pre_rmspe: float
    post_rmspe: float
    rmspe_ratio: float
    pvalue: float
    treatment_year: int
    post_end_year: int

    def summary_row(self) -> dict:
        row = {
            "Country": self.country,
            "Outcome": self.outcome,
            "Scale": self.scale,
            "TreatmentYear": self.treatment_year,
            "PostEndYear": self.post_end_year,
            "MeanPostGap": self.mean_post_gap,
            "MeanPostGapPctOfSynthetic": self.mean_post_gap_pct_of_synthetic,
            "PreRMSPE": self.pre_rmspe,
            "PostRMSPE": self.post_rmspe,
            "RMSPERatio": self.rmspe_ratio,
            "PseudoPValue": self.pvalue,
        }
        if self.scale == "share":
            row["MeanPostGapPP"] = self.mean_post_gap * 100.0
        else:
            row["MeanPostGapPP"] = np.nan
        return row


def _safe_rms(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def _fit_scm_model(
    *,
    df: pd.DataFrame,
    country: str,
    outcome: str,
    scale: str,
    predictors: list[str],
    special_predictors: list[tuple[str, Iterable[int], str]],
    time_predictors_prior: range,
    unit_col: str,
    time_col: str,
    treated: str,
    controls: list[str],
    time_optimize_ssr: range,
    years: list[int],
    treatment_year: int,
    post_end_year: int,
    optim_method: str,
    optim_initial: str,
) -> tuple[ModelResult, PlaceboTest]:
    dataprep = Dataprep(
        foo=df,
        predictors=predictors,
        predictors_op="mean",
        time_predictors_prior=time_predictors_prior,
        special_predictors=special_predictors,
        dependent=outcome,
        unit_variable=unit_col,
        time_variable=time_col,
        treatment_identifier=treated,
        controls_identifier=controls,
        time_optimize_ssr=time_optimize_ssr,
    )

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method=optim_method, optim_initial=optim_initial)

    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    synthetic = pd.Series(synth._synthetic(z0).values.astype(float), index=years)
    treated_series = pd.Series(z1.values.astype(float), index=years)
    gap = treated_series - synthetic

    pre_mask = gap.index < treatment_year
    post_mask = (gap.index >= treatment_year) & (gap.index <= post_end_year)

    pre_gap = gap[pre_mask]
    post_gap = gap[post_mask]

    pre_rmspe = _safe_rms(pre_gap)
    post_rmspe = _safe_rms(post_gap)
    rmspe_ratio = float(post_rmspe / pre_rmspe) if pre_rmspe else np.nan
    mean_post_gap = float(post_gap.mean())
    mean_post_gap_pct_of_synthetic = float(
        ((post_gap / synthetic[post_mask]) * 100.0).mean()
    )

    placebo_test = PlaceboTest()
    placebo_test.fit(
        dataprep=dataprep,
        scm=synth,
        scm_options={"optim_method": optim_method, "optim_initial": optim_initial},
    )
    pvalue = float(placebo_test.pvalue(treatment_time=treatment_year))

    result = ModelResult(
        country=country,
        outcome=outcome,
        scale=scale,
        years=years,
        treated=treated_series,
        synthetic=synthetic,
        gap=gap,
        mean_post_gap=mean_post_gap,
        mean_post_gap_pct_of_synthetic=mean_post_gap_pct_of_synthetic,
        pre_rmspe=pre_rmspe,
        post_rmspe=post_rmspe,
        rmspe_ratio=rmspe_ratio,
        pvalue=pvalue,
        treatment_year=treatment_year,
        post_end_year=post_end_year,
    )
    return result, placebo_test


def _plot_paths(result: ModelResult, output_name: str, y_label: str) -> None:
    is_share = result.scale == "share"
    scale_factor = 100.0 if is_share else 1.0
    y_suffix = " (%)" if is_share else ""

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(
        result.years,
        result.treated.values * scale_factor,
        color="#d62728",
        linewidth=1.8,
        label=f"{result.country} actual",
    )
    ax.plot(
        result.years,
        result.synthetic.values * scale_factor,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=f"Synthetic {result.country}",
    )
    ax.axvline(result.treatment_year, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{y_label}{y_suffix}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, output_name), dpi=220)
    plt.close(fig)


def _plot_placebo_gaps(
    result: ModelResult,
    placebo_test: PlaceboTest,
    output_name: str,
    y_label: str,
    mspe_threshold: float = 100.0,
) -> None:
    gaps = placebo_test.gaps.copy()
    treated_gap = placebo_test.treated_gap.copy()
    years = result.years

    if mspe_threshold:
        # Use treatment_year - 1 to exclude treatment year from pre-period
        # (treatment_year is the first post-treatment year per post_mask logic)
        pre_mspe = gaps.loc[: result.treatment_year - 1].pow(2).sum(axis=0)
        pre_mspe_treated = treated_gap.loc[: result.treatment_year - 1].pow(2).sum(axis=0)
        keep_units = pre_mspe[pre_mspe < mspe_threshold * pre_mspe_treated].index
        gaps = gaps[keep_units]

    gaps = gaps[gaps.index.isin(years)]
    treated_gap = treated_gap[treated_gap.index.isin(years)]

    scale_factor = 100.0 if result.scale == "share" else 1.0
    y_suffix = " (%)" if result.scale == "share" else ""

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(gaps.index, gaps.values * scale_factor, color="gray", alpha=0.15)
    ax.plot(
        treated_gap.index,
        treated_gap.values * scale_factor,
        color="#d62728",
        linewidth=1.8,
        label="Treated gap",
    )
    ax.axvline(result.treatment_year, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{y_label}{y_suffix}")
    ax.legend(loc="best")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, output_name), dpi=220)
    plt.close(fig)


def _build_nz_dataset() -> tuple[pd.DataFrame, list[str]]:
    df = nz_util.clean_data_for_synthetic_control().copy()
    df["Tertiary Share"] = df["Tertiary"] / df["Population"]
    df["Construction_share"] = df["Construction"].astype(float)
    df["NonConstruction_share"] = 1.0 - df["Construction_share"]
    df["Construction_level"] = df["Construction_share"] * df["Gross Domestic Product"]
    df["NonConstruction_level"] = df["Gross Domestic Product"] - df["Construction_level"]
    df["Goods_non_construction_share"] = (
        df["Agriculture"].astype(float) + df["Manufacturing"].astype(float)
    )
    df["Services_non_construction_share"] = (
        df["NonConstruction_share"] - df["Goods_non_construction_share"]
    )
    return df, nz_util.SECTORIAL_GDP_VARIABLES


def _build_chile_dataset() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(os.path.join(_PROJECT_ROOT, "data", "scm_2010.csv")).copy()
    rename_map = {
        "nom_region": "Region",
        "administración pública": "adm_pub_share_pct",
        "agropecuario-silvícola": "agro_share_pct",
        "comercio, restaurantes y hoteles": "commerce_share_pct",
        "construcción": "construction_share_pct",
        "electricidad, gas, agua y gestión de desechos": "electricity_share_pct",
        "industria manufacturera": "industry_share_pct",
        "minería": "mining_share_pct",
        "pesca": "fishery_share_pct",
        "servicios de vivienda e inmobiliarios": "housing_share_pct",
        "servicios financieros y empresariales": "finance_share_pct",
        "servicios personales": "personal_share_pct",
        "transporte, información y comunicaciones": "transport_share_pct",
        "producto interno bruto pct": "gdp_share_pct_total",
    }
    df = df.rename(columns=rename_map)

    share_pct_cols = [
        "adm_pub_share_pct",
        "agro_share_pct",
        "commerce_share_pct",
        "construction_share_pct",
        "electricity_share_pct",
        "industry_share_pct",
        "mining_share_pct",
        "fishery_share_pct",
        "housing_share_pct",
        "finance_share_pct",
        "personal_share_pct",
        "transport_share_pct",
    ]
    df["sector_share_pct_total"] = df[share_pct_cols].sum(axis=1)

    normalized_share_cols = []
    for col in share_pct_cols:
        norm_col = col.replace("_pct", "")
        df[norm_col] = df[col] / df["sector_share_pct_total"]
        normalized_share_cols.append(norm_col)

    df["Construction_share"] = df["construction_share"]  # normalized
    df["NonConstruction_share"] = 1.0 - df["Construction_share"]

    level_cols_non_construction = [
        "adm_publica",
        "agropecuario",
        "comercio",
        "electricidad",
        "industria",
        "mineria",
        "pesca_pct",
        "vivienda",
        "financieros",
        "personales",
        "transporte",
    ]
    df["Construction_level"] = df["construccion"]
    df["NonConstruction_level"] = df[level_cols_non_construction].sum(axis=1)

    df["Goods_non_construction_share"] = (
        df["agro_share"]
        + df["mining_share"]
        + df["industry_share"]
        + df["fishery_share"]
        + df["electricity_share"]
    )
    df["Services_non_construction_share"] = (
        df["adm_pub_share"]
        + df["commerce_share"]
        + df["housing_share"]
        + df["finance_share"]
        + df["personal_share"]
        + df["transport_share"]
    )
    return df, normalized_share_cols


def run_sectoral_appendix_analysis(output_dir: str = FIGURES_DIR) -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    nz_df, nz_share_cols = _build_nz_dataset()
    chile_df, chile_share_cols = _build_chile_dataset()

    nz_years = list(range(NZ_START_YEAR, NZ_END_YEAR + 1))
    chile_years = list(range(CHILE_START_YEAR, CHILE_END_YEAR + 1))

    baseline_results: list[ModelResult] = []
    path_rows: list[dict] = []
    grouping_results: list[ModelResult] = []

    def _store_series(result: ModelResult) -> None:
        for year in result.years:
            path_rows.append(
                {
                    "Country": result.country,
                    "Outcome": result.outcome,
                    "Scale": result.scale,
                    "Year": year,
                    "Treated": float(result.treated.loc[year]),
                    "Synthetic": float(result.synthetic.loc[year]),
                    "Gap": float(result.gap.loc[year]),
                }
            )

    nz_special = [
        ("GDP per capita", range(2005, 2009), "mean"),
        ("Tertiary Share", range(2008, 2009), "mean"),
    ]
    chile_special = [("gdp_pc", range(2005, 2009), "mean")]

    # New Zealand baseline models.
    nz_construction_share, nz_construction_share_placebo = _fit_scm_model(
        df=nz_df,
        country="New Zealand",
        outcome="Construction_share",
        scale="share",
        predictors=[col for col in nz_share_cols if col != "Construction"],
        special_predictors=nz_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="Year",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2010),
        years=nz_years,
        treatment_year=NZ_TREATMENT_YEAR,
        post_end_year=NZ_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="equal",
    )
    baseline_results.append(nz_construction_share)
    _store_series(nz_construction_share)
    _plot_paths(
        nz_construction_share,
        output_name="nz_scm_Construction.png",
        y_label="Construction share",
    )
    _plot_placebo_gaps(
        nz_construction_share,
        nz_construction_share_placebo,
        output_name="nz_sector_placebo_construction_share.png",
        y_label="Gap",
    )

    nz_other_share, _ = _fit_scm_model(
        df=nz_df,
        country="New Zealand",
        outcome="NonConstruction_share",
        scale="share",
        predictors=["Construction"],
        special_predictors=nz_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="Year",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2010),
        years=nz_years,
        treatment_year=NZ_TREATMENT_YEAR,
        post_end_year=NZ_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="equal",
    )
    baseline_results.append(nz_other_share)
    _store_series(nz_other_share)
    _plot_paths(
        nz_other_share,
        output_name="nz_scm_Other_Sectors.png",
        y_label="Non-construction share",
    )

    nz_other_level, nz_other_level_placebo = _fit_scm_model(
        df=nz_df,
        country="New Zealand",
        outcome="NonConstruction_level",
        scale="level",
        predictors=nz_share_cols,
        special_predictors=nz_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="Year",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2010),
        years=nz_years,
        treatment_year=NZ_TREATMENT_YEAR,
        post_end_year=NZ_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="equal",
    )
    baseline_results.append(nz_other_level)
    _store_series(nz_other_level)
    _plot_paths(
        nz_other_level,
        output_name="nz_scm_nonconstruction_level.png",
        y_label="Non-construction GDP level",
    )
    _plot_placebo_gaps(
        nz_other_level,
        nz_other_level_placebo,
        output_name="nz_sector_placebo_nonconstruction_level.png",
        y_label="Gap",
    )

    # Chile baseline models.
    chile_construction_share, chile_construction_share_placebo = _fit_scm_model(
        df=chile_df,
        country="Chile",
        outcome="Construction_share",
        scale="share",
        predictors=[col for col in chile_share_cols if col != "construction_share"],
        special_predictors=chile_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="year",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, 2009),
        years=chile_years,
        treatment_year=CHILE_TREATMENT_YEAR,
        post_end_year=CHILE_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="ols",
    )
    baseline_results.append(chile_construction_share)
    _store_series(chile_construction_share)
    _plot_paths(
        chile_construction_share,
        output_name="chile_scm_Construction.png",
        y_label="Construction share",
    )
    _plot_placebo_gaps(
        chile_construction_share,
        chile_construction_share_placebo,
        output_name="chile_sector_placebo_construction_share.png",
        y_label="Gap",
    )

    chile_other_share, _ = _fit_scm_model(
        df=chile_df,
        country="Chile",
        outcome="NonConstruction_share",
        scale="share",
        predictors=["construction_share"],
        special_predictors=chile_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="year",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, 2009),
        years=chile_years,
        treatment_year=CHILE_TREATMENT_YEAR,
        post_end_year=CHILE_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="ols",
    )
    baseline_results.append(chile_other_share)
    _store_series(chile_other_share)
    _plot_paths(
        chile_other_share,
        output_name="chile_scm_Other_Sectors.png",
        y_label="Non-construction share",
    )

    chile_other_level, chile_other_level_placebo = _fit_scm_model(
        df=chile_df,
        country="Chile",
        outcome="NonConstruction_level",
        scale="level",
        predictors=["construccion"],
        special_predictors=chile_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="year",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, 2009),
        years=chile_years,
        treatment_year=CHILE_TREATMENT_YEAR,
        post_end_year=CHILE_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="ols",
    )
    baseline_results.append(chile_other_level)
    _store_series(chile_other_level)
    _plot_paths(
        chile_other_level,
        output_name="chile_scm_nonconstruction_level.png",
        y_label="Non-construction GDP level",
    )
    _plot_placebo_gaps(
        chile_other_level,
        chile_other_level_placebo,
        output_name="chile_sector_placebo_nonconstruction_level.png",
        y_label="Gap",
    )

    # Alternative grouping sensitivity.
    nz_goods_share, _ = _fit_scm_model(
        df=nz_df,
        country="New Zealand",
        outcome="Goods_non_construction_share",
        scale="share",
        predictors=[col for col in nz_share_cols if col not in ["Agriculture", "Manufacturing"]],
        special_predictors=nz_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="Year",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2010),
        years=nz_years,
        treatment_year=NZ_TREATMENT_YEAR,
        post_end_year=NZ_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="equal",
    )
    grouping_results.append(nz_goods_share)

    nz_services_share, _ = _fit_scm_model(
        df=nz_df,
        country="New Zealand",
        outcome="Services_non_construction_share",
        scale="share",
        predictors=[col for col in nz_share_cols if col != "Construction"],
        special_predictors=nz_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="Year",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2010),
        years=nz_years,
        treatment_year=NZ_TREATMENT_YEAR,
        post_end_year=NZ_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="equal",
    )
    grouping_results.append(nz_services_share)

    chile_goods_share, _ = _fit_scm_model(
        df=chile_df,
        country="Chile",
        outcome="Goods_non_construction_share",
        scale="share",
        predictors=[
            col
            for col in chile_share_cols
            if col
            not in [
                "agro_share",
                "mining_share",
                "industry_share",
                "fishery_share",
                "electricity_share",
            ]
        ],
        special_predictors=chile_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="year",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, 2009),
        years=chile_years,
        treatment_year=CHILE_TREATMENT_YEAR,
        post_end_year=CHILE_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="ols",
    )
    grouping_results.append(chile_goods_share)

    chile_services_share, _ = _fit_scm_model(
        df=chile_df,
        country="Chile",
        outcome="Services_non_construction_share",
        scale="share",
        predictors=[
            col
            for col in chile_share_cols
            if col
            not in [
                "adm_pub_share",
                "commerce_share",
                "housing_share",
                "finance_share",
                "personal_share",
                "transport_share",
            ]
        ],
        special_predictors=chile_special,
        time_predictors_prior=range(2005, 2009),
        unit_col="Region",
        time_col="year",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        time_optimize_ssr=range(1990, 2009),
        years=chile_years,
        treatment_year=CHILE_TREATMENT_YEAR,
        post_end_year=CHILE_END_YEAR,
        optim_method="Nelder-Mead",
        optim_initial="ols",
    )
    grouping_results.append(chile_services_share)

    summary_df = pd.DataFrame([row.summary_row() for row in baseline_results])
    grouping_df = pd.DataFrame([row.summary_row() for row in grouping_results])

    baseline_lookup = {f"{row.country}|{row.outcome}": row for row in baseline_results}
    crowding_rows = []
    for country in ["Chile", "New Zealand"]:
        c_share = baseline_lookup[f"{country}|Construction_share"]
        nc_share = baseline_lookup[f"{country}|NonConstruction_share"]
        nc_level = baseline_lookup[f"{country}|NonConstruction_level"]

        if nc_share.mean_post_gap < 0 and nc_level.mean_post_gap >= 0:
            interpretation = "Share crowding-out without absolute contraction"
        elif nc_share.mean_post_gap < 0 and nc_level.mean_post_gap < 0:
            interpretation = "Share and absolute crowding-out"
        else:
            interpretation = "No evidence of crowding-out in either metric"

        crowding_rows.append(
            {
                "Country": country,
                "ConstructionShareGapPP": c_share.mean_post_gap * 100.0,
                "NonConstructionShareGapPP": nc_share.mean_post_gap * 100.0,
                "NonConstructionLevelGap": nc_level.mean_post_gap,
                "NonConstructionLevelGapPctOfSynthetic": nc_level.mean_post_gap_pct_of_synthetic,
                "Interpretation": interpretation,
            }
        )
    crowding_df = pd.DataFrame(crowding_rows)

    path_df = pd.DataFrame(path_rows)

    summary_path = os.path.join(output_dir, "sectoral_inference_summary.csv")
    crowding_path = os.path.join(output_dir, "sectoral_crowding_out_summary.csv")
    grouping_path = os.path.join(output_dir, "sectoral_grouping_sensitivity.csv")
    series_path = os.path.join(output_dir, "sectoral_appendix_series.xlsx")

    summary_df.to_csv(summary_path, index=False)
    crowding_df.to_csv(crowding_path, index=False)
    grouping_df.to_csv(grouping_path, index=False)

    with pd.ExcelWriter(series_path, engine="xlsxwriter") as writer:
        path_df.to_excel(writer, sheet_name="Series", index=False)
        summary_df.to_excel(writer, sheet_name="InferenceSummary", index=False)
        crowding_df.to_excel(writer, sheet_name="CrowdingOut", index=False)
        grouping_df.to_excel(writer, sheet_name="GroupingSensitivity", index=False)

    return {
        "summary": summary_df,
        "crowding": crowding_df,
        "grouping": grouping_df,
        "series": path_df,
    }


if __name__ == "__main__":
    outputs = run_sectoral_appendix_analysis()
    print("Sectoral inference summary")
    print(outputs["summary"].to_string(index=False))
    print("\nCrowding-out summary")
    print(outputs["crowding"].to_string(index=False))

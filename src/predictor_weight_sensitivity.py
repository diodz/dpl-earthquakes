"""Predictor-weight and cross-country predictor symmetry sensitivity checks.

Issue addressed: #39

This script provides a reproducible robustness layer for two implementation choices
raised by reviewers:
1) How predictor weights (v_m in SCM notation) are selected in practice.
2) How results change when country-specific predictor sets are harmonized.

Outputs (written to article_assets/):
  - predictor_spec_sensitivity.csv

Rows are generated for each country under:
  - baseline predictor specification (country-specific)
  - harmonized reduced specification (same broad sectors in both countries)
and under multiple optimization initializations/methods to probe v_m tuning
stability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth

import nz_util

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_ROOT, "article_assets")
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
CHILE_BASELINE_PREDICTORS = [
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


@dataclass(frozen=True)
class SCMConfig:
    label: str
    optim_method: str
    optim_initial: str


SCM_CONFIGS = [
    SCMConfig("baseline_vm", "Nelder-Mead", "equal"),
    SCMConfig("vm_alt_init", "Nelder-Mead", "ols"),
    SCMConfig("vm_alt_method", "Powell", "equal"),
]


def _rms(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(arr))))


def _add_harmonized_predictors(chile_df: pd.DataFrame, nz_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create a reduced, conceptually symmetric predictor set for both countries."""
    chile = chile_df.copy()
    nz = nz_df.copy()

    # Chile broad sectors
    chile["harm_agriculture"] = chile["agropecuario"]
    chile["harm_manufacturing"] = chile["industria_m"]
    chile["harm_construction"] = chile["construccion"]
    chile["harm_trade_hospitality"] = chile["comercio"]
    chile["harm_transport_info"] = chile["transporte"]
    chile["harm_finance_business"] = chile["servicios_financieros"] + chile["vivienda"]
    chile["harm_personal_services"] = chile["personales"]
    chile["harm_public_admin"] = chile["publica"]

    # New Zealand broad sectors
    nz["harm_agriculture"] = nz["Agriculture"]
    nz["harm_manufacturing"] = nz["Manufacturing"]
    nz["harm_construction"] = nz["Construction"]
    nz["harm_trade_hospitality"] = nz["Wholesale Trade"] + nz["Retail Trade"] + nz["Food and beverage services"]
    nz["harm_transport_info"] = (
        nz["Transport, Postal and Warehousing"]
        + nz["Information Media, Telecommunications and Other Services"]
    )
    nz["harm_finance_business"] = (
        nz["Financial and Insurance Services"]
        + nz["Professional, Scientific, and Technical Services"]
        + nz["Administrative and Support Services"]
        + nz["Rental, Hiring and Real Estate Services"]
    )
    nz["harm_personal_services"] = nz["Health Care and Social Assistance"] + nz["Education and Training"]
    nz["harm_public_admin"] = nz["Public Administration and Safety"]

    harmonized_cols = [
        "harm_agriculture",
        "harm_manufacturing",
        "harm_construction",
        "harm_trade_hospitality",
        "harm_transport_info",
        "harm_finance_business",
        "harm_personal_services",
        "harm_public_admin",
    ]
    return chile, nz, harmonized_cols


def _fit_and_summarize(
    *,
    country: str,
    spec_label: str,
    config: SCMConfig,
    df: pd.DataFrame,
    predictors: list[str],
    special_predictors: list[tuple[str, range, str]],
    treated: str,
    controls: list[str],
    unit_col: str,
    time_col: str,
    outcome_col: str,
    time_predictors_prior: range,
    time_optimize_ssr: range,
    treatment_year: int,
    analysis_end_year: int,
) -> dict:
    dataprep = Dataprep(
        foo=df,
        predictors=predictors,
        predictors_op="mean",
        time_predictors_prior=time_predictors_prior,
        special_predictors=special_predictors,
        dependent=outcome_col,
        unit_variable=unit_col,
        time_variable=time_col,
        treatment_identifier=treated,
        controls_identifier=controls,
        time_optimize_ssr=time_optimize_ssr,
    )

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method=config.optim_method, optim_initial=config.optim_initial)

    years = list(range(int(df[time_col].min()), int(df[time_col].max()) + 1))
    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    synthetic = pd.Series(np.asarray(synth._synthetic(z0)).flatten().astype(float), index=years)
    treated_series = pd.Series(np.asarray(z1).flatten().astype(float), index=years)
    gap = treated_series - synthetic

    pre_gap = gap[gap.index < treatment_year]
    post_gap = gap[(gap.index >= treatment_year) & (gap.index <= analysis_end_year)]

    pre_rmspe = _rms(pre_gap)
    post_rmspe = _rms(post_gap)
    rmspe_ratio = float(post_rmspe / pre_rmspe) if pre_rmspe else np.nan
    mean_post_gap_pct = float(((post_gap / synthetic.loc[post_gap.index]) * 100.0).mean())

    return {
        "country": country,
        "specification": spec_label,
        "vm_config": config.label,
        "optim_method": config.optim_method,
        "optim_initial": config.optim_initial,
        "n_predictors": len(predictors),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "post_pre_rmspe_ratio": rmspe_ratio,
        "mean_post_gap_pct": mean_post_gap_pct,
    }


def main() -> None:
    chile_df = pd.read_csv(os.path.join(PROJECT_ROOT, "inter", "processed_chile.csv"))
    nz_df = nz_util.clean_data_for_synthetic_control()
    chile_df, nz_df, harmonized_cols = _add_harmonized_predictors(chile_df, nz_df)

    rows: list[dict] = []

    for config in SCM_CONFIGS:
        rows.append(
            _fit_and_summarize(
                country="Chile",
                spec_label="baseline_country_specific",
                config=config,
                df=chile_df,
                predictors=CHILE_BASELINE_PREDICTORS,
                special_predictors=[("gdp_cap", range(2005, 2009), "mean"), ("ed_superior_cap", range(2007, 2009), "mean")],
                treated=CHILE_TREATED,
                controls=CHILE_CONTROLS,
                unit_col="region_name",
                time_col="year",
                outcome_col="gdp_cap",
                time_predictors_prior=range(2005, 2009),
                time_optimize_ssr=range(1990, 2009),
                treatment_year=2010,
                analysis_end_year=2019,
            )
        )
        rows.append(
            _fit_and_summarize(
                country="Chile",
                spec_label="harmonized_reduced",
                config=config,
                df=chile_df,
                predictors=harmonized_cols,
                special_predictors=[("gdp_cap", range(2005, 2009), "mean"), ("ed_superior_cap", range(2007, 2009), "mean")],
                treated=CHILE_TREATED,
                controls=CHILE_CONTROLS,
                unit_col="region_name",
                time_col="year",
                outcome_col="gdp_cap",
                time_predictors_prior=range(2005, 2009),
                time_optimize_ssr=range(1990, 2009),
                treatment_year=2010,
                analysis_end_year=2019,
            )
        )

        rows.append(
            _fit_and_summarize(
                country="New Zealand",
                spec_label="baseline_country_specific",
                config=config,
                df=nz_df,
                predictors=nz_util.SECTORIAL_GDP_VARIABLES,
                special_predictors=[("GDP per capita", range(2006, 2011), "mean"), ("Tertiary Share", range(2008, 2011), "mean")],
                treated=NZ_TREATED,
                controls=NZ_CONTROLS,
                unit_col="Region",
                time_col="Year",
                outcome_col="GDP per capita",
                time_predictors_prior=range(2006, 2011),
                time_optimize_ssr=range(2000, 2010),
                treatment_year=2011,
                analysis_end_year=2019,
            )
        )
        rows.append(
            _fit_and_summarize(
                country="New Zealand",
                spec_label="harmonized_reduced",
                config=config,
                df=nz_df,
                predictors=harmonized_cols,
                special_predictors=[("GDP per capita", range(2006, 2011), "mean"), ("Tertiary Share", range(2008, 2011), "mean")],
                treated=NZ_TREATED,
                controls=NZ_CONTROLS,
                unit_col="Region",
                time_col="Year",
                outcome_col="GDP per capita",
                time_predictors_prior=range(2006, 2011),
                time_optimize_ssr=range(2000, 2010),
                treatment_year=2011,
                analysis_end_year=2019,
            )
        )

    out_df = pd.DataFrame(rows).sort_values(["country", "specification", "vm_config"])
    out_path = os.path.join(FIGURES_DIR, "predictor_spec_sensitivity.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


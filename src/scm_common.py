"""
Shared SCM specification: constants, dataprep builders, and RMSPE.

Used by treatment_timing_sensitivity.py and rolling_in_time_placebo.py so that
predictor windows, donor pools, and optimization ranges are defined in one place.
Changes to the SCM specification only need to be made here.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from pysyncon import Dataprep

import nz_util

ANALYSIS_END_YEAR = 2019

# New Zealand
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

# Chile
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


def rmspe(x: Union[np.ndarray, pd.Series]) -> float:
    """Root mean squared prediction error. Accepts 1d array or Series."""
    if isinstance(x, pd.Series):
        vals = x.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    else:
        vals = np.asarray(x, dtype=float)
        vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(vals))))


def build_nz_dataprep(df: pd.DataFrame, treatment_year: int) -> Dataprep:
    """Build Dataprep for New Zealand SCM with treatment_year-parameterized windows."""
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


def build_chile_dataprep(df: pd.DataFrame, treatment_year: int) -> Dataprep:
    """Build Dataprep for Chile SCM with treatment_year-parameterized windows."""
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

"""
Shared mathematical utilities and constants for the SCM analysis pipeline.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared region and treatment constants
# ---------------------------------------------------------------------------

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
NZ_START_YEAR = 2000
NZ_TREATMENT_YEAR = 2011
NZ_END_YEAR = 2019

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
CHILE_TREATMENT_YEAR = 2010
CHILE_END_YEAR = 2019
# Note: CHILE_START_YEAR is defined per-module because it depends on data source:
#   - GDP analysis uses 1990 (first year of processed_chile.csv)
#   - NTL analysis uses 1992 (first year of DMSP-OLS satellite data)


def project_to_simplex(vec: np.ndarray) -> np.ndarray:
    """Project vec onto the unit simplex (non-negative, sum-to-one).

    Implements the algorithm from Duchi et al. (2008) for Euclidean
    projection onto the probability simplex.

    Parameters
    ----------
    vec : np.ndarray
        Input vector to project.

    Returns
    -------
    np.ndarray
        Projected vector with non-negative entries summing to one.
    """
    sorted_vec = np.sort(vec)[::-1]
    cumsum = np.cumsum(sorted_vec)
    rho_candidates = np.nonzero(
        sorted_vec * np.arange(1, len(vec) + 1) > (cumsum - 1)
    )[0]
    rho = int(rho_candidates[-1])
    theta = (cumsum[rho] - 1) / (rho + 1)
    return np.maximum(vec - theta, 0.0)

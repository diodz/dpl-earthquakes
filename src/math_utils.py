"""
Shared mathematical utilities for the SCM analysis pipeline.
"""

import numpy as np


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

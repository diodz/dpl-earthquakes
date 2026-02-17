import os
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, PenalizedSynth, Synth

import nz_util
from math_utils import project_to_simplex

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
NZ_START_YEAR = 2000
NZ_TREATMENT_YEAR = 2011
NZ_END_YEAR = 2019

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
CHILE_START_YEAR = 1990
CHILE_TREATMENT_YEAR = 2010
CHILE_END_YEAR = 2019


@dataclass
class GapStats:
    mean_post_gap_pct: float
    pre_rmspe_pct: float
    post_rmspe_pct: float


def _solve_simplex_with_intercept(
    design_matrix: np.ndarray,
    target_vector: np.ndarray,
    regularization: float,
    max_iter: int = 40_000,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Solve:
      min_w,a ||a + Xw - y||^2 + reg ||w||^2
      s.t.   w >= 0, sum(w)=1

    We eliminate the intercept by centering rows, then optimize w on the simplex
    with projected gradient descent.
    """
    x_centered = design_matrix - design_matrix.mean(axis=0, keepdims=True)
    y_centered = target_vector - target_vector.mean()

    n_weights = design_matrix.shape[1]
    weights = np.full(n_weights, 1.0 / n_weights, dtype=float)

    lipschitz = 2.0 * (np.linalg.norm(x_centered, ord=2) ** 2 + regularization)
    step_size = 1.0 / max(lipschitz, 1e-12)

    prev_obj = np.inf
    for iteration in range(max_iter):
        gradient = 2.0 * (
            x_centered.T @ (x_centered @ weights - y_centered)
            + regularization * weights
        )
        weights = project_to_simplex(weights - step_size * gradient)

        if iteration % 300 == 0:
            obj = float(
                np.sum((x_centered @ weights - y_centered) ** 2)
                + regularization * np.sum(weights * weights)
            )
            if abs(prev_obj - obj) < tol:
                break
            prev_obj = obj

    intercept = float(target_vector.mean() - design_matrix.mean(axis=0) @ weights)
    return weights, intercept


def _extract_weight_vector(model: object) -> np.ndarray:
    for attr_name in ("W", "w"):
        if hasattr(model, attr_name):
            arr = np.asarray(getattr(model, attr_name), dtype=float).reshape(-1)
            return arr
    weight_series = model.weights(round=12, threshold=None)
    return np.asarray(weight_series.values, dtype=float).reshape(-1)


def _compute_gap_stats(
    treated_series: np.ndarray,
    synthetic_series: np.ndarray,
    pre_indices: list[int],
    post_indices: list[int],
) -> GapStats:
    gap_pct = (treated_series - synthetic_series) / synthetic_series * 100.0
    pre_gaps = gap_pct[pre_indices]
    post_gaps = gap_pct[post_indices]
    return GapStats(
        mean_post_gap_pct=float(np.mean(post_gaps)),
        pre_rmspe_pct=float(np.sqrt(np.mean(np.square(pre_gaps)))),
        post_rmspe_pct=float(np.sqrt(np.mean(np.square(post_gaps)))),
    )


def _add_gap_rows(
    rows: list[dict],
    country: str,
    estimator: str,
    years: Iterable[int],
    gap_pct: np.ndarray,
) -> None:
    for year, gap in zip(years, gap_pct, strict=True):
        rows.append(
            {
                "Country": country,
                "Estimator": estimator,
                "Year": int(year),
                "GapPct": float(gap),
            }
        )


def _run_scm_nz(nz_df: pd.DataFrame) -> tuple[GapStats, pd.DataFrame]:
    work = nz_df.copy()
    work["Tertiary Share"] = work["Tertiary"] / work["Population"]

    dataprep = Dataprep(
        foo=work,
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
        controls_identifier=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2009),
    )

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="equal")

    years = list(range(NZ_START_YEAR, NZ_END_YEAR + 1))
    pre_years = list(range(NZ_START_YEAR, NZ_TREATMENT_YEAR))
    post_years = list(range(NZ_TREATMENT_YEAR, NZ_END_YEAR + 1))
    pre_idx = [years.index(y) for y in pre_years]
    post_idx = [years.index(y) for y in post_years]

    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    synthetic = synth._synthetic(z0).values.astype(float)
    treated = z1.values.astype(float)

    stats = _compute_gap_stats(treated, synthetic, pre_idx, post_idx)
    out = pd.DataFrame({"Year": years, "Treated": treated, "Synthetic": synthetic})
    out["GapPct"] = (out["Treated"] - out["Synthetic"]) / out["Synthetic"] * 100.0
    return stats, out


def _run_scm_chile(chile_df: pd.DataFrame) -> tuple[GapStats, pd.DataFrame]:
    dataprep = Dataprep(
        foo=chile_df,
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

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="ols")

    years = list(range(CHILE_START_YEAR, CHILE_END_YEAR + 1))
    pre_years = list(range(CHILE_START_YEAR, CHILE_TREATMENT_YEAR))
    post_years = list(range(CHILE_TREATMENT_YEAR, CHILE_END_YEAR + 1))
    pre_idx = [years.index(y) for y in pre_years]
    post_idx = [years.index(y) for y in post_years]

    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    synthetic = synth._synthetic(z0).values.astype(float)
    treated = z1.values.astype(float)

    stats = _compute_gap_stats(treated, synthetic, pre_idx, post_idx)
    out = pd.DataFrame({"Year": years, "Treated": treated, "Synthetic": synthetic})
    out["GapPct"] = (out["Treated"] - out["Synthetic"]) / out["Synthetic"] * 100.0
    return stats, out


def _prepare_pivot(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    treated: str,
    controls: list[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    subset = df[df[unit_col].isin([treated] + controls)].copy()
    subset = subset[(subset[time_col] >= start_year) & (subset[time_col] <= end_year)]
    pivot = subset.pivot(index=unit_col, columns=time_col, values=outcome_col)
    pivot = pivot.loc[[treated] + controls, years]
    keep_years = pivot.columns[pivot.notna().all(axis=0)].tolist()
    return pivot[keep_years]


def _run_sdid_from_pivot(
    pivot: pd.DataFrame,
    treated: str,
    controls: list[str],
    treatment_year: int,
    regularization_multiplier: float = 1.0,
) -> tuple[GapStats, pd.DataFrame, dict]:
    years = [int(y) for y in pivot.columns.tolist()]
    pre_years = [year for year in years if year < treatment_year]
    post_years = [year for year in years if year >= treatment_year]
    pre_idx = [years.index(year) for year in pre_years]
    post_idx = [years.index(year) for year in post_years]

    treated_series = pivot.loc[treated].to_numpy(dtype=float)
    controls_matrix = pivot.loc[controls].to_numpy(dtype=float)

    controls_pre = controls_matrix[:, pre_idx]
    controls_post = controls_matrix[:, post_idx]
    treated_pre = treated_series[pre_idx]

    noise_level = max(float(np.std(np.diff(controls_pre, axis=1))), 1e-8)
    zeta_unit = regularization_multiplier * (
        (1 * len(post_idx)) ** 0.25
    ) * noise_level
    zeta_time = regularization_multiplier * (
        (len(controls) * len(post_idx)) ** 0.25
    ) * noise_level
    unit_reg = (zeta_unit**2) * len(pre_idx)
    time_reg = (zeta_time**2) * len(controls)

    unit_weights, unit_intercept = _solve_simplex_with_intercept(
        controls_pre.T, treated_pre, regularization=unit_reg
    )
    time_weights, time_intercept = _solve_simplex_with_intercept(
        controls_pre, controls_post.mean(axis=1), regularization=time_reg
    )

    synthetic_no_intercept = unit_weights @ controls_matrix
    raw_gap = treated_series - synthetic_no_intercept
    pre_bias = float(time_weights @ raw_gap[pre_idx])
    adjusted_gap = raw_gap - pre_bias

    synthetic_with_intercept = unit_intercept + synthetic_no_intercept
    pre_fit_pct = (
        (treated_series[pre_idx] - synthetic_with_intercept[pre_idx])
        / synthetic_with_intercept[pre_idx]
        * 100.0
    )
    post_gap_pct = adjusted_gap[post_idx] / synthetic_no_intercept[post_idx] * 100.0

    stats = GapStats(
        mean_post_gap_pct=float(np.mean(post_gap_pct)),
        pre_rmspe_pct=float(np.sqrt(np.mean(np.square(pre_fit_pct)))),
        post_rmspe_pct=float(np.sqrt(np.mean(np.square(post_gap_pct)))),
    )

    out = pd.DataFrame(
        {
            "Year": years,
            "Treated": treated_series,
            "Synthetic": synthetic_no_intercept,
            "AdjustedGap": adjusted_gap,
            "GapPct": adjusted_gap / synthetic_no_intercept * 100.0,
        }
    )
    diagnostics = {
        "unit_intercept": unit_intercept,
        "time_intercept": time_intercept,
        "top_unit_weights": sorted(
            zip(controls, unit_weights, strict=True), key=lambda x: x[1], reverse=True
        )[:5],
        "top_time_weights": sorted(
            zip(pre_years, time_weights, strict=True), key=lambda x: x[1], reverse=True
        )[:5],
    }
    return stats, out, diagnostics


def _run_penalized_from_pivot(
    pivot: pd.DataFrame,
    treated: str,
    controls: list[str],
    treatment_year: int,
    lambda_grid: np.ndarray | None = None,
    tolerance: float = 0.05,
) -> tuple[GapStats, pd.DataFrame, dict, pd.DataFrame]:
    if lambda_grid is None:
        lambda_grid = np.logspace(-6, 2, 17)

    years = [int(y) for y in pivot.columns.tolist()]
    pre_years = [year for year in years if year < treatment_year]
    post_years = [year for year in years if year >= treatment_year]
    pre_idx = [years.index(year) for year in pre_years]
    post_idx = [years.index(year) for year in post_years]

    pre_controls = pivot.loc[controls, pre_years].T
    pre_treated = pivot.loc[treated, pre_years]
    all_controls = pivot.loc[controls, years].T
    all_treated = pivot.loc[treated, years].to_numpy(dtype=float)

    candidates: list[dict] = []
    for lambda_value in lambda_grid:
        estimator = PenalizedSynth()
        estimator.fit(X0=pre_controls, X1=pre_treated, lambda_=float(lambda_value))
        weights = _extract_weight_vector(estimator)

        synthetic = all_controls.to_numpy(dtype=float) @ weights
        gap_pct = (all_treated - synthetic) / synthetic * 100.0
        pre_gap = gap_pct[pre_idx]
        post_gap = gap_pct[post_idx]

        candidates.append(
            {
                "Lambda": float(lambda_value),
                "PreRMSPEPct": float(np.sqrt(np.mean(np.square(pre_gap)))),
                "MeanPostGapPct": float(np.mean(post_gap)),
                "PostRMSPEPct": float(np.sqrt(np.mean(np.square(post_gap)))),
                "Weights": weights,
                "GapPct": gap_pct,
                "Synthetic": synthetic,
            }
        )

    min_pre_rmspe = min(item["PreRMSPEPct"] for item in candidates)
    max_allowed = (1.0 + tolerance) * min_pre_rmspe
    eligible = [item for item in candidates if item["PreRMSPEPct"] <= max_allowed]
    selected = max(eligible, key=lambda item: item["Lambda"])

    stats = GapStats(
        mean_post_gap_pct=selected["MeanPostGapPct"],
        pre_rmspe_pct=selected["PreRMSPEPct"],
        post_rmspe_pct=selected["PostRMSPEPct"],
    )
    out = pd.DataFrame(
        {
            "Year": years,
            "Treated": all_treated,
            "Synthetic": selected["Synthetic"],
            "GapPct": selected["GapPct"],
        }
    )
    diagnostics = {
        "selected_lambda": selected["Lambda"],
        "min_pre_rmspe_pct": min_pre_rmspe,
        "top_weights": sorted(
            zip(controls, selected["Weights"], strict=True),
            key=lambda x: x[1],
            reverse=True,
        )[:5],
    }
    lambda_df = pd.DataFrame(
        [
            {
                "Lambda": item["Lambda"],
                "PreRMSPEPct": item["PreRMSPEPct"],
                "MeanPostGapPct": item["MeanPostGapPct"],
                "PostRMSPEPct": item["PostRMSPEPct"],
            }
            for item in candidates
        ]
    )
    return stats, out, diagnostics, lambda_df


def _plot_gap_comparison(gap_df: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    countries = [
        ("Chile", CHILE_TREATMENT_YEAR),
        ("New Zealand", NZ_TREATMENT_YEAR),
    ]
    estimator_order = ["SCM", "SDID", "Penalized SCM"]
    estimator_labels = {
        "SCM": "Baseline SCM",
        "SDID": "SDID",
        "Penalized SCM": "Penalized SCM (bias-corrected)",
    }
    colors = {
        "SCM": "#444444",
        "SDID": "#d62728",
        "Penalized SCM": "#1f77b4",
    }

    for axis, (country, treatment_year) in zip(axes, countries, strict=True):
        country_df = gap_df[gap_df["Country"] == country]
        for estimator in estimator_order:
            line_df = country_df[country_df["Estimator"] == estimator]
            axis.plot(
                line_df["Year"],
                line_df["GapPct"],
                label=estimator_labels[estimator],
                linewidth=1.9 if estimator != "SCM" else 1.5,
                color=colors[estimator],
            )
        axis.axvline(treatment_year, color="black", linestyle="--", linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_title(country)
        axis.set_xlabel("Year")
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Gap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_sdid_bias_corrected_analysis(output_dir: str = FIGURES_DIR) -> pd.DataFrame:
    nz_df = pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "nz.csv"))
    chile_df = pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv"))

    scm_nz_stats, scm_nz_path = _run_scm_nz(nz_df)
    scm_chile_stats, scm_chile_path = _run_scm_chile(chile_df)

    nz_pivot = _prepare_pivot(
        nz_df,
        unit_col="Region",
        time_col="Year",
        outcome_col="GDP per capita",
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        start_year=NZ_START_YEAR,
        end_year=NZ_END_YEAR,
    )
    chile_pivot = _prepare_pivot(
        chile_df,
        unit_col="region_name",
        time_col="year",
        outcome_col="gdp_cap",
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        start_year=CHILE_START_YEAR,
        end_year=CHILE_END_YEAR,
    )

    sdid_nz_stats, sdid_nz_path, sdid_nz_diag = _run_sdid_from_pivot(
        nz_pivot,
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        treatment_year=NZ_TREATMENT_YEAR,
        regularization_multiplier=1.0,
    )
    sdid_chile_stats, sdid_chile_path, sdid_chile_diag = _run_sdid_from_pivot(
        chile_pivot,
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        treatment_year=CHILE_TREATMENT_YEAR,
        regularization_multiplier=1.0,
    )

    pen_nz_stats, pen_nz_path, pen_nz_diag, pen_nz_grid = _run_penalized_from_pivot(
        nz_pivot,
        treated=NZ_TREATED,
        controls=NZ_CONTROLS,
        treatment_year=NZ_TREATMENT_YEAR,
    )
    pen_chile_stats, pen_chile_path, pen_chile_diag, pen_chile_grid = _run_penalized_from_pivot(
        chile_pivot,
        treated=CHILE_TREATED,
        controls=CHILE_CONTROLS,
        treatment_year=CHILE_TREATMENT_YEAR,
    )

    summary_rows = [
        {
            "Country": "Chile",
            "Estimator": "SCM",
            "MeanPostGapPct": scm_chile_stats.mean_post_gap_pct,
            "PreRMSPEPct": scm_chile_stats.pre_rmspe_pct,
            "PostRMSPEPct": scm_chile_stats.post_rmspe_pct,
            "SelectedLambda": np.nan,
        },
        {
            "Country": "Chile",
            "Estimator": "SDID",
            "MeanPostGapPct": sdid_chile_stats.mean_post_gap_pct,
            "PreRMSPEPct": sdid_chile_stats.pre_rmspe_pct,
            "PostRMSPEPct": sdid_chile_stats.post_rmspe_pct,
            "SelectedLambda": np.nan,
        },
        {
            "Country": "Chile",
            "Estimator": "Penalized SCM",
            "MeanPostGapPct": pen_chile_stats.mean_post_gap_pct,
            "PreRMSPEPct": pen_chile_stats.pre_rmspe_pct,
            "PostRMSPEPct": pen_chile_stats.post_rmspe_pct,
            "SelectedLambda": pen_chile_diag["selected_lambda"],
        },
        {
            "Country": "New Zealand",
            "Estimator": "SCM",
            "MeanPostGapPct": scm_nz_stats.mean_post_gap_pct,
            "PreRMSPEPct": scm_nz_stats.pre_rmspe_pct,
            "PostRMSPEPct": scm_nz_stats.post_rmspe_pct,
            "SelectedLambda": np.nan,
        },
        {
            "Country": "New Zealand",
            "Estimator": "SDID",
            "MeanPostGapPct": sdid_nz_stats.mean_post_gap_pct,
            "PreRMSPEPct": sdid_nz_stats.pre_rmspe_pct,
            "PostRMSPEPct": sdid_nz_stats.post_rmspe_pct,
            "SelectedLambda": np.nan,
        },
        {
            "Country": "New Zealand",
            "Estimator": "Penalized SCM",
            "MeanPostGapPct": pen_nz_stats.mean_post_gap_pct,
            "PreRMSPEPct": pen_nz_stats.pre_rmspe_pct,
            "PostRMSPEPct": pen_nz_stats.post_rmspe_pct,
            "SelectedLambda": pen_nz_diag["selected_lambda"],
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    gap_rows: list[dict] = []
    _add_gap_rows(gap_rows, "Chile", "SCM", scm_chile_path["Year"], scm_chile_path["GapPct"].values)
    _add_gap_rows(
        gap_rows, "Chile", "SDID", sdid_chile_path["Year"], sdid_chile_path["GapPct"].values
    )
    _add_gap_rows(
        gap_rows, "Chile", "Penalized SCM", pen_chile_path["Year"], pen_chile_path["GapPct"].values
    )
    _add_gap_rows(gap_rows, "New Zealand", "SCM", scm_nz_path["Year"], scm_nz_path["GapPct"].values)
    _add_gap_rows(
        gap_rows, "New Zealand", "SDID", sdid_nz_path["Year"], sdid_nz_path["GapPct"].values
    )
    _add_gap_rows(
        gap_rows, "New Zealand", "Penalized SCM", pen_nz_path["Year"], pen_nz_path["GapPct"].values
    )
    gap_df = pd.DataFrame(gap_rows)

    summary_path = os.path.join(output_dir, "sdid_bias_corrected_summary.csv")
    gaps_path = os.path.join(output_dir, "sdid_bias_corrected_gaps.csv")
    fig_path = os.path.join(output_dir, "sdid_bias_corrected_gaps.png")
    lambda_path = os.path.join(output_dir, "sdid_penalized_lambda_grid.csv")
    diag_path = os.path.join(output_dir, "sdid_diagnostics.txt")

    summary_df.to_csv(summary_path, index=False)
    gap_df.to_csv(gaps_path, index=False)
    lambda_df = pd.concat(
        [
            pen_chile_grid.assign(Country="Chile"),
            pen_nz_grid.assign(Country="New Zealand"),
        ],
        ignore_index=True,
    )
    lambda_df.to_csv(lambda_path, index=False)
    _plot_gap_comparison(gap_df, fig_path)

    with open(diag_path, "w", encoding="utf-8") as handle:
        handle.write("SDID diagnostics (top weights and intercepts)\n")
        handle.write("=========================================\n\n")
        handle.write("Chile SDID\n")
        handle.write(f"Unit intercept: {sdid_chile_diag['unit_intercept']:.6f}\n")
        handle.write(f"Time intercept: {sdid_chile_diag['time_intercept']:.6f}\n")
        handle.write("Top unit weights:\n")
        for unit, weight in sdid_chile_diag["top_unit_weights"]:
            handle.write(f"  - {unit}: {weight:.6f}\n")
        handle.write("Top time weights:\n")
        for year, weight in sdid_chile_diag["top_time_weights"]:
            handle.write(f"  - {year}: {weight:.6f}\n")
        handle.write("\nNew Zealand SDID\n")
        handle.write(f"Unit intercept: {sdid_nz_diag['unit_intercept']:.6f}\n")
        handle.write(f"Time intercept: {sdid_nz_diag['time_intercept']:.6f}\n")
        handle.write("Top unit weights:\n")
        for unit, weight in sdid_nz_diag["top_unit_weights"]:
            handle.write(f"  - {unit}: {weight:.6f}\n")
        handle.write("Top time weights:\n")
        for year, weight in sdid_nz_diag["top_time_weights"]:
            handle.write(f"  - {year}: {weight:.6f}\n")

    return summary_df


if __name__ == "__main__":
    result = run_sdid_bias_corrected_analysis()
    print(result.to_string(index=False))

"""Spillover / SUTVA diagnostics for the SCM earthquake analysis.

Addresses Reviewer #1 Comment 3.4 and Reviewer #2 Major Comment 1.
Implements:
  1. Explicit spillover proxy construction (population flows, labour-sector
     shifts, inter-regional economic linkage indices).
  2. Geographically structured (ring / adjacency) progressive donor
     exclusions for both Chile (Maule) and New Zealand (Canterbury).
  3. Re-estimation of SCM under each progressively restricted donor pool.
  4. Side-by-side comparison of baseline vs spillover-robust estimates.
  5. Summary tables, CSV exports, and publication-quality figures.
"""

import os
from dataclasses import dataclass, field
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth

import nz_util

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

ANALYSIS_END_YEAR = 2019

# ── Chile configuration ─────────────────────────────────────────────────

CHILE_TREATED = "VII Del Maule"
CHILE_TREATMENT_YEAR = 2010
CHILE_FIT_START = 1990

CHILE_PREDICTORS = [
    "agropecuario", "pesca", "mineria", "industria_m", "electricidad",
    "construccion", "comercio", "transporte", "servicios_financieros",
    "vivienda", "personales", "publica",
]

CHILE_ALL_CONTROLS = [
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

# Geographic adjacency rings around Maule (VII).
# Ring 0 = baseline (VIII Biobío already excluded from analysis).
# Ring 1 = exclude O'Higgins (VI) — immediate northern neighbour.
# Ring 2 = also exclude Araucanía (IX) — immediate southern neighbour
#          and RM Santiago — northern second-ring economic hub.
# Ring 3 = also exclude Valparaíso (V) and Los Lagos (X) — most restrictive.
CHILE_EXCLUSION_RINGS = {
    "Baseline (excl. Biobío only)": [],
    "Ring 1: excl. O'Higgins": [
        "VI Del Libertador General Bernardo OHiggins",
    ],
    "Ring 2: excl. O'Higgins + Araucanía + RM": [
        "VI Del Libertador General Bernardo OHiggins",
        "IX De La Araucanía",
        "RMS Región Metropolitana de Santiago",
    ],
    "Ring 3: excl. O'Higgins + Araucanía + RM + Valparaíso + Los Lagos": [
        "VI Del Libertador General Bernardo OHiggins",
        "IX De La Araucanía",
        "RMS Región Metropolitana de Santiago",
        "V De Valparaíso",
        "X De Los Lagos",
    ],
}

# ── New Zealand configuration ────────────────────────────────────────────

NZ_TREATED = "Canterbury"
NZ_TREATMENT_YEAR = 2011
NZ_FIT_START = 2000

NZ_ALL_CONTROLS = [
    "Auckland", "Bay of Plenty", "Gisborne", "Hawke's Bay",
    "Manawatu-Whanganui", "Marlborough", "Northland", "Otago",
    "Southland", "Taranaki", "Tasman/Nelson", "Waikato",
    "Wellington", "West Coast",
]

# Geographic adjacency rings around Canterbury.
# Ring 0 = baseline — all donor regions.
# Ring 1 = exclude West Coast (shares Canterbury's alpine border).
# Ring 2 = also exclude Otago + Marlborough (adjacent South Island).
# Ring 3 = also exclude Wellington (gov-sector spillover) + Southland.
# Ring 4 = also exclude Auckland (largest baseline donor weight; migration hub).
NZ_EXCLUSION_RINGS = {
    "Baseline (all donors)": [],
    "Ring 1: excl. West Coast": [
        "West Coast",
    ],
    "Ring 2: excl. West Coast + Otago + Marlborough": [
        "West Coast", "Otago", "Marlborough",
    ],
    "Ring 3: excl. W.Coast + Otago + Marl. + Wellington + Southland": [
        "West Coast", "Otago", "Marlborough", "Wellington", "Southland",
    ],
    "Ring 4: Ring 3 + Auckland (most restrictive)": [
        "West Coast", "Otago", "Marlborough", "Wellington", "Southland",
        "Auckland",
    ],
}


# ── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class ExclusionResult:
    country: str
    ring_label: str
    excluded_regions: list[str]
    remaining_donors: list[str]
    n_donors: int
    mean_post_gap_pct: float
    pre_rmspe_pct: float
    post_rmspe_pct: float
    rmspe_ratio: float
    gap_series_pct: pd.Series = field(repr=False)
    treated_series: pd.Series = field(repr=False)
    synthetic_series: pd.Series = field(repr=False)
    weights: dict[str, float] = field(default_factory=dict, repr=False)


# ── SCM helpers ──────────────────────────────────────────────────────────

def _rmspe(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(valid))))


def _build_chile_dataprep(
    df: pd.DataFrame,
    controls: list[str],
    treatment_year: int = CHILE_TREATMENT_YEAR,
) -> Dataprep:
    # Use hardcoded baseline windows matching the paper specification.
    # Predictors: range(2005, 2009) (years 2005-2008)
    # Education: range(2008, 2009) (year 2008)
    # Optimization: range(1990, 2009) (years 1990-2008)
    # These windows exclude 2009+ to avoid contamination from the Feb 2010 earthquake.
    pred_window = range(2005, 2009)
    edu_window = range(2008, 2009)
    return Dataprep(
        foo=df,
        predictors=CHILE_PREDICTORS,
        predictors_op="mean",
        time_predictors_prior=pred_window,
        special_predictors=[
            ("gdp_cap", pred_window, "mean"),
            ("ed_superior_cap", edu_window, "mean"),
        ],
        dependent="gdp_cap",
        unit_variable="region_name",
        time_variable="year",
        treatment_identifier=CHILE_TREATED,
        controls_identifier=controls,
        time_optimize_ssr=range(1990, 2009),
    )


def _build_nz_dataprep(
    df: pd.DataFrame,
    controls: list[str],
    treatment_year: int = NZ_TREATMENT_YEAR,
) -> Dataprep:
    # Use hardcoded baseline windows matching the paper specification.
    # Predictors: range(2005, 2009) (years 2005-2008)
    # Tertiary: range(2008, 2009) (year 2008)
    # Optimization: range(2000, 2009) (years 2000-2008)
    # These windows exclude 2009+ to avoid contamination from earthquake effects
    # (the Sep 2010 Darfield earthquake partially affected 2010 data).
    pred_window = range(2005, 2009)
    tert_window = range(2008, 2009)
    return Dataprep(
        foo=df,
        predictors=nz_util.SECTORIAL_GDP_VARIABLES,
        predictors_op="mean",
        time_predictors_prior=pred_window,
        special_predictors=[
            ("GDP per capita", pred_window, "mean"),
            ("Tertiary Share", tert_window, "mean"),
        ],
        dependent="GDP per capita",
        unit_variable="Region",
        time_variable="Year",
        treatment_identifier=NZ_TREATED,
        controls_identifier=controls,
        time_optimize_ssr=range(2000, 2009),
    )


def _run_scm_exclusion(
    df: pd.DataFrame,
    country: str,
    ring_label: str,
    excluded_regions: list[str],
    all_controls: list[str],
    treated: str,
    treatment_year: int,
    fit_start: int,
    dataprep_builder,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    optim_method: str,
    optim_initial: str,
) -> ExclusionResult:
    """Fit SCM with a restricted donor pool and return diagnostics."""
    controls = [r for r in all_controls if r not in excluded_regions]
    if len(controls) < 2:
        raise ValueError(
            f"Ring '{ring_label}' leaves only {len(controls)} donors; need at least 2."
        )

    dataprep = dataprep_builder(df, controls, treatment_year)
    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method=optim_method, optim_initial=optim_initial)

    all_years = list(range(fit_start, ANALYSIS_END_YEAR + 1))
    z0, z1 = synth.dataprep.make_outcome_mats(time_period=all_years)
    synthetic = pd.Series(
        np.asarray(synth._synthetic(z0)).flatten().astype(float), index=all_years
    )
    treated_s = pd.Series(np.asarray(z1).flatten().astype(float), index=all_years)
    gap_pct = (treated_s - synthetic) / synthetic * 100.0

    year_arr = np.array(all_years)
    pre_mask = year_arr < treatment_year
    post_mask = (year_arr >= treatment_year) & (year_arr <= ANALYSIS_END_YEAR)

    pre_rmspe = _rmspe(gap_pct.values[pre_mask])
    post_rmspe = _rmspe(gap_pct.values[post_mask])
    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe and not np.isclose(pre_rmspe, 0.0) else float("nan")
    mean_post = float(np.nanmean(gap_pct.values[post_mask]))

    try:
        w_df = synth.weights(round=6)
        weight_dict = dict(zip(w_df.index, w_df.values.flatten()))
    except Exception:
        weight_dict = {}

    return ExclusionResult(
        country=country,
        ring_label=ring_label,
        excluded_regions=excluded_regions,
        remaining_donors=controls,
        n_donors=len(controls),
        mean_post_gap_pct=mean_post,
        pre_rmspe_pct=pre_rmspe,
        post_rmspe_pct=post_rmspe,
        rmspe_ratio=rmspe_ratio,
        gap_series_pct=gap_pct,
        treated_series=treated_s,
        synthetic_series=synthetic,
        weights=weight_dict,
    )


# ── Spillover proxy diagnostics ─────────────────────────────────────────

def _compute_population_flow_diagnostics(
    df: pd.DataFrame,
    treated: str,
    treatment_year: int,
    unit_col: str,
    time_col: str,
    pop_col: str,
    adjacent_regions: list[str],
    non_adjacent_regions: list[str],
    pre_window: int = 5,
    post_window: int = 5,
) -> pd.DataFrame:
    """Compute pre/post population growth rates for treated, adjacent, and
    non-adjacent region groups as a proxy for migration flows."""
    pre_years = list(range(treatment_year - pre_window, treatment_year))
    post_years = list(range(treatment_year, treatment_year + post_window))

    def _avg_growth(region_list, year_range):
        sub = df[df[unit_col].isin(region_list) & df[time_col].isin(year_range)]
        if sub.empty or pop_col not in sub.columns:
            return float("nan")
        pivot = sub.pivot(index=time_col, columns=unit_col, values=pop_col)
        growth_rates = pivot.pct_change(fill_method=None).dropna()
        if growth_rates.empty:
            return float("nan")
        return float(growth_rates.mean().mean() * 100.0)

    rows = []
    for group_label, group_regions in [
        ("Treated", [treated]),
        ("Adjacent", adjacent_regions),
        ("Non-adjacent", non_adjacent_regions),
    ]:
        pre_growth = _avg_growth(group_regions, pre_years)
        post_growth = _avg_growth(group_regions, post_years)
        rows.append({
            "Group": group_label,
            "Regions": ", ".join(group_regions) if len(group_regions) <= 4 else f"{len(group_regions)} regions",
            "Pre-treatment avg pop growth (%/yr)": pre_growth,
            "Post-treatment avg pop growth (%/yr)": post_growth,
            "Change (p.p.)": post_growth - pre_growth if np.isfinite(pre_growth) and np.isfinite(post_growth) else float("nan"),
        })
    return pd.DataFrame(rows)


def _compute_sector_spillover_diagnostics(
    df: pd.DataFrame,
    treated: str,
    treatment_year: int,
    unit_col: str,
    time_col: str,
    construction_col: str,
    adjacent_regions: list[str],
    non_adjacent_regions: list[str],
    pre_window: int = 5,
    post_window: int = 5,
) -> pd.DataFrame:
    """Compute pre/post construction-share changes for adjacent vs non-adjacent
    regions as a proxy for labour reallocation spillovers."""
    pre_years = list(range(treatment_year - pre_window, treatment_year))
    post_years = list(range(treatment_year, treatment_year + post_window))

    def _avg_constr(region_list, year_range):
        sub = df[df[unit_col].isin(region_list) & df[time_col].isin(year_range)]
        if sub.empty or construction_col not in sub.columns:
            return float("nan")
        return float(sub[construction_col].astype(float).mean() * 100.0)

    rows = []
    for group_label, group_regions in [
        ("Treated", [treated]),
        ("Adjacent", adjacent_regions),
        ("Non-adjacent", non_adjacent_regions),
    ]:
        pre_avg = _avg_constr(group_regions, pre_years)
        post_avg = _avg_constr(group_regions, post_years)
        rows.append({
            "Group": group_label,
            "Regions": ", ".join(group_regions) if len(group_regions) <= 4 else f"{len(group_regions)} regions",
            "Pre-treatment avg construction share (%)": pre_avg,
            "Post-treatment avg construction share (%)": post_avg,
            "Change (p.p.)": post_avg - pre_avg if np.isfinite(pre_avg) and np.isfinite(post_avg) else float("nan"),
        })
    return pd.DataFrame(rows)


def _compute_economic_linkage_index(
    df: pd.DataFrame,
    treated: str,
    treatment_year: int,
    unit_col: str,
    time_col: str,
    sector_cols: list[str],
    all_other_regions: list[str],
) -> pd.DataFrame:
    """Compute pairwise sector-structure correlation between the treated region
    and all other regions as a proxy for inter-regional economic linkages.
    Higher correlation implies greater potential for economic spillovers."""
    pre_df = df[df[time_col] < treatment_year]

    treated_profile = (
        pre_df[pre_df[unit_col] == treated][sector_cols]
        .astype(float).mean()
    )

    rows = []
    for region in all_other_regions:
        region_profile = (
            pre_df[pre_df[unit_col] == region][sector_cols]
            .astype(float).mean()
        )
        if treated_profile.notna().sum() >= 3 and region_profile.notna().sum() >= 3:
            corr = float(treated_profile.corr(region_profile))
        else:
            corr = float("nan")
        rows.append({"Region": region, "Sector correlation with treated": corr})

    return pd.DataFrame(rows).sort_values(
        "Sector correlation with treated", ascending=False
    )


# ── Progressive exclusion runner ─────────────────────────────────────────

def _run_progressive_exclusions(
    df: pd.DataFrame,
    country: str,
    treated: str,
    treatment_year: int,
    fit_start: int,
    all_controls: list[str],
    exclusion_rings: dict[str, list[str]],
    dataprep_builder,
    unit_col: str,
    time_col: str,
    outcome_col: str,
    optim_method: str,
    optim_initial: str,
) -> list[ExclusionResult]:
    results = []
    for ring_label, excluded in exclusion_rings.items():
        try:
            result = _run_scm_exclusion(
                df=df,
                country=country,
                ring_label=ring_label,
                excluded_regions=excluded,
                all_controls=all_controls,
                treated=treated,
                treatment_year=treatment_year,
                fit_start=fit_start,
                dataprep_builder=dataprep_builder,
                unit_col=unit_col,
                time_col=time_col,
                outcome_col=outcome_col,
                optim_method=optim_method,
                optim_initial=optim_initial,
            )
            results.append(result)
        except Exception as exc:
            print(f"Warning: {country} ring '{ring_label}' failed: {exc}")
    return results


# ── Plotting ─────────────────────────────────────────────────────────────

def _plot_side_by_side_gaps(
    chile_results: list[ExclusionResult],
    nz_results: list[ExclusionResult],
    output_path: str,
) -> None:
    """Two-panel figure comparing gap paths across exclusion rings."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(chile_results), len(nz_results))))

    for ax, results, title, treatment_year in [
        (axes[0], chile_results, "Chile (Maule)", CHILE_TREATMENT_YEAR),
        (axes[1], nz_results, "New Zealand (Canterbury)", NZ_TREATMENT_YEAR),
    ]:
        for i, res in enumerate(results):
            lw = 2.5 if i == 0 else 1.5
            ls = "-" if i == 0 else "--"
            label_short = res.ring_label.split(":")[0] if ":" in res.ring_label else res.ring_label
            gap_trimmed = res.gap_series_pct[
                (res.gap_series_pct.index >= treatment_year - 5)
                & (res.gap_series_pct.index <= ANALYSIS_END_YEAR)
            ]
            ax.plot(
                gap_trimmed.index, gap_trimmed.values,
                color=colors[i], linewidth=lw, linestyle=ls,
                label=f"{label_short} (n={res.n_donors})",
            )
        ax.axvline(treatment_year, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7.5, loc="best")

    axes[0].set_ylabel("Gap (treated − synthetic) in %")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_treated_vs_synthetic_rings(
    results: list[ExclusionResult],
    country: str,
    treatment_year: int,
    output_path: str,
) -> None:
    """Multi-line figure showing treated path and synthetic under each ring."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(results)))

    plot_start = treatment_year - 8
    treated_trimmed = results[0].treated_series[
        (results[0].treated_series.index >= plot_start)
        & (results[0].treated_series.index <= ANALYSIS_END_YEAR)
    ]
    ax.plot(
        treated_trimmed.index, treated_trimmed.values,
        color="red", linewidth=2.2, label=f"Actual {country.split('(')[-1].rstrip(')')}"
    )

    for i, res in enumerate(results):
        synth_trimmed = res.synthetic_series[
            (res.synthetic_series.index >= plot_start)
            & (res.synthetic_series.index <= ANALYSIS_END_YEAR)
        ]
        label_short = res.ring_label.split(":")[0] if ":" in res.ring_label else res.ring_label
        ax.plot(
            synth_trimmed.index, synth_trimmed.values,
            color=colors[i], linewidth=1.4, linestyle="--",
            label=f"Synth. — {label_short} (n={res.n_donors})",
        )

    ax.axvline(treatment_year, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
    ax.set_title(f"{country}: treated vs synthetic under progressive donor exclusions", fontsize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("GDP per capita")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_sensitivity_bar(
    chile_results: list[ExclusionResult],
    nz_results: list[ExclusionResult],
    output_path: str,
) -> None:
    """Bar chart showing mean post-gap under each exclusion ring."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    for ax, results, title in [
        (axes[0], chile_results, "Chile (Maule)"),
        (axes[1], nz_results, "New Zealand (Canterbury)"),
    ]:
        labels = []
        vals = []
        for res in results:
            short = res.ring_label.split(":")[0] if ":" in res.ring_label else res.ring_label
            labels.append(f"{short}\n(n={res.n_donors})")
            vals.append(res.mean_post_gap_pct)
        x = np.arange(len(labels))
        bar_colors = ["#4c78a8" if i > 0 else "#d62728" for i in range(len(labels))]
        bars = ax.bar(x, vals, color=bar_colors, alpha=0.85, edgecolor="white")
        for xi, vi in zip(x, vals):
            ax.text(xi, vi + (0.3 if vi >= 0 else -0.8), f"{vi:+.1f}%", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Mean post-treatment gap (%)")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ── Main runner ──────────────────────────────────────────────────────────

def run_spillover_diagnostics(output_dir: str = FIGURES_DIR) -> dict[str, pd.DataFrame]:
    """Execute the full spillover / SUTVA diagnostic battery."""
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    chile_df = pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv"))
    nz_df = nz_util.clean_data_for_synthetic_control().copy()
    nz_df["Tertiary Share"] = nz_df["Tertiary"] / nz_df["Population"]

    # ── 1. Spillover proxy diagnostics ───────────────────────────────────

    # Chile: population flow diagnostics
    chile_adjacent = [
        "VI Del Libertador General Bernardo OHiggins",
        "IX De La Araucanía",
        "VIII Del Biobío",
    ]
    chile_non_adjacent = [
        r for r in CHILE_ALL_CONTROLS + ["VIII Del Biobío"]
        if r not in chile_adjacent and r != CHILE_TREATED
    ]
    chile_pop_diag = _compute_population_flow_diagnostics(
        df=chile_df, treated=CHILE_TREATED,
        treatment_year=CHILE_TREATMENT_YEAR,
        unit_col="region_name", time_col="year", pop_col="population",
        adjacent_regions=chile_adjacent,
        non_adjacent_regions=chile_non_adjacent,
    )

    chile_sector_diag = _compute_sector_spillover_diagnostics(
        df=chile_df, treated=CHILE_TREATED,
        treatment_year=CHILE_TREATMENT_YEAR,
        unit_col="region_name", time_col="year",
        construction_col="construccion",
        adjacent_regions=chile_adjacent,
        non_adjacent_regions=chile_non_adjacent,
    )

    chile_linkage = _compute_economic_linkage_index(
        df=chile_df, treated=CHILE_TREATED,
        treatment_year=CHILE_TREATMENT_YEAR,
        unit_col="region_name", time_col="year",
        sector_cols=CHILE_PREDICTORS,
        all_other_regions=CHILE_ALL_CONTROLS + ["VIII Del Biobío"],
    )

    # NZ: population and sector diagnostics
    nz_adjacent = ["West Coast", "Otago", "Marlborough"]
    nz_non_adjacent = [r for r in NZ_ALL_CONTROLS if r not in nz_adjacent]
    nz_pop_diag = _compute_population_flow_diagnostics(
        df=nz_df, treated=NZ_TREATED,
        treatment_year=NZ_TREATMENT_YEAR,
        unit_col="Region", time_col="Year", pop_col="Population",
        adjacent_regions=nz_adjacent,
        non_adjacent_regions=nz_non_adjacent,
    )

    nz_sector_diag = _compute_sector_spillover_diagnostics(
        df=nz_df, treated=NZ_TREATED,
        treatment_year=NZ_TREATMENT_YEAR,
        unit_col="Region", time_col="Year",
        construction_col="Construction",
        adjacent_regions=nz_adjacent,
        non_adjacent_regions=nz_non_adjacent,
    )

    nz_linkage = _compute_economic_linkage_index(
        df=nz_df, treated=NZ_TREATED,
        treatment_year=NZ_TREATMENT_YEAR,
        unit_col="Region", time_col="Year",
        sector_cols=nz_util.SECTORIAL_GDP_VARIABLES,
        all_other_regions=NZ_ALL_CONTROLS,
    )

    # ── 2. Progressive geographic donor exclusions ───────────────────────

    chile_results = _run_progressive_exclusions(
        df=chile_df, country="Chile", treated=CHILE_TREATED,
        treatment_year=CHILE_TREATMENT_YEAR, fit_start=CHILE_FIT_START,
        all_controls=CHILE_ALL_CONTROLS,
        exclusion_rings=CHILE_EXCLUSION_RINGS,
        dataprep_builder=_build_chile_dataprep,
        unit_col="region_name", time_col="year", outcome_col="gdp_cap",
        optim_method="Nelder-Mead", optim_initial="ols",
    )

    nz_results = _run_progressive_exclusions(
        df=nz_df, country="New Zealand", treated=NZ_TREATED,
        treatment_year=NZ_TREATMENT_YEAR, fit_start=NZ_FIT_START,
        all_controls=NZ_ALL_CONTROLS,
        exclusion_rings=NZ_EXCLUSION_RINGS,
        dataprep_builder=_build_nz_dataprep,
        unit_col="Region", time_col="Year", outcome_col="GDP per capita",
        optim_method="Nelder-Mead", optim_initial="equal",
    )

    # ── 3. Build summary tables ──────────────────────────────────────────

    exclusion_summary_rows = []
    for res in chile_results + nz_results:
        exclusion_summary_rows.append({
            "Country": res.country,
            "Ring": res.ring_label,
            "Excluded regions": "; ".join(res.excluded_regions) if res.excluded_regions else "None",
            "N donors": res.n_donors,
            "Mean post gap (%)": res.mean_post_gap_pct,
            "Pre RMSPE (%)": res.pre_rmspe_pct,
            "Post RMSPE (%)": res.post_rmspe_pct,
            "Post/Pre RMSPE": res.rmspe_ratio,
        })
    exclusion_df = pd.DataFrame(exclusion_summary_rows)

    # Compute sensitivity range: max absolute change from baseline
    sensitivity_rows = []
    for country, results in [("Chile", chile_results), ("New Zealand", nz_results)]:
        if not results:
            continue
        baseline_gap = results[0].mean_post_gap_pct
        max_shift = 0.0
        all_gaps = [res.mean_post_gap_pct for res in results]
        sign_consistent = all(g > 0 for g in all_gaps) or all(g <= 0 for g in all_gaps)
        for res in results[1:]:
            shift = abs(res.mean_post_gap_pct - baseline_gap)
            if shift > max_shift:
                max_shift = shift
        min_gap = min(all_gaps)
        max_gap = max(all_gaps)
        if sign_consistent and max_shift < 5.0:
            conclusion = "Estimates qualitatively stable; sign unchanged across all exclusion rings"
        elif sign_consistent:
            conclusion = (
                "Sign of effect robust across exclusions; magnitude varies moderately"
            )
        else:
            conclusion = "Sign reversal under strictest exclusion; interpret with caution"
        sensitivity_rows.append({
            "Country": country,
            "Baseline mean gap (%)": baseline_gap,
            "Min gap across rings (%)": min_gap,
            "Max gap across rings (%)": max_gap,
            "Max absolute shift (p.p.)": max_shift,
            "Sign consistent": sign_consistent,
            "Conclusion": conclusion,
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows)

    # ── 4. Export CSVs ───────────────────────────────────────────────────

    exclusion_df.to_csv(os.path.join(output_dir, "spillover_exclusion_summary.csv"), index=False)
    sensitivity_df.to_csv(os.path.join(output_dir, "spillover_sensitivity_statement.csv"), index=False)

    chile_pop_diag.to_csv(os.path.join(output_dir, "chile_population_flow_diagnostics.csv"), index=False)
    chile_sector_diag.to_csv(os.path.join(output_dir, "chile_sector_spillover_diagnostics.csv"), index=False)
    chile_linkage.to_csv(os.path.join(output_dir, "chile_economic_linkage_index.csv"), index=False)

    nz_pop_diag.to_csv(os.path.join(output_dir, "nz_population_flow_diagnostics.csv"), index=False)
    nz_sector_diag.to_csv(os.path.join(output_dir, "nz_sector_spillover_diagnostics.csv"), index=False)
    nz_linkage.to_csv(os.path.join(output_dir, "nz_economic_linkage_index.csv"), index=False)

    # Gap series for each ring
    gap_records = []
    for res in chile_results + nz_results:
        for year, val in res.gap_series_pct.items():
            gap_records.append({
                "Country": res.country,
                "Ring": res.ring_label,
                "Year": int(year),
                "Gap_pct": float(val),
            })
    gap_series_df = pd.DataFrame(gap_records)
    gap_series_df.to_csv(os.path.join(output_dir, "spillover_exclusion_gap_series.csv"), index=False)

    # ── 5. Generate figures ──────────────────────────────────────────────

    _plot_side_by_side_gaps(
        chile_results, nz_results,
        os.path.join(output_dir, "spillover_geographic_exclusion_gaps.png"),
    )

    _plot_treated_vs_synthetic_rings(
        chile_results, "Chile (Maule)", CHILE_TREATMENT_YEAR,
        os.path.join(output_dir, "chile_spillover_paths.png"),
    )

    _plot_treated_vs_synthetic_rings(
        nz_results, "New Zealand (Canterbury)", NZ_TREATMENT_YEAR,
        os.path.join(output_dir, "nz_spillover_paths.png"),
    )

    _plot_sensitivity_bar(
        chile_results, nz_results,
        os.path.join(output_dir, "spillover_sensitivity_bar.png"),
    )

    # Consolidated Excel workbook
    with pd.ExcelWriter(
        os.path.join(output_dir, "spillover_diagnostics_workbook.xlsx"),
        engine="xlsxwriter",
    ) as writer:
        exclusion_df.to_excel(writer, sheet_name="Exclusion Summary", index=False)
        sensitivity_df.to_excel(writer, sheet_name="Sensitivity Statement", index=False)
        chile_pop_diag.to_excel(writer, sheet_name="Chile Pop Flows", index=False)
        chile_sector_diag.to_excel(writer, sheet_name="Chile Sector Spillover", index=False)
        chile_linkage.to_excel(writer, sheet_name="Chile Linkage Index", index=False)
        nz_pop_diag.to_excel(writer, sheet_name="NZ Pop Flows", index=False)
        nz_sector_diag.to_excel(writer, sheet_name="NZ Sector Spillover", index=False)
        nz_linkage.to_excel(writer, sheet_name="NZ Linkage Index", index=False)
        gap_series_df.to_excel(writer, sheet_name="Gap Series", index=False)

    return {
        "exclusion_summary": exclusion_df,
        "sensitivity": sensitivity_df,
        "chile_pop_flows": chile_pop_diag,
        "chile_sector_spillover": chile_sector_diag,
        "chile_linkage": chile_linkage,
        "nz_pop_flows": nz_pop_diag,
        "nz_sector_spillover": nz_sector_diag,
        "nz_linkage": nz_linkage,
        "gap_series": gap_series_df,
    }


if __name__ == "__main__":
    outputs = run_spillover_diagnostics()
    print("\n=== Exclusion summary ===")
    print(outputs["exclusion_summary"].to_string(index=False))
    print("\n=== Sensitivity statement ===")
    print(outputs["sensitivity"].to_string(index=False))
    print("\n=== Chile population flow diagnostics ===")
    print(outputs["chile_pop_flows"].to_string(index=False))
    print("\n=== NZ population flow diagnostics ===")
    print(outputs["nz_pop_flows"].to_string(index=False))
    print("\n=== Chile economic linkage index ===")
    print(outputs["chile_linkage"].to_string(index=False))
    print("\n=== NZ economic linkage index ===")
    print(outputs["nz_linkage"].to_string(index=False))

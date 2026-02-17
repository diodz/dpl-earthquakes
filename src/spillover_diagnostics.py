#!/usr/bin/env python3
"""
Spillover / SUTVA diagnostics for Chile (Maule) and New Zealand (Canterbury).

Addresses reviewer concerns about spatial spillovers by:
  1. Assembling explicit spillover diagnostic proxies (population flow,
     sectoral similarity as inter-regional linkage proxy).
  2. Running SCM under progressively restrictive geographic donor
     exclusion rules (rings / adjacency).
  3. Producing side-by-side baseline vs spillover-robust estimates.

Outputs (all written to article_assets/):
  - chile_spillover_summary.csv
  - nz_spillover_summary.csv
  - spillover_population_diagnostics.csv
  - spillover_sectoral_linkage.csv
  - spillover_donor_exclusion_comparison.png
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pysyncon import Dataprep, Synth

import nz_util
import process_chile_gdp_data as pcd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

ANALYSIS_END_YEAR = 2019

# ── Chile constants ──────────────────────────────────────────────────────────

CHILE_TREATED = "VII Del Maule"
CHILE_BASELINE_CONTROLS = [
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
    "agropecuario", "pesca", "mineria", "industria_m",
    "electricidad", "construccion", "comercio", "transporte",
    "servicios_financieros", "vivienda", "personales", "publica",
]
CHILE_TREATMENT_YEAR = 2010

# Geographic adjacency rings for Maule (VII)
# Ring 1: directly adjacent (O'Higgins shares northern border)
# Note: Biobío (VIII) is already excluded from the baseline donor pool
CHILE_RING1 = [
    "VI Del Libertador General Bernardo OHiggins",
]
# Ring 2: next closest regions (RM Santiago, Valparaíso, Araucanía)
CHILE_RING2 = [
    "RMS Región Metropolitana de Santiago",
    "V De Valparaíso",
    "IX De La Araucanía",
]

CHILE_EXCLUSION_SCENARIOS = [
    {
        "label": "Baseline (all donors)",
        "exclude": [],
    },
    {
        "label": "Excl. adjacent (O'Higgins)",
        "exclude": CHILE_RING1,
    },
    {
        "label": "Excl. Ring 1+2 (O'Higgins, RM, Valparaíso, Araucanía)",
        "exclude": CHILE_RING1 + CHILE_RING2,
    },
]

# ── NZ constants ─────────────────────────────────────────────────────────────

NZ_TREATED = "Canterbury"
NZ_BASELINE_CONTROLS = [
    "Auckland", "Bay of Plenty", "Gisborne", "Hawke's Bay",
    "Manawatu-Whanganui", "Marlborough", "Northland", "Otago",
    "Southland", "Taranaki", "Tasman/Nelson", "Waikato",
    "Wellington", "West Coast",
]

NZ_TREATMENT_YEAR = 2011

# Geographic adjacency rings for Canterbury
# Ring 1: South Island neighbours sharing a border
NZ_RING1 = ["West Coast", "Otago", "Marlborough", "Tasman/Nelson"]
# Ring 2: remaining South Island region + major linked North Island centres
NZ_RING2 = ["Southland", "Wellington", "Auckland"]

NZ_EXCLUSION_SCENARIOS = [
    {
        "label": "Baseline (all donors)",
        "exclude": [],
    },
    {
        "label": "Excl. adjacent South Island",
        "exclude": NZ_RING1,
    },
    {
        "label": "Excl. SI adjacent + Auckland/Wellington",
        "exclude": NZ_RING1 + ["Auckland", "Wellington"],
    },
    {
        "label": "North Island only (excl. all SI + Auckland + Wellington)",
        "exclude": NZ_RING1 + NZ_RING2,
    },
]


# ── Helper: RMSPE ────────────────────────────────────────────────────────────

def _rmspe(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(valid))))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(den) or np.isclose(den, 0.0):
        return float("nan")
    return float(num / den)


# ── Chile data / Dataprep builder ────────────────────────────────────────────

def _load_chile_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_PROJECT_ROOT, "inter", "processed_chile.csv"))


def _build_chile_dataprep(
    df: pd.DataFrame,
    controls: list[str],
    treatment_year: int = CHILE_TREATMENT_YEAR,
) -> Dataprep:
    pred_window = range(max(1990, treatment_year - 5), treatment_year)
    edu_window = range(max(1990, treatment_year - 2), treatment_year)
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
        time_optimize_ssr=range(1990, treatment_year),
    )


# ── NZ data / Dataprep builder ──────────────────────────────────────────────

def _load_nz_data() -> pd.DataFrame:
    nz = nz_util.clean_data_for_synthetic_control().copy()
    nz["Tertiary Share"] = nz["Tertiary"] / nz["Population"]
    return nz


def _build_nz_dataprep(
    df: pd.DataFrame,
    controls: list[str],
    treatment_year: int = NZ_TREATMENT_YEAR,
) -> Dataprep:
    pred_window = range(max(2000, treatment_year - 5), treatment_year)
    tert_window = range(max(2000, treatment_year - 2), treatment_year)
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
        time_optimize_ssr=range(2000, treatment_year),
    )


# ── Core: run one SCM scenario and extract summary statistics ────────────────

def _run_scenario(
    df: pd.DataFrame,
    dataprep: Dataprep,
    time_col: str,
    treatment_year: int,
    fit_start_year: int,
    optim_method: str,
    optim_initial: str,
) -> dict:
    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method=optim_method, optim_initial=optim_initial)

    all_years = list(range(int(df[time_col].min()), int(df[time_col].max()) + 1))
    z0, z1 = synth.dataprep.make_outcome_mats(time_period=all_years)
    synthetic = pd.Series(
        np.asarray(synth._synthetic(z0)).flatten().astype(float), index=all_years,
    )
    treated_outcome = pd.Series(
        np.asarray(z1).flatten().astype(float), index=all_years,
    )
    gap_level = treated_outcome - synthetic
    gap_pct = gap_level / synthetic * 100.0

    year_arr = np.asarray(all_years, dtype=int)
    pre_mask = (year_arr >= fit_start_year) & (year_arr < treatment_year)
    post_mask = (year_arr >= treatment_year) & (year_arr <= ANALYSIS_END_YEAR)

    gap_pct_values = gap_pct.to_numpy(dtype=float)
    pre_rmspe = _rmspe(gap_pct_values[pre_mask])
    post_rmspe = _rmspe(gap_pct_values[post_mask])
    rmspe_ratio = _safe_ratio(post_rmspe, pre_rmspe)
    mean_post_gap_pct = float(np.nanmean(gap_pct_values[post_mask]))

    gap_level_values = gap_level.to_numpy(dtype=float)
    mean_post_gap_level = float(np.nanmean(gap_level_values[post_mask]))

    return {
        "PreRMSPE_Pct": pre_rmspe,
        "PostRMSPE_Pct": post_rmspe,
        "RMSPERatio": rmspe_ratio,
        "MeanPostGapPct": mean_post_gap_pct,
        "MeanPostGapLevel": mean_post_gap_level,
        "gap_pct_series": gap_pct,
        "gap_level_series": gap_level,
        "treated_series": treated_outcome,
        "synthetic_series": synthetic,
    }


# ── Spillover proxy: population-flow diagnostics ────────────────────────────

def _population_flow_diagnostics(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    pop_col: str,
    treated: str,
    adjacent_regions: list[str],
    treatment_year: int,
    pre_window: int = 5,
    post_window: int = 5,
) -> pd.DataFrame:
    """Compare population growth rates in adjacent vs non-adjacent regions
    before and after the treatment year as a proxy for migration flows."""
    records = []
    all_regions = df[unit_col].unique()
    for region in all_regions:
        rdf = df[df[unit_col] == region].sort_values(time_col).copy()
        rdf["pop_growth"] = rdf[pop_col].pct_change(fill_method=None) * 100.0
        pre_years = range(treatment_year - pre_window, treatment_year)
        post_years = range(treatment_year, treatment_year + post_window + 1)
        pre_growth = rdf[rdf[time_col].isin(pre_years)]["pop_growth"].mean()
        post_growth = rdf[rdf[time_col].isin(post_years)]["pop_growth"].mean()
        change = post_growth - pre_growth if np.isfinite(pre_growth) and np.isfinite(post_growth) else np.nan

        if region == treated:
            ring = "Treated"
        elif region in adjacent_regions:
            ring = "Adjacent"
        else:
            ring = "Non-adjacent"

        records.append({
            "Region": region,
            "RingCategory": ring,
            "PrePopGrowthPctAvg": pre_growth,
            "PostPopGrowthPctAvg": post_growth,
            "PopGrowthChange_pp": change,
        })
    return pd.DataFrame(records)


# ── Spillover proxy: sectoral linkage (cosine similarity) ───────────────────

def _sectoral_linkage_diagnostics(
    df: pd.DataFrame,
    unit_col: str,
    time_col: str,
    sector_cols: list[str],
    treated: str,
    adjacent_regions: list[str],
    treatment_year: int,
    pre_window: int = 5,
) -> pd.DataFrame:
    """Compute sectoral composition similarity (cosine) between the treated
    region and each donor as a proxy for economic linkage / trade exposure."""
    pre_years = range(treatment_year - pre_window, treatment_year)
    treated_vec = (
        df[(df[unit_col] == treated) & (df[time_col].isin(pre_years))][sector_cols]
        .mean()
        .to_numpy(dtype=float)
    )
    treated_norm = np.linalg.norm(treated_vec)
    if treated_norm == 0:
        treated_norm = 1e-12

    records = []
    all_regions = [r for r in df[unit_col].unique() if r != treated]
    for region in all_regions:
        rvec = (
            df[(df[unit_col] == region) & (df[time_col].isin(pre_years))][sector_cols]
            .mean()
            .to_numpy(dtype=float)
        )
        rnorm = np.linalg.norm(rvec)
        if rnorm == 0:
            rnorm = 1e-12
        cos_sim = float(np.dot(treated_vec, rvec) / (treated_norm * rnorm))
        ring = "Adjacent" if region in adjacent_regions else "Non-adjacent"
        records.append({
            "Region": region,
            "RingCategory": ring,
            "CosineSimilarity": cos_sim,
        })
    return pd.DataFrame(records).sort_values("CosineSimilarity", ascending=False)


# ── Main: Chile spillover analysis ──────────────────────────────────────────

def run_chile_spillover(output_dir: str = FIGURES_DIR) -> pd.DataFrame:
    print("  Loading Chile data...")
    chile_df = _load_chile_data()
    summary_rows = []

    for scenario in CHILE_EXCLUSION_SCENARIOS:
        controls = [c for c in CHILE_BASELINE_CONTROLS if c not in scenario["exclude"]]
        if len(controls) < 2:
            continue
        print(f"  Chile scenario: {scenario['label']} ({len(controls)} donors)")
        dp = _build_chile_dataprep(chile_df, controls)
        result = _run_scenario(
            chile_df, dp,
            time_col="year",
            treatment_year=CHILE_TREATMENT_YEAR,
            fit_start_year=1990,
            optim_method="Nelder-Mead",
            optim_initial="ols",
        )
        summary_rows.append({
            "Country": "Chile",
            "Scenario": scenario["label"],
            "ExcludedRegions": "; ".join(scenario["exclude"]) if scenario["exclude"] else "None",
            "NumDonors": len(controls),
            "PreRMSPE_Pct": result["PreRMSPE_Pct"],
            "PostRMSPE_Pct": result["PostRMSPE_Pct"],
            "RMSPERatio": result["RMSPERatio"],
            "MeanPostGapPct": result["MeanPostGapPct"],
            "MeanPostGapLevel": result["MeanPostGapLevel"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "chile_spillover_summary.csv"), index=False)

    # Population flow diagnostics
    pop_diag = _population_flow_diagnostics(
        chile_df, "region_name", "year", "population",
        CHILE_TREATED, CHILE_RING1 + ["VIII Del Biobío"],
        CHILE_TREATMENT_YEAR,
    )
    pop_diag.insert(0, "Country", "Chile")

    # Sectoral linkage
    sect_diag = _sectoral_linkage_diagnostics(
        chile_df, "region_name", "year", CHILE_PREDICTORS,
        CHILE_TREATED, CHILE_RING1 + ["VIII Del Biobío"],
        CHILE_TREATMENT_YEAR,
    )
    sect_diag.insert(0, "Country", "Chile")

    return summary_df, pop_diag, sect_diag


# ── Main: NZ spillover analysis ─────────────────────────────────────────────

def run_nz_spillover(output_dir: str = FIGURES_DIR) -> pd.DataFrame:
    print("  Loading NZ data...")
    nz_df = _load_nz_data()
    summary_rows = []

    for scenario in NZ_EXCLUSION_SCENARIOS:
        controls = [c for c in NZ_BASELINE_CONTROLS if c not in scenario["exclude"]]
        if len(controls) < 2:
            continue
        print(f"  NZ scenario: {scenario['label']} ({len(controls)} donors)")
        dp = _build_nz_dataprep(nz_df, controls)
        result = _run_scenario(
            nz_df, dp,
            time_col="Year",
            treatment_year=NZ_TREATMENT_YEAR,
            fit_start_year=2000,
            optim_method="Nelder-Mead",
            optim_initial="equal",
        )
        summary_rows.append({
            "Country": "New Zealand",
            "Scenario": scenario["label"],
            "ExcludedRegions": "; ".join(scenario["exclude"]) if scenario["exclude"] else "None",
            "NumDonors": len(controls),
            "PreRMSPE_Pct": result["PreRMSPE_Pct"],
            "PostRMSPE_Pct": result["PostRMSPE_Pct"],
            "RMSPERatio": result["RMSPERatio"],
            "MeanPostGapPct": result["MeanPostGapPct"],
            "MeanPostGapLevel": result["MeanPostGapLevel"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "nz_spillover_summary.csv"), index=False)

    nz_sector_cols = nz_util.SECTORIAL_GDP_VARIABLES
    available_sectors = [c for c in nz_sector_cols if c in nz_df.columns]

    pop_diag = _population_flow_diagnostics(
        nz_df, "Region", "Year", "Population",
        NZ_TREATED, NZ_RING1,
        NZ_TREATMENT_YEAR,
    )
    pop_diag.insert(0, "Country", "New Zealand")

    sect_diag = _sectoral_linkage_diagnostics(
        nz_df, "Region", "Year", available_sectors,
        NZ_TREATED, NZ_RING1,
        NZ_TREATMENT_YEAR,
    )
    sect_diag.insert(0, "Country", "New Zealand")

    return summary_df, pop_diag, sect_diag


# ── Plotting ─────────────────────────────────────────────────────────────────

def _plot_comparison(
    chile_summary: pd.DataFrame,
    nz_summary: pd.DataFrame,
    output_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    for ax, (title, sdf) in zip(
        axes,
        [("Chile (Maule)", chile_summary), ("New Zealand (Canterbury)", nz_summary)],
    ):
        x = np.arange(len(sdf))
        bars = ax.bar(x, sdf["MeanPostGapPct"], color="#4c78a8", alpha=0.85, width=0.55)
        for i, (_, row) in enumerate(sdf.iterrows()):
            sign = "+" if row["MeanPostGapPct"] >= 0 else ""
            ax.text(
                i, row["MeanPostGapPct"] + (0.3 if row["MeanPostGapPct"] >= 0 else -0.6),
                f"{sign}{row['MeanPostGapPct']:.2f}%\nRMSPE ratio: {row['RMSPERatio']:.2f}",
                ha="center", va="bottom", fontsize=7,
            )
            if i == 0:
                bars[i].set_color("#d62728")
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["Scenario"], rotation=18, ha="right", fontsize=7)
        ax.set_title(title, fontsize=10)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Mean post-treatment gap (%)")
    fig.suptitle("Spillover-robust donor pool comparison", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def run_spillover_diagnostics(output_dir: str = FIGURES_DIR) -> None:
    print("Running spillover / SUTVA diagnostics...")
    os.makedirs(output_dir, exist_ok=True)

    chile_summary, chile_pop, chile_sect = run_chile_spillover(output_dir)
    nz_summary, nz_pop, nz_sect = run_nz_spillover(output_dir)

    # Combine and save population diagnostics
    pop_all = pd.concat([chile_pop, nz_pop], ignore_index=True)
    pop_all.to_csv(os.path.join(output_dir, "spillover_population_diagnostics.csv"), index=False)

    # Combine and save sectoral linkage
    sect_all = pd.concat([chile_sect, nz_sect], ignore_index=True)
    sect_all.to_csv(os.path.join(output_dir, "spillover_sectoral_linkage.csv"), index=False)

    # Plot side-by-side comparison
    _plot_comparison(
        chile_summary, nz_summary,
        os.path.join(output_dir, "spillover_donor_exclusion_comparison.png"),
    )

    print("Spillover diagnostics complete.")
    print(f"  Chile summary:\n{chile_summary.to_string(index=False)}")
    print(f"  NZ summary:\n{nz_summary.to_string(index=False)}")


if __name__ == "__main__":
    run_spillover_diagnostics()

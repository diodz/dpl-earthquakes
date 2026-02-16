"""
SCM robustness analysis: SDID and Bias-Corrected (Penalized) SCM.

Addresses Reviewer 1 Comment 3.3:
- SDID (Arkhangelsky et al., 2021): Verify Canterbury overshoot persists with unit and time weights.
- Bias-corrected SCM (Abadie & L'Hour, 2021): Mitigate interpolation bias for Maule as agrarian outlier.
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Chile: Maule treated 2010, T0=2009
CHILE_TREATED = "VII Del Maule"
CHILE_T0 = 2009
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

# NZ: Canterbury treated 2011, T0=2010
NZ_TREATED = "Canterbury"
NZ_T0 = 2010
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


def _prepare_chile_panel() -> pd.DataFrame:
    """Load Chile data and build panel for SDID (unit, time, outcome, quota)."""
    import process_chile_gdp_data as pcd

    df = pcd.process_data_for_synth()
    df = df[df["region_name"].isin([CHILE_TREATED] + CHILE_CONTROLS)]
    df = df.rename(columns={"region_name": "unit", "year": "time", "gdp_per_capita": "outcome"})
    # Block design: quota=1 only for treated unit in post-treatment periods (time > T0)
    df["quota"] = ((df["unit"] == CHILE_TREATED) & (df["time"] > CHILE_T0)).astype(int)
    return df[["unit", "time", "outcome", "quota"]].dropna()


def _prepare_nz_panel() -> pd.DataFrame:
    """Load NZ data and build panel for SDID."""
    import nz_util

    df = nz_util.clean_data_for_synthetic_control()
    df = df[df["Region"].isin([NZ_TREATED] + NZ_CONTROLS)]
    df = df.rename(columns={"Region": "unit", "Year": "time", "gdp_per_capita": "outcome"})
    # Block design: quota=1 only for treated unit in post-treatment periods (time > T0)
    df["quota"] = ((df["unit"] == NZ_TREATED) & (df["time"] > NZ_T0)).astype(int)
    return df[["unit", "time", "outcome", "quota"]].dropna()


def _run_sdid(df: pd.DataFrame, unit_col: str, time_col: str, quota_col: str, outcome_col: str) -> Optional[object]:
    """Run synthdid and return fitted model or None if import fails."""
    try:
        from synthdid import Synthdid

        model = Synthdid(df, unit_col, time_col, quota_col, outcome_col)
        model.fit()
        return model
    except ImportError:
        print("synthdid not installed; skipping SDID. Install with: pip install synthdid")
        return None
    except Exception as e:
        print(f"SDID failed: {e}")
        return None


def _run_penalized_synth(dataprep, lambda_: float = 0.01) -> Optional[object]:
    """Run PenalizedSynth (Abadie & L'Hour) and return fitted model or None."""
    try:
        from pysyncon import PenalizedSynth

        model = PenalizedSynth()
        model.fit(dataprep=dataprep, lambda_=lambda_)
        return model
    except Exception as e:
        print(f"PenalizedSynth failed: {e}")
        return None


def run_sdid_chile() -> Optional[float]:
    """Run SDID for Maule. Returns ATT or None."""
    df = _prepare_chile_panel()
    model = _run_sdid(df, "unit", "time", "quota", "outcome")
    if model is None:
        return None
    try:
        att = float(model.summary2.iloc[0, 0])
        return att
    except Exception:
        return None


def run_sdid_nz() -> Optional[float]:
    """Run SDID for Canterbury. Returns ATT or None."""
    df = _prepare_nz_panel()
    model = _run_sdid(df, "unit", "time", "quota", "outcome")
    if model is None:
        return None
    try:
        att = float(model.summary2.iloc[0, 0])
        return att
    except Exception:
        return None


def run_penalized_maule() -> Optional[dict]:
    """Run PenalizedSynth for Maule. Returns dict with weights summary or None."""
    import process_chile_gdp_data as pcd
    from pysyncon import Dataprep

    df = pcd.process_data_for_synth()
    dataprep = Dataprep(
        foo=df,
        predictors=["agropecuario", "pesca", "mineria", "industria_m", "electricidad", "construccion",
                    "comercio", "transporte", "servicios_financieros", "vivienda", "personales", "publica"],
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
    model = _run_penalized_synth(dataprep)
    if model is None:
        return None
    try:
        w = model.weights()
        top = w.nlargest(5)
        return {"top_weights": top.to_dict(), "n_donors": len(CHILE_CONTROLS)}
    except Exception:
        return None


def run_penalized_canterbury() -> Optional[dict]:
    """Run PenalizedSynth for Canterbury. Returns dict with weights summary or None."""
    import nz_util
    from pysyncon import Dataprep

    df = nz_util.clean_data_for_synthetic_control()
    df["Tertiary Share"] = df["Tertiary"] / df["total_population"]
    dataprep = Dataprep(
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
        controls_identifier=NZ_CONTROLS,
        time_optimize_ssr=range(2000, 2009),
    )
    model = _run_penalized_synth(dataprep)
    if model is None:
        return None
    try:
        w = model.weights()
        top = w.nlargest(5)
        return {"top_weights": top.to_dict(), "n_donors": len(NZ_CONTROLS)}
    except Exception:
        return None


def _pre_treatment_mean(df: pd.DataFrame, unit_col: str, time_col: str, outcome_col: str, treated_id: str, t0: int) -> float:
    """Average outcome for treated unit in pre-treatment period (time <= t0)."""
    pre = df[(df[unit_col] == treated_id) & (df[time_col] <= t0)]
    return float(pre[outcome_col].mean())


def plot_sdid_comparison(
    att_chile: Optional[float],
    att_nz: Optional[float],
    baseline_chile: Optional[float] = None,
    baseline_nz: Optional[float] = None,
) -> None:
    """Plot SDID vs baseline SCM ATT comparison. All values shown in percentage of pre-treatment mean."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Chile: baseline SCM ~0% (null), SDID as %
    att_chile_pct = (100 * att_chile / baseline_chile) if (att_chile is not None and baseline_chile and baseline_chile != 0) else (att_chile or 0)
    axes[0].bar(["Baseline SCM\n(avg post gap)", "SDID"], [0, att_chile_pct], color=["gray", "steelblue"])
    axes[0].set_title("Maule (Chile)")
    axes[0].set_ylabel("ATT (%)")
    # NZ: baseline ~6.8% gap, SDID as % (same scale)
    att_nz_pct = (100 * att_nz / baseline_nz) if (att_nz is not None and baseline_nz and baseline_nz != 0) else (att_nz or 0)
    baseline_scm_pct = 6.8
    axes[1].bar(["Baseline SCM\n(avg post gap)", "SDID"], [baseline_scm_pct, att_nz_pct], color=["gray", "steelblue"])
    axes[1].set_title("Canterbury (NZ)")
    axes[1].set_ylabel("ATT (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sdid_comparison.png"))
    plt.close()


def save_summary_csv(
    sdid_chile: Optional[float],
    sdid_nz: Optional[float],
    penalized_maule: Optional[dict],
    penalized_canterbury: Optional[dict],
) -> None:
    """Save summary CSV of robustness results."""
    rows = [
        {"case": "Maule", "method": "SDID", "att": sdid_chile},
        {"case": "Canterbury", "method": "SDID", "att": sdid_nz},
        {"case": "Maule", "method": "Penalized SCM", "top_weights": str(penalized_maule.get("top_weights", "")) if penalized_maule else ""},
        {"case": "Canterbury", "method": "Penalized SCM", "top_weights": str(penalized_canterbury.get("top_weights", "")) if penalized_canterbury else ""},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(FIGURES_DIR, "scm_robustness_summary.csv"), index=False)


def main() -> None:
    print("Running SDID for Chile (Maule)...")
    sdid_chile = run_sdid_chile()
    print(f"  Maule SDID ATT: {sdid_chile}")

    print("Running SDID for New Zealand (Canterbury)...")
    sdid_nz = run_sdid_nz()
    print(f"  Canterbury SDID ATT: {sdid_nz}")

    print("Running PenalizedSynth for Maule...")
    penalized_maule = run_penalized_maule()
    print(f"  Maule penalized top weights: {penalized_maule}")

    print("Running PenalizedSynth for Canterbury...")
    penalized_canterbury = run_penalized_canterbury()
    print(f"  Canterbury penalized top weights: {penalized_canterbury}")

    # Pre-treatment mean outcomes for converting level ATT to percentage (same scale as baseline SCM gap %)
    df_chile = _prepare_chile_panel()
    df_nz = _prepare_nz_panel()
    baseline_chile = _pre_treatment_mean(df_chile, "unit", "time", "outcome", CHILE_TREATED, CHILE_T0)
    baseline_nz = _pre_treatment_mean(df_nz, "unit", "time", "outcome", NZ_TREATED, NZ_T0)

    plot_sdid_comparison(sdid_chile, sdid_nz, baseline_chile, baseline_nz)
    save_summary_csv(sdid_chile, sdid_nz, penalized_maule, penalized_canterbury)
    print("Outputs saved to article_assets/")


if __name__ == "__main__":
    main()

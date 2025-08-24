# scripts/run_chile.py
from pathlib import Path
import pandas as pd

from scripts.config import CHILE_MAIN, CHILE_TREATED, CHILE_EVENT_YEAR, CHILE_DONORS, FIG_DIR
from scripts.loaders import load_chile_gdppc
from scripts.scm_core import fit_scm, rmspe, placebo_in_space, leave_one_out
from scripts.plots import save_lineplot, save_gapplot, save_placebo_plot, save_gap_with_band, save_loo_plot

def main(base_dir: Path = Path(".")):
    y = load_chile_gdppc(base_dir / CHILE_MAIN)
    print(y)
    years = y.index.tolist()
    pre_years = [t for t in years if t < CHILE_EVENT_YEAR]
    post_years = [t for t in years if t >= CHILE_EVENT_YEAR]

    donors = [d for d in CHILE_DONORS if d in y.columns]
    res = fit_scm(y, CHILE_TREATED, donors, pre_years)
    y_synth = res["y_synth"]
    pre_r = res["pre_rmspe"]
    post_r = rmspe(y[CHILE_TREATED], y_synth, post_years)

    # Figures: level paths
    save_lineplot(
        {"Actual": y[CHILE_TREATED], "Synthetic": y_synth},
        f"Chile – {CHILE_TREATED} vs. Synthetic",
        "Year", "GDP per capita",
        FIG_DIR / "fig_maule_scm"
    )
    gap = y[CHILE_TREATED] - y_synth
    save_gapplot(gap, CHILE_EVENT_YEAR, f"Chile – {CHILE_TREATED}: Gap (Actual − Synthetic)", FIG_DIR / "fig_maule_gap")

    # Placebos and 90% band
    donors_all = [c for c in y.columns if c != CHILE_TREATED]
    plc = placebo_in_space(y, CHILE_TREATED, donors_all, CHILE_EVENT_YEAR, pre_years, post_years)
    save_placebo_plot(plc, CHILE_TREATED, CHILE_EVENT_YEAR, FIG_DIR / "fig_maule_placebos")

    plc_non = plc[plc["unit"] != CHILE_TREATED]
    if not plc_non.empty:
        gaps_mat = pd.DataFrame({row["unit"]: row["gaps"] for _, row in plc_non.iterrows()})
        low = gaps_mat.quantile(0.05, axis=1)
        high = gaps_mat.quantile(0.95, axis=1)
        save_gap_with_band(gap, low, high, CHILE_EVENT_YEAR, f"Chile – {CHILE_TREATED}: Gap with 90% placebo band", FIG_DIR / "fig_maule_gap_band")

    # Leave-one-out for influential donors (weights > 0.05)
    loo_res = leave_one_out(y, CHILE_TREATED, donors, pre_years)
    gaps_dict = {"base": gap}
    for d, w in res["weights"].items():
        if w > 0.05 and f"drop_{d}" in loo_res and "y_synth" in loo_res[f"drop_{d}"]:
            y_synth_alt = loo_res[f"drop_{d}"]["y_synth"]
            gaps_dict[f"drop {d}"] = y[CHILE_TREATED] - y_synth_alt
    save_loo_plot(gaps_dict, CHILE_EVENT_YEAR, f"Chile – {CHILE_TREATED}: Leave-one-out gaps", FIG_DIR / "fig_maule_loo")


    # Pseudo p-values based on RMSPE ratio (treated vs donors with good pre-fit)
    treated_row = plc[plc["unit"] == CHILE_TREATED].iloc[0]
    donors_ok = plc[plc["unit"] != CHILE_TREATED]
    p_ratio = float((donors_ok["ratio"] >= treated_row["ratio"]).mean()) if not donors_ok.empty else 1.0
    p_table = pd.DataFrame({"Statistic":["RMSPE Ratio (treated)","Pseudo p-value"], "Value":[treated_row["ratio"], p_ratio]})
    p_table.to_latex((FIG_DIR.parent / "tables" / "table_pvalues_maule.tex"), index=False, float_format="%.3f")

    # Tables
    w = res["weights"].reset_index()
    w.columns = ["Region", "Weight"]
    (FIG_DIR.parent / "tables").mkdir(parents=True, exist_ok=True)
    w.to_latex((FIG_DIR.parent / "tables" / "table_weights_maule.tex"), index=False, float_format="%.3f")

    stats = pd.DataFrame({
        "Metric": ["Pre RMSPE", "Post RMSPE", "RMSPE Ratio"],
        "Value": [pre_r, post_r, post_r / (pre_r + 1e-12)]
    })
    stats.to_latex((FIG_DIR.parent / "tables" / "table_fit_maule.tex"), index=False, float_format="%.3f")

if __name__ == "__main__":
    main()

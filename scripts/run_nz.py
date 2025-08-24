# scripts/run_nz.py
from pathlib import Path
import pandas as pd

from scripts.config import NZ_MAIN, NZ_TREATED, NZ_EVENT_YEAR, NZ_EXCLUDE, FIG_DIR
from scripts.loaders import load_nz_gdppc
from scripts.scm_core import fit_scm, rmspe, placebo_in_space, leave_one_out
from scripts.plots import save_lineplot, save_gapplot, save_placebo_plot, save_gap_with_band, save_loo_plot

def main(base_dir: Path = Path(".")):
    y = load_nz_gdppc(base_dir / NZ_MAIN)
    donors = [r for r in y.columns if r not in NZ_EXCLUDE and r != NZ_TREATED]

    years = y.index.tolist()
    pre_years = [t for t in years if t < NZ_EVENT_YEAR]
    post_years = [t for t in years if t >= NZ_EVENT_YEAR]

    res = fit_scm(y, NZ_TREATED, donors, pre_years)
    y_synth = res["y_synth"]

    pre_r = res["pre_rmspe"]
    post_r = rmspe(y[NZ_TREATED], y_synth, post_years)

    # Figures: level paths
    save_lineplot(
        {"Actual": y[NZ_TREATED], "Synthetic": y_synth},
        f"NZ – {NZ_TREATED} vs. Synthetic",
        "Year", "GDP per capita",
        FIG_DIR / "fig_canterbury_scm"
    )

    gap = y[NZ_TREATED] - y_synth
    save_gapplot(gap, NZ_EVENT_YEAR, f"NZ – {NZ_TREATED}: Gap (Actual − Synthetic)", FIG_DIR / "fig_canterbury_gap")

    # Placebos and 90% band
    donors_all = [c for c in y.columns if c != NZ_TREATED]
    plc = placebo_in_space(y, NZ_TREATED, donors_all, NZ_EVENT_YEAR, pre_years, post_years)
    save_placebo_plot(plc, NZ_TREATED, NZ_EVENT_YEAR, FIG_DIR / "fig_canterbury_placebos")

    plc_non = plc[plc["unit"] != NZ_TREATED]
    if not plc_non.empty:
        gaps_mat = pd.DataFrame({row["unit"]: row["gaps"] for _, row in plc_non.iterrows()})
        low = gaps_mat.quantile(0.05, axis=1)
        high = gaps_mat.quantile(0.95, axis=1)
        save_gap_with_band(gap, low, high, NZ_EVENT_YEAR, f"NZ – {NZ_TREATED}: Gap with 90% placebo band", FIG_DIR / "fig_canterbury_gap_band")

    # Leave-one-out for influential donors (weights > 0.05)
    loo_res = leave_one_out(y, NZ_TREATED, donors, pre_years)
    gaps_dict = {"base": gap}
    for d, w in res["weights"].items():
        if w > 0.05 and f"drop_{d}" in loo_res and "y_synth" in loo_res[f"drop_{d}"]:
            y_synth_alt = loo_res[f"drop_{d}"]["y_synth"]
            gaps_dict[f"drop {d}"] = y[NZ_TREATED] - y_synth_alt
    save_loo_plot(gaps_dict, NZ_EVENT_YEAR, f"NZ – {NZ_TREATED}: Leave-one-out gaps", FIG_DIR / "fig_canterbury_loo")


    # Pseudo p-values based on RMSPE ratio (treated vs donors with good pre-fit)
    treated_row = plc[plc["unit"] == NZ_TREATED].iloc[0]
    donors_ok = plc[plc["unit"] != NZ_TREATED]
    p_ratio = float((donors_ok["ratio"] >= treated_row["ratio"]).mean()) if not donors_ok.empty else 1.0
    p_table = pd.DataFrame({"Statistic":["RMSPE Ratio (treated)","Pseudo p-value"], "Value":[treated_row["ratio"], p_ratio]})
    p_table.to_latex((FIG_DIR.parent / "tables" / "table_pvalues_nz.tex"), index=False, float_format="%.3f")

    # Tables
    w = res["weights"].reset_index()
    w.columns = ["Region", "Weight"]
    (FIG_DIR.parent / "tables").mkdir(parents=True, exist_ok=True)
    w.to_latex((FIG_DIR.parent / "tables" / "table_weights_canterbury.tex"), index=False, float_format="%.3f")

    stats = pd.DataFrame({
        "Metric": ["Pre RMSPE", "Post RMSPE", "RMSPE Ratio"],
        "Value": [pre_r, post_r, post_r / (pre_r + 1e-12)]
    })
    stats.to_latex((FIG_DIR.parent / "tables" / "table_fit_canterbury.tex"), index=False, float_format="%.3f")

if __name__ == "__main__":
    main()

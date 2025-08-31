# scripts/spec_curve.py
from pathlib import Path
import pandas as pd

from scripts.config import CHILE_MAIN, CHILE_TREATED, CHILE_EVENT_YEAR, CHILE_DONORS, NZ_MAIN, NZ_TREATED, NZ_EVENT_YEAR, FIG_DIR
from scripts.loaders import load_chile_gdppc, load_nz_gdppc
from scripts.scm_core import fit_scm
import matplotlib.pyplot as plt

def run_curve(y, treated, donors, event_year, target_year):
    years = y.index.tolist()
    pre_years = [t for t in years if t < event_year]
    base = fit_scm(y, treated, donors, pre_years)
    base_gap = (y[treated] - base["y_synth"]).loc[target_year]
    rows = []
    for d in donors:
        dd = [x for x in donors if x != d]
        try:
            res = fit_scm(y, treated, dd, pre_years)
            gap = (y[treated] - res["y_synth"]).loc[target_year]
            rows.append({"drop": d, "gap": gap})
        except Exception:
            continue
    df = pd.DataFrame(rows).sort_values("gap")
    return base_gap, df

def main(base_dir: Path = Path(".")):
    # NZ
    y_nz = load_nz_gdppc(base_dir/NZ_MAIN)
    donors_nz = [c for c in y_nz.columns if c not in {"Marlborough", "Canterbury", "New Zealand", "North Island", "South Island"}]
    base_gap, df = run_curve(y_nz, NZ_TREATED, donors_nz, NZ_EVENT_YEAR, target_year=2016)
    plt.figure()
    plt.hlines(base_gap, 0, len(df), linestyles="--", label="Baseline")
    plt.plot(range(len(df)), df["gap"].to_numpy(), marker="o", linestyle="None", label="drop-one")
    plt.title("NZ – Gap in 2016 across leave-one-out donor specs")
    plt.xlabel("Specification (donor dropped)")
    plt.ylabel("Gap (per capita)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig_spec_curve_nz.pdf")
    plt.savefig(FIG_DIR/"fig_spec_curve_nz.png", dpi=200)
    plt.close()

    # Chile
    y_ch = load_chile_gdppc(base_dir/CHILE_MAIN)
    donors_ch = [d for d in CHILE_DONORS if d in y_ch.columns]
    base_gap, df = run_curve(y_ch, CHILE_TREATED, donors_ch, CHILE_EVENT_YEAR, target_year=2012)
    plt.figure()
    plt.hlines(base_gap, 0, len(df), linestyles="--", label="Baseline")
    if not df.empty:
        plt.plot(range(len(df)), df["gap"].to_numpy(), marker="o", linestyle="None", label="drop-one")
    plt.title("Chile – Gap in 2012 across leave-one-out donor specs")
    plt.xlabel("Specification (donor dropped)")
    plt.ylabel("Gap (per capita)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR/"fig_spec_curve_maule.pdf")
    plt.savefig(FIG_DIR/"fig_spec_curve_maule.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()

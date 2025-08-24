# scripts/run_placebos.py
from pathlib import Path
from scripts.config import FIG_DIR, CHILE_MAIN, CHILE_TREATED, CHILE_EVENT_YEAR, NZ_MAIN, NZ_TREATED, NZ_EVENT_YEAR
from scripts.loaders import load_chile_gdppc, load_nz_gdppc
from scripts.scm_core import placebo_in_space
from scripts.plots import save_placebo_plot

def main(base_dir: Path = Path(".")):
    y_ch = load_chile_gdppc(base_dir/CHILE_MAIN)
    pre_ch = [t for t in y_ch.index if t < CHILE_EVENT_YEAR]
    post_ch = [t for t in y_ch.index if t >= CHILE_EVENT_YEAR]
    plc_ch = placebo_in_space(y_ch, CHILE_TREATED, [c for c in y_ch.columns if c != CHILE_TREATED], CHILE_EVENT_YEAR, pre_ch, post_ch)
    save_placebo_plot(plc_ch, CHILE_TREATED, CHILE_EVENT_YEAR, FIG_DIR/"fig_maule_placebos")

    y_nz = load_nz_gdppc(base_dir/NZ_MAIN)
    pre_nz = [t for t in y_nz.index if t < NZ_EVENT_YEAR]
    post_nz = [t for t in y_nz.index if t >= NZ_EVENT_YEAR]
    plc_nz = placebo_in_space(y_nz, NZ_TREATED, [c for c in y_nz.columns if c != NZ_TREATED], NZ_EVENT_YEAR, pre_nz, post_nz)
    save_placebo_plot(plc_nz, NZ_TREATED, NZ_EVENT_YEAR, FIG_DIR/"fig_canterbury_placebos")

if __name__ == "__main__":
    main()

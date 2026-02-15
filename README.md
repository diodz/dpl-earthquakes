# Earthquakes and the Wealth of Nations

This repository contains the analysis and code for the paper "Earthquakes and the Wealth of Nations: The cases of Chile and New Zealand" (Díaz, Paniagua, Larroulet).

## Reproducibility

### Regenerating all figures

All figures used by `main.tex` are produced by Python code and saved directly to `article_assets/`. To regenerate all figures:

```bash
python run_analysis.py
```

This script:
1. Runs `create_maps.py` (Maule and Canterbury maps)
2. Executes the Maule SCM notebook
3. Executes the Canterbury SCM notebook

### Figure output

Figures are written to `article_assets/`, which is the canonical location expected by `main.tex`:

- Maps: `Maule_map.png`, `Canterbury_map.png`
- GDP paths: `maule_gdp_paths.png`, `nz_gdp_paths.png`
- Gaps: `maule_gap.png`, `nz_gap.png`
- Placebos: `maule_placebos.png`, `nz_placebos.png`
- Sectoral: `nz_scm_Construction.png`, `nz_scm_Other_Sectors.png`
- Jackknife: `chile_jacknife.png`, `nz_jacknife.png`

## Requirements

See `requirements.txt`. Key dependencies: `pysyncon`, `pandas`, `matplotlib`, `geopandas`, `jupyter`, `nbconvert`.

## Running notebooks manually

Run the notebooks from the `notebooks/` directory so that relative paths (e.g. `../src`, `../data`) resolve correctly.

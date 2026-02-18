# Earthquakes and the Wealth of Nations

This repository contains the analysis and code for the paper "Earthquakes and the Wealth of Nations: The cases of Chile and New Zealand" (Díaz, Paniagua, Larroulet).

## Reproducibility

### Regenerating all figures

All figures used by `main.tex` are produced by Python code and saved directly to `article_assets/`. To regenerate all figures:

```bash
python run_analysis.py
```

This script:
1. Runs `src/create_maps.py` (Maule and Canterbury maps)
2. Runs `src/nz_outcome_extensions.py` (GDP / population decomposition)
3. Runs `src/sdid_bias_corrected_analysis.py` (SDID + penalized SCM robustness)
4. Runs `src/uniform_confidence_analysis.py` (uniform confidence sets + sensitivity checks)
5. Runs `src/treatment_timing_sensitivity.py` (2010/2011 timing diagnostics + placebo ranks)
6. Runs `src/nighttime_lights_validation.py` (independent night-time lights validation)
7. Runs `src/main_scm_figures.py` (main SCM paths, gaps, placebos, jackknife for NZ and Chile)
8. Runs `src/sectoral_appendix_analysis.py` (parallel Chile/NZ sectoral SCM appendix outputs)
9. Runs `src/predictor_weight_sensitivity.py` (predictor-weight tuning + harmonized predictor-set sensitivity)

> Note: the night-time lights validation step downloads yearly global rasters on first run and caches them under `data/ntl/rasters/` (large, ignored by git). Subsequent runs reuse the local cache.

### Figure output

Figures are written to `article_assets/`, which is the canonical location expected by `main.tex`:

- Maps: `Maule_map.png`, `Canterbury_map.png`
- GDP paths: `maule_gdp_paths.png`, `nz_gdp_paths.png`
- Gaps: `maule_gap.png`, `nz_gap.png`
- Placebos: `maule_placebos.png`, `nz_placebos.png`
- Uniform confidence sets: `scm_uniform_confidence_sets.png`, `chile_uniform_threshold_sensitivity.png`, `nz_uniform_treatment_timing_sensitivity.png`
- Sectoral: `nz_scm_Construction.png`, `nz_scm_Other_Sectors.png`
- Sectoral appendix (new): `chile_scm_Construction.png`, `chile_scm_Other_Sectors.png`,
  `sectoral_inference_summary.csv`, `sectoral_crowding_out_summary.csv`,
  `sectoral_grouping_sensitivity.csv`, `sectoral_appendix_series.xlsx`
- Predictor-weight / predictor-symmetry sensitivity: `predictor_spec_sensitivity.csv`
- Jackknife: `chile_jacknife.png`, `nz_jacknife.png`
- SDID / bias-corrected robustness: `sdid_bias_corrected_summary.csv`, `sdid_bias_corrected_gaps.png`
- Uniform confidence tables: `scm_uniform_confidence_sets.csv`, `scm_uniform_confidence_summary.csv`
- Timing sensitivity appendix outputs: `timing_sensitivity_summary.csv`, `timing_sensitivity_gap_paths.png`, `timing_sensitivity_rmspe_ratios.png`
- Night-time lights validation:
  `ntl_validation_paths_gaps.png`,
  `ntl_sensor_processing_robustness.png`,
  `ntl_spatial_sensitivity.png`,
  `ntl_validation_summary.csv`,
  `ntl_scm_summary.csv`,
  `ntl_scm_gaps.csv`

## Requirements

See `requirements.txt`. Key dependencies: `pysyncon`, `pandas`, `matplotlib`, `geopandas`.

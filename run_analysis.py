#!/usr/bin/env python3
"""
Regenerate all figures for the article (main.tex).

Runs:
  1. create_maps.py - generates Maule_map.png and Canterbury_map.png
  2. src/nz_outcome_extensions.py - generates NZ decomposition outcome figures/tables
  2b. src/chile_outcome_extensions.py - generates Maule (Chile) decomposition outcome figures/tables
  3. src/sdid_bias_corrected_analysis.py - generates SDID / penalized SCM robustness outputs
  4. src/uniform_confidence_analysis.py - uniform confidence sets + sensitivity checks
  5. src/treatment_timing_sensitivity.py - treatment-year sensitivity diagnostics/figures
  6. src/nighttime_lights_validation.py - generates independent NTL validation outputs
  6b. src/spillover_diagnostics.py - generates SUTVA/spillover diagnostics outputs
  7. Maule SCM.ipynb - generates maule_*, chile_jacknife figures
  8. Canterbury SCM.ipynb - regenerates nz_*, nz_scm_Construction, nz_scm_Other_Sectors
  9. src/sectoral_appendix_analysis.py - sectoral SCM appendix outputs/inference (runs last
     so its nz_scm_Construction.png and nz_scm_Other_Sectors.png are the final versions)
 10. src/predictor_weight_sensitivity.py - predictor weighting and cross-country
     harmonized predictor sensitivity table

All figures are written to article_assets/ (used by main.tex).
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")


def main():
    os.chdir(PROJECT_ROOT)

    # 1. Run create_maps.py (needs to run from notebooks/ for data paths, or we cd)
    print("Running create_maps.py...")
    create_maps_path = os.path.join(NOTEBOOKS_DIR, "create_maps.py")
    subprocess.run([sys.executable, create_maps_path], check=True, cwd=NOTEBOOKS_DIR)

    # 2. Run explicit NZ outcome extensions (per-capita, population, total GDP).
    print("Running NZ outcome extension SCM script...")
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "src", "nz_outcome_extensions.py")], check=True, cwd=PROJECT_ROOT)

    # 2b. Run Chile (Maule) outcome extensions (same decomposition for Y/L interpretability).
    print("Running Chile (Maule) outcome extension SCM script...")
    subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "src", "chile_outcome_extensions.py")], check=True, cwd=PROJECT_ROOT)

    # 3. Run SDID + penalized (bias-corrected) SCM robustness analysis.
    print("Running SDID / bias-corrected SCM robustness script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "sdid_bias_corrected_analysis.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 4. Run placebo-based uniform confidence set inference.
    print("Running SCM uniform confidence set script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "uniform_confidence_analysis.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 5. Run treatment timing sensitivity outputs (2010/2011/sequence-aware).
    print("Running treatment timing sensitivity script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "treatment_timing_sensitivity.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 6. Run nighttime-lights validation as an independent proxy robustness check.
    print("Running nighttime lights validation script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "nighttime_lights_validation.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 6b. Run spillover / SUTVA diagnostics (geographic donor exclusions).
    print("Running spillover diagnostics script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "spillover_diagnostics.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 7–8. Execute notebooks (nbconvert --execute runs from notebook's directory)
    # Note: Notebooks run before sectoral appendix script so that the sectoral script's
    # outputs (nz_scm_Construction.png, nz_scm_Other_Sectors.png) are the final versions.
    for nb_name in ["Maule SCM.ipynb", "Canterbury SCM.ipynb"]:
        nb_path = os.path.join(NOTEBOOKS_DIR, nb_name)
        if not os.path.exists(nb_path):
            print(f"Warning: {nb_path} not found, skipping")
            continue
        print(f"Executing {nb_name}...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                nb_path,
            ],
            check=True,
            cwd=NOTEBOOKS_DIR,
        )

    # 9. Run sectoral appendix SCM outputs (parallel Chile/NZ sectoral diagnostics).
    # Runs last so its outputs are not overwritten by the notebooks.
    print("Running sectoral SCM appendix script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "sectoral_appendix_analysis.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 10. Run predictor-weight and cross-country harmonized predictor sensitivity checks.
    print("Running predictor-weight sensitivity script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "predictor_weight_sensitivity.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    print("Done. Figures saved to article_assets/")


if __name__ == "__main__":
    main()

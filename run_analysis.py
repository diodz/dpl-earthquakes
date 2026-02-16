#!/usr/bin/env python3
"""
Regenerate all figures for the article (main.tex).

Runs:
  1. create_maps.py - generates Maule_map.png and Canterbury_map.png
  2. src/nz_outcome_extensions.py - generates NZ decomposition outcome figures/tables
  3. src/sdid_bias_corrected_analysis.py - generates SDID / penalized SCM robustness outputs
  4. Maule SCM.ipynb - generates maule_*, chile_jacknife figures
  5. Canterbury SCM.ipynb - regenerates nz_*, nz_scm_Construction, nz_scm_Other_Sectors

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

    # 3. Run SDID + penalized (bias-corrected) SCM robustness analysis.
    print("Running SDID / bias-corrected SCM robustness script...")
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "sdid_bias_corrected_analysis.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )

    # 4. Execute notebooks (nbconvert --execute runs from notebook's directory)
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
                "jupyter",
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

    print("Done. Figures saved to article_assets/")


if __name__ == "__main__":
    main()

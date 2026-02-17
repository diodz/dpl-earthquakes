"""
Maule (Chile) outcome decomposition: GDP per capita only.

Runs SCM with the same donor pool and pre-treatment setup as the baseline Maule analysis,
with dependent variable gdp_cap (GDP per capita). Regional population (and thus total GDP)
in our Chilean source (scm_chile_2010.xlsx) are available only for a short year window
(about 7 years), so we do not run population or total_gdp outcomes for Maule; those
would produce misleading paths/gaps. The GDP-per-capita decomposition alone addresses
Comment 3.1 by showing that the Maule null result holds for the main Y/L outcome.
"""
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from pysyncon import Dataprep, Synth

import process_chile_gdp_data as pcd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

CHILE_TREATED = "VII Del Maule"
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

CHILE_PREDICTORS = [
    "agropecuario",
    "pesca",
    "mineria",
    "industria_m",
    "electricidad",
    "construccion",
    "comercio",
    "transporte",
    "servicios_financieros",
    "vivienda",
    "personales",
    "publica",
]

# Only GDP per capita: regional population (and thus total_gdp) are not available
# over the full SCM window in the Chile source, so we do not run those outcomes.
OUTCOME_SPECS: Dict[str, Dict[str, object]] = {
    "gdp_per_capita": {
        "column": "gdp_cap",
        "label": "GDP per Capita (CLP)",
        "scale": 1.0,
        "path_figure": "chile_maule_gdp_paths.png",
        "gap_figure": "chile_maule_gdp_gap.png",
    },
}

TREATMENT_YEAR = 2010
TIME_OPTIMIZE = range(1990, 2009)
YEARS_OUTCOME = range(1990, 2020)


def _build_dataprep(df: pd.DataFrame, dependent: str) -> Dataprep:
    return Dataprep(
        foo=df,
        predictors=CHILE_PREDICTORS,
        predictors_op="mean",
        time_predictors_prior=range(2005, 2009),
        special_predictors=[
            (dependent, range(2005, 2009), "mean"),
            ("ed_superior_cap", range(2008, 2009), "mean"),
        ],
        dependent=dependent,
        unit_variable="region_name",
        time_variable="year",
        treatment_identifier=CHILE_TREATED,
        controls_identifier=CHILE_CONTROLS,
        time_optimize_ssr=TIME_OPTIMIZE,
    )


def _plot_paths(
    series_df: pd.DataFrame,
    label: str,
    scale: float,
    treatment_time: int,
    filename: str,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(series_df["Treated"] / scale, color="blue", linewidth=1.5, label="Maule")
    plt.plot(
        series_df["Synthetic"] / scale,
        color="blue",
        linewidth=1,
        linestyle="dashed",
        label="Synthetic Control",
    )
    plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()


def _plot_gap(series_df: pd.DataFrame, treatment_time: int, filename: str) -> None:
    gap_pct = (series_df["Treated"] - series_df["Synthetic"]) / series_df["Synthetic"] * 100
    plt.figure(figsize=(8, 5))
    plt.plot(gap_pct, color="blue", linewidth=1.5, label="Gap (%)")
    plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.axhline(y=0, color="black")
    plt.ylabel("Gap (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()


def run_chile_outcome_extensions() -> Dict[str, pd.DataFrame]:
    df = pcd.process_data_for_synth()

    output_tables: Dict[str, pd.DataFrame] = {}

    for outcome_name, spec in OUTCOME_SPECS.items():
        dependent_col = spec["column"]
        dataprep = _build_dataprep(df, dependent=dependent_col)
        synth = Synth()
        synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="ols")

        Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=YEARS_OUTCOME)
        synthetic = synth._synthetic(Z0=Z0)
        out = pd.concat([Z1.rename("Treated"), synthetic.rename("Synthetic")], axis=1)
        out["Absolute Gap"] = out["Treated"] - out["Synthetic"]
        out["Percent Gap"] = out["Absolute Gap"] / out["Synthetic"] * 100
        output_tables[outcome_name] = out

        _plot_paths(
            out,
            label=spec["label"],
            scale=spec["scale"],
            treatment_time=TREATMENT_YEAR,
            filename=spec["path_figure"],
        )
        _plot_gap(out, treatment_time=TREATMENT_YEAR, filename=spec["gap_figure"])

    with pd.ExcelWriter(
        os.path.join(FIGURES_DIR, "chile_outcome_extension_tables.xlsx"),
        engine="xlsxwriter",
    ) as writer:
        for sheet, table in output_tables.items():
            table.to_excel(writer, sheet_name=sheet)

    summary_rows = []
    for name, table in output_tables.items():
        post = table[table.index >= TREATMENT_YEAR]
        summary_rows.append(
            {
                "outcome": name,
                "avg_post_gap_percent": post["Percent Gap"].mean(),
                "max_post_gap_percent": post["Percent Gap"].max(),
                "gap_2016_percent": table.loc[2016, "Percent Gap"] if 2016 in table.index else None,
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(FIGURES_DIR, "chile_outcome_extension_summary.csv"),
        index=False,
    )
    return output_tables


if __name__ == "__main__":
    run_chile_outcome_extensions()

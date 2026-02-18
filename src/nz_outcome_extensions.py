import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from pysyncon import Dataprep, Synth

import nz_util

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(_PROJECT_ROOT, "article_assets")
os.makedirs(FIGURES_DIR, exist_ok=True)

CONTROLS_IDENTIFIER = [
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

POST_TREATMENT_START_YEAR = 2011
POST_TREATMENT_END_YEAR = 2019

OUTCOME_SPECS: Dict[str, Dict[str, object]] = {
    "gdp_per_capita": {
        "column": "gdp_per_capita",
        "label": "GDP per Capita (Thousands of NZD)",
        "scale": 1000,
        "path_figure": "nz_gdp_paths.png",
        "gap_figure": "nz_gap.png",
    },
    "population": {
        "column": "total_population",
        "label": "Population (Thousands)",
        "scale": 1000,
        "path_figure": "nz_population_paths.png",
        "gap_figure": "nz_population_gap.png",
    },
    "total_gdp": {
        "column": "total_gdp",
        "label": "Gross Domestic Product (Millions of NZD)",
        "scale": 1000,
        "path_figure": "nz_total_gdp_paths.png",
        "gap_figure": "nz_total_gdp_gap.png",
    },
}


def _build_dataprep(df: pd.DataFrame, dependent: str) -> Dataprep:
    return Dataprep(
        foo=df,
        predictors=nz_util.SECTORIAL_GDP_VARIABLES,
        predictors_op="mean",
        time_predictors_prior=range(2005, 2009),
        special_predictors=[
            (dependent, range(2005, 2009), "mean"),
            ("Tertiary Share", range(2008, 2009), "mean"),
        ],
        dependent=dependent,
        unit_variable="Region",
        time_variable="Year",
        treatment_identifier="Canterbury",
        controls_identifier=CONTROLS_IDENTIFIER,
        time_optimize_ssr=range(2000, 2009),
    )


def _plot_paths(series_df: pd.DataFrame, label: str, scale: float, treatment_time: int, filename: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(series_df["Treated"] / scale, color="red", linewidth=1.5, label="Canterbury")
    plt.plot(
        series_df["Synthetic"] / scale,
        color="red",
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
    plt.plot(gap_pct, color="red", linewidth=1.5, label="Gap (%)")
    plt.axvline(x=treatment_time, color="black", ymin=0.05, ymax=0.95, linestyle="dashed")
    plt.axhline(y=0, color="black")
    plt.ylabel("Gap (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename))
    plt.close()


def run_nz_outcome_extensions() -> Dict[str, pd.DataFrame]:
    df = nz_util.clean_data_for_synthetic_control().copy()
    df["Tertiary Share"] = df["Tertiary"] / df["total_population"]

    output_tables: Dict[str, pd.DataFrame] = {}

    for outcome_name, spec in OUTCOME_SPECS.items():
        dependent_col = spec["column"]
        dataprep = _build_dataprep(df, dependent=dependent_col)
        synth = Synth()
        synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="equal")

        Z0, Z1 = synth.dataprep.make_outcome_mats(time_period=range(2000, 2023))
        synthetic = synth._synthetic(Z0=Z0)
        out = pd.concat([Z1.rename("Treated"), synthetic.rename("Synthetic")], axis=1)
        out["Absolute Gap"] = out["Treated"] - out["Synthetic"]
        out["Percent Gap"] = out["Absolute Gap"] / out["Synthetic"] * 100
        output_tables[outcome_name] = out

        _plot_paths(
            out,
            label=spec["label"],
            scale=spec["scale"],
            treatment_time=2010,
            filename=spec["path_figure"],
        )
        _plot_gap(out, treatment_time=2010, filename=spec["gap_figure"])

    with pd.ExcelWriter(os.path.join(FIGURES_DIR, "nz_outcome_extension_tables.xlsx"), engine="xlsxwriter") as writer:
        for sheet, table in output_tables.items():
            table.to_excel(writer, sheet_name=sheet)

    summary_rows: List[dict] = []
    for name, table in output_tables.items():
        post = table[
            (table.index >= POST_TREATMENT_START_YEAR)
            & (table.index <= POST_TREATMENT_END_YEAR)
        ]
        summary_rows.append(
            {
                "outcome": name,
                "avg_post_gap_percent": post["Percent Gap"].mean(),
                "max_post_gap_percent": post["Percent Gap"].max(),
                "gap_2016_percent": table.loc[2016, "Percent Gap"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(os.path.join(FIGURES_DIR, "nz_outcome_extension_summary.csv"), index=False)
    return output_tables


if __name__ == "__main__":
    run_nz_outcome_extensions()

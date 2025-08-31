# scripts/plots.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def save_lineplot(series_dict, title, xlabel, ylabel, outpath: Path):
    plt.figure()
    for label, s in series_dict.items():
        s.sort_index().plot(label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.savefig(outpath.with_suffix(".png"), dpi=200)
    plt.close()

def save_gapplot(gap: pd.Series, event_year: int, title: str, outpath: Path):
    plt.figure()
    gap.sort_index().plot()
    plt.axvline(event_year, linestyle="--")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Actual - Synthetic (per capita)")
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.savefig(outpath.with_suffix(".png"), dpi=200)
    plt.close()

def save_placebo_plot(placebo_df: pd.DataFrame, treated: str, event_year: int, outpath: Path):
    plt.figure()
    for _, row in placebo_df.iterrows():
        g = row["gaps"].copy()
        if row["unit"] == treated:
            g.plot(linewidth=2.5, label=f"{treated} (treated)")
        else:
            g.plot(alpha=0.5)
    plt.axvline(event_year, linestyle="--")
    plt.title("In-space placebo gaps")
    plt.xlabel("Year")
    plt.ylabel("Actual - Synthetic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.savefig(outpath.with_suffix(".png"), dpi=200)
    plt.close()

def save_gap_with_band(gap: pd.Series, band_low: pd.Series, band_high: pd.Series, event_year: int, title: str, outpath: Path):
    plt.figure()
    gap = gap.sort_index()
    band_low = band_low.reindex(gap.index)
    band_high = band_high.reindex(gap.index)
    plt.plot(gap.index, gap.values, label="Gap")
    plt.fill_between(gap.index, band_low.values, band_high.values, alpha=0.2, label="Placebo 90% band")
    plt.axvline(event_year, linestyle="--")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Actual - Synthetic (per capita)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.savefig(outpath.with_suffix(".png"), dpi=200)
    plt.close()

def save_loo_plot(gaps_dict: dict, event_year: int, title: str, outpath: Path):
    plt.figure()
    # First item assumed to be 'base'
    for label, series in gaps_dict.items():
        series = series.sort_index()
        if label == "base":
            plt.plot(series.index, series.values, linewidth=2.5, label="Baseline")
        else:
            plt.plot(series.index, series.values, alpha=0.6, label=label)
    plt.axvline(event_year, linestyle="--")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Actual - Synthetic (per capita)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath.with_suffix(".pdf"))
    plt.savefig(outpath.with_suffix(".png"), dpi=200)
    plt.close()

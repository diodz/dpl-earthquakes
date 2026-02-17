import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from pysyncon import Dataprep, Synth
from rasterio.mask import mask
from shapely import affinity
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union
import rasterio


_PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = _PROJECT_ROOT / "data" / "ntl"
_BOUNDARY_DIR = _DATA_DIR / "boundaries"
_RASTER_DIR = _DATA_DIR / "rasters"
_FIGURES_DIR = _PROJECT_ROOT / "article_assets"

for _path in (_BOUNDARY_DIR, _RASTER_DIR, _FIGURES_DIR):
    _path.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2000, 2020))

CHILE_TREATED = "VII Del Maule"
CHILE_CONTROLS = [
    "I De Tarapaca",
    "II De Antofagasta",
    "III De Atacama",
    "IV De Coquimbo",
    "V De Valparaiso",
    "RMS Region Metropolitana de Santiago",
    "VI Del Libertador General Bernardo OHiggins",
    "IX De La Araucania",
    "X De Los Lagos",
    "XI Aysen del General Carlos Ibanez del Campo",
    "XII De Magallanes y de la Antartica Chilena",
]

NZ_TREATED = "Canterbury"
NZ_CONTROLS = [
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

CHILE_BOUNDARY_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/"
    "gbOpen/CHL/ADM1/geoBoundaries-CHL-ADM1_simplified.geojson"
)
NZ_BOUNDARY_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/"
    "gbOpen/NZL/ADM1/geoBoundaries-NZL-ADM1_simplified.geojson"
)


@dataclass
class ProductSpec:
    name: str
    zenodo_record_id: str
    output_subdir: str
    key_selector: Callable[[list[str], int], str]


def _select_pcnl_key(keys: list[str], year: int) -> str:
    target = f"PCNL{year}.tif"
    if target not in keys:
        raise KeyError(f"{target} missing in PCNL record.")
    return target


def _select_viirs_extrapolated_key(keys: list[str], year: int) -> str:
    pattern = re.compile(
        rf"^nightlights\.average_viirs\..*_s_{year}0101_{year}1231_.*\.tif$"
    )
    matches = sorted([key for key in keys if pattern.match(key)])
    if not matches:
        raise KeyError(f"No VIIRS-extrapolated file found for year {year}.")
    # Keep the latest-version key if multiple versions are present.
    return matches[-1]


PRODUCTS = {
    "pcnl_harmonized": ProductSpec(
        name="pcnl_harmonized",
        zenodo_record_id="17013414",
        output_subdir="pcnl_harmonized",
        key_selector=_select_pcnl_key,
    ),
    "viirs_extrapolated": ProductSpec(
        name="viirs_extrapolated",
        zenodo_record_id="17294744",
        output_subdir="viirs_extrapolated",
        key_selector=_select_viirs_extrapolated_key,
    ),
}


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower()
    normalized = normalized.replace("'", "")
    normalized = normalized.replace(".", " ")
    normalized = normalized.replace("-", " ")
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _fix_mojibake(value: str) -> str:
    try:
        return value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value


def _download_file(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    # Download to a temporary file first to avoid caching partial downloads
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    try:
        with temp_destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        # Atomically move the completed download to the final destination
        temp_destination.rename(destination)
    except BaseException:
        # Clean up partial temp file on any error
        if temp_destination.exists():
            temp_destination.unlink()
        raise


def _fetch_zenodo_file_map(record_id: str) -> dict[str, str]:
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=120)
    response.raise_for_status()
    payload = response.json()
    return {item["key"]: item["links"]["self"] for item in payload.get("files", [])}


def _resolve_yearly_rasters(spec: ProductSpec, years: list[int]) -> dict[int, Path]:
    target_dir = _RASTER_DIR / spec.output_subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    key_to_url = _fetch_zenodo_file_map(spec.zenodo_record_id)
    all_keys = list(key_to_url.keys())
    year_to_path: dict[int, Path] = {}

    for year in years:
        key = spec.key_selector(all_keys, year)
        destination = target_dir / key
        _download_file(key_to_url[key], destination, timeout=600)
        year_to_path[year] = destination
    return year_to_path


def _load_geojson(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_chile_geometries(chile_geojson: dict) -> dict[str, object]:
    source_geoms: dict[str, object] = {}
    for feature in chile_geojson["features"]:
        raw_name = _fix_mojibake(feature["properties"]["shapeName"])
        key = _normalize_text(raw_name)
        source_geoms[key] = shape(feature["geometry"])

    target_to_sources = {
        "I De Tarapaca": [
            "region de arica y parinacota",
            "region de tarapaca",
        ],
        "II De Antofagasta": ["region de antofagasta"],
        "III De Atacama": ["region de atacama"],
        "IV De Coquimbo": ["region de coquimbo"],
        "V De Valparaiso": ["region de valparaiso"],
        "RMS Region Metropolitana de Santiago": ["region metropolitana de santiago"],
        "VI Del Libertador General Bernardo OHiggins": [
            "region del libertador bernardo ohiggins"
        ],
        "VII Del Maule": ["region del maule"],
        "VIII Del Biobio": [
            "region del bio bio",
            "region de nuble",
        ],
        "IX De La Araucania": ["region de la araucania"],
        "X De Los Lagos": [
            "region de los lagos",
            "region de los rios",
        ],
        "XI Aysen del General Carlos Ibanez del Campo": [
            "region de aysen del gral ibanez del campo"
        ],
        "XII De Magallanes y de la Antartica Chilena": [
            "region de magallanes y antartica chilena"
        ],
    }

    missing_keys = [
        source
        for source_list in target_to_sources.values()
        for source in source_list
        if source not in source_geoms
    ]
    if missing_keys:
        raise KeyError(f"Missing Chile boundary keys: {missing_keys}")

    output: dict[str, object] = {}
    for target, source_list in target_to_sources.items():
        output[target] = unary_union([source_geoms[source] for source in source_list])
    return output


def _build_nz_geometries(nz_geojson: dict) -> dict[str, object]:
    source_geoms: dict[str, object] = {}
    for feature in nz_geojson["features"]:
        key = _normalize_text(feature["properties"]["shapeName"])
        source_geoms[key] = shape(feature["geometry"])

    target_to_sources = {
        "Northland": ["northland region"],
        "Auckland": ["auckland region"],
        "Waikato": ["waikato region"],
        "Bay of Plenty": ["bay of plenty region"],
        "Gisborne": ["gisborne region"],
        "Hawke's Bay": ["hawkes bay region"],
        "Taranaki": ["taranaki region"],
        "Manawatu-Whanganui": ["manawatu wanganui region"],
        "Wellington": ["wellington region"],
        "Tasman/Nelson": ["tasman region", "nelson region"],
        "Marlborough": ["marlborough region"],
        "West Coast": ["west coast region"],
        "Canterbury": ["canterbury region"],
        "Otago": ["otago region"],
        "Southland": ["southland region"],
    }

    missing_keys = [
        source
        for source_list in target_to_sources.values()
        for source in source_list
        if source not in source_geoms
    ]
    if missing_keys:
        raise KeyError(f"Missing New Zealand boundary keys: {missing_keys}")

    output: dict[str, object] = {}
    for target, source_list in target_to_sources.items():
        output[target] = unary_union([source_geoms[source] for source in source_list])
    return output


def _extract_values(dataset: rasterio.DatasetReader, geom: object) -> np.ndarray:
    raster, _ = mask(dataset, [mapping(geom)], crop=True, filled=False)
    values = raster[0].compressed().astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    if dataset.nodata is not None:
        values = values[values != float(dataset.nodata)]
    values = np.clip(values, a_min=0.0, a_max=None)
    return values


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "ntl_mean": np.nan,
            "ntl_median": np.nan,
            "ntl_p90": np.nan,
            "lit_share": np.nan,
        }
    return {
        "ntl_mean": float(np.mean(values)),
        "ntl_median": float(np.median(values)),
        "ntl_p90": float(np.quantile(values, 0.9)),
        "lit_share": float(np.mean(values > 0)),
    }


def _build_regional_panel(
    product_name: str,
    year_to_path: dict[int, Path],
    chile_geoms: dict[str, object],
    nz_geoms: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict] = []
    for year in sorted(year_to_path):
        raster_path = year_to_path[year]
        print(f"Processing {product_name} {year}: {raster_path.name}")
        with rasterio.open(raster_path) as dataset:
            for region, geom in chile_geoms.items():
                stats = _summarize_values(_extract_values(dataset, geom))
                rows.append(
                    {
                        "product": product_name,
                        "country": "Chile",
                        "unit": region,
                        "year": year,
                        **stats,
                    }
                )

            for region, geom in nz_geoms.items():
                stats = _summarize_values(_extract_values(dataset, geom))
                rows.append(
                    {
                        "product": product_name,
                        "country": "New Zealand",
                        "unit": region,
                        "year": year,
                        **stats,
                    }
                )
    panel = pd.DataFrame(rows)
    panel["ntl_log_mean"] = np.log1p(panel["ntl_mean"])
    return panel


@dataclass
class SCMOutput:
    summary_row: dict
    path_df: pd.DataFrame
    weights: pd.Series


def _run_ntl_scm(
    panel: pd.DataFrame,
    country: str,
    treated: str,
    controls: list[str],
    treatment_year: int,
    start_year: int,
    end_year: int,
    product_name: str,
) -> SCMOutput:
    sample = panel[
        (panel["country"] == country)
        & (panel["unit"].isin([treated] + controls))
        & (panel["year"] >= start_year)
        & (panel["year"] <= end_year)
    ].copy()

    pre_years = list(range(start_year, treatment_year))
    years = list(range(start_year, end_year + 1))
    special_years = sorted({start_year, max(start_year + 4, start_year), treatment_year - 1})
    special_predictors = [("ntl_log_mean", [year], "mean") for year in special_years]

    dataprep = Dataprep(
        foo=sample,
        predictors=["ntl_log_mean"],
        predictors_op="mean",
        time_predictors_prior=pre_years,
        special_predictors=special_predictors,
        dependent="ntl_log_mean",
        unit_variable="unit",
        time_variable="year",
        treatment_identifier=treated,
        controls_identifier=controls,
        time_optimize_ssr=pre_years,
    )

    synth = Synth()
    synth.fit(dataprep=dataprep, optim_method="Nelder-Mead", optim_initial="equal")

    z0, z1 = synth.dataprep.make_outcome_mats(time_period=years)
    treated_log = z1.astype(float)
    synthetic_log = synth._synthetic(z0).astype(float)
    treated_level = np.expm1(treated_log.to_numpy())
    synthetic_level = np.expm1(synthetic_log.to_numpy())

    denominator = np.where(synthetic_level > 1e-9, synthetic_level, np.nan)
    gap_pct = (treated_level - synthetic_level) / denominator * 100.0

    path_df = pd.DataFrame(
        {
            "product": product_name,
            "country": country,
            "year": years,
            "treated_level": treated_level,
            "synthetic_level": synthetic_level,
            "gap_pct": gap_pct,
        }
    )

    post = path_df[path_df["year"] >= treatment_year]
    pre = path_df[path_df["year"] < treatment_year]

    summary_row = {
        "product": product_name,
        "country": country,
        "treated_unit": treated,
        "avg_post_gap_pct": float(post["gap_pct"].mean()),
        "max_post_gap_pct": float(post["gap_pct"].max()),
        "gap_2016_pct": float(path_df.loc[path_df["year"] == 2016, "gap_pct"].iloc[0]),
        "pre_rmspe_pct": float(np.sqrt(np.nanmean(np.square(pre["gap_pct"])))),
        "post_rmspe_pct": float(np.sqrt(np.nanmean(np.square(post["gap_pct"])))),
    }

    weights = synth.weights(round=10, threshold=None).rename("weight")
    return SCMOutput(summary_row=summary_row, path_df=path_df, weights=weights)


def _km_buffer_ellipse(lon: float, lat: float, radius_km: float) -> object:
    point = Point(lon, lat)
    unit_circle = point.buffer(1.0, resolution=128)
    lat_scale = radius_km / 111.32
    lon_scale = radius_km / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return affinity.scale(unit_circle, xfact=lon_scale, yfact=lat_scale, origin=point)


def _compute_maule_urban_series(
    year_to_path: dict[int, Path],
    maule_geom: object,
    pre_years: list[int],
    quantile_threshold: float = 0.8,
) -> pd.DataFrame:
    pre_values: list[np.ndarray] = []
    for year in pre_years:
        with rasterio.open(year_to_path[year]) as dataset:
            values = _extract_values(dataset, maule_geom)
            positive = values[values > 0]
            if positive.size:
                pre_values.append(positive)
    if not pre_values:
        raise ValueError("Could not compute Maule urban threshold: no positive pre-period values.")

    threshold = float(np.quantile(np.concatenate(pre_values), quantile_threshold))

    rows: list[dict] = []
    for year in sorted(year_to_path):
        with rasterio.open(year_to_path[year]) as dataset:
            values = _extract_values(dataset, maule_geom)
            full_mean = float(np.mean(values)) if values.size else np.nan
            urban_values = values[values >= threshold]
            urban_mean = float(np.mean(urban_values)) if urban_values.size else np.nan
            rows.append(
                {
                    "year": year,
                    "maule_full_mean": full_mean,
                    "maule_urban_core_mean": urban_mean,
                    "urban_threshold": threshold,
                }
            )
    return pd.DataFrame(rows)


def _compute_christchurch_buffer_series(
    year_to_path: dict[int, Path],
    canterbury_geom: object,
    radii_km: list[int],
    center_lon: float = 172.6362,
    center_lat: float = -43.5321,
) -> pd.DataFrame:
    rows: list[dict] = []
    for radius in radii_km:
        buffer_geom = _km_buffer_ellipse(center_lon, center_lat, radius)
        clipped = buffer_geom.intersection(canterbury_geom)
        for year in sorted(year_to_path):
            with rasterio.open(year_to_path[year]) as dataset:
                values = _extract_values(dataset, clipped)
                rows.append(
                    {
                        "year": year,
                        "radius_km": radius,
                        "ntl_mean": float(np.mean(values)) if values.size else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    base = out[out["year"] == 2010][["radius_km", "ntl_mean"]].rename(columns={"ntl_mean": "base_2010"})
    out = out.merge(base, on="radius_km", how="left")
    out["index_2010_100"] = 100.0 * out["ntl_mean"] / out["base_2010"]
    return out


def _plot_main_validation(path_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    countries = ["Chile", "New Zealand"]
    treatment_year = {"Chile": 2010, "New Zealand": 2011}
    labels = {"Chile": "Maule", "New Zealand": "Canterbury"}

    for col, country in enumerate(countries):
        country_df = path_df[(path_df["country"] == country) & (path_df["product"] == "pcnl_harmonized")]
        base = float(country_df.loc[country_df["year"] == treatment_year[country] - 1, "treated_level"].iloc[0])
        treated_idx = 100.0 * country_df["treated_level"] / base
        synth_idx = 100.0 * country_df["synthetic_level"] / base

        axes[0, col].plot(country_df["year"], treated_idx, color="#d62728", linewidth=1.8, label=labels[country])
        axes[0, col].plot(
            country_df["year"],
            synth_idx,
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
            label="Synthetic",
        )
        axes[0, col].axvline(treatment_year[country], color="black", linestyle="--", linewidth=1.0)
        axes[0, col].set_title(country)
        axes[0, col].grid(alpha=0.2)

        axes[1, col].plot(country_df["year"], country_df["gap_pct"], color="#1f77b4", linewidth=1.8)
        axes[1, col].axhline(0.0, color="black", linewidth=0.8)
        axes[1, col].axvline(treatment_year[country], color="black", linestyle="--", linewidth=1.0)
        axes[1, col].grid(alpha=0.2)
        axes[1, col].set_xlabel("Year")

    axes[0, 0].set_ylabel("NTL index (pre-treatment=100)")
    axes[1, 0].set_ylabel("Gap (%)")
    handles, labels_legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_sensor_robustness(path_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    product_labels = {
        "pcnl_harmonized": "PCNL (DMSP+VIIRS harmonized)",
        "viirs_extrapolated": "VIIRS-extrapolated alternative",
    }
    color_map = {"pcnl_harmonized": "#1f77b4", "viirs_extrapolated": "#ff7f0e"}
    countries = ["Chile", "New Zealand"]
    treatment_year = {"Chile": 2010, "New Zealand": 2011}

    for axis, country in zip(axes, countries, strict=True):
        for product in ("pcnl_harmonized", "viirs_extrapolated"):
            line = path_df[(path_df["country"] == country) & (path_df["product"] == product)]
            axis.plot(
                line["year"],
                line["gap_pct"],
                label=product_labels[product],
                color=color_map[product],
                linewidth=1.8,
            )
        axis.axvline(treatment_year[country], color="black", linestyle="--", linewidth=1.0)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.grid(alpha=0.2)
        axis.set_title(country)
        axis.set_xlabel("Year")

    axes[0].set_ylabel("Gap (%)")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_spatial_sensitivity(
    maule_series: pd.DataFrame,
    canterbury_buffer_series: pd.DataFrame,
    maule_synthetic: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)

    maule_full_base = float(maule_series.loc[maule_series["year"] == 2009, "maule_full_mean"].iloc[0])
    maule_urban_base = float(maule_series.loc[maule_series["year"] == 2009, "maule_urban_core_mean"].iloc[0])
    synth_base = float(maule_synthetic.loc[maule_synthetic["year"] == 2009, "synthetic_level"].iloc[0])

    axes[0].plot(
        maule_series["year"],
        100.0 * maule_series["maule_full_mean"] / maule_full_base,
        label="Maule full region",
        color="#1f77b4",
    )
    axes[0].plot(
        maule_series["year"],
        100.0 * maule_series["maule_urban_core_mean"] / maule_urban_base,
        label="Maule urban-core mask",
        color="#2ca02c",
    )
    axes[0].plot(
        maule_synthetic["year"],
        100.0 * maule_synthetic["synthetic_level"] / synth_base,
        label="Synthetic Maule",
        color="#d62728",
        linestyle="--",
    )
    axes[0].axvline(2010, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_title("Chile: urban-mask sensitivity (Maule)")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Index (2009=100)")
    axes[0].grid(alpha=0.2)

    for radius, group in canterbury_buffer_series.groupby("radius_km"):
        axes[1].plot(
            group["year"],
            group["index_2010_100"],
            label=f"{radius} km buffer",
            linewidth=1.7,
        )
    axes[1].axvline(2011, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("New Zealand: Christchurch buffer sensitivity")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Index (2010=100)")
    axes[1].grid(alpha=0.2)

    handles_left, labels_left = axes[0].get_legend_handles_labels()
    handles_right, labels_right = axes[1].get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_nighttime_lights_validation(output_dir: Path = _FIGURES_DIR) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    chile_boundary_path = _BOUNDARY_DIR / "chile_adm1_simplified.geojson"
    nz_boundary_path = _BOUNDARY_DIR / "nz_adm1_simplified.geojson"
    _download_file(CHILE_BOUNDARY_URL, chile_boundary_path, timeout=180)
    _download_file(NZ_BOUNDARY_URL, nz_boundary_path, timeout=180)

    chile_geojson = _load_geojson(chile_boundary_path)
    nz_geojson = _load_geojson(nz_boundary_path)
    chile_geoms = _build_chile_geometries(chile_geojson)
    nz_geoms = _build_nz_geometries(nz_geojson)

    all_panels: list[pd.DataFrame] = []
    all_paths: list[pd.DataFrame] = []
    all_weights_rows: list[dict] = []
    all_summaries: list[dict] = []

    product_rasters: dict[str, dict[int, Path]] = {}
    for product_name, spec in PRODUCTS.items():
        print(f"Resolving rasters for {product_name}")
        product_rasters[product_name] = _resolve_yearly_rasters(spec, YEARS)
        panel = _build_regional_panel(
            product_name=product_name,
            year_to_path=product_rasters[product_name],
            chile_geoms=chile_geoms,
            nz_geoms=nz_geoms,
        )
        all_panels.append(panel)

        chile_scm = _run_ntl_scm(
            panel=panel,
            country="Chile",
            treated=CHILE_TREATED,
            controls=CHILE_CONTROLS,
            treatment_year=2010,
            start_year=2000,
            end_year=2019,
            product_name=product_name,
        )
        nz_scm = _run_ntl_scm(
            panel=panel,
            country="New Zealand",
            treated=NZ_TREATED,
            controls=NZ_CONTROLS,
            treatment_year=2011,
            start_year=2000,
            end_year=2019,
            product_name=product_name,
        )

        for country, scm_output in (("Chile", chile_scm), ("New Zealand", nz_scm)):
            all_paths.append(scm_output.path_df)
            all_summaries.append(scm_output.summary_row)
            for donor, weight in scm_output.weights.items():
                all_weights_rows.append(
                    {
                        "product": product_name,
                        "country": country,
                        "donor_unit": donor,
                        "weight": float(weight),
                    }
                )

    panel_df = pd.concat(all_panels, ignore_index=True)
    path_df = pd.concat(all_paths, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    weights_df = pd.DataFrame(all_weights_rows)

    baseline_summary = summary_df[summary_df["product"] == "pcnl_harmonized"].copy()
    baseline_summary["gdp_inference_alignment"] = baseline_summary.apply(
        lambda row: (
            "confirm"
            if (row["country"] == "Chile" and abs(row["avg_post_gap_pct"]) <= 2.0)
            or (
                row["country"] == "New Zealand"
                and row["avg_post_gap_pct"] > 0
                and row["gap_2016_pct"] > 0
            )
            else "diverge"
        ),
        axis=1,
    )
    def _interpretation_note(row: pd.Series) -> str:
        if row["country"] == "Chile":
            if row["gdp_inference_alignment"] == "confirm":
                return "Maule NTL gap remains close to zero (null-like)."
            return "Maule NTL gap is positive, diverging from GDP-null inference."
        if row["gdp_inference_alignment"] == "confirm":
            return "Canterbury NTL gap is positive in rebuild years."
        return "Canterbury NTL does not reproduce the GDP overshoot."

    baseline_summary["interpretation_note"] = baseline_summary.apply(_interpretation_note, axis=1)

    # Spatial robustness outputs based on the harmonized baseline product.
    baseline_rasters = product_rasters["pcnl_harmonized"]
    maule_series = _compute_maule_urban_series(
        year_to_path=baseline_rasters,
        maule_geom=chile_geoms[CHILE_TREATED],
        pre_years=list(range(2005, 2010)),
        quantile_threshold=0.8,
    )
    canterbury_buffer_series = _compute_christchurch_buffer_series(
        year_to_path=baseline_rasters,
        canterbury_geom=nz_geoms[NZ_TREATED],
        radii_km=[30, 50, 80],
    )
    maule_baseline_scm = path_df[
        (path_df["product"] == "pcnl_harmonized") & (path_df["country"] == "Chile")
    ].copy()

    _plot_main_validation(path_df, output_dir / "ntl_validation_paths_gaps.png")
    _plot_sensor_robustness(path_df, output_dir / "ntl_sensor_processing_robustness.png")
    _plot_spatial_sensitivity(
        maule_series=maule_series,
        canterbury_buffer_series=canterbury_buffer_series,
        maule_synthetic=maule_baseline_scm,
        output_path=output_dir / "ntl_spatial_sensitivity.png",
    )

    panel_df.to_csv(output_dir / "ntl_regional_panel.csv", index=False)
    path_df.to_csv(output_dir / "ntl_scm_gaps.csv", index=False)
    summary_df.to_csv(output_dir / "ntl_scm_summary.csv", index=False)
    baseline_summary.to_csv(output_dir / "ntl_validation_summary.csv", index=False)
    weights_df.to_csv(output_dir / "ntl_scm_weights.csv", index=False)
    maule_series.to_csv(output_dir / "ntl_maule_urban_mask_series.csv", index=False)
    canterbury_buffer_series.to_csv(output_dir / "ntl_nz_buffer_sensitivity.csv", index=False)

    return baseline_summary


if __name__ == "__main__":
    result = run_nighttime_lights_validation()
    print(result.to_string(index=False))

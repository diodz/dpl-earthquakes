# scripts/loaders.py
import pandas as pd
import numpy as np
from pathlib import Path

def load_chile_gdppc(chile_main: Path, chile_pop: Path | None = None):
    def _from_excel(path):
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = xls.parse(sheet)
            cols = [c.lower() for c in df.columns]
            df.columns = cols
            if "region" in cols and "year" in cols:
                if "gdp_per_capita" in cols:
                    tmp = df[["region","year","gdp_per_capita"]].dropna()
                    tmp["year"] = tmp["year"].astype(int)
                    piv = tmp.pivot(index="year", columns="region", values="gdp_per_capita").sort_index()
                    return piv
                if "gdp" in cols and "population" in cols:
                    tmp = df[["region","year","gdp","population"]].dropna()
                    tmp["year"] = tmp["year"].astype(int)
                    tmp["gdp_per_capita"] = tmp["gdp"]/tmp["population"]
                    piv = tmp.pivot(index="year", columns="region", values="gdp_per_capita").sort_index()
                    return piv
            if "year" in cols or "año" in cols:
                year_col = "year" if "year" in cols else "año"
                df.rename(columns={year_col:"year"}, inplace=True)
                if df["year"].dropna().empty: 
                    continue
                df["year"] = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
                region_cols = [c for c in df.columns if c != "year"]
                numeric_cols = df[region_cols].select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    piv = df.set_index("year")[numeric_cols]
                    return piv
        raise ValueError("Could not parse Chile Excel file")
    if chile_main.suffix.lower() in [".xlsx",".xls"] and chile_main.exists():
        return _from_excel(chile_main)
    elif chile_main.with_suffix(".xlsx").exists():
        return _from_excel(chile_main.with_suffix(".xlsx"))
    if chile_main.exists():
        df = pd.read_csv(chile_main)
        cols = [c.lower() for c in df.columns]
        df.columns = cols
        if all(c in cols for c in ["region","year"]) and ("gdp_per_capita" in cols or ("gdp" in cols and "population" in cols)):
            if "gdp_per_capita" in cols:
                tmp = df[["region","year","gdp_per_capita"]].dropna()
            else:
                tmp = df[["region","year","gdp","population"]].dropna()
                tmp["gdp_per_capita"] = tmp["gdp"]/tmp["population"]
            tmp["year"] = tmp["year"].astype(int)
            return tmp.pivot(index="year", columns="region", values="gdp_per_capita").sort_index()
    raise FileNotFoundError(f"Could not load Chile GDP per capita from {chile_main}")

def load_nz_gdppc(nz_path: Path):
    def _parse(df):
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols
        if "region" in cols and "year" in cols:
            candidates = [c for c in cols if "per" in c and "cap" in c]
            if not candidates:
                if "gdp" in cols and "population" in cols:
                    df["gdp_per_capita"] = df["gdp"] / df["population"]
                    candidates = ["gdp_per_capita"]
            if candidates:
                c = candidates[0]
                tmp = df[["region","year",c]].dropna().copy()
                tmp["year"] = pd.to_numeric(tmp["year"], errors="coerce").dropna().astype(int)
                piv = tmp.pivot(index="year", columns="region", values=c).sort_index()
                return piv
        if ("year" in cols) or ("march year" in cols) or ("year_ended_march" in cols):
            year_col = "year" if "year" in cols else ("march year" if "march year" in cols else "year_ended_march")
            df.rename(columns={year_col:"year"}, inplace=True)
            df["year"] = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
            region_cols = [c for c in df.columns if c != "year"]
            numeric_cols = df[region_cols].select_dtypes(include="number").columns.tolist()
            if numeric_cols:
                return df.set_index("year")[numeric_cols]
        raise ValueError("Could not parse NZ file")
    if nz_path.suffix.lower() in [".csv",".txt"] and nz_path.exists():
        df = pd.read_csv(nz_path)
        return _parse(df)
    if nz_path.with_suffix(".csv").exists():
        df = pd.read_csv(nz_path.with_suffix(".csv"))
        return _parse(df)
    if nz_path.suffix.lower() in [".xlsx",".xls"] and nz_path.exists():
        xls = pd.ExcelFile(nz_path)
        for s in xls.sheet_names:
            try:
                df = xls.parse(s)
                return _parse(df)
            except Exception:
                continue
    if nz_path.with_suffix(".xlsx").exists():
        xls = pd.ExcelFile(nz_path.with_suffix(".xlsx"))
        for s in xls.sheet_names:
            try:
                df = xls.parse(s)
                return _parse(df)
            except Exception:
                continue
    raise FileNotFoundError(f"Could not load NZ GDP per capita from {nz_path}")

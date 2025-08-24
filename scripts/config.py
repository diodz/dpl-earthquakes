# scripts/config.py
from pathlib import Path

# Base filenames (adjust if your paths differ)
CHILE_MAIN = Path("data/scm_chile_2010.xlsx")   # fallback: "scm_2010.csv"
CHILE_POP = Path("data/poblacion regional.xlsx")
CHILE_SECTORS = Path("data/pib sectorial y regional.xlsx")

NZ_MAIN = Path("..data/regional-gross-domestic-product-year-ended-march-2023.csv")  # fallback .xlsx

# Output folders
FIG_DIR = Path("earthquakes_pack/figures")
TAB_DIR = Path("earthquakes_pack/tables")
TEX_DIR = Path("earthquakes_pack/tex")

# Analysis config
CHILE_TREATED = "VII Del Maule"
CHILE_EVENT_YEAR = 2010
# Donors excluding Biobío, O'Higgins, and later large-quake regions (I, IV)
CHILE_DONORS = [
    "Antofagasta", "Atacama", "Valparaíso", "Metropolitana",
    "Los Lagos", "Aysén", "Magallanes"
]

NZ_TREATED = "Canterbury"
NZ_EVENT_YEAR = 2011
# Donors: all others except Marlborough (Kaikōura 2016 affected)
NZ_EXCLUDE = {"Marlborough", "Canterbury", "New Zealand", "North Island", "South Island"}
# If your file uses different region labels, the loader will infer candidates.

# run_all.py
from pathlib import Path
from scripts.run_chile import main as run_chile
from scripts.run_nz import main as run_nz
from scripts.run_placebos import main as run_placebos

if __name__ == "__main__":
    base = Path(".")
    run_chile(base)
    run_nz(base)
    run_placebos(base)
    print("All done. See earthquakes_pack/figures and earthquakes_pack/tables.")

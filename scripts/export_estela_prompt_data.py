#!/usr/bin/env python3
"""
Export the estela database dict to notebooks/estela_prompt_data.pkl.

notebooks/estela_prompt_data.pkl is the SOURCE OF TRUTH for all experiments.
It is committed to git and should only be updated deliberately by running this
script and committing the result.  Do NOT regenerate the pkl mid-experiment.

Loads data/estela/estela_db.py (which defines database_estela_dict and uses
datetime.date objects). The pickle preserves those native Python types and is
the format expected by load_estela_dict() across all sweep and analysis scripts.

To update the snapshot:
    python scripts/export_estela_prompt_data.py
    git add notebooks/estela_prompt_data.pkl
    git commit -m "data: update estela snapshot (<reason>)"
"""

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ESTELA_DB_PY = REPO_ROOT / "data" / "estela" / "estela_db.py"
OUT_PKL = REPO_ROOT / "notebooks" / "estela_prompt_data.pkl"


def main():
    if not ESTELA_DB_PY.exists():
        print(f"ERROR: {ESTELA_DB_PY} not found. Add estela_db.py to data/estela/ first.")
        sys.exit(1)

    # Run the estela_db.py module with datetime in scope (it uses datetime.date)
    import datetime as dt
    globals_ = runpy.run_path(str(ESTELA_DB_PY), init_globals={"datetime": dt})

    if "database_estela_dict" not in globals_:
        print("ERROR: database_estela_dict not found in", ESTELA_DB_PY)
        sys.exit(1)

    database_estela_dict = globals_["database_estela_dict"]
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(OUT_PKL, "wb") as f:
        pickle.dump(database_estela_dict, f)

    print(f"Exported {len(database_estela_dict)} entries to {OUT_PKL}")


if __name__ == "__main__":
    main()

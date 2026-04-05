"""
CSV scenario size and environment factory.

Edit NUM_HUBS / NUM_ZONES / NUM_TRUCKS here — used by main.py and visualization.py.
None (or 0) means load every row from that CSV.
"""

from __future__ import annotations

from pathlib import Path

from csv_loader import load_environment_from_csv

# ---------------------------------------------------------------------------
# First N data rows from each CSV (CSV order). None = load all rows.
# ---------------------------------------------------------------------------

NUM_HUBS: int | None = 10
NUM_ZONES: int | None = 20
NUM_TRUCKS: int | None = 3


def make_env(seed: int = 42):
    """Build the environment from CSVs using NUM_HUBS / NUM_ZONES / NUM_TRUCKS."""
    base = Path(__file__).resolve().parent
    return load_environment_from_csv(
        base / "hubs.csv",
        base / "zones.csv",
        base / "trucks.csv",
        seed=seed,
        max_hubs=NUM_HUBS,
        max_zones=NUM_ZONES,
        max_trucks=NUM_TRUCKS,
    )

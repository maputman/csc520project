"""
CSV scenario size and environment factory.

Edit NUM_HUBS / NUM_ZONES / NUM_TRUCKS here — used by main.py and visualization.py.
None (or 0) means load every row from that CSV.

CSV lookup order (first triple where all three files exist):

  1. ``csvdata/hubs.csv``, ``csvdata/zones.csv``, ``csvdata/trucks.csv``
  2. project root ``hubs.csv``, ``zones.csv``, ``trucks.csv``

If no CSV bundle is found, ``make_env()`` falls back to ``environment.build_scenario()``
(small built-in graph).
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


def _csv_triplet(base: Path) -> tuple[Path, Path, Path] | None:
    """Return (hubs, zones, trucks) paths if a complete set exists."""
    for sub in ("csvdata", "."):
        d = base / sub if sub != "." else base
        h, z, t = d / "hubs.csv", d / "zones.csv", d / "trucks.csv"
        if h.is_file() and z.is_file() and t.is_file():
            return h, z, t
    return None


def make_env(seed: int = 42):
    """
    Build the environment from CSVs using NUM_HUBS / NUM_ZONES / NUM_TRUCKS,
    or the built-in toy scenario if CSVs are missing.
    """
    base = Path(__file__).resolve().parent
    triple = _csv_triplet(base)
    if triple is not None:
        hubs_csv, zones_csv, trucks_csv = triple
        return load_environment_from_csv(
            hubs_csv,
            zones_csv,
            trucks_csv,
            seed=seed,
            max_hubs=NUM_HUBS,
            max_zones=NUM_ZONES,
            max_trucks=NUM_TRUCKS,
        )

    from environment import build_scenario

    return build_scenario()

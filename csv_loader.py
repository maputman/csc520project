"""
Load hubs, zones, and trucks from CSV files into a DisasterEnvironment.

Expected columns (see sample hubs.csv, zones.csv, trucks.csv in repo root):

  hubs:   Hub ID, X Coord, Y Coord, Water (units), Food (units), Medical (units)
  zones:  Zone ID, X Coord, Y Coord, Urgency (1-10), Water (units), ...
          Critical Resource (empty or water|food|medical)
  trucks: Truck ID, Home Hub, Current Location, Water on Board, Food on Board,
          Medical on Board, Capacity, Status

Roads: after hubs and zones are loaded, build_roads_by_proximity() adds a sparse
network; if that leaves disconnected components, bridge edges (Euclidean cost) are
added until the graph is connected. Use max_neighbors / max_distance on
load_environment_from_csv to tune the sparse layer, or sparse_roads=False for a full
distance-complete graph (small scenarios only).

Use `load_environment_from_csv(..., max_hubs=, max_zones=, max_trucks=)` to load
only the first N data rows from each file (CSV order).
"""

from __future__ import annotations

import csv
import math
import warnings
from pathlib import Path

from environment import DisasterEnvironment, build_roads_by_proximity


def _int_cell(value: str, default: int = 0) -> int:
    s = (value or "").strip()
    if not s:
        return default
    return int(float(s))


def _float_cell(value: str) -> float:
    s = (value or "").strip()
    if not s:
        return 0.0
    return float(s)


def _norm_key(name: str) -> str:
    # Strip BOM so Excel-exported "﻿Hub ID" still maps to hub_id
    return name.strip().strip("\ufeff").lower().replace(" ", "_")


def load_hubs_csv(
    path: str | Path,
    env: DisasterEnvironment,
    *,
    max_hubs: int | None = None,
) -> None:
    """Load hub rows in CSV order. If max_hubs is set, stop after that many data rows."""
    path = Path(path)
    loaded = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        keys = [_norm_key(h) for h in header]
        for row in reader:
            if max_hubs is not None and max_hubs > 0 and loaded >= max_hubs:
                break
            if not row or all(not c.strip() for c in row):
                continue
            cells = list(row) + [""] * max(0, len(keys) - len(row))
            d = dict(zip(keys, cells[: len(keys)]))
            hub_id = (d.get("hub_id") or "").strip()
            if not hub_id or hub_id.upper() == "TOTAL":
                continue
            x = _float_cell(d.get("x_coord", "0"))
            y = _float_cell(d.get("y_coord", "0"))
            inventory = {
                "water":   _int_cell(d.get("water_(units)", d.get("water", "0"))),
                "food":    _int_cell(d.get("food_(units)",  d.get("food",  "0"))),
                "medical": _int_cell(d.get("medical_(units)", d.get("medical", "0"))),
            }
            env.add_hub(hub_id, x=x, y=y, inventory=inventory)
            loaded += 1


def load_zones_csv(
    path: str | Path,
    env: DisasterEnvironment,
    *,
    max_zones: int | None = None,
) -> None:
    """Load zone rows in CSV order. If max_zones is set, stop after that many rows."""
    path = Path(path)
    loaded = 0
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        keys = [_norm_key(h) for h in header]
        for row in reader:
            if max_zones is not None and max_zones > 0 and loaded >= max_zones:
                break
            if not row or all(not c.strip() for c in row):
                continue
            cells = list(row) + [""] * max(0, len(keys) - len(row))
            d = dict(zip(keys, cells[: len(keys)]))
            zone_id = (d.get("zone_id") or "").strip()
            if not zone_id:
                continue
            x = _float_cell(d.get("x_coord", "0"))
            y = _float_cell(d.get("y_coord", "0"))
            urgency = _int_cell(d.get("urgency_(1-10)", d.get("urgency", "1")), 1)
            urgency = max(1, min(10, urgency))
            needs = {
                "water":   _int_cell(d.get("water_(units)", d.get("water", "0"))),
                "food":    _int_cell(d.get("food_(units)",  d.get("food",  "0"))),
                "medical": _int_cell(d.get("medical_(units)", d.get("medical", "0"))),
            }
            crit = (d.get("critical_resource") or "").strip().lower()
            critical = None if not crit else crit
            if critical is not None and critical not in ("water", "food", "medical"):
                critical = None
            env.add_zone(zone_id, x, y, urgency, needs, critical_resource=critical)
            loaded += 1


def load_trucks_csv(
    path: str | Path,
    env: DisasterEnvironment,
    *,
    max_trucks: int | None = None,
) -> None:
    """
    Load trucks in CSV order.

    If max_trucks is a positive N: take the first N trucks whose Home Hub exists among
    loaded hubs (skip earlier rows that reference hubs you did not load — no spam).
    If max_trucks is None/0: load every CSV row with a valid home hub.
    """
    path = Path(path)
    loaded = 0
    skipped_missing_home = 0
    loaded_hub_ids = set(env.hubs.keys())
    want_cap = max_trucks is not None and max_trucks > 0

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        keys = [_norm_key(h) for h in header]
        for row in reader:
            if want_cap and loaded >= max_trucks:
                break
            if not row or all(not c.strip() for c in row):
                continue
            cells = list(row) + [""] * max(0, len(keys) - len(row))
            d = dict(zip(keys, cells[: len(keys)]))

            tid = (d.get("truck_id") or "").strip()
            if not tid:
                continue

            home = (d.get("home_hub") or "").strip()
            loc = (d.get("current_location") or home).strip()
            cap = _int_cell(d.get("capacity", "5"), 5)
            if cap < 1:
                cap = 5

            if home not in loaded_hub_ids:
                skipped_missing_home += 1
                continue

            if loc not in env.graph.nodes:
                loc = home

            w = min(_int_cell(d.get("water_on_board", "0")), cap)
            fd = min(_int_cell(d.get("food_on_board", "0")), cap)
            m = min(_int_cell(d.get("medical_on_board", "0")), cap)
            cargo = {}
            if w:
                cargo["water"] = w
            if fd:
                cargo["food"] = fd
            if m:
                cargo["medical"] = m

            status = (d.get("status") or "idle").strip().lower()
            if status not in ("idle", "en_route", "delivering"):
                status = "idle"

            t = env.add_truck(tid, home_hub_id=home, capacity=cap)
            t.current_node = loc
            if cargo:
                t.cargo = cargo
            if status != "idle":
                t.status = status

            loaded += 1

    if want_cap and loaded < max_trucks:
        warnings.warn(
            f"Only loaded {loaded} truck(s), fewer than max_trucks={max_trucks}: "
            f"not enough CSV rows with a home hub among the {len(loaded_hub_ids)} loaded hubs.",
            UserWarning,
            stacklevel=2,
        )
    elif not want_cap and skipped_missing_home:
        warnings.warn(
            f"Skipped {skipped_missing_home} truck row(s) whose Home Hub is not among loaded hubs.",
            UserWarning,
            stacklevel=2,
        )


def add_roads_from_distances(env: DisasterEnvironment) -> None:
    """
    Connect every pair of graph nodes with an edge whose cost is their
    Euclidean distance. Produces a complete graph — fine for small scenarios
    but use build_roads_by_proximity() instead for 20+ nodes.
    """
    nodes = list(env.graph.nodes())
    pos = {
        n: (float(env.graph.nodes[n]["x"]), float(env.graph.nodes[n]["y"]))
        for n in nodes
    }
    for i, a in enumerate(nodes):
        xa, ya = pos[a]
        for b in nodes[i + 1:]:
            xb, yb = pos[b]
            dist = math.hypot(xa - xb, ya - yb)
            env.add_road(a, b, cost=round(dist, 2))


def load_environment_from_csv(
    hubs_csv: str | Path,
    zones_csv: str | Path,
    trucks_csv: str | Path,
    *,
    seed: int = 42,
    max_hubs:   int | None = None,
    max_zones:  int | None = None,
    max_trucks: int | None = None,
    sparse_roads: bool = True,
    max_neighbors: int = 3,
    max_distance: float = 12.0,
) -> DisasterEnvironment:
    """
    Build a DisasterEnvironment from the three entity CSVs.

    max_*       : if set to a positive N, only the first N data rows from that
                  file (CSV order). None / 0 / negative means load all rows.
    sparse_roads: if True (default), use build_roads_by_proximity() for a
                  realistic sparse network. Set False to get the old complete
                  graph (every node connected to every other node) — only
                  suitable for very small scenarios.
    max_neighbors / max_distance: tuning knobs for the sparse road builder.
    """
    env = DisasterEnvironment(seed=seed)
    load_hubs_csv(hubs_csv,   env, max_hubs=max_hubs)
    load_zones_csv(zones_csv, env, max_zones=max_zones)
    load_trucks_csv(trucks_csv, env, max_trucks=max_trucks)

    if sparse_roads:
        build_roads_by_proximity(env, max_neighbors=max_neighbors,
                                 max_distance=max_distance)
    else:
        add_roads_from_distances(env)

    return env


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    env = load_environment_from_csv(
        base / "hubs.csv",
        base / "zones.csv",
        base / "trucks.csv",
    )
    print(f"Loaded {len(env.hubs)} hubs, {len(env.zones)} zones, {len(env.trucks)} trucks")
    print(f"Graph: {env.graph.number_of_nodes()} nodes, {env.graph.number_of_edges()} edges")
    t = env.trucks.get("T1")
    if t:
        print(f"T1: home={t.home_hub_id}, at={t.current_node}, cap={t.capacity}, status={t.status}")

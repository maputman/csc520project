import networkx as nx
import random
import matplotlib.pyplot as plt


# resource kinds trucks and zones use (order stable for printing / unload)
RESOURCE_TYPES = ("water", "food", "medical")

# Default max units of each resource on a truck; max 30 total if all three are full.
DEFAULT_TRUCK_CAPACITY_PER_RESOURCE = 10

# Max units of each resource a zone can buffer per delivery cycle (site storage cap).
DEFAULT_ZONE_HOLD_PER_RESOURCE = 10


#  data classes

class Zone:
    # disaster-affected area that needs supplies

    def __init__(
        self,
        zone_id,
        x,
        y,
        urgency,
        needs,
        critical_resource=None,
        *,
        max_hold_per_resource: int | None = None,
    ):
        """
        zone_id         : unique string "Z1"
        x, y            : coordinates (used for A* heuristic later)
        urgency         : int 1-10, severity of situation
        needs           : dict, {"water": 3, "food": 2, "medical": 0}
        critical_resource: resource type that is at zero (triggers hard constraint),
                           or None if no critical shortage
        max_hold_per_resource : max units of each type the site can take into buffer
                                before applying to needs (default 10 each).
        """
        self.zone_id = zone_id
        self.x = x
        self.y = y
        self.urgency = urgency
        self.max_hold_per_resource = (
            DEFAULT_ZONE_HOLD_PER_RESOURCE
            if max_hold_per_resource is None
            else int(max_hold_per_resource)
        )
        merged = {r: 0 for r in RESOURCE_TYPES}
        merged.update(needs)
        self.needs = merged
        self.stocked = {r: 0 for r in RESOURCE_TYPES}
        self.critical_resource = critical_resource
        self.served = False

    def is_critical(self):
        # returns True if this zone has a zero-supply critical resource
        return self.critical_resource is not None

    def _flush_stock_to_needs(self, resource_type: str) -> None:
        while True:
            st = self.stocked.get(resource_type, 0)
            nd = self.needs.get(resource_type, 0)
            if st <= 0 or nd <= 0:
                break
            u = min(st, nd)
            self.stocked[resource_type] = st - u
            self.needs[resource_type] = nd - u
        if self.stocked.get(resource_type, 0) == 0:
            self.stocked.pop(resource_type, None)

    def update_needs(self, resource_type, amount) -> int:
        """
        Accept up to ``amount`` units from a delivery, respecting remaining need and
        per-type site capacity. Returns how many units were actually taken off the truck.
        """
        if resource_type not in RESOURCE_TYPES or amount <= 0:
            return 0
        need = self.needs.get(resource_type, 0)
        space = self.max_hold_per_resource - self.stocked.get(resource_type, 0)
        incoming = min(int(amount), need, space)
        if incoming <= 0:
            return 0
        self.stocked[resource_type] = self.stocked.get(resource_type, 0) + incoming
        self._flush_stock_to_needs(resource_type)
        if self.critical_resource and self.needs.get(self.critical_resource, 0) == 0:
            self.critical_resource = None
        if all(self.needs[r] == 0 for r in RESOURCE_TYPES):
            self.served = True
        return incoming

    def has_unmet_needs(self):
        return not self.served

    def needs_summary(self) -> str:
        parts = [f"{r}:{self.needs[r]}" for r in RESOURCE_TYPES]
        st = [f"{r}:{self.stocked.get(r, 0)}" for r in RESOURCE_TYPES if self.stocked.get(r, 0)]
        extra = f" | buffer {', '.join(st)}" if st else ""
        return "{" + ", ".join(parts) + "}" + extra

    def __repr__(self):
        return (f"Zone({self.zone_id}, urgency={self.urgency}, "
                f"needs={self.needs}, critical={self.critical_resource})")


class Hub:
    # supply depot that holds inventory and dispatches trucks

    def __init__(self, hub_id, x, y, inventory):
        """
        hub_id    : unique string, "H1"
        x, y      : coordinates
        inventory : dict, {"water": 10, "food": 8, "medical": 5}
        """
        self.hub_id = hub_id
        self.x = x
        self.y = y
        self.inventory = inventory

    def can_dispatch(self, resource_type, amount=1):
        return self.inventory.get(resource_type, 0) >= amount

    def dispatch(self, resource_type, amount=1):
        # remove resources from inventory when dispatched
        if not self.can_dispatch(resource_type, amount):
            raise ValueError(f"{self.hub_id} cannot dispatch {amount} of {resource_type}")
        self.inventory[resource_type] -= amount

    def __repr__(self):
        return f"Hub({self.hub_id}, inventory={self.inventory})"


class Truck:
    # supply vehicle that moves between hubs and zones

    def __init__(self, truck_id, home_hub_id, capacity_per_resource=DEFAULT_TRUCK_CAPACITY_PER_RESOURCE):
        """
        truck_id              : unique string, "T1"
        home_hub_id           : the hub this truck operates from
        capacity_per_resource : max units of each resource type on board at once
                                (same cap for water, food, medical). Aliased as .capacity
                                for checks like "shipment size <= truck capacity".
        """
        self.truck_id = truck_id
        self.home_hub_id = home_hub_id
        self.capacity_per_resource = int(capacity_per_resource)
        self.capacity = self.capacity_per_resource
        self.current_node = home_hub_id   # current location on the graph
        self.status = "idle"              # "idle", "en_route", "delivering"
        self.route = []                   # remaining nodes in current path
        self.cargo = {}                   # e.g. {"water": 2, "food": 1}

    def _amount(self, resource_type: str) -> int:
        if resource_type not in RESOURCE_TYPES:
            return 0
        return int(self.cargo.get(resource_type, 0))

    def can_load(self, resource_type: str, amount: int) -> bool:
        if resource_type not in RESOURCE_TYPES or amount < 0:
            return False
        return self._amount(resource_type) + amount <= self.capacity_per_resource

    def load(self, resource_type: str, amount: int):
        """Add units picked up at a hub (must not exceed per-type capacity)."""
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(f"Unknown resource type {resource_type!r}; expected one of {RESOURCE_TYPES}")
        if amount < 0:
            raise ValueError("amount must be non-negative")
        if amount == 0:
            return
        if not self.can_load(resource_type, amount):
            raise ValueError(
                f"Cannot load {amount} {resource_type}: already carrying "
                f"{self._amount(resource_type)}/{self.capacity_per_resource} of that type"
            )
        self.cargo[resource_type] = self._amount(resource_type) + amount
        self.status = "en_route"

    def unload(self):
        """
        Deliver everything on board (e.g. at a zone). Returns amounts per resource type.
        """
        delivered = {r: self.cargo.pop(r, 0) for r in RESOURCE_TYPES}
        self.cargo.clear()
        self.status = "idle"
        return delivered

    def unload_amount(self, resource_type: str, amount: int) -> int:
        """Drop off up to ``amount`` of one resource at a zone; returns how many were unloaded."""
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(f"Unknown resource type {resource_type!r}")
        if amount <= 0:
            return 0
        have = self._amount(resource_type)
        take = min(amount, have)
        if take <= 0:
            return 0
        self.cargo[resource_type] = have - take
        if self.cargo[resource_type] == 0:
            self.cargo.pop(resource_type, None)
        if not self.cargo:
            self.status = "idle"
        return take

    def cargo_summary(self) -> str:
        parts = [f"{r}:{self._amount(r)}/{self.capacity_per_resource}" for r in RESOURCE_TYPES]
        return "{" + ", ".join(parts) + "}"

    def total_cargo_units(self) -> int:
        return sum(self._amount(r) for r in RESOURCE_TYPES)

    def __repr__(self):
        return (f"Truck({self.truck_id}, at={self.current_node}, "
                f"status={self.status}, cargo={self.cargo})")


#  environment

class DisasterEnvironment:
    """
    Holds the graph, all zones, hubs, and trucks
    Provides methods to update road conditions and inject new distress calls
    """

    def __init__(self, seed=42):
        random.seed(seed)
        self.graph = nx.Graph()
        self.zones = {}   # zone_id  -> Zone
        self.hubs = {}    # hub_id   -> Hub
        self.trucks = {}  # truck_id -> Truck
        self.blocked_edges = set()   # set of (u, v) tuples currently blocked
        self.time_step = 0

    # constructing graph

    def add_hub(self, hub_id, x, y, inventory):
        hub = Hub(hub_id, x, y, inventory)
        self.hubs[hub_id] = hub
        self.graph.add_node(hub_id, type="hub", x=x, y=y)
        return hub

    def add_zone(
        self,
        zone_id,
        x,
        y,
        urgency,
        needs,
        critical_resource=None,
        *,
        max_hold_per_resource=None,
    ):
        zone = Zone(
            zone_id,
            x,
            y,
            urgency,
            needs,
            critical_resource,
            max_hold_per_resource=max_hold_per_resource,
        )
        self.zones[zone_id] = zone
        self.graph.add_node(zone_id, type="zone", x=x, y=y)
        return zone

    def add_road(self, node_a, node_b, cost):
        # add a bidirectional road with a travel cost
        self.graph.add_edge(node_a, node_b, weight=cost, blocked=False)

    def add_truck(
        self,
        truck_id,
        home_hub_id,
        capacity=DEFAULT_TRUCK_CAPACITY_PER_RESOURCE,
        *,
        capacity_per_resource=None,
    ):
        cap = capacity_per_resource if capacity_per_resource is not None else capacity
        truck = Truck(truck_id, home_hub_id, capacity_per_resource=cap)
        self.trucks[truck_id] = truck
        return truck

    # dynamic environment updates

    def block_road(self, node_a, node_b):
        # block a road due to damage, triggers replanning for any truck using it
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]["blocked"] = True
            self.blocked_edges.add((node_a, node_b))
            self.blocked_edges.add((node_b, node_a))
            print(f"  [ENV] Road blocked: {node_a} <-> {node_b}")

    def unblock_road(self, node_a, node_b):
        # clear a road blockage after rescue teams clear debris
        if self.graph.has_edge(node_a, node_b):
            self.graph[node_a][node_b]["blocked"] = False
            self.blocked_edges.discard((node_a, node_b))
            self.blocked_edges.discard((node_b, node_a))
            print(f"  [ENV] Road cleared: {node_a} <-> {node_b}")

    def add_distress_call(
        self, zone_id, x, y, urgency, needs, critical_resource=None, **kwargs
    ):
        # inject a new distress zone mid-simulation
        zone = self.add_zone(
            zone_id, x, y, urgency, needs, critical_resource, **kwargs
        )
        print(f"  [ENV] New distress call: {zone}")
        return zone

    def get_active_graph(self):
        # return a view of the graph with blocked edges removed, for use by A*
        active = nx.Graph()
        for node, data in self.graph.nodes(data=True):
            active.add_node(node, **data)
        for u, v, data in self.graph.edges(data=True):
            if not data.get("blocked", False):
                active.add_edge(u, v, **data)
        return active

    def tick(self):
        # advance simulation time by one step
        self.time_step += 1

    # getters

    def get_unserved_zones(self):
        return [z for z in self.zones.values() if z.has_unmet_needs()]

    def get_critical_zones(self):
        return [z for z in self.get_unserved_zones() if z.is_critical()]

    def get_idle_trucks(self):
        return [t for t in self.trucks.values() if t.status == "idle"]

    def get_node_coords(self, node_id):
        data = self.graph.nodes[node_id]
        return data["x"], data["y"]

    def print_status(self):
        print(f"\n{'='*50}")
        print(f"  Time Step: {self.time_step}")
        print(f"  Unserved zones : {len(self.get_unserved_zones())}")
        print(f"  Critical zones : {len(self.get_critical_zones())}")
        print(f"  Blocked roads  : {len(self.blocked_edges) // 2}")
        print(f"\n  Zones:")
        for z in self.zones.values():
            status = "SERVED" if z.served else ("CRITICAL" if z.is_critical() else "active")
            print(
                f"    {z.zone_id}: urgency={z.urgency}, {z.needs_summary()}, [{status}]"
            )
        print(f"\n  Hubs:")
        for h in self.hubs.values():
            print(f"    {h.hub_id}: inventory={h.inventory}")
        print(f"\n  Trucks:")
        for t in self.trucks.values():
            print(
                f"    {t.truck_id}: at={t.current_node}, status={t.status}, "
                f"cargo={t.cargo_summary()}"
            )
        print(f"{'='*50}\n")

    def truck_fill_from_home_hub(self, truck_id: str) -> None:
        """
        Load one truck from its home hub up to per-type caps (10/10/10 by default),
        limited by hub inventory.
        """
        truck = self.trucks[truck_id]
        hub = self.hubs[truck.home_hub_id]
        if truck.current_node != truck.home_hub_id:
            raise ValueError(
                f"Truck {truck_id} must be at home hub {truck.home_hub_id} to load "
                f"(currently at {truck.current_node})"
            )
        for r in RESOURCE_TYPES:
            room = truck.capacity_per_resource - truck._amount(r)
            if room <= 0:
                continue
            take = min(room, hub.inventory.get(r, 0))
            if take <= 0:
                continue
            hub.dispatch(r, take)
            truck.load(r, take)

    def run_truck_delivery_tour(self, truck_id: str, verbose: bool = False) -> dict:
        """
        Fill the truck at its home hub, then visit zones (shortest paths on the active
        graph) and unload until the truck is empty or no zone can accept remaining cargo.

        Returns a dict with visit list, per-resource totals delivered, and any cargo left.
        """
        truck = self.trucks[truck_id]
        G = self.get_active_graph()
        self.truck_fill_from_home_hub(truck_id)
        visits: list[tuple[str, dict[str, int]]] = []
        delivered_total = {r: 0 for r in RESOURCE_TYPES}
        full_node_path: list[str] = []

        def _append_segment(route: list[str], segment: list[str]) -> None:
            if not segment:
                return
            if not route:
                route.extend(segment)
                return
            if route[-1] == segment[0]:
                route.extend(segment[1:])
            else:
                route.extend(segment)

        while truck.total_cargo_units() > 0:
            targets = []
            for z in sorted(self.zones.values(), key=lambda zz: zz.zone_id):
                if z.served:
                    continue
                for r in RESOURCE_TYPES:
                    if truck._amount(r) > 0 and z.needs.get(r, 0) > 0:
                        targets.append(z.zone_id)
                        break
            if not targets:
                break
            target = targets[0]
            if not nx.has_path(G, truck.current_node, target):
                if verbose:
                    print(
                        f"  [ENV] No path from {truck.current_node} to {target}; "
                        f"stopping tour with cargo {truck.cargo}"
                    )
                break
            path = nx.shortest_path(G, truck.current_node, target, weight="weight")
            _append_segment(full_node_path, path)
            if verbose:
                print(f"  [ENV] {truck_id}: {' -> '.join(path)} → unload @ {target}")
            truck.current_node = target
            visit_deliver = {r: 0 for r in RESOURCE_TYPES}
            while True:
                progressed = False
                for r in RESOURCE_TYPES:
                    if truck._amount(r) <= 0:
                        continue
                    avail = truck._amount(r)
                    used = self.zones[target].update_needs(r, avail)
                    if used > 0:
                        truck.unload_amount(r, used)
                        visit_deliver[r] += used
                        delivered_total[r] += used
                        progressed = True
                if not progressed:
                    break
            visits.append((target, visit_deliver))
            truck.status = "idle" if truck.total_cargo_units() == 0 else "en_route"

        truck.status = "idle" if truck.total_cargo_units() == 0 else truck.status
        return {
            "visits": visits,
            "delivered": delivered_total,
            "cargo_left": dict(truck.cargo),
            "full_node_path": full_node_path,
        }


#  build a default scenario

def build_scenario():
    """
    Constructs a sample disaster region:
      - 2 supply hubs (H1, H2)
      - 5 affected zones (Z1-Z5), one critical
      - 2 trucks
      - A connected road network with varied costs
    """
    env = DisasterEnvironment(seed=42)

    # Supply hubs
    env.add_hub("H1", x=0,  y=0,  inventory={"water": 10, "food": 8,  "medical": 6})
    env.add_hub("H2", x=10, y=0,  inventory={"water": 6,  "food": 10, "medical": 4})

    # affected zones
    # Z1 is critical : zero water
    env.add_zone("Z1", x=2,  y=5,  urgency=9, needs={"water": 3, "food": 1, "medical": 0},
                 critical_resource="water")
    env.add_zone("Z2", x=5,  y=8,  urgency=7, needs={"water": 2, "food": 2, "medical": 1})
    env.add_zone("Z3", x=8,  y=6,  urgency=5, needs={"water": 1, "food": 3, "medical": 0})
    env.add_zone("Z4", x=3,  y=2,  urgency=6, needs={"water": 0, "food": 2, "medical": 2},
                 critical_resource="medical")
    env.add_zone("Z5", x=7,  y=3,  urgency=4, needs={"water": 2, "food": 1, "medical": 1})

    # Road network
    env.add_road("H1", "Z1", cost=3)
    env.add_road("H1", "Z4", cost=2)
    env.add_road("H1", "Z2", cost=6)
    env.add_road("Z1", "Z2", cost=4)
    env.add_road("Z1", "Z4", cost=3)
    env.add_road("Z2", "Z3", cost=3)
    env.add_road("Z2", "Z5", cost=4)
    env.add_road("Z3", "H2", cost=3)
    env.add_road("Z4", "Z5", cost=4)
    env.add_road("Z5", "H2", cost=3)
    env.add_road("H2", "Z3", cost=3)

    # Trucks
    env.add_truck("T1", home_hub_id="H1")
    env.add_truck("T2", home_hub_id="H2")

    return env


def bridge_disconnected_components(env) -> int:
    """
    Repeatedly add a road between the closest pair of nodes in different connected
    components until the active (non-blocked) graph is connected.
    Returns how many new roads were added.
    """
    import math

    added = 0
    while True:
        active = env.get_active_graph()
        comps = list(nx.connected_components(active))
        if len(comps) <= 1:
            break
        best_dist = None
        best_u = best_v = None
        for i, ca in enumerate(comps):
            for cb in comps[i + 1 :]:
                for u in ca:
                    xu, yu = env.get_node_coords(u)
                    for v in cb:
                        xv, yv = env.get_node_coords(v)
                        d = math.hypot(xu - xv, yu - yv)
                        if best_dist is None or d < best_dist:
                            best_dist, best_u, best_v = d, u, v
        if best_u is None:
            break
        if not env.graph.has_edge(best_u, best_v):
            env.add_road(best_u, best_v, cost=round(best_dist, 2))
            added += 1
    return added


def build_roads_by_proximity(env, max_neighbors=3, max_distance=10.0):
    """
    Builds a realistic sparse road network by connecting each node only to its
    nearest neighbors within a distance threshold.

    This avoids the dense "everything connected to everything" problem that
    occurs when roads are added naively for large scenarios.

    Parameters
    ----------
    env           : DisasterEnvironment (nodes already added)
    max_neighbors : max edges per node (default 3 — keeps graph sparse)
    max_distance  : hard cutoff; nodes farther apart are never connected
    """
    import math

    nodes = list(env.graph.nodes(data=True))

    for node_id, data in nodes:
        x1, y1 = data["x"], data["y"]

        # compute distances to every other node
        distances = []
        for other_id, other_data in nodes:
            if other_id == node_id:
                continue
            dx = x1 - other_data["x"]
            dy = y1 - other_data["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= max_distance:
                distances.append((dist, other_id))

        # connect to the closest N neighbors (skip if edge already exists)
        distances.sort()
        connected = 0
        for dist, neighbor_id in distances:
            if connected >= max_neighbors:
                break
            if not env.graph.has_edge(node_id, neighbor_id):
                env.add_road(node_id, neighbor_id, cost=round(dist, 2))
                connected += 1

    active = env.get_active_graph()
    components = list(nx.connected_components(active))
    n_bridge = 0
    if len(components) > 1:
        n_bridge = bridge_disconnected_components(env)
        active = env.get_active_graph()
        components = list(nx.connected_components(active))

    edges = active.number_of_edges()
    nn = active.number_of_nodes()
    if len(components) > 1:
        print(
            f"  [ENV] Warning: graph has {len(components)} disconnected components "
            f"after {n_bridge} bridge edge(s)."
        )
    elif n_bridge:
        print(
            f"  [ENV] Road network: {nn} nodes, {edges} edges "
            f"(sparse + {n_bridge} bridge edge(s) for connectivity)"
        )
    else:
        avg = f"{edges / nn:.1f}" if nn else "n/a"
        print(
            f"  [ENV] Road network built: {nn} nodes, {edges} edges (avg {avg} per node)"
        )


def build_large_scenario(hub_csv=None, zone_csv=None, truck_csv=None,
                         max_neighbors=3, max_distance=12.0, seed=42):
    """
    Builds a large scenario either from CSV files (exported from the xlsx)
    or from the default 30-node random data if no CSVs are provided.

    CSV column expectations
    -----------------------
    Hubs  : Hub ID, X Coord, Y Coord, Water (units), Food (units), Medical (units)
    Zones : Zone ID, X Coord, Y Coord, Urgency (1-10), Water (units),
            Food (units), Medical (units), Critical Resource
    Trucks: Truck ID, Home Hub, Current Location, Water on Board,
            Food on Board, Medical on Board, Capacity, Status
    """
    import math
    import csv

    env = DisasterEnvironment(seed=seed)

    # ── load hubs ──────────────────────────────────────────────────────────
    if hub_csv:
        with open(hub_csv) as f:
            for row in csv.DictReader(f):
                env.add_hub(
                    row["Hub ID"].strip(),
                    x=float(row["X Coord"]),
                    y=float(row["Y Coord"]),
                    inventory={
                        "water":   int(row["Water (units)"]),
                        "food":    int(row["Food (units)"]),
                        "medical": int(row["Medical (units)"]),
                    }
                )
    else:
        # built-in 30-hub default (mirrors xlsx seed=99 data)
        random.seed(99)
        used = set()
        for i in range(1, 31):
            while True:
                x, y = random.randint(0, 30), random.randint(0, 30)
                if (x, y) not in used:
                    used.add((x, y)); break
            env.add_hub(f"H{i}", x=x, y=y,
                        inventory={"water":   random.randint(5, 20),
                                   "food":    random.randint(5, 20),
                                   "medical": random.randint(3, 15)})

    # ── load zones ─────────────────────────────────────────────────────────
    if zone_csv:
        with open(zone_csv) as f:
            for row in csv.DictReader(f):
                critical = row.get("Critical Resource", "").strip() or None
                env.add_zone(
                    row["Zone ID"].strip(),
                    x=float(row["X Coord"]),
                    y=float(row["Y Coord"]),
                    urgency=int(row["Urgency (1-10)"]),
                    needs={
                        "water":   int(row["Water (units)"]),
                        "food":    int(row["Food (units)"]),
                        "medical": int(row["Medical (units)"]),
                    },
                    critical_resource=critical if critical else None
                )
    else:
        random.seed(99)
        used = set()
        # re-advance rng past hub generation
        for _ in range(30):
            while True:
                x, y = random.randint(0, 30), random.randint(0, 30)
                if (x, y) not in used:
                    used.add((x, y)); break
        used2 = set()
        for i in range(1, 31):
            while True:
                x, y = random.randint(0, 30), random.randint(0, 30)
                if (x, y) not in used2:
                    used2.add((x, y)); break
            urgency = random.randint(1, 10)
            water   = 0 if random.random() < 0.25 else random.randint(1, 6)
            food    = 0 if random.random() < 0.20 else random.randint(1, 6)
            medical = 0 if random.random() < 0.20 else random.randint(1, 5)
            critical = ("water" if water == 0
                        else "food" if food == 0
                        else "medical" if medical == 0
                        else None)
            env.add_zone(f"Z{i}", x=x, y=y, urgency=urgency,
                         needs={"water": water, "food": food, "medical": medical},
                         critical_resource=critical)

    # ── load trucks ────────────────────────────────────────────────────────
    if truck_csv:
        with open(truck_csv) as f:
            for row in csv.DictReader(f):
                t = env.add_truck(row["Truck ID"].strip(),
                                  home_hub_id=row["Home Hub"].strip(),
                                  capacity=int(row["Capacity"]))
                t.current_node = row["Current Location"].strip()
    else:
        random.seed(99)
        hub_ids = list(env.hubs.keys())
        for i in range(1, 31):
            home = random.choice(hub_ids)
            env.add_truck(f"T{i}", home_hub_id=home,
                          capacity=random.choice([4, 5, 5, 5, 6, 8]))

    # ── build sparse road network ──────────────────────────────────────────
    build_roads_by_proximity(env, max_neighbors=max_neighbors,
                             max_distance=max_distance)

    return env


#  test

if __name__ == "__main__":
    env = build_scenario()

    print("=== Initial State ===")
    env.print_status()

    print("--- Simulating a delivery to Z1 ---")
    truck = env.trucks["T1"]
    hub = env.hubs["H1"]
    hub.dispatch("water", amount=3)
    truck.load("water", 3)
    truck.current_node = "Z1"
    accepted = env.zones["Z1"].update_needs("water", truck._amount("water"))
    truck.unload_amount("water", accepted)
    env.tick()

    print("\n=== After delivering water to Z1 ===")
    env.print_status()

    print("--- Blocking road H1 <-> Z1 ---")
    env.block_road("H1", "Z1")

    print("\n--- Active graph edges (blocked roads removed) ---")
    active = env.get_active_graph()
    for u, v, d in active.edges(data=True):
        print(f"  {u} <-> {v}  cost={d['weight']}")

    print("\n--- Injecting new distress call ---")
    env.add_distress_call("Z6", x=4, y=7, urgency=8,
                          needs={"water": 0, "food": 2, "medical": 3},
                          critical_resource="water")
    env.tick()
    env.print_status()


    # visualization
    pos = {node: (data["x"], data["y"]) for node, data in env.graph.nodes(data=True)}
    node_colors = ["lightblue" if env.graph.nodes[n]["type"] == "hub" else "salmon" for n in env.graph.nodes]
    nx.draw(env.graph, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=10)
    plt.title("Disaster Region")
    plt.show()

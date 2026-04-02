import networkx as nx
import random
import matplotlib.pyplot as plt


#  data classes

class Zone:
    # disaster-affected area that needs supplies

    def __init__(self, zone_id, x, y, urgency, needs, critical_resource=None):
        """
        zone_id         : unique string "Z1"
        x, y            : coordinates (used for A* heuristic later)
        urgency         : int 1-10, severity of situation
        needs           : dict, {"water": 3, "food": 2, "medical": 0}
        critical_resource: resource type that is at zero (triggers hard constraint),
                           or None if no critical shortage
        """
        self.zone_id = zone_id
        self.x = x
        self.y = y
        self.urgency = urgency
        self.needs = needs                          # remaining unmet needs
        self.critical_resource = critical_resource  # "water" if water == 0
        self.served = False                         # boolean flipped when all needs are met

    def is_critical(self):
        # returns True if this zone has a zero-supply critical resource
        return self.critical_resource is not None

    def update_needs(self, resource_type, amount):
        # reduce the need for a resource after a delivery, mark served if done
        if resource_type in self.needs:
            self.needs[resource_type] = max(0, self.needs[resource_type] - amount)
        # clear critical flag if the critical resource has now been delivered
        if self.critical_resource and self.needs.get(self.critical_resource, 0) == 0:
            self.critical_resource = None
        # mark zone as fully served when all needs are zero
        if all(v == 0 for v in self.needs.values()):
            self.served = True

    def has_unmet_needs(self):
        return not self.served

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

    def __init__(self, truck_id, home_hub_id, capacity=5):
        """
        truck_id    : unique string, "T1"
        home_hub_id : the hub this truck operates from
        capacity    : max units the truck can carry per trip
        """
        self.truck_id = truck_id
        self.home_hub_id = home_hub_id
        self.capacity = capacity
        self.current_node = home_hub_id   # current location on the graph
        self.status = "idle"              # "idle", "en_route", "delivering"
        self.route = []                   # remaining nodes in current path
        self.cargo = {}                   # what the truck is carrying

    def load(self, resource_type, amount):
        self.cargo = {resource_type: amount}
        self.status = "en_route"

    def unload(self):
        delivered = self.cargo.copy()
        self.cargo = {}
        self.status = "idle"
        return delivered

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

    def add_zone(self, zone_id, x, y, urgency, needs, critical_resource=None):
        zone = Zone(zone_id, x, y, urgency, needs, critical_resource)
        self.zones[zone_id] = zone
        self.graph.add_node(zone_id, type="zone", x=x, y=y)
        return zone

    def add_road(self, node_a, node_b, cost):
        # add a bidirectional road with a travel cost
        self.graph.add_edge(node_a, node_b, weight=cost, blocked=False)

    def add_truck(self, truck_id, home_hub_id, capacity=5):
        truck = Truck(truck_id, home_hub_id, capacity)
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

    def add_distress_call(self, zone_id, x, y, urgency, needs, critical_resource=None):
        # inject a new distress zone mid-simulation
        zone = self.add_zone(zone_id, x, y, urgency, needs, critical_resource)
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
            print(f"    {z.zone_id}: urgency={z.urgency}, needs={z.needs}, [{status}]")
        print(f"\n  Hubs:")
        for h in self.hubs.values():
            print(f"    {h.hub_id}: inventory={h.inventory}")
        print(f"\n  Trucks:")
        for t in self.trucks.values():
            print(f"    {t.truck_id}: at={t.current_node}, status={t.status}, cargo={t.cargo}")
        print(f"{'='*50}\n")



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
    env.add_truck("T1", home_hub_id="H1", capacity=5)
    env.add_truck("T2", home_hub_id="H2", capacity=5)

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
    delivered = truck.unload()
    env.zones["Z1"].update_needs("water", delivered["water"])
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
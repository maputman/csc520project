import math
from dataclasses import dataclass
from typing import List

from astar import path_for_dispatch

#  CSP Solver for dispatch ordering

#  models the dispatch assignment problem as a CSP:
#    Variables : (zone, hub, resource) one assignment per dispatch slot
#    Domains   : all valid (zone, hub, resource) combinations
#    Hard constraints : must ALL be satisfied 
#    Soft constraints : scored and used to rank candidates (best first)

#  the solver returns the single best next dispatch assignment
#  re-called every planning cycle as the environment changes


# Euclidean distance helper
def euclidean_distance(env, node_a, node_b):
    # straight-line distance between two nodes using their xy coordinates
    x1, y1 = env.get_node_coords(node_a)
    x2, y2 = env.get_node_coords(node_b)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


#  build domain
#  all (zone, hub, resource) combos that are worth considering before applying constraints
def build_domain(env):
    """
    generate all candidate assignments: (zone_id, hub_id, resource_type)
    candidate is included only if:
      - zone has an unmet need for that resource
      - hub has that resource in stock
    """
    candidates = []
    for zone in env.get_unserved_zones():
        for resource, amount in zone.needs.items():
            if amount > 0:  # zone actually needs this resource
                for hub in env.hubs.values():
                    if hub.can_dispatch(resource):
                        candidates.append((zone.zone_id, hub.hub_id, resource))
    return candidates



#  hard constraints
#  filter out any assignment that violates a mandatory rule

def passes_hard_constraints(assignment, env):
    """
    return true if the assignment satisfies ALL hard constraints
    assignment is a tuple: (zone_id, hub_id, resource_type)

    hard constraints:
      HC1: Critical zones must be served before non-critical zones.
            If any critical zone exists, only assignments targeting a
            critical zone are valid
      HC2: Hub must have the resource in stock (at least 1 unit)
      HC3: Zone must actually need the resource being dispatched
      HC4: Zone must not already be fully served
    """
    zone_id, hub_id, resource = assignment

    zone = env.zones[zone_id]
    hub = env.hubs[hub_id]
    critical_zones = env.get_critical_zones()

    # HC1: if critical zones exist, only serve critical zones
    if critical_zones and not zone.is_critical():
        return False

    # HC2: hub must have the resource
    if not hub.can_dispatch(resource):
        return False

    # HC3: zone must need this resource
    if zone.needs.get(resource, 0) == 0:
        return False

    # HC4: zone must not already be served
    if zone.served:
        return False

    return True


# soft constraints
# score each valid assignment where higher is better
# used to rank candidates so the best is tried first
# weights control how much each soft constraint matters
WEIGHT_URGENCY   = 3.0   # urgency score of the target zone
WEIGHT_DISTANCE  = 2.0   # prefer shorter hub-to-zone distance
WEIGHT_INVENTORY = 1.0   # prefer hubs with more remaining stock

def soft_constraint_score(assignment, env):
    """
    returns a score for a valid assignment
    higher score = better assignment = explored first

    soft constraints:
      SC1: Prefer zones with higher urgency scores
      SC2: Prefer assignments with shorter hub-to-zone distance
      SC3: Prefer hubs with higher remaining inventory (avoid depleting one hub)
    """
    zone_id, hub_id, resource = assignment
    zone = env.zones[zone_id]
    hub = env.hubs[hub_id]

    # SC1: urgency score (higher is better, 1-10)
    urgency_score = zone.urgency * WEIGHT_URGENCY

    # SC2: distance penalty (shorter = better, so we negate it)
    distance = euclidean_distance(env, hub_id, zone_id)
    # normalize distance to a 0-10 scale for fair weighting
    # we subtract so closer hubs score higher
    distance_score = -(distance / 10.0) * WEIGHT_DISTANCE

    # SC3: hub inventory (more stock = better, use total remaining)
    total_inventory = sum(hub.inventory.values())
    inventory_score = (total_inventory / 30.0) * WEIGHT_INVENTORY  # normalize by max ~30

    return urgency_score + distance_score + inventory_score


#  backtracking search
#  combines domain, hard constraints, & soft constraint scores to find the single best assignment

def backtrack(candidates, env):
    """
    backtracking search over the candidate list
    - filters candidates by hard constraints
    - ranks remaining by soft constraint score
    - return the best valid assignment, or None if no valid assignment exists

    selecting ONE dispatch assignment per cycle, so backtracking tries candidates 
    in score order & returns the first one that passes all hard constraints
    """
    # keep only hard-constraint-satisfying candidates
    valid = [c for c in candidates if passes_hard_constraints(c, env)]

    if not valid:
        return None  # no valid assignment found this cycle

    # rank by soft constraint score (highest first)
    valid.sort(key=lambda c: soft_constraint_score(c, env), reverse=True)

    # return the best candidate
    # (backtracking would recurse here in a full multi-variable CSP;
    #  for single-slot dispatch we just return the top-ranked valid assignment)
    return valid[0]


#  CSP entry point
def solve_csp(env, claimed_zones=None, verbose=True):
    """
    Main entry point for the CSP solver.
    claimed_zones: a set of zone_ids already assigned to another truck this cycle.
    """
    if claimed_zones is None:
        claimed_zones = set()

    candidates = build_domain(env)

    # Exclude zones already claimed by another truck this cycle
    candidates = [c for c in candidates if c[0] not in claimed_zones]

    if not candidates:
        if verbose:
            print("  [CSP] No candidates available — all zones served or no inventory.")
        return None

    assignment = backtrack(candidates, env)

    if assignment:
        zone_id, hub_id, resource = assignment
        zone = env.zones[zone_id]
        score = soft_constraint_score(assignment, env)
        critical_tag = " [CRITICAL]" if zone.is_critical() else ""
        if verbose:
            print(f"  [CSP] Best assignment: dispatch {resource} from {hub_id} "
                  f"to {zone_id} (urgency={zone.urgency}, score={score:.2f}){critical_tag}")
    else:
        if verbose:
            print("  [CSP] No valid assignment found after applying hard constraints.")

    return assignment


#  agent integration (CSP + A* path for DisasterReliefAgent)

@dataclass
class DispatchAssignment:
    zone: object
    hub: object
    resource_type: str
    amount: int
    path: List[str]


def print_csp_state(env) -> None:
    n_unserved = len(env.get_unserved_zones())
    n_crit = len(env.get_critical_zones())
    n_idle = len(env.get_idle_trucks())
    print(
        f"  [CSP] t={env.time_step} | unserved_zones={n_unserved} | "
        f"critical={n_crit} | idle_trucks={n_idle}"
    )


def _dispatch_amount(zone, resource_type: str) -> int:
    need = int(zone.needs.get(resource_type, 0))
    return min(1, need) if need > 0 else 0


def solve_next_dispatch(
    env,
    max_assignments: int = 1,
    *,
    verbose: bool = False,
) -> List[DispatchAssignment]:
    """
    For each idle truck slot (up to ``max_assignments``), run ``solve_csp`` with
    ``claimed_zones`` so different trucks target different zones when possible,
    build an A* path hub→zone, and return ``DispatchAssignment`` objects.
    """
    n_slots = min(max_assignments, len(env.get_idle_trucks()))
    if n_slots <= 0:
        return []

    out: List[DispatchAssignment] = []
    claimed: set = set()

    for _ in range(n_slots):
        raw = solve_csp(env, claimed_zones=claimed, verbose=verbose)
        if not raw:
            break

        zone_id, hub_id, resource_type = raw
        zone = env.zones[zone_id]
        hub = env.hubs[hub_id]
        amount = _dispatch_amount(zone, resource_type)
        if amount <= 0:
            break

        path, cost = path_for_dispatch(env, hub_id, zone_id)
        if not path or math.isinf(cost):
            if verbose:
                print(f"  [CSP] No route from {hub_id} to {zone_id}; stopping multi-plan.")
            break

        out.append(
            DispatchAssignment(
                zone=zone,
                hub=hub,
                resource_type=resource_type,
                amount=amount,
                path=path,
            )
        )
        claimed.add(zone_id)

    return out


#  test
#  Note: solve_next_dispatch is the function used by the agent
#  solve_csp is a lower-level helper it calls internally

if __name__ == "__main__":
    # Import environment from the same folder
    import sys, os
    sys.path.append(os.path.dirname(__file__))
    from environment import build_scenario

    env = build_scenario()

    print("=== Initial Environment ===")
    env.print_status()

    print(">>> Cycle 1 — both trucks assigned simultaneously (Z1 and Z4 are CRITICAL)")
    assignments = solve_next_dispatch(env, max_assignments=2, verbose=True)
    for a in assignments:
        a.hub.dispatch(a.resource_type, a.amount)
        a.zone.update_needs(a.resource_type, a.amount)
    env.tick()
 
    print("\n>>> Cycle 2 — after first deliveries")
    assignments = solve_next_dispatch(env, max_assignments=2, verbose=True)
    for a in assignments:
        a.hub.dispatch(a.resource_type, a.amount)
        a.zone.update_needs(a.resource_type, a.amount)
    env.tick()
 
    print("\n>>> Manually serve all critical zones to test soft constraints")
    env.zones["Z1"].needs = {"water": 0, "food": 0, "medical": 0}
    env.zones["Z1"].served = True
    env.zones["Z4"].needs = {"water": 0, "food": 0, "medical": 0}
    env.zones["Z4"].served = True
    env.zones["Z4"].critical_resource = None
 
    print("\n>>> Cycle 3 — no critical zones, soft constraints decide")
    assignments = solve_next_dispatch(env, max_assignments=2, verbose=True)
 
    print("\n=== Final State ===")
    env.print_status()
 

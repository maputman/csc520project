"""
Greedy and random dispatch baselines for benchmarking against the CSP + A* agent.

Same simulation loop and ``_execute_dispatch`` as ``DisasterReliefAgent``; only
the assignment policy differs.
"""

from __future__ import annotations

import math
import random
from typing import List, Set

from agent import DisasterReliefAgent
from astar import path_for_dispatch
from csp import DispatchAssignment, build_domain, euclidean_distance, passes_hard_constraints


def _dispatch_amount(zone, hub, resource_type: str, max_amount: int = 10) -> int:
    need = int(zone.needs.get(resource_type, 0))
    stock = int(hub.inventory.get(resource_type, 0))
    return min(need, stock, max_amount) if need > 0 else 0


def _passes_greedy_constraints(assignment, env) -> bool:
    """
    Greedy-only feasibility check: HC2, HC3, HC4 only.
    Deliberately omits HC1 (critical-first) so greedy ranks purely by proximity,
    not by zone criticality. This is the key differentiator vs CSP.
    """
    zone_id, hub_id, resource = assignment
    zone = env.zones[zone_id]
    hub  = env.hubs[hub_id]
    if not hub.can_dispatch(resource):       # HC2: hub must have stock
        return False
    if zone.needs.get(resource, 0) == 0:     # HC3: zone must need this resource
        return False
    if zone.served:                          # HC4: zone not already fully served
        return False
    return True


def _assignment_from_tuple(env, zone_id: str, hub_id: str, resource: str, truck=None) -> DispatchAssignment | None:
    zone = env.zones[zone_id]
    hub = env.hubs[hub_id]
    max_amount = truck.capacity_per_resource if truck is not None else 10
    amount = _dispatch_amount(zone, hub, resource, max_amount=max_amount)
    if amount <= 0:
        return None
    path, cost = path_for_dispatch(env, hub_id, zone_id)
    if not path or math.isinf(cost):
        return None
    return DispatchAssignment(
        zone=zone,
        hub=hub,
        resource_type=resource,
        amount=amount,
        path=path,
    )


def _valid_candidates(env, claimed: Set[str]) -> List[tuple]:
    out = []
    for c in build_domain(env):
        if c[0] in claimed:
            continue
        if _passes_greedy_constraints(c, env):
            out.append(c)
    return out


def _shortest_path_cost(env, zone_id: str, hub_id: str) -> float:
    """A* shortest-path weight hub→zone; +inf if unreachable (same as dispatch)."""
    path, cost = path_for_dispatch(env, hub_id, zone_id)
    if not path or math.isinf(cost):
        return float("inf")
    return float(cost)


class GreedyBaseline(DisasterReliefAgent):
    """
    Among CSP-feasible (hub, zone, resource) triples, uses a **myopic** rule that
    differs from the CSP agent:

    CSP ranks by a weighted mix that puts **urgency first**, then distance and
    hub inventory. This baseline is **proximity-only** for ordering among feasible
    assignments: shortest **road** cost (A* on the current graph), then
    straight-line hub–zone distance, then lexicographic tie-breakers. **Urgency
    is not used** to rank candidates (only hard constraints, e.g. critical-first,
    filter the feasible set).

    That mimics a common greedy failure mode (minimize immediate driving while
    under-serving the most urgent zones when cost and urgency disagree), so
    benchmark gaps vs CSP + A* are usually visible.
    """

    def _planning_cycle(self):
        env = self.env
        idle = env.get_idle_trucks()
        if not idle:
            env.tick()
            self.metrics["total_steps"] += 1
            self._fire_events()
            return

        n_slots = min(len(env.trucks), len(idle))
        claimed: Set[str] = set()
        assignments: List[DispatchAssignment] = []

        for i in range(n_slots):
            candidates = _valid_candidates(env, claimed)
            if not candidates:
                break
            # Proximity-only ordering (road cost, then euclidean); no urgency term.
            candidates.sort(
                key=lambda c: (
                    _shortest_path_cost(env, c[0], c[1]),
                    euclidean_distance(env, c[1], c[0]),
                    c[0],
                    c[1],
                    c[2],
                ),
            )
            truck = idle[i]
            chosen = None
            for cand in candidates:
                da = _assignment_from_tuple(env, cand[0], cand[1], cand[2], truck=truck)
                if da is not None:
                    chosen = da
                    claimed.add(cand[0])
                    break
            if chosen is None:
                break
            assignments.append(chosen)

        if not assignments:
            if self.verbose:
                print("  [GREEDY] No valid dispatch — advancing time.")
            env.tick()
            self.metrics["total_steps"] += 1
            self._fire_events()
            return

        pairs = list(zip(idle, assignments))
        for truck, assignment in pairs:
            self._execute_dispatch(truck, assignment)
        env.tick()
        self.metrics["total_steps"] += 1
        self._fire_events()


class RandomBaseline(DisasterReliefAgent):
    """Uniform random choice among CSP-feasible assignments each slot."""

    def __init__(
        self,
        env,
        max_steps=100,
        verbose=True,
        seed=None,
        events=None,
        record_paths=False,
        dynamic_roadblock_chance=0.0,
    ):
        super().__init__(
            env,
            max_steps=max_steps,
            verbose=verbose,
            events=events,
            record_paths=record_paths,
            dynamic_roadblock_chance=dynamic_roadblock_chance,
        )
        self._rng = random.Random(seed)

    def _planning_cycle(self):
        env = self.env
        idle = env.get_idle_trucks()
        if not idle:
            env.tick()
            self.metrics["total_steps"] += 1
            self._fire_events()
            return

        n_slots = min(len(env.trucks), len(idle))
        claimed: Set[str] = set()
        assignments: List[DispatchAssignment] = []

        for i in range(n_slots):
            candidates = _valid_candidates(env, claimed)
            if not candidates:
                break
            self._rng.shuffle(candidates)
            truck = idle[i]
            chosen = None
            for cand in candidates:
                da = _assignment_from_tuple(env, cand[0], cand[1], cand[2], truck=truck)
                if da is not None:
                    chosen = da
                    claimed.add(cand[0])
                    break
            if chosen is None:
                break
            assignments.append(chosen)

        if not assignments:
            if self.verbose:
                print("  [RANDOM] No valid dispatch — advancing time.")
            env.tick()
            self.metrics["total_steps"] += 1
            self._fire_events()
            return

        pairs = list(zip(idle, assignments))
        for truck, assignment in pairs:
            self._execute_dispatch(truck, assignment)
        env.tick()
        self.metrics["total_steps"] += 1
        self._fire_events()

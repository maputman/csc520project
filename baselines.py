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


def _dispatch_amount(zone, resource_type: str) -> int:
    need = int(zone.needs.get(resource_type, 0))
    return min(1, need) if need > 0 else 0


def _assignment_from_tuple(env, zone_id: str, hub_id: str, resource: str) -> DispatchAssignment | None:
    zone = env.zones[zone_id]
    hub = env.hubs[hub_id]
    amount = _dispatch_amount(zone, resource)
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
        if passes_hard_constraints(c, env):
            out.append(c)
    return out


class GreedyBaseline(DisasterReliefAgent):
    """
    Among CSP-feasible (hub, zone, resource) triples, prefer higher zone urgency,
    then shorter straight-line hub–zone distance.
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

        for _ in range(n_slots):
            candidates = _valid_candidates(env, claimed)
            if not candidates:
                break
            candidates.sort(
                key=lambda c: (
                    -env.zones[c[0]].urgency,
                    euclidean_distance(env, c[1], c[0]),
                ),
            )
            chosen = None
            for cand in candidates:
                da = _assignment_from_tuple(env, cand[0], cand[1], cand[2])
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
        dynamic_roadblock_chance=0.0,
    ):
        super().__init__(
            env,
            max_steps=max_steps,
            verbose=verbose,
            events=events,
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

        for _ in range(n_slots):
            candidates = _valid_candidates(env, claimed)
            if not candidates:
                break
            self._rng.shuffle(candidates)
            chosen = None
            for cand in candidates:
                da = _assignment_from_tuple(env, cand[0], cand[1], cand[2])
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

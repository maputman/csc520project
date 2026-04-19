import random

from csp import solve_next_dispatch, print_csp_state
from astar import astar, replan


class DisasterReliefAgent:
    """
    The main AI agent that drives the simulation loop.

    Each planning cycle:
      1. CSP solver selects the next best dispatch assignment.
      2. A* computes the optimal route for that assignment.
      3. The truck is dispatched step-by-step along the path.
      4. If a road blockage is detected mid-route, A* replans from the
         truck's current position.
      5. On arrival the truck delivers resources and the environment updates.
      6. Any registered events for the new time step are fired.
      7. Repeat until all needs are met or the step limit is reached.
    """

    def __init__(
        self,
        env,
        max_steps=100,
        verbose=True,
        events=None,
        record_paths=False,
        dynamic_roadblock_chance=0.0,
    ):
        """
        env       : DisasterEnvironment
        max_steps : cap on planning cycles
        verbose   : print step by step output
        events    : dict mapping time_step -> list of callables(env)
                    events fire after env.tick() advances to that time step
        record_paths : if True, append each completed delivery's walked node list
                       (hub→...→zone, including replans) to ``delivery_paths`` for viz.
        dynamic_roadblock_chance : probability before each move step of attempting to
                       block an edge on the remaining route, but only if an alternate
                       path still exists so replanning can be demonstrated.
        """
        self.env = env
        self.max_steps = max_steps
        self.verbose = verbose
        self.events = events or {} # time_step -> [callable, ...]
        self.record_paths = record_paths
        self.dynamic_roadblock_chance = max(0.0, float(dynamic_roadblock_chance))
        self._rng = random.Random(42)
        self.delivery_paths: list[dict] = []

        # metrics collected during the run
        self.metrics = {
            "total_steps":           0,
            "deliveries_made":       0,
            "zones_served":          0,
            "critical_zones_served": 0,
            "critical_cleared_step": None,  # step when last critical zone was served
            "replan_events":         0,
            "failed_deliveries":     0,
            "total_travel_cost":     0.0,
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self):
        """Run the full simulation until all needs are met or step limit hit."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("  DISASTER RELIEF AGENT — starting simulation")
            print("=" * 60)
            self.env.print_status()

        while self.env.get_unserved_zones() and self.metrics["total_steps"] < self.max_steps:
            critical_before = bool(self.env.get_critical_zones())
            self._planning_cycle()
            # record the step the last critical zone transitioned from active to cleared
            if (critical_before
                    and not self.env.get_critical_zones()
                    and self.metrics["critical_cleared_step"] is None):
                self.metrics["critical_cleared_step"] = self.env.time_step

        self._print_summary()
        return self.metrics

    # ------------------------------------------------------------------
    # Internal cycle
    # ------------------------------------------------------------------

    def _planning_cycle(self):
        env = self.env

        if self.verbose:
            print_csp_state(env)

        # --- Step 1: CSP solver picks the best dispatch assignment ----------
        assignments = solve_next_dispatch(
            env, max_assignments=len(env.trucks), verbose=self.verbose
        )

        if not assignments:
            if self.verbose:
                print("  [AGENT] No valid dispatch found — advancing time.")
            env.tick()
            self.metrics["total_steps"] += 1
            return

        # Dispatch as many assignments as there are idle trucks
        idle_trucks = env.get_idle_trucks()
        dispatched_pairs = list(zip(idle_trucks, assignments))

        for truck, assignment in dispatched_pairs:
            self._execute_dispatch(truck, assignment)

        # Advance time after all dispatches in this cycle
        env.tick()
        self.metrics["total_steps"] += 1

        # fire any registered events for this time step
        self._fire_events()

    def _fire_events(self):
        """fire any callables registered for the current time step"""
        callbacks = self.events.get(self.env.time_step, [])
        for callback in callbacks:
            callback(self.env)

    def _maybe_trigger_dynamic_roadblock(self, truck, goal, remaining_path):
        """
        Occasionally block an edge on the truck's remaining route to trigger A* replanning.
        Keep the blockage only if the destination is still reachable without that edge.
        """
        if self.dynamic_roadblock_chance <= 0 or len(remaining_path) < 2:
            return
        if self._rng.random() >= self.dynamic_roadblock_chance:
            return

        candidates = [
            (truck.current_node, remaining_path[0]),
            *list(zip(remaining_path[:-1], remaining_path[1:])),
        ]
        self._rng.shuffle(candidates)

        for node_a, node_b in candidates:
            if (node_a, node_b) in self.env.blocked_edges:
                continue
            if not self.env.graph.has_edge(node_a, node_b):
                continue

            self.env.block_road(node_a, node_b)
            new_path, _ = astar(self.env, truck.current_node, goal)
            if new_path:
                if self.verbose:
                    print(
                        f"  [ENV] Dynamic blockage introduced on {node_a}<->{node_b} "
                        f"while {truck.truck_id} is en route."
                    )
                return

            self.env.unblock_road(node_a, node_b)

    def _execute_dispatch(self, truck, assignment):
        """
        Moves a truck along the A* path, handling mid-route replanning,
        then delivers the cargo.
        """
        env   = self.env
        zone  = assignment.zone
        hub   = assignment.hub
        rtype = assignment.resource_type
        amt   = assignment.amount
        path  = assignment.path

        if self.verbose:
            print(f"\n  [AGENT] Dispatching {truck.truck_id}: "
                  f"{hub.hub_id} -> {zone.zone_id} | "
                  f"{amt}x {rtype} | path: {' -> '.join(path)}")
        
        # mark zone as having been dispatched while actively critical (once set, stays True)
        if zone.is_critical():
            zone.dispatched_as_critical = True
        # was_critical uses the persistent flag so subsequent deliveries to the same
        # zone (after the critical resource was already delivered) are still counted
        was_critical = zone.dispatched_as_critical

        if amt > truck.capacity_per_resource:
            if self.verbose:
                print(
                    f"  [AGENT] Dispatch skipped: {amt}x {rtype} exceeds truck "
                    f"{truck.truck_id} per-type capacity {truck.capacity_per_resource}"
                )
            self.metrics["failed_deliveries"] += 1
            return

        if not truck.can_load(rtype, amt):
            if self.verbose:
                print(
                    f"  [AGENT] Dispatch skipped: truck {truck.truck_id} cannot load "
                    f"{amt}x {rtype} (already {truck.cargo.get(rtype, 0)}/"
                    f"{truck.capacity_per_resource} of that type on board)"
                )
            self.metrics["failed_deliveries"] += 1
            return

        # Pick up at hub, then deduct hub inventory
        try:
            hub.dispatch(rtype, amt)
        except ValueError as e:
            if self.verbose:
                print(f"  [AGENT] Dispatch failed: {e}")
            self.metrics["failed_deliveries"] += 1
            return

        truck.load(rtype, amt)
        truck.current_node = hub.hub_id

        dispatch_time_step = env.time_step
        visited_nodes: list[str] = [hub.hub_id] if self.record_paths else []

        # --- Step 2: Move truck step-by-step along the path ----------------
        remaining_path = path[1:]   # skip the starting hub node
        goal = zone.zone_id

        while remaining_path:
            self._maybe_trigger_dynamic_roadblock(truck, goal, remaining_path)
            next_node = remaining_path[0]
            edge = (truck.current_node, next_node)

            # Check if the upcoming road is blocked (dynamic environment)
            if edge in env.blocked_edges:
                if self.verbose:
                    print(f"  [AGENT] Road {edge[0]}<->{edge[1]} blocked mid-route!")
                self.metrics["replan_events"] += 1

                new_path, new_cost = replan(env, truck, goal)
                if not new_path:
                    if self.verbose:
                        print(f"  [AGENT] No alternate path — delivery of {rtype} aborted.")
                    self.metrics["failed_deliveries"] += 1
                    truck.unload()
                    # return resource to hub
                    hub.inventory[rtype] = hub.inventory.get(rtype, 0) + amt
                    return

                remaining_path = new_path[1:]   # new path starts at truck's current node

            # Move one step
            prev_node = truck.current_node
            truck.current_node = remaining_path.pop(0)
            if env.graph.has_edge(prev_node, truck.current_node):
                self.metrics["total_travel_cost"] += env.graph[prev_node][truck.current_node]["weight"]
            if self.record_paths:
                visited_nodes.append(truck.current_node)

        # --- Step 3: Deliver (zone may accept less than on truck due to hold caps) ---
        avail = truck._amount(rtype)
        accepted = zone.update_needs(rtype, avail)
        if accepted > 0:
            truck.unload_amount(rtype, accepted)

        self.metrics["deliveries_made"] += 1

        # check if zone is now fully served & update metrics
        if all(v == 0 for v in zone.needs.values()):
            self.metrics["zones_served"] += 1
            if was_critical:
                self.metrics["critical_zones_served"] += 1
            if self.verbose:
                print(f"  [AGENT] ✓ Zone {zone.zone_id} fully served.")

        if self.verbose:
            print(f"  [AGENT] Delivered {accepted}x {rtype} to {zone.zone_id}")

        if self.record_paths and len(visited_nodes) >= 2:
            blocked_unique = sorted(
                {
                    tuple(sorted((a, b)))
                    for (a, b) in env.blocked_edges
                    if env.graph.has_edge(a, b)
                }
            )
            self.delivery_paths.append(
                {
                    "time_step": dispatch_time_step,
                    "truck_id": truck.truck_id,
                    "path": list(visited_nodes),
                    "resource": rtype,
                    "zone_id": zone.zone_id,
                    "hub_id": hub.hub_id,
                    "amount": amt,
                    "blocked_edges": blocked_unique,
                }
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _print_summary(self):
        m = self.metrics
        remaining = self.env.get_unserved_zones()
        print("\n" + "=" * 60)
        print("  SIMULATION COMPLETE")
        print("=" * 60)
        print(f"  Steps taken        : {m['total_steps']}")
        print(f"  Deliveries made    : {m['deliveries_made']}")
        print(f"  Zones fully served : {m['zones_served']}")
        print(f"  Critical zones served : {m['critical_zones_served']}")
        print(f"  Replan events      : {m['replan_events']}")
        print(f"  Failed deliveries  : {m['failed_deliveries']}")
        print(f"  Total travel cost  : {m['total_travel_cost']:.2f}")
        if remaining:
            print(f"  Unserved zones     : {[z.zone_id for z in remaining]}")
        else:
            print("  All zones served!")
        print("=" * 60)

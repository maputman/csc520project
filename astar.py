"""
Graph search for disaster relief routing (CSC520 proposal).

Integration with ``csp.py`` (see project proposal):

  (1) The CSP selects the next dispatch: target area A, supply hub H, resource R.
  (2) **This module** computes the lowest-cost route from H to A on the current road
      graph (edges may be blocked; costs are edge weights).
  (3) The agent moves the truck along that path; the environment updates.
  (4) If a road blocks mid-route, ``replan`` runs A* again from the truck’s current
      node to the same goal with updated ``get_active_graph()``.

A* uses f(n) = g(n) + h(n): g is cumulative travel cost from the start node; h is an
admissible straight-line (Euclidean) heuristic to the goal, consistent with the proposal.
"""

from __future__ import annotations

import heapq
import math


def heuristic(env, node_a, node_b):
    """
    h(n): straight-line distance from node_a toward node_b (admissible for routing
    when edge weights are actual travel costs in the plane).
    """
    x1, y1 = env.get_node_coords(node_a)
    x2, y2 = env.get_node_coords(node_b)
    return math.hypot(x1 - x2, y1 - y2)


def astar(env, start, goal):
    """
    A* on the active (non-blocked) graph: f(n) = g(n) + h(n, goal).

    Parameters
    ----------
    env   : DisasterEnvironment
    start : node id (e.g. hub id when leaving a depot)
    goal  : node id (e.g. affected area)

    Returns
    -------
    path : node ids from start to goal inclusive, or [] if unreachable
    cost : sum of edge weights along path, or math.inf if unreachable
    """
    active = env.get_active_graph()

    if start == goal:
        return [start], 0.0

    if start not in active or goal not in active:
        return [], math.inf

    counter = 0
    open_set: list[tuple[float, int, str]] = []
    heapq.heappush(
        open_set,
        (heuristic(env, start, goal), counter, start),
    )

    came_from: dict[str, str] = {}
    g_score: dict[str, float] = {start: 0.0}
    visited: set[str] = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path: list[str] = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        for neighbor in active.neighbors(current):
            edge_data = active[current][neighbor]
            tentative_g = g_score[current] + float(edge_data["weight"])

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(env, neighbor, goal)
                counter += 1
                heapq.heappush(open_set, (f_score, counter, neighbor))

    return [], math.inf


def path_for_dispatch(env, hub_id: str, zone_id: str):
    """
    Proposal step (2): optimal route for a CSP dispatch assignment.

    Call this after ``solve_csp`` returns ``(zone_id, hub_id, resource)``: it runs A*
    from the chosen hub to the chosen zone on the current road network (including
    any blocked roads removed via ``DisasterEnvironment.get_active_graph``).

    Returns
    -------
    path : list[str]
        Hub → … → zone, inclusive, or [] if no path exists.
    cost : float
        Total travel cost along the path, or ``math.inf`` if unreachable.
    """
    return astar(env, hub_id, zone_id)


def replan(env, truck, goal):
    """
    Proposal steps (4)–(5): dynamic replanning when conditions change en route.

    Re-runs A* from ``truck.current_node`` to ``goal`` (original zone) using the
    updated graph so edge weight / blockage changes are respected.
    """
    print(f"  [A*] Replanning for {truck.truck_id} from {truck.current_node} to {goal}")
    new_path, cost = astar(env, truck.current_node, goal)

    if not new_path:
        print(f"  [A*] No path found from {truck.current_node} to {goal} — delivery aborted")
    else:
        print(f"  [A*] New path: {' -> '.join(new_path)}  (cost={cost:.2f})")

    return new_path, cost


def path_uses_edge(path, node_a, node_b):
    """True if ``path`` traverses the undirected edge between ``node_a`` and ``node_b``."""
    for i in range(len(path) - 1):
        if (path[i] == node_a and path[i + 1] == node_b) or (
            path[i] == node_b and path[i + 1] == node_a
        ):
            return True
    return False

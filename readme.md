# csc520 project
astar.py — astar(env, start, goal) runs A* on the active (non-blocked) graph using Euclidean straight-line distance as the admissible heuristic. 

replan(env, truck, goal) re-runs A* from the truck's current node when a road becomes blocked mid-route. path_uses_edge() detects whether a new blockage affects a truck in transit.

csp_solver.py — Encodes all hard and soft constraints from your proposal. Hard constraints (HC1–HC4) are checked before any assignment is considered. Soft constraints (urgency, travel distance, hub inventory balance) are combined into a score and used to rank candidates. solve_next_dispatch() runs the backtracking search with constraint propagation and returns the best assignment(s).

agent.py — The main simulation loop. Each cycle: CSP picks assignments → A* routes trucks → trucks move step-by-step → blockages trigger replanning → delivery updates zone needs. Collects metrics throughout.

baselines.py — GreedyBaseline (highest urgency → nearest hub, no constraint awareness) and RandomBaseline (random zone/hub/resource) for benchmarking.

main.py — Run python main.py for a demo, or python main.py benchmark for a side-by-side comparison table of all three strategies.

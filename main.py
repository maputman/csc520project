"""
main.py – Entry point for the Disaster Relief Resource Allocation project.

Runs three scenarios:
  1. CSP + A* agent  (your proposed system)
  2. Greedy baseline
  3. Random baseline

Then prints a side-by-side comparison table.

Edit scenario size in scenario.py (NUM_HUBS / NUM_ZONES / NUM_TRUCKS).

Commands::

  python main.py              # CSP + A* agent (text)
  python main.py benchmark    # compare CSP vs baselines
  python main.py visualize    # same CSV scenario as demo, matplotlib animation
  python main.py viz          # alias for visualize
"""

from __future__ import annotations

from agent import DisasterReliefAgent
from baselines import GreedyBaseline, RandomBaseline
from scenario import make_env

# Re-export so `from main import NUM_HUBS` still works if needed
from scenario import NUM_HUBS, NUM_TRUCKS, NUM_ZONES  # noqa: F401

DYNAMIC_ROADBLOCK_CHANCE = 0.08


# ---------------------------------------------------------------------------
# Optional: inject mid-simulation events to test dynamic replanning
# ---------------------------------------------------------------------------

def register_events(env):
    """
    Returns a dict mapping time_step -> list of callables that mutate env.
    The simulation runner calls these at the right step.
    """
    return {
        2: [lambda e: e.block_road("H1", "Z1")],
        4: [lambda e: e.add_distress_call(
                "Z6", x=4, y=7, urgency=8,
                needs={"water": 0, "food": 2, "medical": 3},
                critical_resource="water")],
        6: [lambda e: e.unblock_road("H1", "Z1")],
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark():
    results = {}

    # ---- 1. CSP + A* agent ------------------------------------------------
    print("\n" + "#" * 60)
    print("  RUNNING: CSP + A* Agent")
    print("#" * 60)
    env_agent = make_env()
    agent = DisasterReliefAgent(
        env_agent, max_steps=50, verbose=True,
        events=register_events(env_agent),
        dynamic_roadblock_chance=DYNAMIC_ROADBLOCK_CHANCE,
    )
    results["CSP + A*"] = agent.run()

    # ---- 2. Greedy baseline -----------------------------------------------
    print("\n" + "#" * 60)
    print("  RUNNING: Greedy Baseline")
    print("#" * 60)
    env_greedy = make_env()
    greedy = GreedyBaseline(
        env_greedy,
        max_steps=50,
        verbose=False,
        events=register_events(env_greedy),
        dynamic_roadblock_chance=DYNAMIC_ROADBLOCK_CHANCE,
    )
    results["Greedy"] = greedy.run()

    # ---- 3. Random baseline -----------------------------------------------
    print("\n" + "#" * 60)
    print("  RUNNING: Random Baseline")
    print("#" * 60)
    env_random = make_env()
    rand = RandomBaseline(
        env_random,
        max_steps=50,
        seed=7,
        verbose=False,
        events=register_events(env_random),
        dynamic_roadblock_chance=DYNAMIC_ROADBLOCK_CHANCE,
    )
    results["Random"] = rand.run()

    # ---- Comparison table -------------------------------------------------
    print("\n\n" + "=" * 65)
    print(f"  {'METRIC':<28} {'CSP + A*':>10} {'Greedy':>10} {'Random':>10}")
    print("=" * 65)
    metrics_to_show = [
        ("total_steps",       "Steps taken"),
        ("deliveries_made",   "Deliveries made"),
        ("zones_served",      "Zones fully served"),
        ("replan_events",     "Replan events"),
        ("failed_deliveries", "Failed deliveries"),
        ("total_travel_cost", "Total travel cost"),
    ]
    for key, label in metrics_to_show:
        vals = {name: m.get(key, 0) for name, m in results.items()}
        row = f"  {label:<28}"
        for name in ["CSP + A*", "Greedy", "Random"]:
            v = vals[name]
            row += f" {v:>10.2f}" if isinstance(v, float) else f" {v:>10}"
        print(row)
    print("=" * 65)


# ---------------------------------------------------------------------------
# Quick smoke-test: just run the agent on the default scenario
# ---------------------------------------------------------------------------

def demo():
    env = make_env()
    agent = DisasterReliefAgent(
        env,
        max_steps=50,
        verbose=True,
        events=register_events(env),
        dynamic_roadblock_chance=DYNAMIC_ROADBLOCK_CHANCE,
    )
    agent.run()


def visualize():
    """Open the graph animation for the same ``make_env()`` scenario as ``demo()``."""
    from visualization import run_scenario_animation

    env = make_env()
    run_scenario_animation(
        env,
        events=register_events(env),
        dynamic_roadblock_chance=DYNAMIC_ROADBLOCK_CHANCE,
        max_steps=50,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if cmd == "benchmark":
        benchmark()
    elif cmd in ("visualize", "viz"):
        visualize()
    else:
        demo()

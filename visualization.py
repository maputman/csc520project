"""
Animated graph visualization for DisasterEnvironment.

Running this file (``python visualization.py``) uses the same CSP + A* replay as
``python main.py visualize``: zones change color (critical / active / served),
truck motion along planned routes, and a per-dispatch status line.

Scenario size (hubs, zones, trucks) comes from ``scenario.py`` (``NUM_HUBS``,
``NUM_ZONES``, ``NUM_TRUCKS``) via ``make_env()``.

Helper functions such as ``tour_path_from_simulation`` / ``animate_path`` are
still available for tour-style demos if you call them from code.
"""

from __future__ import annotations

import copy

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

from environment import DisasterEnvironment


def _routing_graph(env: DisasterEnvironment) -> nx.Graph:
    """Graph used for paths: same connectivity the simulation uses (no blocked edges)."""
    return env.get_active_graph()


def assert_path_on_graph(env: DisasterEnvironment, path: list[str]) -> None:
    """Every step must follow an existing road."""
    if len(path) < 2:
        return
    G = _routing_graph(env)
    for u, v in zip(path, path[1:]):
        if not G.has_edge(u, v):
            raise ValueError(
                f"Invalid path: no road between {u!r} and {v!r}. "
                "Consecutive nodes must be neighbors in the graph."
            )


def interpolate_along_path(
    env: DisasterEnvironment,
    path: list[str],
    steps_per_edge: int = 28,
) -> list[tuple[float, float]]:
    """Linear interpolation along each edge; avoids duplicating joint vertices."""
    if len(path) < 2:
        x, y = env.get_node_coords(path[0])
        return [(x, y)]

    coords: list[tuple[float, float]] = []
    pos = {n: env.get_node_coords(n) for n in path}

    for i in range(len(path) - 1):
        x0, y0 = pos[path[i]]
        x1, y1 = pos[path[i + 1]]
        for s in range(steps_per_edge):
            if i > 0 and s == 0:
                continue
            t = s / (steps_per_edge - 1) if steps_per_edge > 1 else 1.0
            coords.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return coords


def draw_static_graph(
    ax,
    env: DisasterEnvironment,
    *,
    title: str = "Disaster Region — animated route",
) -> None:
    """Plot nodes and edges like environment.py (nx.draw + lightblue/salmon)."""
    G = env.graph
    pos = {node: (data["x"], data["y"]) for node, data in G.nodes(data=True)}
    node_colors = [
        "lightblue" if G.nodes[n]["type"] == "hub" else "salmon" for n in G.nodes
    ]
    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_colors,
        node_size=1000,
        font_size=10,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)


def graph_axis_limits(env: DisasterEnvironment, margin: float = 0.8) -> tuple[float, float, float, float]:
    """Bounds that include every node (so hubs off the animation path stay visible)."""
    G = env.graph
    xs = [G.nodes[n]["x"] for n in G.nodes]
    ys = [G.nodes[n]["y"] for n in G.nodes]
    return (
        min(xs) - margin,
        max(xs) + margin,
        min(ys) - margin,
        max(ys) + margin,
    )


def _format_visited_nodes(path: list[str]) -> str:
    """One line with arrows if short; otherwise a numbered list."""
    joined = " → ".join(path)
    if len(joined) <= 64:
        return joined
    return "\n".join(f"{i + 1}. {n}" for i, n in enumerate(path))


def animate_path(
    env: DisasterEnvironment,
    path: list[str],
    *,
    steps_per_edge: int = 28,
    interval_ms: int = 35,
    trail_length: int = 18,
    title: str | None = None,
    caption_footer: str | None = None,
) -> FuncAnimation:
    """
    Animate a dot moving along `path` (node ids in order), once (no loop).

    On the last frame, the fading trail clears and the full route is drawn along
    graph edges so the path stays visible. trail_length: past positions in the
    moving tail (0 = dot only) before the final highlight.
    """
    assert_path_on_graph(env, path)

    frames = interpolate_along_path(env, path, steps_per_edge=steps_per_edge)
    if not frames:
        raise ValueError("No frames to animate.")

    # Polyline through path vertices (highlights edges taken after animation ends)
    path_x = [env.get_node_coords(n)[0] for n in path]
    path_y = [env.get_node_coords(n)[1] for n in path]

    fig, ax = plt.subplots(figsize=(9, 7))
    plot_title = title if title is not None else "Disaster Region — animated route"
    draw_static_graph(ax, env, title=plot_title)

    visit_body = _format_visited_nodes(path)
    cap_lines = f"Nodes visited ({len(path)})\n{visit_body}"
    if caption_footer:
        cap_lines += f"\n{caption_footer}"
    ax.text(
        0.02,
        0.98,
        cap_lines,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        zorder=20,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", alpha=0.92),
        clip_on=False,
    )

    # Moving marker (above nx.draw artists)
    (dot,) = ax.plot([], [], "o", color="#1c71d8", markersize=14, zorder=10, label="vehicle")
    (trail_line,) = ax.plot(
        [], [], "-", color="#1c71d8", alpha=0.45, linewidth=3, zorder=9
    )
    (path_highlight,) = ax.plot(
        [],
        [],
        "-",
        color="#1c71d8",
        linewidth=5,
        alpha=0.65,
        zorder=8,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    xs = [p[0] for p in frames]
    ys = [p[1] for p in frames]
    last_i = len(frames) - 1

    xmin, xmax, ymin, ymax = graph_axis_limits(env)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    def init():
        dot.set_data([], [])
        trail_line.set_data([], [])
        path_highlight.set_data([], [])
        return dot, trail_line, path_highlight

    def update(i: int):
        x, y = xs[i], ys[i]
        dot.set_data([x], [y])
        at_end = i == last_i

        if at_end:
            trail_line.set_data([], [])
            if len(path) >= 2:
                path_highlight.set_data(path_x, path_y)
            else:
                path_highlight.set_data([], [])
        else:
            path_highlight.set_data([], [])
            if trail_length > 0:
                i0 = max(0, i - trail_length + 1)
                trail_line.set_data(xs[i0 : i + 1], ys[i0 : i + 1])
            else:
                trail_line.set_data([], [])

        return dot, trail_line, path_highlight

    # blit=False: keeps full nx.draw background stable; blit can interact badly with many artists
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    return anim


def demo_path_for_env(env: DisasterEnvironment) -> list[str]:
    """
    Shortest weighted route from a hub to a zone along real roads (graph edges).

    Alphabetically sorting hub/zone ids does *not* imply adjacency; we use
    ``networkx.shortest_path`` on the active road network instead.
    """
    G = _routing_graph(env)
    hubs = [n for n, d in env.graph.nodes(data=True) if d["type"] == "hub"]
    zones = [n for n, d in env.graph.nodes(data=True) if d["type"] == "zone"]
    if not hubs or not zones:
        return []

    hubs_s = sorted(hubs)
    zones_s = sorted(zones)

    # Prefer a path from first hub lexicographically to a zone that reaches it
    for start in hubs_s:
        for end in reversed(zones_s):
            if nx.has_path(G, start, end):
                return nx.shortest_path(G, start, end, weight="weight")

    for start in hubs_s:
        for end in zones_s:
            if nx.has_path(G, start, end):
                return nx.shortest_path(G, start, end, weight="weight")

    return []


def _format_tour_caption(result: dict, truck_id: str) -> str:
    visits = result.get("visits") or []
    deliv = result.get("delivered") or {}
    left = result.get("cargo_left") or {}
    lines = [f"Truck {truck_id}: {len(visits)} delivery stop(s)"]
    dparts = [f"{k}:{v}" for k, v in deliv.items() if v]
    if dparts:
        lines.append("Delivered: " + ", ".join(dparts))
    if any(left.get(r, 0) for r in ("water", "food", "medical")):
        lines.append(f"Remaining cargo: {left}")
    return "\n".join(lines)


def tour_path_from_simulation(env: DisasterEnvironment, truck_id: str) -> tuple[list[str], dict]:
    """
    Run ``run_truck_delivery_tour`` on a deep copy so the original ``env`` is unchanged.
    Returns ``(full_node_path, result_dict)``.
    """
    sim = copy.deepcopy(env)
    result = sim.run_truck_delivery_tour(truck_id, verbose=False)
    return result.get("full_node_path") or [], result


def _expand_path_on_active_graph(env: DisasterEnvironment, path: list[str]) -> list[str]:
    """
    Ensure consecutive nodes are graph neighbors by splicing shortest paths on the
    active (non-blocked) graph. Keeps the animation from drawing chords across the map.
    """
    if len(path) < 2:
        return list(path)
    G = env.get_active_graph()
    out: list[str] = [path[0]]
    for v in path[1:]:
        u = out[-1]
        if u == v:
            continue
        if G.has_edge(u, v):
            out.append(v)
            continue
        if u not in G or v not in G or not nx.has_path(G, u, v):
            # No valid road path between these nodes; stop the animated path here
            # so the truck never appears to drive along a non-existent edge.
            break
        seg = nx.shortest_path(G, u, v, weight="weight")
        out.extend(seg[1:])
    return out


def _show_run_summary_popup(
    *,
    metrics: dict,
    total_zones: int,
    total_trucks: int,
    unique_blocked_edges: set[tuple[str, str]],
    unserved_zone_ids: list[str],
) -> None:
    """Show a full-screen summary panel after animation playback completes."""
    fig, ax = plt.subplots(figsize=(16, 10))
    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "full_screen_toggle"):
            manager.full_screen_toggle()
    except Exception:
        pass

    ax.set_axis_off()
    ax.set_title("Disaster Relief Run Summary", fontsize=28, fontweight="bold", pad=24)

    zones_served = int(metrics.get("zones_served", 0))
    total_steps = int(metrics.get("total_steps", 0))
    deliveries = int(metrics.get("deliveries_made", 0))
    replans = int(metrics.get("replan_events", 0))
    failed = int(metrics.get("failed_deliveries", 0))
    travel_cost = float(metrics.get("total_travel_cost", 0.0))

    lines = [
        f"Total steps (time): {total_steps}",
        f"Total path cost: {travel_cost:.2f}",
        f"Zones served: {zones_served} / {total_zones}",
        f"Trucks active in scenario: {total_trucks}",
        f"Deliveries made: {deliveries}",
        f"Replan events: {replans}",
        f"Failed deliveries: {failed}",
        f"Blocked roads encountered (unique): {len(unique_blocked_edges)}",
    ]
    if unserved_zone_ids:
        lines.append(f"Unserved zones: {', '.join(unserved_zone_ids)}")
    else:
        lines.append("Unserved zones: none (all served)")

    body = "\n".join(lines)
    ax.text(
        0.5,
        0.5,
        body,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=20,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#333333", alpha=0.98),
    )
    plt.tight_layout()
    plt.show()


def run_scenario_animation(
    env,
    *,
    events=None,
    dynamic_roadblock_chance=0.0,
    max_steps=50,
    agent_cls=None,
    agent_kwargs=None,
    run_label="CSP + A*",
    frame_label_prefix="[CSP+A*]",
    plot_title=None,
):
    """
    Run an agent on ``env`` and animate its deliveries.

    By default runs ``DisasterReliefAgent`` (CSP + A*). Pass ``agent_cls`` (e.g.
    ``GreedyBaseline`` or ``RandomBaseline`` from ``baselines``) and optional
    ``agent_kwargs`` to visualize a baseline with the same animation pipeline.

    Shows:
      - Trucks moving along A*-computed routes (per-truck colors when multiple trucks)
      - Zone node colors updating in real time:
          red   = critical zone (zero supply of a resource)
          orange = active zone (has unmet needs)
          green  = served zone (all needs met)
          blue   = supply hub
      - Status box showing the current dispatch decision
      - Path highlight after each delivery completes

    The agent is constructed with ``record_paths=True`` so each leg uses the
    **walked** hub→…→zone sequence (including A* replans), not the original planned
    path alone — otherwise the dot can cut across non-edges after replanning.
    """
    import copy
    from agent import DisasterReliefAgent

    if agent_cls is None:
        agent_cls = DisasterReliefAgent
    kw = dict(agent_kwargs or {})
    if "verbose" not in kw:
        kw["verbose"] = False
    kw["record_paths"] = True
    kw["events"] = events
    kw["dynamic_roadblock_chance"] = dynamic_roadblock_chance

    sim_env = copy.deepcopy(env)
    print(f"[VIZ] Running {run_label} agent to record dispatch history...")
    agent = agent_cls(sim_env, max_steps=max_steps, **kw)
    agent.run()

    dispatch_log = agent.delivery_paths

    if not dispatch_log:
        print("[VIZ] No dispatches recorded — nothing to animate")
        return None

    print(f"[VIZ] Recorded {len(dispatch_log)} delivery path(s) — building animation...")
 
    # Replay deliveries on a fresh copy to get zone snapshots
    # snapshots[i] = zone states AFTER dispatch i-1 (snapshots[0] = initial)
    snap_env = copy.deepcopy(env)
 
    def _zone_states(e):
        out = {}
        for zid, z in e.zones.items():
            if z.served:
                out[zid] = "served"
            elif z.is_critical():
                out[zid] = "critical"
            else:
                out[zid] = "active"
        return out
 
    snapshots = [_zone_states(snap_env)]
    blocked_snapshots: list[list[tuple[str, str]]] = [[]]
    for d in dispatch_log:
        z = snap_env.zones.get(d["zone_id"])
        h = snap_env.hubs.get(d["hub_id"])
        if z and h:
            try:
                h.dispatch(d["resource"], d["amount"])
            except Exception:
                pass
            z.update_needs(d["resource"], d["amount"])
        snapshots.append(_zone_states(snap_env))
        blocked_snapshots.append(d.get("blocked_edges", blocked_snapshots[-1]))
    unique_blocked_edges = {
        tuple(edge)
        for snapshot in blocked_snapshots
        for edge in snapshot
    }
 
    # Build per-node positions and color helpers
    G = env.graph
    node_list = list(G.nodes())
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_list}
 
    NODE_COLORS = {
        "hub":      "#aed6f1",   # light blue
        "critical": "#e74c3c",   # red
        "active":   "#f0b27a",   # orange
        "served":   "#58d68d",   # green
    }
 
    def _node_color_list(state_idx):
        state = snapshots[min(state_idx, len(snapshots) - 1)]
        return [
            NODE_COLORS["hub"] if G.nodes[n]["type"] == "hub"
            else NODE_COLORS.get(state.get(n, "active"), NODE_COLORS["active"])
            for n in node_list
        ]

    def _blocked_polyline(state_idx):
        edges = blocked_snapshots[min(state_idx, len(blocked_snapshots) - 1)]
        xs: list[float] = []
        ys: list[float] = []
        for u, v in edges:
            if u in pos and v in pos:
                xs.extend([pos[u][0], pos[v][0], float("nan")])
                ys.extend([pos[u][1], pos[v][1], float("nan")])
        return xs, ys
 
    # Build global frame list
    # Each frame: truck position + which snapshot to show + status label
    STEPS_PER_EDGE = 24
    TRAIL_LEN = 16
 
    global_frames = []   # list of dicts
    truck_ids = sorted({d["truck_id"] for d in dispatch_log})
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("tab20").colors)
    truck_colors = {
        truck_id: palette[i % len(palette)]
        for i, truck_id in enumerate(truck_ids)
    }
 
    for di, d in enumerate(dispatch_log):
        path = _expand_path_on_active_graph(env, d["path"])
        if len(path) < 2:
            continue

        # Interpolate (x,y) positions along path using node coordinates
        interp = []
        for i in range(len(path) - 1):
            x0, y0 = pos[path[i]]
            x1, y1 = pos[path[i + 1]]
            for s in range(STEPS_PER_EDGE):
                if i > 0 and s == 0:
                    continue   # avoid duplicate joint points
                t = s / (STEPS_PER_EDGE - 1)
                interp.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
 
        ph_x = [pos[n][0] for n in path]
        ph_y = [pos[n][1] for n in path]
        label = (f"{frame_label_prefix} t={d['time_step']} | "
                 f"{d['truck_id']}: {d['resource']} → {d['zone_id']}")
 
        for fi, (x, y) in enumerate(interp):
            is_last = fi == len(interp) - 1
            global_frames.append({
                "x":         x,
                "y":         y,
                "truck_id":  d["truck_id"],
                "dispatch":  di,
                "state_idx": di + 1 if is_last else di,
                "label":     label,
                "ph_x":      ph_x if is_last else None,
                "ph_y":      ph_y if is_last else None,
            })
 
    if not global_frames:
        print("[VIZ] No valid paths to animate.")
        return None
 
    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_axis_off()
    if plot_title is None:
        plot_title = f"Disaster Relief — {run_label}"
    ax.set_title(plot_title, fontsize=13, fontweight="bold", pad=12)
 
    # Draw edges once (static background)
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], "-", color="#aaaaaa", linewidth=1.5, zorder=1)
 
    # Draw nodes as a scatter plot (colors updated each frame via set_facecolor)
    xs = [pos[n][0] for n in node_list]
    ys = [pos[n][1] for n in node_list]
    scatter = ax.scatter(xs, ys, c=_node_color_list(0), s=900, zorder=4,
                         edgecolors="#333333", linewidth=1.2)
 
    # Draw node labels (static)
    for n in node_list:
        ax.text(pos[n][0], pos[n][1], n, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=5)
 
    # Axis bounds
    xmin, xmax, ymin, ymax = graph_axis_limits(env)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
 
    # Animated artists
    default_truck_color = truck_colors[truck_ids[0]] if truck_ids else "#1c71d8"
    (dot,)       = ax.plot([], [], "o",  color=default_truck_color, markersize=14, zorder=10)
    (trail_line,)= ax.plot([], [], "-",  color=default_truck_color, alpha=0.4, linewidth=3, zorder=9)
    (highlight,) = ax.plot([], [], "-",  color=default_truck_color, linewidth=4,
                           alpha=0.65, zorder=8,
                           solid_capstyle="round", solid_joinstyle="round")
    (blocked_line,) = ax.plot(
        [],
        [],
        "-",
        color="#e74c3c",
        linewidth=4,
        alpha=0.9,
        zorder=7,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
 
    status_box = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, fontsize=8,
        verticalalignment="top", family="monospace", zorder=20,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#333333", alpha=0.92),
    )
 
    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=NODE_COLORS["hub"],      label="Supply Hub"),
        Patch(facecolor=NODE_COLORS["critical"],  label="Critical Zone"),
        Patch(facecolor=NODE_COLORS["active"],    label="Active Zone"),
        Patch(facecolor=NODE_COLORS["served"],    label="Served Zone"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=3, label="Blocked Road"),
    ]
    legend_elems.extend(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=truck_colors[tid],
            markersize=9,
            label=tid,
        )
        for tid in truck_ids
    )
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8, framealpha=0.9)
 
    gxs = [f["x"] for f in global_frames]
    gys = [f["y"] for f in global_frames]
 
    # FuncAnimation
    def init():
        dot.set_data([], [])
        trail_line.set_data([], [])
        highlight.set_data([], [])
        blocked_line.set_data([], [])
        status_box.set_text("")
        return dot, trail_line, highlight, blocked_line, status_box, scatter
 
    def update(i):
        frame = global_frames[i]
 
        # Update zone node colors
        scatter.set_facecolor(_node_color_list(frame["state_idx"]))
        bx, by = _blocked_polyline(frame["state_idx"])
        blocked_line.set_data(bx, by)
 
        # Truck position
        truck_color = truck_colors.get(frame["truck_id"], default_truck_color)
        dot.set_color(truck_color)
        trail_line.set_color(truck_color)
        highlight.set_color(truck_color)
        dot.set_data([frame["x"]], [frame["y"]])
 
        # Fading trail — only within the same dispatch leg so the line
        # never stretches across non-existent edges between deliveries.
        cur_dispatch = frame["dispatch"]
        i0 = i
        while i0 > 0 and (i - i0 + 1) < TRAIL_LEN and global_frames[i0 - 1]["dispatch"] == cur_dispatch:
            i0 -= 1
        trail_line.set_data(gxs[i0: i + 1], gys[i0: i + 1])
 
        # Path highlight on last frame of each dispatch
        if frame["ph_x"] is not None:
            highlight.set_data(frame["ph_x"], frame["ph_y"])
        else:
            highlight.set_data([], [])
 
        status_box.set_text(frame["label"])
        return dot, trail_line, highlight, blocked_line, status_box, scatter
 
    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(global_frames), interval=40,
        blit=False, repeat=False,
    )
 
    plt.tight_layout()
    plt.show()
    _show_run_summary_popup(
        metrics=agent.metrics,
        total_zones=len(sim_env.zones),
        total_trucks=len(sim_env.trucks),
        unique_blocked_edges=unique_blocked_edges,
        unserved_zone_ids=sorted(z.zone_id for z in sim_env.get_unserved_zones()),
    )
    return anim
 
 
if __name__ == "__main__":
    import sys

    from main import DYNAMIC_ROADBLOCK_CHANCE, register_events
    from scenario import make_env

    env = make_env()
    if not env.trucks:
        raise SystemExit("No trucks in scenario — check trucks.csv and NUM_TRUCKS.")

    mode = (sys.argv[1].lower() if len(sys.argv) > 1 else "csp")
    agent_kwargs = None
    if mode in ("greedy", "g"):
        from baselines import GreedyBaseline

        agent_cls = GreedyBaseline
        run_label = "Greedy baseline"
        frame_label_prefix = "[GREEDY]"
    elif mode in ("random", "rand", "r"):
        from baselines import RandomBaseline

        agent_cls = RandomBaseline
        run_label = "Random baseline"
        frame_label_prefix = "[RANDOM]"
        agent_kwargs = {"seed": 7}
    else:
        agent_cls = None
        run_label = "CSP + A*"
        frame_label_prefix = "[CSP+A*]"

    run_scenario_animation(
        env,
        events=register_events(env),
        dynamic_roadblock_chance=DYNAMIC_ROADBLOCK_CHANCE,
        max_steps=50,
        agent_cls=agent_cls,
        agent_kwargs=agent_kwargs,
        run_label=run_label,
        frame_label_prefix=frame_label_prefix,
    )

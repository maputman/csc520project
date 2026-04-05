"""
Animated graph visualization for DisasterEnvironment.

By default, runs a simulated multi-stop delivery tour (load at hub, visit zones
until the truck is empty) on a copy of the scenario, then animates the full
route on the original map. Run: python visualization.py

Does not use CSP or A*.
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


if __name__ == "__main__":
    from scenario import make_env

    env = make_env()
    if not env.trucks:
        raise SystemExit("No trucks in scenario — check trucks.csv and NUM_TRUCKS.")

    truck_id = sorted(env.trucks.keys())[0]
    path, tour_result = tour_path_from_simulation(env, truck_id)
    caption_footer: str | None = None
    plot_title: str | None = None

    if len(path) >= 2:
        caption_footer = _format_tour_caption(tour_result, truck_id)
        plot_title = f"Disaster Region — delivery tour ({truck_id})"
    else:
        path = demo_path_for_env(env)
        if len(path) < 2:
            nh = sum(1 for _, d in env.graph.nodes(data=True) if d.get("type") == "hub")
            nz = sum(1 for _, d in env.graph.nodes(data=True) if d.get("type") == "zone")
            raise SystemExit(
                f"No animation path: tour was empty (no cargo after load at hub?) and no "
                f"hub→zone shortest path. Hubs={nh}, zones={nz}. Check CSVs / scenario limits."
            )
        caption_footer = (
            "Fallback: single shortest hub→zone path (tour had no multi-stop route)."
        )
        plot_title = "Disaster Region — animated route (fallback)"

    # Slightly faster stepping when the tour visits many edges
    steps = 24 if len(path) > 25 else 32
    anim = animate_path(
        env,
        path,
        steps_per_edge=steps,
        interval_ms=28,
        trail_length=22,
        title=plot_title,
        caption_footer=caption_footer,
    )
    plt.show()


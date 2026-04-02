"""
Animated graph visualization for DisasterEnvironment.

Draws hubs/zones and roads, then moves a dot along a sequence of nodes
(interpolated along each edge). Run: python visualize.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

from environment import DisasterEnvironment, build_scenario


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


def draw_static_graph(ax, env: DisasterEnvironment) -> None:
    """Plot nodes and edges using the environment's x,y layout."""
    G = env.graph
    pos = {n: (G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes}

    # Edges: blocked vs open
    blocked = env.blocked_edges
    edge_open = [(u, v) for u, v in G.edges() if (u, v) not in blocked and (v, u) not in blocked]
    edge_blocked = [(u, v) for u, v in G.edges() if (u, v) in blocked or (v, u) in blocked]

    nx.draw_networkx_edges(
        G, pos, edgelist=edge_open, ax=ax, edge_color="#555555", width=2.0, alpha=0.85
    )
    if edge_blocked:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edge_blocked,
            ax=ax,
            edge_color="#cc3333",
            width=2.5,
            style="dashed",
            alpha=0.9,
        )

    hubs = [n for n in G.nodes if G.nodes[n]["type"] == "hub"]
    zones = [n for n in G.nodes if G.nodes[n]["type"] == "zone"]
    nx.draw_networkx_nodes(
        G, pos, nodelist=hubs, ax=ax, node_color="#8ec8ff", edgecolors="#1a5fb4",
        node_size=900, linewidths=2,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=zones, ax=ax, node_color="#ffb4a2", edgecolors="#c01c28",
        node_size=850, linewidths=1.5,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Disaster region — animated route")


def animate_path(
    env: DisasterEnvironment,
    path: list[str],
    *,
    steps_per_edge: int = 28,
    interval_ms: int = 35,
    trail_length: int = 18,
) -> FuncAnimation:
    """
    Animate a dot moving along `path` (node ids in order).

    trail_length: number of past positions drawn as a fading tail (0 = dot only).
    """
    frames = interpolate_along_path(env, path, steps_per_edge=steps_per_edge)
    if not frames:
        raise ValueError("No frames to animate.")

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_static_graph(ax, env)

    # Moving marker
    (dot,) = ax.plot([], [], "o", color="#1c71d8", markersize=14, zorder=10, label="vehicle")
    (trail_line,) = ax.plot(
        [], [], "-", color="#1c71d8", alpha=0.45, linewidth=3, zorder=9
    )

    xs = [p[0] for p in frames]
    ys = [p[1] for p in frames]

    def init():
        dot.set_data([], [])
        trail_line.set_data([], [])
        return dot, trail_line

    def update(i: int):
        x, y = xs[i], ys[i]
        dot.set_data([x], [y])
        if trail_length > 0:
            i0 = max(0, i - trail_length + 1)
            trail_line.set_data(xs[i0 : i + 1], ys[i0 : i + 1])
        else:
            trail_line.set_data([], [])
        return dot, trail_line

    margin = 0.8
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=interval_ms,
        blit=True,
    )
    return anim


def demo_path() -> list[str]:
    """Example route that exists in build_scenario(): H1 → Z1 → Z2 → Z3."""
    return ["H1", "Z1", "Z2", "Z3"]


if __name__ == "__main__":
    env = build_scenario()
    path = demo_path()
    anim = animate_path(env, path, steps_per_edge=32, interval_ms=30, trail_length=22)
    plt.show()


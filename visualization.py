"""
Animated graph visualization for DisasterEnvironment.

Draws hubs/zones and roads (same style as environment.py static plot), then
moves a dot along a sequence of nodes. Run: python visualization.py
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
    ax.set_title("Disaster Region — animated route")


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


def animate_path(
    env: DisasterEnvironment,
    path: list[str],
    *,
    steps_per_edge: int = 28,
    interval_ms: int = 35,
    trail_length: int = 18,
) -> FuncAnimation:
    """
    Animate a dot moving along `path` (node ids in order), once (no loop).

    On the last frame, the fading trail clears and the full route is drawn along
    graph edges so the path stays visible. trail_length: past positions in the
    moving tail (0 = dot only) before the final highlight.
    """
    frames = interpolate_along_path(env, path, steps_per_edge=steps_per_edge)
    if not frames:
        raise ValueError("No frames to animate.")

    # Polyline through path vertices (highlights edges taken after animation ends)
    path_x = [env.get_node_coords(n)[0] for n in path]
    path_y = [env.get_node_coords(n)[1] for n in path]

    fig, ax = plt.subplots(figsize=(9, 7))
    draw_static_graph(ax, env)

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


def demo_path() -> list[str]:
    """Example route that exists in build_scenario(): H1 → Z1 → Z2 → Z3."""
    return ["H1", "Z1", "Z2", "Z3"]


if __name__ == "__main__":
    env = build_scenario()
    path = demo_path()
    anim = animate_path(env, path, steps_per_edge=32, interval_ms=30, trail_length=22)
    plt.show()


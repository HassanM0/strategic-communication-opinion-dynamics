"""
export.py — Data export and animation utilities.

Functions
---------
export_history_csv(history, path)
    Save the opinion trajectory as a CSV file.

export_run_json(sim, history, path, bin_indices)
    Save a complete run record (config + trajectory + derived quantities).

animate_opinion_on_network(sim, history, interval_ms, save_path)
    Animate node colours on the network as opinions evolve.
    Displayable in Jupyter via HTML(anim.to_jshtml()).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .simulator import StrategicCODASimulator


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_history_csv(
    history: np.ndarray,
    path: str,
) -> None:
    """Save the private-opinion trajectory to a CSV file.

    The file has columns ``step, agent_1, ..., agent_N`` (agents are
    numbered from 1, as in the paper).

    Parameters
    ----------
    history : np.ndarray, shape (T+1, N)
        Opinion trajectory.
    path : str
        Destination file path (e.g. 'run.csv').
    """
    history = np.asarray(history, dtype=float)
    N = history.shape[1]

    try:
        import pandas as pd
        df = pd.DataFrame(
            history,
            columns=[f"agent_{i + 1}" for i in range(N)],
        )
        df.index.name = "step"
        df.to_csv(path)
    except ImportError:
        # Fallback: plain numpy savetxt
        header = "step," + ",".join(f"agent_{i + 1}" for i in range(N))
        step_col = np.arange(history.shape[0], dtype=float).reshape(-1, 1)
        data = np.hstack([step_col, history])
        np.savetxt(path, data, delimiter=",", header=header, comments="")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_run_json(
    sim: "StrategicCODASimulator",
    history: np.ndarray,
    path: str,
    bin_indices: np.ndarray | None = None,
) -> None:
    """Save a complete run record to a JSON file.

    The record includes the full SimConfig, derived quantities (ξ, η,
    thresholds τ, reconstruction levels r, W) and the private-opinion
    trajectory; optionally the zero-based bin indices.

    Parameters
    ----------
    sim : StrategicCODASimulator
    history : np.ndarray, shape (T+1, N)
    path : str
        Destination file path (e.g. 'run.json').
    bin_indices : np.ndarray or None
        Bin trajectory from ``run(steps, track_bins=True)``.
    """
    import json

    record: dict = {
        "schema_version": 3,
        "indexing": "zero-based arrays",
        "config": sim.config.to_dict(),
        "metadata": sim.metadata,
        "xi": sim.xi.tolist(),
        "eta_vec": None if sim.eta_vec is None else sim.eta_vec.tolist(),
        "alpha_vec": sim.alpha_vec.tolist(),
        "thresholds_matrix": sim.thresholds_matrix.tolist(),
        "reconstruction_levels_matrix": sim.reconstruction_levels_matrix.tolist(),
        "W": sim.W.tolist(),
        "history": np.asarray(history, dtype=float).tolist(),
    }
    if bin_indices is not None:
        record["bin_indices"] = np.asarray(bin_indices, dtype=int).tolist()

    with open(path, "w") as fh:
        json.dump(record, fh)


# ---------------------------------------------------------------------------
# Network animation
# ---------------------------------------------------------------------------

def animate_opinion_on_network(
    sim: "StrategicCODASimulator",
    history: np.ndarray,
    interval_ms: int = 150,
    save_path: str | None = None,
    figsize: tuple[float, float] = (6, 5),
) -> "matplotlib.animation.FuncAnimation":  # type: ignore[name-defined]
    """Animate node colours on the network graph as opinions evolve.

    The animation can be displayed in a Jupyter notebook via:

        from IPython.display import HTML
        HTML(anim.to_jshtml())

    Parameters
    ----------
    sim : StrategicCODASimulator
    history : np.ndarray, shape (T+1, N)
    interval_ms : int
        Delay between frames in milliseconds.
    save_path : str or None
        If given, save the animation as .gif or .mp4 (requires ffmpeg/pillow).
    figsize : tuple[float, float]

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.animation import FuncAnimation

    cfg = sim.config
    history = np.asarray(history, dtype=float)
    pos = sim.layout_pos or nx.circular_layout(sim.G)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    plt.close(fig)  # prevent immediate display

    def _update(frame: int) -> None:
        ax.clear()
        nx.draw_networkx_edges(
            sim.G, pos, ax=ax, edge_color="#888888", width=0.6, alpha=0.6,
            arrows=cfg.directed,
        )
        node_colors = [cmap(norm(history[frame, i])) for i in range(sim.N)]
        nx.draw_networkx_nodes(
            sim.G, pos, ax=ax,
            node_color=node_colors,
            node_size=max(20, 500 // sim.N),
            linewidths=0.4, edgecolors="#333333",
        )
        ax.set_title(f"t = {frame}  (N={sim.N}, L={sim.L})", fontsize=10)
        ax.axis("off")

    anim = FuncAnimation(
        fig, _update, frames=len(history),
        interval=interval_ms, blit=False,
    )

    if save_path is not None:
        anim.save(save_path)

    return anim

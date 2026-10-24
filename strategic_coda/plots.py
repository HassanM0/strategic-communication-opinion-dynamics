"""
plots.py — Visualization functions for the strategic-CODA simulator.

Top-level entry points:
  plot_simulation(sim, history, ...)
      Signaling maps s_i / private opinions x_i(t) / [public signals
      s_i(x_i(t))] / network panels.
  plot_public_signal_evolution(sim, bin_indices)
      Public signals q_i(x_i(t)) from tracked bin indices.
  get_agent_colors(N, cmap_name)
      Identity colours used by all panels.

Helper functions are prefixed with underscore and not exported.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection


if TYPE_CHECKING:
    from .simulator import StrategicCODASimulator


def get_agent_colors(
    num_agents: int,
    cmap_name: str = "tab20",
) -> np.ndarray:
    """Return an (N, 4) array of RGBA identity colours, one per agent."""
    N = int(num_agents)
    cmap = plt.get_cmap(cmap_name)
    if N <= 1:
        return np.array([cmap(0.0)])
    return np.array([cmap(i / (N - 1)) for i in range(N)])


# ===========================================================================
# Internal helpers
# ===========================================================================

def _should_show_labels(
    sim: "StrategicCODASimulator",
    graph_type: str,
    label_policy: str = "auto",
) -> bool:
    if label_policy == "always":
        return True
    if label_policy == "never":
        return False
    return (graph_type == "star") or (sim.N <= 25)


def _make_xi_linear_norm(xi_values: np.ndarray):
    """Return a simple linear color norm spanning the finite xi range."""
    xi_values = np.asarray(xi_values, dtype=float)
    finite = xi_values[np.isfinite(xi_values)]
    if finite.size == 0:
        return plt.Normalize(vmin=0.0, vmax=1.0), [0.0, 1.0]

    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if not vmin < vmax:
        pad = max(abs(vmin) * 0.05, 0.5)
        return plt.Normalize(vmin=vmin - pad, vmax=vmax + pad), [vmin]

    return plt.Normalize(vmin=vmin, vmax=vmax), [vmin, vmax]


def _draw_network_identity(
    ax: plt.Axes,
    sim: "StrategicCODASimulator",
    pos: dict,
    show_labels: bool,
    graph_type: str,
) -> Any:
    """Draw the network with nodes coloured by agent identity (fixed colour)."""
    nodelist = list(range(sim.N))
    show_edges = getattr(sim.config, "show_edges", True)
    if show_edges:
        nx.draw_networkx_edges(
            sim.G, pos, ax=ax, edge_color="#555555", width=0.85,
            alpha=0.98, arrows=sim.config.directed,
        )
    show_community = getattr(sim.config, "show_community", False) and graph_type == "karate_club"
    if show_community:
        nodelist_hi = [i for i in range(sim.N) if getattr(sim, 'base_graph', None) is not None and sim.base_graph.nodes[i].get('club', 'Mr. Hi') != 'Officer']
        nodelist_officer = [i for i in range(sim.N) if getattr(sim, 'base_graph', None) is not None and sim.base_graph.nodes[i].get('club', 'Mr. Hi') == 'Officer']
        node_collection = None
        if nodelist_hi:
            node_collection = nx.draw_networkx_nodes(
                sim.G, pos, ax=ax, nodelist=nodelist_hi,
                node_color=[sim.agent_colors[i] for i in nodelist_hi], node_size=95,
                node_shape="o",
                linewidths=0.6, edgecolors="#222222", alpha=1.0,
            )
        if nodelist_officer:
            node_collection_officer = nx.draw_networkx_nodes(
                sim.G, pos, ax=ax, nodelist=nodelist_officer,
                node_color=[sim.agent_colors[i] for i in nodelist_officer], node_size=95,
                node_shape="s",
                linewidths=0.6, edgecolors="#222222", alpha=1.0,
            )
            if node_collection is None:
                node_collection = node_collection_officer
    else:
        node_collection = nx.draw_networkx_nodes(
            sim.G, pos, ax=ax, nodelist=nodelist,
            node_color=sim.agent_colors, node_size=95,
            linewidths=0.6, edgecolors="#222222", alpha=1.0,
        )
    if show_labels:
        nx.draw_networkx_labels(
            sim.G, pos, labels={i: str(i + 1) for i in nodelist},
            ax=ax, font_size=8,
        )
    ax.set_aspect("equal")
    if graph_type == "grid":
        ax.autoscale()
    else:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Identity ({graph_type.replace('_', ' ').title()})"
    )
    return node_collection


def _draw_network_opinion(
    ax: plt.Axes,
    sim: "StrategicCODASimulator",
    pos: dict,
    show_labels: bool,
    graph_type: str,
    fig: plt.Figure,
) -> Any:
    """Draw the network with nodes coloured by final opinion value."""
    nodelist = list(range(sim.N))
    cfg = sim.config
    # Dynamically scale bounds to the actual data, keeping 0 centered to preserve coolwarm logic
    max_abs = max(abs(float(np.min(sim.opinions))), abs(float(np.max(sim.opinions))))
    if max_abs < 1e-6:
        max_abs = 0.1
    node_vmin = -max_abs
    node_vmax = max_abs

    show_edges = getattr(cfg, "show_edges", True)
    if show_edges:
        nx.draw_networkx_edges(
            sim.G, pos, ax=ax, edge_color="#555555", width=0.85,
            alpha=0.98, arrows=cfg.directed,
        )
    show_community = getattr(cfg, "show_community", False) and graph_type == "karate_club"
    if show_community:
        nodelist_hi = [i for i in range(sim.N) if getattr(sim, 'base_graph', None) is not None and sim.base_graph.nodes[i].get('club', 'Mr. Hi') != 'Officer']
        nodelist_officer = [i for i in range(sim.N) if getattr(sim, 'base_graph', None) is not None and sim.base_graph.nodes[i].get('club', 'Mr. Hi') == 'Officer']
        node_collection = None
        if nodelist_hi:
            node_collection = nx.draw_networkx_nodes(
                sim.G, pos, ax=ax, nodelist=nodelist_hi,
                node_color=[sim.opinions[i] for i in nodelist_hi], cmap=plt.cm.coolwarm,
                node_size=95, vmin=node_vmin, vmax=node_vmax,
                node_shape="o",
                linewidths=0.6, edgecolors="#222222", alpha=0.98,
            )
        if nodelist_officer:
            node_collection_officer = nx.draw_networkx_nodes(
                sim.G, pos, ax=ax, nodelist=nodelist_officer,
                node_color=[sim.opinions[i] for i in nodelist_officer], cmap=plt.cm.coolwarm,
                node_size=95, vmin=node_vmin, vmax=node_vmax,
                node_shape="s",
                linewidths=0.6, edgecolors="#222222", alpha=0.98,
            )
            if node_collection is None:
                node_collection = node_collection_officer
    else:
        node_collection = nx.draw_networkx_nodes(
            sim.G, pos, ax=ax, nodelist=nodelist,
            node_color=sim.opinions, cmap=plt.cm.coolwarm,
            node_size=95, vmin=node_vmin, vmax=node_vmax,
            linewidths=0.6, edgecolors="#222222", alpha=0.98,
        )
    if show_labels:
        nx.draw_networkx_labels(
            sim.G, pos, labels={i: str(i + 1) for i in nodelist},
            ax=ax, font_size=8,
        )
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.coolwarm,
        norm=plt.Normalize(vmin=node_vmin, vmax=node_vmax),
    )
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.1, label="Opinion")
    ax.set_aspect("equal")
    if graph_type == "grid":
        ax.autoscale()
    else:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"({graph_type.replace('_', ' ').title()})"
    )
    return node_collection


def _draw_network_exag_profile(
    ax: plt.Axes,
    sim: "StrategicCODASimulator",
    pos: dict,
    show_labels: bool,
    graph_type: str,
    fig: plt.Figure,
    inverse_xi: bool = False,
) -> Any:
    """Draw the network with nodes coloured by exaggeration factor."""
    nodelist = list(range(sim.N))
    cfg = sim.config
    xi_values = np.asarray(sim.xi, dtype=float)
    if inverse_xi:
        profile_values = np.divide(
            1.0,
            xi_values,
            out=np.full_like(xi_values, np.nan, dtype=float),
            where=np.isfinite(xi_values) & (xi_values != 0.0),
        )
        cbar_label = r"$1/\xi_i$"
        title_prefix = "1/xi Profile"
    else:
        profile_values = xi_values
        cbar_label = r"$\xi_i$"
        title_prefix = "Exag. Profile"

    cmap = plt.get_cmap("viridis")
    norm, ticks = _make_xi_linear_norm(profile_values)

    node_colors = np.tile(mcolors.to_rgba("#d9d9d9"), (sim.N, 1))
    finite = np.isfinite(profile_values)
    if np.any(finite):
        node_colors[finite] = cmap(norm(profile_values[finite]))

    show_edges = getattr(cfg, "show_edges", True)
    if show_edges:
        nx.draw_networkx_edges(
            sim.G, pos, ax=ax, edge_color="#555555", width=0.85,
            alpha=0.98, arrows=cfg.directed,
        )

    show_community = getattr(cfg, "show_community", False) and graph_type == "karate_club"
    if show_community:
        nodelist_hi = [i for i in range(sim.N) if getattr(sim, 'base_graph', None) is not None and sim.base_graph.nodes[i].get('club', 'Mr. Hi') != 'Officer']
        nodelist_officer = [i for i in range(sim.N) if getattr(sim, 'base_graph', None) is not None and sim.base_graph.nodes[i].get('club', 'Mr. Hi') == 'Officer']
        node_collection = None
        if nodelist_hi:
            node_collection = nx.draw_networkx_nodes(
                sim.G, pos, ax=ax, nodelist=nodelist_hi,
                node_color=node_colors[nodelist_hi],
                node_size=95,
                node_shape="o",
                linewidths=0.6, edgecolors="#222222", alpha=0.98,
            )
        if nodelist_officer:
            node_collection_officer = nx.draw_networkx_nodes(
                sim.G, pos, ax=ax, nodelist=nodelist_officer,
                node_color=node_colors[nodelist_officer],
                node_size=95,
                node_shape="s",
                linewidths=0.6, edgecolors="#222222", alpha=0.98,
            )
            if node_collection is None:
                node_collection = node_collection_officer
    else:
        node_collection = nx.draw_networkx_nodes(
            sim.G, pos, ax=ax, nodelist=nodelist,
            node_color=node_colors,
            node_size=95,
            linewidths=0.6, edgecolors="#222222", alpha=0.98,
        )

    if show_labels:
        nx.draw_networkx_labels(
            sim.G, pos, labels={i: str(i + 1) for i in nodelist},
            ax=ax, font_size=8,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.1, label=cbar_label)
    if ticks:
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(
            [f"{tick:.0f}" if abs(tick) >= 10 else f"{tick:.1f}" for tick in ticks]
        )

    ax.set_aspect("equal")
    if graph_type == "grid":
        ax.autoscale()
    else:
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title_prefix} ({graph_type.replace('_', ' ').title()})"
    )
    return node_collection


# ===========================================================================
# LineCollection for the signaling maps s_i
# ===========================================================================

def _public_signal_line_collection(
    sim: "StrategicCODASimulator",
    line_colors: list | np.ndarray | None = None,
) -> LineCollection:
    """Build a LineCollection with every agent's signaling map s_i."""
    segments = []
    if sim.config.signal_function == "quantized":
        for i in range(sim.N):
            thresh = sim.thresholds_matrix[i]
            levels = sim.reconstruction_levels_matrix[i]
            # Piecewise constant: one horizontal segment per bin.
            x_pts = np.repeat(thresh, 2)[1:-1]  # shape (2L,)
            y_pts = np.repeat(levels, 2)         # shape (2L,)
            segments.append(np.column_stack([x_pts, y_pts]))
    else:
        grid = np.linspace(-1.0, 1.0, 401)
        values = np.array([sim.get_public_signals(np.full(sim.N, x)) for x in grid])
        for i in range(sim.N):
            segments.append(np.column_stack([grid, values[:, i]]))
    return LineCollection(
        segments,
        colors=sim.agent_colors if line_colors is None else line_colors,
        linewidths=1.5,
        alpha=0.85,
    )

# ===========================================================================
# Main simulation plot (3-panel)
# ===========================================================================

def plot_simulation(
    sim: "StrategicCODASimulator",
    history: np.ndarray,
    graph_type: str | None = None,
    show: bool = True,
    return_artists: bool = False,
    fig3_mode: str = "opinion",
    label_policy: str = "auto",
    show_exag_profile: bool = False,
    show_inverse_xi_profile: bool = False,
    layout: str = "horizontal",
    show_signals: bool = False,
    bin_indices: np.ndarray | None = None,
) -> dict | None:
    """Simulation figure: signaling maps / private opinions / network.

    Parameters
    ----------
    sim : StrategicCODASimulator
    history : np.ndarray, shape (T+1, N)
    graph_type : str or None
        If None, uses sim.config.graph_type.
    show : bool
        Call plt.show() at the end.
    return_artists : bool
        Return dict of artist references instead of None.
    fig3_mode : str
        'identity', 'opinion', or 'dual'.
    label_policy : str
        'auto', 'always', or 'never'.
    show_exag_profile : bool
        Append an exaggeration-profile network panel when True.
    show_inverse_xi_profile : bool
        Colour the exaggeration-profile panel by 1/xi_i when True.
    layout : str
        'split' (two figures), 'horizontal', or 'vertical'.
    show_signals : bool
        Add a panel with the public signals s_i(x_i(t)).
    bin_indices : np.ndarray or None
        Tracked zero-based bin indices (speeds up the public-signal panel
        for the quantized map).

    Returns
    -------
    artists dict (if return_artists=True) or None.
    """
    cfg = sim.config
    if graph_type is None:
        graph_type = cfg.graph_type
    history = np.asarray(history)
    # Continuous-time runs: x-axis in real time t = k*dt
    _dt_axis = (float(getattr(cfg, "dt", 1.0))
                if getattr(cfg, "continuous_time", False) else 1.0)

    if fig3_mode not in {"identity", "opinion", "dual"}:
        raise ValueError("fig3_mode must be 'identity', 'opinion', or 'dual'.")
    if label_policy not in {"auto", "always", "never"}:
        raise ValueError("label_policy must be 'auto', 'always', or 'never'.")
    show_exag_profile = bool(show_exag_profile)
    show_inverse_xi_profile = bool(show_inverse_xi_profile)
    base_network_panels = 2 if fig3_mode == "dual" else 1
    network_panel_count = base_network_panels + int(show_exag_profile)
    ax3_xi = None

    base_plot_panels = 3 if show_signals else 2
    row_or_col_offset = 1 if show_signals else 0
    ax_signals = None

    if layout == "split":
        figheight = 10.5 if show_signals else 7.5
        fig = plt.figure(figsize=(4.5, figheight))
        gs = fig.add_gridspec(base_plot_panels, 1, hspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        if show_signals:
            ax_signals = fig.add_subplot(gs[2, 0])
        
        # Second figure for Panel 3
        if fig3_mode == "dual":
            fig_net = plt.figure(figsize=(4.2 * network_panel_count, 3.6))
            gs_net = fig_net.add_gridspec(1, network_panel_count, wspace=0.3)
            ax3_id = fig_net.add_subplot(gs_net[0, 0])
            ax3_op = fig_net.add_subplot(gs_net[0, 1])
            if show_exag_profile:
                ax3_xi = fig_net.add_subplot(gs_net[0, 2])
        else:
            fig_net = plt.figure(figsize=(4.2 * network_panel_count, 3.6))
            gs_net = fig_net.add_gridspec(1, network_panel_count)
            ax3 = fig_net.add_subplot(gs_net[0, 0])
            if show_exag_profile:
                ax3_xi = fig_net.add_subplot(gs_net[0, 1])
    elif layout == "vertical":
        base_height = 12.5 if show_signals else 9
        if fig3_mode == "dual":
            base_height = 17.5 if show_signals else 14
        figheight = base_height + (3.5 if show_exag_profile else 0)
        fig = plt.figure(figsize=(6, figheight))
        row_count = base_plot_panels + network_panel_count
        gs = fig.add_gridspec(row_count, 1, hspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        if show_signals:
            ax_signals = fig.add_subplot(gs[2, 0])
        offset = row_or_col_offset
        if fig3_mode == "dual":
            ax3_id = fig.add_subplot(gs[2 + offset, 0])
            ax3_op = fig.add_subplot(gs[3 + offset, 0])
            if show_exag_profile:
                ax3_xi = fig.add_subplot(gs[4 + offset, 0])
        else:
            ax3 = fig.add_subplot(gs[2 + offset, 0])
            if show_exag_profile:
                ax3_xi = fig.add_subplot(gs[3 + offset, 0])
    else:
        base_width = 20 if show_signals else 16
        if fig3_mode == "dual":
            base_width = 22 if show_signals else 18
        figwidth = base_width + (4 if show_exag_profile else 0)
        fig = plt.figure(figsize=(figwidth, 5))
        col_count = base_plot_panels + network_panel_count
        gs = fig.add_gridspec(1, col_count, width_ratios=[1.0] * col_count)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        if show_signals:
            ax_signals = fig.add_subplot(gs[0, 2])
        offset = row_or_col_offset
        if fig3_mode == "dual":
            ax3_id = fig.add_subplot(gs[0, 2 + offset])
            ax3_op = fig.add_subplot(gs[0, 3 + offset])
            if show_exag_profile:
                ax3_xi = fig.add_subplot(gs[0, 4 + offset])
        else:
            ax3 = fig.add_subplot(gs[0, 2 + offset])
            if show_exag_profile:
                ax3_xi = fig.add_subplot(gs[0, 3 + offset])

    # --- Determine line colors for opinion and quantizer plots ---
    color_community = getattr(cfg, "color_community_opinion", False) and graph_type == "karate_club"
    if color_community:
        line_colors = []
        for i in range(sim.N):
            club = sim.base_graph.nodes[i].get('club', 'Mr. Hi') if getattr(sim, 'base_graph', None) is not None else 'Mr. Hi'
            color = 'royalblue' if club == 'Officer' else 'crimson'
            line_colors.append(color)
    else:
        line_colors = sim.agent_colors

    # --- Panel 1: signaling maps s_i via LineCollection ---
    lc = _public_signal_line_collection(sim, line_colors=line_colors)
    ax1.add_collection(lc)
    ax1.autoscale()
    ax1.plot(
        [-1.0, 1.0], [-1.0, 1.0],
        "k--", alpha=0.35, label="$y=x$" if sim.N <= 12 else None,
    )

    # Title reflects the active signaling map
    signal_fn = cfg.signal_function
    if signal_fn == "quantized":
        family = cfg.quantizer_family
        if family == "uniform":
            ax1.set_title(f"Uniform quantizers $q_i$ (L={sim.L})")
        elif family == "sign":
            ax1.set_title("Sign quantizers (classical CODA)")
        elif family == "custom":
            ax1.set_title(f"Custom quantizers $q_i$ (L={sim.L})")
        else:
            ax1.set_title(f"Credible codes $q_i$ (L={sim.L})")
    elif signal_fn == "linear":
        ax1.set_title("Linear signaling $s_i(x)=x$")
    elif signal_fn == "tanh":
        ax1.set_title(fr"$s_i(x)=\tanh(Kx)$ (K={cfg.tanh_K:g})")
    elif signal_fn == "saturated":
        ax1.set_title(fr"Saturated $s_i(x)=\operatorname{{sat}}(\xi_i x)$ ($\bar{{\xi}}$={np.nanmean(sim.xi):.3f})")
    else:
        ax1.set_title(signal_fn)
    _pad = 0.1
    ax1.set_xlim(-1.0 - _pad, 1.0 + _pad)
    ax1.set_ylim(-1.0 - _pad, 1.0 + _pad)
    ax1.set_xlabel("Private opinion $x$")
    ax1.set_ylabel("Public signal $s_i(x)$")
    if sim.N <= 12:
        ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # Keep individual Line2D references for return_artists compatibility
    public_signal_lines = []
    for i in range(sim.N):
        ln, = ax1.plot([], [], color=line_colors[i], lw=0, alpha=0)
        public_signal_lines.append(ln)

    # --- Panel 2: Opinion evolution ---
    evolution_lines = []
    for i in range(sim.N):
        ln, = ax2.plot(np.arange(history.shape[0]) * _dt_axis, history[:, i],
                       color=line_colors[i], alpha=0.9, lw=1.2)
        evolution_lines.append(ln)

    y_min = min(float(np.min(history)), -1.0)
    y_max = max(float(np.max(history)), 1.0)
    y_pad = 0.05 * max(y_max - y_min, 1e-6)
    if np.allclose(sim.alpha_vec, sim.alpha_vec[0]):
        alpha_summary = fr"$\alpha_i={sim.alpha_vec[0]:.2f}$"
    else:
        alpha_summary = (fr"$\alpha_i\in[{sim.alpha_vec.min():.2f}, "
                         fr"{sim.alpha_vec.max():.2f}]$")
    ax2.set_title(f"Private opinions ({alpha_summary})")

    ax2.set_xlabel("$t$")
    ax2.set_ylabel(r"Private opinion $x_i(t)$")
    ax2.set_ylim(y_min - y_pad, y_max + y_pad)
    ax2.grid(True, alpha=0.25)

    # --- Panel: Public signals ---
    signal_lines = []
    if show_signals and ax_signals is not None:
        T, N = history.shape
        if cfg.signal_function == "quantized" and bin_indices is not None:
            public_signals = np.zeros_like(bin_indices, dtype=float)
            for i in range(N):
                public_signals[:, i] = sim.reconstruction_levels_matrix[
                    i,
                    bin_indices[:, i],
                ]
        else:
            public_signals = np.zeros((T, N), dtype=float)
            for t in range(T):
                public_signals[t, :] = sim.get_public_signals(history[t])

        for i in range(N):
            ln, = ax_signals.plot(np.arange(public_signals.shape[0]) * _dt_axis, public_signals[:, i],
                                  color=line_colors[i], alpha=0.9, lw=1.2, drawstyle='steps-post')
            signal_lines.append(ln)

        ax_signals.set_title("Public signals")
        ax_signals.set_xlabel("$t$")
        ax_signals.set_ylabel(r"Public signal $s_i(x_i(t))$")
        ax_signals.set_ylim(-1.05, 1.05)
        ax_signals.grid(True, alpha=0.25)

    # --- Panel 3: Network ---
    pos = sim.layout_pos or nx.circular_layout(sim.G)
    show_labels = _should_show_labels(sim, graph_type, label_policy)
    fig3_identity_nodes = None
    fig3_axes = []

    if fig3_mode == "identity":
        target_fig = fig_net if layout == "split" else fig
        fig3_identity_nodes = _draw_network_identity(ax3, sim, pos, show_labels, graph_type)
        fig3_axes = [ax3]
    elif fig3_mode == "opinion":
        target_fig = fig_net if layout == "split" else fig
        _draw_network_opinion(ax3, sim, pos, show_labels, graph_type, target_fig)
        fig3_axes = [ax3]
    else:
        target_fig = fig_net if layout == "split" else fig
        fig3_identity_nodes = _draw_network_identity(ax3_id, sim, pos, show_labels, graph_type)
        _draw_network_opinion(ax3_op, sim, pos, show_labels, graph_type, target_fig)
        fig3_axes = [ax3_id, ax3_op]

    if show_exag_profile and ax3_xi is not None:
        _draw_network_exag_profile(
            ax3_xi,
            sim,
            pos,
            show_labels,
            graph_type,
            target_fig,
            inverse_xi=show_inverse_xi_profile,
        )
        fig3_axes.append(ax3_xi)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout.*",
        )
        fig.tight_layout()
        if layout == "split":
            fig_net.tight_layout()

    artists = {
        "fig": fig,
        "public_signal_lines": public_signal_lines,
        "evolution_lines": evolution_lines,
        "signal_lines": signal_lines,
        "fig3_axes": fig3_axes,
        "fig3_mode": fig3_mode,
        "show_exag_profile": show_exag_profile,
        "show_inverse_xi_profile": show_inverse_xi_profile,
        "label_policy": label_policy,
        "show_labels": show_labels,
        "fig3_identity_facecolors": (
            None if fig3_identity_nodes is None
            else fig3_identity_nodes.get_facecolors()
        ),
    }

    if layout == "split":
        artists["fig_net"] = fig_net

    if show:
        if layout == "split":
            plt.figure(fig.number)
            plt.show(block=False)
            plt.figure(fig_net.number)
            plt.show()
        else:
            plt.show()

    if not return_artists:
        plt.close(fig)
        if layout == "split":
            plt.close(fig_net)
        return None
    
    return artists



def plot_public_signal_evolution(
    sim: "StrategicCODASimulator",
    bin_indices: np.ndarray,
    title: str = "Public signals",
) -> None:
    """Plot the public signal q_i(x_i(t)) of every agent over time.

    ``bin_indices`` comes from ``sim.run(steps, track_bins=True)``
    (quantized signaling).
    """
    T, N = bin_indices.shape
    fig, ax = plt.subplots(figsize=(16, 4))

    public_signals = np.zeros_like(bin_indices, dtype=float)
    for i in range(N):
        public_signals[:, i] = sim.reconstruction_levels_matrix[
            i,
            bin_indices[:, i],
        ]

    for i in range(N):
        ax.plot(
            public_signals[:, i],
            color=sim.agent_colors[i],
            alpha=0.9,
            lw=1.2,
            drawstyle="steps-post",
        )

    ax.set_title(f"{title} (N={N})")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"Public signal $q_i(x_i(t))$")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


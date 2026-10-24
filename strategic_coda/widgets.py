"""
widgets.py — ipywidgets interface for the strategic-CODA simulator.

Architecture
------------
  build_ui()         — construct and return the full widget layout + output area
  _collect_config()  — read all widget values into a SimConfig
  _apply_visibility_rules(cfg) — hide/show irrelevant controls declaratively

UI Layout (Tab-based)
---------------------
  Tab 0 "Model"           — N, inertia α (+ heterogeneity), horizon T,
                            signaling map s_i, graph + per-graph params,
                            seed, node layout, perturbation
  Tab 1 "Quantizers"      — family, number of bins L, custom per-agent rules
                            (e.g. '1-11, 14') to thresholds / reconstruction levels
  Tab 2 "Network"         — influence weights, directed toggle, W viewer
  Tab 3 "Initial Cond."   — initial-opinion mode and its parameters
  Tab 4 "Display"         — figure layout, colors, network view, labels, edges
  Tab 5 "Analysis"        — ξ bar chart, public-signal panel
  Tab 6 "Presets"         — load/save JSON presets (presets/ folder)
  Run bar                 — Run button + auto-run toggle
  Status line             — preset load/save messages
  Output areas            — main simulation figure, network figure, W matrix
"""

from __future__ import annotations

import io
import json
import os
import time
import warnings
from contextlib import contextmanager

import ipywidgets as widgets
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from IPython.display import display
from matplotlib.ticker import MaxNLocator

from .config import SimConfig
from .presets import (BUILTIN_PRESETS, list_presets, load_preset,
                      preset_path, save_preset)


# ===========================================================================
# Widget definitions
# ===========================================================================

def _make_widgets() -> dict:
    """Create all ipywidgets and return them in a flat dict keyed by name."""
    W = {}
    _DEF = SimConfig()  # single source of truth for default values

    # ---- Tab 0: Simulation ----
    W["N"] = widgets.IntSlider(min=2, max=500, step=1, value=20,
                               description="Agents", style={"description_width": "90px"})
    W["alpha"] = widgets.FloatSlider(
        min=0.0, max=0.999, step=0.01, value=0.5,
        description="Inertia α",
        style={"description_width": "90px"},
    )
    W["L"] = widgets.IntSlider(min=1, max=50, step=1, value=4,
                               description="Bins (L)", style={"description_width": "90px"})
    W["steps"] = widgets.IntSlider(min=1, max=500, step=1, value=60,
                                   description="Horizon T", style={"description_width": "90px"})
    W["dt"] = widgets.FloatText(
        value=1.0, step=0.05, description="Time step dt",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="200px"),
    )
    W["continuous_time"] = widgets.Checkbox(
        value=False, description="Continuous time (Euler; beyond the paper)",
        style={"description_width": "initial"},
    )
    W["signal_function"] = widgets.Dropdown(
        options=[("Quantized qᵢ (Bayesian receivers)", "quantized"),
                 ("Linear s(x)=x (aligned)", "linear"),
                 ("Saturated sat(ξᵢx) (naive receivers)", "saturated"),
                 ("tanh(Kx) (beyond the paper)", "tanh")],
        value="quantized", description="Signaling sᵢ",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="370px"),
    )
    W["quantizer_family"] = widgets.Dropdown(
        options=[("Credible codes (paper)", "strategic"),
                 ("Uniform rating scale (beyond the paper)", "uniform"),
                 ("Sign, classical CODA (beyond the paper)", "sign"),
                 ("Custom (beyond the paper)", "custom")],
        value="strategic", description="Quantizer",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="370px"),
    )
    W["custom_rules_box"] = widgets.VBox([])
    W["add_rule_btn"] = widgets.Button(
        description="Add rule", icon="plus",
        layout=widgets.Layout(width="120px"),
    )
    W["custom_base"] = widgets.Dropdown(
        options=[("Credible codes", "strategic"),
                 ("Uniform rating scale", "uniform"),
                 ("Sign (classical CODA)", "sign")],
        value="strategic", description="Others use",
        style={"description_width": "90px"},
    )
    W["tanh_K"] = widgets.FloatSlider(
        min=0.01, max=10.0, step=0.01, value=1.0,
        description="tanh K", style={"description_width": "90px"}
    )
    W["graph_type"] = widgets.Dropdown(
        options=[
            ("Watts-Strogatz", "watts_strogatz"),
            ("Complete", "complete"),
            ("Star", "star"),
            ("Wheel", "wheel"),
            ("Line (path)", "line"),
            ("Ring (cycle)", "ring"),
            ("Grid", "grid"),
            ("Stochastic Block", "stochastic_block"),
            ("D-Regular", "d_regular"),
            ("Bipartite", "bipartite"),
            ("Powerlaw Cluster", "powerlaw_cluster"),
            ("Erdős–Rényi", "erdos_renyi"),
            ("Karate Club", "karate_club"),
            ("Petersen", "petersen"),
            ("Dodecahedral", "dodecahedral"),
            ("Two groups", "two_group"),
        ],
        value="star", description="Graph", style={"description_width": "90px"},
    )
    W["regular_degree"] = widgets.IntSlider(min=0, max=19, step=1, value=_DEF.regular_degree,
                                            description="Degree d",
                                            style={"description_width": "90px"})
    W["seed"] = widgets.IntText(value=11, description="Graph Seed",
                                style={"description_width": "90px"})
    W["sbm_sizes"] = widgets.Text(value="10,10", description="SBM Sizes",
                                    style={"description_width": "90px"},
                                    placeholder="e.g. 10,10")
    W["sbm_p_in"] = widgets.FloatSlider(min=0.001, max=1.0, step=0.0001, value=_DEF.sbm_p_in,
                                          description="SBM P(in)",readout_format=".4f", style={"description_width": "90px"})
    W["sbm_p_out"] = widgets.FloatSlider(min=0.0, max=1.0, step=0.0001, value=0.06,
                                           description="SBM P(out)",readout_format=".4f", style={"description_width": "90px"})
    W["ws_k"] = widgets.IntSlider(min=2, max=20, step=1, value=4,
                                  description="K (Degree)", style={"description_width": "90px"})
    W["ws_p"] = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.1,
                                    description="P (Rewire)", style={"description_width": "90px"})
    W["group_sizes"] = widgets.Text(value=_DEF.group_sizes, description="Group sizes",
                                    style={"description_width": "90px"},
                                    placeholder="e.g. 2,3")
    W["cross_group_weight"] = widgets.FloatSlider(
        min=0.0, max=1.0, step=0.01, value=_DEF.cross_group_weight,
        description="Cross weight ρ", style={"description_width": "90px"})
    W["layout_algo"] = widgets.Dropdown(
        options=[
            ("Spring", "spring"), ("Circular", "circular"), ("Shell", "shell"),
            ("Spectral", "spectral"), ("Kamada-Kawai", "kamada_kawai"),
        ],
        value=_DEF.layout_algo, description="Layout",
        style={"description_width": "90px"},
    )
    W["show_edges"] = widgets.Checkbox(
        value=True, description="Show Edges",
        style={"description_width": "90px"},
    )
    W["show_community"] = widgets.Checkbox(
        value=False, description="Show Community",
        style={"description_width": "initial"}
    )
    W["color_community_opinion"] = widgets.Checkbox(
        value=False, description="Color Comm. Opinion",
        style={"description_width": "initial"}
    )
    W["ensure_valid_graph"] = widgets.Checkbox(
        value=False, description="Require connected / strongly connected",
        style={"description_width": "initial"},
    )
    W["model_profile"] = widgets.Dropdown(
        options=[("Extended (all options)", "extended"),
                 ("Paper assumptions", "paper")],
        value=_DEF.model_profile,
        description="Model profile",
        style={"description_width": "90px"},
    )
    W["connectivity_html"] = widgets.HTML(value="")
    W["show_exag_profile"] = widgets.Checkbox(
        value=False, description="ξ-profile network panel",
        style={"description_width": "initial"},
    )
    W["show_inverse_xi_profile"] = widgets.Checkbox(
        value=False, description="Color ξ-panel by 1/ξ",
        style={"description_width": "initial"},
    )
    W["signal_plot_out"] = widgets.Output()

    # ---- Tab 1: Initial Conditions ----
    W["initial_opinion_mode"] = widgets.Dropdown(
        options=[
            ("Random", "random"), ("Symmetric Zero", "symmetric_zero"),
            ("Manual", "manual"), ("Identical", "identical"),
            ("Eigenvector", "eigen"), ("Eigenvector + Noise", "eigen_noise"),
        ],
        value="random", description="Init Mode",
        style={"description_width": "110px"},
    )
    W["opinion_seed"] = widgets.IntText(value=_DEF.opinion_seed, description="Opinion Seed",
                                        style={"description_width": "110px"})
    W["symmetric_scale"] = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.3,
                                               description="Spread (±)",
                                               style={"description_width": "110px"})
    W["initial_opinions_manual"] = widgets.Text(
        value="", description="Opinions (CSV)",
        placeholder="comma-separated N values",
        style={"description_width": "110px"},
    )
    W["identical_value"] = widgets.FloatSlider(
        min=-1.0, max=1.0, step=0.01, value=0.0,
        description="Identical Val",
        style={"description_width": "110px"}
    )
    W["eigen_index"] = widgets.IntSlider(min=0, max=19, step=1, value=1,
                                         description="Eigen Index",
                                         style={"description_width": "110px"})
    W["eigen_scale"] = widgets.FloatSlider(min=0.01, max=1.0, step=0.01, value=0.8,
                                           description="Eigen Scale",
                                           style={"description_width": "110px"})
    W["eigen_noise_snr"] = widgets.FloatSlider(min=0.0, max=40.0, step=0.5, value=10.0,
                                               description="Noise SNR (dB)",
                                               style={"description_width": "110px"})

    # ---- Tab 2: Advanced ----
    W["directed"] = widgets.Checkbox(value=False, description="Directed",
                                     style={"description_width": "110px"})
    W["directed_mode"] = widgets.Dropdown(
        options=[
            ("Strong-preserving asymmetric", "strongly_connected"),
            ("Random Asymmetric (Any)", "random_asymmetric"),
        ],
        value="strongly_connected", description="Directed Mode",
        style={"description_width": "110px"},
    )
    W["weight_mode"] = widgets.Dropdown(
        options=[
            ("Uniform Incoming", "uniform_in"),
            ("Random Incoming", "random_in"),
            ("Out-Degree Prop.", "out_degree"),
            ("In-Degree Prop.", "in_degree"),
            ("Eigenvector Cent.", "eigenvector"),
            ("Karate Weights", "karate_weights"),
            ("Group Weights (ρ)", "group_weights"),
        ],
        value="uniform_in", description="Weights",
        style={"description_width": "110px"},
    )
    W["alpha_mode"] = widgets.Dropdown(
        options=[
            ("Common αᵢ = α", "scalar"),
            ("Heterogeneous random", "heterogeneous-random"),
            ("Manual vector", "manual"),
        ],
        value="scalar", description="Inertia mode",
        style={"description_width": "110px"},
    )
    W["alpha_seed"] = widgets.IntText(value=23, description="α seed",
                                      style={"description_width": "110px"})
    W["alpha_min"] = widgets.FloatSlider(min=0.0, max=0.95, step=0.01, value=0.2,
                                         description="α min",
                                         style={"description_width": "110px"})
    W["alpha_max"] = widgets.FloatSlider(min=0.05, max=0.99, step=0.01, value=0.8,
                                         description="α max",
                                         style={"description_width": "110px"})
    W["alpha_manual"] = widgets.Text(
        value="", description="α list",
        placeholder="comma-separated N values",
        style={"description_width": "110px"},
    )

    # ---- Tab 3: Analysis / Visualisation ----
    W["color_map"] = widgets.Dropdown(
        options=["tab20", "turbo", "viridis", "plasma", "rainbow"],
        value="turbo", description="Line Colors",
        style={"description_width": "120px"},
    )
    W["fig3_mode"] = widgets.Dropdown(
        options=[("Identity", "identity"), ("Opinion", "opinion"), ("Dual", "dual")],
        value="opinion", description="Network view",
        style={"description_width": "120px"},
    )
    W["label_policy"] = widgets.Dropdown(
        options=[("Auto", "auto"), ("Always", "always"), ("Never", "never")],
        value="auto", description="Node Labels",
        style={"description_width": "120px"},
    )
    W["show_xi_bar"] = widgets.Checkbox(
        value=False, description="Show ξ bar chart",
        style={"description_width": "200px"},
    )
    W["show_public_signals"] = widgets.Checkbox(
        value=False, description="Show public signals sᵢ(xᵢ(t))",
        style={"description_width": "200px"},
    )
    W["add_perturbation"] = widgets.Checkbox(
        value=False, description="Add perturbation (beyond the paper)",
        style={"description_width": "200px"},
    )
    W["perturb_type"] = widgets.Dropdown(
        options=["Random", "Deterministic"], value="Random", description="Perturb Type",
        style={"description_width": "120px"},
    )
    W["perturb_scale"] = widgets.FloatSlider(
        value=_DEF.perturb_scale, min=0.0, max=1.0, step=0.001, description="Noise Std. Dev.",
        readout_format='.4f',
        style={"description_width": "120px"},
    )
    W["perturb_time"] = widgets.IntSlider(
        min=0, max=500, step=1, value=30,
        description="Perturb time",
        style={"description_width": "120px"},
    )

    # ---- Preset bar ----
    _opts = list(BUILTIN_PRESETS) + [n for n in list_presets()
                                     if n not in BUILTIN_PRESETS]
    W["preset_dropdown"] = widgets.Dropdown(
        options=_opts,
        value="Default", description="Preset",
        layout=widgets.Layout(width="260px"),
    )
    W["load_preset_btn"] = widgets.Button(
        description="Load Preset", button_style="info", icon="download",
        layout=widgets.Layout(width="130px"),
    )
    W["save_preset_btn"] = widgets.Button(
        description="Save Preset", button_style="", icon="save",
        layout=widgets.Layout(width="130px"),
    )
    W["preset_name"] = widgets.Text(
        value="", description="",
        layout=widgets.Layout(width="200px"),
        placeholder="preset name",
    )
    W["delete_preset_btn"] = widgets.Button(
        icon="trash", tooltip="Delete the selected preset file",
        layout=widgets.Layout(width="36px"),
    )

    # ---- Matrix Viewer Toggle ----
    W["show_matrix"] = widgets.Checkbox(
        value=False, description="Show W (LaTeX)",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="160px"),
    )

    # ---- Xi info ----
    W["xi_info_html"] = widgets.HTML(
        value="<b>ξ info:</b> press Run to compute."
    )

    # ---- Run bar ----
    W["run_btn"] = widgets.Button(
        description="Run", button_style="primary", icon="play",
        layout=widgets.Layout(width="100px"),
    )
    W["auto_run"] = widgets.Checkbox(
        value=True, description="Auto-run on change",
        style={"description_width": "initial"},
    )

    # ---- Display: figure layout (was hardcoded to "split") ----
    W["fig_layout"] = widgets.Dropdown(
        options=[("Split", "split"), ("Horizontal", "horizontal"), ("Vertical", "vertical")],
        value="split", description="Fig Layout",
        style={"description_width": "120px"},
    )

    # ---- Status line (preset load/save messages) ----
    W["status_html"] = widgets.HTML(value="")

    # Sliders update on release, text fields on submit/blur —
    # avoids one simulation per pixel of drag or per keystroke
    for wdg in W.values():
        if isinstance(wdg, (widgets.IntSlider, widgets.FloatSlider,
                            widgets.IntText, widgets.FloatText, widgets.Text)):
            wdg.continuous_update = False

    return W


# ===========================================================================
# Collect config from widgets
# ===========================================================================

def _collect_config(W: dict) -> SimConfig:
    """Read all widget values and return a SimConfig."""
    return SimConfig(
        N=int(W["N"].value),
        alpha=float(W["alpha"].value),
        L=int(W["L"].value),
        steps=int(W["steps"].value),
        dt=float(W["dt"].value),
        continuous_time=bool(W["continuous_time"].value),
        model_profile=str(W["model_profile"].value),
        signal_function=str(W["signal_function"].value),
        quantizer_family=str(W["quantizer_family"].value),
        custom_rules=_serialize_rules(W),
        custom_base=str(W["custom_base"].value),
        tanh_K=float(W["tanh_K"].value),
        graph_type=W["graph_type"].value,
        seed=int(W["seed"].value),
        sbm_sizes=str(W["sbm_sizes"].value),
        sbm_p_in=float(W["sbm_p_in"].value),
        sbm_p_out=float(W["sbm_p_out"].value),
        ws_k=int(W["ws_k"].value),
        ws_p=float(W["ws_p"].value),
        group_sizes=str(W["group_sizes"].value),
        cross_group_weight=float(W["cross_group_weight"].value),
        regular_degree=int(W["regular_degree"].value),
        directed=bool(W["directed"].value),
        directed_mode=str(W["directed_mode"].value),
        weight_mode=W["weight_mode"].value,
        ensure_valid_graph=bool(W["ensure_valid_graph"].value),
        alpha_mode=W["alpha_mode"].value,
        alpha_seed=int(W["alpha_seed"].value),
        alpha_min=float(W["alpha_min"].value),
        alpha_max=float(W["alpha_max"].value),
        alpha_manual=str(W["alpha_manual"].value),
        initial_opinion_mode=W["initial_opinion_mode"].value,
        opinion_seed=int(W["opinion_seed"].value),
        symmetric_scale=float(W["symmetric_scale"].value),
        initial_opinions_manual=str(W["initial_opinions_manual"].value),
        identical_value=float(W["identical_value"].value),
        eigen_index=int(W["eigen_index"].value),
        eigen_scale=float(W["eigen_scale"].value),
        eigen_noise_snr=float(W["eigen_noise_snr"].value),
        color_map=W["color_map"].value,
        fig_layout=str(W["fig_layout"].value),
        show_exag_profile=bool(W["show_exag_profile"].value),
        show_inverse_xi_profile=bool(W["show_inverse_xi_profile"].value),
        show_edges=bool(W["show_edges"].value),
        show_community=bool(W["show_community"].value),
        color_community_opinion=bool(W["color_community_opinion"].value),
        fig3_mode=W["fig3_mode"].value,
        label_policy=W["label_policy"].value,
        layout_algo=W["layout_algo"].value,
        add_perturbation=bool(W["add_perturbation"].value),
        perturb_type=str(W["perturb_type"].value),
        perturb_scale=float(W["perturb_scale"].value),
        perturb_time=int(W["perturb_time"].value),
    )


# ===========================================================================
# Dynamic widget visibility (Phase 5.3)
# ===========================================================================

# Maps widget name → condition function(config) → should be visible
_VISIBILITY_MAP: dict[str, object] = {
    "alpha":                  lambda cfg: cfg.alpha_mode == "scalar",
    "alpha_seed":             lambda cfg: cfg.alpha_mode == "heterogeneous-random",
    "alpha_min":              lambda cfg: cfg.alpha_mode == "heterogeneous-random",
    "alpha_max":              lambda cfg: cfg.alpha_mode == "heterogeneous-random",
    "alpha_manual":           lambda cfg: cfg.alpha_mode == "manual",
    "opinion_seed":           lambda cfg: cfg.initial_opinion_mode == "random",
    "symmetric_scale":        lambda cfg: cfg.initial_opinion_mode == "symmetric_zero",
    "initial_opinions_manual":lambda cfg: cfg.initial_opinion_mode == "manual",
    "identical_value":        lambda cfg: cfg.initial_opinion_mode == "identical",
    "eigen_index":            lambda cfg: cfg.initial_opinion_mode in ("eigen", "eigen_noise"),
    "eigen_scale":            lambda cfg: cfg.initial_opinion_mode in ("eigen", "eigen_noise"),
    "eigen_noise_snr":        lambda cfg: cfg.initial_opinion_mode == "eigen_noise",
    "directed_mode":          lambda cfg: cfg.directed,
    "regular_degree":         lambda cfg: cfg.graph_type in ("d_regular", "powerlaw_cluster"),
    "sbm_sizes":              lambda cfg: cfg.graph_type == "stochastic_block",
    "sbm_p_in":               lambda cfg: cfg.graph_type == "stochastic_block",
    "sbm_p_out":              lambda cfg: cfg.graph_type == "stochastic_block",
    "ws_k":                   lambda cfg: cfg.graph_type == "watts_strogatz",
    "ws_p":                   lambda cfg: cfg.graph_type == "watts_strogatz",
    "group_sizes":            lambda cfg: cfg.graph_type == "two_group",
    "cross_group_weight":     lambda cfg: (cfg.graph_type == "two_group"
                                           and cfg.weight_mode == "group_weights"),
    "show_community":         lambda cfg: cfg.graph_type == "karate_club",
    "color_community_opinion": lambda cfg: cfg.graph_type == "karate_club",
    "perturb_type":           lambda cfg: cfg.add_perturbation,
    "perturb_scale":          lambda cfg: cfg.add_perturbation,
    "perturb_time":           lambda cfg: cfg.add_perturbation,
    "tanh_K":                 lambda cfg: cfg.signal_function == "tanh",
    "quantizer_family":       lambda cfg: cfg.signal_function == "quantized",
    "custom_base":            lambda cfg: (cfg.signal_function == "quantized"
                                           and cfg.quantizer_family == "custom"),
    "custom_rules_box":       lambda cfg: (cfg.signal_function == "quantized"
                                           and cfg.quantizer_family == "custom"),
    "add_rule_btn":           lambda cfg: (cfg.signal_function == "quantized"
                                           and cfg.quantizer_family == "custom"),
    "show_inverse_xi_profile": lambda cfg: bool(getattr(cfg, "show_exag_profile", False)),
}


def _apply_visibility_rules(W: dict, cfg: SimConfig) -> None:
    """Show/hide widgets according to _VISIBILITY_MAP."""
    for name, condition in _VISIBILITY_MAP.items():
        if name in W:
            visible = bool(condition(cfg))
            W[name].layout.display = "" if visible else "none"


# ===========================================================================
# Auto-run suspension
# ===========================================================================

@contextmanager
def _autorun_suspended(W: dict):
    """Suppress auto-run while programmatically mutating widgets.

    Prevents one-simulation-per-widget storms during preset application and
    cross-widget synchronisation (the guard list is installed by build_ui)."""
    guard = W.get("_autorun_guard")
    if guard is None:
        yield
        return
    prev = guard[0]
    guard[0] = True
    try:
        yield
    finally:
        guard[0] = prev


# ===========================================================================
# Custom quantizer rules (Quantizers tab)
# ===========================================================================

def _uniform_strings(L: int) -> tuple[str, str]:
    """Threshold and level strings for a uniform L-bin quantizer."""
    t = np.round(np.linspace(-1.0, 1.0, int(L) + 1), 12) + 0.0
    v = np.round((t[:-1] + t[1:]) / 2.0, 12) + 0.0
    def fmt(arr):
        return ", ".join(f"{x:g}" for x in arr)
    return fmt(t), fmt(v)


def _parse_agent_spec(spec: str, N: int) -> list[int]:
    """Parse paper-indexed agent labels into zero-based internal indices.

    A specification containing zero is treated as a deprecated zero-based
    legacy specification so existing saved rules remain loadable.
    """
    spec = str(spec).strip().lower()
    if spec in ("", "all", "*"):
        return list(range(N))
    idxs: set[int] = set()
    for part in spec.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a > b:
                a, b = b, a
            idxs.update(range(a, b + 1))
        else:
            idxs.add(int(part))
    if 0 in idxs:
        warnings.warn(
            "Zero-based custom-rule agent labels are deprecated; use "
            f"paper-indexed labels 1..{N}",
            DeprecationWarning,
            stacklevel=2,
        )
        bad = sorted(i for i in idxs if not 0 <= i < N)
        if bad:
            raise ValueError(
                f"legacy zero-based agent indices {bad} out of range "
                f"0..{N - 1}"
            )
        return sorted(idxs)

    bad = sorted(i for i in idxs if not 1 <= i <= N)
    if bad:
        raise ValueError(
            f"paper-indexed agent labels {bad} out of range 1..{N}"
        )
    return sorted(i - 1 for i in idxs)


def _make_rule_row(W: dict, agents: str = "all",
                   thr: str = "", vals: str = "") -> widgets.HBox:
    """One custom-quantizer rule: agents + thresholds + reconstruction levels."""
    agents_w = widgets.Text(
        value=agents, description="Agents",
        placeholder="all, rest, or 1-11, 14",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="170px"),
    )
    thr_w = widgets.Text(
        value=thr, description="Thr.",
        placeholder="L+1 values in [-1, 1]",
        style={"description_width": "35px"},
        layout=widgets.Layout(width="225px"),
    )
    val_w = widgets.Text(
        value=vals, description="Lvl.",
        placeholder="L values",
        style={"description_width": "35px"},
        layout=widgets.Layout(width="205px"),
    )
    rm_btn = widgets.Button(
        icon="trash", tooltip="Remove this rule",
        layout=widgets.Layout(width="36px"),
    )
    on_change = getattr(W["custom_rules_box"], "_on_rule_change", None)
    for t in (agents_w, thr_w, val_w):
        t.continuous_update = False
        if on_change is not None:
            t.observe(on_change, names="value")
    row = widgets.HBox([agents_w, thr_w, val_w, rm_btn])

    def _remove(_b: object = None) -> None:
        box = W["custom_rules_box"]
        box.children = tuple(c for c in box.children if c is not row)
        cb = getattr(box, "_on_rule_change", None)
        if cb is not None:
            cb()

    rm_btn.on_click(_remove)
    return row


def _serialize_rules(W: dict) -> str:
    """JSON-encode the current rule rows (for SimConfig / presets)."""
    rules = []
    for row in W["custom_rules_box"].children:
        rules.append({
            "agents": str(row.children[0].value),
            "thresholds": str(row.children[1].value),
            "reconstruction_levels": str(row.children[2].value),
        })
    return json.dumps(rules)


# ===========================================================================
# Widget synchronisation helpers
# ===========================================================================

_DIRECTED_MODE_COMMON = (
    ("Strong-preserving asymmetric", "strongly_connected"),
    ("Random asymmetric", "random_asymmetric"),
)


def _directed_mode_options(graph_type: str) -> tuple[tuple[str, str], ...]:
    """Return direction schemes compatible with a base topology."""
    options = list(_DIRECTED_MODE_COMMON)
    if graph_type == "line":
        options.append(("Directed Path (endpoint-safe)", "line_directed"))
    if graph_type == "ring":
        options.append(("Directed Cycle", "ring_directed"))
    return tuple(options)


def _sync_directed_mode_options(W: dict, announce: bool = True) -> None:
    """Keep the directed-mode dropdown compatible with the selected graph."""
    graph_type = str(W["graph_type"].value)
    options = _directed_mode_options(graph_type)
    allowed = {value for _, value in options}
    current = str(W["directed_mode"].value)
    replacement = current if current in allowed else "strongly_connected"
    W["directed_mode"].options = options
    W["directed_mode"].value = replacement
    if current not in allowed and bool(W["directed"].value):
        notice = (
            f"Direction mode changed to Strong-preserving because "
            f"'{current}' is incompatible with {graph_type.replace('_', ' ')}."
        )
        guard = W.get("_directed_notice")
        if guard is not None:
            guard[0] = notice
        if announce and "status_html" in W:
            W["status_html"].value = (
                f"<span style='color:#b58900'>⚠ {notice}</span>")


def _connectivity_status(sim: object) -> tuple[str, str]:
    """Return Network-tab HTML and a compact run-status suffix."""
    cfg = sim.config
    strong = bool(getattr(sim, "is_strongly_connected", False))
    weak = bool(getattr(sim, "is_weakly_connected", False))
    scc_count = int(getattr(sim, "scc_count", 0))
    largest = int(getattr(sim, "largest_scc_size", 0))

    if bool(getattr(cfg, "directed", False)):
        if strong:
            detail = ""
            base_graph = getattr(sim, "base_graph", None)
            if base_graph is not None and nx.is_tree(base_graph):
                detail = (" Strong connectivity requires reciprocal links "
                          "on this tree topology.")
            html = ("<span style='color:#2e7d32'><b>Directed · strongly "
                    f"connected</b></span><span>{detail}</span>")
            return html, "&nbsp; Directed · strongly connected"
        html = (
            "<span style='color:#b58900'><b>Directed · not strongly connected"
            f"</b> — {scc_count} SCCs; largest {largest}/{sim.N}.</span>"
        )
        return html, ("&nbsp; <span style='color:#b58900'>⚠ Directed · not "
                      "strongly connected</span>")

    if weak:
        return ("<span style='color:#2e7d32'><b>Undirected · connected</b></span>",
                "")
    return ("<span style='color:#b58900'><b>Undirected · disconnected</b></span>",
            "&nbsp; <span style='color:#b58900'>⚠ graph is disconnected</span>")


def _sync_d_range(W: dict, change: dict | None = None) -> None:
    n = int(W["N"].value)
    steps = int(W["steps"].value)
    new_max = max(0, n - 1)
    W["regular_degree"].max = new_max
    if W["regular_degree"].value > new_max:
        W["regular_degree"].value = new_max
    W["eigen_index"].max = max(0, n - 1)
    if W["eigen_index"].value > n - 1:
        W["eigen_index"].value = 0
    W["perturb_time"].max = max(0, steps - 1)
    if not W["add_perturbation"].value and isinstance(change, dict) and change.get("owner") is W["steps"]:
        # When steps changes and perturbation is OFF, auto-center perturb_time
        W["perturb_time"].value = max(0, steps // 2)
        
    ws_new_max = max(2, n - 1)
    W["ws_k"].max = ws_new_max
    if W["ws_k"].value > ws_new_max:
        W["ws_k"].value = ws_new_max


def _sync_alpha_bounds(W: dict, change: dict | None = None) -> None:
    if W["alpha_min"].value >= W["alpha_max"].value:
        W["alpha_max"].value = min(0.99, W["alpha_min"].value + 0.01)

def _sync_default_layout(W: dict, change: dict | None = None) -> None:
    if not change or change.get("name") != "value":
        return
    
    gt = change.get("new")
    # Leaving Karate Club / Two groups: drop the graph-specific weight mode
    if change.get("old") == "karate_club" and W["weight_mode"].value == "karate_weights":
        W["weight_mode"].value = "uniform_in"
    if change.get("old") == "two_group" and W["weight_mode"].value == "group_weights":
        W["weight_mode"].value = "uniform_in"
    if gt == "complete":
        W["layout_algo"].value = "circular"
    elif gt == "star":
        W["layout_algo"].value = "shell"
    elif gt == "ring":
        W["layout_algo"].value = "circular"
    elif gt == "line":
        W["layout_algo"].value = "spring"
    elif gt == "petersen":
        W["N"].value = 10
    elif gt == "dodecahedral":
        W["N"].value = 20
    elif gt == "karate_club":
        W["N"].value = 34
        W["weight_mode"].value = "karate_weights"
    elif gt == "two_group":
        try:
            W["N"].value = sum(int(v) for v in str(W["group_sizes"].value).split(",") if v.strip())
        except ValueError:
            pass
        W["weight_mode"].value = "group_weights"
        W["layout_algo"].value = "circular"
    _sync_directed_mode_options(W)


# ===========================================================================
# Xi bar chart (Phase 5.5)
# ===========================================================================

def _render_xi_bar(W: dict, sim: object | None) -> None:
    """Render a small bar chart of ξ_i values in the xi_info_html widget."""
    if sim is None:
        W["xi_info_html"].value = "<b>ξ info:</b> press Run to compute."
        return
    xi = sim.xi  # type: ignore[attr-defined]
    N = len(xi)

    if N > 100:
        # Compact text summary for large N
        eta_text = ""
        if sim.eta_vec is not None:  # type: ignore[attr-defined]
            eta_text = (f" &nbsp;|&nbsp; <b>η:</b> min={sim.eta_vec.min():.3f}, "  # type: ignore[attr-defined]
                        f"mean={sim.eta_vec.mean():.3f}")  # type: ignore[attr-defined]
        W["xi_info_html"].value = (
            f"<b>ξ:</b> min={np.nanmin(xi):.3f}, mean={np.nanmean(xi):.3f}, "
            f"max={np.nanmax(xi):.3f}{eta_text}"
        )
        return

    fig, ax = plt.subplots(figsize=(min(N * 0.25 + 1, 10), 1.6))
    ax.bar(range(N), xi, color=sim.agent_colors, edgecolor="none")  # type: ignore[attr-defined]
    ax.axhline(np.nanmean(xi), color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Agent", fontsize=7)
    ax.set_xticks(range(N))
    ax.set_xticklabels([str(i + 1) for i in range(N)])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(r"$\xi_i$", fontsize=7)
    ax.set_title(f"Exaggeration factors (mean={np.nanmean(xi):.2f})", fontsize=7)
    ax.tick_params(labelsize=6)
    plt.tight_layout(pad=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    import base64
    b64 = base64.b64encode(buf.read()).decode("ascii")
    W["xi_info_html"].value = f'<img src="data:image/png;base64,{b64}" />'


# ===========================================================================
# Preset loading into widgets
# ===========================================================================

def _apply_preset_to_widgets(W: dict, cfg: SimConfig) -> None:
    """Write all SimConfig field values back to their widgets."""
    # Temporarily block observers to prevent cascading updates
    if W.get("_directed_notice") is not None:
        W["_directed_notice"][0] = None
    # Set this before graph_type so topology observers use the preset's final
    # directed state rather than the state left by the previous simulation.
    W["directed"].value = cfg.directed
    W["N"].value = cfg.N
    W["alpha"].value = cfg.alpha
    W["L"].value = cfg.L
    W["steps"].value = cfg.steps
    W["dt"].value = getattr(cfg, "dt", 1.0)
    W["continuous_time"].value = cfg.continuous_time
    W["model_profile"].value = cfg.model_profile
    W["signal_function"].value = cfg.signal_function
    W["quantizer_family"].value = cfg.quantizer_family
    try:
        _rules = json.loads(cfg.custom_rules or "[]")
    except Exception:
        _rules = []
    W["custom_rules_box"].children = tuple(
        _make_rule_row(
            W,
            str(r.get("agents", "all")),
            str(r.get("thresholds", "")),
            str(r.get("reconstruction_levels", "")),
        )
        for r in _rules)
    W["custom_base"].value = cfg.custom_base
    W["tanh_K"].value = cfg.tanh_K
    W["group_sizes"].value = cfg.group_sizes
    W["cross_group_weight"].value = cfg.cross_group_weight
    W["graph_type"].value = cfg.graph_type
    W["seed"].value = cfg.seed
    W["sbm_sizes"].value = cfg.sbm_sizes
    W["sbm_p_in"].value = cfg.sbm_p_in
    W["sbm_p_out"].value = cfg.sbm_p_out
    W["ws_k"].value = min(cfg.ws_k, W["ws_k"].max)
    W["ws_p"].value = cfg.ws_p
    W["regular_degree"].value = min(cfg.regular_degree, W["regular_degree"].max)
    _sync_directed_mode_options(W, announce=False)
    requested_mode = getattr(cfg, "directed_mode", "strongly_connected")
    allowed_modes = {value for _, value in W["directed_mode"].options}
    if requested_mode in allowed_modes:
        W["directed_mode"].value = requested_mode
    else:
        W["directed_mode"].value = "strongly_connected"
        if cfg.directed and W.get("_directed_notice") is not None:
            W["_directed_notice"][0] = (
                f"Preset direction mode '{requested_mode}' is incompatible "
                f"with {cfg.graph_type.replace('_', ' ')}; migrated to "
                "Strong-preserving."
            )
    W["weight_mode"].value = cfg.weight_mode
    W["alpha_mode"].value = cfg.alpha_mode
    W["alpha_seed"].value = cfg.alpha_seed
    W["alpha_min"].value = cfg.alpha_min
    W["alpha_max"].value = cfg.alpha_max
    W["alpha_manual"].value = cfg.alpha_manual
    W["initial_opinion_mode"].value = cfg.initial_opinion_mode
    W["opinion_seed"].value = cfg.opinion_seed
    W["symmetric_scale"].value = cfg.symmetric_scale
    W["initial_opinions_manual"].value = cfg.initial_opinions_manual
    W["identical_value"].value = cfg.identical_value
    W["eigen_index"].value = min(cfg.eigen_index, W["eigen_index"].max)
    W["eigen_scale"].value = cfg.eigen_scale
    W["eigen_noise_snr"].value = cfg.eigen_noise_snr
    if cfg.color_map not in W["color_map"].options:
        W["color_map"].options = tuple(W["color_map"].options) + (cfg.color_map,)
    W["color_map"].value = cfg.color_map
    W["fig_layout"].value = getattr(cfg, "fig_layout", "split")
    W["ensure_valid_graph"].value = getattr(cfg, "ensure_valid_graph", False)
    W["show_exag_profile"].value = getattr(cfg, "show_exag_profile", False)
    W["show_inverse_xi_profile"].value = getattr(cfg, "show_inverse_xi_profile", False)
    W["show_edges"].value = getattr(cfg, "show_edges", True)
    W["show_community"].value = getattr(cfg, "show_community", False)
    W["color_community_opinion"].value = getattr(cfg, "color_community_opinion", False)
    W["fig3_mode"].value = cfg.fig3_mode
    W["label_policy"].value = cfg.label_policy
    W["layout_algo"].value = cfg.layout_algo
    W["add_perturbation"].value = cfg.add_perturbation
    W["perturb_type"].value = cfg.perturb_type
    W["perturb_scale"].value = cfg.perturb_scale
    W["perturb_time"].value = min(cfg.perturb_time, W["perturb_time"].max)


# ===========================================================================
# Main public entry point
# ===========================================================================

def build_ui() -> widgets.VBox:
    """Build and return the interactive strategic-CODA interface.

    Returns
    -------
    ui : widgets.VBox
        The complete widget tree.  Pass to IPython.display.display().

    Example
    -------
    >>> from strategic_coda import build_ui
    >>> from IPython.display import display
    >>> ui = build_ui()
    >>> display(ui)
    """
    from .simulator import StrategicCODASimulator
    from .plots import plot_simulation
    from .quantizers import Quantizer

    W = _make_widgets()
    _current_sim: list[StrategicCODASimulator | None] = [None]
    _suspend_autorun = [False]
    W["_autorun_guard"] = _suspend_autorun  # shared with _autorun_suspended(W)
    W["_directed_notice"] = [None]
    _prev_L = [int(W["L"].value)]

    def _sync_custom_defaults(change: dict | None = None) -> None:
        """Changing the number of bins L: untouched auto-seeded uniform rules
        defaults follow the new L; hand-edited rules are left untouched."""
        L_old, L_new = _prev_L[0], int(W["L"].value)
        _prev_L[0] = L_new
        if W["quantizer_family"].value != "custom" or L_old == L_new:
            return
        old_t, old_v = _uniform_strings(L_old)
        new_t, new_v = _uniform_strings(L_new)
        _suspend_autorun[0] = True
        try:
            for row in W["custom_rules_box"].children:
                if (str(row.children[1].value).strip() == old_t
                        and str(row.children[2].value).strip() == old_v):
                    row.children[1].value = new_t
                    row.children[2].value = new_v
        finally:
            _suspend_autorun[0] = False

    def _on_continuous_change(change: dict | None = None) -> None:
        """Convenience: entering continuous mode with dt still at 1.0 sets a
        sensible integration step (0.1); leaving it restores 1.0."""
        _suspend_autorun[0] = True
        try:
            if W["continuous_time"].value and float(W["dt"].value) == 1.0:
                W["dt"].value = 0.1
            elif not W["continuous_time"].value and float(W["dt"].value) == 0.1:
                W["dt"].value = 1.0
        finally:
            _suspend_autorun[0] = False

    def _on_family_change(change: dict | None = None) -> None:
        """Entering Custom with no rules: seed one all-agents uniform rule."""
        if W["quantizer_family"].value != "custom":
            return
        if len(W["custom_rules_box"].children) == 0:
            t_str, v_str = _uniform_strings(int(W["L"].value))
            W["custom_rules_box"].children = (
                _make_rule_row(W, "all", t_str, v_str),)

    # ---- Output areas ----
    out_main = widgets.Output()
    out_network = widgets.Output(layout=widgets.Layout(display="flex", justify_content="center", width="100%"))
    out_matrix = widgets.Output(layout=widgets.Layout(width="95%", overflow="auto", margin="0px 0px 0px 0px"))

    # ---- Observers for sync (guarded: their widget mutations must not
    #      trigger auto-run — only the user's original change does) ----
    def _guarded(sync_fn):
        def _handler(change=None):
            with _autorun_suspended(W):
                sync_fn(W, change)
        return _handler

    W["N"].observe(_guarded(_sync_d_range), names="value")
    W["steps"].observe(_guarded(_sync_d_range), names="value")
    W["alpha_min"].observe(_guarded(_sync_alpha_bounds), names="value")
    W["graph_type"].observe(_guarded(_sync_default_layout), names="value")
    W["L"].observe(_sync_custom_defaults, names="value")
    W["quantizer_family"].observe(_on_family_change, names="value")
    W["continuous_time"].observe(_on_continuous_change, names="value")

    def _on_any_change(change: dict | None = None) -> None:
        """Re-apply visibility rules whenever any key widget changes."""
        cfg_preview = _collect_config(W)
        _apply_visibility_rules(W, cfg_preview)

    for name in ("alpha_mode", "initial_opinion_mode", "graph_type", "weight_mode",
                 "add_perturbation", "directed", "signal_function",
                 "quantizer_family", "show_exag_profile"):
        W[name].observe(lambda c: _on_any_change(c), names="value")

    # ---- Run function ----
    def _run(b: object = None) -> None:
        cfg = _collect_config(W)
        track_bins = bool(W["show_public_signals"].value)

        # Custom quantizer rules: each rule = (agent spec, thresholds, levels);
        # later rules win on overlap, uncovered agents use the base family
        quantizers = None
        overrides = None
        if cfg.signal_function == "quantized" and cfg.quantizer_family == "custom":
            try:
                rules = json.loads(getattr(cfg, "custom_rules", "") or "[]")
            except Exception as exc:
                with out_main:
                    out_main.clear_output(wait=True)
                    print(f"❌  Custom rules error: {exc}")
                return
            overrides = {}
            rest_rules = []  # 'rest'/'others': agents covered by no other rule
            for k, rule in enumerate(rules, start=1):
                try:
                    thr = [float(s) for s in str(rule.get("thresholds", "")).replace(";", ",").split(",") if s.strip()]
                    level_text = rule.get("reconstruction_levels", "")
                    levels = [
                        float(s)
                        for s in str(level_text).replace(";", ",").split(",")
                        if s.strip()
                    ]
                    q_rule = Quantizer.custom(thr, levels)
                    spec = str(rule.get("agents", "all")).strip().lower()
                    if spec in ("rest", "others", "other"):
                        rest_rules.append(q_rule)
                        continue
                    agents = _parse_agent_spec(spec, int(cfg.N))
                except ValueError as exc:
                    with out_main:
                        out_main.clear_output(wait=True)
                        print(f"❌  Custom rule {k}: {exc}")
                    return
                for i in agents:
                    overrides[i] = q_rule
            if rest_rules:
                uncovered = [i for i in range(int(cfg.N)) if i not in overrides]
                for q_rule in rest_rules:  # later 'rest' rules win
                    for i in uncovered:
                        overrides[i] = q_rule
            overrides = overrides or None
            cfg = cfg.replace(quantizer_family=str(cfg.custom_base))

        with out_main:
            out_main.clear_output(wait=True)
            
        dt = float(cfg.dt)
        if not 0.0 < dt <= 1.0:
            with out_main:
                out_main.clear_output(wait=True)
                print("❌  Time step dt must be in (0, 1].")
            return
        n_iter = int(cfg.steps)
        if cfg.continuous_time:
            # Horizon T is a continuous time; integrate round(T/dt) steps
            n_iter = max(1, int(round(cfg.steps / dt)))
            if n_iter > 200_000:
                with out_main:
                    out_main.clear_output(wait=True)
                    print(f"❌  T/dt = {n_iter} iterations is too large; "
                          "increase dt or reduce Time Steps.")
                return

        W["run_btn"].disabled = True
        W["status_html"].value = "⏳ Running…"
        _t0 = time.time()
        try:
            sim = StrategicCODASimulator(
                cfg,
                quantizers=quantizers,
                quantizer_overrides=overrides,
            )
            history, bin_indices = sim.run(n_iter, track_bins=track_bins)
            _current_sim[0] = sim
            if W["show_xi_bar"].value:
                _render_xi_bar(W, sim)
            artists = plot_simulation(
                sim, history,
                fig3_mode=cfg.fig3_mode,
                label_policy=cfg.label_policy,
                show=False,
                layout=str(cfg.fig_layout),
                show_exag_profile=bool(cfg.show_exag_profile),
                show_inverse_xi_profile=bool(cfg.show_inverse_xi_profile),
                return_artists=True,
                show_signals=track_bins,
                bin_indices=bin_indices,
            )
            
            fig = artists["fig"]
            fig_net = artists.get("fig_net", None)
            
            with out_main:
                display(fig)
                plt.close(fig)
            
            if fig_net is not None:
                with out_network:
                    out_network.clear_output(wait=True)
                    display(fig_net)
                    plt.close(fig_net)
            
            # Expose to UI object
            ui.latest_sim = sim
            ui.latest_history = history
            ui.latest_bin_indices = bin_indices if track_bins else None
            connectivity_html, connectivity_note = _connectivity_status(sim)
            W["connectivity_html"].value = connectivity_html

            _render_matrix()
            _render_signal_plot()

            notice = W["_directed_notice"][0]
            notice_html = (f"&nbsp; <span style='color:#b58900'>⚠ {notice}</span>"
                           if notice else "")
            W["status_html"].value = (
                f"✓ {n_iter} iterations in {time.time() - _t0:.2f} s"
                f"{connectivity_note}{notice_html}")
            W["_directed_notice"][0] = None

        except ValueError as exc:
            with out_main:
                print(f"❌  Configuration error: {exc}")
            W["status_html"].value = "<span style='color:red'>✗ configuration error</span>"
            return
        except Exception as exc:  # surface anything unexpected instead of dying silently
            with out_main:
                print(f"❌  Unexpected {type(exc).__name__}: {exc}")
            W["status_html"].value = f"<span style='color:red'>✗ {type(exc).__name__}</span>"
            return
        finally:
            W["run_btn"].disabled = False

    # ---- Preset helpers ----
    def _refresh_preset_options(select: str | None = None) -> None:
        opts = list(BUILTIN_PRESETS) + [n for n in list_presets()
                                        if n not in BUILTIN_PRESETS]
        current = W["preset_dropdown"].value
        W["preset_dropdown"].options = opts
        if select and select in opts:
            W["preset_dropdown"].value = select
        elif current in opts:
            W["preset_dropdown"].value = current

    # ---- Load preset function ----
    def _load_preset(b: object = None) -> None:
        _refresh_preset_options()  # pick up files added outside the UI
        name = W["preset_dropdown"].value
        with _autorun_suspended(W):  # apply all values without run storms
            try:
                _apply_preset_to_widgets(W, load_preset(name))
            except Exception as exc:
                W["status_html"].value = f"<span style='color:red'>Load error: {exc}</span>"
                return
            _on_any_change()
        notice = W["_directed_notice"][0]
        notice_html = (f" — <span style='color:#b58900'>⚠ {notice}</span>"
                       if notice else "")
        W["status_html"].value = f"<b>Loaded</b> preset '{name}'{notice_html}"
        _maybe_auto_run()  # exactly one run for the whole preset

    # ---- Save preset function ----
    def _save_preset(b: object = None) -> None:
        name = str(W["preset_name"].value).strip()
        if not name:
            W["status_html"].value = "<span style='color:red'>Enter a preset name first.</span>"
            return
        if name in BUILTIN_PRESETS:
            W["status_html"].value = ("<span style='color:red'>That name is a "
                                      "built-in preset; choose another.</span>")
            return
        cfg = _collect_config(W)
        try:
            save_preset(cfg, name)
        except Exception as exc:
            W["status_html"].value = f"<span style='color:red'>Save error: {exc}</span>"
            return
        _refresh_preset_options(select=name)
        W["status_html"].value = (f"<b>Saved</b> preset '{name}' — it is now "
                                  "in the dropdown, ready to load.")

    # ---- Delete preset function ----
    def _delete_preset(b: object = None) -> None:
        name = W["preset_dropdown"].value
        if name in BUILTIN_PRESETS:
            W["status_html"].value = ("<span style='color:red'>Built-in presets "
                                      "cannot be deleted.</span>")
            return
        try:
            os.remove(preset_path(name))
        except Exception as exc:
            W["status_html"].value = f"<span style='color:red'>Delete error: {exc}</span>"
            return
        _refresh_preset_options()
        W["status_html"].value = f"<b>Deleted</b> preset '{name}'"

    # ---- Show Matrix function ----
    def _render_matrix() -> None:
        if not W["show_matrix"].value:
            out_matrix.clear_output(wait=False)
            return

        if _current_sim[0] is None:
            return
        
        W_mat = _current_sim[0].W
        latex_str = "\\begin{bmatrix}\n"
        for row in W_mat:
            fmt_row = []
            for val in row:
                if val == 0:
                    fmt_row.append("0")
                elif val == 1:
                    fmt_row.append("1")
                else:
                    fmt_row.append(f"{val:.3f}")
            latex_str += " & ".join(fmt_row) + " \\\\\n"
        latex_str += "\\end{bmatrix}"
        
        with out_matrix:
            out_matrix.clear_output(wait=True)
            from IPython.display import Math, display
            display(Math(latex_str))

    def _render_signal_plot() -> None:
        """Small preview of a non-quantized signaling map (Display tab)."""
        signal_fn = W["signal_function"].value
        if signal_fn == "quantized":
            W["signal_plot_out"].clear_output(wait=False)
            return

        with W["signal_plot_out"]:
            W["signal_plot_out"].clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(2, 2))
            x_min, x_max = -1.0, 1.0
            x = np.linspace(x_min, x_max, 200)
            if signal_fn == "linear":
                ax.plot(x, x, color='blue', linewidth=2)
            elif signal_fn == "tanh":
                K = float(W["tanh_K"].value)
                y = np.tanh(K * x)
                ax.plot(x, y, color='blue', linewidth=2)
            elif signal_fn == "saturated":
                sim = _current_sim[0]
                if sim is not None:
                    colors = sim.agent_colors
                    for i in range(sim.N):
                        y = np.clip(sim.xi[i] * x, -1.0, 1.0)
                        ax.plot(x, y, color=colors[i], linewidth=0.5)
                else:
                    y = np.clip(1.0 * x, -1.0, 1.0)
                    ax.plot(x, y, color='blue', linewidth=2)
            else:
                ax.plot(x, x, color='blue', linewidth=2)
            ax.plot(x,x, color='red', linewidth=2, alpha=0.6, linestyle='--')
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(x_min, x_max)
            ticks = [x_min, 0.0, x_max]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.grid(True, color='gray', linestyle='--', linewidth=0.2, alpha=0.6, which='major')
            plt.tight_layout(pad=0.0)
            display(fig)
            plt.close(fig)


    # ---- Dynamic toggles ----
    def _toggle_xi_bar(change: dict | None = None) -> None:
        if W["show_xi_bar"].value:
            W["xi_info_html"].layout.display = ""
            if _current_sim[0] is not None:
                _render_xi_bar(W, _current_sim[0])
        else:
            W["xi_info_html"].layout.display = "none"

    W["show_xi_bar"].observe(_toggle_xi_bar, names="value")
    _toggle_xi_bar()  # Initialize to hidden

    # ---- Auto-run observer ----
    def _maybe_auto_run(change: dict | None = None) -> None:
        if _suspend_autorun[0]:
            return
        if W["auto_run"].value:
            _run()

    _NO_AUTO_RUN = (
        "run_btn", "auto_run",                              # run bar itself
        "show_matrix", "show_xi_bar",                       # have their own handlers
        "load_preset_btn", "save_preset_btn",               # buttons (no value trait)
        "preset_dropdown", "preset_name",                   # act on Load/Save click only
        "xi_info_html", "status_html",                      # passive HTML
        "connectivity_html",                                # passive HTML
        "signal_plot_out",                                  # Output area
        "custom_rules_box", "add_rule_btn",                 # rules: own handlers
        "delete_preset_btn", "_autorun_guard",              # button / guard list
        "_directed_notice",                                  # migration notice list
    )
    for name in list(W.keys()):
        if name not in _NO_AUTO_RUN:
            W[name].observe(_maybe_auto_run, names="value")

    # ---- Button handlers ----
    W["run_btn"].on_click(lambda b: _run())

    # rule rows re-run on edit; Add rule appends a uniform-L seeded row
    W["custom_rules_box"]._on_rule_change = _maybe_auto_run

    def _add_rule(b: object = None) -> None:
        t_str, v_str = _uniform_strings(int(W["L"].value))
        box = W["custom_rules_box"]
        box.children = box.children + (_make_rule_row(W, "all", t_str, v_str),)
        _maybe_auto_run()

    W["add_rule_btn"].on_click(_add_rule)

    # seed one default rule (hidden unless the Custom family is active)
    _t0, _v0 = _uniform_strings(int(W["L"].value))
    W["custom_rules_box"].children = (_make_rule_row(W, "all", _t0, _v0),)
    W["show_matrix"].observe(lambda c: _render_matrix(), names="value")
    W["load_preset_btn"].on_click(_load_preset)
    W["save_preset_btn"].on_click(_save_preset)
    W["delete_preset_btn"].on_click(_delete_preset)

    # ==================================================================
    # Build tab layout
    # ==================================================================

    fixed_layout = widgets.Layout(width="100%", overflow="auto", padding="5px")

    # Tab 0: Model — what is being simulated (dynamics + graph)
    tab_model = widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                W["N"], W["alpha"], W["steps"], W["model_profile"],
                W["signal_function"], W["tanh_K"],
                W["alpha_mode"], W["alpha_seed"], W["alpha_min"],
                W["alpha_max"], W["alpha_manual"],
                W["dt"], W["continuous_time"],
            ]),
            widgets.VBox([
                W["graph_type"], W["regular_degree"], W["ws_k"], W["ws_p"],
                W["sbm_sizes"], W["sbm_p_in"], W["sbm_p_out"],
                W["group_sizes"], W["cross_group_weight"], W["seed"],
                W["layout_algo"],
                W["add_perturbation"], W["perturb_type"],
                W["perturb_scale"], W["perturb_time"],
            ]),
        ])
    ], layout=fixed_layout)

    # Tab 1: Quantizers — signalling map (family + manual design)
    quant_help = widgets.HTML(
        "<i>Used when the signaling map is 'Quantized'. The paper's model is "
        "'Credible codes': every agent uses its L-bin credible code "
        "(Theorem 4), with thresholds τ and midpoint reconstruction levels r. "
        "Custom: each rule maps agents ('all', '3', or ranges like "
        "'1-11, 14', numbered from 1) to a quantizer given by L+1 increasing "
        "thresholds spanning [-1, 1] and L strictly increasing reconstruction "
        "levels in [-1, 1]. Later rules override earlier ones; 'rest' (or "
        "'others') targets every agent no other rule covers. Agents covered "
        "by no rule use the 'Others use' family. Changing the number of bins "
        "L reseeds rules that still hold the untouched uniform defaults.</i>"
    )
    tab_quant = widgets.VBox([
        W["quantizer_family"],
        W["L"],
        W["custom_base"],
        W["custom_rules_box"],
        W["add_rule_btn"],
        quant_help,
    ], layout=fixed_layout)

    # Tab 2: Network — weights, directedness, W matrix
    tab_net = widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                W["weight_mode"], W["directed"], W["directed_mode"],
            ]),
            widgets.VBox([
                W["show_matrix"], W["ensure_valid_graph"],
                W["connectivity_html"],
            ]),
        ])
    ], layout=fixed_layout)

    # Tab 3: Initial Conditions
    tab_init = widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                W["initial_opinion_mode"],
                W["opinion_seed"],
                W["symmetric_scale"],
                W["initial_opinions_manual"],
                W["identical_value"],
                W["eigen_index"],
                W["eigen_scale"],
                W["eigen_noise_snr"],
            ]),
        ]),
    ], layout=fixed_layout)

    # Tab 4: Display — how results are drawn
    tab_disp = widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                W["fig_layout"], W["color_map"],
                W["fig3_mode"], W["label_policy"],
            ]),
            widgets.VBox([
                W["show_edges"], W["show_community"],
                W["color_community_opinion"],
                W["show_exag_profile"], W["show_inverse_xi_profile"],
                W["signal_plot_out"],
            ]),
        ])
    ], layout=fixed_layout)

    # Tab 5: Analysis — what is measured
    tab_an = widgets.VBox([
        widgets.HBox([W["show_xi_bar"], W["show_public_signals"]]),
    ], layout=fixed_layout)

    # Tab 6: Presets
    tab_presets = widgets.VBox([
        widgets.HBox([W["preset_dropdown"], W["load_preset_btn"],
                      W["delete_preset_btn"]]),
        widgets.HBox([W["preset_name"], W["save_preset_btn"]]),
    ], layout=fixed_layout)

    tabs = widgets.Tab(children=[tab_model, tab_quant, tab_net, tab_init,
                                 tab_disp, tab_an, tab_presets],
                       layout=widgets.Layout(width="100%"))
    for i, title in enumerate(["Model", "Quantizers", "Network", "Initial Cond.",
                               "Display", "Analysis", "Presets"]):
        tabs.set_title(i, title)

    # Run bar + status line
    run_bar = widgets.HBox([W["run_btn"], W["auto_run"]])

    # Xi info
    xi_bar = W["xi_info_html"]

    network_host = widgets.VBox(
        [out_network], layout=widgets.Layout(width="100%"))

    control_panel = widgets.VBox([
        run_bar,
        tabs,
        W["status_html"],
        xi_bar,
        network_host,
        out_matrix
    ], layout=widgets.Layout(width="800px", min_width="700px", min_height="1000px", padding="5px"))

    # Full UI
    ui = widgets.HBox([
        out_main,
        control_panel,
    ], layout=widgets.Layout(
        width="100%", flex_flow="row wrap", align_items="flex-start",
    ))

    # Initialize exported data
    ui.latest_sim = None
    ui.latest_history = None
    ui.latest_bin_indices = None

    # Initial visibility pass
    _sync_directed_mode_options(W, announce=False)
    _on_any_change()
    _sync_d_range(W)
    
    # Run once automatically
    _run()

    return ui
